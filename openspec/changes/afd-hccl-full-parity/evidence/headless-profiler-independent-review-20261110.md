# Independent review: explicit headless F profiler control

The reviewer inspected the plugin-owned Unix control service and its existing
FFN EngineCore loop integration. No source blocker remains in the reviewed
bytes below. This is static review; local Windows socket skips do not qualify
the Unix listener, and no device or new service was started by this reviewer.

| Source | Reviewed SHA256 |
| --- | --- |
| `afd_plugin/compat/npu/profiler_control.py` | `753dbc11acbdfbc6b1df640b165b4c87af79cade8a903cea039b6c5e88c654ba` |
| `afd_plugin/compat/patches/engine_core.py` | `c7b5972f26b29d3bb68441fe94faad3dadc0be1f8cbfec7252c8f943ed63beb8` |
| `tests/unit/compat/npu/test_profiler_control.py` | `62bf087ebcef1aa193350b5307fd9f39e592d76a371cce42be413382a4cc4a69` |
| `tests/unit/compat/patches/test_engine_core.py` | `5e6053b855a92abf75b4738cf6b33f59fd407ee57fb4e54a4d305735cfcefc50` |
| `tests/unit/fakes/profiler_control.py` | `91feec1ea62fba245004a4f375a6742cdb04195495a9c2ce5218ef760ca527d4` |
| `afd_plugin/compat/npu/profiler.py` (retained session owner) | `4dcd664757a788f26efbdf1a3dfaf8acbf7a0a007f26bd2e53c1b6a896d1c117` |

The upstream reference is vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
`vllm/v1/executor/abstract.py:Executor.collective_rpc` accepts the existing
`method`, `timeout`, `args`, `kwargs` and optional `non_block` API. The selected
blocking overload documents `TimeoutError`; this adapter passes a fixed
30-second timeout and only `profile` or `afd_profiler_status`.
`vllm/v1/engine/core.py:EngineCoreProc._process_input_queue` uses
`process_input_queue_block`; the existing HCCL F loop sets it false while
the FFN worker thread owns the blocking receive/compute loop. The new socket
is polled on that management thread and never directly starts or stops a
profiler backend there.

## Fixed lifecycle and resource boundary

`create_ffn_profiler_control` returns immediately when its explicit path env
is absent. An enabled endpoint requires the actual NPU/HCCL F role, the F role
profiler env, and no native profiler config. The absolute-path listener binds
before worker startup, so an invalid explicit endpoint cannot leave a newly
started worker behind. Binding refuses existing paths, including dangling
symlinks. The pathname is chmod 0600 before listening.

The service accepts at most four clients, one newline JSON request per
connection, a bounded request size and one executor request per poll. It
retains partial writes rather than blocking on a full response. The five-second
deadline concerns incomplete client IO. A synchronous native collective RPC
can occupy the management loop until its own timeout; this is not a hard
inference-latency guarantee or an independent RPC cancellation facility.

`_run_ffn_busy_loop` constructs the optional listener before
`start_ffn_server_loop`, polls only after that RPC succeeds, and closes the
local endpoint in its existing finalizer before stopping the worker service.
No HCCL control/data message or worker execution owner is replaced. The
normal role session still applies a requested start/stop at its next real
owner boundary; accepted control responses do not claim applied state.

Close attempts every local client and listener, and unlinks only a socket
whose pathname device/inode still match this instance. A replacement file is
preserved. An unlink failure retains the recorded identity so a later close
can retry. EngineCore logs a cleanup `OSError` without masking the primary
service failure.

## Finding and correction

The initial `_reply` could raise on non-JSON, cyclic or non-finite values
returned by the executor status RPC. Such a response could escape `poll` and
terminate the F daemon rather than reject the management response. The
reviewed implementation catches `TypeError`, `ValueError` and
`RecursionError`, returns a fixed `invalid_rpc_response` frame, and also
bounds the encoded response. The public test sends invalid executor results
and then reaches the same live listener with a fresh valid status request.
This closes the source finding without changing the native RPC or profiler.

The mirror also covers real socket request/response behavior, partial writes,
malformed and oversized requests, capacity/deadline release, fixed RPC errors,
pending owner state and replacement-path preservation. The EngineCore seam
exercises control through its own retained executor and cleanup on both STOP
and service error. These are CPU/API tests, not NPU trace evidence. Actual
Linux execution is recorded separately below. The table records the final
mirror formatting and named partial-write constant; the earlier reviewed
mirror hash was `8dd0707d5a1e9dae7cc7f2f9c8ca0a24acc0a33c66b2966227e5afd88c8cd746`.
The reviewer re-read the final partial-write, invalid-RPC and recursive-JSON
seams, the source's one-request-per-poll budget, and the real-socket shared
factory. The mirror does not contain a separate multiple-ready-client test
for that budget. Neither runtime hash changed during that follow-up.

## Actual Linux CPU follow-up

The reviewer independently read the immutable identity, summary, installed
source checks and JUnit XML for
`a3-profiler-cpu-20261110/headless-profiler-027`. The snapshot at HEAD
`bdf4b3e3eeef2781713a0b63591eb0ce79e94843` has manifest
`1cad334ebdf3f51de7f4a74fdc99992d8bce4527bb01cefea7b0065527f61db1`.
All six source hashes above match that snapshot and current files. The
identity JSON itself is
`567854f88f43ed37ea198386ffbd4efeac89862b7d34334b4ba40653b2c15fb7`.

The selected complete modules passed **91 tests**, with zero failures,
errors or skips, JUnit 0.411 s: real Unix control 31, EngineCore 29 and retained
profiler 31. All 17 installed upstream source checks matched, the harness
returned zero and explicitly recorded `device_probe_started: false`.
This establishes actual Linux AF_UNIX/JSON and management-loop seam behavior
with the existing dependencies. The separate local Windows outcome of 65
passed and 26 AF_UNIX skips remains a distinct collection/execution result;
it is not substituted for the Linux run. The RPC boundary in those tests is
an explicit executor/session seam, so this does not establish full worker
process RPC, live model serving, NPU timeline capture or A5 acceptance.

## Cost and removal

Disabled mode adds no listener or socket IO. Enabled mode owns one listener,
up to four small client buffers and at most one fixed collective RPC per
management poll. It introduces no new background thread, global registry,
generic RPC method or HCCL protocol. Remove this local adapter when native
headless EngineCore exposes an externally usable fixed profiler-management
entry for its already-owned executor. HTTP endpoint validation and the
controlled NPU profiler export probe remain distinct evidence scopes.
