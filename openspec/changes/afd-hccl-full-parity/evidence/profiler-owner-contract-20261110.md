# H14 profiler control and resident FFN execution owner

This records source inspection and a plugin implementation seam. CPU management
tests do not prove that torch-npu generated a complete NPU timeline. No profiler,
model service, or device experiment was started for this implementation.

## Pinned source and the original gap

| Source | Exact revision / symbol | Contract |
| --- | --- | --- |
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, `vllm/entrypoints/serve/profile/api_router.py:22,30` | HTTP awaits the engine start/stop methods. The router is attached only when native `ProfilerConfig.profiler` is configured. |
| vLLM | Same revision, `vllm/v1/engine/async_llm.py:1044,1050`, `engine/core.py:774`, `executor/abstract.py:266` | `start_profile/stop_profile` reach executor `collective_rpc("profile", args=(is_start, profile_prefix))`. They do not issue an AFD FORWARD. |
| vLLM | Same revision, `vllm/v1/executor/multiproc_executor.py`, `WorkerProc.worker_busy_loop/_execute_worker_rpc` | The worker process dequeues and invokes management methods on its RPC thread. `UniProcExecutor.collective_rpc` invokes its worker directly. Neither path waits on the plugin daemon's connector receive. |
| Ascend | `7e37e29a7803d6897a7113088e2b771f83205bc0`, `vllm_ascend/worker/worker.py:1223`, `NPUWorker.profile(self, is_start: bool = True, profile_prefix: str \| None = None)` | Native profiler configuration validation, rank trace names, lazy native wrapper start/restart, and stop. The function has no return annotation. |
| Ascend | Same revision, `worker/worker.py:788`, `NPUWorker.execute_model` | Advances the native worker profiler, then delegates model execution. The connector-driven F worker rejects scheduler `execute_model`, so this native step is not the F parent loop. |
| Ascend | Same revision, `vllm_ascend/profiler/torch_npu_profiler.py`, `TorchNPUProfilerWrapper._start/_stop/_profiler_step` | Wraps a separate `torch_npu.profiler.profile`; `_profiler_step` returns `True` without calling the role profiler. |
| vLLM | `ced6857a`, `vllm/profiler/wrapper.py`, `WorkerProfiler.start/step/stop` | Owns native active/running flags and delayed/max-iteration policy. Successful control state alone is not evidence that the F daemon was recorded. |
| Donor | `4ddd69a391dfb5b3e362c7601399e538a31de702`, `afd_plugin/v1/worker/npu/ffn_worker.py:141`, `profile`; `ffn_model_runner.py:200`, `start_afd_profiler/stop_afd_profiler` | Routes RPC to the runner's role profiler. Creation/stop occur in RPC, while the resident F daemon advances execution; the donor does not establish a single thread owner for these operations. |
| Donor | Same revision, `afd_plugin/compat/npu/profiler.py`, `_DEFAULT_PROFILE_RANKS`, `_env_ranks`, `create_afd_npu_profiler` | Role-owned enable/schedule/trace settings; rank default `{0}`, `all`, or a comma-separated set of nonnegative ranks. |

Before this change, the current F model runner created **and started** its role
profiler during construction. It advanced that profiler from legacy model
execution, but HCCL U1 and staged U1/U2 execution did not advance it. The F worker
inherited native `NPUWorker.profile`, controlling a second object. Its resident
receive is in `afd-npu-ffn-worker-loop`, so the direct problem is the mismatched
profiler/execution owner, rather than the RPC thread being inside F receive.

## Installed Python evidence and its limit

The read-only stdlib collection is
`../../afd-hccl-v2-multistream/evidence/torch-npu-profiler-source-20261110.json`.
It records torch-npu `2.10.0.post4`, no torch/torch-npu imports, and full source
for installed `torch_npu/profiler/profiler.py`, SHA-256
`1a04902572383b74a401449e73929b2c4ce728d6c81bb59341876b03c7a82168`.

At lines 296–305, scheduled `profile.start` enters an autograd
`record_function` and an MSTX range on the current NPU stream. Lines 308–331
exit/re-enter these during stop/step. Lines 343–379 expose explicit child-thread
enable/disable APIs. These Python operations justify retaining one execution
owner. They do **not** establish whether the C++ collector is global or thread
local, or prove that every auxiliary thread appears in exported traces.

## Implemented public boundary

`create_afd_npu_profiler` retains the immediate-start behavior used by A. The new
`create_deferred_afd_npu_profiler(role, *, role_rank=0)` freezes the role settings
and rank selection without importing torch-npu. Disabled mode returns `None`.
Selected and excluded ranks retain an honest management session, but excluded
ranks do not create the backend profiler. F runner constructors select this
factory; they no longer start a role profiler on the constructor thread.

`AFDNPUProfilerSession.request(is_start, profile_prefix)` only records a desired
state. `before_step`, `step`, and `finish` enforce one thread identity and perform
backend creation/start/step/stop/export there. A short instance lock protects
request/status snapshots and is released before backend calls. An RPC during a
blocked receive or export can therefore return without waiting for that work.
The next execution boundary applies the request. Normal STOP finalizes on the
daemon; it does not apply a pending start to an already exited owner.

The F worker override preserves the exact native `profile` signature. Explicit
role-env mode submits requests and exposes `afd_profiler_status()` with
`requested`, `applied`, `pending`, selected rank, request IDs, owner thread ID,
steps, closed, and sticky failure fields. Those fields describe Python control
state, not device completion or a validated trace. Requests in idle receive may
remain pending until the next real parent or STOP; there is no new management
wire, group destruction, fake abort, or extra F FORWARD.

The daemon checks requests before receive; successful HCCL U1/staged parents
advance one profiler step, and STOP does not count as a parent. Legacy/Async F
runner step sites remain the single advance for those execution paths. Daemon
exit finalizes the session in its owner context. A profiler failure is retained
and surfaced as a service failure; references are not released and no retry is
claimed. If serving and cleanup both fail, the original serving failure remains
the propagated cause, with the profiler error also visible in status/logging.
The legacy CAM normal-shutdown receive-error suppression explicitly excludes
a failed profiler session, so export failure cannot be mistaken for an expected
interrupted receive.
Starting a new resident loop also rejects a failed or closed deferred session
before connector initialization. This prevents a consumed legacy loop error
from permitting a new connector under an unusable profiler owner. Disabled
role mode retains the original legacy restart behavior.

Native `--profiler-config` remains the default `super().profile` path when the
role-env session is absent. Enabling both native and role-env configurations is
an explicit configuration error, including on an excluded role rank. This
increment does not change native HTTP router registration: role-env control is
available through worker/executor RPC and status; an HTTP router for role-env
alone is a separate integration concern. Native-only `control_running` does not
claim native F step or timeline parity. A's owner/control migration also remains
with the A runner owner.

## Scope, cost, removal, and validation

Disabled role mode adds no backend imports, profiler objects, threads, events,
or wire messages. Enabled mode adds an instance lock/status state and short
boundary checks; export/start/stop costs stay on the execution owner. Profiling
can itself add latency and is diagnostic use; no throughput claim is made.

The subclass `profile` wrapper documents its source, signature, delegation
exception, differences, and removal condition. Remove this adaptation when
native Worker offers a profiler lifecycle for a connector-driven execution
owner. The sole upstream signature table adds the F `profile` row; no second
signature-testing module was introduced.

`profiler-owner-local-validation-20261110.json` records exact file hashes,
commands, tool-result transcriptions, and the local result: 26 passed, 18
dependency skips. The worker/API tests cover receive-blocked request/status,
one parent/one step, owner-thread finalization, native delegation, configuration
conflict, and failed-export retention. The skips require A3 actual dependencies;
no skipped test is counted as exercised. Independent parent review and actual
dependency regression are recorded separately after the immutable snapshot.
The JSON also records a separately hashed static follow-up for the legacy
shutdown exception filter and its additional worker lifecycle case; it does not
claim that case ran on the dependency-free local host.
The further restart preflight has its own three-file hash/static result in the
same JSON; the actual A3 worker result is recorded in the integration evidence,
separately from the original local snapshot.

## Actual-dependency CPU follow-up

The immutable A3 `a3-profiler-cpu-20261110/profiler-owner-012` run completed
99 cases with no failures/errors/skips: profiler module 26, existing NPU runtime
55, Async FFN runtime 4, and the **14 old** signature rows. The identity, XML,
source check, and summary were read independently for this follow-up. Its
profiler module matches SHA `89bfee66?` and its worker is the earlier
`4f915142?` snapshot. It does not execute the new F worker module.

Separately, `../../afd-hccl-v2-multistream/evidence/a3-pipeline-cpu-20261110/worker-followup-014`
completed 114 cases with no failures/errors/skips: F worker 29, Graph service 60,
and native execution 25. Its identity fixes worker
`6f36918afc2dbd0d0afedc7a908be676f111d9398a4858a34556a3beaf0d2e7e`
and worker mirror
`abae9eb4054374a7df9037774300681e7f83a0aa4e710612f2e3b5cd72910411`;
the XML contains both failed/closed pre-initialization restart negatives. It
closes the old 013 Graph fixture setup errors without overwriting that run.

The newly added `ffn-profile` signature row is present in 014's snapshot but
was not a selected target; it remains a separate targeted follow-up. None of
these counts are added together and called a single full regression. These are
CPU API/ownership tests using installed dependencies, not NPU profiler recording
or complete A/F timeline qualification.

The remaining signature row was then executed in
`../../afd-hccl-v2-multistream/evidence/a3-pipeline-cpu-20261110/binding-guards-015`.
Its identity fixes table
`acd0046014fcea1ab920d207e991d99f2b9f319dff25c9bc388aa40e68616ed2`;
the JUnit contains the passing `ffn-profile` case against locked native
`NPUWorker.profile`. That distinct 134-case CPU run passed without any
failure/error/skip and started no device probe. The reviewer read the identity,
summary, and exact case rather than inferring alignment from source text.
