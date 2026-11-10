# C7 profiler ownership: actual-dependency CPU evidence

`profiler-api-025` separately validates the new native endpoint plugin with
**62 passed**, no failures, errors or skips (JUnit 0.254 s): 31 complete HTTP/API
cases and the existing 31 profiler cases. The actual installed dependency run
includes the native endpoint loader's allowlist/task gates. All 15 upstream
pins match, including the five endpoint/config contracts added in
[`profiler-api-source-identity-20261110.json`](../profiler-api-source-identity-20261110.json).
The [immutable identity](profiler-api-025/identity.json) records HEAD
`36f52994217f4caeb4ae20d5b4783892da4b555d`, manifest
`3fd88c1f933395e50d21d801e369db6377ef4255f38ceb2c0916b725918b5d26`, and the
exact source/fake/test hashes. Raw [JUnit](profiler-api-025/results/cpu.xml),
[stdout](profiler-api-025/results/cpu.stdout.log) and
[summary](profiler-api-025/results/summary.json) qualify ASGI validation, fixed
existing-engine RPC requests, pending/applied state and app-local ownership.
They do not start a server, real worker RPC, native model or NPU task. Local
collection had no FastAPI and was skipped; it is not behavior evidence.
The three previously frozen profiler/Attention runner/Attention worker source
hashes remain unchanged. This run does not replace the older attempts below.

`profiler-owner-012` completed with **99 passed**, no failures, setup errors or
skips (JUnit 2.124 s). This independent A3 snapshot used actual installed runtime
packages, matched all ten upstream source pins, and started **no NPU probe**.
No packages or upstream files were changed.

| Selected module | Passed |
| --- | ---: |
| `tests/unit/compat/npu/test_profiler.py` | 26 |
| `tests/unit/v1/worker/test_npu_runtime.py` | 55 |
| `tests/unit/v1/worker/test_async_ffn_runtime.py` | 4 |
| `tests/unit/compat/patches/test_signatures.py` | 14 |

The profiler seams exercise the owner's scheduled start/stop/step lifecycle,
thread ownership checks, failure propagation and runner/worker integration.
Native profiler device calls are explicit CPU seam boundaries. This does not
prove torch-npu's C++ profiler is thread-local or global, qualify on-device
recording, or establish device performance. The separately collected installed
Python profiler source remains
[source-only evidence](../../../afd-hccl-v2-multistream/evidence/torch-npu-profiler-source-20261110.json).

The actual checkout HEAD was `8a6eeb2a81506621f8571784179a46e3f4680d95`; the immutable manifest digest is
`30363caa92f1d143a75fb8ca916dc432d94bdd3a69e90507bf8208e30af72f1b`. The [source identity](profiler-owner-012/identity.json)
contains the complete production, fake and mirrored-test SHA256s and all ten
expected upstream source hashes. The [index](validation-summary.json) records
per-module outcomes, the exact command and every raw artifact hash. Its later
comparison with current source bytes is advisory: an older run does not accept
changed or missing bytes.

Raw evidence: [summary](profiler-owner-012/results/summary.json),
[JUnit](profiler-owner-012/results/cpu.xml),
[stdout](profiler-owner-012/results/cpu.stdout.log),
[stderr](profiler-owner-012/results/cpu.stderr.log),
[delivery](profiler-owner-012/delivery.json),
[remote transcript](profiler-owner-012/remote-run.log),
[versions](profiler-owner-012/results/versions.json),
[environment](profiler-owner-012/results/effective-env.json) and
[source checks](profiler-owner-012/results/source-check.json).
Use a fresh label and the exact target list in the identity when reproducing
with `run_a3_graph_cpu.py`; never overwrite this attempt.

The [C6 index](../../../afd-hccl-v2-multistream/evidence/a3-pipeline-cpu-20261110/README.md)
retains the independently executed Graph/eager consumer attempts, including
44 owner fixture failures in 009, 13 session fixture failures in 010 and two
Graph worker setup errors in 013 with their separate follow-ups. Counts from
those attempts are not added to this 99-case run. Full-model and A5 acceptance
remain separate manual validation work.

## A execute owner and strict shutdown follow-up

`attention-profiler-022` ran seven complete modules with actual installed
dependencies: **299 passed, 12 failed, 8 setup errors**, no skips. Both the
shared fixture and the complete failure history are retained. Passing modules
were profiler 31, legacy NPU A runner 156, NPU runtime 55, native execution 36
and the complete single signature table 18; three cases in the newer A runner
mirror also passed. This closes the preceding execution module's stale
profiler-symbol fixtures without rerunning unrelated modules.

All 20 failed/error cases came from the two newer Attention mirrors. A lower
level C6 builder had started preselecting its own eager owner, while their
shared runner builder creates a replacement control/eager for explicit binder
tests. It failed to reset that owner to unbound. The real binder correctly
rejected this already-bound object; one intended-unbound case instead entered
native execute without its native execution setup. The fix is solely in
`tests/unit/fakes/attention_model_runner_v2.py`
(`ac4ad83cff30608869502c3ac7df4adb90c1d5e57c767cedb62a10f09e93734d`).
Production admission and binder checks were preserved.

`attention-profiler-023` then ran both affected mirrors in full: **23 passed**
(worker 12, A runner 11), no failures/errors/skips. Its three profiler/runner/
worker production hashes are identical to 022. This is a directed closure,
not a new all-green 319-case run. Both runs matched all ten upstream pins,
started no device probe, and changed no installed package or upstream source.

The [022 identity](attention-profiler-022/identity.json) fixes manifest
`9011ef40b51f54e3c4aad620dced6f414ce768748e4499987b90122ca28b3f9a`;
the [023 identity](attention-profiler-023/identity.json) fixes
`92d725d7575f60d8f1ce10d24d25b144ad2704f86d3c794a0fe244295d2c6bbc`.
Each directory retains raw JUnit, stdout/stderr, versions, source checks and
the exact selected command. The JSON index contains each run's outcomes and
raw artifact hashes separately. These CPU checks qualify request/status,
execute-step semantics, thread ownership and STOP/cache/export/native KV/
model/group cleanup order; actual profiler traces remain separate.

## Fixed HTTP management entry

`profiler-api-025` passed **62 cases**, no failures/errors/skips: API 31 and
profiler lifecycle 31. All fifteen installed source pins matched. Its ASGI
and native endpoint-loader CPU seams preserve independent app/executor
binding, strict request validation and requested/applied state. It started
no device probe and does not qualify a deployed native server or NPU trace.
The [immutable identity](profiler-api-025/identity.json) and raw results are
listed separately in the JSON index.

## Headless F local management entry

`headless-profiler-027` passed **91 cases**, no failures/errors/skips: local
control 31, EngineCore 29 and profiler lifecycle 31. All seventeen installed
source pins matched. Actual Unix sockets cover strict commands, partial IO,
connection/size/deadline bounds, nonserializable RPC responses and owned-inode
cleanup. EngineCore cases exercise requests during its existing management
loop and cleanup on normal STOP and service failure. Executor/device work
remains a CPU seam; the attempt started no NPU probe or model and changed no
installed package or upstream source.

The [identity](headless-profiler-027/identity.json) fixes the source snapshot
and exact target list; [summary](headless-profiler-027/results/summary.json),
[JUnit](headless-profiler-027/results/cpu.xml), source checks, versions and
stdout/stderr remain intact. The JSON index records every raw hash. Local
Windows verification separately passed 65 and skipped 26 because AF_UNIX is
unavailable; those skips are not accepted socket behavior. Native full-model
headless management, NPU timeline, pressure and A5 remain unaccepted. See the
[source contract](../headless-profiler-control-contract-20261110.md) and
[A5 guide](../../../../../docs/npu/DSV41_A5_PROFILER_VALIDATION_ZH.md).
