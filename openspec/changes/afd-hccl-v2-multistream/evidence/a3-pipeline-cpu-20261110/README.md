# C6 pipeline component: actual-dependency CPU evidence

The separate
[`graph-fallback-diagnostics-021`](graph-fallback-diagnostics-021/results/summary.json)
retains **212 passed, 2 failed and 26 setup errors**, zero skips. The complete
Graph eager/pre-KV module passed 49 cases, coordinator passed 81 and UBatch
passed 74. Execution passed eight cases; its two failures and 26 setup errors
all reference the retired `create_afd_npu_profiler` or `step_afd_npu_profiler`
fixture patch names after the H14 explicit-session rename. The original failed
run is preserved; the H14 owner will repair and independently rerun only the
complete Execution module. This is not an all-green four-module result.
All ten upstream pins and uploaded sources matched; no device probe started.
Its [JUnit](graph-fallback-diagnostics-021/results/cpu.xml),
[identity](graph-fallback-diagnostics-021/identity.json),
[stdout](graph-fallback-diagnostics-021/results/cpu.stdout.log) and
[independent review](../stage-diagnostics-graph-fallback-review-20261110.md)
retain the exact source and failure scope. The
[021 integrity check](graph-fallback-diagnostics-021-index-check.json)
rehashes eighteen independent CPU attempts and 180 raw artifacts without
combining their test counts.

The later independent
[`diagnostic-boundaries-020`](diagnostic-boundaries-020/results/summary.json)
passed **247 cases**, zero failures, errors or skips: A execution 36, rank
binding 32, UBatch 74, F model entry 17, F serial runner 59 and F worker 29.
It verifies the outer executor preserves a sink exception after real ticket
COMPLETE, rejects configuration sink failures before U1/U2 model producers,
and rejects both orders of incompatible Graph/diagnostic binding. All ten
installed source pins matched; no device probe was started. Its
[JUnit](diagnostic-boundaries-020/results/cpu.xml),
[identity](diagnostic-boundaries-020/identity.json) and all original raw files
remain separate from 018b. The [latest integrity check](diagnostic-boundaries-020-index-check.json)
covers seventeen independent CPU attempts and 170 raw hashes.

The latest independent diagnostic family run is
[`diagnostics-family-018b`](diagnostics-family-018b/results/summary.json):
**512 passed, zero failures, errors or skips** across eleven selected modules.
All ten installed upstream source pins match. It used the existing venv,
including FastAPI 0.123.10, and started no device probe. The complete command,
actual HEAD and frozen working-tree hashes are in its
[identity](diagnostics-family-018b/identity.json); the manifest digest is
`3254497cb19e97cdb842af769aa239ca23ed83fe1a6d5723815480237334dced`.
Later runtime changes are compared separately in the index and are not
qualified by this historical run.

| 018b selected module | Passed |
| --- | ---: |
| connector diagnostic helper | 56 |
| eager ticket owner | 70 |
| A native stage model seam | 51 |
| F eager model consumer seam | 49 |
| rank diagnostic binding | 32 |
| A execution | 33 |
| UBatch runner | 74 |
| F serial runner | 59 |
| native NPU runtime seam | 55 |
| async FFN runtime seam | 4 |
| FFN worker | 29 |

This run verifies immutable actual serial/Graph/eager execution descriptions,
real stream alias reporting at the owner seam, exact rank/transaction/layer
progress ownership, all-rank configuration and host post/loan/reuse/finish
records. It preserves native model/context CPU outputs and the live serial
FFN U1-only synchronization rule. Sink failures retain the exact live owner;
a sink error after COMPLETE preserves its original error without failing a
retired owner. Default-disabled diagnostics add no observable stream operations
in the CPU seam. These checks do not execute device HCCL, native MoE/Engram
operators, graph replay logging, device overlap or performance measurements.

The initial local [`diagnostics-family-018` launch](diagnostics-family-018/local-launch-failure.json)
could not read the existing SSH configuration within the child sandbox. It
created no remote directory, uploaded no snapshot and ran no tests. The
authorized sandbox escalation ran 018b under a new label. Both outcomes remain
separate, and the launch failure is not counted as a pytest failure or pass.

Raw 018b artifacts are [JUnit](diagnostics-family-018b/results/cpu.xml),
[stdout](diagnostics-family-018b/results/cpu.stdout.log),
[stderr](diagnostics-family-018b/results/cpu.stderr.log),
[source checks](diagnostics-family-018b/results/source-check.json),
[versions](diagnostics-family-018b/results/versions.json),
[environment](diagnostics-family-018b/results/effective-env.json),
[delivery](diagnostics-family-018b/delivery.json) and
[remote transcript](diagnostics-family-018b/remote-run.log).
The [index integrity check](diagnostics-family-018b-index-check.json) verifies
the raw hashes of all sixteen independent indexed CPU attempts. It does not
sum their counts into one test result. The
[implementation contract](../stage-diagnostics-family-implementation-20261110.md)
describes the new API and its execution boundaries.

`component-001` completed with **31 passed**, no failures, errors or skips
(`pytest` 1.06 s; JUnit 1.065 s). The existing A3 environment supplied the real
installed dependencies. The test uses CPU bank arithmetic and host event/stream
traces: **no NPU probe was started**. This does not establish device stream/event
execution, HCCL overlap, graph capture/replay, allocator safety, full-model output
or performance.

The frozen candidate is `hccl_graph_pipeline.py` SHA256
`12eab892912906240d116ac064e8bc1b449b7a8a278433547143f6e1041f7bfd`.
The mirrored test and shared fake hashes are recorded in
[the source manifest](component-001/identity.json), with actual HEAD
`a897223f1d3489b588c68ed3cf3d099df4dbf0d4` and manifest digest
`e7b9d01e907f1d9486e53c86d7ce1f4a0bb53502726892ead44af786e2b21462`.
The immutable snapshot contains other Python files; the only selected pytest
module was `tests/unit/connectors/npu/test_hccl_graph_pipeline.py`.

The 31 cases cover CPU outputs for both roles and zero stages, FP32/IDs and wire
order, matching ready/receive/send frontiers and final joins, foreign-stream and
incomplete-phase rejection, distinct event ownership, failure retention,
replacement-frontier release and phase-reuse rejection. These are host contracts
with explicit device boundaries, rather than simulated device acceptance.

The environment remained Python 3.12.9, torch 2.10.0, torch-npu 2.10.0.post4,
vLLM 0.30.0+empty, vLLM-Ascend 0.1.dev1+g7e37e29a7 and FastAPI 0.123.10.
All ten pinned installed upstream source checks matched. No packages or upstream
sources were changed.

Reproduce the immutable upload and CPU selection from the checkout root:

```powershell
.venv/Scripts/python.exe -B openspec/changes/afd-hccl-v2-graph/evidence/a3-control-cpu-20261109/run_a3_graph_cpu.py --label component-001 --evidence-dir openspec/changes/afd-hccl-v2-multistream/evidence/a3-pipeline-cpu-20261110 --targets tests/unit/connectors/npu/test_hccl_graph_pipeline.py
```

Use a fresh label for another attempt. The reused uploader's C6 delta adds only
`--evidence-dir` and an identity output-directory field; CPU execution, source
checking, environment setup and timeouts are unchanged. Its executed SHA256 was
`1496e88ce8b6060b866b561b9cd1954dc1e956908b4bbdb8faa6e305bba2acc1`.
The remote scratch retains its generic historical `c5-control-cpu-20261109-`
prefix; the test result belongs exclusively to this C6 evidence directory.

Raw files are [summary](component-001/results/summary.json),
[JUnit](component-001/results/cpu.xml),
[stdout](component-001/results/cpu.stdout.log),
[stderr](component-001/results/cpu.stderr.log),
[remote invocation](component-001/delivery.json),
[remote transcript](component-001/remote-run.log),
[versions](component-001/results/versions.json),
[effective environment](component-001/results/effective-env.json) and
[source checks](component-001/results/source-check.json).
The 14 warnings are torch's existing `torch.jit.script_method` deprecation.

An additional, separate **source-only** SSH read collected installed torch-npu
stream/event APIs without importing `torch` or `torch_npu`, starting a device
job, or writing remote files. The [raw JSON](torch-npu-stream-api-20261110.json)
contains the full installed sources, hashes and line-numbered AST excerpts;
[the collector](collect_torch_npu_stream_api.py) is reproducible.
`streams.py` implements `Stream.__new__` at lines 24–26 and `Event.__new__`
at 134–136; `Event.record` and `Event.wait` are at 143–151 and 153–161.
`npu/utils.py` defines `StreamContext` at 176–229, `stream` at 232–242 and
`current_stream` at 260–272. The queried `_tensor_methods.py` candidate did not
exist; this read does not validate tensor allocator/`record_stream` runtime
behavior. Source inspection remains separate from device acceptance.

## Later consumer snapshots

The following are separate immutable runs, all with zero failures, errors or
skips and all ten installed upstream pins matching. Their counts must not be
combined into one pytest result. The [index](validation-summary.json) records
each command, module count, source manifest, raw-artifact hashes and comparison
with the later source tree.

| Snapshot | Passed | Selected seams | JUnit seconds |
| --- | --- | --- | --- |
| `consumers-002` | 130 | Stream TLS 10, pipeline 31, actual FFN body 43, FFN service 46 | 3.785 |
| `native-consumers-003` | 240 | A body 42, native graph execution 16/coordinator 60/session 19, runner entry 11, eager bridge 27, eager execution 22, dispatch 23, signatures 14, new policy cases 6 | 4.113 |
| `parent-stream-004` | 122 | Updated A parent 46, native graph execution 16, coordinator 60 | 3.709 |
| `five-axis-005` | 460 | Immutable role/phase selection, five Graph axes, actual A/F bodies, native coordinator/service/execution | 8.367 |

`consumers-002` tests the shared native/vLLM current-stream scope, both side
compute policies in the actual FFN tensor body, unchanged FP32/IDs/output
messages and native context selection. The service tests verify capture-time
pipeline retention until STOP/reset and reject a same-bank/same-policy
replacement through the public cache API. It predates the six added immutable
policy cases; those six were selected explicitly in `native-consumers-003`.

`native-consumers-003` adds actual A continuation/aux/MTP CPU comparisons,
single unsliced parent Engram producer seams, U1/U2 native preparation,
pipeline/cache identity, replay reuse and policy-replacement rejection. It
also checks the unchanged explicit pre-KV and normal native lifecycle seams.
It describes the earlier A source `dcae2bc4...`, before the common parent TLS
follow-up. It does not establish execution of device Attention/Engram/MoE.

`parent-stream-004` verifies A source
`a3140845755cae90d5f7927596c5a50248d829db15aac8c7d9230ef4d37a92fb`.
The common graph parent scopes the actual parent stream and existing vLLM TLS
around producer, embedding, side DAG and final collapse. Both U1/U2 and hybrid
policies start with a foreign cached stream and must restore it after success
or an injected payload failure. Only the affected A/execution/coordinator
modules were rerun. The full original runs remain unchanged in their named
directories; each has `identity.json`, `delivery.json` and `results/`.

The separate [local collection record](../local-cpu-collection-20261110.json)
checks Python 3.11 grammar, Ruff and collection without torch/vLLM installed.
The [settings record](../local-settings-20261110.json) documents 77 local
stdlib parser cases as a delegated tool-output transcription. Neither local
record is hardware evidence. Actual controlled NPU graph tests are recorded
under [their own directory](../a3-pipeline-device-20261110/README.md).

`five-axis-005` freezes the independent Graph settings consumers. Its 460
cases include role/phase projection, U1/U2 in one native capture session,
compute-off Graph replay, logical Attention send/receive choices under the
physical parent override, all three legal FFN receive/cross-layer combinations,
and rejection of changed settings even when the captured pipeline is absent.
The actual A/F CPU bodies preserve their output and BF16/FP32/IDs wire contract.
All ten installed upstream source pins match. This run predates the subsequent
stage-diagnostics integration; its immutable source manifest is authoritative.
It does not qualify physical receive/compute overlap or native device MoE.

## Diagnostic integration follow-up

`diagnostics-006` selected the diagnostic helper and the real FFN eager/Graph,
coordinator and service seams: **282 passed, 3 failed**, no errors or skips
(JUnit 5.949 s). All three failures were new service test fixtures missing the
real bank's public `layer_indices`; the fixture was corrected without changing
production code. The affected complete service module was then rerun as
`diagnostic-service-007`: **60 passed**, no failures, errors or skips
(JUnit 2.725 s). Both immutable attempts remain in the index and raw directories.
These are separate runs, rather than a combined all-green pytest result.

The checks cover capture-time configuration records, configured/effective and
logical/actual stream distinctions, replay retention, host enqueue milestones,
default-disabled diagnostics, and live eager U1-only FFN compute synchronization.
The synchronization callback binds the runner's explicit device. Profiling,
warmup, dummy and Graph paths do not enable that synchronization. Each run
matched all ten upstream pins and started no NPU probe. Diagnostic log entries
describe host enqueue points; they do not measure device overlap or latency.

## Eager ticket owner, consumers and worker follow-up

These are additional independent CPU attempts using actual installed A3
packages. Every attempt matched all ten installed source pins and started no
NPU probe. The index records exact commands, per-module outcomes, complete raw
artifact SHA256s and upstream expected hashes. Its current-tree comparison is
only a later source sample: changed or newly added bytes are not accepted by an
older run. Do not add the following counts together or turn failed attempts
into a combined green result.

| Snapshot | Passed | Failed | Setup errors | Skipped | Per-module observations |
| --- | ---: | ---: | ---: | ---: | --- |
| [eager-attention-008](eager-attention-008/identity.json) | 38 | 0 | 0 | 0 | Staged A model 38 |
| [eager-transport-009](eager-transport-009/identity.json) | 89 | 44 | 0 | 0 | Existing staged transport 84; owner 5 passed / 44 failed |
| [eager-owner-010](eager-owner-010/identity.json) | 143 | 13 | 0 | 0 | Owner 49, execution 24 and ubatch runner 70 passed; session 13 failed |
| [eager-session-011](eager-session-011/identity.json) | 119 | 0 | 0 | 0 | Session 13, execution 25, A runner 11, ubatch runner 70 |
| [eager-ffn-013](eager-ffn-013/identity.json) | 231 | 0 | 2 | 0 | Owner 50, F consumer 43, worker 25 passed / 2 setup errors, F runner 58, NPU runtime 55 |
| [worker-followup-014](worker-followup-014/identity.json) | 114 | 0 | 0 | 0 | Worker 29 including two restart guards, Graph service 60, execution 25 |
| [binding-guards-015](binding-guards-015/identity.json) | 134 | 0 | 0 | 0 | Complete FFN parent module 59, complete ubatch runner module 74, one `ffn-profile` signature case |
| [parent-identity-016](parent-identity-016/identity.json) | 28 | 0 | 0 | 0 | Complete eager execution module, including three direct-parent identity rejection cases |

`eager-transport-009` retained the first real ACCEPT-frontier failures: its
scripted peer omitted the required `world_rank`. The shared fake was corrected;
the real wire validation was preserved. `eager-owner-010` then passed all 49
owner cases. The new session fake in that run lacked `parallel_config.rank`,
so its 13 cases failed before opening a parent. The fake was corrected, and
`eager-session-011` passed its complete 13-case module alongside the updated A
entry seams. These are test-fixture repairs, rather than hidden relaxations of
the production acceptance or rank contracts.

`eager-ffn-013` uses owner source
`b758b9e8cb398cebdf91c8bd36087b060e2d10e7cbfc6636ff0ba385e0cb6e31` as recorded by its immutable manifest and the F consumer source
`5e7183dea3436868aa1e22cb04056ffabb0ca714532b7090b623f0d56d2624ac`.
The F consumer's 43 cases cover complete native-call CPU context and output
seams, independent FP32/IDs, padding-only and empty stages, TP1 and both TP2
role ranks, one COMPLETE, invalid topology/model/stream rejection, sticky live
failure, stale-owner rejection without failing another parent, and actual
vLLM stream TLS restoration around compute and final completion. The native
MoE boundary is controlled CPU arithmetic; no device kernel is simulated as
acceptance. The full run has two **setup errors**, not failed F consumer tests:
old Graph worker fixtures supplied `enforce_eager=True` and Graph NONE, which
real Graph binding correctly rejects. The shared fixture was corrected to
`enforce_eager=False` / `FULL_DECODE_ONLY`. Its complete Graph service module
and worker/execution seams passed in `worker-followup-014`; that follow-up also
checks two new worker restart guards. The original 013 XML, logs and summary
remain unchanged.

`binding-guards-015` is a separate **134 passed** run (JUnit 2.228 s), with
zero failures, setup errors or skips. It closes three binding-identity review
findings: reject an FFN eager session from a different connector wire; reject
an A staged eager run on a foreign wire before native producer work; and reject
an uninitialized, closed or terminal previously warmed A session. The complete
FFN parent and ubatch runner modules were selected, plus only the single
`ffn-profile` upstream-signature row. All ten installed source pins and the
uploaded source manifest matched, and no device probe was started. This does
not replace any earlier immutable attempt or qualify native device execution.

`parent-identity-016` separately passed **28 cases** with zero failures,
errors or skips. The public `execute_hccl_v41_stages` entry itself now rejects
a foreign or stale eager owner before preparing Engram, including callers that
do not enter through `UBatch.run`. Three added negative cases exercise exact
transport/plan identity and active/finished/failed ownership. All ten upstream
pins and uploaded sources matched, and no NPU probe was started. Its source
identity and complete module result are independent of the 015 binding checks.

The separate C7 profiler-owner snapshot is
[profiler-owner-012](../../../afd-hccl-full-parity/evidence/a3-profiler-cpu-20261110/README.md):
**99 passed**, no failures, errors or skips. It is indexed under its own change
rather than counted as another C6 run.

For each label, inspect `results/cpu.xml` and `results/cpu.stdout.log` together
with `identity.json` and `results/summary.json`. The
[index](validation-summary.json) hashes every local raw file, including stderr,
remote invocation/transcript, environment, versions and source checks. All
raw files were rehashed after indexing with no changes. Reproduction uses the
same CPU harness and a **new label** with the exact `targets` list recorded in
the immutable identity; the command in each summary is the actual executed
command. No package installation or upstream modification was part of these
runs. CPU stream/event/Work boundaries, profiler state and native model-call
seams do not establish NPU overlap, native device MoE, profiler C++ thread
semantics, full-model correctness or performance.
