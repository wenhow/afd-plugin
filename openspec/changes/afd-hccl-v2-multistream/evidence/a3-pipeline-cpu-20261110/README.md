# C6 pipeline component: actual-dependency CPU evidence

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
