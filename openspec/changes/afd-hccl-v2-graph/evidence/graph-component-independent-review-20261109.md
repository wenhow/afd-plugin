# Static HCCL graph component independent review, 2026-11-09

The reviewer read `hccl_graph.py`, its mirrored CPU tests/fakes, the native NPU
probe, the C5 source-contract report and the installed torch-npu schema/source
transcription. The reviewer did not edit those runtime or test files. Two
storage/lifetime findings were reported to their author and fixed. The final
source has no unresolved static blocker for this explicit paired, static-shape,
synchronous communication component. No production Graph/model guard is opened.

## Findings and verified fixes

1. The initial constructor admitted overlapping payload storage, including
   `output=hidden_states`. It now validates all four contiguous byte intervals
   outside capture and rejects overlap before any payload. Different disjoint
   views of one allocation remain valid. The added output/input alias test
   protects the observable rejection, rather than inspecting private fields.
2. The initial probe cached graphs across cases without retaining each graph's
   external exchange/count/index buffers. Older graphs were not replayed again,
   so this was not an observed wrong-result claim, but their captured addresses
   lacked one explicit lifetime owner. The final probe retains
   `(graph, exchanges, valid_counts, row_indices)` until device synchronization
   and graph reset; only afterwards does it clear those references and destroy
   the caller-owned groups. This matches the intended cached-graph contract.

The initial native run remains useful evidence for its original bytes, not a
substitute for validating these final source changes.

## Coordinates, shape and capture body

`HcclGraphEndpoint.from_process_group` resolves membership, self/peer identity,
group tag and group-relative peer coordinate before capture. Negative/non-int,
self and out-of-group peers are rejected. Calls to `npu_define._send/_recv`
pass `None` for the global rank argument and the explicit relative coordinate,
along with the actual group's rank list/tag. This matches the installed
torch-npu 2.10.0.post4 schema and the converter's c10d canonicalization. A
relative rank is never passed as global `dst/src`.

Every call retains its static shape tuple, including int32 IDs. In particular,
the GE receive converter uses the shape and dtype; the older donor's omitted
shape workaround is not copied on the assumption that optional schema fields
are unused. The installed-source transcription identifies full file hashes and
labels its normalized/paraphrased excerpts. It is source evidence for that
installed wheel, not a lock or an A5 wheel equivalence result.

The capture body contains only the four typed tensor exchanges and controlled
tensor arithmetic/copies. Group lookup/import, buffer construction, shape/layout
validation, input refresh, Gloo agreement/barriers and result checks are outside
capture. Static Python role/stage loops unroll the agreed schedule during
capture; replay does not advance a Python cursor or send a parent packet. It
does not call eager `receive_layer`, `_advance` or `finish_forward`.

The endpoint retains a strong reference to its caller-owned group and never
destroys it. The component owns no parent plan, graph cache, graph selection,
Engram state or TP reduction. Callers must agree the positive row count, stage
order and empty-skip pattern before invoking it. Allocation rejects zero rows;
the probe agrees empty stages on Gloo and skips them on both ranks. A stage
with physical rows and zero valid rows still communicates, with its device
valid-count tensor masking the output.

## Storage and CPU seam

`allocate` enforces an indexed NPU device and allocation outside capture. The
constructor validates fixed shape/dtype, same device, contiguous layout and
non-overlap. Its documented CPU tensor injection is a layout/protocol seam only:
the tests intercept the real operator boundary and do not create a fake NPU
device or claim a CPU tensor can communicate on HCCL. No test-only enable flag
or second constructor abstraction was added.

BF16 inputs, independently retained FP32 router values, int32 IDs and BF16
output have fixed addresses. Input refresh validates every source before
copying, so a validation rejection cannot partly replace a prior replay's
payload. Received output is copied into the existing destination. Device copy
failure/recovery, ID range/zero-padding policy and parent protocol validation
remain caller/session responsibilities; this standalone body does not promise
a recoverable session or broad model support.

## Scope of actual evidence

The reviewer read the original immutable
`a3-static-graph-20261109/native-001` manifest, run summary, CPU XML and both rank
reports. All four original source hashes match that manifest. Its CPU suite
records 22 tests with zero failures/errors/skips. Both native subprocesses exit
0 without a forced kill; the summary reports eight idle boards afterwards.
Each rank captures U1 `(4,)`, U2 `(3,2)` and U2 with an empty stage `(0,2)`, then
replays each three times. The agreed fully empty case performs no capture.
Exact output, original FP32 deltas, IDs and all four persistent addresses are
checked while valid counts change. This is actual `torch.npu.NPUGraph` evidence
for the arithmetic communication component on devices 0 and 2.

That run's process-group ranks are `[0,1]`. The nonzero-global-rank mapping is
supported by the CPU `(10,20)` protocol seam and source inspection; an offset
real HCCL process group was not separately exercised. The probe performs native
graph capture, not Dynamo tracing, npugraph_ex compilation or GE compilation.
Successful native capture/replay cannot qualify either compiler converter or
the donor dynamic-shape workaround. It also does not execute the native V2
manager, a parent count/plan protocol, native MoE/Engram/attention, checkpoint
inference, miss/fallback consensus, Graph/U2 overlap or performance acceptance.

## Final reviewed source and cost

| File | SHA256 |
| --- | --- |
| `afd_plugin/connectors/npu/hccl_graph.py` | `b98e5f0f8c115a2a1f6e5c1ddc25e4f532425eb02649a0d3d1c08602d99b4d37` |
| `tests/unit/connectors/npu/test_hccl_graph.py` | `a03e7f7b8635e683b567a9d40cd0648b9b7c3913f90f508cb00bb3bfb104f165` |
| `tests/unit/fakes/hccl_graph.py` | `7fd357da41dd902933c91c9e8954d691084a256ee493187b3337bf1ea55b7775` |
| `tests/npu/hccl_graph_probe.py` | `691d1bd3ed6e318f1f3d82f109b08b193a15bf78d80dab4f412e1a2610e24b35` |

The final sources parse with Python 3.11 grammar. Their directed CPU and native
graph follow-up must be matched to these exact hashes before appending final
runtime results. The old 001 result must retain its original identity.

Four persistent buffers require `rows * (8 * hidden_size + 4)` bytes per stage,
apart from graph pool, compute temporaries, count/mask/index tensors and native
model storage. Every additional resident graph must retain its full input
bundle. The receive operation includes an in-place copy and the input refresh
copies three source tensors; no overlap or throughput improvement is claimed.
Replace private `npu_define`/c10d entry use when the target wheel exposes a
stable public graph collective API. Native manager callbacks, complete parent
schedule consensus and model capture remain subsequent work.

## Final directed follow-up

The reviewer read the final immutable
`a3-static-graph-20261109/native-002/identity.json`, `results/run-summary.json`,
`results/cpu.xml`, native supervisor summary and both rank JSON reports. All
four final hashes in the table match that manifest; both rank reports identify
the final component hash. CPU records **23 passed**, zero failures/errors/skips,
exit 0. Native subprocesses both exit 0; U1, U2 and U2-empty-stage each capture
and replay three times on both ranks, while fully empty is an agreed skip.
Every per-case payload address remains stable, and the final cached-graph
bundles retain their external storage through reset. The run summary reports
eight idle boards after execution. No timeout or supervisor kill is reported.

This validates the final independent static graph communication component on
the tested A3 pair. It retains all previously listed scope limits: no native
model/manager integration, checkpoint inference, GE/Dynamo compilation, actual
offset process-group ranks, graph-miss consensus or overlap/performance claim.
