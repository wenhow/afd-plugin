# A-side V4.1 graph compute consumer, 2026-11-10

This increment connects the explicit compute pipeline to the actual
`DeepseekV41GraphStages` model driver. CPU contract arithmetic is not native
HC/Attention/Engram numerical or device acceptance. The production model guard
remains closed; no user overlap environment switch is enabled here.

## Source and ownership

Native reference: Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`,
`DeepseekV41Model.forward` at model.py:1561-1651 and native decoder continuation
semantics. Scheduling reference: donor
`4ddd69a391dfb5b3e362c7601399e538a31de702`,
`deepseek_v4.py:1067-1244`. Exact donor blob hashes and event dependencies are
recorded in `source-scheduling-contract-20261110.md`.

`DeepseekV41GraphStages(..., bank, *, pipeline=None)` preserves the synchronous
path when no pipeline is supplied. The composed path requires
`pipeline.bank is bank` before payload or producer work. The class remains a
tensor body: phase reservation, actual counts, graph cache, native model wrapper,
one parent Engram producer and sole ModelState retirement stay with their
existing owners. No capture factory or native execute body is copied here.

The root owns F-side, GraphExecution/coordinator/service policy wiring. This
report describes the A driver and shared pipeline/TLS integration; those other
owner deltas need their own review and snapshot.

Authorship: this is the implementer's contract and self-review record, not an
independent approval. The root independently read the actual A driver and
reported no blocker. The separate state-audit reviewer also checked the frozen
`dcae2bc4` source against the pinned native layer/Engram/HC ownership and found
no remaining hard blocker within this CPU/component scope.

## Actual model schedule

For each layer, all positive stages first enqueue compute on the side stream.
For layer 1 onward, this consumes that stage's preceding received output and
saved `DeepseekV41FFNContinuation` before observing the next native layer entry.
Auxiliary hidden states are observed before Engram injection, preserving the
native order. The ordinary parent mask/lookup readiness events are waited on
the selected compute stream; no child producer or ExternalEvent reset is added.

The host then enqueues every stage's parent-stream bank exchange in unchanged
layer/stage/peer order. Only after the entire layer's wire construction does
the host enqueue next-layer compute. Hybrid mode records a dependency on that
stage's own receive; non-hybrid mode uses the parent's whole-layer frontier.
These host construction rules do not claim that the resulting device timeline
has already overlapped.

The final continuations run per stage and all tail events join the parent
before auxiliary concatenation, parent MTP-buffer writes, delayed HC collapse
and final normalization. Static zero-row stages execute no HC/Attention kernel,
wire or event wait, but retain empty auxiliary entries for every observed layer
so zero-first and zero-last splits preserve parent output coordinates. Positive
padded/zero-valid stages keep the agreed fixed shape.

Native child forward contexts and the independent shared-Attention row views
remain bound while that child's model work is enqueued. They restore on error.
Bank exchanges have the corresponding child context but remain on the parent
stream. Parent producer scope cleanup remains outside the driver and executes
on failure as before.

## Cross-stream allocation and cached stream ownership

The compute input tuple explicitly includes positions, IDs, mask, the lookup
consumed at this layer, optional embedding inputs, hidden state/pre_mix, pending
received output and continuation residual/post/comb/pre_mix. Bank copies/send
receive their independent BF16, original FP32 and IDs tuple after the compute
event. Returned hidden/pre_mix and every auxiliary output are recorded on the
parent at the final join. The pipeline keeps only each stage's latest declared
frontier rather than retaining all layers' temporary allocations.

Both side-compute contexts now use
`afd_plugin.compat.npu.stream.use_npu_current_stream`. The helper scopes the
actual NPU context and the existing vLLM stream TLS together, then restores the
precise previous cache value or absent state. The driver does not add a thread
registry, worker thread, global stream, producer or retirement call.

`HcclGraphPipelinePolicy(hybrid=True)` is a frozen, slots-based policy with a
strict bool field. `None` means synchronous execution. A/F owners must hold one
policy for the session/cache/manager lifetime; changing policy requires fresh
owners because the physical key does not encode the DAG. A generation miss
does not authorize recapturing an existing native graph.

## Frozen candidate and validation

| Path | SHA256 |
| --- | --- |
| `afd_plugin/model_executor/models/npu/deepseek_v41_graph.py` | `dcae2bc40118426a515fedfc002b121ad2e4764acd0b150e170212d9aaff63f7` |
| `afd_plugin/connectors/npu/hccl_graph_pipeline.py` | `6473cc7ad538ddeaf45a5a9233eef7d5bae79b1cb1e6c95b883953ef05db73cd` |
| `tests/unit/fakes/deepseek_v41_graph.py` | `4d9db7be88c96535ae0ab03798518b0b00a817b54c4888e45913e686d5a92659` |
| `tests/unit/fakes/hccl_graph_pipeline.py` | `6ea974b8922ce5ebdad953fcc0e240204d68551668e9fee4c51f3188ec40f718` |
| `tests/unit/model_executor/models/npu/test_deepseek_v41_graph.py` | `f4f224d468c4ae78dff2442e7b4ef1e324dc6ccaa48bcb95518cb1a4bdcf0bd6` |
| `tests/unit/connectors/npu/test_hccl_graph_pipeline.py` | `ad68f6c8001479dd0847cddfebbbb77fc28f6899db6500f3829fa404b5091ed5` |

The TLS helper `compat/npu/stream.py` is independently owned and reviewed;
its frozen source is `6a881fa579e9c775f3b7142c7c456936e3d05c2ea37f9a87f3bcc386e7656771`.
Shared `fakes/npu_stream.py` source is
`8c8d2568aa6c562012e760e67474d2bd919d8aedc14a5df4d6e21058f111edb4`.

Ruff and Python compilation passed on the six files above. Actual installed
dependency CPU validation is recorded in two immutable snapshots:

| Snapshot | Result and relevant scope | Manifest SHA256 |
| --- | --- | --- |
| `a3-pipeline-cpu-20261110/consumers-002` | 130 passed: TLS 10, existing pipeline 31, F graph 43, service 46 | `90a6099351021b0b4ba5a1cd8bc1ecaab610d09f44c2da3c32c0c7b5455760f8` |
| `a3-pipeline-cpu-20261110/native-consumers-003` | 240 passed: A graph 42, GraphExecution 16, coordinator 60, session 19, main runner 11, eager bridge 27, C4 execution 22, dispatch 23, signatures 14, additional policy 6 | `362b7dd86680baabb90fbb66e383da391fb1b01b1ca0cdfe1a29755946e7e266` |

Both summary files report return code 0, source-files match and no device
probe. The second snapshot's manifest matches all six frozen files above.
The first snapshot contains the earlier A driver/test bytes and pre-policy
test module; it validates its selected TLS/pipeline/F/service targets only.
Together the snapshots cover the existing 31 pipeline cases and six new policy
cases without claiming that all 370 cases ran in one command. The old
component-001 result of 31 passes names an earlier pipeline implementation.

New CPU seams compare the actual native forward/continuation bodies over
explicit CPU arithmetic with the graph driver for U1/U2, both hybrid settings,
Engram on/off and zero-first/zero-last splits. They check HC state/output,
auxiliary/MTP coordinates, whole-layer wire order, stream-query consistency,
one unsliced producer, final join and error restoration through observable
outputs/protocol traces. They do not simulate NPU kernels or replay a CPU handle
as graph evidence. Device allocator safety, graph capture/replay and useful
overlap still require an actual component probe and later full-model evidence.

Cost: one optional pipeline object per native factory phase, its retained
compute stream/events and bounded per-stage continuations. Extra host tuple
construction and stream recording replace immediate continuation consumption;
no per-layer device synchronization or new wire bank is introduced. Native
graph resource bundles must retain the captured pipeline, and replay must not
allocate a fresh unused pipeline on every request.

Removal: replace the composed driver scheduling with native stage-aware
remote-expert/stream hooks when they preserve V4.1 HC/pre_mix, parent producer,
metadata and final joins. Upstream checkouts remain read-only.

## Parent stream cache audit, before any additional patch

Pinned vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`,
`gpu/cudagraph_utils.py:390-395`, selects the microbatch capture stream for U2.
Its capture body at lines 472-475 enters `torch.cuda.graph` on that stream.
Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`,
`worker/v2/utils.py:24-42`, aliases CUDA stream/set-stream functions directly
to NPU functions. This bypasses the cached setter in vLLM
`utils/torch_utils.py:792-801`; the lazy public query at lines 814-848 then
returns an already populated cache without verifying the actual NPU stream.
Native `ModelWithContext.forward`, Ascend
`worker/v2/aclgraph_utils.py:318-342`, does not refresh this cache.

Consequently, a side-compute context restoring the old TLS value cannot prove
that the surrounding parent TLS equals its graph capture stream. The selected
V4.1/Engram event producers explicitly use `torch.npu.current_stream` (native
model.py:1204,1269,1282,1335,1365,1415,1505 and
engram/embedding.py:360). The selected embedding/default TP communicator and
HC-collapse path do not directly query the vLLM cached stream. This audit
therefore identifies an owner invariant to close, not an observed numerical
failure in those native operations.

Any parent synchronization should cover U1 and U2. Wrapping only
`HcclV41GraphExecution.forward_parent` misses the U1 sidecar, which calls the
same retained `DeepseekV41GraphParent` directly. A common parent tensor-body
entry can scope the actual parent stream and existing TLS around the unsliced
Engram producer, embeddings, pipeline, and final collapse while keeping native
`ModelWithContext` device-metadata begin/finish and ModelState retirement with
their owners. This is a read-only recommendation; the hashes and CPU results
above describe the implementation before such a follow-up patch.

## Follow-up: common parent stream scope

The follow-up implements that recommendation at the common
`DeepseekV41GraphParent.__call__` entry. Only a non-None pipeline enters
`use_npu_current_stream(torch.npu.current_stream())`; the existing producer and
tensor body move together into `_run_parent`. U1 and U2 therefore cover the
same unsliced producer, initial embedding, side-compute DAG, parent merge and
final collapse. The None branch directly invokes the unchanged synchronous
body. Native wrapper device-metadata begin/finish and ModelState retirement
remain outside this composition boundary. The stage class description now
states the actual parent exchange and fenced side-continuation ownership.

| Follow-up path | SHA256 |
| --- | --- |
| `afd_plugin/model_executor/models/npu/deepseek_v41_graph.py` | `a3140845755cae90d5f7927596c5a50248d829db15aac8c7d9230ef4d37a92fb` |
| `tests/unit/fakes/deepseek_v41_graph.py` | `3d3572338082a250e581ff22a5ce5b3044ba98e6885f3018e846052d1f92bd43` |
| `tests/unit/model_executor/models/npu/test_deepseek_v41_graph.py` | `7fce12a187a562a03ceceacd3cda59679e8715295bc6942f15d53bfba1e6ca6c` |

The CPU seam uses the existing shared stream/TLS fixture with an explicitly
foreign populated cache. Producer and parent-output operator boundaries query
the actual vLLM public API and actual selected host stream; both must observe
the parent. Normal return and injected wire failure must restore the foreign
cached stream and original native stream/context. Both hybrid policies and U1
and U2 layouts are covered. The fake observes those public boundaries; it does
not model NPU event execution or graph capture.

Ruff check/format and Python compilation passed for these three files.
`a3-pipeline-cpu-20261110/parent-stream-004` then passed 122 actual installed
dependency CPU cases: A graph 46, GraphExecution 16 and coordinator 60, with
zero failures/errors/skips in 3.71 seconds. Its summary reports return code 0,
matching source files and no device probe. The root and separate state-audit
reviewer read this common parent scope and reported no remaining blocker at
the stated static boundary. `native-consumers-003` continues to name its
earlier immutable bytes. The helper, pipeline component and physical graph
key have no changes in this follow-up. Native NPU stream/allocator behavior,
model graph acceptance and overlap performance remain separate evidence.

The follow-up snapshot manifest SHA256 is
`f75afc44b8d10d970e88856201c63527d1a3200609945dd0c3481018778d0f51`;
its summary and JUnit XML remain under that immutable `parent-stream-004`
directory.
