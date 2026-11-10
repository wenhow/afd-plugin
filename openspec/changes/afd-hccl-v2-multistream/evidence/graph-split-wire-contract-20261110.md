# Next Graph exchange increment: paired bulk inputs on the capture parent

This is a source contract and proposed implementation boundary after C6 delivery
`24c8d51`. It changes no runtime, codec, task status or device state. It separates
an implementable receive/compute DAG from independent physical HCCL streams,
which have not been qualified on this locked backend.

## Actual source boundary

The donor identities and source ranges remain in
[the five-axis audit](graph-five-axis-wire-design-20261110.md). Donor D01
`_graph_parent_transport_active` selects the capture/warmup parent for both
directions even when the configured logical plan names send/receive streams.
D04 still receives, computes and sends each stage before advancing. The existing
five-axis implementation preserves that behavior and its diagnostics honestly.

Current target source bytes read for this proposal:

| File | SHA256 |
| --- | --- |
| `connectors/npu/hccl_graph.py` | `95a9ae01aef550fd0dd5967aa35458455c74c898ff9909316d89003bedf34ab7` |
| `connectors/npu/hccl_graph_stage.py` | `b55e8dce0a72109bfcceaca52163ac2177ca0d0d6697664af64ed3dd4460a00f` |
| `connectors/npu/hccl_graph_pipeline.py` | `2105627f04b4693c44d5fc12c4ffac318cf540f6709fbb86228b60576540c066` |
| `model_executor/models/npu/deepseek_v41_graph.py` | `a3140845755cae90d5f7927596c5a50248d829db15aac8c7d9230ef4d37a92fb` |
| `v1/worker/npu/hccl_v41_ffn_graph.py` | `b82fc1a39979e9d5e206fca565d5f1c83e7b11d5557958d08a74e80df1c096cf` |
| `distributed/hccl_graph.py` | `ed9f425e0fd5a486a7d66adce77a17e57c7dd5ea2103b78a3d53740188804742` |
| `distributed/hccl_plan.py` | `fa4b08b0179d992732d1a5f7243ae635d27c5992281c04ce2abe84eaa6041186` |

`HcclGraphExchange` already exposes separate tensor-only `send_inputs` and
`receive_output`, using the same existing endpoint/group/shape contract. The
stage bank combines them in `attention_exchange`. `DeepseekV41GraphStages` wraps
that combined operation in `pipeline.attention_exchange`. The F body similarly
waits its compute in `ffn_send` before enqueueing the next parent receive.
Consequently the present parent stream serializes F receive1 behind compute0.
Changing a diagnostic value or creating an unused receive stream cannot alter
that dependency.

## Smallest useful paired operation order

Keep both HCCL directions on their actual capture parent. Split the A combined
operation and change both A/F host schedules together:

```text
Attention parent: SEND inputs0, SEND inputs1, RECEIVE output0, RECEIVE output1
FFN parent:       RECEIVE inputs0, RECEIVE inputs1, SEND output0, SEND output1
FFN compute:     wait inputs0 -> compute0; wait inputs1 -> compute1
```

The F host enqueues compute0 after receive0 and then receive1 before waiting
compute0 for send0. Receive1 can therefore proceed while compute0 runs on its
side stream. Actual concurrency remains a device/timeline question, but this
is a different, observable dependency DAG, rather than a relabelled stream.
Blocking warmup is acyclic because Attention never waits output0 before sending
inputs1. A unilateral F prefetch against the current A serial schedule remains
invalid and must be rejected before warmup.

No native batched endpoint is required for this first increment. Individual
existing graph-visible sends/receives can express the new order. Batching alone
would not resolve a mismatched peer order and is not a prerequisite. Changing
global/group-relative peer numbering, tensor headers, payload types, static
buffer addresses, or native TP groups is unnecessary.

The initial implementation should remain paired native TP1/TP2, DP1/PP1/CP1,
non-SP, uniform q1 Graph U2, one peer per role rank, and separate existing stage
groups. U1 has the same one-stage order under either schedule. Globally zero
static stages have no operator/event; positive stages with valid count zero
retain their static communication and the existing Graph masks. BF16, original
FP32, raw IDs, and output all retain separate ownership.

## Concrete plugin seams

In `HcclGraphStageBank`, introduce `attention_send(stage_idx, hidden, fp32, ids)`
and `attention_receive(stage_idx) -> Tensor`. Move the current shape/dtype/device/
contiguity checks, copies and ID padding mask into the former. The latter only
receives into retained output. Keep `attention_exchange` as their serial
composition. Do not bypass validation by having the new scheduler reach into a
private exchange object.

The pipeline needs distinct A send and receive scopes, with one preallocated
event per operation and a static layer/stage traversal. Send waits that stage's
compute and its preceding consumption/reuse frontier. Receive records its own
readiness event; the next Attention compute waits that event for hybrid mode or
the full parent receive frontier for barrier mode. Warmup and formal capture
each own fresh event bundles. Replay reuses the captured device DAG and must not
allocate host tickets, mutate a dynamic pending cursor, query events, send
control packets, or consume a one-shot Python loan.

A model consumer prepares every positive stage's native continuation, sends all
inputs, then receives all outputs. Preserve existing final continuation/MTP/
collapse ownership and single parent Engram producer. F's body becomes two
frontiers per layer: receive+enqueue compute for all positive stages, then send
all outputs after their individual compute events. Keep full native FFN/MoE and
TP reduction order layer-major/stage-minor. No second TP reduction or output
semantic change is added.

Retain both F outputs until the send wave. That raises the lifetime of F native
output activations from the current single-stage send frontier to at most U
outputs; it needs an explicit memory budget and `record_stream` on the parent
copy/send consumer. Existing static bank bytes remain unchanged. Native A
continuation tensors and original FP32/IDs need the same compute/parent fences
as the current body. The final parent join still precedes COMPLETE and reuse.

## Peer agreement and cache identity

A/F rank-local stream flags cannot silently select this wire change. The
smallest robust negotiation is an explicit session capability such as
`graph_exchange_order="stage_serial" | "bulk_inputs_v1"`. Encode only the
non-default value in HELLO so the default serial bytes stay unchanged. Validate
it with graph capability v1 and paired topology, and require exact equality in
negotiation; do not use a minimum-version downgrade. A strict older reader
rejects the unknown extension rather than entering a different tensor order.
HELLO disagreement must be drained and globally rejected before stage groups
or warmup, following the existing control contract.

The same immutable session choice must reach controller, bank/pipeline
construction, A model body and F service. Graph phases use that agreed order;
eager fallback retains its existing accepted eager executor. Do not derive
network order independently from A-only or F-only *effective* selection fields.
The raw configuration may be mapped to the new capability once at caller
construction, but both roles must propose the same result or reject. That
projection is a reviewed implementation decision, not a new environment flag
in this proposal.

An existing shape key need not acquire stream objects or runtime IDs. Its cache
entry must, however, belong to the exact session/capability and retained schedule
that captured it. Current resource checks already compare bank capabilities to
the transport; extend those checks to the agreed order and frozen pipeline
schedule. A policy/order change requires a fresh transport/session, cache and
native manager. Generation increment or cache miss does not authorize
recapturing the same occupied key under another wire order. Include the static
order in capture-manifest agreement/diagnostics if it is not already protected
by the exact session capability and resource validation; do not hide it in
metadata shapes or transaction text.

## Cross-layer and physical-stream limits

Bulk inputs remove the *within-layer* F receive1 dependency on compute0. They do
not implement full cross-layer receive overlap: the parent still sends output1
before receiving the next layer's inputs0. A stronger static diagonal schedule
would have to pair, for example, A receive output0 → send next inputs0 before
receive output1, with F send output0 → receive next inputs0 before send output1.
Both native compute streams must still preserve collective layer/stage order,
and same-stage bank reuse must wait the prior native consumer/send. Such a DAG
needs a separate complete schedule proof, phase agreement and experiment; the
current cross-layer flag must not silently authorize it. Bulk alone cannot mark
the complete physical cross-layer requirement done.

Independent physical send/receive streams during native capture are also not
established by this design. The source-pinned donor deliberately overrides them
to the parent, and only that placement has actual component evidence here.
Multi-stream Graph event support is not proof that the locked HCCL endpoint is
legal on an arbitrary side stream. If task 1.6 specifically requires distinct
physical HCCL streams, first perform a separate endpoint capture legality gate
with explicit fork/join and matched warmup; never loosen the production parent
assertion as an experiment. A backend/API change may be required, but absence of
current evidence is not itself proof that no implementation exists.

## Failure and validation boundary

Before any tensor operation, unavailable/mismatched session capability fails
global negotiation. A pre-prepare graph cache miss still selects the existing
all-peer eager fallback. A failure after warmup/capture/tensor posting is
terminal: retain bank, inputs/outputs, stream/event bundles, model/metadata and
group owners for external supervision. No late fallback, new abort packet,
optimistic COMPLETE or in-process retry is justified.

First CPU seams should observe exact paired operator sequences, all positive
payloads, selected static order, separate send/receive dependencies, masks,
same-stage lifetime, full final join, U1/U2/empty stages, and rejection of a
serial/bulk peer mismatch before communication. Existing serial byte/protocol
tests remain unchanged. Then an isolated A3 paired TP1 component can warm up,
capture and replay the same existing static bank with the actual split order;
check full FP32/IDs/output, stable addresses, valid-count changes, zero stages,
normal STOP and native TP preservation. TP2 and allocator pressure follow.
Only subsequent A/F timelines can establish useful receive/compute overlap.
Native HC/Engram/MoE acceptance remains separate from arithmetic component gates.
