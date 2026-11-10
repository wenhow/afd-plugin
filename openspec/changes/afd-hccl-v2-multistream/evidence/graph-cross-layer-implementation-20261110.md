# cross_layer_v1 implementation contract

This is the implementer's contract record, not an independent review. It
implements the prior static proposal without changing production model
admission, native sources or the default `stage_serial` wire bytes. Evidence
and final source identities are in
[the CPU index](a3-graph-cross-layer-20261110/README.md) and
[the final snapshot](graph-cross-layer-final-snapshot-20261110.json).

`HcclPlanCapabilities.graph_exchange_order` now accepts explicit
`cross_layer_v1`. HELLO must match it exactly; neither serial nor bulk peers
can downgrade it. Graph/counts v1, U2 capacity and paired native TP1/TP2 remain
required. The negotiated capability and the retained bank/pipeline identify
the session; changing order needs a fresh session/cache/manager. A U1 bank
still uses serial exchange, including a U1 bucket before a U2 bucket.

The new pure-stdlib `HcclGraphSchedule` constructs immutable shared INPUT and
OUTPUT actions outside capture. For three layers its paired order is:

```text
I00 I01 O00 I10 O01 I11 O10 I20 O11 I21 O20 O21
```

INPUT means A sends BF16/original FP32/raw IDs and F receives those packets;
OUTPUT means F sends its fixed stage output and A receives it. Both native
compute streams retain `C00 C01 C10 C11 C20 C21`. Static zero-row stages appear
in the retained geometry but produce no action, compute or unrecorded-event
wait. Positive padding-only stages execute the full static body.

The pipeline exposes this same schedule to both model bodies. A strict common
action cursor advances only after a successful parent frontier record. The
existing independent compute/input/output cursors remain. Only the new order
allows Attention's next-layer compute after its own preceding receive, and
allows its input send before its sibling next compute. Serial/bulk retain
their whole-layer guards. F keeps previous same-stage send -> next receive ->
compute dependencies, and every output send waits its compute completion.

A reuses the native continuation methods and stage-bound shared-Attention
views. Its single parent Engram producer, ordinary event waits, layer-entry
aux timing, residual/post/comb/pre_mix, original FP32 and IDs are retained.
The preceding borrowed reply and continuation stay in the compute-consumer
tuple until consumed; compute_done -> next input send -> later same-stage
reply prevents premature overwrite. Final joins precede parent MTP writes,
HC collapse, aux merge and native ModelState retirement.

F calls complete native MoE once for every positive layer/stage and adds no
TP reduction. It validates/copies each native return to the existing fixed
stage output bank inside the compute scope, before compute_done. This is
necessary even when native consecutive returns alias one workspace. Its
bounded pending map holds stage banks, and each next same-stage receive waits
the preceding send before that output bank is reused.

The whole ordered capture-list preflight requires real U2 compute support
even when the first descriptor is U1. For cross-layer, A additionally requires
hybrid dependencies; F requires receive and cross-layer dependencies. Unused
opposite-role gates do not select the local schedule. Any unsupported peer
makes everyone agree and retire the same empty eager parent before skipping
FULL capture. Graph all-off with serial stays synchronous Graph; explicitly
cross-layer plus all-off cannot silently change to serial within the session.

Failed runs retain current bank/tensors/streams/events for supervision and
reject reentry. There is no graph-body control wire, late eager retry, abort
claim, package change or new mutable global state. The CPU evidence protects
these public protocol/output/config seams. Matching blocking HCCL warmup,
actual multi-group capture/replay, native model safety and useful overlap are
separate validation gates; no physical side receive/send stream is claimed.

The later `cross-layer-tp1-001` session passed the matched blocking warmup and
actual NPUGraph/HCCL capture/replay component gate at the same eight production
file hashes. Both ranks completed four warmups, four captures, twelve replays
and six eager fallbacks across three layers. The original payload/precision,
IDs, address and recurrence oracle remained unchanged. See
[the device assertions](graph-cross-layer-device-validation-20261110.json)
and the index for the full immutable identity and raw results. This is a
controlled-arithmetic TP1 component result; native model state, TP2 device,
physical side-stream communication and performance remain separate gates.
