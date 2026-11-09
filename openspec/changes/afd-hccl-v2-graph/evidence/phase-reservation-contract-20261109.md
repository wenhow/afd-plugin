# C5 graph phase reservation and early mode agreement (2026-11-09)

Status: implemented standalone software contract, pending independent review and
actual-dependency transport regression. This record does not enable the model
guard, production graph selection, GE tracing, or V4.1 model capture. The existing
static tensor exchange acceptance remains independently scoped in
[native-002](a3-static-graph-20261109/native-002/README.md).

## Source and owner

Native source remains read-only at vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607` and vLLM-Ascend
`7e37e29a7803d6897a7113088e2b771f83205bc0`. The exact native pre-prepare
seam, real batch ordering, FULL descriptor geometry, U2 cache lookup, and
Engram timing are established in the separate
[pre-prepare review](pre-prepare-dispatch-contract-review-20261109.md).

The native selected gather runs once and supplies its actual returned CPU batch
state. A graph controller resolves a proposed descriptor from that state before
native input/attention/Engram preparation. The transport owns only bounded wire
records, reservation identity, all-peer agreement, and parent retirement. The
native manager/cache owner owns graph availability, model identity, storage
generation, actual launch, and retained input/event/buffer lifetime.

The cache is bound to one transport session and one model instance. Its key is
not reusable across transports or models. Negotiated hidden/vocabulary sizes,
target layers, native TP topology, payload dtypes and layer-major schedule are
fixed by that session; they need not be repeated in each static key. Local
storage generations are checked by the local cache and are not compared as
integers across peers. Physical addresses never cross the control wire.

## Public records

New CPU-only records live in `afd_plugin/distributed/hccl_graph.py`:

| Record | Fields |
| --- | --- |
| `HcclGraphStage` | `request_start`, `request_stop`, `token_start`, `token_stop` |
| `HcclGraphKey` | `num_tokens`, `num_reqs`, `uniform_token_count`, `max_query_len`, `num_active_loras`, `num_ubatches`, `stage_layouts: tuple[HcclGraphStage, ...]`, `metadata_shapes: tuple[tuple[int, ...], ...] = ()` |
| `HcclGraphCandidate` | `phase`, `mode`, `key: HcclGraphKey | None`, `eager_padded_tokens: tuple[int, ...]` |
| `HcclGraphReport` | `availability` |
| `HcclGraphDecision` | `transaction_id`, `phase`, `mode`, `key`, `padded_tokens`, `reason` |

The initial key deliberately admits only uniform q=1 decode, no active LoRA,
and descriptor `num_tokens == num_reqs`, with U1/U2. Token bounds continuously
cover the padded parent; zero-length stages are retained. Native request slices
may overlap at a token split and therefore are not forced into disjoint ranges.
`metadata_shapes` is an ordered projection supplied by the native owner; the
owner must use the same defined ordering for capture and replay. It carries
shape information, not metadata tensor values.

`key.padded_tokens` derives per-stage token extents. Its canonical SHA256 digest
covers every static field, including request bounds and zero-stage placement.
It excludes runtime valid counts, history, token IDs, transaction identity and
local generation. The realR=4/paddedR=8/U2 native trailing request slice `3:8`
therefore differs from dummy capture's `4:8` despite identical token rows.

`phase` is one of `eager/warmup/capture/replay`; `mode` retains the existing
`target/dummy/warmup/profile` purpose. A native capture commonly has
`phase=capture, mode=dummy`. Profile remains eager. Only eager omits the key.
Whole-empty parents use eager plus actual empty counts; every peer still takes
the reservation and actual-count rounds.

`availability` is `ready/missing/unsupported`. Ready means a completely captured,
device-synchronized, live entry in the current local generation. Warmup alone,
an allocated graph object, or an in-progress capture does not mean ready.
Replay requires every A and F report ready. Any miss selects eager with the
independently previewed `eager_padded_tokens`; reason is `cache_miss` or
`unsupported`. Warmup and capture require every report missing. An unsupported
peer or already-ready duplicate key rejects that phase before preparation;
there is no unilateral skip, recapture, or capture/replay mixture.

## Capability and exact wire fields

`HcclPlanCapabilities.graph_execution_version` is strict integer 0 or 1. Zero
omits the HELLO field and retains pre-extension C4 serialization. One requires
`local_counts_version=1`, which already enforces
`attention_size == ffn_size == native_tp_size`. This is the current DP1 paired
TP scope. All peers must advertise exactly the same graph version; there is no
minimum downgrade. An old strict decoder rejects the new HELLO field before
data-group execution. The separate old v2 protocol is untouched.

All envelopes retain version 3, current payload format and schedule. Each new
message has exactly these fields in addition to that common envelope:

| Kind | Required fields |
| --- | --- |
| `GRAPH_QUERY` | `transaction_id`, `graph_candidate` |
| `GRAPH_REPORT` | `transaction_id`, `world_rank`, `graph_digest`, `graph_report` |
| `GRAPH_DECISION` | `graph_decision` |
| `GRAPH_ACK` | `transaction_id`, `world_rank`, `graph_digest` |

Candidate/decision serialize their named record fields. A key is an object with
all named key fields; its stage layouts are objects with the four bounds.
Tuples serialize to arrays. `graph_report` contains only `availability`.
QUERY reports bind to the candidate digest; ACKs bind to the decision digest.
Unknown/duplicate fields, non-finite values, booleans used as integers, invalid
digests, invalid bounds and oversized frames retain strict rejection.

## Solicited reservation and state machine

The public runtime APIs are:

```python
reserve_attention_forward(
    resolve: Callable[[HcclGraphCandidate], HcclGraphReport],
    *,
    candidate: HcclGraphCandidate | None = None,
    transaction_id: str | None = None,
) -> HcclGraphDecision | None

receive_graph_reservation(
    resolve: Callable[[HcclGraphCandidate], HcclGraphReport],
) -> HcclGraphDecision | None

complete_graph_forward(key: HcclGraphKey) -> None
```

All A ranks call `reserve_attention_forward`; only A0 supplies candidate/ID.
F calls `receive_graph_reservation`. Non-A0 A and F receive QUERY or idle STOP
before sending any report. The resolver checks actual local geometry and cache
generation without native input preparation. An A local `manager.dispatch`
result of NONE can simply mean its own cache is missing. The resolver previews
the descriptor proposed by QUERY against its actual native batch; it must not
treat that local cache miss as an inconsistent native descriptor. Actual
incompatible batch/fallback geometry raises a report-round error.

The ordered protocol is:

1. Idle A0 validates the candidate and its own resolver report, reserves the
   transaction ID, then sends QUERY to every other A/F.
2. Every requested peer validates the candidate, resolves it locally and sends
   REPORT. A0 drains all complete REPORT frames, validating actual sender rank,
   transaction ID and candidate digest.
3. A0 computes the deterministic decision and broadcasts DECISION.
4. Every peer validates that decision against its query and own availability,
   sends ACK, then waits. A0 drains every complete ACK and verifies the selected
   decision digest.
5. A0 broadcasts a global ACK with A0's world rank. Only after that ACK may an A
   return the selected native descriptor and begin native input preparation.

The final global ACK cannot be dropped: otherwise one peer could begin FULL
preparation while another still rejects its decision. Reservation has its own
transaction state, separate from an accepted parent and actual-count state.
The scope has one owning host thread; concurrent entry on the same transport is
not supported.

After global ACK, A performs unchanged native preparation and invokes existing
`submit_forward(actual_local, transaction_id=reserved_id)`; A0 must reuse the
reserved ID. Other A still wait for COUNTS_REQUEST before reporting. F waits in
`receive_forward()`. Real valid counts remain runtime values and must still be
collected from every A; paired TP agreement remains validated by C4 layout.
Actual mode and padded stage extents must match the selected reservation.
There is no fallback or second preparation at this late boundary.

The parent adds `graph_execution: HcclGraphDecision` only for graph capability
1. Its ID/mode/rows must match the full actual plan, and transport compares it
against its accepted reservation. Consequently the existing plan digest now
also binds phase, key, selected fallback, and reservation ID. FORWARD/all-peer
ACCEPT/global ACCEPT and COMPLETE remain unchanged. Capability 0 omits this
nested field entirely.

## Launch, completion, STOP and failure

For eager decisions the existing per-layer C4 APIs and native model path run.
For warmup/capture/replay the four eager data APIs reject the graph phase, so
the cursor cannot become partially eager before a whole graph is committed.
The tensor-only graph body uses static `HcclGraphExchange` operations and never
calls control JSON, `.item()`, allocations, host schedule iteration or COMPLETE.

After successful actual launch, the graph owner invokes
`complete_graph_forward(accepted_key)` outside capture. It requires a non-eager
accepted decision and exact static key, retires the whole local validated
schedule atomically, then uses existing device synchronize and COMPLETE/digest
handshake. This API cannot prove that a caller really launched a graph; its
caller/manager probe must prove device execution and correctness independently.
It is not a replacement for native Engram or model retirement.

Idle STOP remains the existing STOP/ACK/global STOP_ACK handshake. A0
`stop_attention()` broadcasts; other A can wait there, and F can receive it
through `receive_graph_reservation()`. Receiving STOP there returns None and
sends no availability report. STOP is rejected while any reservation, count
round or parent remains live, including F between reservation and FORWARD.
There is no single-sided stop after reservation acceptance.

A complete bad REPORT is drained with the other requested reports before
global ERROR. A bad DECISION is answered ERROR, and A0 drains all decision ACKs
before global ERROR. A complete bad ACK is likewise drained before ERROR.
Backend I/O errors or invalid frame lengths mark terminal and require external
supervision; the code does not claim a coordinated network abort. Native
preparation failure between rounds, capture failure and device failure also
require terminal supervision with resources retained until safe teardown.
No asynchronous CANCEL or late eager recovery is introduced.

## Actual consumer, cost and remaining validation

The immediate consumer under development is the controlled native manager and
F graph-cache probe, using the real `CudaGraphManager` ctor/dispatch/capture
with a small backend subclass and real static exchange/arithmetic. It must
exercise warmup/capture with distinct transaction IDs, fresh callbacks, replay
that bypasses the Python body, missing F cache => early global eager, actual
counts binding, and idle STOP. This proves a manager/transport component; it
does not prove full V4.1 attention, HC, MoE weights or Engram capture. The native
preview helper is a separate owner and projects its concrete geometry to these
CPU records without importing native modules into the codec.

Native full-model integration must finish each prior capture phase before a
fresh native factory writes buffers, reserve before native preparation, retain
one parent Engram producer, preserve Ascend's finish_execution owner and use an
explicit replay hook because FULL bypasses the eager model/provider. Current
C4 synchronous stage model execution cannot simply be put inside a graph.

The additional agreement uses `5 * (world_size - 1)` bounded logical control
frames per forward before the existing actual-count round. With current
length/body framing that is twice as many point-to-point control operations.
TP1 adds five logical frames; TP2 adds fifteen. Key hashing/preview is host work
and introduces no device readback. These costs are accepted for initial
correctness and have not been benchmarked. They can be reduced only after the
native preview and cache ownership contracts permit safely combining rounds.

Removal/consolidation requires a native pre-input-preparation dispatch hook
and a supported model/manager graph lifecycle owner carrying the all-peer
decision. The transport protocol remains an AFD responsibility until an
upstream equivalent provides the same role and failure contracts.

## Initial frozen bytes and validation boundary

| File | SHA256 |
| --- | --- |
| `afd_plugin/distributed/hccl_graph.py` | `111e6f5ca34280ec3e1339f8b57e8d42dbc17edca11099b1e06ef0d1de50de3b` |
| `afd_plugin/distributed/hccl_plan.py` | `fa4b08b0179d992732d1a5f7243ae635d27c5992281c04ce2abe84eaa6041186` |
| `afd_plugin/connectors/npu/hccl_staged_transport.py` | `f728121cef806f682ca0d2e92bbfd8bf4966a7fec8c9e5d6f87842f1dd1d7170` |
| `tests/unit/distributed/test_hccl_graph.py` | `f01146ea1c1671c660a211129c9155c08ce7860087e19e8c6798469fbe7015d5` |
| `tests/unit/distributed/test_hccl_plan.py` | `6fc41329a8de169c9c3120ada94b820d19beb47358e81660ad810c08a09a170f` |
| `tests/unit/connectors/npu/test_hccl_staged_transport.py` | `c02ec7c387abe8afa4384bee89410bec5aa6898fc462a6dc710bffc67cbd898b` |
| `tests/unit/fakes/hccl_graph_control.py` | `c8f166c5fdfa5fe8ec12c4373873d7441945179bb5f1d6012df47764b2422587` |

Local Python CPU run: `175 passed` across schema and plan tests, including all
116 previous plan cases. Ruff check/format and diff whitespace checks passed.
The local host lacks torch, so transport unit collection is gated; its new and
old runtime seam tests await the A3 actual-dependency run. No NPU or native
manager acceptance is asserted for this increment at this point. Independent
review and actual probe evidence must be appended separately, with final byte
hashes if further fixes are needed.

### Actual-dependency CPU follow-up and independent review finding

The immutable A3 [control-001](a3-control-cpu-20261109/control-001/identity.json)
snapshot ran the five schema/plan/transport/cache/geometry modules with real
installed dependencies: **317 passed, zero failures/errors/skips**. XML records
40 schema, 135 plan, 82 transport, 14 cache and 46 geometry cases. This covers
the initial frozen bytes listed above; its working-tree identity records
`head_commit=45356b2ff0f81e1d3fcf9b660026bdc904738a19`. It is a CPU result,
without device capture or native model execution.

Independent review by the native-geometry owner then found that A1's existing
`submit_forward` STOP branch still acknowledged STOP after a graph reservation
had been accepted. F's equivalent receive branch already rejected that order.
A0's normal STOP API could not emit it, so the positive path was unaffected,
but an incorrect coordinator message could make A1 leave a live reservation.
This has been fixed: A1 rejects that STOP without ACK and becomes terminal.
The same review also moved COUNTS_REQUEST/reservation-ID comparison before A1
sends its actual local report. Two observable protocol negatives were added.

Only these two initial frozen files changed for that finding:

| File | Final SHA256 after independent review |
| --- | --- |
| `afd_plugin/connectors/npu/hccl_staged_transport.py` | `a8ce468558a82b577e2fad8119030aa8dea33008bc621618b50d8c717ccc9dd3` |
| `tests/unit/connectors/npu/test_hccl_staged_transport.py` | `3086e232c8d234bfdfcb54d90fa24c42d07c396c37f42be6b1adbec737fb373c` |

The other five listed files are unchanged. The 317-case result is retained for
its exact initial snapshot and does not silently qualify the two new cases;
the final actual-dependency follow-up is recorded separately when available.
