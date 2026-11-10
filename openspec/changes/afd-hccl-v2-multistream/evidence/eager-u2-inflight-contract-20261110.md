# C6 eager/U2 tickets and in-flight frontiers

This is a read-only design. No transport, model, worker, tests, codec, flags or
admission guards are changed, and no device experiment is run. The goal is a
bounded eager consumer of the existing accepted parent plan, not reuse of a
Graph one-shot body or a claim that creating streams provides overlap.

## Exact source boundary

Donor Git objects were read at
`4ddd69a391dfb5b3e362c7601399e538a31de702`. Source hashes refer to original blob
bytes; donor/upstream checkout files remain read-only.

| Source | SHA256 | Relevant owner / rows |
| --- | --- | --- |
| Donor `connectors/npu/p2p_hccl.py` | `594ac53c8c8008ba05da4d12db1c38511e94fdcb8e13340322a0587e940df0ef` | Eager eligibility 595-623; compute/send/recv events, pending transfer and deferred consumer 985-1099; streamed F helpers 1241-1317; real eager dist.send/recv 1730-1778. |
| Donor `model_executor/models/deepseek_v4.py` | `d513a463d159a0a2c9dae18d566805d1f30a8bb9504909c29c7f1d04bb920d3b` | Eager layer-major delayed continuation 1004-1065. This V4 tuple is not the target V4.1 HC contract. |
| Donor `v1/worker/npu/ffn_model_runner.py` | `f95532370a9a5157b16817c5879d78af7405c321d5739d975cd6e2785fe54ccb` | Receive/compute/send streams 168-193; eager selection 918-950; stage scheduling 1002-1122; final sends join 1196-1218. |
| Target `connectors/npu/hccl_staged_transport.py` | `9ae40a90dab184fc4f1d17f934520318c2bd50eea75b5d082ed284e9a4e79128` | Single pending/cursor 674-682,806-829,853-997; synchronization + COMPLETE 1001-1027; idle STOP 1053-1102. |
| Target `model_executor/models/npu/deepseek_v41_staged.py` | `62d47c8686f4257047a5290803f88231bdcc83c1ef526a314df60d7a53b12322` | Parent-derived stages, IDs, native aux/Engram/HC, blocking exchange and merge 142-265. |
| Target `v1/worker/npu/hccl_v41_execution.py` | `a5fa30c78f3bb6696662f2c708bd6a9e7c71f150c81d43e293735ca550b1f200` | One parent producer and native owner 214-294. |
| Target `v1/worker/npu/hccl_v41_ffn_runner.py` | `9628ce7861526c503d0aeebe6254f3a5c37147bb7335e811b7d36a3fc0d90eec` | Real eager native F context, complete MoE, unchanged TP and parent COMPLETE. |
| Target `v1/worker/npu/hccl_v41_ubatch_runner.py` | `2e0940e9c3a07c43d0b4a942a8dca19c6f217dbff019a7832a53f12ae65663c5` | Native preparation and sole native finish_execution remain outside the tensor stage engine. |

The separate `overlap-switch-contract-20261110.md` records exact settings.
`graph-five-axis-wire-design-20261110.md` owns Graph logical-plan parity and
the optional stronger bulk-input Graph schedule. This proposal concerns eager
in-flight ownership only. Target HEAD was `e8557317dd66efd8971566c758456ad2308c6e00`
when the file hashes above were collected; working-tree hashes are authoritative.

## Why the present cursor cannot provide donor-style eager overlap

Current A sends BF16, original FP32 and raw int32 IDs for the current step,
retains one `_pending_inputs`, receives all output chunks, then advances.
Current F receives all three tensors, computes native MoE, sends output and
advances. `_require_step` rejects a second layer/stage coordinate before that
advance. The A model completes HC immediately after each receive. Putting
these same blocking calls in a stream context does not create two in-flight
stages or a deferred HC consumer.

Donor eager instead distinguishes send posted, receive posted and receive
consumed: the receive dependency survives the connector call until the next
layer of that stage needs it. Final continuations wait the remaining receives.
F uses independent receive/compute/send streams and same-stage previous-send
dependencies. The donor source establishes this intent; its use of ordinary
`dist.send/recv` in side contexts is not independent proof of host nonblocking
progress for the installed HCCL backend.

There are two separate opportunities:

| Increment | What can overlap | What remains serialized |
| --- | --- | --- |
| Conservative A compute fork, retaining the C4 cursor | Enqueue both stages' preparations on one compute stream; parent waits only S0's compute before its current exchange, potentially allowing S1 compute while S0 communicates. | A/F wire remains one completed step at a time; F cannot prepost S1 receive while S0 compute/send is pending. ID `.item()` checks or backend host waits may reduce this opportunity, so it still needs a timeline. |
| Eager tickets with ordered nonblocking posting | A posts S0's send/receive then S1's send/receive before consuming S0. F posts S0 receive, enqueues S0 compute/send, then posts S1 receive without waiting for S0's device send to finish. A next-layer S0 can consume S0 while S1 remains in flight. | Native TP compute stays on one compute stream in layer/stage order. Each stage buffer cannot be reused before its own prior consumers finish. Backend legality/progress remains a required gate. |

The first increment may be worthwhile but is not the full eager-overlap flag.
The second needs a real posting primitive whose CPU call can return with work
pending. Wrapping a possibly host-blocking `dist.recv` in a Python context
manager or calling an event `record()` after it is not a substitute.

## Smallest transport-owned lifecycle API

Keep the accepted parent, peer layout, tensor dtypes and per-stage process
groups. Add a separate eager parent owner through a public transport entry,
for example `transport.open_eager_pipeline(streams) -> HcclEagerParent`.
It is exclusive with legacy eager methods and non-eager Graph execution for
that accepted plan. No caller may mutate the transport's private cursor to
simulate completion. The serial C4 surface remains the default.

The initial owner admits actual U2, eager target mode, DP1/PP1/CP1/non-SP,
paired `attention_size == ffn_size == native_tp_size` with native TP1/TP2 and
actual equal TP counts. U1/profile/warmup and unequal/fanout layouts remain
on the existing serial engine until separately designed and validated.
Accepted Graph-capable decisions with phase eager may use it only through
explicit eager eligibility; the donor Graph recipe forces eager overlap off.

Suggested public methods, all outside Graph capture:

```python
parent.post_attention(layer_idx, stage_idx, bf16, original_fp32, raw_ids,
                      *, producer_stream) -> HcclAttentionTicket
with parent.attention_output(ticket, *, consumer_stream) as output:
    # Native complete_remote_ffn plus this stage's next required consumer.

ticket = parent.post_ffn_receive(layer_idx, stage_idx)
with parent.ffn_inputs(ticket, *, consumer_stream) as inputs:
    # Existing ascend_forward_context and complete native MoE, once.
parent.post_ffn_output(ticket, output, *, producer_stream)

parent.finish() -> None
parent.fail() -> None
```

Tickets are opaque owner-issued handles. Read-only diagnostic coordinates
may expose transaction/digest/sequence/layer/stage; identical fields on a
forged or old object confer no authority. Validate exact parent/transport,
accepted plan identity and ticket identity before I/O. Reject duplicate posts,
duplicate loans, wrong role/stage/layer, tickets from another parent, and
legacy calls while this owner is active. A ticket holds all input/output
storage, native work handles and readiness/consumer events until their last
use; it is not an abort token or a claim that output is already ready.

Two frontiers replace the single completion cursor inside this opt-in owner:

1. The posting frontier remains exactly layer-major/stage-minor. A advances it
   after three input sends and the output receive are posted for that step;
   F advances after its three receives and output send are posted. A native
   producer-complete event precedes sends, receive completion precedes F
   compute, and F compute completion precedes its output send.
2. The consumption/reuse frontier advances only when the matching output/input
   loan has recorded its real consumer completion and both wire directions are
   accounted for. One occupied slot per stage bounds active payload ownership
   by U, not layers times U. Reusing stage S at layer L+1 waits S's own prior
   send and consumer events; it cannot overwrite a borrowed output before HC
   has consumed it. A final continuation is a real consumer too.

The returned A output is borrowed inside its loan. Retaining it elsewhere
requires an explicit copy; an HC continuation that retains the view extends
the ticket lifetime. F input storage survives native asynchronous routing and
TP kernels, and the native output survives its send. Record every BF16, FP32,
IDs and output allocation on actual consuming streams, using the existing
scoped native/vLLM stream helper. A tensor tuple kept in Python without the
matching stream dependency is insufficient.

Create unique events per parent/layer/stage initially rather than re-recording
a stage event while another stream may still wait on its old record. Keep
event/work bookkeeping until proven complete, but release obsolete native
output tensor references after the recorded last consumer; do not accumulate
all layer outputs in a parent-wide retained payload list. Backend work-handle
storage ownership must be checked before promising that release is bounded.

## Wire order and the split-wire boundary

The ticket proposal preserves each current tensor match, peer traversal and
logical posting order:

```text
A host posts: input3(L,S0), output-recv(L,S0), input3(L,S1), output-recv(L,S1)
F host posts: input3-recv(L,S0), compute/send-output(L,S0),
              input3-recv(L,S1), compute/send-output(L,S1)
```

Independent streams may complete a later input transfer before an earlier
output finishes. That relaxation of device completion order is the intended
overlap; it must not be represented as preserving C4's strict end-to-end
completion order. No extra tensor/header or parent-control frame is required
merely to separate post from consume. Existing per-group order still has a
matching proof at the payload level; installed backend cross-stream behavior
and compatibility with a serial peer still require actual tests before normal
selection. Both service owners must freeze an admitted scheduling choice for
the session; no mid-parent flag change or late fallback is allowed.

If the backend cannot post without a CPU wait, moving only F's S1 receive
before S0 send deadlocks the current A sequence:

```text
A waits F output S0 before it sends S1.
F waits A input S1 before it sends output S0.
```

The alternative `A SEND S0,S1 -> RECV S0,S1` / matching F bulk-input wave
changes logical wire order. It is a paired split-wire protocol, not an
implementation detail of a rank-local eager flag. It needs explicit A/F
session agreement, its own admission/version boundary preserving old bytes,
and a full deadlock/peer-traversal proof. It must not be hidden inside an eager
ticket API. No stronger cross-layer bulk schedule is proposed here.

## Native model, IDs and TP ownership

The A stage engine keeps native V4.1 continuations, not the donor's V4 tuple.
For each next layer, wait/loan the same stage's preceding output, complete HC,
then observe auxiliary hidden state at native layer entry, wait ordinary
parent Engram mask/lookup events and perform native prepare_remote_ffn. Only
then send that layer's BF16 and original FP32 outputs. Final HC, auxiliary
merge, MTP buffer, collapse and norm follow all final-stage consumers.

One unsliced parent prepares Engram and lookback once. The existing U1
pre-hook/U2 manual producer distinction and sole native finish_execution are
unchanged; the ticket owner neither creates child history nor retires native
ModelState. Shared-Attention views remain stage-specific, and only one compute
stream queues their native Attention/TP operations in deterministic order.

Maintain current C4 raw ID semantics: private int32 stage clones, valid token
IDs unchanged and padded IDs zero. A can validate its owned stable IDs once
before asynchronous posting rather than adding per-layer `.item()` checks on
the communication frontier. This must be a real owned-storage invariant, not
an externally supplied `already_validated=True` bypass. F retains vocabulary
and padding validation for received data; a check may delay that stage's
compute but must not prevent already posted valid peer sends from progressing.
Changes to the validation placement need their own negative tests.

F consumes original FP32 directly and runs the entire native compute_ffn_output
with existing per-stage ascend_forward_context, raw IDs and native TP once.
Never add another TP reduction. All native TP ranks execute the same
layer/stage compute sequence and actual counts; plugin HCCL groups remain
separate from native TP. Preserve serial C4 output values, including its
padding behavior: do not import the Graph bank's padded-output zeroing policy
into eager execution without a separate equivalence decision. A positive
padded bucket with valid=0 still computes/transfers its fixed rows; an extent
that is globally zero has no payload step. A wholly empty parent completes
honestly without creating tickets or loading missing model weights.

## Finish, cancellation, failure and STOP

`parent.finish()` requires the posting schedule exhausted, every final output
consumed, all required sends/receives accounted for, and every compute/consumer
branch joined to the parent. It then invokes the transport-owned normal
finish path once: device synchronization plus all-world transaction/digest
COMPLETE. A host posting cursor reaching the end is not this proof. Native
request/Engram retirement remains the original runner's responsibility.

Normal request cancellation before acceptance can avoid a parent. After an
accepted parent starts, a healthy graceful cancellation must still execute
the agreed matching schedule, drain its genuine work, join and COMPLETE before
idle STOP. It may discard the final result; it cannot omit a peer's expected
send or manufacture zero MoE responses to fill the wire.

Pre-acceptance, completely received control reports can use the existing
drain-all-reports then all-world ERROR path. After data posting, a local
validation/kernel/I/O exception makes the owner and transport terminal. Retain
streams, events, tickets, works, model and groups and propagate the original
error to process supervision. Do not issue unsolicited best-effort ERROR or
STOP to peers already blocked in payload I/O, clear in-flight state, release
buffers, destroy groups, or declare COMPLETE. Draining already completed valid
work can be observed; waiting for an absent peer or launching more traffic
after an arbitrary backend failure is not a safe automatic drain strategy.

STOP remains an idle parent-boundary handshake after COMPLETE with no live
ticket/loan/reservation. Only acknowledged global STOP permits normal caller
group cleanup. A ticket `fail()` retains ownership and does not cancel HCCL.
Existing compute-cancel or peer-exit device probes demonstrate their bounded
supervised cases; they do not prove interruption/abort of a process blocked
inside HCCL, in-process communicator recovery, or safe same-process restart.

## First concrete validation and costs

Before coding a production backend adapter, resolve the installed concrete
isend/irecv signatures, Work completion/wait-to-stream semantics and HCCL
nonblocking progress with a bounded two-stage probe. Compare posted order,
consumer readiness and exact changed BF16/FP32/IDs/output across iterations;
hold one stage's compute deliberately while the other transfer progresses.
Source availability, return of a Python method and CPU event recorders do not
establish this behavior. No such async-backend result is asserted here.

CPU behavior seams should use actual ticket/frontier objects and public peer
records: two outstanding stages, duplicate/wrong/stale/foreign handles, same
stage premature reuse, wrong dtype/shape/IDs, missing or repeated output loan,
stage-zero skips, positive valid-zero rows, native TP mapping and unchanged
MoE outputs, final-join/COMPLETE/STOP rejection with a live ticket, and sticky
failure retaining the unresolved buffers. Test original FP32 values that
differ from BF16.float and continuation reuse across layers. Actual NPU tests
then cover TP1 and TP2, all-off versus eager-on, changed counts, allocator
reuse pressure, normal STOP and bounded peer/compute faults. A real A/F trace
must establish useful overlap and its cost; full V4.1/Engram acceptance still
needs the target environment.

For paired layout with R total padded rows and H hidden size, two retained
stage slots need approximately `(8*H + 4)*R` bytes for BF16, original FP32,
int32 IDs and BF16 output, excluding native activations/continuations and any
intentional extra ID copy. Streams and initial unique events cost O(1) streams
and O(L*U) lightweight event/work records; active payload slots stay O(U).
There must be no per-layer device synchronization for safety. Removal can use
native stage-aware asynchronous expert hooks only if they preserve ticket
ownership, native continuation/TP, exact payload matching and final retirement.
