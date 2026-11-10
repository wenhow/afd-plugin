# Eager ticket owner: implementation contract, 2026-11-10

This is the implementer's source review, not an independent approval or device
acceptance. The actual warmed endpoint gate supports developing this opt-in
component. It does not prove the complete multi-stage owner, model scheduling,
or overlap. Production admission remains guarded. The native A/F consumers are
a separate implementation increment owned by the parent agent.

## Frozen initial implementation

| File | SHA256 |
| --- | --- |
| `afd_plugin/connectors/npu/hccl_eager_pipeline.py` | `b758b9e8cb398cebdf91c8bd36087b060e2d10e7cbfc6636ff0ba385e0cb6e31` |
| `afd_plugin/connectors/npu/hccl_staged_transport.py` | `ae4c38de38cb90d5e511fe0c7d0dbd6fda9fc318292f2d10784d33b3c8d422f1` |
| `tests/unit/fakes/hccl_eager_pipeline.py` | `2d3564c479cf5fa530b9706649239cf6b78f91dfc387e56e5f66471e40642a8a` |
| `tests/unit/connectors/npu/test_hccl_eager_pipeline.py` | `a28bc135b2e0f05e488234468a0c5b3246e0080d56e61a770b73e806bdacbb9e` |

Prerequisites and source reasoning are in
[eager-u2-inflight-contract](eager-u2-inflight-contract-20261110.md) and
[eager-backend-next-api](eager-backend-next-api-20261110.md). Pinned PyTorch and
torch_npu collection and the real endpoint result remain under
`a3-eager-backend-20261110/`; this implementation does not replace that evidence.
No upstream source was changed. This is composition using public PyTorch P2P
and stream/event APIs, not a copied upstream function or new monkey patch.

## Public lifecycle

All roles explicitly call `transport.warmup_eager_pipeline(parent_stream=...)`
once, after group initialization and before any parent/reservation. On both
stage groups the Attention peer sends BF16, FP32, and int32 packets; FFN receives
them and sends a BF16 response. Device synchronization precedes an all-world
Gloo barrier. Only successful completion marks the exact group objects warm.
A failed warmup is terminal and retains its storage, stream, and groups; close
refuses to destroy them. Replacing a warmed data group does not transfer this
authorization. No HELLO extension or implicit startup action was introduced.

On an accepted actual target U2 eager plan, each rank explicitly opens the
owner with parent/send/receive streams. It checks paired native TP1/TP2, plan
capacity and shapes, equal native TP counts, indexed NPU devices, an actual
parent current stream, and absence of graph capture. DP1/PP1/CP1/non-SP and model
eligibility remain caller validation responsibilities. Profile, dummy, warmup,
U1, and static Graph phases retain their existing execution paths.

Attention calls `post_attention` in layer-major/stage-minor order and later
enters `attention_output(ticket, consumer_stream=..., retain=(...))`. FFN calls
`post_ffn_receive`, consumes `ffn_inputs(ticket, ...)` as a loan, and only after
closing that loan calls `post_ffn_output`. Three input operations and one output
operation preserve the existing paired wire order. FFN cannot post its next
stage receive until its preceding output has been posted. This does not create
a bulk-send or unilateral FFN-prefetch wire schedule.

Tickets have readable coordinates, but authorization uses the exact ticket
identity held in its live stage slot. Forged, foreign, replaced, consumed, and
duplicate tickets do not authorize a loan. A loan must cover the caller's full
asynchronous native use; a borrowed output must not be retained for additional
unfenced asynchronous use after closing it. Source arguments must remain stable
until their ordered copy/consumer frontier, rather than being mutated on an
unrelated stream after a host post returns.

Each active stage owns one fixed BF16/FP32/IDs/output payload slot. Producer
events fence copies and sends. Native `Work.wait()` runs in the actual loan
consumer stream; an event after the native consumer closes that loan. Reuse
first waits that consumer event and all Works on the parent, then records a new
reuse event. Both send and receive streams wait the reuse event before touching
the old slot. Merely inserting a wait in the parent stream would not protect a
different stream's write. `record_stream` plus strong references cover current
inputs, output, original FP32 and IDs, and explicitly retained continuation
tensors. Historical Work/Event bookkeeping remains parent-owned; payload slots
stay bounded by U. This relies on the tested backend dependency semantics and
still needs an actual multi-stage pressure test.

Globally empty stages issue no ticket or tensor operations. A whole-empty parent
can finish directly. Positive padding-only stages still exchange/compute full
native shapes. BF16 and original FP32 are copied separately; IDs remain int32
with the existing vocabulary and padding checks. No Graph-specific output
padding zeroing is introduced. These checks include host-visible ID reductions,
whose synchronization cost remains visible and has not been optimized away.

`finish()` requires all ordered posts and closed native loans, joins the last
stage consumers and Works on the parent, and invokes the existing synchronized
COMPLETE exchange once. The transport owns the cursor transition internally.
While this owner is live, serial tensor APIs, serial/Graph COMPLETE, a second
owner, and group destruction reject. STOP continues to require an idle parent.
Failure is sticky, retains the current owner and dependencies, and emits no
invented abort, STOP, or recovery packet. An external supervisor bounds peers
that failed in the tensor phase.

## Validation and remaining gates

Local Ruff check/format and Python 3.11 grammar parsing pass for the four files.
The local checkout has no torch; runtime CPU collection therefore cannot accept
the owner here. Actual-dependency CPU execution is pending the parent's immutable
snapshot run. The new tests use actual CPU tensor arithmetic with explicit host
allocation/Work/stream observation seams. Queued packet copies establish public
protocol and lifetime behavior; they simulate no NPU progress, HCCL timeout,
allocator capture, event timing, or overlap. Existing connector module validation
globs own both the mirror and shared fake.

The first device gate must use this actual owner, paired warmup of both groups,
two in-flight stages, full independent FP32 and IDs, per-stage reuse, delayed
native consumers, failure supervision, and then TP2. Native V4.1 HC/pre_mix,
Engram, FFN/MoE, TP collectives, and real A/F timelines remain separate gates.
This document marks no migration task complete and makes no performance claim.

## First actual-dependency CPU snapshot

`a3-pipeline-cpu-20261110/eager-transport-009` completed with 44 failed and 89
passed: 84 existing transport cases and five new-owner cases passed. The 44
failures began in the shared fake's invalid ACCEPT frame: the initial fake
omitted the required `world_rank`. The production owner and codec were not
changed to accommodate this fixture error. The corrected fake above supplies
each actual scripted sender's rank and the coordinator's rank on global ACCEPT.
The first raw result remains retained.

The parent reports the next actual-dependency snapshot `eager-owner-010` passed
all 49 owner cases on production `d67e22f6...` and mirror `e7f50ae1...`. Its wider
run had 13 unrelated new execution-session fixture failures, separately fixed
by their owner; this is not described as a single all-green suite. Exact raw
manifests remain authoritative.

A subsequent narrow source review found that `fail()` on a successfully
finished old owner could incorrectly terminal its transport after retirement.
The current source rejects finished/foreign fail calls and keeps live failed
owners sticky and idempotent. A public negative test verifies that rejecting
the old call preserves the idle transport. The earlier 49-case result is not
silently relabelled for these new bytes. The parent reports `eager-ffn-013`
passed all 50 owner cases and 43 new FFN consumer cases on their exact frozen
bytes. The wider 013 run contained two unrelated old Graph worker fixture setup
errors, closed by the separately retained `worker-followup-014` run with 114
passed. The new actual-owner device artifact and limits are described in
[its probe design](eager-owner-probe-design-20261110.md).

The separately authorized `owner-tp1-001` actual A3 session subsequently passed:
33 installed source hashes matched, all eight boards were idle before and after,
both ranks exited zero with no timeout or forced kill, six accepted U2 parents
completed, and each rank checked 16 positive layer/stage steps. Native TP
all-reduce returned `[1.0]` before and after plugin close. Exact BF16/original
FP32/IDs, full padding output, same-stage recurrence/reuse, whole-empty parent,
and normal STOP passed. Manifest:
`ccf53188e398340b70e4975fc2b856e1ddce8f0a06056438f12b2199d591ea54`.
Raw results remain under `a3-eager-pipeline-20261110/owner-tp1-001/`. This accepts
only this TP1 actual owner component; native delayed HC/MoE/model and overlap
remain separate.

The separately authorized `owner-tp2-001` then passed on four idle A3 devices:
33 installed source hashes matched, 8 idle boards before and after, four zero
exit codes, no timeout/forced kill, and each rank's six parents/16 positive
checks passed. Native TP all-reduce returned `[2.0]` before and after plugin
close for every rank. Its independent snapshot manifest is
`4c58361ff41090bc58e5d87e1f69f124753be97c703a5c36102f540157ce27c7`.
Owner/transport/probe bytes match TP1. This extends the actual owner component
scope to paired TP2 and preserves the same model/overlap limitations.
