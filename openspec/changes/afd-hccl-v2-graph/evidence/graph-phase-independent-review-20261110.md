# C5 graph phase protocol — independent review, 2026-11-10

Scope: independent source review of the neutral schema, cap/version extension,
and staged transport reservation/actual-plan/whole-graph completion paths.
The reviewer did not author these three modules. The separate A geometry module
is reviewer-authored and is not covered by an independent approval here.

## Findings and required closure

Two related Attention peer boundaries were found in the initial frozen transport
`f728121cef806f682ca0d2e92bbfd8bf4966a7fec8c9e5d6f87842f1dd1d7170`:

1. The non-coordinator `submit_forward` STOP branch acknowledged STOP even after
   a graph reservation had been accepted. FFN already rejected the corresponding
   STOP between reservation and FORWARD. A malformed/misordered STOP must not let
   an Attention peer acknowledge shutdown and leave the reserved transaction.
   Normal coordinator STOP preflight prevents this in an ordinary successful
   call sequence, but the receive-side protocol contract also needs enforcement.
2. With the normal non-coordinator `transaction_id=None` calling convention,
   COUNTS_REQUEST did not compare its ID with the accepted graph reservation
   before replying. The final FORWARD/decision binding would reject a different
   ID, so this did not admit a mismatched device launch; it should be rejected in
   the solicited counts round before emitting an inconsistent counts report.

Both findings were sent to the protocol author and root. I independently
reviewed their final narrow fix: Attention rejects STOP while its graph
transaction is live, without STOP_ACK; mismatched COUNTS_REQUEST reservation IDs
produce ERROR rather than COUNTS_REPORT. The two new tests observe exactly those
wire outcomes and terminal state. No blocking source finding remains in this
standalone protocol scope. The changed bytes still require their targeted
actual-dependency regression result; the previous 317-case run is not relabeled.

## Verified control and ownership properties

- `graph_execution_version=1` strictly requires counts version 1 and the existing
  paired native TP topology. Negotiation compares graph versions exactly; it
  does not silently downgrade the mode/key contract.
- Neutral records reject unexpected fields, invalid enum values, bool-as-int,
  non-finite JSON, unbounded values and invalid digests. Static keys retain stage
  request/token bounds, zero-stage placement and ordered metadata shape tuples.
  Request overlap remains legal for native straddling/clamped slices.
- All non-coordinator Attention and FFN endpoints receive QUERY or idle STOP
  before sending a report. Reports bind actual sender, transaction ID and query
  digest. The coordinator drains complete semantic failures from all requested
  peers before broadcasting ERROR.
- DECISION is checked against each peer's query/readiness and acknowledged with
  its full digest. The coordinator drains all ACKs, then emits the global ACK;
  successful API return occurs only after that commit boundary.
- Replay requires all participants ready. Missing/unsupported replay selects
  the independently proposed eager dimensions. Warmup/capture require missing
  on every participant, rejecting unsupported or already-ready duplicate keys
  before input preparation.
- Actual local reports must match the accepted mode and stage padding. The
  accepted full decision is embedded in the final parent plan; parent identity,
  mode and all Attention padding columns are checked against it. The canonical
  plan digest therefore binds the phase/key/reservation as well as the actual
  valid/padded counts and target layers.
- A non-eager graph phase rejects each public eager tensor step. Whole-graph
  completion accepts only the exact agreed key, retires the complete local
  schedule, synchronizes the device, then retains the existing all-rank
  COMPLETE/digest handshake. It cannot itself prove graph launch; that is the
  cache/manager consumer's device acceptance responsibility.
- Successful COMPLETE clears both parent and reservation state. Failure marks
  the session terminal and does not automatically destroy caller-owned groups.
  Broken I/O/framing and preparation/device failure rely on process supervision;
  they are not described as coordinated recoverable cancellation. No new CANCEL
  wire kind or unilateral late fallback was introduced.

The key intentionally does not encode physical addresses, dynamic valid counts,
history values or request IDs. Its model/session namespace, local storage
generation and lifetime bundle are the cache owner's responsibilities. Native
request/query geometry must be checked by each Attention resolver from its own
actual batch. Transport has no model state and cannot perform that comparison
on behalf of a resolver that incorrectly claims ready.

## Backward bytes and initial regression evidence

I loaded the committed pre-extension C4 plan implementation from
`45356b2ff0f81e1d3fcf9b660026bdc904738a19` through its exact Git object and compared
serialization with the new module. All seven checks matched byte for byte:
HELLO for each role with counts versions 0 and 1 (graph capability zero), and
FORWARD plus canonical plan digest for stage rows `(3)`, `(2,3)` and `(0,2)`.
The graph capability field and nested parent decision are omitted when disabled;
this comparison does not assert compatibility for a graph-enabled peer with an
old strict decoder, which should reject the new capability during HELLO.

[control-001](a3-control-cpu-20261109/control-001/results/summary.json) records an
actual-dependency CPU run with exit 0, **317 passed, zero failures/errors/skips**.
I inspected its JUnit XML and independently matched the initial schema, plan,
transport, their mirrored tests and the shared control fake to `identity.json`.
The module counts were schema 40, plan 135, transport 82, cache 14 and geometry
46. This run predates the two receive-side fixes above; it is not final evidence
for their changed transport bytes.

Initial reviewed production hashes:

| File | SHA256 |
| --- | --- |
| `distributed/hccl_graph.py` | `111e6f5ca34280ec3e1339f8b57e8d42dbc17edca11099b1e06ef0d1de50de3b` |
| `distributed/hccl_plan.py` | `fa4b08b0179d992732d1a5f7243ae635d27c5992281c04ce2abe84eaa6041186` |
| `connectors/npu/hccl_staged_transport.py` | `f728121cef806f682ca0d2e92bbfd8bf4966a7fec8c9e5d6f87842f1dd1d7170` |

This source/CPU review does not validate native NPU manager construction,
capture/replay, GE/Dynamo, production model Graph dispatch, V4.1 Engram/events,
native attention/MoE or checkpoint inference. Those consumers and evidence must
be reviewed and accepted separately.

## Independently reviewed receive-side fix

The schema and plan remain at the hashes above. Final fixed transport SHA256 is
`a8ce468558a82b577e2fad8119030aa8dea33008bc621618b50d8c717ccc9dd3`;
its mirrored test is
`3086e232c8d234bfdfcb54d90fa24c42d07c396c37f42be6b1adbec737fb373c`.
The fix is limited to the two reviewed Attention receive boundaries and their
protocol tests. No cancellation handshake, late fallback or new feature
admission was added.

## Fixed-byte CPU regression

I inspected [control-002](a3-control-cpu-20261109/control-002/results/summary.json)
and its JUnit XML: **319 passed, zero failures/errors/skips**, exit 0, with no
device probe. Its manifest identifies the exact fixed transport `a8ce4685...`
and mirrored test `3086e232...` above; schema and plan hashes remain unchanged.
The subsequent [control-003](a3-control-cpu-20261109/control-003/results/summary.json)
also reports exit 0 and **519 passed, zero failures/errors/skips** for that same
transport, adding existing native model-runner and UBatchRunner CPU regressions.
These results close the two receive-boundary regression requirements for those
source bytes. Any later transport changes need their own source comparison;
these records do not imply device, model capture or checkpoint acceptance.

The root author subsequently clarified the `complete_graph_forward` docstring:
capture recording is a phase boundary and does not execute or validate captured
payload output. Current transport SHA256 is
`666e8ffab77ab15f9b5a9d1e9c004b2450349c3465fa2fb9e1c6a3e733d391df`.
The author confirms this is a documentation-only change from the reviewed
`a8ce4685...` body. The clarified text narrows the evidence claim and introduces
no additional runtime protocol behavior.
