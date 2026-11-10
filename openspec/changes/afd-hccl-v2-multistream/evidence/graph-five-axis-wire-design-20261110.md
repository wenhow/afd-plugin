# C6 Graph five-axis parity and separate split-wire design

This is a read-only design and source audit. It changes no runtime, wire codec,
admission guard or task status and runs no device experiment. Its immediate
recommendation is exact donor logical-plan and dependency parity. A stronger
FFN receive/compute pipeline is described separately and is not a prerequisite
for calling that donor behavior equivalent.

## Pinned source identity

All donor objects below come from
`4ddd69a391dfb5b3e362c7601399e538a31de702`. Original blob bytes were hashed
again during this audit; no donor checkout was changed.

| Alias and donor path | SHA256 |
| --- | --- |
| D01 `connectors/npu/p2p_hccl.py` | `594ac53c8c8008ba05da4d12db1c38511e94fdcb8e13340322a0587e940df0ef` |
| D02 `model_executor/models/deepseek_v4.py` | `d513a463d159a0a2c9dae18d566805d1f30a8bb9504909c29c7f1d04bb920d3b` |
| D04 `v1/worker/npu/ffn_model_runner.py` | `f95532370a9a5157b16817c5879d78af7405c321d5739d975cd6e2785fe54ccb` |

Target files inspected, before any change proposed here:

| Plugin path | SHA256 |
| --- | --- |
| `connectors/npu/hccl_graph.py` | `95a9ae01aef550fd0dd5967aa35458455c74c898ff9909316d89003bedf34ab7` |
| `connectors/npu/hccl_graph_stage.py` | `b55e8dce0a72109bfcceaca52163ac2177ca0d0d6697664af64ed3dd4460a00f` |
| `connectors/npu/hccl_graph_pipeline.py` | `6473cc7ad538ddeaf45a5a9233eef7d5bae79b1cb1e6c95b883953ef05db73cd` |
| `model_executor/models/npu/deepseek_v41_graph.py` | `a3140845755cae90d5f7927596c5a50248d829db15aac8c7d9230ef4d37a92fb` |
| `v1/worker/npu/hccl_v41_ffn_graph.py` | `3c708b4912fd8e72c6e16fa3ef59d9769440629c9673dfeacf1c5e26aaa84f37` |

The five Graph settings, exact parser/defaults, eager separation and diagnostic
settings are independently documented in
`overlap-switch-contract-20261110.md`. H09/H10/H14 in the capability matrix
require observable dependencies and honest effective behavior, beyond parsing.

## What the donor actually enqueues

D01:194-230 defines a logical compute/send/receive stream plan. Construction
at lines 629-646 chooses optional send/receive entries from the three-stream
flag. Compute selection and hybrid dependencies are gated at lines 668-700;
lines 726-807 connect parent-ready, own-stage receive-ready, compute-done and
final parent joins.

There is a separate, higher-priority communication override:

* D01:893-978 performs A send/receive on the current parent whenever
  `_graph_parent_transport_active()` is true. Compute-done is waited before
  the send; send-done precedes the receive; receive-done is recorded afterward.
* D01:1241-1317 applies the same parent override to F streamed receive/send.
  Receive can wait the same stage's preceding send. Send waits its compute.
* D01:1780-1789 defines this override as warmup **or** graph transport active.
  D01:1730-1778 uses ordinary `dist.send/recv` for warmup and graph-visible
  operators for capture/tracing. Its comment explicitly requires a quiescent
  communicator before capture and forbids moving these operations to side
  communication streams merely because a logical stream was configured.

D02:1180-1244 constructs A in this order: enqueue every stage's first compute;
for each layer, dispatch then receive each stage in order; only after all
stage exchanges enqueue the following layer's compute. Hybrid lets next-layer
S0 compute depend on its own previous receive rather than the parent's entire
layer frontier. Host enqueue order and device dependencies are distinct.

D04:900-946 gates F compute, receive and cross-layer selection independently.
D04:1002-1122 still iterates each stage as receive, enqueue compute, then send.
With the Graph override, that send waits compute on the parent before the next
stage's parent receive is enqueued. D04:1178-1195 adds per-layer waits for all
send events when cross-layer is off; lines 1196-1218 use final-stage-send joins
when cross-layer is on. Every send event was recorded on the parent in this
Graph path, so the extra per-layer waits do not establish a new independent
physical communication stream.

| Graph axis | Required donor parity | Honest parent-transport consequence |
| --- | --- | --- |
| Compute overlap | Gate the side compute stream on/off; off keeps Graph/U2 and its bank/key. | A hybrid compute can depend only on its own preceding receive. F still waits matching compute before its parent send. |
| Hybrid DAG | Select own-stage receive dependency versus whole-layer parent-ready, including final continuations. | This changes the A DAG even though communication remains on the parent. |
| Attention three-stream | Preserve independent logical send/receive selection and its configured value. | Warmup/capture override both physical communication choices to the parent. Do not advertise three simultaneous communication streams. |
| FFN receive stream | On waits same-stage previous send before bank reuse and connects receive-done to compute; off uses direct parent receive and records its completion. | Both perform the actual Graph receive on the parent. Turning on does not itself prepost S1 receive before S0 send. |
| FFN cross-layer | On uses same-stage reuse dependencies plus final joins; off additionally performs the donor per-layer send-event joins. Raw cross=1/recv=0 remains rejected. | Under the parent override these extra waits follow already ordered parent sends. Their existence is not evidence of physical cross-layer receive overlap. |

These are source-level dependency conclusions. The backend's internal HCCL
streams, graph lowering, device completion semantics and useful overlap still
require actual trace/timeline evidence. Python return or multiple stream
objects alone prove none of those properties.

## Smallest next implementation: fixed projection, then real consumers

First add an immutable, pure-stdlib selection next to the existing immutable
settings, with one explicit function such as:

```python
select_hccl_overlap(
    settings: HcclOverlapSettings,
    *,
    role: Literal["attention", "ffn"],
    phase: Literal["eager", "graph_warmup", "graph_capture", "graph_replay"],
    num_stages: int,
) -> HcclOverlapSelection
```

The frozen result must retain all five independently selected Graph axes,
their role/mode/U2 gates, logical communication choices and the actual
`parent` communication override. Keep configured values, effective logical
choices and physical transport ownership distinguishable in its public
diagnostic record. A Graph warmup callback's runtime mode NONE must not turn
this into the eager schedule. Capture/replay share the same retained selection;
replay does not reread environment or reconstruct events.

This first pure seam is configuration/plan validation only. It becomes runtime
parity only when existing A/F bodies consume the selected dependencies:

1. Graph compute off selects the synchronous C5 tensor body, with Graph phase,
   native descriptor, bank, wire and native lifecycle unchanged.
2. Graph compute on passes hybrid to the existing A DAG. A communication
   scopes resolve the configured logical plan and report the parent override.
   Do not allocate unused NPU communication streams to imply effectiveness.
3. F receive on/off selects the preceding-same-stage-send wait independently;
   both retain receive-done before compute. Off does not remove that readiness.
4. F cross-layer off adds an observable per-layer send-event frontier; on
   keeps the existing final-only frontier. Preserve final quiescence in both.
5. Record configured/effective/physical values outside capture once per rank
   when diagnostics are requested. These records must not claim speedup.

No axis should be collapsed into `hybrid`. `None` is a valid synchronous
Graph plan, not eager fallback. Eager U2 overlap remains its own unimplemented
schedule. A/F cache/session/manager owners must retain one immutable policy;
the current physical key excludes scheduling policy, and a generation miss
does not authorize recapturing an existing key under another plan.

Pure checks can cover independent axes, role/U1/U2/mode gating, all-off still
Graph/U2, malformed phase/role/U, dependency rejection and independent public
diagnostic snapshots. Runtime CPU seams then need to observe the selected
parent/compute and event order. Exact donor parity can truthfully report a
logical flag that is overridden for capture; an unused parsed flag cannot be
presented as an implemented runtime consumer. NPU all-on/off and one-axis
comparisons remain separate acceptance.

## Separate stronger schedule: paired split exchange

This section is an optional future enhancement, beyond the immediate donor
parity increment. The existing target `attention_exchange` at
`hccl_graph_stage.py:238-270` copies BF16/original FP32/IDs, sends those three
payloads and receives BF16 output immediately. The F body performs receive,
compute and send for that same stage before advancing. Static positive rows
with zero valid rows still communicate; static zero-row stages skip both peers.

Changing only F to receive S1 before sending S0 creates this blocking-warmup
cycle, regardless of the number of configured stream objects:

```text
A: SEND S0 -> RECEIVE S0 [waits for F SEND S0] -> SEND S1
F: RECEIVE S0 -> RECEIVE S1 [waits for A SEND S1] -> SEND S0
```

It must not be attempted with the current A order. A paired bulk-input wave
can instead use the same parent HCCL groups and payload types:

```text
A: COMPUTE S0,S1 -> SEND S0,S1 -> RECEIVE S0,S1
F: RECEIVE S0 -> enqueue COMPUTE S0 -> RECEIVE S1 -> enqueue COMPUTE S1
   -> SEND S0,S1 after each matching compute event
```

This is an opportunity for F S0 compute to coexist with the S1 receive. It is
not proof that the target backend actually overlaps them. Both warmup and
capture must follow this matching order; no design may depend on eager
`dist.send/recv` returning before its documented operation has completed.

The smallest bank seam would split A's existing operation into a validated
copy/send and a separate receive, while retaining `attention_exchange` as a
compatibility composition of the two. A retained frozen transfer coordinate
can name `(exact bank, layer_idx, stage_idx, exact buffers)`; it is not an
asynchronous Work handle or a network cancellation promise. Coordinates and
events should be prepared outside capture. Graph replay must not mutate a
Python pending cursor, allocate tickets or consume a one-shot host state.

The scheduler must own matching static send/receive frontiers and reject a
foreign bank, layer or stage before communication. A same-stage next-layer
copy must wait until its previous borrowed output has been consumed by the
native continuation. F same-stage receive must wait the previous send before
buffer overwrite. Original FP32 and IDs need the same lifetime/event coverage
as BF16, and all final outputs/sends must join before COMPLETE or storage reuse.
Errors retain unresolved dependencies for external supervision; they do not
invent an in-flight HCCL abort.

This can initially stay within the current paired topology
`attention_size == ffn_size == native_tp_size`, DP1/PP1/CP1, one peer per rank
and one distinct group per stage. It does not change native TP collective
order: every F rank still computes the same layer/stage order. Aggregation or
fanout would additionally need a complete peer traversal proof for all payloads
and output chunks; transport-only support is not model admission for those
topologies.

A stronger cross-layer F receive schedule cannot simply skip the existing
parent sends: with one communication parent, the next-layer A send still
depends on the prior layer's returned output, and all ranks must preserve
matching operation order. A future schedule would need its own static DAG and
backend legality evidence. The donor cross-layer flag alone authorizes none
of that change.

Before enabling split wire, both roles must agree the same immutable wire
order. It cannot be silently selected by a rank-local flag because the current
HELLO/key does not encode this distinction. Any future protocol capability or
explicit paired admission belongs to that enhancement, not this source audit.
CPU endpoint traces can prove matching sequences and reject the deadlock
construction; actual A3 warmup/capture/replay must prove device legality,
stable addresses and exact outputs. Native TP2, allocator stress, empty
stages, repeated replay and timeline measurements must then qualify the
selected scope. No such split implementation or experiment is recorded here.

## Authorized follow-up: pure immutable selection

After the separately delivered C6 base commit
`8a6eeb2a81506621f8571784179a46e3f4680d95`, the pure selection increment is
implemented in the existing stdlib-only `connectors/npu/hccl_overlap.py`.
This section is the implementer's contract and validation record, not an
independent runtime review. None of the split-wire design is implemented.

`HcclOverlapSelection` is frozen with slots and has only four constructor
fields: `settings`, `role`, `phase`, and `num_stages`. The named
`select_hccl_overlap(settings, *, role, phase, num_stages)` API constructs it.
All six overlap axes and both diagnostics are read-only properties projected
from those inputs; callers cannot supply inconsistent effective booleans.
The existing settings API and its 77 tests are retained.

The public effective axis names match the settings names. Graph compute is
gated by explicit Graph phase and U>1 for either role; hybrid/three-stream are
Attention-only; recv-stream/cross-layer are FFN-only. Cross-layer also gates on
effective recv-stream. The settings constructor continues to reject raw
cross=1/recv=0 independently of role/mode. Live eager U>1 selects only the
eager axis. Graph U1 selects no overlap axes. Compute off or Graph all-off
preserves the supplied Graph phase and never enables eager fallback.

Public `logical_compute_stream`, `logical_send_stream`, and
`logical_recv_stream` retain the donor role-specific choices. The corresponding
`physical_*` properties apply `parent_transport_override` for every Graph
phase. In particular a true Attention three-stream or F receive-stream choice
can report a logical communication stream alongside an actual parent stream.
Replay describes the same retained plan as capture; no replay reads process
environment or creates a stream/event. `diagnostic_record()` returns a fresh
flat mapping with role/phase/U, `configured.<ENV>`, `effective.<ENV>`, logical
streams, physical streams and the override. Export mutation changes no policy.

Runtime initialization, eligible layer-major DBO contexts, non-Dynamo entry
and backend support remain caller preconditions. The four-phase API treats
`eager` as live eager for the serial FFN compute-sync diagnostic: it selects
that diagnostic only for FFN U1 with the diagnostic configured. A profiling
caller must suppress it separately; these inputs do not distinguish live
eager from an eager profiling call. No synchronization is performed here.

| Follow-up path | SHA256 |
| --- | --- |
| `afd_plugin/connectors/npu/hccl_overlap.py` | `04391ecd16bbf4210bb978f6c7be7a16ab53a68f74dc9ad5e1ae1d452865a34c` |
| `tests/unit/connectors/npu/test_hccl_overlap.py` | `c4bd7e59e057fa8da13a0221c4a1f13beec2bd2b3a90aeeef2ebafc20faaa895` |

Local Python 3.14.3 ran
`.venv/Scripts/python.exe -m pytest -o addopts= -q tests/unit/connectors/npu/test_hccl_overlap.py`:
131 passed, zero skips/failures/errors, in 0.54 seconds. This is the original
77 settings cases plus 54 selection cases, loaded as the actual standalone
module on a host without torch/vLLM. They cover U1/U2, both roles, all four
phases, independent axes, all-off, configured/effective/physical maps, invalid
inputs, diagnostics, frozen configuration and independent exports. Ruff
check/format and Python 3.11 AST grammar parsing passed for both files.

No device experiment, runtime consumer, policy/cache mutation, main runner,
wire codec or task checklist changed in this increment. This costs pure host
construction/property reads and diagnostic mapping allocation outside capture.
It does not prove that any newly projected axis is consumed by the A/F body.
The next runtime increment must bind this fixed configuration to each actual
descriptor while preserving the explicit legacy pipeline API and preventing
U2 settings from forcing side compute for U1 cache entries.
