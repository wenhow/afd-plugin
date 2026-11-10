# Independent review: cross_layer_v1 static Graph wavefront

The reviewer inspected the eight final production files below, their changes
from the frozen bulk implementation, the common action constructor and both
role consumers. No source blocker was found for this controlled paired
component. This is not approval to open the native-model guard or a device
acceptance result. The reviewer changed no runtime, test, Git object or device.

## Additional probe and harness review

The reviewer separately inspected the two new controlled-device artifacts:

| Artifact | Reviewed SHA256 |
| --- | --- |
| `tests/npu/hccl_graph_manager_probe.py` | `8e64e60927cdaefd692a3bf82d4482317013db24a28807d0baff151939a2b904` |
| `openspec/changes/afd-hccl-v2-graph/evidence/a3-native-manager-20261109/run_a3_graph_manager.py` | `1084c6d330902ba1ebc3a99bbb552671081434cdef3fd85f25f54dba791ab734` |

Both explicitly propagate `cross_layer_v1` and a bounded layer count of two
or three through local harness, remote harness, supervisor and both workers.
The new tensor callback consumes `pipeline.schedule.actions` on both roles.
It does not reconstruct another wire order. Attention prepares first-layer
inputs for both positive stages, then prepares a later input only after its
own received output. FFN receives, computes and copies into that stage's
fixed bank before compute-done; pending stage outputs are popped by their
matching output action. Native compute order remains layer-major. A stage
with zero physical rows creates no schedule action; positive padding still
uses the actual per-parent validity tensors.

Per-layer observation buffers preserve each BF16/independent-FP32/ID/output
result across later stage-bank reuse. Attention's next compute retains the
previous borrowed reply until the side stream consumes it, and its later
input action waits for that computation before a subsequent reply can
overwrite the bank. Final continuation uses the same final-compute/join
boundary as the existing probe. No JSON, host readback or COMPLETE was added
inside the captured callback.

An independent local AST comparison against HEAD confirmed unchanged
`payload`, `verify`, `observe_bank`, `native_check` and `complete_active`
functions. The reviewer also recomputed all twenty frozen source hashes;
all match `graph-cross-layer-final-snapshot-20261110.json`. This supplements
the author's static checks rather than claiming a new CPU or device run.
The existing 26-transaction lifecycle, native TP check, source pins, idle
checks and owned-process supervisor remain in use. The additional layer
dimension reaches their dynamic loops rather than leaving a two-layer
constant in the numerical check.

No additional source blocker was found for one separately authorized,
bounded TP1 three-layer component session. Preparation manifest
`4dd1a582eeeefdfe5a785ca1ebf175cb92e0f315cb396a76614dfc077246229b`
records 31 installed source pins and an immutable archive; prepare-only is
not HCCL, Graph or model acceptance. The reviewer did not launch a device
session. TP2, native HC/Engram/MoE, GE, physical side-stream HCCL and measured
overlap or performance remain outside this artifact review.

## Independently checked TP1 device follow-up

After the separate parent authorization, the implementation agent completed
`a3-graph-cross-layer-20261110/cross-layer-tp1-001`. The reviewer independently
read its immutable identity, outer and inner summaries, both rank reports,
installed source checks and before/after NPU logs. Manifest
`8539a7e899605d997f825c9fdcc7eee8084bd802928844aa190d6b1962516830`
contains the eight reviewed production files and the exact reviewed probe
and harness. The production/probe hashes were recomputed against that
manifest; all match. The separate local twenty-file CPU snapshot still
matches its own final manifest and is not described as part of the device
archive, which excludes unit tests.

The actual run selected `cross_layer_v1`, three layers, stage banks and all
five Graph axes enabled. Both ranks report passed and exit zero, each with
26 transactions: four warmup, four capture, twelve replay and six eager.
Their four U2 warmup/capture resource records contain six positive compute
steps and the same twelve common actions, including the cross-layer input
between two previous-layer outputs. Both native TP checks are `[1.0]`
before and after close. All 31 installed source pins match; all eight boards
are idle before and after; there is no supervisor cleanup, timeout or killed
owned process. The outer bounded session took 30.942 s.

The unchanged exact numerical oracle runs for all three layers and checks
each independent BF16/FP32/ID/output observation and final continuation across
warmup/replay, changing valid counts and fallback. A passing result therefore
qualifies this TP1 controlled arithmetic Graph/HCCL wavefront component at
these source bytes. It does not establish native model/HC/Engram/MoE,
checkpoint loading, TP2, GE lowering, physical side-stream HCCL, overlap or
performance. This reviewer performed no device launch.

| Source | Reviewed SHA256 |
| --- | --- |
| `connectors/npu/hccl_graph_schedule.py` | `45c3904a6a8d523b3564b665894431b63b27309da5817c3577d495e4cc978c76` |
| `connectors/npu/hccl_graph_pipeline.py` | `94929ab44c1a27271fc544675e64b757ffe0ab3a1331b4fd3b8bfe66242468ae` |
| `connectors/npu/hccl_graph_stage.py` | `6e4110a23226b6a39354bd9aeb6f17c61d2f93166cef8c13f1997f80a95a00a0` |
| `distributed/hccl_plan.py` | `43434ec9bfc11a8bc7d4b57d97d2c1a2b2c1f928c4fe74ac78e66d042acdb762` |
| `model_executor/models/npu/deepseek_v41_graph.py` | `2a502b8d248f22ab168d901c58aa5b49a476deea7ca9df599cdc35f94817ef5f` |
| `v1/worker/npu/hccl_v41_ffn_graph.py` | `5b048db49931eef11d3741709538d949a87538a4dee04a7778b6576f11c9a19b` |
| `v1/worker/npu/hccl_v41_graph_coordinator.py` | `35adf2b22af1ba80938c91dd5bea8670c8f99f951c767abd9a45b6f26ee420f4` |
| `v1/worker/npu/hccl_v41_graph_service.py` | `3f5b104b22dcd2be6d1303f7bd4e90b5d0d3a216c8b36991100271bc852b1a43` |

Paths are relative to `afd_plugin/`. The native and donor ownership contracts
remain those in `sources.lock.json`: vLLM `ced6857a`, Ascend `7e37e29a`, donor
`4ddd69a3`. This component changes plugin scheduling, not those upstream
implementations or native TP reduction.

## Matched order and dependency proof

`HcclGraphSchedule` builds one immutable common action list outside capture.
For two positive stages and three layers it gives
`I00,I01,O00,I10,O01,I11,O10,I20,O11,I21,O20,O21`, where INPUT is A send-three /
F receive-three and OUTPUT is F send-one / A receive-one. Both consumers use
this same list; they do not independently invent their wire sequence.
The pipeline checks a distinct action cursor as well as its compute/input/
output frontiers, advancing the action only after the corresponding parent
operation records its event.

The first two input actions make F compute C00/C01 available. O00 waits only
C00; I10 depends on O00 and A's own C10. O01 then consumes C01, already queued
before I10. Thus matched blocking warmup has no new circular dependency.
Both roles retain native compute order C00,C01,C10,C11,C20,C21, preserving the
order of native TP collectives. The negotiated capability limits this new
order to Graph v1, U2, paired native TP1/TP2 and existing count agreement.
The default capability bytes remain omitted/serial; peers with a different
exchange order cannot silently share a session.

Only the cross-layer schedule relaxes A's requirement to enqueue every previous
layer receive before the next compute, and F's whole-layer output frontier
before receiving the next layer. Own-stage receive/send dependencies remain.
Attention requires hybrid policy; F requires receive and cross-layer policy.
Coordinator and service report unsupported before the first capture decision
if the session cannot supply them, including a sequence whose first key is U1.
Actual U1 banks retain serial behavior. No late local fallback is introduced.

## Storage, state and failure boundary

For A, borrowed O00 and the four old continuation tensors enter C10's retained
input tuple before `_complete_stage` clears execution fields. C10 consumes
them, prepares the next native HC/pre_mix continuation, and records compute
completion. I10 waits that event and copies/sends its BF16, original FP32 and
raw IDs on the parent. Only a later parent O10 can overwrite the same output
bank. Strong bank ownership plus recorded actual stream consumers preserve
the asynchronous allocation lifetime; a Python return alone is not the fence.

For F, every nonserial native result is copied by `prepare_output` into its
fixed stage-owned output before compute_done. The pending map holds that bank,
not borrowed native MoE scratch. C01 and C10 can therefore coexist even when
successive native calls return the same writable workspace. The next receive
of a stage follows that stage's prior send event; its compute waits the new
receive event. Masking and send remain on the existing parent. No reduction,
FP32 reconstruction, generic fanout or new payload bank is added.

Static zero-row stages contribute auxiliary coordinates but no wire/compute
events; padding-only positive stages retain the full common trace regardless
of live valid count. The existing single parent Engram producer, ordinary
lookup-ready events, shared-Attention stage bindings and final continuation
join/merge remain owned by the original parent/side scopes. Exceptions retain
the failed owner/frontiers and are not translated into a network abort.

The audit initially identified outdated proposal wording about merely retaining
native F outputs. The implementer updated the proposal to require the stage
copy already present in the final code; this closed a documentation risk,
without introducing an extra runtime delta during this review.

## Actual CPU evidence and limits

The reviewer directly parsed both immutable identities, summaries and JUnit
files under `a3-graph-cross-layer-20261110`:

* `graph-cross-layer-025`, manifest
  `766374c7a843e7e02a1ccf823774e3e428131771d105d9273fb78f28eb79782c`:
  647 passed, five failed, no skips/errors (652 selected tests). The failures
  were two bank message expectations and three new pipeline test fixtures.
  Six other complete mirrors passed 485 tests; their production bytes still
  match the final snapshot.
* `graph-cross-layer-026`, manifest
  `13bea45b7c4596f968cf5d525724f90ec90a1aa6adb3dba991ec5a825e9075ba`:
  the complete bank 43 and pipeline 124 mirrors passed, 167 total, no
  skips/errors/failures. All eight final production bytes match this snapshot.
  The bank change restored the previous bulk error wording while distinguishing
  cross-layer split calls; the pipeline fixture corrections changed no runtime.

This is a directed closure, not a fabricated full 652-test green rerun. Both
summaries confirm installed source checks and no device probe. CPU state,
wire-order, shared-workspace and lifetime fakes do not establish actual HCCL
capture/replay. Previous bulk device evidence does not qualify cross_layer_v1.
The independently reviewed fresh three-layer TP1 device session is recorded
in the dedicated follow-up above; its result qualifies only that controlled
component. Native model, TP2, MoE/Engram timeline, GE, A5 and performance
acceptance remain separate.

The implementation adds a small static action list and host frontier checks
per recorded communication step, with at most one retained frontier/bank per
positive stage. Replay reuses the captured device work and does not run this
Python cursor. Remove the plugin wavefront when native stage-aware Graph hooks
offer the same negotiated order and state/storage ownership.
