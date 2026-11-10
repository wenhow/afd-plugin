# Proposed cross_layer_v1: a paired static Graph wavefront

Date: 2026-11-10. This is an implementation proposal and source audit, not a
runtime change or acceptance result. The preceding bulk implementation and its
raw evidence are frozen in
[the final manifest](graph-bulk-v1-final-snapshot-20261110.json) and
[the bulk evidence index](a3-graph-bulk-20261110/README.md). The implementer wrote
this proposal; independent review is still required before a new implementation.
No task checkbox changes here.

## Concrete next boundary

Add an explicit session exchange order `cross_layer_v1`. Keep every HCCL
operator on its existing capture parent, the native stage groups and peer
coordinates unchanged. This would allow the next layer's first input receive
to be enqueued before the preceding layer's last output send. It does not
claim an independent physical receive stream or measured overlap.

The smallest target is the current paired native TP1/TP2, DP1/PP1/CP1, non-SP,
spec-off, uniform q1 Graph U2 component. U1 remains serial. The existing
`stage_serial` and `bulk_v1` paths must retain their bytes, schedules and tests.
Use actual static key geometry and layer indices; no tensor-dependent branch,
new dynamic ticket, control packet in the graph, or native batched endpoint is
needed for this first component.

## Source and present blockers

Read source identities for the relevant implementation are:

| Source | SHA256 |
| --- | --- |
| `afd_plugin/distributed/hccl_plan.py` | `68fbfa4b5ce38c8dfdd6ad3cebe8dc797f71a7b634d68b876375eaa56bdaec7d` |
| `afd_plugin/connectors/npu/hccl_graph_stage.py` | `8c79cebfa7eac4d206d4a3e8e53b1a1bfa70cf5a447faa34bf791aaef45b11e9` |
| `afd_plugin/connectors/npu/hccl_graph_pipeline.py` | `af043a551fdb40fbdb665bd78112b9100a6c6b84ba8c831fee8939d858149405` |
| `afd_plugin/model_executor/models/npu/deepseek_v41_graph.py` | `09065402433f51cef2378ad984b29f2d0cec2b48f4961fb7bbca4315ee8f9347` |
| `afd_plugin/v1/worker/npu/hccl_v41_ffn_graph.py` | `0660618d0d2497d67b5f3f46b5fdfe0e52356dd716a7ad367e9dce5369ea5ea8` |

The donor remains `4ddd69a391dfb5b3e362c7601399e538a31de702`; its
parent-transport override and logical selection are documented in
[the five-axis audit](graph-five-axis-wire-design-20261110.md). The target
native model remains Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`,
`vllm_ascend/models/deepseek_v41/model.py`, particularly native model forward
1561–1651. No upstream source is changed.

Current blockers are explicit plugin scheduling constraints, rather than a
proven lack of endpoint support:

* `HcclGraphComputePipeline.compute` requires all preceding-layer Attention
  receives to have been enqueued, even in hybrid mode.
* `attention_send` requires every current-layer compute before any send;
  `attention_receive` requires every current-layer send before any receive.
* F `ffn_receive` requires the output frontier to be exactly the start of its
  layer. Thus it cannot receive layer1/S0 after sending only layer0/S0.
* A `_run_pipelined` and F `run` explicitly finish the whole layer's wire wave
  before constructing the next layer. Setting `ffn_cross_layer=True` does not
  change that host order.

The bank already exposes split input-send/output-receive and F input-receive/
output-send. Its payload and endpoint APIs need no additional split. New work
must change both paired schedules and the event guards together; weakening an
individual guard is insufficient and can deadlock warmup.

## One common paired action sequence

Define `INPUT(layer, stage)` as A sends BF16/original-FP32/raw-IDs and F receives
those same three packets. Define `OUTPUT(layer, stage)` as F sends output and A
receives it. These are full packet groups, not interchangeable single payloads.
Let positive stages be the ordered static stages with `stage_rows > 0`.

```text
for stage in positive_stages:
    INPUT(first_layer, stage)
for layer in layer_indices:
    for stage in positive_stages:
        OUTPUT(layer, stage)
        if a next layer exists:
            INPUT(next_layer, stage)
```

For two layers and two positive stages, both roles follow this same ordinal
sequence, interpreting input/output in their respective directions:

| Ordinal | Common action | A parent | F parent |
| --- | --- | --- | --- |
| 0 | INPUT(0,S0) | send 3 | receive 3 |
| 1 | INPUT(0,S1) | send 3 | receive 3 |
| 2 | OUTPUT(0,S0) | receive 1 | send 1 |
| 3 | INPUT(1,S0) | send 3 | receive 3 |
| 4 | OUTPUT(0,S1) | receive 1 | send 1 |
| 5 | INPUT(1,S1) | send 3 | receive 3 |
| 6 | OUTPUT(1,S0) | receive 1 | send 1 |
| 7 | OUTPUT(1,S1) | receive 1 | send 1 |

Bulk currently has ordinals `I00,I01,O00,O01,I10,I11,O10,O11`. The proposed
sequence is `I00,I01,O00,I10,O01,I11,O10,O11`. F enqueues compute0/S1 before
receiving input1/S0 and waits compute0/S1 only when subsequently sending
output0/S1. Consequently that next input receive is not downstream of the
preceding layer's entire compute/send frontier.

The compute enqueue order stays **layer-major, stage-minor** on both roles:
`C00,C01,C10,C11`. A first prepares both layer0 stages. After receiving O00 it
completes that stage's layer0 continuation and prepares layer1/S0, then sends
I10; after O01 it does the corresponding S1 work. F first receives/enqueues
C00/C01, then sends O00, receives/enqueues C10, sends O01, and receives/enqueues
C11. Native TP collectives retain their existing compute order. Cross-role
native groups remain distinct, and all native TP ranks must follow the same
static trace; transport support for uneven peers is not evidence for different
native TP scheduler batches.

The concrete F call remains plugin `AFDDeepseekV41FFNForCausalLM.compute_ffn_output`
to `self.model.layers[layer_idx].mlp`. Locked native
`DeepseekV41MoE.forward` at 274–340 passes the supplied original FP32/router
input and IDs into its experts. At 335–338 it reduces legacy tuple output when
TP>1, while tensor output has already undergone the native MoERunner reduction.
Therefore the schedule must retain one complete native call per layer/stage,
without adding another reduction or substituting BF16 for the router input.
The returned tensor may be a view of the native output at line 340. Retain and
record that source through its immediate validated `prepare_output` copy into
the fixed stage bank, before compute_done. The later pending send must retain
that bank tensor. References alone do not prevent writes through a native
workspace alias, as the subsequent output-ownership audit explains.

This dependency graph is acyclic with blocking matched warmup: I00/I01 make
C00/C01 possible; O00 depends only on C00; I10 depends on O00 and A's own next
compute; O01 depends on C01 already queued from I01, not on C10. The same
induction applies to following layers. The paired parent action order must be
validated as a complete sequence, rather than inferred independently on A/F.

One positive static stage degenerates to `I0,O0,I1,O1,...`; it remains correct
but offers no cross-stage overlap. Zero-row static stages generate no wire,
compute or event. Positive padding-only stages still generate the entire
sequence regardless of runtime valid counts. A full-empty parent retains the
existing accepted eager zero-parent retirement path.

## Minimal API and state changes

Use a small immutable, typed action plan shared by the A/F consumers and the
pipeline's guards, for example `HcclGraphAction(kind, layer_idx, stage_idx)` and
`HcclGraphSchedule(actions, compute_steps)`, built outside capture from the
bank's full immutable capabilities, key geometry and layer indices. A single
public factory should derive the plan. Avoid copying the sequence algorithm
into the model and FFN bodies. The action plan is static host construction;
replay only executes the recorded operators.

Existing public `attention_send`, `attention_receive`, `ffn_receive`,
`ffn_send`, `compute`, `final_compute`, and `join` can remain the consumer seams.
Add one strict action cursor for the new schedule. Each direction checks and
advances its common action ordinal only after successfully recording its
parent frontier. Keep input and output frontiers separately layer-major and
retain the existing independent compute cursor. Do not make `_wire_index`
alone authorize a next-layer receive.

For the new order only, A compute may follow its **own** preceding receive
before the other stage's receive has been enqueued, and A send may follow its
own compute before the other next-layer compute. The common action cursor
still dictates when that operation may occur. Existing serial/bulk all-stage
guards stay intact. F's matching action cursor replaces the bulk layer-start
restriction, while keeping same-stage preceding-send and compute checks.
Final Attention continuation can remain deferred until all final receives;
there is no need to optimize that tail in this increment.

The two model bodies should consume the action plan while their native compute
helpers retain the existing semantics. For A, INPUT of a non-first layer means
complete that stage's previous continuation, observe the next layer's aux
entry, apply its Engram lookup, prepare the new continuation, then send its
three original payloads. For F, INPUT means receive and enqueue one full native
MoE compute followed by its stage-owned output copy; OUTPUT means wait and
send that fixed bank result. No FFN TP
reduction is added or moved to another stream.

## Stage storage, Engram and final ownership

Each stage retains its own execution record, disjoint shared-Attention row
views, hidden/pre_mix, auxiliary list, native continuation and borrowed output.
The single parent captured Engram producer remains outside the stage schedule.
Its mask/lookup slices and ordinary ready events can be consumed by either
stage; no producer re-entry or ExternalEvent reset is added.

A must not overwrite its borrowed output bank before the next same-stage
continuation has consumed it. Compute1/S0 waits receive0/S0; its compute_done
then dominates I10, and the later receive1/S0 cannot overtake I10 on the parent.
This preserves the stage-bank reuse frontier. F receive1/S0 follows send0/S0
and its send_done fence; send0/S0 itself waited compute0/S0. Thus neither the
three input buffers nor the output buffer are reused while a previous native
consumer or HCCL send still accesses them.

More explicitly, an A stage frontier must preserve this chain:

```text
parent O00 writes bank.output[S0]
  -> receive_done[0,S0]
  -> side C10 waits that event
  -> retain inputs includes borrowed O00 + continuation0/S0
  -> complete_remote_ffn consumes O00/continuation0/S0
  -> prepare_remote_ffn produces x10, original_fp32_10, continuation1/S0
  -> record compute_done[1,S0] on the same side stream
  -> parent I10 waits compute_done[1,S0], copies/sends new inputs
  -> later parent O10 may overwrite bank.output[S0]
```

`_complete_stage` clearing its execution fields must not release the input
frontier: the tuple retained on entry to C10 still owns its old output and all
four continuation tensors. `retain_compute_outputs` must then retain the new
outputs alongside those inputs. When the subsequent send replaces
`_held[S0]`, every discarded side-stream consumer was already recorded via
`record_stream`, and the parent has waited compute_done. The bank output itself
remains strongly owned by the bank. Host return from a kernel or HCCL call is
not the storage-release criterion. Keep this proof local to cross_layer_v1;
do not relax the existing all-preceding-receives guard in serial or bulk mode.

On F, C01's result is copied into S1's disjoint output bank before compute_done;
C10 similarly copies its result into S0's bank. Their native return tensors may
share one workspace, so retaining them as pending outputs would be incorrect.
Keep a bounded per-stage pending-output map of fixed bank tensors: remove an entry only
after its matching output-send has recorded the parent consumer, with the
allocator stream record and same-stage send fence still protecting asynchronous
use. Receiving I11 must not overwrite S1's input/output storage before O01's
send event; receiving I20 similarly follows O10. If C10 or I11 fails, both
currently live frontiers remain reachable from the failed owner. CPU tests
should deliberately fail at these partial frontiers and check retained public
outputs/lifetime; a successful final output alone would miss the defect.

Retain and record actual stream consumers for positions, IDs, lookup/mask,
embeddings, BF16, original FP32, hidden/pre_mix, all four continuation tensors,
borrowed output, native F output through its immediate copy, fixed F output
banks and aux values. References are bounded by the
two stage frontiers; failure preserves current inputs, outputs, events, streams,
bank and model resources. Captured native graphs and all these resources stay
in the same bundle. Parent output merging, final HC collapse, MTP writes and
native ModelState retirement remain after the existing final joins. No control
COMPLETE runs inside the graph.

The current disjoint row-view implementation and single compute stream reduce
the new ownership surface. They do not prove every V4.1 operator workspace or
shared-KV access is cross-layer safe. A source audit of native stage-bound
workspace/metadata consumers, followed by actual model validation, remains
required. Current bulk arithmetic evidence does not settle native HC/Engram/MoE.

The subsequent
[MoE output audit](native-moe-output-ownership-audit-20261110.md) also identified
an output-ownership assumption at the deferred native return seam. The approved
024 follow-up fixes bulk by validating/copying each native result into the
existing stage output bank **inside compute, before compute_done**. Cross-layer
must build on this stronger seam: its pending-output map contains fixed bank
tensors, not arbitrary native return views. The preceding same-stage send event
must dominate each new compute-stream bank copy. References and allocator
records alone are insufficient to stop native workspace writes through aliases.

## Capability agreement and five-axis participation

Extend the existing strict `graph_exchange_order` value set with
`cross_layer_v1`; do not introduce a local environment-only wire selector.
Retain exact HELLO equality and reject unsupported/mismatched values before
stage groups/warmup. Default `stage_serial` is still omitted from HELLO, so its
bytes remain unchanged. Require graph/counts v1, capacity for U2 and paired
native TP1/TP2. The graph key stays a static geometry key; exact negotiated
capabilities plus the frozen resource/schedule identify its session order.
Changing order or scheduling policy needs a fresh session, cache and manager,
not a generation bump or recapture of an occupied key.

The smallest eligible local policy is:

| Role/setting | Requirement for cross_layer_v1 Graph U2 |
| --- | --- |
| Both roles: Graph compute | actual pipeline exists and compute is enabled |
| A: hybrid DAG | true, so next S0 compute can wait only its own preceding receive |
| F: receive dependency | true, with same-stage preceding-send fence |
| F: cross-layer | true, so no full-layer send join is required before the next receive |
| A: attention-three-stream | either value; logical choice remains observable, physical HCCL remains parent |
| Eager overlap | not a selector for this Graph protocol |

Use role-specific effective gates. Do not require A's unused F gate or F's
unused hybrid gate merely because the immutable policy also retains the other
role's configured fields. A pipeline alone is insufficient: its selected
policy must meet the relevant row. Hybrid=false or F cross=false implements a
different barrier contract and cannot be silently ignored in this order.

All A/F ranks participate in the existing complete-capture-list preflight,
including when the first key is U1. A local missing policy reports unsupported;
after all reports, everyone accepts and retires the same zero-row EAGER parent
and skips FULL capture. No rank enters a warmup action sequence while another
rank locally falls back to serial or returns early. A native graph replay miss
still uses the existing all-peer eager fallback before input preparation; it
does not change the captured wire order.

Graph all-off **with the default stage_serial session** remains synchronous
Graph U2, as today. An explicitly requested cross_layer_v1 session with all
compute flags off is ineligible and uniformly skips FULL through preflight;
it cannot locally turn into a different Graph wire protocol. For an all-off
Graph comparison, create a fresh stage_serial session. Supporting a negotiated
menu of per-key wire orders would require another explicit phase-contract
extension and is outside this minimal proposal.

## Implementation split and validation gate

The concrete next change can stay within the existing capabilities/codec,
graph stage bank/pipeline, A/F Graph bodies, coordinator/service preflight and
their mirrored tests/shared fakes. A small neutral static schedule module is
reasonable if it removes the two copies of the action algorithm; it must have
an actual pipeline/model consumer in the same change. No runner, main factory,
native capture patch, production guard or tensor payload format needs to
change for the component increment.

CPU seams must compare the paired full action trace for at least three layers,
not just two, and preserve native compute order. Cover static one/zero-positive
stages, runtime padding-only stages, BF16/original FP32/raw IDs, continuation/
aux/MTP output, same-stage reuse fences, all-off and role-policy preflight,
HELLO mismatch/default bytes, missing/duplicate/out-of-order actions and failed
output retention. Pure host event traces are dependency evidence only.

Only after independent review should a new bounded A3 TP1 session exercise the
actual new order with matched warmup, capture/replay, changing valid counts,
two/three-layer recurrence, fixed addresses, normal STOP and native TP checks.
TP2 follows separately. The existing bulk TP1 result cannot substitute for
this cross-layer capture gate. Timeline evidence is needed to establish useful
receive/compute overlap; native model and checkpoint acceptance are separate.
An error after posting is terminal and retains owners for supervision: no
late eager retry, unnegotiated abort, optimistic COMPLETE or package change.
