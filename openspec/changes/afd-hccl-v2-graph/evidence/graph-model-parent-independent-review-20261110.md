# Independent static review of the graph parent and tensor stages

Date: 2026-11-10. Reviewer: transport architecture agent. The graph parent,
tensor-stage driver and execution owner were authored by the root agent. This
review does not independently approve the reviewer's own UBatchRunner or
capture-scope patches.

## Snapshot and result

| File | Reviewed SHA256 |
| --- | --- |
| `afd_plugin/model_executor/models/npu/deepseek_v41_graph.py` | `f7520ab3109478469c890d0b07419c1b01b5c6d1e4ef094037d1a7c798c492e7` |
| `afd_plugin/v1/worker/npu/hccl_v41_graph_execution.py` | `c021fe036f8083a9bcbd7e17d21bcebbff36eeb46373035d0056a66e88be2f10` |

No remaining static blocker was identified within this explicit component
scope after the two findings below were resolved. This is not native model
Graph, SFA/DSAV41 operator, full-checkpoint or production admission acceptance.

The comparison used read-only Git objects vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607` and Ascend
`7e37e29a7803d6897a7113088e2b771f83205bc0`, including native V4.1
`model.py:1297-1360,1561-1651`, Engram ModelState, native ModelWithContext and
the original UBatchRunner/model capture factory.

## Resolved findings

1. A prepared callback originally validated only its prepared-state identity.
   Completing or replacing the reservation before calling that callback could
   still execute the stale HCCL tensor body. The final owner checks terminal,
   closed, active plan and exact public reservation identity before executing
   the supplied wrapper. The regression now creates the real callback,
   completes its reservation, and requires the callback to reject before any
   endpoint payload. It does not merely test `require_full` after retirement.
2. Any nn.Module could originally be supplied as the native parent wrapper.
   The final owner requires native ModelWithContext, its original model to be
   this owner's model, and its metadata owner to be this ModelState's metadata.
   Tests cover both foreign model and foreign metadata. Parent context and
   platform kwargs are constructed during preparation, outside capture.

The owner also compares the complete bank capabilities to this transport and
requires each positive exchange group to be the exact corresponding group
owned by this session. Matching dimensions and rank numbers alone cannot
substitute another session's communicator.

## Source contract examined

The static driver retains native BF16 activations and the independent original
FP32 router input. It uses native layer continuations with the matching
residual/post/comb/pre_mix ownership. Attention buffer bindings are scoped and
restored per child. Auxiliary outputs observe layer entry before Engram; MTP
receives the reconstructed parent's HC tensor before final native collapse and
normalization. The bank's borrowed output is consumed by its continuation
before the next exchange overwrites it.

The parent enters native captured_engram_inputs once for the unsliced parent,
then exposes child token views of that storage. The same ordinary events may
be waited by both children without ExternalEvent reset. The native producer
scope joins its auxiliary stream on success and failure. Unsupported native
producer layouts reject before preparation and require early eager selection.
The actual ModelWithContext remains around one parent invocation and owns the
native metadata begin/finish frontier.

Only tensor copy/mask/exchange and model kernels belong in the graph body.
Reservation, actual count collection, device-scalar count updates, capture
phase completion and resource teardown stay outside. Positive padded stages
execute even with zero valid tokens; a statically zero-row stage records no
wire on either peer. The driver does not advance eager transport cursors or
retire request history.

## Evidence boundary and remaining integration work

The immutable actual-dependency CPU snapshot `model-seams-007` ran 198 tests:
187 passed, 11 failed, zero skipped/errors. Its XML was independently read.
The model tensor driver's 19 cases and bank/F tensor seams passed. Ten owner
failures were a duplicate shared-fake constructor keyword; one runner failure
was an expected CPU integer dtype mismatch. These are preserved as a failing
intermediate snapshot, not a green final run. Corrected snapshots must be
recorded separately.

CPU arithmetic, event handles, task executors and endpoint peers prove module
and ownership contracts only. Actual native V4.1 layers, captured Engram,
A5 packed metadata, repeated NPU graph execution and native model-manager
output retention remain separate validation requirements. Earlier native base
manager/HCCL graph probes do not establish this full model body.

Native capture must reserve before its hidden factory's prepare_dummy_inputs;
U1 needs the actual selected prepare_attn batch to submit counts before model
execution. Runtime early mode agreement must precede all input preparation.
Native ModelAclGraphManager.run_fullgraph also directly reads the parent
ModelState.attn_metadata; U2 child metadata must be handed to that route
explicitly rather than relying on stale parent metadata. SFA/DSAV41's pinned
update methods being no-ops does not remove that Python ownership obligation.
