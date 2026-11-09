# Native capture coordinator independent review

Static review on 2026-11-10. No upstream edits, device experiments, production
Graph admission or full-model acceptance are part of this review.

## Reviewed source

- `afd_plugin/v1/worker/npu/hccl_v41_graph_coordinator.py`, SHA256
  `475b42ff057c4f4c68885c509fabfd88245a828cb338f1a647b958d94a4b47a1`.
- Native vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`:
  `vllm/v1/worker/gpu/model_runner.py:998-1071`,
  `vllm/v1/worker/gpu/cudagraph_utils.py:398-490,644-767`.
- Native Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`:
  `vllm_ascend/worker/v2/aclgraph_utils.py:138-169,256-337`,
  `vllm_ascend/worker/device_metadata.py:188-252`,
  `vllm_ascend/models/deepseek_v41/engram/model_state.py:48-75`.
- Neutral capture-sequence schema consumed by this version:
  `afd_plugin/distributed/hccl_graph.py`, SHA256
  `ed9f425e0fd5a486a7d66adce77a17e57c7dd5ea2103b78a3d53740188804742`.
  This reviewer implemented that neutral schema; its acceptance requires the
  separate root/architecture review rather than this coordinator review.

## Findings and closure

Two earlier ownership/agreement findings are closed in the reviewed version.

1. The first draft exposed an outer capture scope without constraining actual
   native model-capture parameters. U1 called its accepted parent body directly,
   so it did not inherit U2's ModelWithContext identity check. Foreign native
   capture arguments could bypass the intended bank or omit storage from the
   retained dependency bundle. The public API now accepts only
   `capture_native(*, profile_only: bool = False) -> int`, which delegates to
   the locked `NPUModelRunner.capture_model(owner)`. That method supplies the
   owning model/state/input/block-table/cache fields itself. The coordinator
   additionally checks manager.model_runner, ModelState.model, frozen input/KV
   storage and Attention group identities before factories and callbacks.
   EngramModelState.model is the outer native model supplied to its constructor,
   so the new identity check targets the correct owner.
2. Checking only membership of the first key did not agree the subsequent
   ordered capture schedule. Two A ranks with equal first keys but different
   later descriptors, ordering, count or profiling prefixes could leave one
   rank waiting for a QUERY after another had returned. Preflight now compares
   the complete ordered effective key sequence and explicit native prefix
   digest, and separately checks the first key. Mismatch reports unsupported
   before any factory writes. An empty A0 list honestly requests eager with no
   fabricated graph key; empty/nonempty peer mismatches also select a common
   skip. Metadata shapes remain model geometry rather than control payload.

No remaining hard blocker was found in this explicit capture-assembly scope.
This does not accept the separate later replay/execute lifecycle, which was
not implemented in this snapshot.

## Native lifecycle and resource ownership

Preflight reserves one zero-row eager parent and retires it through the real
counts/ACCEPT/COMPLETE protocol. Capture begins only after its common result
enables the complete native schedule. Each before-factory callback completes
the preceding phase, reserves the new one and arms one parent owner before
native dummy preparation. Native base capture creates separate warmup and
capture factories and stores each graph before the coordinator retains and
publishes it. The final phase is flushed after native base capture returns.
Host agreement, cursor completion and publication stay outside the tensor body.

The selected native capture entry still constructs ModelWithContext, whose
device metadata begin/finish calls and Ascend outer finally-retirement remain
authoritative. The U1 context hook installs only on the exact prepared native
metadata object. U2 retains its owning-parent wrapper path. Retained bundles
hold the bank, execution/body, ModelState, input/block/KV storage, native
Attention groups, manager and actual native output/intermediate buffers.
Frozen storage references are used instead of reading replaced runner fields
at handoff. Failure retains resources and marks the session terminal; it does
not claim to cancel a peer blocked in device communication.

Costs: preflight adds one control reservation plus an empty parent/device
completion; each native warmup/capture has graph-external reservation, counts
and COMPLETE rounds. Complete sequence hashing runs before capture. Storage
identity checks walk the retained native groups/KV references and descriptor
list on owned callbacks. Per-key static bank storage and native graph/output
resources remain retained until supervised cleanup; none of this establishes
a performance result. Removal depends on upstream exposing owning native
before-factory, prepared-U1 and after-capture callbacks.

Validation remains separate: mirrored CPU cases and the final immutable A3
manifest establish interface/protocol behavior when reported. Native V4.1
Attention, weights, offset shared-buffer views, Engram graph producers and
full-model capture/replay require their actual device evidence. The production
Graph guard and factory selection remain closed.

## Later explicit runtime/replay review

Separate static follow-up on 2026-11-10, after the capture-only snapshot above:

- Coordinator SHA256
  `b36558a1ca4d92d67628e17ac79736e76acea549f9d9b4139859a47af630e0b5`.
- Mirrored coordinator tests SHA256
  `77e491a82efa18184818aea351ba7b5ccc62caf170d6822b16826dd6bc07e425`;
  shared coordinator fixture SHA256
  `16ad4a9510228f3daaea76ad278394bd402f25cb2942b4acbf3abb5305eaba97`.
- Per-instance replay compatibility scope
  `afd_plugin/compat/patches/npu/hccl_graph_replay.py`, SHA256
  `d243acc272cbdd5d2f97d08a41e44e094f42ae34ee73aed0ac4755cf4b08b16f`.

A medium ownership finding in the first runtime draft is closed. Merely
checking the key/generation and native manager's graph handle did not prove
that the bank updated for this replay was the bank recorded by that graph.
Publishing the same graph through the public cache API with a replaced bundle
bank could otherwise update new valid-count/ID storage while the graph read
the old bank. `_require_cached_resources` now validates the retained
execution bank and recorded parent-body bank against the bundle bank, plus
model/state/control/transport/key/generation, frozen input/block/KV/group
references, native graph handle and native output/intermediate references.
The check runs before constructing/preparing the runtime owner or sending its
actual FORWARD. Six mirrored public-cache replacement cases exercise bank,
recorded-body bank, state, input storage, native output and generation
mismatches; they do not mutate private cache dictionaries.

`execution_scope` captures the actual selected native gather tuple and makes
the all-peer decision before input preparation. Eager selection enters the
existing eager preparation scope only after agreement. FULL selection checks
the retained bundle, arms the parent owner and validates actual preparation;
there is no fallback after FULL preparation. A genuine zero-work early return
reserves an explicit zero-row eager parent and completes the real handshake;
a nonempty return without dispatch is rejected.

The replay patch wraps only the selected instance's native
`ModelAclGraphManager.run_fullgraph(self, desc: BatchExecutionDescriptor) ->
torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]` at locked Ascend
`aclgraph_utils.py:174-207`. Its before-replay context temporarily supplies the
prepared first child's native metadata, restores it on every path, and calls
COMPLETE after the selected native method returns. Native `prepare_engram`,
graph-parameter update/replay and `device_metadata.finish_replay()` remain in
that method. Native execute retains its sole `finish_execution` retirement;
the coordinator does not duplicate it. The first-child metadata projection is
restricted to the pinned V4.1 SFA/DSA update boundary, whose relevant graph
parameter updater is a no-op; this is not permission to generalize the same
projection to other Attention backends or CP execution.

The scope delegates the selected upstream method because its native
Engram/update/replay/finally lifecycle must remain intact, and restores the
exact prior instance binding even after an error. Nested/concurrent/foreign
uses are rejected. Costs include one early reservation, the separate actual
counts/ACCEPT round, completion synchronization and retained-reference checks
per parent; no performance measurement is claimed. Removal requires native
preparation/after-replay owning callbacks. No remaining hard blocker was found
in this explicit host runtime/replay scope after the bundle fix. This static
review does not establish actual V4.1 model capture/replay, Engram producer,
offset-view operator support or production admission. The subsequent A3 CPU
manifest and any actual device evidence must identify their own exact bytes.
