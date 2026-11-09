# Native model graph coordinator contract, 2026-11-10

This records the Attention capture and post-KV runtime assembly component. It does not qualify a
checkpoint, NPU model graph, Engram producer, replay, performance, or production
Graph admission. The production guard remains closed. Pre-KV profile bridging
and main-runner bindings are a separate increment from the tested snapshots here.

## Frozen capture snapshot

- `afd_plugin/v1/worker/npu/hccl_v41_graph_coordinator.py`:
  `475b42ff057c4f4c68885c509fabfd88245a828cb338f1a647b958d94a4b47a1`.
- Mirrored test: `8e3f5cbaead1f419b681bb5ee02f5739cd15d8f7b3e3a34c9a0e9fffd4018da1`.
- Shared coordinator fake:
  `2046e8b13e3d5d635b307edcb8d22e03b391e4ce7f776cc79ed1b0cf3beac6e4`.

Ruff and Python compilation passed. A3 actual-dependency CPU snapshot
`a3-control-cpu-20261109/capture-seams-013` passed the 33 coordinator cases,
9 replay-scope cases, and all 14 signature rows. The entire snapshot retained
489 passes and two unrelated F-service test-fixture failures; that combined run
was not wholly green. Its source manifest is
`54d9b795f68c9e1496a51c3c209b424afc530f2c078b5f36534138364834a616`.
Raw results remain in that snapshot's `results` directory. Integration-012
explicitly excluded these three new coordinator paths.

## Native owner and call chain

The locked vLLM revision is
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`; Ascend is
`7e37e29a7803d6897a7113088e2b771f83205bc0`.

Ascend `NPUModelRunner` inherits GPU `capture_model(*, profile_only=False)`
unchanged. GPU `model_runner.py:998-1084` supplies the selected runner's model,
ModelState, input buffers, intermediate tensors, block tables, attention groups,
and KV configuration to its graph manager. Ascend
`ModelAclGraphManager.capture` constructs the real `ModelWithContext` around
that model and `model_state.device_metadata`, delegates the native factory, and
retains its final metadata cleanup.

The coordinator therefore exposes `capture_native(*, profile_only=False)` and
keeps its capture scope private. A caller cannot supply foreign model/storage
arguments to the manager through this API. Construction requires
`manager.model_runner is owner`, `model_state.model is owner.model`, the selected
U2 runner/session identity, native V4.1 SFA/DSA metadata, and the negotiated
model/device layout. Frozen input/KV/model/builder references, native descriptor
order, and profiling prefix are checked before preflight, every factory, every
U1 parent callback, and resource publication. PCP is excluded.

This component retains native `ModelWithContext`, input preparation, graph/output
storage, memory accounting, workspace locking, and final ModelState ownership.
It does not copy the native capture factory or invoke the model a second time.

`profile_only=True` is only an explicit fresh-session component parameter here.
Pinned Ascend `Worker.determine_available_memory` at worker.py:557-655 calls
`profile_run`, then native startup proceeds to real KV initialization and a
single capture. It does not call GPU `profile_cudagraph_memory`; Git grep found
no invocation of that helper in pinned Ascend worker.py or worker/v2. A profile
epoch/reset is therefore not required by this normal NPU startup chain. An
explicit invocation of the separate GPU temporary-KV graph-memory helper would
need independent analysis and remains unsupported here.

The genuine normal-startup gap is earlier: pre-KV `profile_run` has no native
graph manager or final KV/storage configuration. This coordinator deliberately
requires those concrete owners and covers post-KV execution. A separate
graph-aware EAGER bridge must reserve with manager=None before native profile
input preparation, then preserve the native profile retirement. That bridge
and ordinary main-runner binding are subsequent software work, not A5-only
acceptance tasks.

## All-peer startup agreement

Every Attention rank must call `preflight_capture`, including a rank whose
native capture list is empty or whose Engram producer cannot be captured.
Native `needs_capture()` must not cause a rank to skip this agreement locally.

The query includes `capture_sequence_digest`, a separate control identity over
the complete effective ordered `key_for(geometry)` list and the exact native
`_max_full_descs_to_capture` field. It includes actual metadata schema in each
key. It does not encode this schedule in a physical graph key, metadata shape,
transaction identifier, or fabricated request state.

When Attention0 has a real descriptor, it proposes replay availability for that
genuinely uncaptured key, warmup mode, and an empty eager fallback. A responder
reports missing only if its complete sequence and first key match and its model
is eligible; otherwise it reports unsupported. Order, count, later keys,
metadata schema, and profiling-prefix differences cannot leave one rank exiting
a shorter capture loop while another sends its next factory query.

When Attention0 has no effective descriptor, it proposes eager/warmup with no
key and zero rows. This names no fake graph and uniformly skips capture. All
paths accept and retire a zero-row eager parent before returning. There is no
model input preparation, Engram producer, tensor payload, or graph body in this
preflight. F-side handling requires fresh graph cache/model generation and
reports an honest miss or unsupported result.

## Factory phases and lifetime

The selected base manager callback retires the preceding phase, reserves the
next warmup/capture before native dummy preparation, allocates its bank, and
arms a concrete GraphExecution through the persistent GraphSession. U1's native
prepare hook and U2's native runner prepare submit the actual prepared layout
before any tensor callback. U1's context installer attaches only the guarded
tensor callback; it sends no counts or control packets from the model body.

Warmup and formal capture receive fresh phase owners/banks. The final phase is
retired after native graph storage has returned. Formal capture retains the
native graph plus the bank, concrete execution, ModelState/input/block/KV
storage, selected U2 runner/builders, manager, and native output references as
one cache entry before completion. Publication follows successful COMPLETE and
device synchronization. Failed capture marks the session/cache terminal and
retains available references for process supervision; it claims no device
cancellation or safe network abort.

Capture adds one startup availability/empty-parent exchange, one agreement and
completion per native warmup/capture phase, and a publication synchronization
per captured descriptor. The coverage helper reports bank and wire storage per
paired native TP rank; native model buffers, graph pools, and output storage are
additional. No overlap or performance claim follows from this host design.

## Review and validation boundary

The independent state-audit agent reviewed snapshot `475b42ff` and found the
foreign owner/storage boundary and complete-sequence findings resolved within
this explicit capture component. Its source review is not hardware acceptance.
The shared fake uses real control codec/state transitions, native CPU batch
preparation/slicing, the actual U1 hook, and native ModelWithContext around CPU
arithmetic. Device capture, memory accounting, allocation and attention kernels
are explicit external boundaries; their inert graph handle is never replayed as
evidence of NPU behavior.

The companion selected replay scope `hccl_graph_replay.py` snapshot `d243acc2`
matches pinned Ascend `ModelAclGraphManager.run_fullgraph` arguments and return
annotation by AST comparison. Its source comment uses an approximate line range;
the locked function is at lines 174-207. It retains native replay and cleanup,
restores the supplied metadata context, then performs the caller's completion.
Its CPU boundary validates host ownership/order, not graph/device execution.

Removal: replace the scoped capture/U1/replay hooks when native model managers
offer before-factory, prepared-input, explicit child-metadata, and completion
callbacks with the same lifecycle ownership. Upstream sources remain read-only.

## Frozen post-KV runtime snapshot

The final coordinator additionally exposes
`execution_scope(*, mode, valid_tokens, dummy_run)`. It observes the actual
native request state and dispatch before input preparation. Only after global
mode agreement does it select the existing EAGER scope or FULL graph preparation;
the native `execute_model` remains the caller and sole `finish_execution` owner.
Its constructor and storage checks also require
`owner.hccl_v41_transport is control.transport`.

FULL replay reuses the cached bank and graph, updates actual prepared counts
outside replay, and temporarily hands the first child's metadata to the native
V4.1 SFA/DSA lifecycle context. Each child retains its recorded metadata. This
is restricted to the pinned no-op graph-update backends; it does not flatten
children or claim generic updatable-backend support. Native replay must execute
exactly once before COMPLETE, and the metadata field is restored on success or
failure. Zero-row native early returns accept and retire an honest empty parent
without inventing request state or consuming an Engram producer.

Independent review identified a cache-bundle ownership defect: a replacement
resource bundle could point at a new bank while the retained graph body still
referenced the old bank. The final code rejects such bundles before preparation
or FORWARD. It checks the captured execution's bank, actual parent tensor body's
bank/model, control/model/state/session identity, frozen native storage and
outputs, key/generation, and the native manager's graph identity. Public-cache
reserve/publish replacement cases cover these rejected combinations.

Final source hashes:

- `afd_plugin/v1/worker/npu/hccl_v41_graph_coordinator.py`:
  `b36558a1ca4d92d67628e17ac79736e76acea549f9d9b4139859a47af630e0b5`.
- Mirrored test:
  `77e491a82efa18184818aea351ba7b5ccc62caf170d6822b16826dd6bc07e425`.
- Shared coordinator fake:
  `16ad4a9510228f3daaea76ad278394bd402f25cb2942b4acbf3abb5305eaba97`.
- Shared native geometry helper:
  `7e7f56c640b01d7e101b39747096ee520f46e08370b7deac7a22426ac595015a`.

A3 snapshot `a3-control-cpu-20261109/runtime-coordinator-015` ran the coordinator
mirror with the installed runtime dependencies: **54 passed, 14 warnings,
0 failed/error/skipped, 2.46 seconds**, exit 0. Its manifest is
`d385badf03b81391f3bd1ab2efee70ef2fa2e7cb66741165ffa2621ac4f190fb`.
The local final four files above match its identity manifest byte for byte.
Raw command, source check, stdout, XML, and summary remain under that snapshot.
This targeted run did not repeat the broad suite or start a device probe.

The runtime tests exercise native CPU preparation, actual control codec/state,
retained resources, and host replay boundaries. Nonempty EAGER transition tests
stop before eager input preparation; their numerical execution is covered by
the separate C4 module rather than attributed to this target. Inert CPU graph
handles provide no NPU capture/replay, stream, attention, memory, or checkpoint
acceptance. Pre-KV EAGER bridging and new main-runner bindings require their own
subsequent snapshot; no profile epoch is needed by the pinned normal NPU startup.
