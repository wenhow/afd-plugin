# C5 native model Graph stage routing audit, 2026-11-10

This read-only audit identifies the next model-specific work after the native
base-manager, phase protocol, geometry, controller and early-dispatch components.
It does not change runtime code or promote those components to V4.1 model Graph
support. The production feature guard remains closed.

The inspected native objects are vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607` and vLLM-Ascend
`7e37e29a7803d6897a7113088e2b771f83205bc0`. File references below use those locked
Git objects, not the current upstream working tree.

## Backend route: V4.1 does not need a generic task-source patch

`vllm_ascend/utils.py:1907-1912` selects UpdatableGraph only for subclasses of
`AscendAttentionBackend`. `AscendSFABackend` directly inherits AttentionBackend
at `attention/sfa_v1.py:365`; V4.1's `DeepseekV41CacheBackend` does too, at
`attention/dsa_v41.py:1433`. Their update methods explicitly require no pointer
patch: SFA `update_graph_params` at 1159-1168 is a no-op, and
`AscendDSAV41Impl.update_graph_params` at 692-694 says the model owns stable
metadata buffers.

Therefore the V4.1 route is
`ModelAclGraphManager.run_fullgraph` → `_graph_relay` → native replay → the
backend's no-op `update_graph_params`, not `_updatable_graph_replay` →
`resolve_tasks(ContextSource(...))`. This distinction applies even when the
selected executable backend comes from a V4.1 cache resource rather than SFA.
Changing ContextSource would not implement V4.1's missing model capture path.

For the separate FIA/PA UpdatableGraph route, the suspected stage collision is a
real contract limitation, but not silent replacement by the last child:

- `GraphUpdateTask.provider_index` and `provider_sizes` at
  `compilation/updatable_graph.py:120-154` count repeated equal providers.
  FIA/PA providers are frozen value records keyed by layer and operator options
  (`attention/attention_v1.py:480-524`), without a stage coordinate.
- A synchronous U2 capture recording the same provider twice has size two.
  `ContextSource.get` at 30-37 returns exactly one resolved mapping, so the
  native `len(params_by_provider[provider]) == size` assertion fails. Reading
  one parent/last-child dictionary cannot satisfy that contract.
- The active UpdatableGraph is a Python ContextVar. Native UBatchRunner starts
  ordinary threads in `gpu/ubatch_utils.py:494`; this path does not explicitly
  copy that capture context into them. Threaded task registration therefore
  cannot be assumed to work by merely adding a replay source adapter.

If a future generic backend needs this path, capture must record provider-to-stage
identity, and replay must resolve that exact identity from the corresponding
child context. A stage-tagged provider plus a source that returns its own child's
mapping avoids guessing multiplicity or registration order. That is separate
work; it should not be added to the initial V4.1 SFA/DSA implementation.

## Existing stage storage is the correct native base

Native UBatchRunner owns stable per-child query and sequence-length buffers
(`gpu/ubatch_utils.py:304-319`), slices the actual native batch, and carries each
child metadata mapping in its own ForwardContext (`329-387`). It deliberately
does not replace these mappings with a single parent dictionary.

AFD's existing `NPUUBatchRunner` already creates independent metadata builders
for each stage, stable child block-table buffers, and persistent stage RoPE
buffers. These are meaningful prerequisites for SFA/DSA replay, not reasons to
flatten metadata back into `model_state.attn_metadata`.

The native V4.1 metadata builder owns persistent `_seq_lens`, slot-coordinate
and flat-slot buffers, SMLA/QLI metadata, C2 ring/complete/source-position buffers,
and A5 window/length buffers (`attention/dsa_v41.py:779-813`). `build` updates
their contents and returns views (`908-933`, `935-1020`, `1395-1429`). The SFA
NoPE builder likewise retains its block table and operator workspace buffers
(`attention/sfa_v1.py:261-334`). These builders and the exact stage views must
outlive the graph; replay preparation refreshes their contents in place.

The cache's storage generation must cover the full retained bundle: native input
buffers, child query/length buffers, stage block tables, RoPE, V4.1 builder
buffers, shared-attention slices, Engram buffers/events, exchange buffers and
native graph/output objects. Matching shapes alone cannot prove address
stability. Runtime geometry remains independently checked against capture
geometry; the real-R4/padded-T8/U2 request-boundary mismatch must still select
fresh eager before preparation.

## Missing capture entry and minimal owning hook

`ModelCudaGraphManager.capture.create_forward_fn` calls
`ubatch_runner.prepare(..., cg_mode=FULL, for_capture=True)` and then
`begin_capturable_run(...)` directly at
`gpu/cudagraph_utils.py:673-695`. It does not call `ubatch_runner.run`.
Consequently the existing eager HCCL `run` override is not a model capture
integration. `HcclV41UBatchRunner.prepare` currently rejects FULL/capture, and its
inherited NPU `begin_capturable_run` starts the native full-model child threads.

The smallest actual model entry is a controlled HCCL runner override with the
exact native contract:

```python
def begin_capturable_run(
    self,
    model: Any,
    model_inputs: dict[str, Any],
    ubatch_state: UBatchState,
    for_capture: bool = False,
) -> Callable[[], Any]: ...
```

For the accepted graph path this returns one owning-thread callback, without
starting the full-model handoff threads. The existing native factory still
calls that callback once and writes its result into native persistent hidden
outputs. Allocation, exact static geometry, group/endpoint resolution and early
phase acceptance happen outside the returned tensor body. The runner needs a
separate FULL preparation branch that preserves the parent's Engram mode until
native ModelState input preparation; changing the eager mode to FULL after
preparation is invalid.

The native model wrapper and lifecycle remain in the call chain. In particular,
`ModelAclGraphManager.capture` wraps the model with ModelWithContext, whose
forward calls `device_metadata.begin_forward()` and finally `finish()` once
around the owning parent (`worker/v2/aclgraph_utils.py:273-300`, `315-337`). A
new synchronous body must not invoke that wrapper separately for each child.
The native output factory, capture pools, graph registration and replay/output
slicing should remain delegated to the native managers.

## A5 device-metadata tasks need one parent build frontier

The native runner installs `TargetDeviceMetadata` only for A5 packed-cache V4.1
(`worker/v2/model_runner.py:418-422`). Native ModelState uses its `run_build`
around attention preparation; AFD's child preparation currently directly calls
the metadata builder instead.

Naively invoking the same TargetDeviceMetadata `run_build` once per child is
incorrect: `TargetDeviceMetadata.build` rejects a second build while `_tasks` or
an earlier submission is retained (`worker/device_metadata.py:197-220`). It
does not accumulate independent child builds automatically.

A minimal native-preserving route is one parent
`device_metadata.build(all_stage_attn_groups, full_graph=True)` around both
children's actual native preparation. The combined groups retain their distinct
builders; task group identities use their persistent buffer identity. Each
returned V4.1 metadata resource also receives that same native executor, as
`run_build` normally does. The one native ModelWithContext callback consumes and
joins the combined tasks; native replay calls `finish_replay` without resubmitting
them on the host. Preserve the builder's C2 policy: the native in-graph path
explicitly keeps compressor preparation outside the graph.

This route requires a reviewed integration patch and real A5 component
acceptance. A3 base Graph success cannot validate A5 packed-cache task routing.
The initially simpler synchronous metadata path must be explicitly distinguished
from the native asynchronous producer route if retained during development.

## Engram must remain one parent producer

EngramModelState FULL input preparation at
`models/deepseek_v41/engram/model_state.py:130-141` fills persistent parent query
boundaries and a device valid count, and intentionally does not create the eager
pending producer. There is no valid late switch to eager after that branch.

For the native supported captured-producer layout,
`AscendDeepseekV41LLMForCausalLM.forward` enters
`model.captured_engram_inputs(...)` once at `model.py:1754-1765`.
`can_capture_engram_producer` at 1297-1311 explicitly requires slotless history,
overlap enabled, non-SP, PP1, and local Engram table DP/TP sizes of one. That is
the actual native eligibility test; an A/F transport TP pairing is not proof of
Engram producer eligibility.

The synchronous AFD parent body should enter this same captured context once
using unsliced parent IDs, positions, lookback, query boundaries and valid count.
It then gives each child token views of the produced lookup/mask buffers. The
native context records hash/lookup production on its auxiliary stream, publishes
ordinary events and joins in its `finally` (`model.py:1313-1360`). It preserves
request history across a boundary; calling it once per child would duplicate
hash work and use the wrong request/lookback coordinates.

The native ordinary-event path supports multiple waits without resetting the
event. The external-producer fallback has a different contract:
`AscendParallelEngramEmbedding.wait_lookup(..., external=True)` performs both
wait and reset (`engram/embedding.py:358-365`). Sending the same external event
to both children and independently resetting it would consume one producer
record twice. Either the parent waits/resets once before all stage consumers of
that mask/layer, or a separately validated producer supplies stage-specific
events. For a first model Graph increment, requiring the native captured-producer
eligibility (or no Engram) and choosing early eager otherwise is the smaller
correct route. External-event FULL support must not be inferred from the eager
driver's plain-event sharing.

Native Ascend execute remains the sole runtime `finish_execution` owner
(`worker/v2/model_runner.py:514-530`). It clears pending inputs and retires lookup
work with the failure flag. Capture has its own native ModelWithContext and
manager cleanup; do not invent an execute retirement inside every child or
capture callback. On replay, native ModelAclGraphManager's Engram and device
metadata lifecycle must remain around the single whole-parent graph.

## Tensor-body boundary and implementation order

The current eager `run_deepseek_v41_stages` cannot simply run inside capture. It
uses Python plan-valid counts to zero IDs, eager transport methods with host
cursors, and eager-only contexts/events. A graph body must instead use static
stage shapes and agreed endpoints, fixed exchange storage, tensor valid counts
and tensor masks. All positive static stages record the same layer/stage tensor
sequence even when capture is dummy or a replay has no valid rows. Preserve
independent FP32 router input, BF16 activation/output and int32 IDs. A tensor
callback may record deterministic tensor operations; it must not send host
control packets, advance eager protocol state, synchronize the host or decide
fallback while capture is active.

Concrete next changes, in dependency order:

1. Add the controlled synchronous `begin_capturable_run` and FULL preparation
   branch for the owning HCCL runner. Keep eager behavior and admission guard
   unchanged; retain the exact native capture-factory caller and output owner.
2. Add one parent graph-tensor driver consuming existing decoder continuations,
   native child contexts and stable Attention/Engram views, plus static graph
   exchanges. Keep its static tensor body separate from early phase/counts and
   COMPLETE control. F needs its corresponding actual FFN tensor body; the
   current arithmetic graph callback is not that body.
3. Integrate the parent metadata-task build and single Engram captured context,
   with explicit early eager fallback for unsupported producer layouts.
4. Bind the real ModelAclGraphManager capture/replay entry to those owners, then
   validate metadata address stability and changing contents across U1/U2,
   padding/empty replay, parent-event reuse and exceptions. A3 can exercise
   native non-model components; A5-specific cache/metadata/operators and full
   checkpoint output still require their separate hardware evidence.

No production change or device experiment was performed for this audit. The
already passed native base-manager probe demonstrates its own graph/exchange
component only, and does not close any missing model-specific item above.
