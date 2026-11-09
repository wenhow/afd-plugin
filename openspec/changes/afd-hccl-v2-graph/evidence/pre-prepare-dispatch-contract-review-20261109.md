# C5 pre-prepare dispatch contract review — 2026-11-09

Status: source-grounded design review. This document changes no runtime code,
tests, native lifecycle, feature admission or task completion state. It complements
[the manager control proposal](manager-control-contract-proposal-20261109.md).
The existing native NPUGraph tensor-component results do not validate this
proposed native model/manager integration.

## Source and ownership

All line references below refer to the locked Git objects, inspected with
`git show <commit>:<path>`, rather than the mutable upstream worktree contents:

| Source | Commit | Relevant symbols |
| --- | --- | --- |
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | `vllm/v1/worker/gpu/model_runner.py`: `execute_model`, `gather_batch_req_state`, `BatchReqState`, `prepare_attn` |
| vLLM | same | `vllm/v1/worker/gpu/cudagraph_utils.py`: `BatchExecutionDescriptor`, manager `dispatch`, `run_fullgraph`, capture factory |
| vLLM | same | `vllm/v1/worker/gpu/dp_utils.py`: `dispatch_cg_and_sync_dp`, DP U2 graph dispatch |
| vLLM | same | `vllm/v1/worker/gpu/ubatch_utils.py`: `create_ubatch_slices`, `_slice_input_batch`, `UBatchRunner.prepare` |
| vLLM | same | `vllm/v1/worker/ubatch_utils.py`: `maybe_create_ubatch_slices`; `gpu/block_table.py`: persistent block/slot storage |
| vLLM-Ascend | `7e37e29a7803d6897a7113088e2b771f83205bc0` | `vllm_ascend/worker/v2/model_runner.py`: selected gather override, `prepare_inputs`, FIA query padding, slot extension |
| vLLM-Ascend | same | `vllm_ascend/models/deepseek_v41/engram/model_state.py`: Engram mode and pending producer |

Native request-state updates and staged block-table writes already occur at
`execute_model:1641–1648`. The proposed boundary is before per-forward input,
attention metadata and Engram preparation, not before every native state effect.

## The usable early boundary

The actual native call chain is:

1. `execute_model:1660–1662` calls the selected instance method
   `self.gather_batch_req_state(scheduler_output, dummy_run)`.
2. `1685–1700` calls the module-imported `dispatch_cg_and_sync_dp`, obtaining the
   descriptor that controls every subsequent preparation branch.
3. Real inputs start at `1714`; dummy inputs at `1740`; child metadata preparation
   at `1790`; parent ModelState input preparation at `1862`.
4. FULL calls manager replay directly at `1901–1909`. Eager creates its parent
   forward context only in the separate branch at `1918`.

Thus a final mode decision made at the current C4 eager parent-context provider
is too late. Engram `prepare_engram_inputs:130–141` takes the FULL graph-input
route without creating an eager pending producer. A later NONE decision cannot
repair this by changing only the forward context.

The smallest viable hook is inside the existing execute scope: retain the
**actual return** of the selected instance gather method, then wrap the native
module's imported dispatch alias. Return the globally selected descriptor from
that wrapper. Do not copy `execute_model`, call gather a second time, or infer
the native request order from the scheduler dictionary.

The selected Ascend method has the exact signature, including its absent return
annotation:

```python
def gather_batch_req_state(self, scheduler_output: SchedulerOutput, dummy_run: bool):
```

It delegates the GPU method at `gpu/model_runner.py:1217–1267`, whose signature is:

```python
def gather_batch_req_state(
    self, scheduler_output: SchedulerOutput, dummy_run: bool
) -> tuple["BatchReqState | None", int | None]:
```

The GPU method explicitly sorts requests before creating the CPU scheduled-token
array. `BatchReqState:2318–2330` retains `req_ids`, `num_scheduled_tokens`,
`num_tokens`, `idx_mapping_np`, prefill/history coordinates and `has_prefill`.
The Ascend override at `615–638` only adds the optional PD recompute correction,
which the initial C5 scope excludes. Recording its real return preserves even
that selected override's ownership.

The dispatch wrapper must keep the exact `dp_utils.py:214–228` signature and
return annotation. Its delegation exception is justified by native graph
selection/DP ownership, not convenience. Both wrappers belong in the existing
single signature-alignment table. Bindings must be restored in `finally`, with
no additional global registry or lasting class mutation. The present
single-execute scope is the supported concurrency boundary; concurrent runner
dispatches in one process would require a different upstream hook.

## What is knowable before input preparation

For the initial graph scope, require text, BF16, spec off, no LoRA/PD/EP,
DP1/PP1/CP1, one paired native TP world per role, FULL_DECODE_ONLY, uniform decode
query length one. Unsupported candidates choose eager before preparation.

For a real batch in that scope, let `state` be the recorded `BatchReqState` and
`desc` the correctly selected native candidate:

| Quantity | Exact source before preparation |
| --- | --- |
| Actual request count | `len(state.req_ids)` |
| Actual tokens | `state.num_tokens` |
| Native ordered scheduled counts | `state.num_scheduled_tokens` |
| Padded tokens | `max(state.num_tokens, desc.num_tokens)` |
| Initial padded requests | `max(len(state.req_ids), desc.num_reqs or 0)` |
| FULL uniform padded requests | `desc.num_reqs`, when all query lengths are one and `desc.num_tokens == desc.num_reqs` |
| Stage token/request bounds | Native common `maybe_create_ubatch_slices` plus the native GPU request-bound clamp |
| Logical valid rows | Intersection of each token slice with `[0, state.num_tokens)`; zero for a dummy parent |
| Padded stage rows | `token_slice.stop - token_slice.start` |

The FULL padded-request statement is not a generic property of Ascend inputs.
Ascend `prepare_inputs:727–739` preserves the recorded scheduled vector with
speculation disabled, and its FIA helper at `1158–1195` preserves descriptor
request padding only when the uniform query and padded token/request shape
match. Mixed/varlen/adaptive/PCP paths can have different padding and are outside
the first supported graph scope.

For U2 call the existing native common helper with the recorded vector and the
chosen padded dimensions:

```python
_, slices = maybe_create_ubatch_slices(True, scheduled, padded_tokens,
                                      padded_requests, num_ubatches)
```

Then apply the two exact request-bound clamps from native
`gpu/ubatch_utils.py:66–75`: start is at most `actual_requests - 1`; stop is at
most `padded_requests`. This is a small explicit native layout adaptation, not
an independently invented midpoint algorithm. U1 uses one token prefix and one
request prefix. A typed preview should retain both request and token bounds;
`HcclLocalForward` alone is insufficient for a model graph key.

Dummy execution has no `BatchReqState`. Profile remains eager. For uniform decode
warmup/capture, the capture callback already receives the full descriptor;
`InputBatch.make_dummy:138–148` has a deterministic CPU scheduled-vector formula.
With query length one and `num_tokens == num_reqs`, it is all ones. Generic dummy
query distributions should remain unsupported until their native formula is
explicitly covered; do not construct a device `InputBatch` merely to preview it.
Native capture uses its own factory and does not pass through execute dispatch.

## Two graph selection details that must not be missed

First, the existing C4 DP1 threshold scope changes `num_ubatches` after native
dispatch. Native manager dispatch defaults to U1 and its compatibility predicate
checks U exactly (`cudagraph_utils.py:115–128,496–526`). Renaming a returned FULL
U1 descriptor to U2 does not select a captured U2 graph. The C5 path must determine
native U2 eligibility, then actually dispatch with `num_ubatches=2`, as native
DP U2 does at `dp_utils.py:118–129`. Keep `need_eager`/profile forcing NONE.

Second, the graph key must compare actual static stage request shapes as well
as token rows. For query-one decode with real requests/tokens four, padded
requests/tokens eight and U2, the native request clamp makes the second child
request slice `3:8` (five padded request rows). Capturing the eight-request dummy
shape makes that slice `4:8` (four rows). The same parent token bucket and U
therefore do not establish matching child metadata shapes. This candidate must
miss or use an explicitly captured matching layout. The native DP U2
`2 * min_tokens < bucket_tokens` check is not a replacement for this equality.

On a global miss, prefer the native eager descriptor form with actual token and
request counts and the native threshold-selected U. Recompute its preview before
preparation. Merely replacing FULL with NONE while retaining graph request/token
padding is a separate behavior requiring tests; it should not be an unproven
shortcut in the first integration.

## Slots, persistent addresses and TP

Early agreement needs slot layout and storage generation, not current slot IDs.
Native `BlockTables` allocates persistent input block tables and slot mappings at
`gpu/block_table.py:72–83`. `prepare_attn:1478–1491` refreshes their contents and
returns fixed prefix views. Dummy variants explicitly retain those addresses
(`block_table.py:178–189,224–233`). Ascend slot extension at `916–920` selects the
same persistent padded prefix.

The preview therefore knows a stage slot shape from token bounds and the KV
group count, and each stage block-table shape from request bounds and the
configured table width. Native preparation must still calculate every current
value. A graph cache entry must own its fixed views/events and be invalidated
when their storage generation changes. Reallocation or different view shape
cannot be covered by a token-count-only key.

All native TP Attention coordinates must contribute their own preview. Existing
`HcclStageLayout` rejects unequal valid/padded token counts within a native TP
world (`distributed/hccl.py:141–144`). C5 must additionally agree the descriptor
and static request/query layout, then pair each Attention coordinate with its
matching FFN coordinate. An Attention disagreement is a state/configuration
error; an FFN cache miss is a legitimate all-peer eager decision. A0's local
counts cannot stand in for the other coordinates.

## Concrete implementation boundary

Keep the existing `HcclV41Execution` as the owner of one execute invocation:

- Record the returned native batch state in its scope's instance-method wrapper.
- In a scoped exact-signature dispatch wrapper, select the correct U candidate,
  derive its typed layout, ask the graph/cache controller for an all-peer decision,
  and return the chosen native descriptor and native DP sync result.
- Keep actual native input/child metadata preparation unchanged. Compare the
  resulting `UBatchState.slices` and descriptor-derived U1 layout with the reserved
  layout before any device launch. A mismatch is terminal; do not prepare again.
- Preserve one native Engram parent preparation and Ascend's sole
  `finish_execution`. FULL replay needs a separate graph-manager owner hook
  because the eager parent context does not exist on that branch.

A useful proposed owner method is
`select_dispatch(candidate: BatchExecutionDescriptor, *, num_reqs: int,
num_tokens: int, uniform_token_count: int | None, need_eager: bool)
-> BatchExecutionDescriptor`. It consumes the recorded state and invokes the
controller; this is a design signature, not a claim of an existing API.
The typed candidate/decision and its wire serialization remain the transport
owner's responsibility. Agreement must be solicit-first and STOP-compatible.
It may reserve early then verify actual preparation, or combine those phases
once the complete early preview is tested; this review does not freeze an
unnecessary second network round.

Suggested file ownership: execute scope and preview in
`hccl_v41_execution.py`; exact dispatch compatibility in the existing NPU V2
seam; graph callback/retirement integration with the model/UBatch runner owner;
phase/cache agreement in staged transport and the FFN service. Mirror each
changed module's unit tests; extend the sole upstream-signature table.

CPU validation should compare previews against the actual native slice helper
for ordered/misaligned/odd/padded/padding-only inputs, verify a real selected
gather return reaches agreement before native preparation, and verify a global
miss prepares eager Engram exactly once. Include U1→U2 transitions, same token
bucket with different request slices, forced profile eager, and restoration on
failure. These are software contracts. NPU manager capture/replay, real Engram/
attention/MoE, device metadata address stability, and full checkpoint comparison
remain separate required acceptance.

Cost: retaining the already-produced CPU state introduces no device readback.
Preview uses O(requests + stages) host work and bounded layout serialization;
control agreement adds a host round trip before preparation. Native TP/group
creation and per-forward device buffers need not be duplicated. The removal
condition is an upstream pre-input-preparation dispatch callback carrying the
native ordered CPU batch state and a supported per-manager capture/replay hook.
