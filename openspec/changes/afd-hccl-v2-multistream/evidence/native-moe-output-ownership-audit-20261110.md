# Native MoE output ownership across deferred Graph sends

Date: 2026-11-10. This is a read-only source audit requested before freezing the
bulk delivery and before implementing `cross_layer_v1`. No runtime or upstream
source is changed and no device experiment is performed by this audit.

**Conclusion:** no definite fixed-output-workspace overwrite was found in the
locked, non-EP native path. With the checkpoint's separate shared experts,
Ascend's non-in-place shared/routed add produces the final per-call tensor.
However, the plugin's general `compute_ffn_output -> Tensor` seam does not
require an owned, immutable result. Its deferred bulk sends must not rely on
Python references or `record_stream` to prevent a native implementation from
rewriting borrowed storage. Stage-owned output copies are a small, robust way
to remove that unexpressed assumption before full-model acceptance. This is a
contract gap and defensive fix recommendation, not a claim that the actual
checkpoint or the successful arithmetic probe has been observed corrupting
data.

## Locked source examined

All upstream reads used `git show <locked SHA>:<path>`, not the mutable checkout
contents. Ascend is `7e37e29a7803d6897a7113088e2b771f83205bc0`; vLLM is
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`.

| Source | Blob content SHA256 |
| --- | --- |
| Ascend `models/deepseek_v41/model.py` | `989aec6db1e344dca76159dcbef0d5876874facedaa36b47e993146d5b8a2b9f` |
| Ascend `ops/fused_moe/fused_moe.py` | `29a02e758cd347cd9382e86f214d64c0feea7c2df4ff559c293b9e5d6c16344d` |
| Ascend `ops/fused_moe/routed_experts.py` | `568ad7db6547c3eed6e1c286d53af4f0a558ad71a107eb5d694b4e4d345fe56d` |
| Ascend `ops/fused_moe/moe_comm_method.py` | `7040f691355bd9acb3c668e698de3f7d3854edbc22e3f979f9b03d99800364d7` |
| Ascend `ops/fused_moe/token_dispatcher.py` | `1dcfd8181cb2d34c584d8336c3379dbeefba75c0bf664f924063701b93295cbb` |
| Ascend `ops/fused_moe/prepare_finalize.py` | `dff4fac292200fc06f60aef48505f1ee38314ea5adb53560b325901db308eb24` |
| Ascend `device/device_op.py` | `a603e21fff2dabc06877fe6a1e52fe3637dec316b1a4cae7228a54037ec9766a` |
| Ascend `distributed/device_communicators/npu_communicator.py` | `90aaa39c4d53caecb5493ba00a45d953e89992cff5d2d5996388781aa302d5a8` |
| Ascend `ascend_forward_context.py` | `848112214c9f56371b4bf43a59d39fadbed0e0e864581e300a3a89778d4083cb` |
| vLLM `model_executor/layers/fused_moe/runner/moe_runner.py` | `05893188402fe37229bf72d75088d66193c417d5d502daafd3a55e1bfedaa60c` |
| vLLM `distributed/device_communicators/base_device_communicator.py` | `4a7670f12ed7c7621b25d476e2687c24ca01f3be183fff4a6a25337596fe506b` |

Paths prefixed “Ascend” are relative to `vllm_ascend/`; “vLLM” paths are relative
to `vllm/`. The separate quantization audit covers the MXFP4 MLP operator path.

## Concrete call chain and what it proves

1. Plugin `AFDDeepseekV41FFNForCausalLM.compute_ffn_output` passes the original
   BF16, FP32 and IDs directly to its selected layer's native `mlp`. It does not
   allocate an owning output or specify a borrowed-output lifetime.
2. Native `DeepseekV41MoE.forward`, model.py:274–340, passes through its experts,
   then returns `final_hidden_states.view(num_tokens, hidden_dim)`. The view
   does not establish fresh storage. A retained view protects allocation
   lifetime, not value stability under writes through another alias.
3. `ascend_forward_context.py:397–398` selects ALLGATHER when EP is disabled or
   the EP group has size one. Current graph admission excludes EP, so MC2 and
   fused-MC2 fixed buffers are not a justification for a confirmed bug in this
   supported branch.
4. `AscendMoERunner._forward_impl`, fused_moe.py:494–549, calls
   `AscendRoutedExperts.forward_impl` and, when enabled, the native shared
   experts. This overrides the generic vLLM `_forward_impl`; a generic CUDA
   modular-expert workspace must not be assumed to be this NPU return path.
5. `routed_experts.py:724–789` prepares, applies the selected quant method,
   obtains `FusedExpertsResult.routed_out`, and finalizes it. The shared
   communication object is long-lived, but this Python code does not assign a
   persistent output tensor or return a named fixed output workspace.
6. `MoECommMethod.fused_experts`, moe_comm_method.py:154–194, dispatches,
   computes MLP, and combines. ALLGATHER's `token_combine`, token_dispatcher.py:
   429–439, returns `DeviceOperator.npu_moe_token_unpermute(...)`, optionally a
   view. The A2/A3 and A5 adaptors at device_op.py:992–995 and 1976–1979 call
   the functional torch-npu interface without an `out` or plugin-supplied
   workspace argument. This is not visible evidence of fixed output aliasing.
   Its binary/kernel implementation is not contained in these locked sources.
7. DP1/CP1/non-SP `PrepareAndFinalizeWithAllGather` returns that tensor without
   an allocating gather, at prepare_finalize.py:552–569. Thus this step alone
   does not remove possible upstream aliases.
8. With separate shared experts, `AscendMoERunner.forward` uses the functional
   `torch._foreach_add([shared_output], [fused_output])[0]` at fused_moe.py:
   240–241. It is not `_foreach_add_`. This produces a per-call output rather
   than returning the routed tensor directly. With no separate shared output,
   line 243 returns `fused_output` instead, and that stronger ownership argument
   does not apply.
9. The current checkpoint inventories, including the returned A5 preflight,
   record `n_shared_experts=1`. Native model.py:205–207 can nevertheless disable
   separate shared experts under `mix_placement`; ownership must be stated
   conditionally on the actual native configuration, not just the checkpoint
   field. The graph binder does not express a universal separate-shared-output
   requirement.
10. For native TP, Ascend's `NPUCommunicator` does not override `all_reduce`.
    Locked `DeviceCommunicatorBase.all_reduce`, at 231–233, calls
    `dist.all_reduce(input_, ...)` and returns the same input. This does not
    replace the per-call combined result with a Python-visible fixed output
    buffer. Native model.py:335–338 only adds its legacy tuple-output reduction;
    the normal tensor result has already passed through the native runner's
    final reduction. No extra reduction is appropriate in the plugin.

The strong source-level argument for the ordinary separate-shared-expert
configuration is the functional final add, not the lifetime of the MoE return
view. No universal ownership proof is established for all allowed quantization,
mix-placement or opaque operator branches. There is no demonstrated corruption
to attribute to a particular CANN or torch-npu kernel.

## Why the deferred-send seam needs an explicit contract

Bulk enqueues C0 and C1 before copying either native result into the send bank.
Cross-layer would also keep C01 and C10 pending concurrently. A legal callable
at the current `Tensor` return seam can return a view of a reusable workspace.
On one compute stream, C1 can overwrite that workspace after C0 returns but
before the parent performs the deferred C0 copy. Both Python references would
then observe C1's values. `record_stream` prevents allocator reuse; it does not
prevent writes by the original owner of still-live storage.

The controlled arithmetic probe returns separately owned outputs. Its passing
results therefore do not test this counterexample or prove native output
ownership. The issue is relevant to bulk's model seam even though the current
checkpoint's usual final-add path provides a stronger result.

## Minimal defensive preparation

For the new bulk/cross deferred-send paths, copy each native output to its
already allocated `bank.buffers(stage_idx).output` **inside the corresponding
compute scope**, before its `compute_done` event is recorded. Retain the fixed
bank tensor as the pending result. Native C1 may then overwrite a reusable
workspace only after C0's copy on the same compute stream has consumed it.
No extra buffer, TP collective, model call or tensor wire packet is required.

Preserve source validation before this copy: exact static shape, BF16 dtype,
same device and contiguous layout. An unchecked `Tensor.copy_` would silently
broadcast or cast an invalid native output and bypass the present
`bank.send_output` rejection. A small public bank method can share this
validation/copy with `send_output`; it should not mask or send in the compute
scope. Later parent `send_output` may copy the bank to itself, apply the existing
padding mask, and send the same fixed address after waiting compute_done.

The existing same-stage chain remains the bank-reuse proof:

```text
native compute -> stage-owned output copy -> compute_done
  -> parent padding mask/send -> send_done
  -> next same-stage receive/reuse fence -> next compute/copy
```

The S0/S1 output banks are already disjoint. Warmup/capture resources retain
the same bank addresses through replay. The native return tensor remains held
and recorded through its immediate compute-stream copy, but its values are no
longer required at the later parent send. Success, partial failure and final
join ownership must still follow the existing pipeline contract.

The additional copy moves the existing native-to-bank copy earlier. The later
self-copy can be kept for a small first fix; a measured optimization may
remove it after semantics and graph behavior are checked. Do not invent a new
copy-free guarantee from pointer inequality alone.

## Verification required for that fix

A shared fake native FFN should overwrite and return the **same** workspace
on consecutive calls. Its two stage results must remain different and exact
at the peer's public output after deferred sends. Add a failure during the
second call, malformed output shape/dtype/layout cases, and an additional
layer to exercise same-stage reuse. These are meaningful CPU protocol/output
seams, not NPU workspace evidence.

Then rerun the affected F Graph mirror and the bounded bulk NPUGraph component
with the new output-copy placement. Preserve the original passing device raw
as an older snapshot. Native MXFP4 MoE/full-checkpoint behavior and real
overlap remain separate A5/timeline acceptance items.

## Authorized defensive implementation follow-up

The parent approved the defensive preparation without claiming an observed
native kernel bug. `HcclGraphStageBank.prepare_output` now preserves the exact
old output validation and returns the copied fixed stage tensor. `send_output`
reuses it; the default serial path still copies once. Only bulk F compute
records the actual native-output consumer and prepares that output inside the
compute scope, before the existing compute_done event. The previous native
weak-reference lifetime assertion now checks the retained fixed-bank values
that a failed phase actually needs.

The independent actual-dependency CPU snapshot `graph-owned-output-024`
completed **114 passed**, no skips/errors/failures, 14 warnings: bank 41 and
F Graph 73. Its full hashes, source checks and raw XML/logs are indexed in
`a3-graph-bulk-20261110/`. The controlled probe's bulk compute uses matching
preparation. Its independent `bulk-owned-output-002` NPU session then passed:
both ranks exited zero after 4 warmup, 4 capture, 12 replay and 6 eager
transactions; 31 installed source pins matched, all eight devices were idle
before and after, and no timeout, cleanup or kill occurred. The device snapshot
manifest is `17b6cb1d0c5c92d60679ccb89e62da357aaeb90f08180d99d6a82929614a50cd`.

The new raw and rank/address/oracle checks are indexed separately from the
earlier `bulk-tp1-001` snapshot. CPU shared-workspace coverage proves the
defensive seam; the NPU arithmetic probe proves the copy's graph placement.
Neither establishes native MoE workspace reuse, full-model acceptance or
overlap performance.
