# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in DS MXFP4 MoE scale packing correction."""

import torch
import torch_npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.moe_utils import (
    cumsum_group_list,
    maybe_normalize_mxfp_scale_layout,
)
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
    SITU_MX_DST_TYPE_E4M3FN,
    AscendW4A8MXFPDSDynamicFusedMoEMethod,
)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, dispose_tensor

from afd_plugin.compat.patches.npu.mxfp_scales import (
    align_mxfp4_weight,
    align_mxfp_gmm_input,
    align_mxfp_swiglu_input,
    pack_mxfp_scale_pairs,
)


# Upstream: vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py,
# AscendW4A8MXFPDSDynamicFusedMoEMethod @ 7e37e29a (identical at 2db8549c).
# Patch reason: TP8 routed-expert W2 has K=288, hence nine scale columns;
# upstream truncates the pair count and cannot reshape the loaded tensor.
# Patch functionality: pad the unused final scale byte for W13/W2, preserving
# valid FP4 values, logical dimensions, NZ conversion and the native EPLB branch.
# W2's physical K is zero-padded to 64 for the target WeightNz kernel.
# Signature matches upstream, including absent annotations; no added args.
# Cost: loading-only W2/scale allocations for unaligned K; apply_gmm2 separately
# pads activations at runtime. K=288 adds 16 packed weight bytes per output row.
# Removal: delete when pinned DS MoE supports odd pairs and WeightNz K alignment.
def process_weights_after_loading(self, layer):
    # ### PATCH START: align physical W2 K before packed/list NZ conversion
    layer.w2_weight.data = align_mxfp4_weight(layer.w2_weight.data.view(torch.uint8))
    # ### PATCH END: align physical W2 K before packed/list NZ conversion
    if self.dynamic_eplb:
        self._build_expert_lists(layer)
        return
    layer.w13_weight.data = torch_npu.npu_format_cast(
        layer.w13_weight.data.view(torch.uint8),
        ACL_FORMAT_FRACTAL_NZ,
        customize_dtype=torch.float8_e4m3fn,
        input_dtype=torch_npu.float4_e2m1fn_x2,
    )
    layer.w2_weight.data = torch_npu.npu_format_cast(
        layer.w2_weight.data.view(torch.uint8),
        ACL_FORMAT_FRACTAL_NZ,
        customize_dtype=torch.float8_e4m3fn,
        input_dtype=torch_npu.float4_e2m1fn_x2,
    )
    layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
    layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
    # ### PATCH START: preserve odd W13 and W2 scale columns
    layer.w13_weight_scale.data = pack_mxfp_scale_pairs(
        layer.w13_weight_scale.data
    ).transpose(-3, -2)
    layer.w2_weight_scale.data = pack_mxfp_scale_pairs(
        layer.w2_weight_scale.data
    ).transpose(-3, -2)
    # ### PATCH END: preserve odd W13 and W2 scale columns


# Upstream: same class/SHA, static method _process_scale_2d(scale_2d).
# Patch reason: the native EPLB expert-list path has the same odd-column bug.
# Patch functionality: use the same lossless pairs for individual experts.
# Signature matches upstream, including absent annotations; no added args.
# Cost/removal: same loading-only padding and upstream removal condition above.
def _process_scale_2d(scale_2d):
    # ### PATCH START: preserve the final expert-list scale column
    return pack_mxfp_scale_pairs(scale_2d).transpose(-3, -2)
    # ### PATCH END: preserve the final expert-list scale column


# Upstream: same file/SHA, inherited from AscendW4A8MXFPDynamicFusedMoEMethod.
# Patch reason: target WeightNz requires K == 64 * per-token scale pair count;
# a nine-group K=288 activation must match the zero-padded physical W2 K=320.
# Patch functionality: zero-pad quantized activation bytes, initialize only the
# unused scale byte, then execute the copied GMM2 with native dtype/peer layout.
# Signature matches upstream, including return annotation absence; no new args.
# Cost: unaligned K adds activation padding plus a scale clone per GMM2 call.
# Removal: remove with the post-load padding once upstream handles this shape.
def apply_gmm2(
    self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale
):
    layer = mlp_compute_input.layer
    assert layer is not None
    input_dtype = mlp_compute_input.hidden_states.dtype
    use_bf16 = input_dtype in [torch.bfloat16, torch.float8_e4m3fn]
    output_dtype = (
        input_dtype
        if input_dtype in [torch.bfloat16, torch.float16]
        else (torch.bfloat16 if use_bf16 else torch.float16)
    )
    # ### PATCH START: match physical W2 K without changing valid MX values
    hidden_states, act_out_scale = align_mxfp_gmm_input(hidden_states, act_out_scale)
    # ### PATCH END: match physical W2 K without changing valid MX values
    return torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=self._resolve_gmm_weights(layer, "w2_weight", "w2_weight_list"),
        scale=None,
        antiquant_scale=self._resolve_gmm_scales(
            layer, "w2_weight_scale", "w2_weight_scale_list"
        ),
        bias=None,
        per_token_scale=[act_out_scale],
        split_item=2,
        group_list_type=mlp_compute_input.group_list_type,
        group_type=0,
        group_list=mlp_compute_input.group_list,
        output_dtype=output_dtype,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        x_dtype=torch.float8_e4m3fn,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
    )[0]


# Upstream: same file/SHA, inherited apply_gmm1_act_quant.
# Patch reason: MX SwiGLU requires its combined input width divisible by 128;
# TP8 W13 emits 2*288=576 before the already-aligned physical W2 K=320.
# Patch functionality: pad each gate/up half independently to 320 immediately
# before fused SwiGLU. Preserve valid pairs, clamp/group semantics and SITU.
# Signature matches upstream, including return annotation absence; no new args.
# Cost: unaligned halves add one BF16 padding allocation/copy per fused call;
# quantized output is physical K=320 and matches the existing W2 padding.
# Removal: remove with W2 alignment once upstream handles this fused shape.
def apply_gmm1_act_quant(self, mlp_compute_input: MoEMlpComputeInput):
    hidden_states = mlp_compute_input.hidden_states
    hidden_states, pertoken_scale = self._quant_hidden_states(
        hidden_states, mlp_compute_input.dynamic_scale
    )
    layer = mlp_compute_input.layer
    assert layer is not None
    if (
        mlp_compute_input.activation == MoEActivation.SITU
        and mlp_compute_input.group_list_type in (0, 1)
        and self.group_size == 32
        and (mlp_compute_input.activation_situ_linear_beta or 0.0) > 0.0
    ):
        hidden_states, out_scale, _ = DeviceOperator.npu_grouped_matmul_situ_quant(
            x=hidden_states,
            weight=layer.w13_weight,
            weight_scale=layer.w13_weight_scale,
            x_scale=pertoken_scale,
            group_list=mlp_compute_input.group_list,
            group_list_type=mlp_compute_input.group_list_type,
            beta=(
                1.0
                if mlp_compute_input.activation_situ_beta is None
                else mlp_compute_input.activation_situ_beta
            ),
            linear_beta=mlp_compute_input.activation_situ_linear_beta or 0.0,
            mxfp_quant_dtype=self.quant_type,
        )
        dispose_tensor(mlp_compute_input.hidden_states)
        return hidden_states, maybe_normalize_mxfp_scale_layout(out_scale)
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=self._resolve_gmm_weights(layer, "w13_weight", "w13_weight_list"),
        scale=None,
        antiquant_scale=self._resolve_gmm_scales(
            layer, "w13_weight_scale", "w13_weight_scale_list"
        ),
        scale_dtype=None,
        per_token_scale=[pertoken_scale],
        per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        split_item=2,
        group_type=0,
        group_list=mlp_compute_input.group_list,
        group_list_type=mlp_compute_input.group_list_type,
        x_dtype=torch.float8_e4m3fn,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        output_dtype=torch.bfloat16,
    )[0]
    dispose_tensor(mlp_compute_input.hidden_states)
    if mlp_compute_input.activation == MoEActivation.SITU:
        # SituAndMul: run the dequantized gmm1 first, then fuse the situ
        # activation with MXFP output quantization (Kimi K3 SITU activation).

        hidden_states, swiglu_out_scale = torch.ops._C_ascend.situ_mx_quant(
            x=hidden_states,
            beta=1.0
            if mlp_compute_input.activation_situ_beta is None
            else mlp_compute_input.activation_situ_beta,
            linear_beta=mlp_compute_input.activation_situ_linear_beta or 0.0,
            activate_left=True,
            dst_type=SITU_MX_DST_TYPE_E4M3FN,
        )
        return hidden_states, maybe_normalize_mxfp_scale_layout(swiglu_out_scale)

    # The `group_index` input for the `npu_swiglu_group_quant` operator
    # currently only supports the `count` type. In the current version, the
    # `npu_swiglu_group_quant` operator performs a summation calculation on
    # `group_index`. Therefore, under the cumsum type, the last value is directly taken
    # and Avoid executing two small operators.
    if mlp_compute_input.group_list_type == 0:
        group_index = mlp_compute_input.group_list[-1:]
    else:
        group_index = cumsum_group_list(
            mlp_compute_input.group_list, mlp_compute_input.group_list_type, 1
        )
    # ### PATCH START: align each gate/up half for fused MX SwiGLU
    hidden_states = align_mxfp_swiglu_input(hidden_states)
    # ### PATCH END: align each gate/up half for fused MX SwiGLU
    hidden_states, out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
        hidden_states,
        topk_weight=None,
        group_index=group_index,
        dst_type=torch.float8_e4m3fn,
        quant_mode=2,
        clamp_value=mlp_compute_input.swiglu_limit,
    )
    return hidden_states, maybe_normalize_mxfp_scale_layout(out_scale)


def apply_dsv41_mxfp4_scale_patch() -> None:
    AscendW4A8MXFPDSDynamicFusedMoEMethod.process_weights_after_loading = (
        process_weights_after_loading
    )
    AscendW4A8MXFPDSDynamicFusedMoEMethod._process_scale_2d = staticmethod(
        _process_scale_2d
    )
    AscendW4A8MXFPDSDynamicFusedMoEMethod.apply_gmm2 = apply_gmm2
    AscendW4A8MXFPDSDynamicFusedMoEMethod.apply_gmm1_act_quant = apply_gmm1_act_quant
