# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in DS MXFP4 MoE scale packing correction."""

import torch
import torch_npu
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
    AscendW4A8MXFPDSDynamicFusedMoEMethod,
)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

from afd_plugin.compat.patches.npu.mxfp_scales import (
    align_mxfp4_weight,
    align_mxfp_gmm_input,
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


def apply_dsv41_mxfp4_scale_patch() -> None:
    AscendW4A8MXFPDSDynamicFusedMoEMethod.process_weights_after_loading = (
        process_weights_after_loading
    )
    AscendW4A8MXFPDSDynamicFusedMoEMethod._process_scale_2d = staticmethod(
        _process_scale_2d
    )
    AscendW4A8MXFPDSDynamicFusedMoEMethod.apply_gmm2 = apply_gmm2
