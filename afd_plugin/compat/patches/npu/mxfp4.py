# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in DS MXFP4 MoE scale packing correction."""

import torch
import torch_npu
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
    AscendW4A8MXFPDSDynamicFusedMoEMethod,
)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

from afd_plugin.compat.patches.npu.mxfp_scales import pack_mxfp_scale_pairs


# Upstream: vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py,
# AscendW4A8MXFPDSDynamicFusedMoEMethod @ 7e37e29a (identical at 2db8549c).
# Patch reason: TP8 routed-expert W2 has K=288, hence nine scale columns;
# upstream truncates the pair count and cannot reshape the loaded tensor.
# Patch functionality: pad the unused final scale byte for W13/W2, preserving
# FP4 weights, logical dimensions, NZ conversion and the native EPLB branch.
# Signature matches upstream, including absent annotations; no added args.
# Cost: loading-only scale allocation for odd columns; no forward overhead.
# Removal: delete when pinned DS MoE post-load supports odd scale pairs.
def process_weights_after_loading(self, layer):
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


def apply_dsv41_mxfp4_scale_patch() -> None:
    AscendW4A8MXFPDSDynamicFusedMoEMethod.process_weights_after_loading = (
        process_weights_after_loading
    )
    AscendW4A8MXFPDSDynamicFusedMoEMethod._process_scale_2d = staticmethod(
        _process_scale_2d
    )
