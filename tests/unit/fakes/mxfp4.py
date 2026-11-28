# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small tensors at the native DS MXFP4 MoE post-load boundary."""

import torch
from torch import nn
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
    AscendW4A8MXFPDSDynamicFusedMoEMethod,
)

GROUP_SIZE = 32
EXPERT_COUNT = 2
HIDDEN_SIZE = 64


def ds_moe_layer(scale_columns, dynamic_eplb):
    intermediate = scale_columns * GROUP_SIZE
    layer = nn.Module()
    shapes = {
        "w13_weight": (EXPERT_COUNT, 2 * intermediate, HIDDEN_SIZE // 2),
        "w2_weight": (EXPERT_COUNT, HIDDEN_SIZE, intermediate // 2),
        "w13_weight_scale": (
            EXPERT_COUNT,
            2 * intermediate,
            HIDDEN_SIZE // GROUP_SIZE,
        ),
        "w2_weight_scale": (EXPERT_COUNT, HIDDEN_SIZE, scale_columns),
    }
    for name, shape in shapes.items():
        tensor = torch.arange(shape[-1], dtype=torch.uint8).expand(shape).clone()
        if name.endswith("scale"):
            tensor = tensor.add(125).view(torch.float8_e8m0fnu)
        layer.register_parameter(name, nn.Parameter(tensor, requires_grad=False))
    scheme = AscendW4A8MXFPDSDynamicFusedMoEMethod.__new__(
        AscendW4A8MXFPDSDynamicFusedMoEMethod
    )
    scheme.dynamic_eplb = dynamic_eplb
    return scheme, layer
