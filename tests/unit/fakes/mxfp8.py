# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small CPU tensors at the native MXFP8 post-load boundary."""

from types import SimpleNamespace

import torch
from torch import nn


def ds_linear_layer(block_size, scale_columns, prefix, scale_dtype):
    rows = 2 * block_size
    scale_values = torch.arange(2 * scale_columns).reshape(2, scale_columns) % 3
    scales = torch.pow(2.0, scale_values).float()
    if scale_dtype == "e8m0":
        scales = (scales.view(torch.int32) >> 23).to(torch.uint8)
        scales = scales.view(torch.float8_e8m0fnu)
    weight = (
        torch.arange(rows * scale_columns * block_size)
        .reshape(rows, scale_columns * block_size)
        .float()
    )
    layer = nn.Module()
    layer.prefix = prefix
    layer.weight = nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = nn.Parameter(scales, requires_grad=False)
    scheme = SimpleNamespace(
        block_size=block_size, group_size=32, o_lora_rank=block_size
    )
    return scheme, layer
