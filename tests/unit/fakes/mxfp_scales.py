# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU MX tensor values and an independent dequantized dot-product reference."""

import torch

FP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def mxfp_gmm_tensors(width, rows=2):
    x = torch.arange(rows * width).reshape(rows, width).remainder(7).sub(3)
    x = x.float().to(torch.float8_e4m3fn)
    x_scale = torch.full((rows, (width + 63) // 64, 2), 127, dtype=torch.uint8)
    groups = (width + 31) // 32
    if groups % 2:
        x_scale[:, -1, 1] = 255  # Unused E8M0 NaN must not enter padded math.
    weights = torch.arange(2 * 3 * width // 2).remainder(256).to(torch.uint8)
    weights = weights.reshape(2, 3, width // 2)
    w_scale = torch.arange(groups).remainder(3).add(126).to(torch.uint8)
    return x, x_scale, weights, w_scale.expand(2, 3, groups).clone()


def dequantized_mxfp_gmm(x, x_scale, packed_weight, w_scale):
    width = x.shape[-1]
    x_scales = x_scale.view(torch.uint8).flatten(-2).repeat_interleave(32, -1)
    x_float = x.float() * torch.pow(2.0, x_scales[..., :width].float() - 127)
    nibbles = torch.stack((packed_weight & 15, packed_weight >> 4), -1).flatten(-2)
    weights = torch.tensor(FP4_E2M1_VALUES)[nibbles.long()]
    w_scales = w_scale.view(torch.uint8).repeat_interleave(32, -1)
    weights = weights * torch.pow(2.0, w_scales[..., :width].float() - 127)
    return torch.einsum("mk,enk->emn", x_float, weights)
