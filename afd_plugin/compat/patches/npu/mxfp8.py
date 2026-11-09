# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in DS MXFP8 scale packing correction; no AFD model registration."""

import torch
import torch.nn.functional as F  # noqa: N812 - match the copied upstream code
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DSDynamicLinearMethod,
)

SCALE_PAIR_WIDTH = 2


# Upstream: vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py,
# AscendW8A8MXFP8DSDynamicLinearMethod @ 7e37e29a (also identical at 2db8549c).
# Patch reason: TP8 shared_experts.down_proj has K=288, hence nine 32-wide
# scales, which cannot be reshaped into pairs without one metadata padding byte.
# Patch functionality: use the base MXFP8 linear method's odd-column packing;
# retain all valid scales, weights, TP reduction and wo_a/wo_b transformations.
# Signature matches upstream, including its absent annotations; no new args.
# Cost: one scale-buffer allocation only for odd widths, during weight loading.
# Removal: delete once the pinned DS linear implementation packs odd widths.
def process_weights_after_loading(self, layer):
    if layer.weight_scale.data.dtype == torch.float8_e8m0fnu:
        layer.weight_scale.data = layer.weight_scale.data.view(torch.uint8)
    else:
        layer.weight_scale.data = layer.weight_scale.data.view(torch.int32) >> 23 & 0xFF
        layer.weight_scale.data = layer.weight_scale.data.to(torch.uint8)
    if self.block_size % self.group_size != 0:
        raise ValueError(
            "MXFP8 weight block size must be divisible by the "
            f"runtime group size, got block_size={self.block_size}, "
            f"group_size={self.group_size}"
        )
    input_repeat = self.block_size // self.group_size
    layer.weight_scale.data = layer.weight_scale.data.repeat_interleave(
        input_repeat, dim=1
    )
    layer.weight_scale.data = layer.weight_scale.data.repeat_interleave(
        self.block_size, dim=0
    )
    n_dim, k_dim = layer.weight_scale.data.shape
    # ### PATCH START: pad only the unused byte of the final scale pair
    if k_dim % SCALE_PAIR_WIDTH:
        layer.weight_scale.data = F.pad(layer.weight_scale.data, (0, 1), value=0)
        k_dim += 1
    # ### PATCH END: pad only the unused byte of the final scale pair
    layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
    layer.weight.data = layer.weight.data.transpose(0, 1)
    layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)

    if layer.prefix.endswith("wo_a"):
        # The layer may use OTP instead of the model's ordinary TP group.
        # Derive the group count from the loaded output shard.
        n_local_groups = layer.weight.shape[1] // self.o_lora_rank
        layer.weight.data = (
            layer.weight.data.T.reshape(n_local_groups, self.o_lora_rank, -1)
            .transpose(1, 2)
            .contiguous()
        )
        layer.weight_scale.data = (
            layer.weight_scale.data.transpose(0, 1)
            .reshape(n_local_groups, self.o_lora_rank, -1, 2)
            .transpose(1, 2)
            .contiguous()
        )
    elif layer.prefix.endswith("wo_b"):
        # DSA PCP gathers these tensors directly along the leading
        # dimension, which requires contiguous HCCL inputs.
        layer.weight.data = layer.weight.data.contiguous()
        layer.weight_scale.data = layer.weight_scale.data.contiguous()


def apply_dsv41_mxfp8_scale_patch() -> None:
    AscendW8A8MXFP8DSDynamicLinearMethod.process_weights_after_loading = (
        process_weights_after_loading
    )
