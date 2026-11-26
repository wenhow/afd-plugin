# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bound the startup memory used to compare V4.1 DSpark vocabulary weights."""

import torch
from torch import nn

WEIGHT_COMPARE_CHUNK_ROWS = 1024


def equal_weight_rows_on_cpu(draft: torch.Tensor, target: torch.Tensor) -> bool:
    if draft.shape != target.shape:
        return False
    return all(
        torch.equal(draft_rows.cpu(), target_rows.cpu())
        for draft_rows, target_rows in zip(
            draft.split(WEIGHT_COMPARE_CHUNK_ROWS),
            target.split(WEIGHT_COMPARE_CHUNK_ROWS),
            strict=True,
        )
    )


# Upstream: _should_share, vllm/v1/worker/gpu/spec_decode/eagle/utils.py
# @ ced6857afa0ea7b2e3f0846a62e1394e90f15607.
# Reason: the CUDA low-memory fallback does not cover the native Ascend
# compare, which can exhaust HBM before the shared embedding is released.
# Functionality: keep exact equality, with bounded CPU copies of NPU
# vocabulary weights during the diagnostic worker's native model load.
# Signature: matches upstream; no added parameters.
# Cost: synchronous device-to-host copies only during model loading.
# Removal: use the native comparison when it bounds temporary NPU memory.
def should_share(eagle: nn.Module, flag: str, draft, target) -> bool:
    """Share when the draft has no own copy, or its copy matches the target."""

    if not getattr(eagle, flag, False) or draft is None:
        return True
    if target is None:
        return False
    # torch.equal on GPU allocates a bool mask the size of the input.
    # Use the faster GPU path when there is plenty of headroom;
    # otherwise compare on CPU.
    w = draft.weight
    # ### PATCH START: bound NPU vocabulary comparison
    if w.device.type == "npu" and w.ndim == 2 and target.weight.device == w.device:
        return equal_weight_rows_on_cpu(w, target.weight)
    # ### PATCH END: bound NPU vocabulary comparison
    if w.is_cuda and torch.accelerator.get_memory_info(w.device)[0] < w.numel() * 2:
        return torch.equal(w.cpu(), target.weight.cpu())
    return torch.equal(w, target.weight)
