# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pair E8M0 scale bytes without dropping an odd final MX group."""

import torch
import torch.nn.functional as F  # noqa: N812

SCALE_PAIR_WIDTH = 2


def pack_mxfp_scale_pairs(scales: torch.Tensor) -> torch.Tensor:
    """Pack (..., N, K_scale) E8M0/uint8 into (..., N, ceil(K_scale/2), 2)."""
    scale_bytes = scales.view(torch.uint8)
    if scale_bytes.shape[-1] % SCALE_PAIR_WIDTH:
        scale_bytes = F.pad(scale_bytes, (0, 1), value=0)
    return scale_bytes.reshape(
        *scale_bytes.shape[:-1],
        scale_bytes.shape[-1] // SCALE_PAIR_WIDTH,
        SCALE_PAIR_WIDTH,
    )
