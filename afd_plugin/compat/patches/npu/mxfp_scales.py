# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Preserve MX values while pairing scales and aligning the FP4 GMM K axis."""

import torch
import torch.nn.functional as F  # noqa: N812

SCALE_PAIR_WIDTH = 2
MXFP_GROUP_SIZE = 32
MXFP_GMM_K_ALIGNMENT = MXFP_GROUP_SIZE * SCALE_PAIR_WIDTH
FP4_VALUES_PER_BYTE = 2
E8M0_UNIT_EXPONENT = 127


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


def align_mxfp4_weight(weight: torch.Tensor) -> torch.Tensor:
    """Zero-pad packed ND FP4 (..., N, K/2) before native NZ conversion."""
    padding = -weight.shape[-1] % (MXFP_GMM_K_ALIGNMENT // FP4_VALUES_PER_BYTE)
    if padding == 0:
        return weight
    return F.pad(weight, (0, padding), value=0)


def align_mxfp_gmm_input(
    hidden_states: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad quantized FP8 K with zeros and make only unused scales finite."""
    width = hidden_states.shape[-1]
    padding = -width % MXFP_GMM_K_ALIGNMENT
    if padding == 0:
        return hidden_states, scales
    padded = F.pad(hidden_states.view(torch.uint8), (0, padding), value=0)
    scale_bytes = scales.view(torch.uint8).clone(memory_format=torch.contiguous_format)
    valid_groups = (width + MXFP_GROUP_SIZE - 1) // MXFP_GROUP_SIZE
    # The quantizer may leave the unused E8M0 byte unspecified, including NaN.
    # Preserve valid bytes; zero activations in the new group use unit scale.
    scale_bytes.flatten(-2)[..., valid_groups:] = E8M0_UNIT_EXPONENT
    return padded.view(hidden_states.dtype), scale_bytes.view(scales.dtype)
