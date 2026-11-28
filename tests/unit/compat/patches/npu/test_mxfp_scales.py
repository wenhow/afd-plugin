# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Scale bytes are a CPU tensor contract, separate from NZ/GMM acceptance."""

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.compat.patches.npu.mxfp_scales import (  # noqa: E402
    align_mxfp4_weight,
    align_mxfp_gmm_input,
    align_mxfp_swiglu_input,
    pack_mxfp_scale_pairs,
)
from tests.unit.fakes.mxfp_scales import (  # noqa: E402
    dequantized_mxfp_gmm,
    mxfp_gmm_tensors,
)


@pytest.mark.parametrize("shape", [(3, 1), (3, 8), (3, 9), (2, 3, 9), (2, 3, 160)])
@pytest.mark.parametrize("semantic_dtype", [False, True])
def test_scale_pair_packing_preserves_every_valid_byte(shape, semantic_dtype):
    values = torch.arange(shape[-1], dtype=torch.uint8).add(120).expand(shape)
    scales = values.contiguous()
    if semantic_dtype:
        scales = scales.view(torch.float8_e8m0fnu)

    packed = pack_mxfp_scale_pairs(scales)

    assert packed.dtype == torch.uint8
    assert packed.shape == (*shape[:-1], (shape[-1] + 1) // 2, 2)
    unpacked = packed.flatten(-2)
    torch.testing.assert_close(unpacked[..., : shape[-1]], values)
    assert torch.count_nonzero(unpacked[..., shape[-1] :]) == 0


@pytest.mark.parametrize("width", [256, 288, 320])
@pytest.mark.parametrize("rows", [0, 2])
@pytest.mark.parametrize("noncontiguous_scales", [False, True])
def test_gmm_input_alignment_preserves_valid_bytes_and_cleans_unused_scales(
    width, rows, noncontiguous_scales
):
    x, scales, _, _ = mxfp_gmm_tensors(width, rows)
    if noncontiguous_scales:
        scales = scales.transpose(0, 1).contiguous().transpose(0, 1)
    original_scales = scales.clone()
    valid_groups = (width + 31) // 32

    aligned_x, aligned_scales = align_mxfp_gmm_input(x, scales)

    assert aligned_x.shape == (rows, ((width + 63) // 64) * 64)
    torch.testing.assert_close(
        aligned_x.view(torch.uint8)[..., :width], x.view(torch.uint8)
    )
    assert torch.count_nonzero(aligned_x.view(torch.uint8)[..., width:]) == 0
    torch.testing.assert_close(
        aligned_scales.flatten(-2)[..., :valid_groups],
        original_scales.flatten(-2)[..., :valid_groups],
    )
    assert torch.all(aligned_scales.flatten(-2)[..., valid_groups:] == 127)
    torch.testing.assert_close(scales, original_scales)


@pytest.mark.parametrize("width", [256, 288, 320])
def test_fp4_weight_alignment_preserves_both_nibbles_and_zero_pads(width):
    _, _, weight, _ = mxfp_gmm_tensors(width)

    aligned = align_mxfp4_weight(weight)

    assert aligned.shape[-1] * 2 == ((width + 63) // 64) * 64
    torch.testing.assert_close(aligned[..., : weight.shape[-1]], weight)
    assert torch.count_nonzero(aligned[..., weight.shape[-1] :]) == 0


@pytest.mark.parametrize("width", [256, 288, 320])
def test_zero_padding_preserves_dequantized_mxfp_dot_products(width):
    x, x_scale, weight, w_scale = mxfp_gmm_tensors(width)
    expected = dequantized_mxfp_gmm(x, x_scale, weight, w_scale)

    aligned_x, aligned_x_scale = align_mxfp_gmm_input(x, x_scale)
    aligned_weight = align_mxfp4_weight(weight)
    aligned_w_scale = pack_mxfp_scale_pairs(w_scale).flatten(-2)
    actual = dequantized_mxfp_gmm(
        aligned_x, aligned_x_scale, aligned_weight, aligned_w_scale
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.all(torch.isfinite(actual))


@pytest.mark.parametrize("width", [256, 288, 320])
@pytest.mark.parametrize("rows", [0, 3])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_swiglu_alignment_preserves_gate_up_pairs(width, rows, noncontiguous):
    values = torch.arange(rows * 2 * width).reshape(rows, 2 * width)
    values = values.remainder(61).sub(30).to(torch.bfloat16)
    if noncontiguous:
        values = values.T.contiguous().T
    gate, up = values.chunk(2, dim=-1)

    aligned = align_mxfp_swiglu_input(values)

    assert aligned.shape == (rows, ((width + 63) // 64) * 128)
    padded_gate, padded_up = aligned.chunk(2, dim=-1)
    torch.testing.assert_close(padded_gate[..., :width], gate, rtol=0, atol=0)
    torch.testing.assert_close(padded_up[..., :width], up, rtol=0, atol=0)
    assert torch.count_nonzero(padded_gate[..., width:]) == 0
    assert torch.count_nonzero(padded_up[..., width:]) == 0


@pytest.mark.parametrize("width", [256, 288, 320])
@pytest.mark.parametrize("limit", [0, 6])
def test_swiglu_alignment_preserves_clamped_activation_values(width, limit):
    values = torch.arange(2 * width).remainder(61).sub(30).float().unsqueeze(0)
    gate, up = values.chunk(2, dim=-1)
    if limit:
        gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
    expected = torch.nn.functional.silu(gate) * up

    padded_gate, padded_up = align_mxfp_swiglu_input(values).chunk(2, dim=-1)
    if limit:
        padded_gate = padded_gate.clamp(max=limit)
        padded_up = padded_up.clamp(min=-limit, max=limit)
    actual = torch.nn.functional.silu(padded_gate) * padded_up

    torch.testing.assert_close(actual[..., :width], expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[..., width:]) == 0
