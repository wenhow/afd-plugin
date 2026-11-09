# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Scale bytes are a CPU tensor contract, separate from NZ/GMM acceptance."""

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.compat.patches.npu.mxfp_scales import (  # noqa: E402
    pack_mxfp_scale_pairs,
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
