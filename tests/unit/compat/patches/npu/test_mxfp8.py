# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Validate post-load tensor values; no device operator acceptance."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (  # noqa: E402
    AscendW4A8MXFPDSDynamicFusedMoEMethod,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (  # noqa: E402
    AscendW8A8MXFP8DSDynamicLinearMethod,
)

from afd_plugin import register_dsv41_mxfp8_scale_patch  # noqa: E402
from afd_plugin.compat.patches.npu.mxfp8 import (  # noqa: E402
    process_weights_after_loading,
)
from tests.unit.fakes.mxfp8 import ds_linear_layer  # noqa: E402


@pytest.mark.parametrize("block_size,columns", [(32, 9), (32, 8), (128, 3)])
@pytest.mark.parametrize("scale_dtype", ["float32", "e8m0"])
@pytest.mark.parametrize("prefix", ["shared_experts.down_proj", "wo_a", "wo_b"])
def test_packing_preserves_valid_weights_and_scales(
    block_size, columns, scale_dtype, prefix
):
    scheme, layer = ds_linear_layer(block_size, columns, prefix, scale_dtype)
    weight = layer.weight.detach().clone()
    raw = layer.weight_scale.detach().clone()
    encoded = (
        raw.view(torch.uint8)
        if scale_dtype == "e8m0"
        else (raw.view(torch.int32) >> 23 & 0xFF).to(torch.uint8)
    )
    expected = encoded.repeat_interleave(block_size, 0).repeat_interleave(
        block_size // scheme.group_size, 1
    )

    process_weights_after_loading(scheme, layer)

    if prefix == "wo_a":
        output_weight = layer.weight.transpose(1, 2).reshape_as(weight)
        output_scale = layer.weight_scale.transpose(1, 2).flatten(0, 1)
    else:
        output_weight = layer.weight.T
        output_scale = layer.weight_scale.transpose(0, 1)
    packed = output_scale.reshape(expected.shape[0], -1)
    torch.testing.assert_close(output_weight, weight, rtol=0, atol=0)
    torch.testing.assert_close(packed[:, : expected.shape[1]], expected)
    assert packed.shape[1] == (expected.shape[1] + 1) // 2 * 2
    assert torch.count_nonzero(packed[:, expected.shape[1] :]) == 0


def test_nonintegral_block_to_group_ratio_is_rejected():
    scheme, layer = ds_linear_layer(32, 9, "shared_experts.down_proj", "float32")
    scheme.group_size = 64
    with pytest.raises(ValueError, match="block_size=32, group_size=64"):
        process_weights_after_loading(scheme, layer)


def test_repeated_plugin_registration_preserves_odd_scale_packing(monkeypatch):
    target = AscendW8A8MXFP8DSDynamicLinearMethod
    monkeypatch.setattr(
        target, "process_weights_after_loading", target.process_weights_after_loading
    )
    moe = AscendW4A8MXFPDSDynamicFusedMoEMethod
    monkeypatch.setattr(
        moe, "process_weights_after_loading", moe.process_weights_after_loading
    )
    monkeypatch.setattr(moe, "_process_scale_2d", staticmethod(moe._process_scale_2d))
    monkeypatch.setattr(moe, "apply_gmm2", moe.apply_gmm2)
    monkeypatch.setattr(moe, "apply_gmm1_act_quant", moe.apply_gmm1_act_quant)
    monkeypatch.setenv("AFD_DSV41_MXFP8_SCALE_PATCH", "1")
    register_dsv41_mxfp8_scale_patch()
    register_dsv41_mxfp8_scale_patch()
    scheme, layer = ds_linear_layer(32, 9, "shared_experts.down_proj", "float32")

    target.process_weights_after_loading(scheme, layer)

    assert tuple(layer.weight.shape) == (288, 64)
    assert tuple(layer.weight_scale.shape) == (5, 64, 2)
    assert torch.count_nonzero(layer.weight_scale[-1, :, 1]) == 0
    assert torch.count_nonzero(layer.weight_scale[-1, :, 0]) == 64
