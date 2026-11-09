# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Verify native post-load tensor output; fake NZ is not device acceptance."""

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (  # noqa: E402
    AscendW4A8MXFPDSDynamicFusedMoEMethod,
)

from afd_plugin.compat.patches.npu import mxfp4  # noqa: E402
from tests.unit.fakes.mxfp4 import ds_moe_layer  # noqa: E402


@pytest.fixture
def patched_mxfp4(monkeypatch):
    target = AscendW4A8MXFPDSDynamicFusedMoEMethod
    monkeypatch.setattr(
        target, "process_weights_after_loading", target.process_weights_after_loading
    )
    monkeypatch.setattr(
        target, "_process_scale_2d", staticmethod(target._process_scale_2d)
    )
    monkeypatch.setattr(torch_npu, "npu_format_cast", lambda tensor, *a, **k: tensor)
    mxfp4.apply_dsv41_mxfp4_scale_patch()


@pytest.mark.parametrize("columns", [8, 9, 10])
@pytest.mark.parametrize("dynamic_eplb", [False, True])
def test_moe_post_load_preserves_expert_weights_and_scales(
    patched_mxfp4, columns, dynamic_eplb
):
    scheme, layer = ds_moe_layer(columns, dynamic_eplb)
    weights = {
        name: value.detach().view(torch.uint8).clone()
        for name, value in layer.named_parameters()
    }

    scheme.process_weights_after_loading(layer)

    for stem in ("w13", "w2"):
        if dynamic_eplb:
            actual_weight = torch.stack(getattr(layer, f"{stem}_weight_list"))
            scale = torch.stack(getattr(layer, f"{stem}_weight_scale_list"))
        else:
            actual_weight = getattr(layer, f"{stem}_weight").transpose(1, 2)
            scale = getattr(layer, f"{stem}_weight_scale")
        expected = weights[f"{stem}_weight_scale"]
        unpacked = scale.transpose(-3, -2).flatten(-2)
        torch.testing.assert_close(actual_weight, weights[f"{stem}_weight"])
        torch.testing.assert_close(unpacked[..., : expected.shape[-1]], expected)
        assert torch.count_nonzero(unpacked[..., expected.shape[-1] :]) == 0
