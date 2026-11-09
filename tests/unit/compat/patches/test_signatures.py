# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""One table owns upstream signature checks for scoped runtime patches."""

import inspect
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "hook",
    [
        "capture-inputs",
        "camp2p-dispatch",
        "npu-model-loader",
        "ds-mxfp8",
        "ds-mxfp4",
        "ds-mxfp4-expert-scale",
        "ds-mxfp4-gmm2",
    ],
)
def test_scoped_patch_preserves_upstream_signature(hook):  # long-test: shared table
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    if hook == "capture-inputs":
        from vllm.v1.worker.gpu import cudagraph_utils as target
        from vllm.v1.worker.gpu.pcp_manager import PCPManager  # noqa: F401

        from afd_plugin.v1.worker import attention_model_runner_v2 as adapter

        symbol = "prepare_inputs_to_capture"
        tracker = adapter._AFDCaptureEventTracker([])
        scope = adapter._use_afd_capture_input_preparation(None, tracker)
    elif hook == "camp2p-dispatch":
        pytest.importorskip("torch_npu")
        pytest.importorskip("vllm_ascend")
        from vllm.v1.worker.gpu import model_runner as target

        from afd_plugin.v1.worker.npu import attention_model_runner_v2 as adapter

        symbol = "dispatch_cg_and_sync_dp"
        scope = adapter._use_camp2p_dp_padding(4, 2)
    elif hook == "ds-mxfp8":
        pytest.importorskip("torch_npu")
        pytest.importorskip("vllm_ascend")
        from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
            AscendW8A8MXFP8DSDynamicLinearMethod,
        )

        from afd_plugin.compat.patches.npu import mxfp8 as adapter

        target = AscendW8A8MXFP8DSDynamicLinearMethod
        symbol = "process_weights_after_loading"
        scope = patch.object(target, symbol, adapter.process_weights_after_loading)
    elif hook in ("ds-mxfp4", "ds-mxfp4-expert-scale", "ds-mxfp4-gmm2"):
        pytest.importorskip("torch_npu")
        pytest.importorskip("vllm_ascend")
        from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import (  # noqa: F401
            MoEMlpComputeInput,
        )
        from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
            AscendW4A8MXFPDSDynamicFusedMoEMethod,
        )

        from afd_plugin.compat.patches.npu import mxfp4 as adapter

        target = AscendW4A8MXFPDSDynamicFusedMoEMethod
        if hook == "ds-mxfp4":
            symbol = "process_weights_after_loading"
            replacement = adapter.process_weights_after_loading
        elif hook == "ds-mxfp4-expert-scale":
            symbol = "_process_scale_2d"
            replacement = staticmethod(adapter._process_scale_2d)
        else:
            symbol = "apply_gmm2"
            replacement = adapter.apply_gmm2
        scope = patch.object(target, symbol, replacement)
    else:
        from torch import nn  # noqa: F401
        from vllm.config import ModelConfig, VllmConfig  # noqa: F401
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader

        from afd_plugin.compat.patches.npu import model_loader as adapter

        target = BaseModelLoader
        symbol = "create_model"
        scope = patch.object(target, symbol, adapter.create_model)
    original = getattr(target, symbol)
    with scope:
        patched = getattr(target, symbol)
        assert patched is not original
        assert inspect.signature(
            patched, eval_str=True, locals=locals()
        ) == inspect.signature(original, eval_str=True, locals=locals())
    assert getattr(target, symbol) is original
