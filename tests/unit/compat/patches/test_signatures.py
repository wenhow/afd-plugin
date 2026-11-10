# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

"""One table owns upstream signature checks for scoped runtime patches."""

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "hook",
    [
        "capture-inputs",
        "camp2p-dispatch",
        "hccl-native-prepare",
        "hccl-graph-gather",
        "hccl-graph-dispatch",
        "hccl-graph-begin",
        "hccl-graph-capture",
        "hccl-graph-replay",
        "hccl-u1-prepare",
        "ffn-profile",
        "attention-profile",
        "attention-shutdown",
        "npu-model-loader",
        "ds-mxfp8",
        "ds-mxfp4",
        "ds-mxfp4-expert-scale",
        "ds-mxfp4-gmm2",
        "ds-mxfp4-gmm1-act-quant",
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
    elif hook == "hccl-native-prepare":
        pytest.importorskip("torch_npu")
        pytest.importorskip("vllm_ascend")
        from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner

        from afd_plugin.v1.worker.npu.hccl_v41_execution import HcclV41Execution
        from tests.unit.fakes.deepseek_v41 import ParentEngramState

        target = object.__new__(UBatchRunner)
        # Keep a stable bound-method alias so restoration can use the table's
        # identity assertion as well as its exact signature comparison.
        target.prepare = target.prepare
        owner = SimpleNamespace(model_state=ParentEngramState({}), ubatch_runner=target)
        symbol = "prepare"
        scope = HcclV41Execution(
            owner,
            SimpleNamespace(terminal=False),
            mode="target",
            valid_tokens=0,
            dummy_run=False,
        ).scope()
    elif hook in ("hccl-graph-gather", "hccl-graph-dispatch"):
        pytest.importorskip("torch_npu")
        pytest.importorskip("vllm_ascend")
        from vllm.v1.worker.gpu import model_runner
        from vllm_ascend.worker.v2.model_runner import NPUModelRunner

        from afd_plugin.compat.patches.npu.hccl_graph_dispatch import (
            use_hccl_graph_dispatch,
        )

        owner = object.__new__(NPUModelRunner)
        owner.cudagraph_manager = None
        owner.parallel_config = SimpleNamespace()
        owner.gather_batch_req_state = owner.gather_batch_req_state
        if hook == "hccl-graph-gather":
            target, symbol = owner, "gather_batch_req_state"
        else:
            target, symbol = model_runner, "dispatch_cg_and_sync_dp"
        scope = use_hccl_graph_dispatch(owner, lambda state, args, desc: desc)
    elif hook == "hccl-graph-capture":
        from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

        from afd_plugin.compat.patches.npu.hccl_graph_capture import (
            use_hccl_graph_capture,
        )

        target = CudaGraphManager
        symbol = "capture"
        scope = use_hccl_graph_capture(
            object.__new__(CudaGraphManager),
            before_factory=lambda descriptor, warmup: None,
            after_capture=lambda: None,
            fail=lambda: None,
        )
    elif hook == "hccl-graph-replay":
        pytest.importorskip("torch_npu")
        pytest.importorskip("vllm_ascend")
        from contextlib import nullcontext

        from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager

        from afd_plugin.compat.patches.npu.hccl_graph_replay import (
            use_hccl_graph_replay,
        )

        target = object.__new__(ModelAclGraphManager)
        target.run_fullgraph = target.run_fullgraph
        symbol = "run_fullgraph"
        scope = use_hccl_graph_replay(
            target,
            before_replay=lambda descriptor: nullcontext(),
            after_replay=lambda descriptor: None,
            fail=lambda: None,
        )
    elif hook == "hccl-u1-prepare":
        pytest.importorskip("torch_npu")
        pytest.importorskip("vllm_ascend")
        from afd_plugin.compat.patches.npu.hccl_v41_u1_graph import (
            use_hccl_v41_u1_preparation,
        )
        from tests.unit.fakes.deepseek_v41 import ParentEngramState

        target = ParentEngramState({})
        target.prepare_attn = target.prepare_attn
        symbol = "prepare_attn"
        scope = use_hccl_v41_u1_preparation(target, None, lambda: None)
    elif hook == "hccl-graph-begin":
        pytest.importorskip("torch_npu")
        pytest.importorskip("vllm_ascend")
        from vllm.v1.worker.gpu.ubatch_utils import UBatchRunner

        from afd_plugin.v1.worker.npu.hccl_v41_ubatch_runner import HcclV41UBatchRunner

        target = UBatchRunner
        symbol = "begin_capturable_run"
        scope = patch.object(target, symbol, HcclV41UBatchRunner.begin_capturable_run)
    elif hook in ("ffn-profile", "attention-profile", "attention-shutdown"):
        pytest.importorskip("torch_npu")
        pytest.importorskip("vllm_ascend")
        from vllm_ascend.worker.worker import NPUWorker

        from afd_plugin.v1.worker.npu.attention_worker import AFDNPUAttentionWorker
        from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker

        target = NPUWorker
        symbol = "shutdown" if hook == "attention-shutdown" else "profile"
        profile = (
            AFDNPUFFNWorker.profile
            if hook == "ffn-profile"
            else AFDNPUAttentionWorker.shutdown
            if hook == "attention-shutdown"
            else AFDNPUAttentionWorker.profile
        )
        scope = patch.object(target, symbol, profile)
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
    elif hook in (
        "ds-mxfp4",
        "ds-mxfp4-expert-scale",
        "ds-mxfp4-gmm2",
        "ds-mxfp4-gmm1-act-quant",
    ):
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
        elif hook == "ds-mxfp4-gmm2":
            symbol = "apply_gmm2"
            replacement = adapter.apply_gmm2
        else:
            symbol = "apply_gmm1_act_quant"
            replacement = adapter.apply_gmm1_act_quant
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
