# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native TP4 control with the same A5 query-head adapter as AFD."""

from vllm.logger import init_logger
from vllm_ascend.models.deepseek_v41.mixed_quant_attention import (
    MixedQuantPackedCacheOps,
)
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Attention
from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.model_executor.models.npu.dsv41_packed_cache import (
    AFDMixedQuantPackedCacheOps,
)


class DSV41NativeTP4ControlWorker(NPUWorker):
    """Retain native execution and adapt only each packed Attention backend."""

    # Upstream: NPUWorker.load_model, worker/worker.py @ Ascend 7e37e29a.
    # Reason: the TP4 native control supplies 16 heads to a 64-head A5 wheel.
    # Functionality: select the existing head adapter after native weight load,
    # before profiling or inference, without changing native classes globally.
    # Signature: matches upstream exactly; no added parameters.
    # Delegation exception: native load owns sleep pools and weight transfer;
    # duplicating those owners for per-instance backend selection is unsuitable.
    # Cost: one module walk at load; qsmla has the documented 4x head work.
    # Removal: retire this control worker when the A5 wheel accepts 16 heads.
    def load_model(self) -> None:
        # ### PATCH START: native TP4 control scope
        if self.vllm_config.parallel_config.tensor_parallel_size != 4:
            raise ValueError("DSV41 native query-head control requires TP4")
        # ### PATCH END: native TP4 control scope
        super().load_model()
        # ### PATCH START: per-Attention control head adapter
        adapted = 0
        for module in self.model_runner.model.modules():
            if (
                isinstance(module, DeepseekV41Attention)
                and module.packed_cache_ops is MixedQuantPackedCacheOps
            ):
                module.packed_cache_ops = AFDMixedQuantPackedCacheOps
                adapted += 1
        if not adapted:
            raise ValueError("DSV41 native TP4 control found no A5 packed Attention")
        init_logger("vllm.afd_dsv41_native_control").info(
            "Native TP4 control: installed 16-to-64 query-head adapter on %d "
            "Attention layers; native model, MoE and Engram retained.",
            adapted,
        )
        # ### PATCH END: per-Attention control head adapter
