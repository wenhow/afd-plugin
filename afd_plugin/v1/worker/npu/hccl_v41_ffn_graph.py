# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit native FFN tensor body over a retained static stage bank.

The caller supplies initialized native TP and fully prepared native per-stage
contexts. Context construction, graph phase agreement, counts updates, graph
launch/synchronization and COMPLETE stay outside this tensor body. There is no
factory selection or production Graph admission here.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

from vllm.config import CUDAGraphMode, ParallelConfig
from vllm.forward_context import ForwardContext, override_forward_context

from afd_plugin.connectors.npu.hccl_graph_pipeline import HcclGraphComputePipeline
from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank
from afd_plugin.v1.worker.ffn_metadata import _set_moe_layer_index

if TYPE_CHECKING:
    from afd_plugin.model_executor.models.npu.deepseek_v41 import (
        AFDDeepseekV41FFNForCausalLM,
    )


class HcclV41FFNGraph:
    """Record layer-major receive/native MoE/send on the owning compute stream.

    Native ``compute_ffn_output`` already owns shared experts and its TP
    reduction. Its result is copied and padding rows are zeroed without any
    additional reduction. Every positive padded stage computes, including a
    zero-valid stage. A static zero-row stage records no wire operations.
    """

    def __init__(
        self,
        model: AFDDeepseekV41FFNForCausalLM,
        contexts: tuple[ForwardContext, ...],
        bank: HcclGraphStageBank,
        parallel_config: ParallelConfig,
        *,
        pipeline: HcclGraphComputePipeline | None = None,
    ) -> None:
        parallel = parallel_config
        if (
            bank.role != "ffn"
            or parallel.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.use_sequence_parallel_moe
            or parallel.tensor_parallel_size != bank.native_tp_size
            or parallel.rank != bank.role_rank
        ):
            raise ValueError(
                "V4.1 FFN graph requires paired native TP, DP1/PP1/CP1 and non-SP"
            )
        if model.get_experts_layer_indices() != bank.layer_indices or len(
            contexts
        ) != len(bank.stage_rows):
            raise ValueError(
                "V4.1 FFN graph requires the complete static layer/stage layout"
            )
        for stage_idx, (rows, context) in enumerate(
            zip(bank.stage_rows, contexts, strict=True)
        ):
            if (
                context.cudagraph_runtime_mode != CUDAGraphMode.FULL
                or context.ubatch_idx != stage_idx
                or (rows and context.input_ids is not bank.buffers(stage_idx).input_ids)
            ):
                raise ValueError(
                    "V4.1 FFN graph requires FULL contexts with retained stage IDs"
                )
        self.model, self.contexts, self.bank = model, contexts, bank
        if pipeline is not None and pipeline.bank is not bank:
            raise ValueError("FFN graph pipeline must own the retained stage bank")
        self.pipeline = pipeline

    def run(self) -> None:
        """Execute/record tensor work; replay does not enter this Python loop."""
        bank, pipeline = self.bank, self.pipeline
        with pipeline.run() if pipeline is not None else nullcontext():
            for layer_idx in bank.layer_indices:
                for stage_idx, (rows, context) in enumerate(
                    zip(bank.stage_rows, self.contexts, strict=True)
                ):
                    if not rows:
                        continue
                    with override_forward_context(context):
                        _set_moe_layer_index(context, layer_idx)
                        with (
                            pipeline.ffn_receive(layer_idx, stage_idx)
                            if pipeline is not None
                            else nullcontext()
                        ):
                            buffers = bank.receive_inputs(stage_idx)
                        with (
                            pipeline.compute(
                                layer_idx,
                                stage_idx,
                                inputs=(
                                    buffers.hidden_states,
                                    buffers.hidden_states_fp32,
                                    buffers.input_ids,
                                ),
                            )
                            if pipeline is not None
                            else nullcontext()
                        ):
                            output = self.model.compute_ffn_output(
                                hidden_states=buffers.hidden_states,
                                layer_idx=layer_idx,
                                input_ids=buffers.input_ids,
                                hidden_states_fp32=buffers.hidden_states_fp32,
                                already_sequence_parallel=False,
                            )
                        with (
                            pipeline.ffn_send(layer_idx, stage_idx, tensors=(output,))
                            if pipeline is not None
                            else nullcontext()
                        ):
                            bank.send_output(stage_idx, output)
            if pipeline is not None:
                pipeline.join()
