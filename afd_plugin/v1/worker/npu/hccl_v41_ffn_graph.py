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

import torch
from vllm.config import CUDAGraphMode, ParallelConfig
from vllm.forward_context import ForwardContext, override_forward_context

from afd_plugin.connectors.npu.hccl_diagnostics import HcclStageDiagnostics
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
        diagnostics: HcclStageDiagnostics | None = None,
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
        if bank.exchange_order != "stage_serial" and pipeline is None:
            raise ValueError(
                f"{bank.exchange_order} Graph U2 requires an actual compute pipeline"
            )
        self.pipeline = pipeline
        self.diagnostics = None
        if diagnostics is not None:
            selection = diagnostics.selection
            if (
                selection.role != "ffn"
                or selection.phase == "eager"
                or selection.num_stages != len(bank.stage_rows)
                or diagnostics.role_rank != bank.role_rank
                or (
                    pipeline is not None
                    and pipeline.selection is not None
                    and pipeline.selection != selection
                )
            ):
                raise ValueError(
                    "FFN graph diagnostics must match the retained role/stages/phase"
                )
            self.diagnostics = diagnostics.bind_execution(
                layer_indices=bank.layer_indices,
                actual_compute_stream="compute" if pipeline is not None else "parent",
                is_warmup=selection.phase == "graph_warmup",
            )

    def run(self) -> None:
        """Execute/record tensor work; replay does not enter this Python loop."""
        bank, pipeline, progress = self.bank, self.pipeline, self.diagnostics
        with pipeline.run() if pipeline is not None else nullcontext():
            if pipeline is not None and pipeline.schedule is not None:
                outputs: dict[int, torch.Tensor] = {}
                for action in pipeline.schedule.actions:
                    if action.kind == "input":
                        outputs[action.stage_idx] = self._receive_compute(
                            action.layer_idx, action.stage_idx
                        )
                    else:
                        with override_forward_context(self.contexts[action.stage_idx]):
                            self._send_output(
                                action.layer_idx,
                                action.stage_idx,
                                outputs.pop(action.stage_idx),
                            )
            else:
                for layer_idx in bank.layer_indices:
                    pending: list[tuple[int, torch.Tensor]] = []
                    for stage_idx, rows in enumerate(bank.stage_rows):
                        if not rows:
                            continue
                        output = self._receive_compute(layer_idx, stage_idx)
                        if bank.exchange_order == "bulk_v1":
                            pending.append((stage_idx, output))
                        else:
                            with override_forward_context(self.contexts[stage_idx]):
                                self._send_output(layer_idx, stage_idx, output)
                    for stage_idx, output in pending:
                        with override_forward_context(self.contexts[stage_idx]):
                            self._send_output(layer_idx, stage_idx, output)
                    if pipeline is not None and not pipeline.policy.ffn_cross_layer:
                        if progress is not None:
                            progress.record("layer_send_wait_begin", layer_idx, 0)
                        pipeline.join_layer(layer_idx)
                        if progress is not None:
                            progress.record("layer_send_wait_returned", layer_idx, 0)
            if pipeline is not None:
                if progress is not None:
                    progress.record("final_send_wait_begin", bank.layer_indices[-1], 0)
                pipeline.join()
                if progress is not None:
                    progress.record(
                        "final_send_wait_returned", bank.layer_indices[-1], 0
                    )

    def _receive_compute(self, layer_idx: int, stage_idx: int) -> torch.Tensor:
        """Receive one stage and publish its complete native MoE result."""
        bank, pipeline, progress = self.bank, self.pipeline, self.diagnostics
        with override_forward_context(self.contexts[stage_idx]):
            _set_moe_layer_index(self.contexts[stage_idx], layer_idx)
            if progress is not None:
                progress.record("recv_begin", layer_idx, stage_idx)
            with (
                pipeline.ffn_receive(layer_idx, stage_idx)
                if pipeline is not None
                else nullcontext()
            ):
                buffers = bank.receive_inputs(stage_idx)
            if progress is not None:
                progress.record("recv_returned", layer_idx, stage_idx)
                progress.record("compute_begin", layer_idx, stage_idx)
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
                if bank.exchange_order != "stage_serial":
                    # A native return may borrow a reusable MoE workspace.
                    # Consume it into stage-owned storage before compute_done.
                    if output.device.type == "npu":
                        output.record_stream(torch.npu.current_stream())
                    output = bank.prepare_output(stage_idx, output)
            if pipeline is not None and bank.exchange_order != "stage_serial":
                pipeline.retain_compute_outputs(layer_idx, stage_idx, tensors=(output,))
            if progress is not None:
                progress.record("compute_returned", layer_idx, stage_idx)
            return output

    def _send_output(
        self, layer_idx: int, stage_idx: int, output: torch.Tensor
    ) -> None:
        """The shared serial/bulk parent send frontier and host diagnostics."""
        pipeline, progress = self.pipeline, self.diagnostics
        if progress is not None:
            progress.record("send_begin", layer_idx, stage_idx)
        with (
            pipeline.ffn_send(layer_idx, stage_idx, tensors=(output,))
            if pipeline is not None
            else nullcontext()
        ):
            self.bank.send_output(stage_idx, output)
        if progress is not None:
            progress.record("send_returned", layer_idx, stage_idx)
