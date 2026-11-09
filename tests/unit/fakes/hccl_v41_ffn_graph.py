# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native compute_ffn_output with CPU MoE/operator seams; no device simulation."""

from types import SimpleNamespace

import torch
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils.torch_utils import current_stream

from afd_plugin.connectors.npu.hccl_transport import HcclLayerInputs
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    AFDDeepseekV41FFNForCausalLM,
)
from afd_plugin.v1.worker.npu.hccl_v41_ffn_graph import HcclV41FFNGraph
from tests.unit.fakes.hccl import staged_ffn_expected_output
from tests.unit.fakes.hccl_graph_stage import StageGraphEndpoint, make_graph_stage_bank
from tests.unit.fakes.vllm_config import hccl_vllm_config


class FFNGraphEndpoint(StageGraphEndpoint):
    def __init__(self, stage_idx, events, **kwargs):
        super().__init__(**kwargs)
        self.stage_idx, self.events = stage_idx, events

    def send(self, tensor, shape):
        self.events.append(("send", self.stage_idx, shape))
        super().send(tensor, shape)

    def receive(self, tensor, shape):
        self.events.append(("receive", self.stage_idx, shape))
        super().receive(tensor, shape)


class GraphContractMoE(torch.nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx, self.fail = layer_idx, False
        self.current_stream = self.expected_stream = None

    def forward(
        self, hidden_states, *, input_ids, hidden_states_fp32, already_sequence_parallel
    ):
        if self.fail:
            raise RuntimeError("controlled graph MoE failure")
        if (
            self.current_stream is not None
            and self.current_stream() is not self.expected_stream
        ):
            raise RuntimeError("native MoE helper observed the wrong compute stream")
        context = get_forward_context()
        if (
            context.cudagraph_runtime_mode != CUDAGraphMode.FULL
            or context.input_ids is not input_ids
            or context.moe_layer_index != self.layer_idx
            or already_sequence_parallel
        ):
            raise ValueError("graph MoE requires its prepared native context")
        return staged_ffn_expected_output(
            HcclLayerInputs(hidden_states, hidden_states_fp32, input_ids),
            self.layer_idx,
        )


def make_ffn_graph_case(
    *,
    rows=(3, 2),
    valid=(2, 0),
    native_tp_size=1,
    role_rank=0,
    pipeline_factory=None,
):
    events = []
    bank = make_graph_stage_bank(
        role="ffn",
        rows=rows,
        valid=valid,
        native_tp_size=native_tp_size,
        role_rank=role_rank,
        endpoint_factory=lambda index: FFNGraphEndpoint(
            index,
            events,
            role="ffn",
            role_rank=role_rank,
            native_tp_size=native_tp_size,
        ),
    )
    contexts = tuple(
        ForwardContext(
            {},
            {},
            {},
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            all_moe_layers=[
                f"model.layers.{index}.mlp" for index in bank.layer_indices
            ],
        )
        for _ in rows
    )
    for index, (count, context) in enumerate(zip(rows, contexts, strict=True)):
        context.ubatch_idx = index
        context.input_ids = (
            bank.buffers(index).input_ids
            if count
            else torch.empty(0, dtype=torch.int32)
        )
    model = AFDDeepseekV41FFNForCausalLM.__new__(AFDDeepseekV41FFNForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = SimpleNamespace(
        start_layer=0,
        end_layer=len(bank.layer_indices),
        layers=[
            SimpleNamespace(mlp=GraphContractMoE(index)) for index in bank.layer_indices
        ],
    )
    config = hccl_vllm_config(
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL,
        role="ffn",
        tensor_parallel_size=native_tp_size,
    )
    config.parallel_config.rank = role_rank
    pipeline = pipeline_factory(bank) if pipeline_factory is not None else None
    driver = HcclV41FFNGraph(
        model, contexts, bank, config.parallel_config, pipeline=pipeline
    )
    if pipeline is not None:
        for layer in model.model.layers:
            layer.mlp.current_stream = current_stream
            layer.mlp.expected_stream = pipeline.compute_stream
    return SimpleNamespace(
        bank=bank,
        model=model,
        contexts=contexts,
        parallel=config.parallel_config,
        driver=driver,
        events=events,
    )


def queue_ffn_graph_inputs(case, *, offset=0):
    payloads = {}
    for layer in case.bank.layer_indices:
        for stage, rows in enumerate(case.bank.stage_rows):
            if not rows:
                continue
            hidden = torch.ones(
                (rows, case.bank.capabilities.hidden_size), dtype=torch.bfloat16
            )
            fp32 = hidden.float() + (layer + stage + offset + 1) * 0.00031
            ids = torch.arange(1, rows + 1, dtype=torch.int32) + offset
            payload = HcclLayerInputs(hidden, fp32, ids)
            case.bank.exchanges[stage].endpoint.incoming.extend((hidden, fp32, ids))
            payloads[layer, stage] = payload
    return payloads
