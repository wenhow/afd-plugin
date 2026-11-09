# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Public graph payload and producer seams, with CPU contract arithmetic."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch import nn
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.utils import torch_utils

from tests.unit.fakes.deepseek_v41 import (
    StageAttention,
    contract_ffn,
    make_staged_case,
)
from tests.unit.fakes.hccl_graph_pipeline import attach_pipeline
from tests.unit.fakes.hccl_graph_stage import make_graph_stage_bank


class GraphFFNEndpoint:
    """A protocol peer for one paired stage, without device graph simulation."""

    def __init__(self, stage_idx=0):
        self.group = SimpleNamespace(rank=lambda: 1)
        self.ranks, self.peer_group_rank = (0, 1), 0
        self.messages = []
        self.pending = []
        self.runtime, self.stage_idx = None, stage_idx

    def send(self, tensor, shape):
        assert tuple(tensor.shape) == shape
        message = tensor.clone()
        self.messages.append(message)
        self.pending.append(message)
        if self.runtime is not None:
            self.runtime.trace.append(
                ("wire", "send", self.stage_idx, self.runtime.current.label)
            )

    def receive(self, tensor, shape):
        assert tuple(tensor.shape) == shape
        hidden, original_fp32, ids = self.pending
        tensor.copy_(contract_ffn(hidden, original_fp32, ids))
        self.pending.clear()
        if self.runtime is not None:
            self.runtime.trace.append(
                ("wire", "receive", self.stage_idx, self.runtime.current.label)
            )


class GraphStreamAttention(StageAttention):
    """Observe native public stream queries at the CPU Attention boundary."""

    def __init__(self, original, runtime):
        super().__init__(original.layer_idx, original.shared_state)
        self.runtime = runtime

    def forward(self, positions, hidden_states, scaling=None):
        self.runtime.trace.append(
            (
                "attention",
                self.layer_idx,
                get_forward_context().ubatch_idx,
                self.runtime.current is self.runtime.compute,
                torch_utils.current_stream() is self.runtime.compute,
            )
        )
        return super().forward(positions, hidden_states, scaling)


class GraphParentNorm(nn.Module):
    def __init__(self, runtime):
        super().__init__()
        self.runtime = runtime

    def forward(self, hidden_states):
        self.runtime.trace.append(("parent_output", self.runtime.current.label))
        self.runtime.trace.append(
            (
                "parent_stream",
                "output",
                torch.npu.current_stream() is self.runtime.parent,
                torch_utils.current_stream() is self.runtime.parent,
            )
        )
        return hidden_states


def attach_graph_pipeline(monkeypatch, module, case, *, hybrid=True, runtime=None):
    """Use the actual graph model/bank with shared host stream/event boundaries."""
    pipeline_case = attach_pipeline(
        monkeypatch, module, case.bank, hybrid=hybrid, runtime=runtime
    )
    case.pipeline, case.runtime = pipeline_case.pipeline, pipeline_case.runtime
    for endpoint in case.endpoints:
        endpoint.runtime = case.runtime
    for layer in case.model.layers:
        layer.self_attn = GraphStreamAttention(layer.self_attn, case.runtime)
    case.model.norm = GraphParentNorm(case.runtime)
    return case


def make_graph_model_case(split=(2, 3), *, valid=None, engram=True):
    case = make_staged_case(split, valid=valid, engram=engram)
    for context in case.contexts:
        context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    endpoints = tuple(GraphFFNEndpoint(index) for index in range(len(split)))
    case.bank = make_graph_stage_bank(
        rows=split,
        valid=valid,
        endpoint_factory=lambda index: endpoints[index],
    )
    case.endpoints = endpoints
    case.producer_parents = []
    case.joined = []
    case.runtime = None
    case.model.can_capture_engram_producer = lambda: True

    @contextmanager
    def produce(ids, positions, lookback, query, valid_tokens):
        if case.runtime is not None:
            case.runtime.trace.append(
                (
                    "parent_stream",
                    "producer",
                    torch.npu.current_stream() is case.runtime.parent,
                    torch_utils.current_stream() is case.runtime.parent,
                )
            )
        case.producer_parents.append((ids.clone(), positions.clone(), query.clone()))
        try:
            yield {
                "engram_lookups": case.parent_inputs["engram_lookups"],
                "engram_mask": case.parent_inputs["engram_mask"],
            }
        finally:
            case.joined.append(True)

    case.model.captured_engram_inputs = produce
    case.graph_inputs = dict(case.parent_inputs)
    if engram:
        rows = sum(split)
        case.graph_inputs.update(
            lookback_token_ids=torch.full((rows, 2), -1, dtype=torch.int32),
            engram_query_start_loc=torch.arange(rows + 1, dtype=torch.int32),
            engram_valid_token_count=torch.tensor([rows], dtype=torch.int32),
        )
    return case
