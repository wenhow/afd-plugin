# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Public graph payload and producer seams, with CPU contract arithmetic."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode

from tests.unit.fakes.deepseek_v41 import contract_ffn, make_staged_case
from tests.unit.fakes.hccl_graph_stage import make_graph_stage_bank


class GraphFFNEndpoint:
    """A protocol peer for one paired stage, without device graph simulation."""

    def __init__(self):
        self.group = SimpleNamespace(rank=lambda: 1)
        self.ranks, self.peer_group_rank = (0, 1), 0
        self.messages = []
        self.pending = []

    def send(self, tensor, shape):
        assert tuple(tensor.shape) == shape
        message = tensor.clone()
        self.messages.append(message)
        self.pending.append(message)

    def receive(self, tensor, shape):
        assert tuple(tensor.shape) == shape
        hidden, original_fp32, ids = self.pending
        tensor.copy_(contract_ffn(hidden, original_fp32, ids))
        self.pending.clear()


def make_graph_model_case(split=(2, 3), *, valid=None, engram=True):
    case = make_staged_case(split, valid=valid, engram=engram)
    for context in case.contexts:
        context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    endpoints = tuple(GraphFFNEndpoint() for _ in split)
    case.bank = make_graph_stage_bank(
        rows=split,
        valid=valid,
        endpoint_factory=lambda index: endpoints[index],
    )
    case.endpoints = endpoints
    case.producer_parents = []
    case.joined = []
    case.model.can_capture_engram_producer = lambda: True

    @contextmanager
    def produce(ids, positions, lookback, query, valid_tokens):
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
