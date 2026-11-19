# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Engram producer and readiness boundaries; no hashing or device simulation."""

from types import SimpleNamespace

import torch
from vllm.config import CUDAGraphMode

from afd_plugin.model_executor.models.npu.deepseek_v41_engram import (
    AFDDeepseekV41EngramModelState,
)
from tests.unit.fakes.deepseek_v41 import ParentEngramState


class ExternalParentEngramState(ParentEngramState, AFDDeepseekV41EngramModelState):
    """Graph admission identity over the existing one-shot parent boundary."""

    def __init__(self, inputs):
        ParentEngramState.__init__(self, inputs)
        self.uses_external_engram_graph_producer = True


class EngramParentStream:
    def __init__(self):
        self.messages = []

    def wait_event(self, event):
        self.messages.append(("wait", event))


class EngramProducer:
    def __init__(self, stream):
        self.stream = stream
        self.mask = torch.ones(8, dtype=torch.bool)
        self.lookups = {1: torch.arange(16).reshape(8, 2)}
        self.language_model = SimpleNamespace(model=self)
        self.inputs = None

    def _get_engram_input_buffers(self):
        return self.lookups, self.mask

    def prepare_engram_inputs(self, **inputs):
        self.inputs = inputs
        # The boundary exposes native mode choice; no lookup calculation or
        # NPU event behavior is implemented by this host-only fake.
        self.stream.messages.append(("produce", inputs["cg_mode"]))
        if inputs["cg_mode"] != CUDAGraphMode.NONE:
            raise AssertionError("external graph production needs ordinary events")
        return {
            "engram_lookups": self.lookups,
            "engram_mask": self.mask,
            "engram_mask_ready_event": "mask-ready",
            "engram_pending": {1: "layer-ready"},
        }


def make_external_engram_state(state_class):
    stream = EngramParentStream()
    producer = EngramProducer(stream)
    state = object.__new__(state_class)
    state.model = producer
    state.uses_external_engram_graph_producer = True
    state.lookback_token_ids = torch.full((2, 3), 12, dtype=torch.int32)
    state._engram_inputs = {
        "cg_mode": CUDAGraphMode.FULL,
        "lookback_token_ids": state.lookback_token_ids,
    }
    return SimpleNamespace(state=state, producer=producer, stream=stream)
