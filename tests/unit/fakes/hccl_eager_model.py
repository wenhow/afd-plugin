# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU loan/protocol collaborator for native HC, with no device simulation."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from afd_plugin.compat.npu.stream import use_npu_current_stream
from afd_plugin.connectors.npu.hccl_eager_pipeline import HcclAttentionTicket
from tests.unit.fakes.deepseek_v41 import contract_ffn, make_staged_case
from tests.unit.fakes.hccl import StagedModelPeer
from tests.unit.fakes.hccl_graph_pipeline import HostEvent
from tests.unit.fakes.npu_stream import HostStreamRecorder


class EagerModelPipeline:
    def __init__(self, peer, runtime, *, fail_step=None):
        self.transport, self.plan = peer, peer.plan
        self.parent_stream = runtime.parent
        self.runtime, self.fail_step = runtime, fail_step
        self.finished = self.failed = False
        self.messages = []
        self.outputs = {}

    def post_attention(
        self, layer, stage, hidden, fp32, ids, *, producer_stream, retain=()
    ):
        ticket = HcclAttentionTicket(layer, stage)
        self.outputs[ticket] = contract_ffn(hidden, fp32, ids)
        self.messages.append(
            ("post", layer, stage, hidden.clone(), fp32.clone(), ids.clone())
        )
        return ticket

    @contextmanager
    def attention_output(self, ticket, *, consumer_stream, retain=()):
        self.messages.append(("consume", ticket.layer_idx, ticket.stage_idx))
        with use_npu_current_stream(consumer_stream):
            if self.fail_step == (ticket.layer_idx, ticket.stage_idx):
                raise RuntimeError("injected eager output failure")
            yield self.outputs.pop(ticket)

    def finish(self):
        assert not self.outputs
        self.finished = self.transport.completed = True

    def fail(self):
        self.failed = self.transport.terminal = True


def make_eager_model_case(
    monkeypatch, module, *, split=(2, 3), engram=True, valid=None, fail_step=None
):
    case = make_staged_case(split, engram=engram, valid=valid)
    peer = StagedModelPeer(case.plan)
    runtime = HostStreamRecorder(torch.device("cpu"))
    runtime.Event = lambda: HostEvent(runtime, f"event-{len(runtime.trace)}")
    monkeypatch.setattr(module.torch, "npu", runtime, raising=False)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    pipeline = EagerModelPipeline(peer, runtime, fail_step=fail_step)
    peer.eager_pipeline = pipeline
    return SimpleNamespace(case=case, peer=peer, pipeline=pipeline, runtime=runtime)
