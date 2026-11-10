# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU loan/protocol collaborator for native HC, with no device simulation."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from afd_plugin.compat.npu.stream import use_npu_current_stream
from afd_plugin.connectors.npu import hccl_eager_pipeline, hccl_staged_transport
from afd_plugin.connectors.npu.hccl_diagnostics import HcclDiagnosticExecution
from afd_plugin.connectors.npu.hccl_eager_pipeline import HcclAttentionTicket
from tests.unit.fakes.deepseek_v41 import contract_ffn, make_staged_case
from tests.unit.fakes.hccl import StagedModelPeer
from tests.unit.fakes.hccl_eager_pipeline import (
    make_eager_pair,
    make_eager_progress,
    open_eager_pair,
)
from tests.unit.fakes.hccl_graph_pipeline import HostEvent
from tests.unit.fakes.npu_stream import HostStreamRecorder


class EagerModelPipeline:
    def __init__(self, peer, runtime, *, fail_step=None):
        self.transport, self.plan = peer, peer.plan
        self.parent_stream = runtime.parent
        self.send_stream = self.receive_stream = runtime.parent
        self.runtime, self.fail_step = runtime, fail_step
        self.finished = self.failed = False
        self.messages = []
        self.outputs = {}

    def bind_progress(self, progress, *, compute_stream):
        progress.validate_parent(
            role=self.transport.role,
            world_rank=self.transport.world_rank,
            role_rank=self.transport.role_rank,
            num_stages=len(self.plan.stages),
            layer_indices=self.plan.layer_indices,
            transaction_id=self.plan.transaction_id,
        )
        progress.validate_execution(
            HcclDiagnosticExecution(
                "eager-streamed",
                "parent" if compute_stream == self.parent_stream else "compute",
                stream_aliases=(False, False, True)
                if compute_stream != self.parent_stream
                else (True, True, True),
            )
        )

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


def make_empty_ticket_execution_case(monkeypatch, module):
    """Real ticket COMPLETE plus a zero-row native-call seam, without device IO."""
    pair = make_eager_pair(
        monkeypatch, hccl_eager_pipeline, hccl_staged_transport, rows=(0, 0)
    )
    attention, _ = open_eager_pair(pair)
    monkeypatch.setattr(module.torch, "npu", pair.runtime, raising=False)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    case = make_staged_case((0, 0))
    diagnostic = make_eager_progress(attention, pair.runtime.compute)
    return SimpleNamespace(
        case=case,
        peer=pair.attention,
        pipeline=attention,
        runtime=pair.runtime,
        diagnostic=diagnostic,
    )
