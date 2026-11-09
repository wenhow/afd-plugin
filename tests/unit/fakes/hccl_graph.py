# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU protocol seam for the static HCCL graph exchange; no device simulation."""

from collections import deque
from types import SimpleNamespace

import torch

from afd_plugin.connectors.npu.hccl_graph import (
    HcclGraphBuffers,
    HcclGraphEndpoint,
    HcclGraphExchange,
)


class GraphCollectiveRecorder:
    def __init__(self):
        self.messages = []
        self.incoming = deque()

    def send(self, tensor, peer, ranks, group_tag, tag, group_peer, shape):
        self.messages.append(
            (
                "send",
                peer,
                tuple(ranks),
                group_tag,
                tag,
                group_peer,
                shape,
                tensor.clone(),
            )
        )

    def receive(self, tensor, peer, ranks, group_tag, tag, group_peer, shape):
        self.messages.append(
            (
                "receive",
                peer,
                tuple(ranks),
                group_tag,
                tag,
                group_peer,
                shape,
                tensor.clone(),
            )
        )
        return self.incoming.popleft().clone()


def graph_process_group(monkeypatch, graph_module, *, backend="hccl", ranks=(10, 20)):
    group = SimpleNamespace()
    monkeypatch.setattr(
        graph_module,
        "dist",
        SimpleNamespace(
            get_backend=lambda group: backend,
            get_process_group_ranks=lambda group: ranks,
            get_rank=lambda group: 0,
        ),
    )
    monkeypatch.setattr(
        graph_module, "c10d", SimpleNamespace(_get_group_tag=lambda group: "graph-test")
    )
    return group


def make_graph_exchange(
    monkeypatch, graph_module, *, role="attention", rows=3, hidden=2
):
    group = graph_process_group(monkeypatch, graph_module)
    endpoint = HcclGraphEndpoint.from_process_group(group, 20)
    recorder = GraphCollectiveRecorder()
    monkeypatch.setattr(
        graph_module.torch.ops.npu_define._send, "default", recorder.send
    )
    monkeypatch.setattr(
        graph_module.torch.ops.npu_define._recv, "default", recorder.receive
    )
    buffers = HcclGraphBuffers(
        torch.zeros((rows, hidden), dtype=torch.bfloat16),
        torch.zeros((rows, hidden), dtype=torch.float32),
        torch.zeros(rows, dtype=torch.int32),
        torch.zeros((rows, hidden), dtype=torch.bfloat16),
    )
    return HcclGraphExchange(endpoint, buffers, role=role), recorder
