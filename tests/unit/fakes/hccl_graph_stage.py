# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU operator queues for an actual graph stage bank; no device emulation."""

from collections import deque
from types import SimpleNamespace

from afd_plugin.connectors.npu.hccl_graph import HcclGraphExchange
from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_graph import HcclGraphDecision
from afd_plugin.distributed.hccl_plan import HcclParentPlan, HcclPlanCapabilities
from tests.unit.fakes.hccl_graph import make_graph_buffers
from tests.unit.fakes.hccl_graph_control import make_graph_key


class StageGraphEndpoint:
    """Public send/receive seam: retain payload snapshots and consume a queue."""

    def __init__(self, *, role, role_rank=0, native_tp_size=1):
        world_rank = role_rank + (native_tp_size if role == "attention" else 0)
        self.group = SimpleNamespace(rank=lambda: world_rank)
        self.ranks = tuple(range(native_tp_size * 2))
        self.peer_group_rank = role_rank + (native_tp_size if role == "ffn" else 0)
        self.messages, self.incoming = [], deque()

    def send(self, tensor, shape):
        self.messages.append(("send", shape, tensor.clone()))

    def receive(self, tensor, shape):
        self.messages.append(("receive", shape))
        tensor.copy_(self.incoming.popleft())


def make_graph_stage_plan(bank, *, valid=None, transaction_id="stage-bank-1"):
    valid = bank.stage_rows if valid is None else valid
    tp = bank.native_tp_size
    decision = HcclGraphDecision(
        transaction_id, "replay", "target", bank.key, bank.stage_rows, "requested"
    )
    return HcclParentPlan(
        transaction_id,
        bank.layer_indices,
        tuple(
            HcclStageLayout(index, (count,) * tp, (rows,) * tp, tp, tp)
            for index, (count, rows) in enumerate(
                zip(valid, bank.stage_rows, strict=True)
            )
        ),
        graph_execution=decision,
    )


def make_graph_stage_bank(
    *,
    role="attention",
    rows=(2, 2),
    valid=None,
    layers=(0, 1),
    hidden=4,
    native_tp_size=1,
    role_rank=0,
    endpoint_factory=None,
    exchange_order="stage_serial",
):
    key = make_graph_key(rows=rows)
    capabilities = HcclPlanCapabilities(
        hidden,
        128,
        sum(rows),
        layers,
        native_tp_size,
        native_tp_size,
        native_tp_size,
        local_counts_version=1,
        graph_execution_version=1,
        graph_exchange_order=exchange_order,
    )
    exchanges = tuple(
        HcclGraphExchange(
            StageGraphEndpoint(
                role=role, role_rank=role_rank, native_tp_size=native_tp_size
            )
            if endpoint_factory is None
            else endpoint_factory(index),
            make_graph_buffers(rows=count, hidden=hidden),
            role=role,
        )
        if count
        else None
        for index, count in enumerate(rows)
    )
    bank = HcclGraphStageBank(
        role=role,
        role_rank=role_rank,
        capabilities=capabilities,
        key=key,
        exchanges=exchanges,
    )
    bank.update_counts(make_graph_stage_plan(bank, valid=valid))
    return bank
