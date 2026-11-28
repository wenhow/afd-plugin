# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Rank/parent diagnostic inputs only; no device scheduling is simulated."""

from types import SimpleNamespace

from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.v1.worker.npu.hccl_v41_diagnostics import HcclV41DiagnosticBinding
from tests.unit.fakes.hccl_v41_eager import (
    accept_session_parent,
    make_eager_session_case,
)


def make_diagnostic_case(*, role="attention", stages=2, mode="target", enabled=True):
    case = make_eager_session_case(role=role)
    transport = case.transport
    transport.world_rank = 1 if role == "attention" else 0
    transport.eager_pipeline = None
    records, fences = [], []
    binding = HcclV41DiagnosticBinding(
        transport,
        HcclOverlapSettings(
            stage_diagnostics=enabled, ffn_compute_sync_diagnostics=enabled
        ),
        emit=records.append,
        synchronize=lambda: fences.append("synchronized"),
    )
    accept_session_parent(case, stages=stages, mode=mode)
    return SimpleNamespace(
        case=case, transport=transport, binding=binding, records=records, fences=fences
    )


def attach_diagnostic_owner(case, *, alias_parent=False, alias_sides=False):
    streams = case.case.streams
    parent, compute = streams["parent_stream"], streams["compute_stream"]
    send, recv = streams["send_stream"], streams["receive_stream"]
    if alias_parent:
        compute = send = recv = parent
    elif alias_sides:
        send = recv = compute
    owner = SimpleNamespace(
        transport=case.transport,
        plan=case.transport.plan,
        finished=False,
        failed=False,
        parent_stream=parent,
        send_stream=send,
        receive_stream=recv,
    )
    case.transport.eager_pipeline = owner
    return owner, compute
