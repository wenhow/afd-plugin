# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Actual route/rank diagnostic contracts, without device acceptance claims."""

from dataclasses import replace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings  # noqa: E402
from tests.unit.fakes.hccl_v41_diagnostics import (  # noqa: E402
    attach_diagnostic_owner,
    make_diagnostic_case,
)


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("mode", ["target", "profile", "warmup", "dummy"])
def test_serial_parent_records_actual_rank_and_only_live_ffn_u1_can_sync(
    role,
    stages,
    mode,
):
    case = make_diagnostic_case(role=role, stages=stages, mode=mode)

    progress = case.binding.bind_parent()
    progress.synchronize_compute(case.transport.plan.layer_indices[0], 0)

    config = case.records[0]
    expected_sync = role == "ffn" and stages == 1 and mode == "target"
    assert config["event"] == "configuration"
    assert config["world_rank"] == case.transport.world_rank
    assert config["role_rank"] == case.transport.role_rank
    assert config["actual.mode"] == "serial"
    assert config["actual.send_stream"] == config["actual.recv_stream"] == "parent"
    assert config["actual.ffn_compute_sync"] == expected_sync
    assert case.fences == (["synchronized"] if expected_sync else [])
    assert case.transport.messages == []


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize(
    "alias_parent,alias_sides", [(False, False), (True, False), (False, True)]
)
def test_ticket_parent_reports_real_stream_aliases_without_claiming_overlap(
    role,
    alias_parent,
    alias_sides,
):
    case = make_diagnostic_case(role=role)
    owner, compute = attach_diagnostic_owner(
        case,
        alias_parent=alias_parent,
        alias_sides=alias_sides,
    )

    progress = case.binding.bind_parent(eager_pipeline=owner, compute_stream=compute)
    progress.synchronize_compute(case.transport.plan.layer_indices[0], 0)

    config = case.records[0]
    assert config["actual.mode"] == "eager-streamed"
    assert config["actual.compute_stream"] == ("parent" if alias_parent else "compute")
    assert config["actual.send_stream"] == ("parent" if alias_parent else "send")
    assert config["actual.recv_stream"] == ("parent" if alias_parent else "recv")
    assert config["actual.ffn_compute_sync"] is False
    assert case.fences == []
    assert case.transport.messages == []


@pytest.mark.parametrize("invalid", ["foreign", "stale", "finished", "failed"])
def test_ticket_diagnostics_reject_wrong_owner_before_emitting_configuration(invalid):
    case = make_diagnostic_case()
    owner, compute = attach_diagnostic_owner(case)
    if invalid == "foreign":
        owner.transport = make_diagnostic_case().transport
    elif invalid == "stale":
        owner.plan = replace(owner.plan, transaction_id="other-parent")
    else:
        setattr(owner, invalid, True)

    with pytest.raises(ValueError, match="exact active owner"):
        case.binding.bind_parent(eager_pipeline=owner, compute_stream=compute)

    assert case.records == []
    assert case.transport.messages == []
    assert not case.transport.terminal


@pytest.mark.parametrize("invalid", ["foreign", "closed", "terminal", "active"])
def test_diagnostic_setup_rejects_foreign_or_nonidle_transport(invalid):
    case = make_diagnostic_case()
    transport = case.transport
    if invalid != "active":
        transport.plan = None
    if invalid == "foreign":
        transport = make_diagnostic_case().transport
    elif invalid in ("closed", "terminal"):
        setattr(transport, invalid, True)

    with pytest.raises(ValueError, match="exact idle live transport"):
        case.binding.require_idle(transport)

    assert case.records == []


def test_disabled_diagnostics_emit_no_records_or_fences():
    case = make_diagnostic_case(role="ffn", stages=1, enabled=False)
    assert case.binding.bind_parent() is None
    assert case.records == case.fences == []


def test_diagnostics_reject_settings_that_conflict_with_selected_execution():
    case = make_diagnostic_case()
    with pytest.raises(ValueError, match="settings must match"):
        case.binding.require_settings(HcclOverlapSettings())
    assert case.records == []
