# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow the CPU dependency gates.
# ruff: noqa: E402
"""CPU protocol/model seams only; these do not execute native NPU MoE."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from afd_plugin.connectors.npu import hccl_diagnostics as diagnostics_module
from afd_plugin.connectors.npu.hccl_overlap import (
    HcclOverlapSettings,
    select_hccl_overlap,
)
from afd_plugin.connectors.npu.hccl_staged import HcclStagedAFDConnector
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_plan import HcclParentPlan
from afd_plugin.v1.worker.npu import hccl_v41_ffn_runner as runner_module
from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner
from tests.unit.fakes.hccl import (
    make_staged_ffn_case,
    staged_ffn_expected_output,
    unexpected_hccl_forward_context,
)
from tests.unit.fakes.hccl_diagnostics import make_diagnostics_case


@pytest.fixture
def plan():
    return HcclParentPlan(
        "staged-ffn-parent",
        (0, 2),
        (
            HcclStageLayout(0, (2,), (3,), 1),
            HcclStageLayout(1, (1,), (2,), 1),
        ),
    )


@pytest.fixture
def case(monkeypatch, plan):
    case = make_staged_ffn_case(AFDNPUFFNModelRunner, plan)
    monkeypatch.setattr(runner_module, "ascend_forward_context", case.context.activate)
    return case


def test_eager_binding_rejects_a_session_from_another_connector_wire(case):
    connector = object.__new__(HcclStagedAFDConnector)
    connector.transport = case.peer
    case.runner.connector = connector
    foreign = SimpleNamespace(role="ffn")
    session = SimpleNamespace(transport=foreign, initialized=True)

    with pytest.raises(RuntimeError, match="idle eager FFN session"):
        case.runner.bind_hccl_v41_eager_session(session)

    assert case.peer.messages == []


@pytest.mark.parametrize("mode", ["target", "dummy", "warmup", "profile"])
@pytest.mark.parametrize("num_stages", [1, 2])
def test_stage_diagnostics_report_actual_serial_execution_and_only_sync_live_u1(
    monkeypatch, plan, mode, num_stages
):
    current = replace(plan, mode=mode, stages=plan.stages[:num_stages])
    case = make_staged_ffn_case(AFDNPUFFNModelRunner, current)
    monkeypatch.setattr(runner_module, "ascend_forward_context", case.context.activate)
    selection = select_hccl_overlap(
        HcclOverlapSettings(stage_diagnostics=True, ffn_compute_sync_diagnostics=True),
        role="ffn",
        phase="eager",
        num_stages=num_stages,
    )
    diagnostic = make_diagnostics_case(diagnostics_module, selection)
    runner_module.execute_hccl_v41_parent(
        case.runner, case.peer, diagnostics=diagnostic.diagnostics
    )
    config = diagnostic.records[0]
    assert config["event"] == "configuration"
    assert config["actual.mode"] == "serial"
    assert config["actual.compute_stream"] == config["actual.recv_stream"] == "parent"
    assert config["actual.compute_pipeline_enabled"] is False
    assert config["effective.AFD_HCCL_EAGER_U2_STREAM_OVERLAP"] is (num_stages == 2)
    expected_sync = mode == "target" and num_stages == 1
    assert config["actual.ffn_compute_sync"] is expected_sync
    steps = len(current.layer_indices) * num_stages
    assert diagnostic.events.count("device_sync") == (steps if expected_sync else 0)
    for record in diagnostic.records[1:]:
        assert record["layer"] in current.layer_indices
        assert record["stage"] < num_stages
    assert case.peer.messages[-1] == ("COMPLETE", current.transaction_id)


def test_requested_compute_sync_failure_keeps_session_terminal_without_send(
    monkeypatch, plan
):
    current = replace(plan, stages=plan.stages[:1])
    case = make_staged_ffn_case(AFDNPUFFNModelRunner, current)
    monkeypatch.setattr(runner_module, "ascend_forward_context", case.context.activate)
    selection = select_hccl_overlap(
        HcclOverlapSettings(ffn_compute_sync_diagnostics=True),
        role="ffn",
        phase="eager",
        num_stages=1,
    )
    diagnostic = make_diagnostics_case(diagnostics_module, selection)
    diagnostic.recorder.fail_sync = True
    with pytest.raises(RuntimeError, match="requested diagnostic device sync failed"):
        runner_module.execute_hccl_v41_parent(
            case.runner, case.peer, diagnostics=diagnostic.diagnostics
        )
    assert diagnostic.events == ["configuration", "compute_sync_begin", "device_sync"]
    assert case.peer.messages == [("RECEIVE", current.layer_indices[0], 0)]
    assert case.peer.terminal


@pytest.mark.parametrize("invalid", ["role", "phase", "stages", "rank"])
def test_foreign_eager_diagnostics_are_rejected_before_any_payload(case, invalid):
    selection = select_hccl_overlap(
        HcclOverlapSettings(stage_diagnostics=True),
        role="attention" if invalid == "role" else "ffn",
        phase="graph_capture" if invalid == "phase" else "eager",
        num_stages=1 if invalid == "stages" else 2,
    )
    diagnostic = make_diagnostics_case(
        diagnostics_module, selection, role_rank=1 if invalid == "rank" else 0
    )
    with pytest.raises(ValueError, match="diagnostics must match"):
        runner_module.execute_hccl_v41_parent(
            case.runner, case.peer, diagnostics=diagnostic.diagnostics
        )
    assert diagnostic.records == case.peer.messages == []
    assert case.peer.terminal


@pytest.mark.parametrize("mode", ["target", "dummy", "warmup", "profile"])
@pytest.mark.parametrize(
    "valid,padded,ffn_size,tp_size,role_rank",
    [
        ((1,), (3,), 1, 1, 0),
        ((0, 2), (0, 3), 1, 1, 0),
        ((2, 2), (3, 3), 2, 2, 0),
        ((2, 2), (3, 3), 2, 2, 1),
    ],
)
def test_layer_major_outputs_preserve_aggregate_fp32_ids_and_native_context(
    monkeypatch, plan, mode, valid, padded, ffn_size, tp_size, role_rank
):
    current = replace(
        plan,
        mode=mode,
        stages=(
            HcclStageLayout(0, valid, padded, ffn_size, tp_size),
            HcclStageLayout(1, (0,) * len(valid), (1,) * len(valid), ffn_size, tp_size),
        ),
    )
    case = make_staged_ffn_case(AFDNPUFFNModelRunner, current, role_rank=role_rank)
    monkeypatch.setattr(runner_module, "ascend_forward_context", case.context.activate)

    assert runner_module.execute_hccl_v41_parent(case.runner, case.peer) is None

    steps = [(layer, stage) for layer in current.layer_indices for stage in (0, 1)]
    assert [message[:3] for message in case.peer.messages[:-1]] == [
        (kind, layer, stage) for layer, stage in steps for kind in ("RECEIVE", "OUTPUT")
    ]
    for kind, layer, stage, *output in case.peer.messages[:-1]:
        if kind == "OUTPUT":
            expected = staged_ffn_expected_output(case.peer.inputs[layer, stage], layer)
            torch.testing.assert_close(output[0], expected, rtol=0, atol=0)
    assert case.peer.messages[-1] == ("COMPLETE", current.transaction_id)
    assert case.peer.completed
    assert case.context.current is None


@pytest.mark.parametrize("empty_stage", [0, 1])
def test_wholly_empty_stage_has_no_tensor_messages(monkeypatch, plan, empty_stage):
    stages = tuple(
        HcclStageLayout(index, (0,), (0,), 1) if index == empty_stage else stage
        for index, stage in enumerate(plan.stages)
    )
    case = make_staged_ffn_case(AFDNPUFFNModelRunner, replace(plan, stages=stages))
    monkeypatch.setattr(runner_module, "ascend_forward_context", case.context.activate)

    runner_module.execute_hccl_v41_parent(case.runner, case.peer)

    assert [message[:3] for message in case.peer.messages[:-1]] == [
        (kind, layer, 1 - empty_stage)
        for layer in plan.layer_indices
        for kind in ("RECEIVE", "OUTPUT")
    ]
    assert case.peer.completed


@pytest.mark.parametrize("stage_count", [1, 2])
def test_empty_parent_completes_without_model_or_context(
    monkeypatch, plan, stage_count
):
    empty = replace(
        plan,
        stages=tuple(
            HcclStageLayout(index, (0,), (0,), 1) for index in range(stage_count)
        ),
    )
    case = make_staged_ffn_case(AFDNPUFFNModelRunner, empty)
    case.runner.model = None
    monkeypatch.setattr(
        runner_module, "ascend_forward_context", unexpected_hccl_forward_context
    )

    runner_module.execute_hccl_v41_parent(case.runner, case.peer)

    assert case.peer.messages == [("COMPLETE", empty.transaction_id)]
    assert case.peer.completed


@pytest.mark.parametrize(
    "attribute,value,message",
    [
        ("data_parallel_size", 2, "DP1/PP1"),
        ("pipeline_parallel_size", 2, "DP1/PP1"),
        ("prefill_context_parallel_size", 2, "CP1"),
        ("decode_context_parallel_size", 2, "CP1"),
        ("use_sequence_parallel_moe", True, "non-SP"),
        ("tensor_parallel_size", 2, "native TP size"),
        ("rank", 1, "native rank"),
    ],
)
@pytest.mark.parametrize("empty_parent", [False, True])
def test_invalid_native_coordinates_fail_before_tensor_work(
    case, attribute, value, message, empty_parent
):
    if empty_parent:
        empty = replace(case.plan, stages=(HcclStageLayout(0, (0,), (0,), 1),))
        case = make_staged_ffn_case(AFDNPUFFNModelRunner, empty)
    setattr(case.runner.vllm_config.parallel_config, attribute, value)

    with pytest.raises(ValueError, match=message):
        runner_module.execute_hccl_v41_parent(case.runner, case.peer)

    assert case.peer.messages == []
    assert case.peer.terminal


@pytest.mark.parametrize(
    "attribute,value,message",
    [
        ("role", "attention", "active FFN transport"),
        ("initialized", False, "active FFN transport"),
        ("closed", True, "active FFN transport"),
        ("terminal", True, "active FFN transport"),
        ("plan", None, "accepted before FFN execution"),
    ],
)
def test_invalid_transport_cannot_execute_or_complete(case, attribute, value, message):
    setattr(case.peer, attribute, value)

    with pytest.raises(RuntimeError, match=message):
        runner_module.execute_hccl_v41_parent(case.runner, case.peer)

    assert case.peer.messages == []
    assert case.peer.terminal


def test_separate_tp1_ffn_worlds_require_a_native_group_mapping(monkeypatch):
    plan = HcclParentPlan(
        "unmapped-ffn-world", (0,), (HcclStageLayout(0, (1,), (1,), 2),)
    )
    case = make_staged_ffn_case(AFDNPUFFNModelRunner, plan)
    monkeypatch.setattr(
        runner_module, "ascend_forward_context", unexpected_hccl_forward_context
    )

    with pytest.raises(ValueError, match="role size must match its native TP world"):
        runner_module.execute_hccl_v41_parent(case.runner, case.peer)

    assert case.peer.messages == [] and case.peer.terminal


def test_nonempty_parent_requires_loaded_weights(case):
    case.runner.model = None

    with pytest.raises(RuntimeError, match="NPU FFN model has not been loaded"):
        runner_module.execute_hccl_v41_parent(case.runner, case.peer)

    assert case.peer.messages == []
    assert case.peer.terminal


@pytest.mark.parametrize("failure", ["receive", "compute", "send", "complete"])
def test_failure_escapes_and_marks_session_terminal_without_complete(case, failure):
    if failure == "compute":
        case.runner.model.fail = True
        error, message = ValueError, "controlled staged FFN compute failure"
    else:
        case.peer.fail_at = failure
        error = RuntimeError
        message = (
            "injected staged COMPLETE"
            if failure == "complete"
            else f"injected staged {failure}"
        )

    with pytest.raises(error, match=message):
        runner_module.execute_hccl_v41_parent(case.runner, case.peer)

    assert not any(message[0] == "COMPLETE" for message in case.peer.messages)
    assert case.peer.terminal
    assert not case.peer.completed
    assert case.context.current is None
