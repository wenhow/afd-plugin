# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow the CPU dependency gates.
# ruff: noqa: E402
"""CPU execution seams; no native MoE or NPU correctness is represented."""

from __future__ import annotations

from copy import copy
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.hccl_protocol import HcclForwardPlan
from afd_plugin.v1.worker.npu import ffn_model_runner as runner_module
from afd_plugin.v1.worker.npu import hccl_v41_graph_service as service_module
from tests.unit.fakes.hccl import (
    ROUTER_DELTA_SCALE,
    HcclContractModel,
    HcclCpuForwardContext,
    HcclFailingModel,
    hccl_layer_tensors,
    make_hccl_connector_pair,
    make_hccl_runner,
    unexpected_hccl_forward_context,
)
from tests.unit.fakes.hccl_v41_graph_service import make_graph_service_case


@pytest.fixture
def pair(monkeypatch):
    connectors = make_hccl_connector_pair(monkeypatch, layer_count=1)
    yield connectors
    for connector in connectors:
        connector.close()


@pytest.fixture
def execution_context(monkeypatch):
    context = HcclCpuForwardContext()
    monkeypatch.setattr(runner_module, "ascend_forward_context", context.activate)
    return context


@pytest.fixture
def plan():
    return HcclForwardPlan("existing-runner-forward", (0,), 2, 3)


@pytest.fixture
def graph_case(monkeypatch):
    return make_graph_service_case(service_module, monkeypatch)


def test_bound_graph_runner_serves_capture_replay_and_stop_before_cache_release(
    graph_case,
):
    case = graph_case
    case.runner.bind_hccl_v41_graph_service(case.service)
    case.transport.queue_parent("capture", identifier="capture-1")
    assert case.runner.execute_hccl_v41_step(case.transport)
    case.transport.queue_parent("replay", identifier="replay-1", valid=(1, 0))
    assert case.runner.execute_hccl_v41_step(case.transport)
    assert "graph-replay" in case.events and not case.cache.closed
    case.transport.queue_stop()
    assert case.runner.execute_hccl_v41_step(case.transport) is False
    case.runner.close_hccl_v41_graph_service(case.transport)
    assert case.cache.closed and not case.transport.closed
    assert case.events[-2:] == ["device-sync", "graph-reset"]


def test_graph_runner_rejects_unbound_service_before_receiving_peer_messages(
    graph_case,
):
    case = graph_case
    with pytest.raises(RuntimeError, match="has not been bound"):
        case.runner.execute_hccl_v41_step(case.transport)
    assert case.transport.terminal and case.events == []


def test_graph_runner_rejects_foreign_transport_before_receiving_peer_messages(
    graph_case,
):
    case = graph_case
    case.runner.bind_hccl_v41_graph_service(case.service)
    other = copy(case.transport)
    with pytest.raises(ValueError, match="bound session"):
        case.runner.execute_hccl_v41_step(other)
    assert other.terminal and not case.transport.terminal
    assert case.events == []


def test_graph_runner_rejects_a_service_owned_by_another_runner(graph_case):
    case = graph_case
    service = copy(case.service)
    service.runner = copy(case.runner)
    with pytest.raises(ValueError, match="this FFN runner/session"):
        case.runner.bind_hccl_v41_graph_service(service)
    assert case.events == [] and not case.transport.terminal


def test_graph_runner_rejects_rebinding_before_peer_work(graph_case):
    case = graph_case
    case.runner.bind_hccl_v41_graph_service(case.service)
    with pytest.raises(RuntimeError, match="already bound"):
        case.runner.bind_hccl_v41_graph_service(case.service)
    assert case.events == []


def test_graph_runner_retains_cached_capture_when_cleanup_precedes_stop(graph_case):
    case = graph_case
    case.runner.bind_hccl_v41_graph_service(case.service)
    case.transport.queue_parent("capture", identifier="capture-1")
    assert case.runner.execute_hccl_v41_step(case.transport)
    with pytest.raises(RuntimeError, match="STOP"):
        case.runner.close_hccl_v41_graph_service(case.transport)
    assert not case.cache.closed
    assert case.cache.contains(case.transport.base_bank.key)
    assert "graph-reset" not in case.events


@pytest.mark.parametrize(
    ("mode", "valid_tokens"),
    [("target", 2), ("dummy", 0), ("warmup", 0), ("profile", 0)],
)
def test_ffn_step_returns_output_from_fp32_ids_and_matching_context(
    pair, execution_context, plan, mode, valid_tokens
):
    attention, ffn = pair
    plan = replace(plan, mode=mode, valid_tokens=valid_tokens)
    model = HcclContractModel(execution_context, plan)
    runner = make_hccl_runner(runner_module.AFDNPUFFNModelRunner, ffn, model)
    hidden, fp32, ids = hccl_layer_tensors()
    attention.begin_forward_plan(plan)
    assert ffn.receive_forward_plan() == plan
    context = AFDTransferContext(AFDTransferMetadata(0, 0, [plan.padded_tokens]))
    attention.send_attn_output(hidden, context, hidden_states_fp32=fp32, input_ids=ids)

    assert runner.execute_hccl_step(plan) is None

    output = attention.recv_ffn_output(torch.empty_like(hidden), context=context)
    expected = ((fp32 - hidden.float()) * ROUTER_DELTA_SCALE + ids[:, None].float()).to(
        torch.bfloat16
    )
    assert torch.equal(output, expected)
    assert attention.active_forward_plan is None
    assert ffn.active_forward_plan is None


def test_ffn_step_accepts_consecutive_existing_forward_transactions(
    pair, execution_context, plan
):
    attention, ffn = pair
    hidden, fp32, ids = hccl_layer_tensors()
    for transaction_id in ("first-existing-forward", "second-existing-forward"):
        current = replace(plan, transaction_id=transaction_id)
        runner = make_hccl_runner(
            runner_module.AFDNPUFFNModelRunner,
            ffn,
            HcclContractModel(execution_context, current),
        )
        attention.begin_forward_plan(current)
        assert ffn.receive_forward_plan() == current
        context = AFDTransferContext(AFDTransferMetadata(0, 0, [3]))
        attention.send_attn_output(
            hidden, context, hidden_states_fp32=fp32, input_ids=ids
        )
        runner.execute_hccl_step(current)
        output = attention.recv_ffn_output(torch.empty_like(hidden), context=context)
        assert output.shape == hidden.shape
        assert ffn.active_forward_plan is None


def test_empty_ffn_step_needs_no_model_or_native_context(pair, monkeypatch, plan):
    attention, ffn = pair
    empty = replace(plan, valid_tokens=0, padded_tokens=0)
    monkeypatch.setattr(
        runner_module, "ascend_forward_context", unexpected_hccl_forward_context
    )
    runner = make_hccl_runner(runner_module.AFDNPUFFNModelRunner, ffn, None)
    attention.begin_forward_plan(empty)
    assert ffn.receive_forward_plan() == empty

    assert runner.execute_hccl_step(empty) is None

    assert attention.active_forward_plan is None
    assert ffn.active_forward_plan is None


def test_ffn_step_rejects_a_plan_that_has_not_been_accepted(
    pair, execution_context, plan
):
    _, ffn = pair
    runner = make_hccl_runner(
        runner_module.AFDNPUFFNModelRunner,
        ffn,
        HcclContractModel(execution_context, plan),
    )

    with pytest.raises(RuntimeError, match="accepted before execution"):
        runner.execute_hccl_step(plan)


def test_ffn_step_requires_a_loaded_model(pair, plan):
    attention, ffn = pair
    runner = make_hccl_runner(runner_module.AFDNPUFFNModelRunner, ffn, None)
    attention.begin_forward_plan(plan)
    assert ffn.receive_forward_plan() == plan

    with pytest.raises(RuntimeError, match="model has not been loaded"):
        runner.execute_hccl_step(plan)


def test_ffn_compute_error_escapes_and_prevents_the_next_plan(
    pair, execution_context, plan
):
    attention, ffn = pair
    runner = make_hccl_runner(
        runner_module.AFDNPUFFNModelRunner, ffn, HcclFailingModel()
    )
    hidden, fp32, ids = hccl_layer_tensors()
    attention.begin_forward_plan(plan)
    assert ffn.receive_forward_plan() == plan
    context = AFDTransferContext(AFDTransferMetadata(0, 0, [3]))
    attention.send_attn_output(hidden, context, hidden_states_fp32=fp32, input_ids=ids)

    with pytest.raises(ValueError, match="controlled FFN compute failure"):
        runner.execute_hccl_step(plan)
    with pytest.raises(RuntimeError, match="previous HCCL forward"):
        ffn.receive_forward_plan()
