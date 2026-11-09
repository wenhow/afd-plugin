# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU worker lifecycle contracts; device callbacks are explicit no-ops."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.hccl_protocol import HcclForwardPlan
from afd_plugin.connectors.npu.hccl_staged import HcclStagedAFDConnector
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_plan import HcclParentPlan
from afd_plugin.v1.worker.npu import ffn_model_runner as runner_module
from afd_plugin.v1.worker.npu import ffn_worker as worker_module
from tests.unit.fakes.hccl import (
    ExitedHcclWorkerThread,
    HcclCpuDeviceRuntime,
    HcclCpuForwardContext,
    HcclFailingModel,
    PendingHcclWorkerThread,
    hccl_layer_tensors,
    join_hccl_worker,
    make_hccl_connector_pair,
    make_hccl_runner,
    make_hccl_worker,
    make_staged_ffn_service_case,
)


@pytest.fixture
def staged_service(monkeypatch):
    plans = [
        HcclParentPlan(
            f"service-parent-{index}",
            (0,),
            tuple(HcclStageLayout(stage, (0,), (0,), 1) for stage in range(count)),
        )
        for index, count in enumerate((1, 2, 1))
    ]
    case = make_staged_ffn_service_case(
        runner_module.AFDNPUFFNModelRunner, worker_module.AFDNPUFFNWorker, plans
    )
    monkeypatch.setattr(
        worker_module,
        "torch",
        SimpleNamespace(
            npu=HcclCpuDeviceRuntime(), bfloat16=runner_module.torch.bfloat16
        ),
    )
    return case


def test_bound_staged_service_completes_u1_u2_parents_and_acknowledges_stop(
    staged_service,
):
    case = staged_service
    case.worker.bind_hccl_v41_transport(case.peer)
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    case.worker.stop_ffn_server_loop()

    assert case.peer.messages == [
        message
        for index in range(3)
        for message in (
            ("FORWARD", f"service-parent-{index}"),
            ("COMPLETE", f"service-parent-{index}"),
        )
    ] + [("STOP_ACK",)]
    assert case.peer.closed
    assert case.worker.poll_hccl_ffn_service() is False
    assert case.worker.stop_ffn_server_loop() is None


def test_bound_staged_service_preserves_compute_failure_and_retains_resources(
    staged_service,
):
    case = staged_service
    case.peer.parents[0] = HcclParentPlan(
        "unloaded-model", (0,), (HcclStageLayout(0, (1,), (1,), 1),)
    )
    case.worker.bind_hccl_v41_transport(case.peer)
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)

    for action in (case.worker.poll_hccl_ffn_service, case.worker.stop_ffn_server_loop):
        with pytest.raises(RuntimeError, match="worker loop failed") as failure:
            action()
        assert "model has not been loaded" in str(failure.value.__cause__)
    assert case.peer.terminal and not case.peer.closed
    assert case.peer.messages == [("FORWARD", "unloaded-model")]


def test_bound_staged_service_cannot_release_initialized_transport_before_peer_stop(
    staged_service,
):
    case = staged_service
    case.worker.bind_hccl_v41_transport(case.peer)
    case.peer.initialize()

    with pytest.raises(RuntimeError, match="requires peer STOP; retain resources"):
        case.worker.stop_ffn_server_loop()

    assert not case.peer.closed


@pytest.mark.parametrize(
    "boundary",
    [
        "role",
        "version",
        "native-tp",
        "dp",
        "eager",
        "device",
        "hidden",
        "vocab",
        "layers",
        "capacity",
    ],
)
def test_staged_service_rejects_unsupported_binding_before_protocol(
    staged_service, boundary
):
    case = staged_service
    if boundary == "role":
        case.peer.role = "attention"
    elif boundary == "version":
        case.peer.negotiated_capabilities = replace(
            case.peer.negotiated_capabilities, local_counts_version=0
        )
    elif boundary == "native-tp":
        case.worker.vllm_config.parallel_config.tensor_parallel_size = 2
    elif boundary == "dp":
        case.worker.vllm_config.parallel_config.data_parallel_size = 2
    elif boundary == "eager":
        case.worker.vllm_config.model_config.enforce_eager = False
    elif boundary == "device":
        case.peer.local_device = 1
    else:
        field, value = {
            "hidden": ("hidden_size", 32),
            "vocab": ("vocab_size", 2048),
            "layers": ("layer_indices", (0, 1)),
            "capacity": ("max_tokens", 17),
        }[boundary]
        case.peer.negotiated_capabilities = replace(
            case.peer.negotiated_capabilities, **{field: value}
        )

    with pytest.raises(ValueError, match="(?i)hccl"):
        case.worker.bind_hccl_v41_transport(case.peer)

    assert not case.peer.initialized
    assert case.peer.messages == []


def test_staged_service_rejects_a_second_transport_for_its_connector(staged_service):
    case = staged_service
    config = case.runner.vllm_config
    connector = HcclStagedAFDConnector(0, 0, config, parse_afd_config(config), 0)
    case.runner.connector = connector

    with pytest.raises(ValueError, match="must use its connector transport"):
        case.worker.bind_hccl_v41_transport(case.peer)

    assert not case.peer.initialized
    assert not connector.is_initialized
    assert case.peer.messages == []


@pytest.fixture
def lifecycle(monkeypatch):
    connectors = make_hccl_connector_pair(monkeypatch, layer_count=1)
    context = HcclCpuForwardContext()
    monkeypatch.setattr(runner_module, "ascend_forward_context", context.activate)
    monkeypatch.setattr(
        worker_module, "torch", SimpleNamespace(npu=HcclCpuDeviceRuntime())
    )
    attention, ffn = connectors
    runner = make_hccl_runner(runner_module.AFDNPUFFNModelRunner, ffn, None)
    worker = make_hccl_worker(worker_module.AFDNPUFFNWorker, runner)
    yield attention, ffn, runner, worker
    for connector in connectors:
        connector.close()


def test_worker_acknowledges_peer_stop_before_releasing_connector(lifecycle):
    attention, ffn, _, worker = lifecycle
    attention.request_stop()

    worker.start_ffn_server_loop()
    join_hccl_worker(worker)
    assert worker.stop_ffn_server_loop() is None

    assert not attention.is_initialized
    assert not ffn.is_initialized
    assert worker.raise_ffn_loop_error_if_any() is None
    assert worker.poll_hccl_ffn_service() is False
    assert worker.stop_ffn_server_loop() is None


def test_worker_keeps_compute_failure_terminal_across_raise_start_and_stop(lifecycle):
    attention, ffn, runner, worker = lifecycle
    runner.model = HcclFailingModel()
    plan = HcclForwardPlan("failing-existing-forward", (0,), 2, 3)
    hidden, fp32, ids = hccl_layer_tensors()
    attention.begin_forward_plan(plan)
    context = AFDTransferContext(AFDTransferMetadata(0, 0, [3]))
    attention.send_attn_output(hidden, context, hidden_states_fp32=fp32, input_ids=ids)

    worker.start_ffn_server_loop()
    join_hccl_worker(worker)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="FFN worker loop failed") as failure:
            worker.raise_ffn_loop_error_if_any()
        assert isinstance(failure.value.__cause__, ValueError)
        assert str(failure.value.__cause__) == "controlled FFN compute failure"
    with pytest.raises(RuntimeError, match="FFN worker loop failed"):
        worker.start_ffn_server_loop()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="FFN worker loop failed") as failure:
            worker.poll_hccl_ffn_service()
        assert isinstance(failure.value.__cause__, ValueError)
    with pytest.raises(RuntimeError, match="FFN worker loop failed"):
        worker.stop_ffn_server_loop()
    assert ffn.is_initialized


@pytest.mark.parametrize("thread", [None, ExitedHcclWorkerThread()])
def test_worker_poll_rejects_service_without_a_live_thread_or_peer_stop(
    lifecycle, thread
):
    _, ffn, _, worker = lifecycle
    worker._ffn_thread = thread

    with pytest.raises(RuntimeError, match="HCCL"):
        worker.poll_hccl_ffn_service()

    assert ffn.is_initialized


def test_worker_retains_resources_while_peer_stop_or_device_is_pending(lifecycle):
    _, ffn, _, worker = lifecycle
    worker._ffn_thread = PendingHcclWorkerThread()
    assert worker.poll_hccl_ffn_service() is True

    with pytest.raises(RuntimeError, match="retain resources"):
        worker.stop_ffn_server_loop()

    assert ffn.is_initialized
