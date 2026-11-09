# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU worker lifecycle contracts; device callbacks are explicit no-ops."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.hccl_protocol import HcclForwardPlan
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
)


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
