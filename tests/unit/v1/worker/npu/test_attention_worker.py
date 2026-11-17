# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# ruff: noqa: E402
"""Attention management/execute seams; no NPU timeline is simulated."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

import torch
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm_ascend.worker import worker as native_worker
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.v1.worker.npu.attention_worker import AFDNPUAttentionWorker
from tests.unit.fakes.attention_model_runner_v2 import queue_graph_runner_stop
from tests.unit.fakes.kv_connector import make_native_pd_shutdown_case
from tests.unit.fakes.profiler import (
    PROFILER_TEST_TIMEOUT_SECONDS,
    make_profiled_attention_case,
)


@pytest.fixture
def attention_case():
    return make_profiled_attention_case(AFDNPUAttentionWorker)


@pytest.fixture
def native_pd_shutdown_case(monkeypatch, request):
    stages, failed = request.param
    case = make_native_pd_shutdown_case(
        monkeypatch, stages=stages, fail_sampling=failed
    )
    request.addfinalizer(case.close)
    return case


@pytest.mark.parametrize(
    "native_pd_shutdown_case", [(1, True), (2, True)], indirect=True
)
def test_failed_pd_sampling_rejects_shutdown_before_native_release(
    native_pd_shutdown_case,
):
    case = native_pd_shutdown_case
    assert case.owner.prof is None and case.owner.dspark_pd_consumer.terminal
    assert str(case.cause) == "controlled native D shutdown proposal failure"
    assert not case.receiver.retired and case.receiver.is_alive()
    connector = kv_transfer_state._KV_CONNECTOR_AGENT
    protocol = tuple(case.transfer.sent)

    with pytest.raises(RuntimeError, match="DSpark.*terminal|failed.*DSpark"):
        case.worker.shutdown()

    assert kv_transfer_state._KV_CONNECTOR_AGENT is connector is case.transfer.connector
    assert not case.receiver.retired and case.receiver.is_alive()
    assert not case.transport.closed
    assert not any(
        message.kind in ("STOP", "STOP_ACK")
        for _, _, _, message in case.transport.transcript
    )
    assert case.release_observation is None
    assert tuple(case.transfer.sent) == protocol
    assert case.owner.model is case.target
    assert case.owner.speculator.model is case.draft
    for layer, views in zip(
        case.target.language_model.model.layers, case.target_views, strict=True
    ):
        assert layer.self_attn.dsa_attn.swa_cache_layer.kv_cache is views
    for layer, views in zip(
        case.draft.model.layers.values(), case.draft_views, strict=True
    ):
        assert layer.self_attn.dsa_attn.swa_cache_layer.kv_cache is views
    for actual, expected in zip(
        case.transfer.destination_cache.pages, case.pages, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "native_pd_shutdown_case", [(1, False), (2, False)], indirect=True
)
def test_successful_pd_shutdown_drains_before_native_release(native_pd_shutdown_case):
    case = native_pd_shutdown_case
    assert case.owner.prof is None and not case.owner.dspark_pd_consumer.terminal
    assert case.output.sampled_token_ids == [[7]]
    assert kv_transfer_state._KV_CONNECTOR_AGENT is case.transfer.connector

    case.worker.shutdown()

    assert kv_transfer_state._KV_CONNECTOR_AGENT is None
    assert case.receiver.retired and not case.receiver.is_alive()
    assert case.transport.closed
    assert case.release_observation == {
        "kv_owner_cleared": True,
        "receiver_retired": True,
        "receiver_alive": False,
        "stop_acknowledged": True,
    }
    assert case.owner.kv_caches == []
    for layer in case.target.language_model.model.layers:
        assert layer.self_attn.dsa_attn.swa_cache_layer.kv_cache == []
    for layer in case.draft.model.layers.values():
        assert layer.self_attn.dsa_attn.swa_cache_layer.kv_cache == []


@pytest.mark.parametrize("runner_present", [False, True])
def test_partial_native_shutdown_without_pd_preserves_native_cleanup(
    monkeypatch, runner_present
):
    worker = AFDNPUAttentionWorker.__new__(AFDNPUAttentionWorker)
    worker.profiler = worker.weight_transfer_engine = None
    if runner_present:
        worker.model_runner = None
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", None)
    worker.shutdown()
    assert kv_transfer_state._KV_CONNECTOR_AGENT is None


@pytest.mark.parametrize("result", [None, "native-result"])
def test_successful_native_execute_is_one_step_including_legal_none(
    attention_case, monkeypatch, result
):
    case = attention_case

    def native_execute(*args, **kwargs):
        assert case.worker.afd_profiler_status()["applied"]
        assert case.worker.afd_profiler_status()["steps"] == 0
        return result

    monkeypatch.setattr(NPUModelRunner, "execute_model", native_execute)
    assert case.worker.afd_profiler_status()["pending"]
    assert (
        case.owner.execute_model(SimpleNamespace(total_num_scheduled_tokens=0))
        == result
    )
    status = case.worker.afd_profiler_status()
    assert status["steps"] == 1 and status["step_unit"] == "execute"
    assert not status["closed"] and not status["pending"]
    case.profile.session.finish()


def test_idle_profile_request_is_pending_until_the_next_execute(
    attention_case, monkeypatch
):
    case = attention_case
    monkeypatch.setattr(NPUModelRunner, "execute_model", lambda *a, **k: None)
    case.owner.execute_model(SimpleNamespace(total_num_scheduled_tokens=0))
    case.worker.profile(False)
    status = case.worker.afd_profiler_status()
    assert status["pending"] and status["applied"] and not status["requested"]
    case.owner.execute_model(SimpleNamespace(total_num_scheduled_tokens=0))
    status = case.worker.afd_profiler_status()
    assert not status["pending"] and not status["applied"]
    assert status["steps"] == 1
    case.profile.session.finish()


@pytest.mark.parametrize("export_failure", [False, True])
def test_execute_failure_preserves_native_error_and_exports_on_owner(
    monkeypatch, export_failure
):
    case = make_profiled_attention_case(
        AFDNPUAttentionWorker, failure="stop" if export_failure else None
    )

    def failed_execute(*args, **kwargs):
        raise ValueError("original native execute failure")

    monkeypatch.setattr(NPUModelRunner, "execute_model", failed_execute)
    with pytest.raises(ValueError, match="original native execute failure"):
        case.owner.execute_model(SimpleNamespace(total_num_scheduled_tokens=0))
    status = case.worker.afd_profiler_status()
    assert status["steps"] == 0 and status["closed"] is not export_failure
    assert status["failed"] is export_failure


@pytest.mark.parametrize("boundary", ["foreign-thread", "export-failure"])
def test_profiler_shutdown_failure_retains_native_storage_and_groups(
    monkeypatch, boundary
):
    case = make_profiled_attention_case(
        AFDNPUAttentionWorker, failure="stop" if boundary == "export-failure" else None
    )
    case.profile.session.before_step()
    queue_graph_runner_stop(case)
    monkeypatch.setattr(
        NPUModelRunner, "shutdown", lambda owner: case.events.append("native-close")
    )
    monkeypatch.setattr(
        native_worker,
        "ensure_kv_transfer_shutdown",
        lambda: case.events.append("kv-close"),
    )
    if boundary == "foreign-thread":
        with ThreadPoolExecutor(max_workers=1) as foreign:
            result = foreign.submit(case.worker.shutdown)
            with pytest.raises(RuntimeError, match="execution owner thread"):
                result.result(PROFILER_TEST_TIMEOUT_SECONDS)
        case.profile.session.finish()
    else:
        with pytest.raises(ValueError, match="controlled profiler stop"):
            case.worker.shutdown()
    assert ("graph-sync" in case.events) is (boundary == "export-failure")
    assert "kv-close" not in case.events
    assert "native-close" not in case.events and "groups-close" not in case.events
    assert not case.transport.closed


def test_shutdown_preserves_stop_cache_export_native_group_order(
    attention_case, monkeypatch
):
    case = attention_case
    case.profile.session.before_step()
    queue_graph_runner_stop(case)

    def native_shutdown(owner):
        assert case.profile.session.status()["closed"]
        assert case.cache.closed
        case.events.append("native-close")

    monkeypatch.setattr(NPUModelRunner, "shutdown", native_shutdown)
    monkeypatch.setattr(
        native_worker,
        "ensure_kv_transfer_shutdown",
        lambda: case.events.append("kv-close"),
    )
    case.owner.prepare_shutdown()
    prepared_events = list(case.events)
    case.owner.prepare_shutdown()
    assert case.events == prepared_events
    case.worker.shutdown()
    case.owner.prepare_shutdown()
    stop = next(i for i, event in enumerate(case.events) if event[0] == "stop")
    assert case.events.index("graph-sync") < stop < case.events.index("kv-close")
    assert case.events.index("kv-close") < case.events.index("native-close")
    assert case.events[-1] == "groups-close" and case.transport.closed
    assert any(
        direction == "recv" and message.kind == "STOP_ACK"
        for direction, _, _, message in case.transport.transcript
    )


def test_role_and_native_profiler_conflict_is_an_explicit_management_error(
    attention_case,
):
    case = attention_case
    case.worker.profiler_config = SimpleNamespace(profiler="torch")
    with pytest.raises(ValueError, match="cannot be enabled together"):
        case.worker.profile()
    assert case.worker.afd_profiler_status()["owner_thread_id"] is None


def test_disabled_role_profiler_preserves_native_rpc_result(
    attention_case, monkeypatch
):
    case = attention_case
    case.owner.prof = None
    case.worker.profiler_config = SimpleNamespace(profiler="torch")

    def native_profile(worker, is_start=True, profile_prefix=None):
        assert worker is case.worker
        return is_start, profile_prefix

    monkeypatch.setattr(NPUWorker, "profile", native_profile)
    assert case.worker.profile(False, "round-2") == (False, "round-2")
    assert case.worker.afd_profiler_status() == {
        "mode": "native",
        "configured": True,
        "control_running": False,
    }


@pytest.mark.parametrize("initialized", [False, True])
def test_disabled_role_preserves_native_shutdown_including_partial_worker(
    attention_case, monkeypatch, initialized
):
    worker = (
        attention_case.worker if initialized else object.__new__(AFDNPUAttentionWorker)
    )
    if initialized:
        worker.model_runner.prof = None

    def native_shutdown(owner):
        raise ValueError("native shutdown boundary")

    monkeypatch.setattr(NPUWorker, "shutdown", native_shutdown)
    with pytest.raises(ValueError, match="native shutdown boundary"):
        worker.shutdown()
