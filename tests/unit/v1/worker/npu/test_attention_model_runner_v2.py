# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Host lifecycle entry seams; these tests do not execute NPU graphs or KV."""

from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from afd_plugin.v1.worker.npu import attention_model_runner_v2 as runner_module
from afd_plugin.v1.worker.npu import hccl_v41_graph_coordinator as coordinator_module
from tests.unit.fakes.attention_model_runner_v2 import (
    install_graph_runner_coordinator,
    make_graph_runner_case,
    queue_graph_runner_stop,
)


@pytest.fixture
def runner_case(monkeypatch):
    monkeypatch.setattr(runner_module, "step_afd_npu_profiler", lambda _: None)
    monkeypatch.setattr(runner_module, "stop_afd_npu_profiler", lambda _: None)
    monkeypatch.setattr(
        coordinator_module, "_get_graph_update_backend", lambda _: AscendSFABackend
    )
    return make_graph_runner_case


@pytest.mark.parametrize("post_kv", [False, True])
def test_native_zero_return_uses_the_current_graph_lifecycle(
    runner_case, monkeypatch, post_kv
):
    case = runner_case()
    case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    if post_kv:
        case.owner.bind_hccl_v41_graph_coordinator(
            install_graph_runner_coordinator(case)
        )
    result = object()
    monkeypatch.setattr(NPUModelRunner, "execute_model", lambda *a, **k: result)

    assert (
        case.owner.execute_model(
            SimpleNamespace(total_num_scheduled_tokens=0),
            dummy_run=True,
            is_profile=True,
        )
        is result
    )

    plans = [
        message.plan
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == "FORWARD"
    ]
    assert len(plans) == 1
    assert plans[0].mode == "profile"
    assert plans[0].graph_execution.phase == "eager"
    assert plans[0].stages[0].attention_padded_tokens == (0,)
    assert case.transport.plan is None and case.transport.graph_reservation is None


@pytest.mark.parametrize("entry", ["execute", "capture", "kv"])
def test_unbound_graph_lifecycle_fails_before_native_work(runner_case, entry):
    case = runner_case()
    with pytest.raises(RuntimeError, match="bind.*graph"):
        if entry == "execute":
            case.owner.execute_model(SimpleNamespace(total_num_scheduled_tokens=4))
        elif entry == "capture":
            case.owner.capture_model()
        else:
            case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    assert case.transport.transcript == []


def test_pre_kv_binding_rejects_another_runner_before_protocol_work(runner_case):
    case, foreign = runner_case(), runner_case()
    with pytest.raises(RuntimeError, match="idle pre-KV owners"):
        case.owner.bind_hccl_v41_graph_eager(foreign.eager, case.session)
    assert case.transport.transcript == []


def test_graph_binding_cannot_replace_an_already_bound_lifecycle(runner_case):
    case = runner_case()
    case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    with pytest.raises(RuntimeError, match="idle pre-KV owners"):
        case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    assert case.transport.transcript == []


def test_real_kv_coordinator_rejects_a_different_pre_kv_session(runner_case):
    case, foreign = runner_case(), runner_case()
    case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    coordinator = install_graph_runner_coordinator(case)
    coordinator.session = foreign.session
    with pytest.raises(RuntimeError, match="pre-KV session owners"):
        case.owner.bind_hccl_v41_graph_coordinator(coordinator)
    assert case.transport.transcript == []


@pytest.mark.parametrize("failure", [None, "cache", "native"])
def test_shutdown_retains_native_owners_until_graph_reset_succeeds(
    runner_case, monkeypatch, failure
):
    case = runner_case(fail_close=failure == "cache")
    case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    queue_graph_runner_stop(case)

    def native_shutdown(owner):
        case.events.append("native-close")
        if failure == "native":
            raise RuntimeError("native shutdown failed")

    monkeypatch.setattr(NPUModelRunner, "shutdown", native_shutdown)
    if failure:
        with pytest.raises(RuntimeError, match="synchronization|shutdown"):
            case.owner.shutdown()
        assert case.owner.hccl_v41_graph_eager is case.eager
        assert not case.transport.closed
    else:
        case.owner.shutdown()
        assert case.owner.hccl_v41_graph_eager is None
        assert case.cache.closed and case.transport.closed

    expected = ["graph-sync"]
    if failure != "cache":
        expected.append("native-close")
    if failure is None:
        expected.append("groups-close")
    assert case.events == expected
    assert any(
        direction == "recv" and message.kind == "STOP_ACK"
        for direction, _, _, message in case.transport.transcript
    )
