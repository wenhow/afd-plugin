# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.profiler import (
    afd_npu_profiler_config,
    afd_npu_profiler_execute,
    afd_npu_profiler_owner,
    create_afd_npu_profiler,
    create_deferred_afd_npu_profiler,
    step_afd_npu_profiler,
    stop_afd_npu_profiler,
)
from tests.unit.fakes.profiler import (
    PROFILER_TEST_TIMEOUT_SECONDS,
    FakeTorchNPUProfiler,
    StepProfiler,
    make_profiler_session_case,
)

_ENV_NAMES = (
    "AFD_NPU_ATTENTION_PROFILER_ENABLE",
    "AFD_NPU_ATTENTION_PROFILER_WAIT",
    "AFD_NPU_ATTENTION_PROFILER_WARMUP",
    "AFD_NPU_ATTENTION_PROFILER_ACTIVE",
    "AFD_NPU_ATTENTION_PROFILER_REPEAT",
    "AFD_NPU_ATTENTION_PROFILER_SKIP_FIRST",
    "AFD_NPU_ATTENTION_PROFILER_DIR",
    "AFD_NPU_ATTENTION_PROFILER_WITH_STACK",
    "AFD_NPU_FFN_PROFILER_ENABLE",
    "AFD_NPU_FFN_PROFILER_WAIT",
    "AFD_NPU_FFN_PROFILER_WARMUP",
    "AFD_NPU_FFN_PROFILER_ACTIVE",
    "AFD_NPU_FFN_PROFILER_REPEAT",
    "AFD_NPU_FFN_PROFILER_SKIP_FIRST",
    "AFD_NPU_FFN_PROFILER_DIR",
    "AFD_NPU_FFN_PROFILER_WITH_STACK",
    "AFD_NPU_ATTENTION_PROFILER_RANKS",
    "AFD_NPU_FFN_PROFILER_RANKS",
    "VLLM_ASCEND_MODEL_RUNNER_PROFILER_ENABLE",
    "VLLM_ASCEND_FFN_PROFILER_ENABLE",
    "VLLM_TORCH_PROFILER_DIR",
)


@pytest.fixture(autouse=True)
def _clear_profiler_env(monkeypatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_npu_profiler_defaults_are_disabled():
    attention = afd_npu_profiler_config("attention")
    ffn = afd_npu_profiler_config("ffn")

    assert attention.enabled is False
    assert attention.wait == 2
    assert attention.warmup == 1
    assert attention.active == 10
    assert attention.repeat == 1
    assert attention.skip_first == 1500
    assert attention.trace_dir == "/tmp/profile/attn"
    assert attention.with_stack is False
    assert ffn.enabled is False
    assert ffn.active == 20
    assert ffn.trace_dir == "/tmp/profile/ffn"
    assert ffn.with_stack is False


def test_npu_profiler_uses_only_plugin_owned_enable_env(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_FFN_PROFILER_ENABLE", "1")

    assert afd_npu_profiler_config("ffn").enabled is False

    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "1")

    assert afd_npu_profiler_config("ffn").enabled is True


def test_npu_profiler_dir_falls_back_to_vllm_torch_profiler_dir(monkeypatch):
    monkeypatch.setenv("VLLM_TORCH_PROFILER_DIR", "/tmp/vllm-profile")

    assert afd_npu_profiler_config("attention").trace_dir == "/tmp/vllm-profile"

    monkeypatch.setenv("AFD_NPU_ATTENTION_PROFILER_DIR", "/tmp/afd-attn")

    assert afd_npu_profiler_config("attention").trace_dir == "/tmp/afd-attn"


def test_create_npu_profiler_uses_configured_schedule(monkeypatch):
    profiler_module = FakeTorchNPUProfiler()
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(profiler=profiler_module),
    )
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "true")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_WAIT", "3")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_WARMUP", "4")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ACTIVE", "5")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_REPEAT", "6")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_SKIP_FIRST", "7")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_DIR", "/tmp/afd-ffn")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_WITH_STACK", "true")

    profiler = create_afd_npu_profiler("ffn")

    assert profiler is profiler_module.created_profiler
    assert profiler.started is True
    assert profiler_module.schedule_kwargs == {
        "wait": 3,
        "warmup": 4,
        "active": 5,
        "repeat": 6,
        "skip_first": 7,
    }
    assert profiler_module.profile_kwargs["record_shapes"] is True
    assert profiler_module.profile_kwargs["with_stack"] is True
    assert profiler_module.profile_kwargs["with_modules"] is True
    assert profiler_module.trace_dir == "/tmp/afd-ffn"


def test_step_npu_profiler_ignores_disabled_profiler():
    step_afd_npu_profiler(None)

    profiler = StepProfiler()
    step_afd_npu_profiler(profiler)

    assert profiler.steps == 1


def test_stop_npu_profiler_ignores_disabled_profiler():
    stop_afd_npu_profiler(None)

    profiler = StepProfiler()
    stop_afd_npu_profiler(profiler)

    assert profiler.stopped is True


@pytest.mark.parametrize(
    "value,expected",
    [(None, frozenset({0})), ("all", None), (" 2,0,2 ", frozenset({0, 2}))],
)
def test_role_profiler_rank_selection_matches_donor(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv("AFD_NPU_FFN_PROFILER_RANKS", value)
    assert afd_npu_profiler_config("ffn").ranks == expected


@pytest.mark.parametrize("value", ["", "-1", "0,no", "0,"])
def test_role_profiler_rejects_invalid_rank_selection(monkeypatch, value):
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_RANKS", value)
    with pytest.raises(ValueError, match="AFD_NPU_FFN_PROFILER_RANKS"):
        afd_npu_profiler_config("ffn")


def test_deferred_profiler_is_disabled_without_loading_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    assert create_deferred_afd_npu_profiler("ffn") is None


@pytest.mark.parametrize("rank,selected", [(0, True), (1, False)])
@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_deferred_profiler_freezes_rank_settings_and_trace_name(
    monkeypatch, rank, selected, role
):
    backend = FakeTorchNPUProfiler()
    prefix = f"AFD_NPU_{role.upper()}_PROFILER"
    monkeypatch.setenv(f"{prefix}_ENABLE", "1")
    session = create_deferred_afd_npu_profiler(role, role_rank=rank)
    assert session.status()["pending"]
    assert not backend.created_profiler.started
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=backend))
    monkeypatch.setenv(f"{prefix}_RANKS", "all")
    session.before_step()
    assert session.status()["selected"] is selected
    assert session.status()["applied"] is selected
    assert backend.created_profiler.started is selected
    if selected:
        assert backend.worker_name == f"{role}_rank{rank}"
    session.finish()


def test_management_request_remains_pending_until_execution_owner_applies_it():
    case = make_profiler_session_case()
    with ThreadPoolExecutor(max_workers=1) as owner:
        owner.submit(case.session.before_step).result(PROFILER_TEST_TIMEOUT_SECONDS)
        case.session.request(False)
        status = case.session.status()
        assert status["requested"] is False and status["applied"] is True
        assert status["pending"]
        owner.submit(case.session.before_step).result(PROFILER_TEST_TIMEOUT_SECONDS)
        assert not case.session.status()["applied"]
        assert not case.session.status()["pending"]
        case.session.request(True, "round2")
        owner.submit(case.session.step).result(PROFILER_TEST_TIMEOUT_SECONDS)
        owner.submit(case.session.finish).result(PROFILER_TEST_TIMEOUT_SECONDS)
    assert [event[0] for event in case.events] == [
        "create",
        "start",
        "stop",
        "create",
        "start",
        "step",
        "stop",
    ]
    assert len({event[1] for event in case.events}) == 1
    assert case.events[3][2] == "round2"
    assert case.session.status()["owner_thread_id"] != threading.get_ident()
    assert case.session.status()["closed"]


def test_management_requests_do_not_wait_for_owner_trace_export():
    case = make_profiler_session_case()
    with ThreadPoolExecutor(max_workers=1) as owner:
        owner.submit(case.session.before_step).result(PROFILER_TEST_TIMEOUT_SECONDS)
        backend = case.profilers[0]
        backend.stop_release.clear()
        case.session.request(False)
        pending = owner.submit(case.session.before_step)
        assert backend.stop_entered.wait(PROFILER_TEST_TIMEOUT_SECONDS)
        try:
            case.session.request(True, "next")
            assert case.session.status()["pending"]
        finally:
            backend.stop_release.set()
        pending.result(PROFILER_TEST_TIMEOUT_SECONDS)
        owner.submit(case.session.step).result(PROFILER_TEST_TIMEOUT_SECONDS)
        owner.submit(case.session.finish).result(PROFILER_TEST_TIMEOUT_SECONDS)
    assert case.session.status()["steps"] == 1


@pytest.mark.parametrize("operation", ["before_step", "step", "finish"])
def test_profiler_backend_rejects_a_foreign_execution_thread(operation):
    case = make_profiler_session_case()
    case.session.before_step()
    with ThreadPoolExecutor(max_workers=1) as foreign:
        result = foreign.submit(getattr(case.session, operation))
        with pytest.raises(RuntimeError, match="execution owner thread"):
            result.result(PROFILER_TEST_TIMEOUT_SECONDS)
    case.session.finish()
    assert [event[0] for event in case.events] == ["create", "start", "stop"]


@pytest.mark.parametrize("failure", ["start", "step", "stop"])
def test_profiler_failure_remains_visible_and_cannot_restart(failure):
    case = make_profiler_session_case(failure=failure)
    with (
        pytest.raises(ValueError, match=f"controlled profiler {failure}"),
        afd_npu_profiler_owner(case.session),
    ):
        case.session.step()
    status = case.session.status()
    assert status["failed"] and not status["closed"]
    assert status["error"] == f"controlled profiler {failure} failure"
    with pytest.raises(RuntimeError, match="session failed"):
        case.session.request(True)


def test_serving_failure_is_preserved_when_profiler_export_also_fails():
    case = make_profiler_session_case(failure="stop")
    with (
        pytest.raises(ValueError, match="original serving failure"),
        afd_npu_profiler_owner(case.session),
    ):
        case.session.before_step()
        raise ValueError("original serving failure")
    assert case.session.status()["error"] == "controlled profiler stop failure"


def test_exited_profiler_owner_cannot_accept_a_new_start():
    case = make_profiler_session_case()
    case.session.finish()
    case.session.stop()
    with pytest.raises(RuntimeError, match="owner has exited"):
        case.session.start()


def test_attention_execute_counts_success_without_closing_its_session():
    case = make_profiler_session_case()
    with afd_npu_profiler_execute(case.session):
        assert case.session.status()["applied"]
        assert case.session.status()["steps"] == 0
    assert case.session.status()["steps"] == 1
    assert not case.session.status()["closed"]
    case.session.finish()


@pytest.mark.parametrize("export_failure", [False, True])
def test_attention_execute_preserves_failure_and_never_counts_it(export_failure):
    case = make_profiler_session_case(failure="stop" if export_failure else None)
    with (
        pytest.raises(ValueError, match="original Attention execute"),
        afd_npu_profiler_execute(case.session),
    ):
        raise ValueError("original Attention execute")
    status = case.session.status()
    assert status["steps"] == 0
    assert status["closed"] is not export_failure
    assert status["failed"] is export_failure
