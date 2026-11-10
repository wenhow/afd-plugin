# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import asyncio
import importlib.metadata
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI

from afd_plugin.compat.npu.profiler_api import PROFILER_RPC_TIMEOUT_SECONDS
from tests.unit.fakes.profiler_api import (
    ProfilerEndpointEntryPoint,
    make_profiler_api_case,
    profiler_http_request,
)


@pytest.fixture(autouse=True)
def _enable_role_profilers(monkeypatch):
    monkeypatch.setenv("AFD_NPU_ATTENTION_PROFILER_ENABLE", "1")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "1")


def test_application_without_the_endpoint_retains_its_http_surface():
    assert profiler_http_request(FastAPI(), "GET").status_code == 404


@pytest.mark.parametrize(
    "role,step_unit", [("attention", "execute"), ("ffn", "parent")]
)
def test_status_returns_each_actual_rpc_worker_snapshot(role, step_unit):
    case = make_profiler_api_case(role=role)
    case.profiles[0].session.before_step()
    response = profiler_http_request(case.app, "GET")
    expected = [
        {"mode": "afd-role", "step_unit": step_unit, **item.session.status()}
        for item in case.profiles
    ]
    assert response.status_code == 200
    assert response.json() == {"role": role, "workers": expected}


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_request_is_accepted_before_execution_owner_applies_it(role):
    case = make_profiler_api_case(role=role)
    case.profiles[0].session.before_step()
    response = profiler_http_request(case.app, "POST", json={"is_start": False})
    assert response.status_code == 200
    assert response.json() == {"accepted": True, "role": role}
    pending = profiler_http_request(case.app, "GET").json()["workers"][0]
    assert (pending["requested"], pending["applied"], pending["pending"]) == (
        False,
        True,
        True,
    )
    case.profiles[0].session.before_step()
    applied = profiler_http_request(case.app, "GET").json()["workers"][0]
    assert (applied["requested"], applied["applied"], applied["pending"]) == (
        False,
        False,
        False,
    )


def test_restart_prefix_reaches_the_existing_session_without_creating_a_backend():
    case = make_profiler_api_case()
    profiler_http_request(case.app, "POST", json={"is_start": False})
    response = profiler_http_request(
        case.app, "POST", json={"is_start": True, "profile_prefix": "sample2"}
    )
    assert response.json()["accepted"] is True
    status = profiler_http_request(case.app, "GET").json()["workers"][0]
    assert status["profile_prefix"] == "sample2"
    assert status["pending"] is True
    assert case.profiles[0].events == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"is_start": "false"},
        {"is_start": 0},
        {"is_start": None},
        {"is_start": True, "profile_prefix": 123},
        {"is_start": True, "method": "shutdown"},
    ],
)
def test_invalid_http_control_does_not_send_an_engine_rpc(body):
    case = make_profiler_api_case()
    response = profiler_http_request(case.app, "POST", json=body)
    assert response.status_code == 422
    assert case.engine.requests == []


@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize(
    "failure,status", [(TimeoutError("late"), 504), (RuntimeError("failed"), 502)]
)
def test_rpc_failure_is_an_http_error_without_false_acceptance(method, failure, status):
    case = make_profiler_api_case()
    case.engine.failure = failure
    response = profiler_http_request(case.app, method, json={"is_start": False})
    assert response.status_code == status
    assert "accepted" not in response.json()


def test_each_app_controls_only_its_own_existing_engine():
    attention, ffn = make_profiler_api_case(), make_profiler_api_case(role="ffn")
    # The native loader normally creates a new plugin for every app. Reusing
    # the same protocol object must also preserve app-local engine ownership.
    asyncio.run(attention.plugin.init_state(ffn.engine, ffn.app.state, ffn.args))
    profiler_http_request(attention.app, "POST", json={"is_start": False})
    assert profiler_http_request(attention.app, "GET").json()["role"] == "attention"
    assert (
        profiler_http_request(ffn.app, "GET").json()["workers"][0]["requested"] is True
    )


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_public_engine_rpc_has_a_bounded_timeout_and_fixed_method(method):
    case = make_profiler_api_case()
    response = profiler_http_request(case.app, method, json={"is_start": False})
    assert response.status_code == 200
    expected_method = "afd_profiler_status" if method == "GET" else "profile"
    assert case.engine.requests == [
        (
            expected_method,
            PROFILER_RPC_TIMEOUT_SECONDS,
            (),
            None if method == "GET" else {"is_start": False, "profile_prefix": None},
        )
    ]


@pytest.mark.parametrize(
    "boundary,error",
    [
        ("no-engine", "existing engine client"),
        ("foreign-engine", "this app's engine client"),
        ("foreign-config", "this app's engine client"),
        ("cpu", "NPU AFD"),
        ("not-afd", "AFD config requires"),
        ("disabled", "environment to be enabled"),
        ("native-profiler", "cannot be enabled together"),
    ],
)
def test_endpoint_initialization_rejects_incompatible_runtime(
    boundary, error, monkeypatch
):
    case = make_profiler_api_case(initialize=False)
    engine = case.engine
    if boundary == "no-engine":
        engine = None
    elif boundary == "foreign-engine":
        case.app.state.engine_client = make_profiler_api_case().engine
    elif boundary == "foreign-config":
        case.app.state.vllm_config = make_profiler_api_case().config
    elif boundary == "cpu":
        case.config.device_config.device_type = "cpu"
    elif boundary == "not-afd":
        case.config.additional_config = {}
    elif boundary == "disabled":
        monkeypatch.setenv("AFD_NPU_ATTENTION_PROFILER_ENABLE", "0")
    elif boundary == "native-profiler":
        case.config.profiler_config = SimpleNamespace(profiler="torch")
    with pytest.raises(ValueError, match=error):
        asyncio.run(case.plugin.init_state(engine, case.app.state, case.args))


@pytest.mark.parametrize(
    "allowlist,tasks,loaded",
    [
        (None, ("generate",), False),
        ([], ("generate",), False),
        (["afd"], ("generate",), False),
        (["afd", "afd_npu_profiler"], ("generate",), True),
        (["afd_npu_profiler"], ("render",), False),
    ],
)
def test_native_endpoint_loader_requires_explicit_allowlist_and_generate_task(
    allowlist, tasks, loaded, monkeypatch
):
    pytest.importorskip("torch")
    plugins = pytest.importorskip("vllm.plugins")
    envs = pytest.importorskip("vllm.envs")
    monkeypatch.setattr(envs, "VLLM_PLUGINS", allowlist)
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda **kwargs: [ProfilerEndpointEntryPoint()],
    )
    result = plugins.load_endpoint_plugins(tasks)
    assert [item.name for item in result] == (["afd_npu_profiler"] if loaded else [])
