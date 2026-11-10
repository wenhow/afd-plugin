# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real Unix socket/JSON management seams; no device or worker is simulated."""

from __future__ import annotations

import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.profiler_control import (
    CLIENT_IDLE_SECONDS,
    CONTROL_PATH_ENV,
    MAX_CONTROL_CLIENTS,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    RPC_TIMEOUT_SECONDS,
    SOCKET_PERMISSIONS,
    FfnProfilerControl,
    create_ffn_profiler_control,
    request_ffn_profiler,
)
from tests.unit.fakes.profiler_control import (
    PartialWriteSocket,
    ProfilerControlExecutor,
    control_exchange,
    make_control_config,
    make_control_socket_path,
    profiler_control_case,
)


@pytest.fixture
def control_path(tmp_path_factory):
    return make_control_socket_path(tmp_path_factory)


def test_status_reports_actual_worker_session_and_bounded_fixed_rpc(control_path):
    with profiler_control_case(control_path) as case:
        response = control_exchange(case, {"operation": "status"})
        assert response == {
            "ok": True,
            "role": "ffn",
            "workers": [
                {
                    "mode": "afd-role",
                    "step_unit": "parent",
                    **case.executor.profile.session.status(),
                }
            ],
        }
        assert case.executor.requests == [
            ("afd_profiler_status", RPC_TIMEOUT_SECONDS, (), None)
        ]


@pytest.mark.parametrize("operation,requested", [("start", True), ("stop", False)])
def test_control_accepts_desired_state_before_owner_applies_it(
    control_path, operation, requested
):
    with profiler_control_case(control_path) as case:
        session = case.executor.profile.session
        session.request(not requested)
        session.before_step()
        response = control_exchange(case, {"operation": operation})
        assert response == {"ok": True, "role": "ffn", "accepted": True}
        assert session.status()["requested"] is requested
        assert session.status()["pending"] is True
        session.before_step()
        assert session.status()["pending"] is False


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json\n",
        b"[]\n",
        b'{"operation":"shutdown"}\n',
        b'{"operation":"status","profile_prefix":"x"}\n',
        b'{"operation":"start","profile_prefix":1}\n',
        b'{"operation":"stop","method":"shutdown"}\n',
        b'{"operation":"status"}\n{"operation":"stop"}\n',
        b"[" * 1_100 + b"0" + b"]" * 1_100 + b"\n",
    ],
)
def test_malformed_or_unrelated_commands_never_reach_executor(control_path, payload):
    with profiler_control_case(control_path) as case:
        assert control_exchange(case, payload) == {
            "ok": False,
            "error": "invalid_request",
        }
        assert case.executor.requests == []


def test_oversized_request_is_rejected_before_json_or_rpc(control_path):
    with profiler_control_case(control_path) as case:
        assert control_exchange(case, b"x" * (MAX_REQUEST_BYTES + 1)) == {
            "ok": False,
            "error": "request_too_large",
        }
        assert case.executor.requests == []


def test_partial_request_waits_for_newline_without_blocking_poll(control_path):
    with profiler_control_case(control_path) as case:
        case.client.sendall(b'{"operation":')
        case.server.poll()
        assert case.executor.requests == []
        assert control_exchange(case, b'"status"}\n')["ok"] is True


def test_partial_response_writes_preserve_the_complete_json_message(
    control_path, monkeypatch
):
    monkeypatch.setattr(socket, "socket", PartialWriteSocket)
    with profiler_control_case(control_path) as case:
        case.client.setblocking(False)
        case.client.sendall(b'{"operation":"status"}\n')
        reply = bytearray()
        for _ in range(MAX_RESPONSE_BYTES):
            case.server.poll()
            with suppress(BlockingIOError):
                reply.extend(case.client.recv(65_536))
            if b"\n" in reply:
                break
        assert json.loads(reply) == {
            "ok": True,
            "role": "ffn",
            "workers": [
                {
                    "mode": "afd-role",
                    "step_unit": "parent",
                    **case.executor.profile.session.status(),
                }
            ],
        }


@pytest.mark.parametrize(
    "failure,code",
    [(TimeoutError("pending"), "rpc_timeout"), (RuntimeError("failed"), "rpc_failed")],
)
def test_rpc_error_is_reported_without_false_acceptance(control_path, failure, code):
    with profiler_control_case(control_path) as case:
        case.executor.failure = failure
        assert control_exchange(case, {"operation": "start"}) == {
            "ok": False,
            "error": code,
        }


@pytest.mark.parametrize("response", [[{1, 2}], [float("nan")]])
def test_invalid_rpc_output_does_not_stop_the_management_service(
    control_path, response
):
    with profiler_control_case(control_path) as case:
        case.executor.response = response
        assert control_exchange(case, {"operation": "status"}) == {
            "ok": False,
            "error": "invalid_rpc_response",
        }
        case.executor.response = None
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as fresh:
            fresh.settimeout(1)
            fresh.connect(str(control_path))
            fresh.sendall(b'{"operation":"status"}\n')
            case.server.poll()
            assert json.loads(fresh.recv(65_536))["ok"] is True


def test_unfinished_client_expires_without_blocking_other_clients(control_path):
    with profiler_control_case(control_path) as case:
        case.client.sendall(b'{"operation":')
        case.server.poll()
        case.clock.now = CLIENT_IDLE_SECONDS
        case.server.poll()
        assert case.client.recv(1) == b""
        assert case.executor.requests == []


def test_client_limit_releases_capacity_after_expiry(control_path):
    with profiler_control_case(control_path) as case:
        clients = [case.client]
        try:
            for _ in range(MAX_CONTROL_CLIENTS - 1):
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.settimeout(1)
                client.connect(str(control_path))
                clients.append(client)
            case.server.poll()
            case.clock.now = CLIENT_IDLE_SECONDS
            case.server.poll()
            assert all(client.recv(1) == b"" for client in clients)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as fresh:
                fresh.settimeout(1)
                fresh.connect(str(control_path))
                fresh.sendall(b'{"operation":"status"}\n')
                case.server.poll()
                assert json.loads(fresh.recv(65_536))["ok"] is True
        finally:
            for client in clients[1:]:
                client.close()


def test_close_removes_only_its_own_socket_and_is_idempotent(control_path):
    with profiler_control_case(control_path) as case:
        if os.name != "nt":
            assert control_path.stat().st_mode & 0o777 == SOCKET_PERMISSIONS
        case.server.close()
        case.server.close()
        assert not control_path.exists()


def test_existing_endpoint_is_preserved_when_a_second_server_fails(control_path):
    with profiler_control_case(control_path) as case:
        with pytest.raises(OSError):
            FfnProfilerControl(control_path, ProfilerControlExecutor())
        assert control_exchange(case, {"operation": "status"})["ok"] is True


def test_close_preserves_a_replacement_path(control_path):
    with profiler_control_case(control_path) as case:
        control_path.unlink()
        control_path.write_text("replacement", encoding="utf-8")
        case.server.close()
        assert control_path.read_text(encoding="utf-8") == "replacement"


def test_disabled_control_has_no_filesystem_side_effects(tmp_path_factory):
    control_path = tmp_path_factory.mktemp("pc") / "unused.sock"
    assert (
        create_ffn_profiler_control(
            make_control_config(), ProfilerControlExecutor(), environment={}
        )
        is None
    )
    assert not control_path.exists()


@pytest.mark.parametrize(
    "boundary,error",
    [
        ("disabled", "role profiler enabled"),
        ("cpu", "NPU HCCL"),
        ("connector", "NPU HCCL"),
        ("role", "role"),
        ("native", "conflicts"),
        ("relative", "absolute"),
    ],
)
def test_explicit_control_requires_compatible_runtime_before_binding(
    tmp_path_factory, monkeypatch, boundary, error
):
    control_path = tmp_path_factory.mktemp("pc") / "unused.sock"
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "1")
    config = make_control_config()
    raw = str(control_path)
    if boundary == "disabled":
        monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "0")
    elif boundary == "cpu":
        config.device_config.device_type = "cpu"
    elif boundary == "connector":
        config.additional_config["afd"]["connector"] = "CAMP2pAFDConnector"
    elif boundary == "role":
        config.additional_config["afd"]["role"] = "attention"
    elif boundary == "native":
        config.profiler_config = SimpleNamespace(profiler="torch")
    elif boundary == "relative":
        raw = "relative.sock"
    with pytest.raises(ValueError, match=error):
        create_ffn_profiler_control(
            config,
            ProfilerControlExecutor(),
            environment={CONTROL_PATH_ENV: raw},
        )
    assert not control_path.exists()


def test_client_sends_one_fixed_command_and_returns_server_status(control_path):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(control_path))
        listener.listen(1)
        listener.settimeout(1)
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(
                request_ffn_profiler, control_path, "start", profile_prefix="sample2"
            )
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(1)
                assert json.loads(connection.recv(MAX_REQUEST_BYTES)) == {
                    "operation": "start",
                    "profile_prefix": "sample2",
                }
                connection.sendall(b'{"ok":true,"accepted":true}\n')
            assert result.result(timeout=1) == {"ok": True, "accepted": True}
