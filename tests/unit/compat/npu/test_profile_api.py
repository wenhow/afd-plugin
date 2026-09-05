from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.profile_api import afd_profile_control_middleware


class _Response:
    def __init__(self, *, status_code: int):
        self.status_code = status_code


@pytest.fixture(autouse=True)
def _fake_starlette(monkeypatch):
    package = types.ModuleType("starlette")
    package.__path__ = []
    responses = types.ModuleType("starlette.responses")
    responses.Response = _Response
    monkeypatch.setitem(sys.modules, "starlette", package)
    monkeypatch.setitem(sys.modules, "starlette.responses", responses)


def _request(*, path: str, host: str = "127.0.0.1"):
    start_calls: list[bool] = []
    stop_calls: list[bool] = []

    async def start_profile() -> None:
        start_calls.append(True)

    async def stop_profile() -> None:
        stop_calls.append(True)

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path=path),
        client=SimpleNamespace(host=host),
        app=SimpleNamespace(
            state=SimpleNamespace(
                engine_client=SimpleNamespace(
                    start_profile=start_profile,
                    stop_profile=stop_profile,
                ),
            ),
        ),
    )
    return request, start_calls, stop_calls


def test_profile_middleware_starts_profiler_from_loopback():
    request, start_calls, stop_calls = _request(path="/afd/profile/start")

    async def unused_call_next(_request):
        raise AssertionError("plugin endpoint must not fall through")

    response = asyncio.run(
        afd_profile_control_middleware(request, unused_call_next),
    )

    assert response.status_code == 200
    assert start_calls == [True]
    assert stop_calls == []


def test_profile_finalize_middleware_stops_profiler_from_loopback():
    request, start_calls, stop_calls = _request(path="/afd/profile/finalize")

    async def unused_call_next(_request):
        raise AssertionError("plugin endpoint must not fall through")

    response = asyncio.run(
        afd_profile_control_middleware(request, unused_call_next),
    )

    assert response.status_code == 200
    assert start_calls == []
    assert stop_calls == [True]


def test_profile_middleware_stops_profiler_from_loopback():
    request, start_calls, stop_calls = _request(path="/afd/profile/stop")

    async def unused_call_next(_request):
        raise AssertionError("plugin endpoint must not fall through")

    response = asyncio.run(
        afd_profile_control_middleware(request, unused_call_next),
    )

    assert response.status_code == 200
    assert start_calls == []
    assert stop_calls == [True]


def test_profile_finalize_middleware_rejects_remote_client():
    request, start_calls, stop_calls = _request(
        path="/afd/profile/finalize",
        host="192.0.2.10",
    )

    async def unused_call_next(_request):
        raise AssertionError("plugin endpoint must not fall through")

    response = asyncio.run(
        afd_profile_control_middleware(request, unused_call_next),
    )

    assert response.status_code == 403
    assert start_calls == []
    assert stop_calls == []


def test_profile_finalize_middleware_preserves_other_routes():
    request, start_calls, stop_calls = _request(path="/health")
    expected = _Response(status_code=204)

    async def call_next(_request):
        return expected

    response = asyncio.run(afd_profile_control_middleware(request, call_next))

    assert response is expected
    assert start_calls == []
    assert stop_calls == []
