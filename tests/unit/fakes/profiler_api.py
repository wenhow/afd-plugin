# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Existing-engine RPC boundary for HTTP tests, with real CPU profiler state."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import httpx
from fastapi import FastAPI

from afd_plugin.compat.npu.profiler_api import AFDNPUProfilerEndpointPlugin
from tests.unit.fakes.profiler import make_profiler_session_case
from tests.unit.fakes.vllm_config import npu_vllm_config


class ProfilerEngineClient:
    """An API client double; no native worker, device or timeline is simulated."""

    def __init__(self, config, sessions):
        self.vllm_config = config
        self.sessions = sessions
        self.requests = []
        self.failure = None

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.requests.append((method, timeout, args, kwargs))
        if self.failure is not None:
            raise self.failure
        if method == "profile":
            for session in self.sessions:
                session.request(**kwargs)
            return [None for session in self.sessions]
        if method == "afd_profiler_status":
            step_unit = (
                "execute"
                if self.vllm_config.additional_config["afd"]["role"] == "attention"
                else "parent"
            )
            return [
                {"mode": "afd-role", "step_unit": step_unit, **session.status()}
                for session in self.sessions
            ]
        raise ValueError("unexpected profiler RPC method")


class ProfilerEndpointEntryPoint:
    name = "afd_npu_profiler"
    value = "afd_plugin.compat.npu.profiler_api:AFDNPUProfilerEndpointPlugin"

    def load(self):
        return AFDNPUProfilerEndpointPlugin


def make_profiler_api_case(
    *, role="attention", initialize=True, selected=(True, False)
):
    config = npu_vllm_config(role=role)
    config.device_config = SimpleNamespace(device_type="npu")
    config.profiler_config = None
    profiles = tuple(
        make_profiler_session_case(selected=selection) for selection in selected
    )
    engine = ProfilerEngineClient(config, tuple(item.session for item in profiles))
    app = FastAPI()
    app.state.engine_client, app.state.vllm_config = engine, config
    plugin, args = AFDNPUProfilerEndpointPlugin(), Namespace()
    plugin.attach_router(app)
    if initialize:
        asyncio.run(plugin.init_state(engine, app.state, args))
    return SimpleNamespace(
        app=app,
        engine=engine,
        plugin=plugin,
        config=config,
        args=args,
        profiles=profiles,
    )


def profiler_http_request(app, method, *, json=None):
    async def send():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.request(method, "/afd/profiler", json=json)

    return asyncio.run(send())
