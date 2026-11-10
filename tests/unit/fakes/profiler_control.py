# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real local socket fixture around the fixed worker profiler RPC boundary."""

from __future__ import annotations

import errno
import json
import socket
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.profiler_control import FfnProfilerControl
from afd_plugin.config import HCCL_P2P_CONNECTOR
from tests.unit.fakes.profiler import make_profiler_session_case
from tests.unit.fakes.vllm_config import npu_vllm_config

PARTIAL_WRITE_BYTES = 7


class ProfilerControlExecutor:
    def __init__(self):
        self.profile = make_profiler_session_case()
        self.requests = []
        self.failure = None
        self.response = None

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.requests.append((method, timeout, args, kwargs))
        if self.failure is not None:
            raise self.failure
        if self.response is not None:
            return self.response
        if method == "profile":
            self.profile.session.request(**kwargs)
            return [None]
        if method == "afd_profiler_status":
            return [
                {
                    "mode": "afd-role",
                    "step_unit": "parent",
                    **self.profile.session.status(),
                }
            ]
        raise ValueError("unexpected control method")


class PartialWriteSocket(socket.socket):
    """Real Unix socket with a deliberately short public send operation."""

    def accept(self):
        connection, address = super().accept()
        return type(self)(
            self.family, self.type, self.proto, fileno=connection.detach()
        ), address

    def send(self, data, flags=0):
        return super().send(data[:PARTIAL_WRITE_BYTES], flags)


def make_control_config():
    config = npu_vllm_config(role="ffn", connector=HCCL_P2P_CONNECTOR)
    config.device_config = SimpleNamespace(device_type="npu")
    config.profiler_config = None
    return config


def make_control_socket_path(tmp_path_factory):
    if not hasattr(socket, "AF_UNIX"):
        pytest.skip("real Unix sockets unavailable on this CPU host")
    path = tmp_path_factory.mktemp("pc") / "p.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(str(path))
        except OSError as error:
            if error.errno in (errno.EAFNOSUPPORT, errno.EPROTONOSUPPORT):
                pytest.skip("real Unix sockets unsupported by this CPU kernel")
            raise
    path.unlink()
    return path


@contextmanager
def profiler_control_case(path):
    executor = ProfilerControlExecutor()
    clock = SimpleNamespace(now=0.0)
    server = FfnProfilerControl(path, executor, clock=lambda: clock.now)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(1)
    client.connect(str(path))
    try:
        yield SimpleNamespace(
            server=server, client=client, executor=executor, clock=clock, path=path
        )
    finally:
        client.close()
        server.close()


def control_exchange(case, request):
    payload = (
        request if isinstance(request, bytes) else (json.dumps(request) + "\n").encode()
    )
    case.client.sendall(payload)
    case.server.poll()
    return json.loads(case.client.recv(65_536))
