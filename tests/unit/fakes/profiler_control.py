# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real local socket fixture around the fixed worker profiler RPC boundary."""

from __future__ import annotations

import errno
import json
import socket
from collections import deque
from contextlib import contextmanager
from queue import Empty, Queue
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


class NativeRpcQueueBoundary:
    """Deterministic pending wire replies; no delay or device is simulated."""

    def __init__(self):
        self.responses = Queue()
        self.requests = deque()
        self.transmitted = []

    def enqueue(self, value):
        self.responses.put_nowait(value)

    def dequeue(self, timeout=None):
        try:
            return self.responses.get_nowait()
        except Empty as error:
            raise TimeoutError("controlled native RPC reply remains pending") from error

    def transmit(self, request):
        self.transmitted.append(request)
        self.requests.append(request)


def make_native_profiler_rpc_case():
    """Retain pinned executor futures, WorkerProc dispatch and AFD worker RPC.

    Lazy imports keep the stdlib control suite collectable without torch/vLLM.
    Constructors, IPC startup, serving owner and device work are excluded. The
    explicit queue leaf withholds the first response until the caller releases
    its actual WorkerProc dispatch, after native collective_rpc has timed out.
    """
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc

    from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker

    profile = make_profiler_session_case()
    worker = AFDNPUFFNWorker.__new__(AFDNPUFFNWorker)
    worker.model_runner = SimpleNamespace(prof=profile.session)
    worker.profiler_config = None
    worker._ffn_loop_error = None
    queues = NativeRpcQueueBoundary()
    worker_process = WorkerProc.__new__(WorkerProc)
    worker_process.worker = worker
    worker_process.rank = 0
    worker_process.use_async_scheduling = False
    worker_process.worker_response_mq = queues
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.failure_callback = None
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=queues.transmit)
    executor.response_mqs = [queues]
    executor.futures_queue = deque()
    return SimpleNamespace(
        executor=executor,
        queues=queues,
        profile=profile,
        dispatch_pending=lambda: worker_process._execute_worker_rpc(
            queues.requests.popleft()
        ),
        success=WorkerProc.ResponseStatus.SUCCESS,
    )


def make_native_profiler_engine_case():
    """Keep native executor dispatch and the core fatal request handler real.

    Only construction, queues and the registered fatal-notification callback are
    host boundaries. No worker process or EngineCore serving loop is started.
    """
    from vllm.v1.engine.core import EngineCoreProc, EngineCoreRequestType

    case = make_native_profiler_rpc_case()
    engine = EngineCoreProc.__new__(EngineCoreProc)
    engine.vllm_config = make_control_config()
    engine.model_executor = case.executor
    engine.input_queue = Queue()
    case.executor.failure_callback = lambda: engine.input_queue.put_nowait(
        (EngineCoreRequestType.EXECUTOR_FAILED, b"")
    )
    case.executor.output_rank = 0
    case.executor.kv_output_aggregator = None
    case.executor.ec_output_aggregator = None
    case.engine = engine
    return case


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
def profiler_control_case(path, *, executor=None):
    if executor is None:
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
