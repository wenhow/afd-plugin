# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host RPC queue boundaries; no worker startup, device or model is simulated."""

from collections import deque
from queue import Queue
from types import SimpleNamespace


class ProfilerRpcQueueBoundary:
    """Expose failure gating and an unconsumed response as observable seams.

    The runtime-gated case uses the real MultiprocExecutor instead. This small
    boundary keeps configuration and callback tests collectable without vLLM.
    """

    def __init__(self, failure):
        self.is_failed = False
        self.failure_callback = None
        self.failure = failure
        self.responses = deque()
        self.transmitted = []
        self.shutdown_complete = False

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        if self.is_failed:
            raise RuntimeError("Executor failed.")
        self.transmitted.append((method, timeout, args, kwargs))
        if self.failure is not None:
            failure, self.failure = self.failure, None
            raise failure
        return self.responses.popleft()

    def execute_model(self, scheduler_output):
        return self.collective_rpc("execute_model", args=(scheduler_output,))

    def shutdown(self):
        self.shutdown_complete = True


class ProfilerRpcExecutor(ProfilerRpcQueueBoundary):
    """The explicit Multiproc class boundary used by the CPU-only core suite."""


class NonMultiprocProfilerExecutor(ProfilerRpcQueueBoundary):
    """Separate runtime class: queue behavior alone must not qualify a backend."""


class MultiprocProfilerSubclass(ProfilerRpcExecutor):
    """A backend subclass keeps the real Multiproc failure contract."""


def make_profiler_rpc_engine_case(core_type, config, failure_request, *, failure):
    engine = core_type.__new__(core_type)
    engine.vllm_config = config
    engine._afd_ffn_rpc_failed = False
    engine._afd_hccl_peer_stop_complete = False
    engine.input_queue = Queue()
    executor = ProfilerRpcExecutor(failure)
    engine.model_executor = executor
    executor.failure_callback = lambda: engine.input_queue.put_nowait(
        (failure_request, b"")
    )
    return SimpleNamespace(engine=engine, executor=executor)
