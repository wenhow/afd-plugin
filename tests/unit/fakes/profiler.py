# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import threading
from types import SimpleNamespace

from afd_plugin.compat.npu.profiler import AFDNPUProfilerSession

PROFILER_TEST_TIMEOUT_SECONDS = 5


class StepProfiler:
    """Profiler double recording the start/step/stop calls the runners make."""

    def __init__(self):
        self.steps = 0
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def step(self):
        self.steps += 1


class FakeTorchNPUProfiler:
    class ExportType:
        Text = "text"

    class ProfilerLevel:
        Level2 = "level2"

    class AiCMetrics:
        AiCoreNone = "aicore_none"

    class ProfilerActivity:
        CPU = "cpu"
        NPU = "npu"

    def __init__(self):
        self.created_profiler = StepProfiler()
        self.schedule_kwargs = {}
        self.profile_kwargs = {}
        self.trace_dir = self.worker_name = None
        self._ExperimentalConfig = self._experimental_config

    def _experimental_config(self, **kwargs):
        return kwargs

    def schedule(self, **kwargs):
        self.schedule_kwargs = kwargs
        return kwargs

    def tensorboard_trace_handler(self, trace_dir, *, worker_name=None):
        self.trace_dir, self.worker_name = trace_dir, worker_name
        return ("handler", trace_dir, worker_name)

    def profile(self, **kwargs):
        self.profile_kwargs = kwargs
        return self.created_profiler


class RecordingProfiler:
    """Public profiler API observations, with no device/timeline simulation."""

    def __init__(self, events, failure=None):
        self.events, self.failure = events, failure
        self.stop_entered = threading.Event()
        self.stop_release = threading.Event()
        self.stop_release.set()

    def _record(self, action):
        self.events.append((action, threading.get_ident()))
        if self.failure == action:
            raise ValueError(f"controlled profiler {action} failure")

    def start(self):
        self._record("start")

    def step(self):
        self._record("step")

    def stop(self):
        self.stop_entered.set()
        if not self.stop_release.wait(PROFILER_TEST_TIMEOUT_SECONDS):
            raise TimeoutError("test did not release profiler export")
        self._record("stop")


def make_profiler_session_case(*, selected=True, failure=None):
    events, profilers = [], []

    def create(prefix):
        events.append(("create", threading.get_ident(), prefix))
        profiler = RecordingProfiler(events, failure)
        profilers.append(profiler)
        return profiler

    return SimpleNamespace(
        session=AFDNPUProfilerSession(create, selected=selected),
        events=events,
        profilers=profilers,
    )


class ProfiledReceiveGate:
    """Hold actual CPU-protocol receives, without simulating device progress."""

    def __init__(self, receive):
        self.receive = receive
        self.entered = (threading.Event(), threading.Event())
        self.release = (threading.Event(), threading.Event())
        self.index = 0

    def __call__(self):
        index = self.index
        self.index += 1
        self.entered[index].set()
        if not self.release[index].wait(PROFILER_TEST_TIMEOUT_SECONDS):
            raise TimeoutError("test did not release parent receive")
        return self.receive()


def make_profiled_worker_case(worker_module, runner_module, monkeypatch):
    # Device-dependent fake concepts stay lazy so profiler-only tests collect
    # on a host without torch/vLLM.
    from afd_plugin.distributed.hccl import HcclStageLayout
    from afd_plugin.distributed.hccl_plan import HcclParentPlan
    from tests.unit.fakes.hccl import (
        HcclCpuDeviceRuntime,
        make_staged_ffn_service_case,
    )

    plan = HcclParentPlan("profile-parent", (0,), (HcclStageLayout(0, (0,), (0,), 1),))
    case = make_staged_ffn_service_case(
        runner_module.AFDNPUFFNModelRunner, worker_module.AFDNPUFFNWorker, [plan]
    )
    case.profile = make_profiler_session_case()
    case.runner.prof = case.profile.session
    case.gate = ProfiledReceiveGate(case.peer.receive_forward)
    case.peer.receive_forward = case.gate
    case.worker.bind_hccl_v41_transport(case.peer)
    monkeypatch.setattr(
        worker_module,
        "torch",
        SimpleNamespace(
            npu=HcclCpuDeviceRuntime(), bfloat16=runner_module.torch.bfloat16
        ),
    )
    return case


def make_profiled_shutdown_case(worker_module, runner_module, monkeypatch):
    case = make_profiled_worker_case(worker_module, runner_module, monkeypatch)
    case.profile = make_profiler_session_case(failure="stop")
    case.runner.prof = case.profile.session
    case.runner.connector = SimpleNamespace(
        is_initialized=True, control_plane=None, close=lambda: None
    )
    case.worker._hccl_v41_transport = None

    def shutdown_loop():
        case.profile.session.before_step()
        case.worker._ffn_shutdown_event.set()

    case.worker._run_ffn_server_loop = shutdown_loop
    return case


def make_profiled_restart_case(worker_module, runner_module, monkeypatch):
    case = make_profiled_worker_case(worker_module, runner_module, monkeypatch)
    case.worker._hccl_v41_transport = None
    case.runner.connector = SimpleNamespace(is_initialized=False, control_plane=None)

    def initialize_connector():
        case.runner.connector.is_initialized = True

    case.runner.initialize_afd_connector = initialize_connector
    return case


def make_profiled_attention_case(worker_class, *, failure=None):
    # Keep native model/transport builders lazy for the stdlib profiler suite.
    from tests.unit.fakes.attention_model_runner_v2 import make_graph_runner_case

    case = make_graph_runner_case()
    case.profile = make_profiler_session_case(failure=failure)
    case.events = case.profile.events
    case.owner.prof = case.profile.session
    case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    case.worker = object.__new__(worker_class)
    case.worker.model_runner = case.owner
    case.worker.profiler_config = None
    case.worker.profiler = None
    return case
