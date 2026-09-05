from __future__ import annotations

import sys
from functools import wraps
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.profiler import (
    _fail_if_msmonitor_is_enabled,
    afd_npu_profiler_config,
    create_afd_npu_profiler,
    step_afd_npu_profiler,
    stop_afd_npu_profiler,
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
    "AFD_NPU_ATTENTION_PROFILER_RANKS",
    "AFD_NPU_FFN_PROFILER_ENABLE",
    "AFD_NPU_FFN_PROFILER_WAIT",
    "AFD_NPU_FFN_PROFILER_WARMUP",
    "AFD_NPU_FFN_PROFILER_ACTIVE",
    "AFD_NPU_FFN_PROFILER_REPEAT",
    "AFD_NPU_FFN_PROFILER_SKIP_FIRST",
    "AFD_NPU_FFN_PROFILER_DIR",
    "AFD_NPU_FFN_PROFILER_WITH_STACK",
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
    assert attention.trace_dir == "/tmp/profile/attn"
    assert attention.with_stack is False
    assert attention.ranks == frozenset({0})
    assert ffn.enabled is False
    assert ffn.trace_dir == "/tmp/profile/ffn"
    assert ffn.with_stack is False
    assert ffn.ranks == frozenset({0})


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


def test_create_npu_profiler_starts_immediate_manual_window(monkeypatch):
    profiler_module = _FakeTorchNPUProfiler()
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(profiler=profiler_module, npu=profiler_module.npu),
    )
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "true")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_DIR", "/tmp/afd-ffn")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_WITH_STACK", "true")

    profiler = create_afd_npu_profiler("ffn")

    assert profiler is profiler_module.created_profiler
    assert profiler.started is True
    assert profiler_module.schedule_kwargs is None
    assert "schedule" not in profiler_module.profile_kwargs
    assert profiler_module.profile_kwargs["record_shapes"] is False
    assert profiler_module.profile_kwargs["profile_memory"] is False
    assert profiler_module.profile_kwargs["with_stack"] is True
    assert profiler_module.profile_kwargs["with_modules"] is True
    assert profiler_module.experimental_config_kwargs == {
        "export_type": "text",
        "profiler_level": "level1",
        "msprof_tx": False,
        "aic_metrics": "pipe_utilization",
        "l2_cache": False,
        "op_attr": False,
        "data_simplification": True,
        "record_op_args": False,
        "gc_detect_threshold": None,
    }
    assert profiler_module.trace_dir == "/tmp/afd-ffn"
    assert profiler_module.trace_handler_kwargs == {"analyse_flag": False}
    assert profiler_module.npu.synchronize_calls == 1


def test_create_npu_profiler_propagates_wrapped_start_error(monkeypatch):
    created_profiler = _ExceptionSuppressingProfiler(fail_action="start")
    profiler_module = _FakeTorchNPUProfiler(created_profiler)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(profiler=profiler_module, npu=profiler_module.npu),
    )
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "true")

    with pytest.raises(RuntimeError, match="low-level start failed"):
        create_afd_npu_profiler("ffn")

    assert created_profiler.stopped is True


def test_create_npu_profiler_cleans_up_invalid_capture_root(monkeypatch):
    created_profiler = _StepProfiler()
    created_profiler.prof_if = SimpleNamespace(prof_path="/missing/afd-profile")
    profiler_module = _FakeTorchNPUProfiler(created_profiler)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(profiler=profiler_module, npu=profiler_module.npu),
    )
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "true")

    with pytest.raises(RuntimeError, match="capture root"):
        create_afd_npu_profiler("ffn")

    assert created_profiler.started is True
    assert created_profiler.stopped is True


def test_create_npu_profiler_skips_nonzero_role_rank(monkeypatch):
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "true")

    assert create_afd_npu_profiler("ffn", role_rank=1) is None


@pytest.mark.parametrize("configured_ranks", ["1,3", "all"])
def test_create_npu_profiler_supports_configured_role_ranks(
    monkeypatch,
    configured_ranks,
):
    profiler_module = _FakeTorchNPUProfiler()
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(profiler=profiler_module, npu=profiler_module.npu),
    )
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "true")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_RANKS", configured_ranks)

    profiler = create_afd_npu_profiler("ffn", role_rank=1)

    assert profiler is profiler_module.created_profiler
    assert profiler.started is True


@pytest.mark.parametrize("configured_ranks", ["", "1,bad", "-1"])
def test_npu_profiler_rejects_invalid_role_ranks(monkeypatch, configured_ranks):
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_RANKS", configured_ranks)

    with pytest.raises(ValueError, match="AFD_NPU_FFN_PROFILER_RANKS"):
        afd_npu_profiler_config("ffn")


def test_step_npu_profiler_ignores_disabled_profiler():
    step_afd_npu_profiler(None)

    profiler = _StepProfiler()
    step_afd_npu_profiler(profiler)

    assert profiler.steps == 1


def test_stop_npu_profiler_ignores_disabled_profiler(monkeypatch):
    stop_afd_npu_profiler(None)

    profiler_module = _FakeTorchNPUProfiler()
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(profiler=profiler_module, npu=profiler_module.npu),
    )
    profiler = _StepProfiler()
    stop_afd_npu_profiler(profiler)

    assert profiler.stopped is True
    assert profiler_module.npu.synchronize_calls == 1


def test_stop_npu_profiler_propagates_wrapped_stop_error(monkeypatch):
    profiler_module = _FakeTorchNPUProfiler()
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(profiler=profiler_module, npu=profiler_module.npu),
    )
    profiler = _ExceptionSuppressingProfiler(fail_action="stop")

    with pytest.raises(RuntimeError, match="low-level stop failed"):
        stop_afd_npu_profiler(profiler)

    assert profiler_module.npu.synchronize_calls == 1


def test_profiler_rejects_msmonitor_enabled_only_by_environment(monkeypatch):
    ascend_config = SimpleNamespace(
        get_ascend_config=lambda: SimpleNamespace(msmonitor_use_daemon=False),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_config",
        ascend_config,
    )
    monkeypatch.setenv("MSMONITOR_USE_DAEMON", "1")

    with pytest.raises(RuntimeError, match="MSMONITOR_USE_DAEMON"):
        _fail_if_msmonitor_is_enabled()


class _StepProfiler:
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


class _ExceptionSuppressingProfiler(_StepProfiler):
    def __init__(self, *, fail_action):
        super().__init__()
        self.fail_action = fail_action

    def _start(self):
        if self.fail_action == "start":
            raise RuntimeError("low-level start failed")
        super().start()

    @wraps(_start)
    def start(self):
        try:
            self._start()
        except RuntimeError:
            return

    def _stop(self):
        if self.fail_action == "stop":
            raise RuntimeError("low-level stop failed")
        super().stop()

    @wraps(_stop)
    def stop(self):
        try:
            self._stop()
        except RuntimeError:
            return


class _FakeNPU:
    def __init__(self):
        self.synchronize_calls = 0

    def current_device(self):
        return 0

    def synchronize(self):
        self.synchronize_calls += 1


class _FakeTorchNPUProfiler:
    class ExportType:
        Text = "text"

    class ProfilerLevel:
        Level1 = "level1"

    class AiCMetrics:
        PipeUtilization = "pipe_utilization"

    class ProfilerActivity:
        CPU = "cpu"
        NPU = "npu"

    def __init__(self, created_profiler=None):
        self.created_profiler = created_profiler or _StepProfiler()
        self.npu = _FakeNPU()
        self.schedule_kwargs = None
        self.profile_kwargs = None
        self.trace_dir = None
        self.trace_handler_kwargs = None
        self.experimental_config_kwargs = None
        self._ExperimentalConfig = self._experimental_config

    def _experimental_config(self, **kwargs):
        self.experimental_config_kwargs = kwargs
        return kwargs

    def schedule(self, **kwargs):
        self.schedule_kwargs = kwargs
        return kwargs

    def tensorboard_trace_handler(self, trace_dir, **kwargs):
        self.trace_dir = trace_dir
        self.trace_handler_kwargs = kwargs
        return ("handler", trace_dir)

    def profile(self, **kwargs):
        self.profile_kwargs = kwargs
        return self.created_profiler
