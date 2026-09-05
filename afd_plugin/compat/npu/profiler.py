# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Plugin-owned NPU profiler helpers for AFD Ascend runners."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Final, Literal, Protocol

logger = logging.getLogger(__name__)

AFDNPUProfilerRole = Literal["attention", "ffn"]

_ENV_PREFIX: Final[dict[AFDNPUProfilerRole, str]] = {
    "attention": "AFD_NPU_ATTENTION_PROFILER",
    "ffn": "AFD_NPU_FFN_PROFILER",
}
_DEFAULT_DIR: Final[dict[AFDNPUProfilerRole, str]] = {
    "attention": "/tmp/profile/attn",
    "ffn": "/tmp/profile/ffn",
}
_DEFAULT_PROFILE_RANKS: Final[frozenset[int]] = frozenset({0})
_VLLM_TORCH_PROFILER_DIR_ENV: Final[str] = "VLLM_TORCH_PROFILER_DIR"


@dataclass(frozen=True)
class AFDNPUProfilerConfig:
    enabled: bool
    trace_dir: str
    with_stack: bool
    ranks: frozenset[int] | None


class AFDNPUProfiler(Protocol):
    def start(self) -> None: ...

    def step(self) -> None: ...

    def stop(self) -> None: ...


def afd_npu_profiler_config(role: AFDNPUProfilerRole) -> AFDNPUProfilerConfig:
    """Read plugin-owned profiler settings for an AFD NPU runner role."""

    prefix = _ENV_PREFIX[role]
    return AFDNPUProfilerConfig(
        enabled=_env_bool(f"{prefix}_ENABLE", default=False),
        trace_dir=_env_dir(f"{prefix}_DIR", default=_DEFAULT_DIR[role]),
        with_stack=_env_bool(f"{prefix}_WITH_STACK", default=False),
        ranks=_env_ranks(
            f"{prefix}_RANKS",
            default=_DEFAULT_PROFILE_RANKS,
        ),
    )


def create_afd_npu_profiler(
    role: AFDNPUProfilerRole,
    *,
    role_rank: int = 0,
) -> AFDNPUProfiler | None:
    """Create and explicitly start a profiler for an enabled role-local rank.

    The caller invokes this only after service readiness. No step schedule is
    installed: ``start()`` enters RECORD immediately and ``stop()`` closes the
    exact manually controlled window.
    """

    config = afd_npu_profiler_config(role)
    if not config.enabled or (
        config.ranks is not None and int(role_rank) not in config.ranks
    ):
        return None

    import torch_npu

    _fail_if_msmonitor_is_enabled()
    os.makedirs(config.trace_dir, exist_ok=True)
    _synchronize_npu(torch_npu, role=role, boundary="start")

    # Match the vLLM-Ascend TorchNPUProfilerWrapper configuration that is used
    # by the proven External-DP /start_profile and /stop_profile path. Level1 is
    # sufficient for kernel, task, stream, and communication timelines, while
    # avoiding the very large framework payload produced by Level2 shape data.
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=True,
        record_op_args=False,
        gc_detect_threshold=None,
    )
    logger.warning(
        "AFD NPU %s profiler started manually for role rank %d. Traces will "
        "be saved to: %s; with_stack=%s; online_analysis=False; %s",
        role,
        role_rank,
        config.trace_dir,
        config.with_stack,
        _storage_diagnostics(config.trace_dir),
    )
    trace_handler_factory = torch_npu.profiler.tensorboard_trace_handler
    unwrapped_factory = getattr(trace_handler_factory, "__wrapped__", None)
    if unwrapped_factory is not None:
        trace_handler = unwrapped_factory(
            config.trace_dir,
            analyse_flag=False,
        )
    else:
        trace_handler = trace_handler_factory(
            config.trace_dir,
            analyse_flag=False,
        )
    if trace_handler is None:
        raise RuntimeError("torch_npu profiler trace handler initialization failed")

    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        with_stack=config.with_stack,
        with_modules=config.with_stack,
        record_shapes=False,
        profile_memory=False,
        experimental_config=experimental_config,
        on_trace_ready=trace_handler,
    )
    try:
        _invoke_profiler_lifecycle(profiler, "start")
        prof_if = getattr(profiler, "prof_if", None)
        if prof_if is not None:
            prof_path = getattr(prof_if, "prof_path", None)
            if not prof_path or not os.path.isdir(prof_path):
                raise RuntimeError(
                    "torch_npu profiler start returned without creating its CANN "
                    f"capture root: {prof_path!r}"
                )
            logger.warning(
                "AFD NPU %s CANN capture root is ready: %s; %s",
                role,
                prof_path,
                _profiler_state(profiler),
            )
    except Exception:
        try:
            _invoke_profiler_lifecycle(profiler, "stop")
        except Exception:
            logger.exception(
                "AFD NPU %s profiler cleanup failed after start failure",
                role,
            )
        raise
    return profiler


def step_afd_npu_profiler(profiler: AFDNPUProfiler | None) -> None:
    if profiler is not None:
        profiler.step()


def stop_afd_npu_profiler(profiler: AFDNPUProfiler | None) -> None:
    if profiler is not None:
        import torch_npu

        prof_path = _profiler_path(profiler)
        logger.warning(
            "AFD NPU profiler is finalizing raw capture: path=%s; %s; %s",
            prof_path,
            _profiler_state(profiler),
            _raw_capture_snapshot(prof_path),
        )
        # Graph U2 launches work on multiple NPU streams. The profile utility
        # RPC can run after the model step has returned while kernels are still
        # in flight, so establish a device-wide completion boundary before
        # disabling CANN collection.
        _synchronize_npu(torch_npu, role="active", boundary="stop")
        started_at = time.monotonic()
        _invoke_profiler_lifecycle(profiler, "stop")
        logger.warning(
            "AFD NPU profiler raw capture finalization returned in %.3fs: "
            "path=%s; %s; %s",
            time.monotonic() - started_at,
            prof_path,
            _profiler_state(profiler),
            _raw_capture_snapshot(prof_path),
        )


def _invoke_profiler_lifecycle(
    profiler: AFDNPUProfiler,
    action: Literal["start", "stop"],
) -> None:
    """Invoke torch_npu lifecycle code without its exception-swallowing wrapper."""

    method = getattr(profiler, action)
    unwrapped = getattr(method, "__wrapped__", None)
    if unwrapped is None:
        method()
        return
    unwrapped(profiler)


def _synchronize_npu(
    torch_npu: object,
    *,
    role: str,
    boundary: Literal["start", "stop"],
) -> None:
    npu = getattr(torch_npu, "npu", None)
    synchronize = getattr(npu, "synchronize", None)
    if not callable(synchronize):
        raise RuntimeError("torch_npu.npu.synchronize is unavailable")
    current_device = getattr(npu, "current_device", None)
    device = current_device() if callable(current_device) else "unknown"
    thread = threading.current_thread()
    started_at = time.monotonic()
    logger.warning(
        "AFD NPU %s profiler %s synchronization started: pid=%d, "
        "thread=%s/%d, device=%s",
        role,
        boundary,
        os.getpid(),
        thread.name,
        threading.get_ident(),
        device,
    )
    synchronize()
    logger.warning(
        "AFD NPU %s profiler %s synchronization completed in %.3fs",
        role,
        boundary,
        time.monotonic() - started_at,
    )


def _fail_if_msmonitor_is_enabled() -> None:
    enabled = _env_bool("MSMONITOR_USE_DAEMON", default=False)
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        enabled = enabled or bool(get_ascend_config().msmonitor_use_daemon)
    except (AttributeError, ImportError, RuntimeError):
        pass
    if enabled:
        raise RuntimeError(
            "MSMONITOR_USE_DAEMON and torch profiler cannot be enabled together"
        )


def _storage_diagnostics(path: str) -> str:
    try:
        stat = os.statvfs(path)
        free_bytes = stat.f_bavail * stat.f_frsize
    except OSError as exc:
        return f"profile_storage=unavailable({exc})"

    return (
        f"profile_storage_path={os.path.realpath(path)}, "
        f"free_gib={free_bytes / (1024**3):.2f}"
    )


def _profiler_path(profiler: AFDNPUProfiler) -> str | None:
    prof_if = getattr(profiler, "prof_if", None)
    prof_path = getattr(prof_if, "prof_path", None)
    return str(prof_path) if prof_path else None


def _profiler_state(profiler: AFDNPUProfiler) -> str:
    current_action = getattr(profiler, "current_action", None)
    action_name = getattr(current_action, "name", current_action)
    return (
        f"current_action={action_name}, "
        f"step_num={getattr(profiler, 'step_num', 'unknown')}, "
        f"stopped={getattr(profiler, 'stopped', 'unknown')}"
    )


def _raw_capture_snapshot(prof_path: str | None) -> str:
    if not prof_path or not os.path.isdir(prof_path):
        return "raw_root=missing"

    raw_roots = []
    try:
        raw_roots = sorted(
            entry.path
            for entry in os.scandir(prof_path)
            if entry.is_dir() and entry.name.startswith("PROF_")
        )
    except OSError as exc:
        return f"raw_root=unreadable({exc})"

    file_count = 0
    total_bytes = 0
    device_data_count = 0
    device_data_bytes = 0
    device_end_count = 0
    host_end_count = 0
    for raw_root in raw_roots:
        for dirpath, _, filenames in os.walk(raw_root):
            relative_dir = os.path.relpath(dirpath, raw_root)
            parts = relative_dir.split(os.sep)
            in_device_data = (
                len(parts) >= 2
                and parts[0].startswith("device_")
                and parts[1] == "data"
            )
            in_device = bool(parts and parts[0].startswith("device_"))
            in_host = bool(parts and parts[0] == "host")
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                file_count += 1
                total_bytes += size
                if in_device_data and size > 0:
                    device_data_count += 1
                    device_data_bytes += size
                if (
                    in_device
                    and filename.startswith("end_info")
                    and filename.endswith(".done")
                ):
                    device_end_count += 1
                if in_host and filename == "end_info.done":
                    host_end_count += 1
    return (
        f"raw_roots={len(raw_roots)}, files={file_count}, bytes={total_bytes}, "
        f"device_data_files={device_data_count}, "
        f"device_data_bytes={device_data_bytes}, "
        f"device_end_markers={device_end_count}, "
        f"host_end_markers={host_end_count}"
    )


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _env_dir(name: str, *, default: str) -> str:
    return os.getenv(name) or os.getenv(_VLLM_TORCH_PROFILER_DIR_ENV) or default


def _env_ranks(
    name: str,
    *,
    default: frozenset[int],
) -> frozenset[int] | None:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized == "all":
        return None
    if not normalized:
        raise ValueError(f"{name} must be 'all' or comma-separated ranks")
    try:
        ranks = frozenset(int(rank.strip()) for rank in normalized.split(","))
    except ValueError as exc:
        raise ValueError(
            f"{name} must be 'all' or comma-separated non-negative ranks, got {value!r}"
        ) from exc
    if any(rank < 0 for rank in ranks):
        raise ValueError(f"{name} must contain non-negative ranks, got {value!r}")
    return ranks


__all__ = [
    "AFDNPUProfilerConfig",
    "AFDNPUProfiler",
    "afd_npu_profiler_config",
    "create_afd_npu_profiler",
    "step_afd_npu_profiler",
    "stop_afd_npu_profiler",
]
