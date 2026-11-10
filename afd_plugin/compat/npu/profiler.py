# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Plugin-owned NPU profiler helpers for AFD Ascend runners."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
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
_DEFAULT_ACTIVE_STEPS: Final[dict[AFDNPUProfilerRole, int]] = {
    "attention": 10,
    "ffn": 20,
}
_DEFAULT_WAIT_STEPS: Final[int] = 2
_DEFAULT_WARMUP_STEPS: Final[int] = 1
_DEFAULT_REPEAT: Final[int] = 1
_DEFAULT_SKIP_FIRST_STEPS: Final[int] = 1500
_DEFAULT_PROFILE_RANKS: Final[frozenset[int]] = frozenset({0})
_VLLM_TORCH_PROFILER_DIR_ENV: Final[str] = "VLLM_TORCH_PROFILER_DIR"


@dataclass(frozen=True)
class AFDNPUProfilerConfig:
    enabled: bool
    wait: int
    warmup: int
    active: int
    repeat: int
    skip_first: int
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
        wait=_env_int(f"{prefix}_WAIT", default=_DEFAULT_WAIT_STEPS),
        warmup=_env_int(f"{prefix}_WARMUP", default=_DEFAULT_WARMUP_STEPS),
        active=_env_int(f"{prefix}_ACTIVE", default=_DEFAULT_ACTIVE_STEPS[role]),
        repeat=_env_int(f"{prefix}_REPEAT", default=_DEFAULT_REPEAT),
        skip_first=_env_int(
            f"{prefix}_SKIP_FIRST",
            default=_DEFAULT_SKIP_FIRST_STEPS,
        ),
        trace_dir=_env_dir(f"{prefix}_DIR", default=_DEFAULT_DIR[role]),
        with_stack=_env_bool(f"{prefix}_WITH_STACK", default=False),
        ranks=_env_ranks(f"{prefix}_RANKS"),
    )


def create_afd_npu_profiler(role: AFDNPUProfilerRole) -> AFDNPUProfiler | None:
    """Create a torch-npu profiler when the plugin-owned env enables it."""

    config = afd_npu_profiler_config(role)
    if not config.enabled:
        return None

    profiler = _create_profiler(role, config)
    profiler.start()
    return profiler


def _create_profiler(
    role: AFDNPUProfilerRole,
    config: AFDNPUProfilerConfig,
    *,
    trace_name: str | None = None,
) -> AFDNPUProfiler:
    """Build without starting; deferred F sessions call this on their owner."""

    import torch_npu

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    )
    logger.info(
        "AFD NPU %s profiler enabled. Traces will be saved to: %s; with_stack=%s",
        role,
        config.trace_dir,
        config.with_stack,
    )
    trace_handler = (
        torch_npu.profiler.tensorboard_trace_handler(config.trace_dir)
        if trace_name is None
        else torch_npu.profiler.tensorboard_trace_handler(
            config.trace_dir, worker_name=trace_name
        )
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=config.wait,
            warmup=config.warmup,
            active=config.active,
            repeat=config.repeat,
            skip_first=config.skip_first,
        ),
        with_stack=config.with_stack,
        with_modules=config.with_stack,
        record_shapes=True,
        experimental_config=experimental_config,
        on_trace_ready=trace_handler,
    )


class AFDNPUProfilerSession:
    """Nonblocking management requests, applied by one execution thread.

    The lock protects request/status snapshots only. It is never held across
    profiler calls, trace export, connector receive, or device work. A request
    while receive is blocked remains visibly pending until the next owner
    boundary or normal STOP; it does not wake or abort a communicator.
    """

    def __init__(
        self,
        factory: Callable[[str | None], AFDNPUProfiler],
        *,
        selected: bool = True,
    ) -> None:
        self._factory = factory
        self._selected = selected
        self._lock = threading.Lock()
        self._owner: int | None = None
        self._requested = True
        self._prefix: str | None = None
        self._request_id = 0
        self._applied_id: int | None = None
        self._running = False
        self._closed = False
        self._error: Exception | None = None
        self._profiler: AFDNPUProfiler | None = None
        self._steps = 0

    def request(self, is_start: bool, profile_prefix: str | None = None) -> None:
        """Accept a desired state without waiting for the serving thread."""
        if type(is_start) is not bool:
            raise TypeError("profiler start state must be bool")
        if profile_prefix is not None and not isinstance(profile_prefix, str):
            raise TypeError("profiler prefix must be a string or None")
        with self._lock:
            if self._error is not None:
                raise RuntimeError(
                    "AFD profiler session failed; retain owner resources"
                ) from self._error
            if self._closed:
                if is_start:
                    raise RuntimeError("AFD profiler owner has exited")
                return
            # Like native WorkerProfiler, an already requested start is
            # idempotent; a different prefix does not restart an active trace.
            if self._requested == is_start:
                return
            self._requested = is_start
            self._prefix = profile_prefix if is_start else None
            self._request_id += 1

    def status(self) -> dict[str, bool | int | str | None]:
        """A JSON-safe snapshot; applied does not assert a valid NPU timeline."""
        with self._lock:
            return {
                "enabled": True,
                "selected": self._selected,
                "requested": self._requested,
                "applied": self._running,
                "pending": not self._closed and self._request_id != self._applied_id,
                "request_id": self._request_id,
                "applied_request_id": self._applied_id,
                "steps": self._steps,
                "owner_thread_id": self._owner,
                "profile_prefix": self._prefix,
                "closed": self._closed,
                "failed": self._error is not None,
                "error": None if self._error is None else str(self._error),
            }

    def _require_owner(self) -> None:
        thread_id = threading.get_ident()
        with self._lock:
            if self._owner is None:
                self._owner = thread_id
            elif self._owner != thread_id:
                raise RuntimeError(
                    "AFD profiler operations require their execution owner thread"
                )
            if self._error is not None:
                raise RuntimeError(
                    "AFD profiler session failed; retain owner resources"
                ) from self._error

    def _fail(self, error: Exception) -> None:
        with self._lock:
            self._error = error

    def before_step(self) -> None:
        """Apply the latest request at a safe owner boundary, outside capture."""
        self._require_owner()
        with self._lock:
            if self._closed:
                raise RuntimeError("AFD profiler owner has exited")
            requested, prefix, request_id = (
                self._requested,
                self._prefix,
                self._request_id,
            )
            running = self._running
        try:
            if requested and not running and self._selected:
                self._profiler = self._factory(prefix)
                self._profiler.start()
                running = True
            elif not requested and running:
                assert self._profiler is not None
                self._profiler.stop()
                self._profiler = None
                running = False
        except Exception as error:
            self._fail(error)
            raise
        with self._lock:
            self._running = running
            self._applied_id = request_id

    def step(self) -> None:
        """Advance one actual F parent, including HCCL capture/replay parents."""
        self.before_step()
        if self._profiler is not None:
            try:
                self._profiler.step()
            except Exception as error:
                self._fail(error)
                raise
            with self._lock:
                self._steps += 1

    def stop(self) -> None:
        """Compatibility with runner shutdown: request, never cross-thread stop."""
        self.request(False)

    def start(self) -> None:
        self.request(True)

    def finish(self) -> None:
        """Stop/export on the owner after STOP or loop exit; retain on failure."""
        self._require_owner()
        with self._lock:
            if self._closed:
                return
        try:
            if self._profiler is not None:
                self._profiler.stop()
        except Exception as error:
            self._fail(error)
            raise
        self._profiler = None
        with self._lock:
            self._requested = self._running = False
            self._applied_id = self._request_id
            self._closed = True


def _create_rank_profiler(
    role: AFDNPUProfilerRole,
    config: AFDNPUProfilerConfig,
    role_rank: int,
    prefix: str | None,
) -> AFDNPUProfiler:
    name = f"{role}_rank{role_rank}"
    return _create_profiler(
        role, config, trace_name=f"{prefix}_{name}" if prefix else name
    )


def create_deferred_afd_npu_profiler(
    role: AFDNPUProfilerRole, *, role_rank: int = 0
) -> AFDNPUProfilerSession | None:
    """Freeze enabled/rank settings without importing or starting torch-npu."""
    config = afd_npu_profiler_config(role)
    if not config.enabled:
        return None
    if type(role_rank) is not int or role_rank < 0:
        raise ValueError("profiler role rank must be a non-negative integer")
    return AFDNPUProfilerSession(
        partial(_create_rank_profiler, role, config, role_rank),
        selected=config.ranks is None or role_rank in config.ranks,
    )


@contextmanager
def afd_npu_profiler_owner(
    profiler: AFDNPUProfiler | None,
) -> Iterator[None]:
    """Finalize only deferred sessions; preserve the original serving failure."""
    try:
        yield
    except BaseException:
        if isinstance(profiler, AFDNPUProfilerSession):
            try:
                profiler.finish()
            except Exception:
                logger.exception("AFD profiler cleanup failed after serving failure")
        raise
    else:
        if isinstance(profiler, AFDNPUProfilerSession):
            profiler.finish()


def step_afd_npu_profiler(profiler: AFDNPUProfiler | None) -> None:
    if profiler is not None:
        profiler.step()


def stop_afd_npu_profiler(profiler: AFDNPUProfiler | None) -> None:
    if profiler is not None:
        profiler.stop()


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


def _env_int(name: str, *, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_dir(name: str, *, default: str) -> str:
    return os.getenv(name) or os.getenv(_VLLM_TORCH_PROFILER_DIR_ENV) or default


def _env_ranks(name: str) -> frozenset[int] | None:
    value = os.getenv(name)
    if value is None:
        return _DEFAULT_PROFILE_RANKS
    normalized = value.strip().lower()
    if normalized == "all":
        return None
    try:
        ranks = frozenset(int(rank.strip()) for rank in normalized.split(","))
    except ValueError as error:
        raise ValueError(
            f"{name} must be 'all' or comma-separated non-negative ranks"
        ) from error
    if any(rank < 0 for rank in ranks):
        raise ValueError(f"{name} must contain non-negative ranks")
    return ranks


__all__ = [
    "AFDNPUProfilerConfig",
    "AFDNPUProfiler",
    "AFDNPUProfilerSession",
    "afd_npu_profiler_owner",
    "afd_npu_profiler_config",
    "create_afd_npu_profiler",
    "create_deferred_afd_npu_profiler",
    "step_afd_npu_profiler",
    "stop_afd_npu_profiler",
]
