# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit host-stage diagnostics; enqueue boundaries are not device completion.

No process environment, tensors, communicator or global logger are accessed.
The caller supplies a selected plan, actual execution and an optional sink and
synchronizer. Diagnostic defaults remain in the immutable overlap settings.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSelection


@dataclass(frozen=True, slots=True)
class HcclStageDiagnostics:
    """One explicit rank's sink and configured/logical selection, without IO.

    Source: donor 4ddd69a3 ffn_model_runner.py:957-1218. This composition
    preserves first/last-layer progress and live serial-U1 compute-sync
    diagnostics without moving native execution or adding a global logger.
    Remove when native structured stage diagnostics expose the same actual
    execution and explicit synchronization boundary.
    """

    selection: HcclOverlapSelection
    world_rank: int
    role_rank: int
    emit: Callable[[dict[str, str | int | bool]], None] | None = None
    synchronize: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if any(
            type(rank) is not int or rank < 0
            for rank in (self.world_rank, self.role_rank)
        ):
            raise ValueError("HCCL diagnostic ranks must be nonnegative integers")

    def bind_execution(
        self,
        *,
        layer_indices: tuple[int, ...],
        actual_compute_stream: Literal["parent", "compute"],
        in_profile_run: bool = False,
        is_warmup: bool = False,
    ) -> HcclStageDiagnosticRun:
        """Bind actual host execution and emit its configuration before capture."""
        run = HcclStageDiagnosticRun(
            self, layer_indices, actual_compute_stream, in_profile_run, is_warmup
        )
        settings = self.selection.settings
        if settings.stage_diagnostics or settings.ffn_compute_sync_diagnostics:
            record = self.selection.diagnostic_record()
            record.update(run._fields())
            record["event"] = "configuration"
            run._emit(record)
        return run


@dataclass(frozen=True, slots=True)
class HcclStageDiagnosticRun:
    """Actual enqueue boundaries for one prepared body; no replay Python claim."""

    diagnostics: HcclStageDiagnostics
    layer_indices: tuple[int, ...]
    actual_compute_stream: Literal["parent", "compute"]
    in_profile_run: bool = False
    is_warmup: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.layer_indices) is not tuple
            or not self.layer_indices
            or any(type(layer) is not int or layer < 0 for layer in self.layer_indices)
            or any(
                left >= right
                for left, right in zip(
                    self.layer_indices, self.layer_indices[1:], strict=False
                )
            )
        ):
            raise ValueError(
                "HCCL diagnostic layers must be ordered nonnegative indices"
            )
        if self.actual_compute_stream not in ("parent", "compute"):
            raise ValueError("HCCL diagnostic actual compute stream is unsupported")
        if (
            self.diagnostics.selection.phase == "eager"
            and self.actual_compute_stream != "parent"
        ):
            raise ValueError(
                "current eager diagnostics require actual serial parent streams"
            )
        if type(self.in_profile_run) is not bool or type(self.is_warmup) is not bool:
            raise ValueError("HCCL diagnostic profile/warmup markers must be bool")
        if self.compute_sync_active and self.diagnostics.synchronize is None:
            raise ValueError(
                "active FFN compute-sync diagnostics require a synchronizer"
            )

    @property
    def compute_sync_active(self) -> bool:
        return (
            self.diagnostics.selection.ffn_compute_sync_diagnostics
            and self.actual_compute_stream == "parent"
            and not self.in_profile_run
            and not self.is_warmup
        )

    def _fields(self) -> dict[str, str | int | bool]:
        selection = self.diagnostics.selection
        return {
            "world_rank": self.diagnostics.world_rank,
            "role_rank": self.diagnostics.role_rank,
            "role": selection.role,
            "phase": selection.phase,
            "num_stages": selection.num_stages,
            "actual.mode": "serial"
            if self.actual_compute_stream == "parent"
            else "graph-streamed",
            "actual.compute_stream": self.actual_compute_stream,
            "actual.send_stream": "parent",
            "actual.recv_stream": "parent",
            "actual.compute_pipeline_enabled": self.actual_compute_stream != "parent",
            "actual.ffn_compute_sync": self.compute_sync_active,
            "in_profile_run": self.in_profile_run,
            "is_warmup": self.is_warmup,
            "boundary": "host_enqueue",
        }

    def _emit(self, record: dict[str, str | int | bool]) -> None:
        sink = self.diagnostics.emit
        if sink is None:
            logging.getLogger(__name__).warning("AFD HCCL stage diagnostic: %s", record)
        else:
            sink(record)

    def record(self, event: str, layer_idx: int, stage_idx: int) -> None:
        """Emit first/last-layer enqueue progress, without synchronizing devices."""
        if event not in (
            "recv_begin",
            "recv_returned",
            "compute_begin",
            "compute_returned",
            "send_begin",
            "send_returned",
            "layer_send_wait_begin",
            "layer_send_wait_returned",
            "final_send_wait_begin",
            "final_send_wait_returned",
        ):
            raise ValueError("HCCL diagnostic stage event is unsupported")
        self._validate_coordinates(layer_idx, stage_idx)
        if self.diagnostics.selection.stage_diagnostics and layer_idx in (
            self.layer_indices[0],
            self.layer_indices[-1],
        ):
            self._emit(
                {
                    **self._fields(),
                    "event": event,
                    "layer": layer_idx,
                    "stage": stage_idx,
                }
            )

    def _validate_coordinates(self, layer_idx: int, stage_idx: int) -> None:
        if (
            type(layer_idx) is not int
            or layer_idx not in self.layer_indices
            or type(stage_idx) is not int
            or not 0 <= stage_idx < self.diagnostics.selection.num_stages
        ):
            raise ValueError("HCCL diagnostic coordinates are outside this body")

    def synchronize_compute(self, layer_idx: int, stage_idx: int) -> None:
        """Only explicit live serial-U1 FFN diagnostics add this device fence."""
        self._validate_coordinates(layer_idx, stage_idx)
        if not self.compute_sync_active:
            return
        fields = {
            **self._fields(),
            "layer": layer_idx,
            "stage": stage_idx,
            "boundary": "explicit_synchronize",
        }
        self._emit({**fields, "event": "compute_sync_begin"})
        assert self.diagnostics.synchronize is not None
        self.diagnostics.synchronize()
        self._emit({**fields, "event": "compute_sync_complete"})
