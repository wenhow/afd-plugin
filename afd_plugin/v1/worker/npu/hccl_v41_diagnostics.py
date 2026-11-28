# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit eager diagnostic settings bound to one transport, without globals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from afd_plugin.connectors.npu.hccl_diagnostics import (
    HcclDiagnosticExecution,
    HcclStageDiagnosticRun,
    HcclStageDiagnostics,
)
from afd_plugin.connectors.npu.hccl_eager_pipeline import HcclEagerPipeline
from afd_plugin.connectors.npu.hccl_overlap import (
    HcclOverlapSettings,
    select_hccl_overlap,
)
from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport


@dataclass(frozen=True, slots=True)
class HcclV41DiagnosticBinding:
    """One immutable rank's opt-in sink; native owners retain execution state.

    Unlike an eager U2 stream session, this binding also serves serial U1.
    Graph owners retain their separately captured immutable diagnostics.
    """

    transport: HcclStagedTransport
    settings: HcclOverlapSettings
    emit: Callable[[dict[str, str | int | bool]], None] | None = None
    synchronize: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.settings, HcclOverlapSettings):
            raise ValueError("HCCL diagnostics require immutable overlap settings")

    def require_idle(self, transport: HcclStagedTransport) -> None:
        """Reject a foreign or in-flight binding before producer or wire work."""
        if self.transport is not transport or (
            not transport.initialized
            or transport.closed
            or transport.terminal
            or transport.plan is not None
            or transport.graph_reservation is not None
        ):
            raise ValueError("bind diagnostics to their exact idle live transport")
        if transport.negotiated_capabilities.graph_execution_version:
            raise ValueError("Graph diagnostics belong to the captured Graph owner")

    def require_settings(self, settings: HcclOverlapSettings) -> None:
        if self.settings != settings:
            raise ValueError("HCCL diagnostic and execution settings must match")

    def bind_parent(
        self,
        *,
        eager_pipeline: HcclEagerPipeline | None = None,
        compute_stream: torch.npu.Stream | None = None,
    ) -> HcclStageDiagnosticRun | None:
        """Describe the accepted actual route, including aliased eager streams."""
        transport, plan = self.transport, self.transport.plan
        if (
            not transport.initialized
            or transport.closed
            or transport.terminal
            or plan is None
        ):
            raise ValueError("diagnostics require their accepted live parent")
        actual = HcclDiagnosticExecution(mode="serial", compute_stream="parent")
        if eager_pipeline is None:
            if compute_stream is not None or transport.eager_pipeline is not None:
                raise ValueError("serial diagnostics require the actual serial route")
        else:
            if (
                eager_pipeline.transport is not transport
                or eager_pipeline.plan is not plan
                or transport.eager_pipeline is not eager_pipeline
                or eager_pipeline.finished
                or eager_pipeline.failed
                or compute_stream is None
            ):
                raise ValueError("eager diagnostics require their exact active owner")
            parent = eager_pipeline.parent_stream
            send, recv = eager_pipeline.send_stream, eager_pipeline.receive_stream
            actual = HcclDiagnosticExecution(
                mode="eager-streamed",
                compute_stream="parent" if compute_stream == parent else "compute",
                send_stream="parent" if send == parent else "send",
                recv_stream="parent" if recv == parent else "recv",
                stream_aliases=(
                    compute_stream == send,
                    compute_stream == recv,
                    send == recv,
                ),
            )
        if not (
            self.settings.stage_diagnostics
            or self.settings.ffn_compute_sync_diagnostics
        ):
            return None
        selection = select_hccl_overlap(
            self.settings,
            role=transport.role,
            phase="eager",
            num_stages=len(plan.stages),
        )
        return HcclStageDiagnostics(
            selection,
            transport.world_rank,
            transport.role_rank,
            emit=self.emit,
            synchronize=self.synchronize,
        ).bind_execution(
            layer_indices=plan.layer_indices,
            actual_execution=actual,
            transaction_id=plan.transaction_id,
            in_profile_run=plan.mode == "profile",
            is_warmup=plan.mode in ("warmup", "dummy"),
        )
