# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Prepare one eager stream family before profiling and bind after native KV.

The connector supplies its immutable settings snapshot and negotiated transport.
This owner neither admits a model nor changes native KV/Engram ownership. The
worker retains STOP, synchronization, process-group cleanup and failed-resource
supervision; preparing or failing this object performs none of that cleanup.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from vllm.config import VllmConfig

from afd_plugin.connectors.npu.hccl_config import validate_staged_hccl_binding
from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.v1.worker.npu.hccl_v41_diagnostics import HcclV41DiagnosticBinding
from afd_plugin.v1.worker.npu.hccl_v41_eager import HcclV41EagerSession

EAGER_SESSION_STREAM_COUNT = 3


class HcclV41EagerBootstrap:
    """Retain preparation resources across the loaded-model/KV handoff."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        transport: HcclStagedTransport,
        *,
        overlap_settings: HcclOverlapSettings,
        synchronize: Callable[[], None],
        diagnostic_sink: Callable[[dict[str, str | int | bool]], None] | None = None,
    ) -> None:
        if not isinstance(overlap_settings, HcclOverlapSettings):
            raise ValueError("eager bootstrap requires immutable overlap settings")
        self.vllm_config, self.transport = vllm_config, transport
        self.overlap_settings = overlap_settings
        self.synchronize, self.diagnostic_sink = synchronize, diagnostic_sink
        self.session: HcclV41EagerSession | None = None
        self.diagnostics: HcclV41DiagnosticBinding | None = None
        # Retain even a partially allocated stream family for process cleanup.
        self._streams: tuple[torch.npu.Stream, ...] = ()
        self._prepared = self._failed = False
        self._kv_started = self._kv_completed = False

    def _validate_binding(self) -> bool:
        transport = self.transport
        local, negotiated = transport.capabilities, transport.negotiated_capabilities
        local.validate()
        negotiated.validate()
        parallel = self.vllm_config.parallel_config
        enabled = (
            self.overlap_settings.eager_u2_stream_overlap and parallel.use_ubatching
        )
        if (
            local.graph_execution_version != 0
            or negotiated.graph_execution_version != 0
            or local.eager_pipeline_version != int(enabled)
            or negotiated.eager_pipeline_version != int(enabled)
            or local.negotiate(negotiated) != negotiated
        ):
            raise ValueError("eager bootstrap settings and HELLO family must match")
        if enabled and parallel.num_ubatches != 2:
            raise ValueError("eager bootstrap requires native U2 when enabled")
        if (
            self._failed
            or not transport.initialized
            or transport.closed
            or transport.terminal
            or transport.plan is not None
            or transport.graph_reservation is not None
        ):
            raise RuntimeError("eager bootstrap requires its idle live transport")
        validate_staged_hccl_binding(
            self.vllm_config,
            transport,
            torch.device("npu", transport.local_device),
        )
        return enabled

    def prepare(self, parent_stream: torch.npu.Stream) -> None:
        """Check HELLO before allocation/warmup; retain any failed resources."""
        try:
            if self._prepared or self._failed:
                raise RuntimeError(
                    "eager bootstrap prepares one loaded model only once"
                )
            enabled = self._validate_binding()
            if enabled:
                device = torch.device("npu", self.transport.local_device)
                if (
                    parent_stream.device != device
                    or torch.npu.current_stream(device) != parent_stream
                    or torch.npu.is_current_stream_capturing()
                ):
                    raise ValueError(
                        "eager bootstrap requires its actual NPU parent stream"
                    )
                for _ in range(EAGER_SESSION_STREAM_COUNT):
                    self._streams += (torch.npu.Stream(device=device),)
                compute, send, receive = self._streams
                self.session = HcclV41EagerSession(
                    self.vllm_config,
                    self.transport,
                    self.overlap_settings,
                    parent_stream=parent_stream,
                    compute_stream=compute,
                    send_stream=send,
                    receive_stream=receive,
                )
                # Publish the session before matched warmup can fail or block.
                self.session.initialize()
            if (
                self.overlap_settings.stage_diagnostics
                or self.overlap_settings.ffn_compute_sync_diagnostics
            ):
                self.diagnostics = HcclV41DiagnosticBinding(
                    self.transport,
                    self.overlap_settings,
                    emit=self.diagnostic_sink,
                    synchronize=self.synchronize,
                )
                self.diagnostics.require_idle(self.transport)
                if self.session is not None:
                    self.diagnostics.require_settings(self.session.settings)
            self._prepared = True
        except BaseException:
            self.fail()
            raise

    def begin_kv_init(self) -> None:
        """Reject a second KV lifetime before the native allocator writes."""
        try:
            if not self._prepared or self._kv_started or self._failed:
                raise RuntimeError(
                    "eager KV initialization requires its first prepared load"
                )
            self._validate_binding()
            self._kv_started = True
        except BaseException:
            self.fail()
            raise

    def complete_kv_init(
        self, bind_session: Callable[[HcclV41EagerSession], None]
    ) -> None:
        """Hand the same warmed session to the final native U2 runner once."""
        try:
            if not self._kv_started or self._kv_completed or self._failed:
                raise RuntimeError("eager KV completion requires its first allocation")
            self._validate_binding()
            if self.session is not None:
                bind_session(self.session)
            self._kv_completed = True
        except BaseException:
            self.fail()
            raise

    def fail(self) -> None:
        """Keep streams/session/diagnostics for supervision without peer work."""
        self._failed = True
        self.transport.terminal = True


__all__ = ["HcclV41EagerBootstrap"]
