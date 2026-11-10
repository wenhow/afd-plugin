# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit eager session streams, shared by the native A/F service owners.

The caller constructs one session on each matched role, initializes it while
idle, and keeps its immutable settings through shutdown. This does not enable
production model admission or create a peer recovery path.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig

from afd_plugin.connectors.npu.hccl_config import validate_staged_hccl_binding
from afd_plugin.connectors.npu.hccl_eager_pipeline import (
    HcclEagerPipeline,
    validate_eager_topology,
)
from afd_plugin.connectors.npu.hccl_overlap import (
    HcclOverlapSettings,
    select_hccl_overlap,
)
from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport


class HcclV41EagerSession:
    """One fixed stream family; U1/profile/warmup retain the serial executor."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        transport: HcclStagedTransport,
        settings: HcclOverlapSettings,
        *,
        parent_stream: torch.npu.Stream,
        compute_stream: torch.npu.Stream,
        send_stream: torch.npu.Stream,
        receive_stream: torch.npu.Stream,
    ) -> None:
        device = torch.device("npu", transport.local_device)
        validate_staged_hccl_binding(vllm_config, transport, device)
        validate_eager_topology(transport)
        if (
            not vllm_config.parallel_config.use_ubatching
            or vllm_config.parallel_config.num_ubatches != 2
            or transport.negotiated_capabilities.graph_execution_version != 0
        ):
            raise ValueError(
                "eager stream session requires native eager U2 configuration"
            )
        if not isinstance(settings, HcclOverlapSettings):
            raise ValueError("eager stream session requires immutable settings")
        if any(
            stream.device != device
            for stream in (parent_stream, compute_stream, send_stream, receive_stream)
        ):
            raise ValueError("eager session streams must use its indexed NPU device")
        self.transport = transport
        self._settings = settings
        self.parent_stream, self.compute_stream = parent_stream, compute_stream
        self.send_stream, self.receive_stream = send_stream, receive_stream
        self._initialized = False

    @property
    def settings(self) -> HcclOverlapSettings:
        return self._settings

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        """Matched roles warm all stage groups before accepting any parent."""
        transport = self.transport
        if (
            self._initialized
            or not transport.initialized
            or transport.closed
            or transport.terminal
            or transport.plan is not None
            or transport.graph_reservation is not None
        ):
            raise RuntimeError(
                "initialize eager streams once with an idle live transport"
            )
        if self.settings.eager_u2_stream_overlap:
            transport.warmup_eager_pipeline(parent_stream=self.parent_stream)
        self._initialized = True

    def open_parent(self) -> HcclEagerPipeline | None:
        """Select before data posting; never switch an in-flight parent."""
        transport, plan = self.transport, self.transport.plan
        if (
            not self._initialized
            or not transport.initialized
            or transport.closed
            or transport.terminal
            or plan is None
        ):
            raise RuntimeError(
                "eager selection requires its initialized accepted parent"
            )
        selection = select_hccl_overlap(
            self.settings,
            role=transport.role,
            phase="eager",
            num_stages=len(plan.stages),
        )
        if not selection.eager_u2_stream_overlap or plan.mode != "target":
            return None
        return transport.open_eager_pipeline(
            parent_stream=self.parent_stream,
            send_stream=self.send_stream,
            receive_stream=self.receive_stream,
        )
