# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Assemble native Graph owners for one loaded model and one KV lifetime.

The normal runners call this after rendezvous, before pre-KV profiling and
after native KV initialization respectively. No model admission, environment
read, graph capture or communication occurs during assembly. Native workers
retain STOP, ModelState retirement and process-group cleanup ownership.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from afd_plugin.connectors.npu.hccl_graph_cache import HcclGraphCache
from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.v1.worker.npu.hccl_v41_execution import validate_hccl_v41_binding
from afd_plugin.v1.worker.npu.hccl_v41_graph_control import HcclV41GraphControl
from afd_plugin.v1.worker.npu.hccl_v41_graph_coordinator import (
    HcclV41GraphCoordinator,
    HcclV41GraphResources,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph_eager import HcclV41GraphEager
from afd_plugin.v1.worker.npu.hccl_v41_graph_service import (
    HcclV41FFNGraphResources,
    HcclV41GraphService,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph_session import HcclV41GraphSession

if TYPE_CHECKING:
    from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )
    from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner

NATIVE_GRAPH_STORAGE_GENERATION = 0


def _storage_generation() -> int:
    # The normal startup supports one real KV allocation, never a generation
    # bump that would incorrectly authorize recapture into an existing cache.
    return NATIVE_GRAPH_STORAGE_GENERATION


class HcclV41AttentionGraphBootstrap:
    """Retain assembly state before bindings or native KV writes can fail."""

    def __init__(
        self,
        owner: AFDNPUAttentionModelRunnerV2,
        *,
        synchronize: Callable[[], None],
        overlap_settings: HcclOverlapSettings,
        diagnostic_sink: Callable[[dict[str, str | int | bool]], None] | None = None,
    ) -> None:
        transport = owner.hccl_v41_transport
        if transport is None:
            raise ValueError("Graph bootstrap requires its bound Attention transport")
        self.owner, self.transport = owner, transport
        self.overlap_settings = overlap_settings
        self.diagnostic_sink = diagnostic_sink
        self.cache: HcclGraphCache[HcclV41GraphResources] = HcclGraphCache(synchronize)
        self.session = HcclV41GraphSession(transport)
        self.control = HcclV41GraphControl(
            transport, None, self.cache, _storage_generation
        )
        self.eager: HcclV41GraphEager | None = None
        self.coordinator: HcclV41GraphCoordinator | None = None
        self._kv_started = self._bound = self._failed = False
        self._transaction_index = 0

    def _next_transaction_id(self) -> str:
        self._transaction_index += 1
        return f"native-graph-{self._transaction_index}"

    def bind_loaded_model(self) -> None:
        if self._bound or self._failed:
            raise RuntimeError("Graph bootstrap binds one loaded model only once")
        try:
            self.eager = HcclV41GraphEager(
                self.owner,
                self.control,
                next_transaction_id=self._next_transaction_id,
                overlap_settings=self.overlap_settings,
                diagnostic_sink=self.diagnostic_sink,
            )
            self.owner.bind_hccl_v41_graph_eager(self.eager, self.session)
            self._bound = True
        except BaseException:
            self.fail()
            raise

    def begin_kv_init(self) -> None:
        """Reject reinitialization before the native allocator changes storage."""
        if (
            not self._bound
            or self._failed
            or self._kv_started
            or self.owner.hccl_v41_graph_eager is not self.eager
            or self.owner.hccl_v41_graph_session is not self.session
            or self.owner.hccl_v41_transport is not self.transport
            or self.eager is None
            or self.owner.model is not self.eager.model
            or self.owner.model_state is not self.eager.model_state
            or self.owner.model_state.model is not self.owner.model
            or self.owner.cudagraph_manager is not None
            or self.cache.closed
            or self.cache.terminal
            or not self.transport.initialized
            or self.transport.closed
            or self.transport.terminal
            or self.transport.plan is not None
            or self.transport.graph_reservation is not None
            or not self.cache.is_empty
        ):
            raise RuntimeError(
                "native Graph KV initialization requires its first idle load"
            )
        validate_hccl_v41_binding(
            self.owner.vllm_config, self.transport, self.owner.device
        )
        self._kv_started = True

    def complete_kv_init(self) -> None:
        """Bind only after Ascend's final real-KV/metadata owner replacement."""
        if not self._kv_started or self.coordinator is not None or self._failed:
            raise RuntimeError(
                "native Graph KV completion requires its first allocation"
            )
        try:
            control = HcclV41GraphControl(
                self.transport,
                self.owner.cudagraph_manager,
                self.cache,
                self.control.generation,
                metadata_shapes=self.control.metadata_shapes,
            )
            self.coordinator = HcclV41GraphCoordinator(
                self.owner,
                control,
                self.session,
                next_transaction_id=self._next_transaction_id,
                overlap_settings=self.overlap_settings,
                diagnostic_sink=self.diagnostic_sink,
            )
            self.owner.bind_hccl_v41_graph_coordinator(self.coordinator)
        except BaseException:
            self.fail()
            raise

    def fail(self) -> None:
        """Keep dependencies for supervision; this sends no abort or STOP."""
        self._failed = True
        self.transport.terminal = True
        self.cache.fail()


def create_hccl_v41_ffn_graph_service(
    runner: AFDNPUFFNModelRunner,
    transport: HcclStagedTransport,
    *,
    synchronize: Callable[[], None],
    overlap_settings: HcclOverlapSettings,
    diagnostic_sink: Callable[[dict[str, str | int | bool]], None] | None = None,
) -> HcclV41GraphService:
    """Assemble after F HELLO, before its headless service thread can receive.

    F retains the original target shape despite stripping its draft config.
    GraphService constructs banks lazily from accepted A descriptors; it does
    not independently infer capture buckets or initialize Attention state.
    """
    runner.get_model()
    cache: HcclGraphCache[HcclV41FFNGraphResources] = HcclGraphCache(synchronize)
    return HcclV41GraphService(
        runner,
        transport,
        cache,
        _storage_generation,
        overlap_settings=overlap_settings,
        diagnostic_sink=diagnostic_sink,
    )


__all__ = [
    "HcclV41AttentionGraphBootstrap",
    "create_hccl_v41_ffn_graph_service",
]
