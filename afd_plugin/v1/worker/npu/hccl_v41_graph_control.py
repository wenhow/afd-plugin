# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Attention graph agreement before native input preparation.

This explicit controller consumes native CPU geometry and the real solicited
transport protocol. It does not prepare ModelState inputs, capture a model,
launch graphs, or admit the production Graph feature. Cache/model/transport
identity is fixed for the lifetime of this instance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

from vllm.config import CUDAGraphMode, ParallelConfig
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.model_runner import BatchReqState

from afd_plugin.connectors.npu.hccl_graph_cache import HcclGraphCache
from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.distributed.hccl_graph import (
    HcclGraphCandidate,
    HcclGraphDecision,
    HcclGraphKey,
    HcclGraphReport,
    HcclGraphStage,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph import (
    HcclV41GraphDispatch,
    HcclV41GraphGeometry,
    capture_hccl_v41_geometry,
    dispatch_hccl_v41_graph,
    preview_hccl_v41_geometry,
)

ResourcesT = TypeVar("ResourcesT")
MetadataShapes = tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class HcclV41GraphReservation:
    decision: HcclGraphDecision
    geometry: HcclV41GraphGeometry
    generation: int


class HcclV41GraphControl(Generic[ResourcesT]):
    """One owning host thread and one model/cache/transport session.

    A local NONE or different graph bucket reports a cache miss after proving
    that the authoritative query can serve the actual native batch. It never
    fabricates a FULL cache hit or performs late fallback after preparation.
    """

    def __init__(
        self,
        transport: HcclStagedTransport,
        manager: CudaGraphManager | None,
        cache: HcclGraphCache[ResourcesT],
        generation: Callable[[], int],
        *,
        metadata_shapes: Callable[[HcclV41GraphGeometry], MetadataShapes] | None = None,
    ) -> None:
        if transport.role != "attention":
            raise ValueError("Attention graph control requires attention role")
        self.transport = transport
        self.manager = manager
        self.cache = cache
        self.generation = generation
        self.metadata_shapes = metadata_shapes
        self._current_reservation: HcclV41GraphReservation | None = None

    def _generation(self) -> int:
        generation = self.generation()
        if type(generation) is not int or generation < 0:
            raise ValueError("graph storage generation must be a nonnegative integer")
        return generation

    def key_for(self, geometry: HcclV41GraphGeometry) -> HcclGraphKey:
        descriptor = geometry.descriptor
        if descriptor.cg_mode != CUDAGraphMode.FULL:
            raise ValueError("static graph key requires a native FULL descriptor")
        assert descriptor.num_reqs is not None
        assert descriptor.uniform_token_count is not None
        return HcclGraphKey(
            descriptor.num_tokens,
            descriptor.num_reqs,
            descriptor.uniform_token_count,
            1 if descriptor.max_query_len is None else descriptor.max_query_len,
            descriptor.num_active_loras,
            descriptor.num_ubatches,
            tuple(
                HcclGraphStage(
                    stage.request_start,
                    stage.request_stop,
                    stage.token_start,
                    stage.token_stop,
                )
                for stage in geometry.stages
            ),
            self.metadata_shapes(geometry) if self.metadata_shapes is not None else (),
        )

    @staticmethod
    def _descriptor(key: HcclGraphKey) -> BatchExecutionDescriptor:
        return BatchExecutionDescriptor(
            CUDAGraphMode.FULL,
            key.num_tokens,
            key.num_reqs,
            uniform_token_count=key.uniform_token_count,
            max_query_len=key.max_query_len,
            num_active_loras=key.num_active_loras,
            num_ubatches=key.num_ubatches,
        )

    @staticmethod
    def _rows(geometry: HcclV41GraphGeometry) -> tuple[int, ...]:
        return tuple(stage.token_stop - stage.token_start for stage in geometry.stages)

    def _fail(self) -> None:
        # No network abort is claimed. The owner/supervisor must bound peers
        # waiting for a query, preparation, device work, or final completion.
        self.transport.terminal = True
        self.cache.fail()
        self._current_reservation = None

    def reserve_batch(
        self,
        batch_state: BatchReqState,
        *,
        parallel_config: ParallelConfig,
        allow_ubatching: bool,
        need_eager: bool = False,
        max_query_len: int | None = None,
        num_active_loras: int = 0,
        transaction_id: str | None = None,
        mode: str = "target",
    ) -> HcclV41GraphReservation | None:
        """Reserve from the actual ordered native gather result, exactly once."""
        self._current_reservation = None
        local: HcclV41GraphDispatch | None = None
        generation: int | None = None

        def select() -> HcclV41GraphDispatch:
            return dispatch_hccl_v41_graph(
                self.manager,
                batch_state,
                parallel_config=parallel_config,
                allow_ubatching=allow_ubatching,
                need_eager=need_eager or mode == "profile",
                max_query_len=max_query_len,
                num_active_loras=num_active_loras,
            )

        def resolve(candidate: HcclGraphCandidate) -> HcclGraphReport:
            nonlocal local, generation
            generation = self._generation()
            if local is None:
                local = select()
            if candidate.mode != mode or candidate.eager_padded_tokens != self._rows(
                local.eager
            ):
                raise ValueError(
                    "graph query does not match local native eager geometry or mode"
                )
            if candidate.phase == "eager":
                return HcclGraphReport("unsupported")
            if candidate.phase != "replay" or candidate.key is None:
                raise ValueError(
                    "native batch reservation requires replay or eager phase"
                )
            authoritative = preview_hccl_v41_geometry(
                batch_state, self._descriptor(candidate.key)
            )
            if self.key_for(authoritative) != candidate.key:
                raise ValueError(
                    "graph query does not match actual native request/token geometry"
                )
            if local.candidate.descriptor.cg_mode != CUDAGraphMode.FULL:
                return HcclGraphReport("missing")
            if self.key_for(local.candidate) != candidate.key:
                return HcclGraphReport("missing")
            entry = self.cache.find(candidate.key, generation)
            if entry is None:
                return HcclGraphReport("missing")
            assert self.manager is not None
            if self.manager.graphs.get(local.candidate.descriptor) is not entry.graph:
                return HcclGraphReport("missing")
            return HcclGraphReport("ready")

        try:
            candidate = None
            if self.transport.is_coordinator:
                local = select()
                full = local.candidate.descriptor.cg_mode == CUDAGraphMode.FULL
                candidate = HcclGraphCandidate(
                    "replay" if full else "eager",
                    mode,
                    self.key_for(local.candidate) if full else None,
                    self._rows(local.eager),
                )
            decision = self.transport.reserve_attention_forward(
                resolve,
                candidate=candidate,
                transaction_id=transaction_id,
            )
            if decision is None:
                return None
            assert local is not None and generation is not None
            geometry = local.eager if decision.phase == "eager" else local.candidate
            reservation = HcclV41GraphReservation(decision, geometry, generation)
            self._current_reservation = reservation
            return reservation
        except BaseException:
            self._fail()
            raise

    def reserve_capture(
        self,
        descriptor: BatchExecutionDescriptor,
        *,
        warmup: bool,
        transaction_id: str | None = None,
        mode: str = "warmup",
    ) -> HcclV41GraphReservation | None:
        """Agree a fresh native dummy factory phase before its input writes."""
        self._current_reservation = None
        geometry: HcclV41GraphGeometry | None = None
        generation: int | None = None

        def resolve(candidate: HcclGraphCandidate) -> HcclGraphReport:
            nonlocal geometry, generation
            generation = self._generation()
            geometry = capture_hccl_v41_geometry(descriptor)
            key = self.key_for(geometry)
            if (
                candidate.phase != ("warmup" if warmup else "capture")
                or candidate.mode != mode
                or candidate.key != key
                or candidate.eager_padded_tokens != self._rows(geometry)
            ):
                raise ValueError(
                    "capture query does not match the native local factory"
                )
            if self.cache.contains(key):
                raise ValueError(
                    "graph capture key already pending or retained; "
                    "restart before recapture"
                )
            if self.manager is not None and any(
                captured.max_query_len in (None, 1)
                and replace(captured, max_query_len=1)
                == replace(descriptor, max_query_len=1)
                for captured in self.manager.graphs
            ):
                raise ValueError(
                    "native manager already retains the graph capture descriptor"
                )
            return HcclGraphReport("missing")

        try:
            if type(warmup) is not bool:
                raise ValueError("warmup must be a boolean")
            candidate = None
            if self.transport.is_coordinator:
                geometry = capture_hccl_v41_geometry(descriptor)
                candidate = HcclGraphCandidate(
                    "warmup" if warmup else "capture",
                    mode,
                    self.key_for(geometry),
                    self._rows(geometry),
                )
            decision = self.transport.reserve_attention_forward(
                resolve,
                candidate=candidate,
                transaction_id=transaction_id,
            )
            if decision is None:
                return None
            assert geometry is not None and generation is not None
            reservation = HcclV41GraphReservation(decision, geometry, generation)
            self._current_reservation = reservation
            return reservation
        except BaseException:
            self._fail()
            raise

    def reserve_eager(
        self,
        geometry: HcclV41GraphGeometry,
        *,
        mode: str,
        transaction_id: str | None = None,
    ) -> HcclV41GraphReservation | None:
        """Reserve caller-owned native eager/dummy geometry, including empty.

        Whole-empty batches deliberately avoid a fabricated BatchReqState or
        the nonempty-only native preview helper.
        """
        self._current_reservation = None
        generation: int | None = None

        def resolve(candidate: HcclGraphCandidate) -> HcclGraphReport:
            nonlocal generation
            generation = self._generation()
            descriptor = geometry.descriptor
            if (
                descriptor.cg_mode != CUDAGraphMode.NONE
                or type(descriptor.num_tokens) is not int
                or descriptor.num_tokens < 0
                or type(descriptor.num_reqs) is not int
                or descriptor.num_reqs < 0
                or type(descriptor.num_ubatches) is not int
                or not 1 <= descriptor.num_ubatches <= 2
                or len(geometry.stages) != descriptor.num_ubatches
            ):
                raise ValueError(
                    "explicit eager reservation requires native NONE geometry"
                )
            previous = 0
            for stage in geometry.stages:
                if (
                    any(
                        type(value) is not int or value < 0
                        for value in (
                            stage.request_start,
                            stage.request_stop,
                            stage.token_start,
                            stage.token_stop,
                        )
                    )
                    or stage.token_start != previous
                    or not stage.token_start
                    <= stage.token_stop
                    <= descriptor.num_tokens
                    or not 0
                    <= stage.request_start
                    <= stage.request_stop
                    <= descriptor.num_reqs
                ):
                    raise ValueError(
                        "explicit eager geometry must cover the actual native parent"
                    )
                previous = stage.token_stop
            if previous != descriptor.num_tokens:
                raise ValueError(
                    "explicit eager geometry must cover the actual native parent"
                )
            if (
                candidate.phase != "eager"
                or candidate.mode != mode
                or candidate.eager_padded_tokens != self._rows(geometry)
            ):
                raise ValueError("eager query does not match the native local geometry")
            return HcclGraphReport("unsupported")

        try:
            candidate = (
                HcclGraphCandidate("eager", mode, None, self._rows(geometry))
                if self.transport.is_coordinator
                else None
            )
            decision = self.transport.reserve_attention_forward(
                resolve,
                candidate=candidate,
                transaction_id=transaction_id,
            )
            if decision is None:
                return None
            assert generation is not None
            reservation = HcclV41GraphReservation(decision, geometry, generation)
            self._current_reservation = reservation
            return reservation
        except BaseException:
            self._fail()
            raise

    def verify_prepared(
        self, reservation: HcclV41GraphReservation, actual: HcclV41GraphGeometry
    ) -> None:
        """Check the current owned handle and native geometry before launch.

        Equal values are insufficient: another controller or a retired parent
        may have the same dimensions. Both the returned handle and transport's
        still-accepted decision must retain this reservation's exact identity.
        """
        try:
            if (
                reservation is not self._current_reservation
                or self.transport.graph_reservation is not reservation.decision
            ):
                raise ValueError(
                    "native preparation requires the current owned reservation"
                )
            if actual != reservation.geometry:
                raise ValueError(
                    "actual native preparation does not match reserved geometry"
                )
            if reservation.decision.phase != "eager" and (
                self._generation() != reservation.generation
                or self.key_for(actual) != reservation.decision.key
            ):
                raise ValueError(
                    "native graph storage or metadata changed after reservation"
                )
        except BaseException:
            self._fail()
            raise
