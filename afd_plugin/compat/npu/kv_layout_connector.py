# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in V4.1 eager PD attestation on the pinned native Mooncake lifecycle.

Native source: Ascend 7e37e29a, mooncake_hybrid_connector.py. The external
connector factory selects this module explicitly; no upstream/global patches.
Remove the copies when upstream exposes worker/thread factories and a typed
metadata validation hook. Layout compatibility is not remote authentication.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from _thread import LockType
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
    DONE_RECVING_MSG,
    GET_META_MSG,
    KVCacheRecvingThread,
    KVCacheSendingThread,
    MambaSpec,
    MooncakeAgentMetadata,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    RemotePortInfo,
    _reconstruct_shared_pages,
    as_kv_cache_tensors,
    collect_configured_register_regions,
    ensure_zmq_recv,
    ensure_zmq_send,
    get_ip,
    get_kv_cache_tensor_layers,
    global_te,
    group_concurrent_contiguous,
    logger,
    validate_register_region_count,
)

from afd_plugin.compat.npu.dspark_pd import (
    DsparkPdPrefixConsumer,
    DsparkPdReadLease,
    DsparkPdReadQuery,
    DsparkPdReady,
    DsparkPdSourceRequest,
    draft_schema_digest,
    layout_digest,
    prompt_digest,
    read_lease_from_payload,
    read_lease_payload,
    read_query_from_payload,
    read_query_payload,
    source_request_from_payload,
    source_request_payload,
)
from afd_plugin.compat.npu.kv_layout import (
    MAX_LAYOUT_BYTES,
    KVLayoutSnapshot,
    compare_kv_layouts,
    layout_from_json,
    validate_kv_transfer_blocks,
)
from afd_plugin.compat.npu.kv_layout_native import collect_deepseek_v41_kv_layout
from afd_plugin.compat.npu.runtime_config import (
    is_deepseek_v41_model,
    uses_deepseek_v41_dspark_pd,
    validate_deepseek_v41_pd,
)

if TYPE_CHECKING:
    from mooncake.engine import TransferEngine

MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_ATTESTED_PEERS = 16_000
MAX_TRANSFER_ADDRESS = (1 << 64) - 1
GET_DSPARK_READY_MSG = b"afd_dspark_get_ready_v1"
DONE_DSPARK_RECV_MSG = b"afd_dspark_done_v1"
DSPARK_SOURCE_PARAM = "afd_dspark_source_v1"
DSPARK_PROTOCOL_VERSION = 1
MAX_DSPARK_ERROR_LENGTH = 1024
DSPARK_SHUTDOWN_TIMEOUT = 30.0


class AttestedMooncakeAgentMetadata(MooncakeAgentMetadata, kw_only=True):
    """A required extension; legacy peers never silently downgrade."""

    afd_kv_layout: bytes
    afd_dspark_version: int = 0
    afd_tp_rank: int = 0
    afd_tp_size: int = 1


@dataclass(frozen=True, slots=True)
class _AttestedPeer:
    layout: KVLayoutSnapshot
    addresses: tuple[int, ...]
    host: str
    port: int
    dspark_version: int = 0
    tp_rank: int = 0
    tp_size: int = 1


@dataclass(frozen=True, slots=True)
class DsparkPdDestination:
    """The original destination allocation, captured before native queueing."""

    source: DsparkPdSourceRequest
    generation: str
    block_ids: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, DsparkPdSourceRequest):
            raise ValueError("DSpark destination requires a typed source declaration")
        if not isinstance(self.generation, str) or not self.generation:
            raise ValueError("DSpark destination generation must be nonempty")
        if (
            not isinstance(self.block_ids, tuple)
            or len(self.block_ids) != len(self.source.transfer_block_ids)
            or any(
                not isinstance(group, tuple)
                or any(type(block) is not int or block <= 0 for block in group)
                or len(group) != len(set(group))
                for group in self.block_ids
            )
        ):
            raise ValueError("DSpark destination requires nonzero all-group block IDs")


class DsparkMooncakeConnectorMetadata(MooncakeConnectorMetadata):
    def __init__(self):
        super().__init__()
        self.dspark_sources: tuple[DsparkPdSourceRequest, ...] = ()
        self.dspark_destinations: dict[str, DsparkPdDestination] = {}
        self.dspark_cancelled: tuple[tuple[str, str], ...] = ()


@dataclass(slots=True)
class _DsparkPublication:
    source: DsparkPdSourceRequest
    ready: DsparkPdReady | None = None
    lease: DsparkPdReadLease | None = None
    cancelled: bool = False
    completed: bool = False


class DsparkPdReadyRegistry:
    """P request leases, independent of native timeout-based block release.

    The worker owns this instance for its one registration lifetime. A completed
    publication is retained as a bounded tombstone for identical DONE retries.
    Locks protect host descriptors only; no model/device operation runs here.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._publications: OrderedDict[tuple[str, str], _DsparkPublication] = (
            OrderedDict()
        )
        self._current: dict[str, str] = {}
        self._finished: set[str] = set()
        self._closed = False
        self._failed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("DSpark source registry admission is closed")

    def close(self) -> None:
        """Close admission before checking leases; a failed close is terminal.

        Existing typed DONE remains admissible so a real in-flight read can
        drain, but it never revives this owner after a failed shutdown.
        """
        with self._lock:
            self._closed = True
            if self._failed or any(
                not item.completed for item in self._publications.values()
            ):
                self._failed = True
                raise RuntimeError("DSpark source/lease has not drained for shutdown")

    def admit(self, sources: tuple[DsparkPdSourceRequest, ...]) -> None:
        with self._lock:
            self._require_open()
            pending = []
            request_ids: set[str] = set()
            for source in sources:
                if source.request_id in request_ids:
                    raise ValueError("duplicate request in DSpark source batch")
                request_ids.add(source.request_id)
                if source.request_id in self._finished:
                    raise RuntimeError(
                        "previous DSpark completion is not yet delivered"
                    )
                previous = self._current.get(source.request_id)
                if previous is not None and previous != source.source_generation:
                    raise RuntimeError("previous DSpark source generation is retained")
                key = (source.request_id, source.source_generation)
                publication = self._publications.get(key)
                if publication is not None:
                    old = publication.source
                    if publication.cancelled or publication.completed:
                        raise RuntimeError("DSpark source generation is closed")
                    if (
                        old.request_id != source.request_id
                        or old.source_generation != source.source_generation
                        or old.original_prompt_tokens != source.original_prompt_tokens
                        or old.computed_prefix_tokens != source.computed_prefix_tokens
                        or old.prompt_digest != source.prompt_digest
                        or old.source_engine_id != source.source_engine_id
                        or old.draft_schema_digest != source.draft_schema_digest
                        or len(old.allocation_block_ids)
                        != len(source.allocation_block_ids)
                        or (publication.ready is not None and old != source)
                    ):
                        raise ValueError(
                            "DSpark source changed its contract or published allocation"
                        )
                    # Native SWA can replace retired prefix slots with null 0.
                    # The actual prefix consumer validates current write slots;
                    # only Ready/leased mappings must be wholly immutable here.
                    pending.append((key, publication, source))
                else:
                    pending.append((key, None, source))
            new_count = sum(publication is None for _, publication, _ in pending)
            self._evict_completed(required=new_count)
            if len(self._publications) + new_count > MAX_ATTESTED_PEERS:
                raise RuntimeError("DSpark source registry capacity exhausted")
            for key, publication, source in pending:
                if publication is None:
                    self._publications[key] = _DsparkPublication(source)
                else:
                    publication.source = source
                self._current[source.request_id] = source.source_generation

    def _evict_completed(self, *, required: int) -> None:
        for key in tuple(self._publications):
            if len(self._publications) + required <= MAX_ATTESTED_PEERS:
                break
            if self._publications[key].completed:
                del self._publications[key]

    def publish(self, ready_batch: tuple[DsparkPdReady, ...]) -> None:
        with self._lock:
            self._require_open()
            publications = []
            for ready in ready_batch:
                publication = self._publications[
                    (ready.source_request_id, ready.source_generation)
                ]
                source = publication.source
                if (
                    publication.cancelled
                    or publication.completed
                    or ready.source_engine_id != source.source_engine_id
                    or ready.original_prompt_tokens != source.original_prompt_tokens
                    or ready.computed_prefix_tokens != source.computed_prefix_tokens
                    or ready.prompt_digest != source.prompt_digest
                    or ready.draft_schema_digest != source.draft_schema_digest
                    or ready.transfer_block_ids != source.transfer_block_ids
                    or publication.ready not in (None, ready)
                ):
                    raise ValueError("DSpark ready does not match its admitted source")
                publications.append((publication, ready))
            for publication, ready in publications:
                publication.ready = ready

    def acquire(self, query: DsparkPdReadQuery) -> DsparkPdReadLease:
        with self._lock:
            self._require_open()
            ready = query.expected_ready
            publication = self._publications[
                (ready.source_request_id, ready.source_generation)
            ]
            if (
                publication.ready != ready
                or publication.cancelled
                or publication.completed
            ):
                raise RuntimeError("DSpark source is not ready for this query")
            lease = publication.lease
            if lease is not None:
                if (
                    lease.query_id != query.query_id
                    or lease.destination_engine_id != query.destination_engine_id
                    or lease.destination_generation != query.destination_generation
                    or lease.destination_block_ids != query.destination_block_ids
                ):
                    raise RuntimeError("DSpark source already belongs to another read")
                return lease
            lease = DsparkPdReadLease(
                ready,
                query.query_id,
                query.destination_engine_id,
                query.destination_generation,
                query.destination_block_ids,
                uuid.uuid4().hex,
            )
            publication.lease = lease
            return lease

    def complete(self, lease: DsparkPdReadLease) -> None:
        with self._lock:
            key = (lease.ready.source_request_id, lease.ready.source_generation)
            publication = self._publications[key]
            if publication.lease != lease:
                raise ValueError("DSpark DONE does not match the acquired read lease")
            if publication.completed:
                return
            self._finish(key, publication)

    def cancel(self, request_id: str, generation: str) -> None:
        with self._lock:
            key = (request_id, generation)
            publication = self._publications[key]
            publication.cancelled = True
            if publication.lease is None and not publication.completed:
                self._finish(key, publication)

    def _finish(self, key: tuple[str, str], publication: _DsparkPublication) -> None:
        publication.completed = True
        if self._current.get(key[0]) == key[1]:
            del self._current[key[0]]
        self._finished.add(key[0])

    def owns(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._current

    def is_dspark(self, request_id: str) -> bool:
        with self._lock:
            return any(key[0] == request_id for key in self._publications)

    def cancel_current(self, request_id: str) -> None:
        with self._lock:
            generation = self._current.get(request_id)
            if generation is not None:
                self.cancel(request_id, generation)

    def take_finished(self) -> set[str]:
        with self._lock:
            finished = self._finished.copy()
            self._finished.clear()
            return finished


@dataclass(slots=True)
class _DsparkReadAttempt:
    destination: DsparkPdDestination
    lease: DsparkPdReadLease | None = None
    read_started: bool = False
    read_drained: bool = False


class _ReceiverControl(Enum):
    RETIRE = "retire"


@contextmanager
def _before_retirement_deadline(
    lock: LockType | threading.Condition, deadline: float
) -> Iterator[None]:
    if not lock.acquire(timeout=max(0, deadline - time.monotonic())):
        raise TimeoutError("receiver retirement frontier lock did not drain")
    try:
        yield
    finally:
        lock.release()


def _validate_layout_role(
    layout: KVLayoutSnapshot, *, draft_layer_names: tuple[str, ...] = ()
) -> None:
    if layout.device_type != "npu":
        raise ValueError("attested V4.1 PD requires materialized NPU storage")
    actual_draft = {
        name for group in layout.groups if group.is_draft for name in group.layer_names
    }
    all_names = {name for group in layout.groups for name in group.layer_names}
    if (
        not set(draft_layer_names).issubset(all_names)
        or not actual_draft.issubset(draft_layer_names)
        or any(g.host_resident or not g.enable_transfer for g in layout.groups)
    ):
        raise ValueError(
            "attested V4.1 PD requires device transferable target groups only"
        )


def _validate_native_arrays(
    layout: KVLayoutSnapshot,
    addresses: Sequence[int],
    block_lengths: Sequence[int],
) -> None:
    if (
        tuple(block_lengths) != tuple(p.block_stride_bytes for p in layout.pages)
        or any(type(n) is not int for n in block_lengths)
        or len(addresses) != len(layout.pages)
        or any(
            type(a) is not int or not 0 < a <= MAX_TRANSFER_ADDRESS for a in addresses
        )
    ):
        raise ValueError("Mooncake metadata arrays disagree with the typed layout")
    bases = {}
    aliases = {}
    ranges = set()
    for address, page in zip(addresses, layout.pages, strict=True):
        end = address + layout.num_blocks * page.block_stride_bytes
        if end > MAX_TRANSFER_ADDRESS:
            raise ValueError("Mooncake page address span exceeds uint64")
        base = address - page.storage_offset_bytes
        if base <= 0:
            raise ValueError("Mooncake page storage base must be a positive address")
        if bases.setdefault(page.storage_alias, base) != base:
            raise ValueError("Mooncake address aliases disagree with the typed layout")
        identity = (page.storage_alias, page.storage_offset_bytes)
        if aliases.setdefault(address, identity) != identity:
            raise ValueError("Mooncake address aliases disagree with the typed layout")
        ranges.add((address, end))
    previous_end = 0
    for start, end in sorted(ranges):
        if start < previous_end:
            raise ValueError("Mooncake page addresses overlap outside declared aliases")
        previous_end = end


def _decode_attestation(
    metadata: AttestedMooncakeAgentMetadata,
    local: KVLayoutSnapshot,
    block_size: int,
    *,
    draft_layer_names: tuple[str, ...] = (),
) -> KVLayoutSnapshot:
    layout = layout_from_json(metadata.afd_kv_layout.decode("utf-8"))
    _validate_layout_role(layout, draft_layer_names=draft_layer_names)
    comparison = compare_kv_layouts(layout, local)
    if not comparison.compatible:
        raise ValueError(
            "incompatible KV layouts: " + "; ".join(comparison.differences)
        )
    if (
        type(metadata.num_blocks) is not int
        or metadata.num_blocks != layout.num_blocks
        or type(metadata.block_size) is not int
        or metadata.block_size != block_size
        or metadata.ssm_sizes != (0, 0)
    ):
        raise ValueError("Mooncake metadata arrays disagree with the typed layout")
    _validate_native_arrays(layout, metadata.kv_caches_base_addr, metadata.block_lens)
    return layout


def _validate_local_registration(
    worker: AFDMooncakeConnectorWorker,
    layout: KVLayoutSnapshot,
    pointers: list[int],
    lengths: list[int],
) -> None:
    expected_addresses = tuple(
        min(
            t.data_ptr()
            for name in page.layers
            for t in as_kv_cache_tensors(worker.kv_caches[name])
        )
        for page in layout.pages
    )
    expected_strides = tuple(page.block_stride_bytes for page in layout.pages)
    _validate_native_arrays(layout, expected_addresses, expected_strides)
    layer_groups = {
        name: i for i, g in enumerate(layout.groups) for name in g.layer_names
    }
    expected_groups = tuple(
        tuple(sorted({layer_groups[name] for name in page.layers}))
        for page in layout.pages
    )
    if (
        tuple(worker.kv_caches_base_addr) != expected_addresses
        or tuple(worker.block_len_per_addr) != expected_strides
        or tuple(worker.block_stride_per_addr) != expected_strides
        or tuple(tuple(g) for g in worker.addr_group_idx) != expected_groups
        or len(pointers) != len(lengths)
    ):
        raise ValueError(
            "native registration arrays disagree with the materialized layout"
        )
    for address, stride in zip(expected_addresses, expected_strides, strict=True):
        if not any(
            p <= address and address + layout.num_blocks * stride <= p + n
            for p, n in zip(pointers, lengths, strict=True)
        ):
            raise ValueError(
                "native registration does not cover an attested physical page"
            )


class AFDMooncakeConnectorScheduler(MooncakeConnectorScheduler):
    # Native 7e37e29a:1418-1496. Keep native allocation/topology setup; only
    # instance-owned DSpark declarations and generation state are added.
    # Delegation exception: the unchanged native constructor is 79 lines.
    # Remove this extension when native carries typed generation metadata.
    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        # ### PATCH START: request-owned DSpark declarations
        self._dspark_enabled = uses_deepseek_v41_dspark_pd(vllm_config)
        self._dspark_schema = (
            draft_schema_digest(vllm_config, kv_cache_config)
            if self._dspark_enabled
            else None
        )
        self._dspark_original: dict[str, tuple[int, str, str]] = {}
        self._dspark_sources: dict[str, DsparkPdSourceRequest] = {}
        self._dspark_allocations: dict[str, KVCacheBlocks] = {}
        self._dspark_remote_sources: dict[str, DsparkPdSourceRequest] = {}
        self._dspark_destinations: dict[str, DsparkPdDestination] = {}
        self._dspark_cancelled: list[tuple[str, str]] = []
        self._dspark_failed = False
        # ### PATCH END: request-owned DSpark declarations

    # Native 7e37e29a:1566-1571, exact signature. Preserve the original prompt
    # before native P truncation; D validates that same prompt before any pull.
    # Remove when native preserves an authenticated original-prompt declaration.
    def on_new_request(self, request: Request) -> None:
        params = request.kv_transfer_params
        # ### PATCH START: original prompt and source generation
        if (
            self._dspark_enabled
            and params is not None
            and (params.get("do_remote_decode") or params.get("do_remote_prefill"))
        ):
            if request.prompt_token_ids is None:
                raise ValueError("DSpark PD requires original text prompt IDs")
            tokens = tuple(request.prompt_token_ids)
            digest = prompt_digest(tokens)
            self._dspark_original[request.request_id] = (
                len(tokens),
                digest,
                uuid.uuid4().hex,
            )
            if params.get("do_remote_prefill"):
                source = source_request_from_payload(params[DSPARK_SOURCE_PARAM])
                required_prefix = self._state_prefill_token_count(len(tokens))
                if (
                    source.original_prompt_tokens != len(tokens)
                    or source.prompt_digest != digest
                    or source.computed_prefix_tokens != required_prefix
                    or source.draft_schema_digest != self._dspark_schema
                    or source.request_id != params["remote_request_id"]
                    or source.source_engine_id != params["remote_engine_id"]
                    or source.transfer_block_ids
                    != tuple(tuple(ids) for ids in params["remote_block_ids"])
                ):
                    raise ValueError(
                        "DSpark source declaration does not match this prompt"
                    )
                self._dspark_remote_sources[request.request_id] = source
        # ### PATCH END: original prompt and source generation
        # Native truncation must remain after the original-prompt capture.
        if params is not None and params.get("do_remote_decode") and self.need_truncate:
            self._truncate_request_for_prefill(request)

    # Native 7e37e29a:1607-1628, exact signature. The native body keeps its
    # unhashed all-group mapping and one-transfer flags. The extra declarations
    # capture the actual allocation before queue admission, not a later lookup.
    # Delegation exception: the native dictionary/rank flags remain unchanged.
    # Remove when native owns immutable allocation envelopes and chunk views.
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        params = request.kv_transfer_params
        is_receive = params is not None and params.get("do_remote_prefill", False)
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        # ### PATCH START: immutable allocation envelope
        if not self._dspark_enabled or request.request_id not in self._dspark_original:
            return
        original, digest, generation = self._dspark_original[request.request_id]
        if params is not None and params.get("do_remote_decode"):
            # Native SWA retains absolute coordinates with retired prefix
            # nulls. Filtering them would shift the draft writer's positions.
            groups = tuple(tuple(ids) for ids in blocks.get_block_ids())
            computed_prefix = request.num_prompt_tokens
            exported = self.get_sw_clipped_blocks(
                self._compute_transfer_block_ids(groups, computed_prefix)
            )
            assert self._dspark_schema is not None
            self._dspark_sources[request.request_id] = DsparkPdSourceRequest(
                request.request_id,
                generation,
                original,
                computed_prefix,
                digest,
                groups,
                tuple(tuple(ids) for ids in exported),
                self.engine_id,
                self._dspark_schema,
            )
            # KVCacheManager.get_blocks preserves the manager-owned per-group
            # lists. RUNNING chunk allocations extend/retire these lists without
            # calling this admission hook again. Snapshot them in each batch.
            self._dspark_allocations[request.request_id] = blocks
        elif is_receive and num_external_tokens > 0:
            groups = tuple(
                tuple(ids) for ids in blocks.get_unhashed_block_ids_all_groups()
            )
            self._dspark_destinations[request.request_id] = DsparkPdDestination(
                self._dspark_remote_sources[request.request_id], generation, groups
            )
        # ### PATCH END: immutable allocation envelope

    # Native 7e37e29a:1630-1656, exact signature. Delegate the unchanged native
    # batch construction and add typed P/D records to the resulting metadata.
    # Delegation exception: preserve the complete native receive/send batch.
    # Remove when native transports typed ready/cancel generations itself.
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        # ### PATCH START: preemption invalidates the in-progress projection
        if self._dspark_enabled:
            if self._dspark_failed:
                raise RuntimeError("DSpark PD scheduler is terminal after preemption")
            if (
                set(scheduler_output.preempted_req_ids or ())
                & self._dspark_sources.keys()
            ):
                # Native preemption pops the KV manager's lists. Reusing the
                # retained view would name freed blocks; resumption requires a
                # new generation and complete re-projection, outside this slice.
                self._dspark_failed = True
                raise RuntimeError("DSpark PD does not resume a preempted source")
        # ### PATCH END: preemption invalidates the in-progress projection
        native = super().build_connector_meta(scheduler_output)
        if not self._dspark_enabled:
            return native
        # ### PATCH START: consumed DSpark batch metadata
        metadata = DsparkMooncakeConnectorMetadata()
        metadata.requests = native.requests
        metadata.requests_to_send = native.requests_to_send
        metadata.reqs_in_batch = native.reqs_in_batch
        for request_id in scheduler_output.num_scheduled_tokens:
            if request_id in self._dspark_allocations:
                source = self._dspark_sources[request_id]
                allocation = tuple(
                    tuple(ids)
                    for ids in self._dspark_allocations[request_id].get_block_ids()
                )
                exported = self.get_sw_clipped_blocks(
                    self._compute_transfer_block_ids(
                        allocation, source.computed_prefix_tokens
                    )
                )
                self._dspark_sources[request_id] = replace(
                    source,
                    allocation_block_ids=allocation,
                    transfer_block_ids=tuple(tuple(ids) for ids in exported),
                )
        cancelled_ids = {request_id for request_id, _ in self._dspark_cancelled}
        selected = set(scheduler_output.num_scheduled_tokens) | cancelled_ids
        metadata.dspark_sources = tuple(
            source
            for request_id, source in self._dspark_sources.items()
            if request_id in selected
        )
        metadata.dspark_destinations = {
            request_id: self._dspark_destinations[request_id]
            for request_id in metadata.requests
        }
        metadata.dspark_cancelled = tuple(self._dspark_cancelled)
        self._dspark_cancelled.clear()
        for request_id in cancelled_ids:
            self._release_dspark_tracking(request_id)
        return metadata
        # ### PATCH END: consumed DSpark batch metadata

    # Native 7e37e29a:1658-1705, exact signature. Keep native prefix/SWA export
    # and delay-free behavior; attach a source declaration, never a ready claim.
    # Delegation exception: native owns exact status/truncation/response fields.
    # Remove when native implements matching request-ready leases.
    def request_finished_all_groups(
        self, request: Request, block_ids: BlockIds
    ) -> tuple[bool, dict[str, Any] | None]:
        delayed, params = super().request_finished_all_groups(request, block_ids)
        # ### PATCH START: generation-aware source publication/cancellation
        source = self._dspark_sources.get(request.request_id)
        if not self._dspark_enabled or source is None:
            if self._dspark_enabled:
                self._release_dspark_tracking(request.request_id)
            return delayed, params
        if params is not None:
            if (
                tuple(tuple(ids) for ids in params["remote_block_ids"])
                != source.transfer_block_ids
            ):
                raise ValueError(
                    "native exported blocks changed after DSpark allocation"
                )
            params[DSPARK_SOURCE_PARAM] = source_request_payload(source)
            self._release_dspark_tracking(request.request_id)
            return delayed, params
        self._dspark_cancelled.append((source.request_id, source.source_generation))
        # The worker completes cancellation only after a no-read/lease drain.
        return any(source.allocation_block_ids), None
        # ### PATCH END: generation-aware source publication/cancellation

    def _release_dspark_tracking(self, request_id: str) -> None:
        """Worker Ready/lease owns P retention; scheduler snapshots are retired."""
        self._dspark_original.pop(request_id, None)
        self._dspark_sources.pop(request_id, None)
        self._dspark_allocations.pop(request_id, None)
        self._dspark_remote_sources.pop(request_id, None)
        self._dspark_destinations.pop(request_id, None)


class MooncakeHybridConnector(MooncakeConnector):
    # Upstream: MooncakeConnector.__init__ @ 7e37e29a:1303-1318.
    # Reason/functionality: external selection must construct the attesting
    # worker before registration starts native threads; scheduler stays native.
    # Signature matches upstream, including its omitted return annotation.

    def __init__(  # type: ignore[misc]
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        assert vllm_config.kv_transfer_config is not None
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeConnectorMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                AFDMooncakeConnectorScheduler(
                    vllm_config, str(self.engine_id), kv_cache_config
                )
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            # ### PATCH START: attesting worker
            self.connector_worker = AFDMooncakeConnectorWorker(
                vllm_config, str(self.engine_id), kv_cache_config
            )
            # ### PATCH END: attesting worker

    def bind_dspark_model(self, consumer: DsparkPdPrefixConsumer) -> None:
        if not isinstance(self.connector_worker, AFDMooncakeConnectorWorker):
            raise RuntimeError(
                "DSpark model binding requires the actual worker connector"
            )
        self.connector_worker.bind_dspark_model(consumer)

    def get_dspark_prefill_sources(
        self, metadata: KVConnectorMetadata | None
    ) -> tuple[DsparkPdSourceRequest, ...]:
        if not isinstance(self.connector_worker, AFDMooncakeConnectorWorker):
            raise RuntimeError("DSpark source consumption requires the actual worker")
        if self.connector_worker.kv_role != "kv_producer" or metadata is None:
            return ()
        if not isinstance(metadata, DsparkMooncakeConnectorMetadata):
            raise ValueError("DSpark producer requires its typed allocation metadata")
        cancelled = set(metadata.dspark_cancelled)
        return tuple(
            source
            for source in metadata.dspark_sources
            if (source.request_id, source.source_generation) not in cancelled
        )

    def publish_dspark_ready(self, ready_batch: tuple[DsparkPdReady, ...]) -> None:
        if not isinstance(self.connector_worker, AFDMooncakeConnectorWorker):
            raise RuntimeError(
                "DSpark publication requires the actual worker connector"
            )
        self.connector_worker.publish_dspark_ready(ready_batch)

    # Native MooncakeConnector.shutdown(self), 7e37e29a, is a no-op. Native
    # Ascend Worker.shutdown calls ensure_kv_transfer_shutdown before releasing
    # runner/KV resources. Keep target-only behavior, but DSpark cannot release
    # storage until real read/DONE frontiers drain. Signature is exact (no ann).
    # Remove when native owns request leases and cooperative receiver retirement.
    def shutdown(self):
        # ### PATCH START: retire DSpark leases before native storage release
        worker = self.connector_worker
        if worker is not None and worker._dspark_enabled:
            if worker.kv_role == "kv_producer":
                worker._dspark_registry.close()
            else:
                worker.retire_receiver(timeout=DSPARK_SHUTDOWN_TIMEOUT)
        # ### PATCH END: retire DSpark leases before native storage release
        return super().shutdown()


class AFDMooncakeConnectorWorker(MooncakeConnectorWorker):
    # Upstream: MooncakeConnectorWorker.start_load_kv @ 7e37e29a:2065-2158.
    # Reason/functionality: serialize the entire native multi-pull batch with
    # receiver retirement, so shutdown cannot admit half a native request.
    # Delegation exception: retain native topology/pull/tracker construction;
    # the 94-line body is unchanged and only this admission scope is added.
    # Signature matches upstream, including its omitted return annotation.
    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        # ### PATCH START: atomic consumer batch admission
        if self._dspark_enabled:
            if not isinstance(metadata, DsparkMooncakeConnectorMetadata):
                raise ValueError("DSpark worker requires typed allocation metadata")
            if self._dspark_consumer is None:
                raise RuntimeError("DSpark loaded model is not bound")
            if self.kv_role == "kv_producer":
                self._dspark_registry.admit(metadata.dspark_sources)
                for request_id, generation in metadata.dspark_cancelled:
                    self._dspark_registry.cancel(request_id, generation)
        if self.kv_role == "kv_consumer":
            receiver = self.kv_recv_thread
            if not isinstance(receiver, AFDMooncakeRecvingThread):
                raise RuntimeError("attested receiver is not registered")
            destinations = (
                metadata.dspark_destinations if self._dspark_enabled else None
            )
            with receiver.admission_scope(destinations=destinations):
                return super().start_load_kv(metadata)
        # ### PATCH END: atomic consumer batch admission
        return super().start_load_kv(metadata)

    def retire_receiver(self, *, timeout: float) -> None:
        """Retire this consumer, retaining its engine and fixed registration.

        ``timeout`` bounds admission/drain/join before executor shutdown. Python
        executor shutdown has no timeout; the calling process supervisor must
        provide the final bound. This never closes/unregisters the engine.
        """
        if self.kv_role != "kv_consumer" or not isinstance(
            self.kv_recv_thread, AFDMooncakeRecvingThread
        ):
            raise RuntimeError("retirement requires a registered attested consumer")
        self.kv_recv_thread.retire(timeout=timeout)

    # Upstream: MooncakeConnectorWorker.__init__ @ 7e37e29a:1750-1862.
    # Reason/functionality: reject unsupported configuration before TE setup.
    # Delegation exception: keep the 113-line native engine/topology constructor
    # intact; the extra behavior is only validation and instance registration state.
    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig
    ):
        # ### PATCH START: constrained external connector
        if not is_deepseek_v41_model(vllm_config):
            raise ValueError("attested PD requires the exact DeepSeek V4.1 model")
        validate_deepseek_v41_pd(vllm_config)
        if global_te.transfer_engine is not None:
            raise RuntimeError("attested PD requires a fresh native engine owner")
        # ### PATCH END: constrained external connector
        super().__init__(vllm_config, engine_id, kv_cache_config)
        # ### PATCH START: single registration lifetime
        self._registration_started = False
        self._registered_layout: KVLayoutSnapshot | None = None
        self._dspark_enabled = uses_deepseek_v41_dspark_pd(vllm_config)
        self._dspark_consumer: DsparkPdPrefixConsumer | None = None
        self._dspark_registry = DsparkPdReadyRegistry()
        # ### PATCH END: single registration lifetime

    def bind_dspark_model(self, consumer: DsparkPdPrefixConsumer) -> None:
        if not self._dspark_enabled or self._registration_started:
            raise RuntimeError(
                "DSpark binding requires pre-registration DSpark configuration"
            )
        if self._dspark_consumer is not None:
            raise RuntimeError("DSpark model binding is single-use")
        self._dspark_consumer = consumer

    def publish_dspark_ready(self, ready_batch: tuple[DsparkPdReady, ...]) -> None:
        if self.kv_role != "kv_producer" or self._dspark_consumer is None:
            if ready_batch:
                raise RuntimeError("only the bound producer can publish DSpark ready")
            return
        assert self._registered_layout is not None
        expected_layout = layout_digest(self._registered_layout)
        for ready in ready_batch:
            if (
                ready.source_engine_id != self.engine_id
                or ready.layout_digest != expected_layout
                or ready.source_tp_rank != self.tp_rank
                or ready.tp_size != self.tp_size
            ):
                raise ValueError(
                    "DSpark ready does not belong to this registered worker"
                )
        self._dspark_registry.publish(ready_batch)

    # Upstream: MooncakeConnectorWorker.register_kv_caches @ 7e37e29a:1886-2037.
    # Reason/functionality: validate actual views before memory registration,
    # publish required proof, then start the original sender or attesting receiver.
    # Native returns no annotation; signature intentionally preserves that contract.

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""
        # ### PATCH START: validate materialized layout before TE registration
        if self._registration_started:
            raise RuntimeError("attested KV registration is single-use")
        if global_te.is_register_buffer:
            raise RuntimeError(
                "the native transfer engine already registered KV storage"
            )
        self._registration_started = True
        self._registered_layout = collect_deepseek_v41_kv_layout(
            self.kv_cache_config, kv_caches
        )
        draft_names = ()
        if self._dspark_enabled:
            if self._dspark_consumer is None:
                raise RuntimeError("DSpark model must bind before KV registration")
            draft_names = self._dspark_consumer.binding.draft_layer_names
        _validate_layout_role(self._registered_layout, draft_layer_names=draft_names)
        if not self.use_hybrid or self.use_mamba or not self.use_compress:
            raise ValueError(
                "attested V4.1 PD requires the native compressed hybrid path"
            )
        # ### PATCH END: validate materialized layout before TE registration
        self.use_mla = self.vllm_config.model_config.is_deepseek_mla
        self.use_sparse = hasattr(
            self.vllm_config.model_config.hf_text_config, "index_topk"
        )

        self.num_blocks = self.kv_cache_config.num_blocks
        logger.info("num_blocks: %s", self.num_blocks)
        self.kv_caches = kv_caches
        self.kv_caches_base_addr = []
        self.block_len_per_addr: list[int] = []
        self.block_stride_per_addr: list[int] = []
        self.addr_group_idx: list[list[int]] = []
        ptrs = []
        lengths = []
        if not self.use_hybrid:
            for layer_name, kv_cache_tuple in kv_caches.items():  # noqa: B007
                if isinstance(kv_cache_tuple, (list, tuple)) is False:
                    kv_cache_tuple = [kv_cache_tuple]
                for single_kv_cache in kv_cache_tuple:
                    tensor_num_blocks = single_kv_cache.shape[0]
                    block_size_scale = tensor_num_blocks // self.num_blocks
                    block_shape = single_kv_cache.shape[1:]
                    self.block_len_per_addr.append(
                        single_kv_cache.element_size()
                        * math.prod(block_shape)
                        * block_size_scale
                    )
                    self.kv_caches_base_addr.append(single_kv_cache.data_ptr())
                    ptrs.append(single_kv_cache.data_ptr())
                    lengths.append(
                        single_kv_cache.element_size()
                        * math.prod(single_kv_cache.shape)
                    )
        elif self.use_mamba:
            for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
                for layer_name in get_kv_cache_tensor_layers(kv_cache_tensor):
                    kv_cache_tuple = kv_caches[layer_name]
                    if isinstance(kv_cache_tuple, (list, tuple)) is False:
                        kv_cache_tuple = [kv_cache_tuple]
                    for single_kv_cache in kv_cache_tuple:
                        if single_kv_cache.data_ptr() in self.kv_caches_base_addr:
                            continue
                        tensor_num_blocks = single_kv_cache.shape[0]
                        block_size_scale = tensor_num_blocks // self.num_blocks
                        block_shape = single_kv_cache.shape[1:]
                        self.block_len_per_addr.append(
                            single_kv_cache.element_size()
                            * math.prod(block_shape)
                            * block_size_scale
                        )
                        self.kv_caches_base_addr.append(single_kv_cache.data_ptr())
            self.block_stride_per_addr.extend(self.block_len_per_addr)
        elif self.use_compress:
            layer_group_idx = dict[str, int]()
            for i, group in enumerate(self.kv_cache_config.kv_cache_groups):
                for layer_name in group.layer_names:
                    layer_group_idx[layer_name] = i
            # Rebuild physical pages from the descriptor sharing relation and
            # runtime views. One entry is one complete padded page, so inner
            # views such as indexer K and scale must not become independent
            # full-page transfers.
            for shared_page in _reconstruct_shared_pages(
                self.kv_cache_config, kv_caches
            ):
                page_base_addr: int | None = None
                page_groups: set[int] = set()
                shared_by = [layer_name for layer_name, _ in shared_page.placements]
                for layer_name, placement_offset in shared_page.placements:
                    page_groups.add(layer_group_idx[layer_name])
                    layer_tensors = as_kv_cache_tensors(kv_caches[layer_name])
                    if not layer_tensors:
                        raise ValueError(
                            "DeepSeek-V4 shared KV cache layer has no materialized "
                            f"tensor: {layer_name}."
                        )

                    layer_page_base = (
                        min(tensor.data_ptr() for tensor in layer_tensors)
                        - placement_offset
                    )
                    if page_base_addr is not None and layer_page_base != page_base_addr:
                        raise ValueError(
                            "DeepSeek-V4 layers with the same placement and block "
                            "stride "
                            "do not resolve to the same page base: "
                            f"layers={shared_by}, layer={layer_name}, "
                            f"placement_offset={placement_offset}."
                        )
                    page_base_addr = layer_page_base
                assert page_base_addr is not None
                self.kv_caches_base_addr.append(page_base_addr)
                self.addr_group_idx.append(sorted(page_groups))
                self.block_stride_per_addr.append(shared_page.block_stride)
                self.block_len_per_addr.append(shared_page.block_stride)
        else:
            raise TypeError(
                "Mooncake connector does not support this type kv_cache now."
            )

        if self.use_hybrid:
            register_regions = collect_configured_register_regions(
                self.kv_cache_config, kv_caches
            )
            validate_register_region_count(register_regions)
            ptrs = register_regions.ptrs
            lengths = register_regions.lengths

        # After KV Caches registered, start the sending or receiving thread.
        # ### PATCH START: required layout metadata
        _validate_local_registration(self, self._registered_layout, ptrs, lengths)
        layout_bytes = self._registered_layout.to_json().encode("utf-8")
        if len(layout_bytes) > MAX_LAYOUT_BYTES:
            raise ValueError("layout metadata exceeds the byte limit")
        metadata = AttestedMooncakeAgentMetadata(
            afd_kv_layout=layout_bytes,
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            block_size=self.block_size,
            kv_caches_base_addr=self.kv_caches_base_addr,
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_addr,
            ssm_sizes=self._mamba_ssm_size,
            local_ip=get_ip(),
            runtime_id=self.mooncake_runtime_id,
            afd_dspark_version=(DSPARK_PROTOCOL_VERSION if self._dspark_enabled else 0),
            afd_tp_rank=self.tp_rank if self._dspark_enabled else 0,
            afd_tp_size=self.tp_size if self._dspark_enabled else 1,
        )
        if len(msgspec.msgpack.encode(metadata)) > MAX_METADATA_BYTES:
            raise ValueError("Mooncake metadata exceeds the byte limit")
        if self._dspark_consumer is not None:
            self._dspark_consumer.bind_registered_layout(
                self._registered_layout, self.kv_cache_config, self.kv_caches
            )
        global_te.register_buffer(ptrs, lengths)
        self.xfer_handshake_metadata = metadata
        # ### PATCH END: required layout metadata

        ready_event = threading.Event()
        if self.kv_role == "kv_producer":
            sender_type = (
                AFDMooncakeSendingThread
                if self._dspark_enabled
                else KVCacheSendingThread
            )
            self.kv_send_thread = sender_type(
                self.vllm_config,
                self.tp_rank,
                self._prefill_tp_size,
                self.engine_id,
                self.side_channel_host,
                self.side_channel_port,
                metadata,
                ready_event,
                self.kv_caches,
                self.pcp_rank,
            )
            if self._dspark_enabled:
                self.kv_send_thread.bind_dspark_registry(self._dspark_registry)
            self.kv_send_thread.start()
        else:
            # ### PATCH START: receiver owns the same proof
            self.kv_recv_thread = AFDMooncakeRecvingThread(
                self.tp_rank,
                self.tp_size,
                self._prefill_pp_size,
                self.engine,
                self.engine_id,
                self.handshake_port,
                self.side_channel_port,
                self.kv_caches_base_addr,
                self.block_len_per_addr,
                self.block_stride_per_addr,
                self.addr_group_idx,
                self._mamba_ssm_size,
                self.use_hybrid,
                self.use_mamba,
                self.hma_group_size,
                ready_event,
                self.vllm_config,
                self.kv_cache_config,
                self.kv_caches,
                self._prefill_pp_layer_partition,
                self.mooncake_runtime_id,
                local_layout=self._registered_layout,
                draft_layer_names=draft_names,
            )
            # ### PATCH END: receiver owns the same proof
            self.kv_recv_thread.start()

        start_wait_time = time.time()
        thread = (
            self.kv_send_thread
            if self.kv_role == "kv_producer"
            else self.kv_recv_thread
        )
        assert thread is not None
        while not ready_event.is_set():
            if not thread.is_alive():
                raise RuntimeError("KV Cache sending/receiving thread failed to start.")
            if time.time() - start_wait_time > 5 * 60:
                raise RuntimeError("Timeout waiting for KV Cache thread to be ready.")
            time.sleep(3)


class AFDMooncakeSendingThread(KVCacheSendingThread):
    """Native socket owner, with request-specific DSpark ready/read leases."""

    def bind_dspark_registry(self, registry: DsparkPdReadyRegistry) -> None:
        if self.ident is not None:
            raise RuntimeError("DSpark registry must bind before sender thread start")
        self.dspark_registry = registry

    # Native 7e37e29a:324-330. Retired DSpark generations become native finished
    # requests only after their lease is acknowledged or a no-read cancellation.
    def get_and_clear_finished_requests(self) -> set[str]:
        # ### PATCH START: completed generation ownership
        for request_id in self.dspark_registry.take_finished():
            self.task_tracker.add_not_transfer_request(request_id)
        # ### PATCH END: completed generation ownership
        return self.task_tracker.get_and_clear_finished_requests()

    # Native 7e37e29a:332-333. An unused native P replica cannot remain open to
    # a future GETREADY after native block release. Close its exact generation.
    def add_not_transfer_request(self, request_id: str):
        # ### PATCH START: close an unused DSpark source
        if self.dspark_registry.is_dspark(request_id):
            self.dspark_registry.cancel_current(request_id)
            return
        # ### PATCH END: close an unused DSpark source
        self.task_tracker.add_not_transfer_request(request_id)

    # Native 7e37e29a:335-336. Native expiry force-frees blocks without proving
    # read completion; DSpark generations retain ownership until matching DONE.
    def add_delayed_request(self, request_id: str, delay_start_time: float):
        # ### PATCH START: timeout is not a read drain
        if self.dspark_registry.is_dspark(request_id):
            return
        # ### PATCH END: timeout is not a read drain
        return self.task_tracker.add_delayed_request(request_id, delay_start_time)

    # Native 7e37e29a:368-451, exact signature. Keep the native ROUTER owner and
    # GETMETA loop. Replace legacy DONE with strict lease-DONE; its handlers
    # touch host records only. Remove when native exposes request-ready hooks.
    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
        size_in_bytes = len(encoded_data)
        logger.debug(
            "Size of encoded MooncakeAgentMetadata: %s bytes", str(size_in_bytes)
        )

        decoder = msgspec.msgpack.Decoder(type=tuple)
        while True:
            try:
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    logger.error(
                        "Invalid message format in KVCacheSendingThread. "
                        "Expected: at least 2 frames (identity + payload). "
                        "Actual: %d frames. "
                        "Frames: %s. "
                        "Check: Verify message sender implementation.",
                        len(frames),
                        frames,
                    )
                    continue

                identity = frames[0]
                payload = [f for f in frames[1:] if f != b""]
                if len(payload) != 1:
                    logger.error(
                        "Invalid message format in KVCacheSendingThread. "
                        "Expected: exactly 1 payload frame. "
                        "Actual: %d payload frames. "
                        "Frames: %s. "
                        "Check: Verify message sender removes empty frames correctly.",
                        len(payload),
                        frames,
                    )
                    continue

                # ### PATCH START: bounded request-ready protocol
                if len(payload[0]) > MAX_METADATA_BYTES:
                    sock.send_multipart(
                        (
                            identity,
                            b"",
                            encoder.encode({"ok": False, "error": "message too large"}),
                        )
                    )
                    continue
                # ### PATCH END: bounded request-ready protocol
                msg = decoder.decode(payload[0])
                if msg[0] == GET_META_MSG:
                    sock.send_multipart((identity, b"", encoded_data))
                # ### PATCH START: request-specific acquire/complete
                elif msg[0] in (GET_DSPARK_READY_MSG, DONE_DSPARK_RECV_MSG):
                    try:
                        if len(msg) != 2:
                            raise ValueError("invalid DSpark request message")
                        if msg[0] == GET_DSPARK_READY_MSG:
                            lease = self.dspark_registry.acquire(
                                read_query_from_payload(msg[1])
                            )
                            reply = encoder.encode(
                                {"ok": True, "lease": read_lease_payload(lease)}
                            )
                        else:
                            self.dspark_registry.complete(
                                read_lease_from_payload(msg[1])
                            )
                            reply = b"ACK"
                    except (ValueError, RuntimeError, KeyError, TypeError) as error:
                        reply = encoder.encode(
                            {"ok": False, "error": str(error)[:MAX_DSPARK_ERROR_LENGTH]}
                        )
                    sock.send_multipart((identity, b"", reply))
                # ### PATCH END: request-specific acquire/complete
                elif msg[0] == DONE_RECVING_MSG:
                    logger.debug("Got DONE_RECVING_MSG for request %s", msg[1])
                    # ### PATCH START: old DONE cannot release a new generation
                    # This sender is selected only for DSpark PD. Even an old
                    # evicted tombstone must never enter native reqid-only DONE.
                    sock.send_multipart(
                        (
                            identity,
                            b"",
                            encoder.encode(
                                {
                                    "ok": False,
                                    "error": "DSpark requires lease-aware DONE",
                                }
                            ),
                        )
                    )
                    continue
                    # ### PATCH END: old DONE cannot release a new generation
                else:
                    logger.error(
                        "Connection listener received unexpected message type. "
                        "Expected: GET_META_MSG or DONE_RECVING_MSG. "
                        "Actual: %s. "
                        "Full message: %s. "
                        "Check: Verify message protocol implementation.",
                        msg[0] if msg else "empty",
                        msg,
                    )
            except Exception as e:
                logger.error(
                    "Connection listener encountered exception during "
                    "message processing. "
                    "Exception type: %s. "
                    "Error: %s. "
                    "Context: Processing frames from socket. "
                    "Check: Review message handling logic and socket state.",
                    type(e).__name__,
                    e,
                )


class AFDMooncakeRecvingThread(KVCacheRecvingThread):
    # Upstream: KVCacheRecvingThread.__init__ @ 7e37e29a:455-559.
    # Reason/functionality: bind one immutable local registration proof and its
    # strict decoder before native thread startup. local_layout is a required
    # plugin-only keyword supplied by the worker's pre-registration collector.
    # Delegation exception: retain the 105-line native queue/executor/device setup.
    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        _prefill_pp_size: int,
        engine: TransferEngine,
        local_engine_id: str,
        local_handshake_port: int,
        side_channel_port: int,
        local_kv_caches_base_addr: list[int],
        block_len_per_addr: list[int],
        block_stride_per_addr: list[int],
        addr_group_idx: list[list[int]],
        mamba_ssm_size: tuple[int, int],
        use_hybrid,
        has_mamba,
        hma_group_size,
        ready_event: threading.Event,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        kv_caches: dict[str, Any],
        prefill_pp_layer_partition: str | None = None,
        mooncake_runtime_id: str = "",
        *,
        local_layout: KVLayoutSnapshot,
        draft_layer_names: tuple[str, ...] = (),
    ):
        # ### PATCH START: required registration proof before construction
        _validate_layout_role(local_layout, draft_layer_names=draft_layer_names)
        if not use_hybrid or has_mamba or _prefill_pp_size != 1:
            raise ValueError("attested PD requires the native V4.1 hybrid PP1 receiver")
        _validate_native_arrays(
            local_layout, local_kv_caches_base_addr, block_len_per_addr
        )
        expected_groups = tuple(
            tuple(
                i
                for i, group in enumerate(local_layout.groups)
                if any(layer in group.layer_names for layer in page.layers)
            )
            for page in local_layout.pages
        )
        if (
            kv_cache_config.num_blocks != local_layout.num_blocks
            or tuple(block_stride_per_addr)
            != tuple(p.block_stride_bytes for p in local_layout.pages)
            or tuple(tuple(g) for g in addr_group_idx) != expected_groups
            or hma_group_size != len(local_layout.groups)
            or mamba_ssm_size != (0, 0)
        ):
            raise ValueError(
                "native receiver arrays disagree with its registration proof"
            )
        self._local_layout = local_layout
        self._local_addresses = tuple(local_kv_caches_base_addr)
        self._local_block_lengths = tuple(block_len_per_addr)
        self._local_block_strides = tuple(block_stride_per_addr)
        self._local_address_groups = tuple(tuple(g) for g in addr_group_idx)
        self._local_block_size = vllm_config.cache_config.block_size
        self._draft_layer_names = draft_layer_names
        self._attested_peers: OrderedDict[tuple[str, int], _AttestedPeer] = (
            OrderedDict()
        )
        # ### PATCH END: required registration proof before construction
        super().__init__(
            tp_rank,
            tp_size,
            _prefill_pp_size,
            engine,
            local_engine_id,
            local_handshake_port,
            side_channel_port,
            local_kv_caches_base_addr,
            block_len_per_addr,
            block_stride_per_addr,
            addr_group_idx,
            mamba_ssm_size,
            use_hybrid,
            has_mamba,
            hma_group_size,
            ready_event,
            vllm_config,
            kv_cache_config,
            kv_caches,
            prefill_pp_layer_partition,
            mooncake_runtime_id,
        )
        # ### PATCH START: strict decoder and instance retirement owner
        self.decoder = msgspec.msgpack.Decoder(AttestedMooncakeAgentMetadata)
        self._admission_lock = threading.RLock()
        self._retire_lock = threading.Lock()
        self._retire_condition = threading.Condition(self._admission_lock)
        self._accepted_requests: dict[str, int] = {}
        self._retirement_failed = threading.Event()
        self._admission_closed = False
        self._retired = False
        self._batch_destinations: dict[str, DsparkPdDestination] | None = None
        self._retained_dspark_attempts: list[_DsparkReadAttempt] = []
        # ### PATCH END: strict decoder and instance retirement owner

    @property
    def retired(self) -> bool:
        """Only true after native ACKs, receiver join and executor shutdown."""
        with self._retire_condition:
            return self._retired

    @contextmanager
    def admission_scope(
        self, *, destinations: dict[str, DsparkPdDestination] | None = None
    ) -> Iterator[None]:
        """One native batch and retirement share this instance-owned lock."""
        with self._retire_condition:
            if self._admission_closed or self._retirement_failed.is_set():
                raise RuntimeError("attested receiver admission is closed")
            previous = self._batch_destinations
            if destinations is not None:
                self._batch_destinations = destinations
            try:
                yield
            except BaseException:
                self._retirement_failed.set()
                self._retire_condition.notify_all()
                raise
            finally:
                self._batch_destinations = previous

    # Upstream: KVCacheRecvingThread.add_request @ 7e37e29a:561-593.
    # Reason/functionality: count every admitted native task before its queue
    # publication and prevent new tasks once retirement closes admission.
    # Delegation exception: preserve the complete native rank/port dictionary;
    # the wrapper owns admission only, not a second request representation.
    # Signature matches upstream, including its omitted return annotation.
    def add_request(
        self,
        request_id: str,
        remote_request_id: str,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        remote_engine_id: str,
        remote_host: str,
        remote_handshake_port: int,
        offset: int,
        tp_num_need_pulls: int,
        remote_port_send_num: dict[int, RemotePortInfo] | None = None,
        all_task_done: bool = False,
    ):
        # ### PATCH START: exact task admission before native publication
        with self.admission_scope():
            self._accepted_requests[request_id] = (
                self._accepted_requests.get(request_id, 0) + 1
            )
            try:
                # ### PATCH START: immutable per-queue DSpark allocation
                if self._batch_destinations is not None:
                    if tp_num_need_pulls != 1 or offset != 0 or not all_task_done:
                        raise ValueError("DSpark PD requires one same-TP source peer")
                    destination = self._batch_destinations[request_id]
                    if (
                        destination.source.request_id != remote_request_id
                        or destination.block_ids
                        != tuple(tuple(ids) for ids in local_block_ids)
                        or destination.source.transfer_block_ids
                        != tuple(tuple(ids) for ids in remote_block_ids)
                    ):
                        raise ValueError(
                            "native queue mapping changed its DSpark allocation"
                        )
                    self.request_queue.put(
                        {
                            "request_id": request_id,
                            "local_block_ids": local_block_ids,
                            "remote_block_ids": remote_block_ids,
                            "remote_engine_id": remote_engine_id,
                            "remote_request_id": remote_request_id,
                            "remote_host": remote_host,
                            "remote_handshake_port": remote_handshake_port,
                            "offset": offset,
                            "tp_num_need_pulls": tp_num_need_pulls,
                            "remote_port_send_num": remote_port_send_num or {},
                            "all_task_done": all_task_done,
                            "afd_dspark_attempt": _DsparkReadAttempt(destination),
                        }
                    )
                    return
                # ### PATCH END: immutable per-queue DSpark allocation
                return super().add_request(
                    request_id,
                    remote_request_id,
                    local_block_ids,
                    remote_block_ids,
                    remote_engine_id,
                    remote_host,
                    remote_handshake_port,
                    offset,
                    tp_num_need_pulls,
                    remote_port_send_num,
                    all_task_done,
                )
            except BaseException:
                self._retirement_failed.set()
                self._retire_condition.notify_all()
                raise
        # ### PATCH END: exact task admission before native publication

    # Upstream: KVCacheRecvingThread.run @ 7e37e29a:623-635.
    # Reason/functionality: a private local retirement sentinel exits the input
    # thread after admission/drain; native None and normal requests are unchanged.
    # Signature matches upstream. Remove when native owns cooperative retirement.
    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                # ### PATCH START: local receiver retirement
                if request_data is _ReceiverControl.RETIRE:
                    self.request_queue.task_done()
                    return
                # ### PATCH END: local receiver retirement
                if request_data is None:
                    logger.warning("Received a None request. ")
                    self.request_queue.task_done()
                    continue
                self._submit_request(request_data)
            except Exception as e:
                # ### PATCH START: failed dispatch cannot retire successfully
                self._retirement_failed.set()
                with self._retire_condition:
                    self._retire_condition.notify_all()
                # ### PATCH END: failed dispatch cannot retire successfully
                logger.error("Error in KVCacheTransferThread. error=%s. ", e)

    # Upstream: KVCacheRecvingThread._handle_request @ 7e37e29a:706-745.
    # Reason/functionality: native task_done precedes DONE/ACK; retain acceptance
    # through the complete finalization, with a sticky failure on any exception.
    # Delegation exception: preserve native failed-transfer retirement and both
    # DONE routes as one owner; only the post-finalization counter is new.
    # Signature matches upstream, including its omitted return annotation.
    def _handle_request(self, req_meta: dict[str, Any]):
        # ### PATCH START: task remains live through final native ACK
        request_id = req_meta["request_id"]
        with self._retire_condition:
            if request_id not in self._accepted_requests:
                # Native component callers may directly execute one queued task.
                if self._admission_closed:
                    raise RuntimeError("unadmitted task after receiver retirement")
                self._accepted_requests[request_id] = 1
        try:
            # ### PATCH START: exact queued allocation owns DSpark finalization
            attempt = req_meta.get("afd_dspark_attempt")
            if attempt is None:
                result = super()._handle_request(req_meta)
            else:
                result = self._handle_dspark_request(req_meta, attempt)
            # ### PATCH END: exact queued allocation owns DSpark finalization
        except BaseException:
            if req_meta.get("afd_dspark_attempt") is not None:
                self._retained_dspark_attempts.append(req_meta["afd_dspark_attempt"])
            self._retirement_failed.set()
            with self._retire_condition:
                self._retire_condition.notify_all()
            raise
        with self._retire_condition:
            count = self._accepted_requests[request_id] - 1
            if count:
                self._accepted_requests[request_id] = count
            else:
                del self._accepted_requests[request_id]
            self._retire_condition.notify_all()
        return result
        # ### PATCH END: task remains live through final native ACK

    def _handle_dspark_request(
        self, req_meta: dict[str, Any], attempt: _DsparkReadAttempt
    ) -> None:
        """Native 706-745 lifecycle with stricter read-drain/DONE ownership.

        The native handler swallows read errors and publishes completion in its
        finally. DSpark cannot release either allocation if DMA drain is unknown.
        Keep its queue/task frontiers then. ACK successful/no-read leases before
        publishing local finished-receive. This helper adds no network abort.
        """
        # ### PATCH START: lease/read-drain ownership replaces native finally
        request_id = req_meta["request_id"]
        remote_request_id = req_meta["remote_request_id"]
        all_task_done = req_meta["all_task_done"]
        transfer_failed = self._is_failed_recv_request(request_id)
        try:
            if transfer_failed:
                self._mark_failed_recv_request(request_id, req_meta["local_block_ids"])
            else:
                try:
                    self._transfer_kv_cache_all_groups(req_meta)
                except Exception:
                    if attempt.read_started and not attempt.read_drained:
                        raise
                    self._mark_failed_recv_request(
                        request_id, req_meta["local_block_ids"]
                    )
        finally:
            if not attempt.read_started or attempt.read_drained:
                if attempt.lease is not None:
                    self._finish_dspark_read(req_meta, attempt.lease)
                if self._mark_request_task_done(request_id, all_task_done):
                    self.task_tracker.update_done_task_count(request_id)
                    with self.proc_not_transfer_request_lock:
                        self.proc_not_transfer_request.pop(remote_request_id, None)
                    self._clear_failed_recv_request(request_id)
                self.request_queue.task_done()
        # ### PATCH END: lease/read-drain ownership replaces native finally

    def _finish_dspark_read(
        self, req_meta: dict[str, Any], lease: DsparkPdReadLease
    ) -> None:
        host, port = req_meta["remote_host"], req_meta["remote_handshake_port"]
        sock = self._get_remote_socket(host, port)
        try:
            ensure_zmq_send(
                sock,
                self.encoder.encode((DONE_DSPARK_RECV_MSG, read_lease_payload(lease))),
                f"{host}:{port}",
            )
            if ensure_zmq_recv(sock, f"{host}:{port}") != b"ACK":
                raise RuntimeError("DSpark read completion was not acknowledged")
        except BaseException:
            sock.close()
            raise
        else:
            self._return_remote_socket(sock, host, port)

    # Upstream: KVCacheRecvingThread._send_done_recv_signal @ 7e37e29a:1199-1232.
    # Reason/functionality: a missing/invalid ACK must prevent successful retire;
    # preserve native framing/socket cleanup but make its swallowed error sticky
    # and observable to the native per-peer exception handler.
    # Signature matches upstream, including its omitted return annotation.
    def _send_done_recv_signal(
        self,
        request_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_port_send_num: dict[int, RemotePortInfo],
    ):
        logger.debug(
            "Sending done recving signal for request %s to %s:%d",
            request_id,
            remote_host,
            remote_handshake_port,
        )
        sock: zmq.Socket | None = None
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            data_bytes = self.encoder.encode(
                (DONE_RECVING_MSG, request_id, remote_port_send_num)
            )
            ensure_zmq_send(sock, data_bytes, f"{remote_host}:{remote_handshake_port}")
            resp = ensure_zmq_recv(sock, f"{remote_host}:{remote_handshake_port}")
            logger.debug(
                "Received response for request %s: %s", request_id, resp.decode("utf-8")
            )
            if resp != b"ACK":
                logger.error(
                    "Failed to receive ACK for request. request_id=%s, source=%s:%d. ",
                    request_id,
                    remote_host,
                    remote_handshake_port,
                )
                raise RuntimeError(
                    f"Failed to receive ACK, resp: {resp.decode('utf-8')}"
                )
        except RuntimeError as e:
            if isinstance(sock, zmq.Socket):
                sock.close()
                sock = None
                logger.warning("Unexpected error occurred in socket. error=%s. ", e)
            # ### PATCH START: ACK failure forbids successful retirement
            self._retirement_failed.set()
            raise
            # ### PATCH END: ACK failure forbids successful retirement
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)
                logger.debug(
                    "Returned socket to pool for %s:%d",
                    remote_host,
                    remote_handshake_port,
                )

    def retire(self, *, timeout: float) -> None:
        """Close admission, drain real ACKs and join this consumer's input thread.

        One deadline bounds admission, accepted tasks and receiver join. Native
        executor shutdown has no timeout and is entered only after those tasks
        have completed; a process supervisor must bound that final stdlib join.
        No memory is unregistered and no native engine/global owner is reset.
        """
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("retirement timeout must be finite and positive")
        if threading.current_thread() is self or self.ident is None:
            raise RuntimeError("retirement requires a started external receiver")
        deadline = time.monotonic() + timeout
        # A second caller must not poison an in-progress retirement's state.
        if not self._retire_lock.acquire(blocking=False):
            raise RuntimeError("receiver retirement is already in progress")
        try:
            self._retire_before_deadline(deadline)
        finally:
            self._retire_lock.release()

    def _retire_before_deadline(self, deadline: float) -> None:
        if not self._admission_lock.acquire(
            timeout=max(0, deadline - time.monotonic())
        ):
            # Admission is still owned by a live native batch. Do not poison
            # its remaining pulls before this caller can close admission.
            raise TimeoutError("receiver batch admission did not retire")
        try:
            if self._retired:
                return
            if self._retirement_failed.is_set():
                raise RuntimeError("receiver retirement has a retained failure")
            if self._admission_closed:
                raise RuntimeError("receiver retirement is already in progress")
            self._admission_closed = True
            drained = self._retire_condition.wait_for(
                lambda: not self._accepted_requests or self._retirement_failed.is_set(),
                timeout=max(0, deadline - time.monotonic()),
            )
            if not drained:
                raise TimeoutError("accepted native requests did not finish DONE/ACK")
            if self._retirement_failed.is_set():
                raise RuntimeError("native DONE/ACK or dispatch failed")
            with _before_retirement_deadline(
                self.task_tracker.done_task_lock, deadline
            ):
                if self.task_tracker.reqs_to_process:
                    raise RuntimeError(
                        "native request retirement frontier is incomplete"
                    )
            with _before_retirement_deadline(self.request_task_counts_lock, deadline):
                if self.request_task_counts or self.finished_request_markers:
                    raise RuntimeError("native pull retirement frontier is incomplete")
            self.request_queue.put(_ReceiverControl.RETIRE)
        except BaseException:
            self._retirement_failed.set()
            raise
        finally:
            self._admission_lock.release()
        try:
            self.join(timeout=max(0, deadline - time.monotonic()))
            if self.is_alive():
                raise TimeoutError("native receiver input thread did not exit")
            with _before_retirement_deadline(
                self.request_queue.all_tasks_done, deadline
            ):
                if self.request_queue.unfinished_tasks:
                    raise RuntimeError("native input queue has unfinished work")
            self.executor.shutdown(wait=True)
            with _before_retirement_deadline(self.peer_request_queues_lock, deadline):
                if self.peer_request_queues or self.active_peer_request_handlers:
                    raise RuntimeError("native peer handlers did not retire")
            with self._retire_condition:
                self._retired = True
        except BaseException:
            self._retirement_failed.set()
            raise

    def _validated_request(
        self, request: dict[str, Any]
    ) -> tuple[dict[str, Any], _AttestedPeer | None]:
        groups = len(self._local_layout.groups)
        remote = tuple(tuple(ids) for ids in request["remote_block_ids"])
        local = tuple(tuple(ids) for ids in request["local_block_ids"])
        if len(remote) != groups or len(local) != groups:
            raise ValueError("block IDs must cover every native transferable group")
        for ids in remote + local:
            if any(type(i) is not int or i < 0 for i in ids):
                raise ValueError("block IDs must be nonnegative integers")
        # A true empty native request reads no address and needs no peer lookup.
        if not any(remote) and not any(local):
            if request.get("afd_dspark_attempt") is not None:
                raise ValueError("DSpark read requires materialized draft blocks")
            return {
                **request,
                "remote_block_ids": remote,
                "local_block_ids": local,
            }, None
        key = (request["remote_engine_id"], request["remote_handshake_port"])
        with self.remote_metadata_lock:
            peer = self._attested_peers.get(key)
            if peer is not None:
                addresses = self.kv_caches_base_addr.get(key[0], {}).get(key[1])
                host = self.remote_te_host.get(key[0], {}).get(key[1])
                port = self.remote_te_port.get(key[0], {}).get(key[1])
                if addresses is None or host is None or port is None:
                    self._retire_peer(key)
                    peer = None
                elif (
                    tuple(addresses) != peer.addresses
                    or host != peer.host
                    or port != peer.port
                ):
                    self._retire_peer(key)
                    raise RuntimeError(
                        "native peer metadata changed outside its attestation"
                    )
        if peer is None:
            self._get_remote_metadata(key[0], request["remote_host"], key[1])
            with self.remote_metadata_lock:
                peer = self._attested_peers[key]
        if any(i >= peer.layout.num_blocks for ids in remote for i in ids):
            raise ValueError("source block ID is out of bounds")
        effective = []
        for src, dst in zip(remote, local, strict=True):
            if len(src) < len(dst):
                raise ValueError("source block IDs do not cover the local destination")
            effective.append(src[-len(dst) :] if dst else ())
        effective = tuple(effective)
        validate_kv_transfer_blocks(peer.layout, self._local_layout, effective, local)
        # ### PATCH START: request-ready lease before the actual read
        attempt = request.get("afd_dspark_attempt")
        if attempt is not None:
            if not isinstance(attempt, _DsparkReadAttempt):
                raise ValueError("invalid queued DSpark allocation")
            source = attempt.destination.source
            if (
                peer.dspark_version != DSPARK_PROTOCOL_VERSION
                or peer.tp_size != self.tp_size
                or peer.tp_rank != self.tp_rank
                or source.source_engine_id != request["remote_engine_id"]
                or source.transfer_block_ids != remote
                or attempt.destination.block_ids != local
            ):
                raise ValueError("DSpark source/allocation does not match its peer")
            if any(
                not remote[index] or not local[index]
                for index, group in enumerate(self._local_layout.groups)
                if set(group.layer_names) & set(self._draft_layer_names)
            ):
                raise ValueError("DSpark read requires every loaded draft group")
            ready = DsparkPdReady(
                source.source_engine_id,
                source.request_id,
                source.source_generation,
                source.original_prompt_tokens,
                source.computed_prefix_tokens,
                source.prompt_digest,
                layout_digest(peer.layout),
                source.draft_schema_digest,
                source.transfer_block_ids,
                peer.tp_rank,
                peer.tp_size,
            )
            query = DsparkPdReadQuery(
                ready,
                self.local_engine_id,
                attempt.destination.generation,
                local,
                f"{attempt.destination.generation}:{request['remote_handshake_port']}",
            )
            attempt.lease = self._acquire_dspark_read(request, query)
        # ### PATCH END: request-ready lease before the actual read
        return {
            **request,
            "remote_block_ids": effective,
            "local_block_ids": local,
        }, peer

    def _acquire_dspark_read(
        self, request: dict[str, Any], query: DsparkPdReadQuery
    ) -> DsparkPdReadLease:
        host, port = request["remote_host"], request["remote_handshake_port"]
        sock = self._get_remote_socket(host, port)
        try:
            ensure_zmq_send(
                sock,
                self.encoder.encode((GET_DSPARK_READY_MSG, read_query_payload(query))),
                f"{host}:{port}",
            )
            encoded = ensure_zmq_recv(sock, f"{host}:{port}")
            if len(encoded) > MAX_METADATA_BYTES:
                raise ValueError("DSpark ready response exceeds byte limit")
            response = msgspec.msgpack.decode(encoded)
            if (
                not isinstance(response, dict)
                or set(response) != {"ok", "lease"}
                or response["ok"] is not True
            ):
                raise RuntimeError("DSpark source did not grant the exact ready lease")
            lease = read_lease_from_payload(response["lease"])
            if (
                lease.ready != query.expected_ready
                or lease.query_id != query.query_id
                or lease.destination_engine_id != query.destination_engine_id
                or lease.destination_generation != query.destination_generation
                or lease.destination_block_ids != query.destination_block_ids
            ):
                raise ValueError("DSpark ready reply belongs to another allocation")
        except BaseException:
            sock.close()
            raise
        else:
            self._return_remote_socket(sock, host, port)
            return lease

    def _retire_peer(self, key: tuple[str, int]) -> None:
        """Retire every mirror while the caller holds remote_metadata_lock."""
        self._attested_peers.pop(key, None)
        for mirror in (
            self.kv_caches_base_addr,
            self.remote_te_host,
            self.remote_te_port,
        ):
            if key[0] in mirror:
                mirror[key[0]].pop(key[1], None)

    # Upstream: KVCacheRecvingThread._get_remote_metadata @ 7e37e29a:1157-1197.
    # Reason/functionality: require compatible typed layout and exact native array
    # association before atomically publishing addresses; preserve socket cleanup.
    # Signature exactly matches upstream.

    def _get_remote_metadata(
        self,
        expected_engine_id: str,
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
        """Get the metadata from the remote host."""
        if not remote_host or not remote_host.strip():
            raise RuntimeError("Mooncake remote handshake host is empty.")
        if remote_handshake_port <= 0 or remote_handshake_port > 65535:
            raise RuntimeError(
                f"Mooncake remote handshake port is invalid: {remote_handshake_port}."
            )

        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(
                sock,
                self.encoder.encode((GET_META_MSG, "")),
                f"{remote_host}:{remote_handshake_port}",
            )
            metadata_bytes = ensure_zmq_recv(
                sock, f"{remote_host}:{remote_handshake_port}"
            )
            # ### PATCH START: bounded required attestation
            if len(metadata_bytes) > MAX_METADATA_BYTES:
                raise ValueError("Mooncake metadata exceeds the byte limit")
            agent_meta = self.decoder.decode(metadata_bytes)
            layout = _decode_attestation(
                agent_meta,
                self._local_layout,
                self._local_block_size,
                draft_layer_names=self._draft_layer_names,
            )
            # ### PATCH END: bounded required attestation
            engine_id = agent_meta.engine_id
            assert engine_id != self.local_engine_id, (
                f"Conflict engine id {engine_id} with local engine id "
                f"{self.local_engine_id}."
            )
            if engine_id != expected_engine_id:
                raise RuntimeError(
                    "Mooncake remote metadata engine ID mismatch: expected "
                    f"{expected_engine_id}, received {engine_id}."
                )
            self._validate_remote_runtime(agent_meta)
            transfer_host, transfer_port = self._resolve_remote_transfer_endpoint(
                agent_meta, remote_host
            )
            # ### PATCH START: publish one immutable peer bundle
            peer = _AttestedPeer(
                layout,
                tuple(agent_meta.kv_caches_base_addr),
                transfer_host,
                transfer_port,
                agent_meta.afd_dspark_version,
                agent_meta.afd_tp_rank,
                agent_meta.afd_tp_size,
            )
            with self.remote_metadata_lock:
                key = (engine_id, remote_handshake_port)
                previous = self._attested_peers.get(key)
                if previous is not None and previous != peer:
                    raise RuntimeError(
                        "attested peer storage changed during its registration lifetime"
                    )
                self._attested_peers[key] = peer
                while len(self._attested_peers) > MAX_ATTESTED_PEERS:
                    self._retire_peer(next(iter(self._attested_peers)))
                self.kv_caches_base_addr[engine_id][remote_handshake_port] = (
                    agent_meta.kv_caches_base_addr
                )
                self.remote_te_host[engine_id][remote_handshake_port] = transfer_host
                self.remote_te_port[engine_id][remote_handshake_port] = transfer_port
            # ### PATCH END: publish one immutable peer bundle
        except Exception:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
            raise
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)
                logger.debug(
                    "Returned socket to pool for %s:%d",
                    remote_host,
                    remote_handshake_port,
                )

    # Upstream: KVCacheRecvingThread._transfer_kv_cache_all_groups @ 7e37e29a:764-884.
    # Reason/functionality: validate all effective block mappings before the one
    # read, and consume the same immutable peer bundle despite concurrent lookups.
    # Native coalescing, error/DONE lifecycle and read semantics stay native.
    # Signature exactly matches upstream; copies avoid a mutable engine proxy.

    def _transfer_kv_cache_all_groups(self, req_meta: dict[str, Any]):
        """Handle a KV cache transfer request."""
        # ### PATCH START: validate the whole immutable request before reads
        req_meta, peer = self._validated_request(req_meta)
        # ### PATCH END: validate the whole immutable request before reads
        remote_request_id = req_meta["remote_request_id"]
        remote_block_ids = req_meta["remote_block_ids"]
        local_block_ids = req_meta["local_block_ids"]
        remote_engine_id = req_meta["remote_engine_id"]  # noqa: F841
        remote_host = req_meta["remote_host"]  # noqa: F841
        remote_handshake_port = req_meta["remote_handshake_port"]
        # offset = req_meta["offset"]
        # tp_num_need_pulls = req_meta["tp_num_need_pulls"]

        # Full prefix cache hit: do not need to read remote blocks, just notify
        # P worker that we have the blocks we need.
        num_local_blocks = sum(
            len(group_block_ids) for group_block_ids in local_block_ids
        )
        if num_local_blocks == 0:
            return

        # ### PATCH START: consume the same proof-bound addresses
        assert peer is not None
        remote_kv_caches_base_addrs = peer.addresses
        local_kv_caches_base_addrs = self._local_addresses
        remote_transfer_host = peer.host
        remote_transfer_port = peer.port
        # ### PATCH END: consume the same proof-bound addresses
        session_id = f"{remote_transfer_host}:{remote_transfer_port}"

        # With P-side PP the remote worker only owns its stage's
        # kv_cache_tensors. Offset the local (D) group-major flat addr list to
        # the remote stage's global group range before zipping, so stage-1
        # data lands in D's groups 8-15 instead of overwriting groups 0-7.
        local_addrs = local_kv_caches_base_addrs
        # ### PATCH START: consume the same proof-bound local page arrays
        block_len_arr = self._local_block_lengths
        block_stride_arr = self._local_block_strides
        addr_group_arr = self._local_address_groups
        # ### PATCH END: consume the same proof-bound local page arrays
        if self._prefill_pp_size > 1:
            tp_num_need_pulls = req_meta["tp_num_need_pulls"]
            remote_pp_rank = req_meta["offset"] // tp_num_need_pulls
            num_groups = len(self.kv_cache_config.kv_cache_tensors)
            groups_per_stage = num_groups // self._prefill_pp_size
            addrs_per_group, rem = divmod(len(local_addrs), num_groups)
            if rem != 0:
                raise ValueError("non-uniform per-group address counts unsupported")
            if num_groups % self._prefill_pp_size != 0:
                raise ValueError("pp split not group-aligned")
            start = remote_pp_rank * groups_per_stage * addrs_per_group
            end = (remote_pp_rank + 1) * groups_per_stage * addrs_per_group
            local_addrs = local_addrs[start:end]
            block_len_arr = self.block_len_per_addr[start:end]
            block_stride_arr = self.block_stride_per_addr[start:end]
            if addr_group_arr:
                addr_group_arr = self.addr_group_idx[start:end]
            logger.debug(
                "Cross-PP pull alignment: remote_port=%d -> pp_rank=%d "
                "local_addrs[%d:%d]=%d remote=%d",
                remote_handshake_port,
                remote_pp_rank,
                start,
                end,
                len(local_addrs),
                len(remote_kv_caches_base_addrs),
            )

        req_start_time = time.perf_counter()
        src_list, dst_list, length_list = [], [], []
        # ### PATCH START: validated logical group count
        for i in range(len(self._local_layout.groups)):
            # ### PATCH END: validated logical group count
            if not remote_block_ids[i] or not local_block_ids[i]:
                continue
            cur_remote_block_ids = remote_block_ids[i]
            cur_local_block_ids = local_block_ids[i]
            if not isinstance(self.kv_cache_specs[i], MambaSpec) and len(
                cur_local_block_ids
            ) < len(cur_remote_block_ids):
                cur_remote_block_ids = cur_remote_block_ids[-len(cur_local_block_ids) :]
            grouped_remote_block_ids, grouped_local_block_ids = (
                group_concurrent_contiguous(cur_remote_block_ids, cur_local_block_ids)
            )
            for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
                zip(local_addrs, remote_kv_caches_base_addrs)  # noqa: B905
            ):
                if addr_group_arr and i not in addr_group_arr[k]:
                    continue
                block_len = block_len_arr[k]
                block_stride = block_stride_arr[k]
                for remote_block_id, local_block_id in zip(  # noqa: B905
                    grouped_remote_block_ids, grouped_local_block_ids
                ):
                    src = src_layer_base_addr + local_block_id[0] * block_stride
                    dst = dst_layer_base_addr + remote_block_id[0] * block_stride
                    length = block_len * len(local_block_id)
                    src_list.append(src)
                    dst_list.append(dst)
                    length_list.append(length)

        # ### PATCH START: negative/exceptional reads do not prove DMA drain
        attempt = req_meta.get("afd_dspark_attempt")
        if attempt is not None:
            attempt.read_started = True
        # ### PATCH END: negative/exceptional reads do not prove DMA drain
        ret = self.engine.batch_transfer_sync_read(
            session_id, src_list, dst_list, length_list
        )
        if ret < 0:
            logger.error(
                "Mooncake transfer failed for request. remote_request_id=%s, "
                "session_id=%s, ret=%d.",
                req_meta["remote_request_id"],
                session_id,
                ret,
            )
            raise RuntimeError(
                f"Mooncake transfer failed for session {session_id}, ret: {ret}"
            )

        req_end_time = time.perf_counter()
        # ### PATCH START: only the successful synchronous result retires data
        if attempt is not None:
            attempt.read_drained = True
        # ### PATCH END: only the successful synchronous result retires data
        req_transfer_elapsed = (req_end_time - req_start_time) * 1000
        logger.info(
            "KV cache transfer for request %s took %.2f ms. local_ip %s "
            "local_device_id %s remote_session_id %s",
            remote_request_id,
            req_transfer_elapsed,
            get_ip(),
            self.tp_rank,
            session_id,
        )
