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
from _thread import LockType
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import msgspec
import torch
import zmq
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import KVCacheConfig
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
    validate_deepseek_v41_pd,
)

if TYPE_CHECKING:
    from mooncake.engine import TransferEngine

MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_ATTESTED_PEERS = 16_000
MAX_TRANSFER_ADDRESS = (1 << 64) - 1


class AttestedMooncakeAgentMetadata(MooncakeAgentMetadata, kw_only=True):
    """A required extension; legacy peers never silently downgrade."""

    afd_kv_layout: bytes


@dataclass(frozen=True, slots=True)
class _AttestedPeer:
    layout: KVLayoutSnapshot
    addresses: tuple[int, ...]
    host: str
    port: int


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


def _validate_layout_role(layout: KVLayoutSnapshot) -> None:
    if layout.device_type != "npu":
        raise ValueError("attested V4.1 PD requires materialized NPU storage")
    if any(
        g.is_draft or g.host_resident or not g.enable_transfer for g in layout.groups
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
) -> KVLayoutSnapshot:
    layout = layout_from_json(metadata.afd_kv_layout.decode("utf-8"))
    _validate_layout_role(layout)
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
                MooncakeConnectorScheduler(
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


class AFDMooncakeConnectorWorker(MooncakeConnectorWorker):
    # Upstream: MooncakeConnectorWorker.start_load_kv @ 7e37e29a:2065-2158.
    # Reason/functionality: serialize the entire native multi-pull batch with
    # receiver retirement, so shutdown cannot admit half a native request.
    # Delegation exception: retain native topology/pull/tracker construction;
    # the 94-line body is unchanged and only this admission scope is added.
    # Signature matches upstream, including its omitted return annotation.
    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        # ### PATCH START: atomic consumer batch admission
        if self.kv_role == "kv_consumer":
            receiver = self.kv_recv_thread
            if not isinstance(receiver, AFDMooncakeRecvingThread):
                raise RuntimeError("attested receiver is not registered")
            with receiver.admission_scope():
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
        # ### PATCH END: single registration lifetime

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
        _validate_layout_role(self._registered_layout)
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
        )
        if len(msgspec.msgpack.encode(metadata)) > MAX_METADATA_BYTES:
            raise ValueError("Mooncake metadata exceeds the byte limit")
        global_te.register_buffer(ptrs, lengths)
        self.xfer_handshake_metadata = metadata
        # ### PATCH END: required layout metadata

        ready_event = threading.Event()
        if self.kv_role == "kv_producer":
            self.kv_send_thread = KVCacheSendingThread(
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
    ):
        # ### PATCH START: required registration proof before construction
        _validate_layout_role(local_layout)
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
        # ### PATCH END: strict decoder and instance retirement owner

    @property
    def retired(self) -> bool:
        """Only true after native ACKs, receiver join and executor shutdown."""
        with self._retire_condition:
            return self._retired

    @contextmanager
    def admission_scope(self) -> Iterator[None]:
        """One native batch and retirement share this instance-owned lock."""
        with self._retire_condition:
            if self._admission_closed or self._retirement_failed.is_set():
                raise RuntimeError("attested receiver admission is closed")
            try:
                yield
            except BaseException:
                self._retirement_failed.set()
                self._retire_condition.notify_all()
                raise

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
            result = super()._handle_request(req_meta)
        except BaseException:
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
        return {
            **request,
            "remote_block_ids": effective,
            "local_block_ids": local,
        }, peer

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
                agent_meta, self._local_layout, self._local_block_size
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
