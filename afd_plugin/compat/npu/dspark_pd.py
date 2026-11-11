# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Loaded DSpark prompt projection and immutable request-ready PD records.

This component owns no socket, block allocator or transfer-engine operation.
The connector owns read leases and storage release. Projection uses the pinned
Ascend lower context writer, including its real model and stream fence; the
MLA-only upper initialize_local_context entry is deliberately not substituted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from enum import Enum
from fractions import Fraction
from typing import TypeAlias

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.model_runner import ExecuteModelState
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.dspark_context import (
    DSparkContextChunk,
    DSparkContextDescriptor,
    initialize_draft_context_chunk,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator

from afd_plugin.compat.npu.kv_layout import JsonValue, KVLayoutSnapshot
from afd_plugin.compat.npu.kv_layout_native import (
    KVCacheValue,
    collect_deepseek_v41_kv_layout,
)
from afd_plugin.compat.npu.runtime_config import (
    uses_deepseek_v41_dspark_pd,
    validate_deepseek_v41_pd,
)

MAX_PD_TEXT = 2_048
MAX_PD_GROUPS = 16_384
MAX_PD_BLOCKS = 262_144
MAX_PD_INTEGER = (1 << 63) - 1
MAX_PD_ERROR = 1_024
NULL_BLOCK_ID = 0
SpecScalar: TypeAlias = str | int | bool | None | Enum | Fraction | torch.dtype


def _integer(value: int, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= MAX_PD_INTEGER:
        raise ValueError(f"{name} must be a bounded integer >= {minimum}")


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not 0 < len(value) <= MAX_PD_TEXT:
        raise ValueError(f"{name} must be bounded nonempty text")


def _digest(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != hashlib.sha256().digest_size * 2
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def _blocks(
    value: tuple[tuple[int, ...], ...], name: str, *, allow_null: bool = False
) -> None:
    if (
        not isinstance(value, tuple)
        or not 0 < len(value) <= MAX_PD_GROUPS
        or any(not isinstance(group, tuple) for group in value)
        or sum(map(len, value)) > MAX_PD_BLOCKS
    ):
        raise ValueError(f"{name} must be bounded immutable group block arrays")
    for group in value:
        for block in group:
            _integer(block, name, minimum=int(not allow_null))
        nonnull = tuple(block for block in group if block != NULL_BLOCK_ID)
        if len(nonnull) != len(set(nonnull)):
            raise ValueError(f"{name} contains repeated blocks in one group")


def _sha(payload: JsonValue) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prompt_digest(token_ids: tuple[int, ...]) -> str:
    if not isinstance(token_ids, tuple) or not 0 < len(token_ids) <= MAX_PD_BLOCKS:
        raise ValueError("original prompt must be a bounded nonempty token tuple")
    for token in token_ids:
        _integer(token, "prompt token")
    return _sha(list(token_ids))


def layout_digest(layout: KVLayoutSnapshot) -> str:
    return hashlib.sha256(layout.to_json().encode("utf-8")).hexdigest()


def _spec_scalar(value: SpecScalar) -> str:
    if isinstance(value, Enum):
        return f"enum:{type(value).__module__}.{type(value).__qualname__}:{value.name}"
    if isinstance(value, torch.dtype):
        return f"dtype:{value}"
    if isinstance(value, Fraction):
        return f"fraction:{value.numerator}/{value.denominator}"
    if value is None or type(value) in (str, int, bool):
        return f"{type(value).__name__}:{value}"
    raise ValueError("unsupported native DSpark KV schema field")


def draft_schema_digest(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> str:
    """Config/spec agreement without capacity, pointers or checkpoint paths."""
    validate_deepseek_v41_pd(vllm_config)
    if not uses_deepseek_v41_dspark_pd(vllm_config):
        raise ValueError("DSpark schema requires the exact plugin PD protocol")
    target = vllm_config.model_config.hf_text_config
    draft = vllm_config.speculative_config.draft_model_config.hf_text_config
    groups = []
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        specs = []
        for layer in group.layer_names:
            spec = (
                group.kv_cache_spec.kv_cache_specs[layer]
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else group.kv_cache_spec
            )
            values = vars(spec)
            specs.append(
                [
                    layer,
                    f"{type(spec).__module__}.{type(spec).__qualname__}",
                    [
                        [field.name, _spec_scalar(values[field.name])]
                        for field in fields(spec)
                    ],
                ]
            )
        groups.append(
            [
                group_id,
                group.is_eagle_group,
                group.host_resident,
                group.enable_kv_transfer,
                group.role.value,
                specs,
            ]
        )
    return _sha(
        {
            "version": 1,
            "hidden_size": target.hidden_size,
            "vocab_size": target.vocab_size,
            "target_layers": target.num_hidden_layers,
            "aux_layers": list(target.dspark_target_layer_ids),
            "block_size": target.dspark_block_size,
            "draft_layers": draft.n_mtp_layers,
            "draft_experts": draft.n_routed_experts,
            "draft_topk": draft.num_experts_per_tok,
            "steps": vllm_config.speculative_config.num_speculative_tokens,
            "tp_size": vllm_config.parallel_config.tensor_parallel_size,
            "groups": groups,
        }
    )


@dataclass(frozen=True, slots=True)
class DsparkPdSourceRequest:
    request_id: str
    source_generation: str
    original_prompt_tokens: int
    computed_prefix_tokens: int
    prompt_digest: str
    allocation_block_ids: tuple[tuple[int, ...], ...]
    transfer_block_ids: tuple[tuple[int, ...], ...]
    source_engine_id: str
    draft_schema_digest: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.request_id, "request ID"),
            (self.source_generation, "source generation"),
            (self.source_engine_id, "source engine"),
        ):
            _text(value, name)
        _integer(self.original_prompt_tokens, "original prompt", minimum=1)
        _integer(self.computed_prefix_tokens, "computed prefix", minimum=1)
        if self.computed_prefix_tokens > self.original_prompt_tokens:
            raise ValueError("computed prefix exceeds original prompt")
        _digest(self.prompt_digest, "prompt digest")
        _digest(self.draft_schema_digest, "draft schema digest")
        _blocks(self.allocation_block_ids, "source allocation", allow_null=True)
        _blocks(self.transfer_block_ids, "source transfer")
        if len(self.allocation_block_ids) != len(self.transfer_block_ids):
            raise ValueError("source transfer must preserve every original group")
        for allocated, transfer in zip(
            self.allocation_block_ids, self.transfer_block_ids, strict=True
        ):
            exported = set(transfer)
            if tuple(block for block in allocated if block in exported) != transfer:
                raise ValueError("source transfer must retain allocation block order")


@dataclass(frozen=True, slots=True)
class DsparkPdReady:
    source_engine_id: str
    source_request_id: str
    source_generation: str
    original_prompt_tokens: int
    computed_prefix_tokens: int
    prompt_digest: str
    layout_digest: str
    draft_schema_digest: str
    transfer_block_ids: tuple[tuple[int, ...], ...]
    source_tp_rank: int
    tp_size: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_engine_id, "source engine"),
            (self.source_request_id, "source request"),
            (self.source_generation, "source generation"),
        ):
            _text(value, name)
        _integer(self.original_prompt_tokens, "original prompt", minimum=1)
        _integer(self.computed_prefix_tokens, "computed prefix", minimum=1)
        if self.computed_prefix_tokens > self.original_prompt_tokens:
            raise ValueError("computed prefix exceeds original prompt")
        for value, name in (
            (self.prompt_digest, "prompt digest"),
            (self.layout_digest, "layout digest"),
            (self.draft_schema_digest, "draft schema digest"),
        ):
            _digest(value, name)
        _blocks(self.transfer_block_ids, "ready transfer")
        _integer(self.tp_size, "TP size", minimum=1)
        _integer(self.source_tp_rank, "source TP rank")
        if self.source_tp_rank >= self.tp_size:
            raise ValueError("source TP rank exceeds TP size")


@dataclass(frozen=True, slots=True)
class DsparkPdReadQuery:
    expected_ready: DsparkPdReady
    destination_engine_id: str
    destination_generation: str
    destination_block_ids: tuple[tuple[int, ...], ...]
    query_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.expected_ready, DsparkPdReady):
            raise ValueError("read query requires typed Ready")
        for value, name in (
            (self.destination_engine_id, "destination engine"),
            (self.destination_generation, "destination generation"),
            (self.query_id, "query ID"),
        ):
            _text(value, name)
        _blocks(self.destination_block_ids, "destination blocks")
        if tuple(map(len, self.destination_block_ids)) != tuple(
            map(len, self.expected_ready.transfer_block_ids)
        ):
            raise ValueError(
                "destination blocks must match every source transfer group"
            )


@dataclass(frozen=True, slots=True)
class DsparkPdReadLease:
    ready: DsparkPdReady
    query_id: str
    destination_engine_id: str
    destination_generation: str
    destination_block_ids: tuple[tuple[int, ...], ...]
    lease_id: str

    def __post_init__(self) -> None:
        DsparkPdReadQuery(
            self.ready,
            self.destination_engine_id,
            self.destination_generation,
            self.destination_block_ids,
            self.query_id,
        )
        _text(self.lease_id, "lease ID")


def _groups_payload(blocks: tuple[tuple[int, ...], ...]) -> list[JsonValue]:
    return [list(group) for group in blocks]


def source_request_payload(source: DsparkPdSourceRequest) -> dict[str, JsonValue]:
    return {
        "request_id": source.request_id,
        "source_generation": source.source_generation,
        "original_prompt_tokens": source.original_prompt_tokens,
        "computed_prefix_tokens": source.computed_prefix_tokens,
        "prompt_digest": source.prompt_digest,
        "allocation_block_ids": _groups_payload(source.allocation_block_ids),
        "transfer_block_ids": _groups_payload(source.transfer_block_ids),
        "source_engine_id": source.source_engine_id,
        "draft_schema_digest": source.draft_schema_digest,
    }


def ready_payload(ready: DsparkPdReady) -> dict[str, JsonValue]:
    return {
        "source_engine_id": ready.source_engine_id,
        "source_request_id": ready.source_request_id,
        "source_generation": ready.source_generation,
        "original_prompt_tokens": ready.original_prompt_tokens,
        "computed_prefix_tokens": ready.computed_prefix_tokens,
        "prompt_digest": ready.prompt_digest,
        "layout_digest": ready.layout_digest,
        "draft_schema_digest": ready.draft_schema_digest,
        "transfer_block_ids": _groups_payload(ready.transfer_block_ids),
        "source_tp_rank": ready.source_tp_rank,
        "tp_size": ready.tp_size,
    }


def read_query_payload(query: DsparkPdReadQuery) -> dict[str, JsonValue]:
    return {
        "expected_ready": ready_payload(query.expected_ready),
        "destination_engine_id": query.destination_engine_id,
        "destination_generation": query.destination_generation,
        "destination_block_ids": _groups_payload(query.destination_block_ids),
        "query_id": query.query_id,
    }


def read_lease_payload(lease: DsparkPdReadLease) -> dict[str, JsonValue]:
    return {
        "ready": ready_payload(lease.ready),
        "query_id": lease.query_id,
        "destination_engine_id": lease.destination_engine_id,
        "destination_generation": lease.destination_generation,
        "destination_block_ids": _groups_payload(lease.destination_block_ids),
        "lease_id": lease.lease_id,
    }


def _record(payload: JsonValue, names: tuple[str, ...]) -> dict[str, JsonValue]:
    if not isinstance(payload, dict) or set(payload) != set(names):
        raise ValueError("DSpark PD record has missing or unexpected fields")
    return payload


def _read_groups(value: JsonValue) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or any(
        not isinstance(group, list) for group in value
    ):
        raise ValueError("DSpark PD block arrays must be lists")
    return tuple(tuple(group) for group in value)


def source_request_from_payload(payload: JsonValue) -> DsparkPdSourceRequest:
    value = _record(
        payload, tuple(field.name for field in fields(DsparkPdSourceRequest))
    )
    return DsparkPdSourceRequest(
        value["request_id"],
        value["source_generation"],
        value["original_prompt_tokens"],
        value["computed_prefix_tokens"],
        value["prompt_digest"],
        _read_groups(value["allocation_block_ids"]),
        _read_groups(value["transfer_block_ids"]),
        value["source_engine_id"],
        value["draft_schema_digest"],
    )


def ready_from_payload(payload: JsonValue) -> DsparkPdReady:
    value = _record(payload, tuple(field.name for field in fields(DsparkPdReady)))
    return DsparkPdReady(
        value["source_engine_id"],
        value["source_request_id"],
        value["source_generation"],
        value["original_prompt_tokens"],
        value["computed_prefix_tokens"],
        value["prompt_digest"],
        value["layout_digest"],
        value["draft_schema_digest"],
        _read_groups(value["transfer_block_ids"]),
        value["source_tp_rank"],
        value["tp_size"],
    )


def read_query_from_payload(payload: JsonValue) -> DsparkPdReadQuery:
    value = _record(payload, tuple(field.name for field in fields(DsparkPdReadQuery)))
    return DsparkPdReadQuery(
        ready_from_payload(value["expected_ready"]),
        value["destination_engine_id"],
        value["destination_generation"],
        _read_groups(value["destination_block_ids"]),
        value["query_id"],
    )


def read_lease_from_payload(payload: JsonValue) -> DsparkPdReadLease:
    value = _record(payload, tuple(field.name for field in fields(DsparkPdReadLease)))
    return DsparkPdReadLease(
        ready_from_payload(value["ready"]),
        value["query_id"],
        value["destination_engine_id"],
        value["destination_generation"],
        _read_groups(value["destination_block_ids"]),
        value["lease_id"],
    )


@dataclass(frozen=True, slots=True)
class DsparkPdModelBinding:
    target_model: torch.nn.Module
    draft_model: torch.nn.Module
    draft_layer_names: tuple[str, ...]
    aux_layer_ids: tuple[int, ...]
    hidden_size: int
    device: torch.device


@dataclass(frozen=True, slots=True)
class DsparkPdProjectedPrefix:
    source: DsparkPdSourceRequest
    projected_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, DsparkPdSourceRequest):
            raise ValueError("projected prefix requires a typed source declaration")
        _integer(self.projected_tokens, "projected prefix", minimum=1)
        if self.projected_tokens > self.source.computed_prefix_tokens:
            raise ValueError("projected prefix exceeds required source prefix")


@dataclass(frozen=True, slots=True)
class DsparkPdWriteReport:
    source_tp_rank: int
    tp_size: int
    prefixes: tuple[DsparkPdProjectedPrefix, ...]
    ready: tuple[DsparkPdReady, ...]
    error: str | None = None

    def __post_init__(self) -> None:
        _integer(self.tp_size, "report TP size", minimum=1)
        _integer(self.source_tp_rank, "report TP rank")
        if self.source_tp_rank >= self.tp_size:
            raise ValueError("report rank exceeds TP size")
        for records, record_type in (
            (self.prefixes, DsparkPdProjectedPrefix),
            (self.ready, DsparkPdReady),
        ):
            if (
                not isinstance(records, tuple)
                or len(records) > MAX_PD_BLOCKS
                or any(not isinstance(record, record_type) for record in records)
            ):
                raise ValueError("projection report requires bounded typed records")
        if self.error is not None and (
            not isinstance(self.error, str) or not 0 < len(self.error) <= MAX_PD_ERROR
        ):
            raise ValueError("projection error must be bounded nonempty text")


@dataclass(frozen=True, slots=True)
class _BoundDraftView:
    name: str
    tensor: torch.Tensor
    address: int
    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]
    strides: tuple[int, ...]


class DsparkPdPrefixConsumer:
    """One loaded/storage generation, with no allocator or transfer ownership."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        speculator: AscendDSparkSpeculator,
        *,
        target_model: torch.nn.Module,
        device: torch.device,
        source_engine_id: str,
        tp_rank: int,
        tp_size: int,
        agree_batch: Callable[[DsparkPdWriteReport], tuple[DsparkPdWriteReport, ...]],
    ) -> None:
        validate_deepseek_v41_pd(vllm_config)
        if not uses_deepseek_v41_dspark_pd(vllm_config):
            raise ValueError(
                "loaded DSpark prefix requires the exact plugin PD connector"
            )
        _text(source_engine_id, "source engine")
        _integer(tp_size, "TP size", minimum=1)
        _integer(tp_rank, "TP rank")
        if (
            tp_rank >= tp_size
            or tp_size != vllm_config.parallel_config.tensor_parallel_size
        ):
            raise ValueError("loaded DSpark prefix TP does not match config")
        names = tuple(speculator.model.get_draft_kv_cache_layer_names())
        if (
            not names
            or len(set(names)) != len(names)
            or set(names) != speculator.draft_attn_layer_names
        ):
            raise ValueError("loaded draft layer names do not match the speculator")
        if speculator.device != device:
            raise ValueError("loaded draft device does not match prefix consumer")
        self.vllm_config = vllm_config
        self.speculator = speculator
        self.source_engine_id = source_engine_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.agree_batch = agree_batch
        self._binding = DsparkPdModelBinding(
            target_model,
            speculator.model,
            names,
            tuple(vllm_config.model_config.hf_text_config.dspark_target_layer_ids),
            vllm_config.model_config.hf_text_config.hidden_size,
            device,
        )
        self._layout: KVLayoutSnapshot | None = None
        self._kv_config: KVCacheConfig | None = None
        self._caches: dict[str, KVCacheValue] = {}
        self._progress: dict[str, DsparkPdProjectedPrefix] = {}
        self._draft_views: tuple[_BoundDraftView, ...] = ()
        self._group_ids: tuple[int, ...] = ()
        self._layer_group_ids: tuple[int, ...] = ()
        self._block_sizes: dict[int, int] = {}
        self._draft_windows: dict[int, tuple[int, int]] = {}
        self._schema_digest: str | None = None
        self._layout_digest: str | None = None
        self._terminal = False

    @property
    def binding(self) -> DsparkPdModelBinding:
        return self._binding

    @property
    def terminal(self) -> bool:
        return self._terminal

    def fail(self) -> None:
        self._terminal = True

    def bind_registered_layout(
        self,
        layout: KVLayoutSnapshot,
        kv_cache_config: KVCacheConfig,
        kv_caches: Mapping[str, KVCacheValue],
    ) -> None:
        if self._terminal or self._layout is not None:
            raise RuntimeError("DSpark prefix storage cannot be rebound")
        if collect_deepseek_v41_kv_layout(kv_cache_config, kv_caches) != layout:
            raise ValueError("DSpark prefix layout differs from materialized storage")
        groups, layer_groups, sizes = self.speculator.get_draft_context_group_layout()
        # V4.1's native make_cache_groups does not set is_eagle_group. Actual
        # loaded names plus native speculator mapping define draft ownership.
        expected_groups = tuple(
            index
            for index, group in enumerate(kv_cache_config.kv_cache_groups)
            if set(group.layer_names) & set(self.binding.draft_layer_names)
        )
        if groups != expected_groups or len(layer_groups) != len(
            self.binding.draft_layer_names
        ):
            raise ValueError("loaded DSpark draft groups differ from registered groups")
        for name, group_id in zip(
            self.binding.draft_layer_names, layer_groups, strict=True
        ):
            group = kv_cache_config.kv_cache_groups[group_id]
            if (
                name not in group.layer_names
                or sizes[group_id] != group.kv_cache_spec.block_size
            ):
                raise ValueError("loaded DSpark layer/group/block mapping differs")
        registered_names = tuple(
            name
            for index in groups
            for name in kv_cache_config.kv_cache_groups[index].layer_names
        )
        if set(registered_names) != set(self.binding.draft_layer_names):
            raise ValueError(
                "registered draft groups must contain exactly the loaded draft"
            )
        if draft_schema_digest(
            self.vllm_config, self.speculator.kv_cache_config
        ) != draft_schema_digest(self.vllm_config, kv_cache_config):
            raise ValueError("speculator and registered native KV schemas differ")
        self._layout = layout
        self._kv_config = kv_cache_config
        self._caches = dict(kv_caches)
        self._group_ids, self._layer_group_ids = groups, layer_groups
        self._block_sizes = dict(sizes)
        for group_id in groups:
            group = kv_cache_config.kv_cache_groups[group_id]
            spec = group.kv_cache_spec
            native_spec = (
                spec.kv_cache_specs[group.layer_names[0]]
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else spec
            )
            if not isinstance(native_spec, SlidingWindowSpec):
                self.fail()
                raise ValueError("V4.1 draft prompt writer requires native SWA groups")
            self._draft_windows[group_id] = (
                native_spec.sliding_window,
                native_spec.extra_retained_tokens,
            )
        self._schema_digest = draft_schema_digest(self.vllm_config, kv_cache_config)
        self._layout_digest = layout_digest(layout)
        views = []
        for name in self.binding.draft_layer_names:
            tensor = self._caches[name]
            if not isinstance(tensor, torch.Tensor):
                self.fail()
                raise ValueError("loaded V4.1 draft requires one SWA cache view")
            views.append(
                _BoundDraftView(
                    name,
                    tensor,
                    tensor.data_ptr(),
                    tensor.device,
                    tensor.dtype,
                    tuple(tensor.shape),
                    tensor.stride(),
                )
            )
        self._draft_views = tuple(views)
        try:
            self._require_storage()
        except Exception:
            self.fail()
            raise

    def _require_storage(self) -> None:
        if self._layout is None or self._kv_config is None:
            raise RuntimeError("DSpark prefix storage is not registered")
        if self.speculator.model is not self.binding.draft_model:
            raise ValueError("loaded draft ownership changed after binding")
        # Full target layout/schema checks belong to registration. Admission
        # forbids reload/sleep/KV reinit; each chunk only checks the actual
        # draft writers' retained views, without reserializing every target page.
        context = self.vllm_config.compilation_config.static_forward_context
        for view in self._draft_views:
            actual = context[view.name].kv_cache[0]
            if (
                actual is not view.tensor
                or actual.device != view.device
                or actual.data_ptr() != view.address
                or tuple(actual.shape) != view.shape
                or actual.stride() != view.strides
                or actual.dtype != view.dtype
                or actual.device != self.binding.device
            ):
                raise ValueError("loaded draft cache is not the registered KV view")

    def reap_finished(self, finished_req_ids: set[str]) -> None:
        for request_id in finished_req_ids:
            self._progress.pop(request_id, None)

    @torch.inference_mode()
    def consume_prefill(
        self,
        scheduler_output: SchedulerOutput,
        execute_state: ExecuteModelState | None,
        sources: tuple[DsparkPdSourceRequest, ...],
        *,
        source_error: Exception | None = None,
    ) -> tuple[DsparkPdReady, ...]:
        if self._terminal:
            raise RuntimeError("DSpark prefix consumer is terminal")
        prefixes: list[DsparkPdProjectedPrefix] = []
        ready: list[DsparkPdReady] = []
        error: Exception | None = None
        try:
            if source_error is not None:
                raise source_error
            self._require_storage()
            self.reap_finished(scheduler_output.finished_req_ids)
            self._project_batch(execute_state, sources, prefixes, ready)
        except Exception as exc:
            error = exc
        report = DsparkPdWriteReport(
            self.tp_rank,
            self.tp_size,
            tuple(prefixes),
            tuple(ready),
            None
            if error is None
            else f"{type(error).__name__}: {error}"[:MAX_PD_ERROR],
        )
        try:
            reports = self.agree_batch(report)
            if len(reports) != self.tp_size or any(
                peer.source_tp_rank != index
                or peer.tp_size != self.tp_size
                or peer.error is not None
                or peer.prefixes != report.prefixes
                for index, peer in enumerate(reports)
            ):
                raise RuntimeError("DSpark prompt projection TP agreement failed")
            for index, peer in enumerate(reports):
                expected = tuple(
                    DsparkPdReady(
                        local.source_engine_id,
                        local.source_request_id,
                        local.source_generation,
                        local.original_prompt_tokens,
                        local.computed_prefix_tokens,
                        local.prompt_digest,
                        remote.layout_digest,
                        local.draft_schema_digest,
                        local.transfer_block_ids,
                        index,
                        self.tp_size,
                    )
                    for local, remote in zip(report.ready, peer.ready, strict=True)
                )
                if peer.ready != expected:
                    raise RuntimeError("DSpark Ready TP agreement failed")
            if error is not None:
                raise error
        except Exception:
            self.fail()
            raise
        for prefix in prefixes:
            self._progress[prefix.source.request_id] = prefix
        return tuple(ready)

    def _project_batch(
        self,
        execute_state: ExecuteModelState | None,
        sources: tuple[DsparkPdSourceRequest, ...],
        prefixes: list[DsparkPdProjectedPrefix],
        ready: list[DsparkPdReady],
    ) -> None:
        if not isinstance(sources, tuple) or any(
            not isinstance(source, DsparkPdSourceRequest) for source in sources
        ):
            raise ValueError("DSpark source batch must be typed immutable declarations")
        if len({source.request_id for source in sources}) != len(sources):
            raise ValueError("DSpark source batch contains duplicate requests")
        if not sources:
            return
        if execute_state is None or execute_state.aux_hidden_states is None:
            raise ValueError(
                "DSpark prefill requires actual native execute auxiliary states"
            )
        batch = execute_state.input_batch
        aux = execute_state.aux_hidden_states
        if len(aux) != len(self.binding.aux_layer_ids) or any(
            tensor.ndim != 2
            or tensor.shape[1] != self.binding.hidden_size
            or tensor.dtype != torch.bfloat16
            or tensor.device != self.binding.device
            for tensor in aux
        ):
            raise ValueError(
                "DSpark prefill requires ordered BF16 auxiliary feature rows"
            )
        features = torch.cat(aux, dim=-1)
        by_request = {source.request_id: source for source in sources}
        groups, layer_groups, sizes = (
            self._group_ids,
            self._layer_group_ids,
            self._block_sizes,
        )
        schema = self._schema_digest
        for row, request_id in enumerate(batch.req_ids):
            source = by_request.pop(request_id, None)
            if source is None:
                continue
            if (
                source.source_engine_id != self.source_engine_id
                or source.draft_schema_digest != schema
            ):
                raise ValueError(
                    "DSpark source engine or draft schema differs from loaded storage"
                )
            if len(source.allocation_block_ids) != len(self._layout.groups) or any(
                block >= self._layout.num_blocks
                for group in source.allocation_block_ids
                for block in group
            ):
                raise ValueError(
                    "DSpark source allocation exceeds registered group capacity"
                )
            offset = int(batch.num_computed_tokens_np[row])
            scheduled = int(batch.num_scheduled_tokens[row])
            count = min(scheduled, max(int(batch.prefill_len_np[row]) - offset, 0))
            if (
                int(batch.prefill_len_np[row]) != source.computed_prefix_tokens
                or count <= 0
            ):
                raise ValueError(
                    "DSpark source does not name an actual prompt prefill chunk"
                )
            previous = self._progress.get(request_id)
            expected_offset = 0 if previous is None else previous.projected_tokens
            if previous is not None:
                old = previous.source
                if (
                    source.source_generation != old.source_generation
                    or source.original_prompt_tokens != old.original_prompt_tokens
                    or source.computed_prefix_tokens != old.computed_prefix_tokens
                    or source.prompt_digest != old.prompt_digest
                    or source.draft_schema_digest != old.draft_schema_digest
                ):
                    raise ValueError(
                        "DSpark source generation/schema changed during prefill"
                    )
            if offset != expected_offset:
                raise ValueError("DSpark prompt chunks must form a contiguous prefix")
            stop = offset + count
            blocks = {}
            for group_id in groups:
                allocated = source.allocation_block_ids[group_id]
                if previous is not None:
                    prior = previous.source.allocation_block_ids[group_id]
                    window, extra = self._draft_windows[group_id]
                    # ced6857a SlidingWindowManager.get_num_skipped_tokens:
                    # only whole blocks before the next-token window may be
                    # retired to native null block 0. Live/future mapping stays
                    # exact; a freed physical ID may appear at an appended index.
                    retired = max(0, offset - window + 1 - extra) // sizes[group_id]
                    if len(allocated) < len(prior) or any(
                        current != old
                        and not (index < retired and current == NULL_BLOCK_ID)
                        for index, (old, current) in enumerate(
                            zip(prior, allocated, strict=False)
                        )
                    ):
                        raise ValueError(
                            "DSpark active draft allocation changed during prefill"
                        )
                needed = (stop + sizes[group_id] - 1) // sizes[group_id]
                blocks[group_id] = allocated[:needed]
                if len(blocks[group_id]) != needed:
                    raise ValueError(
                        "DSpark draft allocation does not cover the prompt chunk"
                    )
                if any(
                    block == NULL_BLOCK_ID
                    for block in allocated[offset // sizes[group_id] : needed]
                ):
                    raise ValueError("DSpark current chunk cannot write a null block")
            begin = int(batch.query_start_loc_np[row])
            if begin + scheduled > features.shape[0]:
                raise ValueError(
                    "DSpark auxiliary rows do not cover the scheduled chunk"
                )
            descriptor = DSparkContextDescriptor(
                request_id,
                source.source_generation,
                source.computed_prefix_tokens,
                self.binding.aux_layer_ids,
                self.binding.hidden_size,
            )
            with set_current_vllm_config(self.speculator.attn_vllm_config):
                initialize_draft_context_chunk(
                    self.binding.draft_model,
                    DSparkContextChunk(descriptor, offset, count),
                    features[begin : begin + count].contiguous(),
                    draft_group_ids=groups,
                    draft_block_ids_by_group=blocks,
                    block_sizes_by_group=sizes,
                    layer_group_ids=layer_groups,
                    device=self.binding.device,
                )
            prefixes.append(DsparkPdProjectedPrefix(source, stop))
            if stop == source.computed_prefix_tokens:
                ready.append(
                    DsparkPdReady(
                        source.source_engine_id,
                        source.request_id,
                        source.source_generation,
                        source.original_prompt_tokens,
                        source.computed_prefix_tokens,
                        source.prompt_digest,
                        self._layout_digest,
                        schema,
                        source.transfer_block_ids,
                        self.tp_rank,
                        self.tp_size,
                    )
                )
        if by_request:
            raise ValueError(
                "DSpark source batch names requests absent from native execute"
            )
