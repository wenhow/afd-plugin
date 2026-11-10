# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Collect native typed V4.1 shared pages and materialized tensor views.

Read-only metadata access only. No tensor contents, device synchronization,
connector registration, transfer thread, or model execution is invoked.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from enum import Enum
from fractions import Fraction
from typing import TypeAlias

import torch
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
    get_storage_block_size,
)

from afd_plugin.compat.npu.kv_layout import (
    KVLayoutGroup,
    KVLayoutPage,
    KVLayoutSnapshot,
    KVLayoutSpec,
    KVLayoutView,
)

KVCacheValue: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]
SpecValue: TypeAlias = str | int | bool | None | Enum | Fraction | torch.dtype


def _spec_value(value: SpecValue) -> str:
    if isinstance(value, Enum):
        return f"enum:{type(value).__module__}.{type(value).__qualname__}:{value.name}"
    if isinstance(value, torch.dtype):
        return f"dtype:{value}"
    if isinstance(value, Fraction):
        return f"fraction:{value.numerator}/{value.denominator}"
    if value is None:
        return "none"
    if type(value) is bool:
        return f"bool:{int(value)}"
    if type(value) is int:
        return f"int:{value}"
    if isinstance(value, str):
        return "str:" + value
    raise ValueError("unsupported native KV spec field type")


def _spec(
    layer: str, spec: KVCacheSpec, tensors: tuple[torch.Tensor, ...]
) -> KVLayoutSpec:
    if isinstance(spec, (AscendMLAAttentionSpec, AscendSlidingWindowMLASpec)):
        if spec.model_version != "deepseek_v41":
            raise ValueError("collector requires native DeepSeek V4.1 specs")
    elif not isinstance(spec, CircularBufferSpec):
        raise ValueError("unsupported native V4.1 cache spec type")
    expected_tensors = (
        2 if isinstance(spec, AscendMLAAttentionSpec) and spec.scale_dim > 0 else 1
    )
    if len(tensors) != expected_tensors:
        raise ValueError("runtime tensor count disagrees with the native scale spec")
    # The native V4.1 backend exposes [blocks, stored rows, heads, width];
    # attn_utils._adjust_dsv4_kv_layout only replaces its first byte stride.
    # Preserve all actual strides below, and reject a view whose typed shape
    # or dtype already disagrees with its local spec before remote comparison.
    for index, tensor in enumerate(tensors):
        width = spec.head_size if index == 0 else spec.scale_dim
        dtype = spec.dtype if index == 0 else spec.scale_dtype
        if tensor.shape[1:] != (get_storage_block_size(spec), spec.num_kv_heads, width):
            raise ValueError("runtime tensor shape disagrees with the native KV spec")
        if tensor.dtype != dtype:
            raise ValueError("runtime tensor dtype disagrees with the native KV spec")
    # This explicit dataclass serialization boundary records every declared
    # native spec field, including newly added fields. Unsupported values fail
    # visibly instead of silently omitting layout semantics during an upgrade.
    values = vars(spec)
    return KVLayoutSpec(
        layer,
        f"{type(spec).__module__}.{type(spec).__qualname__}",
        tuple((field.name, _spec_value(values[field.name])) for field in fields(spec)),
        spec.page_size_bytes,
        expected_tensors,
    )


def _tensors(value: KVCacheValue) -> tuple[torch.Tensor, ...]:
    tensors = (value,) if isinstance(value, torch.Tensor) else tuple(value)
    if not tensors or any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("every cache layer requires materialized tensors")
    for tensor in tensors:
        if (
            tensor.layout != torch.strided
            or tensor.device.type not in ("cpu", "npu")
            or tensor.numel() == 0
            or tensor.data_ptr() == 0
        ):
            raise ValueError(
                "cache views must have real nonempty strided CPU/NPU storage"
            )
    return tensors


def collect_deepseek_v41_kv_layout(
    kv_cache_config: KVCacheConfig,
    kv_caches: Mapping[str, KVCacheValue],
) -> KVLayoutSnapshot:
    """Snapshot the actual config/views supplied to native KV registration.

    Source: KVCacheTensor/Group/Config @ vLLM ced6857a:1361-1434 and V4.1
    cache_config @ Ascend 7e37e29a:147-178,248-274. Initial support deliberately
    covers native shared-page layer_stride=0, not generic strided descriptors.
    Capacities are local; do not interpret equal configs as remote attestation.
    """
    groups = []
    has_v41_mla = False
    for group in kv_cache_config.kv_cache_groups:
        specs = []
        for layer in group.layer_names:
            native_spec = (
                group.kv_cache_spec.kv_cache_specs[layer]
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else group.kv_cache_spec
            )
            has_v41_mla |= isinstance(native_spec, AscendMLAAttentionSpec)
            specs.append(_spec(layer, native_spec, _tensors(kv_caches[layer])))
        groups.append(
            KVLayoutGroup(
                tuple(group.layer_names),
                tuple(specs),
                group.is_eagle_group,
                group.host_resident,
                group.enable_kv_transfer,
                _spec_value(group.role),
            )
        )
    if not has_v41_mla:
        raise ValueError("collector requires the native V4.1 main/index cache group")
    expected = tuple(
        layer
        for descriptor in kv_cache_config.kv_cache_tensors
        for layer in descriptor.layers
    )
    if set(kv_caches) != set(expected):
        raise ValueError(
            "runtime cache names must exactly match all native descriptors"
        )
    aliases: dict[tuple[torch.device, int], int] = {}
    devices: set[torch.device] = set()
    pages, views = [], []
    for page_index, descriptor in enumerate(kv_cache_config.kv_cache_tensors):
        if descriptor.layer_stride != 0 or descriptor.host_resident:
            raise ValueError("collector requires device shared-page layer_stride=0")
        tensors_by_layer = tuple(
            (layer, _tensors(kv_caches[layer])) for layer in descriptor.layers
        )
        all_tensors = tuple(
            tensor for _, tensors in tensors_by_layer for tensor in tensors
        )
        storage_keys = {
            (tensor.device, tensor.untyped_storage().data_ptr())
            for tensor in all_tensors
        }
        if len(storage_keys) != 1:
            raise ValueError("one native shared page must resolve to the same storage")
        storage_key = next(iter(storage_keys))
        alias = aliases.setdefault(storage_key, len(aliases))
        page_address = min(tensor.data_ptr() for tensor in all_tensors)
        storage_offset = page_address - storage_key[1]
        pages.append(
            KVLayoutPage(
                tuple(descriptor.layers),
                descriptor.size,
                descriptor.offset,
                descriptor.layer_stride,
                descriptor.block_stride,
                alias,
                storage_offset,
            )
        )
        for layer, tensors in tensors_by_layer:
            for tensor_index, tensor in enumerate(tensors):
                devices.add(tensor.device)
                element_bytes = tensor.element_size()
                local_offset = tensor.storage_offset() * element_bytes
                if tensor.data_ptr() - storage_key[1] != local_offset:
                    raise ValueError(
                        "opaque tensor address disagrees with its storage offset"
                    )
                views.append(
                    KVLayoutView(
                        layer,
                        tensor_index,
                        page_index,
                        str(tensor.dtype),
                        element_bytes,
                        tuple(tensor.shape),
                        tuple(stride * element_bytes for stride in tensor.stride()),
                        alias,
                        local_offset,
                        tensor.untyped_storage().nbytes(),
                        tensor.data_ptr() - page_address,
                    )
                )
    if len(devices) != 1:
        raise ValueError("one snapshot requires one actual device")
    return KVLayoutSnapshot(
        kv_cache_config.num_blocks,
        next(iter(devices)).type,
        kv_cache_config.kv_cache_layout,
        tuple(groups),
        tuple(pages),
        tuple(views),
    )
