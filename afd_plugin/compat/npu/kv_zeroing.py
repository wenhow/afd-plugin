# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Expose native V4.1 physical cache tiles to the KV block zeroing kernel."""

from dataclasses import dataclass, replace

import torch
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheTensor
from vllm.v1.worker.utils import KVBlockZeroer
from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheLayer
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

ZERO_WORD_BYTES = 4


@dataclass(frozen=True)
class KVCacheTensorBinding:
    kv_cache: torch.Tensor


def _physical_block_view(
    descriptor: KVCacheTensor,
    layers: list[DeepseekV41CacheLayer],
    num_blocks: int,
) -> torch.Tensor:
    # Pinned V4.1 get_layer_tuples puts long KV first, at byte offset zero.
    # Other resources overlay this same global block-ID tile. In particular,
    # the ratio-128 index scale is only two bytes; exposing components alone
    # cannot satisfy the unchanged native zeroer's four-byte segment ABI.
    if (
        descriptor.offset != 0
        or descriptor.layer_stride != 0
        or descriptor.size != num_blocks * descriptor.block_stride
        or descriptor.block_stride % ZERO_WORD_BYTES
        or not descriptor.layers[0].endswith(".long_kv_cache")
    ):
        raise ValueError("V4.1 zeroing requires native overlaid physical block tiles")
    anchor = layers[0].kv_cache[0]
    if not isinstance(anchor, torch.Tensor):
        raise ValueError("V4.1 physical tile must start with the long KV Tensor")
    storage = anchor.untyped_storage()
    base = anchor.data_ptr() - storage.data_ptr()
    # PD allocation can return a 2 MiB-aligned slice of a larger storage. Its
    # first bound byte, rather than storage.data_ptr(), is the descriptor base.
    if (
        base != anchor.storage_offset() * anchor.element_size()
        or base % ZERO_WORD_BYTES
        or base < 0
        or base + descriptor.size > storage.nbytes()
    ):
        raise ValueError("V4.1 physical tile exceeds its bound storage")
    for layer in layers:
        cache = layer.kv_cache[0]
        components = cache if isinstance(cache, tuple) else (cache,)
        for component in components:
            element_bytes = component.element_size()
            offset = component.data_ptr() - anchor.data_ptr()
            span = element_bytes + sum(
                (component.shape[dim] - 1) * component.stride(dim) * element_bytes
                for dim in range(1, component.ndim)
            )
            if (
                component.device != anchor.device
                or component.untyped_storage().data_ptr() != storage.data_ptr()
                or component.shape[0] != num_blocks
                or component.stride(0) * element_bytes != descriptor.block_stride
                or offset < 0
                or offset + span > descriptor.block_stride
            ):
                raise ValueError("V4.1 cache component escapes its physical block tile")
    return torch.empty(0, dtype=torch.uint8, device=anchor.device).set_(
        storage,
        base,
        (num_blocks, descriptor.block_stride),
        (descriptor.block_stride, 1),
    )


class DeepseekV41KVBlockZeroer(KVBlockZeroer):
    """Describe physical tiles without changing native carriers or their views.

    Native V4.1 keeps ``kv_cache=[Tensor/tuple[Tensor, ...]]``. The upstream
    zeroer accepts direct Tensors only. Temporary byte views expose complete
    physical tiles to its unchanged segment construction and zeroing kernel.
    Fresh global block IDs own the whole tile, including overlaid resources.
    """

    def __init__(self, runner: NPUModelRunner) -> None:
        context = runner.compilation_config.static_forward_context
        bindings: dict[str, KVCacheTensorBinding] = {}
        groups = []
        eligible = {
            name: group
            for cache_groups in runner.attn_groups
            for group in cache_groups
            if isinstance(group.kv_cache_spec, AttentionSpec)
            and group.kv_cache_group_id < len(runner.kernel_block_sizes)
            for name in group.layer_names
        }
        for descriptor in runner.kv_cache_config.kv_cache_tensors:
            if descriptor.host_resident:
                continue
            group = next(
                (eligible[name] for name in descriptor.layers if name in eligible), None
            )
            if group is None:
                continue
            layers = [context[name] for name in descriptor.layers]
            if not all(isinstance(layer, DeepseekV41CacheLayer) for layer in layers):
                raise ValueError("V4.1 zeroing requires native bound cache layers")
            view = _physical_block_view(
                descriptor, layers, runner.kv_cache_config.num_blocks
            )
            if view.device != runner.device:
                raise ValueError("V4.1 zeroing requires runner-local device storage")
            name = descriptor.layers[0]
            bindings[name] = KVCacheTensorBinding(view)
            groups.append(replace(group, layer_names=[name]))
        super().__init__(
            runner.device,
            attn_groups_iter=groups,
            kernel_block_sizes=runner.kernel_block_sizes,
            static_forward_context=bindings,
            num_blocks=runner.kv_cache_config.num_blocks,
        )
