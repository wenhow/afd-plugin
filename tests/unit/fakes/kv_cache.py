# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Actual native KV config/specs with real CPU shared-storage views.

Native grouping is reused. CPU storage construction exercises metadata only;
it is not the NPU allocator, connector registration or a PD transfer.
"""

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheConfig, KVCacheTensor
from vllm.v1.worker.utils import AttentionGroup
from vllm_ascend.attention.dsa_v41 import (
    DeepseekV41CacheBackend,
    DeepseekV41CacheLayer,
)
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
)
from vllm_ascend.models.deepseek_v41.cache_config import (
    get_layer_tuples,
    group_cache_specs,
    make_cache_groups,
)
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def make_native_kv_cache_case(
    *, num_blocks=3, allocation_offset=0, include_draft=False
):
    prefix = "model.layers.0.self_attn"
    names = SimpleNamespace(
        main=prefix + ".long_kv_cache",
        index=prefix + ".indexer.k_cache",
        state=prefix + ".compressor.state",
        swa=prefix + ".swa_cache",
        draft="model.mtp.0.self_attn.swa_cache",
    )
    specs = {
        names.main: AscendMLAAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.bfloat16,
            tokens_per_state=2,
            model_version="deepseek_v41",
            storage_block_size=2,
        ),
        names.index: AscendMLAAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=2,
            dtype=torch.int8,
            tokens_per_state=2,
            model_version="deepseek_v41",
            storage_block_size=2,
            scale_dim=1,
            scale_dtype=torch.float16,
        ),
        names.state: CircularBufferSpec(
            block_size=2, num_kv_heads=1, head_size=3, dtype=torch.bfloat16
        ),
        names.swa: AscendSlidingWindowMLASpec(
            block_size=4,
            num_kv_heads=1,
            head_size=2,
            dtype=torch.bfloat16,
            sliding_window=4,
            model_version="deepseek_v41",
        ),
    }
    if include_draft:
        specs[names.draft] = specs[names.swa]
    groups = make_cache_groups(group_cache_specs(specs))
    page_sizes, layer_tuples = get_layer_tuples(specs)
    page_size = page_sizes[0]
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_blocks * page_size,
                layers=list(layer_tuples[0]),
                layer_stride=0,
                block_stride=page_size,
            )
        ],
        kv_cache_groups=groups,
    )
    raw = torch.zeros(allocation_offset + num_blocks * page_size, dtype=torch.uint8)
    caches = {}
    for name, dtype, width, rows, offset in (
        (names.main, torch.bfloat16, 4, 2, 0),
        (names.index, torch.int8, 2, 2, 16),
        (names.state, torch.bfloat16, 3, 2, 0),
        (names.swa, torch.bfloat16, 2, 4, 0),
    ):
        size = dtype.itemsize
        caches[name] = torch.empty(0, dtype=dtype).set_(
            raw.untyped_storage(),
            (allocation_offset + offset) // size,
            (num_blocks, rows, 1, width),
            (page_size // size, width, width, 1),
        )
    scale = torch.empty(0, dtype=torch.float16).set_(
        raw.untyped_storage(),
        (allocation_offset + 20) // 2,
        (num_blocks, 2, 1, 1),
        (page_size // 2, 1, 1, 1),
    )
    caches[names.index] = (caches[names.index], scale)
    if include_draft:
        caches[names.draft] = caches[names.swa]
    return SimpleNamespace(config=config, caches=caches, names=names, raw=raw)


def make_native_zeroing_case(*, ratio=128, num_blocks=3, allocation_offset=64):
    """Native specs/group planner/carriers over guarded CPU byte storage.

    CPU views model the documented descriptor layout only. This does not call
    the NPU allocator, native Worker initializer or the zeroing device kernel.
    """
    logical_block_size, head_size, index_head_size = 128, 16, 8
    rows, state_rows, suffix_bytes = logical_block_size // ratio, 32, 96
    prefix = "model.layers.0.self_attn"
    names = SimpleNamespace(
        main=prefix + ".long_kv_cache",
        index=prefix + ".indexer.k_cache",
        state=prefix + ".compressor.state_cache",
        swa=prefix + ".swa_cache",
    )
    specs = {
        names.main: AscendMLAAttentionSpec(
            block_size=logical_block_size,
            num_kv_heads=1,
            head_size=head_size,
            dtype=torch.bfloat16,
            tokens_per_state=ratio,
            storage_block_size=rows,
            model_version="deepseek_v41",
        ),
        names.index: AscendMLAAttentionSpec(
            block_size=logical_block_size,
            num_kv_heads=1,
            head_size=index_head_size,
            dtype=torch.int8,
            tokens_per_state=ratio,
            storage_block_size=rows,
            scale_dim=1,
            scale_dtype=torch.float16,
            model_version="deepseek_v41",
        ),
        names.state: CircularBufferSpec(
            block_size=state_rows,
            num_kv_heads=1,
            head_size=head_size,
            dtype=torch.bfloat16,
        ),
        names.swa: AscendSlidingWindowMLASpec(
            block_size=logical_block_size,
            num_kv_heads=1,
            head_size=head_size,
            dtype=torch.bfloat16,
            sliding_window=logical_block_size,
            model_version="deepseek_v41",
        ),
    }
    page_sizes, layer_tuples = get_layer_tuples(specs)
    page_size = page_sizes[0]
    descriptor = KVCacheTensor(
        size=num_blocks * page_size,
        layers=list(layer_tuples[0]),
        layer_stride=0,
        block_stride=page_size,
    )
    groups = make_cache_groups(group_cache_specs(specs))
    context = {}
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=context),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v41", architectures=["DeepseekV41ForCausalLM"]
            )
        ),
        additional_config={},
        kv_transfer_config=None,
    )
    raw = torch.full(
        (allocation_offset + descriptor.size + suffix_bytes,),
        93,
        dtype=torch.uint8,
    )
    caches = {}
    index_offset = specs[names.main].page_size_bytes
    for name, dtype, width, count, offset in (
        (names.main, torch.bfloat16, head_size, rows, 0),
        (names.index, torch.int8, index_head_size, rows, index_offset),
        (names.state, torch.bfloat16, head_size, state_rows, 0),
        (names.swa, torch.bfloat16, head_size, logical_block_size, 0),
    ):
        element_bytes = dtype.itemsize
        caches[name] = torch.empty(0, dtype=dtype).set_(
            raw.untyped_storage(),
            (allocation_offset + offset) // element_bytes,
            (num_blocks, count, 1, width),
            (page_size // element_bytes, width, width, 1),
        )
    scale_offset = index_offset + rows * index_head_size
    scale = torch.empty(0, dtype=torch.float16).set_(
        raw.untyped_storage(),
        (allocation_offset + scale_offset) // 2,
        (num_blocks, rows, 1, 1),
        (page_size // 2, 1, 1, 1),
    )
    caches[names.index] = (caches[names.index], scale)
    carriers = {}
    for name, spec in specs.items():
        carrier = DeepseekV41CacheLayer(config, name, spec)
        carrier.bind_kv_cache(caches[name])
        carriers[name] = carrier
    owner = NPUModelRunner.__new__(NPUModelRunner)
    owner.vllm_config = config
    owner.compilation_config = config.compilation_config
    owner.device = torch.device("cpu")
    owner.attn_groups = [
        [
            AttentionGroup(
                backend=DeepseekV41CacheBackend,
                layer_names=[name],
                kv_cache_spec=group.kv_cache_spec.kv_cache_specs[name],
                kv_cache_group_id=index,
            )
            for name in group.layer_names
        ]
        for index, group in enumerate(groups)
    ]
    owner.kernel_block_sizes = [group.kv_cache_spec.block_size for group in groups]
    owner.kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[descriptor],
        kv_cache_groups=groups,
    )
    owner.kv_block_zeroer = None
    return SimpleNamespace(
        owner=owner,
        config=config,
        descriptor=descriptor,
        raw=raw,
        caches=caches,
        carriers=carriers,
        layers=[carriers[name] for name in descriptor.layers],
        names=names,
        num_blocks=num_blocks,
        allocation_offset=allocation_offset,
        scale_offset=scale_offset,
    )
