# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Actual native KV config/specs with real CPU shared-storage views.

Native grouping is reused. CPU storage construction exercises metadata only;
it is not the NPU allocator, connector registration or a PD transfer.
"""

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheConfig, KVCacheTensor
from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
)
from vllm_ascend.models.deepseek_v41.cache_config import (
    get_layer_tuples,
    group_cache_specs,
    make_cache_groups,
)


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
