# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Controlled immutable snapshots; no tensors or device claims."""

from dataclasses import replace

from afd_plugin.compat.npu.kv_layout import (
    KVLayoutGroup,
    KVLayoutPage,
    KVLayoutSnapshot,
    KVLayoutSpec,
    KVLayoutView,
)


def make_kv_layout(*, num_blocks=4, allocation_offset=0, page_count=1):
    groups, pages, views = [], [], []
    for index in range(page_count):
        names = (
            f"model.layers.{index}.self_attn.long_kv_cache",
            f"model.layers.{index}.self_attn.indexer.k_cache",
        )
        for part, name in enumerate(names):
            spec = KVLayoutSpec(
                name, "controlled.V41Spec", (("block_size", "int:4"),), 16, 1 + part
            )
            groups.append(
                KVLayoutGroup((name,), (spec,), False, False, True, "default")
            )
        pages.append(
            KVLayoutPage(names, num_blocks * 32, 0, 0, 32, index, allocation_offset)
        )
        for name, tensor_index, dtype, size, shape, strides, offset in (
            (names[0], 0, "torch.bfloat16", 2, (num_blocks, 2, 4), (32, 8, 2), 0),
            (names[1], 0, "torch.int8", 1, (num_blocks, 2, 2), (32, 2, 1), 16),
            (names[1], 1, "torch.float16", 2, (num_blocks, 2, 1), (32, 2, 2), 20),
        ):
            views.append(
                KVLayoutView(
                    name,
                    tensor_index,
                    index,
                    dtype,
                    size,
                    shape,
                    strides,
                    index,
                    allocation_offset + offset,
                    allocation_offset + num_blocks * 32,
                    offset,
                )
            )
    return KVLayoutSnapshot(
        num_blocks, "cpu", None, tuple(groups), tuple(pages), tuple(views)
    )


def make_slab_kv_layout(*, num_blocks=4, page_offsets=None):
    snapshot = make_kv_layout(num_blocks=num_blocks, page_count=2)
    offsets = (0, num_blocks * 32) if page_offsets is None else page_offsets
    capacity = max(offsets) + num_blocks * 32
    return replace(
        snapshot,
        pages=tuple(
            replace(page, storage_alias=0, storage_offset_bytes=offset)
            for page, offset in zip(snapshot.pages, offsets, strict=True)
        ),
        views=tuple(
            replace(
                view,
                storage_alias=0,
                storage_offset_bytes=view.page_offset_bytes + offsets[view.page_index],
                storage_size_bytes=capacity,
            )
            for view in snapshot.views
        ),
    )
