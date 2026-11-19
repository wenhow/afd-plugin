# SPDX-License-Identifier: Apache-2.0
"""Target verification graph geometry for the two Window transport slots."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
        AscendDSparkSpeculator,
    )


@contextmanager
def window_dspark_draft_graph_sizes(
    speculator: AscendDSparkSpeculator,
) -> Iterator[None]:
    """Give the native draft manager whole query rows within each token budget."""
    original = speculator.vllm_config
    draft_config = copy(original)
    compilation = copy(original.compilation_config)
    width = speculator.num_query_per_req
    sizes = {
        min(budget // width, speculator.max_num_reqs) * width
        for budget in (compilation.cudagraph_capture_sizes or ())
        if budget >= width
    }
    if not sizes:
        raise ValueError(
            f"Window DSpark needs a draft graph budget of at least {width} tokens"
        )
    compilation.cudagraph_capture_sizes = sorted(sizes)
    compilation.max_cudagraph_capture_size = max(sizes)
    draft_config.compilation_config = compilation
    # The Ascend manager constructs both its dispatch table and its per-size
    # graph update records during KV initialization. It retains draft_config.
    speculator.vllm_config = draft_config
    try:
        yield
    finally:
        speculator.vllm_config = original


def window_dspark_capture_descs(
    descs: list[BatchExecutionDescriptor], width: int, max_reqs: int
) -> list[BatchExecutionDescriptor]:
    """Each graph stage must hold whole target verification rows."""
    buckets = set()
    for desc in descs:
        rows = min(desc.num_tokens // width, max_reqs)
        rows -= rows % 2
        if rows:
            buckets.add(
                replace(
                    desc,
                    num_reqs=rows,
                    num_tokens=rows * width,
                    uniform_token_count=width,
                    max_query_len=None,
                    num_ubatches=2,
                )
            )
    if not buckets:
        raise ValueError(
            f"Window DSpark U2 needs max_num_seqs >= 2 and a target "
            f"graph bucket of at least {2 * width} tokens"
        )
    return sorted(buckets, key=lambda desc: desc.num_tokens, reverse=True)


def verification_stage_ranges(
    rows: int, tokens: int, width: int
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    if rows < 2 or rows % 2 or width < 1 or tokens != rows * width:
        raise ValueError(f"Invalid U2 verification shape: {rows} rows, {tokens} tokens")
    return (
        (slice(0, rows // 2), slice(0, tokens // 2)),
        (slice(rows // 2, rows), slice(tokens // 2, tokens)),
    )
