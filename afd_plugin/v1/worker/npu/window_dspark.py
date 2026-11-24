# SPDX-License-Identifier: Apache-2.0
"""Window DSpark draft configuration and verification graph geometry."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
        AscendDSparkSpeculator,
    )


def configure_window_dspark_draft(target: ModelConfig, draft: ModelConfig) -> None:
    """Resolve the checkpoint's draft MoE dimensions for the native Ascend loader."""
    text = draft.hf_text_config
    if draft.model == target.model:
        # The embedded checkpoint describes two MoEs. Only the layer numbering
        # follows the backbone depth; the drafter has its own router and weights.
        text.n_routed_experts = text.dspark_n_routed_experts
        text.num_experts_per_tok = text.dspark_num_experts_per_tok
        text.num_hidden_layers = target.hf_text_config.num_hidden_layers
    else:
        # A standalone draft may already use the native MoE field names.
        if hasattr(text, "dspark_n_routed_experts"):
            text.n_routed_experts = text.dspark_n_routed_experts
        if hasattr(text, "dspark_num_experts_per_tok"):
            text.num_experts_per_tok = text.dspark_num_experts_per_tok
    draft.hf_config.architectures = ["AFDDeepseekV41DSparkModel"]


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
    descs: list[BatchExecutionDescriptor],
    width: int,
    max_reqs: int,
    num_ubatches: int,
) -> list[BatchExecutionDescriptor]:
    """Align whole verification rows to the target's chosen stage count."""
    if width < 1 or num_ubatches < 1:
        raise ValueError("Verification width and target stage count must be positive")
    buckets = set()
    for desc in descs:
        rows = min(desc.num_tokens // width, max_reqs)
        rows -= rows % num_ubatches
        if rows:
            buckets.add(
                replace(
                    desc,
                    num_reqs=rows,
                    num_tokens=rows * width,
                    uniform_token_count=width,
                    max_query_len=None,
                    num_ubatches=num_ubatches,
                )
            )
    if not buckets:
        raise ValueError(
            f"Window DSpark needs max_num_seqs >= {num_ubatches} and a target "
            f"graph bucket of at least {num_ubatches * width} tokens"
        )
    return sorted(buckets, key=lambda desc: desc.num_tokens, reverse=True)


def verification_stage_ranges(
    rows: int, tokens: int, width: int, num_ubatches: int
) -> tuple[tuple[slice, slice], ...]:
    if (
        num_ubatches < 1
        or rows < num_ubatches
        or rows % num_ubatches
        or width < 1
        or tokens != rows * width
    ):
        raise ValueError(f"Invalid verification shape: {rows} rows, {tokens} tokens")
    rows_per_stage = rows // num_ubatches
    return tuple(
        (
            slice(start, start + rows_per_stage),
            slice(start * width, (start + rows_per_stage) * width),
        )
        for start in range(0, rows, rows_per_stage)
    )
