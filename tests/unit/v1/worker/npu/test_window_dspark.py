# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from afd_plugin.v1.worker.npu.window_dspark import (
    verification_stage_ranges,
    window_dspark_capture_descs,
    window_dspark_draft_graph_sizes,
)


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    num_reqs: int = 1
    num_ubatches: int = 1
    uniform_token_count: int | None = None
    max_query_len: int | None = None


@pytest.mark.parametrize("width", [2, 5, 9])
@pytest.mark.parametrize(
    "stages,expected", [(1, [7, 5, 3, 2, 1]), (2, [6, 4, 2]), (4, [4])]
)
def test_graphs_keep_whole_rows_and_respect_capacity(width, stages, expected):
    descs = [Descriptor(i * width) for i in (1, 2, 3, 5, 8)]
    graphs = window_dspark_capture_descs(descs, width, max_reqs=7, num_ubatches=stages)
    assert [(d.num_reqs, d.num_tokens) for d in graphs] == [
        (rows, rows * width) for rows in expected
    ]
    assert all(
        d.num_ubatches == stages and d.uniform_token_count == width for d in graphs
    )


@pytest.mark.parametrize("stages", [1, 2, 4])
@pytest.mark.parametrize("live,width", [(1, 5), (3, 9), (4, 2)])
def test_replay_partition_matches_capture_with_padding(live, width, stages):
    padded = ((live + stages - 1) // stages) * stages
    rows = list(range(live)) + [None] * (padded - live)
    tokens = [owner for owner in rows for _ in range(width)]
    ranges = verification_stage_ranges(padded, len(tokens), width, stages)
    for requests, span in ranges:
        assert len(rows[requests]) == padded // stages
        assert tokens[span] == [owner for owner in rows[requests] for _ in range(width)]
    assert [token for _, span in ranges for token in tokens[span]] == tokens


def test_graph_capacity_too_small_for_two_verify_rows():
    with pytest.raises(ValueError, match="at least 10 tokens"):
        window_dspark_capture_descs(
            [Descriptor(5)], width=5, max_reqs=16, num_ubatches=2
        )


def test_single_request_target_graph_needs_no_second_stage():
    assert window_dspark_capture_descs(
        [Descriptor(5)],
        width=5,
        max_reqs=1,
        num_ubatches=1,
    ) == [Descriptor(5, num_reqs=1, uniform_token_count=5)]


@pytest.mark.parametrize(
    "rows,tokens,width,stages",
    [
        (3, 15, 5, 2),
        (1, 5, 5, 4),
        (4, 19, 5, 4),
        (4, 20, 5, 0),
    ],
)
def test_invalid_verification_geometry_is_rejected(rows, tokens, width, stages):
    with pytest.raises(ValueError, match="verification shape"):
        verification_stage_ranges(rows, tokens, width, stages)


@pytest.mark.parametrize("width,expected", [(5, [5, 10]), (6, [6, 12])])
def test_draft_manager_retains_its_own_query_buckets(width, expected):
    compilation = SimpleNamespace(
        cudagraph_capture_sizes=[1, 6, 12, 18], max_cudagraph_capture_size=18
    )
    target = SimpleNamespace(compilation_config=compilation)
    draft = SimpleNamespace(vllm_config=target, num_query_per_req=width, max_num_reqs=2)
    with window_dspark_draft_graph_sizes(draft):
        manager_config = draft.vllm_config
    assert manager_config.compilation_config.cudagraph_capture_sizes == expected
    assert manager_config.compilation_config.max_cudagraph_capture_size == expected[-1]
    assert draft.vllm_config is target
    assert compilation.cudagraph_capture_sizes == [1, 6, 12, 18]


def test_failed_draft_initialization_restores_config():
    target = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[12])
    )
    draft = SimpleNamespace(vllm_config=target, num_query_per_req=5, max_num_reqs=16)
    with (
        pytest.raises(RuntimeError, match="initialization failed"),
        window_dspark_draft_graph_sizes(draft),
    ):
        raise RuntimeError("initialization failed")
    assert draft.vllm_config is target
