# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Rank and row projection contracts; no tensor or device simulation."""

from dataclasses import replace

import pytest

from afd_plugin.distributed.hccl import HcclStageLayout, build_hccl_rank_mapping
from afd_plugin.distributed.topology import build_rank_mapping
from tests.unit.fakes.afd_config import hccl_config


@pytest.mark.parametrize(
    "attention_size,ffn_size,groups",
    [
        (2, 2, ((0, 2), (1, 3))),
        (4, 2, ((0, 2, 3), (1, 4, 5))),
        (2, 4, ((0, 1, 4), (2, 3, 5))),
    ],
)
@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_hccl_peers_agree_on_ffn_first_group_order(
    attention_size, ffn_size, groups, role
):
    config = hccl_config(role=role, attention_size=attention_size, ffn_size=ffn_size)
    size = attention_size if role == "attention" else ffn_size

    for role_rank in range(size):
        mapping = build_hccl_rank_mapping(config, role_rank)
        group = next(group for group in groups if mapping.world_rank in group)
        assert mapping.subgroup_ranks == group
        assert mapping.rank_in_subgroup == group.index(mapping.world_rank)
        assert mapping.p2p_rank == mapping.world_rank
        if role == "attention" and ffn_size > attention_size:
            assert mapping.dp_metadata_destinations == tuple(
                rank for rank in group if rank < ffn_size
            )


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("native_tp_size", [1, 2, 4])
def test_equal_native_tp_keeps_the_common_rank_coordinates(role, native_tp_size):
    config = hccl_config(role=role, attention_size=4, ffn_size=4)

    for rank in range(4):
        assert build_hccl_rank_mapping(
            config, rank, native_tp_size=native_tp_size
        ) == build_rank_mapping(config, rank)


def test_hccl_fanout_does_not_change_other_connectors_mapping_rules():
    config = hccl_config(attention_size=2, ffn_size=4)

    assert build_hccl_rank_mapping(config, 0).subgroup_ranks == (0, 1, 4)
    with pytest.raises(ValueError, match="num_attention_ranks >="):
        build_rank_mapping(replace(config, connector="P2pNcclAFDConnector"), 0)


@pytest.mark.parametrize("attention_size,ffn_size", [(2, 4), (4, 2)])
def test_prepared_mapping_does_not_enable_unequal_runtime_topologies(
    attention_size, ffn_size
):
    config = hccl_config(attention_size=attention_size, ffn_size=ffn_size)

    assert build_hccl_rank_mapping(config, 0).subgroup_ranks
    with pytest.raises(ValueError, match="equal positive A/F"):
        config.validate()


@pytest.mark.parametrize(
    "changes,native_tp_size,role_rank,reason",
    [
        ({"connector": "P2pNcclAFDConnector"}, 1, 0, "P2pHccl"),
        ({"num_attention_ranks": 0}, 1, 0, "positive integer"),
        ({"num_ffn_ranks": True}, 1, 0, "positive integer"),
        ({"num_attention_ranks": 3}, 1, 0, "integral A/F"),
        ({}, 3, 0, "complete native TP"),
        ({"num_attention_ranks": 4}, 2, 0, "native TP1"),
        ({}, False, 0, "positive integer"),
        ({"role": "other"}, 1, 0, "attention or ffn"),
        ({}, 1, True, "role_rank"),
        ({}, 1, 2, "role_rank"),
    ],
)
def test_hccl_mapping_rejects_unimplemented_or_ambiguous_layouts(
    changes, native_tp_size, role_rank, reason
):
    config = replace(hccl_config(attention_size=2, ffn_size=2), **changes)

    with pytest.raises(ValueError, match=reason):
        build_hccl_rank_mapping(config, role_rank, native_tp_size=native_tp_size)


@pytest.mark.parametrize(
    "valid,padded,ffn_size,expected",
    [
        ((3,), (4,), 1, [(1, 0, 0, 0, 3, 4)]),
        ((0, 3), (0, 4), 1, [(1, 0, 0, 0, 0, 1), (2, 0, 0, 1, 3, 4)]),
        ((5,), (5,), 2, [(2, 0, 0, 0, 3, 3), (2, 1, 3, 0, 2, 2)]),
        ((3,), (6,), 2, [(2, 0, 0, 0, 3, 3), (2, 1, 3, 0, 0, 3)]),
        ((1,), (1,), 3, [(3, 0, 0, 0, 1, 1), (3, 1, 1, 0, 0, 1), (3, 2, 2, 0, 0, 1)]),
        ((0,), (0,), 2, [(2, 0, 0, 0, 0, 0), (2, 1, 0, 0, 0, 0)]),
        ((0,), (4,), 2, [(2, 0, 0, 0, 0, 2), (2, 1, 2, 0, 0, 2)]),
    ],
)
@pytest.mark.parametrize("stage_idx", [0, 1])
def test_stage_projects_native_padding_and_valid_prefixes(
    valid, padded, ffn_size, expected, stage_idx
):
    layout = HcclStageLayout(stage_idx, valid, padded, ffn_size)

    assert [
        (
            peer.attention_world_rank,
            peer.ffn_world_rank,
            peer.attention_offset,
            peer.ffn_offset,
            peer.valid_tokens,
            peer.padded_tokens,
        )
        for peer in layout.peer_slices
    ] == expected


@pytest.mark.parametrize("attention_size,ffn_size", [(1, 1), (2, 2), (4, 2), (2, 4)])
@pytest.mark.parametrize("token_counts", [(0, 1, 3, 7), (1, 0, 5, 2)])
def test_valid_rows_return_to_their_original_attention_coordinates(
    attention_size, ffn_size, token_counts
):
    valid = token_counts[:attention_size]
    padded = tuple(count + 1 for count in valid)
    layout = HcclStageLayout(1, valid, padded, ffn_size)
    rows = tuple(
        tuple((rank, row) for row in range(count)) for rank, count in enumerate(valid)
    )
    buffers = [
        [None] * sum(peer.padded_tokens for peer in layout.for_ffn(rank))
        for rank in range(ffn_size)
    ]
    returned = [[None] * count for count in valid]

    for peer in layout.peer_slices:
        attention_rank = peer.attention_world_rank - ffn_size
        buffers[peer.ffn_world_rank][peer.ffn_output_slice] = rows[attention_rank][
            peer.attention_output_slice
        ]
    for peer in layout.peer_slices:
        attention_rank = peer.attention_world_rank - ffn_size
        returned[attention_rank][peer.attention_output_slice] = buffers[
            peer.ffn_world_rank
        ][peer.ffn_output_slice]

    assert tuple(tuple(rank_rows) for rank_rows in returned) == rows
    for rank in range(attention_size):
        assert (
            sum(peer.valid_tokens for peer in layout.for_attention(rank)) == valid[rank]
        )


def test_aggregation_returns_only_each_peers_valid_rows():
    layout = HcclStageLayout(0, (2, 0, 1), (3, 0, 2), 1)
    aggregate = [11, 12, 0, 0, 21, 0]

    assert [aggregate[peer.ffn_output_slice] for peer in layout.for_ffn(0)] == [
        [11, 12],
        [],
        [21],
    ]


def test_two_stages_keep_independent_peer_offsets():
    first = HcclStageLayout(0, (3, 1), (4, 2), 1)
    second = HcclStageLayout(1, (1, 0), (2, 0), 1)

    assert [(peer.ffn_offset, peer.padded_tokens) for peer in first.for_ffn(0)] == [
        (0, 4),
        (4, 2),
    ]
    assert [(peer.ffn_offset, peer.padded_tokens) for peer in second.for_ffn(0)] == [
        (0, 2),
        (2, 1),
    ]


@pytest.mark.parametrize("native_tp_size", [2, 4])
def test_tp_layout_keeps_each_coordinate_on_its_own_peer(native_tp_size):
    layout = HcclStageLayout(0, (3,) * 4, (4,) * 4, 4, native_tp_size)

    assert [
        (peer.ffn_world_rank, peer.attention_world_rank) for peer in layout.peer_slices
    ] == [(rank, 4 + rank) for rank in range(4)]


def test_tp_layout_keeps_distinct_dp_batches_on_their_native_coordinates():
    layout = HcclStageLayout(0, (3, 3, 1, 1), (4, 4, 2, 2), 4, 2)

    assert [(peer.valid_tokens, peer.padded_tokens) for peer in layout.peer_slices] == [
        (3, 4),
        (3, 4),
        (1, 2),
        (1, 2),
    ]


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"stage_idx": 2}, "stage_idx"),
        ({"stage_idx": False}, "stage_idx"),
        ({"attention_valid_tokens": []}, "nonempty tuples"),
        ({"attention_valid_tokens": ()}, "nonempty tuples"),
        ({"attention_padded_tokens": (4,)}, "matching"),
        ({"attention_valid_tokens": (True, 1)}, "valid <= padded"),
        ({"attention_valid_tokens": (-1, 1)}, "valid <= padded"),
        ({"attention_valid_tokens": (5, 1)}, "valid <= padded"),
        ({"attention_padded_tokens": (4.0, 4)}, "valid <= padded"),
        ({"attention_padded_tokens": (1 << 31, 4)}, "INT32_MAX"),
        ({"native_tp_size": 2}, "TP coordinates"),
        (
            {
                "attention_valid_tokens": (1, 1),
                "attention_padded_tokens": (2, 4),
                "native_tp_size": 2,
            },
            "TP coordinates",
        ),
    ],
)
def test_stage_rejects_ambiguous_counts_and_tp_ownership(changes, reason):
    kwargs = dict(
        stage_idx=0,
        attention_valid_tokens=(3, 1),
        attention_padded_tokens=(4, 4),
        ffn_size=2,
    )

    with pytest.raises(ValueError, match=reason):
        HcclStageLayout(**(kwargs | changes))


def test_aggregation_rejects_a_total_larger_than_the_wire_count_capacity():
    with pytest.raises(ValueError, match="FFN padded count exceeds INT32"):
        HcclStageLayout(0, (1, 1), ((1 << 31) - 1, 1), 1)


@pytest.mark.parametrize("role_rank", [True, -1, 2])
@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_stage_peer_lookup_rejects_an_invalid_role_coordinate(role, role_rank):
    layout = HcclStageLayout(0, (1, 1), (1, 1), 2)

    with pytest.raises(ValueError, match="role_rank"):
        if role == "attention":
            layout.for_attention(role_rank)
        else:
            layout.for_ffn(role_rank)
