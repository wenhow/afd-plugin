# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Graph reservation geometry and global decision seams, CPU-only."""

from dataclasses import replace

import pytest

from afd_plugin.distributed.hccl_graph import (
    HcclGraphCandidate,
    HcclGraphReport,
    HcclGraphStage,
    decide_graph_execution,
    decode_graph_candidate,
    decode_graph_decision,
    decode_graph_key,
    graph_candidate_payload,
    graph_decision_payload,
    graph_key_payload,
)
from tests.unit.fakes.hccl_graph_control import make_graph_candidate, make_graph_key


@pytest.mark.parametrize("rows", [(4,), (2, 2), (0, 4), (4, 0)])
def test_static_key_round_trip_preserves_empty_stage_pattern(rows):
    key = make_graph_key(rows=rows, metadata_shapes=((1, 4), (2, 0)))
    assert decode_graph_key(graph_key_payload(key)) == key
    assert key.padded_tokens == rows


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_tokens", True),
        ("num_tokens", 0),
        ("num_reqs", 3),
        ("uniform_token_count", 2),
        ("max_query_len", 2),
        ("num_active_loras", 1),
        ("num_ubatches", True),
        ("num_ubatches", 1),
        ("stage_layouts", (HcclGraphStage(0, 2, 1, 2), HcclGraphStage(2, 4, 2, 4))),
        ("stage_layouts", (HcclGraphStage(0, 2, 0, 2), HcclGraphStage(2, 5, 2, 4))),
        ("metadata_shapes", ((True,),)),
        ("metadata_shapes", ((),)),
        ("metadata_shapes", [[1]]),
    ],
)
def test_static_key_rejects_unsupported_or_inconsistent_geometry(field, value):
    with pytest.raises(ValueError):
        replace(make_graph_key(), **{field: value})


def test_request_layout_is_part_of_static_key_even_with_equal_token_rows():
    key = make_graph_key(rows=(4, 4))
    different_requests = replace(
        key, stage_layouts=(key.stage_layouts[0], HcclGraphStage(3, 8, 4, 8))
    )
    assert key.digest != different_requests.digest


@pytest.mark.parametrize(
    "phase,states,expected,reason",
    [
        ("replay", ("ready", "ready"), "replay", "requested"),
        ("replay", ("ready", "missing"), "eager", "cache_miss"),
        ("replay", ("missing", "unsupported"), "eager", "unsupported"),
        ("eager", ("unsupported", "unsupported"), "eager", "requested"),
        ("warmup", ("missing", "missing"), "warmup", "requested"),
        ("capture", ("missing", "missing"), "capture", "requested"),
    ],
)
def test_phase_selection_requires_all_peers_ready_before_replay(
    phase, states, expected, reason
):
    candidate = make_graph_candidate(phase=phase, eager_rows=(1, 1))
    decision = decide_graph_execution(
        "forward-1", candidate, tuple(HcclGraphReport(state) for state in states)
    )
    assert decision.phase == expected
    assert decision.reason == reason
    assert decision.padded_tokens == ((1, 1) if expected == "eager" else (2, 2))
    assert decode_graph_candidate(graph_candidate_payload(candidate)) == candidate
    assert decode_graph_decision(graph_decision_payload(decision)) == decision


@pytest.mark.parametrize("phase", ["warmup", "capture"])
@pytest.mark.parametrize("state", ["ready", "unsupported"])
def test_capture_phase_rejects_duplicate_ready_or_unsupported_peer(phase, state):
    with pytest.raises(ValueError, match="uncaptured supported key"):
        decide_graph_execution(
            "forward-1", make_graph_candidate(phase=phase), (HcclGraphReport(state),)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "FULL"),
        ("mode", "capture"),
        ("key", None),
        ("eager_padded_tokens", (True,)),
        ("eager_padded_tokens", ()),
        ("mode", "profile"),
    ],
)
def test_candidate_rejects_ambiguous_phase_or_unbounded_fallback(field, value):
    with pytest.raises(ValueError):
        replace(make_graph_candidate(), **{field: value})


def test_phase_is_separate_from_native_forward_purpose():
    candidate = HcclGraphCandidate("capture", "dummy", make_graph_key(), (4,))
    assert candidate.phase == "capture"
    assert candidate.mode == "dummy"


def test_runtime_transaction_identity_does_not_change_static_key():
    candidate = make_graph_candidate()
    first = decide_graph_execution("one", candidate, (HcclGraphReport("ready"),))
    second = replace(first, transaction_id="two")
    assert first.key == second.key
    assert first.digest != second.digest


@pytest.mark.parametrize(
    "change",
    [
        {"unexpected": 1},
        {"num_tokens": False},
        {"stage_layouts": {}},
        {"metadata_shapes": [1]},
    ],
)
def test_wire_key_rejects_unknown_fields_and_wrong_types(change):
    with pytest.raises(ValueError):
        decode_graph_key({**graph_key_payload(make_graph_key()), **change})
