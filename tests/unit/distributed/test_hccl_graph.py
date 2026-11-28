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
    graph_capture_sequence_digest,
    graph_decision_payload,
    graph_key_payload,
)
from afd_plugin.distributed.hccl_plan import (
    HcclPlanMessage,
    decode_hccl_plan_message,
    encode_hccl_plan_message,
)
from tests.unit.fakes.hccl_graph_control import make_graph_candidate, make_graph_key


@pytest.mark.parametrize("query_len", [2, 3, 6])
def test_uniform_target_key_keeps_request_and_token_axes_distinct(query_len):
    tokens = 3 * query_len
    key = make_graph_key(
        rows=(tokens // 2, tokens - tokens // 2), target_query_len=query_len
    )

    assert decode_graph_key(graph_key_payload(key)) == key
    assert key.num_reqs == 3 and key.num_tokens == tokens
    assert key.max_query_len == key.uniform_token_count == query_len
    assert key.stage_layouts[0].request_stop >= key.stage_layouts[1].request_start
    with pytest.raises(ValueError, match="uniform target"):
        replace(key, num_reqs=key.num_tokens)


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


@pytest.mark.parametrize("change", ["order", "count", "later_key", "profile_prefix"])
def test_capture_sequence_binds_every_native_key_in_order_and_profile_prefix(change):
    keys = (make_graph_key(rows=(4,)), make_graph_key(rows=(2, 2)))
    original = graph_capture_sequence_digest(keys, max_full_descs_to_capture=None)
    alternative = {
        "order": (keys[::-1], None),
        "count": (keys[:1], None),
        "later_key": ((keys[0], make_graph_key(rows=(3, 3))), None),
        "profile_prefix": (keys, 2),
    }[change]
    assert original != graph_capture_sequence_digest(
        alternative[0], max_full_descs_to_capture=alternative[1]
    )
    assert original == graph_capture_sequence_digest(
        keys, max_full_descs_to_capture=None
    )


def test_preflight_sequence_round_trip_binds_query_without_changing_model_key():
    key = make_graph_key()
    ordinary = HcclGraphCandidate("replay", "warmup", key, (0,))
    candidate = replace(
        ordinary,
        capture_sequence_digest=graph_capture_sequence_digest(
            (key,), max_full_descs_to_capture=None
        ),
    )
    assert decode_graph_candidate(graph_candidate_payload(candidate)) == candidate
    assert candidate.digest != ordinary.digest
    assert candidate.key == ordinary.key
    assert "capture_sequence_digest" not in graph_candidate_payload(ordinary)
    assert decode_graph_candidate(graph_candidate_payload(ordinary)) == ordinary


def test_capture_sequence_survives_the_framed_graph_query_protocol():
    key = make_graph_key()
    candidate = HcclGraphCandidate(
        "replay",
        "warmup",
        key,
        (0,),
        graph_capture_sequence_digest((key,), max_full_descs_to_capture=1),
    )
    query = HcclPlanMessage(
        "GRAPH_QUERY", transaction_id="capture-preflight", graph_candidate=candidate
    )
    assert decode_hccl_plan_message(encode_hccl_plan_message(query)) == query


def test_empty_native_capture_list_uses_eager_preflight_without_an_invented_key():
    candidate = HcclGraphCandidate(
        "eager",
        "warmup",
        None,
        (0,),
        graph_capture_sequence_digest((), max_full_descs_to_capture=None),
    )
    assert decode_graph_candidate(graph_candidate_payload(candidate)) == candidate
    decision = decide_graph_execution(
        "empty-capture-preflight", candidate, (HcclGraphReport("unsupported"),)
    )
    assert (decision.phase, decision.key, decision.padded_tokens) == (
        "eager",
        None,
        (0,),
    )


@pytest.mark.parametrize("value", [None, True, 1, "", "A" * 64, "0" * 63])
def test_wire_preflight_rejects_noncanonical_capture_sequence_digest(value):
    payload = graph_candidate_payload(
        HcclGraphCandidate("replay", "warmup", make_graph_key(), (0,))
    )
    with pytest.raises(ValueError, match="capture sequence digest"):
        decode_graph_candidate({**payload, "capture_sequence_digest": value})


@pytest.mark.parametrize(
    "change", [{"phase": "capture"}, {"mode": "target"}, {"eager_padded_tokens": (1,)}]
)
def test_capture_sequence_is_only_valid_for_empty_capture_preflight(change):
    candidate = HcclGraphCandidate("replay", "warmup", make_graph_key(), (0,), "0" * 64)
    with pytest.raises(ValueError, match="empty capture preflight"):
        replace(candidate, **change)


@pytest.mark.parametrize("prefix", [-1, True, 1.0, "1"])
def test_capture_sequence_rejects_invalid_native_profile_prefix(prefix):
    with pytest.raises(ValueError, match="max_full_descs_to_capture"):
        graph_capture_sequence_digest(
            (make_graph_key(),), max_full_descs_to_capture=prefix
        )


@pytest.mark.parametrize("keys", [[], (None,)])
def test_capture_sequence_requires_concrete_graph_keys(keys):
    with pytest.raises(ValueError, match="tuple of graph keys"):
        graph_capture_sequence_digest(keys, max_full_descs_to_capture=None)
