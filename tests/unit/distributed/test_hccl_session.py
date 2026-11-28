# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Observable global control order; these tests execute no device behavior."""

from dataclasses import replace

import pytest

from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_plan import (
    MAX_ROLE_RANKS,
    MAX_TARGET_LAYERS,
    HcclParentPlan,
    HcclPlanCapabilities,
    HcclPlanMessage,
    decode_hccl_plan_message,
    encode_hccl_plan_message,
)
from afd_plugin.distributed.hccl_session import HcclPlanSession


@pytest.fixture
def session(capabilities):
    return HcclPlanSession(capabilities)


def accept_parent(session, plan, capabilities):
    for world_rank in range(capabilities.attention_size + capabilities.ffn_size):
        session.accept(
            HcclPlanMessage(
                "ACCEPT",
                world_rank=world_rank,
                transaction_id=plan.transaction_id,
                plan_digest=plan.digest,
            )
        )


def finish_parent(session, plan, capabilities):
    accept_parent(session, plan, capabilities)
    while session.next_input is not None:
        layer, stage = session.next_input.layer_idx, session.next_input.stage_idx
        exchanges = []
        while session.next_input is not None and (
            session.next_input.layer_idx,
            session.next_input.stage_idx,
        ) == (layer, stage):
            exchange = session.next_input
            session.begin_exchange(exchange)
            exchanges.append(exchange)
        for exchange in reversed(exchanges):
            session.finish_exchange(exchange)
    return session.complete(plan.transaction_id)


def test_parent_forward_and_complete_cross_the_actual_wire_codec(
    session, parent_plan, capabilities
):
    forward = session.begin(parent_plan)
    assert (
        decode_hccl_plan_message(encode_hccl_plan_message(forward)).plan == parent_plan
    )
    complete = finish_parent(session, parent_plan, capabilities)
    assert decode_hccl_plan_message(
        encode_hccl_plan_message(complete)
    ) == HcclPlanMessage(
        "COMPLETE",
        transaction_id=parent_plan.transaction_id,
        plan_digest=parent_plan.digest,
    )


def test_tensor_exchange_waits_for_every_parent_ack(session, parent_plan, capabilities):
    session.begin(parent_plan)
    exchange = session.next_input
    with pytest.raises(RuntimeError, match="ACCEPT"):
        session.begin_exchange(exchange)
    accept_parent(session, parent_plan, capabilities)
    session.begin_exchange(exchange)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("transaction_id", "wrong-parent", "transaction_id"),
        ("world_rank", 3, "topology"),
        ("plan_digest", "0" * 64, "digest"),
    ],
)
def test_peer_ack_binds_rank_to_the_complete_parent(
    session, parent_plan, field, value, error
):
    session.begin(parent_plan)
    ack = HcclPlanMessage(
        "ACCEPT",
        world_rank=0,
        transaction_id=parent_plan.transaction_id,
        plan_digest=parent_plan.digest,
    )
    with pytest.raises(ValueError, match=error):
        session.accept(replace(ack, **{field: value}))


def test_duplicate_ack_cannot_replace_another_rank(session, parent_plan):
    session.begin(parent_plan)
    ack = HcclPlanMessage(
        "ACCEPT",
        world_rank=0,
        transaction_id=parent_plan.transaction_id,
        plan_digest=parent_plan.digest,
    )
    session.accept(ack)
    with pytest.raises(ValueError, match="already accepted"):
        session.accept(ack)
    with pytest.raises(RuntimeError, match="ACCEPT"):
        session.begin_exchange(session.next_input)


@pytest.mark.parametrize(
    "field,value",
    [
        ("transaction_id", "wrong"),
        ("layer_idx", 3),
        ("stage_idx", 1),
        ("attention_world_rank", 1),
        ("ffn_world_rank", 1),
        ("valid_tokens", 0),
        ("padded_tokens", 5),
    ],
)
def test_wrong_input_identity_or_shape_is_rejected_before_advancing(
    session, parent_plan, capabilities, field, value
):
    session.begin(parent_plan)
    accept_parent(session, parent_plan, capabilities)
    exchange = session.next_input
    with pytest.raises(ValueError):
        session.begin_exchange(replace(exchange, **{field: value}))
    assert session.next_input == exchange
    session.begin_exchange(exchange)


def test_two_stages_can_have_pending_outputs_together(
    session, parent_plan, capabilities
):
    session.begin(parent_plan)
    accept_parent(session, parent_plan, capabilities)
    pending = []
    while session.next_input.layer_idx == 0:
        exchange = session.next_input
        session.begin_exchange(exchange)
        pending.append(exchange)
    assert [exchange.stage_idx for exchange in pending] == [0, 0, 1, 1]
    assert [exchange.valid_tokens for exchange in pending] == [2, 1, 0, 0]
    for exchange in reversed(pending):
        session.finish_exchange(exchange)
    assert session.next_input.layer_idx == 1


def test_fanout_waits_for_all_previous_outputs_before_next_attention_layer(
    session, parent_plan, capabilities
):
    plan = replace(parent_plan, stages=parent_plan.stages[:1])
    session.begin(plan)
    accept_parent(session, plan, capabilities)
    first = session.next_input
    session.begin_exchange(first)
    second = session.next_input
    session.begin_exchange(second)
    session.finish_exchange(first)
    with pytest.raises(RuntimeError, match="all previous"):
        session.begin_exchange(session.next_input)
    session.finish_exchange(second)
    session.begin_exchange(session.next_input)


@pytest.mark.parametrize("native_tp_size", [1, 2])
def test_ffn_requires_every_aggregate_or_native_tp_input_before_output(native_tp_size):
    capabilities = HcclPlanCapabilities(
        16, 128, 8, (0, 1), 2, native_tp_size, native_tp_size
    )
    plan = HcclParentPlan(
        "aggregate",
        (0, 1),
        (HcclStageLayout(0, (3, 3), (4, 4), native_tp_size, native_tp_size),),
    )
    session = HcclPlanSession(capabilities)
    session.begin(plan)
    accept_parent(session, plan, capabilities)
    first = session.next_input
    session.begin_exchange(first)
    with pytest.raises(RuntimeError, match="all aggregate"):
        session.finish_exchange(first)
    second = session.next_input
    session.begin_exchange(second)
    session.finish_exchange(first)
    session.finish_exchange(second)
    session.begin_exchange(session.next_input)


def test_native_tp_attention_waits_for_all_previous_coordinate_outputs():
    capabilities = HcclPlanCapabilities(16, 128, 8, (0, 1), 2, 2, 2)
    plan = HcclParentPlan(
        "tp2",
        (0, 1),
        (HcclStageLayout(0, (3, 3), (4, 4), 2, 2),),
    )
    session = HcclPlanSession(capabilities)
    session.begin(plan)
    accept_parent(session, plan, capabilities)
    first = session.next_input
    session.begin_exchange(first)
    second = session.next_input
    session.begin_exchange(second)
    session.finish_exchange(first)
    with pytest.raises(RuntimeError, match="all previous"):
        session.begin_exchange(session.next_input)
    session.finish_exchange(second)
    session.begin_exchange(session.next_input)


@pytest.mark.parametrize("padded", [(0,), (2,)])
def test_empty_stage_skips_only_when_it_has_no_transport_rows(
    session, parent_plan, capabilities, padded
):
    plan = replace(parent_plan, stages=(HcclStageLayout(0, (0,), padded, 2),))
    session.begin(plan)
    with pytest.raises(RuntimeError, match="ACCEPT"):
        session.complete(plan.transaction_id)
    accept_parent(session, plan, capabilities)
    if any(padded):
        with pytest.raises(RuntimeError, match="every stage/peer"):
            session.complete(plan.transaction_id)
        while session.next_input is not None:
            exchange = session.next_input
            session.begin_exchange(exchange)
            session.finish_exchange(exchange)
    assert session.complete(plan.transaction_id).plan_digest == plan.digest


@pytest.mark.parametrize("stage_idx", [0, 1])
def test_zero_native_rows_still_exchange_dummy_rows_when_other_peers_are_active(
    stage_idx,
):
    capabilities = HcclPlanCapabilities(16, 128, 8, (0,), 2, 1)
    stages = tuple(HcclStageLayout(i, (0, 0), (0, 0), 1) for i in range(stage_idx)) + (
        HcclStageLayout(stage_idx, (0, 1), (0, 1), 1),
    )
    plan = HcclParentPlan("empty-peer", (0,), stages)
    session = HcclPlanSession(capabilities)
    session.begin(plan)
    accept_parent(session, plan, capabilities)
    first = session.next_input
    assert (first.valid_tokens, first.padded_tokens) == (0, 1)
    session.begin_exchange(first)
    with pytest.raises(RuntimeError, match="aggregate"):
        session.finish_exchange(first)


@pytest.mark.parametrize(
    "operation", ["begin", "complete", "request_stop", "receive_stop"]
)
def test_active_parent_rejects_overlap_and_early_stop(session, parent_plan, operation):
    session.begin(parent_plan)
    with pytest.raises(RuntimeError):
        if operation == "begin":
            session.begin(replace(parent_plan, transaction_id="overlap"))
        elif operation == "complete":
            session.complete(parent_plan.transaction_id)
        elif operation == "request_stop":
            session.request_stop()
        else:
            session.receive_stop()


def test_output_must_match_a_pending_exchange(session, parent_plan, capabilities):
    session.begin(parent_plan)
    accept_parent(session, parent_plan, capabilities)
    first = session.next_input
    with pytest.raises(ValueError, match="pending"):
        session.finish_exchange(first)
    session.begin_exchange(first)
    with pytest.raises(ValueError, match="pending"):
        session.finish_exchange(replace(first, stage_idx=1))
    session.finish_exchange(first)
    with pytest.raises(ValueError, match="pending"):
        session.finish_exchange(first)


def test_last_pending_stage_prevents_parent_complete(
    session, parent_plan, capabilities
):
    session.begin(parent_plan)
    accept_parent(session, parent_plan, capabilities)
    while session.next_input is not None:
        exchange = session.next_input
        session.begin_exchange(exchange)
        if session.next_input is not None:
            session.finish_exchange(exchange)
    with pytest.raises(RuntimeError, match="every stage/peer"):
        session.complete(parent_plan.transaction_id)
    session.finish_exchange(exchange)
    assert session.complete(parent_plan.transaction_id).kind == "COMPLETE"


def test_completed_parent_cannot_be_replayed(session, parent_plan, capabilities):
    session.begin(parent_plan)
    finish_parent(session, parent_plan, capabilities)
    with pytest.raises(ValueError, match="already completed"):
        session.begin(parent_plan)
    next_plan = replace(parent_plan, transaction_id="parent-2")
    session.begin(next_plan)
    assert finish_parent(session, next_plan, capabilities).transaction_id == "parent-2"


def test_transaction_limit_requires_reconnect_without_forgetting_replay(
    capabilities, parent_plan
):
    session = HcclPlanSession(capabilities, max_transactions=1)
    session.begin(parent_plan)
    finish_parent(session, parent_plan, capabilities)
    with pytest.raises(ValueError, match="already completed"):
        session.begin(parent_plan)
    with pytest.raises(ValueError, match="reconnect"):
        session.begin(replace(parent_plan, transaction_id="parent-2"))


@pytest.mark.parametrize("operation", ["request_stop", "receive_stop"])
def test_stop_handshake_ends_the_session(session, parent_plan, operation):
    stop = (
        session.request_stop()
        if operation == "request_stop"
        else session.receive_stop()
    )
    assert stop.kind == ("STOP" if operation == "request_stop" else "STOP_ACK")
    if operation == "request_stop":
        with pytest.raises(RuntimeError, match="terminal or stopping"):
            session.begin(parent_plan)
        session.acknowledge_stop(HcclPlanMessage("STOP_ACK"))
    with pytest.raises(RuntimeError, match="terminal or stopping"):
        session.begin(parent_plan)


def test_unsolicited_stop_ack_is_rejected(session):
    with pytest.raises(RuntimeError, match="expected STOP_ACK"):
        session.acknowledge_stop(HcclPlanMessage("STOP_ACK"))


def test_abort_discards_pending_work_and_permanently_ends_session(
    session, parent_plan, capabilities
):
    session.begin(parent_plan)
    accept_parent(session, parent_plan, capabilities)
    exchange = session.next_input
    session.begin_exchange(exchange)
    error = session.abort("peer disconnected")
    assert (error.kind, error.transaction_id, error.error) == (
        "ERROR",
        parent_plan.transaction_id,
        "peer disconnected",
    )
    assert session.next_input is None
    with pytest.raises(RuntimeError, match="terminal"):
        session.finish_exchange(exchange)


@pytest.mark.parametrize("reason", ["", "x" * 4_097, "\ud800"])
def test_invalid_error_text_still_terminates_the_failed_session(
    session, parent_plan, reason
):
    session.begin(parent_plan)
    with pytest.raises(ValueError):
        session.abort(reason)
    assert session.next_input is None
    with pytest.raises(RuntimeError, match="terminal"):
        session.begin(parent_plan)
    with pytest.raises(RuntimeError, match="terminal"):
        session.request_stop()


def test_oversized_parent_is_rejected_before_starting_a_transaction():
    layers = tuple(range(1_000_000_000, 1_000_000_000 + MAX_TARGET_LAYERS))
    counts = (1_000_000,) * MAX_ROLE_RANKS
    capabilities = HcclPlanCapabilities(
        16, 128, 1_000_000, layers, MAX_ROLE_RANKS, MAX_ROLE_RANKS
    )
    plan = HcclParentPlan(
        "too-large",
        layers,
        tuple(HcclStageLayout(i, counts, counts, MAX_ROLE_RANKS) for i in range(2)),
    )
    session = HcclPlanSession(capabilities)
    with pytest.raises(ValueError, match="maximum size"):
        session.begin(plan)
    assert session.receive_stop().kind == "STOP_ACK"
