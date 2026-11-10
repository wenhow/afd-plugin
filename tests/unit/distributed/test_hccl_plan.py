# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Versioned parent-plan wire contracts, without device or runtime imports."""

import json
from dataclasses import replace

import pytest

from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_plan import (
    MAX_CONTROL_MESSAGE_BYTES,
    MAX_ROLE_RANKS,
    HcclLocalForward,
    HcclParentPlan,
    HcclPlanCapabilities,
    HcclPlanMessage,
    decode_hccl_plan_message,
    encode_hccl_plan_message,
)
from tests.unit.fakes.hccl_graph_control import make_graph_control_round


@pytest.fixture
def graph_round():
    return make_graph_control_round()


@pytest.mark.parametrize(
    "kind", ["GRAPH_QUERY", "GRAPH_REPORT", "GRAPH_DECISION", "GRAPH_ACK", "FORWARD"]
)
def test_graph_capability_messages_round_trip_with_reserved_geometry(graph_round, kind):
    case = graph_round
    messages = {
        "GRAPH_QUERY": case.query,
        "GRAPH_REPORT": case.report_messages[0],
        "GRAPH_DECISION": HcclPlanMessage(
            "GRAPH_DECISION", graph_decision=case.decision
        ),
        "GRAPH_ACK": case.acks[0],
        "FORWARD": HcclPlanMessage("FORWARD", plan=case.plan),
    }
    assert (
        decode_hccl_plan_message(encode_hccl_plan_message(messages[kind]))
        == messages[kind]
    )


def test_graph_capability_requires_exact_peer_version_and_local_counts(graph_round):
    cap = graph_round.capabilities
    with pytest.raises(ValueError, match="graph_execution_version"):
        cap.negotiate(replace(cap, graph_execution_version=0))
    with pytest.raises(ValueError, match="local_counts_version=1"):
        replace(cap, local_counts_version=0).validate()


@pytest.mark.parametrize("value", [True, 2, -1, "1"])
def test_graph_capability_rejects_non_version_integer(graph_round, value):
    with pytest.raises(ValueError, match="graph_execution_version"):
        replace(graph_round.capabilities, graph_execution_version=value).validate()


def test_graph_zero_keeps_pre_extension_hello_fields(capabilities):
    message = HcclPlanMessage(
        "HELLO", role="attention", world_rank=2, capabilities=capabilities
    )
    wire = json.loads(encode_hccl_plan_message(message))
    assert "graph_execution_version" not in wire["capabilities"]


@pytest.mark.parametrize("order", ["stage_serial", "bulk_v1", "cross_layer_v1"])
def test_graph_exchange_order_roundtrips_without_changing_default_hello(
    graph_round, order
):
    cap = replace(graph_round.capabilities, graph_exchange_order=order)
    message = HcclPlanMessage("HELLO", role="attention", world_rank=2, capabilities=cap)
    encoded = encode_hccl_plan_message(message)
    assert decode_hccl_plan_message(encoded) == message
    payload = json.loads(encoded)["capabilities"]
    assert ("graph_exchange_order" in payload) == (order != "stage_serial")
    if order == "stage_serial":
        assert encoded == encode_hccl_plan_message(
            replace(message, capabilities=graph_round.capabilities)
        )
    else:
        assert cap.negotiate(cap) == cap
        with pytest.raises(ValueError, match="graph_exchange_order"):
            cap.negotiate(graph_round.capabilities)
        with pytest.raises(ValueError, match="graph_exchange_order"):
            graph_round.capabilities.negotiate(cap)


@pytest.mark.parametrize("value", [None, True, 1, "bulk", "BULK_V1", [], {}])
def test_graph_exchange_order_rejects_unknown_or_non_string_wire_values(
    graph_round, value
):
    message = HcclPlanMessage(
        "HELLO", role="attention", world_rank=2, capabilities=graph_round.capabilities
    )
    payload = json.loads(encode_hccl_plan_message(message))
    payload["capabilities"]["graph_exchange_order"] = value
    with pytest.raises(ValueError, match="graph_exchange_order"):
        decode_hccl_plan_message(json.dumps(payload).encode())


@pytest.mark.parametrize(
    "change",
    [
        {"graph_execution_version": 0},
        {"max_stages": 1},
        {"native_tp_size": 4, "attention_size": 4, "ffn_size": 4},
    ],
)
@pytest.mark.parametrize("order", ["bulk_v1", "cross_layer_v1"])
def test_graph_bulk_requires_the_supported_static_session(graph_round, change, order):
    with pytest.raises(ValueError, match=order):
        replace(
            graph_round.capabilities, graph_exchange_order=order, **change
        ).validate()


def test_target_query_len_one_preserves_the_original_hello_bytes(capabilities):
    message = HcclPlanMessage(
        "HELLO", role="attention", world_rank=2, capabilities=capabilities
    )
    expected = (
        b'{"capabilities":{"attention_size":1,"ffn_size":2,"hidden_size":16,'
        b'"layer_indices":[0,1,3],"max_stages":2,"max_tokens":8,'
        b'"native_tp_size":1,"vocab_size":128},"kind":"HELLO",'
        b'"payload_format":"bf16-fp32-int32-per-layer","role":"attention",'
        b'"schedule":"layer-major","version":3,"world_rank":2}'
    )
    assert encode_hccl_plan_message(message) == expected
    assert decode_hccl_plan_message(expected) == message
    assert capabilities.target_query_len == 1


@pytest.mark.parametrize("query_len", [1, 2, 3, 8])
@pytest.mark.parametrize("native_tp", [1, 2])
@pytest.mark.parametrize("order", ["stage_serial", "bulk_v1", "cross_layer_v1"])
def test_target_query_len_roundtrips_explicit_target_session(
    graph_round, query_len, native_tp, order
):
    cap = replace(
        graph_round.capabilities,
        attention_size=native_tp,
        ffn_size=native_tp,
        native_tp_size=native_tp,
        target_query_len=query_len,
        graph_exchange_order=order,
    )
    message = HcclPlanMessage(
        "HELLO", role="attention", world_rank=native_tp, capabilities=cap
    )
    encoded = encode_hccl_plan_message(message)
    assert decode_hccl_plan_message(encoded) == message
    assert cap.negotiate(cap) == cap
    wire_cap = json.loads(encoded)["capabilities"]
    assert ("target_query_len" in wire_cap) == (query_len != 1)
    if query_len != 1:
        assert wire_cap["target_query_len"] == query_len


@pytest.mark.parametrize("query_len", [True, False, None, 0, -1, 9, 1.5, "3", [], {}])
@pytest.mark.parametrize("through_wire", [False, True])
def test_target_query_len_rejects_invalid_or_over_capacity_values(
    graph_round, query_len, through_wire
):
    cap = graph_round.capabilities
    with pytest.raises(ValueError, match="target_query_len"):
        if through_wire:
            message = HcclPlanMessage(
                "HELLO", role="attention", world_rank=2, capabilities=cap
            )
            wire = json.loads(encode_hccl_plan_message(message))
            wire["capabilities"]["target_query_len"] = query_len
            decode_hccl_plan_message(json.dumps(wire).encode())
        else:
            replace(cap, target_query_len=query_len).validate()


@pytest.mark.parametrize(
    "change",
    [
        {"graph_execution_version": 0},
        {"local_counts_version": 0},
        {"native_tp_size": 4, "attention_size": 4, "ffn_size": 4},
        {"native_tp_size": 1},
        {"attention_size": 1},
        {"ffn_size": 1},
    ],
)
def test_target_query_len_above_one_requires_the_supported_target_world(
    graph_round, change
):
    with pytest.raises(ValueError, match="target_query_len > 1 requires"):
        replace(graph_round.capabilities, target_query_len=3, **change).validate()


@pytest.mark.parametrize("local_query_len,peer_query_len", [(1, 3), (3, 1), (2, 3)])
def test_target_query_len_cannot_downgrade_or_change_across_peers(
    graph_round, local_query_len, peer_query_len
):
    local = replace(graph_round.capabilities, target_query_len=local_query_len)
    peer = replace(graph_round.capabilities, target_query_len=peer_query_len)
    with pytest.raises(ValueError, match="peer target_query_len does not match"):
        local.negotiate(peer)
    with pytest.raises(ValueError, match="peer target_query_len does not match"):
        peer.negotiate(local)


def test_target_query_len_is_preserved_when_capacity_and_stages_are_bounded(
    graph_round,
):
    local = replace(graph_round.capabilities, target_query_len=3)
    peer = replace(local, max_tokens=3, max_stages=1)
    assert local.negotiate(peer) == peer
    assert peer.negotiate(local) == peer


def test_graph_parent_digest_binds_phase_and_static_key(graph_round):
    plan = graph_round.plan
    changed = replace(
        plan, graph_execution=replace(plan.graph_execution, phase="capture")
    )
    assert changed.digest != plan.digest


@pytest.mark.parametrize("change", ["id", "mode", "rows", "capability"])
def test_graph_parent_rejects_unreserved_actual_plan(graph_round, change):
    case = graph_round
    with pytest.raises(ValueError, match="graph"):
        if change == "id":
            replace(case.plan, transaction_id="stale")
        elif change == "mode":
            replace(case.plan, mode="dummy")
        elif change == "rows":
            replace(
                case.plan,
                stages=(
                    HcclStageLayout(0, (1, 1), (1, 1), 2, 2),
                    *case.plan.stages[1:],
                ),
            )
        else:
            case.plan.validate(replace(case.capabilities, graph_execution_version=0))


@pytest.mark.parametrize("kind", ["GRAPH_QUERY", "GRAPH_REPORT", "GRAPH_ACK"])
def test_graph_message_rejects_unknown_nested_fields(graph_round, kind):
    message = {
        "GRAPH_QUERY": graph_round.query,
        "GRAPH_REPORT": graph_round.report_messages[0],
        "GRAPH_ACK": graph_round.acks[0],
    }[kind]
    wire = json.loads(encode_hccl_plan_message(message))
    wire["unexpected"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        decode_hccl_plan_message(json.dumps(wire).encode())


@pytest.fixture
def forward_wire(parent_plan):
    return json.loads(
        encode_hccl_plan_message(HcclPlanMessage("FORWARD", plan=parent_plan))
    )


def test_capabilities_negotiate_stage_and_buffer_limits_symmetrically(capabilities):
    peer = replace(capabilities, max_tokens=4, max_stages=1)
    assert capabilities.negotiate(peer) == peer
    assert peer.negotiate(capabilities) == peer


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 32),
        ("vocab_size", 256),
        ("layer_indices", (0, 1)),
        ("attention_size", 2),
        ("ffn_size", 1),
    ],
)
def test_capabilities_reject_peer_contract_mismatch(capabilities, field, value):
    with pytest.raises(ValueError, match=field):
        capabilities.negotiate(replace(capabilities, **{field: value}))


def test_capabilities_reject_different_native_tp_coordinate_groups():
    local = HcclPlanCapabilities(16, 128, 8, (0,), 2, 2, native_tp_size=1)
    with pytest.raises(ValueError, match="native_tp_size"):
        local.negotiate(replace(local, native_tp_size=2))


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", True),
        ("vocab_size", (1 << 31) + 1),
        ("max_tokens", 0),
        ("max_tokens", 1.0),
        ("attention_size", MAX_ROLE_RANKS + 1),
        ("ffn_size", False),
        ("native_tp_size", 0),
        ("max_stages", 3),
        ("max_stages", True),
        ("local_counts_version", True),
        ("local_counts_version", 2),
        ("layer_indices", (1, 0)),
        ("layer_indices", (0, 0)),
        ("layer_indices", ()),
        ("layer_indices", (False,)),
    ],
)
def test_capabilities_reject_invalid_wire_dimensions(capabilities, field, value):
    with pytest.raises(ValueError):
        replace(capabilities, **{field: value}).validate()


@pytest.mark.parametrize("version", [0, 1])
def test_local_counts_extension_roundtrip_preserves_baseline_hello(
    capabilities, version
):
    local = replace(
        capabilities,
        attention_size=2,
        ffn_size=2,
        native_tp_size=2,
        local_counts_version=version,
    )
    message = HcclPlanMessage(
        "HELLO", role="attention", world_rank=2, capabilities=local
    )
    encoded = encode_hccl_plan_message(message)
    assert ("local_counts_version" in json.loads(encoded)["capabilities"]) == bool(
        version
    )
    assert decode_hccl_plan_message(encoded) == message
    assert local.negotiate(local) == local


@pytest.mark.parametrize("local_version", [0, 1])
def test_local_counts_extension_cannot_silently_downgrade(capabilities, local_version):
    local = replace(
        capabilities,
        attention_size=2,
        ffn_size=2,
        native_tp_size=2,
        local_counts_version=local_version,
    )
    peer = replace(local, local_counts_version=1 - local_version)
    with pytest.raises(ValueError, match="local_counts_version"):
        local.negotiate(peer)


@pytest.mark.parametrize("attention,ffn,tp", [(2, 2, 1), (1, 2, 1), (2, 1, 1)])
def test_local_counts_extension_rejects_unproven_native_role_worlds(
    capabilities,
    attention,
    ffn,
    tp,
):
    local = replace(
        capabilities,
        attention_size=attention,
        ffn_size=ffn,
        native_tp_size=tp,
        local_counts_version=1,
    )
    with pytest.raises(ValueError, match="one native TP world per role"):
        local.validate()


@pytest.mark.parametrize(
    "local",
    [
        HcclLocalForward("target", (1,), (2,)),
        HcclLocalForward("dummy", (0, 0), (4, 2)),
        HcclLocalForward("profile", (0,), (0,)),
    ],
)
def test_solicited_count_messages_roundtrip_actual_local_extents(local):
    request = HcclPlanMessage("COUNTS_REQUEST", transaction_id="counts-1")
    report = HcclPlanMessage(
        "COUNTS_REPORT", transaction_id="counts-1", world_rank=3, local_forward=local
    )
    assert decode_hccl_plan_message(encode_hccl_plan_message(request)) == request
    assert decode_hccl_plan_message(encode_hccl_plan_message(report)) == report


@pytest.mark.parametrize(
    "local",
    [
        HcclLocalForward("graph", (1,), (2,)),
        HcclLocalForward("target", (), ()),
        HcclLocalForward("target", [1], (2,)),
        HcclLocalForward("target", (1,), (2, 2)),
        HcclLocalForward("target", (0, 0, 0), (0, 0, 0)),
        HcclLocalForward("target", (True,), (2,)),
        HcclLocalForward("target", (1,), (False,)),
        HcclLocalForward("target", (3,), (2,)),
        HcclLocalForward("target", (-1,), (2,)),
        HcclLocalForward("target", (1,), ((1 << 31),)),
    ],
)
def test_count_report_rejects_invalid_native_extents(local):
    with pytest.raises(ValueError):
        encode_hccl_plan_message(
            HcclPlanMessage(
                "COUNTS_REPORT",
                transaction_id="counts-1",
                world_rank=3,
                local_forward=local,
            )
        )


@pytest.mark.parametrize("field", ["transaction_id", "world_rank", "local_forward"])
def test_count_report_decoder_rejects_missing_identity_or_counts(field):
    encoded = json.loads(
        encode_hccl_plan_message(
            HcclPlanMessage(
                "COUNTS_REPORT",
                transaction_id="counts-1",
                world_rank=3,
                local_forward=HcclLocalForward("target", (1,), (2,)),
            )
        )
    )
    del encoded[field]
    with pytest.raises(ValueError, match="missing or unexpected"):
        decode_hccl_plan_message(json.dumps(encoded).encode())


@pytest.mark.parametrize("attention,ffn,tp", [(2, 3, 1), (4, 2, 2), (2, 2, 3)])
def test_capabilities_reject_unsupported_topologies(capabilities, attention, ffn, tp):
    with pytest.raises(ValueError):
        replace(
            capabilities, attention_size=attention, ffn_size=ffn, native_tp_size=tp
        ).validate()


def test_parent_plan_roundtrip_reconstructs_peer_rows_and_padding(
    capabilities, parent_plan
):
    message = HcclPlanMessage("FORWARD", plan=parent_plan)
    decoded = decode_hccl_plan_message(encode_hccl_plan_message(message))
    assert decoded == message
    decoded.plan.validate(capabilities)
    assert tuple(
        (p.valid_tokens, p.padded_tokens) for p in decoded.plan.stages[1].peer_slices
    ) == ((0, 1), (0, 1))
    assert decoded.plan.digest == parent_plan.digest


@pytest.mark.parametrize("stage_count", [1, 2])
def test_empty_parent_is_a_valid_explicit_zero_tensor_plan(capabilities, stage_count):
    plan = HcclParentPlan(
        "empty",
        (0, 1, 3),
        tuple(HcclStageLayout(i, (0,), (0,), 2) for i in range(stage_count)),
    )
    plan.validate(capabilities)
    assert all(
        peer.padded_tokens == 0 for stage in plan.stages for peer in stage.peer_slices
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"transaction_id": ""},
        {"transaction_id": "界" * 86},
        {"transaction_id": "\ud800"},
        {"mode": "draft"},
        {"layer_indices": (1, 0)},
        {"stages": ()},
        {"stages": []},
        {"stages": (HcclStageLayout(1, (1,), (1,), 2),)},
        {
            "stages": (
                HcclStageLayout(0, (1,), (1,), 1),
                HcclStageLayout(1, (1,), (1,), 2),
            )
        },
        {
            "stages": (
                HcclStageLayout(0, (1,), (1,), 2),
                HcclStageLayout(0, (1,), (1,), 2),
            )
        },
    ],
)
def test_parent_plan_rejects_ambiguous_stage_identity(parent_plan, changes):
    with pytest.raises(ValueError):
        replace(parent_plan, **changes)


@pytest.mark.parametrize(
    "field,value",
    [("max_stages", 1), ("max_tokens", 3), ("layer_indices", (0,)), ("ffn_size", 1)],
)
def test_parent_plan_rejects_negotiated_contract_overrun(
    capabilities, parent_plan, field, value
):
    with pytest.raises(ValueError):
        parent_plan.validate(replace(capabilities, **{field: value}))


def test_parent_capacity_includes_ffn_aggregate_padding():
    cap = HcclPlanCapabilities(16, 128, 4, (0,), 2, 1)
    plan = HcclParentPlan("aggregate", (0,), (HcclStageLayout(0, (0, 4), (0, 4), 1),))
    with pytest.raises(ValueError, match="transport rows"):
        plan.validate(cap)


@pytest.mark.parametrize(
    "field,value",
    [
        ("transaction_id", "different"),
        ("mode", "dummy"),
        ("layer_indices", (0, 2)),
        ("stages", (HcclStageLayout(0, (2,), (4,), 2),)),
    ],
)
def test_parent_acceptance_digest_binds_full_plan(parent_plan, field, value):
    assert replace(parent_plan, **{field: value}).digest != parent_plan.digest


@pytest.mark.parametrize(
    "kind", ["HELLO", "FORWARD", "ACCEPT", "COMPLETE", "STOP", "STOP_ACK", "ERROR"]
)
def test_all_parent_control_messages_roundtrip(capabilities, parent_plan, kind):
    messages = {
        "HELLO": HcclPlanMessage(
            "HELLO", role="attention", world_rank=2, capabilities=capabilities
        ),
        "FORWARD": HcclPlanMessage("FORWARD", plan=parent_plan),
        "ACCEPT": HcclPlanMessage(
            "ACCEPT",
            transaction_id=parent_plan.transaction_id,
            plan_digest=parent_plan.digest,
            world_rank=0,
        ),
        "COMPLETE": HcclPlanMessage(
            "COMPLETE",
            transaction_id=parent_plan.transaction_id,
            plan_digest=parent_plan.digest,
        ),
        "STOP": HcclPlanMessage("STOP"),
        "STOP_ACK": HcclPlanMessage("STOP_ACK"),
        "ERROR": HcclPlanMessage(
            "ERROR", error="错误", transaction_id=parent_plan.transaction_id
        ),
    }
    message = messages[kind]
    assert decode_hccl_plan_message(encode_hccl_plan_message(message)) == message


def test_stop_wire_pins_version_payload_precision_and_schedule():
    assert (
        encode_hccl_plan_message(HcclPlanMessage("STOP"))
        == b'{"kind":"STOP","payload_format":"bf16-fp32-int32-per-layer",'
        b'"schedule":"layer-major","version":3}'
    )


@pytest.mark.parametrize(
    "role,rank",
    [("attention", 0), ("ffn", 2), ("attention", True), ("ffn", -1), ("unknown", 0)],
)
def test_hello_rejects_wrong_role_coordinate(capabilities, role, rank):
    with pytest.raises(ValueError):
        encode_hccl_plan_message(
            HcclPlanMessage(
                "HELLO", role=role, world_rank=rank, capabilities=capabilities
            )
        )


@pytest.mark.parametrize(
    "message",
    [
        HcclPlanMessage("STOP", transaction_id="unexpected"),
        HcclPlanMessage("ACCEPT"),
        HcclPlanMessage("FORWARD", plan="wrong"),
        HcclPlanMessage("ERROR", error=""),
        HcclPlanMessage(
            "ACCEPT", transaction_id="t", world_rank=0, plan_digest="A" * 64
        ),
        HcclPlanMessage(
            "ACCEPT", transaction_id="t", world_rank=True, plan_digest="0" * 64
        ),
        HcclPlanMessage("COMPLETE", transaction_id="t", plan_digest="0" * 63),
        HcclPlanMessage("ERROR", error="x" * 4097),
    ],
)
def test_control_messages_reject_missing_or_wrong_payload(message):
    with pytest.raises(ValueError):
        encode_hccl_plan_message(message)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("version", 3.0),
        ("version", True),
        ("payload_format", "bf16-only"),
        ("schedule", "stage-major"),
        ("unexpected", 1),
        ("kind", "UNKNOWN"),
    ],
)
def test_decoder_rejects_incompatible_envelope(forward_wire, field, value):
    forward_wire[field] = value
    with pytest.raises(ValueError):
        decode_hccl_plan_message(json.dumps(forward_wire).encode())


@pytest.mark.parametrize(
    "field,value",
    [
        ("peer_slices", []),
        ("attention_padded_tokens", [True]),
        ("attention_valid_tokens", [5]),
        ("native_tp_size", 0),
        ("ffn_size", MAX_ROLE_RANKS + 1),
        ("stage_idx", 1),
    ],
)
def test_decoder_rejects_untrusted_stage_offsets_and_dimensions(
    forward_wire, field, value
):
    forward_wire["plan"]["stages"][0][field] = value
    with pytest.raises(ValueError):
        decode_hccl_plan_message(json.dumps(forward_wire).encode())


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\xff",
        b"[]",
        b"{",
        b"null",
        b" " * (MAX_CONTROL_MESSAGE_BYTES + 1),
        b'{"version":3,"version":3}',
        b'{"version":3,"value":NaN}',
        b'{"version":3,"value":Infinity}',
        b"[" * 1500 + b"]" * 1500,
    ],
)
def test_decoder_rejects_malformed_or_ambiguous_json(data):
    with pytest.raises(ValueError):
        decode_hccl_plan_message(data)


def test_decoder_rejects_duplicate_native_count_fields(forward_wire):
    encoded = (
        json.dumps(forward_wire)
        .encode()
        .replace(
            b'"attention_valid_tokens": [3]',
            b'"attention_valid_tokens": [3], "attention_valid_tokens": [2]',
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        decode_hccl_plan_message(encoded)


def test_encoder_rejects_parent_larger_than_control_frame():
    layers = tuple(range((1 << 31) - 4096, 1 << 31))
    counts = ((1 << 31) - 1,) * MAX_ROLE_RANKS
    plan = HcclParentPlan(
        "bounded",
        layers,
        tuple(
            HcclStageLayout(index, counts, counts, MAX_ROLE_RANKS) for index in range(2)
        ),
    )
    with pytest.raises(ValueError, match="maximum size"):
        encode_hccl_plan_message(HcclPlanMessage("FORWARD", plan=plan))


@pytest.mark.parametrize(
    "left,right", [("bulk_v1", "cross_layer_v1"), ("cross_layer_v1", "bulk_v1")]
)
def test_graph_peers_never_downgrade_between_bulk_and_cross_layer(
    graph_round, left, right
):
    cap = graph_round.capabilities
    with pytest.raises(ValueError, match="graph_exchange_order"):
        replace(cap, graph_exchange_order=left).negotiate(
            replace(cap, graph_exchange_order=right)
        )
