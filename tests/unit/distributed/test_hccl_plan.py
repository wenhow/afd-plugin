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
    HcclParentPlan,
    HcclPlanCapabilities,
    HcclPlanMessage,
    decode_hccl_plan_message,
    encode_hccl_plan_message,
)


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
        ("layer_indices", (1, 0)),
        ("layer_indices", (0, 0)),
        ("layer_indices", ()),
        ("layer_indices", (False,)),
    ],
)
def test_capabilities_reject_invalid_wire_dimensions(capabilities, field, value):
    with pytest.raises(ValueError):
        replace(capabilities, **{field: value}).validate()


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
