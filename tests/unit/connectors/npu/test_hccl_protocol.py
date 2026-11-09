# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Wire contracts without importing the runtime-heavy connector namespaces."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def protocol():
    # Both parent connector __init__ modules import device runtimes. Load the
    # actual standalone protocol so these CPU tests also run without torch.
    module_name = "_afd_unit_hccl_protocol"
    module_path = (
        Path(__file__).resolve().parents[4]
        / "afd_plugin/connectors/npu/hccl_protocol.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[module_name]


@pytest.fixture
def capabilities(protocol):
    return protocol.HcclCapabilities(
        hidden_size=16, vocab_size=128, max_tokens=8, layer_indices=(0, 1, 3)
    )


@pytest.fixture
def forward_wire():
    return {
        "version": 2,
        "kind": "FORWARD",
        "plan": {
            "transaction_id": "target-1",
            "layer_indices": [0, 1, 3],
            "valid_tokens": 3,
            "padded_tokens": 4,
            "mode": "target",
            "stage_idx": 0,
        },
    }


def test_capabilities_negotiate_smaller_capacity_symmetrically(protocol, capabilities):
    peer = replace(capabilities, max_tokens=4)

    assert capabilities.negotiate(peer) == peer
    assert peer.negotiate(capabilities) == peer


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden_size", 0),
        ("hidden_size", True),
        ("hidden_size", 16.0),
        ("vocab_size", -1),
        ("vocab_size", (1 << 31) + 1),
        ("max_tokens", 0),
        ("max_tokens", 1 << 31),
        ("native_tp_size", 0),
        ("native_tp_size", True),
        ("pair_rank", -1),
        ("pair_rank", 1),
        ("pair_rank", False),
        ("layer_indices", ()),
        ("layer_indices", [0, 1]),
        ("layer_indices", (0, 0)),
        ("layer_indices", (1, 0)),
        ("layer_indices", (-1,)),
        ("layer_indices", (True,)),
        ("layer_indices", (0, 1 << 31)),
    ],
)
def test_capabilities_reject_invalid_dimensions(protocol, capabilities, field, value):
    with pytest.raises(ValueError):
        replace(capabilities, **{field: value}).validate()


def test_capabilities_allow_full_nonnegative_int32_id_range(protocol, capabilities):
    replace(capabilities, vocab_size=1 << 31).validate()


def test_capabilities_bound_layer_count(protocol, capabilities):
    with pytest.raises(ValueError, match="layer_indices"):
        replace(
            capabilities, layer_indices=tuple(range(protocol.MAX_TARGET_LAYERS + 1))
        ).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [("hidden_size", 32), ("vocab_size", 256), ("layer_indices", (0, 1))],
)
def test_capabilities_reject_peer_model_mismatch(protocol, capabilities, field, value):
    with pytest.raises(ValueError, match=field):
        capabilities.negotiate(replace(capabilities, **{field: value}))


@pytest.mark.parametrize("field,value", [("native_tp_size", 4), ("pair_rank", 1)])
def test_handshake_rejects_a_different_native_tp_group_or_pair(
    protocol, capabilities, field, value
):
    local = replace(capabilities, native_tp_size=2)
    peer = replace(local, **{field: value})
    with pytest.raises(ValueError, match=field):
        local.negotiate(peer)


def test_hello_roundtrip_preserves_native_tp_coordinate(protocol, capabilities):
    capabilities = replace(capabilities, native_tp_size=8, pair_rank=7)
    message = protocol.HcclControlMessage(
        protocol.HcclMessageType.HELLO,
        role=protocol.HcclRole.ATTENTION,
        capabilities=capabilities,
    )
    assert (
        protocol.decode_hccl_control_message(
            protocol.encode_hccl_control_message(message)
        )
        == message
    )


def test_decoder_rejects_previous_single_pair_wire_version(protocol):
    with pytest.raises(ValueError, match="unsupported HCCL protocol version"):
        protocol.decode_hccl_control_message(b'{"version":1,"kind":"STOP"}')


@pytest.mark.parametrize(
    ("mode", "valid_tokens", "padded_tokens"),
    [
        ("target", 3, 4),
        ("target", 0, 0),
        ("dummy", 0, 4),
        ("warmup", 0, 4),
        ("profile", 0, 4),
        ("profile", 4, 4),
    ],
)
def test_forward_modes_preserve_valid_and_execution_rows(
    protocol, capabilities, mode, valid_tokens, padded_tokens
):
    plan = protocol.HcclForwardPlan(
        transaction_id="target-1",
        layer_indices=capabilities.layer_indices,
        valid_tokens=valid_tokens,
        padded_tokens=padded_tokens,
        mode=mode,
    )
    plan.validate(capabilities)

    message = protocol.HcclControlMessage(protocol.HcclMessageType.FORWARD, plan=plan)
    assert (
        protocol.decode_hccl_control_message(
            protocol.encode_hccl_control_message(message)
        )
        == message
    )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("transaction_id", "", "transaction_id"),
        ("transaction_id", "  ", "transaction_id"),
        ("transaction_id", 1, "transaction_id"),
        ("transaction_id", "汉" * 86, "transaction_id"),
        ("transaction_id", "\ud800", "UTF-8"),
        ("layer_indices", (0, 1), "layer_indices"),
        ("layer_indices", (1, 0), "strictly increasing"),
        ("valid_tokens", -1, "valid_tokens"),
        ("valid_tokens", 5, "valid_tokens"),
        ("valid_tokens", False, "valid_tokens"),
        ("padded_tokens", 9, "max_tokens"),
        ("padded_tokens", 4.0, "padded_tokens"),
        ("stage_idx", 1, "stage_idx"),
        ("stage_idx", False, "stage_idx"),
        ("mode", "graph", "mode"),
    ],
)
def test_forward_plan_rejects_invalid_contract(
    protocol, capabilities, field, value, reason
):
    plan = protocol.HcclForwardPlan(
        transaction_id="target-1",
        layer_indices=capabilities.layer_indices,
        valid_tokens=3,
        padded_tokens=4,
    )

    with pytest.raises(ValueError, match=reason):
        replace(plan, **{field: value}).validate(capabilities)


def test_empty_target_cannot_execute_padding_rows(protocol, capabilities):
    plan = protocol.HcclForwardPlan(
        transaction_id="target-1",
        layer_indices=capabilities.layer_indices,
        valid_tokens=0,
        padded_tokens=1,
    )

    with pytest.raises(ValueError, match="empty target"):
        plan.validate(capabilities)


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_hello_wire_contains_model_and_capacity_contract(protocol, capabilities, role):
    message = protocol.HcclControlMessage(
        protocol.HcclMessageType.HELLO,
        role=protocol.HcclRole(role),
        capabilities=capabilities,
    )
    encoded = protocol.encode_hccl_control_message(message)

    assert json.loads(encoded) == {
        "version": 2,
        "kind": "HELLO",
        "role": role,
        "capabilities": {
            "hidden_size": 16,
            "vocab_size": 128,
            "max_tokens": 8,
            "layer_indices": [0, 1, 3],
            "native_tp_size": 1,
            "pair_rank": 0,
        },
    }
    assert protocol.decode_hccl_control_message(encoded) == message


def test_forward_wire_interoperates_with_primitive_json(
    protocol, capabilities, forward_wire
):
    decoded = protocol.decode_hccl_control_message(json.dumps(forward_wire).encode())

    assert decoded.kind is protocol.HcclMessageType.FORWARD
    assert decoded.plan == protocol.HcclForwardPlan(
        transaction_id="target-1",
        layer_indices=(0, 1, 3),
        valid_tokens=3,
        padded_tokens=4,
    )
    decoded.plan.validate(capabilities)
    assert json.loads(protocol.encode_hccl_control_message(decoded)) == forward_wire


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("ACCEPT", {"transaction_id": "target-1"}),
        ("COMPLETE", {"transaction_id": "target-1"}),
        ("STOP", {}),
        ("STOP_ACK", {}),
        ("ERROR", {"error": "invalid capacity"}),
        ("ERROR", {"error": "计算失败", "transaction_id": "target-1"}),
    ],
)
def test_control_reply_round_trip(protocol, kind, payload):
    message = protocol.HcclControlMessage(protocol.HcclMessageType(kind), **payload)

    assert (
        protocol.decode_hccl_control_message(
            protocol.encode_hccl_control_message(message)
        )
        == message
    )


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("HELLO", {}),
        ("HELLO", {"role": "attention", "capabilities": 1}),
        ("FORWARD", {}),
        ("FORWARD", {"plan": "target-1"}),
        ("ACCEPT", {}),
        ("ACCEPT", {"transaction_id": "target-1", "error": "extra"}),
        ("COMPLETE", {"transaction_id": ""}),
        ("STOP", {"transaction_id": "target-1"}),
        ("STOP_ACK", {"error": "extra"}),
        ("ERROR", {}),
        ("ERROR", {"error": ""}),
        ("ERROR", {"error": "failed", "plan": "extra"}),
        ("ERROR", {"error": "failed", "transaction_id": False}),
    ],
)
def test_control_message_rejects_mismatched_payload(protocol, kind, payload):
    message = protocol.HcclControlMessage(protocol.HcclMessageType(kind), **payload)

    with pytest.raises(ValueError):
        protocol.encode_hccl_control_message(message)


def test_control_message_requires_typed_kind(protocol):
    with pytest.raises(ValueError, match="kind"):
        protocol.HcclControlMessage("STOP").validate()


def test_error_text_is_bounded_before_encoding(protocol):
    message = protocol.HcclControlMessage(
        protocol.HcclMessageType.ERROR, error="x" * (protocol.MAX_ERROR_BYTES + 1)
    )

    with pytest.raises(ValueError, match="error exceeds"):
        protocol.encode_hccl_control_message(message)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\xff",
        b"[]",
        b"null",
        b"{",
        b'{"version":2,"kind":"STOP","kind":"STOP_ACK"}',
        b'{"version":2,"kind":"STOP","extra":NaN}',
        b'{"version":2,"kind":"STOP","extra":Infinity}',
        b'{"version":2,"kind":"STOP","extra":-Infinity}',
        b"[" * 2_000 + b"]" * 2_000,
    ],
)
def test_decoder_rejects_invalid_or_ambiguous_json(protocol, data):
    with pytest.raises(ValueError):
        protocol.decode_hccl_control_message(data)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "STOP"},
        {"version": True, "kind": "STOP"},
        {"version": 2.0, "kind": "STOP"},
        {"version": 3, "kind": "STOP"},
        {"version": 2, "kind": "unknown"},
        {"version": 2, "kind": False},
        {"version": 2, "kind": "STOP", "error": None},
        {"version": 2, "kind": "ACCEPT"},
        {"version": 2, "kind": "ACCEPT", "transaction_id": 1},
        {"version": 2, "kind": "ERROR", "error": "failed", "transaction_id": None},
        {"version": 2, "kind": "HELLO", "role": "invalid", "capabilities": {}},
    ],
)
def test_decoder_rejects_invalid_envelope(protocol, payload):
    with pytest.raises(ValueError):
        protocol.decode_hccl_control_message(json.dumps(payload).encode())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transaction_id", None),
        ("valid_tokens", True),
        ("valid_tokens", "3"),
        ("padded_tokens", 4.0),
        ("layer_indices", {"0": 0}),
        ("layer_indices", [0, False]),
        ("stage_idx", 1),
        ("mode", "graph"),
        ("unexpected", None),
    ],
)
def test_decoder_rejects_invalid_plan_fields(protocol, forward_wire, field, value):
    forward_wire["plan"][field] = value

    with pytest.raises(ValueError):
        protocol.decode_hccl_control_message(json.dumps(forward_wire).encode())


def test_decoder_requires_all_plan_fields(protocol, forward_wire):
    del forward_wire["plan"]["valid_tokens"]

    with pytest.raises(ValueError, match="missing"):
        protocol.decode_hccl_control_message(json.dumps(forward_wire).encode())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hidden_size", True),
        ("vocab_size", "128"),
        ("max_tokens", 0),
        ("layer_indices", [1, 0]),
        ("extra", 1),
    ],
)
def test_decoder_rejects_invalid_capabilities(protocol, field, value):
    wire_capabilities = {
        "hidden_size": 16,
        "vocab_size": 128,
        "max_tokens": 8,
        "layer_indices": [0, 1, 3],
        "native_tp_size": 1,
        "pair_rank": 0,
    }
    wire_capabilities[field] = value
    payload = {
        "version": 2,
        "kind": "HELLO",
        "role": "attention",
        "capabilities": wire_capabilities,
    }

    with pytest.raises(ValueError):
        protocol.decode_hccl_control_message(json.dumps(payload).encode())


def test_decoder_rejects_duplicate_nested_fields(protocol):
    encoded = (
        b'{"version":2,"kind":"HELLO","role":"ffn","capabilities":'
        b'{"hidden_size":16,"vocab_size":128,"max_tokens":8,'
        b'"max_tokens":4,"layer_indices":[0]}}'
    )

    with pytest.raises(ValueError, match="duplicate"):
        protocol.decode_hccl_control_message(encoded)


def test_decoder_accepts_exact_size_boundary_and_rejects_larger(protocol):
    encoded = b'{"version":2,"kind":"STOP"}'
    bounded = encoded + b" " * (protocol.MAX_CONTROL_MESSAGE_BYTES - len(encoded))

    assert protocol.decode_hccl_control_message(bounded) == protocol.HcclControlMessage(
        protocol.HcclMessageType.STOP
    )
    with pytest.raises(ValueError, match="bounded"):
        protocol.decode_hccl_control_message(bounded + b" ")
