# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only wire contracts for the eager, U1 HCCL component.

Version 2 verifies each pair's native TP size and coordinate. The data plane uses
BF16 activations, independently produced FP32
router inputs, INT32 raw token IDs, and complete BF16 FFN outputs. All tensors
use the plan's padded token count; valid tokens occupy its leading rows. This
module owns validation and encoding, while the transport owns message order,
transaction reuse, and process-group lifetime. It does not extend the existing
DP-metadata protocol or enable the model runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

HCCL_PROTOCOL_VERSION = 2
MAX_CONTROL_MESSAGE_BYTES = 65_536
MAX_TRANSACTION_ID_BYTES = 256
MAX_ERROR_BYTES = 4_096
MAX_TARGET_LAYERS = 4_096
MAX_INT32 = (1 << 31) - 1
FORWARD_MODES = frozenset({"target", "dummy", "warmup", "profile"})

JsonValue: TypeAlias = (
    bool | int | float | str | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class HcclRole(str, Enum):
    ATTENTION = "attention"
    FFN = "ffn"


class HcclMessageType(str, Enum):
    HELLO = "HELLO"
    FORWARD = "FORWARD"
    ACCEPT = "ACCEPT"
    COMPLETE = "COMPLETE"
    STOP = "STOP"
    STOP_ACK = "STOP_ACK"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class HcclCapabilities:
    hidden_size: int
    vocab_size: int
    max_tokens: int
    layer_indices: tuple[int, ...]
    native_tp_size: int = 1
    pair_rank: int = 0

    def validate(self) -> None:
        _validate_integer("hidden_size", self.hidden_size, minimum=1)
        # The largest vocabulary ID, rather than its size, must fit INT32.
        _validate_integer(
            "vocab_size", self.vocab_size, minimum=1, maximum=MAX_INT32 + 1
        )
        _validate_integer("max_tokens", self.max_tokens, minimum=1)
        _validate_layers(self.layer_indices)
        _validate_integer("native_tp_size", self.native_tp_size, minimum=1)
        _validate_integer("pair_rank", self.pair_rank, maximum=self.native_tp_size - 1)

    def negotiate(self, peer: HcclCapabilities) -> HcclCapabilities:
        """Match the model contract and use the smaller receive capacity."""
        self.validate()
        peer.validate()
        for name, local_value, peer_value in (
            ("hidden_size", self.hidden_size, peer.hidden_size),
            ("vocab_size", self.vocab_size, peer.vocab_size),
            ("layer_indices", self.layer_indices, peer.layer_indices),
            ("native_tp_size", self.native_tp_size, peer.native_tp_size),
            ("pair_rank", self.pair_rank, peer.pair_rank),
        ):
            if local_value != peer_value:
                raise ValueError(f"HCCL peer {name} does not match")
        return HcclCapabilities(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            max_tokens=min(self.max_tokens, peer.max_tokens),
            layer_indices=self.layer_indices,
            native_tp_size=self.native_tp_size,
            pair_rank=self.pair_rank,
        )


@dataclass(frozen=True, slots=True)
class HcclForwardPlan:
    transaction_id: str
    layer_indices: tuple[int, ...]
    valid_tokens: int
    padded_tokens: int
    mode: str = "target"
    stage_idx: int = 0

    def _validate_fields(self) -> None:
        _validate_text("transaction_id", self.transaction_id, MAX_TRANSACTION_ID_BYTES)
        _validate_layers(self.layer_indices)
        _validate_integer("valid_tokens", self.valid_tokens)
        _validate_integer("padded_tokens", self.padded_tokens)
        _validate_integer("stage_idx", self.stage_idx, maximum=0)
        if not isinstance(self.mode, str) or self.mode not in FORWARD_MODES:
            raise ValueError("mode must be target, dummy, warmup, or profile")
        if self.valid_tokens > self.padded_tokens:
            raise ValueError("valid_tokens cannot exceed padded_tokens")
        if self.mode == "target" and self.valid_tokens == 0 and self.padded_tokens:
            raise ValueError("an empty target must have padded_tokens=0")

    def validate(self, capabilities: HcclCapabilities) -> None:
        self._validate_fields()
        capabilities.validate()
        if self.layer_indices != capabilities.layer_indices:
            raise ValueError("plan layer_indices do not match negotiated target layers")
        if self.padded_tokens > capabilities.max_tokens:
            raise ValueError("padded_tokens exceeds negotiated max_tokens")


@dataclass(frozen=True, slots=True)
class HcclControlMessage:
    kind: HcclMessageType
    role: HcclRole | None = None
    capabilities: HcclCapabilities | None = None
    plan: HcclForwardPlan | None = None
    transaction_id: str | None = None
    error: str | None = None

    def validate(self) -> None:
        if not isinstance(self.kind, HcclMessageType):
            raise ValueError("kind must be an HcclMessageType")
        present = {
            name
            for name, value in (
                ("role", self.role),
                ("capabilities", self.capabilities),
                ("plan", self.plan),
                ("transaction_id", self.transaction_id),
                ("error", self.error),
            )
            if value is not None
        }
        if self.kind is HcclMessageType.HELLO:
            if present != {"role", "capabilities"}:
                raise ValueError("HELLO requires only role and capabilities")
            if not isinstance(self.role, HcclRole):
                raise ValueError("HELLO role must be an HcclRole")
            if not isinstance(self.capabilities, HcclCapabilities):
                raise ValueError("HELLO capabilities must be HcclCapabilities")
            self.capabilities.validate()
        elif self.kind is HcclMessageType.FORWARD:
            if present != {"plan"} or not isinstance(self.plan, HcclForwardPlan):
                raise ValueError("FORWARD requires only an HcclForwardPlan")
            self.plan._validate_fields()
        elif self.kind in (HcclMessageType.ACCEPT, HcclMessageType.COMPLETE):
            if present != {"transaction_id"}:
                raise ValueError(f"{self.kind.value} requires only transaction_id")
            _validate_text(
                "transaction_id", self.transaction_id, MAX_TRANSACTION_ID_BYTES
            )
        elif self.kind in (HcclMessageType.STOP, HcclMessageType.STOP_ACK):
            if present:
                raise ValueError(f"{self.kind.value} does not accept payload fields")
        else:
            if "error" not in present or present - {"error", "transaction_id"}:
                raise ValueError("ERROR requires error and optional transaction_id")
            _validate_text("error", self.error, MAX_ERROR_BYTES)
            if self.transaction_id is not None:
                _validate_text(
                    "transaction_id", self.transaction_id, MAX_TRANSACTION_ID_BYTES
                )


def _validate_integer(
    name: str, value: JsonValue, *, minimum: int = 0, maximum: int = MAX_INT32
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _validate_text(name: str, value: JsonValue, max_bytes: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain valid UTF-8 text") from exc
    if size > max_bytes:
        raise ValueError(f"{name} exceeds {max_bytes} UTF-8 bytes")
    return value


def _validate_layers(layer_indices: tuple[int, ...]) -> None:
    if not isinstance(layer_indices, tuple) or not (
        1 <= len(layer_indices) <= MAX_TARGET_LAYERS
    ):
        raise ValueError(
            f"layer_indices must be a tuple of 1 to {MAX_TARGET_LAYERS} layers"
        )
    previous = -1
    for layer_idx in layer_indices:
        _validate_integer("layer_idx", layer_idx)
        if layer_idx <= previous:
            raise ValueError("layer_indices must be strictly increasing")
        previous = layer_idx


def encode_hccl_control_message(message: HcclControlMessage) -> bytes:
    """Encode a validated message without serializing runtime-owned objects."""
    message.validate()
    payload: dict[str, JsonValue] = {
        "version": HCCL_PROTOCOL_VERSION,
        "kind": message.kind.value,
    }
    if message.role is not None:
        payload["role"] = message.role.value
    if message.capabilities is not None:
        capabilities = message.capabilities
        payload["capabilities"] = {
            "hidden_size": capabilities.hidden_size,
            "vocab_size": capabilities.vocab_size,
            "max_tokens": capabilities.max_tokens,
            "layer_indices": list(capabilities.layer_indices),
            "native_tp_size": capabilities.native_tp_size,
            "pair_rank": capabilities.pair_rank,
        }
    if message.plan is not None:
        plan = message.plan
        payload["plan"] = {
            "transaction_id": plan.transaction_id,
            "layer_indices": list(plan.layer_indices),
            "valid_tokens": plan.valid_tokens,
            "padded_tokens": plan.padded_tokens,
            "mode": plan.mode,
            "stage_idx": plan.stage_idx,
        }
    if message.transaction_id is not None:
        payload["transaction_id"] = message.transaction_id
    if message.error is not None:
        payload["error"] = message.error
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > MAX_CONTROL_MESSAGE_BYTES:
        raise ValueError("HCCL control message exceeds maximum size")
    return encoded


def _unique_json_fields(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate HCCL JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite HCCL JSON value: {value}")


def _require_fields(
    payload: dict[str, JsonValue], required: set[str], optional: set[str] | None = None
) -> None:
    present = set(payload)
    if required - present or present - required - (optional or set()):
        raise ValueError("HCCL message has missing or unexpected fields")


def _decode_capabilities(value: JsonValue) -> HcclCapabilities:
    if not isinstance(value, dict):
        raise ValueError("capabilities must be a JSON object")
    _require_fields(
        value,
        {
            "hidden_size",
            "vocab_size",
            "max_tokens",
            "layer_indices",
            "native_tp_size",
            "pair_rank",
        },
    )
    capabilities = HcclCapabilities(
        hidden_size=_validate_integer("hidden_size", value["hidden_size"], minimum=1),
        vocab_size=_validate_integer(
            "vocab_size", value["vocab_size"], minimum=1, maximum=MAX_INT32 + 1
        ),
        max_tokens=_validate_integer("max_tokens", value["max_tokens"], minimum=1),
        layer_indices=_decode_layers(value["layer_indices"]),
        native_tp_size=_validate_integer(
            "native_tp_size", value["native_tp_size"], minimum=1
        ),
        pair_rank=_validate_integer("pair_rank", value["pair_rank"]),
    )
    capabilities.validate()
    return capabilities


def _decode_layers(value: JsonValue) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError("layer_indices must be a JSON array")
    layers = tuple(_validate_integer("layer_idx", layer_idx) for layer_idx in value)
    _validate_layers(layers)
    return layers


def _decode_plan(value: JsonValue) -> HcclForwardPlan:
    if not isinstance(value, dict):
        raise ValueError("plan must be a JSON object")
    _require_fields(
        value,
        {
            "transaction_id",
            "layer_indices",
            "valid_tokens",
            "padded_tokens",
            "mode",
            "stage_idx",
        },
    )
    plan = HcclForwardPlan(
        transaction_id=_validate_text(
            "transaction_id", value["transaction_id"], MAX_TRANSACTION_ID_BYTES
        ),
        layer_indices=_decode_layers(value["layer_indices"]),
        valid_tokens=_validate_integer("valid_tokens", value["valid_tokens"]),
        padded_tokens=_validate_integer("padded_tokens", value["padded_tokens"]),
        mode=_validate_text("mode", value["mode"], MAX_CONTROL_MESSAGE_BYTES),
        stage_idx=_validate_integer("stage_idx", value["stage_idx"], maximum=0),
    )
    plan._validate_fields()
    return plan


def decode_hccl_control_message(data: bytes) -> HcclControlMessage:
    """Reject oversized or ambiguous JSON before interpreting its fields."""
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_CONTROL_MESSAGE_BYTES:
        raise ValueError("HCCL control message must be bounded nonempty bytes")
    try:
        payload: JsonValue = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_json_fields,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid HCCL control JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("HCCL control message must be a JSON object")
    if type(payload.get("version")) is not int or (
        payload["version"] != HCCL_PROTOCOL_VERSION
    ):
        raise ValueError("unsupported HCCL protocol version")
    if not isinstance(payload.get("kind"), str):
        raise ValueError("HCCL message kind must be a string")
    kind = HcclMessageType(payload["kind"])
    envelope_fields = {"version", "kind"}
    if kind is HcclMessageType.HELLO:
        _require_fields(payload, envelope_fields | {"role", "capabilities"})
        if not isinstance(payload["role"], str):
            raise ValueError("HELLO role must be a string")
        message = HcclControlMessage(
            kind=kind,
            role=HcclRole(payload["role"]),
            capabilities=_decode_capabilities(payload["capabilities"]),
        )
    elif kind is HcclMessageType.FORWARD:
        _require_fields(payload, envelope_fields | {"plan"})
        message = HcclControlMessage(kind=kind, plan=_decode_plan(payload["plan"]))
    elif kind in (HcclMessageType.ACCEPT, HcclMessageType.COMPLETE):
        _require_fields(payload, envelope_fields | {"transaction_id"})
        message = HcclControlMessage(
            kind=kind,
            transaction_id=_validate_text(
                "transaction_id", payload["transaction_id"], MAX_TRANSACTION_ID_BYTES
            ),
        )
    elif kind in (HcclMessageType.STOP, HcclMessageType.STOP_ACK):
        _require_fields(payload, envelope_fields)
        message = HcclControlMessage(kind=kind)
    else:
        _require_fields(payload, envelope_fields | {"error"}, {"transaction_id"})
        transaction_id = (
            _validate_text(
                "transaction_id", payload["transaction_id"], MAX_TRANSACTION_ID_BYTES
            )
            if "transaction_id" in payload
            else None
        )
        message = HcclControlMessage(
            kind=kind,
            transaction_id=transaction_id,
            error=_validate_text("error", payload["error"], MAX_ERROR_BYTES),
        )
    message.validate()
    return message


__all__ = [
    "HCCL_PROTOCOL_VERSION",
    "MAX_CONTROL_MESSAGE_BYTES",
    "MAX_TRANSACTION_ID_BYTES",
    "MAX_ERROR_BYTES",
    "MAX_TARGET_LAYERS",
    "HcclRole",
    "HcclMessageType",
    "HcclCapabilities",
    "HcclForwardPlan",
    "HcclControlMessage",
    "encode_hccl_control_message",
    "decode_hccl_control_message",
]
