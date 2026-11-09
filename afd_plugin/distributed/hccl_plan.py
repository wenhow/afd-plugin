# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only version 3 parent plans; not selected by the current HCCL runtime.

Version 2 remains a separate, strict U1 wire contract. Peers must explicitly
select version 3 before using this codec; an old HELLO cannot negotiate it.
Only native counts cross the wire. Each receiver derives peer offsets locally
using the same layout contract before accepting a parent plan.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TypeAlias

from afd_plugin.distributed.hccl import MAX_HCCL_STAGES, HcclStageLayout
from afd_plugin.distributed.hccl_graph import (
    HcclGraphCandidate,
    HcclGraphDecision,
    HcclGraphReport,
    decode_graph_candidate,
    decode_graph_decision,
    graph_candidate_payload,
    graph_decision_payload,
)

HCCL_PLAN_PROTOCOL_VERSION = 3
MAX_ROLE_RANKS = 1_024
MAX_CONTROL_MESSAGE_BYTES = 65_536
MAX_TRANSACTION_ID_BYTES = 256
MAX_ERROR_BYTES = 4_096
MAX_TARGET_LAYERS = 4_096
MAX_INT32 = (1 << 31) - 1
PAYLOAD_FORMAT = "bf16-fp32-int32-per-layer"
LAYER_SCHEDULE = "layer-major"
FORWARD_MODES = frozenset({"target", "dummy", "warmup", "profile"})
MESSAGE_KINDS = frozenset(
    {
        "HELLO",
        "FORWARD",
        "ACCEPT",
        "COMPLETE",
        "STOP",
        "STOP_ACK",
        "ERROR",
        "COUNTS_REQUEST",
        "COUNTS_REPORT",
        "GRAPH_QUERY",
        "GRAPH_REPORT",
        "GRAPH_DECISION",
        "GRAPH_ACK",
    }
)

JsonValue: TypeAlias = (
    bool | int | float | str | None | list["JsonValue"] | dict[str, "JsonValue"]
)


def _integer(
    name: str, value: JsonValue, *, minimum: int = 0, maximum: int = MAX_INT32
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _text(name: str, value: JsonValue, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must contain valid UTF-8 text") from exc
    if size > maximum:
        raise ValueError(f"{name} exceeds {maximum} UTF-8 bytes")
    return value


def _layers(layers: tuple[int, ...]) -> None:
    if not isinstance(layers, tuple) or not 1 <= len(layers) <= MAX_TARGET_LAYERS:
        raise ValueError("layer_indices must be a bounded nonempty tuple")
    previous = -1
    for layer in layers:
        _integer("layer_idx", layer)
        if layer <= previous:
            raise ValueError("layer_indices must be strictly increasing")
        previous = layer


@dataclass(frozen=True, slots=True)
class HcclPlanCapabilities:
    hidden_size: int
    vocab_size: int
    max_tokens: int
    layer_indices: tuple[int, ...]
    attention_size: int
    ffn_size: int
    native_tp_size: int = 1
    max_stages: int = MAX_HCCL_STAGES
    local_counts_version: int = 0
    graph_execution_version: int = 0

    def validate(self) -> None:
        _integer("hidden_size", self.hidden_size, minimum=1)
        _integer("vocab_size", self.vocab_size, minimum=1, maximum=MAX_INT32 + 1)
        _integer("max_tokens", self.max_tokens, minimum=1)
        _integer(
            "attention_size", self.attention_size, minimum=1, maximum=MAX_ROLE_RANKS
        )
        _integer("ffn_size", self.ffn_size, minimum=1, maximum=MAX_ROLE_RANKS)
        _integer(
            "native_tp_size", self.native_tp_size, minimum=1, maximum=MAX_ROLE_RANKS
        )
        _integer("max_stages", self.max_stages, minimum=1, maximum=MAX_HCCL_STAGES)
        _integer("local_counts_version", self.local_counts_version, maximum=1)
        _integer("graph_execution_version", self.graph_execution_version, maximum=1)
        if self.graph_execution_version and self.local_counts_version != 1:
            raise ValueError("graph execution requires local_counts_version=1")
        if self.local_counts_version == 1 and not (
            self.attention_size == self.ffn_size == self.native_tp_size
        ):
            raise ValueError(
                "local count collection requires one native TP world per role"
            )
        _layers(self.layer_indices)
        empty = (0,) * self.attention_size
        HcclStageLayout(0, empty, empty, self.ffn_size, self.native_tp_size)

    def negotiate(self, peer: HcclPlanCapabilities) -> HcclPlanCapabilities:
        """Match model/topology and symmetrically bound stages and capacity."""
        self.validate()
        peer.validate()
        for name, local, remote in (
            ("hidden_size", self.hidden_size, peer.hidden_size),
            ("vocab_size", self.vocab_size, peer.vocab_size),
            ("layer_indices", self.layer_indices, peer.layer_indices),
            ("attention_size", self.attention_size, peer.attention_size),
            ("ffn_size", self.ffn_size, peer.ffn_size),
            ("native_tp_size", self.native_tp_size, peer.native_tp_size),
            (
                "local_counts_version",
                self.local_counts_version,
                peer.local_counts_version,
            ),
            (
                "graph_execution_version",
                self.graph_execution_version,
                peer.graph_execution_version,
            ),
        ):
            if local != remote:
                raise ValueError(f"HCCL peer {name} does not match")
        return HcclPlanCapabilities(
            self.hidden_size,
            self.vocab_size,
            min(self.max_tokens, peer.max_tokens),
            self.layer_indices,
            self.attention_size,
            self.ffn_size,
            self.native_tp_size,
            min(self.max_stages, peer.max_stages),
            self.local_counts_version,
            self.graph_execution_version,
        )


@dataclass(frozen=True, slots=True)
class HcclLocalForward:
    """One Attention worker's actual ordered native stage extents.

    Validation runs inside the solicited report round, so a bad local report
    can reject every peer instead of raising before its blocking reply.
    """

    mode: str
    valid_tokens: tuple[int, ...]
    padded_tokens: tuple[int, ...]

    def validate(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in FORWARD_MODES:
            raise ValueError("local mode must be target, dummy, warmup, or profile")
        if (
            not isinstance(self.valid_tokens, tuple)
            or not isinstance(self.padded_tokens, tuple)
            or not 1 <= len(self.valid_tokens) <= MAX_HCCL_STAGES
            or len(self.valid_tokens) != len(self.padded_tokens)
        ):
            raise ValueError("local counts require one or two matching stage tuples")
        for valid, padded in zip(self.valid_tokens, self.padded_tokens, strict=True):
            _integer("valid_tokens", valid)
            _integer("padded_tokens", padded)
            if valid > padded:
                raise ValueError("local counts must satisfy valid <= padded")


@dataclass(frozen=True, slots=True)
class HcclParentPlan:
    transaction_id: str
    layer_indices: tuple[int, ...]
    stages: tuple[HcclStageLayout, ...]
    mode: str = "target"
    graph_execution: HcclGraphDecision | None = None

    def __post_init__(self) -> None:
        _text("transaction_id", self.transaction_id, MAX_TRANSACTION_ID_BYTES)
        _layers(self.layer_indices)
        if not isinstance(self.mode, str) or self.mode not in FORWARD_MODES:
            raise ValueError("mode must be target, dummy, warmup, or profile")
        if (
            not isinstance(self.stages, tuple)
            or not 1 <= len(self.stages) <= MAX_HCCL_STAGES
        ):
            raise ValueError("parent plan requires one or two stages")
        topology: tuple[int, int, int] | None = None
        for index, stage in enumerate(self.stages):
            if not isinstance(stage, HcclStageLayout) or stage.stage_idx != index:
                raise ValueError(
                    "parent stages must be ordered consecutively from zero"
                )
            current = (
                len(stage.attention_valid_tokens),
                stage.ffn_size,
                stage.native_tp_size,
            )
            if max(current) > MAX_ROLE_RANKS:
                raise ValueError("parent topology exceeds role rank limit")
            if topology is not None and current != topology:
                raise ValueError("all parent stages must use the same topology")
            topology = current
        if self.graph_execution is not None:
            decision = self.graph_execution
            if not isinstance(decision, HcclGraphDecision):
                raise ValueError("graph_execution must be HcclGraphDecision")
            if (
                decision.transaction_id != self.transaction_id
                or decision.mode != self.mode
                or len(decision.padded_tokens) != len(self.stages)
                or any(
                    any(rows != expected for rows in stage.attention_padded_tokens)
                    for stage, expected in zip(
                        self.stages, decision.padded_tokens, strict=True
                    )
                )
            ):
                raise ValueError("parent plan does not match reserved graph execution")

    def validate(self, capabilities: HcclPlanCapabilities) -> None:
        capabilities.validate()
        if bool(capabilities.graph_execution_version) != (
            self.graph_execution is not None
        ):
            raise ValueError(
                "parent graph execution does not match negotiated capability"
            )
        if self.layer_indices != capabilities.layer_indices:
            raise ValueError("plan layer_indices do not match negotiated target layers")
        if len(self.stages) > capabilities.max_stages:
            raise ValueError("parent stage count exceeds negotiated max_stages")
        for stage in self.stages:
            if (
                len(stage.attention_valid_tokens) != capabilities.attention_size
                or stage.ffn_size != capabilities.ffn_size
                or stage.native_tp_size != capabilities.native_tp_size
            ):
                raise ValueError("plan topology does not match negotiated capabilities")
            attention_rows = [0] * capabilities.attention_size
            ffn_rows = [0] * capabilities.ffn_size
            for peer in stage.peer_slices:
                attention_rows[peer.attention_world_rank - stage.ffn_size] += (
                    peer.padded_tokens
                )
                ffn_rows[peer.ffn_world_rank] += peer.padded_tokens
            if max((*attention_rows, *ffn_rows)) > capabilities.max_tokens:
                raise ValueError("stage transport rows exceed negotiated max_tokens")

    @property
    def digest(self) -> str:
        """Bind every peer's acceptance to the entire canonical parent plan."""
        payload: dict[str, JsonValue] = {
            "version": HCCL_PLAN_PROTOCOL_VERSION,
            "payload_format": PAYLOAD_FORMAT,
            "schedule": LAYER_SCHEDULE,
            "plan": _plan_payload(self),
        }
        return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class HcclPlanMessage:
    kind: str
    role: str | None = None
    world_rank: int | None = None
    capabilities: HcclPlanCapabilities | None = None
    plan: HcclParentPlan | None = None
    transaction_id: str | None = None
    plan_digest: str | None = None
    error: str | None = None
    local_forward: HcclLocalForward | None = None
    graph_candidate: HcclGraphCandidate | None = None
    graph_report: HcclGraphReport | None = None
    graph_decision: HcclGraphDecision | None = None
    graph_digest: str | None = None

    def validate(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in MESSAGE_KINDS:
            raise ValueError("unsupported HCCL plan message kind")
        present = {
            key
            for key, value in (
                ("role", self.role),
                ("world_rank", self.world_rank),
                ("capabilities", self.capabilities),
                ("plan", self.plan),
                ("transaction_id", self.transaction_id),
                ("plan_digest", self.plan_digest),
                ("error", self.error),
                ("local_forward", self.local_forward),
                ("graph_candidate", self.graph_candidate),
                ("graph_report", self.graph_report),
                ("graph_decision", self.graph_decision),
                ("graph_digest", self.graph_digest),
            )
            if value is not None
        }
        required: set[str]
        if self.kind == "HELLO":
            required = {"role", "world_rank", "capabilities"}
        elif self.kind == "FORWARD":
            required = {"plan"}
        elif self.kind == "ACCEPT":
            required = {"transaction_id", "plan_digest", "world_rank"}
        elif self.kind == "COMPLETE":
            required = {"transaction_id", "plan_digest"}
        elif self.kind == "COUNTS_REQUEST":
            required = {"transaction_id"}
        elif self.kind == "COUNTS_REPORT":
            required = {"transaction_id", "world_rank", "local_forward"}
        elif self.kind == "GRAPH_QUERY":
            required = {"transaction_id", "graph_candidate"}
        elif self.kind == "GRAPH_REPORT":
            required = {"transaction_id", "world_rank", "graph_digest", "graph_report"}
        elif self.kind == "GRAPH_DECISION":
            required = {"graph_decision"}
        elif self.kind == "GRAPH_ACK":
            required = {"transaction_id", "world_rank", "graph_digest"}
        elif self.kind == "ERROR":
            required = {"error"} | (
                {"transaction_id"} if self.transaction_id is not None else set()
            )
        else:
            required = set()
        if present != required:
            raise ValueError(f"{self.kind} has missing or unexpected payload fields")
        if self.capabilities is not None:
            if not isinstance(self.capabilities, HcclPlanCapabilities):
                raise ValueError("HELLO capabilities must be HcclPlanCapabilities")
            self.capabilities.validate()
            if self.role not in ("attention", "ffn"):
                raise ValueError("HELLO requires attention or ffn role")
            world_rank = _integer(
                "world_rank", self.world_rank, maximum=2 * MAX_ROLE_RANKS - 1
            )
            first = self.capabilities.ffn_size if self.role == "attention" else 0
            count = (
                self.capabilities.attention_size
                if self.role == "attention"
                else self.capabilities.ffn_size
            )
            if not first <= world_rank < first + count:
                raise ValueError("HELLO world_rank does not match its role")
        elif self.world_rank is not None:
            _integer("world_rank", self.world_rank, maximum=2 * MAX_ROLE_RANKS - 1)
        if self.plan is not None and not isinstance(self.plan, HcclParentPlan):
            raise ValueError("FORWARD requires an HcclParentPlan")
        if self.local_forward is not None:
            if not isinstance(self.local_forward, HcclLocalForward):
                raise ValueError("COUNTS_REPORT requires an HcclLocalForward")
            self.local_forward.validate()
        for value, expected in (
            (self.graph_candidate, HcclGraphCandidate),
            (self.graph_report, HcclGraphReport),
            (self.graph_decision, HcclGraphDecision),
        ):
            if value is not None and not isinstance(value, expected):
                raise ValueError("graph message has an invalid record type")
        if self.graph_digest is not None and (
            not isinstance(self.graph_digest, str)
            or len(self.graph_digest) != 64
            or any(
                character not in "0123456789abcdef" for character in self.graph_digest
            )
        ):
            raise ValueError("graph_digest must be a lowercase SHA256 digest")
        if self.transaction_id is not None:
            _text("transaction_id", self.transaction_id, MAX_TRANSACTION_ID_BYTES)
        if self.plan_digest is not None and (
            not isinstance(self.plan_digest, str)
            or len(self.plan_digest) != hashlib.sha256().digest_size * 2
            or any(
                character not in "0123456789abcdef" for character in self.plan_digest
            )
        ):
            raise ValueError("plan_digest must be a lowercase SHA256 digest")
        if self.error is not None:
            _text("error", self.error, MAX_ERROR_BYTES)


def _canonical_json(payload: dict[str, JsonValue]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _plan_payload(plan: HcclParentPlan) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "transaction_id": plan.transaction_id,
        "layer_indices": list(plan.layer_indices),
        "mode": plan.mode,
        "stages": [
            {
                "stage_idx": stage.stage_idx,
                "attention_valid_tokens": list(stage.attention_valid_tokens),
                "attention_padded_tokens": list(stage.attention_padded_tokens),
                "ffn_size": stage.ffn_size,
                "native_tp_size": stage.native_tp_size,
            }
            for stage in plan.stages
        ],
    }
    if plan.graph_execution is not None:
        payload["graph_execution"] = graph_decision_payload(plan.graph_execution)
    return payload


def encode_hccl_plan_message(message: HcclPlanMessage) -> bytes:
    message.validate()
    payload: dict[str, JsonValue] = {
        "version": HCCL_PLAN_PROTOCOL_VERSION,
        "kind": message.kind,
        "payload_format": PAYLOAD_FORMAT,
        "schedule": LAYER_SCHEDULE,
    }
    for name, value in (
        ("role", message.role),
        ("world_rank", message.world_rank),
        ("transaction_id", message.transaction_id),
        ("plan_digest", message.plan_digest),
        ("error", message.error),
        ("graph_digest", message.graph_digest),
    ):
        if value is not None:
            payload[name] = value
    if message.capabilities is not None:
        cap = message.capabilities
        capability_payload: dict[str, JsonValue] = {
            "hidden_size": cap.hidden_size,
            "vocab_size": cap.vocab_size,
            "max_tokens": cap.max_tokens,
            "layer_indices": list(cap.layer_indices),
            "attention_size": cap.attention_size,
            "ffn_size": cap.ffn_size,
            "native_tp_size": cap.native_tp_size,
            "max_stages": cap.max_stages,
        }
        # Preserve the baseline v3 HELLO byte contract unless opted in. Old
        # strict decoders reject this extension during HELLO, before tensor I/O.
        if cap.local_counts_version:
            capability_payload["local_counts_version"] = cap.local_counts_version
        if cap.graph_execution_version:
            capability_payload["graph_execution_version"] = cap.graph_execution_version
        payload["capabilities"] = capability_payload
    if message.plan is not None:
        payload["plan"] = _plan_payload(message.plan)
    if message.local_forward is not None:
        local = message.local_forward
        payload["local_forward"] = {
            "mode": local.mode,
            "valid_tokens": list(local.valid_tokens),
            "padded_tokens": list(local.padded_tokens),
        }
    if message.graph_candidate is not None:
        payload["graph_candidate"] = graph_candidate_payload(message.graph_candidate)
    if message.graph_report is not None:
        payload["graph_report"] = {"availability": message.graph_report.availability}
    if message.graph_decision is not None:
        payload["graph_decision"] = graph_decision_payload(message.graph_decision)
    encoded = _canonical_json(payload)
    if len(encoded) > MAX_CONTROL_MESSAGE_BYTES:
        raise ValueError("HCCL plan message exceeds maximum size")
    return encoded


def _fields(
    value: JsonValue, required: set[str], optional: set[str] | None = None
) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("HCCL plan fields must be a JSON object")
    present = set(value)
    if required - present or present - required - (optional or set()):
        raise ValueError("HCCL plan has missing or unexpected fields")
    return value


def _decode_layers(value: JsonValue) -> tuple[int, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_TARGET_LAYERS:
        raise ValueError("layer_indices must be a bounded nonempty JSON array")
    layers = tuple(_integer("layer_idx", layer) for layer in value)
    _layers(layers)
    return layers


def _decode_capabilities(value: JsonValue) -> HcclPlanCapabilities:
    cap = _fields(
        value,
        {
            "hidden_size",
            "vocab_size",
            "max_tokens",
            "layer_indices",
            "attention_size",
            "ffn_size",
            "native_tp_size",
            "max_stages",
        },
        {"local_counts_version", "graph_execution_version"},
    )
    result = HcclPlanCapabilities(
        _integer("hidden_size", cap["hidden_size"], minimum=1),
        _integer("vocab_size", cap["vocab_size"], minimum=1, maximum=MAX_INT32 + 1),
        _integer("max_tokens", cap["max_tokens"], minimum=1),
        _decode_layers(cap["layer_indices"]),
        _integer(
            "attention_size", cap["attention_size"], minimum=1, maximum=MAX_ROLE_RANKS
        ),
        _integer("ffn_size", cap["ffn_size"], minimum=1, maximum=MAX_ROLE_RANKS),
        _integer(
            "native_tp_size", cap["native_tp_size"], minimum=1, maximum=MAX_ROLE_RANKS
        ),
        _integer("max_stages", cap["max_stages"], minimum=1, maximum=MAX_HCCL_STAGES),
        _integer("local_counts_version", cap.get("local_counts_version", 0), maximum=1),
        _integer(
            "graph_execution_version", cap.get("graph_execution_version", 0), maximum=1
        ),
    )
    result.validate()
    return result


def _decode_counts(value: JsonValue) -> tuple[int, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_ROLE_RANKS:
        raise ValueError("Attention counts must be a bounded nonempty JSON array")
    return tuple(_integer("token_count", count) for count in value)


def _decode_local_forward(value: JsonValue) -> HcclLocalForward:
    local = _fields(value, {"mode", "valid_tokens", "padded_tokens"})
    result = HcclLocalForward(
        _text("mode", local["mode"], MAX_CONTROL_MESSAGE_BYTES),
        _decode_counts(local["valid_tokens"]),
        _decode_counts(local["padded_tokens"]),
    )
    result.validate()
    return result


def _decode_plan(value: JsonValue) -> HcclParentPlan:
    plan = _fields(
        value,
        {"transaction_id", "layer_indices", "stages", "mode"},
        {"graph_execution"},
    )
    stages = plan["stages"]
    if not isinstance(stages, list) or not 1 <= len(stages) <= MAX_HCCL_STAGES:
        raise ValueError("parent plan requires one or two JSON stages")
    decoded = []
    for value in stages:
        stage = _fields(
            value,
            {
                "stage_idx",
                "attention_valid_tokens",
                "attention_padded_tokens",
                "ffn_size",
                "native_tp_size",
            },
        )
        decoded.append(
            HcclStageLayout(
                _integer("stage_idx", stage["stage_idx"], maximum=MAX_HCCL_STAGES - 1),
                _decode_counts(stage["attention_valid_tokens"]),
                _decode_counts(stage["attention_padded_tokens"]),
                _integer(
                    "ffn_size", stage["ffn_size"], minimum=1, maximum=MAX_ROLE_RANKS
                ),
                _integer(
                    "native_tp_size",
                    stage["native_tp_size"],
                    minimum=1,
                    maximum=MAX_ROLE_RANKS,
                ),
            )
        )
    return HcclParentPlan(
        _text("transaction_id", plan["transaction_id"], MAX_TRANSACTION_ID_BYTES),
        _decode_layers(plan["layer_indices"]),
        tuple(decoded),
        _text("mode", plan["mode"], MAX_CONTROL_MESSAGE_BYTES),
        decode_graph_decision(plan["graph_execution"])
        if "graph_execution" in plan
        else None,
    )


def _unique_fields(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate HCCL JSON field: {name}")
        result[name] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite HCCL JSON value: {value}")


def decode_hccl_plan_message(data: bytes) -> HcclPlanMessage:
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_CONTROL_MESSAGE_BYTES:
        raise ValueError("HCCL plan message must be bounded nonempty bytes")
    try:
        payload: JsonValue = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_fields,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid HCCL plan JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("HCCL plan message must be a JSON object")
    if (
        type(payload.get("version")) is not int
        or payload["version"] != HCCL_PLAN_PROTOCOL_VERSION
    ):
        raise ValueError("unsupported HCCL plan protocol version")
    if (
        payload.get("payload_format") != PAYLOAD_FORMAT
        or payload.get("schedule") != LAYER_SCHEDULE
    ):
        raise ValueError("unsupported HCCL payload format or layer schedule")
    kind = payload.get("kind")
    if not isinstance(kind, str) or kind not in MESSAGE_KINDS:
        raise ValueError("unsupported HCCL plan message kind")
    envelope = {"version", "kind", "payload_format", "schedule"}
    if kind == "HELLO":
        _fields(payload, envelope | {"role", "world_rank", "capabilities"})
        message = HcclPlanMessage(
            kind,
            role=_text("role", payload["role"], MAX_CONTROL_MESSAGE_BYTES),
            world_rank=_integer("world_rank", payload["world_rank"]),
            capabilities=_decode_capabilities(payload["capabilities"]),
        )
    elif kind == "FORWARD":
        _fields(payload, envelope | {"plan"})
        message = HcclPlanMessage(kind, plan=_decode_plan(payload["plan"]))
    elif kind == "GRAPH_QUERY":
        _fields(payload, envelope | {"transaction_id", "graph_candidate"})
        message = HcclPlanMessage(
            kind,
            transaction_id=_text(
                "transaction_id", payload["transaction_id"], MAX_TRANSACTION_ID_BYTES
            ),
            graph_candidate=decode_graph_candidate(payload["graph_candidate"]),
        )
    elif kind == "GRAPH_REPORT":
        _fields(
            payload,
            envelope | {"transaction_id", "world_rank", "graph_digest", "graph_report"},
        )
        report = _fields(payload["graph_report"], {"availability"})
        message = HcclPlanMessage(
            kind,
            transaction_id=_text(
                "transaction_id", payload["transaction_id"], MAX_TRANSACTION_ID_BYTES
            ),
            world_rank=_integer("world_rank", payload["world_rank"]),
            graph_digest=_text(
                "graph_digest", payload["graph_digest"], MAX_CONTROL_MESSAGE_BYTES
            ),
            graph_report=HcclGraphReport(
                _text("availability", report["availability"], MAX_CONTROL_MESSAGE_BYTES)
            ),
        )
    elif kind == "GRAPH_DECISION":
        _fields(payload, envelope | {"graph_decision"})
        message = HcclPlanMessage(
            kind, graph_decision=decode_graph_decision(payload["graph_decision"])
        )
    elif kind == "GRAPH_ACK":
        _fields(payload, envelope | {"transaction_id", "world_rank", "graph_digest"})
        message = HcclPlanMessage(
            kind,
            transaction_id=_text(
                "transaction_id", payload["transaction_id"], MAX_TRANSACTION_ID_BYTES
            ),
            world_rank=_integer("world_rank", payload["world_rank"]),
            graph_digest=_text(
                "graph_digest", payload["graph_digest"], MAX_CONTROL_MESSAGE_BYTES
            ),
        )
    elif kind in ("COUNTS_REQUEST", "COUNTS_REPORT"):
        _fields(
            payload,
            envelope
            | {"transaction_id"}
            | ({"world_rank", "local_forward"} if kind == "COUNTS_REPORT" else set()),
        )
        message = HcclPlanMessage(
            kind,
            transaction_id=_text(
                "transaction_id", payload["transaction_id"], MAX_TRANSACTION_ID_BYTES
            ),
            world_rank=_integer("world_rank", payload["world_rank"])
            if kind == "COUNTS_REPORT"
            else None,
            local_forward=_decode_local_forward(payload["local_forward"])
            if kind == "COUNTS_REPORT"
            else None,
        )
    elif kind in ("ACCEPT", "COMPLETE"):
        _fields(
            payload,
            envelope
            | {"transaction_id", "plan_digest"}
            | ({"world_rank"} if kind == "ACCEPT" else set()),
        )
        message = HcclPlanMessage(
            kind,
            transaction_id=_text(
                "transaction_id", payload["transaction_id"], MAX_TRANSACTION_ID_BYTES
            ),
            plan_digest=_text(
                "plan_digest", payload["plan_digest"], MAX_CONTROL_MESSAGE_BYTES
            ),
            world_rank=_integer("world_rank", payload["world_rank"])
            if kind == "ACCEPT"
            else None,
        )
    elif kind == "ERROR":
        _fields(payload, envelope | {"error"}, {"transaction_id"})
        message = HcclPlanMessage(
            kind,
            error=_text("error", payload["error"], MAX_ERROR_BYTES),
            transaction_id=_text(
                "transaction_id", payload["transaction_id"], MAX_TRANSACTION_ID_BYTES
            )
            if "transaction_id" in payload
            else None,
        )
    else:
        _fields(payload, envelope)
        message = HcclPlanMessage(kind)
    message.validate()
    return message
