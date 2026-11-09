# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU graph reservations; runtime values and local storage never form keys.

These records describe the initial uniform decode, paired-native-TP contract.
The cache owner validates its model identity and local storage generation;
transport validates negotiated geometry and the post-prepare parent counts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TypeAlias

from afd_plugin.distributed.hccl import MAX_HCCL_STAGES

MAX_GRAPH_INTEGER = (1 << 31) - 1
MAX_GRAPH_METADATA_SHAPES = 64
MAX_GRAPH_SHAPE_DIMENSIONS = 8
MAX_GRAPH_TRANSACTION_BYTES = 256
GRAPH_PHASES = frozenset({"eager", "warmup", "capture", "replay"})
GRAPH_AVAILABILITY = frozenset({"ready", "missing", "unsupported"})
GRAPH_REASONS = frozenset({"requested", "cache_miss", "unsupported"})
GRAPH_MODES = frozenset({"target", "dummy", "warmup", "profile"})
JsonValue: TypeAlias = (
    bool | int | float | str | None | list["JsonValue"] | dict[str, "JsonValue"]
)


def _integer(name: str, value: JsonValue, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= MAX_GRAPH_INTEGER:
        raise ValueError(f"{name} must be a bounded integer >= {minimum}")
    return value


def _fields(value: JsonValue, names: set[str]) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError("graph record has missing or unexpected fields")
    return value


def _digest(value: dict[str, JsonValue]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _padded_tokens(value: tuple[int, ...]) -> None:
    if not isinstance(value, tuple) or not 1 <= len(value) <= MAX_HCCL_STAGES:
        raise ValueError("graph padded_tokens requires one or two stages")
    for count in value:
        _integer("padded_tokens", count)


@dataclass(frozen=True, slots=True)
class HcclGraphStage:
    request_start: int
    request_stop: int
    token_start: int
    token_stop: int

    def __post_init__(self) -> None:
        for name, value in (
            ("request_start", self.request_start),
            ("request_stop", self.request_stop),
            ("token_start", self.token_start),
            ("token_stop", self.token_stop),
        ):
            _integer(name, value)
        if self.request_start > self.request_stop or self.token_start > self.token_stop:
            raise ValueError("graph stage bounds must be ordered")


@dataclass(frozen=True, slots=True)
class HcclGraphKey:
    num_tokens: int
    num_reqs: int
    uniform_token_count: int
    max_query_len: int
    num_active_loras: int
    num_ubatches: int
    stage_layouts: tuple[HcclGraphStage, ...]
    metadata_shapes: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("num_tokens", self.num_tokens),
            ("num_reqs", self.num_reqs),
            ("uniform_token_count", self.uniform_token_count),
            ("max_query_len", self.max_query_len),
            ("num_ubatches", self.num_ubatches),
        ):
            _integer(name, value, minimum=1)
        _integer("num_active_loras", self.num_active_loras)
        if (
            self.uniform_token_count != 1
            or self.max_query_len != 1
            or self.num_active_loras != 0
            or self.num_tokens != self.num_reqs
        ):
            raise ValueError("graph key requires uniform q=1 decode without LoRA")
        if (
            not isinstance(self.stage_layouts, tuple)
            or not 1 <= self.num_ubatches <= MAX_HCCL_STAGES
            or len(self.stage_layouts) != self.num_ubatches
        ):
            raise ValueError("graph key stage count must match num_ubatches")
        previous = 0
        for stage in self.stage_layouts:
            if not isinstance(stage, HcclGraphStage):
                raise ValueError("graph key stages must be HcclGraphStage")
            if (
                stage.token_start != previous
                or stage.request_stop > self.num_reqs
                or stage.token_stop > self.num_tokens
            ):
                raise ValueError("graph stage layout must cover the padded parent")
            previous = stage.token_stop
        if previous != self.num_tokens:
            raise ValueError("graph stage layout must cover the padded parent")
        if (
            not isinstance(self.metadata_shapes, tuple)
            or len(self.metadata_shapes) > MAX_GRAPH_METADATA_SHAPES
        ):
            raise ValueError("graph metadata shapes must be a bounded tuple")
        for shape in self.metadata_shapes:
            if (
                not isinstance(shape, tuple)
                or not 1 <= len(shape) <= MAX_GRAPH_SHAPE_DIMENSIONS
            ):
                raise ValueError("graph metadata shape dimensions must be bounded")
            for dimension in shape:
                _integer("metadata dimension", dimension)

    @property
    def padded_tokens(self) -> tuple[int, ...]:
        return tuple(
            stage.token_stop - stage.token_start for stage in self.stage_layouts
        )

    @property
    def digest(self) -> str:
        return _digest(graph_key_payload(self))


@dataclass(frozen=True, slots=True)
class HcclGraphCandidate:
    phase: str
    mode: str
    key: HcclGraphKey | None
    eager_padded_tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.phase, str)
            or self.phase not in GRAPH_PHASES
            or not isinstance(self.mode, str)
            or self.mode not in GRAPH_MODES
        ):
            raise ValueError("unsupported graph phase or forward mode")
        _padded_tokens(self.eager_padded_tokens)
        if (self.phase == "eager") != (self.key is None):
            raise ValueError("only eager graph candidates omit the static key")
        if self.key is not None and not isinstance(self.key, HcclGraphKey):
            raise ValueError("graph candidate key must be HcclGraphKey")
        if self.mode == "profile" and self.phase != "eager":
            raise ValueError("profile graph reservations must remain eager")

    @property
    def digest(self) -> str:
        return _digest(graph_candidate_payload(self))


@dataclass(frozen=True, slots=True)
class HcclGraphReport:
    availability: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.availability, str)
            or self.availability not in GRAPH_AVAILABILITY
        ):
            raise ValueError("unsupported graph cache availability")


@dataclass(frozen=True, slots=True)
class HcclGraphDecision:
    transaction_id: str
    phase: str
    mode: str
    key: HcclGraphKey | None
    padded_tokens: tuple[int, ...]
    reason: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.transaction_id, str)
            or not self.transaction_id.strip()
            or len(self.transaction_id.encode("utf-8")) > MAX_GRAPH_TRANSACTION_BYTES
        ):
            raise ValueError("graph transaction_id must be bounded nonempty UTF-8")
        HcclGraphCandidate(self.phase, self.mode, self.key, self.padded_tokens)
        if not isinstance(self.reason, str) or self.reason not in GRAPH_REASONS:
            raise ValueError("unsupported graph decision reason")
        if self.phase != "eager":
            assert self.key is not None
            if (
                self.reason != "requested"
                or self.padded_tokens != self.key.padded_tokens
            ):
                raise ValueError(
                    "graph decision must retain the requested static geometry"
                )

    @property
    def digest(self) -> str:
        return _digest(graph_decision_payload(self))


def decide_graph_execution(
    transaction_id: str,
    candidate: HcclGraphCandidate,
    reports: tuple[HcclGraphReport, ...],
) -> HcclGraphDecision:
    if not reports or any(
        not isinstance(report, HcclGraphReport) for report in reports
    ):
        raise ValueError("graph decision requires every participant report")
    states = {report.availability for report in reports}
    phase, key, reason = candidate.phase, candidate.key, "requested"
    if phase == "replay" and states != {"ready"}:
        phase, key = "eager", None
        reason = "unsupported" if "unsupported" in states else "cache_miss"
    elif phase in ("warmup", "capture") and states != {"missing"}:
        raise ValueError(
            "warmup/capture requires an uncaptured supported key on every rank"
        )
    padded = candidate.eager_padded_tokens if key is None else key.padded_tokens
    return HcclGraphDecision(transaction_id, phase, candidate.mode, key, padded, reason)


def graph_key_payload(key: HcclGraphKey) -> dict[str, JsonValue]:
    return {
        "num_tokens": key.num_tokens,
        "num_reqs": key.num_reqs,
        "uniform_token_count": key.uniform_token_count,
        "max_query_len": key.max_query_len,
        "num_active_loras": key.num_active_loras,
        "num_ubatches": key.num_ubatches,
        "stage_layouts": [
            {
                "request_start": stage.request_start,
                "request_stop": stage.request_stop,
                "token_start": stage.token_start,
                "token_stop": stage.token_stop,
            }
            for stage in key.stage_layouts
        ],
        "metadata_shapes": [list(shape) for shape in key.metadata_shapes],
    }


def graph_candidate_payload(candidate: HcclGraphCandidate) -> dict[str, JsonValue]:
    return {
        "phase": candidate.phase,
        "mode": candidate.mode,
        "key": graph_key_payload(candidate.key) if candidate.key is not None else None,
        "eager_padded_tokens": list(candidate.eager_padded_tokens),
    }


def graph_decision_payload(decision: HcclGraphDecision) -> dict[str, JsonValue]:
    return {
        "transaction_id": decision.transaction_id,
        "phase": decision.phase,
        "mode": decision.mode,
        "key": graph_key_payload(decision.key) if decision.key is not None else None,
        "padded_tokens": list(decision.padded_tokens),
        "reason": decision.reason,
    }


def _decode_rows(value: JsonValue) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError("graph rows must be a JSON array")
    rows = tuple(_integer("graph rows", row) for row in value)
    _padded_tokens(rows)
    return rows


def decode_graph_key(value: JsonValue) -> HcclGraphKey:
    record = _fields(
        value,
        {
            "num_tokens",
            "num_reqs",
            "uniform_token_count",
            "max_query_len",
            "num_active_loras",
            "num_ubatches",
            "stage_layouts",
            "metadata_shapes",
        },
    )
    layouts, shapes = record["stage_layouts"], record["metadata_shapes"]
    if not isinstance(layouts, list) or not isinstance(shapes, list):
        raise ValueError("graph layouts and metadata shapes must be JSON arrays")
    stages = []
    for value in layouts:
        stage = _fields(
            value, {"request_start", "request_stop", "token_start", "token_stop"}
        )
        stages.append(
            HcclGraphStage(
                *(
                    _integer(name, stage[name])
                    for name in (
                        "request_start",
                        "request_stop",
                        "token_start",
                        "token_stop",
                    )
                )
            )
        )
    decoded_shapes = []
    for shape in shapes:
        if not isinstance(shape, list):
            raise ValueError("graph metadata shape must be a JSON array")
        decoded_shapes.append(
            tuple(_integer("metadata dimension", size) for size in shape)
        )
    return HcclGraphKey(
        *(
            _integer(name, record[name])
            for name in (
                "num_tokens",
                "num_reqs",
                "uniform_token_count",
                "max_query_len",
                "num_active_loras",
                "num_ubatches",
            )
        ),
        tuple(stages),
        tuple(decoded_shapes),
    )


def decode_graph_candidate(value: JsonValue) -> HcclGraphCandidate:
    record = _fields(value, {"phase", "mode", "key", "eager_padded_tokens"})
    if not isinstance(record["phase"], str) or not isinstance(record["mode"], str):
        raise ValueError("graph phase/mode must be text")
    return HcclGraphCandidate(
        record["phase"],
        record["mode"],
        decode_graph_key(record["key"]) if record["key"] is not None else None,
        _decode_rows(record["eager_padded_tokens"]),
    )


def decode_graph_decision(value: JsonValue) -> HcclGraphDecision:
    record = _fields(
        value, {"transaction_id", "phase", "mode", "key", "padded_tokens", "reason"}
    )
    if any(
        not isinstance(record[name], str)
        for name in ("transaction_id", "phase", "mode", "reason")
    ):
        raise ValueError("graph decision identity/phase/mode/reason must be text")
    return HcclGraphDecision(
        record["transaction_id"],
        record["phase"],
        record["mode"],
        decode_graph_key(record["key"]) if record["key"] is not None else None,
        _decode_rows(record["padded_tokens"]),
        record["reason"],
    )
