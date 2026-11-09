# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU control contract for a complete HCCL parent plan.

This coordinator describes the global schedule, including all-world plan ACKs;
it is not a per-rank network transport. A future adapter must distribute that
plan, gather ACKs and abort on wire/I/O failures before integrating this state.
No tensors, process groups or model state are created here. Existing wire-v2
U1 transport does not import this module.

Inputs follow (layer, stage, peer) order. Independent stages may have pending
exchanges together. Fanout must reassemble every output before Attention can
advance; aggregation must receive every input before FFN can return outputs.
Native TP coordinates share those readiness boundaries without summing rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from afd_plugin.distributed.hccl_plan import (
    HcclParentPlan,
    HcclPlanCapabilities,
    HcclPlanMessage,
    encode_hccl_plan_message,
)

MAX_SESSION_TRANSACTIONS = 65_536

PeerKey = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class HcclExchange:
    """A tensor exchange identity and row contract, independent of buffers."""

    transaction_id: str
    layer_idx: int
    stage_idx: int
    attention_world_rank: int
    ffn_world_rank: int
    valid_tokens: int
    padded_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.transaction_id, str) or not self.transaction_id:
            raise ValueError("exchange requires a transaction_id")
        for value in (
            self.layer_idx,
            self.stage_idx,
            self.attention_world_rank,
            self.ffn_world_rank,
            self.valid_tokens,
            self.padded_tokens,
        ):
            if type(value) is not int or value < 0:
                raise ValueError(
                    "exchange coordinates and counts must be nonnegative integers"
                )
        if not 0 <= self.valid_tokens <= self.padded_tokens or not self.padded_tokens:
            raise ValueError(
                "an exchange requires 0 <= valid <= positive padded tokens"
            )

    @property
    def peer_key(self) -> PeerKey:
        return self.stage_idx, self.attention_world_rank, self.ffn_world_rank


class HcclPlanSession:
    """Validate a negotiated parent schedule; local misuse leaves state intact.

    A transport adapter must call abort after a rejected peer message or failed
    tensor operation. Retries are only safe for local validation errors before
    I/O. Replay protection never evicts IDs: reconnect at the session limit.
    """

    def __init__(
        self,
        capabilities: HcclPlanCapabilities,
        *,
        max_transactions: int = MAX_SESSION_TRANSACTIONS,
    ) -> None:
        capabilities.validate()
        if type(max_transactions) is not int or not (
            1 <= max_transactions <= MAX_SESSION_TRANSACTIONS
        ):
            raise ValueError("invalid session transaction limit")
        self._capabilities = capabilities
        self._max_transactions = max_transactions
        self._completed: set[str] = set()
        self._plan: HcclParentPlan | None = None
        self._digest: str | None = None
        self._accepted: set[int] = set()
        self._progress: dict[PeerKey, int] = {}
        self._pending: dict[PeerKey, HcclExchange] = {}
        self._schedule: Iterator[HcclExchange] = iter(())
        self._next_input: HcclExchange | None = None
        self._stopping = False
        self._terminal = False

    @property
    def next_input(self) -> HcclExchange | None:
        """Next scheduled exchange; readiness still requires ACKs/prior outputs."""
        return self._next_input

    def _require_open(self) -> None:
        if self._terminal or self._stopping:
            raise RuntimeError("HCCL plan session is terminal or stopping")

    def _require_plan(self, transaction_id: str) -> HcclParentPlan:
        self._require_open()
        if self._plan is None or self._plan.transaction_id != transaction_id:
            raise ValueError("transaction_id does not match the active parent")
        return self._plan

    def begin(self, plan: HcclParentPlan) -> HcclPlanMessage:
        self._require_open()
        if self._plan is not None:
            raise RuntimeError("previous parent transaction has not completed")
        plan.validate(self._capabilities)
        if plan.transaction_id in self._completed:
            raise ValueError("transaction_id has already completed")
        if len(self._completed) >= self._max_transactions:
            raise ValueError("session transaction limit reached; reconnect")
        # Encode/validate the complete wire contract before changing state.
        message = HcclPlanMessage("FORWARD", plan=plan)
        encode_hccl_plan_message(message)
        digest = plan.digest
        self._plan = plan
        self._digest = digest
        self._accepted.clear()
        self._progress = {
            (stage.stage_idx, peer.attention_world_rank, peer.ffn_world_rank): 0
            for stage in plan.stages
            for peer in stage.peer_slices
            if peer.padded_tokens
        }
        self._pending.clear()
        self._schedule = self._exchanges(plan)
        self._next_input = next(self._schedule, None)
        return message

    @staticmethod
    def _exchanges(plan: HcclParentPlan) -> Iterator[HcclExchange]:
        for layer_idx in plan.layer_indices:
            for stage in plan.stages:
                for peer in stage.peer_slices:
                    if peer.padded_tokens:
                        yield HcclExchange(
                            plan.transaction_id,
                            layer_idx,
                            stage.stage_idx,
                            peer.attention_world_rank,
                            peer.ffn_world_rank,
                            peer.valid_tokens,
                            peer.padded_tokens,
                        )

    def accept(self, message: HcclPlanMessage) -> None:
        message.validate()
        if message.kind != "ACCEPT":
            raise ValueError("expected an ACCEPT message")
        assert message.transaction_id is not None and message.world_rank is not None
        self._require_plan(message.transaction_id)
        if message.plan_digest != self._digest:
            raise ValueError("ACCEPT plan digest does not match")
        world_size = self._capabilities.attention_size + self._capabilities.ffn_size
        if not 0 <= message.world_rank < world_size:
            raise ValueError("ACCEPT world_rank is outside the parent topology")
        if message.world_rank in self._accepted:
            raise ValueError("world_rank has already accepted this parent")
        self._accepted.add(message.world_rank)

    def _require_accepted(self) -> None:
        world_size = self._capabilities.attention_size + self._capabilities.ffn_size
        if len(self._accepted) != world_size:
            raise RuntimeError("all parent peers must ACCEPT before tensor exchange")

    def begin_exchange(self, exchange: HcclExchange) -> None:
        plan = self._require_plan(exchange.transaction_id)
        self._require_accepted()
        if exchange != self._next_input:
            raise ValueError("exchange does not match layer/stage/peer input order")
        layer_position = self._progress[exchange.peer_key]
        if (
            exchange.peer_key in self._pending
            or plan.layer_indices[layer_position] != exchange.layer_idx
        ):
            raise RuntimeError("previous output must complete before the next input")
        tp_size = self._capabilities.native_tp_size
        attention_rank = exchange.attention_world_rank - self._capabilities.ffn_size
        tp_start = attention_rank // tp_size * tp_size + self._capabilities.ffn_size
        # All fanout outputs (and native TP lanes) must be reassembled before
        # Attention can produce any of this stage's next-layer inputs.
        if any(
            progress < layer_position
            for (stage, attention, _), progress in self._progress.items()
            if stage == exchange.stage_idx
            and tp_start <= attention < tp_start + tp_size
        ):
            raise RuntimeError("Attention requires all previous peer/TP outputs")
        self._pending[exchange.peer_key] = exchange
        self._next_input = next(self._schedule, None)

    def finish_exchange(self, exchange: HcclExchange) -> None:
        self._require_plan(exchange.transaction_id)
        if self._pending.get(exchange.peer_key) != exchange:
            raise ValueError("output does not match a pending stage/peer exchange")
        layer_position = self._progress[exchange.peer_key]
        tp_size = self._capabilities.native_tp_size
        tp_start = exchange.ffn_world_rank // tp_size * tp_size
        # Aggregation is one FFN input, including all Attention peers. TP
        # coordinates likewise cannot return before their matching inputs exist.
        if any(
            progress < layer_position
            or (progress == layer_position and key not in self._pending)
            for key, progress in self._progress.items()
            if key[0] == exchange.stage_idx and tp_start <= key[2] < tp_start + tp_size
        ):
            raise RuntimeError("FFN requires all aggregate peer/TP inputs")
        del self._pending[exchange.peer_key]
        self._progress[exchange.peer_key] += 1

    def complete(self, transaction_id: str) -> HcclPlanMessage:
        plan = self._require_plan(transaction_id)
        self._require_accepted()
        if (
            self._next_input is not None
            or self._pending
            or any(
                position != len(plan.layer_indices)
                for position in self._progress.values()
            )
        ):
            raise RuntimeError("COMPLETE requires every stage/peer output")
        message = HcclPlanMessage(
            "COMPLETE", transaction_id=transaction_id, plan_digest=self._digest
        )
        self._completed.add(transaction_id)
        self._clear_parent()
        return message

    def request_stop(self) -> HcclPlanMessage:
        self._require_open()
        if self._plan is not None:
            raise RuntimeError("STOP requires an idle parent transaction boundary")
        self._stopping = True
        return HcclPlanMessage("STOP")

    def acknowledge_stop(self, message: HcclPlanMessage) -> None:
        message.validate()
        if not self._stopping or self._terminal or message.kind != "STOP_ACK":
            raise RuntimeError("expected STOP_ACK for a requested STOP")
        self._terminal = True
        self._stopping = False

    def receive_stop(self) -> HcclPlanMessage:
        self._require_open()
        if self._plan is not None:
            raise RuntimeError("STOP requires an idle parent transaction boundary")
        self._terminal = True
        return HcclPlanMessage("STOP_ACK")

    def abort(self, reason: str) -> HcclPlanMessage:
        transaction_id = None if self._plan is None else self._plan.transaction_id
        # Even an unencodable error must close the failed session. The adapter
        # may be unable to send ERROR and must rely on backend timeout/cleanup.
        self._terminal = True
        self._stopping = False
        self._clear_parent()
        message = HcclPlanMessage(
            "ERROR",
            error=reason,
            transaction_id=transaction_id,
        )
        message.validate()
        return message

    def _clear_parent(self) -> None:
        self._plan = None
        self._digest = None
        self._accepted.clear()
        self._progress.clear()
        self._pending.clear()
        self._schedule = iter(())
        self._next_input = None
