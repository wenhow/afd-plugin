# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in eager Work ownership; device progress requires separate acceptance.

One transport owns one accepted parent and at most one live ticket per stage.
Posting follows the serial wire order, while storage reuse follows the actual
native consumer. A loan is the entire lifetime of the caller's asynchronous
use of its borrowed tensors. Retain continuation tensors explicitly until the
same stage is reused. Failure keeps all references and requires supervision;
there is no backend cancellation or mid-data control recovery.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from afd_plugin.compat.npu.stream import use_npu_current_stream
from afd_plugin.connectors.npu.hccl_diagnostics import (
    HcclDiagnosticExecution,
    HcclStageDiagnosticRun,
)
from afd_plugin.connectors.npu.hccl_transport import HcclLayerInputs

if TYPE_CHECKING:
    from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport


@dataclass(frozen=True, slots=True, eq=False)
class HcclAttentionTicket:
    """Coordinates are diagnostic; only the issuing owner's identity authorizes."""

    layer_idx: int
    stage_idx: int


@dataclass(frozen=True, slots=True, eq=False)
class HcclFFNTicket:
    layer_idx: int
    stage_idx: int


@dataclass(slots=True)
class _StageSlot:
    inputs: HcclLayerInputs
    output: torch.Tensor
    ticket: HcclAttentionTicket | HcclFFNTicket | None = None
    works: list[dist.Work] = field(default_factory=list)
    events: list[torch.npu.Event] = field(default_factory=list)
    consumer_done: torch.npu.Event | None = None
    reuse_ready: torch.npu.Event | None = None
    retained: tuple[torch.Tensor, ...] = ()
    loaned: bool = False
    consumed: bool = False
    output_posted: bool = False

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.inputs.hidden_states,
            self.inputs.hidden_states_fp32,
            self.inputs.input_ids,
            self.output,
            *self.retained,
        )


def validate_eager_topology(transport: HcclStagedTransport) -> None:
    """Config/DP/PP/CP/SP eligibility remains the caller's separate obligation."""
    caps = transport.negotiated_capabilities
    caps.validate()
    if not (
        caps.attention_size == caps.ffn_size == caps.native_tp_size
        and caps.native_tp_size in (1, 2)
        and caps.max_stages == 2
    ):
        raise ValueError(
            "eager pipeline requires paired native TP1/TP2 and U2 capacity"
        )


class HcclEagerPipeline:
    """Transport-created eager parent; construct through open_eager_pipeline.

    Three explicit streams are session inputs, including when some are equal.
    A Work.wait is inserted on the stream that actually consumes its storage.
    Host return is never treated as device completion. Mutable per-stage slots
    bound payload storage; old Work/Event bookkeeping remains parent-owned.
    """

    def __init__(
        self,
        transport: HcclStagedTransport,
        *,
        parent_stream: torch.npu.Stream,
        send_stream: torch.npu.Stream,
        receive_stream: torch.npu.Stream,
    ) -> None:
        validate_eager_topology(transport)
        plan = transport.plan
        if plan is None or plan.mode != "target" or len(plan.stages) != 2:
            raise RuntimeError("eager pipeline requires an accepted target U2 parent")
        if plan.graph_execution is not None and plan.graph_execution.phase != "eager":
            raise RuntimeError("eager pipeline cannot execute a graph phase")
        plan.validate(transport.negotiated_capabilities)
        if any(
            len(set(stage.attention_valid_tokens)) != 1
            or len(set(stage.attention_padded_tokens)) != 1
            for stage in plan.stages
        ):
            raise ValueError("native TP eager pipeline requires matching peer counts")
        device = torch.device("npu", transport.local_device)
        for stream in (parent_stream, send_stream, receive_stream):
            if stream.device != device:
                raise ValueError("eager streams must use the transport's indexed NPU")
        if torch.npu.current_stream(device) != parent_stream:
            raise ValueError("open eager pipeline on its actual parent stream")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("eager pipeline cannot run inside graph capture")
        self.transport, self.plan = transport, plan
        self._capabilities = transport.negotiated_capabilities
        self._groups = tuple(transport.data_groups)
        self.parent_stream = parent_stream
        self.send_stream, self.receive_stream = send_stream, receive_stream
        self._steps = tuple(
            (layer, stage.stage_idx)
            for layer in plan.layer_indices
            for stage in plan.stages
            if any(stage.attention_padded_tokens)
        )
        self._posted = 0
        self._failed = self._finished = False
        self._progress: HcclStageDiagnosticRun | None = None
        self._history: list[tuple[list[dist.Work], list[torch.npu.Event]]] = []
        self._slots: dict[int, _StageSlot] = {}
        self._in_progress: tuple[torch.Tensor, ...] = ()

    def _allocate_slots(self) -> None:
        # Attach the owner to its transport before allocation, retaining a
        # partially allocated bank if allocation itself reports a device error.
        transport = self.transport
        device = self.parent_stream.device
        with use_npu_current_stream(self.parent_stream):
            for stage in self.plan.stages:
                peers = (
                    stage.for_attention(transport.role_rank)
                    if transport.role == "attention"
                    else stage.for_ffn(transport.role_rank)
                )
                rows = sum(peer.padded_tokens for peer in peers)
                if not rows:
                    continue
                shape = (rows, transport.negotiated_capabilities.hidden_size)
                self._slots[stage.stage_idx] = _StageSlot(
                    HcclLayerInputs(
                        torch.empty(shape, dtype=torch.bfloat16, device=device),
                        torch.empty(shape, dtype=torch.float32, device=device),
                        torch.empty(rows, dtype=torch.int32, device=device),
                    ),
                    torch.empty(shape, dtype=torch.bfloat16, device=device),
                )

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def occupied_stages(self) -> tuple[int, ...]:
        return tuple(index for index, slot in self._slots.items() if slot.ticket)

    def bind_progress(
        self,
        progress: HcclStageDiagnosticRun,
        *,
        compute_stream: torch.npu.Stream,
    ) -> None:
        """Bind exact parent progress before posting; equal objects are idempotent."""
        self._active()
        self._stream(compute_stream)
        progress.validate_parent(
            role=self.transport.role,
            world_rank=self.transport.world_rank,
            role_rank=self.transport.role_rank,
            num_stages=len(self.plan.stages),
            layer_indices=self.plan.layer_indices,
            transaction_id=self.plan.transaction_id,
        )
        self._validate_progress_execution(progress, compute_stream)
        if self._progress is progress:
            return
        if (
            self._progress is not None
            or self._posted
            or any(slot.ticket is not None for slot in self._slots.values())
        ):
            raise ValueError("eager diagnostics must bind once before payload posting")
        self._progress = progress

    def _validate_progress_execution(
        self, progress: HcclStageDiagnosticRun, compute_stream: torch.npu.Stream
    ) -> None:
        progress.validate_execution(
            HcclDiagnosticExecution(
                "eager-streamed",
                "parent" if compute_stream == self.parent_stream else "compute",
                "parent" if self.send_stream == self.parent_stream else "send",
                "parent" if self.receive_stream == self.parent_stream else "recv",
                (
                    compute_stream == self.send_stream,
                    compute_stream == self.receive_stream,
                    self.send_stream == self.receive_stream,
                ),
            )
        )

    def _active(self, role: str | None = None) -> None:
        self.transport._require_active(role)
        if (
            self._failed
            or self._finished
            or self.transport.eager_pipeline is not self
            or self.transport.plan is not self.plan
            or self.transport.negotiated_capabilities != self._capabilities
            or len(self.transport.data_groups) != len(self._groups)
            or any(
                current is not captured
                for current, captured in zip(
                    self.transport.data_groups, self._groups, strict=True
                )
            )
        ):
            raise RuntimeError("eager parent is foreign, stale, or terminal")
        if torch.npu.current_stream(self.parent_stream.device) != self.parent_stream:
            raise RuntimeError("eager owner methods require the actual parent stream")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("eager owner cannot run inside graph capture")

    def _stream(self, stream: torch.npu.Stream) -> None:
        if stream.device != self.parent_stream.device:
            raise ValueError("producer/consumer stream must use the transport NPU")

    def _retain(
        self, tensors: tuple[torch.Tensor, ...], stream: torch.npu.Stream
    ) -> None:
        if not isinstance(tensors, tuple) or any(
            not isinstance(tensor, torch.Tensor)
            or tensor.device != self.parent_stream.device
            for tensor in tensors
        ):
            raise ValueError("retain must be a tuple of tensors on the transport NPU")
        for tensor in tensors:
            tensor.record_stream(stream)

    def _step(self, layer_idx: int, stage_idx: int) -> _StageSlot:
        if (
            type(layer_idx) is not int
            or type(stage_idx) is not int
            or self._posted >= len(self._steps)
            or self._steps[self._posted] != (layer_idx, stage_idx)
        ):
            raise ValueError("eager posting must follow the accepted layer/stage order")
        return self._slots[stage_idx]

    def _ticket(self, ticket: HcclAttentionTicket | HcclFFNTicket) -> _StageSlot:
        if (
            not isinstance(ticket, (HcclAttentionTicket, HcclFFNTicket))
            or type(ticket.layer_idx) is not int
            or type(ticket.stage_idx) is not int
        ):
            raise ValueError("expected an owner-issued eager ticket")
        slot = self._slots.get(ticket.stage_idx)
        if slot is None or slot.ticket is not ticket:
            raise ValueError("eager ticket is forged, foreign, or stale")
        return slot

    @staticmethod
    def _wait(works: list[dist.Work]) -> None:
        for work in works:
            if not work.wait():
                raise RuntimeError(
                    "eager HCCL Work.wait did not complete its dependency"
                )

    def _release(self, slot: _StageSlot) -> None:
        if slot.ticket is None:
            return
        if (
            slot.loaned
            or not slot.consumed
            or (self.transport.role == "ffn" and not slot.output_posted)
        ):
            raise RuntimeError("stage reuse requires its closed loan and posted output")
        assert slot.consumer_done is not None
        progress = self._progress
        if progress is not None:
            progress.record(
                "stage_reuse_wait_begin", slot.ticket.layer_idx, slot.ticket.stage_idx
            )
        with use_npu_current_stream(self.parent_stream):
            slot.consumer_done.wait(self.parent_stream)
            self._wait(slot.works)
            self._retain(slot.tensors, self.parent_stream)
            ready = torch.npu.Event()
            ready.record(self.parent_stream)
            slot.events.append(ready)
            slot.reuse_ready = ready
        if progress is not None:
            progress.record(
                "stage_reuse_wait_returned",
                slot.ticket.layer_idx,
                slot.ticket.stage_idx,
            )
        self._history.append((slot.works, slot.events))
        slot.works, slot.events, slot.retained = [], [], ()
        slot.ticket, slot.consumer_done = None, None
        slot.loaned = slot.consumed = slot.output_posted = False

    def post_attention(
        self,
        layer_idx: int,
        stage_idx: int,
        hidden_states: torch.Tensor,
        hidden_states_fp32: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        producer_stream: torch.npu.Stream,
        retain: tuple[torch.Tensor, ...] = (),
    ) -> HcclAttentionTicket:
        self._active("attention")
        self._stream(producer_stream)
        slot = self._step(layer_idx, stage_idx)
        stage = self.plan.stages[stage_idx]
        native_rows = stage.attention_padded_tokens[self.transport.role_rank]
        hidden_size = self.transport.negotiated_capabilities.hidden_size
        self.transport._check_tensor(
            hidden_states, (native_rows, hidden_size), torch.bfloat16
        )
        self.transport._check_tensor(
            hidden_states_fp32, (native_rows, hidden_size), torch.float32
        )
        self.transport._check_tensor(input_ids, (native_rows,), torch.int32)
        self._in_progress = (hidden_states, hidden_states_fp32, input_ids, *retain)
        progress = self._progress
        try:
            with use_npu_current_stream(producer_stream):
                self.transport._check_ids(
                    input_ids, stage.attention_valid_tokens[self.transport.role_rank]
                )
                self._retain(retain, producer_stream)
                produced = torch.npu.Event()
                produced.record(producer_stream)
            self._release(slot)
            ticket = HcclAttentionTicket(layer_idx, stage_idx)
            slot.ticket = ticket
            slot.retained = (hidden_states, hidden_states_fp32, input_ids, *retain)
            slot.events.append(produced)
            if progress is not None:
                progress.record("send_post_begin", layer_idx, stage_idx)
            with use_npu_current_stream(self.send_stream):
                if slot.reuse_ready is not None:
                    slot.reuse_ready.wait(self.send_stream)
                produced.wait(self.send_stream)
                self._retain(slot.tensors, self.send_stream)
                # Private wire padding preserves native rows and zeros wire-only IDs.
                slot.inputs.hidden_states.zero_()
                slot.inputs.hidden_states_fp32.zero_()
                slot.inputs.input_ids.zero_()
                slot.inputs.hidden_states[:native_rows].copy_(hidden_states)
                slot.inputs.hidden_states_fp32[:native_rows].copy_(hidden_states_fp32)
                slot.inputs.input_ids[:native_rows].copy_(input_ids)
                peer = stage.for_attention(self.transport.role_rank)[0]
                for tensor in (
                    slot.inputs.hidden_states,
                    slot.inputs.hidden_states_fp32,
                    slot.inputs.input_ids,
                ):
                    slot.works.append(
                        dist.isend(
                            tensor,
                            group=self.transport.data_groups[stage_idx],
                            group_dst=peer.ffn_world_rank,
                        )
                    )
            if progress is not None:
                progress.record("send_post_returned", layer_idx, stage_idx)
                progress.record("recv_post_begin", layer_idx, stage_idx)
            with use_npu_current_stream(self.receive_stream):
                if slot.reuse_ready is not None:
                    slot.reuse_ready.wait(self.receive_stream)
                slot.output.record_stream(self.receive_stream)
                slot.works.append(
                    dist.irecv(
                        slot.output,
                        group=self.transport.data_groups[stage_idx],
                        group_src=peer.ffn_world_rank,
                    )
                )
            if progress is not None:
                progress.record("recv_post_returned", layer_idx, stage_idx)
            self._posted += 1
            self._in_progress = ()
            return ticket
        except BaseException:
            self.fail()
            raise

    @contextmanager
    def attention_output(
        self,
        ticket: HcclAttentionTicket,
        *,
        consumer_stream: torch.npu.Stream,
        retain: tuple[torch.Tensor, ...] = (),
    ) -> Iterator[torch.Tensor]:
        self._active("attention")
        self._stream(consumer_stream)
        slot = self._ticket(ticket)
        if slot.loaned or slot.consumed:
            raise RuntimeError("Attention output loan can only be consumed once")
        native_rows = self.plan.stages[ticket.stage_idx].attention_padded_tokens[
            self.transport.role_rank
        ]
        progress = self._progress
        if progress is not None:
            self._validate_progress_execution(progress, consumer_stream)
        slot.loaned = True
        try:
            with use_npu_current_stream(consumer_stream):
                if progress is not None:
                    progress.record(
                        "loan_wait_begin", ticket.layer_idx, ticket.stage_idx
                    )
                self._wait(slot.works)
                if progress is not None:
                    progress.record(
                        "loan_wait_returned", ticket.layer_idx, ticket.stage_idx
                    )
                self._retain((*slot.tensors, *retain), consumer_stream)
                slot.retained += retain
                yield slot.output[:native_rows]
                done = torch.npu.Event()
                done.record(consumer_stream)
                slot.consumer_done = done
                slot.events.append(done)
            slot.loaned, slot.consumed = False, True
        except BaseException:
            self.fail()
            raise

    def post_ffn_receive(self, layer_idx: int, stage_idx: int) -> HcclFFNTicket:
        self._active("ffn")
        slot = self._step(layer_idx, stage_idx)
        if slot.ticket is not None and not slot.output_posted:
            raise RuntimeError("post the current FFN output before the next receive")
        progress = self._progress
        try:
            self._release(slot)
            ticket = HcclFFNTicket(layer_idx, stage_idx)
            slot.ticket = ticket
            peer = self.plan.stages[stage_idx].for_ffn(self.transport.role_rank)[0]
            if progress is not None:
                progress.record("recv_post_begin", layer_idx, stage_idx)
            with use_npu_current_stream(self.receive_stream):
                if slot.reuse_ready is not None:
                    slot.reuse_ready.wait(self.receive_stream)
                self._retain(slot.tensors, self.receive_stream)
                for tensor in (
                    slot.inputs.hidden_states,
                    slot.inputs.hidden_states_fp32,
                    slot.inputs.input_ids,
                ):
                    slot.works.append(
                        dist.irecv(
                            tensor,
                            group=self.transport.data_groups[stage_idx],
                            group_src=peer.attention_world_rank,
                        )
                    )
            if progress is not None:
                progress.record("recv_post_returned", layer_idx, stage_idx)
            return ticket
        except BaseException:
            self.fail()
            raise

    @contextmanager
    def ffn_inputs(
        self,
        ticket: HcclFFNTicket,
        *,
        consumer_stream: torch.npu.Stream,
        retain: tuple[torch.Tensor, ...] = (),
    ) -> Iterator[HcclLayerInputs]:
        self._active("ffn")
        self._stream(consumer_stream)
        slot = self._ticket(ticket)
        if slot.loaned or slot.consumed:
            raise RuntimeError("FFN inputs loan can only be consumed once")
        progress = self._progress
        if progress is not None:
            self._validate_progress_execution(progress, consumer_stream)
        slot.loaned = True
        try:
            with use_npu_current_stream(consumer_stream):
                if progress is not None:
                    progress.record(
                        "loan_wait_begin", ticket.layer_idx, ticket.stage_idx
                    )
                self._wait(slot.works)
                if progress is not None:
                    progress.record(
                        "loan_wait_returned", ticket.layer_idx, ticket.stage_idx
                    )
                stage = self.plan.stages[ticket.stage_idx]
                peer = stage.for_ffn(self.transport.role_rank)[0]
                self.transport._check_ids(slot.inputs.input_ids, peer.valid_tokens)
                self._retain((*slot.tensors, *retain), consumer_stream)
                slot.retained += retain
                yield slot.inputs
                done = torch.npu.Event()
                done.record(consumer_stream)
                slot.consumer_done = done
                slot.events.append(done)
            slot.loaned, slot.consumed = False, True
        except BaseException:
            self.fail()
            raise

    def post_ffn_output(
        self,
        ticket: HcclFFNTicket,
        output: torch.Tensor,
        *,
        producer_stream: torch.npu.Stream,
        retain: tuple[torch.Tensor, ...] = (),
    ) -> None:
        self._active("ffn")
        self._stream(producer_stream)
        slot = self._ticket(ticket)
        if slot.loaned or not slot.consumed or slot.output_posted:
            raise RuntimeError("FFN output requires a closed input loan and one send")
        self.transport._check_tensor(output, tuple(slot.output.shape), torch.bfloat16)
        progress = self._progress
        self._in_progress = (output, *retain)
        try:
            with use_npu_current_stream(producer_stream):
                self._retain((output, *retain), producer_stream)
                produced = torch.npu.Event()
                produced.record(producer_stream)
            slot.events.append(produced)
            slot.retained += (output, *retain)
            if progress is not None:
                progress.record("send_post_begin", ticket.layer_idx, ticket.stage_idx)
            with use_npu_current_stream(self.send_stream):
                produced.wait(self.send_stream)
                self._retain(slot.tensors, self.send_stream)
                slot.output.copy_(output)
                peer = self.plan.stages[ticket.stage_idx].for_ffn(
                    self.transport.role_rank
                )[0]
                slot.works.append(
                    dist.isend(
                        slot.output,
                        group=self.transport.data_groups[ticket.stage_idx],
                        group_dst=peer.attention_world_rank,
                    )
                )
            if progress is not None:
                progress.record(
                    "send_post_returned", ticket.layer_idx, ticket.stage_idx
                )
            slot.output_posted = True
            self._posted += 1
            self._in_progress = ()
        except BaseException:
            self.fail()
            raise

    def finish(self) -> None:
        self._active()
        if self._posted != len(self._steps) or any(
            slot.ticket is not None
            and (
                slot.loaned
                or not slot.consumed
                or (self.transport.role == "ffn" and not slot.output_posted)
            )
            for slot in self._slots.values()
        ):
            raise RuntimeError(
                "finish requires all postings and closed native consumer loans"
            )
        progress = self._progress
        final_layer = self.plan.layer_indices[-1]
        try:
            if progress is not None:
                progress.record("parent_finish_begin", final_layer, 0)
            for slot in self._slots.values():
                self._release(slot)
            self.transport._complete_eager_pipeline(self)
            self._finished = True
        except BaseException:
            self.fail()
            raise
        if progress is not None:
            # COMPLETE already retired this owner; a sink error must not fail
            # a finished owner or a different transaction accepted afterward.
            progress.record("parent_finish_returned", final_layer, 0)

    def fail(self) -> None:
        """Sticky failure retains groups, Works, events, slots and consumer refs."""
        if self._finished or self.transport.eager_pipeline is not self:
            raise RuntimeError("a stale eager owner cannot fail a different parent")
        self._failed = True
        self.transport.terminal = True
