# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit v3 eager component, independent of model runtime admission.

All ranks own a Gloo control domain and one HCCL domain per negotiated stage.
Attention0 coordinates readiness/completion; TCPStore is served by world rank
zero (FFN0). Native TP groups remain caller-owned. This synchronous path has
no graph or stream overlap. I/O failure is terminal; backend timeouts and an
external supervisor must bound failed peers. Each instance needs its own port.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import suppress
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup

from afd_plugin.connectors.npu.hccl_transport import HcclLayerInputs
from afd_plugin.distributed.afd_process_group import init_afd_process_group
from afd_plugin.distributed.hccl import HcclPeerSlice, HcclStageLayout
from afd_plugin.distributed.hccl_plan import (
    MAX_CONTROL_MESSAGE_BYTES,
    MAX_ERROR_BYTES,
    HcclLocalForward,
    HcclParentPlan,
    HcclPlanCapabilities,
    HcclPlanMessage,
    decode_hccl_plan_message,
    encode_hccl_plan_message,
)
from afd_plugin.distributed.hccl_session import MAX_SESSION_TRANSACTIONS

DEFAULT_TIMEOUT_SECONDS = 60


class HcclStagedTransport:
    """Rank-local blocking exchange with global control barriers.

    Attention sends every peer's inputs before receiving outputs. FFN receives
    a complete aggregate before caller-owned compute, then returns its slices.
    Both sides advance a layer-major schedule. No native TP reduction occurs.
    """

    def __init__(
        self,
        role: str,
        role_rank: int,
        capabilities: HcclPlanCapabilities,
        *,
        init_method: str,
        local_device: int,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        capabilities.validate()
        if role not in ("attention", "ffn"):
            raise ValueError("role must be attention or ffn")
        role_size = (
            capabilities.attention_size
            if role == "attention"
            else capabilities.ffn_size
        )
        if type(role_rank) is not int or not 0 <= role_rank < role_size:
            raise ValueError("role_rank is outside the declared topology")
        if type(local_device) is not int or local_device < 0:
            raise ValueError("local_device must be a nonnegative integer")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        self.role, self.role_rank = role, role_rank
        self.capabilities = self.negotiated_capabilities = capabilities
        self.world_rank = role_rank + (
            capabilities.ffn_size if role == "attention" else 0
        )
        self.world_size = capabilities.attention_size + capabilities.ffn_size
        self.coordinator_rank = capabilities.ffn_size
        self.local_device, self.init_method = local_device, init_method
        self.timeout = timedelta(seconds=timeout_seconds)
        self.control_group: ProcessGroup | None = None
        self.data_groups: list[ProcessGroup] = []
        self.plan: HcclParentPlan | None = None
        self._plan_digest: str | None = None
        self._steps: Iterator[tuple[int, int]] = iter(())
        self._next_step: tuple[int, int] | None = None
        self._pending_inputs: HcclLayerInputs | None = None
        self._counts_transaction_id: str | None = None
        self._local_forward: HcclLocalForward | None = None
        self._completed: set[str] = set()
        self.initialized = self.terminal = self.closed = False

    @property
    def is_coordinator(self) -> bool:
        return self.world_rank == self.coordinator_rank

    @property
    def _other_ranks(self) -> tuple[int, ...]:
        return tuple(
            rank for rank in range(self.world_size) if rank != self.coordinator_rank
        )

    def initialize(self) -> None:
        if self.initialized or self.closed:
            raise RuntimeError("transport cannot be initialized again")
        # Worker isolation: register the NPU backend in runtime processes only.
        import torch_npu  # noqa: F401

        torch.npu.set_device(self.local_device)
        try:
            self.control_group = self._create_group("gloo", "afd_hccl_v3_control")
            self._hello()
            for stage_idx in range(self.negotiated_capabilities.max_stages):
                self.data_groups.append(
                    self._create_group("hccl", f"afd_hccl_v3_stage_{stage_idx}")
                )
            self.initialized = True
        except BaseException:
            self.terminal = True
            with suppress(Exception):  # Preserve the initialization exception.
                self.close()
            raise

    def _create_group(self, backend: str, name: str) -> ProcessGroup:
        return init_afd_process_group(
            backend=backend,
            init_method=self.init_method,
            world_size=self.world_size,
            rank=self.world_rank,
            group_name=name,
            timeout=self.timeout,
        )

    def _hello(self) -> None:
        if self.is_coordinator:
            negotiated, error = self.capabilities, None
            for rank in self._other_ranks:
                try:
                    message = self._recv_control(rank)
                    role = "ffn" if rank < self.capabilities.ffn_size else "attention"
                    if (
                        message.kind != "HELLO"
                        or message.world_rank != rank
                        or message.role != role
                    ):
                        raise ValueError("HELLO does not match the sending rank")
                    assert message.capabilities is not None
                    negotiated = negotiated.negotiate(message.capabilities)
                except ValueError as exc:
                    error = error or str(exc)
            if error is not None:
                self._reject_peers(error)
            self.negotiated_capabilities = negotiated
            self._broadcast_control(
                HcclPlanMessage(
                    "HELLO",
                    role=self.role,
                    world_rank=self.world_rank,
                    capabilities=negotiated,
                )
            )
        else:
            self._send_control(
                self.coordinator_rank,
                HcclPlanMessage(
                    "HELLO",
                    role=self.role,
                    world_rank=self.world_rank,
                    capabilities=self.capabilities,
                ),
            )
            reply = self._expect(self.coordinator_rank, "HELLO")
            if reply.world_rank != self.coordinator_rank or reply.role != "attention":
                raise ValueError("expected the Attention0 HELLO")
            assert reply.capabilities is not None
            negotiated = self.capabilities.negotiate(reply.capabilities)
            if negotiated != reply.capabilities:
                raise ValueError("coordinator exceeded negotiated capacity")
            self.negotiated_capabilities = negotiated

    def _require_active(self, role: str | None = None) -> None:
        if not self.initialized or self.closed or self.terminal:
            raise RuntimeError("transport session is not active")
        if role is not None and self.role != role:
            raise RuntimeError(f"operation requires {role} role")

    def _validate_plan(self, plan: HcclParentPlan) -> None:
        if self.plan is not None:
            raise RuntimeError("previous parent transaction has not completed")
        plan.validate(self.negotiated_capabilities)
        encode_hccl_plan_message(HcclPlanMessage("FORWARD", plan=plan))
        if plan.transaction_id in self._completed:
            raise ValueError("transaction_id has already completed")
        if len(self._completed) >= MAX_SESSION_TRANSACTIONS:
            raise ValueError("session transaction limit reached; reconnect")
        if self._local_forward is not None:
            local = self._local_forward
            if (
                plan.transaction_id != self._counts_transaction_id
                or plan.mode != local.mode
                or tuple(s.attention_valid_tokens[self.role_rank] for s in plan.stages)
                != local.valid_tokens
                or tuple(s.attention_padded_tokens[self.role_rank] for s in plan.stages)
                != local.padded_tokens
            ):
                raise ValueError(
                    "parent plan does not match this Attention local report"
                )

    def _validate_local_forward(self, local: HcclLocalForward) -> None:
        if not isinstance(local, HcclLocalForward):
            raise ValueError("local report must be HcclLocalForward")
        local.validate()
        capabilities = self.negotiated_capabilities
        if (
            len(local.valid_tokens) > capabilities.max_stages
            or max(local.padded_tokens) > capabilities.max_tokens
        ):
            raise ValueError("local counts exceed negotiated stage or token capacity")

    def submit_forward(
        self,
        local: HcclLocalForward,
        *,
        transaction_id: str | None = None,
    ) -> HcclParentPlan | None:
        """Solicit actual Attention counts, then accept one canonical parent.

        All Attention ranks call this on one owning host thread. Attention0
        supplies the ID; peers receive REQUEST (or idle STOP) before sending.
        FFN ranks continue to call receive_forward. Complete invalid frames
        are drained before global ERROR; broken I/O needs external supervision.
        """
        self._require_active("attention")
        capabilities = self.negotiated_capabilities
        if capabilities.local_counts_version != 1:
            raise RuntimeError("submit_forward requires local_counts_version=1")
        if not (
            capabilities.attention_size
            == capabilities.ffn_size
            == capabilities.native_tp_size
        ):
            raise ValueError(
                "local count collection requires one native TP world per role"
            )
        if self.plan is not None or self._counts_transaction_id is not None:
            raise RuntimeError("previous parent or count round has not completed")
        try:
            if self.is_coordinator:
                try:
                    request = HcclPlanMessage(
                        "COUNTS_REQUEST", transaction_id=transaction_id
                    )
                    encode_hccl_plan_message(request)
                    self._validate_local_forward(local)
                    if transaction_id in self._completed:
                        raise ValueError("transaction_id has already completed")
                    if len(self._completed) >= MAX_SESSION_TRANSACTIONS:
                        raise ValueError("session transaction limit reached; reconnect")
                except (ValueError, RuntimeError) as exc:
                    self._reject_peers(str(exc))
                self._counts_transaction_id = transaction_id
                self._local_forward = local
                attention_peers = range(self.coordinator_rank + 1, self.world_size)
                for rank in attention_peers:
                    self._send_control(rank, request)
                reports = [local]
                error = None
                for rank in attention_peers:
                    try:
                        message = self._recv_control(rank)
                    except ValueError as exc:
                        # The sender's complete frame is drained. Continue so
                        # all requested senders can receive the final ERROR.
                        error = error or str(exc)
                        continue
                    try:
                        self._validate_reply(message, "COUNTS_REPORT", None)
                        if (
                            message.world_rank != rank
                            or message.transaction_id != transaction_id
                        ):
                            raise ValueError(
                                "COUNTS_REPORT identity does not match request"
                            )
                        assert message.local_forward is not None
                        self._validate_local_forward(message.local_forward)
                        reports.append(message.local_forward)
                    except (ValueError, RuntimeError) as exc:
                        error = error or str(exc)
                if error is not None:
                    self._reject_peers(error)
                try:
                    if any(
                        report.mode != local.mode
                        or len(report.valid_tokens) != len(local.valid_tokens)
                        for report in reports
                    ):
                        raise ValueError(
                            "Attention reports disagree on mode or stage count"
                        )
                    assert transaction_id is not None
                    plan = HcclParentPlan(
                        transaction_id,
                        capabilities.layer_indices,
                        tuple(
                            HcclStageLayout(
                                stage_idx,
                                tuple(
                                    report.valid_tokens[stage_idx] for report in reports
                                ),
                                tuple(
                                    report.padded_tokens[stage_idx]
                                    for report in reports
                                ),
                                capabilities.ffn_size,
                                capabilities.native_tp_size,
                            )
                            for stage_idx in range(len(local.valid_tokens))
                        ),
                        local.mode,
                    )
                    self._validate_plan(plan)
                except (ValueError, RuntimeError) as exc:
                    self._reject_peers(str(exc))
                self.begin_forward(plan)
                return plan

            try:
                request = self._recv_control(self.coordinator_rank)
            except ValueError as exc:
                self._report_rejection(str(exc))
                raise
            if request.kind == "STOP":
                self._acknowledge_stop()
                return None
            if request.kind == "ERROR":
                raise RuntimeError(f"coordinator rejected session: {request.error}")
            try:
                if request.kind != "COUNTS_REQUEST":
                    raise ValueError("expected COUNTS_REQUEST or STOP")
                if (
                    transaction_id is not None
                    and transaction_id != request.transaction_id
                ):
                    raise ValueError(
                        "COUNTS_REQUEST transaction_id does not match caller"
                    )
                if request.transaction_id in self._completed:
                    raise ValueError("transaction_id has already completed")
                self._validate_local_forward(local)
                self._counts_transaction_id = request.transaction_id
                self._local_forward = local
                report = HcclPlanMessage(
                    "COUNTS_REPORT",
                    world_rank=self.world_rank,
                    transaction_id=request.transaction_id,
                    local_forward=local,
                )
                encode_hccl_plan_message(report)
            except (ValueError, RuntimeError) as exc:
                self._report_rejection(str(exc))
                raise
            self._send_control(self.coordinator_rank, report)
            return self.receive_forward()
        except BaseException:
            self.terminal = True
            raise

    def _report_rejection(self, reason: str) -> None:
        self._send_control(self.coordinator_rank, self._error_message(reason))
        self._expect(self.coordinator_rank, "ERROR")

    def _acknowledge_stop(self) -> None:
        self._send_control(self.coordinator_rank, HcclPlanMessage("STOP_ACK"))
        self._expect(self.coordinator_rank, "STOP_ACK")
        self.terminal = True

    def _activate_plan(self, plan: HcclParentPlan) -> None:
        self.plan, self._plan_digest = plan, plan.digest
        self._steps = (
            (layer, stage.stage_idx)
            for layer in plan.layer_indices
            for stage in plan.stages
            if any(stage.attention_padded_tokens)
        )
        self._next_step = next(self._steps, None)

    def begin_forward(self, plan: HcclParentPlan) -> None:
        self._require_active()
        if not self.is_coordinator:
            raise RuntimeError("only Attention0 publishes a parent plan")
        if (
            self.negotiated_capabilities.local_counts_version == 1
            and self._local_forward is None
        ):
            raise RuntimeError("counts-enabled Attention must use submit_forward")
        self._validate_plan(plan)
        try:
            self._broadcast_control(HcclPlanMessage("FORWARD", plan=plan))
            self._collect_replies("ACCEPT", plan)
            self._broadcast_control(
                HcclPlanMessage(
                    "ACCEPT",
                    transaction_id=plan.transaction_id,
                    plan_digest=plan.digest,
                    world_rank=self.world_rank,
                )
            )
            self._activate_plan(plan)
        except BaseException:
            self.terminal = True
            raise

    def receive_forward(self) -> HcclParentPlan | None:
        self._require_active()
        if self.is_coordinator:
            raise RuntimeError("Attention0 must publish the parent plan")
        if self.plan is not None:
            raise RuntimeError("previous parent transaction has not completed")
        if (
            self.role == "attention"
            and self.negotiated_capabilities.local_counts_version == 1
            and self._local_forward is None
        ):
            raise RuntimeError("counts-enabled Attention must use submit_forward")
        try:
            try:
                message = self._recv_control(self.coordinator_rank)
            except ValueError as exc:
                self._report_rejection(str(exc))
                raise
            if message.kind == "STOP":
                if self._counts_transaction_id is not None:
                    raise ValueError("STOP requires an idle count boundary")
                self._acknowledge_stop()
                return None
            if message.kind == "ERROR":
                raise RuntimeError(f"coordinator rejected session: {message.error}")
            try:
                if message.kind != "FORWARD" or message.plan is None:
                    raise ValueError("expected FORWARD or STOP")
                plan = message.plan
                self._validate_plan(plan)
            except (ValueError, RuntimeError) as exc:
                self._report_rejection(str(exc))
                raise
            self._send_control(
                self.coordinator_rank,
                HcclPlanMessage(
                    "ACCEPT",
                    transaction_id=plan.transaction_id,
                    plan_digest=plan.digest,
                    world_rank=self.world_rank,
                ),
            )
            ready = self._expect(self.coordinator_rank, "ACCEPT", plan)
            if ready.world_rank != self.coordinator_rank:
                raise ValueError("global ACCEPT requires Attention0")
            self._activate_plan(plan)
            return plan
        except BaseException:
            self.terminal = True
            raise

    def _collect_replies(self, kind: str, plan: HcclParentPlan | None = None) -> None:
        error = None
        for rank in self._other_ranks:
            try:
                message = self._recv_control(rank)
            except ValueError as exc:
                # A complete but invalid JSON frame has been drained. Framing
                # or backend RuntimeError must escape: its sender may still
                # be blocked in I/O and cannot safely receive an ERROR.
                error = error or str(exc)
                continue
            try:
                self._validate_reply(message, kind, plan)
                if kind == "ACCEPT" and message.world_rank != rank:
                    raise ValueError("ACCEPT world_rank does not match sender")
            except (ValueError, RuntimeError) as exc:
                error = error or str(exc)
        # Drain every blocking reply before peers can receive ERROR.
        if error is not None:
            self._reject_peers(error)

    @staticmethod
    def _validate_reply(
        message: HcclPlanMessage, kind: str, plan: HcclParentPlan | None
    ) -> None:
        if message.kind == "ERROR":
            raise RuntimeError(f"peer rejected session: {message.error}")
        if message.kind != kind:
            raise ValueError(f"expected {kind}")
        if plan is not None and (
            message.transaction_id != plan.transaction_id
            or message.plan_digest != plan.digest
        ):
            raise ValueError(f"{kind} does not match the active parent digest")

    def _require_step(
        self, layer_idx: int, stage_idx: int
    ) -> tuple[HcclPeerSlice, ...]:
        if (
            type(layer_idx) is not int
            or type(stage_idx) is not int
            or self._next_step != (layer_idx, stage_idx)
        ):
            raise ValueError("layer/stage does not match the active parent sequence")
        assert self.plan is not None
        stage = self.plan.stages[stage_idx]
        return (
            stage.for_attention(self.role_rank)
            if self.role == "attention"
            else stage.for_ffn(self.role_rank)
        )

    def _check_tensor(
        self, tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != torch.device("npu", self.local_device)
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"expected contiguous {dtype} NPU tensor with shape {shape}"
            )

    def _check_ids(self, ids: torch.Tensor, valid_tokens: int) -> None:
        if ids.numel() and (
            int(ids.min().item()) < 0
            or int(ids.max().item()) >= self.capabilities.vocab_size
        ):
            raise ValueError("input_ids must lie within the model vocabulary")
        if bool(ids[valid_tokens:].any().item()):
            raise ValueError("padding input_ids must be zero")

    def send_layer_inputs(
        self,
        layer_idx: int,
        stage_idx: int,
        hidden_states: torch.Tensor,
        hidden_states_fp32: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> None:
        self._require_active("attention")
        peers = self._require_step(layer_idx, stage_idx)
        if self._pending_inputs is not None:
            raise RuntimeError(
                "receive the previous stage output before sending inputs"
            )
        assert self.plan is not None
        stage = self.plan.stages[stage_idx]
        native_rows = stage.attention_padded_tokens[self.role_rank]
        hidden_size = self.negotiated_capabilities.hidden_size
        self._check_tensor(hidden_states, (native_rows, hidden_size), torch.bfloat16)
        self._check_tensor(
            hidden_states_fp32, (native_rows, hidden_size), torch.float32
        )
        self._check_tensor(input_ids, (native_rows,), torch.int32)
        self._check_ids(input_ids, stage.attention_valid_tokens[self.role_rank])
        rows = sum(peer.padded_tokens for peer in peers)
        device = torch.device("npu", self.local_device)
        wire = HcclLayerInputs(
            torch.zeros((rows, hidden_size), dtype=torch.bfloat16, device=device),
            torch.zeros((rows, hidden_size), dtype=torch.float32, device=device),
            torch.zeros(rows, dtype=torch.int32, device=device),
        )
        try:
            wire.hidden_states[:native_rows].copy_(hidden_states)
            wire.hidden_states_fp32[:native_rows].copy_(hidden_states_fp32)
            wire.input_ids[:native_rows].copy_(input_ids)
            self._pending_inputs = wire
            for peer in peers:
                start, stop = (
                    peer.attention_offset,
                    peer.attention_offset + peer.padded_tokens,
                )
                for tensor in (
                    wire.hidden_states,
                    wire.hidden_states_fp32,
                    wire.input_ids,
                ):
                    dist.send(
                        tensor[start:stop],
                        group_dst=peer.ffn_world_rank,
                        group=self.data_groups[stage_idx],
                    )
        except BaseException:
            self.terminal = True
            raise

    def receive_layer_output(self, layer_idx: int, stage_idx: int) -> torch.Tensor:
        self._require_active("attention")
        peers = self._require_step(layer_idx, stage_idx)
        if self._pending_inputs is None:
            raise RuntimeError("send inputs before receiving a stage output")
        rows = sum(peer.padded_tokens for peer in peers)
        try:
            output = torch.empty(
                (rows, self.negotiated_capabilities.hidden_size),
                dtype=torch.bfloat16,
                device=torch.device("npu", self.local_device),
            )
            for peer in peers:
                start, stop = (
                    peer.attention_offset,
                    peer.attention_offset + peer.padded_tokens,
                )
                dist.recv(
                    output[start:stop],
                    group_src=peer.ffn_world_rank,
                    group=self.data_groups[stage_idx],
                )
            assert self.plan is not None
            native_rows = self.plan.stages[stage_idx].attention_padded_tokens[
                self.role_rank
            ]
            self._advance()
            return output[:native_rows]
        except BaseException:
            self.terminal = True
            raise

    def receive_layer(self, layer_idx: int, stage_idx: int) -> HcclLayerInputs:
        self._require_active("ffn")
        peers = self._require_step(layer_idx, stage_idx)
        if self._pending_inputs is not None:
            raise RuntimeError("send the previous stage output before receiving inputs")
        rows = sum(peer.padded_tokens for peer in peers)
        device = torch.device("npu", self.local_device)
        shape = (rows, self.negotiated_capabilities.hidden_size)
        inputs = HcclLayerInputs(
            torch.empty(shape, dtype=torch.bfloat16, device=device),
            torch.empty(shape, dtype=torch.float32, device=device),
            torch.empty(rows, dtype=torch.int32, device=device),
        )
        try:
            for peer in peers:
                start, stop = peer.ffn_offset, peer.ffn_offset + peer.padded_tokens
                for tensor in (
                    inputs.hidden_states,
                    inputs.hidden_states_fp32,
                    inputs.input_ids,
                ):
                    dist.recv(
                        tensor[start:stop],
                        group_src=peer.attention_world_rank,
                        group=self.data_groups[stage_idx],
                    )
                self._check_ids(inputs.input_ids[start:stop], peer.valid_tokens)
            self._pending_inputs = inputs
            return inputs
        except BaseException:
            self.terminal = True
            raise

    def send_layer_output(
        self, layer_idx: int, stage_idx: int, output: torch.Tensor
    ) -> None:
        self._require_active("ffn")
        peers = self._require_step(layer_idx, stage_idx)
        if self._pending_inputs is None:
            raise RuntimeError("receive all aggregate inputs before sending outputs")
        rows = sum(peer.padded_tokens for peer in peers)
        self._check_tensor(
            output, (rows, self.negotiated_capabilities.hidden_size), torch.bfloat16
        )
        try:
            for peer in peers:
                start, stop = peer.ffn_offset, peer.ffn_offset + peer.padded_tokens
                dist.send(
                    output[start:stop],
                    group_dst=peer.attention_world_rank,
                    group=self.data_groups[stage_idx],
                )
            self._advance()
        except BaseException:
            self.terminal = True
            raise

    def _advance(self) -> None:
        self._pending_inputs = None
        self._next_step = next(self._steps, None)

    def finish_forward(self) -> None:
        self._require_active()
        if (
            self.plan is None
            or self._next_step is not None
            or self._pending_inputs is not None
        ):
            raise RuntimeError("COMPLETE requires all local stage/layer outputs")
        plan = self.plan
        message = HcclPlanMessage(
            "COMPLETE",
            transaction_id=plan.transaction_id,
            plan_digest=self._plan_digest,
        )
        try:
            # A host control barrier alone does not prove device work ended.
            torch.npu.synchronize(self.local_device)
            if self.is_coordinator:
                self._collect_replies("COMPLETE", plan)
                self._broadcast_control(message)
            else:
                self._send_control(self.coordinator_rank, message)
                self._expect(self.coordinator_rank, "COMPLETE", plan)
            self._completed.add(plan.transaction_id)
            self._clear_plan()
        except BaseException:
            self.terminal = True
            raise

    def stop_attention(self) -> None:
        """Stop an idle session from every Attention worker, without reports.

        Attention0 publishes STOP. The other Attention ranks only receive it,
        acknowledge and await the global ACK; both counts extension versions
        use the same shutdown handshake. FFN remains in receive_forward.
        """
        self._require_active("attention")
        if self.plan is not None or self._counts_transaction_id is not None:
            raise RuntimeError("STOP requires an idle parent boundary")
        if self.is_coordinator:
            self.stop()
            return
        try:
            try:
                message = self._recv_control(self.coordinator_rank)
            except ValueError as exc:
                self._report_rejection(str(exc))
                raise
            if message.kind == "ERROR":
                raise RuntimeError(f"coordinator rejected session: {message.error}")
            if message.kind != "STOP":
                self._report_rejection("expected STOP at Attention shutdown")
                raise ValueError("expected STOP at Attention shutdown")
            self._acknowledge_stop()
        except BaseException:
            self.terminal = True
            raise

    def stop(self) -> None:
        self._require_active()
        if not self.is_coordinator:
            raise RuntimeError("only Attention0 requests STOP")
        if self.plan is not None or self._counts_transaction_id is not None:
            raise RuntimeError("STOP requires an idle parent boundary")
        try:
            self._broadcast_control(HcclPlanMessage("STOP"))
            self._collect_replies("STOP_ACK")
            self._broadcast_control(HcclPlanMessage("STOP_ACK"))
            self.terminal = True
        except BaseException:
            self.terminal = True
            raise

    def _send_control(self, rank: int, message: HcclPlanMessage) -> None:
        data = encode_hccl_plan_message(message)
        size = torch.tensor([len(data)], dtype=torch.int64, device="cpu")
        body = torch.tensor(list(data), dtype=torch.uint8, device="cpu")
        dist.send(size, group_dst=rank, group=self.control_group)
        dist.send(body, group_dst=rank, group=self.control_group)

    def _recv_control(self, rank: int) -> HcclPlanMessage:
        size = torch.empty(1, dtype=torch.int64, device="cpu")
        dist.recv(size, group_src=rank, group=self.control_group)
        length = int(size.item())
        if not 0 < length <= MAX_CONTROL_MESSAGE_BYTES:
            # An invalid length can leave an undrained body at the sender.
            # It is a framing/I/O failure, not a recoverable decoded reply.
            raise RuntimeError("invalid HCCL control frame size")
        body = torch.empty(length, dtype=torch.uint8, device="cpu")
        dist.recv(body, group_src=rank, group=self.control_group)
        return decode_hccl_plan_message(bytes(body.tolist()))

    def _broadcast_control(self, message: HcclPlanMessage) -> None:
        for rank in self._other_ranks:
            self._send_control(rank, message)

    def _expect(
        self, rank: int, kind: str, plan: HcclParentPlan | None = None
    ) -> HcclPlanMessage:
        message = self._recv_control(rank)
        self._validate_reply(message, kind, plan)
        return message

    @staticmethod
    def _error_message(reason: str) -> HcclPlanMessage:
        bounded = reason.encode("utf-8", errors="replace")[:MAX_ERROR_BYTES].decode(
            "utf-8", errors="ignore"
        )
        return HcclPlanMessage("ERROR", error=bounded or "HCCL session failed")

    def _reject_peers(self, reason: str) -> None:
        self.terminal = True
        self._broadcast_control(self._error_message(reason))
        raise RuntimeError(reason)

    def _clear_plan(self) -> None:
        self.plan, self._plan_digest = None, None
        self._steps, self._next_step = iter(()), None
        self._pending_inputs = None
        self._counts_transaction_id = None
        self._local_forward = None

    def close(self) -> None:
        """Destroy plugin groups in reverse creation order, never native TP."""
        if self.closed:
            return
        self.closed = self.terminal = True
        self._clear_plan()
        groups = [*reversed(self.data_groups), self.control_group]
        self.data_groups, self.control_group = [], None
        first_error = None
        for group in groups:
            if group is not None:
                try:
                    dist.destroy_process_group(group)
                except Exception as exc:
                    first_error = first_error or exc
        if first_error is not None:
            raise first_error
