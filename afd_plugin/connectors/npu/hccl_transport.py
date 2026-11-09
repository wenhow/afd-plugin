# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Eager two-peer HCCL transport used by each native TP rank pair.

Only the blocking tensor primitives and plugin-owned group semantics come from
the locked donor. Its graph, MTP, cached IDs and runner protocols are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup

from afd_plugin.connectors.npu.hccl_protocol import (
    MAX_CONTROL_MESSAGE_BYTES,
    HcclCapabilities,
    HcclControlMessage,
    HcclForwardPlan,
    HcclMessageType,
    HcclRole,
    decode_hccl_control_message,
    encode_hccl_control_message,
)
from afd_plugin.distributed.afd_process_group import init_afd_process_group

HCCL_WORLD_SIZE = 2
HCCL_FFN_RANK = 0
HCCL_ATTENTION_RANK = 1
DEFAULT_TIMEOUT_SECONDS = 60
# Bound replay tracking. Reconnect at a transaction boundary when exhausted.
MAX_SESSION_TRANSACTIONS = 65536


@dataclass(frozen=True, slots=True)
class HcclLayerInputs:
    hidden_states: torch.Tensor
    hidden_states_fp32: torch.Tensor
    input_ids: torch.Tensor


class HcclEagerTransport:
    """Own two groups and one sequential eager exchange, without model state.

    Device indices are local to each process. F world rank is 0 and A is 1;
    these ranks do not represent native TP/DP groups. No default group is owned
    or destroyed here. Runtime topology validation belongs to the future
    connector adapter. Failed sessions must be closed and re-created.

    Returned tensors follow NPU stream semantics. A caller observing a deferred
    device error while consuming an output must terminate the worker session.
    The group timeout does not replace HCCL's execution/connect timeouts or an
    external process supervisor; close is not a bounded failure-recovery API.
    """

    def __init__(
        self,
        role: HcclRole,
        capabilities: HcclCapabilities,
        *,
        init_method: str,
        local_device: int,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        capabilities.validate()
        if not isinstance(role, HcclRole):
            raise ValueError("role must be HcclRole")
        if type(local_device) is not int or local_device < 0:
            raise ValueError("local_device must be a nonnegative integer")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        self.role = role
        self.capabilities = capabilities
        self.negotiated_capabilities = capabilities
        self.init_method = init_method
        self.local_device = local_device
        self.timeout = timedelta(seconds=timeout_seconds)
        self.world_rank = (
            HCCL_ATTENTION_RANK if role is HcclRole.ATTENTION else HCCL_FFN_RANK
        )
        self.peer_rank = HCCL_WORLD_SIZE - 1 - self.world_rank
        self.data_group: ProcessGroup | None = None
        self.control_group: ProcessGroup | None = None
        self.plan: HcclForwardPlan | None = None
        self.layer_position = 0
        self.awaiting_output = False
        self.pending_inputs: HcclLayerInputs | None = None
        self.completed_transactions: set[str] = set()
        self.initialized = False
        self.terminal = False
        self.closed = False

    def initialize(self) -> None:
        if self.initialized or self.closed:
            raise RuntimeError("transport cannot be initialized again")
        # Worker isolation: importing torch-npu registers HCCL in this process.
        import torch_npu  # noqa: F401

        torch.npu.set_device(self.local_device)
        try:
            self.data_group = init_afd_process_group(
                backend="hccl",
                init_method=self.init_method,
                world_size=HCCL_WORLD_SIZE,
                rank=self.world_rank,
                group_name="afd_hccl_eager_data",
                timeout=self.timeout,
            )
            self.control_group = init_afd_process_group(
                backend="gloo",
                init_method=self.init_method,
                world_size=HCCL_WORLD_SIZE,
                rank=self.world_rank,
                group_name="afd_hccl_eager_control",
                timeout=self.timeout,
            )
            hello = HcclControlMessage(
                HcclMessageType.HELLO, role=self.role, capabilities=self.capabilities
            )
            # Blocking send(size), send(body): never send HELLO simultaneously.
            if self.role is HcclRole.ATTENTION:
                self._send_control(hello)
                peer = self._expect(HcclMessageType.HELLO)
                self._negotiate(peer)
            else:
                peer = self._expect(HcclMessageType.HELLO)
                try:
                    self._negotiate(peer)
                except ValueError as exc:
                    self._send_error(str(exc))
                    raise
                self._send_control(hello)
            self.initialized = True
        except BaseException:
            self.terminal = True
            self.close()
            raise

    def _negotiate(self, message: HcclControlMessage) -> None:
        expected_role = (
            HcclRole.FFN if self.role is HcclRole.ATTENTION else HcclRole.ATTENTION
        )
        if message.role is not expected_role or message.capabilities is None:
            raise ValueError("HELLO requires the opposite role and capabilities")
        self.negotiated_capabilities = self.capabilities.negotiate(message.capabilities)

    def _require_role(self, role: HcclRole) -> None:
        if self.role is not role:
            raise RuntimeError(f"operation requires {role.value} role")
        if not self.initialized or self.terminal or self.closed:
            raise RuntimeError("transport session is not active")

    def _validate_new_plan(self, plan: HcclForwardPlan) -> None:
        if self.plan is not None:
            raise RuntimeError("previous transaction has not completed")
        plan.validate(self.negotiated_capabilities)
        if plan.transaction_id in self.completed_transactions:
            raise ValueError("transaction_id has already completed")
        if len(self.completed_transactions) >= MAX_SESSION_TRANSACTIONS:
            raise ValueError("session transaction limit reached; reconnect")

    def begin_forward(self, plan: HcclForwardPlan) -> None:
        self._require_role(HcclRole.ATTENTION)
        self._validate_new_plan(plan)
        self._send_control(HcclControlMessage(HcclMessageType.FORWARD, plan=plan))
        self._expect(HcclMessageType.ACCEPT, plan.transaction_id)
        self.plan = plan
        self.layer_position = 0
        if plan.padded_tokens == 0:
            self._expect(HcclMessageType.COMPLETE, plan.transaction_id)
            self._finish_transaction()

    def receive_forward(self) -> HcclForwardPlan | None:
        self._require_role(HcclRole.FFN)
        if self.plan is not None:
            raise RuntimeError("previous transaction has not completed")
        message = self._recv_control()
        if message.kind is HcclMessageType.STOP:
            self._send_control(HcclControlMessage(HcclMessageType.STOP_ACK))
            self.terminal = True
            return None
        if message.kind is not HcclMessageType.FORWARD or message.plan is None:
            self._send_error("expected FORWARD or STOP")
            raise RuntimeError("expected FORWARD or STOP")
        plan = message.plan
        try:
            self._validate_new_plan(plan)
        except (ValueError, RuntimeError) as exc:
            self._send_error(str(exc), plan.transaction_id)
            raise
        self._send_control(
            HcclControlMessage(
                HcclMessageType.ACCEPT, transaction_id=plan.transaction_id
            )
        )
        self.plan = plan
        self.layer_position = 0
        if plan.padded_tokens == 0:
            self._complete_ffn_transaction()
        return plan

    def _require_layer(self, layer_idx: int) -> HcclForwardPlan:
        if (
            self.plan is None
            or self.plan.layer_indices[self.layer_position] != layer_idx
        ):
            raise RuntimeError("layer does not match the active transaction sequence")
        return self.plan

    def _validate_tensor(
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

    def _validate_inputs(self, inputs: HcclLayerInputs, plan: HcclForwardPlan) -> None:
        shape = (plan.padded_tokens, self.negotiated_capabilities.hidden_size)
        self._validate_tensor(inputs.hidden_states, shape, torch.bfloat16)
        self._validate_tensor(inputs.hidden_states_fp32, shape, torch.float32)
        self._validate_tensor(inputs.input_ids, (plan.padded_tokens,), torch.int32)
        # Compare Python integers so an int32 vocabulary boundary cannot wrap.
        # These host synchronizations are intentional first-path validation cost.
        if (
            int(inputs.input_ids.min().item()) < 0
            or int(inputs.input_ids.max().item()) >= self.capabilities.vocab_size
        ):
            raise ValueError("input_ids must lie within the model vocabulary")
        if bool(inputs.input_ids[plan.valid_tokens :].any().item()):
            raise ValueError("padding input_ids must be zero")

    def exchange_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        hidden_states_fp32: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        self.send_layer_inputs(layer_idx, hidden_states, hidden_states_fp32, input_ids)
        return self.receive_layer_output(layer_idx)

    def send_layer_inputs(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        hidden_states_fp32: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> None:
        """Send one layer; retain its inputs until the matching receive finishes."""
        self._require_role(HcclRole.ATTENTION)
        plan = self._require_layer(layer_idx)
        if self.awaiting_output:
            raise RuntimeError(
                "receive the previous output before sending another layer"
            )
        inputs = HcclLayerInputs(hidden_states, hidden_states_fp32, input_ids)
        try:
            self._validate_inputs(inputs, plan)
            for tensor in (hidden_states, hidden_states_fp32, input_ids):
                dist.send(tensor, group_dst=self.peer_rank, group=self.data_group)
            self.pending_inputs = inputs
            self.awaiting_output = True
        except BaseException:
            self.terminal = True
            raise

    def receive_layer_output(self, layer_idx: int) -> torch.Tensor:
        self._require_role(HcclRole.ATTENTION)
        plan = self._require_layer(layer_idx)
        if not self.awaiting_output:
            raise RuntimeError("send layer inputs before receiving its output")
        try:
            output = torch.empty(
                (plan.padded_tokens, self.negotiated_capabilities.hidden_size),
                dtype=torch.bfloat16,
                device=torch.device("npu", self.local_device),
            )
            dist.recv(output, group_src=self.peer_rank, group=self.data_group)
            self.awaiting_output = False
            self.pending_inputs = None
            self.layer_position += 1
            if self.layer_position == len(plan.layer_indices):
                self._expect(HcclMessageType.COMPLETE, plan.transaction_id)
                self._finish_transaction()
            return output
        except BaseException:
            self.terminal = True
            raise

    def receive_layer(self, layer_idx: int) -> HcclLayerInputs:
        self._require_role(HcclRole.FFN)
        plan = self._require_layer(layer_idx)
        if self.awaiting_output:
            raise RuntimeError("the previous layer output has not been sent")
        shape = (plan.padded_tokens, self.negotiated_capabilities.hidden_size)
        device = torch.device("npu", self.local_device)
        inputs = HcclLayerInputs(
            torch.empty(shape, dtype=torch.bfloat16, device=device),
            torch.empty(shape, dtype=torch.float32, device=device),
            torch.empty((plan.padded_tokens,), dtype=torch.int32, device=device),
        )
        try:
            for tensor in (
                inputs.hidden_states,
                inputs.hidden_states_fp32,
                inputs.input_ids,
            ):
                dist.recv(tensor, group_src=self.peer_rank, group=self.data_group)
            self._validate_inputs(inputs, plan)
            self.awaiting_output = True
            return inputs
        except BaseException:
            self.terminal = True
            raise

    def send_layer_output(self, layer_idx: int, output: torch.Tensor) -> None:
        self._require_role(HcclRole.FFN)
        plan = self._require_layer(layer_idx)
        if not self.awaiting_output:
            raise RuntimeError("layer inputs must be received before its output")
        try:
            self._validate_tensor(
                output,
                (plan.padded_tokens, self.negotiated_capabilities.hidden_size),
                torch.bfloat16,
            )
            dist.send(output, group_dst=self.peer_rank, group=self.data_group)
            self.awaiting_output = False
            self.layer_position += 1
            if self.layer_position == len(plan.layer_indices):
                self._complete_ffn_transaction()
        except BaseException:
            self.terminal = True
            raise

    def _complete_ffn_transaction(self) -> None:
        assert self.plan is not None
        self._send_control(
            HcclControlMessage(
                HcclMessageType.COMPLETE, transaction_id=self.plan.transaction_id
            )
        )
        self._finish_transaction()

    def _finish_transaction(self) -> None:
        assert self.plan is not None
        self.completed_transactions.add(self.plan.transaction_id)
        self.plan = None
        self.layer_position = 0
        self.pending_inputs = None

    def stop(self) -> None:
        self._require_role(HcclRole.ATTENTION)
        if self.plan is not None:
            raise RuntimeError("STOP requires an idle transaction boundary")
        self._send_control(HcclControlMessage(HcclMessageType.STOP))
        self._expect(HcclMessageType.STOP_ACK)
        self.terminal = True

    def _send_control(self, message: HcclControlMessage) -> None:
        try:
            data = encode_hccl_control_message(message)
            size = torch.tensor([len(data)], dtype=torch.int64)
            body = torch.tensor(list(data), dtype=torch.uint8)
            dist.send(size, group_dst=self.peer_rank, group=self.control_group)
            dist.send(body, group_dst=self.peer_rank, group=self.control_group)
        except BaseException:
            self.terminal = True
            raise

    def _recv_control(self) -> HcclControlMessage:
        try:
            size = torch.empty(1, dtype=torch.int64)
            dist.recv(size, group_src=self.peer_rank, group=self.control_group)
            length = int(size.item())
            if not 0 < length <= MAX_CONTROL_MESSAGE_BYTES:
                raise ValueError("invalid HCCL control frame size")
            body = torch.empty(length, dtype=torch.uint8)
            dist.recv(body, group_src=self.peer_rank, group=self.control_group)
            message = decode_hccl_control_message(bytes(body.tolist()))
            if message.kind is HcclMessageType.ERROR:
                raise RuntimeError(f"peer rejected HCCL session: {message.error}")
            return message
        except BaseException:
            self.terminal = True
            raise

    def _expect(
        self, kind: HcclMessageType, transaction_id: str | None = None
    ) -> HcclControlMessage:
        message = self._recv_control()
        if message.kind is not kind or message.transaction_id != transaction_id:
            self.terminal = True
            raise RuntimeError(
                f"expected {kind.value} for transaction {transaction_id}"
            )
        return message

    def _send_error(self, reason: str, transaction_id: str | None = None) -> None:
        self.terminal = True
        self._send_control(
            HcclControlMessage(
                HcclMessageType.ERROR, error=reason, transaction_id=transaction_id
            )
        )

    def close(self) -> None:
        """Release owned groups once; callers perform STOP before normal close.

        This does not promise to interrupt a peer's blocked data operation.
        Backend timeout and the supervising process bound failed sessions.
        """
        if self.closed:
            return
        self.closed = True
        self.terminal = True
        groups = (self.control_group, self.data_group)
        self.control_group = None
        self.data_group = None
        self.plan = None
        self.pending_inputs = None
        first_error: Exception | None = None
        for group in groups:
            if group is not None:
                try:
                    dist.destroy_process_group(group)
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
        if first_error is not None:
            raise first_error
