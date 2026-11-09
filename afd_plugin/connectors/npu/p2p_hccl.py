# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Typed eager HCCL adapter for corresponding native TP rank pairs.

The target plan is explicit and separate from the legacy DP control plane.
Native V4.1 activation and router tensors remain owned by their model roles.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
)
from afd_plugin.connectors.npu.hccl_config import (
    HcclExtraInfo,
    validate_hccl_runtime,
)
from afd_plugin.connectors.npu.hccl_protocol import HcclForwardPlan, HcclRole
from afd_plugin.connectors.npu.hccl_transport import HcclEagerTransport
from afd_plugin.distributed.hccl import build_hccl_rank_mapping

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(slots=True)
class HCCLTransferState(AFDTransferState):
    """Bind one remote MoE exchange to its existing target transaction."""

    transaction_id: str
    layer_idx: int


class P2pHcclAFDConnector(AFDConnectorBase):
    """Exchange three input tensors and a complete FFN output per target layer.

    ``close`` only releases owned groups. Normal shutdown uses the explicit
    Attention-side ``request_stop`` at an idle transaction boundary first.
    Device faults require worker supervision, as documented by the transport.
    """

    @classmethod
    def parse_extra_config(cls, raw: Mapping[str, Any] | None) -> HcclExtraInfo:
        return HcclExtraInfo.from_mapping(raw)

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        if type(rank) is not int or rank != role_rank:
            raise ValueError(
                "P2pHcclAFDConnector requires caller rank equal to role_rank"
            )
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        capabilities = validate_hccl_runtime(vllm_config, afd_config, role_rank)
        mapping = build_hccl_rank_mapping(
            afd_config, role_rank, native_tp_size=capabilities.native_tp_size
        )
        self._target_layer_indices = capabilities.layer_indices
        self.attn_size = afd_config.num_attention_ranks
        self.ffn_size = afd_config.num_ffn_ranks
        self._role = HcclRole(afd_config.role)
        pair_port = afd_config.port + mapping.subgroup_index
        self._transport = HcclEagerTransport(
            self._role,
            capabilities,
            init_method=f"tcp://{afd_config.host}:{pair_port}",
            local_device=local_rank,
            timeout_seconds=afd_config.afd_process_group_timeout_s,
        )
        self._initialized = False
        self._active_plan: HcclForwardPlan | None = None
        self._layer_position = 0
        self._pending_context: AFDTransferContext | None = None
        self._pending_tensor_spec: (
            tuple[tuple[int, ...], torch.dtype, torch.device] | None
        ) = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def active_forward_plan(self) -> HcclForwardPlan | None:
        return self._active_plan

    @property
    def target_layer_indices(self) -> tuple[int, ...]:
        """The model layers verified by the peer capability handshake."""
        return self._target_layer_indices

    def init_afd_connector(self) -> None:
        self._transport.initialize()
        self._initialized = True

    def close(self) -> None:
        self._initialized = False
        self._active_plan = None
        self._pending_context = None
        self._pending_tensor_spec = None
        self._transport.close()

    def _require_role(self, role: HcclRole) -> None:
        if self._role is not role:
            raise RuntimeError(f"HCCL operation requires {role.value} role")
        if not self.is_initialized:
            raise RuntimeError("HCCL connector is not initialized")

    def begin_forward_plan(self, plan: HcclForwardPlan) -> None:
        """Attention sends a plan carrying the existing forward transaction ID."""
        self._require_role(HcclRole.ATTENTION)
        if self._active_plan is not None or self._pending_context is not None:
            raise RuntimeError("previous HCCL forward has not been consumed")
        self._transport.begin_forward(plan)
        self._active_plan = plan if plan.padded_tokens else None
        self._layer_position = 0

    def receive_forward_plan(self) -> HcclForwardPlan | None:
        """FFN accepts the next target plan, or acknowledges an idle STOP."""
        self._require_role(HcclRole.FFN)
        if self._active_plan is not None or self._pending_context is not None:
            raise RuntimeError("previous HCCL forward has not completed")
        plan = self._transport.receive_forward()
        self._active_plan = plan if plan is not None and plan.padded_tokens else None
        self._layer_position = 0
        if plan is None:
            self._initialized = False
        return plan

    def request_stop(self) -> None:
        self._require_role(HcclRole.ATTENTION)
        if self._active_plan is not None or self._pending_context is not None:
            raise RuntimeError("HCCL STOP requires an idle transaction boundary")
        self._transport.stop()
        self._initialized = False

    def _require_layer(self, layer_idx: int) -> HcclForwardPlan:
        plan = self._active_plan
        if (
            type(layer_idx) is not int
            or plan is None
            or plan.layer_indices[self._layer_position] != layer_idx
        ):
            raise RuntimeError("layer does not match the active HCCL target plan")
        return plan

    def _validate_context(self, context: AFDTransferContext) -> HcclForwardPlan:
        metadata = context.metadata
        plan = self._require_layer(metadata.layer_idx)
        if (
            type(metadata.stage_idx) is not int
            or metadata.stage_idx != 0
            or metadata.seq_lens != [plan.padded_tokens]
            or any(type(size) is not int for size in metadata.seq_lens)
        ):
            raise ValueError(
                "HCCL context must match stage 0 and the padded token count"
            )
        expected_state = HCCLTransferState(plan.transaction_id, metadata.layer_idx)
        state = context.states
        if (
            not isinstance(state, HCCLTransferState)
            or type(state.layer_idx) is not int
            or type(state.transaction_id) is not str
            or state != expected_state
        ):
            raise ValueError("HCCL context does not match the transaction and layer")
        return plan

    def _finish_layer(self) -> None:
        assert self._active_plan is not None
        self._pending_context = None
        self._pending_tensor_spec = None
        self._layer_position += 1
        if self._layer_position == len(self._active_plan.layer_indices):
            self._active_plan = None
            self._layer_position = 0

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        self._require_role(HcclRole.ATTENTION)
        if self._pending_context is not None:
            raise RuntimeError("consume the previous FFN output before sending a layer")
        plan = self._require_layer(context.metadata.layer_idx)
        if context.states is None:
            context.states = HCCLTransferState(
                plan.transaction_id, context.metadata.layer_idx
            )
        self._validate_context(context)
        hidden_states_fp32 = kwargs.get("hidden_states_fp32")
        input_ids = kwargs.get("input_ids")
        if not isinstance(hidden_states_fp32, torch.Tensor):
            raise ValueError(
                "HCCL Attention send requires independent hidden_states_fp32"
            )
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("HCCL Attention send requires token-aligned input_ids")
        self._transport.send_layer_inputs(
            context.metadata.layer_idx, hidden_states, hidden_states_fp32, input_ids
        )
        self._pending_context = context
        self._pending_tensor_spec = (
            tuple(hidden_states.shape),
            hidden_states.dtype,
            hidden_states.device,
        )

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._require_role(HcclRole.ATTENTION)
        context = kwargs.get("context")
        if self._pending_context is None:
            raise RuntimeError("no HCCL FFN output is pending")
        if context is not self._pending_context:
            raise ValueError("FFN receive requires the original transfer context")
        if type(ubatch_idx) is not int or ubatch_idx != 0:
            raise ValueError("HCCL supports only ubatch_idx=0")
        self._validate_context(context)
        tensor_spec = (tuple(ref_tensor.shape), ref_tensor.dtype, ref_tensor.device)
        if tensor_spec != self._pending_tensor_spec or not ref_tensor.is_contiguous():
            raise ValueError("FFN output buffer must match the sent tensor layout")
        output = self._transport.receive_layer_output(context.metadata.layer_idx)
        ref_tensor.copy_(output)
        self._finish_layer()
        return ref_tensor

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        self._require_role(HcclRole.FFN)
        if self._pending_context is not None:
            raise RuntimeError("send the previous FFN output before receiving a layer")
        if type(ubatch_idx) is not int or ubatch_idx != 0:
            raise ValueError("HCCL supports only ubatch_idx=0")
        layer_idx = kwargs.get("layer_idx")
        if type(layer_idx) is not int:
            raise ValueError("HCCL Attention receive requires layer_idx")
        plan = self._require_layer(layer_idx)
        inputs = self._transport.receive_layer(layer_idx)
        context = AFDTransferContext(
            AFDTransferMetadata(layer_idx, 0, [plan.padded_tokens]),
            HCCLTransferState(plan.transaction_id, layer_idx),
        )
        self._pending_context = context
        return AFDA2FTransferPayload(
            hidden_states=inputs.hidden_states,
            context=context,
            input_ids=inputs.input_ids,
            hidden_states_fp32=inputs.hidden_states_fp32,
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        self._require_role(HcclRole.FFN)
        if self._pending_context is None:
            raise RuntimeError(
                "receive the HCCL layer inputs before sending its output"
            )
        if context is not self._pending_context:
            raise ValueError("FFN send requires the original transfer context")
        self._validate_context(context)
        self._transport.send_layer_output(context.metadata.layer_idx, ffn_output)
        self._finish_layer()


__all__ = ["HCCLTransferState", "P2pHcclAFDConnector"]
