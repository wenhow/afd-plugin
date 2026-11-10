# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Connector identity and lifecycle for the V4.1 staged target service.

Normal NPU construction selects this adapter without opening model admission.
Native execution owns parent submit/COMPLETE and the worker owns idle STOP.
Tensor methods implement the base contract against that same accepted parent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from afd_plugin.compat.npu.runtime_config import DeepseekV41TargetShape
from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import AFDConnectorBase
from afd_plugin.connectors.factory import AFDConnectorFactory
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
)
from afd_plugin.connectors.npu.hccl_config import (
    HcclExtraInfo,
    uses_staged_hccl_graph_runtime,
    uses_staged_hccl_runtime,
    validate_staged_hccl_runtime,
)
from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.distributed import resolve_role_rank

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(slots=True)
class HcclStagedTransferState(AFDTransferState):
    transaction_id: str
    layer_idx: int
    stage_idx: int


class HcclStagedAFDConnector(AFDConnectorBase):
    """Share one counts-enabled session across the runner and connector API."""

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
        *,
        target_shape: DeepseekV41TargetShape | None = None,
    ) -> None:
        caps = validate_staged_hccl_runtime(
            vllm_config,
            afd_config,
            role_rank,
            graph=uses_staged_hccl_graph_runtime(vllm_config, afd_config),
            target_shape=target_shape,
        )
        if type(rank) is not int or rank != role_rank:
            raise ValueError("staged HCCL requires caller rank equal to role_rank")
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self.attn_size, self.ffn_size = caps.attention_size, caps.ffn_size
        self.transport = HcclStagedTransport(
            afd_config.role,
            role_rank,
            caps,
            init_method=f"tcp://{afd_config.host}:{afd_config.port}",
            local_device=local_rank,
            timeout_seconds=afd_config.afd_process_group_timeout_s,
        )
        self._pending_context: AFDTransferContext | None = None
        self._pending_tensor_spec: (
            tuple[tuple[int, ...], torch.dtype, torch.device] | None
        ) = None

    @property
    def is_initialized(self) -> bool:
        return (
            self.transport.initialized
            and not self.transport.closed
            and not self.transport.terminal
        )

    def init_afd_connector(self) -> None:
        self.transport.initialize()

    def close(self) -> None:
        self.transport.close()
        self._pending_context = None
        self._pending_tensor_spec = None

    def _validate_context(self, context: AFDTransferContext) -> None:
        plan = self.transport.plan
        if not self.is_initialized or plan is None:
            raise RuntimeError("staged connector requires an accepted parent")
        metadata = context.metadata
        if (
            type(metadata.stage_idx) is not int
            or not 0 <= metadata.stage_idx < len(plan.stages)
            or type(metadata.layer_idx) is not int
            or metadata.layer_idx not in plan.layer_indices
        ):
            raise ValueError("transfer layer/stage must match the accepted parent")
        stage = plan.stages[metadata.stage_idx]
        rows = (
            stage.attention_padded_tokens[self.role_rank]
            if self.afd_config.role == "attention"
            else sum(peer.padded_tokens for peer in stage.for_ffn(self.role_rank))
        )
        if (
            metadata.seq_lens != [rows]
            or any(type(size) is not int for size in metadata.seq_lens)
            or context.states
            != HcclStagedTransferState(
                plan.transaction_id, metadata.layer_idx, metadata.stage_idx
            )
        ):
            raise ValueError("transfer state/counts must match the accepted parent")

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        if self.afd_config.role != "attention":
            raise RuntimeError("Attention send requires the Attention role")
        if self._pending_context is not None:
            raise RuntimeError("receive the previous staged FFN output first")
        plan = self.transport.plan
        if plan is None:
            raise RuntimeError("staged connector requires an accepted parent")
        metadata = context.metadata
        if context.states is None:
            context.states = HcclStagedTransferState(
                plan.transaction_id, metadata.layer_idx, metadata.stage_idx
            )
        self._validate_context(context)
        fp32, ids = kwargs.get("hidden_states_fp32"), kwargs.get("input_ids")
        if not isinstance(fp32, torch.Tensor) or not isinstance(ids, torch.Tensor):
            raise ValueError("staged send requires independent FP32 and raw IDs")
        self.transport.send_layer_inputs(
            metadata.layer_idx, metadata.stage_idx, hidden_states, fp32, ids
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
        if self.afd_config.role != "attention":
            raise RuntimeError("Attention receive requires the Attention role")
        context = kwargs.get("context")
        if context is None or context is not self._pending_context:
            raise ValueError("staged receive requires the original transfer context")
        self._validate_context(context)
        if type(ubatch_idx) is not int or ubatch_idx != context.metadata.stage_idx:
            raise ValueError("ubatch_idx must match the accepted stage")
        if (
            tuple(ref_tensor.shape),
            ref_tensor.dtype,
            ref_tensor.device,
        ) != self._pending_tensor_spec or not ref_tensor.is_contiguous():
            raise ValueError("FFN output buffer must match the sent tensor layout")
        output = self.transport.receive_layer_output(
            context.metadata.layer_idx, ubatch_idx
        )
        try:
            ref_tensor.copy_(output)
        except BaseException:
            # The reply was consumed and the transport cursor advanced.
            # Retrying this exchange would split the peer schedules.
            self.transport.terminal = True
            raise
        self._pending_context = None
        self._pending_tensor_spec = None
        return ref_tensor

    def recv_attn_output(
        self, ubatch_idx: int = 0, **kwargs: Any
    ) -> AFDA2FTransferPayload:
        if self.afd_config.role != "ffn":
            raise RuntimeError("FFN receive requires the FFN role")
        if self._pending_context is not None:
            raise RuntimeError("send the previous staged FFN output first")
        layer_idx = kwargs.get("layer_idx")
        plan = self.transport.plan
        if plan is None:
            raise RuntimeError("staged connector requires an accepted parent")
        if type(layer_idx) is not int or type(ubatch_idx) is not int:
            raise ValueError("staged receive requires integer layer_idx/ubatch_idx")
        if not 0 <= ubatch_idx < len(plan.stages):
            raise ValueError("ubatch_idx must match the accepted stage")
        rows = sum(
            peer.padded_tokens
            for peer in plan.stages[ubatch_idx].for_ffn(self.role_rank)
        )
        context = AFDTransferContext(
            AFDTransferMetadata(layer_idx, ubatch_idx, [rows]),
            HcclStagedTransferState(plan.transaction_id, layer_idx, ubatch_idx),
        )
        self._validate_context(context)
        payload = self.transport.receive_layer(layer_idx, ubatch_idx)
        self._pending_context = context
        return AFDA2FTransferPayload(
            hidden_states=payload.hidden_states,
            hidden_states_fp32=payload.hidden_states_fp32,
            input_ids=payload.input_ids,
            context=context,
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        if self.afd_config.role != "ffn":
            raise RuntimeError("FFN send requires the FFN role")
        if context is not self._pending_context:
            raise ValueError("staged send requires the original transfer context")
        self._validate_context(context)
        self.transport.send_layer_output(
            context.metadata.layer_idx, context.metadata.stage_idx, ffn_output
        )
        self._pending_context = None


def create_npu_runtime_connector(
    rank: int,
    local_rank: int,
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    *,
    target_shape: DeepseekV41TargetShape | None = None,
) -> AFDConnectorBase:
    """Select the staged target adapter; preserve other connector factories."""
    if uses_staged_hccl_runtime(vllm_config, afd_config):
        return HcclStagedAFDConnector(
            rank,
            local_rank,
            vllm_config,
            afd_config,
            resolve_role_rank(vllm_config, afd_config),
            target_shape=target_shape,
        )
    return AFDConnectorFactory.create_connector(
        rank, local_rank, vllm_config, afd_config
    )
