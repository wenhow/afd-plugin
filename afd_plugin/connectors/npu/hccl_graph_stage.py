# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Stage-major tensor storage for one paired-TP graph/session layout.

Only ``allocate`` resolves caller-owned groups and allocates NPU storage. A
manually constructed CPU bank is an operator/layout test seam, not an HCCL
implementation. A bank and every graph retaining it belong to one transport
session. The caller agrees the phase/key, updates counts outside capture, and
keeps all storage alive until its graphs have been synchronized and reset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from afd_plugin.connectors.npu.hccl_graph import (
    HcclGraphBuffers,
    HcclGraphEndpoint,
    HcclGraphExchange,
)
from afd_plugin.distributed.hccl_graph import HcclGraphKey
from afd_plugin.distributed.hccl_plan import HcclParentPlan, HcclPlanCapabilities

if TYPE_CHECKING:
    from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport


class HcclGraphStageBank:
    """One exchange per stage, reused across layers on one compute stream.

    The output returned to Attention is a borrowed view. Its continuation must
    consume it before that stage's next layer exchange overwrites the buffer.
    Zero-row stages are skipped by both callers using the same static key;
    a positive bucket with zero valid tokens still executes its tensor body.
    Native TP, request history, events and parent completion remain caller-owned.
    """

    def __init__(
        self,
        *,
        role: str,
        role_rank: int,
        capabilities: HcclPlanCapabilities,
        key: HcclGraphKey,
        exchanges: tuple[HcclGraphExchange | None, ...],
    ) -> None:
        capabilities.validate()
        if (
            capabilities.graph_execution_version != 1
            or capabilities.local_counts_version != 1
            or not capabilities.attention_size
            == capabilities.ffn_size
            == capabilities.native_tp_size
        ):
            raise ValueError(
                "graph stage bank requires graph/counts v1 paired native TP"
            )
        if (
            role not in ("attention", "ffn")
            or type(role_rank) is not int
            or not 0 <= role_rank < capabilities.native_tp_size
        ):
            raise ValueError("graph stage bank requires a valid paired role/rank")
        if (
            key.num_tokens > capabilities.max_tokens
            or key.num_ubatches > capabilities.max_stages
            or len(exchanges) != key.num_ubatches
        ):
            raise ValueError("graph stage bank key exceeds the session layout")
        world_rank = role_rank + (capabilities.ffn_size if role == "attention" else 0)
        peer_rank = role_rank + (capabilities.ffn_size if role == "ffn" else 0)
        ranks = tuple(range(capabilities.attention_size + capabilities.ffn_size))
        payloads = []
        device = None
        for rows, exchange in zip(key.padded_tokens, exchanges, strict=True):
            if not rows:
                if exchange is not None:
                    raise ValueError("zero-row graph stages must have no exchange")
                continue
            if exchange is None:
                raise ValueError("positive graph stages require an exchange")
            if (
                exchange.role != role
                or exchange.hidden_shape != (rows, capabilities.hidden_size)
                or exchange.endpoint.ranks != ranks
                or exchange.endpoint.peer_group_rank != peer_rank
                or exchange.endpoint.group.rank() != world_rank
            ):
                raise ValueError(
                    "graph stage endpoint does not match paired role/layout"
                )
            buffers = exchange.buffers
            if device is None:
                device = buffers.hidden_states.device
            elif buffers.hidden_states.device != device:
                raise ValueError("graph stage buffers must share one device")
            payloads.extend(
                (
                    buffers.hidden_states,
                    buffers.hidden_states_fp32,
                    buffers.input_ids,
                    buffers.output,
                )
            )
        # A key always has positive total rows, even when one child is empty.
        assert device is not None
        if device.type == "npu" and torch.npu.is_current_stream_capturing():
            raise RuntimeError("construct graph stage bank outside capture")
        spans = sorted(
            (
                tensor.data_ptr(),
                tensor.data_ptr() + tensor.numel() * tensor.element_size(),
            )
            for tensor in payloads
        )
        if any(
            stop > start
            for (_, stop), (start, _) in zip(spans, spans[1:], strict=False)
        ):
            raise ValueError("graph stage payload storage must not overlap")
        self.role, self.role_rank = role, role_rank
        self.capabilities, self.key, self.exchanges = capabilities, key, exchanges
        self.native_tp_size = capabilities.native_tp_size
        self.layer_indices, self.stage_rows = (
            capabilities.layer_indices,
            key.padded_tokens,
        )
        self.device = device
        self.valid_counts = tuple(
            torch.zeros(1, dtype=torch.int32, device=device) for _ in exchanges
        )
        self.row_indices = tuple(
            torch.arange(rows, dtype=torch.int32, device=device)
            for rows in self.stage_rows
        )
        self.padding_masks = tuple(
            torch.empty(rows, dtype=torch.bool, device=device)
            for rows in self.stage_rows
        )

    @classmethod
    def allocate(
        cls, transport: HcclStagedTransport, key: HcclGraphKey, *, device: torch.device
    ) -> HcclGraphStageBank:
        """Resolve fixed endpoints and allocate before warmup/capture begins."""
        if (
            not transport.initialized
            or transport.closed
            or transport.terminal
            or device != torch.device("npu", transport.local_device)
        ):
            raise ValueError(
                "graph bank requires an active transport on its indexed NPU"
            )
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("allocate graph stage bank outside capture")
        capabilities = transport.negotiated_capabilities
        capabilities.validate()
        if (
            capabilities.graph_execution_version != 1
            or key.num_tokens > capabilities.max_tokens
            or key.num_ubatches > capabilities.max_stages
            or len(transport.data_groups) < key.num_ubatches
        ):
            raise ValueError(
                "graph stage bank key exceeds the negotiated graph session"
            )
        peer_rank = transport.role_rank + (
            capabilities.ffn_size if transport.role == "ffn" else 0
        )
        exchanges = tuple(
            HcclGraphExchange(
                HcclGraphEndpoint.from_process_group(
                    transport.data_groups[index], peer_rank
                ),
                HcclGraphBuffers.allocate(rows, capabilities.hidden_size, device),
                role=transport.role,
            )
            if rows
            else None
            for index, rows in enumerate(key.padded_tokens)
        )
        return cls(
            role=transport.role,
            role_rank=transport.role_rank,
            capabilities=capabilities,
            key=key,
            exchanges=exchanges,
        )

    def update_counts(self, plan: HcclParentPlan) -> None:
        """Validate the accepted graph parent before changing any device scalar.

        This does not accept or complete a parent. The control owner must first
        bind this session/key to its accepted reservation and actual geometry.
        IDs must already satisfy the parent's vocabulary/int32 input policy;
        the tensor body only converts their dtype and masks padded rows.
        """
        if self.device.type == "npu" and torch.npu.is_current_stream_capturing():
            raise RuntimeError("update graph stage counts outside capture")
        plan.validate(self.capabilities)
        decision = plan.graph_execution
        if (
            decision is None
            or decision.phase == "eager"
            or decision.key != self.key
            or plan.layer_indices != self.layer_indices
            or decision.padded_tokens != self.stage_rows
        ):
            raise ValueError(
                "graph counts require this bank's accepted graph key/layout"
            )
        counts = tuple(
            stage.attention_valid_tokens[self.role_rank] for stage in plan.stages
        )
        for target, count in zip(self.valid_counts, counts, strict=True):
            target.fill_(count)

    def buffers(self, stage_idx: int) -> HcclGraphBuffers:
        """Return a positive stage's retained storage, without communication."""
        return self._exchange(stage_idx).buffers

    def _exchange(self, stage_idx: int) -> HcclGraphExchange:
        if type(stage_idx) is not int or not 0 <= stage_idx < len(self.exchanges):
            raise ValueError("graph stage index is outside the retained layout")
        exchange = self.exchanges[stage_idx]
        if exchange is None:
            raise ValueError("skip zero-row graph stage on both peers")
        return exchange

    def _padding_mask(self, stage_idx: int) -> torch.Tensor:
        mask = self.padding_masks[stage_idx]
        torch.ge(self.row_indices[stage_idx], self.valid_counts[stage_idx], out=mask)
        return mask

    def attention_exchange(
        self,
        stage_idx: int,
        hidden_states: torch.Tensor,
        hidden_states_fp32: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Record/call only tensor copies, padding mask and paired exchange."""
        if self.role != "attention":
            raise RuntimeError("graph Attention exchange requires Attention role")
        exchange = self._exchange(stage_idx)
        buffers = exchange.buffers
        for source, target, dtypes in (
            (hidden_states, buffers.hidden_states, (torch.bfloat16,)),
            (hidden_states_fp32, buffers.hidden_states_fp32, (torch.float32,)),
            (input_ids, buffers.input_ids, (torch.int32, torch.int64)),
        ):
            if (
                source.shape != target.shape
                or source.device != self.device
                or source.dtype not in dtypes
                or not source.is_contiguous()
            ):
                raise ValueError(
                    "graph Attention inputs require static shape/dtype/device/layout"
                )
        buffers.hidden_states.copy_(hidden_states)
        buffers.hidden_states_fp32.copy_(hidden_states_fp32)
        buffers.input_ids.copy_(input_ids)
        buffers.input_ids.masked_fill_(self._padding_mask(stage_idx), 0)
        exchange.send_inputs()
        exchange.receive_output()
        return buffers.output

    def receive_inputs(self, stage_idx: int) -> HcclGraphBuffers:
        if self.role != "ffn":
            raise RuntimeError("graph input receive requires FFN role")
        exchange = self._exchange(stage_idx)
        exchange.receive_inputs()
        return exchange.buffers

    def send_output(self, stage_idx: int, output: torch.Tensor) -> None:
        if self.role != "ffn":
            raise RuntimeError("graph output send requires FFN role")
        exchange = self._exchange(stage_idx)
        if (
            output.shape != exchange.buffers.output.shape
            or output.dtype != torch.bfloat16
            or output.device != self.device
            or not output.is_contiguous()
        ):
            raise ValueError(
                "graph FFN output requires static BF16 shape/device/layout"
            )
        exchange.buffers.output.copy_(output)
        exchange.buffers.output.masked_fill_(
            self._padding_mask(stage_idx).unsqueeze(1), 0
        )
        exchange.send_output()
