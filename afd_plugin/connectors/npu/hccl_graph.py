# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Static tensor exchange for an explicitly paired HCCL graph component.

This module owns no groups, parent plans, host cursors or graph cache. The caller
must agree the same positive row count and stage order on both peers before
warmup/capture/replay. No production Graph selector uses this component yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed import distributed_c10d as c10d


@dataclass(frozen=True, slots=True)
class HcclGraphEndpoint:
    """Group coordinates resolved outside tracing; the caller keeps ownership."""

    group: ProcessGroup
    ranks: tuple[int, ...]
    group_tag: str
    peer_group_rank: int
    tag: int = 0

    @classmethod
    def from_process_group(
        cls, group: ProcessGroup, peer_world_rank: int, *, tag: int = 0
    ) -> HcclGraphEndpoint:
        if type(peer_world_rank) is not int or peer_world_rank < 0:
            raise ValueError("graph peer must be a nonnegative world rank")
        if type(tag) is not int or tag < 0:
            raise ValueError("graph tag must be a nonnegative integer")
        if dist.get_backend(group) != "hccl":
            raise ValueError("graph exchange requires an HCCL process group")
        ranks = tuple(dist.get_process_group_ranks(group))
        if peer_world_rank not in ranks:
            raise ValueError("graph peer is outside the process group")
        peer_group_rank = ranks.index(peer_world_rank)
        # AFD groups can have a rank domain independent of the native default
        # world. dist.get_rank(group) translates through that default world;
        # ProcessGroup.rank() is the actual coordinate of this domain.
        if peer_group_rank == group.rank():
            raise ValueError("graph exchange requires a different peer")
        # Explicitly register the real torch-npu operators before any capture.
        # This deferred import keeps NPU initialization at the endpoint boundary.
        from torch_npu.dynamo.npugraph_ex.ops import _hcom_send_recv  # noqa: F401

        return cls(group, ranks, c10d._get_group_tag(group), peer_group_rank, tag)

    def send(self, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
        # torch-npu 2.10.0.post4 _hcom_send_recv.py:14-15,62-64. Use the explicit
        # group-relative coordinate; do not pass it as a global dst. Retain the
        # static shape also needed by the target GE receive converter.
        torch.ops.npu_define._send.default(
            tensor,
            None,
            self.ranks,
            self.group_tag,
            self.tag,
            self.peer_group_rank,
            shape,
        )

    def receive(self, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
        received = torch.ops.npu_define._recv.default(
            tensor,
            None,
            self.ranks,
            self.group_tag,
            self.tag,
            self.peer_group_rank,
            shape,
        )
        tensor.copy_(received)


@dataclass(frozen=True, slots=True)
class HcclGraphBuffers:
    """Fixed addresses for BF16 activations, original FP32 router rows and IDs."""

    hidden_states: torch.Tensor
    hidden_states_fp32: torch.Tensor
    input_ids: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(
        cls, rows: int, hidden_size: int, device: torch.device
    ) -> HcclGraphBuffers:
        if type(rows) is not int or rows <= 0:
            raise ValueError(
                "graph rows must be positive; skip empty stages on both peers"
            )
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("graph hidden size must be positive")
        if device.type != "npu" or device.index is None:
            raise ValueError("graph buffers require an indexed NPU device")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("allocate graph buffers outside capture")
        shape = (rows, hidden_size)
        return cls(
            torch.zeros(shape, dtype=torch.bfloat16, device=device),
            torch.zeros(shape, dtype=torch.float32, device=device),
            torch.zeros(rows, dtype=torch.int32, device=device),
            torch.zeros(shape, dtype=torch.bfloat16, device=device),
        )


class HcclGraphExchange:
    """Tensor-only operations over one already agreed positive stage shape.

    The caller invokes Attention send/receive and FFN receive/compute/send in
    matching order. Repeated graph replay does not mutate Python protocol state.
    This component never destroys its endpoint's caller-owned process group.
    Allocation enforces NPU placement; construction validates storage/layout.
    CPU tensor injection is useful only with an external operator test seam;
    the real graph operators require the NPU backend.
    """

    def __init__(
        self, endpoint: HcclGraphEndpoint, buffers: HcclGraphBuffers, *, role: str
    ) -> None:
        if role not in ("attention", "ffn"):
            raise ValueError("graph exchange requires attention or ffn role")
        hidden = buffers.hidden_states
        if hidden.ndim != 2 or min(hidden.shape) <= 0:
            raise ValueError("graph buffers require positive rows and hidden size")
        self.endpoint, self.buffers, self.role = endpoint, buffers, role
        self.hidden_shape = tuple(hidden.shape)
        self.ids_shape = (self.hidden_shape[0],)
        self._inputs = (
            (buffers.hidden_states, self.hidden_shape, torch.bfloat16),
            (buffers.hidden_states_fp32, self.hidden_shape, torch.float32),
            (buffers.input_ids, self.ids_shape, torch.int32),
        )
        payloads = (
            *self._inputs,
            (buffers.output, self.hidden_shape, torch.bfloat16),
        )
        for tensor, shape, dtype in payloads:
            self._validate_tensor(tensor, shape, dtype)
        spans = sorted(
            (
                tensor.data_ptr(),
                tensor.data_ptr() + tensor.numel() * tensor.element_size(),
            )
            for tensor, _, _ in payloads
        )
        if any(
            stop > following
            for (_, stop), (following, _) in zip(spans, spans[1:], strict=False)
        ):
            raise ValueError("graph payload buffers must not overlap")

    def _validate_tensor(
        self, tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != self.buffers.hidden_states.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "graph tensor must match static shape, dtype, device and layout"
            )

    def update_inputs(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp32: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> None:
        """Validate and copy the entire padded stage outside graph execution."""
        if self.role != "attention":
            raise RuntimeError("only Attention updates graph inputs")
        if (
            self.buffers.hidden_states.device.type == "npu"
            and torch.npu.is_current_stream_capturing()
        ):
            raise RuntimeError("update graph inputs outside capture")
        sources = (hidden_states, hidden_states_fp32, input_ids)
        # Validate every source first; a rejected update must not partly replace
        # the previous replay's inputs. IDs/valid-token range remain plan policy.
        for source, (_, shape, dtype) in zip(sources, self._inputs, strict=True):
            self._validate_tensor(source, shape, dtype)
        for source, (target, _, _) in zip(sources, self._inputs, strict=True):
            target.copy_(source)

    def send_inputs(self) -> None:
        if self.role != "attention":
            raise RuntimeError("graph input send requires Attention")
        for tensor, shape, _ in self._inputs:
            self.endpoint.send(tensor, shape)

    def receive_inputs(self) -> None:
        if self.role != "ffn":
            raise RuntimeError("graph input receive requires FFN")
        for tensor, shape, _ in self._inputs:
            self.endpoint.receive(tensor, shape)

    def send_output(self) -> None:
        if self.role != "ffn":
            raise RuntimeError("graph output send requires FFN")
        self.endpoint.send(self.buffers.output, self.hidden_shape)

    def receive_output(self) -> None:
        if self.role != "attention":
            raise RuntimeError("graph output receive requires Attention")
        self.endpoint.receive(self.buffers.output, self.hidden_shape)
