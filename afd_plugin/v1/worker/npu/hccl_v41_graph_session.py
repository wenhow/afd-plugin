# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Persistent runner binding for successive explicitly reserved graph phases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.ubatch_utils import UBatchState

if TYPE_CHECKING:
    from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
    from afd_plugin.v1.worker.npu.hccl_v41_ubatch_runner import HcclV41GraphOwner


class HcclV41GraphSession:
    """Bind a runner before its first native factory reservation exists.

    The caller arms one accepted owner before native dummy/input preparation,
    completes it outside capture, then clears it. Failure retains that owner's
    graph dependencies for external supervision; no cancel or cleanup is sent.
    """

    def __init__(self, transport: HcclStagedTransport) -> None:
        if transport.role != "attention":
            raise ValueError("native graph session requires Attention transport")
        self._transport = transport
        self._current_owner: HcclV41GraphOwner | None = None

    @property
    def transport(self) -> HcclStagedTransport:
        return self._transport

    def arm(self, owner: HcclV41GraphOwner) -> None:
        if (
            self._current_owner is not None
            or isinstance(owner, HcclV41GraphSession)
            or owner.transport is not self.transport
            or not self.transport.initialized
            or self.transport.closed
            or self.transport.terminal
        ):
            self.fail()
            raise ValueError("native graph session requires one active owning phase")
        self._current_owner = owner

    def clear(self) -> None:
        if (
            self._current_owner is None
            or self.transport.terminal
            or self.transport.plan is not None
            or self.transport.graph_reservation is not None
        ):
            self.fail()
            raise ValueError(
                "complete the native graph phase before clearing its owner"
            )
        self._current_owner = None

    def _owner(self) -> HcclV41GraphOwner:
        if self._current_owner is None or self.transport.terminal:
            self.fail()
            raise ValueError("native graph session requires an armed phase owner")
        return self._current_owner

    def require_full(self, input_batch: InputBatch, *, for_capture: bool) -> None:
        try:
            self._owner().require_full(input_batch, for_capture=for_capture)
        except BaseException:
            self.fail()
            raise

    def prepared(
        self,
        input_batch: InputBatch,
        ubatch_state: UBatchState,
        *,
        for_capture: bool,
    ) -> None:
        try:
            self._owner().prepared(input_batch, ubatch_state, for_capture=for_capture)
        except BaseException:
            self.fail()
            raise

    def forward_parent(
        self,
        model: nn.Module,
        model_inputs: dict[str, Any],
        ubatch_state: UBatchState,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        try:
            return self._owner().forward_parent(model, model_inputs, ubatch_state)
        except BaseException:
            self.fail()
            raise

    def fail(self) -> None:
        self.transport.terminal = True
        if self._current_owner is not None:
            self._current_owner.fail()
