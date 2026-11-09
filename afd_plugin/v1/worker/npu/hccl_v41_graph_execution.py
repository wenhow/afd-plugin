# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""One explicitly reserved native V4.1 FULL preparation and parent callback.

Construct this owner before native ModelState input writes. A capture/replay
coordinator retains it with the graph's buffers and completes the accepted
transaction outside capture. This seam does not select production Graph mode,
copy a native capture factory, or replace Ascend's final retirement.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    ForwardContext,
    create_forward_context,
    override_forward_context,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.ubatch_utils import UBatchState
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext

from afd_plugin.connectors.npu.hccl_graph_pipeline import HcclGraphComputePipeline
from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank
from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.distributed.hccl_plan import HcclLocalForward
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    HCCL_V41_PARENT_FORWARD_KEY,
    AFDDeepseekV41AttentionForCausalLM,
)
from afd_plugin.model_executor.models.npu.deepseek_v41_graph import (
    DeepseekV41GraphParent,
    DeepseekV41GraphStages,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph import (
    HcclV41GraphGeometry,
    HcclV41StageGeometry,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph_control import (
    HcclV41GraphControl,
    HcclV41GraphReservation,
)


class HcclV41GraphExecution:
    """Implements the HCCL runner's owning-parent graph callback contract.

    One instance belongs to one reservation. The graph cache must retain the
    bank, native builders/input storage, this owner and the native graph/output
    objects as one generation. A later replay may prepare new metadata objects
    only over that same retained native storage.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: AFDDeepseekV41AttentionForCausalLM,
        model_state: EngramModelState,
        control: HcclV41GraphControl,
        reservation: HcclV41GraphReservation,
        bank: HcclGraphStageBank,
        *,
        pipeline: HcclGraphComputePipeline | None = None,
    ) -> None:
        self.transport: HcclStagedTransport = control.transport
        self.control, self.reservation = control, reservation
        self.vllm_config, self.model, self.model_state = vllm_config, model, model_state
        self.bank = bank
        self.pipeline = pipeline
        self.prepared_state: UBatchState | None = None
        self.parent_body: DeepseekV41GraphParent | None = None
        self.parent_context: ForwardContext | None = None
        geometry = reservation.geometry
        native = model.language_model.model
        if (
            reservation.decision.phase not in ("warmup", "capture", "replay")
            or geometry.descriptor.cg_mode != CUDAGraphMode.FULL
            or bank.role != "attention"
            or bank.role_rank != self.transport.role_rank
            or bank.key != reservation.decision.key
            or bank.capabilities != self.transport.negotiated_capabilities
            or bank.layer_indices
            != self.transport.negotiated_capabilities.layer_indices
            or bank.native_tp_size != vllm_config.parallel_config.tensor_parallel_size
            or not isinstance(model_state, EngramModelState)
            or (pipeline is not None and pipeline.bank is not bank)
        ):
            self.fail()
            raise ValueError(
                "native FULL owner requires its accepted static HCCL session"
            )
        if len(self.transport.data_groups) < len(bank.exchanges) or any(
            exchange is not None
            and exchange.endpoint.group is not self.transport.data_groups[index]
            for index, exchange in enumerate(bank.exchanges)
        ):
            self.fail()
            raise ValueError(
                "native FULL owner requires this session's graph endpoints"
            )
        if native.has_engram and not native.can_capture_engram_producer():
            self.fail()
            raise ValueError(
                "unsupported Engram producer must select eager before preparation"
            )
        self.control.verify_prepared(reservation, geometry)

    def fail(self) -> None:
        """Retain graph dependencies; external supervision owns failed peers."""
        self.transport.terminal = True
        self.control.cache.fail()

    def require_full(self, input_batch: InputBatch, *, for_capture: bool) -> None:
        """Reject a foreign/stale phase before native child preparation."""
        try:
            self.control.verify_prepared(self.reservation, self.reservation.geometry)
            desc = self.reservation.geometry.descriptor
            if (
                self.prepared_state is not None
                or for_capture != (self.reservation.decision.phase != "replay")
                or input_batch.num_tokens_after_padding != desc.num_tokens
                or input_batch.num_reqs_after_padding != desc.num_reqs
            ):
                raise ValueError(
                    "native FULL preparation must use the current parent phase"
                )
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
        """Accept actual native child counts before the parent producer runs."""
        try:
            self.require_full(input_batch, for_capture=for_capture)
            actual = HcclV41GraphGeometry(
                self.reservation.geometry.descriptor,
                tuple(
                    HcclV41StageGeometry(
                        stage.request_slice.start,
                        stage.request_slice.stop,
                        stage.token_slice.start,
                        stage.token_slice.stop,
                    )
                    for stage in ubatch_state.slices
                ),
            )
            self.control.verify_prepared(self.reservation, actual)
            body = DeepseekV41GraphParent(
                DeepseekV41GraphStages(
                    self.model.language_model.model,
                    tuple(ubatch_state.forward_contexts),
                    tuple(stage.token_slice for stage in ubatch_state.slices),
                    self.bank,
                    pipeline=self.pipeline,
                )
            )
            context = create_forward_context(
                None,
                self.vllm_config,
                dp_metadata=None,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
            )
            # Build the native parent kwargs before capture. Attention selects
            # each child's retained metadata only during its tensor operations.
            context.additional_kwargs.update(
                current_platform.set_additional_forward_context(
                    None,
                    self.vllm_config,
                    None,
                    num_tokens=self.reservation.geometry.descriptor.num_tokens,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                )
            )
            context.additional_kwargs[HCCL_V41_PARENT_FORWARD_KEY] = body
            valid = 0 if for_capture else input_batch.num_tokens
            local = HcclLocalForward(
                self.reservation.decision.mode,
                tuple(
                    max(0, min(stage.token_slice.stop, valid) - stage.token_slice.start)
                    for stage in ubatch_state.slices
                ),
                tuple(
                    stage.token_slice.stop - stage.token_slice.start
                    for stage in ubatch_state.slices
                ),
            )
            plan = self.transport.submit_forward(
                local,
                transaction_id=(
                    self.reservation.decision.transaction_id
                    if self.transport.is_coordinator
                    else None
                ),
            )
            if plan is None:
                raise RuntimeError(
                    "native graph session stopped before parent acceptance"
                )
            self.bank.update_counts(plan)
            self.prepared_state, self.parent_body = ubatch_state, body
            self.parent_context = context
        except BaseException:
            self.fail()
            raise

    def forward_parent(
        self,
        model: nn.Module,
        model_inputs: dict[str, Any],
        ubatch_state: UBatchState,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Call the supplied native wrapper once around the whole parent.

        All host agreement and count writes have finished in ``prepared``.
        Only context selection and the model's tensor body occur here; the
        outer manager owns warmup/capture completion and persistent outputs.
        """
        if (
            ubatch_state is not self.prepared_state
            or self.parent_body is None
            or self.parent_context is None
        ):
            self.fail()
            raise ValueError(
                "graph callback requires this owner's prepared native state"
            )
        if (
            self.transport.terminal
            or self.transport.closed
            or self.transport.plan is None
            or self.transport.graph_reservation is not self.reservation.decision
        ):
            self.fail()
            raise ValueError("graph callback requires the current accepted reservation")
        if (
            not isinstance(model, ModelWithContext)
            or model.get_original_model() is not self.model
            or model.device_metadata is not self.model_state.device_metadata
        ):
            self.fail()
            raise ValueError(
                "graph parent requires its native ModelWithContext wrapper"
            )
        try:
            # Native parent input preparation follows child preparation. Check
            # storage generation and metadata shapes again before the first
            # producer/consumer. These callbacks inspect CPU identity and
            # dimensions only; no wire, tensor readback or late fallback.
            self.control.verify_prepared(self.reservation, self.reservation.geometry)
            with override_forward_context(self.parent_context):
                return model(**model_inputs)
        except BaseException:
            self.fail()
            raise
