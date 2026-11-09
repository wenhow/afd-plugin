# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit V2 input/run seams for the unadmitted staged HCCL model path.

The native factory does not select this class. The caller must accept the v3
parent plan and invoke run_staged in the native parent forward context. Native
Ascend execute_model retains the sole finish_execution/Engram retirement.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    ForwardContext,
    create_forward_context,
    get_forward_context,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.ubatch_utils import UBatchState, slice_model_inputs
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState

from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    AFDDeepseekV41AttentionForCausalLM,
)
from afd_plugin.model_executor.models.npu.deepseek_v41_staged import (
    DeepseekV41EngramInputs,
    DeepseekV41StageInputs,
    run_deepseek_v41_stages,
)
from afd_plugin.v1.worker.npu.ubatch_runner import NPUUBatchRunner


class HcclV41UBatchRunner(NPUUBatchRunner):
    # Upstream: UBatchRunner.prepare @ ced6857a and EngramModelState.prepare_attn
    # @ 7e37e29a. Patch reason: NPU child metadata preparation temporarily
    # bypasses Engram's override that records the parent's graph mode.
    # Patch functionality: explicitly set NONE before the native parent's
    # subsequent prepare_inputs prepares its single history/lookup producer.
    # Signature matches native prepare; no added parameters. Delegation keeps
    # native request/KV slicing and Ascend metadata builders in their owners.
    # Removal: use an upstream ModelState parent-mode preparation hook.
    def prepare(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        for_capture: bool = False,
    ) -> UBatchState:
        # ### PATCH START: explicit eager Engram parent mode
        if cg_mode != CUDAGraphMode.NONE or for_capture:
            raise ValueError("V4.1 staged HCCL preparation requires eager execution")
        if not isinstance(self.model_state, EngramModelState):
            raise ValueError("V4.1 staged HCCL requires the native Engram ModelState")
        if (
            self.parallel_config.pipeline_parallel_size != 1
            or self.parallel_config.decode_context_parallel_size != 1
            or self.parallel_config.prefill_context_parallel_size != 1
            or self.parallel_config.use_sequence_parallel_moe
        ):
            raise ValueError("V4.1 staged HCCL requires PP1, CP1 and non-SP execution")
        previous_mode = self.model_state._cg_mode
        self.model_state._cg_mode = CUDAGraphMode.NONE
        try:
            return super().prepare(
                input_batch, block_tables, slot_mappings, cg_mode, for_capture
            )
        except BaseException:
            self.model_state._cg_mode = previous_mode
            raise
        # Keep NONE on success until the native parent prepare_inputs consumes
        # it. Restoring here would silently select a stale producer mode.
        # ### PATCH END: explicit eager Engram parent mode

    # Upstream: UBatchRunner._make_forward_contexts @ ced6857a:389-421.
    # Patch reason: native DPMetadata.make asserts for DP1/non-SP and invents
    # uniform child counts for DP>1, whereas HCCL peers use the accepted plan.
    # Patch functionality: A-only contexts have no local-MoE DP sidecar; native
    # parent Engram preparation retains its real DP context. F owns MoE counts.
    # Signature matches upstream; no added parameters. Remove when the native
    # context factory supports externally scheduled A-only microbatches.
    def _make_forward_contexts(
        self,
        ubatch_slices: UBatchSlices,
        attn_metadata: list[dict[str, Any]],
        slot_mappings_by_layer: list[dict[str, torch.Tensor]],
        is_padding: list[torch.Tensor | None],
    ) -> list[ForwardContext]:
        forward_contexts = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            num_tokens = ubatch_slice.num_tokens
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i],
                    self.vllm_config,
                    # ### PATCH START: A-only child; plan owns actual peer counts
                    dp_metadata=None,
                    # ### PATCH END: A-only child; plan owns actual peer counts
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    slot_mapping=slot_mappings_by_layer[i],
                    is_padding=is_padding[i],
                )
            )
            # ### PATCH START: Ascend child kwargs without native handoff threads
            context = forward_contexts[-1]
            context.additional_kwargs = current_platform.set_additional_forward_context(
                context.attn_metadata,
                self.vllm_config,
                context.dp_metadata,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                ubatch_slices=ubatch_slices,
            )
            context.additional_kwargs.update(
                ubatch_idx=i, num_ubatches=len(ubatch_slices), dbo_enabled=False
            )
            context.ubatch_idx = i
            # ### PATCH END: Ascend child kwargs without native handoff threads
        return forward_contexts

    def run_staged(
        self,
        model: AFDDeepseekV41AttentionForCausalLM,
        model_inputs: dict[str, Any],
        ubatch_state: UBatchState,
        transport: HcclStagedTransport,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Run one accepted parent without native microbatch threads.

        Any is confined to the native ModelState's heterogeneous kwargs at this
        boundary. Model execution uses typed stage inputs. This method neither
        submits child history nor retires the native parent's producer.
        """
        try:
            parent_context = get_forward_context()
            if (
                parent_context.cudagraph_runtime_mode != CUDAGraphMode.NONE
                or model_inputs.get("engram_graph_events", False)
                or model_inputs.get("intermediate_tensors") is not None
            ):
                raise ValueError("V4.1 staged HCCL run requires an eager PP1 parent")
            if not isinstance(self.model_state, EngramModelState):
                raise ValueError(
                    "V4.1 staged HCCL requires the native Engram ModelState"
                )
            if self.model_state._cg_mode != CUDAGraphMode.NONE:
                raise ValueError("V4.1 staged HCCL requires prepared eager Engram mode")
            if transport.role != "attention" or transport.plan is None:
                raise ValueError("V4.1 staged HCCL requires an accepted Attention plan")
            parent_inputs = dict(model_inputs)
            # Native prepare_inputs already built the full parent's lookback
            # and pending producer. Run it here once, in the parent context;
            # bypassing nn.Module.__call__ avoids its per-model pre-hook.
            parent_inputs.update(self.model_state.prepare_engram())
            if parent_inputs.get("engram_graph_events", False):
                raise ValueError(
                    "V4.1 staged HCCL does not consume Graph Engram events"
                )
            if not model.language_model.model.has_engram:
                parent_inputs.setdefault("engram_lookups", {})
                parent_inputs.setdefault(
                    "engram_mask",
                    parent_inputs["positions"].new_empty(0, dtype=torch.bool),
                )
            engram = DeepseekV41EngramInputs(
                parent_inputs["engram_lookups"],
                parent_inputs["engram_mask"],
                parent_inputs.get("engram_pending"),
                parent_inputs.get("engram_mask_ready_event"),
            )
            slices = tuple(stage.token_slice for stage in ubatch_state.slices)
            if not slices or slices[-1].stop != parent_inputs["positions"].shape[0]:
                raise ValueError(
                    "V4.1 child slices must cover the parent padded tokens"
                )
            stages = []
            for token_slice in slices:
                child = slice_model_inputs(parent_inputs, token_slice)
                if child["input_ids"] is None:
                    raise ValueError("V4.1 staged HCCL requires native raw token IDs")
                stages.append(
                    DeepseekV41StageInputs(
                        child["input_ids"],
                        child["positions"],
                        engram.slice(token_slice),
                        child.get("inputs_embeds"),
                    )
                )
            output = run_deepseek_v41_stages(
                model.language_model.model,
                tuple(stages),
                tuple(ubatch_state.forward_contexts),
                slices,
                transport,
            )
            transport.finish_forward()
            return output
        except BaseException:
            # A peer can already be waiting on payload/COMPLETE. Mark this
            # endpoint terminal without blocking on an unsolicited error send;
            # backend timeout/process supervision owns failed peer unwinding.
            transport.terminal = True
            raise
