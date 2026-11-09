# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit synchronous V4.1 stage execution, outside production admission.

The caller owns parent input/Engram preparation, transaction acceptance and
retirement. This entry never calls a model pre-hook or prepares child history.
It requires eager, non-SP, PP1 execution and a complete accepted target plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm_ascend.models.deepseek_v41.model import AscendParallelEngramEmbedding

from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    AFDDeepseekV41AttentionModel,
)
from afd_plugin.model_executor.models.npu.deepseek_v41_stage_state import (
    DeepseekV41StageAttentionStates,
)


@dataclass(frozen=True, slots=True)
class DeepseekV41EngramInputs:
    """Views of one parent producer's storage and shared readiness events."""

    lookups: dict[int, torch.Tensor]
    mask: torch.Tensor
    pending: dict[int, torch.npu.Event] | None = None
    mask_ready_event: torch.npu.Event | None = None

    def slice(self, token_slice: slice) -> DeepseekV41EngramInputs:
        start, stop, step = token_slice.start, token_slice.stop, token_slice.step
        if (
            type(start) is not int
            or type(stop) is not int
            or start < 0
            or stop < start
            or (step is not None and (type(step) is not int or step != 1))
        ):
            raise ValueError("Engram stage requires an explicit bounded token slice")
        if not self.lookups and self.mask.numel() == 0:
            return self
        if stop > self.mask.shape[0] or any(
            stop > lookup.shape[0] for lookup in self.lookups.values()
        ):
            raise ValueError("Engram stage exceeds parent token storage")
        return DeepseekV41EngramInputs(
            {layer: lookup[token_slice] for layer, lookup in self.lookups.items()},
            self.mask[token_slice],
            self.pending,
            self.mask_ready_event,
        )


@dataclass(frozen=True, slots=True)
class DeepseekV41StageInputs:
    input_ids: torch.Tensor
    positions: torch.Tensor
    engram: DeepseekV41EngramInputs
    inputs_embeds: torch.Tensor | None = None


@dataclass(slots=True)
class _StageExecution:
    hidden_states: torch.Tensor
    pre_mix: torch.Tensor
    transfer_ids: torch.Tensor
    mask_ready_event: torch.npu.Event | None
    auxiliary: list[torch.Tensor]


# Upstream: DeepseekV41Model.forward, model.py:1561-1651 @ 7e37e29a.
# Patch reason: native full-stage forwards overwrite the shared index prefix
# and cannot express layer-major remote MoE execution.
# Patch functionality: retain native HC/Engram/aux timing with stage views and
# explicit decoder continuations. New API receives native child contexts and
# already-produced Engram views; no native forward signature is changed.
# The SP/PP/Graph paths are excluded, not replaced. Parent request history,
# readiness production, native TP and final ModelState cleanup stay upstream.
# Removal: use upstream stage-aware model/decoder hooks when available.
def run_deepseek_v41_stages(
    model: AFDDeepseekV41AttentionModel,
    stages: tuple[DeepseekV41StageInputs, ...],
    contexts: tuple[ForwardContext, ...],
    token_slices: tuple[slice, ...],
    transport: HcclStagedTransport,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    # ### PATCH START: admit an explicit accepted eager parent
    plan = transport.plan
    if transport.role != "attention" or plan is None:
        raise ValueError("V4.1 stages require an accepted Attention parent plan")
    if model.use_sequence_parallel:
        raise ValueError("V4.1 stages require non-SP execution")
    layers = tuple(model.layers)
    if (
        model.start_layer != 0
        or model.end_layer != len(layers)
        or tuple(layer.layer_idx for layer in layers) != plan.layer_indices
    ):
        raise ValueError("V4.1 stages require the complete PP1 target layer plan")
    if not len(stages) == len(contexts) == len(token_slices) == len(plan.stages):
        raise ValueError("V4.1 stages must match native child contexts and plan")
    bindings = DeepseekV41StageAttentionStates(model, token_slices)
    for stage_idx, (stage, context, token_slice) in enumerate(
        zip(stages, contexts, token_slices, strict=True)
    ):
        rows = plan.stages[stage_idx].attention_padded_tokens[transport.role_rank]
        if (
            context.cudagraph_runtime_mode != CUDAGraphMode.NONE
            or context.ubatch_idx != stage_idx
            or stage.positions.ndim != 1
            or stage.positions.shape[0] != rows
            or stage.input_ids.shape != (rows,)
            or token_slice.stop - token_slice.start != rows
            or (
                stage.inputs_embeds is not None and stage.inputs_embeds.shape[0] != rows
            )
        ):
            raise ValueError("V4.1 native child coordinates do not match the plan")
        if model.has_engram and (
            stage.engram.mask.shape != (rows,)
            or any(lookup.shape[0] != rows for lookup in stage.engram.lookups.values())
            or any(
                layer.engram is not None and layer.layer_idx not in stage.engram.lookups
                for layer in layers
            )
        ):
            raise ValueError("V4.1 Engram inputs must be parent token views")

    executions = []
    for stage_idx, (stage, context) in enumerate(zip(stages, contexts, strict=True)):
        with override_forward_context(context):
            if stage.inputs_embeds is not None:
                hidden_states = stage.inputs_embeds
            elif stage.input_ids.numel():
                hidden_states = model.embed_input_ids(stage.input_ids)
            else:
                hidden_states = stage.positions.new_empty(
                    (0, model.config.hidden_size), dtype=torch.bfloat16
                )
            if hidden_states.dtype != torch.bfloat16:
                raise ValueError("V4.1 staged HCCL requires BF16 activations")
            hidden_states = hidden_states.unsqueeze(1).repeat(1, model.hc_mult, 1)
            pre_mix = hidden_states.new_zeros(
                hidden_states.shape[0], model.hc_mult, dtype=torch.float32
            )
            pre_mix[:, 0] = 1.0
            ids = stage.input_ids.to(dtype=torch.int32).clone()
            ids[
                plan.stages[stage_idx].attention_valid_tokens[transport.role_rank] :
            ] = 0
            executions.append(
                _StageExecution(
                    hidden_states, pre_mix, ids, stage.engram.mask_ready_event, []
                )
            )
    # ### PATCH END: admit an explicit accepted eager parent

    # ### PATCH START: layer-major replaces native full-stage model calls
    for layer in layers:
        for stage_idx, (stage, context, execution) in enumerate(
            zip(stages, contexts, executions, strict=True)
        ):
            with override_forward_context(context), bindings.bind(stage_idx):
                hidden_states = execution.hidden_states
                if layer.layer_idx + 1 in model.aux_hidden_state_layers:
                    # Native aux observes layer entry, before Engram injection.
                    execution.auxiliary.append(hidden_states.mean(dim=1))
                if layer.engram is not None and stage.engram.mask.numel():
                    if execution.mask_ready_event is not None:
                        AscendParallelEngramEmbedding.wait_lookup(
                            execution.mask_ready_event, external=False
                        )
                        execution.mask_ready_event = None
                    if (
                        stage.engram.pending is not None
                        and layer.layer_idx in stage.engram.pending
                    ):
                        AscendParallelEngramEmbedding.wait_lookup(
                            stage.engram.pending[layer.layer_idx], external=False
                        )
                    hidden_states[:] = layer.engram(
                        hidden_states,
                        stage.engram.lookups[layer.layer_idx],
                        stage.engram.mask,
                        model.engram_rotation if model.engram_rotated else None,
                    )
                rows = hidden_states.shape[0]
                if rows:
                    x, x_fp32, continuation = layer.prepare_remote_ffn(
                        stage.positions, hidden_states, execution.pre_mix
                    )
                else:
                    # A locally empty stage can still participate in a global
                    # fanout/aggregate step; avoid calling HC kernels on zero rows.
                    x = hidden_states.new_empty((0, model.config.hidden_size))
                    x_fp32 = x.float()
                if any(plan.stages[stage_idx].attention_padded_tokens):
                    transport.send_layer_inputs(
                        layer.layer_idx,
                        stage_idx,
                        x.contiguous(),
                        x_fp32.contiguous(),
                        execution.transfer_ids,
                    )
                    ffn_output = transport.receive_layer_output(
                        layer.layer_idx, stage_idx
                    )
                    if rows:
                        execution.hidden_states, execution.pre_mix = (
                            layer.complete_remote_ffn(ffn_output, continuation)
                        )
    # ### PATCH END: layer-major replaces native full-stage model calls

    # ### PATCH START: reconstruct parent coordinates before final native collapse
    hidden_states = torch.cat(
        [execution.hidden_states for execution in executions], dim=0
    )
    pre_mix = torch.cat([execution.pre_mix for execution in executions], dim=0)
    if model._mtp_hidden_buffer is not None:
        model._mtp_hidden_buffer[: hidden_states.shape[0]].copy_(
            hidden_states.flatten(1)
        )
    if hidden_states.shape[0]:
        hidden_states = layers[-1].hc_collapse(hidden_states, pre_mix)
        hidden_states = model.norm(hidden_states)
    else:
        hidden_states = hidden_states.new_empty((0, model.config.hidden_size))
    aux_hidden_states = [
        torch.cat([execution.auxiliary[index] for execution in executions], dim=0)
        for index in range(len(executions[0].auxiliary))
    ]
    # ### PATCH END: reconstruct parent coordinates before final native collapse
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states
