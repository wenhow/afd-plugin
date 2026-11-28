# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit V4.1 eager stage execution, outside production admission.

The caller owns parent input/Engram preparation, transaction acceptance and
retirement. This entry never calls a model pre-hook or prepares child history.
It requires eager, non-SP, PP1 execution and a complete accepted target plan.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm_ascend.models.deepseek_v41.model import AscendParallelEngramEmbedding

from afd_plugin.compat.npu.stream import use_npu_current_stream
from afd_plugin.connectors.npu.hccl_diagnostics import (
    HcclDiagnosticExecution,
    HcclStageDiagnosticRun,
)
from afd_plugin.connectors.npu.hccl_eager_pipeline import (
    HcclAttentionTicket,
    HcclEagerPipeline,
)
from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    AFDDeepseekV41AttentionDecoderLayer,
    AFDDeepseekV41AttentionModel,
    DeepseekV41FFNContinuation,
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
    ticket: HcclAttentionTicket | None = None
    continuation: DeepseekV41FFNContinuation | None = None


# Upstream: DeepseekV41Model.forward, model.py:1561-1651 @ 7e37e29a.
# Patch reason: native full-stage forwards overwrite the shared index prefix
# and cannot express layer-major remote MoE execution.
# Patch functionality: retain native HC/Engram/aux timing with stage views and
# explicit decoder continuations. New API receives native child contexts and
# already-produced Engram views; no native forward signature is changed.
# Optional pipeline/compute_stream explicitly select delayed eager loans;
# their transport owns all wire storage and completion. Defaults stay serial.
# Optional progress describes this accepted parent and its actual executor;
# it adds host enqueue observations only and never enables a compute fence.
# The SP/PP/Graph paths are excluded, not replaced. Parent request history,
# readiness production, native TP and final ModelState cleanup stay upstream.
# Removal: use upstream stage-aware model/decoder hooks when available.
def run_deepseek_v41_stages(
    model: AFDDeepseekV41AttentionModel,
    stages: tuple[DeepseekV41StageInputs, ...],
    contexts: tuple[ForwardContext, ...],
    token_slices: tuple[slice, ...],
    transport: HcclStagedTransport,
    *,
    pipeline: HcclEagerPipeline | None = None,
    compute_stream: torch.npu.Stream | None = None,
    progress: HcclStageDiagnosticRun | None = None,
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
    if (pipeline is None) != (compute_stream is None):
        raise ValueError("eager pipeline and compute stream must be supplied together")
    if pipeline is not None and (
        pipeline.transport is not transport
        or pipeline.plan is not plan
        or pipeline.finished
        or pipeline.failed
        or compute_stream.device != pipeline.parent_stream.device
        or torch.npu.current_stream(pipeline.parent_stream.device)
        != pipeline.parent_stream
    ):
        raise ValueError("eager stages require their active exact parent and streams")
    if progress is not None:
        progress.validate_parent(
            role=transport.role,
            world_rank=transport.world_rank,
            role_rank=transport.role_rank,
            num_stages=len(plan.stages),
            layer_indices=plan.layer_indices,
            transaction_id=plan.transaction_id,
        )
        if pipeline is None:
            progress.validate_execution(
                HcclDiagnosticExecution(
                    "serial", "parent", stream_aliases=(True, True, True)
                )
            )
        else:
            pipeline.bind_progress(progress, compute_stream=compute_stream)
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

    if pipeline is not None:
        # The native parent prepared input/metadata once. Keep its buffers
        # alive through side compute; Engram readiness is still waited at the
        # native first consumer rather than synchronizing its producer here.
        ready = torch.npu.Event()
        ready.record(pipeline.parent_stream)
        ready.wait(compute_stream)
        parent_state = model.shared_attention_state
        tensors = (
            parent_state.topk_indices,
            parent_state.candidates,
            parent_state.candidate_lengths,
            parent_state.topk_lengths,
            model._mtp_hidden_buffer,
        )
        for tensor in tensors:
            if tensor is not None:
                tensor.record_stream(compute_stream)
        for stage in stages:
            for tensor in (
                stage.input_ids,
                stage.positions,
                stage.engram.mask,
                *stage.engram.lookups.values(),
            ):
                tensor.record_stream(compute_stream)
            if stage.inputs_embeds is not None:
                stage.inputs_embeds.record_stream(compute_stream)

    executions = []
    for stage_idx, (stage, context) in enumerate(zip(stages, contexts, strict=True)):
        with (
            override_forward_context(context),
            use_npu_current_stream(compute_stream)
            if compute_stream is not None
            else nullcontext(),
        ):
            if progress is not None:
                _record_progress(
                    progress, pipeline, "prepare_begin", layers[0].layer_idx, stage_idx
                )
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
            if progress is not None:
                _record_progress(
                    progress,
                    pipeline,
                    "prepare_returned",
                    layers[0].layer_idx,
                    stage_idx,
                )
    # ### PATCH END: admit an explicit accepted eager parent

    # ### PATCH START: layer-major replaces native full-stage model calls
    if pipeline is not None:
        _run_pipelined_stages(
            model,
            layers,
            stages,
            contexts,
            executions,
            bindings,
            pipeline,
            compute_stream,
            progress,
        )
    else:
        for layer in layers:
            for stage_idx, (stage, context, execution) in enumerate(
                zip(stages, contexts, executions, strict=True)
            ):
                with override_forward_context(context), bindings.bind(stage_idx):
                    if progress is not None:
                        _record_progress(
                            progress,
                            pipeline,
                            "compute_begin",
                            layer.layer_idx,
                            stage_idx,
                        )
                    x, x_fp32, continuation = _prepare_stage(
                        model, layer, stage, execution
                    )
                    if progress is not None:
                        _record_progress(
                            progress,
                            pipeline,
                            "compute_returned",
                            layer.layer_idx,
                            stage_idx,
                        )
                    if any(plan.stages[stage_idx].attention_padded_tokens):
                        if progress is not None:
                            _record_progress(
                                progress,
                                pipeline,
                                "send_begin",
                                layer.layer_idx,
                                stage_idx,
                            )
                        transport.send_layer_inputs(
                            layer.layer_idx,
                            stage_idx,
                            x.contiguous(),
                            x_fp32.contiguous(),
                            execution.transfer_ids,
                        )
                        if progress is not None:
                            _record_progress(
                                progress,
                                pipeline,
                                "send_returned",
                                layer.layer_idx,
                                stage_idx,
                            )
                            _record_progress(
                                progress,
                                pipeline,
                                "recv_begin",
                                layer.layer_idx,
                                stage_idx,
                            )
                        ffn_output = transport.receive_layer_output(
                            layer.layer_idx, stage_idx
                        )
                        if progress is not None:
                            _record_progress(
                                progress,
                                pipeline,
                                "recv_returned",
                                layer.layer_idx,
                                stage_idx,
                            )
                        if continuation is not None:
                            if progress is not None:
                                _record_progress(
                                    progress,
                                    pipeline,
                                    "continuation_begin",
                                    layer.layer_idx,
                                    stage_idx,
                                )
                            execution.hidden_states, execution.pre_mix = (
                                layer.complete_remote_ffn(ffn_output, continuation)
                            )
                            if progress is not None:
                                _record_progress(
                                    progress,
                                    pipeline,
                                    "continuation_returned",
                                    layer.layer_idx,
                                    stage_idx,
                                )
    # ### PATCH END: layer-major replaces native full-stage model calls

    # ### PATCH START: reconstruct parent coordinates before final native collapse
    if progress is not None:
        _record_progress(
            progress, pipeline, "collapse_begin", layers[-1].layer_idx, len(stages) - 1
        )
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
    if progress is not None:
        _record_progress(
            progress,
            pipeline,
            "collapse_returned",
            layers[-1].layer_idx,
            len(stages) - 1,
        )
    # ### PATCH END: reconstruct parent coordinates before final native collapse
    if aux_hidden_states:
        return hidden_states, aux_hidden_states
    return hidden_states


# Upstream: DeepseekV41Model.forward @ Ascend 7e37e29a:1612-1645.
# Reason/functionality: share the exact aux-before-Engram and decoder-entry
# contract between serial and delayed eager schedules. This new helper returns
# the native HC continuation instead of invoking remote MoE inside the layer.
# Parent producer/history/retirement stay native; remove with stage hooks.
def _prepare_stage(
    model: AFDDeepseekV41AttentionModel,
    layer: AFDDeepseekV41AttentionDecoderLayer,
    stage: DeepseekV41StageInputs,
    execution: _StageExecution,
) -> tuple[torch.Tensor, torch.Tensor, DeepseekV41FFNContinuation | None]:
    """Shared native layer entry for serial and delayed eager schedules."""
    # ### PATCH START: retain native layer entry across a remote MoE seam
    hidden_states = execution.hidden_states
    if layer.layer_idx + 1 in model.aux_hidden_state_layers:
        execution.auxiliary.append(hidden_states.mean(dim=1))
    if layer.engram is not None and stage.engram.mask.numel():
        if execution.mask_ready_event is not None:
            AscendParallelEngramEmbedding.wait_lookup(
                execution.mask_ready_event, external=False
            )
            execution.mask_ready_event = None
        if stage.engram.pending is not None and layer.layer_idx in stage.engram.pending:
            AscendParallelEngramEmbedding.wait_lookup(
                stage.engram.pending[layer.layer_idx], external=False
            )
        hidden_states[:] = layer.engram(
            hidden_states,
            stage.engram.lookups[layer.layer_idx],
            stage.engram.mask,
            model.engram_rotation if model.engram_rotated else None,
        )
    if hidden_states.shape[0]:
        return layer.prepare_remote_ffn(
            stage.positions, hidden_states, execution.pre_mix
        )
    # Preserve C4's zero-row global stage and avoid zero-row HC kernels.
    x = hidden_states.new_empty((0, model.config.hidden_size))
    return x, x.float(), None
    # ### PATCH END: retain native layer entry across a remote MoE seam


def _continuation_tensors(execution: _StageExecution) -> tuple[torch.Tensor, ...]:
    continuation = execution.continuation
    return (
        execution.hidden_states,
        execution.pre_mix,
        *execution.auxiliary,
        *(
            (
                continuation.residual,
                continuation.post,
                continuation.comb,
                continuation.pre_mix,
            )
            if continuation is not None
            else ()
        ),
    )


# Upstream model forward @ Ascend 7e37e29a:1561-1651. Reason: eagerly finishing
# each remote MoE before posting the next stage precludes U2 in-flight work.
# Functionality: delay each HC continuation to its next actual stage consumer,
# retaining native auxiliary/Engram/shared-Attention semantics on one compute
# stream. This is a new internal schedule, not a changed upstream signature.
# Transport owns Work/event/storage and COMPLETE; native ModelState owns history
# and retirement. Remove with upstream stage/remote-expert scheduling hooks.
def _run_pipelined_stages(
    model: AFDDeepseekV41AttentionModel,
    layers: tuple[AFDDeepseekV41AttentionDecoderLayer, ...],
    stages: tuple[DeepseekV41StageInputs, ...],
    contexts: tuple[ForwardContext, ...],
    executions: list[_StageExecution],
    bindings: DeepseekV41StageAttentionStates,
    pipeline: HcclEagerPipeline,
    compute_stream: torch.npu.Stream,
    progress: HcclStageDiagnosticRun | None = None,
) -> None:
    # ### PATCH START: delayed HC consumers with bounded eager wire tickets
    try:
        for layer_index, layer in enumerate(layers):
            for stage_idx, (stage, context, execution) in enumerate(
                zip(stages, contexts, executions, strict=True)
            ):
                if execution.ticket is not None:
                    with (
                        override_forward_context(context),
                        bindings.bind(stage_idx),
                        pipeline.attention_output(
                            execution.ticket,
                            consumer_stream=compute_stream,
                            retain=_continuation_tensors(execution),
                        ) as output,
                    ):
                        if execution.continuation is not None:
                            if progress is not None:
                                _record_progress(
                                    progress,
                                    pipeline,
                                    "continuation_begin",
                                    execution.ticket.layer_idx,
                                    stage_idx,
                                )
                            execution.hidden_states, execution.pre_mix = layers[
                                layer_index - 1
                            ].complete_remote_ffn(output, execution.continuation)
                            if progress is not None:
                                _record_progress(
                                    progress,
                                    pipeline,
                                    "continuation_returned",
                                    execution.ticket.layer_idx,
                                    stage_idx,
                                )
                    execution.ticket = execution.continuation = None
                with (
                    override_forward_context(context),
                    bindings.bind(stage_idx),
                    use_npu_current_stream(compute_stream),
                ):
                    if progress is not None:
                        _record_progress(
                            progress,
                            pipeline,
                            "compute_begin",
                            layer.layer_idx,
                            stage_idx,
                        )
                    x, fp32, execution.continuation = _prepare_stage(
                        model, layer, stage, execution
                    )
                    x, fp32 = x.contiguous(), fp32.contiguous()
                    if progress is not None:
                        _record_progress(
                            progress,
                            pipeline,
                            "compute_returned",
                            layer.layer_idx,
                            stage_idx,
                        )
                if any(pipeline.plan.stages[stage_idx].attention_padded_tokens):
                    execution.ticket = pipeline.post_attention(
                        layer.layer_idx,
                        stage_idx,
                        x,
                        fp32,
                        execution.transfer_ids,
                        producer_stream=compute_stream,
                        retain=_continuation_tensors(execution),
                    )
        for stage_idx, (context, execution) in enumerate(
            zip(contexts, executions, strict=True)
        ):
            if execution.ticket is None:
                continue
            with (
                override_forward_context(context),
                bindings.bind(stage_idx),
                pipeline.attention_output(
                    execution.ticket,
                    consumer_stream=compute_stream,
                    retain=_continuation_tensors(execution),
                ) as output,
            ):
                if execution.continuation is not None:
                    if progress is not None:
                        _record_progress(
                            progress,
                            pipeline,
                            "continuation_begin",
                            execution.ticket.layer_idx,
                            stage_idx,
                        )
                    execution.hidden_states, execution.pre_mix = layers[
                        -1
                    ].complete_remote_ffn(output, execution.continuation)
                    if progress is not None:
                        _record_progress(
                            progress,
                            pipeline,
                            "continuation_returned",
                            execution.ticket.layer_idx,
                            stage_idx,
                        )
            execution.ticket = execution.continuation = None
        # Collapse/MTP/native retirement must see all stage computations. A
        # device dependency plus allocator ownership replaces a host sync here.
        done = torch.npu.Event()
        done.record(compute_stream)
        done.wait(pipeline.parent_stream)
        for execution in executions:
            for tensor in _continuation_tensors(execution):
                tensor.record_stream(pipeline.parent_stream)
    except BaseException:
        pipeline.fail()
        raise
    # ### PATCH END: delayed HC consumers with bounded eager wire tickets


def _record_progress(
    progress: HcclStageDiagnosticRun,
    pipeline: HcclEagerPipeline | None,
    event: str,
    layer_idx: int,
    stage_idx: int,
) -> None:
    """A failed sink retains its live ticket owner, including before stage entry."""
    try:
        progress.record(event, layer_idx, stage_idx)
    except BaseException:
        if (
            pipeline is not None
            and pipeline.transport.eager_pipeline is pipeline
            and pipeline.transport.plan is pipeline.plan
            and not pipeline.finished
            and not pipeline.failed
        ):
            pipeline.fail()
        raise
