# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Synchronous V4.1 tensor body for an explicitly owned static graph.

The owner constructs this object, child metadata and exchange storage before
capture. Python stage bindings select the addresses recorded by the graph;
replay reads those same addresses without re-entering Python. This object does
not negotiate a parent, prepare request history, launch graphs or retire a
ModelState. Production model admission remains separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm_ascend.models.deepseek_v41.model import AscendParallelEngramEmbedding

from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    AFDDeepseekV41AttentionModel,
)
from afd_plugin.model_executor.models.npu.deepseek_v41_stage_state import (
    DeepseekV41StageAttentionStates,
)
from afd_plugin.model_executor.models.npu.deepseek_v41_staged import (
    DeepseekV41EngramInputs,
    DeepseekV41StageInputs,
)


@dataclass(slots=True)
class _GraphStageExecution:
    hidden_states: torch.Tensor
    pre_mix: torch.Tensor
    mask_ready_event: torch.npu.Event | None
    auxiliary: list[torch.Tensor]


class DeepseekV41GraphStages:
    """One retained static layout, using native layer continuations.

    Each stage has independent shared-Attention views and native metadata. Its
    exchange storage is reused across layers only on the owning compute stream,
    after the preceding continuation has consumed the returned output.
    """

    def __init__(
        self,
        model: AFDDeepseekV41AttentionModel,
        contexts: tuple[ForwardContext, ...],
        token_slices: tuple[slice, ...],
        bank: HcclGraphStageBank,
    ) -> None:
        layers = tuple(model.layers)
        if model.use_sequence_parallel or model.start_layer != 0:
            raise ValueError("V4.1 graph stages require non-SP PP1 Attention")
        if (
            bank.role != "attention"
            or model.end_layer != len(layers)
            or tuple(layer.layer_idx for layer in layers) != bank.layer_indices
            or not len(contexts) == len(token_slices) == len(bank.stage_rows)
        ):
            raise ValueError("V4.1 graph stages require one complete static layout")
        # This validates contiguous bounds and creates stable parent-buffer
        # views before any graph body or producer is entered.
        bindings = DeepseekV41StageAttentionStates(model, token_slices)
        for index, (context, token_slice, rows) in enumerate(
            zip(contexts, token_slices, bank.stage_rows, strict=True)
        ):
            if (
                context.cudagraph_runtime_mode != CUDAGraphMode.FULL
                or context.ubatch_idx != index
                or token_slice.stop - token_slice.start != rows
            ):
                raise ValueError("V4.1 graph child coordinates must be static FULL")
        self.model, self.contexts = model, contexts
        self.token_slices, self.bank = token_slices, bank
        self.bindings, self.layers = bindings, layers

    # Upstream: DeepseekV41Model.forward @ Ascend 7e37e29a:1561-1651.
    # Reason: native full-model child forwards cannot record layer-major A/F
    # exchange or disjoint child shared-index addresses.
    # Functionality: retain HC/pre_mix, ordinary Engram waits, aux timing and
    # parent collapse while recording only tensor exchange for each static
    # stage. This is a new API, not a replacement upstream signature. Control,
    # graph cache, producer and native retirement stay with the parent owner.
    # Removal: use upstream stage-aware model/remote-expert hooks when available.
    def run(
        self, stages: tuple[DeepseekV41StageInputs, ...]
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # ### PATCH START: fixed stage inputs, without eager plan cursors
        model, bank = self.model, self.bank
        if len(stages) != len(bank.stage_rows):
            raise ValueError("V4.1 graph inputs must match retained stages")
        for stage, rows in zip(stages, bank.stage_rows, strict=True):
            if (
                stage.positions.shape != (rows,)
                or stage.input_ids.shape != (rows,)
                or stage.input_ids.dtype not in (torch.int32, torch.int64)
                or (
                    stage.inputs_embeds is not None
                    and stage.inputs_embeds.shape[0] != rows
                )
            ):
                raise ValueError("V4.1 graph inputs must match static token rows")
            if model.has_engram and (
                stage.engram.mask.shape != (rows,)
                or any(
                    lookup.shape[0] != rows for lookup in stage.engram.lookups.values()
                )
                or any(
                    layer.engram is not None
                    and layer.layer_idx not in stage.engram.lookups
                    for layer in self.layers
                )
            ):
                raise ValueError("V4.1 graph Engram requires parent token views")

        executions = []
        for stage, context in zip(stages, self.contexts, strict=True):
            with override_forward_context(context):
                if stage.inputs_embeds is not None:
                    hidden = stage.inputs_embeds
                elif stage.input_ids.numel():
                    hidden = model.embed_input_ids(stage.input_ids)
                else:
                    hidden = stage.positions.new_empty(
                        (0, model.config.hidden_size), dtype=torch.bfloat16
                    )
                if hidden.dtype != torch.bfloat16:
                    raise ValueError("V4.1 graph requires BF16 activations")
                hidden = hidden.unsqueeze(1).repeat(1, model.hc_mult, 1)
                pre_mix = hidden.new_zeros(
                    hidden.shape[0], model.hc_mult, dtype=torch.float32
                )
                pre_mix[:, 0] = 1.0
                executions.append(
                    _GraphStageExecution(
                        hidden, pre_mix, stage.engram.mask_ready_event, []
                    )
                )
        # ### PATCH END: fixed stage inputs, without eager plan cursors

        # ### PATCH START: graph records layer-major tensor exchange
        for layer in self.layers:
            for index, (stage, context, execution) in enumerate(
                zip(stages, self.contexts, executions, strict=True)
            ):
                with override_forward_context(context), self.bindings.bind(index):
                    hidden = execution.hidden_states
                    if layer.layer_idx + 1 in model.aux_hidden_state_layers:
                        execution.auxiliary.append(hidden.mean(dim=1))
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
                        hidden[:] = layer.engram(
                            hidden,
                            stage.engram.lookups[layer.layer_idx],
                            stage.engram.mask,
                            model.engram_rotation if model.engram_rotated else None,
                        )
                    if hidden.shape[0]:
                        x, x_fp32, continuation = layer.prepare_remote_ffn(
                            stage.positions, hidden, execution.pre_mix
                        )
                        output = bank.attention_exchange(
                            index, x.contiguous(), x_fp32.contiguous(), stage.input_ids
                        )
                        execution.hidden_states, execution.pre_mix = (
                            layer.complete_remote_ffn(output, continuation)
                        )
                    # A static zero-row stage records no payload on either
                    # peer. Runtime empty/padded stages still use their fixed
                    # positive extent and device-valid masks in the bank.
        # ### PATCH END: graph records layer-major tensor exchange

        # ### PATCH START: native parent aux, MTP and delayed HC collapse
        hidden = torch.cat([item.hidden_states for item in executions], dim=0)
        pre_mix = torch.cat([item.pre_mix for item in executions], dim=0)
        if model._mtp_hidden_buffer is not None:
            model._mtp_hidden_buffer[: hidden.shape[0]].copy_(hidden.flatten(1))
        if hidden.shape[0]:
            hidden = self.layers[-1].hc_collapse(hidden, pre_mix)
            hidden = model.norm(hidden)
        else:
            hidden = hidden.new_empty((0, model.config.hidden_size))
        auxiliary = [
            torch.cat([item.auxiliary[index] for item in executions], dim=0)
            for index in range(len(executions[0].auxiliary))
        ]
        # ### PATCH END: native parent aux, MTP and delayed HC collapse
        return (hidden, auxiliary) if auxiliary else hidden


class DeepseekV41GraphParent:
    """Use one native captured Engram producer for the unsliced parent.

    The native ModelWithContext wrapper calls the model once; an explicitly
    installed parent sidecar can route that call here. The same ordinary event
    can then be waited by both children without ExternalEvent resets.
    """

    def __init__(self, stages: DeepseekV41GraphStages) -> None:
        self.stages = stages

    def __call__(
        self, model_inputs: dict[str, Any]
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        model = self.stages.model
        if model_inputs.get("intermediate_tensors") is not None or model_inputs.get(
            "engram_graph_events", False
        ):
            raise ValueError(
                "V4.1 graph parent requires PP1 and captured producer events"
            )
        ids, positions = model_inputs["input_ids"], model_inputs["positions"]
        parent_rows = self.stages.token_slices[-1].stop
        if (
            ids is None
            or ids.shape != (parent_rows,)
            or positions.shape != (parent_rows,)
        ):
            raise ValueError("V4.1 graph parent must cover retained padded tokens")
        if model.has_engram:
            if not model.can_capture_engram_producer():
                raise ValueError(
                    "unsupported Engram producer must select eager before preparation"
                )
            # These are the native ModelState's persistent FULL inputs. No
            # child history or eager pending producer is constructed here.
            query = model_inputs["engram_query_start_loc"]
            valid = model_inputs["engram_valid_token_count"]
            lookback = model_inputs["lookback_token_ids"]
            with model.captured_engram_inputs(
                ids, positions, lookback, query, valid
            ) as produced:
                engram = DeepseekV41EngramInputs(
                    produced["engram_lookups"],
                    produced["engram_mask"],
                    produced.get("engram_pending"),
                    produced.get("engram_mask_ready_event"),
                )
                return self._run(model_inputs, engram)
        engram = DeepseekV41EngramInputs({}, positions.new_empty(0, dtype=torch.bool))
        return self._run(model_inputs, engram)

    def _run(
        self, model_inputs: dict[str, Any], engram: DeepseekV41EngramInputs
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        embeds = model_inputs.get("inputs_embeds")
        stages = tuple(
            DeepseekV41StageInputs(
                model_inputs["input_ids"][token_slice],
                model_inputs["positions"][token_slice],
                engram.slice(token_slice),
                None if embeds is None else embeds[token_slice],
            )
            for token_slice in self.stages.token_slices
        )
        return self.stages.run(stages)
