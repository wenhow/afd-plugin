# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded diagnostics for the first single-request DSpark step on each DP rank."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch
    from vllm.forward_context import ForwardContext
    from vllm.v1.worker.gpu.model_runner import ExecuteModelState

    from afd_plugin.model_executor.models.npu.deepseek_v41 import (
        AFDDeepseekV41AttentionForCausalLM,
    )
    from afd_plugin.v1.worker.npu.hccl_v41_execution import HcclV41Execution

DSPARK_FIRST_WAVE_TRACE_ENV = "AFD_DSV41_DSPARK_FIRST_WAVE_TRACE"


class DSV41DSparkFirstWaveTrace:
    """Capture actual SWA coordinates and target logits on two eager steps."""

    def __init__(self, dp_rank: int) -> None:
        self.dp_rank = dp_rank
        self.request_id: str | None = None
        self.phases: set[str] = set()
        self.pending: tuple[ExecuteModelState, dict] | None = None

    def record_execute(
        self, state: ExecuteModelState | None, execution: HcclV41Execution
    ) -> None:
        self.pending = None
        if state is None or execution.parent_context is None or len(self.phases) == 2:
            return
        batch = state.input_batch
        if batch.num_reqs != 1:
            return
        request_id = batch.req_ids[0]
        if self.request_id is None:
            if batch.num_computed_tokens_np[0] != 0 or not batch.is_prefilling_np[0]:
                return
            self.request_id = request_id
            phase = "prefill"
        elif request_id == self.request_id and batch.num_draft_tokens > 0:
            phase = "verify"
        else:
            return
        if phase in self.phases:
            return

        prepared = execution.prepared
        if prepared is None:
            stages = [
                (
                    slice(0, batch.num_reqs_after_padding),
                    slice(0, batch.num_tokens_after_padding),
                    execution.parent_context,
                )
            ]
        else:
            stages = [
                (stage.request_slice, stage.token_slice, context)
                for stage, context in zip(
                    prepared.slices, prepared.forward_contexts, strict=True
                )
            ]
        self.pending = (
            state,
            {
                "dp_rank": self.dp_rank,
                "request_id": request_id,
                "phase": phase,
                "prompt_tokens": int(batch.prefill_len_np[0]),
                "computed_tokens": int(batch.num_computed_tokens_np[0]),
                "scheduled_tokens": batch.num_scheduled_tokens.tolist(),
                "num_draft_tokens": batch.num_draft_tokens,
                "input_ids": batch.input_ids[: batch.num_tokens].tolist(),
                "logits_indices": batch.logits_indices.tolist(),
                "stages": [
                    {
                        "request_range": [requests.start, requests.stop],
                        "token_range": [tokens.start, tokens.stop],
                        "positions": batch.positions[tokens].tolist(),
                        "swa": self._swa_coordinates(context),
                    }
                    for requests, tokens, context in stages
                ],
            },
        )

    @staticmethod
    def _swa_coordinates(context: ForwardContext) -> list[dict]:
        # Load only after model initialization, like the child builder setup.
        from vllm_ascend.attention.dsa_v41 import AscendDSAV41Metadata

        records = []
        seen: set[int] = set()
        if context.attn_metadata is None:
            return records
        assert isinstance(context.attn_metadata, dict)
        for layer, metadata in context.attn_metadata.items():
            if (
                not isinstance(metadata, AscendDSAV41Metadata)
                or metadata.cache_kind != "swa"
                or id(metadata) in seen
            ):
                continue
            seen.add(id(metadata))
            records.append(
                {
                    "layer": layer,
                    "num_actual_tokens": metadata.num_actual_tokens,
                    "query_start_loc": metadata.query_start_loc.tolist(),
                    "seq_lens": metadata.seq_lens.tolist(),
                    "slot_mapping": metadata.slot_mapping.tolist(),
                    "flat_slot_mapping": metadata.flat_slot_mapping.tolist()
                    if metadata.flat_slot_mapping is not None
                    else None,
                    "num_prefills": metadata.num_prefills,
                    "num_decodes": metadata.num_decodes,
                }
            )
        return records

    @contextmanager
    def observe_sampling(
        self, state: ExecuteModelState | None, model: AFDDeepseekV41AttentionForCausalLM
    ) -> Iterator[None]:
        pending, self.pending = self.pending, None
        if pending is None or state is not pending[0]:
            yield
            return
        _, record = pending
        top_ids: torch.Tensor | None = None

        def observe(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            logits: torch.Tensor | None,
        ) -> None:
            nonlocal top_ids
            # Native sampling calls the target head before draft proposal.
            if top_ids is None and logits is not None:
                top_ids = logits.argmax(dim=-1)

        with model.language_model.logits_processor.register_forward_hook(observe):
            yield
        record["target_top1_token_ids"] = None if top_ids is None else top_ids.tolist()
        self.phases.add(record["phase"])
        init_logger("vllm.afd_dsv41_dspark_trace").info(
            "AFD DSpark first-wave trace %s", json.dumps(record, separators=(",", ":"))
        )
