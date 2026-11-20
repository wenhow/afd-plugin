# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native TP4 control with the same A5 query-head adapter as AFD."""

import hashlib
import json
import math

import torch
from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm_ascend.models.deepseek_v41.mixed_quant_attention import (
    MixedQuantPackedCacheOps,
)
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Attention
from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.model_executor.models.npu.dsv41_packed_cache import (
    AFDMixedQuantPackedCacheOps,
)

PREFILL_TRACE_STEPS = 2
PREFILL_TRACE_TOP_K = 5
# Include the winning tokens and runners-up from the observed early divergence.
PREFILL_TRACE_CANDIDATES = (1, 14, 223, 270, 455, 1528, 1999)


def _tensor_identity(cpu_tensor: torch.Tensor) -> dict:
    return {
        "shape": list(cpu_tensor.shape),
        "dtype": str(cpu_tensor.dtype),
        "sha256": hashlib.sha256(
            cpu_tensor.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }


def _logit_value(value: float) -> float | str:
    return value if math.isfinite(value) else str(value)


def _summarize_logits(logits: torch.Tensor) -> dict:
    scores = logits.float()
    top_scores, top_ids = scores.topk(PREFILL_TRACE_TOP_K)
    return {
        **_tensor_identity(logits),
        "top5": [
            [token, _logit_value(score)]
            for token, score in zip(top_ids.tolist(), top_scores.tolist(), strict=True)
        ],
        "candidates": {
            token: _logit_value(scores[token].item())
            for token in PREFILL_TRACE_CANDIDATES
        },
    }


class DSV41NativeTP4ControlWorker(NPUWorker):
    """Retain native execution and adapt only each packed Attention backend."""

    # Upstream: NPUWorker.load_model, worker/worker.py @ Ascend 7e37e29a.
    # Reason: the TP4 native control supplies 16 heads to a 64-head A5 wheel.
    # Functionality: select the existing head adapter after native weight load,
    # before profiling or inference, without changing native classes globally.
    # Signature: matches upstream exactly; no added parameters.
    # Delegation exception: native load owns sleep pools and weight transfer;
    # duplicating those owners for per-instance backend selection is unsuitable.
    # Cost: one module walk at load; qsmla has the documented 4x head work.
    # Removal: retire this control worker when the A5 wheel accepts 16 heads.
    def load_model(self) -> None:
        # ### PATCH START: native TP4 control scope
        if self.vllm_config.parallel_config.tensor_parallel_size != 4:
            raise ValueError("DSV41 native query-head control requires TP4")
        # ### PATCH END: native TP4 control scope
        super().load_model()
        # ### PATCH START: per-Attention control head adapter
        adapted = 0
        for module in self.model_runner.model.modules():
            if (
                isinstance(module, DeepseekV41Attention)
                and module.packed_cache_ops is MixedQuantPackedCacheOps
            ):
                module.packed_cache_ops = AFDMixedQuantPackedCacheOps
                adapted += 1
        if not adapted:
            raise ValueError("DSV41 native TP4 control found no A5 packed Attention")
        init_logger("vllm.afd_dsv41_native_control").info(
            "Native TP4 control: installed 16-to-64 query-head adapter on %d "
            "Attention layers; native model, MoE and Engram retained.",
            adapted,
        )
        # ### PATCH END: per-Attention control head adapter


class DSV41NativeTP4PrefillTraceWorker(DSV41NativeTP4ControlWorker):
    """Record the first prefill and decode logits for serial native requests."""

    # Upstream: NPUWorker.sample_tokens, worker/worker.py @ Ascend 7e37e29a.
    # Reason: response-level scores cannot locate changes before the sampler.
    # Functionality: observe the native logits processor for the first two steps.
    # Signature: matches upstream; no added parameters.
    # Delegate to native sampling to preserve its lifecycle and profiling output.
    # Cost: two tensor snapshots per traced step and CPU copies after sampling.
    # Remove this diagnostic worker when the native score drift is located.
    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        # ### PATCH START: scope trace to an actual serial sampling batch
        state = self.model_runner.execute_model_state
        if state is None:
            return super().sample_tokens(grammar_output)
        batch = state.input_batch
        if batch.num_reqs != 1 or batch.num_draft_tokens or grammar_output is not None:
            raise ValueError("native prefill trace requires one plain target request")
        prompt_tokens = int(batch.prefill_len_np[0])
        computed_tokens = int(batch.num_computed_tokens_np[0])
        step = computed_tokens + batch.num_tokens - prompt_tokens + 1
        if not 1 <= step <= PREFILL_TRACE_STEPS:
            return super().sample_tokens(grammar_output)

        snapshots: list[tuple[torch.Tensor, torch.Tensor | None]] = []

        def observe(
            module: torch.nn.Module,
            args: tuple[torch.nn.Module, torch.Tensor],
            logits: torch.Tensor | None,
        ) -> None:
            _, hidden_states = args
            snapshots.append(
                (
                    hidden_states[0].detach().clone(),
                    logits[0].detach().clone() if logits is not None else None,
                )
            )

        # The registered V4.1 wrapper delegates compute_logits to its language model.
        logits_processor = self.model_runner.model.language_model.logits_processor
        with logits_processor.register_forward_hook(observe):
            output = super().sample_tokens(grammar_output)
        if len(snapshots) != 1:
            raise RuntimeError("native prefill trace expected exactly one logits call")
        hidden_states, logits = snapshots[0]
        record = {
            "request_id": batch.req_ids[0],
            "rank": self.rank,
            "step": step,
            "prompt_tokens": prompt_tokens,
            "computed_tokens": computed_tokens,
            "input_ids": batch.input_ids[: batch.num_tokens].tolist(),
            "positions": batch.positions[: batch.num_tokens].tolist(),
            "logits_indices": batch.logits_indices.tolist(),
            "hidden_state": _tensor_identity(hidden_states.cpu()),
            "raw_logits": _summarize_logits(logits.cpu())
            if logits is not None
            else None,
        }
        init_logger("vllm.afd_dsv41_native_control").info(
            "AFD native prefill trace %s", json.dumps(record, allow_nan=False)
        )
        return output
        # ### PATCH END: scope trace to an actual serial sampling batch
