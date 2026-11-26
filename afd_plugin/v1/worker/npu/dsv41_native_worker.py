# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native A5 control workers for V4.1 model loading and diagnostics."""

import hashlib
import json
import math
import os
from contextlib import ExitStack
from functools import partial
from unittest.mock import patch

import torch
from vllm.logger import init_logger
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils
from vllm_ascend.models.deepseek_v41.mixed_quant_attention import (
    MixedQuantPackedCacheOps,
)
from vllm_ascend.models.deepseek_v41.model import DeepseekV41Attention
from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.compat.npu.runtime_config import is_deepseek_v41_model
from afd_plugin.compat.patches.npu.dspark_weight_sharing import should_share
from afd_plugin.model_executor.models.npu.dsv41_packed_cache import (
    AFDMixedQuantPackedCacheOps,
)

PREFILL_TRACE_STEPS = 2
PREFILL_TRACE_TOP_K = 5
NATIVE_LAYER_TRACE_ENV = "AFD_DSV41_NATIVE_LAYER_TRACE"
MAX_LAYER_TRACE_PROMPT_TOKENS = 32
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


class DSV41NativeEPDSparkControlWorker(NPUWorker):
    """Use a bounded vocabulary comparison in the native EP control."""

    # Upstream: NPUWorker.load_model, worker/worker.py @ Ascend 7e37e29a.
    # Reason: native EP DSpark can exhaust HBM comparing loaded weights.
    # Functionality: scope the comparison override to this worker's model load.
    # Signature: matches upstream; no added parameters.
    # Delegate to native load to retain model, sleep and weight-transfer ownership.
    # Cost: startup CPU copies; inference is unchanged.
    # Removal: retire with the DSpark vocabulary comparison patch.
    def load_model(self) -> None:
        # ### PATCH START: scope native DSpark weight comparison
        speculative = self.vllm_config.speculative_config
        if (
            is_deepseek_v41_model(self.vllm_config)
            and self.vllm_config.parallel_config.enable_expert_parallel
            and speculative is not None
            and speculative.use_dspark()
        ):
            init_logger("vllm.afd_dsv41_native_control").info(
                "Native EP DSpark control uses chunked CPU vocabulary comparisons."
            )
            with patch.object(eagle_utils, "_should_share", should_share):
                super().load_model()
            return
        # ### PATCH END: scope native DSpark weight comparison
        super().load_model()


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

    # Upstream: NPUWorker.execute_model, worker/worker.py @ Ascend 7e37e29a.
    # Reason: the prefill hidden-state drift needs per-layer attribution.
    # Functionality: observe full prompt rows at the native decoder boundaries.
    # Signature: matches upstream; no added parameters.
    # Delegate so the runner keeps its scheduling, cache and PP ownership.
    # Cost: bounded device snapshots during forward, CPU hashes after forward.
    # Remove the diagnostic once the prefill drift is located.
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        # ### PATCH START: optional first prefill layer trace
        new_requests = scheduler_output.scheduled_new_reqs
        if len(new_requests) != 1 or os.environ.get(NATIVE_LAYER_TRACE_ENV) != "1":
            return super().execute_model(scheduler_output)
        request = new_requests[0]
        token_count = len(request.prompt_token_ids or ())
        if (
            not 0 < token_count <= MAX_LAYER_TRACE_PROMPT_TOKENS
            or request.num_computed_tokens
            or scheduler_output.num_scheduled_tokens != {request.req_id: token_count}
        ):
            return super().execute_model(scheduler_output)

        captures: list[tuple[str, torch.Tensor, torch.Tensor | None]] = []

        def snapshot(
            boundary: str, hidden: torch.Tensor, pre_mix: torch.Tensor | None = None
        ) -> None:
            captures.append(
                (
                    boundary,
                    hidden[:token_count].detach().clone(),
                    pre_mix[:token_count].detach().clone()
                    if pre_mix is not None
                    else None,
                )
            )

        def decoder_input(
            boundary: str, _module: torch.nn.Module, args: tuple[torch.Tensor, ...]
        ) -> None:
            snapshot(boundary, args[1], args[2])

        def norm_input(
            _module: torch.nn.Module, args: tuple[torch.Tensor, ...]
        ) -> None:
            snapshot("norm.in", args[0])

        def observe_output(
            boundary: str,
            _module: torch.nn.Module,
            _args: tuple[torch.Tensor, ...],
            value: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        ) -> None:
            if isinstance(value, tuple):
                snapshot(boundary, *value)
            else:
                snapshot(boundary, value)

        decoder = self.model_runner.model.language_model.model
        with ExitStack() as hooks:
            hooks.enter_context(
                decoder.embed_tokens.register_forward_hook(
                    partial(observe_output, "embedding.out")
                )
            )
            for layer in decoder.layers:
                boundary = f"layer.{layer.layer_idx}"
                hooks.enter_context(
                    layer.register_forward_pre_hook(
                        partial(decoder_input, f"{boundary}.in")
                    )
                )
                hooks.enter_context(
                    layer.register_forward_hook(
                        partial(observe_output, f"{boundary}.out")
                    )
                )
            hooks.enter_context(decoder.norm.register_forward_pre_hook(norm_input))
            hooks.enter_context(
                decoder.norm.register_forward_hook(partial(observe_output, "norm.out"))
            )
            output = super().execute_model(scheduler_output)
        logger = init_logger("vllm.afd_dsv41_native_control")
        for boundary, hidden, pre_mix in captures:
            record = {
                "request_id": request.req_id,
                "rank": self.rank,
                "prompt_tokens": token_count,
                "boundary": boundary,
                "hidden_state": _tensor_identity(hidden.cpu()),
            }
            if pre_mix is not None:
                record["pre_mix"] = _tensor_identity(pre_mix.cpu())
            logger.info(
                "AFD native layer trace %s", json.dumps(record, allow_nan=False)
            )
        return output
        # ### PATCH END: optional first prefill layer trace

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
