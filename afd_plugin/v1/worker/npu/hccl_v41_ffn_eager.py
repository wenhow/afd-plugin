# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit V4.1 native FFN consumer of an accepted eager U2 ticket owner.

Session construction, matched communicator warmup, stream allocation, request
selection and worker shutdown belong to the caller. Native MoE owns shared
experts and TP reduction; this consumer adds neither a reduction nor a mask.
Failure retains the owner and its storage for external session supervision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from afd_plugin.compat.npu import ascend_forward_context
from afd_plugin.compat.npu.stream import use_npu_current_stream
from afd_plugin.connectors.metadata import AFDForwardContextMetadata
from afd_plugin.connectors.npu.hccl_eager_pipeline import HcclEagerPipeline
from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.v1.worker.ffn_metadata import _set_moe_layer_index

if TYPE_CHECKING:
    from afd_plugin.model_executor.models.npu.deepseek_v41 import (
        AFDDeepseekV41FFNForCausalLM,
    )
    from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner


@torch.inference_mode()
def execute_hccl_v41_eager_parent(
    runner: AFDNPUFFNModelRunner,
    transport: HcclStagedTransport,
    pipeline: HcclEagerPipeline,
    *,
    compute_stream: torch.npu.Stream,
) -> None:
    """Compute each received native child once and finish the owning parent."""
    if (
        not isinstance(pipeline, HcclEagerPipeline)
        or pipeline.transport is not transport
    ):
        raise ValueError("eager FFN requires the exact active transport/plan owner")
    plan = transport.plan
    if (
        transport.eager_pipeline is not pipeline
        or pipeline.plan is not plan
        or plan is None
        or pipeline.failed
        or pipeline.finished
    ):
        raise ValueError("eager FFN requires the exact active transport/plan owner")
    if (
        transport.role != "ffn"
        or not transport.initialized
        or transport.closed
        or transport.terminal
    ):
        raise RuntimeError("eager FFN requires an active FFN transport")
    try:
        device = torch.device("npu", transport.local_device)
        if (
            pipeline.parent_stream.device != device
            or compute_stream.device != device
            or torch.npu.current_stream(device) != pipeline.parent_stream
            or torch.npu.is_current_stream_capturing()
        ):
            raise ValueError(
                "eager FFN requires its actual parent and NPU compute stream"
            )
        with use_npu_current_stream(pipeline.parent_stream):
            if (
                plan.mode != "target"
                or len(plan.stages) != 2
                or (
                    plan.graph_execution is not None
                    and plan.graph_execution.phase != "eager"
                )
                or runner.vllm_config.speculative_config is not None
            ):
                raise ValueError(
                    "eager FFN requires a target U2 parent without speculation"
                )
            parallel = runner.vllm_config.parallel_config
            caps = transport.negotiated_capabilities
            if (
                parallel.data_parallel_size != 1
                or parallel.data_parallel_rank != 0
                or parallel.pipeline_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or parallel.decode_context_parallel_size != 1
                or parallel.use_sequence_parallel_moe
                or parallel.tensor_parallel_size not in (1, 2)
                or not (
                    caps.attention_size
                    == caps.ffn_size
                    == caps.native_tp_size
                    == parallel.tensor_parallel_size
                )
                or parallel.rank != transport.role_rank
            ):
                raise ValueError(
                    "eager FFN requires paired native TP1/TP2, matching rank, "
                    "DP1/PP1/CP1 and non-SP"
                )
            plan.validate(caps)
            if not any(any(stage.attention_padded_tokens) for stage in plan.stages):
                pipeline.finish()
                return
            if runner.model is None:
                raise RuntimeError("NPU FFN model has not been loaded")
            model = cast("AFDDeepseekV41FFNForCausalLM", runner.model)
            if model.get_experts_layer_indices() != plan.layer_indices:
                raise ValueError(
                    "eager FFN requires the complete native MoE layer layout"
                )
            for layer_idx in plan.layer_indices:
                for stage in plan.stages:
                    if not any(stage.attention_padded_tokens):
                        continue
                    peers = stage.for_ffn(transport.role_rank)
                    rows = sum(peer.padded_tokens for peer in peers)
                    metadata = AFDForwardContextMetadata(
                        tokens_start_loc=[0],
                        requests_start_loc=[0],
                        stage_idx=stage.stage_idx,
                        connector=runner.connector,
                        tokens_lens=[rows],
                        num_stages=len(plan.stages),
                        transaction_id=plan.transaction_id,
                        tokens_unpadded_lens=[sum(peer.valid_tokens for peer in peers)],
                    )
                    ticket = pipeline.post_ffn_receive(layer_idx, stage.stage_idx)
                    with (
                        pipeline.ffn_inputs(
                            ticket, consumer_stream=compute_stream
                        ) as payload,
                        ascend_forward_context(
                            vllm_config=runner.vllm_config,
                            afd_metadata=metadata,
                            model_instance=model,
                            num_tokens=rows,
                            num_tokens_across_dp=torch.tensor([rows]),
                            in_profile_run=False,
                            input_ids=payload.input_ids,
                        ) as forward_context,
                    ):
                        _set_moe_layer_index(forward_context, layer_idx)
                        output = model.compute_ffn_output(
                            hidden_states=payload.hidden_states,
                            layer_idx=layer_idx,
                            input_ids=payload.input_ids,
                            hidden_states_fp32=payload.hidden_states_fp32,
                            already_sequence_parallel=False,
                        )
                    pipeline.post_ffn_output(
                        ticket, output, producer_stream=compute_stream
                    )
            pipeline.finish()
    except BaseException:
        if (
            transport.eager_pipeline is pipeline
            and transport.plan is plan
            and not pipeline.finished
            and not pipeline.failed
        ):
            pipeline.fail()
        raise
