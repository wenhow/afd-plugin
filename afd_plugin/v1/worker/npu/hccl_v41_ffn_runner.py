# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit synchronous V4.1 FFN execution, outside production admission.

The caller owns native model/TP initialization and accepts the parent before
entering here. This component keeps C2's DP1/PP1, non-SP, eager MoE context.
It does not select a factory, prepare requests, reduce native outputs or abort
remote ranks. A terminal failure requires external worker supervision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from afd_plugin.compat.npu import ascend_forward_context
from afd_plugin.connectors.metadata import AFDForwardContextMetadata
from afd_plugin.connectors.npu.hccl_diagnostics import HcclStageDiagnostics
from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.v1.worker.ffn_metadata import _set_moe_layer_index

if TYPE_CHECKING:
    from afd_plugin.model_executor.models.npu.deepseek_v41 import (
        AFDDeepseekV41FFNForCausalLM,
    )
    from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner


def execute_hccl_v41_parent(
    runner: AFDNPUFFNModelRunner,
    transport: HcclStagedTransport,
    *,
    diagnostics: HcclStageDiagnostics | None = None,
) -> None:
    """Run all accepted layer/stage aggregates and complete one parent.

    Every active stage computes its wire rows, including padding-only peers.
    Valid counts describe the aggregate but need not form one contiguous
    prefix: padding may separate the individual Attention peer slices.
    Complete native MoE owns its routing, shared experts and TP reduction.
    """
    try:
        if (
            transport.role != "ffn"
            or not transport.initialized
            or transport.closed
            or transport.terminal
        ):
            raise RuntimeError("V4.1 FFN execution requires an active FFN transport")
        plan = transport.plan
        if plan is None:
            raise RuntimeError("HCCL parent must be accepted before FFN execution")
        parallel = runner.vllm_config.parallel_config
        if (
            parallel.data_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.use_sequence_parallel_moe
        ):
            raise ValueError(
                "V4.1 staged FFN requires DP1/PP1/CP1 and non-SP execution"
            )
        if parallel.tensor_parallel_size != plan.stages[0].native_tp_size:
            raise ValueError("V4.1 staged FFN native TP size does not match the plan")
        if plan.stages[0].ffn_size != parallel.tensor_parallel_size:
            raise ValueError("V4.1 staged FFN role size must match its native TP world")
        if parallel.rank != transport.role_rank:
            raise ValueError("V4.1 staged FFN native rank does not match role_rank")
        progress = None
        if diagnostics is not None:
            if (
                diagnostics.selection.role != "ffn"
                or diagnostics.selection.phase != "eager"
                or diagnostics.selection.num_stages != len(plan.stages)
                or diagnostics.role_rank != transport.role_rank
            ):
                raise ValueError(
                    "FFN eager diagnostics must match the accepted role/stages"
                )
            progress = diagnostics.bind_execution(
                layer_indices=plan.layer_indices,
                actual_compute_stream="parent",
                in_profile_run=plan.mode == "profile",
                is_warmup=plan.mode in ("warmup", "dummy"),
            )
        if not any(any(stage.attention_padded_tokens) for stage in plan.stages):
            transport.finish_forward()
            return
        if runner.model is None:
            raise RuntimeError("NPU FFN model has not been loaded")
        model = cast("AFDDeepseekV41FFNForCausalLM", runner.model)
        for layer_idx in plan.layer_indices:
            for stage in plan.stages:
                if not any(stage.attention_padded_tokens):
                    continue
                peers = stage.for_ffn(transport.role_rank)
                rows = sum(peer.padded_tokens for peer in peers)
                # Existing AFD child metadata is stage-projected: the lists
                # contain only this stage, while stage_idx identifies its plan.
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
                if progress is not None:
                    progress.record("recv_begin", layer_idx, stage.stage_idx)
                payload = transport.receive_layer(layer_idx, stage.stage_idx)
                if progress is not None:
                    progress.record("recv_returned", layer_idx, stage.stage_idx)
                with ascend_forward_context(
                    vllm_config=runner.vllm_config,
                    afd_metadata=metadata,
                    model_instance=runner.model,
                    num_tokens=rows,
                    num_tokens_across_dp=torch.tensor([rows]),
                    in_profile_run=plan.mode == "profile",
                    input_ids=payload.input_ids,
                ) as forward_context:
                    _set_moe_layer_index(forward_context, layer_idx)
                    if progress is not None:
                        progress.record("compute_begin", layer_idx, stage.stage_idx)
                    output = model.compute_ffn_output(
                        hidden_states=payload.hidden_states,
                        layer_idx=layer_idx,
                        input_ids=payload.input_ids,
                        hidden_states_fp32=payload.hidden_states_fp32,
                        already_sequence_parallel=False,
                    )
                    if progress is not None:
                        progress.record("compute_returned", layer_idx, stage.stage_idx)
                        progress.synchronize_compute(layer_idx, stage.stage_idx)
                        progress.record("send_begin", layer_idx, stage.stage_idx)
                    transport.send_layer_output(layer_idx, stage.stage_idx, output)
                    if progress is not None:
                        progress.record("send_returned", layer_idx, stage.stage_idx)
        transport.finish_forward()
    except BaseException:
        # No best-effort network abort can release a peer blocked in HCCL.
        # Preserve the original error and let supervision terminate the session.
        transport.terminal = True
        raise
