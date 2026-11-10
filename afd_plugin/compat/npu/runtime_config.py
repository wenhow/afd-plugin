# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Config compatibility adjustments for AFD Ascend workers."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from afd_plugin.config import (
    AFD_ASYNC_CONNECTOR,
    HCCL_P2P_CONNECTOR,
    is_afd_async_dp,
    parse_afd_config,
)

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig, VllmConfig

FLASHINFER_ALL2ALLV_BACKEND = "flashinfer_all2allv"
HCCL_NATIVE_PD_CONNECTOR = "MooncakeHybridConnector"
DEEPSEEK_V2_MODEL_TYPES = frozenset(
    ("deepseek", "deepseek_v2", "deepseek_v3", "deepseek_v32", "glm_moe_dsa")
)


@dataclass(frozen=True, slots=True)
class DeepseekV41TargetShape:
    """Target scheduling shape retained independently of the A-local drafter."""

    target_query_len: int = 1

    def __post_init__(self) -> None:
        if type(self.target_query_len) is not int or self.target_query_len <= 0:
            raise ValueError("V4.1 target query length must be a positive integer")

    def validate_capacity(self, max_num_batched_tokens: int) -> None:
        if self.target_query_len > max_num_batched_tokens:
            raise ValueError("V4.1 target query length exceeds native token capacity")


def resolve_deepseek_v41_target_shape(
    vllm_config: VllmConfig,
) -> DeepseekV41TargetShape | None:
    """Read the original validated target before FFN role isolation.

    The pinned Ascend runner uses num_speculative_tokens + 1 for the target;
    draft query width and HF checkpoint draft block size are separate values.
    No drafter, mutable config child or model state is retained in this object.
    """
    if not is_deepseek_v41_model(vllm_config):
        return None
    validate_deepseek_v41_speculation(vllm_config)
    speculative = vllm_config.speculative_config
    query_len = 1
    if speculative is not None:
        steps = speculative.num_speculative_tokens
        if type(steps) is not int or steps <= 0:
            raise ValueError("V4.1 DSpark requires positive integer speculative steps")
        query_len += steps
    shape = DeepseekV41TargetShape(query_len)
    shape.validate_capacity(vllm_config.scheduler_config.max_num_batched_tokens)
    return shape


def is_deepseek_v41_model(vllm_config: VllmConfig) -> bool:
    """Match the native or AFD V4.1 identity without matching V4 prefixes."""
    hf_config = vllm_config.model_config.hf_config
    return hf_config.model_type == "deepseek_v41" or any(
        architecture.removeprefix("AFD") == "DeepseekV41ForCausalLM"
        for architecture in (hf_config.architectures or ())
    )


def validate_deepseek_v41_speculation(vllm_config: VllmConfig) -> None:
    """Admit native same-checkpoint DSpark components, without enabling serving."""
    speculative = vllm_config.speculative_config
    if speculative is None:
        return
    if not speculative.use_dspark():
        raise ValueError("V4.1 HCCL speculative decoding requires method='dspark'")
    draft = speculative.draft_model_config
    if (
        draft.hf_config.model_type != "deepseek_v41"
        or draft.architectures != ["DeepseekV41DSparkModel"]
        or draft.hf_config.architectures != ["DeepseekV41DSparkModel"]
    ):
        raise ValueError("V4.1 HCCL requires normalized DeepseekV41DSparkModel")
    if draft.model != vllm_config.model_config.model:
        raise ValueError("V4.1 HCCL DSpark requires the same target checkpoint")
    if (
        speculative.draft_parallel_config.tensor_parallel_size
        != vllm_config.parallel_config.tensor_parallel_size
    ):
        raise ValueError("V4.1 HCCL DSpark requires equal target and draft TP")
    if speculative.rejection_sample_method == "synthetic":
        raise ValueError("V4.1 HCCL DSpark requires real acceptance/rejection")
    parallel = vllm_config.parallel_config
    if speculative.enable_adaptive_verification and (
        parallel.enable_dbo or parallel.use_ubatching
    ):
        # Ascend prepare_inputs @ 7e37e29a:723-734 compacts query coordinates,
        # but :864 retains scheduled upper bounds in InputBatch. Native
        # create_ubatch_slices @ ced6857a:49-75 uses those upper bounds to split
        # requests; this component has not reconciled that compacted layout.
        raise ValueError("V4.1 HCCL DSpark U2 requires adaptive verification off")
    target = vllm_config.model_config.hf_text_config
    draft_text = draft.hf_text_config
    layer_ids = target.dspark_target_layer_ids
    if (
        not layer_ids
        or any(type(layer) is not int for layer in layer_ids)
        or list(layer_ids) != sorted(set(layer_ids))
        or not 0 < min(layer_ids) <= max(layer_ids) <= target.num_hidden_layers
        or list(draft_text.dspark_target_layer_ids) != list(layer_ids)
    ):
        raise ValueError("V4.1 HCCL DSpark requires matching ordered one-based aux IDs")
    if (
        type(target.dspark_block_size) is not int
        or target.dspark_block_size <= 0
        or draft_text.dspark_block_size != target.dspark_block_size
        or draft_text.n_mtp_layers != target.num_nextn_predict_layers
        or draft_text.n_routed_experts != target.dspark_n_routed_experts
        or draft_text.num_experts_per_tok != target.dspark_num_experts_per_tok
    ):
        raise ValueError("V4.1 HCCL DSpark requires native checkpoint draft metadata")


def validate_deepseek_v41_pd(vllm_config: VllmConfig) -> None:
    """Limit native PD eligibility without certifying remote cache layouts.

    The caller selects exact V4.1/HCCL. Native MRV2 retains request/history,
    KV transfer and Engram retirement. Configured equal P/D TP is a prerequisite;
    actual checkpoint, physical-page layout and ingress history still require
    separate P/D qualification while the production model guard stays closed.
    """
    transfer: KVTransferConfig | None = vllm_config.kv_transfer_config
    if transfer is None:
        return
    if (
        transfer.kv_connector != HCCL_NATIVE_PD_CONNECTOR
        or transfer.kv_connector_module_path is not None
    ):
        raise ValueError(
            "V4.1 HCCL KV transfer/PD requires native MooncakeHybridConnector"
        )
    if transfer.kv_role not in ("kv_producer", "kv_consumer"):
        raise ValueError("V4.1 HCCL PD requires a separate producer or consumer")
    if (
        not vllm_config.model_config.enforce_eager
        or vllm_config.compilation_config.cudagraph_mode.name != "NONE"
    ):
        raise ValueError("V4.1 HCCL PD requires eager execution without Graph")
    if vllm_config.speculative_config is not None:
        # The pinned SFA draft-ingress path requires offload, which rejects
        # V4.1 compression. Generic Mooncake has no named prompt-draft readiness
        # proof; same-checkpoint DSpark validation alone cannot supply it.
        raise ValueError("V4.1 HCCL PD requires speculation off")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("V4.1 HCCL PD requires local prefix caching disabled")
    if vllm_config.additional_config.get("sparse_kv_offload_config", {}).get(
        "enabled", False
    ):
        raise ValueError("V4.1 HCCL PD requires sparse KV offload disabled")
    parallel = vllm_config.parallel_config
    for name, size in (
        ("pipeline_parallel_size", parallel.pipeline_parallel_size),
        ("data_parallel_size", parallel.data_parallel_size),
        ("prefill_context_parallel_size", parallel.prefill_context_parallel_size),
        ("decode_context_parallel_size", parallel.decode_context_parallel_size),
    ):
        if type(size) is not int or size != 1:
            raise ValueError(f"V4.1 HCCL PD requires {name}=1")
    tp_size = parallel.tensor_parallel_size
    if type(tp_size) is not int or tp_size <= 0:
        raise ValueError("V4.1 HCCL PD requires a positive native TP size")
    # Native MooncakeWorker._get_prefill_decode_size @ 7e37e29a:1864-1884
    # consumes these exact extra fields. PP defaults to one in that native API.
    for side in ("prefill", "decode"):
        layout = transfer.kv_connector_extra_config.get(side, {})
        if (
            type(layout.get("tp_size")) is not int
            or layout.get("tp_size") != tp_size
            or type(layout.get("dp_size")) is not int
            or layout.get("dp_size") != 1
            or type(layout.get("pp_size", 1)) is not int
            or layout.get("pp_size", 1) != 1
        ):
            raise ValueError(
                "V4.1 HCCL PD requires configured equal P/D native TP and DP/PP=1"
            )


def get_npu_ffn_config(vllm_config: VllmConfig) -> VllmConfig:
    """Detach V4.1's A-owned state before native FFN worker construction.

    A shallow copy preserves the native ModelConfig identity and caches without
    rerunning VllmConfig validation. Other configuration children remain shared;
    this view removes F's ownership of Engram and the Attention-local drafter.
    HCCL's validated native PD connector also remains on Attention only.
    """
    if not is_deepseek_v41_model(vllm_config):
        return vllm_config
    afd_config = parse_afd_config(vllm_config, expected_role="ffn")
    validate_deepseek_v41_speculation(vllm_config)
    if afd_config.connector == HCCL_P2P_CONNECTOR:
        # Validate the original target before removing its A-owned fields.
        validate_deepseek_v41_pd(vllm_config)
    parallel = vllm_config.parallel_config
    if (
        vllm_config.speculative_config is not None
        and (parallel.enable_dbo or parallel.use_ubatching)
        and afd_config.connector != HCCL_P2P_CONNECTOR
    ):
        raise ValueError(
            "V4.1 FFN requires DSpark with U1 before role isolation "
            "for non-HCCL connectors"
        )
    # HCCL target ubatching belongs to Attention. F retains target scheduling
    # but must not construct the native Attention-local drafter. Other NPU
    # connectors retain the existing DSpark/U1 restriction above.
    ffn_config = copy(vllm_config)
    ffn_config.engram_config = None
    ffn_config.speculative_config = None
    if afd_config.connector == HCCL_P2P_CONNECTOR:
        ffn_config.kv_transfer_config = None
    return ffn_config


def configure_async_cam_dbo(vllm_config: VllmConfig) -> None:
    """Normalize native ubatching to DBO on every Attention DP size."""
    afd_config = parse_afd_config(vllm_config)
    parallel = vllm_config.parallel_config
    if (
        afd_config.connector != AFD_ASYNC_CONNECTOR
        or not afd_config.is_attention_server
    ):
        return
    if not parallel.use_ubatching:
        return

    parallel.enable_dbo = True
    # Normalizing ubatching happens after ParallelConfig validation.
    # Repeat its native threshold guard so no empty child can be selected.
    if (
        min(
            parallel.dbo_decode_token_threshold,
            parallel.dbo_prefill_token_threshold,
        )
        < parallel.num_ubatches
    ):
        raise ValueError(
            "dbo_decode_token_threshold and dbo_prefill_token_threshold must "
            f"be at least the number of microbatches ({parallel.num_ubatches})."
        )


def uses_async_cam_dbo(vllm_config: VllmConfig) -> bool:
    return is_afd_async_dp(vllm_config) and (
        vllm_config.parallel_config.enable_dbo
        or vllm_config.parallel_config.use_ubatching
    )


def npu_afd_num_ubatches(vllm_config: VllmConfig) -> int:
    parallel_config = vllm_config.parallel_config
    if parallel_config.use_ubatching:
        return int(parallel_config.num_ubatches)
    return 1


def npu_model_uses_sequence_parallel_moe(vllm_config: VllmConfig) -> bool:
    """Match the DSV2 decoder's effective SP setting under pipeline parallelism."""
    parallel_config = vllm_config.parallel_config
    model_type = vllm_config.model_config.hf_text_config.model_type
    return parallel_config.use_sequence_parallel_moe and (
        model_type not in DEEPSEEK_V2_MODEL_TYPES
        or parallel_config.pipeline_parallel_size == 1
    )


def npu_model_uses_sharded_pp_tensors(vllm_config: VllmConfig) -> bool:
    """Return the model's PP wire layout independently of its internal SP.

    Target DSV4 gathers the complete sequence before each PP boundary. Target
    DSV2 disables MoE SP when PP is enabled; its PP tensors are global too.
    """
    return (
        npu_model_uses_sequence_parallel_moe(vllm_config)
        and vllm_config.model_config.hf_text_config.model_type != "deepseek_v4"
    )


def fix_all2all_backend_for_afd(vllm_config: VllmConfig) -> None:
    """Keep configured model SP while preserving the non-SP worker fallback.

    Target Ascend derives SP from ParallelConfig.use_sequence_parallel_moe.
    AscendConfig first applies the effective FlashComm config switch to the
    backend. Preserve that result, including for explicit AFD worker classes.
    Rewriting its all2all backend disables model-owned token sharding even
    when EP/DP/TP requested it. Non-SP workers keep the legacy backend fix.
    """
    parallel_config = vllm_config.parallel_config
    if (
        not parallel_config.use_sequence_parallel_moe
        and not vllm_config.compilation_config.pass_config.enable_sp
        and parallel_config.all2all_backend != FLASHINFER_ALL2ALLV_BACKEND
    ):
        parallel_config.all2all_backend = FLASHINFER_ALL2ALLV_BACKEND


__all__ = [
    "DeepseekV41TargetShape",
    "configure_async_cam_dbo",
    "fix_all2all_backend_for_afd",
    "get_npu_ffn_config",
    "is_deepseek_v41_model",
    "npu_afd_num_ubatches",
    "npu_model_uses_sequence_parallel_moe",
    "npu_model_uses_sharded_pp_tensors",
    "resolve_deepseek_v41_target_shape",
    "uses_async_cam_dbo",
    "validate_deepseek_v41_pd",
    "validate_deepseek_v41_speculation",
]
