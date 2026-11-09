# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Config compatibility adjustments for AFD Ascend workers."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

from afd_plugin.config import (
    AFD_ASYNC_CONNECTOR,
    is_afd_async_dp,
    parse_afd_config,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

FLASHINFER_ALL2ALLV_BACKEND = "flashinfer_all2allv"
DEEPSEEK_V2_MODEL_TYPES = frozenset(
    ("deepseek", "deepseek_v2", "deepseek_v3", "deepseek_v32", "glm_moe_dsa")
)


def is_deepseek_v41_model(vllm_config: VllmConfig) -> bool:
    """Match the native or AFD V4.1 identity without matching V4 prefixes."""
    hf_config = vllm_config.model_config.hf_config
    return hf_config.model_type == "deepseek_v41" or any(
        architecture.removeprefix("AFD") == "DeepseekV41ForCausalLM"
        for architecture in (hf_config.architectures or ())
    )


def get_npu_ffn_config(vllm_config: VllmConfig) -> VllmConfig:
    """Detach V4.1's F Engram binding before native worker construction.

    A shallow copy preserves the native ModelConfig identity and caches without
    rerunning VllmConfig validation. Other configuration children remain shared;
    this view only removes F's ownership of the caller's Engram configuration.
    """
    if not is_deepseek_v41_model(vllm_config):
        return vllm_config
    parse_afd_config(vllm_config, expected_role="ffn")
    ffn_config = copy(vllm_config)
    ffn_config.engram_config = None
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
    "configure_async_cam_dbo",
    "fix_all2all_backend_for_afd",
    "get_npu_ffn_config",
    "is_deepseek_v41_model",
    "npu_afd_num_ubatches",
    "npu_model_uses_sequence_parallel_moe",
    "npu_model_uses_sharded_pp_tensors",
    "uses_async_cam_dbo",
]
