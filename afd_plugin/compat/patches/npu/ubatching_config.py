# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Keep native DBO exclusions except for the plugin-owned text U2 paths.

Upstream: vllm/config/vllm.py at ced6857afa0ea7b2e3f0846a62e1394e90f15607.
"""

from __future__ import annotations

import os

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.engram import EngramConfig, model_has_engram_layers
from vllm.config.vllm import logger
from vllm.platforms import current_platform

from afd_plugin.compat.npu.feature_validation import V41_FULL_QUALIFICATION_ENV
from afd_plugin.compat.npu.runtime_config import (
    is_deepseek_v41_model,
    resolve_deepseek_v41_target_shape,
    validate_deepseek_v41_pd,
)
from afd_plugin.config import (
    HCCL_P2P_CONNECTOR,
    WINDOW_CONNECTOR,
    parse_optional_afd_config,
    use_afd_dspark,
)


def _uses_afd_text_ubatching(config: VllmConfig) -> bool:
    """Select a validated text target before any model or device construction."""
    if current_platform.device_type != "npu":
        return False
    afd_config = parse_optional_afd_config(config)
    model = config.model_config
    if (
        afd_config is None
        or afd_config.connector != HCCL_P2P_CONNECTOR
        or model is None
        or not is_deepseek_v41_model(config)
    ):
        return False
    parallel = config.parallel_config
    speculative = config.speculative_config
    compilation = config.compilation_config
    if (
        envs.VLLM_USE_V2_MODEL_RUNNER is not True
        or not parallel.use_ubatching
        or type(parallel.num_ubatches) is not int
        or parallel.num_ubatches != 2
        or (
            speculative is not None
            and (
                not speculative.use_dspark()
                or speculative.enforce_eager is not True
                or speculative.enable_adaptive_verification
            )
        )
        or (
            (config.engram_config is not None or model.hf_text_config.engram_layer_ids)
            and os.environ.get(V41_FULL_QUALIFICATION_ENV) != "1"
        )
        or config.lora_config is not None
        or parallel.pipeline_parallel_size > 1
        or parallel.decode_context_parallel_size > 1
        or parallel.prefill_context_parallel_size > 1
        or model.is_encoder_decoder
        or model.is_hybrid
        or config.is_mm_encoder_only
        or (
            model.is_multimodal_model
            and (
                model.multimodal_config is None
                or model.multimodal_config.language_model_only is not True
            )
        )
        or compilation.mode.name != "NONE"
        or compilation.use_inductor_graph_partition
        or not (
            (model.enforce_eager and compilation.cudagraph_mode == CUDAGraphMode.NONE)
            or (
                model.enforce_eager is False
                and compilation.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
            )
        )
    ):
        return False
    # The native no-draft target has Q1; an optional native DSpark draft keeps
    # its checkpoint/aux/TP validation. Both retain capacity and PD boundaries,
    # and configuration admission does not open the model serving guard.
    resolve_deepseek_v41_target_shape(config)
    validate_deepseek_v41_pd(config)
    return True


def _uses_window_text_ubatching(config: VllmConfig) -> bool:
    if (
        current_platform.device_type != "npu"
        or envs.VLLM_USE_V2_MODEL_RUNNER is not True
    ):
        return False
    afd_config = parse_optional_afd_config(config)
    if afd_config is None or afd_config.connector != WINDOW_CONNECTOR:
        return False
    model = config.model_config
    parallel = config.parallel_config
    return (
        afd_config.is_attention_server
        and parallel.use_ubatching
        and parallel.num_ubatches == 2
        and model is not None
        and is_deepseek_v41_model(config)
        and model.multimodal_config is not None
        and model.multimodal_config.language_model_only
        and not model.hf_text_config.engram_layer_ids
    )


# Upstream symbol: VllmConfig._get_dbo_unsupported_features at the module's
# pinned vLLM commit; copy the full method again when that source lock changes.
# Patch reason: native DBO excludes all speculation and multimodal identities,
# including language-only PD consumers whose enable_mm_embeds sets MM identity.
# Patch functionality: admit the connector-specific HCCL and Window V4.1
# language-only U2 profiles and their permitted native DSpark combinations.
# Retain every other native exclusion and leave configuration fields intact.
# Cost: host config validation only; no tensor allocation, RPC or device sync.
# Removal: delete when the pinned native gate supports this connector-owned path.
# Signature: matches upstream; no added parameters.
def get_dbo_unsupported_features(self) -> list[str]:
    """Collect what the V2 model runner cannot combine with DBO.

    The V2 runner microbatches a plain decoder forward pass. Anything that
    slices or replays the batch differently (drafting, adapters, pipeline
    stages, context parallelism, encoders) is not handled yet.
    """
    # TODO: DBO with model runner V2 is under development.
    # It should be enabled with explicit VLLM_USE_V2_MODEL_RUNNER environ.
    # Remove it when stable.
    if envs.VLLM_USE_V2_MODEL_RUNNER is None:
        return ["dual batch overlap"]

    unsupported: list[str] = []
    model_config = self.model_config
    parallel_config = self.parallel_config
    # ### PATCH START: AFD text U2 admission
    afd_text_ubatching = _uses_afd_text_ubatching(self)
    window_text_ubatching = _uses_window_text_ubatching(self)
    # ### PATCH END: AFD text U2 admission

    if self.lora_config is not None:
        unsupported.append("dual batch overlap with LoRA")
    # ### PATCH START: AFD text U2 admission
    if self.speculative_config is not None and not (
        afd_text_ubatching or (window_text_ubatching and use_afd_dspark(self))
    ):
        # ### PATCH END: AFD text U2 admission
        unsupported.append("dual batch overlap with speculative decoding")
    if parallel_config.pipeline_parallel_size > 1:
        unsupported.append("dual batch overlap with pipeline parallelism")
    if (
        parallel_config.decode_context_parallel_size > 1
        or parallel_config.prefill_context_parallel_size > 1
    ):
        unsupported.append("dual batch overlap with context parallelism")
    # ### PATCH START: AFD text U2 admission
    if model_config is not None and (
        (
            model_config.is_multimodal_model
            and not (afd_text_ubatching or window_text_ubatching)
        )
        or model_config.is_encoder_decoder
    ):
        # ### PATCH END: AFD text U2 admission
        unsupported.append("dual batch overlap with multimodal models")
    if model_config is not None and model_config.is_hybrid:
        unsupported.append("dual batch overlap with hybrid models")
    if self.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE:
        # DBO captures FULL graphs only.
        unsupported.append("dual batch overlap with PIECEWISE CUDA graphs")
    if self.is_mm_encoder_only:
        unsupported.append("dual batch overlap with encoder only models")

    return unsupported


def apply_afd_dspark_ubatching_config_patch() -> None:
    """Install the NPU-scoped method before native config validation runs."""
    VllmConfig._get_dbo_unsupported_features = get_dbo_unsupported_features
    VllmConfig._resolve_and_verify_engram_config = resolve_and_verify_engram_config


# Upstream: VllmConfig._resolve_and_verify_engram_config @ ced6857a.
# Reason: native excludes Engram U2 even when AFD owns one parent producer and
# token-aligned lookup views. Functionality: retain model/parallel/load checks
# and the native rejection outside the explicit AFD full-checkpoint candidate.
# Signature matches upstream. Cost: host validation only, no config mutation
# beyond native defaults. Remove when native supports AFD parent preparation.
def resolve_and_verify_engram_config(self) -> None:
    """Resolve defaults and validate n-gram embedding settings."""
    from vllm.platforms import current_platform

    model_config = self.model_config
    speculative_config = self.speculative_config
    # Draft configs inherit the target's communication groups and settings.
    # Validate the target because the draft may disable n-gram embeddings.
    if (
        speculative_config is not None
        and model_config is speculative_config.draft_model_config
    ):
        model_config = speculative_config.target_model_config
    if (
        model_config is not None
        and model_config.architecture == "DeepseekV41ForCausalLM"
        and getattr(model_config.hf_text_config, "engram_layer_ids", None)
        and self.parallel_config.use_ubatching
        # ### PATCH START: parent-owned AFD Engram U2 candidate
        and not _uses_afd_text_ubatching(self)
        # ### PATCH END: parent-owned AFD Engram U2 candidate
    ):
        raise ValueError(
            "DeepSeek V4.1 Engram does not support DBO or microbatching. "
            "Disable --enable-dbo and set --ubatch-size to 0."
        )
    if self.engram_config is None:
        if not current_platform.is_cuda() or not model_has_engram_layers(model_config):
            return
        self.engram_config = EngramConfig()
    self.engram_config.verify_model_config(model_config)
    self.engram_config.verify_parallel_config(self.parallel_config)
    self.engram_config.verify_load_config(self.load_config)
    logger.info_once("Resolved Engram configuration: %s", str(self.engram_config))
