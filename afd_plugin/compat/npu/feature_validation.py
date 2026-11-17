# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Validation for AFD features supported by the Ascend runtime."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from afd_plugin.compat.npu.runtime_config import (
    HCCL_ATTESTED_PD_MODULE,
    is_deepseek_v41_model,
)
from afd_plugin.config import (
    AFD_ASYNC_CONNECTOR,
    CAMP2P_CONNECTOR,
    AFDConfig,
    is_afd_async_dp,
    parse_afd_config,
)
from afd_plugin.envs import async_cam_layered_gmm_enabled
from afd_plugin.validation import cam_receive_capacity, requested_ffn_mode

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from afd_plugin.connectors.base import ConnectorExtraInfo


V41_EAGER_QUALIFICATION_ENV = "AFD_DSV41_EAGER_QUALIFICATION"
V41_EAGER_QUALIFICATION_TP_SIZE = 4
V41_EAGER_QUALIFICATION_MAX_TOKENS = 1024
V41_DUMMY_QUALIFICATION_ENV = "AFD_DSV41_DUMMY_QUALIFICATION"
V41_DUMMY_HIDDEN_SIZE = 512
V41_DUMMY_HEAD_SIZE = 512
V41_DUMMY_HC_MULT = 4
V41_DUMMY_QUERY_HEADS = 64
V41_DUMMY_LORA_RANK = 64
V41_DUMMY_LAYER_COUNT = 2
V41_DUMMY_C2_LAYER_COUNT = 3
V41_DUMMY_C2_COMPRESS_RATIOS = (0, 0, 2)
V41_DUMMY_C2_SOURCE_LAYER = 2
V41_DUMMY_CANDIDATE_SOURCE_LAYER = 20
V41_DUMMY_VOCAB_SIZE = 4096
V41_DUMMY_EXPERT_COUNT = 8
V41_DUMMY_EXPERT_TOP_K = 2
V41_DUMMY_INDEX_HEAD_SIZE = 128
V41_DUMMY_MAX_MODEL_LEN = 128
V41_DUMMY_BLOCK_SIZE = 128
V41_DUMMY_BATCH_TOKENS = 16
V41_FUNCTIONAL_QUALIFICATION_ENV = "AFD_DSV41_FUNCTIONAL_QUALIFICATION"
V41_FUNCTIONAL_INDEX_HEADS = 32
V41_FUNCTIONAL_HIDDEN_SIZE = 1024
V41_FUNCTIONAL_EXPERT_COUNT = 64
V41_FUNCTIONAL_MAX_SEQUENCES = 2
V41_FUNCTIONAL_MAX_SPECULATIVE_TOKENS = 2


def _validate_v41_eager_qualification(
    vllm_config: VllmConfig, afd_config: AFDConfig
) -> None:
    """Admit one explicit hardware qualification candidate, not model support.

    Keep the ordinary connector validator in charge of text/BF16/V2, native
    rank pairing, eager U1, F gate and EP/SP/Graph restrictions. This narrower
    candidate adds a fixed topology, real weight loading and A-owned host
    Engram. The worker still runs its normal runtime validation afterwards.
    """
    # Worker/backend isolation: the HCCL validator imports torch and native
    # compilation enums. Unsupported/default V4.1 stays CPU-importable.
    from afd_plugin.connectors.npu.hccl_config import validate_hccl_runtime

    if async_cam_layered_gmm_enabled():
        raise ValueError("V4.1 eager qualification requires Async CAM layered GMM off")
    if vllm_config.speculative_config is not None:
        raise ValueError("V4.1 eager qualification requires speculation off")
    if vllm_config.kv_transfer_config is not None:
        raise ValueError("V4.1 eager qualification requires KV transfer/PD off")
    parallel = vllm_config.parallel_config
    validate_hccl_runtime(vllm_config, afd_config, role_rank=parallel.rank)
    if parallel.tensor_parallel_size != V41_EAGER_QUALIFICATION_TP_SIZE:
        raise ValueError("V4.1 eager qualification requires A4/F4 with native TP4")
    model = vllm_config.model_config
    # The connector validator owns text admission. Native language-only V4.1
    # retains MultiModalConfig, so is_multimodal_model is still True even when
    # its vision path is disabled by multimodal_config.language_model_only.
    if model.enable_sleep_mode or vllm_config.weight_transfer_config is not None:
        raise ValueError("V4.1 eager qualification requires sleep/weight updates off")
    if (
        model.quantization != "deepseek_v4_fp8"
        or vllm_config.quant_config is None
        or vllm_config.quant_config.get_name() != "deepseek_v4_fp8"
    ):
        raise ValueError(
            "V4.1 eager qualification requires deepseek_v4_fp8 quantization"
        )
    if vllm_config.load_config.load_format not in ("auto", "safetensors"):
        raise ValueError(
            "V4.1 eager qualification requires real auto/safetensors loading"
        )
    scheduler = vllm_config.scheduler_config
    if scheduler.max_num_seqs != 1 or scheduler.async_scheduling is not False:
        raise ValueError(
            "V4.1 eager qualification requires one sequence and sync scheduling"
        )
    if (
        not 0 < model.max_model_len <= V41_EAGER_QUALIFICATION_MAX_TOKENS
        or scheduler.max_num_batched_tokens > V41_EAGER_QUALIFICATION_MAX_TOKENS
    ):
        raise ValueError("V4.1 eager qualification requires token/model limits <= 1024")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("V4.1 eager qualification requires prefix caching off")
    if vllm_config.additional_config.get("sparse_kv_offload_config", {}).get(
        "enabled", False
    ):
        raise ValueError("V4.1 eager qualification requires sparse KV offload off")
    if vllm_config.additional_config.get("enable_shared_expert_dp", False) or any(
        int(size) > 0
        for size in vllm_config.additional_config.get(
            "finegrained_tp_config", {}
        ).values()
    ):
        raise ValueError("V4.1 eager qualification requires uniform native TP")
    engram = vllm_config.engram_config
    if afd_config.is_attention_server:
        if (
            engram is None
            or engram.cpu_offload is not True
            or engram.dp_shared_memory is not False
            or engram.embedding_across_dp is not False
        ):
            raise ValueError(
                "V4.1 eager qualification requires Attention CPU Engram "
                "without DP sharing"
            )
    elif engram is not None:
        raise ValueError("V4.1 eager qualification requires FFN Engram isolation")


def _validate_v41_dummy_qualification(
    vllm_config: VllmConfig, afd_config: AFDConfig, *, functional: bool = False
) -> None:
    """Admit only the declared small normal-Worker qualification candidate.

    This default-off entry does not qualify checkpoint accuracy or broaden
    the separate A5 FP8/Engram candidate. Keep explicit U1/no-PD/no-spec
    checks here even if the ordinary HCCL validator gains more combinations.
    The separate functional opt-in reuses this bounded geometry for native
    U2/Graph/DSpark/PD validation; it does not change the old U1 candidate.
    Native dependencies stay deferred for CPU-only default rejection.
    """
    from vllm.config import CompilationMode
    from vllm.config.compilation import CUDAGraphMode

    from afd_plugin.connectors.npu.hccl_config import (
        uses_staged_hccl_runtime,
        validate_hccl_runtime,
        validate_staged_hccl_runtime,
    )

    parallel = vllm_config.parallel_config
    model = vllm_config.model_config
    compilation = vllm_config.compilation_config
    if vllm_config.device_config.device_type != "npu":
        raise ValueError("V4.1 dummy qualification requires NPU")
    if not functional and (
        model.enforce_eager is not True
        or parallel.enable_dbo is not False
        or parallel.use_ubatching is not False
        or type(parallel.num_ubatches) is not int
        or parallel.num_ubatches not in (0, 1)
    ):
        # ced parallel.py reports the raw ubatch_size; Ascend platform.py
        # resets it to 0 without DBO. Both native 0 and explicit 1 are U1.
        raise ValueError("V4.1 dummy qualification requires eager U1")
    if not functional and (
        compilation.mode != CompilationMode.NONE
        or compilation.cudagraph_mode != CUDAGraphMode.NONE
    ):
        raise ValueError(
            "V4.1 dummy qualification requires compilation and graph modes NONE"
        )
    if not functional and vllm_config.speculative_config is not None:
        raise ValueError("V4.1 dummy qualification requires speculation off")
    if not functional and vllm_config.kv_transfer_config is not None:
        raise ValueError("V4.1 dummy qualification requires KV transfer/PD off")
    text = model.hf_text_config
    if vllm_config.engram_config is not None or text.engram_layer_ids != []:
        raise ValueError("V4.1 dummy qualification requires Engram off")
    if functional:
        transfer = vllm_config.kv_transfer_config
        if transfer is not None and (
            transfer.kv_connector_module_path != HCCL_ATTESTED_PD_MODULE
        ):
            raise ValueError(
                "V4.1 functional qualification requires the attested PD connector"
            )
        if (
            type(parallel.enable_dbo) is not bool
            or type(parallel.use_ubatching) is not bool
            or type(parallel.num_ubatches) is not int
            or (parallel.enable_dbo and not parallel.use_ubatching)
            or parallel.num_ubatches not in ((2,) if parallel.use_ubatching else (0, 1))
        ):
            raise ValueError("V4.1 functional qualification requires native U1 or U2")
    if afd_config.async_dp:
        raise ValueError("V4.1 dummy qualification requires async AFD off")
    if async_cam_layered_gmm_enabled():
        raise ValueError("V4.1 dummy qualification requires Async CAM layered GMM off")
    if functional and uses_staged_hccl_runtime(vllm_config, afd_config):
        validate_staged_hccl_runtime(
            vllm_config,
            afd_config,
            role_rank=parallel.rank,
            graph=not model.enforce_eager,
        )
    else:
        validate_hccl_runtime(vllm_config, afd_config, role_rank=parallel.rank)
    if parallel.tensor_parallel_size != 1:
        raise ValueError("V4.1 dummy qualification requires A1/F1 with native TP1")
    if model.enable_sleep_mode or vllm_config.weight_transfer_config is not None:
        raise ValueError("V4.1 dummy qualification requires sleep/weight updates off")
    if (
        model.quantization is not None
        or vllm_config.quant_config is not None
        or text.quantization_config is not None
        or text.expert_dtype != "bf16"
    ):
        raise ValueError("V4.1 dummy qualification requires unquantized BF16")
    if vllm_config.load_config.load_format != "dummy":
        raise ValueError("V4.1 dummy qualification requires dummy loading")
    scheduler = vllm_config.scheduler_config
    maximum_sequences = V41_FUNCTIONAL_MAX_SEQUENCES if functional else 1
    if (
        type(scheduler.max_num_seqs) is not int
        or not 0 < scheduler.max_num_seqs <= maximum_sequences
    ):
        raise ValueError(
            "V4.1 functional qualification requires one or two sequences"
            if functional
            else "V4.1 dummy qualification requires one sequence"
        )
    if scheduler.async_scheduling is not False:
        raise ValueError("V4.1 dummy qualification requires sync scheduling")
    if (
        type(model.max_model_len) is not int
        or model.max_model_len != V41_DUMMY_MAX_MODEL_LEN
    ):
        raise ValueError("V4.1 dummy qualification requires max_model_len=128")
    if (
        type(scheduler.max_num_batched_tokens) is not int
        or scheduler.max_num_batched_tokens != V41_DUMMY_BATCH_TOKENS
    ):
        raise ValueError("V4.1 dummy qualification requires batch capacity=16")
    # A 16-token batch with max length 128 needs native chunked prefill at
    # first EngineArgs construction; disabling it fails SchedulerConfig.
    if scheduler.enable_chunked_prefill is not True:
        raise ValueError("V4.1 dummy qualification requires chunked prefill")
    cache = vllm_config.cache_config
    if type(cache.block_size) is not int or cache.block_size != V41_DUMMY_BLOCK_SIZE:
        raise ValueError("V4.1 dummy qualification requires block_size=128")
    if cache.enable_prefix_caching:
        raise ValueError("V4.1 dummy qualification requires prefix caching off")
    if cache.cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("V4.1 dummy qualification requires BF16 KV cache")
    if vllm_config.additional_config.get("sparse_kv_offload_config", {}).get(
        "enabled", False
    ):
        raise ValueError("V4.1 dummy qualification requires sparse KV offload off")
    if vllm_config.additional_config.get("enable_shared_expert_dp", False) or any(
        int(size) > 0
        for size in vllm_config.additional_config.get(
            "finegrained_tp_config", {}
        ).values()
    ):
        raise ValueError("V4.1 dummy qualification requires uniform native TP")
    for name, actual, expected in (
        ("hc_mult", text.hc_mult, V41_DUMMY_HC_MULT),
        ("num_attention_heads", text.num_attention_heads, V41_DUMMY_QUERY_HEADS),
        ("num_key_value_heads", text.num_key_value_heads, 1),
        ("head_dim", text.head_dim, V41_DUMMY_HEAD_SIZE),
        ("qk_rope_head_dim", text.qk_rope_head_dim, V41_DUMMY_LORA_RANK),
        ("q_lora_rank", text.q_lora_rank, V41_DUMMY_LORA_RANK),
        ("kv_lora_rank", text.kv_lora_rank, V41_DUMMY_HEAD_SIZE),
        ("o_lora_rank", text.o_lora_rank, V41_DUMMY_LORA_RANK),
        ("o_groups", text.o_groups, 1),
        ("sliding_window", text.sliding_window, V41_DUMMY_BLOCK_SIZE),
        ("index_topk", text.index_topk, V41_DUMMY_HEAD_SIZE),
        ("index_head_dim", text.index_head_dim, V41_DUMMY_INDEX_HEAD_SIZE),
        ("vocab_size", text.vocab_size, V41_DUMMY_VOCAB_SIZE),
        ("intermediate_size", text.intermediate_size, V41_DUMMY_HIDDEN_SIZE),
        ("moe_intermediate_size", text.moe_intermediate_size, V41_DUMMY_HIDDEN_SIZE),
        ("num_experts_per_tok", text.num_experts_per_tok, V41_DUMMY_EXPERT_TOP_K),
        ("n_shared_experts", text.n_shared_experts, 1),
        ("n_group", text.n_group, 1),
        ("topk_group", text.topk_group, 1),
        ("num_hash_layers", text.num_hash_layers, 0),
    ):
        if type(actual) is not int or actual != expected:
            raise ValueError(f"V4.1 dummy qualification requires {name}={expected}")
    # The pinned A5 hash router uses full-vector UB loads. The normal Worker
    # fixture declares 64 experts; retain eight for historical components.
    expert_counts = (
        (V41_DUMMY_EXPERT_COUNT, V41_FUNCTIONAL_EXPERT_COUNT)
        if functional
        else (V41_DUMMY_EXPERT_COUNT,)
    )
    if (
        type(text.n_routed_experts) is not int
        or text.n_routed_experts not in expert_counts
    ):
        raise ValueError(
            f"V4.1 dummy qualification requires n_routed_experts in {expert_counts}"
        )
    # Native A5 CompressorV2 requires H >= 1024. Retain the 512-wide
    # historical/controlled components; this opt-in is not hardware acceptance.
    hidden_sizes = (
        (V41_DUMMY_HIDDEN_SIZE, V41_FUNCTIONAL_HIDDEN_SIZE)
        if functional
        else (V41_DUMMY_HIDDEN_SIZE,)
    )
    if type(text.hidden_size) is not int or text.hidden_size not in hidden_sizes:
        raise ValueError(
            f"V4.1 dummy qualification requires hidden_size in {hidden_sizes}"
        )
    # A5 packed QLI requires 32 replicated index heads. Keep the historical
    # 64-head candidate for its separate non-packed component evidence.
    index_heads = (
        (V41_FUNCTIONAL_INDEX_HEADS, V41_DUMMY_QUERY_HEADS)
        if functional
        else (V41_DUMMY_QUERY_HEADS,)
    )
    if type(text.index_n_heads) is not int or text.index_n_heads not in index_heads:
        raise ValueError(
            f"V4.1 dummy qualification requires index_n_heads in {index_heads}"
        )
    layer_count = text.num_hidden_layers
    if type(layer_count) is not int or layer_count not in (
        V41_DUMMY_LAYER_COUNT,
        V41_DUMMY_C2_LAYER_COUNT,
    ):
        raise ValueError("V4.1 dummy qualification requires num_hidden_layers=2 or 3")
    if len(text.compress_ratios) != layer_count:
        raise ValueError(
            "V4.1 dummy qualification requires matching num_hidden_layers "
            "and compress_ratios"
        )
    expected_ratios = (
        V41_DUMMY_C2_COMPRESS_RATIOS
        if layer_count == V41_DUMMY_C2_LAYER_COUNT
        else (0,) * V41_DUMMY_LAYER_COUNT
    )
    if any(
        type(ratio) is not int or ratio != expected
        for ratio, expected in zip(text.compress_ratios, expected_ratios, strict=True)
    ):
        raise ValueError(
            "V4.1 dummy qualification requires compress_ratios=[0, 0] "
            "or the three-layer C2 candidate [0, 0, 2]"
        )
    if layer_count == V41_DUMMY_C2_LAYER_COUNT:
        for name, sources in (
            ("kv_source_layer_ids", text.kv_source_layer_ids),
            ("index_source_layer_ids", text.index_source_layer_ids),
        ):
            if sources != [V41_DUMMY_C2_SOURCE_LAYER] or type(sources[0]) is not int:
                raise ValueError(f"V4.1 dummy C2 qualification requires {name}=[2]")
        if (
            type(text.candidate_source_layer_id) is not int
            or text.candidate_source_layer_id != V41_DUMMY_CANDIDATE_SOURCE_LAYER
        ):
            raise ValueError(
                "V4.1 dummy C2 qualification requires candidate_source_layer_id=20"
            )
    if functional:
        if layer_count != V41_DUMMY_C2_LAYER_COUNT:
            raise ValueError("V4.1 functional qualification requires three-layer C2")
        speculative = vllm_config.speculative_config
        if speculative is not None:
            if (
                speculative.quantization is not None
                or speculative.kv_cache_dtype not in (None, "auto", "bfloat16")
            ):
                raise ValueError(
                    "V4.1 functional qualification requires unquantized BF16 draft/KV"
                )
            if (
                speculative.enforce_eager is not True
                or speculative.enable_adaptive_verification
                or type(speculative.num_speculative_tokens) is not int
                or not 0
                < speculative.num_speculative_tokens
                <= V41_FUNCTIONAL_MAX_SPECULATIVE_TOKENS
            ):
                raise ValueError(
                    "V4.1 functional qualification requires fixed eager DSpark "
                    "with one or two speculative tokens"
                )
            draft = speculative.draft_model_config.hf_text_config
            if draft.engram_layer_ids != []:
                raise ValueError(
                    "V4.1 functional qualification requires draft Engram off"
                )
            if (
                type(text.num_nextn_predict_layers) is not int
                or text.num_nextn_predict_layers != 1
                or type(text.dspark_block_size) is not int
                or text.dspark_block_size != 2
                or text.dspark_target_layer_ids != [1, 2, 3]
                or any(type(layer) is not int for layer in text.dspark_target_layer_ids)
                or type(text.dspark_n_routed_experts) is not int
                or text.dspark_n_routed_experts != V41_DUMMY_EXPERT_COUNT
                or type(text.dspark_num_experts_per_tok) is not int
                or text.dspark_num_experts_per_tok != V41_DUMMY_EXPERT_TOP_K
            ):
                raise ValueError(
                    "V4.1 functional qualification requires the bounded DSpark geometry"
                )
            for name, actual, expected in (
                ("hidden_size", draft.hidden_size, text.hidden_size),
                ("hc_mult", draft.hc_mult, text.hc_mult),
                (
                    "num_attention_heads",
                    draft.num_attention_heads,
                    text.num_attention_heads,
                ),
                ("head_dim", draft.head_dim, text.head_dim),
                ("q_lora_rank", draft.q_lora_rank, text.q_lora_rank),
                ("kv_lora_rank", draft.kv_lora_rank, text.kv_lora_rank),
                ("o_lora_rank", draft.o_lora_rank, text.o_lora_rank),
                ("vocab_size", draft.vocab_size, text.vocab_size),
                ("expert_dtype", draft.expert_dtype, text.expert_dtype),
                (
                    "quantization_config",
                    draft.quantization_config,
                    text.quantization_config,
                ),
            ):
                if type(actual) is not type(expected) or actual != expected:
                    raise ValueError(
                        f"V4.1 functional qualification requires matching draft {name}"
                    )


def fail_if_unsupported_npu_afd_features(
    vllm_config: VllmConfig,
    *,
    afd_config: AFDConfig | None = None,
) -> None:
    """Fail fast for NPU AFD settings that are not currently supported."""

    afd_config = afd_config or parse_afd_config(vllm_config)
    if is_deepseek_v41_model(vllm_config):
        qualification = os.environ.get(V41_EAGER_QUALIFICATION_ENV, "0")
        if qualification not in ("0", "1"):
            raise ValueError(f"{V41_EAGER_QUALIFICATION_ENV} must be 0 or 1")
        dummy_qualification = os.environ.get(V41_DUMMY_QUALIFICATION_ENV, "0")
        if dummy_qualification not in ("0", "1"):
            raise ValueError(f"{V41_DUMMY_QUALIFICATION_ENV} must be 0 or 1")
        functional_qualification = os.environ.get(V41_FUNCTIONAL_QUALIFICATION_ENV, "0")
        if functional_qualification not in ("0", "1"):
            raise ValueError(f"{V41_FUNCTIONAL_QUALIFICATION_ENV} must be 0 or 1")
        if (qualification, dummy_qualification, functional_qualification).count(
            "1"
        ) > 1:
            raise ValueError("V4.1 qualification candidates are mutually exclusive")
        if functional_qualification == "1":
            _validate_v41_dummy_qualification(vllm_config, afd_config, functional=True)
            return
        if dummy_qualification == "1":
            _validate_v41_dummy_qualification(vllm_config, afd_config)
            return
        if qualification == "1":
            _validate_v41_eager_qualification(vllm_config, afd_config)
            return
        raise RuntimeError(
            "DeepSeek V4.1 NPU AFD is not available yet; "
            "full-checkpoint quantization, text eager and state lifecycle "
            "acceptance are pending",
        )
    layered_gmm_requested = async_cam_layered_gmm_enabled()
    if layered_gmm_requested and (
        afd_config.role != "ffn" or afd_config.connector != AFD_ASYNC_CONNECTOR
    ):
        raise RuntimeError(
            "AFD_ASYNC_CAM_LAYERED_GMM requires the async CAM FFN role; "
            f"got role={afd_config.role!r}, connector={afd_config.connector!r}"
        )
    is_dsv4 = _is_dsv4_target(vllm_config)
    if layered_gmm_requested and not is_dsv4:
        raise RuntimeError("AFD_ASYNC_CAM_LAYERED_GMM supports only DeepSeek V4")
    if is_dsv4:
        _fail_if_unsupported_dsv4_connector(afd_config)

    from afd_plugin.connectors.factory import AFDConnectorFactory

    extra_info = AFDConnectorFactory.parse_connector_extra_info(
        afd_config.connector,
        vllm_config,
    )

    if afd_config.connector == AFD_ASYNC_CONNECTOR:
        _fail_if_unsupported_npu_afd_async_features(
            vllm_config,
            afd_config,
            extra_info,
        )
        if is_dsv4:
            _fail_if_unsupported_dsv4_async_features(afd_config, extra_info)
        return

    if afd_config.compute_gate_on_attention:
        raise RuntimeError(
            "AFD NPU runtime does not support compute_gate_on_attention=true yet",
        )
    if afd_config.connector == CAMP2P_CONNECTOR:
        from afd_plugin.connectors.npu.camp2p import CAMP2PExtraInfo

        if not isinstance(extra_info, CAMP2PExtraInfo):
            raise TypeError(
                "CAMP2pAFDConnector requires CAMP2PExtraInfo, got "
                f"{type(extra_info).__name__}",
            )
        extra_info.validate_supported()

    uses_ubatching = bool(vllm_config.parallel_config.use_ubatching)
    if uses_ubatching and int(vllm_config.parallel_config.num_ubatches) != 2:
        raise RuntimeError(
            "AFD NPU runtime supports exactly two ubatches when DBO is enabled",
        )
    model_config = vllm_config.model_config
    # Mirror the pinned NPUModelRunner's attention backend selection: a sparse
    # (SFA) or compressor (DSA) configuration selects a backend whose
    # graph-params update is a no-op, so only plain MLA takes the MLA DBO full
    # graph path that owns the merged registry.
    hf_text_config = model_config.hf_text_config
    uses_sparse_mla = hasattr(hf_text_config, "index_topk")
    uses_mla_compressor = hasattr(hf_text_config, "compress_ratios")
    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    uses_mla_dbo_full_graph = (
        uses_ubatching
        and model_config.use_mla
        and not uses_sparse_mla
        and not uses_mla_compressor
        and cudagraph_mode.has_full_cudagraphs()
    )
    if uses_mla_dbo_full_graph and vllm_config.speculative_config is not None:
        raise RuntimeError(
            "AFD NPU MLA DBO FULL graph does not support speculative decoding",
        )
    if uses_mla_dbo_full_graph and cudagraph_mode.name != "FULL_DECODE_ONLY":
        raise RuntimeError(
            "AFD NPU MLA DBO graph execution requires FULL_DECODE_ONLY",
        )


def _is_dsv4_target(vllm_config: VllmConfig) -> bool:
    """Return whether the target model is DeepSeek V4."""
    hf_config = vllm_config.model_config.hf_config
    if hf_config.model_type == "deepseek_v4":
        return True
    return any(
        architecture.removeprefix("AFD") == "DeepseekV4ForCausalLM"
        for architecture in (hf_config.architectures or ())
    )


def _fail_if_unsupported_dsv4_async_features(
    afd_config: AFDConfig,
    extra_info: ConnectorExtraInfo,
) -> None:
    """Keep the initial DSV4 Async CAM target path deliberately narrow."""
    from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo

    if not afd_config.compute_gate_on_attention:
        raise RuntimeError(
            "DSV4 CAMAsyncAFDConnector requires compute_gate_on_attention=true",
        )
    if not isinstance(extra_info, AFDAsyncExtraInfo):
        raise TypeError(
            "DSV4 CAMAsyncAFDConnector requires AFDAsyncExtraInfo, got "
            f"{type(extra_info).__name__}",
        )
    if extra_info.dynamic_quant != 1:
        raise RuntimeError(
            "DSV4 Flash-INT8 CAMAsyncAFDConnector requires dynamicQuant=1",
        )


def _fail_if_unsupported_dsv4_connector(afd_config: AFDConfig) -> None:
    """Reject DSV4 on connectors that cannot carry token ids.

    A DSV4 Hash layer routes by token identity, so whichever role owns the gate
    needs the tokens' ids. CAM async reads them from its dispatch metadata;
    CAMP2P moves them over the A2E ids channel with the gate left on FFN.
    """

    if afd_config.connector not in (AFD_ASYNC_CONNECTOR, CAMP2P_CONNECTOR):
        raise RuntimeError(
            "DSV4 NPU AFD supports only CAMAsyncAFDConnector and "
            f"CAMP2pAFDConnector; got {afd_config.connector!r}",
        )


def _fail_if_unsupported_npu_afd_async_features(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    extra_info: ConnectorExtraInfo,
) -> None:
    from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo

    if not isinstance(extra_info, AFDAsyncExtraInfo):
        raise TypeError(
            "CAMAsyncAFDConnector requires AFDAsyncExtraInfo, got "
            f"{type(extra_info).__name__}",
        )

    if vllm_config.additional_config.get("mix_placement", False):
        raise RuntimeError(
            "Async CAM uses routed-only expert IDs and does not support mix_placement"
        )

    parallel_config = vllm_config.parallel_config
    if extra_info.ffn_two_stream:
        if not afd_config.is_ffn_server:
            raise RuntimeError("ffn_two_stream may only be enabled on the FFN role")
        if (
            not vllm_config.use_v2_model_runner
            or not async_cam_layered_gmm_enabled()
            or not afd_config.compute_gate_on_attention
            or extra_info.dynamic_quant != 1
            or parallel_config.tensor_parallel_size != 1
            or parallel_config.use_sequence_parallel_moe
            or os.environ.get("VLLM_ASCEND_ENABLE_FLASHCOMM1", "0").strip().lower()
            in ("1", "true")
        ):
            raise RuntimeError(
                "FFN two-stream requires MRV2 layered W4A8, dynamicQuant=1, "
                "Attention-side gate, F TP=1 and SP disabled"
            )
    if not is_afd_async_dp(vllm_config):
        raise RuntimeError(
            "CAMAsyncAFDConnector requires additional_config['afd'] "
            "with async=true and connector='CAMAsyncAFDConnector'",
        )
    if not bool(vllm_config.model_config.enforce_eager):
        if afd_config.is_attention_server:
            raise RuntimeError(
                "CAMAsyncAFDConnector supports only eager Attention execution"
            )
        requested_ffn_mode(vllm_config)
        if not vllm_config.use_v2_model_runner:
            raise RuntimeError("Async FFN FULL requires MRV2")
        if parallel_config.tensor_parallel_size != 1:
            raise RuntimeError("Async FFN FULL requires F TP=1")
    if bool(parallel_config.enable_dbo) or bool(parallel_config.use_ubatching):
        if not afd_config.is_attention_server:
            raise RuntimeError("Async CAM DBO/ubatching requires the Attention role")
        if parallel_config.num_ubatches != 2:
            raise RuntimeError("Async CAM DBO requires exactly two ubatches")
        if not afd_config.compute_gate_on_attention:
            raise RuntimeError("Async CAM DBO requires compute_gate_on_attention=true")
        if (
            parallel_config.pipeline_parallel_size != 1
            or parallel_config.prefill_context_parallel_size != 1
            or parallel_config.decode_context_parallel_size != 1
            or vllm_config.speculative_config is not None
        ):
            raise RuntimeError(
                "Async CAM DBO does not support PP, CP or speculative decoding"
            )
        if not _is_dsv4_target(vllm_config) and (
            vllm_config.model_config.hf_text_config.model_type
            not in {"deepseek_v2", "deepseek_v3"}
            or not vllm_config.model_config.use_mla
        ):
            raise RuntimeError(
                "Async CAM DBO requires DeepSeek-V2/V3 MLA or DeepSeek-V4"
            )
    speculative_config = vllm_config.speculative_config
    # Native draft MoE still uses Attention-side cross-DP collectives.
    if (
        afd_config.is_attention_server
        and speculative_config is not None
        and (speculative_config.use_eagle() or speculative_config.uses_draft_model())
    ):
        raise RuntimeError(
            "CAMAsyncAFDConnector does not support native draft models or MTP",
        )

    factor_value = os.environ.get("BATCH_SIZE_FACTOR", "1")
    try:
        # Python accepts underscores and Unicode digits that std::stof does not.
        if not factor_value.isascii() or "_" in factor_value:
            raise ValueError("Not a native floating-point literal")
        batch_size_factor = float(factor_value)
    except ValueError as exc:
        raise RuntimeError(
            "BATCH_SIZE_FACTOR must be a decimal number in (0, 1]",
        ) from exc
    if not 0 < batch_size_factor <= 1:
        raise RuntimeError("BATCH_SIZE_FACTOR must be a decimal number in (0, 1]")
    receive_capacity = cam_receive_capacity()
    max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    # whole-expert chunks need a full TP group's batch; smaller
    # buffers require native sub-expert chunking.
    if receive_capacity < max_num_batched_tokens:
        raise RuntimeError(
            "CAMAsyncAFDConnector requires BATCH_SIZE_FACTOR to allocate at least "
            f"max_num_batched_tokens={max_num_batched_tokens} "
            f"rows; got {receive_capacity}",
        )
    if vllm_config.use_v2_model_runner and afd_config.is_attention_server:
        # CAM's per-rank window uses integer division, while replicated TP
        # input layout rounds up; reject capacities whose profile batch overflows.
        if (
            vllm_config.scheduler_config.max_num_batched_tokens
            % extra_info.attn_ranks_per_dp
        ):
            raise RuntimeError(
                "AFD NPU ModelRunnerV2 Async CAM requires max_num_batched_tokens "
                "divisible by attn_ranks_per_dp",
            )
        model_config = vllm_config.model_config
        if not _is_dsv4_target(vllm_config) and (
            model_config.hf_text_config.model_type not in {"deepseek_v2", "deepseek_v3"}
            or not model_config.use_mla
        ):
            raise RuntimeError(
                "AFD NPU ModelRunnerV2 Async CAM requires DeepSeek-V2/V3 MLA "
                "or DeepSeek-V4",
            )
        additional_config = vllm_config.additional_config
        if additional_config.get("enable_shared_expert_dp", False) or any(
            int(size) > 0
            for size in additional_config.get("finegrained_tp_config", {}).values()
        ):
            raise RuntimeError(
                "AFD NPU ModelRunnerV2 Async CAM does not support finegrained TP "
                "or shared-expert DP",
            )
    if extra_info.dynamic_quant not in (0, 1):
        raise RuntimeError(
            "CAMAsyncAFDConnector currently supports only dynamicQuant 0 or 1",
        )
    _validate_cam_world_topology(vllm_config, afd_config, extra_info)


def _validate_cam_world_topology(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    extra_info: ConnectorExtraInfo,
) -> None:
    """Require each role's local layout to fill the one CAM world."""
    from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo

    if not isinstance(extra_info, AFDAsyncExtraInfo):
        return
    parallel_config = vllm_config.parallel_config
    attn_ranks_per_dp = int(extra_info.attn_ranks_per_dp)
    if afd_config.role == "attention":
        if int(parallel_config.tensor_parallel_size) != attn_ranks_per_dp:
            raise RuntimeError(
                "CAMAsyncAFDConnector Attention tensor_parallel_size must equal "
                "attn_ranks_per_dp",
            )
        local_world_size = int(parallel_config.data_parallel_size) * attn_ranks_per_dp
        expected_world_size = afd_config.num_attention_ranks
    else:
        local_world_size = int(parallel_config.data_parallel_size) * int(
            parallel_config.tensor_parallel_size
        )
        expected_world_size = afd_config.num_ffn_ranks
    if local_world_size != expected_world_size:
        raise RuntimeError(
            "CAMAsyncAFDConnector "
            f"{afd_config.role} DPxTP world size must equal its configured "
            f"role size, got {local_world_size} and {expected_world_size}",
        )


__all__ = ["fail_if_unsupported_npu_afd_features"]
