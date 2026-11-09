# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Configuration admission for the isolated eager HCCL connector component.

Admission here does not enable DeepSeek V4.1 model execution. The model feature
guard remains closed until the Attention MoE boundary and role equivalence pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import CompilationMode
from vllm.config.compilation import CUDAGraphMode

from afd_plugin.config import HCCL_P2P_CONNECTOR, AFDConfig
from afd_plugin.connectors.base import ConnectorExtraInfo
from afd_plugin.connectors.npu.hccl_protocol import MAX_TARGET_LAYERS, HcclCapabilities

if TYPE_CHECKING:
    from vllm.config import VllmConfig

DEEPSEEK_V41_ARCHITECTURES = frozenset(
    {"DeepseekV41ForCausalLM", "AFDDeepseekV41ForCausalLM"}
)


@dataclass(frozen=True)
class HcclExtraInfo(ConnectorExtraInfo):
    """Optional transfer-row ceiling, bounded by the scheduler on each role."""

    max_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is not None and (
            type(self.max_tokens) is not int or self.max_tokens <= 0
        ):
            raise ValueError("HCCL max_tokens must be a positive integer or None")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> HcclExtraInfo:
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise TypeError("HCCL connector_extra_config must be a mapping")
        unknown = sorted(str(key) for key in raw if key != "max_tokens")
        if unknown:
            raise ValueError(
                "unknown HCCL connector_extra_config field(s): " + ", ".join(unknown)
            )
        return cls(max_tokens=raw.get("max_tokens"))

    def to_mapping(self) -> dict[str, int | None]:
        return {"max_tokens": self.max_tokens}


def validate_hccl_runtime(
    vllm_config: VllmConfig, afd_config: AFDConfig, role_rank: int
) -> HcclCapabilities:
    """Reject unsupported combinations before rendezvous or NPU allocation."""
    afd_config.validate()
    if afd_config.connector != HCCL_P2P_CONNECTOR:
        raise ValueError("HCCL runtime requires P2pHcclAFDConnector")
    if (
        type(role_rank) is not int
        or not 0 <= role_rank < afd_config.num_attention_ranks
    ):
        raise ValueError(
            "P2pHcclAFDConnector requires role_rank in the configured TP range"
        )
    if not vllm_config.use_v2_model_runner:
        raise ValueError("P2pHcclAFDConnector requires ModelRunner V2")

    parallel = vllm_config.parallel_config
    if (
        type(parallel.tensor_parallel_size) is not int
        or parallel.tensor_parallel_size != afd_config.num_attention_ranks
    ):
        raise ValueError(
            "P2pHcclAFDConnector requires tensor_parallel_size equal to A/F ranks"
        )
    for name, size in (
        ("pipeline_parallel_size", parallel.pipeline_parallel_size),
        ("data_parallel_size", parallel.data_parallel_size),
        ("prefill_context_parallel_size", parallel.prefill_context_parallel_size),
        ("decode_context_parallel_size", parallel.decode_context_parallel_size),
    ):
        if size != 1:
            raise ValueError(f"P2pHcclAFDConnector requires {name}=1")
    if parallel.data_parallel_rank != 0:
        raise ValueError("P2pHcclAFDConnector requires data_parallel_rank=0")
    if parallel.enable_expert_parallel:
        raise ValueError("P2pHcclAFDConnector does not support expert parallelism")
    if parallel.enable_eplb or parallel.enable_elastic_ep:
        raise ValueError("P2pHcclAFDConnector does not support EPLB or elastic EP")
    if parallel.enable_fault_tolerance:
        raise ValueError("P2pHcclAFDConnector does not support fault tolerance")
    if parallel.use_sequence_parallel_moe:
        raise ValueError("P2pHcclAFDConnector does not support sequence parallelism")
    # In the pinned upstream, no ubatching is reported as num_ubatches=0;
    # Ascend resets ubatch_size to 0. Both 0 and explicit 1 represent logical U1.
    if (
        parallel.enable_dbo
        or parallel.use_ubatching
        or parallel.num_ubatches not in (0, 1)
    ):
        raise ValueError("P2pHcclAFDConnector requires eager U1 without DBO/ubatching")

    model = vllm_config.model_config
    hf_config = model.hf_config
    architectures = hf_config.architectures or ()
    if (
        hf_config.model_type != "deepseek_v41"
        or not architectures
        or any(arch not in DEEPSEEK_V41_ARCHITECTURES for arch in architectures)
    ):
        raise ValueError("P2pHcclAFDConnector requires exact DeepseekV41ForCausalLM")
    if model.is_encoder_decoder or (
        model.is_multimodal_model
        and (
            model.multimodal_config is None
            or not model.multimodal_config.language_model_only
        )
    ):
        raise ValueError("P2pHcclAFDConnector supports decoder-only text models")
    if not model.enforce_eager:
        raise ValueError("P2pHcclAFDConnector requires enforce_eager=True")
    if model.dtype != torch.bfloat16:
        raise ValueError("P2pHcclAFDConnector requires BF16 model activations")

    compilation = vllm_config.compilation_config
    if compilation.pass_config.enable_sp:
        raise ValueError("P2pHcclAFDConnector does not support sequence parallelism")
    if (
        compilation.mode != CompilationMode.NONE
        or compilation.cudagraph_mode != CUDAGraphMode.NONE
        or compilation.use_inductor_graph_partition
    ):
        raise ValueError(
            "P2pHcclAFDConnector requires compilation and graph modes NONE"
        )
    if vllm_config.speculative_config is not None:
        raise ValueError("P2pHcclAFDConnector does not support speculative decoding")
    if vllm_config.kv_transfer_config is not None:
        raise ValueError("P2pHcclAFDConnector does not support KV transfer/PD")
    if vllm_config.lora_config is not None:
        raise ValueError("P2pHcclAFDConnector does not support LoRA")

    extra = HcclExtraInfo.from_mapping(afd_config.connector_extra_config)
    scheduler_capacity = vllm_config.scheduler_config.max_num_batched_tokens
    if type(scheduler_capacity) is not int or scheduler_capacity <= 0:
        raise ValueError("HCCL scheduler max_num_batched_tokens must be positive")
    text_config = model.hf_text_config
    layer_count = text_config.num_hidden_layers
    if type(layer_count) is not int or not 0 < layer_count <= MAX_TARGET_LAYERS:
        raise ValueError(
            f"HCCL num_hidden_layers must be a positive integer <= {MAX_TARGET_LAYERS}"
        )
    capabilities = HcclCapabilities(
        hidden_size=text_config.hidden_size,
        vocab_size=text_config.vocab_size,
        max_tokens=min(extra.max_tokens or scheduler_capacity, scheduler_capacity),
        layer_indices=tuple(range(layer_count)),
        native_tp_size=parallel.tensor_parallel_size,
        pair_rank=role_rank,
    )
    capabilities.validate()
    return capabilities


__all__ = ["HcclExtraInfo", "validate_hccl_runtime"]
