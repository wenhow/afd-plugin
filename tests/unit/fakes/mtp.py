# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Ordinary MTP configuration and proposal seams, shared across owners."""

from copy import deepcopy
from types import SimpleNamespace

from tests.unit.fakes.vllm_config import npu_vllm_config


def ordinary_mtp_config(*, steps=2, tp_size=1, role="attention"):
    config = npu_vllm_config(
        role=role,
        connector="P2pHcclAFDConnector",
        model_type="deepseek_v4",
        architectures=["DeepseekV4ForCausalLM"],
        tensor_parallel_size=tp_size,
        num_attention_ranks=tp_size,
        num_ffn_ranks=tp_size,
        cudagraph_mode="NONE",
        enable_expert_parallel=False,
    )
    target = config.model_config.hf_config
    target.hidden_size = 8
    target.vocab_size = 32
    target.num_hidden_layers = 2
    target.num_nextn_predict_layers = 1
    target.hc_mult = 2
    target.hc_eps = 0.01
    target.rms_norm_eps = 1e-6
    target.n_routed_experts = 2
    target.n_shared_experts = 0
    draft = deepcopy(target)
    draft.model_type = "deepseek_mtp"
    draft.architectures = ["DeepSeekV4MTPModel"]
    config.speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=steps,
        enforce_eager=True,
        enable_adaptive_verification=False,
        draft_tensor_parallel_size=tp_size,
        draft_model_config=SimpleNamespace(hf_config=draft),
    )
    config.compilation_config.mode = SimpleNamespace(name="NONE")
    config.lora_config = None
    config.quant_config = None
    return config
