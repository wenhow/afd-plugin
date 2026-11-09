# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Role configuration outcomes, without a device or upstream imports."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.runtime_config import (
    get_npu_ffn_config,
    validate_deepseek_v41_speculation,
)
from tests.unit.fakes.vllm_config import npu_vllm_config, v41_dspark_config


@pytest.mark.parametrize("has_engram", [False, True])
@pytest.mark.parametrize(
    "model_type,architectures",
    [
        ("deepseek_v41", None),
        ("deepseek_v41", ["DeepseekV41ForCausalLM"]),
        ("custom", ["DeepseekV41ForCausalLM"]),
        ("custom", ["AFDDeepseekV41ForCausalLM"]),
    ],
)
def test_ffn_config_disables_engram_without_changing_model_identity(
    has_engram, model_type, architectures
):
    engram = SimpleNamespace(dp_shared_memory=False) if has_engram else None
    source = npu_vllm_config(
        role="ffn",
        model_type=model_type,
        architectures=architectures,
        engram_config=engram,
    )

    result = get_npu_ffn_config(source)

    assert result.engram_config is None
    assert result is not source
    assert result.model_config is source.model_config
    assert source.engram_config is engram
    if engram is not None:
        assert engram.dp_shared_memory is False


@pytest.mark.parametrize(
    "model_type,architecture",
    [
        ("deepseek_v2", "DeepseekV2ForCausalLM"),
        ("deepseek_v4", "DeepseekV4ForCausalLM"),
        ("custom", "AFDDeepseekV4ForCausalLM"),
        ("custom", "DeepseekV410ForCausalLM"),
    ],
)
def test_ffn_config_preserves_existing_model_configurations(model_type, architecture):
    source = npu_vllm_config(
        role="ffn",
        model_type=model_type,
        architectures=[architecture],
        engram_config=SimpleNamespace(dp_shared_memory=False),
    )

    assert get_npu_ffn_config(source) is source
    assert source.engram_config.dp_shared_memory is False


def test_dspark_validation_retains_native_target_and_draft_metadata():
    config = v41_dspark_config(tensor_parallel_size=4)
    before = deepcopy(config)

    validate_deepseek_v41_speculation(config)

    assert config == before
    assert config.model_config.hf_text_config.n_routed_experts == 384
    assert (
        config.speculative_config.draft_model_config.hf_text_config.n_routed_experts
        == 128
    )


@pytest.mark.parametrize(
    "owner,field,value,reason",
    [
        ("spec", "method", "mtp", "method='dspark'"),
        ("spec", "method", "eagle3", "method='dspark'"),
        ("spec", "rejection_sample_method", "synthetic", "real acceptance"),
        ("draft", "architectures", ["DSparkDraftModel"], "normalized"),
        ("draft_hf", "architectures", ["DeepseekV41ForCausalLM"], "normalized"),
        ("draft_hf", "model_type", "deepseek_v4", "normalized"),
        ("draft", "model", "other-checkpoint", "same target checkpoint"),
        ("draft_parallel", "tensor_parallel_size", 2, "equal target and draft TP"),
        ("target", "dspark_target_layer_ids", [], "one-based aux IDs"),
        ("target", "dspark_target_layer_ids", [0, 38, 39], "one-based aux IDs"),
        ("target", "dspark_target_layer_ids", [37, 38, 41], "one-based aux IDs"),
        ("target", "dspark_target_layer_ids", [39, 38, 37], "one-based aux IDs"),
        ("target", "dspark_target_layer_ids", [37, 37, 39], "one-based aux IDs"),
        ("target", "dspark_target_layer_ids", [True, 38, 39], "one-based aux IDs"),
        ("draft_hf", "dspark_target_layer_ids", [36, 38, 39], "one-based aux IDs"),
        ("target", "dspark_block_size", 0, "checkpoint draft metadata"),
        ("draft_hf", "dspark_block_size", 3, "checkpoint draft metadata"),
        ("draft_hf", "n_mtp_layers", 5, "checkpoint draft metadata"),
        ("draft_hf", "n_routed_experts", 384, "checkpoint draft metadata"),
        ("draft_hf", "num_experts_per_tok", 6, "checkpoint draft metadata"),
    ],
)
def test_dspark_rejects_configuration_drift(owner, field, value, reason):
    config = v41_dspark_config(tensor_parallel_size=4)
    spec = config.speculative_config
    owners = {
        "spec": spec,
        "draft": spec.draft_model_config,
        "draft_hf": spec.draft_model_config.hf_config,
        "draft_parallel": spec.draft_parallel_config,
        "target": config.model_config.hf_text_config,
    }
    setattr(owners[owner], field, value)

    with pytest.raises(ValueError, match=reason):
        validate_deepseek_v41_speculation(config)


def test_ffn_projection_rejects_classic_mtp_without_mutating_source():
    source = v41_dspark_config(role="ffn")
    source.speculative_config.method = "mtp"
    before = deepcopy(source)

    with pytest.raises(ValueError, match="method='dspark'"):
        get_npu_ffn_config(source)

    assert source == before


def test_ffn_config_rejects_a_v41_attention_role():
    source = npu_vllm_config(model_type="deepseek_v41")

    with pytest.raises(ValueError, match="role"):
        get_npu_ffn_config(source)


@pytest.mark.parametrize("tensor_parallel_size", [1, 4])
def test_ffn_config_leaves_dspark_and_engram_on_attention(tensor_parallel_size):
    source = v41_dspark_config(role="ffn", tensor_parallel_size=tensor_parallel_size)
    source.engram_config = SimpleNamespace(dp_shared_memory=False)
    speculative = source.speculative_config

    result = get_npu_ffn_config(source)

    assert result.speculative_config is None
    assert result.engram_config is None
    assert result.model_config is source.model_config
    assert source.speculative_config is speculative
    assert source.speculative_config.draft_model_config.architectures == [
        "DeepseekV41DSparkModel"
    ]
    assert source.engram_config.dp_shared_memory is False
