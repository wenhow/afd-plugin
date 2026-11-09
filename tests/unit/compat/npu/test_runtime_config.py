# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Role configuration outcomes, without a device or upstream imports."""

from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.runtime_config import get_npu_ffn_config
from tests.unit.fakes.vllm_config import npu_vllm_config


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


def test_ffn_config_rejects_a_v41_attention_role():
    source = npu_vllm_config(model_type="deepseek_v41")

    with pytest.raises(ValueError, match="role"):
        get_npu_ffn_config(source)
