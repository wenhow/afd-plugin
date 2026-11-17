"""Probe real V4.1 config identity and Engram opt-in, without model allocation."""

import json

import vllm_ascend.ops  # noqa: F401
from vllm.config import EngramConfig, ModelConfig, VllmConfig
from vllm.config.device import DeviceConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.utils import get_model_architecture
from vllm.platforms import current_platform
from vllm.plugins import load_general_plugins
from vllm_ascend.models.deepseek_v41.model import (
    _engram_enabled_for_runtime,
    normalize_deepseek_v41_config,
)

from afd_plugin.compat.npu.runtime_config import get_npu_ffn_config

CHECKPOINT = "/mnt/workspace/models/DeepSeek-V4.1-Flash"
MAX_MODEL_LEN = 128


def main():
    current_platform.pre_register_and_update()
    load_general_plugins()
    model_config = ModelConfig(
        model=CHECKPOINT,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        skip_tokenizer_init=True,
    )
    native_class, native_arch = get_model_architecture(model_config)
    source = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(load_format="dummy"),
    )
    source.additional_config = {
        "afd": {"role": "ffn", "connector": "CAMP2pAFDConnector"}
    }
    # Attach an Engram object after native config validation for this API-only
    # cell; this is not evidence of a qualified Engram launch configuration.
    source.engram_config = EngramConfig(cpu_offload=True, dp_shared_memory=False)
    normalized = normalize_deepseek_v41_config(model_config.hf_config)
    ffn = get_npu_ffn_config(source)
    class_after, arch_after = get_model_architecture(ffn.model_config)
    result = {
        "scope": "real config and native Engram opt-in API only",
        "checkpoint": CHECKPOINT,
        "full_model_constructed": False,
        "device_groups_initialized": False,
        "source_engram_retained": source.engram_config is not None,
        "source_shared_memory_unchanged": not source.engram_config.dp_shared_memory,
        "ffn_engram_disabled": ffn.engram_config is None,
        "native_a_engram_enabled": _engram_enabled_for_runtime(normalized, source),
        "native_f_engram_enabled": _engram_enabled_for_runtime(normalized, ffn),
        "model_config_preserved": ffn.model_config is model_config,
        "hf_architectures": list(ffn.model_config.hf_config.architectures),
        "architectures": list(ffn.model_config.architectures),
        "architecture": ffn.model_config.architecture,
        "registry_unchanged": native_class is class_after and native_arch == arch_after,
        "registry_class": class_after.__module__ + "." + class_after.__name__,
    }
    print("ROLE_CONFIG_RESULT " + json.dumps(result), flush=True)
    assert result["source_engram_retained"]
    assert result["source_shared_memory_unchanged"]
    assert result["ffn_engram_disabled"]
    assert result["native_a_engram_enabled"] and not result["native_f_engram_enabled"]
    assert result["model_config_preserved"] and result["registry_unchanged"]
    assert result["hf_architectures"] == result["architectures"] == [native_arch]
    assert result["architecture"] == "DeepseekV41ForCausalLM"
    assert result["registry_class"] == (
        "vllm_ascend.models.deepseek_v41.vl_model.AscendDeepseekV41ForCausalLM"
    )


if __name__ == "__main__":
    main()
