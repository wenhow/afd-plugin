"""Probe the pinned native loader API; this does not construct a V4.1 model."""

from __future__ import annotations

import argparse
import inspect
import json
from copy import copy

import torch
import torch.nn as nn
import vllm_ascend.ops  # noqa: F401
from vllm.config import ModelConfig, VllmConfig
from vllm.config.device import DeviceConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.base_loader import (
    BaseModelLoader,
    log_model_inspection,
    log_online_quantization,
)
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.utils import (
    get_model_architecture,
    initialize_model,
)
from vllm.platforms import current_platform
from vllm.plugins import load_general_plugins

PROBE_WEIGHT_SIZE = 1
PROBE_MAX_MODEL_LEN = 128
EXPECTED_NATIVE_CLASS = (
    "vllm_ascend.models.deepseek_v41.vl_model.AscendDeepseekV41ForCausalLM"
)


class LoaderContractProbe(nn.Module):
    """A single CPU parameter, used only to observe native loader selection."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(PROBE_WEIGHT_SIZE))


class ContractProbeLoader(DummyModelLoader):
    # Upstream: vllm/model_executor/model_loader/base_loader.py,
    # BaseModelLoader.create_model, ced6857afa0ea7b2e3f0846a62e1394e90f15607.
    # Reason: prove explicit class injection preserves native load/eval handling.
    # Functionality: construct the one-parameter probe through initialize_model.
    # Signature matches upstream; no added parameters. This class is probe-only.
    def create_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        model = initialize_model(
            vllm_config=vllm_config,
            model_config=model_config,
            prefix=prefix,
            # ### PATCH START: explicit probe class
            model_class=LoaderContractProbe,
            # ### PATCH END: explicit probe class
        )
        log_online_quantization(vllm_config)
        log_model_inspection(model)
        return model


def identity(config: ModelConfig) -> dict:
    return {
        "hf_architectures": list(config.hf_config.architectures),
        "architectures": list(config.architectures),
        "architecture": config.architecture,
        "model_type": config.hf_config.model_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    current_platform.pre_register_and_update()
    load_general_plugins()
    model_config = ModelConfig(
        model=args.checkpoint,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=PROBE_MAX_MODEL_LEN,
        skip_tokenizer_init=True,
    )
    original_identity = identity(model_config)
    native_class, native_arch = get_model_architecture(model_config)
    runtime = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(load_format="dummy"),
    )
    # A separate, CPU-only API cell; no claim about checkpoint quantization.
    runtime = copy(runtime)
    runtime.quant_config = None
    loader = ContractProbeLoader(runtime.load_config)
    model = loader.load_model(vllm_config=runtime, model_config=model_config)
    class_after, arch_after = get_model_architecture(model_config)
    result = {
        "checkpoint": args.checkpoint,
        "scope": "native config and one-parameter CPU loader API only",
        "checkpoint_weights_loaded": False,
        "v41_model_constructed": False,
        "quantization_for_probe": None,
        "identity_before": original_identity,
        "identity_after": identity(model_config),
        "registry_class_before": native_class.__module__ + "." + native_class.__name__,
        "registry_class_after": class_after.__module__ + "." + class_after.__name__,
        "registry_unchanged": class_after is native_class and arch_after == native_arch,
        "class_injected": isinstance(model, LoaderContractProbe),
        "eval_applied": not model.training,
        "weight_device": str(model.weight.device),
        "weight_shape": list(model.weight.shape),
        "weight_finite": bool(torch.isfinite(model.weight).all()),
        "loader_signature": str(inspect.signature(BaseModelLoader.create_model)),
        "injection_signature": str(inspect.signature(initialize_model)),
    }
    print("LOADER_CONTRACT_RESULT " + json.dumps(result))
    assert result["identity_before"] == result["identity_after"]
    assert result["registry_class_before"] == EXPECTED_NATIVE_CLASS
    assert result["registry_unchanged"]
    assert result["class_injected"] and result["eval_applied"]
    assert result["weight_device"] == "cpu" and result["weight_finite"]


if __name__ == "__main__":
    main()
