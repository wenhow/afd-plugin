# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Keep native model identity while constructing V4.1 Attention/FFN roles."""

from torch import nn
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.model_loader.base_loader import (
    BaseModelLoader,
    log_model_inspection,
    log_online_quantization,
)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.utils import initialize_model

from afd_plugin.compat.npu.runtime_config import is_deepseek_v41_model
from afd_plugin.config import HCCL_P2P_CONNECTOR, parse_optional_afd_config


# Upstream: vllm/model_executor/model_loader/base_loader.py,
# BaseModelLoader.create_model @ ced6857afa0ea7b2e3f0846a62e1394e90f15607.
# Patch reason: native class resolution cannot select AFD roles without changing
# ModelConfig identity; initialize_model already supports explicit class input.
# Patch functionality: select the guarded V4.1 NPU role class at construction.
# Quantized checkpoints require the explicit DS FP4-expert quantization class;
# native V4.1 auto selection currently resolves to generic FP8 on this lock.
# Signature: matches upstream; no added parameters. Native load_model is retained.
# Removal plan: use native role selection and remove the quantization check
# after native V4.1 auto detection selects its checkpoint's expert format.
def create_model(
    self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
) -> nn.Module:
    """Create a model with the given configurations."""
    # ### PATCH START: V4.1 role construction class
    model_class = None
    if vllm_config.device_config.device_type == "npu" and is_deepseek_v41_model(
        vllm_config
    ):
        afd_config = parse_optional_afd_config(vllm_config, validate=False)
        if afd_config is not None:
            if (
                vllm_config.quant_config is not None
                and vllm_config.quant_config.get_name() != "deepseek_v4_fp8"
            ):
                raise ValueError(
                    "V4.1 quantized role loading requires "
                    "--quantization deepseek_v4_fp8; generic FP8 does not "
                    "describe the checkpoint's FP4 experts"
                )
            if afd_config.role not in ("attention", "ffn"):
                raise RuntimeError("V4.1 role loading requires attention or ffn role")
            if afd_config.connector != HCCL_P2P_CONNECTOR:
                raise ValueError("V4.1 role loading requires P2pHcclAFDConnector")
            if not (
                type(self) is DefaultModelLoader
                and self.load_config.load_format in ("auto", "safetensors")
                or type(self) is DummyModelLoader
                and self.load_config.load_format == "dummy"
            ):
                raise ValueError(
                    "V4.1 role loading requires native default(auto/safetensors) "
                    "or dummy loader"
                )
            # Worker/backend isolation: import the NPU model only for this role.
            from afd_plugin.model_executor.models.npu.deepseek_v41 import (
                AFDDeepseekV41AttentionForCausalLM,
                AFDDeepseekV41FFNForCausalLM,
            )

            model_class = (
                AFDDeepseekV41AttentionForCausalLM
                if afd_config.role == "attention"
                else AFDDeepseekV41FFNForCausalLM
            )
    # ### PATCH END: V4.1 role construction class
    model = initialize_model(
        vllm_config=vllm_config,
        model_config=model_config,
        prefix=prefix,
        # ### PATCH START: preserve native identity
        model_class=model_class,
        # ### PATCH END: preserve native identity
    )
    log_online_quantization(vllm_config)
    log_model_inspection(model)
    return model


def apply_afd_model_loader_patch() -> None:
    """Install the role constructor after Ascend worker initialization."""
    BaseModelLoader.create_model = create_model
