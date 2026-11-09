# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small model/config boundary for native loader API tests on CPU."""

from types import SimpleNamespace

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import CompilationConfig, CompilationMode
from vllm.config.device import DeviceConfig
from vllm.config.load import LoadConfig


class NativeModel(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1))
        self.prefix = prefix
        self.vllm_config = vllm_config


class FFNModel(NativeModel):
    pass


class AttentionModel(NativeModel):
    pass


def loader_config(*, model_type="deepseek_v41", device="npu", role="ffn"):
    hf_config = PretrainedConfig(
        architectures=["DeepseekV41ForCausalLM"], tie_word_embeddings=False
    )
    hf_config.model_type = model_type
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=hf_config,
            dtype=torch.float32,
            model="loader-api-only",
            word_embeddings_untied_by_checkpoint=False,
            quantization=None,
        ),
        compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        load_config=LoadConfig(load_format="dummy", device="cpu"),
        device_config=DeviceConfig(device=device),
        quant_config=None,
        additional_config={},
    )
    if role is not None:
        config.additional_config = {
            "afd": {"role": role, "connector": "P2pHcclAFDConnector"}
        }
    return config
