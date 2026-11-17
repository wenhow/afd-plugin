# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Small model/config boundary for native loader API tests on CPU."""

from copy import copy
from types import SimpleNamespace

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import CompilationConfig, CompilationMode
from vllm.config.compilation import CUDAGraphMode
from vllm.config.device import DeviceConfig
from vllm.config.load import LoadConfig

from tests.unit.fakes.mtp import ordinary_mtp_config


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


class MTPModel(NativeModel):
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
        speculative_config=None,
        additional_config={},
    )
    if role is not None:
        config.additional_config = {
            "afd": {"role": role, "connector": "P2pHcclAFDConnector"}
        }
    return config


def dspark_draft_model_config(target_model_config):
    """Native DSpark supplies this separately from the ambient target config."""
    draft = copy(target_model_config)
    draft.hf_config = PretrainedConfig(
        architectures=["DeepseekV41DSparkModel"], tie_word_embeddings=False
    )
    draft.hf_config.model_type = "deepseek_v41"
    draft.hf_text_config = draft.hf_config
    return draft


def mtp_loader_config(*, steps=2, role="attention"):
    native = loader_config(model_type="deepseek_v4", role=role)
    config = ordinary_mtp_config(steps=steps, role=role)
    for field, value in vars(config.model_config).items():
        if field not in ("hf_config", "hf_text_config"):
            setattr(native.model_config, field, value)
    for field, value in vars(config.model_config.hf_config).items():
        setattr(native.model_config.hf_config, field, value)
    config.model_config = native.model_config
    config.compilation_config = native.compilation_config
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    config.load_config = native.load_config
    config.device_config = native.device_config
    draft = copy(native.model_config)
    draft.hf_config = PretrainedConfig(
        **vars(config.speculative_config.draft_model_config.hf_config)
    )
    draft.hf_config.model_type = "deepseek_mtp"
    draft.hf_text_config = draft.hf_config
    config.speculative_config.draft_model_config = draft
    return config
