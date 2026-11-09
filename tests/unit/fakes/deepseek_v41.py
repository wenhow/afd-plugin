# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import torch
from torch import nn
from transformers import PretrainedConfig

from afd_plugin.connectors import AFDA2FTransferPayload

ROUTER_DELTA_SCALE = 4_096


def ffn_config():
    text = PretrainedConfig(
        hidden_size=4,
        num_hidden_layers=2,
        n_routed_experts=2,
        n_shared_experts=1,
        n_group=1,
        num_attention_heads=1,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=text),
        parallel_config=SimpleNamespace(),
        quant_config=None,
        engram_config=None,
        speculative_config=None,
        additional_config={"afd": {"role": "ffn", "connector": "P2pHcclAFDConnector"}},
    )


def attention_config():
    config = ffn_config()
    config.additional_config["afd"]["role"] = "attention"
    config.parallel_config.use_sequence_parallel_moe = False
    config.cache_config = SimpleNamespace()
    text = config.model_config.hf_config
    text.rope_parameters = {"original_max_position_embeddings": 32}
    text.rms_norm_eps = 1e-6
    text.hc_mult = 2
    text.hc_sinkhorn_iters = 3
    text.hc_eps = 1e-6
    return config


class Attention(nn.Module):
    """CPU parameter namespaces only; no Attention device implementation."""

    def __init__(self, *, config, **kwargs):
        super().__init__()
        self.indexer = nn.Module()
        self.indexer.compressor = nn.Module()
        self.indexer.compressor.wgate = nn.Linear(config.hidden_size, 1, bias=False)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps


class AttentionLoaderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config=config)
        self.mlp = nn.Identity()
        self.post_attention_layernorm = LayerNorm(config.hidden_size, 1e-6)
        self.hc_ffn_fn = nn.Parameter(torch.zeros(config.hidden_size))
        self.engram = nn.Module()
        self.engram.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)


def make_attention_loader(model_type, config):
    """Exercise native loading on CPU namespaces, excluding model construction."""

    class AttentionLoaderHarness(model_type):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = config.model_config.hf_config
            self.model = nn.Module()
            self.model.has_engram = True
            self.model.layers = nn.ModuleList([AttentionLoaderLayer(self.config)])
            self.num_moe_layers = self.config.num_hidden_layers
            self.set_moe_parameters()

    return AttentionLoaderHarness()


class RemoteMoEPeer:
    """A public protocol peer, using arithmetic rather than device MoE."""

    def __init__(self):
        self.payloads = []

    def send_attn_output(
        self, hidden_states, context, *, hidden_states_fp32, input_ids
    ):
        self.payloads.append(
            AFDA2FTransferPayload(
                hidden_states=hidden_states.clone(),
                context=context,
                hidden_states_fp32=hidden_states_fp32.clone(),
                input_ids=input_ids.clone(),
            )
        )

    def recv_ffn_output(self, ref_tensor, ubatch_idx=0, *, context):
        payload = self.payloads[-1]
        if context is not payload.context or ubatch_idx != context.metadata.stage_idx:
            raise ValueError("MoE reply requires the original stage and context")
        output = (
            payload.hidden_states_fp32 - payload.hidden_states.float()
        ) * ROUTER_DELTA_SCALE + payload.input_ids[:, None]
        return ref_tensor.copy_(output)


class MoE(nn.Module):
    """CPU API double; its arithmetic is not a native MoE implementation."""

    def __init__(self, config, parallel_config, quant_config=None, prefix=""):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.experts = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.n_logical_experts = self.n_physical_experts = config.n_routed_experts
        self.n_local_physical_experts = self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.n_redundant_experts = 0

    def forward(
        self,
        hidden_states,
        input_ids=None,
        hidden_states_fp32=None,
        already_sequence_parallel=False,
    ):
        return hidden_states_fp32 + input_ids[:, None] + int(already_sequence_parallel)
