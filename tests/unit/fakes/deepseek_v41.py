# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from types import SimpleNamespace

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.models.deepseek_v41.model import (
    DeepseekV41Model,
    DeepseekV41SharedAttentionState,
)

from afd_plugin.connectors import AFDA2FTransferPayload
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_plan import HcclParentPlan
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    AFDDeepseekV41AttentionDecoderLayer,
    AFDDeepseekV41AttentionModel,
)
from afd_plugin.model_executor.models.npu.deepseek_v41_staged import (
    DeepseekV41EngramInputs,
    DeepseekV41StageInputs,
)

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
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
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


def make_shared_attention_model(*, capacity=6, layer_count=3, with_lengths=True):
    """Native state/tensors on a CPU model namespace; no device construction."""
    model = DeepseekV41Model.__new__(DeepseekV41Model)
    nn.Module.__init__(model)
    state = DeepseekV41SharedAttentionState(
        torch.full((capacity, 3), -1, dtype=torch.int32),
        torch.full((capacity, 1, 2), -1, dtype=torch.int32),
        torch.zeros((capacity, 1), dtype=torch.int32) if with_lengths else None,
        torch.zeros((capacity, 1), dtype=torch.int32) if with_lengths else None,
    )
    model.shared_attention_state = state
    model.topk_indices_buffer = state.topk_indices
    model.candidate_indices_buffer = state.candidates
    model.candidate_lengths_buffer = state.candidate_lengths
    model.topk_lengths_buffer = state.topk_lengths
    layers = []
    for _ in range(layer_count):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.shared_state = state
        layers.append(layer)
    model.layers = nn.ModuleList(layers)
    return model


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


def contract_ffn(hidden_states, hidden_states_fp32, input_ids):
    """Observable independent-FP32/raw-ID seam; no device MoE implementation."""
    return (
        (hidden_states_fp32 - hidden_states.float()) * ROUTER_DELTA_SCALE
        + input_ids[:, None]
    ).to(hidden_states.dtype)


class StageAttention(nn.Module):
    """An index source and consumer over the native shared state namespace."""

    def __init__(self, layer_idx, state):
        super().__init__()
        self.layer_idx = layer_idx
        self.shared_state = state

    def forward(self, positions, hidden_states, scaling=None):
        if not positions.numel():
            raise AssertionError("zero-row Attention must be skipped")
        rows = hidden_states.shape[0]
        if self.layer_idx == 0:
            self.shared_state.topk_indices[:rows, 0].copy_(positions)
        indices = self.shared_state.topk_indices[:rows, 0].to(hidden_states.dtype)
        return hidden_states + indices[:, None]


class StageEngram(nn.Module):
    def forward(self, hidden_states, lookup, mask, rotation):
        return hidden_states + lookup[:, None, :] * mask[:, None, None]


class ContractFFN(nn.Module):
    def forward(
        self,
        hidden_states,
        input_ids=None,
        hidden_states_fp32=None,
        already_sequence_parallel=False,
    ):
        assert not already_sequence_parallel
        return contract_ffn(hidden_states, hidden_states_fp32, input_ids)


class HCContractLayer(AFDDeepseekV41AttentionDecoderLayer):
    """Native forward plus real continuation APIs over explicit CPU arithmetic.

    This fake verifies scheduling/state semantics, not HC or Attention numerics.
    """

    def __init__(self, layer_idx, state, *, engram=True):
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.hc_mult = 2
        self.use_sequence_parallel = False
        self.self_attn = StageAttention(layer_idx, state)
        self.input_layernorm = nn.Identity()
        self.mlp = ContractFFN()
        self.engram = StageEngram() if engram else None
        self.hc_attn_fn = 1.0
        self.hc_ffn_fn = 2.0
        self.hc_attn_scale = self.hc_attn_base = 0.0
        self.hc_ffn_scale = self.hc_ffn_base = 0.0

    def hc_pre(self, hidden_states, fn, scale, base, pre_mix=None):
        if not hidden_states.shape[0]:
            raise AssertionError("zero-row HC must be skipped")
        x = hidden_states.float().mean(dim=1) + pre_mix.sum(dim=1, keepdim=True)
        pre = pre_mix + (self.layer_idx + fn) / 8
        post = pre + 0.25
        comb = pre.unsqueeze(-1).expand(-1, -1, self.hc_mult) / 4
        return x.to(hidden_states.dtype), post, comb, pre

    def hc_post(self, x, residual, post, comb):
        return (
            residual.float() / 2
            + x.float().unsqueeze(1) * post.unsqueeze(-1)
            + comb.sum(dim=-1).unsqueeze(-1)
        ).to(residual.dtype)

    def rms_norm_cast(self, hidden_states):
        return hidden_states, hidden_states.float() + 0.00031


def make_staged_model(*, capacity=6, engram=True):
    """Use native model/decoder forward bodies without loading weights/operators."""
    parent = make_shared_attention_model(capacity=capacity, layer_count=0)
    model = AFDDeepseekV41AttentionModel.__new__(AFDDeepseekV41AttentionModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=4)
    model.hc_mult = 2
    model.has_engram = engram
    model.use_sequence_parallel = False
    model.shared_attention_state = parent.shared_attention_state
    model.start_layer, model.end_layer = 0, 2
    model.layers = nn.ModuleList(
        HCContractLayer(i, model.shared_attention_state, engram=engram)
        for i in range(2)
    )
    model.norm = nn.Identity()
    model.embed_tokens = nn.Embedding.from_pretrained(
        torch.arange(128).reshape(32, 4).to(torch.bfloat16), freeze=True
    )
    model.aux_hidden_state_layers = (1, 2)
    model.engram_rotated = False
    model._mtp_hidden_buffer = torch.empty(capacity, 8, dtype=torch.bfloat16)
    return model


def make_staged_case(split=(2, 3), *, valid=None, engram=True, remote_active=False):
    """One native parent and typed children for execution/worker seam tests."""
    rows = sum(split)
    valid = split if valid is None else valid
    model = make_staged_model(capacity=max(rows, 1), engram=engram)
    ids = torch.arange(1, rows + 1, dtype=torch.int32)
    positions = torch.arange(rows, dtype=torch.int64) + 10
    lookups = (
        {
            layer: (torch.arange(rows * 4).reshape(rows, 4) / 8 + layer).bfloat16()
            for layer in range(2)
        }
        if engram
        else {}
    )
    mask = (
        torch.ones(rows, dtype=torch.bool)
        if engram
        else torch.empty(0, dtype=torch.bool)
    )
    slices, stages, contexts, layouts = [], [], [], []
    offset = 0
    parent_engram = DeepseekV41EngramInputs(lookups, mask)
    for i, (padded, count) in enumerate(zip(split, valid, strict=True)):
        token_slice = slice(offset, offset + padded)
        ids[offset + count : offset + padded] = 0
        if engram:
            mask[offset + count : offset + padded] = False
        slices.append(token_slice)
        stages.append(
            DeepseekV41StageInputs(
                ids[token_slice],
                positions[token_slice],
                parent_engram.slice(token_slice),
            )
        )
        contexts.append(
            SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE, ubatch_idx=i)
        )
        layouts.append(
            HcclStageLayout(
                i,
                (count, 1) if remote_active else (count,),
                (padded, 1) if remote_active else (padded,),
                1,
            )
        )
        offset += padded
    return SimpleNamespace(
        model=model,
        stages=tuple(stages),
        contexts=tuple(contexts),
        slices=tuple(slices),
        plan=HcclParentPlan("staged-case", (0, 1), tuple(layouts)),
        parent_inputs={
            "input_ids": ids,
            "positions": positions,
            "intermediate_tensors": None,
            "engram_lookups": lookups,
            "engram_mask": mask,
        },
        mtp_rows=model._mtp_hidden_buffer[:rows],
        wrapper=SimpleNamespace(language_model=SimpleNamespace(model=model)),
    )


class ParentEngramState(EngramModelState):
    """A one-shot parent producer seam; no history or device lookup simulation."""

    def __init__(self, producer_inputs, parent_context=None):
        self._cg_mode = CUDAGraphMode.NONE
        self.producer_inputs = producer_inputs
        self.parent_context = parent_context
        self.consumed = False

    def prepare_engram(self):
        if self.consumed:
            raise AssertionError("parent producer already consumed")
        if (
            self.parent_context is not None
            and get_forward_context() is not self.parent_context
        ):
            raise AssertionError("producer must use the parent forward context")
        self.consumed = True
        return self.producer_inputs

    def prepare_inputs(self, *args, **kwargs):
        return {"producer_mode": self._cg_mode}

    def finish_execution(self, *, failed):
        raise AssertionError("native execute_model owns parent retirement")
