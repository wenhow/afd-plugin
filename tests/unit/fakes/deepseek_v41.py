# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from collections.abc import Mapping
from types import MethodType, SimpleNamespace

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm_ascend.models.deepseek_v41.dspark import DeepseekV41DSparkDecoderLayer
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.models.deepseek_v41.model import (
    DeepseekV41Model,
    DeepseekV41SharedAttentionState,
    build_layer_plan,
)

from afd_plugin.compat.npu.kv_layout import KVLayoutSnapshot
from afd_plugin.connectors import AFDA2FTransferPayload
from afd_plugin.connectors.npu.hccl_diagnostics import (
    HcclDiagnosticExecution,
    HcclStageDiagnostics,
)
from afd_plugin.connectors.npu.hccl_overlap import (
    HcclOverlapSettings,
    select_hccl_overlap,
)
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_plan import HcclParentPlan
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    AFDDeepseekV41AttentionDecoderLayer,
    AFDDeepseekV41AttentionForCausalLM,
    AFDDeepseekV41AttentionLLMForCausalLM,
    AFDDeepseekV41AttentionModel,
)
from afd_plugin.model_executor.models.npu.deepseek_v41_staged import (
    DeepseekV41EngramInputs,
    DeepseekV41StageInputs,
)
from tests.unit.fakes.hccl_diagnostics import StageDiagnosticRecorder

ROUTER_DELTA_SCALE = 4_096
PD_CPU_VOCAB_SIZE = 128
PD_CPU_TARGET_LAYERS = 3
PD_CPU_ATTENTION_SCALE = 16
PD_CPU_TARGET_TOKEN = 7
PD_CPU_HIDDEN_SIZE = 4
PD_CPU_HC_STREAMS = 2
PD_CPU_CACHE_WIDTH = 2
PD_CPU_TARGET_WINDOW = 4
PD_CPU_MIX_INCREMENT = 1 / 8
PD_CPU_POST_OFFSET = 1 / 4
PD_CPU_RESIDUAL_SCALE = 1 / 2
PD_CPU_ROUTER_OFFSET = 0.00031


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


def make_staged_model(*, capacity=6, engram=True, layer_count=2):
    """Use native model/decoder forward bodies without loading weights/operators."""
    parent = make_shared_attention_model(capacity=capacity, layer_count=0)
    model = AFDDeepseekV41AttentionModel.__new__(AFDDeepseekV41AttentionModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=4)
    model.hc_mult = 2
    model.has_engram = engram
    model.use_sequence_parallel = False
    model.shared_attention_state = parent.shared_attention_state
    model.start_layer, model.end_layer = 0, layer_count
    model.layers = nn.ModuleList(
        HCContractLayer(i, model.shared_attention_state, engram=engram)
        for i in range(layer_count)
    )
    model.norm = nn.Identity()
    model.embed_tokens = nn.Embedding.from_pretrained(
        torch.arange(128).reshape(32, 4).to(torch.bfloat16), freeze=True
    )
    model.aux_hidden_state_layers = tuple(range(1, layer_count + 1))
    model.engram_rotated = False
    model._mtp_hidden_buffer = torch.empty(capacity, 8, dtype=torch.bfloat16)
    return model


class PDCacheAttention(nn.Module):
    """CPU Attention operator over the real registered SWA view and block IDs.

    This is finite prefix-read/write arithmetic, not DSA/softmax/compression.
    Its metadata is supplied at the native preparation boundary. It never
    substitutes a receiver map for the runner's absolute block coordinates.
    """

    def __init__(self, cache_layer, group_id, window, *, contexts=None):
        super().__init__()
        self.dsa_attn = nn.Module()
        self.dsa_attn.swa_cache_layer = cache_layer
        self.group_id, self.window = group_id, window
        self.contexts = contexts
        self.batch = None

    def prepare(
        self,
        *,
        num_reqs,
        query_start_loc_gpu,
        positions,
        seq_lens,
        is_prefilling,
        block_tables,
    ):
        query = query_start_loc_gpu[: num_reqs + 1]
        counts = (query[1:] - query[:-1]).tolist()
        prefix = [int(seq_lens[row]) - count for row, count in enumerate(counts)]
        if is_prefilling.numel() < num_reqs:
            raise ValueError("CPU metadata needs the native prefill classifications")
        for row, start in enumerate(query[:-1].tolist()):
            if int(positions[start]) != prefix[row]:
                raise ValueError("CPU metadata needs native contiguous query positions")
        self.batch = SimpleNamespace(
            query_start_loc=query,
            prefix=prefix,
            # Native Attention writes every physical query. Rejection later
            # changes the accepted frontier; metadata cannot infer its result.
            valid=counts,
            block_table=block_tables[self.group_id],
        )

    def prepare_query(self, buffers, num_reqs, width, gathered_tables):
        self.batch = SimpleNamespace(
            query_start_loc=buffers.query_start_loc[: num_reqs + 1],
            prefix=[int(buffers.positions[row * width]) for row in range(num_reqs)],
            valid=[width] * num_reqs,
            block_table=gathered_tables[self.group_id],
        )

    def forward(self, positions, hidden_states, scaling=None):
        if self.batch is None:
            raise RuntimeError("CPU PD Attention needs actual prepared coordinates")
        if self.contexts is not None:
            self.contexts.append(get_forward_context())
        cache_layer = self.dsa_attn.swa_cache_layer
        cache, block_size = cache_layer.kv_cache[0], cache_layer.block_size
        output = hidden_states.clone()
        for request, (start, stop) in enumerate(
            zip(
                self.batch.query_start_loc[:-1].tolist(),
                self.batch.query_start_loc[1:].tolist(),
                strict=True,
            )
        ):
            prefix = self.batch.prefix[request]
            table = self.batch.block_table[request]
            for row in range(start, stop):
                position = int(positions[row])
                context_sum = hidden_states.new_zeros((), dtype=torch.float32)
                for prior in range(
                    max(0, position - self.window + 1), min(position, prefix)
                ):
                    block = int(table[prior // block_size])
                    if block == 0:
                        raise ValueError("live PD prefix maps to the null block")
                    context_sum += cache[block, prior % block_size].float().sum()
                output[row] += (context_sum / PD_CPU_ATTENTION_SCALE).to(output.dtype)
                if row - start < self.batch.valid[request]:
                    block = int(table[position // block_size])
                    if block == 0:
                        raise ValueError("live PD query maps to the null block")
                    values = hidden_states[row, : cache.shape[-1]]
                    cache[block, position % block_size].copy_(
                        values.reshape_as(cache[block, position % block_size])
                    )
        return output


class CPUVocabProjection(nn.Module):
    """Unquantized CPU head leaf; native model compute_logits stays intact."""

    def forward(self, head, hidden_states):
        return nn.functional.linear(hidden_states.float(), head.weight.float())


def complete_pd_cpu_models(config, kv_config, draft_model, target_cache_layers):
    """Complete both native models before prefix-consumer construction/binding.

    Native V4.1 target/draft forward and decoder delayed-HC control remain real.
    Attention, HC operators, MoE, normalization and unquantized head arithmetic
    are declared CPU leaves, not checkpoint/device implementations.
    """
    text = config.model_config.hf_text_config
    draft_text = config.speculative_config.draft_model_config.hf_text_config
    if (
        text.num_hidden_layers != PD_CPU_TARGET_LAYERS
        or text.dspark_target_layer_ids != [1, 2, 3]
    ):
        raise ValueError("complete CPU PD model requires coherent three-layer metadata")
    topology = build_layer_plan(text)
    layer_groups = {
        name: index
        for index, group in enumerate(kv_config.kv_cache_groups)
        for name in group.layer_names
    }
    target = make_staged_model(
        capacity=config.scheduler_config.max_num_batched_tokens,
        engram=False,
        layer_count=text.num_hidden_layers,
    )
    target.config = text
    target.topology = topology
    target.engram_hash = None
    target.engram_rotation = torch.empty(0)
    target.embed_tokens = nn.Embedding.from_pretrained(
        torch.arange(PD_CPU_VOCAB_SIZE * text.hidden_size)
        .reshape(PD_CPU_VOCAB_SIZE, text.hidden_size)
        .to(torch.bfloat16),
        freeze=True,
    )
    for index, layer in enumerate(target.layers):
        name = f"model.layers.{index}.self_attn.swa_cache"
        cache_layer = target_cache_layers[name]
        layer.self_attn = PDCacheAttention(
            cache_layer, layer_groups[name], text.sliding_window
        )
        layer.self_attn.role = topology.layer(index)
        layer.self_attn.shared_state = target.shared_attention_state
    language = AFDDeepseekV41AttentionLLMForCausalLM.__new__(
        AFDDeepseekV41AttentionLLMForCausalLM
    )
    nn.Module.__init__(language)
    language.config, language.model = text, target
    language.lm_head = nn.Linear(text.hidden_size, PD_CPU_VOCAB_SIZE, bias=False)
    language.lm_head.weight.data.zero_()
    language.lm_head.weight.data[PD_CPU_TARGET_TOKEN, 0] = 1
    language.logits_processor = CPUVocabProjection()
    outer = AFDDeepseekV41AttentionForCausalLM.__new__(
        AFDDeepseekV41AttentionForCausalLM
    )
    nn.Module.__init__(outer)
    outer.language_model = language

    draft = draft_model.model
    draft.config, draft.hc_mult = draft_text, text.hc_mult
    draft.hidden_size, draft.block_size = text.hidden_size, text.dspark_block_size
    draft.target_layer_ids = list(text.dspark_target_layer_ids)
    draft.num_dspark_layers = draft_text.n_mtp_layers
    draft.mtp_start_layer_idx = text.num_hidden_layers
    draft.use_sequence_parallel_moe = False
    draft.embed_tokens = nn.Embedding.from_pretrained(
        target.embed_tokens.weight.clone(), freeze=True
    )
    draft.norm = nn.Identity()
    contexts, blocks = [], {}
    for ordinal, old_layer in enumerate(draft.layers.values()):
        attn = old_layer.self_attn
        cache_layer = attn.dsa_attn.swa_cache_layer
        group_id = layer_groups[cache_layer.prefix]
        window = kv_config.kv_cache_groups[group_id].kv_cache_spec.sliding_window
        replacement = PDCacheAttention(cache_layer, group_id, window, contexts=contexts)
        replacement.wkv = attn.wkv
        replacement.kv_norm = attn.kv_norm
        replacement.rotary_emb = attn.rotary_emb
        replacement.nope_head_dim = attn.nope_head_dim
        replacement.rope_head_dim = attn.rope_head_dim
        replacement.head_dim = attn.head_dim
        replacement.packed_cache_ops = attn.packed_cache_ops
        layer = DeepseekV41DSparkDecoderLayer.__new__(DeepseekV41DSparkDecoderLayer)
        nn.Module.__init__(layer)
        arithmetic = HCContractLayer(
            text.num_hidden_layers + ordinal,
            target.shared_attention_state,
            engram=False,
        )
        vars(layer).update(vars(arithmetic))
        layer.self_attn = replacement
        layer.hc_pre = MethodType(HCContractLayer.hc_pre, layer)
        layer.hc_post = MethodType(HCContractLayer.hc_post, layer)
        layer.rms_norm_cast = MethodType(HCContractLayer.rms_norm_cast, layer)
        blocks[str(text.num_hidden_layers + ordinal)] = layer
    draft.layers = nn.ModuleDict(blocks)
    draft_model.config = draft_text
    draft_model.lm_head = nn.Linear(text.hidden_size, PD_CPU_VOCAB_SIZE, bias=False)
    draft_model.lm_head.weight.data.zero_()
    draft_model.logits_processor = CPUVocabProjection()
    markov = DSparkMarkovHead.__new__(DSparkMarkovHead)
    nn.Module.__init__(markov)
    vocabulary = torch.arange(PD_CPU_VOCAB_SIZE).float()
    markov.markov_w1 = nn.Embedding.from_pretrained(
        torch.stack(
            ((vocabulary + 1) % PD_CPU_VOCAB_SIZE, torch.ones_like(vocabulary)), dim=1
        ),
        freeze=True,
    )
    markov.markov_w2 = nn.Linear(2, PD_CPU_VOCAB_SIZE, bias=False)
    markov.markov_w2.weight.data.copy_(
        torch.stack((2 * vocabulary, -vocabulary.square()), dim=1)
    )
    draft.markov_head = markov

    def prepare_target_attention(
        *,
        num_reqs,
        query_start_loc_gpu,
        positions,
        seq_lens,
        is_prefilling,
        block_tables,
    ):
        """Consume native run_build operands without wrapping prepare_attn."""
        for layer in target.layers:
            layer.self_attn.prepare(
                num_reqs=num_reqs,
                query_start_loc_gpu=query_start_loc_gpu,
                positions=positions,
                seq_lens=seq_lens,
                is_prefilling=is_prefilling,
                block_tables=block_tables,
            )
        metadata = SimpleNamespace(
            num_reqs=num_reqs,
            num_tokens=int(query_start_loc_gpu[num_reqs]),
            query_start_loc=query_start_loc_gpu[: num_reqs + 1],
            positions=positions,
            seq_lens=seq_lens,
            is_prefilling=is_prefilling,
            block_tables=block_tables,
        )
        return {
            f"model.layers.{index}.self_attn.v41_attn": metadata
            for index in range(len(target.layers))
        }

    def prepare_draft_attention(buffers, num_reqs, width, gathered_tables):
        for layer in draft.layers.values():
            layer.self_attn.prepare_query(buffers, num_reqs, width, gathered_tables)

    return SimpleNamespace(
        target_model=outer,
        draft_model=draft_model,
        draft_contexts=contexts,
        topology=topology,
        prepare_target_attention=prepare_target_attention,
        prepare_draft_attention=prepare_draft_attention,
    )


def native_pd_target_reference(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    computed: int,
    block_ids: tuple[tuple[int, ...], ...],
    layout: KVLayoutSnapshot,
    pages: tuple[torch.Tensor, ...],
    caches: Mapping[str, torch.Tensor | tuple[torch.Tensor, ...]],
) -> SimpleNamespace:
    """Independent single-request oracle for the three-layer CPU PD target.

    Inputs are the active native prepared rows, absolute per-group block IDs
    (including retired null entries), and D's pre-execute byte-page snapshot.
    H4/HC2/window4 and the allocated embedding/HC/Attention/FFN arithmetic are
    explicit CPU contracts. This never calls a model, forward or operator fake.
    Rebuilt SWA views alias cloned pages, retaining every unrelated byte/gap.
    """
    if (
        input_ids.device.type != "cpu"
        or positions.device.type != "cpu"
        or input_ids.ndim != 1
        or input_ids.shape != positions.shape
        or not input_ids.numel()
        or type(computed) is not int
        or computed < 0
        or not torch.equal(
            positions, torch.arange(computed, computed + input_ids.numel())
        )
        or torch.any((input_ids < 0) | (input_ids >= PD_CPU_VOCAB_SIZE))
    ):
        raise ValueError("target reference requires one contiguous CPU query")
    if len(block_ids) != len(layout.groups) or len(pages) != len(layout.pages):
        raise ValueError("target reference requires D's complete grouped layout")
    if len({page.storage_alias for page in layout.pages}) != len(layout.pages):
        raise ValueError("CPU PD fixture requires independently allocated pages")
    if any(
        page.device.type != "cpu"
        or page.dtype != torch.uint8
        or page.ndim != 1
        or not page.is_contiguous()
        for page in pages
    ):
        raise ValueError("target reference snapshots must be contiguous CPU bytes")
    expected_pages = tuple(page.clone() for page in pages)
    layer_groups = {
        name: index
        for index, group in enumerate(layout.groups)
        for name in group.layer_names
    }
    descriptors = {(view.layer_name, view.tensor_index): view for view in layout.views}
    embedding = (
        input_ids[:, None] * PD_CPU_HIDDEN_SIZE + torch.arange(PD_CPU_HIDDEN_SIZE)
    ).to(torch.bfloat16)
    states = embedding[:, None, :].repeat(1, PD_CPU_HC_STREAMS, 1)
    mix = torch.zeros(input_ids.numel(), PD_CPU_HC_STREAMS, dtype=torch.float32)
    mix[:, 0] = 1
    auxiliary = []
    for layer in range(PD_CPU_TARGET_LAYERS):
        name = f"model.layers.{layer}.self_attn.swa_cache"
        original = caches[name]
        descriptor = descriptors[(name, 0)]
        if (
            not isinstance(original, torch.Tensor)
            or original.dtype != torch.bfloat16
            or original.device.type != "cpu"
            or original.ndim != 4
            or original.shape[-2:] != (1, PD_CPU_CACHE_WIDTH)
            or tuple(original.shape) != descriptor.shape
            or tuple(s * original.element_size() for s in original.stride())
            != descriptor.strides_bytes
        ):
            raise ValueError("target reference needs its registered BF16 SWA view")
        span = (
            descriptor.page_offset_bytes
            + original.element_size()
            + sum(
                (size - 1) * stride
                for size, stride in zip(
                    original.shape, descriptor.strides_bytes, strict=True
                )
            )
        )
        page = expected_pages[descriptor.page_index]
        if span > page.numel():
            raise ValueError("target reference SWA view exceeds the byte snapshot")
        cache = original.new_empty(0).set_(
            page.untyped_storage(),
            descriptor.page_offset_bytes // original.element_size(),
            original.shape,
            original.stride(),
        )
        table = block_ids[layer_groups[name]]
        block_size = cache.shape[1]
        auxiliary.append(states.float().mean(dim=1).to(torch.bfloat16))
        attention_input = (
            states.float().mean(dim=1) + mix.sum(dim=1, keepdim=True)
        ).to(torch.bfloat16)
        attention_mix = mix + (layer + 1) * PD_CPU_MIX_INCREMENT
        attention_output = attention_input.clone()
        for row, position in enumerate(positions.tolist()):
            context = torch.zeros((), dtype=torch.float32)
            for prior in range(
                max(0, position - PD_CPU_TARGET_WINDOW + 1), min(position, computed)
            ):
                block = table[prior // block_size]
                if not 0 < block < cache.shape[0]:
                    raise ValueError("target reference prefix has no live block")
                context += cache[block, prior % block_size].float().sum()
            attention_output[row] = (
                attention_input[row].float()
                + (context / PD_CPU_ATTENTION_SCALE).to(torch.bfloat16).float()
            ).to(torch.bfloat16)
            block = table[position // block_size]
            if not 0 < block < cache.shape[0]:
                raise ValueError("target reference query has no live block")
            cache[block, position % block_size, 0].copy_(
                attention_input[row, :PD_CPU_CACHE_WIDTH]
            )
        after_attention = (
            states.float() * PD_CPU_RESIDUAL_SCALE
            + attention_output.float()[:, None, :]
            * (attention_mix + PD_CPU_POST_OFFSET)[:, :, None]
            + (attention_mix * PD_CPU_RESIDUAL_SCALE)[:, :, None]
        ).to(torch.bfloat16)
        ffn_input = (
            after_attention.float().mean(dim=1) + attention_mix.sum(dim=1, keepdim=True)
        ).to(torch.bfloat16)
        ffn_mix = attention_mix + (layer + 2) * PD_CPU_MIX_INCREMENT
        # Keep the actual FP32 add/subtract rounding; this is not a constant
        # ROUTER_OFFSET*DELTA_SCALE once the HC state grows in magnitude.
        router = ffn_input.float()
        delta = (router + PD_CPU_ROUTER_OFFSET) - router
        ffn_output = (delta * ROUTER_DELTA_SCALE + input_ids[:, None]).to(
            torch.bfloat16
        )
        states = (
            after_attention.float() * PD_CPU_RESIDUAL_SCALE
            + ffn_output.float()[:, None, :]
            * (ffn_mix + PD_CPU_POST_OFFSET)[:, :, None]
            + (ffn_mix * PD_CPU_RESIDUAL_SCALE)[:, :, None]
        ).to(torch.bfloat16)
        mix = ffn_mix
    return SimpleNamespace(
        hidden_states=(mix[:, :, None] * states.float()).sum(dim=1).to(torch.bfloat16),
        aux_hidden_states=tuple(auxiliary),
        mtp_hidden_states=states.flatten(1).clone(),
        pages=expected_pages,
    )


def make_staged_case(
    split=(2, 3), *, valid=None, engram=True, remote_active=False, layer_count=2
):
    """One native parent and typed children for execution/worker seam tests."""
    rows = sum(split)
    valid = split if valid is None else valid
    model = make_staged_model(
        capacity=max(rows, 1), engram=engram, layer_count=layer_count
    )
    ids = torch.arange(1, rows + 1, dtype=torch.int32)
    positions = torch.arange(rows, dtype=torch.int64) + 10
    lookups = (
        {
            layer: (torch.arange(rows * 4).reshape(rows, 4) / 8 + layer).bfloat16()
            for layer in range(layer_count)
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
        plan=HcclParentPlan("staged-case", tuple(range(layer_count)), tuple(layouts)),
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


def make_staged_progress(peer):
    recorder = StageDiagnosticRecorder()
    selection = select_hccl_overlap(
        HcclOverlapSettings(stage_diagnostics=True, ffn_compute_sync_diagnostics=True),
        role="attention",
        phase="eager",
        num_stages=len(peer.plan.stages),
    )
    progress = HcclStageDiagnostics(
        selection,
        world_rank=peer.world_rank,
        role_rank=peer.role_rank,
        emit=recorder.emit,
        synchronize=recorder.synchronize,
    ).bind_execution(
        layer_indices=peer.plan.layer_indices,
        transaction_id=peer.plan.transaction_id,
        actual_execution=HcclDiagnosticExecution(
            "serial", "parent", stream_aliases=(True, True, True)
        ),
    )
    return SimpleNamespace(progress=progress, recorder=recorder)


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
