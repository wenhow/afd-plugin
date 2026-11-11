# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native loaded DSpark context bodies with CPU RoPE/write/sync leaves.

Actual Ascend speculator group discovery and V4.1 combine/project/store bodies
remain intact. These allocated CPU weights are an independent arithmetic
contract, not checkpoint weights, packed SWA writes or a device transfer.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import ModelConfig
from vllm.config.attention import AttentionConfig
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheTensor
from vllm.v1.worker.gpu.model_runner import ExecuteModelState
from vllm_ascend.core.kv_cache_interface import (
    AscendSlidingWindowMLASpec,
    is_circular_kv_cache_spec,
)
from vllm_ascend.models.deepseek_v41 import dspark as dspark_model_module
from vllm_ascend.models.deepseek_v41.dspark import (
    DeepseekV41DSparkModel,
    DSparkDeepseekV41ForCausalLM,
)
from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41LLMForCausalLM
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator

from afd_plugin.compat.npu.dspark_pd import (
    DsparkPdPrefixConsumer,
    DsparkPdReady,
    DsparkPdSourceRequest,
    draft_schema_digest,
    layout_digest,
    prompt_digest,
)
from afd_plugin.compat.npu.kv_layout_native import collect_deepseek_v41_kv_layout
from tests.unit.fakes.deepseek_v41 import complete_pd_cpu_models
from tests.unit.fakes.kv_cache import make_native_kv_cache_case
from tests.unit.fakes.vllm_config import v41_dspark_pd_config

PD_CPU_HIDDEN = 4
PD_CPU_WIDTH = 2
PD_CPU_BLOCKS = 4
PD_CPU_ORIGINAL_TOKENS = 6
PD_CPU_TARGET_LAYERS = 3
PD_CPU_DRAFT_LAYERS = 2
PD_CPU_VOCAB = 128
PD_CPU_PAGE_PATTERN_DIVISOR = 4
PD_CPU_BLOCK_PATTERN_DIVISOR = 16
PD_CPU_WORD_PATTERN_DIVISOR = 256


class PDCPUModelConfig(SimpleNamespace):
    """Metadata-only microconfig with the actual native embedding-size API.

    Checkpoint/model config construction is outside this CPU seam. The native
    ModelState constructor still consumes its usual methods and fields.
    """

    get_hidden_size = ModelConfig.get_hidden_size
    get_inputs_embeds_size = ModelConfig.get_inputs_embeds_size


class PrefixProjectionStream:
    def __init__(self):
        self.drained = True

    def synchronize(self):
        self.drained = True


class PrefixCacheLayer(nn.Module):
    def __init__(self, name, block_size, tensor):
        super().__init__()
        self.prefix, self.block_size = name, block_size
        self.kv_cache = [tensor]


class PrefixAgreement:
    """Public CPU outcome exchange, with optional explicit peer failure."""

    def __init__(self, tp_size, stream, *, peer_error=False):
        self.tp_size, self.stream = tp_size, stream
        self.peer_error = peer_error
        self.reports = []

    def __call__(self, report):
        self.reports.append(report)
        if report.prefixes and not self.stream.drained:
            raise AssertionError("Ready cannot precede the native context write fence")
        replies = []
        for rank in range(self.tp_size):
            ready = tuple(replace(item, source_tp_rank=rank) for item in report.ready)
            error = (
                "peer projection failed" if self.peer_error and rank else report.error
            )
            replies.append(
                replace(report, source_tp_rank=rank, ready=ready, error=error)
            )
        return tuple(replies)


def _projection_model(kv_case, names, *, fail_projection=False):
    model = DeepseekV41DSparkModel.__new__(DeepseekV41DSparkModel)
    nn.Module.__init__(model)
    model.main_proj = nn.Linear(PD_CPU_HIDDEN * 3, PD_CPU_HIDDEN, bias=False).to(
        torch.bfloat16
    )
    model.main_proj.weight.data.copy_(
        torch.cat([torch.eye(PD_CPU_HIDDEN) * scale for scale in (1, 2, 3)], dim=1)
    )
    model.main_norm = nn.Identity()
    blocks = {}
    for index, (name, block_size) in enumerate(reversed(names)):
        block = nn.Module()
        block.self_attn = nn.Module()
        attn = block.self_attn
        attn.wkv = nn.Linear(PD_CPU_HIDDEN, PD_CPU_WIDTH, bias=False).to(torch.bfloat16)
        attn.wkv.weight.data.copy_(torch.eye(PD_CPU_WIDTH, PD_CPU_HIDDEN))
        attn.kv_norm, attn.rotary_emb = nn.Identity(), nn.Identity()
        attn.nope_head_dim, attn.rope_head_dim = 1, 1
        attn.head_dim = PD_CPU_WIDTH
        attn.packed_cache_ops = None
        attn.dsa_attn = nn.Module()
        attn.dsa_attn.swa_cache_layer = PrefixCacheLayer(
            name, block_size, kv_case.caches[name]
        )
        blocks[str(index)] = block
    model.layers = nn.ModuleDict(blocks)
    outer = DSparkDeepseekV41ForCausalLM.__new__(DSparkDeepseekV41ForCausalLM)
    nn.Module.__init__(outer)
    outer.model = model
    if fail_projection:

        def fail_rope(rotary_emb, positions, x, *, inverse=False):
            raise RuntimeError("controlled projection kernel failure")

        rope = fail_rope
    else:

        def identity_rope(rotary_emb, positions, x, *, inverse=False):
            return x

        rope = identity_rope
    return outer, rope


def make_dspark_pd_loaded_case(
    monkeypatch,
    *,
    config,
    kv_cache_config,
    caches,
    draft_names,
    source_engine_id,
    tp_rank=0,
    tp_size=1,
    peer_error=False,
    projection_error=False,
    complete_cpu_models=False,
):
    """Bind native CPU model identities to caller-owned materialized cache views.

    No constructor loads a checkpoint and no target forward runs here. The draft
    projection model retains native combine/project/store bodies with the same
    explicit CPU RoPE/scatter/fence leaves as the producer prefix fixture.
    Registration remains the actual connector's responsibility.
    """
    kv_case = SimpleNamespace(config=kv_cache_config, caches=caches)
    group_ids = tuple(
        index
        for index, group in enumerate(kv_cache_config.kv_cache_groups)
        if set(group.layer_names) & set(draft_names)
    )
    layer_groups = {
        name: index
        for index in group_ids
        for name in kv_cache_config.kv_cache_groups[index].layer_names
    }
    names = tuple(
        (
            name,
            kv_cache_config.kv_cache_groups[
                layer_groups[name]
            ].kv_cache_spec.block_size,
        )
        for name in draft_names
    )
    model, rope = _projection_model(kv_case, names, fail_projection=projection_error)
    model_bundle = None
    if complete_cpu_models:
        target_cache_layers = {
            name: PrefixCacheLayer(
                name,
                kv_cache_config.kv_cache_groups[group_index].kv_cache_spec.block_size,
                caches[name],
            )
            for group_index, group in enumerate(kv_cache_config.kv_cache_groups)
            if group_index not in group_ids
            for name in group.layer_names
            if name.endswith(".swa_cache")
        }
        model_bundle = complete_pd_cpu_models(
            config, kv_cache_config, model, target_cache_layers
        )
    config.compilation_config.static_forward_context = {}
    for block in model.model.layers.values():
        cache = block.self_attn.dsa_attn.swa_cache_layer
        config.compilation_config.static_forward_context[cache.prefix] = cache
    if model_bundle is not None:
        for name, cache in target_cache_layers.items():
            config.compilation_config.static_forward_context[name] = cache
    speculator = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    speculator.model = model
    speculator.vllm_config = config
    # Native DFlash's read-only attention config derives causality from these
    # actual constructor fields; keep that property intact in this CPU seam.
    speculator.requires_non_causal = True
    speculator.device = torch.device("cpu")
    speculator.draft_attn_layer_names = set(model.get_draft_kv_cache_layer_names())
    speculator.draft_kv_cache_group_ids = group_ids
    speculator._layer_group_idx = [
        group_ids.index(layer_groups[block.self_attn.dsa_attn.swa_cache_layer.prefix])
        for block in model.model.layers.values()
    ]
    speculator.kv_cache_config = kv_cache_config
    speculator.cpu_model_bundle = model_bundle
    stream = PrefixProjectionStream()

    def scatter_cpu(cache, slots, values):
        stream.drained = False
        for row, (block, offset) in enumerate(slots.tolist()):
            if block >= 0:
                cache[block, offset].copy_(values[row].reshape_as(cache[block, offset]))

    monkeypatch.setattr(dspark_model_module, "_apply_dsv4_rope", rope)
    monkeypatch.setattr(dspark_model_module, "scatter_cache_sk", scatter_cpu)
    monkeypatch.setattr(torch.npu, "current_stream", lambda device=None: stream)
    agreement = PrefixAgreement(tp_size, stream, peer_error=peer_error)
    if model_bundle is None:
        target_model = AscendDeepseekV41LLMForCausalLM.__new__(
            AscendDeepseekV41LLMForCausalLM
        )
        nn.Module.__init__(target_model)
    else:
        target_model = model_bundle.target_model
    consumer = DsparkPdPrefixConsumer(
        config,
        speculator,
        target_model=target_model,
        device=speculator.device,
        source_engine_id=source_engine_id,
        tp_rank=tp_rank,
        tp_size=tp_size,
        agree_batch=agreement,
    )
    return SimpleNamespace(
        config=config,
        speculator=speculator,
        model=model,
        target_model=target_model,
        consumer=consumer,
        agreement=agreement,
        kv_config=kv_cache_config,
        caches=caches,
        draft_names=tuple(draft_names),
        draft_group_ids=group_ids,
        model_bundle=model_bundle,
        complete_cpu_models=complete_cpu_models,
    )


def _configure_complete_cpu_models(config):
    """Choose coherent micro metadata before either schema or consumer exists."""
    target = PretrainedConfig(**vars(config.model_config.hf_text_config))
    values = dict(
        hidden_size=PD_CPU_HIDDEN,
        vocab_size=PD_CPU_VOCAB,
        num_hidden_layers=PD_CPU_TARGET_LAYERS,
        num_nextn_predict_layers=PD_CPU_DRAFT_LAYERS,
        dspark_target_layer_ids=[1, 2, 3],
        dspark_noise_token_id=0,
        compress_ratios=[2, 2, 2],
        kv_source_layer_ids=[0],
        index_source_layer_ids=[0],
        candidate_source_layer_id=0,
        candidate_topk_blocks=1,
        candidate_block_size=4,
        index_topk=2,
        sliding_window=4,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        engram_layer_ids=[],
        rswa_window=None,
    )
    vars(target).update(values)
    model_config = PDCPUModelConfig(**vars(config.model_config))
    model_config.hf_config = model_config.hf_text_config = target
    model_config.dtype = torch.bfloat16
    model_config.runner_type = "generate"
    model_config.rswa_window = None
    model_config.model_arch_config = SimpleNamespace(hidden_size=PD_CPU_HIDDEN)
    model_config.enable_prompt_embeds = False
    model_config.uses_mrope = False
    config.model_config = model_config
    draft_config = PDCPUModelConfig(
        **vars(config.speculative_config.draft_model_config)
    )
    draft = PretrainedConfig(**vars(draft_config.hf_text_config))
    vars(draft).update(values)
    draft.n_mtp_layers = PD_CPU_DRAFT_LAYERS
    draft.n_routed_experts = target.dspark_n_routed_experts
    draft.num_experts_per_tok = target.dspark_num_experts_per_tok
    draft_config.hf_config = draft_config.hf_text_config = draft
    draft_config.model_arch_config = SimpleNamespace(hidden_size=PD_CPU_HIDDEN)
    draft_config.dtype = model_config.dtype
    draft_config.enable_prompt_embeds = False
    draft_config.uses_mrope = False
    config.speculative_config.draft_model_config = draft_config


def initialize_pd_cpu_source_pages(pages, layout):
    """Finite typed values, before native projection and Ready publication.

    BF16 primary values and their overlaid integer/FP16 index views all remain
    finite. Page, physical block and within-block coordinates are distinct;
    this is not a checkpoint prefill or native compressed-cache arithmetic.
    """
    for page_index, (page, descriptor) in enumerate(
        zip(pages, layout.pages, strict=True)
    ):
        words = descriptor.block_stride_bytes // torch.bfloat16.itemsize
        values = page.view(torch.bfloat16).reshape(-1, words)
        blocks = torch.arange(values.shape[0])[:, None]
        offsets = torch.arange(words)[None, :]
        values.copy_(
            (
                1
                + page_index / PD_CPU_PAGE_PATTERN_DIVISOR
                + blocks / PD_CPU_BLOCK_PATTERN_DIVISOR
                + offsets / PD_CPU_WORD_PATTERN_DIVISOR
            ).to(values.dtype)
        )


def make_dspark_pd_prefix_case(
    monkeypatch,
    *,
    tp_size=1,
    tp_rank=0,
    peer_error=False,
    projection_error=False,
    original_tokens=PD_CPU_ORIGINAL_TOKENS,
    bind=True,
    num_blocks=PD_CPU_BLOCKS,
    complete_cpu_models=False,
):
    """Return actual native model/speculator/KV records, without device setup."""
    config = v41_dspark_pd_config(tensor_parallel_size=tp_size)
    if complete_cpu_models:
        _configure_complete_cpu_models(config)
    text = config.model_config.hf_text_config
    text.hidden_size, text.vocab_size = PD_CPU_HIDDEN, 128
    config.compilation_config.static_forward_context = {}
    config.attention_config = AttentionConfig()
    kv_case = make_native_kv_cache_case(num_blocks=num_blocks)
    if complete_cpu_models:
        source_swa = "model.layers.0.self_attn.swa_cache"
        group = next(
            item
            for item in kv_case.config.kv_cache_groups
            if source_swa in item.layer_names
        )
        # Native source0 shares long/index/compressor only. Every target layer
        # owns its SWA storage; homogeneous SWA specs still share one group.
        for layer in range(1, PD_CPU_TARGET_LAYERS):
            name = f"model.layers.{layer}.self_attn.swa_cache"
            tensor = torch.zeros(
                kv_case.caches[source_swa].shape,
                dtype=kv_case.caches[source_swa].dtype,
            )
            kv_case.caches[name] = tensor
            group.layer_names.append(name)
            kv_case.config.kv_cache_tensors.append(
                KVCacheTensor(
                    size=tensor.numel() * tensor.element_size(),
                    layers=[name],
                    layer_stride=0,
                    block_stride=group.kv_cache_spec.page_size_bytes,
                )
            )
    names = (("draft.layers.0.swa_cache", 4), ("draft.layers.1.swa_cache", 2))
    for name, block_size in names:
        spec = AscendSlidingWindowMLASpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=PD_CPU_WIDTH,
            dtype=torch.bfloat16,
            sliding_window=block_size,
            model_version="deepseek_v41",
        )
        tensor = torch.zeros(
            num_blocks, block_size, 1, PD_CPU_WIDTH, dtype=torch.bfloat16
        )
        kv_case.caches[name] = tensor
        kv_case.config.kv_cache_groups.append(KVCacheGroupSpec([name], spec))
        kv_case.config.kv_cache_tensors.append(
            KVCacheTensor(
                size=tensor.numel() * tensor.element_size(),
                layers=[name],
                layer_stride=0,
                block_stride=spec.page_size_bytes,
            )
        )
    loaded = make_dspark_pd_loaded_case(
        monkeypatch,
        config=config,
        kv_cache_config=kv_case.config,
        caches=kv_case.caches,
        draft_names=tuple(name for name, _ in names),
        source_engine_id="source-engine",
        tp_rank=tp_rank,
        tp_size=tp_size,
        peer_error=peer_error,
        projection_error=projection_error,
        complete_cpu_models=complete_cpu_models,
    )
    consumer = loaded.consumer
    layout = collect_deepseek_v41_kv_layout(kv_case.config, kv_case.caches)
    if bind:
        consumer.bind_registered_layout(layout, kv_case.config, kv_case.caches)
    prefix_tokens = original_tokens - 1 if original_tokens > 1 else 1
    sizes = tuple(
        group.kv_cache_spec.block_size for group in kv_case.config.kv_cache_groups
    )
    blocks = tuple(
        # The actual CircularBufferManager claims one physical ring block,
        # independently of prompt length. Do not invent one block per chunk.
        (1,)
        if is_circular_kv_cache_spec(group.kv_cache_spec)
        else tuple(range(1, 1 + (prefix_tokens + size - 1) // size))
        for group, size in zip(kv_case.config.kv_cache_groups, sizes, strict=True)
    )
    source = DsparkPdSourceRequest(
        "request-1",
        "source-generation-1",
        original_tokens,
        prefix_tokens,
        prompt_digest(tuple(range(original_tokens))),
        blocks,
        blocks,
        "source-engine",
        draft_schema_digest(config, kv_case.config),
    )
    # Circular: the connector fixture reuses this module's loaded-model helper.
    # Its actual native constructor/export methods own SWA tail clipping; keep
    # the projection allocation complete and do not mutate the caller's config.
    from tests.unit.fakes.kv_layout_connector import make_dspark_scheduler_case

    export_case = make_dspark_scheduler_case(
        monkeypatch,
        SimpleNamespace(
            config=deepcopy(config), source=source, kv_config=kv_case.config
        ),
    )
    exported = export_case.scheduler.get_sw_clipped_blocks(
        export_case.scheduler._compute_transfer_block_ids(blocks, prefix_tokens)
    )
    source = replace(source, transfer_block_ids=tuple(tuple(ids) for ids in exported))

    def make_step(offset, count):
        buffers = AscendInputBuffers(1, count, torch.device("cpu"))
        batch = AscendInputBatch.make_dummy(1, count, buffers)
        batch.req_ids[:] = [source.request_id]
        batch.num_computed_tokens_np[:] = offset
        batch.prefill_len_np[:] = prefix_tokens
        batch.positions.copy_(torch.arange(offset, offset + count))
        features = [
            (
                torch.arange(offset, offset + count)[:, None] * 10
                + torch.arange(PD_CPU_HIDDEN)[None, :]
                + layer * 30
            ).to(torch.bfloat16)
            for layer in range(3)
        ]
        state = ExecuteModelState(
            batch,
            None,
            None,
            torch.zeros(count, PD_CPU_HIDDEN),
            features,
            None,
            set(),
            None,
            None,
            None,
        )
        scheduler = SchedulerOutput(
            [],
            CachedRequestData.make_empty(),
            {source.request_id: count},
            count,
            {},
            {},
            [],
            set(),
            [],
        )
        return scheduler, state

    expected_ready = DsparkPdReady(
        source.source_engine_id,
        source.request_id,
        source.source_generation,
        source.original_prompt_tokens,
        source.computed_prefix_tokens,
        source.prompt_digest,
        layout_digest(layout),
        source.draft_schema_digest,
        source.transfer_block_ids,
        tp_rank,
        tp_size,
    )
    return SimpleNamespace(
        config=config,
        speculator=loaded.speculator,
        model=loaded.model,
        target_model=loaded.target_model,
        kv_config=kv_case.config,
        caches=kv_case.caches,
        layout=layout,
        consumer=consumer,
        source=source,
        make_step=make_step,
        agreement=loaded.agreement,
        expected_ready=expected_ready,
        draft_names=tuple(name for name, _ in names),
        draft_group_ids=loaded.draft_group_ids,
        model_bundle=loaded.model_bundle,
        complete_cpu_models=complete_cpu_models,
    )
