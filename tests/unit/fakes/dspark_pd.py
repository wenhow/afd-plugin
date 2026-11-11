# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native loaded DSpark context bodies with CPU RoPE/write/sync leaves.

Actual Ascend speculator group discovery and V4.1 combine/project/store bodies
remain intact. These allocated CPU weights are an independent arithmetic
contract, not checkpoint weights, packed SWA writes or a device transfer.
"""

from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn
from vllm.config.attention import AttentionConfig
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheTensor
from vllm.v1.worker.gpu.model_runner import ExecuteModelState
from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.models.deepseek_v41 import dspark as dspark_model_module
from vllm_ascend.models.deepseek_v41.dspark import (
    DeepseekV41DSparkModel,
    DSparkDeepseekV41ForCausalLM,
)
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
from tests.unit.fakes.kv_cache import make_native_kv_cache_case
from tests.unit.fakes.vllm_config import v41_dspark_pd_config

PD_CPU_HIDDEN = 4
PD_CPU_WIDTH = 2
PD_CPU_BLOCKS = 4
PD_CPU_ORIGINAL_TOKENS = 6


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
):
    """Return actual native model/speculator/KV records, without device setup."""
    config = v41_dspark_pd_config(tensor_parallel_size=tp_size)
    text = config.model_config.hf_text_config
    text.hidden_size, text.vocab_size = PD_CPU_HIDDEN, 128
    config.compilation_config.static_forward_context = {}
    config.attention_config = AttentionConfig()
    kv_case = make_native_kv_cache_case(num_blocks=num_blocks)
    names = (("draft.layers.0.swa_cache", 4), ("draft.layers.1.swa_cache", 2))
    first_draft_group = len(kv_case.config.kv_cache_groups)
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
    model, rope = _projection_model(kv_case, names, fail_projection=projection_error)
    for block in model.model.layers.values():
        cache = block.self_attn.dsa_attn.swa_cache_layer
        config.compilation_config.static_forward_context[cache.prefix] = cache
    speculator = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    speculator.model = model
    speculator.vllm_config = config
    # Native DFlash's read-only attention config derives causality from these
    # actual constructor fields; keep that property intact in this CPU seam.
    speculator.requires_non_causal = True
    speculator.device = torch.device("cpu")
    speculator.draft_attn_layer_names = set(model.get_draft_kv_cache_layer_names())
    speculator.draft_kv_cache_group_ids = (first_draft_group, first_draft_group + 1)
    speculator._layer_group_idx = [1, 0]
    speculator.kv_cache_config = kv_case.config
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
    target_model = nn.Module()
    consumer = DsparkPdPrefixConsumer(
        config,
        speculator,
        target_model=target_model,
        device=speculator.device,
        source_engine_id="source-engine",
        tp_rank=tp_rank,
        tp_size=tp_size,
        agree_batch=agreement,
    )
    layout = collect_deepseek_v41_kv_layout(kv_case.config, kv_case.caches)
    if bind:
        consumer.bind_registered_layout(layout, kv_case.config, kv_case.caches)
    prefix_tokens = original_tokens - 1 if original_tokens > 1 else 1
    sizes = tuple(
        group.kv_cache_spec.block_size for group in kv_case.config.kv_cache_groups
    )
    blocks = tuple(
        tuple(range(1, 1 + (prefix_tokens + size - 1) // size)) for size in sizes
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
        speculator=speculator,
        model=model,
        target_model=target_model,
        kv_config=kv_case.config,
        caches=kv_case.caches,
        layout=layout,
        consumer=consumer,
        source=source,
        make_step=make_step,
        agreement=agreement,
        expected_ready=expected_ready,
        draft_names=tuple(name for name, _ in names),
        draft_group_ids=speculator.draft_kv_cache_group_ids,
    )
