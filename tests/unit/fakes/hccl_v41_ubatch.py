# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit CPU parent graph owner; no model/operator/graph numerics claim."""

from types import SimpleNamespace

import torch
from torch import nn
from vllm.config import CUDAGraphMode
from vllm.forward_context import override_forward_context
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
from vllm.v1.worker.utils import AttentionGroup
from vllm_ascend.attention.dsa_v41 import (
    AscendDSAV41MetadataBuilder,
    DeepseekV41CacheBackend,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

from tests.unit.fakes.deepseek_v41 import ParentEngramState
from tests.unit.fakes.input_batch import (
    make_ascend_decode_batch,
    make_hccl_ubatch_seam,
)

COMPRESSED_METADATA_RATIO = 128
COMPRESSED_METADATA_QUERY_LEN = 3
COMPRESSED_METADATA_REQUESTS = 2
COMPRESSED_METADATA_HEAD_DIM = 4


class GraphUBatchOwner:
    def __init__(self, transport, parent_context, *, reject=None, target_query_len=1):
        self.transport = transport
        self.parent_context = parent_context
        self.reject = reject
        self.target_query_len = target_query_len

    def require_full(self, input_batch, *, for_capture):
        if self.reject == "before":
            raise RuntimeError("reservation is unavailable")
        if input_batch.num_tokens_after_padding != (
            input_batch.num_reqs_after_padding * self.target_query_len
        ):
            raise ValueError("graph parent must use its uniform target query length")

    def prepared(self, input_batch, ubatch_state, *, for_capture):
        if self.reject == "after":
            raise RuntimeError("prepared geometry does not match reservation")
        if any(
            context.cudagraph_runtime_mode != CUDAGraphMode.FULL
            for context in ubatch_state.forward_contexts
        ):
            raise AssertionError("FULL parent lost a child's graph mode")

    def forward_parent(self, model, model_inputs, ubatch_state):
        if self.transport.terminal:
            raise RuntimeError("graph parent is terminal")
        with override_forward_context(self.parent_context):
            return model(**model_inputs)

    def fail(self):
        self.transport.terminal = True


def make_native_compressed_metadata_case(runner_type, *, cache_kind):
    """Real CPU V4.1 builders through native U2 slicing and public prepare.

    Constructor-free runner state reuses the existing preparation fixture.
    Both child builders run their actual constructors/build methods on CPU;
    no metadata function or device operator is substituted. No model forward,
    graph capture/replay, NPU allocation or device equivalence is exercised.
    """
    state = ParentEngramState({})
    runner = make_hccl_ubatch_seam(
        runner_type, model_state=state, max_num_reqs=COMPRESSED_METADATA_REQUESTS
    )
    # The shared slicing fixture normally returns a CPU metadata snapshot.
    # This case selects the actual NPU adapter and native V4.1 builder instead.
    del runner._prepare_attn
    config = runner.vllm_config
    config.cache_config.block_size = COMPRESSED_METADATA_RATIO
    config.scheduler_config.max_num_seqs = COMPRESSED_METADATA_REQUESTS
    config.scheduler_config.enable_chunked_prefill = False
    config.model_config.max_model_len = COMPRESSED_METADATA_RATIO
    config.model_config.runner_type = "generate"
    text = config.model_config.hf_text_config
    text.model_type = "deepseek_v41"
    text.architectures = ["DeepseekV41ForCausalLM"]
    text.num_attention_heads = 1
    text.head_dim = COMPRESSED_METADATA_HEAD_DIM
    text.qk_rope_head_dim = COMPRESSED_METADATA_HEAD_DIM
    text.sliding_window = COMPRESSED_METADATA_HEAD_DIM
    text.index_topk = 1
    state.max_model_len = config.model_config.max_model_len
    state.device_metadata = None
    layer = "model.layers.0.self_attn." + (
        "indexer.k_cache" if cache_kind == "index_k" else "long_kv_cache"
    )
    spec = AscendMLAAttentionSpec(
        block_size=COMPRESSED_METADATA_RATIO,
        num_kv_heads=1,
        head_size=COMPRESSED_METADATA_HEAD_DIM,
        dtype=torch.bfloat16,
        tokens_per_state=COMPRESSED_METADATA_RATIO,
        storage_block_size=1,
        scale_dim=int(cache_kind == "index_k"),
        model_version="deepseek_v41",
    )
    runner.kv_cache_config = KVCacheConfig(
        num_blocks=COMPRESSED_METADATA_REQUESTS,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer], kv_cache_spec=spec)],
    )
    runner._stage_block_tables = [None, None]
    runner._stage_rope_buffers = [{}, {}]
    runner.stage_attn_groups = [
        [
            [
                AttentionGroup(
                    backend=DeepseekV41CacheBackend,
                    layer_names=[layer],
                    kv_cache_spec=spec,
                    kv_cache_group_id=0,
                    metadata_builders=[
                        AscendDSAV41MetadataBuilder(
                            spec, [layer], config, torch.device("cpu")
                        )
                    ],
                )
            ]
        ]
        for _ in range(runner.num_ubatches)
    ]
    runner.attn_groups = runner.stage_attn_groups[0]
    runner.graph_owner = GraphUBatchOwner(
        SimpleNamespace(terminal=False),
        None,
        target_query_len=COMPRESSED_METADATA_QUERY_LEN,
    )
    rows = COMPRESSED_METADATA_REQUESTS * COMPRESSED_METADATA_QUERY_LEN
    batch = make_ascend_decode_batch(
        rows, target_query_len=COMPRESSED_METADATA_QUERY_LEN
    )
    batch.positions.copy_(
        torch.arange(COMPRESSED_METADATA_QUERY_LEN).repeat(COMPRESSED_METADATA_REQUESTS)
    )
    return SimpleNamespace(
        runner=runner,
        layer=layer,
        batch=batch,
        block_tables=(torch.tensor([[1], [2]], dtype=torch.int32),),
        slots=torch.arange(rows, dtype=torch.int64).reshape(1, rows),
    )


class GraphParentModel(nn.Module):
    """Single-consumer arithmetic model; wrapper/lifecycle behavior only."""

    def __init__(self, *, providers=(), fail=False):
        super().__init__()
        self.providers = providers
        self.fail = fail
        self.consumed = False

    def forward(self, input_ids, **kwargs):
        if self.consumed:
            raise AssertionError("native parent wrapper executed twice")
        self.consumed = True
        if self.fail:
            raise RuntimeError("parent model failed")
        if self.providers:
            return torch.cat([provider.output for provider in self.providers])
        return input_ids + 1


class U1GraphOwner(GraphUBatchOwner):
    """Consumer of the actual one-child metadata protocol, on CPU only."""

    def prepared(self, input_batch, ubatch_state, *, for_capture):
        super().prepared(input_batch, ubatch_state, for_capture=for_capture)
        if len(ubatch_state.slices) != 1 or len(ubatch_state.forward_contexts) != 1:
            raise AssertionError("U1 graph owner requires one native child")
        stage, context = ubatch_state.slices[0], ubatch_state.forward_contexts[0]
        if (
            stage.request_slice != slice(0, input_batch.num_reqs_after_padding)
            or stage.token_slice != slice(0, input_batch.num_tokens_after_padding)
            or context.ubatch_idx != 0
            or context.dp_metadata is not None
        ):
            raise AssertionError("U1 graph owner received different parent coordinates")
        child = context.attn_metadata["child"]
        torch.testing.assert_close(child.input_ids, input_batch.input_ids)
        torch.testing.assert_close(context.is_padding, input_batch.is_padding)
        if not torch.equal(context.slot_mapping["layer"], child.slot_mappings[0]):
            raise AssertionError("U1 graph owner received a different slot layout")
        self.parent_context = context
