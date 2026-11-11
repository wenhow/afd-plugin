# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real DSpark proposal with explicit CPU preparation/metadata/model leaves.

Native Ascend/DFlash propose, DSpark generation and sequential Markov sampling
remain unchanged. This excludes device construction, KV kernels, probabilistic
sampling and attention; it is not a captured draft graph or a weight test.
"""

from copy import copy
from types import MethodType, SimpleNamespace

import torch
from torch import nn
from vllm.config import CUDAGraphMode
from vllm.config.attention import AttentionConfig
from vllm.forward_context import get_forward_context
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_module
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator

from tests.unit.fakes.cudagraph_manager import make_native_graph_manager
from tests.unit.fakes.hccl_v41_graph_coordinator import retain_native_target_capture
from tests.unit.fakes.input_batch import make_ascend_decode_batch

DSPARK_CPU_VOCAB_SIZE = 17
DSPARK_CPU_HIDDEN_SIZE = 4
DSPARK_CPU_REQUESTS = 2
DSPARK_CPU_BLOCK_SIZE = 32
DSPARK_CPU_MAX_MODEL_LEN = 64
DSPARK_CPU_MASK_TOKEN = 16
DSPARK_CPU_FIRST_ANCHOR = 3


class DSparkModelBoundary(nn.Module):
    """Observable draft payload/context and deterministic greedy Markov head."""

    def __init__(self, *, fail_backbone=False):
        super().__init__()
        self.error = (
            RuntimeError("controlled DSpark backbone failure")
            if fail_backbone
            else None
        )
        self.contexts = []

    def combine_hidden_states(self, hidden_states):
        return hidden_states.view(
            hidden_states.shape[0], -1, DSPARK_CPU_HIDDEN_SIZE
        ).mean(dim=1)

    def precompute_and_store_context_kv(self, hidden_states, positions, slots=None):
        pass

    def forward(self, input_ids, positions, inputs_embeds):
        self.contexts.append(get_forward_context())
        if self.error is not None:
            raise self.error
        return (
            (input_ids + positions)[:, None].expand(-1, DSPARK_CPU_HIDDEN_SIZE).float()
        )

    def compute_draft_logits(self, hidden_states):
        return hidden_states.new_zeros((hidden_states.shape[0], DSPARK_CPU_VOCAB_SIZE))

    def markov_embed(self, token_ids):
        return token_ids[:, None]

    def markov_bias(self, embeddings):
        next_ids = (embeddings + 1) % DSPARK_CPU_VOCAB_SIZE
        vocabulary = torch.arange(DSPARK_CPU_VOCAB_SIZE)
        return -(vocabulary - next_ids).float().square()

    def map_draft_to_target(self, token_ids):
        return token_ids


def prepare_dspark_inputs_cpu(
    input_buffers,
    query_slot_mapping,
    context_positions,
    context_slot_mapping,
    sample_indices,
    sample_pos,
    sample_idx_mapping,
    temperature,
    seeds,
    input_batch,
    num_sampled,
    num_rejected,
    last_sampled,
    next_prefill_tokens,
    input_temperature,
    input_seeds,
    block_table,
    block_size,
    cp_rank,
    cp_size,
    cp_interleave,
    parallel_drafting_token_id,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    sample_from_anchor=False,
):
    """Explicit CP1 resident-block reference for the device prepare operator.

    Only query/context row preparation is replaced. No DSpark proposal, model
    dispatch or sampling body is copied. Rejected context receives slot -1;
    masks, bonus/prefill anchors and the two native query layouts are explicit.
    """
    if (cp_rank, cp_size, cp_interleave) != (0, 1, 1):
        raise AssertionError("CPU DSpark preparation supports CP1 only")
    query_slot_mapping.fill_(-1)
    context_slot_mapping.fill_(-1)
    sample_indices.zero_()
    sample_pos.zero_()
    sample_idx_mapping.fill_(-1)
    input_buffers.seq_lens.zero_()
    input_buffers.query_start_loc.fill_(input_batch.num_reqs * num_query_per_req)
    for row, request_index in enumerate(input_batch.idx_mapping.tolist()):
        start, stop = input_batch.query_start_loc[row : row + 2].tolist()
        valid_stop = stop - int(num_rejected[row])
        positions = input_batch.positions[start:valid_stop]
        context_positions[start:stop].zero_()
        context_positions[start:valid_stop].copy_(positions)
        block_numbers = (positions // block_size).clamp(max=block_table.shape[1] - 1)
        blocks = block_table[row, block_numbers]
        context_slot_mapping[start:valid_stop].copy_(
            torch.where(blocks != 0, blocks * block_size + positions % block_size, -1)
        )
        query_base = row * num_query_per_req
        query_positions = positions[-1] + 1 + torch.arange(num_query_per_req)
        query_ids = input_buffers.input_ids[query_base : query_base + num_query_per_req]
        query_ids.fill_(parallel_drafting_token_id)
        query_ids[0] = (
            last_sampled[request_index]
            if num_sampled[row] > 0
            else next_prefill_tokens[request_index]
        )
        input_buffers.positions[query_base : query_base + num_query_per_req].copy_(
            query_positions.clamp(max=max_model_len - 1)
        )
        query_blocks = block_table[
            row, (query_positions // block_size).clamp(max=block_table.shape[1] - 1)
        ]
        query_slot_mapping[query_base : query_base + num_query_per_req].copy_(
            torch.where(
                query_blocks != 0,
                query_blocks * block_size + query_positions % block_size,
                -1,
            )
        )
        input_buffers.query_start_loc[row] = query_base
        input_buffers.seq_lens[row] = min(
            int(positions[-1]) + 1 + num_query_per_req, max_model_len
        )
        sample_base = row * num_speculative_steps
        sample_slice = slice(sample_base, sample_base + num_speculative_steps)
        offset = 0 if sample_from_anchor else 1
        sample_indices[sample_slice] = (
            query_base + offset + torch.arange(num_speculative_steps)
        )
        sample_pos[sample_slice] = query_positions[offset:] + int(sample_from_anchor)
        sample_idx_mapping[sample_slice] = request_index
        temperature[request_index] = input_temperature[request_index]
        seeds[request_index] = input_seeds[request_index]


def _draft_metadata_cpu(speculator, *, num_reqs, batch_desc, **kwargs):
    """Metadata/backend kernel leaf; preserve real native forward-context entry."""
    return {
        "dspark-only": SimpleNamespace(
            num_reqs=num_reqs,
            num_tokens=batch_desc.num_tokens,
            query_start_loc=speculator.input_buffers.query_start_loc.clone(),
        )
    }


def make_dspark_proposal_case(
    monkeypatch,
    *,
    vllm_config,
    num_speculative_tokens=2,
    sample_from_anchor=True,
    fail_backbone=False,
    num_reqs=DSPARK_CPU_REQUESTS,
    reverse_requests=False,
):
    """Build a native DSpark instance without its device/weight constructor.

    Normal proposal executes its genuine preparation/dispatch/control flow;
    only prepare_dflash_inputs and metadata generation have CPU leaves. Dummy
    memory-profile mode legitimately skips those leaves in the pinned source.
    """
    n = num_speculative_tokens
    width = n if sample_from_anchor else n + 1
    max_tokens = num_reqs * (n + 1)
    target = make_ascend_decode_batch(max_tokens, target_query_len=n + 1)
    target.is_dummy = False
    target.is_prefilling_np.fill(False)
    if reverse_requests:
        target.idx_mapping.copy_(torch.arange(num_reqs - 1, -1, -1))
    model = DSparkModelBoundary(fail_backbone=fail_backbone)
    config = copy(vllm_config)
    config.model_config = copy(vllm_config.speculative_config.draft_model_config)
    config.model_config.enforce_eager = True
    config.speculative_config = None
    config.compilation_config = copy(config.compilation_config)
    config.compilation_config.static_forward_context = {}
    config.attention_config = AttentionConfig()
    draft = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    draft.vllm_config = config
    draft.device = torch.device("cpu")
    draft.model = model
    draft.replicated_pcp = draft.use_dcp = False
    draft.pcp_manager = None
    draft.attn_architecture = None
    draft.max_num_tokens, draft.max_num_reqs = max_tokens, num_reqs
    draft.max_model_len = DSPARK_CPU_MAX_MODEL_LEN
    draft.num_speculative_steps, draft.num_query_per_req = n, width
    draft.sample_from_anchor = sample_from_anchor
    draft.requires_non_causal = True
    draft.supports_mm_inputs = False
    draft.eplb_state = draft.acceptance_estimator = None
    draft._draft_topk = None
    draft.use_confidence_head = False
    draft.draft_logits = None
    draft._step_cols = torch.arange(n, dtype=torch.int32)
    draft._anchor_idx = torch.arange(num_reqs) * width
    draft.input_buffers = AscendInputBuffers(num_reqs, max_tokens, draft.device)
    draft.hidden_states = torch.zeros(max_tokens, DSPARK_CPU_HIDDEN_SIZE)
    draft.context_positions = torch.zeros(max_tokens, dtype=torch.int64)
    draft.sample_indices = torch.zeros(num_reqs * n, dtype=torch.int64)
    draft.sample_pos = torch.zeros_like(draft.sample_indices)
    draft.sample_idx_mapping = torch.full_like(draft.sample_indices, -1)
    draft.draft_tokens = torch.full((num_reqs, n), -1, dtype=torch.int64)
    draft.temperature = torch.zeros(num_reqs)
    draft.seeds = torch.zeros(num_reqs, dtype=torch.int64)
    draft._context_slot_mappings = torch.full((1, max_tokens), -1, dtype=torch.int64)
    draft._layer_group_idx = None
    draft._group_causal = True
    draft.draft_kv_cache_group_id = 0
    draft.draft_kv_cache_group_ids = [0]
    draft.parallel_drafting_token_id = DSPARK_CPU_MASK_TOKEN
    draft.dp_size, draft.dp_rank = 1, 0
    draft.attn_groups = []
    draft.block_tables = SimpleNamespace(
        slot_mappings=torch.full((1, max_tokens), -1, dtype=torch.int64),
        # Native prepare indexes already-gathered block tables by batch row,
        # whereas last_sampled/seeds use the request-state idx_mapping.
        input_block_tables=[torch.arange(1, num_reqs + 1)[:, None].expand(-1, 2)],
        kernel_block_sizes=[DSPARK_CPU_BLOCK_SIZE],
        cp_rank=0,
        cp_size=1,
        cp_interleave=1,
    )
    draft.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["dspark-only"])]
    )
    draft._build_uniform_attn_metadata = MethodType(_draft_metadata_cpu, draft)
    draft.query_cudagraph_manager = make_native_graph_manager(())
    aux = [
        torch.full((max_tokens, DSPARK_CPU_HIDDEN_SIZE), float(value))
        for value in (2, 6)
    ]
    anchors = torch.arange(num_reqs) + DSPARK_CPU_FIRST_ANCHOR
    kwargs = dict(
        input_batch=target,
        attn_metadata={"target-only": SimpleNamespace()},
        slot_mappings={},
        last_hidden_states=torch.full((max_tokens, DSPARK_CPU_HIDDEN_SIZE), -100.0),
        aux_hidden_states=aux,
        num_sampled=torch.full((num_reqs,), n + 1, dtype=torch.int32),
        num_rejected=torch.zeros(num_reqs, dtype=torch.int32),
        last_sampled=anchors,
        next_prefill_tokens=anchors + 5,
        temperature=torch.zeros(num_reqs),
        seeds=torch.arange(num_reqs, dtype=torch.int64),
    )
    # Seed persistent profile inputs explicitly; native profile skips preparation.
    batch_anchors = anchors[target.idx_mapping.long()]
    for row, anchor in enumerate(batch_anchors):
        draft.input_buffers.input_ids[row * width] = anchor
    monkeypatch.setattr(
        dflash_module, "prepare_dflash_inputs", prepare_dspark_inputs_cpu
    )
    return SimpleNamespace(
        speculator=draft,
        model=model,
        input_batch=target,
        kwargs=kwargs,
        expected_tokens=(batch_anchors[:, None] + torch.arange(1, n + 1))
        % DSPARK_CPU_VOCAB_SIZE,
        expected_anchors=batch_anchors,
        query_width=width,
    )


def bind_dspark_proposal_after_target_capture(case, proposal, *, mode="target"):
    """Run the real proposal while the plugin's target capture hooks are scoped.

    Native NONE draft capture performs no model forward. The explicit proposal
    after that return is an observable host invocation, not captured execution.
    """
    draft = proposal.speculator
    proposal.input_batch.is_dummy = mode != "target"
    manager = DFlashCudaGraphManager.__new__(DFlashCudaGraphManager)
    vars(manager).update(vars(make_native_graph_manager(())))
    manager.cudagraph_mode = CUDAGraphMode.NONE
    manager._capture_descs = {}
    manager.host_descriptors, manager.graphs = (), {}
    # The inherited coordinator CPU capture boundary also serves this empty
    # NONE manager; it must have the same explicit failure selector.
    manager.capture_failure = None
    draft.query_cudagraph_manager = manager

    def capture():
        retain_native_target_capture(case)
        DFlashSpeculator.capture(draft)
        case.capture_order.append("native-dspark-none-capture-returned")
        proposal.output = draft.propose(
            **proposal.kwargs,
            dummy_run=mode != "target",
            skip_attn_for_dummy_run=mode != "target",
            is_profile=mode == "profile",
        )
        case.capture_order.append("native-dspark-proposal-returned")

    draft.capture = capture
    case.owner.speculator = draft
    return proposal
