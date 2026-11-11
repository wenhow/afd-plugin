# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native Ascend batches and CPU metadata snapshots for slicing contracts."""

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu import input_batch as gpu_input_batch
from vllm.v1.worker.gpu.eplb_utils import EPLBController
from vllm_ascend.worker.v2 import model_runner as ascend_runner
from vllm_ascend.worker.v2.eplb import AscendEPLBController
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers

from tests.unit.fakes.request_state import cpu_tensor_copy
from tests.unit.fakes.vllm_config import npu_vllm_config

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )


@dataclass(frozen=True)
class CPUInputKernel:
    """Only the Triton launch boundary; native preparation wrappers stay real."""

    launch: Callable[..., None]

    def __getitem__(self, grid):
        return self.launch


def prepare_prefill_inputs_cpu(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    next_prefill_stride: int,
    num_lookahead: int,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    history_stride: int,
    prefill_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    *,
    BLOCK_SIZE: int,  # noqa: N803 - native launch keyword
    LOOKAHEAD_BLOCK: int,  # noqa: N803 - native launch keyword
) -> None:
    """Copy native prompt-tail queries and their bounded prefill lookahead."""
    if history_stride != all_token_ids.stride(0):
        raise ValueError("CPU prefill fixture requires the native history stride")
    if next_prefill_stride != next_prefill_tokens.stride(0):
        raise ValueError("CPU prefill fixture requires the native lookahead stride")
    for row, request_index in enumerate(idx_mapping.tolist()):
        computed = int(num_computed_tokens[request_index])
        prefill_len = int(prefill_lens[request_index])
        if computed >= prefill_len:
            continue
        start, stop = query_start_loc[row : row + 2].tolist()
        query_len = stop - start
        input_ids[start:stop].copy_(
            all_token_ids[request_index, computed : computed + query_len]
        )
        for lookahead in range(num_lookahead):
            position = computed + query_len + lookahead
            next_prefill_tokens[lookahead, request_index] = (
                all_token_ids[request_index, position] if position < prefill_len else 0
            )


def prepare_pos_seq_lens_cpu(
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    max_num_reqs: int,
    *,
    BLOCK_SIZE: int,  # noqa: N803 - native launch keyword
) -> None:
    """Materialize positions from the actual rejection-corrected frontier."""
    num_reqs = len(idx_mapping)
    seq_lens[num_reqs:max_num_reqs].zero_()
    for row, request_index in enumerate(idx_mapping.tolist()):
        start, stop = query_start_loc[row : row + 2].tolist()
        computed = int(num_computed_tokens[request_index])
        positions[start:stop].copy_(
            torch.arange(computed, computed + stop - start, dtype=positions.dtype)
        )
        seq_lens[row] = computed + stop - start


def combine_sampled_and_draft_tokens_cpu(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_lens: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_stride: int,
    cu_num_logits: torch.Tensor,
    logits_indices: torch.Tensor,
    *,
    BLOCK_SIZE: int,  # noqa: N803 - native launch keyword
    NUM_NEW_SAMPLED_TOKENS: int = 1,  # noqa: N803 - native launch keyword
) -> None:
    """Preserve prompt slots and consume native GPU draft values, not -1 IDs."""
    if draft_stride != draft_tokens.stride(0):
        raise ValueError("CPU combine fixture requires the native draft stride")
    for row, request_index in enumerate(idx_mapping.tolist()):
        logits_start, logits_stop = cu_num_logits[row : row + 2].tolist()
        num_logits = logits_stop - logits_start
        num_drafts = num_logits - NUM_NEW_SAMPLED_TOKENS
        query_end = int(query_start_loc[row + 1])
        query_logits_start = query_end - num_logits
        logits_indices[logits_start:logits_stop].copy_(
            torch.arange(query_logits_start, query_end, dtype=logits_indices.dtype)
        )
        seq_len = int(seq_lens[row])
        prefill_len = int(prefill_lens[request_index])
        if seq_len <= prefill_len:
            continue
        if NUM_NEW_SAMPLED_TOKENS and seq_len - num_logits >= prefill_len:
            input_ids[query_logits_start] = last_sampled_tokens.reshape(-1)[
                request_index
            ]
        if num_drafts:
            input_ids[query_end - num_drafts : query_end].copy_(
                draft_tokens[request_index, :num_drafts]
            )


def expand_idx_mapping_cpu(
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    cu_num_logits: torch.Tensor,
    *,
    BLOCK_SIZE: int,  # noqa: N803 - native launch keyword
) -> None:
    for row, request_index in enumerate(idx_mapping.tolist()):
        start, stop = cu_num_logits[row : row + 2].tolist()
        expanded_idx_mapping[start:stop].fill_(request_index)
        expanded_local_pos[start:stop].copy_(
            torch.arange(stop - start, dtype=expanded_local_pos.dtype)
        )


def skip_rotary_cache_update_cpu(positions: torch.Tensor) -> None:
    """Explicit no-RoPE leaf; this seam does not validate rotary operators."""
    if positions.device.type != "cpu":
        raise ValueError("CPU input fixture requires CPU positions")


def install_native_input_preparation_cpu(
    monkeypatch: pytest.MonkeyPatch,
    owner: "AFDNPUAttentionModelRunnerV2",
) -> None:
    """Enable actual Ascend prepare_inputs on the admitted CPU D owner.

    Four GPU-input kernels and two Ascend copy/RoPE leaves are replaced. Native
    wrappers, attention-state selection, request/frontier ownership and the
    entire prepare_inputs method remain intact. The existing loaded models,
    config, registered cache views and block tables are never replaced.
    """
    if owner.device.type != "cpu":
        raise ValueError("native input fixture requires a CPU owner")
    parallel = owner.vllm_config.parallel_config
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.enable_eplb
    ):
        raise ValueError("native input fixture requires PP1/CP1 without EPLB")
    monkeypatch.setattr(
        gpu_input_batch,
        "_prepare_prefill_inputs_kernel",
        CPUInputKernel(prepare_prefill_inputs_cpu),
    )
    monkeypatch.setattr(
        gpu_input_batch,
        "_prepare_pos_seq_lens_kernel",
        CPUInputKernel(prepare_pos_seq_lens_cpu),
    )
    monkeypatch.setattr(
        gpu_input_batch,
        "_combine_sampled_and_draft_tokens_kernel",
        CPUInputKernel(combine_sampled_and_draft_tokens_cpu),
    )
    monkeypatch.setattr(
        gpu_input_batch,
        "_expand_idx_mapping_kernel",
        CPUInputKernel(expand_idx_mapping_cpu),
    )
    monkeypatch.setattr(ascend_runner, "async_copy_to_gpu", cpu_tensor_copy)
    monkeypatch.setattr(ascend_runner, "update_cos_sin", skip_rotary_cache_update_cpu)
    owner.model_config = owner.vllm_config.model_config
    owner.input_buffers = AscendInputBuffers(
        owner.max_num_reqs, owner.max_num_tokens, owner.device
    )
    owner._finegrained_tp_requires_graph = False
    owner.sync_spec_pp_cpu_counts = False
    owner.pcp_manager = None
    owner.use_fia = False
    # Only native phase bookkeeping is consumed here. Do not initialize an
    # EPLB policy, process group, model registration or device state on CPU.
    owner.eplb = AscendEPLBController.__new__(AscendEPLBController)
    EPLBController.__init__(owner.eplb, parallel, owner.device)
    owner.eplb.load_collection_phase = "all"


def make_ascend_batch(real=5, padded=8, *, num_reqs=1):
    buffers = AscendInputBuffers(max(2, num_reqs), padded, torch.device("cpu"))
    batch = AscendInputBatch.make_dummy(num_reqs, real, buffers)
    batch.num_tokens_after_padding = padded
    batch.input_ids = torch.arange(padded)
    batch.positions = torch.arange(padded)
    batch.is_padding = torch.arange(padded) >= real
    batch.seq_lens += 12
    batch.seq_lens_cpu_upper_bound += 12
    batch.seq_lens_np = batch.num_scheduled_tokens.copy() + 12
    return batch


def make_ascend_speculative_batch(real=5, padded=8):
    """Two native request queries, with all target logits retained for sampling.

    This describes scheduled draft tokens; it performs no acceptance/rejection
    and does not run the native request-history or device preparation kernels.
    """
    batch = make_ascend_batch(real, padded, num_reqs=2)
    batch.num_draft_tokens_per_req = batch.num_scheduled_tokens - 1
    batch.num_draft_tokens = int(batch.num_draft_tokens_per_req.sum())
    counts = torch.from_numpy(batch.num_scheduled_tokens)
    batch.expanded_idx_mapping = torch.repeat_interleave(batch.idx_mapping, counts)
    batch.expanded_local_pos = torch.cat(
        [torch.arange(int(count), dtype=torch.int32) for count in counts]
    )
    batch.logits_indices = torch.arange(real, dtype=torch.int64)
    batch.cu_num_logits = batch.query_start_loc.clone()
    batch.cu_num_logits_np = batch.query_start_loc_np.copy()
    return batch


def make_ascend_decode_batch(rows=8, *, target_query_len=1):
    """Real native uniform-query dummy geometry on CPU, without graph capture."""
    assert rows % target_query_len == 0
    num_reqs = rows // target_query_len
    buffers = AscendInputBuffers(num_reqs, rows, torch.device("cpu"))
    batch = AscendInputBatch.make_dummy(
        num_reqs,
        rows,
        buffers,
        max_query_len=None if target_query_len == 1 else target_query_len,
    )
    batch.input_ids.copy_(torch.arange(rows))
    batch.positions.copy_(torch.arange(rows))
    return batch


def snapshot_ascend_child(
    input_batch,
    cudagraph_mode,
    block_tables,
    slot_mappings,
    attn_groups,
    kv_cache_config,
    for_capture=False,
    ubatch_idx=0,
):
    """Expose sliced coordinates as metadata without building device operators."""
    return {
        "child": SimpleNamespace(
            input_ids=input_batch.input_ids.clone(),
            positions=input_batch.positions.clone(),
            valid_tokens=input_batch.num_tokens,
            padded_tokens=input_batch.num_tokens_after_padding,
            query_start_loc=input_batch.query_start_loc.clone(),
            seq_lens=input_batch.seq_lens.clone(),
            seq_lens_np=input_batch.seq_lens_np.copy(),
            block_tables=tuple(table.clone() for table in block_tables),
            slot_mappings=slot_mappings.clone(),
        )
    }


def make_hccl_ubatch_seam(
    runner_type, *, model_state, dp_size=1, max_num_reqs=2, graph_owner=None
):
    """Construct only CPU preparation state; no runner device constructor."""
    runner = object.__new__(runner_type)
    runner.vllm_config = npu_vllm_config(
        connector="P2pHcclAFDConnector",
        data_parallel_size=dp_size,
        use_ubatching=True,
        num_ubatches=2,
        cp_kv_cache_interleave_size=1,
    )
    runner.vllm_config.compilation_config.static_forward_context = {}
    runner.parallel_config = runner.vllm_config.parallel_config
    runner.model_state = model_state
    runner.graph_owner = graph_owner
    runner.eager_session = None
    runner.diagnostic_binding = None
    runner._preparing_cg_mode = CUDAGraphMode.NONE
    runner.num_ubatches, runner.dcp_size, runner.dcp_rank = 2, 1, 0
    runner.cp_interleave = 1
    runner.ubatch_query_start_loc = [
        torch.zeros(max_num_reqs + 1, dtype=torch.int32) for _ in range(2)
    ]
    runner.ubatch_seq_lens = [
        torch.zeros(max_num_reqs, dtype=torch.int32) for _ in range(2)
    ]
    runner.ubatch_dcp_local_seq_lens = [None, None]
    runner.attn_groups = []
    runner.stage_attn_groups = [[], []]
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layer"])]
    )
    runner._prepare_attn = snapshot_ascend_child
    runner._uses_mla_graph_params = False
    return runner
