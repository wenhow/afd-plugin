# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native Ascend batches and CPU metadata snapshots for slicing contracts."""

from types import SimpleNamespace

import numpy as np
import torch
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers

from tests.unit.fakes.vllm_config import npu_vllm_config


def make_ascend_batch(real=5, padded=8):
    buffers = AscendInputBuffers(2, padded, torch.device("cpu"))
    batch = AscendInputBatch.make_dummy(1, real, buffers)
    batch.num_tokens_after_padding = padded
    batch.input_ids = torch.arange(padded)
    batch.positions = torch.arange(padded)
    batch.is_padding = torch.arange(padded) >= real
    batch.seq_lens += 12
    batch.seq_lens_cpu_upper_bound += 12
    batch.seq_lens_np = np.array([real + 12], dtype=np.int32)
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


def make_hccl_ubatch_seam(runner_type, *, model_state, dp_size=1):
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
    runner.num_ubatches, runner.dcp_size, runner.dcp_rank = 2, 1, 0
    runner.cp_interleave = 1
    runner.ubatch_query_start_loc = [
        torch.zeros(3, dtype=torch.int32) for _ in range(2)
    ]
    runner.ubatch_seq_lens = [torch.zeros(2, dtype=torch.int32) for _ in range(2)]
    runner.ubatch_dcp_local_seq_lens = [None, None]
    runner.attn_groups = []
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layer"])]
    )
    runner._prepare_attn = snapshot_ascend_child
    runner._uses_mla_graph_params = False
    return runner
