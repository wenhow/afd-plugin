# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native Ascend batches and CPU metadata snapshots for slicing contracts."""

from types import SimpleNamespace

import numpy as np
import torch
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers


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
