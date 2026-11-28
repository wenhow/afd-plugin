# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Per-Attention A5 packed-cache compatibility for the TP4 query heads."""

from torch.nn import functional
from vllm_ascend.models.deepseek_v41 import mixed_quant_attention as native

DSV41_NATIVE_QUERY_HEADS = 64
DSV41_TP4_QUERY_HEADS = 16


class AFDMixedQuantPackedCacheOps(native.MixedQuantPackedCacheOps):
    """Retain the native cache/index/metadata ABI and adapt only TP4 qsmla."""

    # Upstream: mixed_quant_attention.py::qsmla @ Ascend 7e37e29a.
    # Patch reason: the deployed A5 ds41 wheel accepts 64 query heads while
    # Attention TP4 supplies 16; the native task metadata already declares 64.
    # Patch functionality: pad independent query/sink heads and crop the output.
    # Signature: matches upstream exactly; no added parameters or annotations.
    # Delegation exception: native qsmla owns cache indices, metadata wait,
    # packed operator kwargs and sink FP32 conversion. Reusing that owner avoids
    # copying its device ABI for a change confined to head shape at this seam.
    # Cost: TP4 computes four times the attention heads plus temporary Q/output
    # and a contiguous crop; KV/cache/metadata and HCCL traffic are not expanded.
    # Removal: drop the adapter when the pinned wheel supports native 16 heads.
    # CPU headwise checks do not qualify the packed A5 kernel or its performance.
    @staticmethod
    def qsmla(
        q,
        ori_kv,
        cmp_kv,
        metadata,
        compressed_indices,
        *,
        window_size,
        sinks,
        softmax_scale,
        compressed_lengths=None,
    ):
        # ### PATCH START: TP4 independent query heads
        local_query_heads = q.shape[1]
        pad_tp4 = local_query_heads == DSV41_TP4_QUERY_HEADS
        if pad_tp4:
            padding_heads = DSV41_NATIVE_QUERY_HEADS - local_query_heads
            q = functional.pad(q, (0, 0, 0, padding_heads))
            sinks = functional.pad(sinks, (0, padding_heads))
        output = native.qsmla(
            q,
            ori_kv,
            cmp_kv,
            metadata,
            compressed_indices,
            window_size=window_size,
            sinks=sinks,
            softmax_scale=softmax_scale,
            compressed_lengths=compressed_lengths,
        )
        if pad_tp4:
            return output[:, :local_query_heads, :].contiguous()
        return output
        # ### PATCH END: TP4 independent query heads
