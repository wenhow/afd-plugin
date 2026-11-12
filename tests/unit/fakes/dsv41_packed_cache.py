# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Headwise CPU arithmetic reference; never a packed NPU kernel emulator."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch


@dataclass
class PackedAttentionInputs:
    q: torch.Tensor
    ori_kv: torch.Tensor
    cmp_kv: torch.Tensor | None
    metadata: SimpleNamespace
    compressed_indices: torch.Tensor
    compressed_lengths: torch.Tensor
    sinks: torch.Tensor
    window_size: int = 3
    softmax_scale: float = 0.5

    def arguments(self):
        return (
            self.q,
            self.ori_kv,
            self.cmp_kv,
            self.metadata,
            self.compressed_indices,
        )

    def options(self):
        return {
            "window_size": self.window_size,
            "sinks": self.sinks,
            "softmax_scale": self.softmax_scale,
            "compressed_lengths": self.compressed_lengths,
        }


def make_packed_attention_inputs(*, tokens=3, heads=16, compressed=True, dtype=None):
    width = 4
    q = torch.arange(tokens * heads * width).reshape(tokens, heads, width).sin()
    ori_kv = torch.arange(7 * width).reshape(7, width).cos()
    cmp_kv = torch.arange(5 * width).reshape(5, width).sin()
    indices = torch.arange(tokens)[:, None] + torch.arange(3)[None, :]
    lengths = (torch.arange(tokens) % 3 + 1).to(torch.int32)
    compressed_indices = torch.arange(tokens)[:, None] + torch.arange(2)[None, :]
    metadata = SimpleNamespace(
        swa=SimpleNamespace(
            ori_sparse_indices=indices[:, None, :].to(torch.int32),
            ori_topk_length=lengths,
        ),
    )
    return PackedAttentionInputs(
        q=q.to(dtype or torch.bfloat16),
        ori_kv=ori_kv,
        cmp_kv=cmp_kv if compressed else None,
        metadata=metadata,
        compressed_indices=compressed_indices.to(torch.int32),
        compressed_lengths=(torch.arange(tokens) % 2 + 1).to(torch.int32),
        sinks=torch.linspace(-0.75, 0.75, heads),
    )


def headwise_attention_with_sink(
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
    """Independent heads over shared K=V rows plus a zero-valued sink."""
    tokens = q.shape[0]
    ori_indices = metadata.swa.ori_sparse_indices[:tokens, 0, :window_size]
    values = ori_kv[ori_indices.long()]
    mask = (
        torch.arange(ori_indices.shape[1])[None, :]
        < metadata.swa.ori_topk_length[:tokens, None]
    )
    if cmp_kv is not None:
        indices = compressed_indices[:tokens]
        values = torch.cat((values, cmp_kv[indices.long()]), dim=1)
        compressed_mask = (
            torch.arange(indices.shape[1])[None, :] < compressed_lengths[:tokens, None]
        )
        mask = torch.cat((mask, compressed_mask), dim=1)
    logits = torch.einsum("thd,tkd->thk", q.float(), values.float()) * softmax_scale
    logits.masked_fill_(~mask[:, None, :], -torch.inf)
    sink_logits = sinks.detach().float()[None, :, None].expand(tokens, -1, -1)
    probabilities = torch.cat((logits, sink_logits), dim=-1).softmax(dim=-1)
    return torch.einsum("thk,tkd->thd", probabilities[:, :, :-1], values.float()).to(
        torch.bfloat16
    )


def packed_64_head_attention_reference(*args, **kwargs):
    """Expose the installed wheel's observed shape gate around CPU arithmetic."""
    heads = args[0].shape[1]
    if heads != 64:
        raise ValueError(
            f"N1 must be 64 (the only supported query-head count), got {heads}"
        )
    return headwise_attention_with_sink(*args, **kwargs)
