# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Eager, single-thread V4.1 shared-attention buffer views for one parent.

The native reset is a no-op. Index sources and consumers instead share four
fixed buffers and address their leading token rows. Binding disjoint parent
views keeps that native per-stage sharing without copying storage or KV caches.

This component is not installed by the production model guard. Offset views
still need validation with the target device operators. The scoped Python
bindings are for a single-thread eager driver, not concurrent model forwards,
graph capture or cross-stream ownership.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

from vllm_ascend.models.deepseek_v41.model import (
    DeepseekV41Model,
    DeepseekV41SharedAttentionState,
)

MAX_ATTENTION_STAGES = 2


class DeepseekV41StageAttentionStates:
    """Preserve native source/consumer sharing within each token slice.

    The model must use the PP1 target layout: each layer has native Attention
    and references the model's parent shared state. Slices cover the parent's
    active padded extent from zero, not the buffers' whole capacity; zero-sized
    children and wholly empty parents are valid.
    """

    def __init__(
        self, model: DeepseekV41Model, token_slices: tuple[slice, ...]
    ) -> None:
        if not isinstance(token_slices, tuple) or not (
            1 <= len(token_slices) <= MAX_ATTENTION_STAGES
        ):
            raise ValueError("V4.1 Attention requires one or two token slices")
        end = 0
        for token_slice in token_slices:
            if (
                not isinstance(token_slice, slice)
                or type(token_slice.start) is not int
                or type(token_slice.stop) is not int
                or token_slice.start != end
                or token_slice.stop < token_slice.start
                or (
                    token_slice.step is not None
                    and (type(token_slice.step) is not int or token_slice.step != 1)
                )
            ):
                raise ValueError(
                    "stage token slices must be bounded and contiguous from zero"
                )
            end = token_slice.stop
        parent = model.shared_attention_state
        for buffer in (
            parent.topk_indices,
            parent.candidates,
            parent.candidate_lengths,
            parent.topk_lengths,
        ):
            if buffer is not None and (
                buffer.ndim == 0 or end > buffer.shape[0] or not buffer.is_contiguous()
            ):
                raise ValueError(
                    "parent shared buffers require contiguous token capacity"
                )
        attentions = tuple(layer.self_attn for layer in model.layers)
        if any(attention.shared_state is not parent for attention in attentions):
            raise ValueError("native Attention layers must share the parent state")
        # Validate the entire layout and binding graph before creating views.
        self._model = model
        self._parent = parent
        self._attentions = attentions
        self._states = tuple(
            DeepseekV41SharedAttentionState(
                parent.topk_indices[token_slice],
                parent.candidates[token_slice],
                None
                if parent.candidate_lengths is None
                else parent.candidate_lengths[token_slice],
                None
                if parent.topk_lengths is None
                else parent.topk_lengths[token_slice],
            )
            for token_slice in token_slices
        )
        self._active = False

    @contextmanager
    def bind(self, stage_idx: int) -> Iterator[DeepseekV41SharedAttentionState]:
        """Bind one child's row views and restore all native references on exit."""
        if type(stage_idx) is not int or not 0 <= stage_idx < len(self._states):
            raise ValueError("stage_idx is outside the prepared Attention stages")
        if self._active:
            raise RuntimeError("Attention stage binding is not reentrant")
        # Snapshot every original binding before changing any object, so a
        # missing/broken layer field cannot leave a partially bound model.
        original_model_state = self._model.shared_attention_state
        original_attention_states = tuple(
            attention.shared_state for attention in self._attentions
        )
        if original_model_state is not self._parent or any(
            state is not self._parent for state in original_attention_states
        ):
            raise RuntimeError("Attention stage binding requires the unbound parent")
        stage_state = self._states[stage_idx]
        self._active = True
        try:
            with ExitStack() as restoration:
                restoration.callback(
                    setattr, self._model, "shared_attention_state", original_model_state
                )
                self._model.shared_attention_state = stage_state
                for attention, original in zip(
                    self._attentions, original_attention_states, strict=True
                ):
                    restoration.callback(setattr, attention, "shared_state", original)
                    attention.shared_state = stage_state
                yield stage_state
        finally:
            self._active = False


__all__ = ["DeepseekV41StageAttentionStates"]
