# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit CPU parent graph owner; no model/operator/graph numerics claim."""

import torch
from torch import nn
from vllm.config import CUDAGraphMode
from vllm.forward_context import override_forward_context


class GraphUBatchOwner:
    def __init__(self, transport, parent_context, *, reject=None):
        self.transport = transport
        self.parent_context = parent_context
        self.reject = reject

    def require_full(self, input_batch, *, for_capture):
        if self.reject == "before":
            raise RuntimeError("reservation is unavailable")
        if input_batch.num_tokens_after_padding != input_batch.num_reqs_after_padding:
            raise ValueError("graph parent must use uniform query length one")

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
