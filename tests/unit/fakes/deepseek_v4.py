# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU protocol peers for the old DeepSeek V4 role boundary."""

from types import SimpleNamespace

from torch import nn
from vllm.forward_context import override_forward_context

from afd_plugin.connectors import AFDA2FTransferPayload

ROUTER_DELTA_SCALE = 4_096


class RoutingMoE(nn.Module):
    """Observable FP32/raw-ID arithmetic; this does not simulate native MoE."""

    def forward(self, hidden_states, input_ids=None, hidden_states_fp32=None):
        router_input = (
            hidden_states.float() if hidden_states_fp32 is None else hidden_states_fp32
        )
        return (
            (router_input - hidden_states.float()) * ROUTER_DELTA_SCALE
            + input_ids[:, None]
        ).to(hidden_states.dtype)


class RoleBoundaryPeer:
    """Exchange public payloads and execute the adapter's real F-side hook."""

    def __init__(self, layer):
        self.layer = layer
        self.payloads = []

    def send_attn_output(self, hidden_states, context, **kwargs):
        payload = AFDA2FTransferPayload(
            hidden_states=hidden_states.clone(),
            context=context,
            input_ids=kwargs["input_ids"].clone(),
            hidden_states_fp32=(
                kwargs["hidden_states_fp32"].clone()
                if "hidden_states_fp32" in kwargs
                else None
            ),
        )
        self.payloads.append(payload)

    def recv_ffn_output(self, *, ref_tensor, ubatch_idx):
        payload = self.payloads[-1]
        if ubatch_idx != payload.context.metadata.stage_idx:
            raise ValueError("reply stage differs from the original request")
        with override_forward_context(SimpleNamespace(input_ids=payload.input_ids)):
            return self.layer.compute_ffn_output(
                payload.hidden_states,
                hidden_states_fp32=payload.hidden_states_fp32,
            )


def ffn_layer(layer_type):
    """Build only the public F compute seam, without native device allocation."""
    layer = layer_type.__new__(layer_type)
    nn.Module.__init__(layer)
    layer.mlp = RoutingMoE()
    layer.compute_gate_on_attention = False
    return layer
