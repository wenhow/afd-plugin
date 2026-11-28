# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU protocol peers for the old DeepSeek V4 role boundary."""

from types import SimpleNamespace

import torch
from torch import nn
from vllm.config import CompilationMode
from vllm.forward_context import override_forward_context

from afd_plugin.connectors import AFDA2FTransferPayload
from tests.unit.fakes.vllm_config import npu_vllm_config

ROUTER_DELTA_SCALE = 4_096
TARGET_HIDDEN_SIZE = 4
TARGET_HC_MULT = 2
TARGET_LAYERS = 2
TARGET_TOKEN_CAPACITY = 8


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


class TargetNormBoundary(nn.Module):
    """Keep model control/copy/HC arithmetic real; isolate the device norm."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs):
        return inputs


class TargetDecoderBoundary(nn.Module):
    def __init__(self, vllm_config, prefix, **kwargs):
        super().__init__()
        self.layer_idx = int(prefix.split(".")[-1])

    def forward(self, positions, hidden_states, residual, scaling, *, input_ids):
        return hidden_states + 1, residual


def make_target_buffer_case(
    monkeypatch, adapter, *, consumer="draft", last_rank=True, role="attention"
):
    """Construct the real role model and inherit its exact native forward.

    CPU constructors replace embedding/decoder/norm allocations and PP topology.
    Native speculative-buffer allocation, copy, HC collapse and public getter
    remain real; there is no native Attention/MoE or device acceptance here.
    """
    config = npu_vllm_config(role=role, model_type="deepseek_v4")
    hf = config.model_config.hf_config
    hf.hidden_size = TARGET_HIDDEN_SIZE
    hf.vocab_size = TARGET_TOKEN_CAPACITY
    hf.num_hidden_layers = TARGET_LAYERS
    hf.hc_mult = TARGET_HC_MULT
    hf.hc_eps = 0.01
    hf.rms_norm_eps = 1e-6
    config.model_config.dtype = torch.bfloat16
    config.compilation_config.mode = CompilationMode.NONE
    config.quant_config = None
    config.parallel_config.eplb_config = SimpleNamespace(num_redundant_experts=0)
    config.scheduler_config.max_num_batched_tokens = TARGET_TOKEN_CAPACITY
    config.speculative_config = (
        None
        if consumer is None
        else SimpleNamespace(
            use_eagle=lambda: consumer == "eagle",
            uses_draft_model=lambda: consumer == "draft",
        )
    )
    pp = SimpleNamespace(is_first_rank=True, is_last_rank=last_rank)

    def make_layers(count, factory, *, prefix):
        return 0, count, nn.ModuleList(factory(f"{prefix}.{i}") for i in range(count))

    native_module = adapter.native
    names = dict(vars(native_module))
    names.update(
        current_platform=SimpleNamespace(device_type="cpu"),
        get_pp_group=lambda: pp,
        VocabParallelEmbedding=lambda vocab, hidden, **kwargs: nn.Embedding(
            vocab, hidden
        ),
        make_layers=make_layers,
        RMSNorm=TargetNormBoundary,
        make_pp_empty_intermediate_tensors=lambda owner, factory: factory,
    )
    monkeypatch.setattr(adapter, "native", SimpleNamespace(**names))
    monkeypatch.setattr(adapter, "AFDDeepseekV4DecoderLayer", TargetDecoderBoundary)
    monkeypatch.setattr(adapter, "_refresh_ascend_fused_moe", lambda: None)
    # Inherited native forward resolves the native module's PP owner directly.
    monkeypatch.setattr(native_module, "get_pp_group", lambda: pp)
    model = adapter.AFDDeepseekV4Model(vllm_config=config)
    with torch.no_grad():
        model.hc_head_fn.zero_()
        model.hc_head_base.zero_()
        model.hc_head_scale.fill_(1)
    wrapper = adapter.AFDDeepseekV4ForCausalLM.__new__(adapter.AFDDeepseekV4ForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    return SimpleNamespace(model=model, wrapper=wrapper, config=config)
