# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU allocation/proposal boundaries; no native attention/MoE/device claim."""

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from vllm.config import CompilationMode
from vllm.config.compilation import CUDAGraphMode
from vllm_ascend.models.deepseek_v4 import model as native_target
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator

from afd_plugin.compat.npu.mtp import resolve_deepseek_v4_mtp_contract
from afd_plugin.v1.worker.npu.hccl_mtp import HcclMtpSession, MtpCompletedTarget
from tests.unit.fakes.mtp import ordinary_mtp_config


class HostLinear(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(output_size, input_size))

    def forward(self, inputs):
        return torch.nn.functional.linear(inputs, self.weight)


class HostNorm(nn.Module):
    def __init__(self, hidden_size, *, has_weight=True, **kwargs):
        super().__init__()
        if has_weight:
            self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, inputs):
        return inputs


class HostEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(vocab_size, hidden_size))

    def forward(self, input_ids):
        return torch.nn.functional.embedding(input_ids.long(), self.weight)


class HostSharedHead(nn.Module):
    def __init__(self, *, config, **kwargs):
        super().__init__()
        self.head = HostEmbedding(config.vocab_size, config.hidden_size)
        self.norm = HostNorm(config.hidden_size)

    def forward(self, inputs):
        return self.norm(inputs)


class HostLogits:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, head, inputs):
        return torch.nn.functional.linear(inputs, head.weight)


class HostMoE(native_target.DeepseekV4MoE):
    """Complete-MoE arithmetic boundary; preserves observable FP32 and IDs."""

    def __init__(self, config, parallel_config, *, prefix, is_draft_layer, **kwargs):
        nn.Module.__init__(self)
        self.is_draft = is_draft_layer
        self.n_logical_experts = self.n_physical_experts = config.n_routed_experts
        self.n_local_physical_experts = config.n_routed_experts
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.n_redundant_experts = 0
        self.gate = HostLinear(config.hidden_size, config.n_routed_experts)
        self.experts = HostLinear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, input_ids=None, hidden_states_fp32=None):
        fp32 = (
            hidden_states.float() if hidden_states_fp32 is None else hidden_states_fp32
        )
        return ((fp32 - hidden_states.float()) * 4096 + input_ids[:, None]).to(
            hidden_states.dtype
        )


class HostAttentionDecoder(native_target.DeepseekV4DecoderLayer):
    def __init__(
        self,
        vllm_config,
        prefix,
        config=None,
        topk_indices_buffer=None,
        is_draft_layer=False,
    ):
        nn.Module.__init__(self)
        self.use_sequence_parallel_moe = False
        self.self_attn = HostLinear(8, 8)
        self.mlp = nn.Identity()

    def forward(self, *, hidden_states, **kwargs):
        return hidden_states + 1, None


def install_host_mtp_allocations(monkeypatch, adapter):
    """Use native role constructors/introspection with explicit CPU operators."""
    names = dict(vars(adapter.native))
    names.update(
        ReplicatedLinear=HostLinear,
        RMSNorm=HostNorm,
        VocabParallelEmbedding=HostEmbedding,
        SharedHead=HostSharedHead,
        LogitsProcessor=HostLogits,
        DeepseekV4MoE=HostMoE,
        current_platform=SimpleNamespace(device_type="cpu"),
    )
    target_names = dict(vars(adapter.native_target))
    target_names.update(DeepseekV4MoE=HostMoE)
    monkeypatch.setattr(adapter, "native", SimpleNamespace(**names))
    monkeypatch.setattr(adapter, "native_target", SimpleNamespace(**target_names))
    monkeypatch.setattr(adapter, "AFDDeepseekV4DecoderLayer", HostAttentionDecoder)
    monkeypatch.setattr(adapter, "_refresh_ascend_fused_moe", lambda: None)


def mtp_runtime_config(*, steps=2, role="attention", tp_size=1):
    config = ordinary_mtp_config(steps=steps, role=role, tp_size=tp_size)
    config.compilation_config.mode = CompilationMode.NONE
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    return config


class HostMtpExchange:
    def __init__(self):
        self.transport = SimpleNamespace(
            role="attention",
            initialized=True,
            closed=False,
            terminal=False,
            plan=None,
            graph_reservation=None,
        )
        self.records = []
        self.acknowledged_stop = False

    @contextmanager
    def draft_forward(self, call, input_ids, positions):
        self.records.append(("forward", call, input_ids.clone(), positions.clone()))
        yield
        self.records.append(("complete", call.logical_step))

    def finish_proposal(self, transaction_id, steps):
        self.records.append(("end", transaction_id, steps))

    def fail(self):
        self.transport.terminal = True
        self.records.append(("failed",))


def host_native_propose(self, **kwargs):
    """Stand-in for native proposal body; tests do not claim sampling coverage."""
    if self.result_only:
        return self.draft_result
    batch = kwargs["input_batch"]
    hidden = kwargs["last_hidden_states"]
    for step in range(self.forward_limit):
        rows = batch.num_tokens_after_padding if step == 0 else batch.num_reqs
        ids = torch.arange(rows, dtype=torch.int32) + step
        positions = torch.arange(rows)
        hidden = self.model(ids, positions, hidden[:rows])
    self.exchange.records.append(("native_return",))
    if self.fail_after_models:
        raise RuntimeError("native feedback/head failed")
    return self.draft_result


@dataclass
class MtpSessionCase:
    owner: SimpleNamespace
    session: HcclMtpSession
    exchange: HostMtpExchange
    input_batch: SimpleNamespace
    propose_kwargs: dict


def make_mtp_session_case(monkeypatch, adapter, *, steps=2, mode="target"):
    install_host_mtp_allocations(monkeypatch, adapter)
    monkeypatch.setattr(AscendAutoRegressiveSpeculator, "propose", host_native_propose)
    config = mtp_runtime_config(steps=steps)
    model = adapter.AFDDeepSeekV4MTP(vllm_config=config)
    exchange = HostMtpExchange()
    speculator = object.__new__(AscendMTPSpeculator)
    speculator.model, speculator.exchange = model, exchange
    speculator.prefill_cudagraph_manager = None
    speculator.decode_cudagraph_manager = None
    speculator.forward_limit = steps
    speculator.fail_after_models = speculator.result_only = False
    speculator.draft_result = torch.arange(2 * steps).view(2, steps)
    owner = SimpleNamespace(
        vllm_config=config, model=nn.Identity(), speculator=speculator
    )
    session = HcclMtpSession(owner, resolve_deepseek_v4_mtp_contract(config), exchange)
    model.bind_mtp_session(session)
    session.bind_proposer()
    batch = SimpleNamespace(num_reqs=2, num_tokens=5, num_tokens_after_padding=6)
    session.record_completed_target(MtpCompletedTarget("target-1", batch, mode))
    vector = torch.zeros(2, dtype=torch.int32)
    kwargs = dict(
        input_batch=batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.ones(6, 16),
        aux_hidden_states=None,
        num_sampled=vector,
        num_rejected=vector,
        last_sampled=vector,
        next_prefill_tokens=vector,
        temperature=vector.float(),
        seeds=vector,
        dummy_run=mode != "target",
        is_profile=mode == "profile",
    )
    return MtpSessionCase(owner, session, exchange, batch, kwargs)
