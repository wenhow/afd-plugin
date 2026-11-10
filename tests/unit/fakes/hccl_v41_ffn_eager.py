# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Actual CPU packet/context seams for the native FFN eager consumer.

The existing real ticket owner executes against queued CPU packet copies and
host-only stream/event observations. No NPU progress or native MoE is simulated
as acceptance; the model boundary checks context and exposes CPU arithmetic.
"""

from threading import local
from types import SimpleNamespace

import torch
from vllm.utils import torch_utils

from afd_plugin.compat.npu.stream import use_npu_current_stream
from afd_plugin.distributed.hccl_plan import HcclPlanMessage
from tests.unit.fakes.hccl import (
    ScriptedStagedControlPeer,
    StagedFFNContractModel,
    make_staged_ffn_case,
)
from tests.unit.fakes.hccl_eager_pipeline import (
    EagerHostGroup,
    eager_payload,
    make_eager_pair,
    open_eager_pair,
)
from tests.unit.fakes.npu_stream import HostStream


class EagerFFNContractModel(StagedFFNContractModel):
    def __init__(self, context, plan, role_rank, runtime):
        super().__init__(context, plan, role_rank)
        self.runtime = runtime
        self.observations = []
        self.layers = plan.layer_indices

    def get_experts_layer_indices(self):
        return self.layers

    def compute_ffn_output(self, hidden_states, layer_idx, **kwargs):
        if not torch.is_inference_mode_enabled():
            raise RuntimeError("native eager FFN requires inference mode")
        if self.runtime.current is not self.runtime.compute:
            raise RuntimeError("native eager FFN requires its selected compute stream")
        if torch_utils.current_stream() is not self.runtime.compute:
            raise RuntimeError("native eager FFN requires its selected vLLM stream")
        output = super().compute_ffn_output(hidden_states, layer_idx, **kwargs)
        context = self.context.current
        self.observations.append(
            {
                "layer": layer_idx,
                "stage": context.afd_metadata.stage_idx,
                "ids": kwargs["input_ids"].clone(),
                "fp32": kwargs["hidden_states_fp32"].clone(),
                "output": output.clone(),
                "stream": self.runtime.current.label,
                "inference": torch.is_inference_mode_enabled(),
            }
        )
        return output


def make_ffn_eager_case(
    monkeypatch,
    owner_module,
    transport_module,
    consumer_module,
    runner_type,
    *,
    rows=(2, 3),
    valid=None,
    tp=1,
    role_rank=0,
):
    pair = make_eager_pair(
        monkeypatch, owner_module, transport_module, rows=rows, valid=valid, tp=tp
    )
    if role_rank:
        pair.ffn = ScriptedStagedControlPeer(
            "ffn",
            role_rank,
            pair.caps,
            {
                tp: [
                    HcclPlanMessage("FORWARD", plan=pair.plan),
                    HcclPlanMessage(
                        "ACCEPT",
                        transaction_id=pair.plan.transaction_id,
                        plan_digest=pair.plan.digest,
                        world_rank=tp,
                    ),
                    HcclPlanMessage(
                        "COMPLETE",
                        transaction_id=pair.plan.transaction_id,
                        plan_digest=pair.plan.digest,
                    ),
                ]
            },
        )
        pair.ffn.data_groups = [EagerHostGroup(i, role_rank) for i in range(2)]
        pair.ffn.control_group = "host-control"
    _, owner = open_eager_pair(pair)
    staged = make_staged_ffn_case(runner_type, pair.plan, role_rank=role_rank)
    model = EagerFFNContractModel(staged.context, pair.plan, role_rank, pair.runtime)
    staged.runner.model = model
    monkeypatch.setattr(consumer_module.torch, "npu", pair.runtime, raising=False)
    monkeypatch.setattr(
        consumer_module, "ascend_forward_context", staged.context.activate
    )
    monkeypatch.setattr(consumer_module, "torch", owner_module.torch)
    # CPU-only consumer needs real tensor construction/inference decorator, but
    # indexed NPU validation resolves through the already shared host facade.
    monkeypatch.setattr(owner_module.torch, "tensor", torch.tensor, raising=False)
    monkeypatch.setattr(owner_module, "use_npu_current_stream", use_npu_current_stream)
    cache = local()
    previous_stream = HostStream("cached-foreign", pair.runtime.parent.device)
    cache.value = previous_stream
    monkeypatch.setattr(torch_utils, "_current_stream_tls", cache)
    inputs = {}
    for layer in pair.plan.layer_indices:
        for stage in pair.plan.stages:
            count = stage.attention_padded_tokens[role_rank]
            if not count:
                continue
            payload = eager_payload(
                count, valid=stage.attention_valid_tokens[role_rank]
            )
            inputs[layer, stage.stage_idx] = payload
            group = EagerHostGroup(stage.stage_idx, tp + role_rank)
            for tensor in payload:
                pair.backend.isend(tensor, group=group, group_dst=role_rank)
    return SimpleNamespace(
        runner=staged.runner,
        model=model,
        context=staged.context,
        transport=pair.ffn,
        owner=owner,
        pair=pair,
        inputs=inputs,
        previous_stream=previous_stream,
        query_stream=torch_utils.current_stream,
    )
