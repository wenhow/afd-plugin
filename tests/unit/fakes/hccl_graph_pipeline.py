# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host stream/event call recorder; it does not execute or simulate NPU tasks."""

from dataclasses import dataclass

import torch

from afd_plugin.connectors.npu.hccl_graph_pipeline import (
    HcclGraphComputePipeline,
    HcclGraphPipelineEvents,
    HcclGraphPipelineTailEvents,
)
from tests.unit.fakes.hccl_graph_stage import (
    StageGraphEndpoint,
    make_graph_stage_bank,
)
from tests.unit.fakes.npu_stream import HostStream as HostStream
from tests.unit.fakes.npu_stream import HostStreamRecorder


class HostEvent:
    def __init__(self, runtime, label):
        self.runtime, self.label = runtime, label
        self.recorded = False

    def record(self, stream):
        if self.recorded:
            raise RuntimeError("host trace event was recorded more than once")
        self.runtime.trace.append(("record", self.label, stream.label))
        self.recorded = True

    def wait(self, stream):
        if not self.recorded:
            raise RuntimeError("host trace waited an unrecorded event")
        self.runtime.trace.append(("wait", self.label, stream.label))


class PipelineEndpoint(StageGraphEndpoint):
    def __init__(self, runtime, stage, role):
        super().__init__(role=role)
        self.runtime, self.stage = runtime, stage

    def send(self, tensor, shape):
        self.runtime.trace.append(
            ("wire", "send", self.stage, self.runtime.current.label)
        )
        super().send(tensor, shape)

    def receive(self, tensor, shape):
        self.runtime.trace.append(
            ("wire", "receive", self.stage, self.runtime.current.label)
        )
        super().receive(tensor, shape)


@dataclass
class PipelineCase:
    pipeline: HcclGraphComputePipeline
    runtime: HostStreamRecorder

    @property
    def bank(self):
        return self.pipeline.bank


def make_pipeline_case(
    monkeypatch, module, *, role="attention", rows=(2, 3), hybrid=True
):
    runtime = HostStreamRecorder(torch.device("cpu"))
    bank = make_graph_stage_bank(
        role=role,
        rows=rows,
        endpoint_factory=lambda stage: PipelineEndpoint(runtime, stage, role),
    )
    return attach_pipeline(monkeypatch, module, bank, hybrid=hybrid, runtime=runtime)


def attach_pipeline(monkeypatch, module, bank, *, hybrid=True, runtime=None):
    """Compose the same event seam with a caller's actual existing bank."""
    runtime = HostStreamRecorder(bank.device) if runtime is None else runtime
    events = {
        (layer, stage): HcclGraphPipelineEvents(
            *(
                HostEvent(runtime, f"L{layer}S{stage}.{name}")
                for name in ("ready", "compute", "receive", "send")
            )
        )
        for layer in bank.layer_indices
        for stage, count in enumerate(bank.stage_rows)
        if count
    }
    tails = {
        stage: HcclGraphPipelineTailEvents(
            HostEvent(runtime, f"S{stage}.final_ready"),
            HostEvent(runtime, f"S{stage}.tail"),
        )
        for stage, count in enumerate(bank.stage_rows)
        if count
    }
    pipeline = HcclGraphComputePipeline(
        bank, runtime.compute, events, tails, hybrid=hybrid
    )
    if hasattr(module.torch, "npu"):
        monkeypatch.setattr(module.torch.npu, "current_stream", runtime.current_stream)
        monkeypatch.setattr(module.torch.npu, "stream", runtime.stream)
    else:
        monkeypatch.setattr(module.torch, "npu", runtime, raising=False)
    return PipelineCase(pipeline, runtime)


def run_attention_arithmetic(case):
    """Actual CPU bank copies and arithmetic, with explicit host event boundaries."""
    bank, pipeline = case.bank, case.pipeline
    hidden = {
        stage: torch.full_like(bank.buffers(stage).hidden_states, stage + 1)
        for stage, rows in enumerate(bank.stage_rows)
        if rows
    }
    returned, prepared = {}, {}
    with pipeline.run():
        for layer in bank.layer_indices:
            for stage in hidden:
                inputs = (hidden[stage],) + ((returned[stage],) if layer else ())
                with pipeline.compute(layer, stage, inputs=inputs):
                    if layer:
                        hidden[stage] = hidden[stage] + returned[stage]
                    x = hidden[stage] + layer + 1
                    prepared[stage] = (
                        x,
                        x.float() + 0.00031,
                        torch.arange(x.shape[0], dtype=torch.int32),
                    )
            for stage in hidden:
                x, router, ids = prepared[stage]
                bank.exchanges[stage].endpoint.incoming.append(x * 2)
                with pipeline.attention_exchange(
                    layer, stage, tensors=(x, router, ids)
                ):
                    returned[stage] = bank.attention_exchange(stage, x, router, ids)
        for stage in hidden:
            with pipeline.final_compute(stage, inputs=(hidden[stage], returned[stage])):
                hidden[stage] = hidden[stage] + returned[stage]
        pipeline.join(outputs=tuple(hidden.values()))
    return tuple(hidden.values())


def run_ffn_arithmetic(case):
    bank, pipeline = case.bank, case.pipeline
    outputs = []
    with pipeline.run():
        for layer in bank.layer_indices:
            for stage, rows in enumerate(bank.stage_rows):
                if not rows:
                    continue
                buffers = bank.buffers(stage)
                endpoint = bank.exchanges[stage].endpoint
                endpoint.incoming.extend(
                    (
                        torch.full_like(buffers.hidden_states, layer + stage + 1),
                        torch.full_like(buffers.hidden_states_fp32, 0.00031),
                        torch.arange(rows, dtype=torch.int32),
                    )
                )
                with pipeline.ffn_receive(layer, stage):
                    inputs = bank.receive_inputs(stage)
                with pipeline.compute(
                    layer,
                    stage,
                    inputs=(
                        inputs.hidden_states,
                        inputs.hidden_states_fp32,
                        inputs.input_ids,
                    ),
                ):
                    output = inputs.hidden_states + inputs.hidden_states_fp32.to(
                        torch.bfloat16
                    )
                with pipeline.ffn_send(layer, stage, tensors=(output,)):
                    bank.send_output(stage, output)
                outputs.append(output)
        pipeline.join(outputs=tuple(outputs))
    return tuple(outputs)
