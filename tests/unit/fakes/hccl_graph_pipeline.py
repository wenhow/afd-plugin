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
    monkeypatch,
    module,
    *,
    role="attention",
    rows=(2, 3),
    hybrid=None,
    policy=None,
    selection=None,
    exchange_order="stage_serial",
    layers=(0, 1),
):
    runtime = HostStreamRecorder(torch.device("cpu"))
    bank = make_graph_stage_bank(
        role=role,
        rows=rows,
        layers=layers,
        endpoint_factory=lambda stage: PipelineEndpoint(runtime, stage, role),
        exchange_order=exchange_order,
    )
    return attach_pipeline(
        monkeypatch,
        module,
        bank,
        hybrid=hybrid,
        policy=policy,
        selection=selection,
        runtime=runtime,
    )


def attach_pipeline(
    monkeypatch,
    module,
    bank,
    *,
    hybrid=None,
    policy=None,
    selection=None,
    runtime=None,
):
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
        bank,
        runtime.compute,
        events,
        tails,
        hybrid=hybrid,
        policy=policy,
        selection=selection,
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
                with (
                    pipeline.attention_send
                    if bank.exchange_order == "bulk_v1"
                    else pipeline.attention_exchange
                )(layer, stage, tensors=(x, router, ids)):
                    if bank.exchange_order == "bulk_v1":
                        bank.attention_send(stage, x, router, ids)
                    else:
                        returned[stage] = bank.attention_exchange(stage, x, router, ids)
            if bank.exchange_order == "bulk_v1":
                for stage in hidden:
                    with pipeline.attention_receive(layer, stage):
                        returned[stage] = bank.attention_receive(stage)
        for stage in hidden:
            with pipeline.final_compute(stage, inputs=(hidden[stage], returned[stage])):
                hidden[stage] = hidden[stage] + returned[stage]
        pipeline.join(outputs=tuple(hidden.values()))
    return tuple(hidden.values())


def run_ffn_layer(case, layer):
    """One actual CPU payload frontier, reusable by negative join seam tests."""
    bank, pipeline = case.bank, case.pipeline
    outputs = []
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
            output = inputs.hidden_states + inputs.hidden_states_fp32.to(torch.bfloat16)
        pipeline.retain_compute_outputs(layer, stage, tensors=(output,))
        if bank.exchange_order == "stage_serial":
            with pipeline.ffn_send(layer, stage, tensors=(output,)):
                bank.send_output(stage, output)
        outputs.append(output)
    if bank.exchange_order == "bulk_v1":
        for stage, output in zip(
            (stage for stage, rows in enumerate(bank.stage_rows) if rows),
            outputs,
            strict=True,
        ):
            with pipeline.ffn_send(layer, stage, tensors=(output,)):
                bank.send_output(stage, output)
    return tuple(outputs)


def run_ffn_arithmetic(case):
    bank, pipeline = case.bank, case.pipeline
    outputs = []
    with pipeline.run():
        for layer in bank.layer_indices:
            outputs.extend(run_ffn_layer(case, layer))
            if not pipeline.policy.ffn_cross_layer:
                pipeline.join_layer(layer)
        pipeline.join(outputs=tuple(outputs))
    return tuple(outputs)


def run_wavefront_arithmetic(case):
    """Run public cross-layer scopes with CPU values and queued peer packets."""
    bank, pipeline = case.bank, case.pipeline
    assert pipeline.schedule is not None
    hidden = {
        stage: torch.full_like(bank.buffers(stage).hidden_states, stage + 1)
        for stage, rows in enumerate(bank.stage_rows)
        if rows
    }
    returned, prepared, outputs = {}, {}, []

    def attention_compute(layer, stage):
        inputs = (hidden[stage],) + ((returned[stage],) if stage in returned else ())
        with pipeline.compute(layer, stage, inputs=inputs):
            if stage in returned:
                hidden[stage] = hidden[stage] + returned[stage]
            x = hidden[stage] + layer + 1
            prepared[stage] = (
                x,
                x.float() + 0.00031,
                torch.arange(x.shape[0], dtype=torch.int32),
            )
        pipeline.retain_compute_outputs(layer, stage, tensors=prepared[stage])

    with pipeline.run():
        if bank.role == "attention":
            for stage in hidden:
                attention_compute(bank.layer_indices[0], stage)
        for action in pipeline.schedule.actions:
            layer, stage = action.layer_idx, action.stage_idx
            if action.kind == "input":
                if bank.role == "attention":
                    if layer != bank.layer_indices[0]:
                        attention_compute(layer, stage)
                    x, router, ids = prepared.pop(stage)
                    bank.exchanges[stage].endpoint.incoming.append(x * 2)
                    with pipeline.attention_send(
                        layer, stage, tensors=(x, router, ids)
                    ):
                        bank.attention_send(stage, x, router, ids)
                else:
                    buffers = bank.buffers(stage)
                    bank.exchanges[stage].endpoint.incoming.extend(
                        (
                            torch.full_like(buffers.hidden_states, layer + stage + 1),
                            torch.full_like(buffers.hidden_states_fp32, 0.00031),
                            torch.arange(bank.stage_rows[stage], dtype=torch.int32),
                        )
                    )
                    with pipeline.ffn_receive(layer, stage):
                        buffers = bank.receive_inputs(stage)
                    with pipeline.compute(
                        layer,
                        stage,
                        inputs=(
                            buffers.hidden_states,
                            buffers.hidden_states_fp32,
                            buffers.input_ids,
                        ),
                    ):
                        output = buffers.hidden_states + buffers.hidden_states_fp32.to(
                            torch.bfloat16
                        )
                        output = bank.prepare_output(stage, output)
                    prepared[stage] = output
                    pipeline.retain_compute_outputs(layer, stage, tensors=(output,))
            elif bank.role == "attention":
                with pipeline.attention_receive(layer, stage):
                    returned[stage] = bank.attention_receive(stage)
            else:
                output = prepared.pop(stage)
                with pipeline.ffn_send(layer, stage, tensors=(output,)):
                    bank.send_output(stage, output)
                outputs.append(output.clone())
        if bank.role == "attention":
            for stage in hidden:
                with pipeline.final_compute(
                    stage, inputs=(hidden[stage], returned[stage])
                ):
                    hidden[stage] = hidden[stage] + returned[stage]
            outputs = list(hidden.values())
        pipeline.join(outputs=tuple(outputs))
    return tuple(outputs)
