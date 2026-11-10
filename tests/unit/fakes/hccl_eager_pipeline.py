# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU storage and packet/Work dependency seams; never an NPU progress model."""

from collections import defaultdict, deque
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from afd_plugin.connectors.npu.hccl_diagnostics import (
    HcclDiagnosticExecution,
    HcclStageDiagnostics,
)
from afd_plugin.connectors.npu.hccl_overlap import (
    HcclOverlapSettings,
    select_hccl_overlap,
)
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_plan import (
    HcclParentPlan,
    HcclPlanCapabilities,
    HcclPlanMessage,
)
from tests.unit.fakes.hccl import ScriptedStagedControlPeer
from tests.unit.fakes.hccl_diagnostics import StageDiagnosticRecorder
from tests.unit.fakes.npu_stream import HostStream, HostStreamRecorder


class HostStorageTensor(torch.Tensor):
    """Real CPU arithmetic with a host-only allocator ownership observation."""

    def record_stream(self, stream):
        stream.trace.append(("record_storage", stream.label, tuple(self.shape)))


@dataclass(frozen=True)
class EagerHostGroup:
    stage: int
    rank: int


class EagerHostEvent:
    def __init__(self, runtime):
        self.runtime = runtime
        self.recorded = False

    def record(self, stream):
        self.recorded = True
        self.runtime.trace.append(("record_event", stream.label))

    def wait(self, stream):
        if not self.recorded:
            raise RuntimeError("host seam waited an unrecorded event")
        self.runtime.trace.append(("wait_event", stream.label))


class EagerHostRuntime(HostStreamRecorder):
    def __init__(self):
        super().__init__(torch.device("cpu"))
        self.send = HostStream("send", self.parent.device)
        self.receive = HostStream("receive", self.parent.device)
        # The tensor subclass observes the destination stream without a global.
        for stream in (self.parent, self.compute, self.send, self.receive):
            object.__setattr__(stream, "trace", self.trace)

    def Event(self):  # noqa: N802 - native constructor boundary
        return EagerHostEvent(self)

    def synchronize(self, device):
        self.trace.append(("synchronize", device))

    def is_current_stream_capturing(self):
        return False


class EagerHostTorch:
    Tensor = torch.Tensor
    bfloat16, float32, int32 = torch.bfloat16, torch.float32, torch.int32

    def __init__(self, runtime):
        self.npu = runtime

    @staticmethod
    def device(device_type, index=None):
        return torch.device("cpu")

    @staticmethod
    def empty(shape, *, dtype, device):
        return torch.empty(shape, dtype=dtype, device=device).as_subclass(
            HostStorageTensor
        )

    @staticmethod
    def zeros(shape, *, dtype, device):
        return torch.zeros(shape, dtype=dtype, device=device).as_subclass(
            HostStorageTensor
        )


class EagerHostWork:
    def __init__(self, backend, action=None):
        self.backend, self.action = backend, action

    def wait(self):
        self.backend.trace.append(("work_wait", self.backend.runtime.current.label))
        if self.backend.fail_wait:
            raise RuntimeError("host Work failure")
        if self.action is not None:
            self.action()
            self.action = None
        return True


class EagerHostBackend:
    """Queued packet copies establish host protocol, not asynchronous device I/O."""

    def __init__(self, runtime):
        self.runtime, self.trace = runtime, []
        self.packets = defaultdict(deque)
        self.fail_wait = self.fail_warmup = False

    def isend(self, tensor, *, group, group_dst):
        self.trace.append(
            ("isend", group.rank, group_dst, group.stage, str(tensor.dtype))
        )
        self.packets[(group.stage, group.rank, group_dst)].append(tensor.clone())
        return EagerHostWork(self)

    def irecv(self, tensor, *, group, group_src):
        self.trace.append(
            ("irecv", group_src, group.rank, group.stage, str(tensor.dtype))
        )

        def deliver():
            tensor.copy_(self.packets[(group.stage, group_src, group.rank)].popleft())

        return EagerHostWork(self, deliver)

    def send(self, tensor, *, group, group_dst):
        self.trace.append(
            ("warm_send", group.rank, group_dst, group.stage, str(tensor.dtype))
        )
        if self.fail_warmup:
            raise RuntimeError("matched warmup failed")

    def recv(self, tensor, *, group, group_src):
        self.trace.append(
            ("warm_recv", group_src, group.rank, group.stage, str(tensor.dtype))
        )
        if self.fail_warmup:
            raise RuntimeError("matched warmup failed")

    def barrier(self, *, group):
        self.trace.append(("warm_barrier",))


def make_eager_pair(
    monkeypatch,
    owner_module,
    transport_module,
    *,
    rows=(2, 3),
    layers=(0, 1),
    valid=None,
    tp=1,
    mode="target",
):
    runtime = EagerHostRuntime()
    host_torch, backend = EagerHostTorch(runtime), EagerHostBackend(runtime)
    for module in (owner_module, transport_module):
        monkeypatch.setattr(module, "torch", host_torch)
        monkeypatch.setattr(module, "dist", backend)
        monkeypatch.setattr(module, "use_npu_current_stream", runtime.stream)
    caps = HcclPlanCapabilities(4, 32, max(1, *rows), layers, tp, tp, tp)
    valid = rows if valid is None else valid
    plan = HcclParentPlan(
        "eager-ticket-parent",
        layers,
        tuple(
            HcclStageLayout(i, (v,) * tp, (p,) * tp, tp, tp)
            for i, (v, p) in enumerate(zip(valid, rows, strict=True))
        ),
        mode,
    )
    incoming = {
        rank: [
            HcclPlanMessage(
                "ACCEPT",
                transaction_id=plan.transaction_id,
                plan_digest=plan.digest,
                world_rank=rank,
            ),
            HcclPlanMessage(
                "COMPLETE", transaction_id=plan.transaction_id, plan_digest=plan.digest
            ),
        ]
        for rank in range(2 * tp)
        if rank != tp
    }
    attention = ScriptedStagedControlPeer("attention", 0, caps, incoming)
    ffn = ScriptedStagedControlPeer(
        "ffn",
        0,
        caps,
        {
            tp: [
                HcclPlanMessage("FORWARD", plan=plan),
                HcclPlanMessage(
                    "ACCEPT",
                    transaction_id=plan.transaction_id,
                    plan_digest=plan.digest,
                    world_rank=tp,
                ),
                HcclPlanMessage(
                    "COMPLETE",
                    transaction_id=plan.transaction_id,
                    plan_digest=plan.digest,
                ),
            ]
        },
    )
    for peer in (attention, ffn):
        peer.data_groups = [EagerHostGroup(i, peer.world_rank) for i in range(2)]
        peer.control_group = "host-control"
    return SimpleNamespace(
        runtime=runtime,
        backend=backend,
        plan=plan,
        attention=attention,
        ffn=ffn,
        caps=caps,
    )


def open_eager_pair(case):
    for peer in (case.attention, case.ffn):
        peer.warmup_eager_pipeline(parent_stream=case.runtime.parent)
    case.attention.begin_forward(case.plan)
    assert case.ffn.receive_forward() == case.plan
    owners = tuple(
        peer.open_eager_pipeline(
            parent_stream=case.runtime.parent,
            send_stream=case.runtime.send,
            receive_stream=case.runtime.receive,
        )
        for peer in (case.attention, case.ffn)
    )
    return owners


def make_eager_progress(owner, compute_stream, *, enabled=True):
    """A real immutable diagnostic run, driven by the host packet seam."""
    recorder = StageDiagnosticRecorder()
    selection = select_hccl_overlap(
        HcclOverlapSettings(
            stage_diagnostics=enabled, ffn_compute_sync_diagnostics=enabled
        ),
        role=owner.transport.role,
        phase="eager",
        num_stages=len(owner.plan.stages),
    )
    actual = HcclDiagnosticExecution(
        "eager-streamed",
        "parent" if compute_stream == owner.parent_stream else "compute",
        "parent" if owner.send_stream == owner.parent_stream else "send",
        "parent" if owner.receive_stream == owner.parent_stream else "recv",
        (
            compute_stream == owner.send_stream,
            compute_stream == owner.receive_stream,
            owner.send_stream == owner.receive_stream,
        ),
    )
    progress = HcclStageDiagnostics(
        selection,
        world_rank=owner.transport.world_rank,
        role_rank=owner.transport.role_rank,
        emit=recorder.emit,
        synchronize=recorder.synchronize,
    ).bind_execution(
        layer_indices=owner.plan.layer_indices,
        transaction_id=owner.plan.transaction_id,
        actual_execution=actual,
    )
    return SimpleNamespace(progress=progress, recorder=recorder)


def eager_payload(rows, *, valid=None):
    valid = rows if valid is None else valid
    raw = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
    return tuple(
        tensor.as_subclass(HostStorageTensor)
        for tensor in (
            raw.to(torch.bfloat16),
            raw + 0.125,
            torch.cat(
                (
                    torch.arange(1, valid + 1, dtype=torch.int32),
                    torch.zeros(rows - valid, dtype=torch.int32),
                )
            ),
        )
    )


def run_eager_layer(case, attention, ffn, layer):
    tickets, expected = [], []
    for stage in case.plan.stages:
        rows = stage.attention_padded_tokens[0]
        if not rows:
            continue
        payload = eager_payload(rows, valid=stage.attention_valid_tokens[0])
        tickets.append(
            attention.post_attention(
                layer, stage.stage_idx, *payload, producer_stream=case.runtime.parent
            )
        )
        expected.append((payload[0].float() + payload[1]).to(torch.bfloat16))
    for stage in case.plan.stages:
        if not any(stage.attention_padded_tokens):
            continue
        ticket = ffn.post_ffn_receive(layer, stage.stage_idx)
        with ffn.ffn_inputs(ticket, consumer_stream=case.runtime.compute) as inputs:
            output = (inputs.hidden_states.float() + inputs.hidden_states_fp32).to(
                torch.bfloat16
            )
        ffn.post_ffn_output(ticket, output, producer_stream=case.runtime.compute)
    outputs = []
    for ticket in tickets:
        with attention.attention_output(
            ticket, consumer_stream=case.runtime.compute
        ) as output:
            outputs.append(output.clone())
    return outputs, expected
