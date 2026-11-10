# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Graph-service control/API seams, without a device capture implementation.

Graph API events describe host lifecycle only. The tensor bank's CPU storage
is exposed through an explicit allocation-boundary adapter; its reported NPU
placement is metadata for service ownership checks, never device evidence.
"""

import weakref
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm_ascend.ascend_forward_context import _EXTRA_CTX

from afd_plugin.connectors.npu.hccl_graph_cache import HcclGraphCache
from afd_plugin.connectors.npu.hccl_graph_pipeline import HcclGraphPipelinePolicy
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_graph import HcclGraphReport, decide_graph_execution
from afd_plugin.distributed.hccl_plan import HcclParentPlan, HcclPlanMessage
from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner
from afd_plugin.v1.worker.npu.hccl_v41_graph_service import HcclV41GraphService
from tests.unit.fakes.graph_cache import ReplayGraph
from tests.unit.fakes.hccl import ScriptedStagedControlPeer
from tests.unit.fakes.hccl_graph_control import make_graph_candidate
from tests.unit.fakes.hccl_graph_stage import (
    StageGraphEndpoint,
    make_graph_stage_bank,
    make_graph_stage_plan,
)
from tests.unit.fakes.hccl_v41_ffn_graph import make_ffn_graph_case
from tests.unit.fakes.vllm_config import hccl_vllm_config


class GraphServiceAPI:
    """Externally observable graph API ordering, with no replayed arithmetic."""

    def __init__(self, events):
        self.events, self.graphs = events, []
        self.body_flags = []
        self.capturing = False
        self.fail_capture = self.fail_replay = self.fail_body = self.fail_sync = False

    def synchronize(self, *args):
        self.events.append("device-sync")
        if self.fail_sync:
            raise RuntimeError("controlled device synchronization failure")

    def create(self):
        graph = ServiceReplayGraph(self)
        self.graphs.append(weakref.ref(graph))
        self.events.append("graph-created")
        return graph

    @contextmanager
    def capture(self, graph):
        self.events.append("capture-enter")
        if self.fail_capture:
            raise RuntimeError("controlled capture entry failure")
        self.capturing = True
        try:
            yield
            self.events.append("capture-exit")
        finally:
            self.capturing = False

    @contextmanager
    def forward_context(self, **kwargs):
        metadata = kwargs["afd_metadata"]
        context = ForwardContext(
            {},
            {},
            {},
            cudagraph_runtime_mode=kwargs["aclgraph_runtime_mode"],
            additional_kwargs={"afd_metadata": metadata},
        )
        context.input_ids = kwargs["input_ids"]
        with override_forward_context(context):
            yield context


class ServiceReplayGraph(ReplayGraph):
    def __init__(self, api):
        super().__init__()
        self.api = api

    def replay(self):
        self.api.events.append("graph-replay")
        if self.api.fail_replay:
            raise RuntimeError("controlled replay failure")
        super().replay()

    def reset(self):
        self.api.events.append("graph-reset")
        super().reset()


class ServiceGraphBanks:
    def __init__(self):
        self.allocations = []

    def allocate(self, transport, key, *, device):
        caps = transport.negotiated_capabilities

        def endpoint(index):
            peer = StageGraphEndpoint(
                role="ffn",
                role_rank=transport.role_rank,
                native_tp_size=caps.native_tp_size,
            )
            peer.group = transport.data_groups[index]
            return peer

        cpu = make_graph_stage_bank(
            role="ffn",
            rows=key.padded_tokens,
            hidden=caps.hidden_size,
            layers=caps.layer_indices,
            native_tp_size=caps.native_tp_size,
            role_rank=transport.role_rank,
            endpoint_factory=endpoint,
            exchange_order=caps.graph_exchange_order,
        )
        bank = SimpleNamespace(
            layer_indices=caps.layer_indices,
            device=device,
            key=key,
            capabilities=caps,
            role="ffn",
            role_rank=transport.role_rank,
            native_tp_size=caps.native_tp_size,
            exchanges=cpu.exchanges,
            stage_rows=cpu.stage_rows,
            valid_counts=cpu.valid_counts,
            buffers=cpu.buffers,
            update_counts=cpu.update_counts,
            exchange_order=cpu.exchange_order,
        )
        # SimpleNamespace cannot be weak-referenced; retain only its actual CPU
        # backing as a lifecycle observation, not the adapter itself.
        self.allocations.append(weakref.ref(cpu))
        return bank


class ServiceGraphBodies:
    def __init__(self, api):
        self.api, self.resources = api, []

    def __call__(
        self, model, contexts, bank, parallel, *, pipeline=None, diagnostics=None
    ):
        body = ServiceGraphBody(
            self.api,
            model,
            contexts,
            bank,
            pipeline=pipeline,
            diagnostics=diagnostics,
        )
        self.resources.append(weakref.ref(body))
        return body


class ServiceGraphBody:
    def __init__(self, api, model, contexts, bank, *, pipeline=None, diagnostics=None):
        self.api, self.model, self.contexts, self.bank = api, model, contexts, bank
        self.pipeline = pipeline
        if diagnostics is not None:
            diagnostics.bind_execution(
                layer_indices=bank.layer_indices,
                actual_compute_stream="compute" if pipeline is not None else "parent",
                is_warmup=diagnostics.selection.phase == "graph_warmup",
            )

    def run(self):
        self.api.events.append("tensor-body")
        if self.api.fail_body:
            raise RuntimeError("controlled tensor body failure")
        for index, (rows, context) in enumerate(
            zip(self.bank.stage_rows, self.contexts, strict=True)
        ):
            if (
                context.cudagraph_runtime_mode != CUDAGraphMode.FULL
                or context.ubatch_idx != index
                or (
                    rows and context.input_ids is not self.bank.buffers(index).input_ids
                )
            ):
                raise ValueError("service did not retain its prepared FULL context")
            with override_forward_context(context):
                self.api.body_flags.append(_EXTRA_CTX.capturing)


@dataclass(frozen=True)
class ServiceGraphPipeline:
    """Resource identity only; no stream, event or device work is simulated."""

    bank: SimpleNamespace
    policy: HcclGraphPipelinePolicy

    @property
    def hybrid(self):
        return self.policy.hybrid


class ServiceGraphPipelines:
    def __init__(self, api):
        self.api = api

    def allocate(self, bank, *, policy=None, selection=None):
        if self.api.capturing:
            raise RuntimeError("pipeline allocation entered graph capture")
        if selection is not None:
            policy = HcclGraphPipelinePolicy.from_selection(selection)
        return ServiceGraphPipeline(bank, policy)


class FFNGraphControlPeer(ScriptedStagedControlPeer):
    def __init__(self, bank, events):
        super().__init__(
            "ffn", bank.role_rank, bank.capabilities, {bank.capabilities.ffn_size: []}
        )
        self.data_groups = [exchange.endpoint.group for exchange in bank.exchanges]
        self.events, self.base_bank = events, bank
        self.before_forward = None

    def close(self):
        """Observe caller-owned group release without simulating device groups."""
        if not self.closed:
            self.events.append("groups-close")
            self.closed = self.terminal = True
            self.data_groups, self.control_group = [], None

    def _recv_control(self, rank):
        message = super()._recv_control(rank)
        if message.kind == "FORWARD" and self.before_forward is not None:
            self.before_forward()
        return message

    def queue_parent(
        self,
        phase,
        *,
        identifier,
        valid=(2, 2),
        fallback=False,
        capture_sequence_digest=None,
        availability=None,
    ):
        candidate = make_graph_candidate(
            phase=phase,
            mode="warmup" if phase in ("warmup", "capture") else "target",
            rows=self.base_bank.stage_rows,
            eager_rows=(0,) * len(valid) if fallback else None,
        )
        if capture_sequence_digest is not None:
            candidate = replace(
                candidate,
                mode="warmup",
                eager_padded_tokens=(0,),
                capture_sequence_digest=capture_sequence_digest,
            )
        state = availability or (
            "missing" if fallback or phase in ("warmup", "capture") else "ready"
        )
        decision = decide_graph_execution(
            identifier, candidate, (HcclGraphReport(state),)
        )
        if fallback:
            stages = tuple(
                HcclStageLayout(index, (0,), (0,), 1) for index in range(len(valid))
            )
            plan = HcclParentPlan(
                identifier,
                self.base_bank.layer_indices,
                stages,
                mode=decision.mode,
                graph_execution=decision,
            )
        else:
            plan = replace(
                make_graph_stage_plan(
                    self.base_bank, valid=valid, transaction_id=identifier
                ),
                mode=decision.mode,
                graph_execution=decision,
            )
        self.incoming[self.coordinator_rank].extend(
            (
                HcclPlanMessage(
                    "GRAPH_QUERY", transaction_id=identifier, graph_candidate=candidate
                ),
                HcclPlanMessage("GRAPH_DECISION", graph_decision=decision),
                HcclPlanMessage(
                    "GRAPH_ACK",
                    transaction_id=identifier,
                    world_rank=self.coordinator_rank,
                    graph_digest=decision.digest,
                ),
                HcclPlanMessage("FORWARD", plan=plan),
                HcclPlanMessage(
                    "ACCEPT",
                    transaction_id=identifier,
                    plan_digest=plan.digest,
                    world_rank=self.coordinator_rank,
                ),
                HcclPlanMessage(
                    "COMPLETE", transaction_id=identifier, plan_digest=plan.digest
                ),
            )
        )
        return candidate, decision, plan

    def queue_stop(self):
        self.incoming[self.coordinator_rank].extend(
            (HcclPlanMessage("STOP"), HcclPlanMessage("STOP_ACK"))
        )

    def queue_rejection(self, phase, *, error, capture_sequence_digest=None, rows=None):
        candidate = make_graph_candidate(
            phase=phase, rows=self.base_bank.stage_rows if rows is None else rows
        )
        if capture_sequence_digest is not None:
            candidate = replace(
                candidate,
                mode="warmup",
                eager_padded_tokens=(0,),
                capture_sequence_digest=capture_sequence_digest,
            )
        self.incoming[self.coordinator_rank].extend(
            (
                HcclPlanMessage(
                    "GRAPH_QUERY", transaction_id="reject-1", graph_candidate=candidate
                ),
                HcclPlanMessage("ERROR", error=error),
            )
        )

    def _send_control(self, rank, message):
        self.events.append(message.kind)
        super()._send_control(rank, message)


def make_graph_service_case(
    service_module,
    monkeypatch,
    *,
    pipeline_policy=None,
    overlap_settings=None,
    rows=(2, 2),
    diagnostic_sink=None,
    exchange_order="stage_serial",
):
    events = []
    model_case = make_ffn_graph_case(
        rows=rows, valid=rows, exchange_order=exchange_order, build_driver=False
    )
    transport = FFNGraphControlPeer(model_case.bank, events)
    config = hccl_vllm_config(
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        role="ffn",
    )
    config.parallel_config.rank = 0
    config.model_config.enforce_eager = False
    config.model_config.hf_text_config.hidden_size = 4
    config.model_config.hf_text_config.vocab_size = 128
    runner = object.__new__(AFDNPUFFNModelRunner)
    runner.vllm_config = config
    runner.model = model_case.model
    runner.device = torch.device("npu", 0)
    runner.connector = SimpleNamespace(is_initialized=False)
    runner.graph_pool = None
    runner.hccl_v41_graph_service = None
    runner.hccl_v41_eager_session = None
    runner.hccl_v41_diagnostics = None
    runner.prof = None
    api, banks = GraphServiceAPI(events), ServiceGraphBanks()
    cache = HcclGraphCache(api.synchronize)
    bodies = ServiceGraphBodies(api)
    monkeypatch.setattr(service_module, "HcclGraphStageBank", banks)
    monkeypatch.setattr(service_module, "HcclV41FFNGraph", bodies)
    if pipeline_policy is not None or overlap_settings is not None:
        monkeypatch.setattr(
            service_module, "HcclGraphComputePipeline", ServiceGraphPipelines(api)
        )
    monkeypatch.setattr(service_module, "ascend_forward_context", api.forward_context)
    monkeypatch.setattr(torch.npu, "synchronize", api.synchronize)
    generation = SimpleNamespace(value=0)
    service = HcclV41GraphService(
        runner,
        transport,
        cache,
        lambda: generation.value,
        pipeline_policy=pipeline_policy,
        overlap_settings=overlap_settings,
        diagnostic_sink=diagnostic_sink,
        graph_factory=api.create,
        graph_capture=api.capture,
    )
    return SimpleNamespace(
        service=service,
        runner=runner,
        transport=transport,
        cache=cache,
        api=api,
        banks=banks,
        bodies=bodies,
        events=events,
        generation=generation,
    )
