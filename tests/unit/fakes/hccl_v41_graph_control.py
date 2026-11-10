# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native CPU dispatch/cache plus scripted peer messages for A graph control."""

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from afd_plugin.connectors.npu.hccl_graph_cache import HcclGraphCache
from afd_plugin.distributed.hccl_graph import HcclGraphReport, decide_graph_execution
from afd_plugin.distributed.hccl_plan import (
    HcclLocalForward,
    HcclPlanCapabilities,
    HcclPlanMessage,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph_control import HcclV41GraphControl
from tests.unit.fakes.batch_req_state import make_batch_req_state
from tests.unit.fakes.cudagraph_manager import make_native_graph_manager
from tests.unit.fakes.graph_cache import DeviceCompletion, GraphResources, ReplayGraph
from tests.unit.fakes.hccl import ScriptedStagedControlPeer
from tests.unit.fakes.hccl_graph_control import make_graph_key
from tests.unit.fakes.vllm_config import npu_parallel_config


@dataclass
class NativeGraphGeneration:
    value: int = 0

    def __call__(self):
        return self.value


@dataclass
class NativeGraphMetadata:
    shapes: tuple[tuple[int, ...], ...] = ()

    def __call__(self, geometry):
        return self.shapes


class AttentionGraphControlPeer(ScriptedStagedControlPeer):
    """External peers reply to real encoded QUERY/DECISION; no graph execution."""

    def __init__(self, role_rank=0, *, states=None, incoming=None, capabilities=None):
        capabilities = capabilities or HcclPlanCapabilities(
            16,
            128,
            16,
            (0, 2),
            2,
            2,
            2,
            local_counts_version=1,
            graph_execution_version=1,
        )
        super().__init__("attention", role_rank, capabilities, incoming or {})
        self.states = states or {0: "ready", 1: "ready", 3: "ready"}

    def _send_control(self, rank, message):
        super()._send_control(rank, message)
        queue = self.incoming.setdefault(rank, deque())
        if message.kind == "GRAPH_QUERY":
            queue.append(
                HcclPlanMessage(
                    "GRAPH_REPORT",
                    transaction_id=message.transaction_id,
                    world_rank=rank,
                    graph_digest=message.graph_candidate.digest,
                    graph_report=HcclGraphReport(self.states[rank]),
                )
            )
        elif message.kind == "GRAPH_DECISION":
            queue.append(
                HcclPlanMessage(
                    "GRAPH_ACK",
                    transaction_id=message.graph_decision.transaction_id,
                    world_rank=rank,
                    graph_digest=message.graph_decision.digest,
                )
            )
        elif message.kind == "COUNTS_REQUEST":
            decision = self.graph_reservation
            queue.append(
                HcclPlanMessage(
                    "COUNTS_REPORT",
                    transaction_id=message.transaction_id,
                    world_rank=rank,
                    local_forward=HcclLocalForward(
                        decision.mode,
                        (0,) * len(decision.padded_tokens),
                        decision.padded_tokens,
                    ),
                )
            )
        elif message.kind == "FORWARD":
            queue.extend(
                (
                    HcclPlanMessage(
                        "ACCEPT",
                        transaction_id=message.plan.transaction_id,
                        world_rank=rank,
                        plan_digest=message.plan.digest,
                    ),
                    HcclPlanMessage(
                        "COMPLETE",
                        transaction_id=message.plan.transaction_id,
                        plan_digest=message.plan.digest,
                    ),
                )
            )


def make_attention_graph_control_case(
    *,
    rows=(2, 2),
    scheduled=(1, 1, 1, 1),
    descriptor=None,
    cached=True,
    pending=False,
    cache_generation=0,
    generation=0,
    local_descriptors=None,
    role_rank=0,
    candidate=None,
    availability="ready",
    metadata_shapes=(),
    rejection=False,
    peer_states=None,
    capabilities=None,
):
    target_query_len = 1 if capabilities is None else capabilities.target_query_len
    descriptor = descriptor or BatchExecutionDescriptor(
        CUDAGraphMode.FULL,
        sum(rows),
        sum(rows) // target_query_len,
        uniform_token_count=target_query_len,
        num_ubatches=len(rows),
    )
    key = make_graph_key(
        rows=rows,
        metadata_shapes=metadata_shapes,
        target_query_len=target_query_len,
    )
    graph = ReplayGraph()
    cache = HcclGraphCache(DeviceCompletion().synchronize)
    if cached or pending:
        cache.reserve_capture(key, cache_generation, graph, GraphResources())
        if not pending:
            cache.publish_capture(key, cache_generation)
    descriptors = (descriptor,) if local_descriptors is None else local_descriptors
    manager = make_native_graph_manager(
        descriptors, {descriptor: graph} if cached else {}
    )
    incoming = None
    if role_rank:
        if candidate is None:
            incoming = {2: [HcclPlanMessage("STOP"), HcclPlanMessage("STOP_ACK")]}
        elif rejection:
            incoming = {
                2: [
                    HcclPlanMessage(
                        "GRAPH_QUERY",
                        transaction_id="batch-1",
                        graph_candidate=candidate,
                    ),
                    HcclPlanMessage("ERROR", error="local native geometry rejected"),
                ]
            }
        else:
            decision = decide_graph_execution(
                "batch-1", candidate, (HcclGraphReport(availability),)
            )
            incoming = {
                2: [
                    HcclPlanMessage(
                        "GRAPH_QUERY",
                        transaction_id="batch-1",
                        graph_candidate=candidate,
                    ),
                    HcclPlanMessage("GRAPH_DECISION", graph_decision=decision),
                    HcclPlanMessage(
                        "GRAPH_ACK",
                        transaction_id="batch-1",
                        world_rank=2,
                        graph_digest=decision.digest,
                    ),
                ]
            }
    transport = AttentionGraphControlPeer(
        role_rank, incoming=incoming, states=peer_states, capabilities=capabilities
    )
    epoch = NativeGraphGeneration(generation)
    metadata = NativeGraphMetadata(metadata_shapes)
    control = HcclV41GraphControl(
        transport,
        manager,
        cache,
        epoch,
        metadata_shapes=metadata,
    )
    return SimpleNamespace(
        control=control,
        transport=transport,
        manager=manager,
        cache=cache,
        generation=epoch,
        metadata=metadata,
        descriptor=descriptor,
        key=key,
        graph=graph,
        batch=make_batch_req_state(scheduled),
        parallel=npu_parallel_config(
            use_ubatching=True,
            num_ubatches=2,
            dbo_decode_token_threshold=1,
            dbo_prefill_token_threshold=1,
        ),
    )
