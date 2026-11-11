# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native runner lifecycle boundaries using shared host graph builders."""

from collections import deque

from afd_plugin.connectors.npu.hccl_graph_cache import HcclGraphCache
from afd_plugin.connectors.npu.hccl_staged import HcclStagedAFDConnector
from afd_plugin.distributed.hccl_plan import HcclPlanMessage
from afd_plugin.v1.worker.npu.hccl_v41_graph_control import HcclV41GraphControl
from afd_plugin.v1.worker.npu.hccl_v41_graph_coordinator import HcclV41GraphCoordinator
from afd_plugin.v1.worker.npu.hccl_v41_graph_eager import HcclV41GraphEager
from tests.unit.fakes.hccl_v41_graph_coordinator import make_coordinator_case
from tests.unit.fakes.hccl_v41_graph_eager import make_graph_eager_case


def make_graph_runner_case(*, fail_close=False, stages=1, target_query_len=1):
    """Use the actual AFD runner with CPU-only model/control collaborators."""
    case = make_graph_eager_case(stages=stages, target_query_len=target_query_len)
    # Binding precedes real KV/native runner construction, even for U2 config.
    case.owner.ubatch_runner = None
    case.events = []

    def synchronize():
        case.events.append("graph-sync")
        if fail_close:
            raise RuntimeError("graph synchronization failed")

    case.cache = HcclGraphCache(synchronize)
    case.control = HcclV41GraphControl(
        case.transport,
        None,
        case.cache,
        case.generation,
        metadata_shapes=case.metadata,
    )
    case.eager = HcclV41GraphEager(
        case.owner, case.control, next_transaction_id=case.next_transaction_id
    )
    # This builder owns explicit binder tests, so its replacement control/eager
    # starts unbound even when the lower-level execution builder preselects one.
    case.owner.hccl_v41_graph_eager = None
    connector = object.__new__(HcclStagedAFDConnector)
    connector.transport = case.transport
    connector._pending_context = connector._pending_tensor_spec = None
    case.owner.connector = connector
    case.owner._afd_pending_metadata = None

    def close_groups():
        case.events.append("groups-close")
        case.transport.closed = True

    case.transport.close = close_groups
    return case


def install_graph_runner_coordinator(case):
    """Install host manager/storage identities; no real KV or NPU capture."""
    native = make_coordinator_case(
        stages=1, target_query_len=case.owner.decode_query_len
    ).manager
    native.model_runner = case.owner
    native.ubatch_runner = None
    case.owner.cudagraph_manager = native
    control = HcclV41GraphControl(
        case.transport,
        native,
        case.cache,
        case.generation,
        metadata_shapes=case.metadata,
    )
    coordinator = HcclV41GraphCoordinator(
        case.owner,
        control,
        case.session,
        next_transaction_id=case.next_transaction_id,
    )
    return coordinator


def queue_graph_runner_stop(case):
    """Script the external FFN acknowledgement to the real STOP handshake."""
    case.transport.incoming.setdefault(0, deque()).append(HcclPlanMessage("STOP_ACK"))
