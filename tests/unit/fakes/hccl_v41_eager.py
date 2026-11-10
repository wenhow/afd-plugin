# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU configuration/session selection collaborator; no stream execution."""

from types import SimpleNamespace

import torch
from vllm.config import CompilationMode, CUDAGraphMode

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.npu.hccl_config import validate_staged_hccl_runtime
from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_plan import HcclParentPlan
from afd_plugin.v1.worker.npu.hccl_v41_eager import HcclV41EagerSession
from tests.unit.fakes.npu_stream import HostStream
from tests.unit.fakes.vllm_config import hccl_vllm_config


class EagerSessionTransport:
    def __init__(self, config, role):
        self.role, self.role_rank, self.local_device = role, 0, 0
        self.initialized, self.closed, self.terminal = True, False, False
        self.plan = self.graph_reservation = None
        self.negotiated_capabilities = validate_staged_hccl_runtime(
            config, parse_afd_config(config), 0
        )
        self.messages = []

    def warmup_eager_pipeline(self, *, parent_stream):
        self.messages.append(("warmup", parent_stream))

    def open_eager_pipeline(self, **streams):
        self.messages.append(("open", streams))
        return SimpleNamespace(transport=self, plan=self.plan)


def make_eager_session_case(*, role="attention", enabled=True):
    config = hccl_vllm_config(
        role=role,
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
    )
    config.parallel_config.use_ubatching = True
    config.parallel_config.num_ubatches = 2
    config.parallel_config.rank = 0
    transport = EagerSessionTransport(config, role)
    streams = {
        name: HostStream(name, torch.device("npu", 0))
        for name in ("parent_stream", "compute_stream", "send_stream", "receive_stream")
    }
    session = HcclV41EagerSession(
        config,
        transport,
        HcclOverlapSettings(eager_u2_stream_overlap=enabled),
        **streams,
    )
    return SimpleNamespace(
        config=config, transport=transport, session=session, streams=streams
    )


def accept_session_parent(case, *, mode="target", stages=2):
    caps = case.transport.negotiated_capabilities
    case.transport.plan = HcclParentPlan(
        "eager-session-case",
        caps.layer_indices,
        tuple(HcclStageLayout(index, (3,), (4,), 1) for index in range(stages)),
        mode,
    )
