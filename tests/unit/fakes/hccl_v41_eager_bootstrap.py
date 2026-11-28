# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real eager assembly over shared CPU stream and transport packet boundaries."""

from dataclasses import replace
from types import SimpleNamespace

import torch
from vllm.config import CompilationMode, CUDAGraphMode

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.npu import hccl_staged_transport as transport_module
from afd_plugin.connectors.npu.hccl_config import validate_staged_hccl_runtime
from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_plan import HcclParentPlan
from afd_plugin.v1.worker.npu import hccl_v41_eager as session_module
from afd_plugin.v1.worker.npu import hccl_v41_eager_bootstrap as bootstrap_module
from tests.unit.fakes.hccl import ScriptedStagedControlPeer
from tests.unit.fakes.hccl_diagnostics import StageDiagnosticRecorder
from tests.unit.fakes.hccl_eager_pipeline import (
    EagerHostBackend,
    EagerHostGroup,
    EagerHostTorch,
)
from tests.unit.fakes.npu_stream import HostStream, HostStreamRecorder
from tests.unit.fakes.vllm_config import hccl_vllm_config


class BootstrapHostRuntime(HostStreamRecorder):
    def __init__(self, device):
        super().__init__(device)
        self.allocated = []
        self.capturing = False

    def Stream(self, *, device):  # noqa: N802 - native allocation boundary
        stream = HostStream(f"allocated-{len(self.allocated)}", device)
        self.allocated.append(stream)
        return stream

    def is_current_stream_capturing(self):
        return self.capturing

    def synchronize(self, device):
        self.trace.append(("synchronize", device))


class BootstrapHostTorch(EagerHostTorch):
    # Validate the real indexed NPU identity while allocating CPU packets only.
    device = staticmethod(torch.device)

    @staticmethod
    def zeros(shape, *, dtype, device):
        return torch.zeros(shape, dtype=dtype, device="cpu")


def make_eager_bootstrap_case(
    monkeypatch, *, role="attention", stages=2, tp=1, enabled=True, diagnostics=False
):
    config = hccl_vllm_config(
        role=role,
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        tensor_parallel_size=tp,
        num_ubatches=stages,
    )
    config.parallel_config.rank = 0
    caps = replace(
        validate_staged_hccl_runtime(config, parse_afd_config(config), 0),
        max_stages=stages,
        eager_pipeline_version=int(enabled and stages == 2),
    )
    transport = ScriptedStagedControlPeer(role, 0, caps, {})
    transport.control_group = "host-control"
    transport.data_groups = [
        EagerHostGroup(stage, transport.world_rank) for stage in range(stages)
    ]
    runtime = BootstrapHostRuntime(torch.device("npu", transport.local_device))
    backend = EagerHostBackend(runtime)
    host_torch = BootstrapHostTorch(runtime)
    for module in (bootstrap_module, session_module, transport_module):
        monkeypatch.setattr(module, "torch", host_torch)
    monkeypatch.setattr(transport_module, "dist", backend)
    monkeypatch.setattr(transport_module, "use_npu_current_stream", runtime.stream)
    settings = HcclOverlapSettings(
        eager_u2_stream_overlap=enabled,
        stage_diagnostics=diagnostics,
        ffn_compute_sync_diagnostics=diagnostics,
    )
    recorder = StageDiagnosticRecorder()
    bootstrap = bootstrap_module.HcclV41EagerBootstrap(
        config,
        transport,
        overlap_settings=settings,
        synchronize=recorder.synchronize,
        diagnostic_sink=recorder.emit,
    )
    return SimpleNamespace(
        config=config,
        transport=transport,
        runtime=runtime,
        backend=backend,
        settings=settings,
        recorder=recorder,
        bootstrap=bootstrap,
        bound_sessions=[],
    )


def accept_bootstrap_parent(case, *, stages=1, mode="target"):
    caps = case.transport.negotiated_capabilities
    case.transport.plan = HcclParentPlan(
        "bootstrap-diagnostic-parent",
        caps.layer_indices,
        tuple(
            HcclStageLayout(
                index,
                (1,) * caps.attention_size,
                (1,) * caps.attention_size,
                caps.ffn_size,
                caps.native_tp_size,
            )
            for index in range(stages)
        ),
        mode,
    )
