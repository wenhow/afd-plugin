# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Reuse loaded CPU models at native load/KV/thread construction boundaries."""

from types import SimpleNamespace

from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from tests.unit.fakes.attention_model_runner_v2 import make_graph_runner_case
from tests.unit.fakes.hccl import HcclCpuDeviceRuntime, make_hccl_worker
from tests.unit.fakes.hccl_v41_graph_coordinator import make_coordinator_case
from tests.unit.fakes.hccl_v41_graph_service import make_graph_service_case


class BootstrapNativeLifecycle:
    """Host allocation boundary; no weights, KV allocation or NPU graph claims."""

    def __init__(self, case):
        self.case = case
        self.fail_load = self.fail_kv = False

    def load_model(self, owner, *args, **kwargs):
        self.case.events.append("native-load")
        if self.fail_load:
            raise RuntimeError("native load failed")

    def initialize_kv_cache(self, owner, config, allocation_context=None):
        self.case.events.append("native-kv")
        # Reuse the single native-manager builder; only the CPU host identities
        # that a real allocation publishes are replaced at this seam.
        stages = 2 if owner.parallel_config.use_ubatching else 1
        manager = make_coordinator_case(
            stages=stages, target_query_len=owner.decode_query_len
        ).manager
        manager.model_runner = owner
        if stages == 2:
            owner.ubatch_runner = self.case.runner
            owner.ubatch_runner.graph_owner = owner.hccl_v41_graph_session
        manager.ubatch_runner = owner.ubatch_runner
        owner.cudagraph_manager = manager
        if self.fail_kv:
            raise RuntimeError("native KV allocation failed")


def make_graph_bootstrap_case(*, stages=1, target_query_len=1):
    case = make_graph_runner_case(stages=stages, target_query_len=target_query_len)
    case.native = BootstrapNativeLifecycle(case)
    return case


def set_graph_bootstrap_environment(monkeypatch):
    for name, value in HcclOverlapSettings().effective_settings().items():
        monkeypatch.setenv(name, "1" if value else "0")


def make_ffn_graph_bootstrap_case(service_module, worker_module, monkeypatch):
    set_graph_bootstrap_environment(monkeypatch)
    case = make_graph_service_case(service_module, monkeypatch)
    case.worker = make_hccl_worker(worker_module.AFDNPUFFNWorker, case.runner)
    case.worker.device = case.runner.device
    runtime = HcclCpuDeviceRuntime()
    runtime.synchronize = case.api.synchronize
    monkeypatch.setattr(
        worker_module,
        "torch",
        SimpleNamespace(npu=runtime, bfloat16=service_module.torch.bfloat16),
    )
    case.worker.bind_hccl_v41_transport(case.transport)
    return case
