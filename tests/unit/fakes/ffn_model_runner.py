# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU worker/loader boundaries for the FFN constructor and capacity seam."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.model_executor import model_loader
from vllm.utils import mem_utils
from vllm_ascend import ascend_forward_context as native_context

from tests.unit.fakes.vllm_config import hccl_vllm_config


class FFNCapacityModel(torch.nn.Module):
    """Allocate CPU storage using the actual native MC2 capacity result."""

    def __init__(self, vllm_config):
        super().__init__()
        self.vllm_config = vllm_config
        self.register_buffer(
            "routing_workspace", torch.zeros(native_context.get_mc2_tokens_capacity())
        )


class FFNCapacityLoader:
    def load_model(self, *, vllm_config, model_config):
        return FFNCapacityModel(vllm_config)


@contextmanager
def cpu_memory_profiler():
    yield SimpleNamespace(consumed_memory=0)


def make_ffn_capacity_case(
    monkeypatch,
    worker_module,
    runner_module,
    *,
    speculative_steps=2,
    tp_size=1,
    capacity_mode="decode",
):
    """Run plugin construction; replace only admission/native device boundaries.

    No NPU initialization, workspace, profiler, weights or MC2 kernel is executed.
    The original Ascend capacity helper remains installed and is called by the
    production runner's load_model before the CPU loader allocates its model.
    """
    source = hccl_vllm_config(
        role="ffn",
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        tensor_parallel_size=tp_size,
        dspark=speculative_steps is not None,
    )
    if speculative_steps is not None:
        source.speculative_config.num_speculative_tokens = speculative_steps
    source.engram_config = SimpleNamespace(dp_shared_memory=False)
    source.parallel_config.rank = 0
    source.parallel_config.all2all_backend = "flashinfer_all2allv"
    source.scheduler_config.max_num_seqs = 3
    source.compilation_config.cudagraph_capture_sizes = (
        [6, 12] if capacity_mode == "capture" else []
    )
    source.compilation_config.max_cudagraph_capture_size = 12
    source.load_config = SimpleNamespace()
    device = SimpleNamespace(type="npu", index=0)

    def native_worker_init(worker, vllm_config, *args, **kwargs):
        worker.vllm_config = vllm_config

    monkeypatch.setattr(worker_module.NPUWorker, "__init__", native_worker_init)
    monkeypatch.setattr(worker_module.NPUWorker, "_init_device", lambda self: device)
    monkeypatch.setattr(
        worker_module, "apply_afd_ascend_patches_if_needed", lambda: None
    )
    monkeypatch.setattr(
        worker_module, "assert_compatible_afd_stack", lambda *a, **k: None
    )
    monkeypatch.setattr(
        worker_module, "fail_if_unsupported_npu_afd_features", lambda *a, **k: None
    )
    monkeypatch.setattr(worker_module, "init_workspace_manager", lambda *a: None)
    monkeypatch.setattr(
        runner_module, "fail_if_unsupported_npu_afd_features", lambda *a, **k: None
    )
    monkeypatch.setattr(runner_module, "_resolve_world_ranks", lambda: (0, 0))
    monkeypatch.setattr(
        runner_module, "create_npu_runtime_connector", lambda *a: SimpleNamespace()
    )
    monkeypatch.setattr(
        runner_module, "create_deferred_afd_npu_profiler", lambda *a, **k: None
    )
    native_config = SimpleNamespace(
        enable_prefill_mc2=capacity_mode == "scheduler", enable_fused_mc2=False
    )
    monkeypatch.setattr(native_context, "get_ascend_config", lambda: native_config)
    monkeypatch.setattr(native_context, "use_cann_megamoe", lambda config: False)
    monkeypatch.setattr(native_context, "_mc2_tokens_capacity", None)
    monkeypatch.setattr(native_context, "set_mc2_mask", lambda *a: None)
    monkeypatch.setattr(mem_utils, "DeviceMemoryProfiler", cpu_memory_profiler)
    monkeypatch.setattr(
        model_loader, "get_model_loader", lambda config: FFNCapacityLoader()
    )
    worker = worker_module.AFDNPUFFNWorker(source, 0, 0, "cpu-constructor-seam")
    return SimpleNamespace(source=source, worker=worker, device=device)
