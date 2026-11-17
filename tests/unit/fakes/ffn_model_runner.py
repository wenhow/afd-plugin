# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU worker/loader boundaries for the FFN constructor and capacity seam."""

from collections import deque
from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace

import torch
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.model_executor import model_loader
from vllm.utils import mem_utils
from vllm_ascend import ascend_forward_context as native_context

from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu import p2p_hccl as p2p_module
from afd_plugin.connectors.npu.hccl_protocol import HcclForwardPlan
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_plan import HcclParentPlan, HcclPlanMessage
from afd_plugin.v1.worker.npu import hccl_v41_ffn_runner as staged_runner_module
from tests.unit.fakes.connector import FakeRoutedConnector
from tests.unit.fakes.hccl import (
    ROUTER_DELTA_SCALE,
    WORKER_JOIN_TIMEOUT_SECONDS,
    HcclContractModel,
    HcclCpuDeviceRuntime,
    HcclCpuForwardContext,
    HcclLoopbackSession,
    StagedFFNContractModel,
    hccl_layer_tensors,
    make_hccl_connector_pair,
    make_hccl_runner,
    make_hccl_worker,
    make_staged_ffn_service_case,
    staged_ffn_expected_output,
)
from tests.unit.fakes.hccl_v41_eager_bootstrap import make_eager_bootstrap_case
from tests.unit.fakes.vllm_config import hccl_vllm_config


def make_ffn_inference_tensor_case(
    monkeypatch, worker_module, runner_module, *, staged
):
    """Real daemon/runner over CPU peers, with a genuine inference tensor.

    Only the forward-context leaf substitutes a CPU bool mask for native MC2
    storage. Its inplace update obeys real PyTorch thread-local rules; this
    case does not simulate NPU execution, graph capture or collectives.
    """
    with torch.inference_mode():
        mask = torch.zeros(4, dtype=torch.bool)
    if staged:
        plan = HcclParentPlan(
            "inference-tensor-staged",
            (0,),
            tuple(HcclStageLayout(stage, (3,), (3,), 1) for stage in range(2)),
        )
        case = make_staged_ffn_service_case(
            runner_module.AFDNPUFFNModelRunner, worker_module.AFDNPUFFNWorker, [plan]
        )
        case.runner.model = StagedFFNContractModel(case.context, plan, 0)
        case.worker.bind_hccl_v41_transport(case.peer)

        def output_pairs():
            return [
                (
                    message[3],
                    staged_ffn_expected_output(
                        case.peer.current.inputs[message[1], message[2]], message[1]
                    ),
                )
                for message in case.peer.messages
                if message[0] == "OUTPUT"
            ]
    else:
        wire = HcclLoopbackSession()
        attention, ffn = make_hccl_connector_pair(monkeypatch, layer_count=1, wire=wire)
        plan = HcclForwardPlan("inference-tensor-legacy", (0,), 3, 3)
        context = HcclCpuForwardContext()
        runner = make_hccl_runner(
            runner_module.AFDNPUFFNModelRunner,
            ffn,
            HcclContractModel(context, plan),
        )
        case = SimpleNamespace(
            context=context,
            runner=runner,
            worker=make_hccl_worker(worker_module.AFDNPUFFNWorker, runner),
            wire=wire,
        )
        hidden, fp32, ids = hccl_layer_tensors()
        attention.begin_forward_plan(plan)
        attention.send_attn_output(
            hidden,
            AFDTransferContext(AFDTransferMetadata(0, 0, [3])),
            hidden_states_fp32=fp32,
            input_ids=ids,
        )
        # The CPU peer scripts one FORWARD followed by STOP. It never starts
        # a group or substitutes the worker/runner execution methods.
        monkeypatch.setattr(
            ffn._transport,
            "receive_forward",
            lambda: wire.plans.popleft() if wire.plans else None,
        )
        expected = (
            (fp32 - hidden.float()) * ROUTER_DELTA_SCALE + ids[:, None].float()
        ).to(torch.bfloat16)

        def output_pairs():
            return [(output, expected) for output in wire.outputs]

    @contextmanager
    def forward_context(**kwargs):
        mask[: kwargs["num_tokens"]] = True
        with case.context.activate(**kwargs) as context:
            yield context

    context_module = staged_runner_module if staged else runner_module
    monkeypatch.setattr(context_module, "ascend_forward_context", forward_context)
    monkeypatch.setattr(
        worker_module,
        "torch",
        SimpleNamespace(npu=HcclCpuDeviceRuntime(), bfloat16=torch.bfloat16),
    )
    case.mask = mask
    case.output_pairs = output_pairs
    return case


def make_ffn_shutdown_interrupt_case(
    worker_module, runner_module, monkeypatch, error: BaseException
):
    # Keep the shared profiler lifecycle builder out of its lazy device import
    # chain; this factory changes only the host loop's shutdown outcome.
    from tests.unit.fakes.profiler import (
        make_profiled_shutdown_case,
        make_profiler_session_case,
    )

    case = make_profiled_shutdown_case(worker_module, runner_module, monkeypatch)
    case.profile = make_profiler_session_case()
    case.runner.prof = case.profile.session

    def interrupted_shutdown_loop():
        case.worker._ffn_shutdown_event.set()
        raise error

    case.worker._run_ffn_server_loop = interrupted_shutdown_loop
    return case


def make_legacy_ffn_startup_case(
    monkeypatch,
    worker_module,
    runner_module,
    *,
    other_connector=False,
    other_model=True,
    manual=False,
):
    """Real normal FFN factory and service, with a bounded host STOP wait.

    The native worker/device constructor and NPU groups are excluded. The queue
    wire's initialize is a host state boundary, not a real HELLO assertion.
    """
    config = hccl_vllm_config(
        role="ffn",
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
    )
    config.parallel_config.rank = 0
    if other_connector:
        config.additional_config["afd"]["connector"] = "CAMP2pAFDConnector"
        config.parallel_config.enable_expert_parallel = True
        # Exercise the model_type selector independently of the generic
        # registration guard; this fixture never constructs either model.
        config.model_config.hf_config.architectures = ["DeepseekV2ForCausalLM"]
        if other_model:
            config.model_config.hf_config.model_type = "deepseek_v2"
        fallback = FakeRoutedConnector()
        monkeypatch.setattr(
            runner_module, "create_npu_runtime_connector", lambda *a, **k: fallback
        )
    monkeypatch.setattr(
        runner_module, "fail_if_unsupported_npu_afd_features", lambda *a, **k: None
    )
    monkeypatch.setattr(runner_module, "_resolve_world_ranks", lambda: (0, 0))
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_ENABLE", "0")
    wire = HcclLoopbackSession()
    monkeypatch.setattr(p2p_module, "HcclEagerTransport", wire.endpoint)
    device = torch.device("npu", 0)
    if manual:
        connector = runner_module.create_npu_runtime_connector(
            0, 0, config, runner_module.AFDNPUFFNModelRunner.parse_config(config)
        )
        runner = runner_module.AFDNPUFFNModelRunner(config, device, connector)
    else:
        runner = worker_module.create_npu_ffn_model_runner(config, device)
    runner.model = torch.nn.Module()
    worker = make_hccl_worker(worker_module.AFDNPUFFNWorker, runner)
    worker.cache_config = config.cache_config
    # The actual serving thread reaches STOP before any device computation.
    monkeypatch.setattr(
        worker_module.torch.npu, "set_device", HcclCpuDeviceRuntime.set_device
    )
    entered, release = Event(), Event()
    if not other_connector:
        original_receive = runner.connector._transport.receive_forward

        def receive_stop():
            entered.set()
            if not release.wait(timeout=WORKER_JOIN_TIMEOUT_SECONDS):
                raise TimeoutError("CPU peer did not release STOP")
            return original_receive()

        monkeypatch.setattr(
            runner.connector._transport, "receive_forward", receive_stop
        )
    wire.stopped = True
    return SimpleNamespace(
        config=config, runner=runner, worker=worker, entered=entered, release=release
    )


class FFNCapacityModel(torch.nn.Module):
    """Allocate CPU storage using the actual native MC2 capacity result."""

    def __init__(self, vllm_config):
        super().__init__()
        self.vllm_config = vllm_config
        self.register_buffer(
            "routing_workspace", torch.zeros(native_context.get_mc2_tokens_capacity())
        )


class FFNCapacityLoader:
    def __init__(self, *, fail=False):
        self.fail = fail

    def load_model(self, *, vllm_config, model_config):
        if self.fail:
            raise RuntimeError("controlled FFN weight loading failure")
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
    graph=False,
    fail_load=False,
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
    if graph:
        source.model_config.enforce_eager = False
        source.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
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
    if not graph:
        monkeypatch.setattr(
            runner_module,
            "create_npu_runtime_connector",
            lambda *a, **k: SimpleNamespace(),
        )
    else:
        monkeypatch.setattr(
            runner_module,
            "current_platform",
            SimpleNamespace(get_global_graph_pool=lambda: None),
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
        model_loader,
        "get_model_loader",
        lambda config: FFNCapacityLoader(fail=fail_load),
    )
    worker = worker_module.AFDNPUFFNWorker(source, 0, 0, "cpu-constructor-seam")
    return SimpleNamespace(source=source, worker=worker, device=device)


def make_ffn_eager_bootstrap_service(
    monkeypatch, worker_module, runner_module, *, enabled=True, diagnostics=False
):
    """Actual plugin factory/bootstrap/daemon over scripted CPU control frames."""
    case = make_eager_bootstrap_case(
        monkeypatch, role="ffn", enabled=enabled, diagnostics=diagnostics
    )
    for name, value in case.settings.effective_settings().items():
        monkeypatch.setenv(name, "1" if value else "0")
    monkeypatch.setattr(
        runner_module, "fail_if_unsupported_npu_afd_features", lambda *a, **k: None
    )
    monkeypatch.setattr(runner_module, "_resolve_world_ranks", lambda: (0, 0))
    monkeypatch.setattr(
        runner_module, "create_deferred_afd_npu_profiler", lambda *a, **k: None
    )
    runner = runner_module.AFDNPUFFNModelRunner(case.config, case.runtime.parent.device)
    # The factory's real capability/settings selection remains observable;
    # substitute only the process-group/peer boundary before HELLO is needed.
    case.factory_capabilities = runner.connector.transport.capabilities
    case.transport.initialized = False
    runner.connector.transport = case.transport
    runner.model = torch.nn.Module()
    case.runner = runner
    case.worker = make_hccl_worker(worker_module.AFDNPUFFNWorker, runner)
    case.worker.device = runner.device
    case.worker.rank = case.config.parallel_config.rank
    case.runtime.set_device = lambda device: None
    monkeypatch.setattr(worker_module.torch, "npu", case.runtime)
    case.worker.bind_hccl_v41_transport(case.transport)
    # The framed-control boundary supplies its completed HELLO only after the
    # real worker has bound the still-uninitialized connector transport.
    case.transport.initialized = True

    def close_groups():
        case.transport.closed = True

    case.transport.close = close_groups
    case.transport.incoming.setdefault(case.transport.coordinator_rank, deque()).extend(
        (HcclPlanMessage("STOP"), HcclPlanMessage("STOP_ACK"))
    )
    return case
