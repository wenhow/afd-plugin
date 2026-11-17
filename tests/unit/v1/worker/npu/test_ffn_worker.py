# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU worker lifecycle contracts; device callbacks are explicit no-ops."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from vllm.v1.worker.gpu import shutdown as native_shutdown_module

from afd_plugin.compat.npu.runtime_config import DeepseekV41TargetShape
from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.connectors.npu.hccl_protocol import HcclForwardPlan
from afd_plugin.connectors.npu.hccl_staged import HcclStagedAFDConnector
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_plan import HcclParentPlan
from afd_plugin.v1.worker.npu import ffn_model_runner as runner_module
from afd_plugin.v1.worker.npu import ffn_worker as worker_module
from afd_plugin.v1.worker.npu import hccl_v41_graph_service as service_module
from tests.unit.fakes.ffn_model_runner import (
    make_ffn_capacity_case,
    make_ffn_eager_bootstrap_service,
    make_ffn_inference_tensor_case,
    make_ffn_shutdown_interrupt_case,
    make_legacy_ffn_startup_case,
)
from tests.unit.fakes.hccl import (
    WORKER_JOIN_TIMEOUT_SECONDS,
    ExitedHcclWorkerThread,
    HcclCpuDeviceRuntime,
    HcclCpuForwardContext,
    HcclFailingModel,
    PendingHcclWorkerThread,
    hccl_layer_tensors,
    join_hccl_worker,
    make_hccl_connector_pair,
    make_hccl_runner,
    make_hccl_worker,
    make_staged_ffn_service_case,
)
from tests.unit.fakes.hccl_v41_graph_bootstrap import make_ffn_graph_bootstrap_case
from tests.unit.fakes.hccl_v41_graph_service import make_graph_service_case
from tests.unit.fakes.logging import (
    SESSION_CONFIGURATION_PREFIX,
    structured_log_records,
)
from tests.unit.fakes.logging import serving_logs as serving_logs
from tests.unit.fakes.profiler import (
    PROFILER_TEST_TIMEOUT_SECONDS,
    make_profiled_restart_case,
    make_profiled_shutdown_case,
    make_profiled_worker_case,
    make_profiler_session_case,
)


@pytest.mark.parametrize("staged", [False, True], ids=["legacy-u1", "staged-u2"])
def test_ffn_daemon_updates_inference_tensor_and_completes_parent(monkeypatch, staged):
    case = make_ffn_inference_tensor_case(
        monkeypatch, worker_module, runner_module, staged=staged
    )

    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    case.worker.raise_ffn_loop_error_if_any()

    assert case.mask.tolist() == [True, True, True, False]
    outputs = case.output_pairs()
    assert len(outputs) == (2 if staged else 1)
    assert all(
        runner_module.torch.equal(output, expected) for output, expected in outputs
    )
    if staged:
        assert case.peer.messages[-2:] == [
            ("COMPLETE", "inference-tensor-staged"),
            ("STOP_ACK",),
        ]
    assert case.worker.poll_hccl_ffn_service() is False
    case.worker.stop_ffn_server_loop()


@pytest.mark.parametrize("steps", [None, 2, 5])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("capacity_mode", ["decode", "capture", "scheduler"])
def test_ffn_worker_retains_target_shape_through_factory_and_native_capacity(
    monkeypatch, steps, tp_size, capacity_mode
):
    case = make_ffn_capacity_case(
        monkeypatch,
        worker_module,
        runner_module,
        speculative_steps=steps,
        tp_size=tp_size,
        capacity_mode=capacity_mode,
    )
    shape = DeepseekV41TargetShape(1 if steps is None else steps + 1)
    assert case.worker.target_shape == shape
    assert case.worker.vllm_config.speculative_config is None
    assert case.worker.vllm_config.engram_config is None

    case.worker.init_device()
    case.worker.model_runner.load_model()

    model = case.worker.model_runner.get_model()
    raw_capacity = {
        "decode": 3 * shape.target_query_len,
        "capture": 12,
        "scheduler": 16,
    }[capacity_mode]
    expected_capacity = ((raw_capacity + tp_size - 1) // tp_size) * tp_size
    assert case.worker.model_runner.target_shape is case.worker.target_shape
    assert model.routing_workspace.numel() == expected_capacity
    assert model.vllm_config.speculative_config is None
    assert model.vllm_config.engram_config is None
    assert case.source.engram_config.dp_shared_memory is False
    if steps is not None:
        assert case.source.speculative_config.num_speculative_tokens == steps


@pytest.fixture
def graph_service(monkeypatch):
    case = make_graph_service_case(service_module, monkeypatch)
    case.worker = make_hccl_worker(worker_module.AFDNPUFFNWorker, case.runner)
    case.worker.device = case.runner.device
    monkeypatch.setattr(
        worker_module,
        "torch",
        SimpleNamespace(
            npu=HcclCpuDeviceRuntime(), bfloat16=runner_module.torch.bfloat16
        ),
    )
    case.worker.bind_hccl_v41_transport(case.transport)
    case.runner.bind_hccl_v41_graph_service(case.service)
    return case


def test_graph_worker_releases_cache_after_stop_before_communicators(graph_service):
    case = graph_service
    case.transport.queue_parent("capture", identifier="capture-1")
    case.transport.queue_parent("replay", identifier="replay-1", valid=(1, 0))
    case.transport.queue_stop()
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    case.worker.stop_ffn_server_loop()
    assert case.worker.poll_hccl_ffn_service() is False
    assert case.cache.closed and case.transport.closed
    assert case.events[-3:] == ["device-sync", "graph-reset", "groups-close"]


def test_graph_worker_preserves_failed_capture_and_communicator_ownership(
    graph_service,
):
    case = graph_service
    case.api.fail_body = True
    case.transport.queue_parent("capture", identifier="capture-1")
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    with pytest.raises(RuntimeError, match="worker loop failed") as failure:
        case.worker.stop_ffn_server_loop()
    assert "controlled tensor body failure" in str(failure.value.__cause__)
    assert case.cache.terminal and not case.cache.closed
    assert case.transport.terminal and not case.transport.closed
    assert "graph-reset" not in case.events and "groups-close" not in case.events


def test_headless_start_assembles_graph_service_without_attention_kv_initialization(
    monkeypatch,
):
    case = make_ffn_graph_bootstrap_case(service_module, worker_module, monkeypatch)
    case.transport.queue_parent("replay", identifier="early-profile", fallback=True)
    case.transport.queue_stop()

    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    service = case.runner.hccl_v41_graph_service
    assert service is not None and service.acknowledged_stop
    assert service.generation() == 0
    assert service.cache.is_empty and not service.cache.closed
    assert not service.overlap_settings.stage_diagnostics
    assert not service.overlap_settings.ffn_compute_sync_diagnostics
    with pytest.raises(RuntimeError, match="acknowledged staged HCCL STOP"):
        case.worker.start_ffn_server_loop()
    assert case.worker.poll_hccl_ffn_service() is False
    case.worker.stop_ffn_server_loop()

    assert service.cache.closed and case.transport.closed
    assert case.events[-2:] == ["device-sync", "groups-close"]
    sent = [
        m.kind
        for direction, _, _, m in case.transport.transcript
        if direction == "send"
    ]
    assert "GRAPH_REPORT" in sent and "ACCEPT" in sent and "COMPLETE" in sent
    assert "STOP_ACK" in sent


def test_headless_graph_start_rejects_unloaded_weights_and_retains_the_session(
    monkeypatch,
):
    case = make_ffn_graph_bootstrap_case(service_module, worker_module, monkeypatch)
    case.runner.model = None
    with pytest.raises(RuntimeError, match="not been loaded"):
        case.worker.start_ffn_server_loop()
    assert case.transport.terminal and not case.transport.closed
    assert case.transport.transcript == []
    with pytest.raises(RuntimeError, match="terminal staged"):
        case.worker.start_ffn_server_loop()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_normal_eager_factory_freezes_family_before_daemon_stop(
    monkeypatch, enabled, diagnostics
):
    case = make_ffn_eager_bootstrap_service(
        monkeypatch,
        worker_module,
        runner_module,
        enabled=enabled,
        diagnostics=diagnostics,
    )
    assert case.factory_capabilities.eager_pipeline_version == int(enabled)
    monkeypatch.setenv("AFD_HCCL_EAGER_U2_STREAM_OVERLAP", "invalid-after-construction")
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    assert case.worker.poll_hccl_ffn_service() is False
    assert (case.runner.hccl_v41_eager_session is not None) == enabled
    assert (case.runner.hccl_v41_diagnostics is not None) == diagnostics
    assert [
        message.kind
        for direction, _, _, message in case.transport.transcript
        if direction == "send"
    ] == ["STOP_ACK"]
    case.worker.stop_ffn_server_loop()
    assert case.transport.closed


def test_ffn_startup_logs_frozen_explicit_zero_settings_once(monkeypatch, serving_logs):
    case = make_ffn_eager_bootstrap_service(
        monkeypatch, worker_module, runner_module, enabled=False
    )
    values = {name: "0" for name in HcclOverlapSettings().effective_settings()}
    case.runner.connector.overlap_settings = HcclOverlapSettings.from_environment(
        values, num_stages=2
    )
    for name in values:
        monkeypatch.setenv(name, "1")
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    with pytest.raises(RuntimeError, match="acknowledged staged HCCL STOP"):
        case.worker.start_ffn_server_loop()
    case.worker.stop_ffn_server_loop()
    records = structured_log_records(serving_logs, SESSION_CONFIGURATION_PREFIX)
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "session_configuration" and record["phase"] == "startup"
    assert record["role"] == "ffn" and record["snapshot_present"] is True
    assert record["negotiated.graph_execution_version"] == 0
    assert record["negotiated.eager_pipeline_version"] == 0
    assert {name: record[f"configured.{name}"] for name in values} == {
        name: False for name in values
    }


def test_ffn_startup_logs_missing_snapshot_without_environment_guesses(
    monkeypatch, serving_logs
):
    case = make_ffn_eager_bootstrap_service(
        monkeypatch, worker_module, runner_module, enabled=False
    )
    case.runner.connector.overlap_settings = None
    monkeypatch.setenv("AFD_HCCL_EAGER_U2_STREAM_OVERLAP", "1")
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    case.worker.stop_ffn_server_loop()
    records = structured_log_records(serving_logs, SESSION_CONFIGURATION_PREFIX)
    assert len(records) == 1 and records[0]["snapshot_present"] is False
    assert records[0]["negotiated.eager_pipeline_version"] == 0
    assert not any(name.startswith("configured.") for name in records[0])


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("preinitialized", [False, True])
def test_legacy_ffn_logs_one_frozen_session_across_successful_start_repeats(
    monkeypatch, serving_logs, enabled, preinitialized
):
    names = HcclOverlapSettings().effective_settings()
    for name in names:
        monkeypatch.setenv(name, str(int(enabled)))
    case = make_legacy_ffn_startup_case(monkeypatch, worker_module, runner_module)
    if preinitialized:
        case.runner.initialize_afd_connector()
    for name in names:
        monkeypatch.setenv(name, "invalid-after-construction")
    case.worker.start_ffn_server_loop()
    try:
        assert case.entered.wait(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
        case.worker.start_ffn_server_loop()
    finally:
        case.release.set()
        join_hccl_worker(case.worker)
    assert case.worker.poll_hccl_ffn_service() is False
    case.worker.stop_ffn_server_loop()

    records = structured_log_records(serving_logs, SESSION_CONFIGURATION_PREFIX)
    assert len(records) == 1
    record = records[0]
    assert record["protocol"] == "legacy_pair" and record["role"] == "ffn"
    assert record["snapshot_present"] is True
    assert record["world_rank"] == 0 and record["pair_index"] == 0
    expected = {name: enabled for name in names}
    expected["AFD_HCCL_EAGER_U2_STREAM_OVERLAP"] = False
    assert {name: record[f"configured.{name}"] for name in names} == expected


def test_legacy_ffn_kv_initialization_keeps_the_successful_session_record(
    monkeypatch, serving_logs
):
    case = make_legacy_ffn_startup_case(monkeypatch, worker_module, runner_module)
    case.worker.initialize_from_config(SimpleNamespace(num_blocks=1))
    try:
        assert case.entered.wait(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
    finally:
        case.release.set()
        join_hccl_worker(case.worker)
    case.worker.stop_ffn_server_loop()

    records = structured_log_records(serving_logs, SESSION_CONFIGURATION_PREFIX)
    assert len(records) == 1 and records[0]["snapshot_present"] is True


def test_legacy_ffn_injected_connector_keeps_unknown_settings(
    monkeypatch, serving_logs
):
    for name in HcclOverlapSettings().effective_settings():
        monkeypatch.setenv(name, "invalid-unused-setting")
    case = make_legacy_ffn_startup_case(
        monkeypatch, worker_module, runner_module, manual=True
    )
    case.worker.start_ffn_server_loop()
    try:
        assert case.entered.wait(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
    finally:
        case.release.set()
        join_hccl_worker(case.worker)
    case.worker.stop_ffn_server_loop()

    records = structured_log_records(serving_logs, SESSION_CONFIGURATION_PREFIX)
    assert len(records) == 1 and records[0]["snapshot_present"] is False
    assert not any(name.startswith("configured.") for name in records[0])


def test_legacy_ffn_hello_failure_emits_no_configuration(monkeypatch, serving_logs):
    case = make_legacy_ffn_startup_case(monkeypatch, worker_module, runner_module)

    def failed_hello():
        raise RuntimeError("controlled legacy HELLO failure")

    monkeypatch.setattr(case.runner.connector, "init_afd_connector", failed_hello)

    with pytest.raises(RuntimeError, match="legacy HELLO"):
        case.worker.start_ffn_server_loop()

    assert structured_log_records(serving_logs, SESSION_CONFIGURATION_PREFIX) == []


@pytest.mark.parametrize("failure", ["weights", "warmup", "hello"])
def test_ffn_failed_startup_emits_no_session_configuration(
    monkeypatch, serving_logs, failure
):
    case = make_ffn_eager_bootstrap_service(monkeypatch, worker_module, runner_module)
    if failure == "weights":
        case.runner.model = None
    elif failure == "warmup":
        case.backend.fail_warmup = True
    else:
        case.transport.initialized = False

        def failed_hello():
            raise RuntimeError("controlled HELLO failure")

        monkeypatch.setattr(case.transport, "initialize", failed_hello)
    with pytest.raises(RuntimeError, match="not been loaded|warmup|HELLO"):
        case.worker.start_ffn_server_loop()
    assert structured_log_records(serving_logs, SESSION_CONFIGURATION_PREFIX) == []


@pytest.mark.parametrize("failure", ["weights", "warmup"])
def test_eager_start_failure_retains_prethread_resources_without_stop(
    monkeypatch, failure
):
    case = make_ffn_eager_bootstrap_service(monkeypatch, worker_module, runner_module)
    if failure == "weights":
        case.runner.model = None
    else:
        case.backend.fail_warmup = True
    with pytest.raises(RuntimeError, match="not been loaded|matched warmup"):
        case.worker.start_ffn_server_loop()
    assert case.transport.terminal and not case.transport.closed
    assert case.transport.transcript == []
    if failure == "warmup":
        assert case.runner.hccl_v41_eager_bootstrap.session is not None
    with pytest.raises(RuntimeError, match="terminal staged"):
        case.worker.start_ffn_server_loop()


@pytest.mark.parametrize("fail_native", [False, True])
def test_eager_worker_cleanup_releases_session_only_after_native_success(
    monkeypatch, fail_native
):
    case = make_ffn_eager_bootstrap_service(monkeypatch, worker_module, runner_module)
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    bootstrap = case.runner.hccl_v41_eager_bootstrap
    model = case.runner.get_model()

    def native_free(config):
        if fail_native:
            raise RuntimeError("native cleanup failed")

    monkeypatch.setattr(native_shutdown_module, "free_before_shutdown", native_free)
    if fail_native:
        with pytest.raises(RuntimeError, match="native cleanup failed"):
            case.worker.shutdown()
        assert case.runner.hccl_v41_eager_bootstrap is bootstrap
        assert case.runner.hccl_v41_eager_session is bootstrap.session
        assert case.runner.get_model() is model
    else:
        case.worker.shutdown()
        assert case.runner.hccl_v41_eager_bootstrap is None
        assert case.runner.hccl_v41_eager_session is None
        with pytest.raises(RuntimeError, match="not been loaded"):
            case.runner.get_model()
    assert case.transport.closed


@pytest.fixture
def staged_service(monkeypatch):
    plans = [
        HcclParentPlan(
            f"service-parent-{index}",
            (0,),
            tuple(HcclStageLayout(stage, (0,), (0,), 1) for stage in range(count)),
        )
        for index, count in enumerate((1, 2, 1))
    ]
    case = make_staged_ffn_service_case(
        runner_module.AFDNPUFFNModelRunner, worker_module.AFDNPUFFNWorker, plans
    )
    monkeypatch.setattr(
        worker_module,
        "torch",
        SimpleNamespace(
            npu=HcclCpuDeviceRuntime(), bfloat16=runner_module.torch.bfloat16
        ),
    )
    return case


def test_bound_staged_service_completes_u1_u2_parents_and_acknowledges_stop(
    staged_service,
):
    case = staged_service
    case.worker.bind_hccl_v41_transport(case.peer)
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    case.worker.stop_ffn_server_loop()

    assert case.peer.messages == [
        message
        for index in range(3)
        for message in (
            ("FORWARD", f"service-parent-{index}"),
            ("COMPLETE", f"service-parent-{index}"),
        )
    ] + [("STOP_ACK",)]
    assert case.peer.closed
    assert case.worker.poll_hccl_ffn_service() is False
    assert case.worker.stop_ffn_server_loop() is None


def test_bound_staged_service_preserves_compute_failure_and_retains_resources(
    staged_service,
):
    case = staged_service
    case.peer.parents[0] = HcclParentPlan(
        "unloaded-model", (0,), (HcclStageLayout(0, (1,), (1,), 1),)
    )
    case.worker.bind_hccl_v41_transport(case.peer)
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)

    for action in (case.worker.poll_hccl_ffn_service, case.worker.stop_ffn_server_loop):
        with pytest.raises(RuntimeError, match="worker loop failed") as failure:
            action()
        assert "model has not been loaded" in str(failure.value.__cause__)
    assert case.peer.terminal and not case.peer.closed
    assert case.peer.messages == [("FORWARD", "unloaded-model")]


def test_bound_staged_service_cannot_release_initialized_transport_before_peer_stop(
    staged_service,
):
    case = staged_service
    case.worker.bind_hccl_v41_transport(case.peer)
    case.peer.initialize()

    with pytest.raises(RuntimeError, match="requires peer STOP; retain resources"):
        case.worker.stop_ffn_server_loop()

    assert not case.peer.closed


@pytest.mark.parametrize(
    "boundary",
    [
        "role",
        "version",
        "native-tp",
        "dp",
        "eager",
        "device",
        "hidden",
        "vocab",
        "layers",
        "capacity",
    ],
)
def test_staged_service_rejects_unsupported_binding_before_protocol(
    staged_service, boundary
):
    case = staged_service
    if boundary == "role":
        case.peer.role = "attention"
    elif boundary == "version":
        case.peer.negotiated_capabilities = replace(
            case.peer.negotiated_capabilities, local_counts_version=0
        )
    elif boundary == "native-tp":
        case.worker.vllm_config.parallel_config.tensor_parallel_size = 2
    elif boundary == "dp":
        case.worker.vllm_config.parallel_config.data_parallel_size = 2
    elif boundary == "eager":
        case.worker.vllm_config.model_config.enforce_eager = False
    elif boundary == "device":
        case.peer.local_device = 1
    else:
        field, value = {
            "hidden": ("hidden_size", 32),
            "vocab": ("vocab_size", 2048),
            "layers": ("layer_indices", (0, 1)),
            "capacity": ("max_tokens", 17),
        }[boundary]
        case.peer.negotiated_capabilities = replace(
            case.peer.negotiated_capabilities, **{field: value}
        )

    with pytest.raises(ValueError, match="(?i)hccl"):
        case.worker.bind_hccl_v41_transport(case.peer)

    assert not case.peer.initialized
    assert case.peer.messages == []


def test_staged_service_rejects_a_second_transport_for_its_connector(staged_service):
    case = staged_service
    config = case.runner.vllm_config
    connector = HcclStagedAFDConnector(0, 0, config, parse_afd_config(config), 0)
    case.runner.connector = connector

    with pytest.raises(ValueError, match="must use its connector transport"):
        case.worker.bind_hccl_v41_transport(case.peer)

    assert not case.peer.initialized
    assert not connector.is_initialized
    assert case.peer.messages == []


@pytest.fixture
def lifecycle(monkeypatch):
    connectors = make_hccl_connector_pair(monkeypatch, layer_count=1)
    context = HcclCpuForwardContext()
    monkeypatch.setattr(runner_module, "ascend_forward_context", context.activate)
    monkeypatch.setattr(
        worker_module, "torch", SimpleNamespace(npu=HcclCpuDeviceRuntime())
    )
    attention, ffn = connectors
    runner = make_hccl_runner(runner_module.AFDNPUFFNModelRunner, ffn, None)
    worker = make_hccl_worker(worker_module.AFDNPUFFNWorker, runner)
    yield attention, ffn, runner, worker
    for connector in connectors:
        connector.close()


def test_worker_acknowledges_peer_stop_before_releasing_connector(lifecycle):
    attention, ffn, _, worker = lifecycle
    attention.request_stop()

    worker.start_ffn_server_loop()
    join_hccl_worker(worker)
    assert worker.stop_ffn_server_loop() is None

    assert not attention.is_initialized
    assert not ffn.is_initialized
    assert worker.raise_ffn_loop_error_if_any() is None
    assert worker.poll_hccl_ffn_service() is False
    assert worker.stop_ffn_server_loop() is None


@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("native_failure", [False, True])
def test_hccl_worker_cleanup_log_requires_successful_native_shutdown(
    lifecycle, monkeypatch, serving_logs, rank, native_failure
):
    attention, ffn, _, worker = lifecycle
    worker.rank = rank

    def native_shutdown(_worker):
        if native_failure:
            raise RuntimeError("native cleanup failed")

    monkeypatch.setattr(worker_module.NPUWorker, "shutdown", native_shutdown)
    attention.request_stop()
    worker.start_ffn_server_loop()
    join_hccl_worker(worker)

    if native_failure:
        with pytest.raises(RuntimeError, match="native cleanup failed"):
            worker.shutdown()
    else:
        worker.shutdown()

    assert not ffn.is_initialized
    messages = [record.getMessage() for record in serving_logs.records]
    marker = f"AFD HCCL FFN worker cleanup complete; rank={rank}."
    assert (marker in messages) is (not native_failure)


def test_hccl_worker_cleanup_log_requires_peer_stop(
    lifecycle, monkeypatch, serving_logs
):
    _, ffn, _, worker = lifecycle
    worker.rank = 0
    monkeypatch.setattr(worker_module.NPUWorker, "shutdown", lambda _worker: None)

    worker.shutdown()

    assert not ffn.is_initialized
    assert "AFD HCCL FFN worker cleanup complete;" not in serving_logs.text


def test_worker_keeps_compute_failure_terminal_across_raise_start_and_stop(lifecycle):
    attention, ffn, runner, worker = lifecycle
    runner.model = HcclFailingModel()
    plan = HcclForwardPlan("failing-existing-forward", (0,), 2, 3)
    hidden, fp32, ids = hccl_layer_tensors()
    attention.begin_forward_plan(plan)
    context = AFDTransferContext(AFDTransferMetadata(0, 0, [3]))
    attention.send_attn_output(hidden, context, hidden_states_fp32=fp32, input_ids=ids)

    worker.start_ffn_server_loop()
    join_hccl_worker(worker)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="FFN worker loop failed") as failure:
            worker.raise_ffn_loop_error_if_any()
        assert isinstance(failure.value.__cause__, ValueError)
        assert str(failure.value.__cause__) == "controlled FFN compute failure"
    with pytest.raises(RuntimeError, match="FFN worker loop failed"):
        worker.start_ffn_server_loop()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="FFN worker loop failed") as failure:
            worker.poll_hccl_ffn_service()
        assert isinstance(failure.value.__cause__, ValueError)
    with pytest.raises(RuntimeError, match="FFN worker loop failed"):
        worker.stop_ffn_server_loop()
    assert ffn.is_initialized


@pytest.mark.parametrize("thread", [None, ExitedHcclWorkerThread()])
def test_worker_poll_rejects_service_without_a_live_thread_or_peer_stop(
    lifecycle, thread
):
    _, ffn, _, worker = lifecycle
    worker._ffn_thread = thread

    with pytest.raises(RuntimeError, match="HCCL"):
        worker.poll_hccl_ffn_service()

    assert ffn.is_initialized


def test_worker_retains_resources_while_peer_stop_or_device_is_pending(lifecycle):
    _, ffn, _, worker = lifecycle
    worker._ffn_thread = PendingHcclWorkerThread()
    assert worker.poll_hccl_ffn_service() is True

    with pytest.raises(RuntimeError, match="retain resources"):
        worker.stop_ffn_server_loop()

    assert ffn.is_initialized


@pytest.fixture
def profiled_service(monkeypatch):
    case = make_profiled_worker_case(worker_module, runner_module, monkeypatch)
    yield case
    for event in case.gate.release:
        event.set()
    if case.worker._ffn_thread is not None:
        join_hccl_worker(case.worker)


def test_role_profiler_rpc_remains_responsive_during_parent_receive(profiled_service):
    case = profiled_service
    case.worker.start_ffn_server_loop()
    assert case.gate.entered[0].wait(PROFILER_TEST_TIMEOUT_SECONDS)
    case.worker.profile(False)
    pending = case.worker.afd_profiler_status()
    assert pending["mode"] == "afd-role"
    assert pending["pending"] and pending["applied"] and not pending["requested"]
    case.gate.release[0].set()
    assert case.gate.entered[1].wait(PROFILER_TEST_TIMEOUT_SECONDS)
    applied = case.worker.afd_profiler_status()
    assert not applied["pending"] and not applied["applied"]
    case.gate.release[1].set()
    join_hccl_worker(case.worker)
    case.worker.stop_ffn_server_loop()
    assert case.worker.afd_profiler_status()["closed"]
    assert case.peer.messages == [
        ("FORWARD", "profile-parent"),
        ("COMPLETE", "profile-parent"),
        ("STOP_ACK",),
    ]


def test_hccl_profiler_steps_one_parent_and_finalizes_on_its_execution_owner(
    profiled_service,
):
    case = profiled_service
    for release in case.gate.release:
        release.set()
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    case.worker.stop_ffn_server_loop()
    status = case.worker.afd_profiler_status()
    assert status["steps"] == 1 and status["closed"]
    assert [event[0] for event in case.profile.events] == [
        "create",
        "start",
        "step",
        "stop",
    ]
    assert {event[1] for event in case.profile.events} == {status["owner_thread_id"]}


@pytest.mark.parametrize("export_failure", [None, "stop"])
def test_hccl_poll_waits_for_stop_export_and_preserves_its_result(
    profiled_service, export_failure
):
    case = profiled_service
    case.profile = make_profiler_session_case(failure=export_failure)
    case.runner.prof = case.profile.session
    case.worker.start_ffn_server_loop()
    assert case.gate.entered[0].wait(PROFILER_TEST_TIMEOUT_SECONDS)
    backend = case.profile.profilers[0]
    backend.stop_release.clear()
    for release in case.gate.release:
        release.set()
    try:
        assert backend.stop_entered.wait(PROFILER_TEST_TIMEOUT_SECONDS)
        assert case.peer.messages[-1] == ("STOP_ACK",)
        assert case.worker.poll_hccl_ffn_service() is True
        assert not case.peer.closed
    finally:
        backend.stop_release.set()
        join_hccl_worker(case.worker)

    if export_failure is None:
        assert case.worker.poll_hccl_ffn_service() is False
        assert case.worker.stop_ffn_server_loop() is None
        assert case.worker.poll_hccl_ffn_service() is False
        assert case.worker.afd_profiler_status()["closed"]
    else:
        for action in (
            case.worker.poll_hccl_ffn_service,
            case.worker.stop_ffn_server_loop,
            case.worker.poll_hccl_ffn_service,
        ):
            with pytest.raises(RuntimeError, match="worker loop failed") as failure:
                action()
            assert isinstance(failure.value.__cause__, ValueError)
            assert str(failure.value.__cause__) == "controlled profiler stop failure"
        assert not case.peer.closed


def test_role_and_native_profilers_cannot_start_together(profiled_service):
    case = profiled_service
    case.worker.profiler_config = SimpleNamespace(profiler="torch")
    with pytest.raises(ValueError, match="cannot be enabled together"):
        case.worker.start_ffn_server_loop()
    with pytest.raises(ValueError, match="cannot be enabled together"):
        case.worker.profile()
    assert case.profile.events == [] and case.peer.messages == []


def test_default_profile_rpc_preserves_native_worker_control(lifecycle, monkeypatch):
    _, _, _, worker = lifecycle
    worker.profiler_config = SimpleNamespace(profiler="torch")
    native = SimpleNamespace(is_running=False, prefix=None)

    def profile_native(self, is_start=True, profile_prefix=None):
        native.is_running, native.prefix = is_start, profile_prefix

    monkeypatch.setattr(worker_module.NPUWorker, "profile", profile_native)
    worker.profile(True, "native-prefix")
    assert native.is_running and native.prefix == "native-prefix"
    worker.profile(False)
    assert not native.is_running
    assert worker.afd_profiler_status() == {
        "mode": "native",
        "configured": True,
        "control_running": False,
    }


def test_profiler_export_failure_remains_terminal_and_preserves_hccl_resources(
    profiled_service,
):
    case = profiled_service
    case.profile = make_profiler_session_case(failure="stop")
    case.runner.prof = case.profile.session
    for release in case.gate.release:
        release.set()
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    with pytest.raises(RuntimeError, match="worker loop failed") as failure:
        case.worker.stop_ffn_server_loop()
    assert "controlled profiler stop failure" in str(failure.value.__cause__)
    status = case.worker.afd_profiler_status()
    assert status["failed"] and not status["closed"]
    assert not case.peer.closed


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_interrupted_stop_export_preserves_original_failure_and_owner_resources(
    profiled_service, serving_logs, error_type
):
    case = profiled_service
    original = error_type("interrupted owner export")
    case.profile = make_profiler_session_case(failure="stop", failure_error=original)
    case.runner.prof = case.profile.session
    for release in case.gate.release:
        release.set()
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    assert case.peer.messages[-1] == ("STOP_ACK",)
    for action in (case.worker.poll_hccl_ffn_service, case.worker.shutdown):
        with pytest.raises(RuntimeError, match="worker loop failed") as failure:
            action()
        assert failure.value.__cause__ is original
    assert case.worker.afd_profiler_status()["failed"]
    assert not case.peer.closed
    assert not any(
        "AFD HCCL FFN worker cleanup complete" in record.getMessage()
        for record in serving_logs.records
    )


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_legacy_shutdown_only_suppresses_ordinary_loop_errors(monkeypatch, error_type):
    original = error_type("host loop interrupted during shutdown")
    case = make_ffn_shutdown_interrupt_case(
        worker_module, runner_module, monkeypatch, original
    )
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    if isinstance(original, Exception):
        assert case.worker.stop_ffn_server_loop() is None
    else:
        with pytest.raises(RuntimeError, match="worker loop failed") as failure:
            case.worker.stop_ffn_server_loop()
        assert failure.value.__cause__ is original


def test_legacy_shutdown_does_not_hide_a_profiler_export_failure(monkeypatch):
    case = make_profiled_shutdown_case(worker_module, runner_module, monkeypatch)
    case.worker.start_ffn_server_loop()
    join_hccl_worker(case.worker)
    with pytest.raises(RuntimeError, match="worker loop failed") as failure:
        case.worker.stop_ffn_server_loop()
    assert "controlled profiler stop failure" in str(failure.value.__cause__)
    assert case.worker.afd_profiler_status()["failed"]


@pytest.mark.parametrize("state", ["failed", "closed"])
def test_terminal_profiler_owner_rejects_restart_before_legacy_connector_init(
    monkeypatch, state
):
    case = make_profiled_restart_case(worker_module, runner_module, monkeypatch)
    profile = make_profiler_session_case(failure="start" if state == "failed" else None)
    case.runner.prof = profile.session
    if state == "failed":
        with pytest.raises(ValueError, match="controlled profiler start failure"):
            profile.session.before_step()
    else:
        profile.session.finish()
    with pytest.raises(RuntimeError, match="failed or closed AFD profiler owner"):
        case.worker.start_ffn_server_loop()
    assert not case.runner.connector.is_initialized
    assert case.peer.messages == []
