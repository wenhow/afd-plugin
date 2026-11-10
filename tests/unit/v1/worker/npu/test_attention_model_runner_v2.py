# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Host lifecycle entry seams; no NPU graph or transfer-engine I/O."""

from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from afd_plugin.v1.worker.npu import attention_model_runner_v2 as runner_module
from afd_plugin.v1.worker.npu import hccl_v41_graph_coordinator as coordinator_module
from tests.unit.fakes.attention_model_runner_v2 import (
    install_graph_runner_coordinator,
    make_graph_runner_case,
    queue_graph_runner_stop,
)
from tests.unit.fakes.hccl_v41_graph_bootstrap import (
    make_graph_bootstrap_case,
    set_graph_bootstrap_environment,
)
from tests.unit.fakes.hccl_v41_graph_eager import run_graph_eager_parent
from tests.unit.fakes.kv_connector import (
    make_native_full_kv_replay_case,
    make_native_kv_runner_case,
)


@pytest.fixture
def runner_case(monkeypatch):
    monkeypatch.setattr(runner_module, "stop_afd_npu_profiler", lambda _: None)
    monkeypatch.setattr(
        coordinator_module, "_get_graph_update_backend", lambda _: AscendSFABackend
    )
    return make_graph_runner_case


@pytest.mark.parametrize("post_kv", [False, True])
def test_native_zero_return_uses_the_current_graph_lifecycle(
    runner_case, monkeypatch, post_kv
):
    case = runner_case()
    case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    if post_kv:
        case.owner.bind_hccl_v41_graph_coordinator(
            install_graph_runner_coordinator(case)
        )
    result = object()
    monkeypatch.setattr(NPUModelRunner, "execute_model", lambda *a, **k: result)

    assert (
        case.owner.execute_model(
            SimpleNamespace(total_num_scheduled_tokens=0),
            dummy_run=True,
            is_profile=True,
        )
        is result
    )

    plans = [
        message.plan
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == "FORWARD"
    ]
    assert len(plans) == 1
    assert plans[0].mode == "profile"
    assert plans[0].graph_execution.phase == "eager"
    assert plans[0].stages[0].attention_padded_tokens == (0,)
    assert case.transport.plan is None and case.transport.graph_reservation is None


@pytest.mark.parametrize("entry", ["execute", "capture", "kv"])
def test_unbound_graph_lifecycle_fails_before_native_work(runner_case, entry):
    case = runner_case()
    with pytest.raises(RuntimeError, match="bind.*graph"):
        if entry == "execute":
            case.owner.execute_model(SimpleNamespace(total_num_scheduled_tokens=4))
        elif entry == "capture":
            case.owner.capture_model()
        else:
            case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    assert case.transport.transcript == []


def test_pre_kv_binding_rejects_another_runner_before_protocol_work(runner_case):
    case, foreign = runner_case(), runner_case()
    with pytest.raises(RuntimeError, match="idle pre-KV owners"):
        case.owner.bind_hccl_v41_graph_eager(foreign.eager, case.session)
    assert case.transport.transcript == []


def test_graph_binding_cannot_replace_an_already_bound_lifecycle(runner_case):
    case = runner_case()
    case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    with pytest.raises(RuntimeError, match="idle pre-KV owners"):
        case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    assert case.transport.transcript == []


def test_real_kv_coordinator_rejects_a_different_pre_kv_session(runner_case):
    case, foreign = runner_case(), runner_case()
    case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    coordinator = install_graph_runner_coordinator(case)
    coordinator.session = foreign.session
    with pytest.raises(RuntimeError, match="pre-KV session owners"):
        case.owner.bind_hccl_v41_graph_coordinator(coordinator)
    assert case.transport.transcript == []


@pytest.mark.parametrize("failure", [None, "cache", "native"])
def test_shutdown_retains_native_owners_until_graph_reset_succeeds(
    runner_case, monkeypatch, failure
):
    case = runner_case(fail_close=failure == "cache")
    case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    queue_graph_runner_stop(case)

    def native_shutdown(owner):
        case.events.append("native-close")
        if failure == "native":
            raise RuntimeError("native shutdown failed")

    monkeypatch.setattr(NPUModelRunner, "shutdown", native_shutdown)
    if failure:
        with pytest.raises(RuntimeError, match="synchronization|shutdown"):
            case.owner.shutdown()
        assert case.owner.hccl_v41_graph_eager is case.eager
        assert not case.transport.closed
    else:
        case.owner.shutdown()
        assert case.owner.hccl_v41_graph_eager is None
        assert case.cache.closed and case.transport.closed

    expected = ["graph-sync"]
    if failure != "cache":
        expected.append("native-close")
    if failure is None:
        expected.append("groups-close")
    assert case.events == expected
    assert any(
        direction == "recv" and message.kind == "STOP_ACK"
        for direction, _, _, message in case.transport.transcript
    )


@pytest.fixture
def automatic_graph_factory(runner_case, monkeypatch):
    set_graph_bootstrap_environment(monkeypatch)
    monkeypatch.setattr(runner_module.torch.npu, "synchronize", lambda *a: None)

    def create(**kwargs):
        case = make_graph_bootstrap_case(**kwargs)
        monkeypatch.setattr(
            NPUModelRunner,
            "load_model",
            lambda owner, *a, **k: case.native.load_model(owner, *a, **k),
        )
        monkeypatch.setattr(
            NPUModelRunner,
            "initialize_kv_cache",
            lambda owner, *a, **k: case.native.initialize_kv_cache(owner, *a, **k),
        )
        return case

    return create


@pytest.fixture
def automatic_graph_case(automatic_graph_factory):
    return automatic_graph_factory()


def test_normal_dspark_u2_load_and_kv_retain_target_query_and_graph_session(
    automatic_graph_factory,
):
    case = automatic_graph_factory(stages=2, target_query_len=3)
    case.owner.load_model()
    bridge = case.owner.hccl_v41_graph_eager
    case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    coordinator = case.owner.hccl_v41_graph_coordinator
    assert coordinator.control.cache is bridge.control.cache
    assert coordinator.session is case.owner.hccl_v41_graph_session
    assert coordinator.transport.negotiated_capabilities.target_query_len == 3
    assert coordinator.owner.decode_query_len == 3
    assert case.owner.ubatch_runner.graph_owner is coordinator.session
    assert coordinator.control.generation() == 0 and not case.transport.terminal


def test_normal_load_profiles_before_kv_then_installs_the_same_graph_owners(
    automatic_graph_case, monkeypatch
):
    case = automatic_graph_case
    case.owner.load_model()
    bridge = case.owner.hccl_v41_graph_eager
    monkeypatch.setattr(
        NPUModelRunner,
        "execute_model",
        lambda owner, *a, **k: run_graph_eager_parent(case, dummy=True, tokens=4)[1],
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda _: None
    )
    output, auxiliary = case.owner.execute_model(
        SimpleNamespace(total_num_scheduled_tokens=4), dummy_run=True, is_profile=True
    )
    case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    coordinator = case.owner.hccl_v41_graph_coordinator
    assert output.shape == (4, 4) and len(auxiliary) == 2
    assert coordinator.control.cache is bridge.control.cache
    assert coordinator.control.generation is bridge.control.generation
    assert coordinator.control.generation() == 0
    assert coordinator.session is case.owner.hccl_v41_graph_session
    assert not coordinator.control.cache.terminal
    plans = [
        message.plan
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == "FORWARD"
    ]
    assert len(plans) == 1 and plans[0].mode == "profile"
    assert plans[0].graph_execution.phase == "eager"
    assert case.transport.plan is None and case.transport.graph_reservation is None


def test_automatic_graph_settings_stay_fixed_through_real_kv_handoff(
    automatic_graph_case, monkeypatch
):
    case = automatic_graph_case
    case.owner.load_model()
    settings = case.owner.hccl_v41_graph_eager.overlap_settings
    monkeypatch.setenv("AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP", "0")
    monkeypatch.setenv("AFD_HCCL_STAGE_DIAGNOSTICS", "1")
    case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    assert case.owner.hccl_v41_graph_coordinator.overlap_settings is settings
    assert settings.graph_u2_compute_overlap and not settings.stage_diagnostics


def test_automatic_kv_cannot_write_after_acknowledged_shutdown(
    automatic_graph_case,
):
    case = automatic_graph_case
    case.owner.load_model()
    queue_graph_runner_stop(case)
    case.owner.prepare_shutdown()
    with pytest.raises(RuntimeError, match="first idle load"):
        case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    assert case.events == ["native-load"]
    assert case.owner.hccl_v41_graph_eager.control.cache.closed
    assert not case.transport.closed


@pytest.mark.parametrize("entry", ["load", "kv"])
def test_automatic_graph_reinitialization_preserves_the_first_valid_storage(
    automatic_graph_case, entry
):
    case = automatic_graph_case
    case.owner.load_model()
    if entry == "kv":
        case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    first = case.owner.hccl_v41_graph_eager.control.cache
    with pytest.raises(RuntimeError, match="single-use|first idle load"):
        if entry == "load":
            case.owner.load_model()
        else:
            case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    assert case.events == (
        ["native-load", "native-kv"] if entry == "kv" else ["native-load"]
    )
    assert case.owner.hccl_v41_graph_eager.control.cache is first
    assert not first.closed and not first.terminal and not case.transport.terminal


@pytest.mark.parametrize("entry", ["load", "kv"])
def test_automatic_graph_native_initialization_failure_retains_terminal_owners(
    automatic_graph_case, entry
):
    case = automatic_graph_case
    if entry == "load":
        case.native.fail_load = True
    else:
        case.owner.load_model()
        case.native.fail_kv = True
    with pytest.raises(RuntimeError, match="native load|native KV"):
        if entry == "load":
            case.owner.load_model()
        else:
            case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    assert case.transport.terminal and not case.transport.closed
    if entry == "kv":
        assert case.owner.hccl_v41_graph_eager.control.cache.terminal
        assert not case.owner.hccl_v41_graph_eager.control.cache.closed
    with pytest.raises(RuntimeError, match="single-use|first idle load"):
        if entry == "load":
            case.owner.load_model()
        else:
            case.owner.initialize_kv_cache(case.owner.kv_cache_config)


@pytest.mark.parametrize("post_kv", [False, True])
@pytest.mark.parametrize("sync_load", [False, True])
def test_real_native_zero_work_preserves_kv_results_and_single_retirement(
    monkeypatch, post_kv, sync_load
):
    case = make_native_kv_runner_case(monkeypatch, post_kv=post_kv, sync_load=sync_load)

    output = case.owner.execute_model(case.scheduler)

    result = output.kv_connector_output
    assert result.finished_sending == {"sent"}
    assert result.finished_recving == {"received", "failed"}
    assert result.failed_recving == {"failed"}
    assert result.invalid_block_ids == {3, 7}
    assert case.kv_transfer.bound_metadata is None
    assert case.kv_transfer.registered_views is case.kv_views
    assert not case.auxiliary_device.lookup_live
    assert not case.auxiliary_device.metadata_live
    assert case.auxiliary_device.reset_events is False
    # This is the native no_forward lifecycle, including no wait_for_save;
    # it is not a rewritten execute body or a hand-invoked pre/post sequence.
    expected = ["kv-bind", "kv-load", "kv-finish"]
    if not sync_load:
        expected = ["kv-bind", "kv-finish", "kv-load"]
    assert case.native_events == expected + [
        "kv-results",
        "kv-clear",
        "engram-finish",
        "metadata-finish",
    ]
    plans = [
        message.plan
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == "FORWARD"
    ]
    assert len(plans) == 1
    assert plans[0].mode == "target"
    assert plans[0].graph_execution.phase == "eager"
    assert plans[0].stages[0].attention_valid_tokens == (0,)
    assert plans[0].stages[0].attention_padded_tokens == (0,)
    assert case.transport.tensor_peer.messages == []
    assert case.transport.plan is None and case.transport.graph_reservation is None
    assert not case.transport.terminal


@pytest.mark.parametrize("post_kv", [False, True])
@pytest.mark.parametrize("sync_load", [False, True])
def test_real_native_zero_work_load_error_retires_engram_without_a_false_parent(
    monkeypatch, post_kv, sync_load
):
    case = make_native_kv_runner_case(
        monkeypatch, post_kv=post_kv, sync_load=sync_load, fail_load=True
    )

    with pytest.raises(RuntimeError, match="controlled KV load failure") as raised:
        case.owner.execute_model(case.scheduler)

    assert raised.value is case.kv_transfer.load_error
    assert case.kv_transfer.bound_metadata is case.scheduler.kv_connector_metadata
    assert case.kv_transfer.registered_views is case.kv_views
    assert not case.auxiliary_device.lookup_live
    assert not case.auxiliary_device.metadata_live
    assert case.auxiliary_device.reset_events is True
    expected = ["kv-bind", "kv-load"]
    if not sync_load:
        expected = ["kv-bind", "kv-finish", "kv-load"]
    assert case.native_events == expected + ["engram-failed", "metadata-finish"]
    assert not any(
        direction == "send" and message.kind in ("FORWARD", "COMPLETE")
        for direction, _, _, message in case.transport.transcript
    )
    assert case.transport.tensor_peer.messages == []
    assert case.transport.terminal and not case.transport.closed
    assert case.cache.terminal and not case.cache.closed


def test_real_native_full_replay_fixture_returns_none_before_sampling(monkeypatch):
    case = make_native_full_kv_replay_case(monkeypatch)
    resources = case.recorded_resources

    assert case.owner.execute_model(case.scheduler) is None

    assert case.kv_transfer.bound_metadata is case.scheduler.kv_connector_metadata
    assert case.kv_transfer.registered_views is case.kv_views
    assert case.native_events == [
        "inputs-ready",
        "attn-ready",
        "kv-bind",
        "kv-load",
        "replay-device-boundary",
        "metadata-replay-finish",
        "kv-finish",
        "engram-finish",
        "metadata-finish",
    ]
    assert not case.auxiliary_device.lookup_live
    assert not case.auxiliary_device.metadata_live
    assert not case.auxiliary_device.replay_live
    assert case.auxiliary_device.reset_events is False
    entry = case.cache.find(resources.bank.key, case.generation())
    assert entry.resources is resources and entry.graph is case.recorded_graph
    assert case.manager.hidden_states is resources.hidden_states
    assert not case.recorded_graph.closed
    messages = [
        message
        for direction, _, _, message in case.transport.transcript[
            len(case.replay_baseline) :
        ]
        if direction == "send" and message.kind in ("FORWARD", "COMPLETE")
    ]
    assert [message.kind for message in messages] == ["FORWARD", "COMPLETE"]
    plan = messages[0].plan
    assert plan.graph_execution.phase == "replay"
    assert plan.stages[0].attention_valid_tokens == (case.batch.num_tokens,)
    assert plan.stages[0].attention_padded_tokens == resources.bank.stage_rows
    assert messages[1].transaction_id == plan.transaction_id
    assert messages[1].plan_digest == plan.digest
    assert case.transport.tensor_peer.messages == []
    assert all(endpoint.messages == [] for endpoint in case.model_case.endpoints)
    assert case.transport.plan is None and case.transport.graph_reservation is None
    assert not case.transport.terminal


def test_real_native_full_replay_fixture_failure_retains_kv_and_graph_owners(
    monkeypatch,
):
    case = make_native_full_kv_replay_case(monkeypatch, fail_replay=True)
    resources = case.recorded_resources

    with pytest.raises(RuntimeError, match="controlled FULL replay failure") as raised:
        case.owner.execute_model(case.scheduler)

    assert raised.value is case.replay_error
    assert case.kv_transfer.bound_metadata is case.scheduler.kv_connector_metadata
    assert case.kv_transfer.registered_views is case.kv_views
    assert case.native_events == [
        "inputs-ready",
        "attn-ready",
        "kv-bind",
        "kv-load",
        "replay-device-boundary",
        "metadata-replay-finish",
        "engram-failed",
        "metadata-finish",
    ]
    assert not case.auxiliary_device.lookup_live
    assert not case.auxiliary_device.metadata_live
    assert not case.auxiliary_device.replay_live
    assert case.auxiliary_device.reset_events is True
    assert case.manager.hidden_states is resources.hidden_states
    assert not case.recorded_graph.closed
    assert not case.cache.is_empty
    assert case.cache.terminal and not case.cache.closed
    assert case.transport.terminal and not case.transport.closed
    messages = [
        message
        for direction, _, _, message in case.transport.transcript[
            len(case.replay_baseline) :
        ]
        if direction == "send" and message.kind in ("FORWARD", "COMPLETE")
    ]
    assert [message.kind for message in messages] == ["FORWARD"]
    assert messages[0].plan.graph_execution.phase == "replay"
    assert case.transport.tensor_peer.messages == []
    assert all(endpoint.messages == [] for endpoint in case.model_case.endpoints)
