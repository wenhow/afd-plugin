# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Real control/CPU preparation and resource ownership; no device graph claim."""

from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2
from vllm_ascend.worker.v2.model_states.default import AscendModelState

from afd_plugin.connectors.npu import hccl_graph_pipeline as pipeline_module
from afd_plugin.connectors.npu.hccl_graph_pipeline import (
    HcclGraphComputePipeline,
    HcclGraphPipelinePolicy,
)
from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank
from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.v1.worker.npu.hccl_v41_graph_coordinator import HcclV41GraphCoordinator
from tests.unit.fakes.hccl_graph_pipeline import attach_pipeline
from tests.unit.fakes.hccl_v41_graph_coordinator import (
    cpu_allocate_bank,
    cpu_retained_capture,
    cpu_runner_capture,
    dispatch_runtime,
    install_runtime_replay_boundary,
    make_coordinator_case,
    native_keys,
    prepare_runtime_full,
    queue_preflight,
    replace_cached_bundle,
    replace_cached_pipeline,
    set_runtime_rows,
)
from tests.unit.fakes.input_batch import snapshot_ascend_child


@pytest.fixture
def coordinator_case(monkeypatch):
    monkeypatch.setattr(
        "afd_plugin.v1.worker.npu.hccl_v41_graph_coordinator._get_graph_update_backend",
        lambda _: AscendSFABackend,
    )
    monkeypatch.setattr(torch.npu, "synchronize", lambda *args: None)

    def build(
        *, pipeline_policy=None, overlap_settings=None, diagnostic_sink=None, **kwargs
    ):
        case = make_coordinator_case(**kwargs)
        case.coordinator = HcclV41GraphCoordinator(
            case.owner,
            case.control,
            case.session,
            next_transaction_id=case.next_transaction_id,
            pipeline_policy=pipeline_policy,
            overlap_settings=overlap_settings,
            diagnostic_sink=diagnostic_sink,
        )
        return case

    return build


@pytest.fixture
def capture_case(coordinator_case, monkeypatch):
    # Explicit CPU boundaries exclude actual manager/device construction,
    # backend metadata kernels, memory accounting and graph capture/replay.
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda _: None
    )
    monkeypatch.setattr(
        "afd_plugin.v1.worker.npu.hccl_v41_ubatch_runner.current_platform.set_additional_forward_context",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        AscendModelState, "prepare_attn", staticmethod(snapshot_ascend_child)
    )
    monkeypatch.setattr(CudaGraphManager, "capture", cpu_retained_capture)

    def build(**kwargs):
        case = coordinator_case(**kwargs)
        monkeypatch.setattr(
            NPUModelRunnerV2, "capture_model", partial(cpu_runner_capture, case)
        )
        monkeypatch.setattr(
            HcclGraphStageBank, "allocate", partial(cpu_allocate_bank, case)
        )
        return case

    return build


@pytest.fixture
def pipeline_capture_case(capture_case, monkeypatch):
    monkeypatch.setattr(
        HcclGraphComputePipeline,
        "allocate",
        lambda bank, **kwargs: (
            attach_pipeline(monkeypatch, pipeline_module, bank, **kwargs).pipeline
        ),
    )

    def build(*, stages=2, hybrid=True, overlap_settings=None, **kwargs):
        return capture_case(
            stages=stages,
            pipeline_policy=HcclGraphPipelinePolicy(hybrid=hybrid)
            if overlap_settings is None
            else None,
            overlap_settings=overlap_settings,
            **kwargs,
        )

    return build


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("hybrid", [False, True])
def test_native_capture_pipeline_bundle_survives_preparation_and_replay(
    pipeline_capture_case, stages, hybrid
):
    case = pipeline_capture_case(stages=stages, hybrid=hybrid)
    assert case.coordinator.preflight_capture()
    assert case.coordinator.capture_native() == 23
    entry = case.cache.find(native_keys(case)[0], 0)
    pipeline = entry.resources.pipeline
    assert pipeline.hybrid is hybrid and pipeline.bank is entry.resources.bank
    assert entry.resources.execution.pipeline is pipeline
    assert entry.resources.execution.parent_body.stages.pipeline is pipeline
    case.transport.states[0] = "ready"
    install_runtime_replay_boundary(case)
    with case.coordinator.execution_scope(
        mode="target", valid_tokens=4, dummy_run=False
    ):
        descriptor = dispatch_runtime(case)
        prepare_runtime_full(case, descriptor)
        output, auxiliary = case.manager.run_fullgraph(descriptor)
    assert output is case.manager.hidden_states
    assert auxiliary is case.manager.aux_hidden_states
    assert case.cache.find(entry.key, 0).resources.pipeline is pipeline
    assert case.transport.plan is None and not case.transport.terminal


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize(
    "compute,hybrid,three,recv,cross",
    [
        (True, True, True, True, True),
        (True, False, False, False, False),
        (False, True, True, True, True),
    ],
)
def test_overlap_settings_preserve_graph_when_compute_pipeline_is_disabled(
    pipeline_capture_case, stages, compute, hybrid, three, recv, cross
):
    settings = HcclOverlapSettings(
        graph_u2_compute_overlap=compute,
        graph_u2_hybrid_dag=hybrid,
        graph_u2_attention_three_stream=three,
        graph_u2_ffn_recv_stream=recv,
        graph_u2_ffn_cross_layer=cross,
    )
    case = pipeline_capture_case(stages=stages, overlap_settings=settings)
    assert case.coordinator.preflight_capture()
    case.coordinator.capture_native()
    entry = case.cache.find(native_keys(case)[0], 0)
    assert entry is not None and entry.resources.overlap_settings == settings
    pipeline = entry.resources.pipeline
    assert (pipeline is not None) == (stages > 1 and compute)
    if pipeline is not None:
        assert pipeline.policy == HcclGraphPipelinePolicy.from_selection(
            pipeline.selection
        )
        assert pipeline.selection.role == "attention"
        assert pipeline.selection.phase == "graph_capture"
        assert pipeline.selection.physical_send_stream == "parent"
    case.transport.states[0] = "ready"
    install_runtime_replay_boundary(case)
    with case.coordinator.execution_scope(
        mode="target", valid_tokens=4, dummy_run=False
    ):
        descriptor = dispatch_runtime(case)
        prepare_runtime_full(case, descriptor)
        output, _ = case.manager.run_fullgraph(descriptor)
    assert output is case.manager.hidden_states
    assert case.cache.find(entry.key, 0).resources.pipeline is pipeline
    assert case.transport.plan is None and not case.transport.terminal


@pytest.mark.parametrize("stages,compute", [(1, True), (2, False), (2, True)])
def test_captured_graph_rejects_changed_settings_even_without_side_pipeline(
    pipeline_capture_case, stages, compute
):
    settings = HcclOverlapSettings(graph_u2_compute_overlap=compute)
    case = pipeline_capture_case(stages=stages, overlap_settings=settings)
    assert case.coordinator.preflight_capture()
    case.coordinator.capture_native()
    case.transport.states[0] = "ready"
    case.coordinator = HcclV41GraphCoordinator(
        case.owner,
        case.control,
        case.session,
        next_transaction_id=case.next_transaction_id,
        overlap_settings=replace(settings, graph_u2_attention_three_stream=False),
    )
    before = tuple(sent(case, "FORWARD"))
    with (
        pytest.raises(ValueError, match="cache bundle does not own"),
        case.coordinator.execution_scope(
            mode="target", valid_tokens=4, dummy_run=False
        ),
    ):
        dispatch_runtime(case)
    assert tuple(sent(case, "FORWARD")) == before
    assert case.transport.plan is None and case.transport.terminal


def test_one_capture_session_selects_compute_per_native_u1_u2_descriptor(
    pipeline_capture_case,
):
    descriptors = tuple(
        BatchExecutionDescriptor(
            CUDAGraphMode.FULL, 4, 4, uniform_token_count=1, num_ubatches=stages
        )
        for stages in (1, 2)
    )
    case = pipeline_capture_case(
        descriptors=descriptors, overlap_settings=HcclOverlapSettings()
    )
    assert case.coordinator.preflight_capture()
    case.coordinator.capture_native()
    entries = tuple(case.cache.find(key, 0) for key in native_keys(case))
    assert len(entries) == 2
    assert entries[0].resources.pipeline is None
    assert entries[1].resources.pipeline.selection.num_stages == 2
    assert not case.transport.terminal


@pytest.mark.parametrize("stages,compute", [(1, True), (2, False), (2, True)])
def test_native_capture_reports_configured_and_actual_rank_schedule(
    pipeline_capture_case, stages, compute
):
    records = []
    case = pipeline_capture_case(
        stages=stages,
        overlap_settings=HcclOverlapSettings(
            graph_u2_compute_overlap=compute,
            graph_u2_attention_three_stream=False,
            stage_diagnostics=True,
            ffn_compute_sync_diagnostics=True,
        ),
        diagnostic_sink=records.append,
    )
    assert case.coordinator.preflight_capture()
    case.coordinator.capture_native()
    assert [record["phase"] for record in records] == ["graph_warmup", "graph_capture"]
    for record in records:
        assert record["event"] == "configuration"
        assert record["role"] == "attention"
        assert record["world_rank"] == case.transport.world_rank
        assert record["actual.compute_stream"] == (
            "compute" if stages > 1 and compute else "parent"
        )
        assert record["actual.recv_stream"] == "parent"
        assert record["actual.ffn_compute_sync"] is False
        assert record["configured.AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"] == compute
        assert record["effective.AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"] == (
            stages > 1 and compute
        )


@pytest.mark.parametrize("mismatch", ["pipeline", "policy"])
def test_same_bank_pipeline_or_policy_replacement_rejects_before_new_payload(
    pipeline_capture_case, mismatch
):
    case = pipeline_capture_case()
    assert case.coordinator.preflight_capture()
    case.coordinator.capture_native()
    case.transport.states[0] = "ready"
    entry = case.cache.find(native_keys(case)[0], 0)
    if mismatch == "pipeline":
        replace_cached_pipeline(case, entry)
    else:
        case.coordinator = HcclV41GraphCoordinator(
            case.owner,
            case.control,
            case.session,
            next_transaction_id=case.next_transaction_id,
            pipeline_policy=HcclGraphPipelinePolicy(hybrid=False),
        )
    before = tuple(sent(case, "FORWARD"))
    with (
        pytest.raises(ValueError, match="cache bundle does not own"),
        case.coordinator.execution_scope(
            mode="target", valid_tokens=4, dummy_run=False
        ),
    ):
        dispatch_runtime(case)
    assert tuple(sent(case, "FORWARD")) == before
    assert case.transport.plan is None and case.transport.terminal


def sent(case, kind):
    return [
        message
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == kind
    ]


def test_capture_preflight_uses_an_honest_miss_and_retires_only_zero_rows(
    coordinator_case,
):
    case = coordinator_case()
    assert case.coordinator.preflight_capture()
    query = sent(case, "GRAPH_QUERY")[0].graph_candidate
    decision = sent(case, "GRAPH_DECISION")[0].graph_decision
    plan = sent(case, "FORWARD")[0].plan

    assert query.phase == "replay" and query.mode == "warmup"
    assert query.key == native_keys(case)[0]
    assert query.capture_sequence_digest is not None
    assert decision.phase == "eager" and decision.reason == "cache_miss"
    assert plan.stages[0].attention_padded_tokens == (0,)
    assert case.transport.plan is None and case.transport.graph_reservation is None
    assert case.cache.is_empty and not case.transport.terminal


@pytest.mark.parametrize("unsupported", ["local-producer", "remote-peer"])
def test_unsupported_peer_makes_all_attention_skip_native_capture(
    coordinator_case, unsupported
):
    case = coordinator_case()
    if unsupported == "local-producer":
        case.model_case.model.has_engram = True
        case.model_case.model.can_capture_engram_producer = lambda: False
    else:
        case.transport.states[0] = "unsupported"

    assert not case.coordinator.preflight_capture()
    assert case.coordinator.capture_native() == 0
    assert sent(case, "GRAPH_DECISION")[0].graph_decision.reason == "unsupported"
    assert case.cache.is_empty and not case.transport.terminal


@pytest.mark.parametrize("prefix", [None, 0])
def test_empty_capture_list_still_agrees_and_retires_the_empty_parent(
    coordinator_case, prefix
):
    case = coordinator_case(descriptors=(), prefix=prefix)
    assert not case.coordinator.preflight_capture()
    assert case.coordinator.capture_native() == 0
    candidate = sent(case, "GRAPH_QUERY")[0].graph_candidate
    assert candidate.phase == "eager" and candidate.key is None
    assert candidate.capture_sequence_digest is not None
    assert case.transport.plan is None and case.transport.graph_reservation is None


@pytest.mark.parametrize(
    "difference", ["order", "count", "second-key", "prefix", "metadata", "empty"]
)
def test_complete_capture_sequence_mismatch_is_unsupported_before_factory(
    coordinator_case, difference
):
    first = BatchExecutionDescriptor(
        CUDAGraphMode.FULL, 4, 4, uniform_token_count=1, num_ubatches=2
    )
    second = replace(first, num_tokens=8, num_reqs=8)
    descriptors = () if difference == "empty" else (first, second)
    case = coordinator_case(role_rank=1, descriptors=descriptors)
    donor = coordinator_case(descriptors=(first, second))
    keys, prefix = list(native_keys(donor)), None
    if difference == "order":
        keys.reverse()
    elif difference == "count":
        keys.pop()
    elif difference == "second-key":
        keys[1] = replace(keys[1], metadata_shapes=((8, 4),))
    elif difference == "prefix":
        prefix = 2
    elif difference == "metadata":
        keys[0] = replace(keys[0], metadata_shapes=((4, 4),))
    queue_preflight(case, keys, prefix=prefix)

    assert not case.coordinator.preflight_capture()
    assert sent(case, "GRAPH_REPORT")[0].graph_report.availability == "unsupported"
    assert case.coordinator.capture_native() == 0
    assert case.transport.plan is None and not case.transport.terminal


def test_empty_coordinator_query_is_unsupported_even_on_a_nonempty_responder(
    coordinator_case,
):
    case = coordinator_case(role_rank=1)
    queue_preflight(case, (), empty=True)
    assert not case.coordinator.preflight_capture()
    assert sent(case, "GRAPH_REPORT")[0].graph_report.availability == "unsupported"
    assert case.transport.plan is None


@pytest.mark.parametrize(
    "foreign", ["manager-owner", "manager-ubatch", "session", "state-model", "wire"]
)
def test_foreign_native_owners_are_rejected_before_protocol(coordinator_case, foreign):
    case = coordinator_case()
    if foreign == "manager-owner":
        case.manager.model_runner = SimpleNamespace()
    elif foreign == "manager-ubatch":
        case.manager.ubatch_runner = None
    elif foreign == "state-model":
        case.owner.model_state.model = torch.nn.Identity()
    elif foreign == "wire":
        case.owner.hccl_v41_transport = SimpleNamespace()
    else:
        case.owner.ubatch_runner.graph_owner = SimpleNamespace()

    with pytest.raises(ValueError, match="owners|layout|session"):
        HcclV41GraphCoordinator(
            case.owner,
            case.control,
            case.session,
            next_transaction_id=case.next_transaction_id,
        )
    assert case.transport.transcript == []


@pytest.mark.parametrize(
    "replacement",
    [
        "model",
        "model_state",
        "input_buffers",
        "block_tables",
        "kv_cache_config",
        "kv_caches",
        "intermediate_tensors",
        "attn_groups",
        "prefix",
        "descriptors",
    ],
)
def test_replaced_native_storage_cannot_enter_the_capture_lifecycle(
    coordinator_case, replacement
):
    case = coordinator_case()
    assert case.coordinator.preflight_capture()
    if replacement == "kv_caches":
        case.owner.kv_caches[0] = torch.zeros(1)
    elif replacement == "prefix":
        case.manager._max_full_descs_to_capture = 1
    elif replacement == "descriptors":
        case.manager._capture_descs[CUDAGraphMode.FULL] = []
    else:
        setattr(
            case.owner,
            replacement,
            SimpleNamespace() if replacement != "attn_groups" else [],
        )

    with pytest.raises(ValueError, match="ownership changed"):
        case.coordinator.capture_native()
    assert case.transport.terminal and case.cache.terminal
    assert len(sent(case, "GRAPH_QUERY")) == 1


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("profile_only", [False, True])
def test_capture_preserves_native_parent_outputs_and_retains_the_whole_generation(
    capture_case, stages, profile_only
):
    case = capture_case(stages=stages)
    assert case.coordinator.preflight_capture()
    parent = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    with override_forward_context(parent):
        assert case.coordinator.capture_native(profile_only=profile_only) == (
            17 if profile_only else 23
        )
        assert get_forward_context() is parent
    key = native_keys(case)[0]
    entry = case.cache.find(key, 0)

    assert entry is not None and entry.graph is case.manager.graphs[case.descriptor]
    assert entry.resources.model_state is case.owner.model_state
    assert entry.resources.input_buffers is case.owner.input_buffers
    assert entry.resources.block_tables is case.owner.block_tables
    assert entry.resources.ubatch_runner is case.owner.ubatch_runner
    assert entry.resources.hidden_states is case.manager.hidden_states
    assert entry.resources.hidden_states.shape == (4, 4)
    assert len(entry.resources.auxiliary) == 2
    assert [
        message.graph_decision.phase for message in sent(case, "GRAPH_DECISION")
    ] == ["eager", "warmup", "capture"]
    assert case.transport.plan is None and case.transport.graph_reservation is None
    assert not case.transport.terminal
    assert case.capture_generations[0][2] is not case.capture_generations[1][2]


@pytest.mark.parametrize("failure", ["native", "prepare"])
def test_capture_failure_retains_terminal_session_and_restores_native_bindings(
    capture_case, failure
):
    case = capture_case()
    assert case.coordinator.preflight_capture()
    if failure == "native":
        case.manager.capture_failure = "native"
    else:
        case.boundary_failure = "prepare"
    original = CudaGraphManager.capture

    with pytest.raises(RuntimeError, match="native .* failed"):
        case.coordinator.capture_native()

    assert CudaGraphManager.capture is original
    assert case.transport.terminal and case.cache.terminal


def test_capture_requires_preflight_before_native_lifecycle(coordinator_case):
    case = coordinator_case()
    with pytest.raises(ValueError, match="eligibility preflight"):
        case.coordinator.capture_native()
    assert case.transport.transcript == []


@pytest.mark.parametrize("stages", [1, 2])
def test_runtime_full_uses_current_child_metadata_and_completes_native_replay(
    capture_case, stages
):
    case = capture_case(stages=stages)
    assert case.coordinator.preflight_capture()
    case.coordinator.capture_native()
    case.transport.states[0] = "ready"
    install_runtime_replay_boundary(case)
    previous_metadata = case.owner.model_state.attn_metadata
    original = case.manager.run_fullgraph

    with case.coordinator.execution_scope(
        mode="target", valid_tokens=4, dummy_run=False
    ):
        descriptor = dispatch_runtime(case)
        assert descriptor.cg_mode == CUDAGraphMode.FULL
        prepare_runtime_full(case, descriptor)
        output, auxiliary = case.manager.run_fullgraph(descriptor)

    assert output is case.manager.hidden_states and len(auxiliary) == 2
    assert case.runtime_metadata[0] is not previous_metadata
    torch.testing.assert_close(
        case.runtime_metadata[0]["child"].input_ids,
        torch.arange(
            4 // stages, dtype=case.runtime_metadata[0]["child"].input_ids.dtype
        ),
    )
    assert case.owner.model_state.attn_metadata is previous_metadata
    assert case.manager.run_fullgraph is original
    entry = case.cache.find(native_keys(case)[0], 0)
    assert [count.item() for count in entry.resources.bank.valid_counts] == [
        4 // stages
    ] * stages
    assert sent(case, "GRAPH_DECISION")[-1].graph_decision.phase == "replay"
    assert case.transport.plan is None and case.transport.graph_reservation is None
    assert not case.transport.terminal


@pytest.mark.parametrize("stages", [1, 2])
def test_replay_failure_restores_metadata_and_retains_the_active_plan(
    capture_case, stages
):
    case = capture_case(stages=stages)
    assert case.coordinator.preflight_capture()
    case.coordinator.capture_native()
    case.transport.states[0] = "ready"
    install_runtime_replay_boundary(case, fail=True)
    previous_metadata = case.owner.model_state.attn_metadata
    original = case.manager.run_fullgraph

    with (
        pytest.raises(RuntimeError, match="native replay failed"),
        case.coordinator.execution_scope(
            mode="target", valid_tokens=4, dummy_run=False
        ),
    ):
        descriptor = dispatch_runtime(case)
        prepare_runtime_full(case, descriptor)
        case.manager.run_fullgraph(descriptor)

    assert case.owner.model_state.attn_metadata is previous_metadata
    assert case.manager.run_fullgraph is original
    assert case.transport.terminal and case.cache.terminal
    assert case.transport.plan is not None


def test_missing_native_replay_does_not_claim_completion(capture_case):
    case = capture_case()
    assert case.coordinator.preflight_capture()
    case.coordinator.capture_native()
    case.transport.states[0] = "ready"
    install_runtime_replay_boundary(case)

    with (
        pytest.raises(RuntimeError, match="did not complete"),
        case.coordinator.execution_scope(
            mode="target", valid_tokens=4, dummy_run=False
        ),
    ):
        prepare_runtime_full(case, dispatch_runtime(case))

    assert case.transport.terminal and case.transport.plan is not None
    assert case.runtime_metadata == []


@pytest.mark.parametrize("change", ["descriptor", "generation"])
def test_foreign_or_stale_replay_is_rejected_before_native_lifecycle(
    capture_case, change
):
    case = capture_case()
    assert case.coordinator.preflight_capture()
    case.coordinator.capture_native()
    case.transport.states[0] = "ready"
    install_runtime_replay_boundary(case)
    previous_metadata = case.owner.model_state.attn_metadata

    with (
        pytest.raises(ValueError, match="descriptor|storage or metadata changed"),
        case.coordinator.execution_scope(
            mode="target", valid_tokens=4, dummy_run=False
        ),
    ):
        descriptor = dispatch_runtime(case)
        prepare_runtime_full(case, descriptor)
        if change == "descriptor":
            descriptor = replace(descriptor, num_ubatches=1)
        else:
            case.generation.value += 1
        case.manager.run_fullgraph(descriptor)

    assert case.owner.model_state.attn_metadata is previous_metadata
    assert case.runtime_metadata == []
    assert case.transport.terminal


def test_global_cache_miss_selects_fresh_eager_inputs_before_preparation(capture_case):
    case = capture_case()
    assert case.coordinator.preflight_capture()
    case.coordinator.capture_native()
    case.transport.states[0] = "ready"
    set_runtime_rows(case, 2)

    with (
        pytest.raises(RuntimeError, match="CPU boundary before eager preparation"),
        case.coordinator.execution_scope(
            mode="target", valid_tokens=2, dummy_run=False
        ),
    ):
        descriptor = dispatch_runtime(case)
        assert descriptor.cg_mode == CUDAGraphMode.NONE
        assert descriptor.num_tokens == descriptor.num_reqs == 2
        assert descriptor.num_ubatches == 2
        assert case.owner.model_state.prepare_inputs(None, None) == {
            "producer_mode": CUDAGraphMode.NONE
        }
        raise RuntimeError("CPU boundary before eager preparation")

    decision = sent(case, "GRAPH_DECISION")[-1].graph_decision
    assert decision.phase == "eager" and decision.reason == "cache_miss"
    assert decision.padded_tokens == (1, 1)
    assert case.transport.terminal


@pytest.mark.parametrize("purpose", ["profile", "dummy", "unsupported-engram"])
def test_unsupported_native_inputs_reserve_eager_before_parent_input_writes(
    capture_case, purpose
):
    case = capture_case()
    dummy = purpose != "unsupported-engram"
    if not dummy:
        case.model_case.model.has_engram = True
        case.model_case.model.can_capture_engram_producer = lambda: False
    mode = purpose if dummy else "target"

    with (
        pytest.raises(RuntimeError, match="CPU boundary before eager preparation"),
        case.coordinator.execution_scope(
            mode=mode, valid_tokens=0 if dummy else 4, dummy_run=dummy
        ),
    ):
        descriptor = dispatch_runtime(
            case, dummy=dummy, tokens=4, need_eager=purpose == "profile"
        )
        assert descriptor.cg_mode == CUDAGraphMode.NONE
        assert case.owner.model_state.prepare_inputs(None, None) == {
            "producer_mode": CUDAGraphMode.NONE
        }
        raise RuntimeError("CPU boundary before eager preparation")

    query = sent(case, "GRAPH_QUERY")[-1].graph_candidate
    assert query.phase == "eager" and query.key is None
    assert sent(case, "GRAPH_DECISION")[-1].graph_decision.mode == mode


@pytest.mark.parametrize("native_dispatch", [False, True])
def test_zero_native_execution_retires_an_honest_empty_parent(
    coordinator_case, native_dispatch
):
    case = coordinator_case()
    set_runtime_rows(case, 0)
    with case.coordinator.execution_scope(
        mode="target", valid_tokens=0, dummy_run=False
    ):
        if native_dispatch:
            descriptor = dispatch_runtime(case)
            assert descriptor.cg_mode == CUDAGraphMode.NONE
            assert descriptor.num_tokens == descriptor.num_reqs == 0

    plan = sent(case, "FORWARD")[-1].plan
    assert plan.stages[0].attention_padded_tokens == (0,)
    assert case.transport.plan is None and case.transport.graph_reservation is None
    assert not case.transport.terminal


def test_nonempty_native_return_without_dispatch_cannot_retire_an_empty_parent(
    coordinator_case,
):
    case = coordinator_case()
    with (
        pytest.raises(RuntimeError, match="before graph dispatch"),
        case.coordinator.execution_scope(
            mode="target", valid_tokens=4, dummy_run=False
        ),
    ):
        pass
    assert case.transport.transcript == []
    assert case.transport.terminal


@pytest.mark.parametrize(
    "mismatch", ["bank", "body-bank", "state", "inputs", "output", "generation"]
)
def test_ready_graph_rejects_a_cache_bundle_with_foreign_captured_resources(
    capture_case, mismatch
):
    case = capture_case()
    assert case.coordinator.preflight_capture()
    case.coordinator.capture_native()
    case.transport.states[0] = "ready"
    entry = case.cache.find(native_keys(case)[0], 0)
    resources = entry.resources
    if mismatch in ("bank", "body-bank"):
        bank = cpu_allocate_bank(
            case, case.transport, entry.key, device=case.owner.device
        )
        resources = replace(resources, bank=bank)
        if mismatch == "body-bank":
            resources.execution.bank = bank
    elif mismatch == "state":
        resources = replace(resources, model_state=SimpleNamespace())
    elif mismatch == "inputs":
        resources = replace(resources, input_buffers=SimpleNamespace())
    elif mismatch == "output":
        resources = replace(
            resources, hidden_states=torch.zeros_like(resources.hidden_states)
        )
    else:
        resources.execution.reservation = replace(
            resources.execution.reservation, generation=1
        )
    replace_cached_bundle(case, entry, resources)
    before = tuple(sent(case, "FORWARD"))

    with (
        pytest.raises(ValueError, match="cache bundle does not own"),
        case.coordinator.execution_scope(
            mode="target", valid_tokens=4, dummy_run=False
        ),
    ):
        dispatch_runtime(case)

    assert tuple(sent(case, "FORWARD")) == before
    assert case.transport.plan is None and case.transport.terminal
