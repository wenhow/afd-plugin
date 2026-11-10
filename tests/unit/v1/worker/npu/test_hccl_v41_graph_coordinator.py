# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Real control/CPU preparation and resource ownership; no device graph claim."""

from contextlib import nullcontext
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
from vllm.forward_context import (
    get_forward_context,
    override_forward_context,
    set_forward_context,
)
from vllm.v1.worker.gpu import model_runner as native_runner
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
from afd_plugin.model_executor.models import forward_context as context_module
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    HCCL_V41_PARENT_FORWARD_KEY,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph_coordinator import HcclV41GraphCoordinator
from tests.unit.fakes.hccl_graph_pipeline import attach_pipeline
from tests.unit.fakes.hccl_v41_graph_coordinator import (
    cpu_allocate_bank,
    cpu_retained_capture,
    cpu_runner_capture,
    cpu_target_manager_capture,
    dispatch_runtime,
    install_runtime_replay_boundary,
    make_coordinator_case,
    native_keys,
    prepare_native_draft_capture,
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


@pytest.fixture
def native_draft_capture_case(capture_case, monkeypatch):
    # Keep GPUModelRunner.capture_model's real target -> draft lifecycle. Only
    # device memory/workspace and model-manager factory execution are bounded
    # CPU collaborators; no CUDA/NPU graph or native draft kernel is simulated.
    monkeypatch.setattr(native_runner, "freeze_gc_for_cudagraph_capture", nullcontext)
    monkeypatch.setattr(native_runner, "use_workspace_lane", lambda _: nullcontext())
    monkeypatch.setattr(native_runner, "lock_workspace", lambda: None)
    monkeypatch.setattr(
        native_runner,
        "compilation_counter",
        SimpleNamespace(num_gpu_runner_capture_triggers=0),
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda: (1024, 2048))

    def build(*, stages, fail_draft):
        case = capture_case(stages=stages, target_query_len=3, engram=True)
        monkeypatch.setattr(
            NPUModelRunnerV2,
            "capture_model",
            native_runner.GPUModelRunner.capture_model,
        )
        monkeypatch.setattr(
            case.manager,
            "capture",
            partial(cpu_target_manager_capture, case, case.manager),
        )
        return prepare_native_draft_capture(case, fail_draft=fail_draft)

    return build


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("fail_draft", [False, True])
def test_native_capture_keeps_draft_local(  # long-test: native lifecycle isolation
    native_draft_capture_case, stages, fail_draft
):
    case = native_draft_capture_case(stages=stages, fail_draft=fail_draft)
    assert case.coordinator.preflight_capture()
    previous = SimpleNamespace(additional_kwargs={"caller": "retained"})
    with override_forward_context(previous):
        if fail_draft:
            with pytest.raises(RuntimeError, match="local draft forward failed"):
                case.coordinator.capture_native()
        else:
            assert case.coordinator.capture_native() == 0
        assert get_forward_context() is previous

    assert case.capture_order == [
        "target-published",
        "native-draft-capture-returned",
        "local-draft-forward",
        *([] if fail_draft else ["native-kv-reset"]),
    ]
    assert not case.draft.query_cudagraph_manager.needs_capture()
    assert case.draft.query_cudagraph_manager.graphs == {}
    assert case.draft.vllm_config.model_config.enforce_eager
    assert case.draft.sample_idx_mapping.tolist() == [-1, -1, -1]
    context = case.draft_contexts[0]
    assert context.attn_metadata is case.draft_metadata
    assert all(context.attn_metadata is not item for item in case.target_metadata)
    assert context.cudagraph_runtime_mode == CUDAGraphMode.NONE
    assert HCCL_V41_PARENT_FORWARD_KEY not in (context.additional_kwargs or {})
    assert context_module.get_afd_metadata_from_forward_context(context) is None
    assert tuple(case.transport.transcript) == case.target_transcript
    assert case.transport.plan is None and case.transport.graph_reservation is None
    assert case.target_produced
    assert len(case.model_case.producer_parents) == len(case.target_produced)
    for actual, expected in zip(
        case.model_case.producer_parents, case.target_produced, strict=True
    ):
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor)
    entry = case.target_entry
    assert (
        case.manager.hidden_states
        is case.target_output
        is entry.resources.hidden_states
    )
    assert case.target_output.data_ptr() == case.target_storage
    torch.testing.assert_close(case.target_output, case.target_value)
    for actual, retained, value in zip(
        case.manager.aux_hidden_states,
        case.target_auxiliary,
        case.target_auxiliary_values,
        strict=True,
    ):
        assert actual is retained
        torch.testing.assert_close(actual, value)
    entry.graph.replay()  # A public retained handle must not have been reset by draft.
    assert not case.cache.is_empty and not case.cache.closed
    if fail_draft:
        assert case.cache.terminal and case.transport.terminal
    else:
        assert case.cache.find(entry.key, entry.generation) is entry
        torch.testing.assert_close(
            case.draft_output, 2 * torch.arange(3)[:, None] + 100
        )
    with set_forward_context(case.target_metadata[-1], case.owner.vllm_config):
        fresh = get_forward_context()
        assert HCCL_V41_PARENT_FORWARD_KEY not in (fresh.additional_kwargs or {})
        assert context_module.get_afd_metadata_from_forward_context() is None


@pytest.mark.parametrize("query_len", [2, 3])
@pytest.mark.parametrize("stages", [1, 2])
def test_target_query_capture_and_replay_retain_native_request_geometry(
    capture_case, query_len, stages
):
    case = capture_case(stages=stages, target_query_len=query_len)
    assert case.coordinator.preflight_capture()
    assert case.coordinator.capture_native() == 23
    entry = case.cache.find(native_keys(case)[0], 0)
    assert entry.key.num_reqs == 3
    assert entry.key.num_tokens == 3 * query_len
    assert entry.key.uniform_token_count == entry.key.max_query_len == query_len
    assert entry.resources.hidden_states.shape == (3 * query_len, 4)
    assert len(entry.resources.auxiliary) == 2
    assert case.capture_generations[0][2].num_reqs == 3
    assert list(case.capture_generations[0][2].num_scheduled_tokens) == [query_len] * 3
    case.transport.states[0] = "ready"
    install_runtime_replay_boundary(case)
    with case.coordinator.execution_scope(
        mode="target", valid_tokens=3 * query_len, dummy_run=False
    ):
        descriptor = dispatch_runtime(case)
        prepare_runtime_full(case, descriptor)
        output, auxiliary = case.manager.run_fullgraph(descriptor)

    assert output is case.manager.hidden_states and len(auxiliary) == 2
    assert case.transport.plan is None and case.transport.graph_reservation is None
    assert not case.transport.terminal


@pytest.mark.parametrize("foreign", ["native-query", "wire-query", "target-query"])
def test_target_query_mismatch_rejects_before_capture_protocol(
    coordinator_case, foreign
):
    case = coordinator_case(target_query_len=3)
    if foreign == "native-query":
        case.owner.decode_query_len = 1
    elif foreign == "wire-query":
        case.transport.negotiated_capabilities = replace(
            case.transport.negotiated_capabilities, target_query_len=2
        )
    else:
        case.owner.vllm_config.speculative_config.num_speculative_tokens = 3
    with pytest.raises(ValueError, match="target query"):
        HcclV41GraphCoordinator(
            case.owner,
            case.control,
            case.session,
            next_transaction_id=case.next_transaction_id,
        )
    assert case.transport.transcript == []


def test_changed_target_query_rejects_before_native_factory(coordinator_case):
    case = coordinator_case(target_query_len=3)
    assert case.coordinator.preflight_capture()
    case.owner.vllm_config.speculative_config.num_speculative_tokens = 3
    with pytest.raises(ValueError, match="ownership changed"):
        case.coordinator.capture_native()
    assert case.transport.terminal and len(sent(case, "GRAPH_QUERY")) == 1


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
    assert [record["phase"] for record in records] == [
        "eager",
        "graph_warmup",
        "graph_capture",
    ]
    assert records[0]["actual.mode"] == "serial"
    assert records[0]["num_stages"] == 1 and records[0]["is_warmup"]
    assert records[0]["actual.ffn_compute_sync"] is False
    for record in records[1:]:
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


@pytest.mark.parametrize("stages,compute", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("order", ["bulk_v1", "cross_layer_v1"])
def test_bulk_preflight_agrees_pipeline_support_before_any_native_factory(
    coordinator_case, stages, compute, order
):
    case = coordinator_case(
        stages=stages,
        exchange_order=order,
        overlap_settings=HcclOverlapSettings(graph_u2_compute_overlap=compute),
    )
    assert case.coordinator.preflight_capture() is compute
    decision = sent(case, "GRAPH_DECISION")[0].graph_decision
    assert decision.reason == ("cache_miss" if compute else "unsupported")
    assert case.cache.is_empty and not case.transport.terminal
    assert case.transport.plan is None


def test_graph_coordinator_rejects_changed_session_wire_order_before_query(
    coordinator_case,
):
    case = coordinator_case(pipeline_policy=HcclGraphPipelinePolicy())
    case.transport.negotiated_capabilities = replace(
        case.transport.negotiated_capabilities, graph_exchange_order="bulk_v1"
    )
    with pytest.raises(ValueError, match="storage"):
        case.coordinator.preflight_capture()
    assert sent(case, "GRAPH_QUERY") == []


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


@pytest.mark.parametrize("native_dispatch", [False, True])
def test_zero_graph_fallback_records_accepted_rank_configuration(
    coordinator_case, native_dispatch
):
    records = []
    case = coordinator_case(
        overlap_settings=HcclOverlapSettings(stage_diagnostics=True),
        diagnostic_sink=records.append,
    )
    set_runtime_rows(case, 0)

    with case.coordinator.execution_scope(
        mode="target", valid_tokens=0, dummy_run=False
    ):
        if native_dispatch:
            assert dispatch_runtime(case).cg_mode == CUDAGraphMode.NONE

    plan = sent(case, "FORWARD")[-1].plan
    assert len(records) == 1 and records[0]["event"] == "configuration"
    assert records[0]["transaction_id"] == plan.transaction_id
    assert records[0]["world_rank"] == case.transport.world_rank
    assert records[0]["actual.mode"] == "serial"
    assert records[0]["actual.ffn_compute_sync"] is False
    assert case.transport.plan is None and not case.transport.terminal


@pytest.mark.parametrize("same_settings", [False, True])
def test_real_kv_coordinator_retains_pre_kv_diagnostic_settings(
    coordinator_case, same_settings
):
    settings = HcclOverlapSettings(stage_diagnostics=True)
    case = coordinator_case()
    case.owner.hccl_v41_graph_eager = SimpleNamespace(overlap_settings=settings)

    if same_settings:
        HcclV41GraphCoordinator(
            case.owner,
            case.control,
            case.session,
            next_transaction_id=case.next_transaction_id,
            overlap_settings=settings,
        )
    else:
        with pytest.raises(ValueError, match="retain the pre-KV settings"):
            HcclV41GraphCoordinator(
                case.owner,
                case.control,
                case.session,
                next_transaction_id=case.next_transaction_id,
            )

    assert case.transport.transcript == []


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


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("hybrid", [False, True])
def test_cross_layer_capture_preflight_checks_attention_gate_even_before_u1_bucket(
    coordinator_case, stages, hybrid
):
    case = coordinator_case(
        stages=stages,
        exchange_order="cross_layer_v1",
        overlap_settings=HcclOverlapSettings(
            graph_u2_hybrid_dag=hybrid,
            graph_u2_ffn_recv_stream=False,
            graph_u2_ffn_cross_layer=False,
        ),
    )
    assert case.coordinator.preflight_capture() is hybrid
    assert sent(case, "GRAPH_DECISION")[0].graph_decision.reason == (
        "cache_miss" if hybrid else "unsupported"
    )
    assert case.transport.plan is None and case.transport.graph_reservation is None
    assert case.cache.is_empty and case.manager.graphs == {}
