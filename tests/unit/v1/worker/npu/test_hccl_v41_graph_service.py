# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# ruff: noqa: E402
"""Real graph wire/cache lifecycle with CPU graph-API seams, not NPU evidence."""

import gc
import weakref
from copy import copy
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)

from afd_plugin.connectors.npu.hccl_graph_cache import HcclGraphCache
from afd_plugin.connectors.npu.hccl_graph_pipeline import HcclGraphPipelinePolicy
from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.distributed.hccl_graph import graph_capture_sequence_digest
from afd_plugin.v1.worker.npu import hccl_v41_graph_service as service_module
from tests.unit.fakes.hccl_v41_ffn_graph import make_ffn_graph_case
from tests.unit.fakes.hccl_v41_graph_service import make_graph_service_case


@pytest.fixture
def case(monkeypatch):
    return make_graph_service_case(service_module, monkeypatch)


@pytest.mark.parametrize(
    "rows,compute", [((4,), False), ((2, 2), False), ((2, 2), True)]
)
@pytest.mark.parametrize("order", ["bulk_v1", "cross_layer_v1"])
def test_bulk_ffn_preflight_reports_pipeline_support_before_warmup(
    monkeypatch, rows, compute, order
):
    case = make_graph_service_case(
        service_module,
        monkeypatch,
        rows=rows,
        exchange_order=order,
        overlap_settings=HcclOverlapSettings(graph_u2_compute_overlap=compute),
    )
    supported = compute
    case.transport.queue_parent(
        "replay",
        identifier="bulk-preflight",
        valid=(0,),
        fallback=True,
        availability="missing" if supported else "unsupported",
        capture_sequence_digest=graph_capture_sequence_digest(
            (case.transport.base_bank.key,), max_full_descs_to_capture=None
        ),
    )
    assert case.service.execute_step()
    reports = [
        message.graph_report.availability
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == "GRAPH_REPORT"
    ]
    assert reports == ["missing" if supported else "unsupported"]
    assert "tensor-body" not in case.events
    assert case.cache.is_empty and not case.transport.terminal


def test_graph_service_rejects_changed_wire_order_before_reservation(case):
    case.transport.negotiated_capabilities = replace(
        case.transport.negotiated_capabilities, graph_exchange_order="bulk_v1"
    )
    with pytest.raises(ValueError, match="idle V4.1"):
        case.service.execute_step()
    assert not any(
        message.kind == "GRAPH_REPORT" for _, _, _, message in case.transport.transcript
    )


@pytest.fixture(params=[False, True])
def pipeline_case(monkeypatch, request):
    return make_graph_service_case(
        service_module,
        monkeypatch,
        pipeline_policy=HcclGraphPipelinePolicy(hybrid=request.param),
    )


def test_graph_pipeline_is_retained_through_replay_until_acknowledged_stop(
    pipeline_case,
):
    case = pipeline_case
    for phase in ("warmup", "capture", "replay"):
        case.transport.queue_parent(phase, identifier=phase, valid=(0, 0))
        assert case.service.execute_step()
    entry = case.cache.find(case.transport.base_bank.key, 0)
    owner = weakref.ref(entry.resources.pipeline)
    assert owner().hybrid == case.service.pipeline_policy.hybrid
    assert entry.resources.body.pipeline is owner()
    assert owner().bank is entry.resources.bank
    del entry
    case.transport.queue_stop()
    assert case.service.execute_step() is False
    case.service.close()
    gc.collect()
    assert case.cache.closed and owner() is None


def test_same_bank_pipeline_replacement_rejects_before_replay(pipeline_case):
    case = pipeline_case
    case.transport.queue_parent("capture", identifier="capture-1")
    assert case.service.execute_step()
    entry = case.cache.find(case.transport.base_bank.key, 0)
    body = copy(entry.resources.body)
    body.pipeline = copy(body.pipeline)
    replacement = HcclGraphCache(case.api.synchronize)
    replacement.reserve_capture(
        entry.key, 0, entry.graph, replace(entry.resources, body=body)
    )
    replacement.publish_capture(entry.key, 0)
    case.service.cache = replacement
    case.events.clear()
    case.transport.queue_rejection(
        "replay", error="FFN graph resources belong to another model/session"
    )
    with pytest.raises(RuntimeError, match="another model/session"):
        case.service.execute_step()
    assert case.events == ["ERROR"]
    assert replacement.terminal and case.transport.terminal


@pytest.mark.parametrize("rows", [(4,), (2, 2)])
@pytest.mark.parametrize(
    "compute,recv,cross",
    [
        (True, True, True),
        (True, True, False),
        (True, False, False),
        (False, True, True),
    ],
)
def test_ffn_settings_select_pipeline_without_disabling_native_graph(
    monkeypatch, rows, compute, recv, cross
):
    settings = HcclOverlapSettings(
        graph_u2_compute_overlap=compute,
        graph_u2_ffn_recv_stream=recv,
        graph_u2_ffn_cross_layer=cross,
    )
    case = make_graph_service_case(
        service_module, monkeypatch, overlap_settings=settings, rows=rows
    )
    for phase in ("warmup", "capture", "replay"):
        case.transport.queue_parent(phase, identifier=phase, valid=rows)
        assert case.service.execute_step()
    entry = case.cache.find(case.transport.base_bank.key, 0)
    assert entry.resources.overlap_settings == settings
    pipeline = entry.resources.pipeline
    assert (pipeline is not None) == (len(rows) > 1 and compute)
    if pipeline is not None:
        assert pipeline.policy.ffn_recv_stream == recv
        assert pipeline.policy.ffn_cross_layer == cross
    assert "graph-replay" in case.events
    case.transport.queue_stop()
    assert case.service.execute_step() is False
    case.service.close()
    assert case.cache.closed


@pytest.mark.parametrize(
    "rows,compute", [((4,), True), ((2, 2), False), ((2, 2), True)]
)
def test_ffn_cached_graph_rejects_changed_settings_before_new_payload(
    monkeypatch, rows, compute
):
    settings = HcclOverlapSettings(graph_u2_compute_overlap=compute)
    case = make_graph_service_case(
        service_module, monkeypatch, overlap_settings=settings, rows=rows
    )
    case.transport.queue_parent("capture", identifier="capture", valid=rows)
    assert case.service.execute_step()
    case.service = service_module.HcclV41GraphService(
        case.runner,
        case.transport,
        case.cache,
        lambda: 0,
        overlap_settings=replace(settings, graph_u2_attention_three_stream=False),
        graph_factory=case.api.create,
        graph_capture=case.api.capture,
    )
    case.events.clear()
    case.transport.queue_rejection(
        "replay", error="FFN graph resources belong to another model/session"
    )
    with pytest.raises(RuntimeError, match="another model/session"):
        case.service.execute_step()
    assert case.events == ["ERROR"]
    assert case.cache.terminal and case.transport.terminal


@pytest.mark.parametrize(
    "rows,compute", [((4,), True), ((2, 2), False), ((2, 2), True)]
)
def test_ffn_configuration_is_reported_before_capture_and_preserved_on_replay(
    monkeypatch, rows, compute
):
    records, capture_states = [], []

    def emit(record):
        records.append(record)
        capture_states.append(case.api.capturing)

    case = make_graph_service_case(
        service_module,
        monkeypatch,
        rows=rows,
        diagnostic_sink=emit,
        overlap_settings=HcclOverlapSettings(
            graph_u2_compute_overlap=compute,
            stage_diagnostics=True,
            ffn_compute_sync_diagnostics=True,
        ),
    )
    for phase in ("warmup", "capture", "replay"):
        case.transport.queue_parent(phase, identifier=phase, valid=rows)
        assert case.service.execute_step()
    assert capture_states == [False, False]
    assert [record["phase"] for record in records] == ["graph_warmup", "graph_capture"]
    for record in records:
        assert record["event"] == "configuration"
        assert record["role"] == "ffn"
        assert record["world_rank"] == case.transport.world_rank
        assert record["actual.compute_stream"] == (
            "compute" if len(rows) > 1 and compute else "parent"
        )
        assert record["actual.recv_stream"] == "parent"
        assert record["actual.ffn_compute_sync"] is False
    assert "graph-replay" in case.events


def test_graph_phase_sequence_keeps_caller_ownership(
    case,
):  # long-test: complete phase/STOP lifecycle
    assert not case.service.acknowledged_stop
    case.transport.queue_parent("warmup", identifier="warmup-1", valid=(0, 0))
    assert case.service.execute_step()
    assert case.events == [
        "GRAPH_REPORT",
        "GRAPH_ACK",
        "ACCEPT",
        "tensor-body",
        "device-sync",
        "COMPLETE",
    ]
    assert not case.cache.contains(case.transport.base_bank.key)
    case.events.clear()
    _, decision, _ = case.transport.queue_parent(
        "capture", identifier="capture-1", valid=(0, 0)
    )
    assert case.service.execute_step()
    assert case.events == [
        "GRAPH_REPORT",
        "GRAPH_ACK",
        "ACCEPT",
        "graph-created",
        "capture-enter",
        "tensor-body",
        "capture-exit",
        "device-sync",
        "device-sync",
        "COMPLETE",
    ]
    entry = case.cache.find(decision.key, 0)
    assert entry is not None
    assert case.bodies.resources[0]() is None
    assert case.bodies.resources[1]() is entry.resources.body
    case.events.clear()
    case.transport.queue_parent("replay", identifier="replay-1", valid=(1, 0))
    assert case.service.execute_step()
    assert case.events == [
        "GRAPH_REPORT",
        "GRAPH_ACK",
        "ACCEPT",
        "graph-replay",
        "device-sync",
        "COMPLETE",
    ]
    assert [count.tolist() for count in entry.resources.bank.valid_counts] == [[1], [0]]
    case.events.clear()
    case.transport.queue_stop()
    assert case.service.execute_step() is False
    assert case.service.acknowledged_stop
    assert case.events == ["STOP_ACK"]
    assert (
        not case.cache.closed and not case.cache.terminal and not case.transport.closed
    )
    case.service.close()
    assert case.cache.closed and case.events[-2:] == ["device-sync", "graph-reset"]


def test_service_close_before_acknowledged_stop_keeps_cache_and_groups_active(case):
    with pytest.raises(RuntimeError, match="acknowledged idle STOP"):
        case.service.close()
    assert not case.cache.closed and not case.cache.terminal
    assert not case.transport.closed and not case.transport.terminal
    assert not case.service.acknowledged_stop
    assert case.events == []


def test_external_cache_close_does_not_replace_the_service_stop_handshake(case):
    case.cache.close()
    with pytest.raises(RuntimeError, match="acknowledged idle STOP"):
        case.service.close()
    assert case.cache.closed and not case.transport.closed
    assert not case.service.acknowledged_stop


def test_service_close_after_stop_keeps_failed_graph_dependencies_for_supervision(case):
    case.transport.queue_parent("capture", identifier="capture-before-close")
    assert case.service.execute_step()
    case.transport.queue_stop()
    assert case.service.execute_step() is False
    case.api.fail_sync = True
    with pytest.raises(RuntimeError, match="controlled device synchronization failure"):
        case.service.close()
    gc.collect()
    assert case.cache.terminal and not case.cache.closed
    assert not case.transport.closed
    assert case.api.graphs[-1]() is not None
    assert case.banks.allocations[-1]() is not None


def test_native_context_capture_flags_and_outer_context_are_restored(case):
    outer = ForwardContext({}, {}, {}, additional_kwargs={"capturing": False})
    with override_forward_context(outer):
        for phase in ("warmup", "capture"):
            case.transport.queue_parent(phase, identifier=phase, valid=(0, 0))
            assert case.service.execute_step()
            assert get_forward_context() is outer
            assert outer.additional_kwargs == {"capturing": False}
    assert case.api.body_flags == [False, False, True, True]


def test_service_owns_inference_mode_across_bank_allocation_and_next_parent_refresh(
    case,
):
    assert not torch.is_inference_mode_enabled()
    case.transport.queue_parent("capture", identifier="capture-1")
    assert case.service.execute_step()
    entry = case.cache.find(case.transport.base_bank.key, 0)
    assert entry.resources.bank.buffers(0).input_ids.is_inference()
    assert not torch.is_inference_mode_enabled()
    case.transport.queue_parent("replay", identifier="replay-1", valid=(1, 0))
    assert case.service.execute_step()
    assert [count.tolist() for count in entry.resources.bank.valid_counts] == [[1], [0]]
    assert not torch.is_inference_mode_enabled()


@pytest.mark.parametrize("foreign", ["bank", "contexts"])
def test_public_cache_bundle_replacement_rejects_body_storage_mismatch(case, foreign):
    case.transport.queue_parent("capture", identifier="capture-1")
    assert case.service.execute_step()
    entry = case.cache.find(case.transport.base_bank.key, 0)
    body = copy(entry.resources.body)
    if foreign == "bank":
        body.bank = case.banks.allocate(
            case.transport, entry.key, device=case.runner.device
        )
    else:
        body.contexts = tuple(list(entry.resources.contexts))
    replacement_cache = HcclGraphCache(case.api.synchronize)
    replacement_cache.reserve_capture(
        entry.key, 0, entry.graph, replace(entry.resources, body=body)
    )
    replacement_cache.publish_capture(entry.key, 0)
    case.service.cache = replacement_cache
    case.events.clear()
    case.transport.queue_rejection(
        "replay", error="FFN graph resources belong to another model/session"
    )
    with pytest.raises(RuntimeError, match="another model/session"):
        case.service.execute_step()
    assert case.events == ["ERROR"]
    assert case.transport.terminal and replacement_cache.terminal


def test_generation_miss_uses_negotiated_fresh_eager_parent_without_replaying_old_graph(
    case,
):
    case.transport.queue_parent("capture", identifier="capture-1")
    assert case.service.execute_step()
    case.generation.value = 1
    case.events.clear()
    # Missing weights still permit the actual eager whole-empty parent. Remove
    # them only after graph readiness has resolved to the eager miss decision.
    case.transport.before_forward = lambda: setattr(case.runner, "model", None)
    _, decision, _ = case.transport.queue_parent(
        "replay", identifier="generation-miss", fallback=True
    )
    assert decision.phase == "eager"

    assert case.service.execute_step()

    assert case.events == [
        "GRAPH_REPORT",
        "GRAPH_ACK",
        "ACCEPT",
        "device-sync",
        "COMPLETE",
    ]
    assert case.cache.find(case.transport.base_bank.key, 1) is None
    assert case.cache.find(case.transport.base_bank.key, 0) is not None


@pytest.mark.parametrize("phase", ["replay", "eager"])
def test_capture_preflight_reports_honest_availability_and_retires_empty_parent(
    case, phase
):
    digest = graph_capture_sequence_digest(
        (case.transport.base_bank.key,) if phase == "replay" else (),
        max_full_descs_to_capture=None,
    )
    _, decision, plan = case.transport.queue_parent(
        phase,
        identifier="capture-preflight",
        valid=(0,),
        fallback=True,
        capture_sequence_digest=digest,
    )
    assert case.service.execute_step()
    assert not any(any(stage.attention_padded_tokens) for stage in plan.stages)
    assert decision.phase == "eager"
    assert case.cache.is_empty
    reports = [
        message.graph_report.availability
        for direction, _sender, _receiver, message in case.transport.transcript
        if direction == "send" and message.kind == "GRAPH_REPORT"
    ]
    assert reports == ["missing" if phase == "replay" else "unsupported"]
    assert case.events == [
        "GRAPH_REPORT",
        "GRAPH_ACK",
        "ACCEPT",
        "device-sync",
        "COMPLETE",
    ]


@pytest.mark.parametrize("state", ["pending", "ready", "stale"])
@pytest.mark.parametrize("phase", ["replay", "eager"])
def test_capture_preflight_rejects_any_retained_cache_dependencies(case, state, phase):
    case.transport.queue_parent("capture", identifier="existing-capture")
    assert case.service.execute_step()
    entry = case.cache.find(case.transport.base_bank.key, 0)
    assert entry is not None
    if state == "pending":
        pending = HcclGraphCache(case.api.synchronize)
        pending.reserve_capture(entry.key, 0, entry.graph, entry.resources)
        case.service.cache = case.cache = pending
    elif state == "stale":
        case.generation.value = 1
    case.events.clear()
    case.transport.queue_rejection(
        phase,
        error="capture preflight requires a fresh FFN graph cache",
        capture_sequence_digest=graph_capture_sequence_digest(
            (entry.key,), max_full_descs_to_capture=None
        ),
    )
    with pytest.raises(RuntimeError, match="fresh FFN graph cache"):
        case.service.execute_step()
    assert case.events == ["ERROR"]
    assert case.transport.terminal and case.cache.terminal


@pytest.mark.parametrize(
    "invalid,reason",
    [
        ("generation", "generation must be a nonnegative integer"),
        ("weights", "NPU FFN model has not been loaded"),
    ],
)
def test_keyless_capture_preflight_still_validates_current_model_and_generation(
    case, invalid, reason
):
    if invalid == "generation":
        case.generation.value = True
    else:
        case.runner.model = None
    case.transport.queue_rejection(
        "eager",
        error=reason,
        capture_sequence_digest=graph_capture_sequence_digest(
            (), max_full_descs_to_capture=None
        ),
    )
    with pytest.raises(RuntimeError, match=reason):
        case.service.execute_step()
    assert case.events == ["ERROR"]
    assert case.transport.terminal and case.cache.terminal


@pytest.mark.parametrize("phase", ["warmup", "capture"])
@pytest.mark.parametrize("generation", [0, 1])
def test_existing_key_rejects_new_capture_phases_even_after_generation_change(
    case, phase, generation
):
    case.transport.queue_parent("capture", identifier="capture-1")
    assert case.service.execute_step()
    case.generation.value = generation
    case.events.clear()
    case.transport.queue_rejection(phase, error="FFN graph key already reserved")

    with pytest.raises(RuntimeError, match="already reserved"):
        case.service.execute_step()

    assert case.events == ["ERROR"]
    assert case.transport.terminal and case.cache.terminal


@pytest.mark.parametrize(
    "change,reason",
    [
        ("generation", "storage changed after reservation"),
        ("model", "model changed inside its transport session"),
    ],
)
def test_changed_owner_after_reservation_rejects_before_any_graph_or_body(
    case, change, reason
):
    case.transport.queue_parent("capture", identifier="capture-1")
    if change == "generation":
        case.transport.before_forward = lambda: setattr(case.generation, "value", 1)
    else:
        replacement = make_ffn_graph_case().model
        case.transport.before_forward = lambda: setattr(
            case.runner, "model", replacement
        )
    with pytest.raises(ValueError, match=reason):
        case.service.execute_step()
    assert case.events == ["GRAPH_REPORT", "GRAPH_ACK", "ACCEPT"]
    assert case.transport.terminal and case.cache.terminal


@pytest.mark.parametrize(
    "invalid,reason",
    [
        ("generation", "generation must be a nonnegative integer"),
        ("weights", "NPU FFN model has not been loaded"),
        ("capacity", "bank's session capacity"),
    ],
)
def test_invalid_graph_offer_rejects_peers_before_parent_acceptance(
    case, invalid, reason
):
    if invalid == "generation":
        case.generation.value = True
    elif invalid == "weights":
        case.runner.model = None
    case.transport.queue_rejection(
        "capture", error=reason, rows=(4, 4) if invalid == "capacity" else None
    )
    with pytest.raises(RuntimeError, match=reason):
        case.service.execute_step()
    assert case.events == ["ERROR"]
    assert case.transport.plan is None
    assert case.transport.terminal and case.cache.terminal


@pytest.mark.parametrize("failure", ["warmup", "capture", "body", "publish", "replay"])
def test_failed_graph_phase_is_terminal_keeps_resources_and_sends_no_complete(
    case, failure
):
    if failure == "replay":
        case.transport.queue_parent("capture", identifier="capture-1")
        assert case.service.execute_step()
        case.events.clear()
    phase = (
        "warmup"
        if failure == "warmup"
        else "replay"
        if failure == "replay"
        else "capture"
    )
    case.transport.queue_parent(phase, identifier="failing-phase")
    case.api.fail_body = failure in ("warmup", "body")
    case.api.fail_capture = failure == "capture"
    case.api.fail_sync = failure == "publish"
    case.api.fail_replay = failure == "replay"

    with pytest.raises(RuntimeError, match="controlled"):
        case.service.execute_step()

    gc.collect()
    assert "COMPLETE" not in case.events
    assert case.transport.terminal and case.cache.terminal
    assert not case.transport.closed and not case.cache.closed
    assert case.banks.allocations[-1]() is not None
    assert case.bodies.resources[-1]() is not None
    if case.api.graphs:
        assert case.api.graphs[-1]() is not None
    with pytest.raises(RuntimeError, match="acknowledged idle STOP"):
        case.service.close()
    assert not case.cache.closed and not case.transport.closed


@pytest.mark.parametrize(
    "invalid,reason",
    [
        ("model", "FFN graph model changed inside its transport session"),
        ("groups", "FFN graph resources belong to another model/session"),
    ],
)
def test_foreign_cached_model_or_session_is_rejected_during_query(
    case, invalid, reason
):
    case.transport.queue_parent("capture", identifier="capture-1")
    assert case.service.execute_step()
    if invalid == "model":
        case.runner.model = make_ffn_graph_case().model
    else:
        case.transport.data_groups = list(reversed(case.transport.data_groups))
    case.events.clear()
    case.transport.queue_rejection("replay", error=reason)
    with pytest.raises(RuntimeError, match=reason):
        case.service.execute_step()
    assert case.events == ["ERROR"]
    assert case.transport.terminal and case.cache.terminal


@pytest.mark.parametrize(
    "invalid",
    ["graph_version", "device", "rank", "hidden", "engram", "speculation", "sp_pass"],
)
def test_invalid_service_binding_is_rejected_before_receiving_peer_control(
    case, invalid
):
    if invalid == "graph_version":
        case.transport.negotiated_capabilities = replace(
            case.transport.negotiated_capabilities, graph_execution_version=0
        )
    elif invalid == "device":
        case.runner.device = torch.device("npu", 1)
    elif invalid == "rank":
        case.runner.vllm_config.parallel_config.rank = 1
    elif invalid == "hidden":
        case.runner.vllm_config.model_config.hf_text_config.hidden_size = 8
    elif invalid == "engram":
        case.runner.vllm_config.engram_config = True
    elif invalid == "sp_pass":
        case.runner.vllm_config.compilation_config.pass_config.enable_sp = True
    else:
        case.runner.vllm_config.speculative_config = True
    with pytest.raises(ValueError, match="idle V4.1 paired-TP session"):
        case.service.execute_step()
    assert case.events == []
    assert case.transport.terminal and case.cache.terminal


@pytest.mark.parametrize("rows", [(4,), (2, 2)])
@pytest.mark.parametrize("recv,cross", [(True, True), (True, False), (False, False)])
def test_cross_layer_ffn_capture_preflight_checks_role_gates_without_partial_capture(
    monkeypatch, rows, recv, cross
):
    case = make_graph_service_case(
        service_module,
        monkeypatch,
        rows=rows,
        exchange_order="cross_layer_v1",
        overlap_settings=HcclOverlapSettings(
            graph_u2_hybrid_dag=False,
            graph_u2_ffn_recv_stream=recv,
            graph_u2_ffn_cross_layer=cross,
        ),
    )
    supported = recv and cross
    case.transport.queue_parent(
        "replay",
        identifier="cross-preflight",
        valid=(0,),
        fallback=True,
        availability="missing" if supported else "unsupported",
        capture_sequence_digest=graph_capture_sequence_digest(
            (case.transport.base_bank.key,), max_full_descs_to_capture=None
        ),
    )
    assert case.service.execute_step()
    assert [
        message.graph_report.availability
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == "GRAPH_REPORT"
    ] == ["missing" if supported else "unsupported"]
    assert case.cache.is_empty
    assert not any(
        event in case.events
        for event in ("tensor-body", "capture-enter", "graph-created")
    )
    assert case.transport.plan is None and case.transport.graph_reservation is None
