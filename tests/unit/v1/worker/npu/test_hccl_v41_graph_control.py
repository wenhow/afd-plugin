# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Actual native dispatch plus graph reservation messages, without device work."""

from dataclasses import replace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from afd_plugin.distributed.hccl_graph import HcclGraphStage
from afd_plugin.distributed.hccl_plan import HcclLocalForward
from afd_plugin.v1.worker.npu.hccl_v41_graph import (
    HcclV41GraphGeometry,
    HcclV41StageGeometry,
)
from tests.unit.fakes.batch_req_state import make_batch_req_state
from tests.unit.fakes.graph_cache import ReplayGraph
from tests.unit.fakes.hccl_graph_control import make_graph_candidate, make_graph_key
from tests.unit.fakes.hccl_v41_graph_control import make_attention_graph_control_case


@pytest.fixture
def reserved_case():
    case = make_attention_graph_control_case()
    reservation = case.control.reserve_batch(
        case.batch,
        parallel_config=case.parallel,
        allow_ubatching=True,
        transaction_id="batch-1",
    )
    return case, reservation


@pytest.mark.parametrize("role_rank", [0, 1])
def test_ready_reservation_preserves_real_native_descriptor_and_normalizes_wire_key(
    role_rank,
):
    case = make_attention_graph_control_case(
        role_rank=role_rank,
        candidate=make_graph_candidate(),
    )
    result = case.control.reserve_batch(
        case.batch,
        parallel_config=case.parallel,
        allow_ubatching=True,
        transaction_id="batch-1" if role_rank == 0 else None,
    )
    assert result.decision.phase == "replay"
    assert result.geometry.descriptor is case.descriptor
    assert result.geometry.descriptor.max_query_len is None
    assert result.decision.key.max_query_len == 1


@pytest.mark.parametrize(
    "boundary", ["missing", "pending", "generation", "native-graph"]
)
def test_local_cache_miss_chooses_fresh_eager_descriptor_before_preparation(boundary):
    case = make_attention_graph_control_case(
        cached=boundary not in ("missing", "pending"),
        pending=boundary == "pending",
        generation=int(boundary == "generation"),
    )
    if boundary == "native-graph":
        case.manager.graphs[case.descriptor] = ReplayGraph()
    result = case.control.reserve_batch(
        case.batch,
        parallel_config=case.parallel,
        allow_ubatching=True,
        transaction_id="batch-1",
    )
    assert result.decision.phase == "eager"
    assert result.geometry.descriptor.cg_mode == CUDAGraphMode.NONE
    assert result.geometry.descriptor.num_tokens == case.batch.num_tokens
    assert result.geometry.descriptor.num_reqs == len(case.batch.req_ids)


def test_remote_cache_miss_uses_native_eager_geometry_on_attention():
    case = make_attention_graph_control_case(
        peer_states={0: "missing", 1: "ready", 3: "ready"}
    )
    result = case.control.reserve_batch(
        case.batch,
        parallel_config=case.parallel,
        allow_ubatching=True,
        transaction_id="batch-1",
    )
    assert result.decision.phase == "eager" and result.decision.reason == "cache_miss"
    assert result.geometry.descriptor.cg_mode == CUDAGraphMode.NONE


def test_request_layout_changes_query_key_with_equal_capture_token_bucket():
    case = make_attention_graph_control_case(scheduled=(1, 1))
    result = case.control.reserve_batch(
        case.batch,
        parallel_config=case.parallel,
        allow_ubatching=True,
        transaction_id="batch-1",
    )
    query = next(
        message
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == "GRAPH_QUERY"
    )
    assert query.graph_candidate.key.stage_layouts[-1] == HcclGraphStage(1, 4, 2, 4)
    assert result.decision.phase == "eager"
    assert result.geometry.descriptor.num_tokens == 2
    assert result.decision.padded_tokens == (1, 1)


@pytest.mark.parametrize("purpose", ["forced", "profile", "prefill", "lora"])
def test_ineligible_native_batch_reserves_fresh_eager_without_a_graph_key(purpose):
    case = make_attention_graph_control_case()
    if purpose == "prefill":
        case.batch = make_batch_req_state((2, 2), has_prefill=True)
    result = case.control.reserve_batch(
        case.batch,
        parallel_config=case.parallel,
        allow_ubatching=True,
        need_eager=purpose == "forced",
        num_active_loras=int(purpose == "lora"),
        mode="profile" if purpose == "profile" else "target",
        transaction_id="batch-1",
    )
    assert result.decision.phase == "eager" and result.decision.key is None
    assert result.geometry.descriptor.cg_mode == CUDAGraphMode.NONE
    assert result.geometry.descriptor.num_active_loras == int(purpose == "lora")


@pytest.mark.parametrize("native_hit", ["none", "different-bucket"])
def test_attention_peer_local_dispatch_miss_is_a_report_not_a_geometry_error(
    native_hit,
):
    descriptors = (
        ()
        if native_hit == "none"
        else (
            BatchExecutionDescriptor(
                CUDAGraphMode.FULL, 8, 8, uniform_token_count=1, num_ubatches=2
            ),
        )
    )
    case = make_attention_graph_control_case(
        role_rank=1,
        candidate=make_graph_candidate(),
        availability="missing",
        local_descriptors=descriptors,
    )
    result = case.control.reserve_batch(
        case.batch, parallel_config=case.parallel, allow_ubatching=True
    )
    report = next(
        message
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == "GRAPH_REPORT"
    )
    assert report.graph_report.availability == "missing"
    assert result.decision.phase == "eager"


@pytest.mark.parametrize("boundary", ["requests", "eager-rows", "metadata", "mode"])
def test_attention_peer_rejects_query_inconsistent_with_native_batch(
    boundary,
):
    candidate = make_graph_candidate()
    if boundary == "requests":
        candidate = replace(
            candidate,
            key=replace(
                candidate.key,
                stage_layouts=(
                    candidate.key.stage_layouts[0],
                    HcclGraphStage(1, 4, 2, 4),
                ),
            ),
        )
    elif boundary == "eager-rows":
        candidate = replace(candidate, eager_padded_tokens=(4,))
    elif boundary == "metadata":
        candidate = replace(candidate, key=make_graph_key(metadata_shapes=((1, 4),)))
    else:
        candidate = replace(candidate, mode="dummy")
    case = make_attention_graph_control_case(
        role_rank=1, candidate=candidate, rejection=True
    )
    with pytest.raises(RuntimeError, match="local native geometry rejected"):
        case.control.reserve_batch(
            case.batch, parallel_config=case.parallel, allow_ubatching=True
        )
    assert [
        message.kind
        for direction, _, _, message in case.transport.transcript
        if direction == "send"
    ] == ["ERROR"]
    assert case.transport.terminal and case.cache.terminal


@pytest.mark.parametrize("warmup", [True, False])
def test_capture_factory_reserves_fresh_native_geometry_before_any_capture_body(warmup):
    case = make_attention_graph_control_case(
        cached=False,
        peer_states={0: "missing", 1: "missing", 3: "missing"},
    )
    result = case.control.reserve_capture(
        case.descriptor, warmup=warmup, transaction_id="factory-1"
    )
    assert result.decision.phase == ("warmup" if warmup else "capture")
    assert result.decision.key == case.key
    assert result.geometry.descriptor is case.descriptor
    assert not case.cache.contains(case.key)


@pytest.mark.parametrize("retained", ["ready", "pending", "stale", "native-only"])
def test_capture_factory_rejects_retained_key_in_query_preflight(retained):
    case = make_attention_graph_control_case(
        cached=retained not in ("pending", "native-only"),
        pending=retained == "pending",
        generation=int(retained == "stale"),
    )
    if retained == "native-only":
        case.manager.graphs[case.descriptor] = case.graph
    with pytest.raises(RuntimeError, match="already"):
        case.control.reserve_capture(
            case.descriptor, warmup=False, transaction_id="factory-1"
        )
    assert [
        message.kind
        for direction, _, _, message in case.transport.transcript
        if direction == "send"
    ] == ["ERROR"] * 3
    assert case.transport.terminal and case.cache.terminal


def test_capture_peer_rejects_retained_key_after_query_without_report_or_capture():
    case = make_attention_graph_control_case(
        role_rank=1,
        candidate=make_graph_candidate(phase="capture"),
        rejection=True,
    )
    with pytest.raises(RuntimeError, match="local native geometry rejected"):
        case.control.reserve_capture(case.descriptor, warmup=False, mode="target")
    assert [
        message.kind
        for direction, _, _, message in case.transport.transcript
        if direction == "send"
    ] == ["ERROR"]


@pytest.mark.parametrize("num_ubatches", [1, 2])
def test_explicit_empty_eager_parent_needs_no_fabricated_native_batch(num_ubatches):
    case = make_attention_graph_control_case(cached=False)
    geometry = HcclV41GraphGeometry(
        BatchExecutionDescriptor(CUDAGraphMode.NONE, 0, 0, num_ubatches=num_ubatches),
        (HcclV41StageGeometry(0, 0, 0, 0),) * num_ubatches,
    )
    result = case.control.reserve_eager(
        geometry, mode="dummy", transaction_id="empty-1"
    )
    assert result.decision.phase == "eager"
    assert result.decision.padded_tokens == (0,) * num_ubatches
    assert result.geometry is geometry
    case.control.verify_prepared(result, geometry)


@pytest.mark.parametrize(
    "boundary", ["requests", "tokens", "descriptor", "generation", "metadata"]
)
def test_postprepare_change_is_terminal_and_cannot_late_fallback(
    reserved_case, boundary
):
    case, result = reserved_case
    actual = result.geometry
    if boundary in ("requests", "tokens"):
        stage = actual.stages[0]
        change = {"request_stop": 1} if boundary == "requests" else {"token_stop": 1}
        actual = replace(actual, stages=(replace(stage, **change), actual.stages[1]))
    elif boundary == "descriptor":
        actual = replace(actual, descriptor=replace(actual.descriptor, max_query_len=1))
    elif boundary == "generation":
        case.generation.value += 1
    else:
        case.metadata.shapes = ((4, 1),)
    with pytest.raises(ValueError, match="native"):
        case.control.verify_prepared(result, actual)
    assert case.transport.terminal and case.cache.terminal


def test_postprepare_matching_native_geometry_keeps_full_reservation_live(
    reserved_case,
):
    case, result = reserved_case
    case.control.verify_prepared(result, result.geometry)
    assert result.decision.phase == "replay"
    assert not case.transport.terminal and not case.cache.terminal


@pytest.mark.parametrize("source", ["copy", "other-controller"])
def test_postprepare_rejects_equal_valued_handle_not_returned_by_this_controller(
    reserved_case, source
):
    case, owned = reserved_case
    if source == "copy":
        foreign = replace(owned)
    else:
        other = make_attention_graph_control_case()
        foreign = other.control.reserve_batch(
            other.batch,
            parallel_config=other.parallel,
            allow_ubatching=True,
            transaction_id="batch-1",
        )
    assert foreign == owned and foreign is not owned
    with pytest.raises(ValueError, match="current owned reservation"):
        case.control.verify_prepared(foreign, foreign.geometry)
    assert case.transport.terminal and case.cache.terminal


def test_postprepare_rejects_handle_after_its_accepted_parent_completed(monkeypatch):
    monkeypatch.setattr("torch.npu.synchronize", lambda device: None)
    case = make_attention_graph_control_case(cached=False)
    geometry = HcclV41GraphGeometry(
        BatchExecutionDescriptor(CUDAGraphMode.NONE, 0, 0),
        (HcclV41StageGeometry(0, 0, 0, 0),),
    )
    owned = case.control.reserve_eager(geometry, mode="dummy", transaction_id="empty-1")
    assert case.transport.graph_reservation is owned.decision
    case.transport.submit_forward(
        HcclLocalForward("dummy", (0,), (0,)), transaction_id="empty-1"
    )
    case.transport.finish_forward()
    assert case.transport.graph_reservation is None
    with pytest.raises(ValueError, match="current owned reservation"):
        case.control.verify_prepared(owned, geometry)
    assert case.transport.terminal and case.cache.terminal


@pytest.mark.parametrize("entry", ["batch", "capture", "eager"])
def test_attention_peer_handles_idle_stop_before_resolving_native_inputs(entry):
    case = make_attention_graph_control_case(role_rank=1)
    if entry == "batch":
        result = case.control.reserve_batch(
            case.batch, parallel_config=case.parallel, allow_ubatching=True
        )
    elif entry == "capture":
        result = case.control.reserve_capture(case.descriptor, warmup=False)
    else:
        geometry = HcclV41GraphGeometry(
            BatchExecutionDescriptor(CUDAGraphMode.NONE, 0, 0),
            (HcclV41StageGeometry(0, 0, 0, 0),),
        )
        result = case.control.reserve_eager(geometry, mode="dummy")
    assert result is None
    assert [
        message.kind
        for direction, _, _, message in case.transport.transcript
        if direction == "send"
    ] == ["STOP_ACK"]
    assert case.cache.find(case.key, case.generation.value) is not None
