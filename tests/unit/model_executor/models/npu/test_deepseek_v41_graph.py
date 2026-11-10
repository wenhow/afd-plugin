# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native continuation, parent-producer and payload seams on CPU tensors."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.forward_context import (  # noqa: E402
    get_forward_context,
    override_forward_context,
)

import afd_plugin.connectors.npu.hccl_graph_pipeline as pipeline_module  # noqa: E402
from afd_plugin.model_executor.models.npu.deepseek_v41_graph import (  # noqa: E402
    DeepseekV41GraphParent,
    DeepseekV41GraphStages,
)
from tests.unit.fakes.deepseek_v41_graph import (  # noqa: E402
    attach_graph_pipeline,
    make_graph_model_case,
)
from tests.unit.fakes.npu_stream import make_npu_stream_case  # noqa: E402


def make_body(case, *, pipeline=None):
    return DeepseekV41GraphStages(
        case.model, case.contexts, case.slices, case.bank, pipeline=pipeline
    )


@pytest.mark.parametrize("split", [(2, 3), (0, 3), (3, 0), (5,)])
@pytest.mark.parametrize("engram", [False, True])
def test_cross_layer_parent_preserves_three_layer_hc_aux_mtp_and_one_engram_producer(
    monkeypatch, split, engram
):
    case = make_graph_model_case(
        split, engram=engram, exchange_order="cross_layer_v1", layer_count=3
    )
    expected = case.model(**case.parent_inputs)
    expected_mtp = case.mtp_rows.clone()
    attach_graph_pipeline(monkeypatch, pipeline_module, case)
    actual = DeepseekV41GraphParent(make_body(case, pipeline=case.pipeline))(
        case.graph_inputs
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(case.mtp_rows, expected_mtp, rtol=0, atol=0)
    assert len(case.producer_parents) == int(engram)
    assert case.joined == ([True] if engram else [])
    positive = [stage for stage, count in enumerate(split) if count]
    assert [event[1:3] for event in case.runtime.trace if event[0] == "attention"] == [
        (layer, stage) for layer in range(3) for stage in positive
    ]
    if len(split) == 2:
        schedule = case.pipeline.schedule
        expected_wire = [
            ("send" if action.kind == "input" else "receive", action.stage_idx)
            for action in schedule.actions
            for _ in range(3 if action.kind == "input" else 1)
        ]
        assert [
            event[1:3] for event in case.runtime.trace if event[0] == "wire"
        ] == expected_wire
    for endpoint in case.endpoints:
        for offset in range(0, len(endpoint.messages), 3):
            hidden, original, ids = endpoint.messages[offset : offset + 3]
            assert original.dtype == torch.float32 and ids.dtype == torch.int32
            assert torch.any(hidden.float() != original)


def test_cross_layer_failure_restores_parent_after_advancing_next_compute(
    monkeypatch,
):
    streams = make_npu_stream_case(monkeypatch, pipeline_module, cached="foreign")
    case = attach_graph_pipeline(
        monkeypatch,
        pipeline_module,
        make_graph_model_case(exchange_order="cross_layer_v1", layer_count=3),
        runtime=streams.runtime,
    )
    original = case.model.shared_attention_state
    parent = SimpleNamespace(ubatch_idx=-1)

    def fail(*args):
        raise RuntimeError("cross layer peer failure")

    monkeypatch.setattr(case.endpoints[1], "receive", fail)
    body = DeepseekV41GraphParent(make_body(case, pipeline=case.pipeline))
    with override_forward_context(parent):
        with pytest.raises(RuntimeError, match="cross layer peer failure"):
            body(case.graph_inputs)
        assert get_forward_context() is parent
    assert ("attention", 1, 0, True, True) in case.runtime.trace
    assert case.model.shared_attention_state is original
    assert all(layer.self_attn.shared_state is original for layer in case.model.layers)
    assert case.joined == [True]
    assert streams.query() is streams.previous
    with pytest.raises(RuntimeError, match="failed"), case.pipeline.run():
        pass


@pytest.mark.parametrize("split", [(5,), (2, 3), (0, 3), (3, 0)])
@pytest.mark.parametrize("engram", [False, True])
def test_bulk_graph_parent_preserves_native_hc_engram_aux_and_all_send_frontier(
    monkeypatch, split, engram
):
    case = make_graph_model_case(split, engram=engram, exchange_order="bulk_v1")
    expected = case.model(**case.parent_inputs)
    expected_mtp = case.mtp_rows.clone()
    attach_graph_pipeline(monkeypatch, pipeline_module, case)
    actual = DeepseekV41GraphParent(make_body(case, pipeline=case.pipeline))(
        case.graph_inputs
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(case.mtp_rows, expected_mtp, rtol=0, atol=0)
    positive = [stage for stage, rows in enumerate(split) if rows]
    one_layer = [("send", stage) for stage in positive for _ in range(3)] + [
        ("receive", stage) for stage in positive
    ]
    assert [
        event[1:3] for event in case.runtime.trace if event[0] == "wire"
    ] == one_layer * 2
    assert case.joined == ([True] if engram else [])
    assert {event[-1] for event in case.runtime.trace if event[0] == "wire"} == {
        "parent"
    }


def test_bulk_graph_u2_requires_a_pipeline_before_parent_production():
    case = make_graph_model_case(exchange_order="bulk_v1")
    with pytest.raises(ValueError, match="actual compute pipeline"):
        make_body(case)
    assert case.producer_parents == []
    assert all(endpoint.messages == [] for endpoint in case.endpoints)


@pytest.mark.parametrize("split", [(5,), (2, 3), (0, 3), (3, 0)])
@pytest.mark.parametrize("engram", [False, True])
def test_static_tensor_stages_preserve_native_hc_aux_and_parent_mtp(split, engram):
    case = make_graph_model_case(split, engram=engram)
    expected = case.model(**case.parent_inputs)
    expected_mtp = case.mtp_rows.clone()

    actual = make_body(case).run(case.stages)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(case.mtp_rows, expected_mtp, rtol=0, atol=0)
    assert [len(peer.messages) for peer in case.endpoints] == [
        6 if rows else 0 for rows in split
    ]


def test_reusing_static_bank_observes_changed_ids_and_original_fp32_values():
    case = make_graph_model_case(engram=False)
    body = make_body(case)
    first = body.run(case.stages)[0].clone()
    case.parent_inputs["input_ids"].add_(3)
    expected = case.model(**case.parent_inputs)

    actual = body.run(case.stages)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual[0], first)
    for peer in case.endpoints:
        bf16, original_fp32, ids = peer.messages[-3:]
        assert original_fp32.dtype == torch.float32
        assert torch.any(original_fp32 != bf16.float())
        assert ids.dtype == torch.int32


def test_padding_masks_wire_ids_without_changing_parent_input_storage():
    case = make_graph_model_case(valid=(1, 1), engram=False)
    case.parent_inputs["input_ids"].fill_(29)
    original = case.parent_inputs["input_ids"].clone()

    make_body(case).run(case.stages)

    torch.testing.assert_close(case.parent_inputs["input_ids"], original)
    for peer, rows in zip(case.endpoints, (2, 3), strict=True):
        assert peer.messages[2].tolist() == [29] + [0] * (rows - 1)
        assert peer.messages[5].tolist() == [29] + [0] * (rows - 1)


def test_graph_parent_produces_unsliced_engram_once_and_joins_on_success():
    case = make_graph_model_case()
    parent = DeepseekV41GraphParent(make_body(case))
    expected = case.model(**case.parent_inputs)

    actual = parent(case.graph_inputs)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(case.producer_parents) == 1
    ids, positions, query = case.producer_parents[0]
    torch.testing.assert_close(ids, case.parent_inputs["input_ids"])
    torch.testing.assert_close(positions, case.parent_inputs["positions"])
    assert query.tolist() == list(range(6))
    assert case.joined == [True]


def test_graph_parent_joins_producer_and_restores_context_after_failure(monkeypatch):
    case = make_graph_model_case()
    body = DeepseekV41GraphParent(make_body(case))
    original = case.model.shared_attention_state
    parent_context = SimpleNamespace(ubatch_idx=-1)

    def fail(*args):
        raise RuntimeError("injected graph payload failure")

    monkeypatch.setattr(case.endpoints[1], "receive", fail)
    with override_forward_context(parent_context):
        with pytest.raises(RuntimeError, match="graph payload failure"):
            body(case.graph_inputs)
        assert get_forward_context() is parent_context

    assert case.joined == [True]
    assert case.model.shared_attention_state is original
    assert all(layer.self_attn.shared_state is original for layer in case.model.layers)


@pytest.mark.parametrize("invalid", ["external_events", "producer", "parent_rows"])
def test_unsupported_parent_is_rejected_before_production(invalid):
    case = make_graph_model_case()
    if invalid == "external_events":
        case.graph_inputs["engram_graph_events"] = True
    elif invalid == "producer":
        case.model.can_capture_engram_producer = lambda: False
    else:
        case.graph_inputs["input_ids"] = case.parent_inputs["input_ids"][:-1]

    with pytest.raises(ValueError, match="V4.1|Engram"):
        DeepseekV41GraphParent(make_body(case))(case.graph_inputs)

    assert case.producer_parents == []
    assert all(peer.messages == [] for peer in case.endpoints)


@pytest.mark.parametrize("invalid", ["context_mode", "child_index", "token_rows"])
def test_incompatible_static_layout_is_rejected_before_payload(invalid):
    case = make_graph_model_case()
    if invalid == "context_mode":
        case.contexts[0].cudagraph_runtime_mode = CUDAGraphMode.NONE
    elif invalid == "child_index":
        case.contexts[0].ubatch_idx = 1
    else:
        case.slices = (slice(0, 1), slice(1, 5))

    with pytest.raises(ValueError, match="V4.1"):
        make_body(case)

    assert all(peer.messages == [] for peer in case.endpoints)


def test_incompatible_second_stage_is_rejected_before_any_payload():
    case = make_graph_model_case()
    stages = (case.stages[0], replace(case.stages[1], input_ids=torch.zeros(1)))

    with pytest.raises(ValueError, match="V4.1 graph inputs"):
        make_body(case).run(stages)

    assert all(peer.messages == [] for peer in case.endpoints)


@pytest.mark.parametrize("split", [(5,), (2, 3), (0, 3), (3, 0)])
@pytest.mark.parametrize("engram", [False, True])
@pytest.mark.parametrize("hybrid", [False, True])
def test_pipeline_preserves_native_continuations_aux_and_parent_mtp(
    monkeypatch, split, engram, hybrid
):
    case = make_graph_model_case(split, engram=engram)
    expected = case.model(**case.parent_inputs)
    expected_mtp = case.mtp_rows.clone()
    attach_graph_pipeline(monkeypatch, pipeline_module, case, hybrid=hybrid)

    actual = make_body(case, pipeline=case.pipeline).run(case.stages)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(case.mtp_rows, expected_mtp, rtol=0, atol=0)
    assert all(
        event[-2:] == (True, True)
        for event in case.runtime.trace
        if event[0] == "attention"
    )


@pytest.mark.parametrize("hybrid", [False, True])
def test_pipeline_enqueues_whole_layer_wire_before_next_compute_and_joins_outputs(
    monkeypatch, hybrid
):
    case = attach_graph_pipeline(
        monkeypatch, pipeline_module, make_graph_model_case(), hybrid=hybrid
    )
    make_body(case, pipeline=case.pipeline).run(case.stages)
    trace = case.runtime.trace
    assert [event[1:3] for event in trace if event[0] == "attention"] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    first_layer_wire = [
        index for index, event in enumerate(trace) if event[0] == "wire"
    ][:8]
    next_compute = trace.index(("attention", 1, 0, True, True))
    assert max(first_layer_wire) < next_compute
    assert {event[-1] for event in trace if event[0] == "wire"} == {"parent"}
    assert trace.index(("wait", "S1.tail", "parent")) < trace.index(
        ("parent_output", "parent")
    )


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("split", [(5,), (2, 3)])
def test_pipeline_parent_aligns_native_stream_queries_and_consumes_one_producer(
    monkeypatch, hybrid, split
):
    case = make_graph_model_case(split)
    expected = case.model(**case.parent_inputs)
    streams = make_npu_stream_case(monkeypatch, pipeline_module, cached="foreign")
    attach_graph_pipeline(
        monkeypatch, pipeline_module, case, hybrid=hybrid, runtime=streams.runtime
    )

    actual = DeepseekV41GraphParent(make_body(case, pipeline=case.pipeline))(
        case.graph_inputs
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(case.producer_parents) == 1
    assert case.joined == [True]
    assert case.producer_parents[0][0].shape == case.parent_inputs["input_ids"].shape
    assert [event for event in case.runtime.trace if event[0] == "parent_stream"] == [
        ("parent_stream", "producer", True, True),
        ("parent_stream", "output", True, True),
    ]
    assert torch.npu.current_stream() is streams.runtime.parent
    assert streams.query() is streams.previous


def test_pipeline_rejects_a_foreign_bank_before_any_payload(monkeypatch):
    case, foreign = make_graph_model_case(), make_graph_model_case()
    attach_graph_pipeline(monkeypatch, pipeline_module, foreign)
    with pytest.raises(ValueError, match="exact stage bank"):
        make_body(case, pipeline=foreign.pipeline)
    assert all(peer.messages == [] for peer in case.endpoints + foreign.endpoints)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("split", [(5,), (2, 3)])
def test_pipeline_failure_restores_parent_context_state_and_native_stream_cache(
    monkeypatch, hybrid, split
):
    streams = make_npu_stream_case(monkeypatch, pipeline_module, cached="foreign")
    case = attach_graph_pipeline(
        monkeypatch,
        pipeline_module,
        make_graph_model_case(split),
        hybrid=hybrid,
        runtime=streams.runtime,
    )
    original = case.model.shared_attention_state
    parent = SimpleNamespace(ubatch_idx=-1)

    def fail(*args):
        raise RuntimeError("injected pipeline payload failure")

    monkeypatch.setattr(case.endpoints[-1], "receive", fail)
    with override_forward_context(parent):
        with pytest.raises(RuntimeError, match="pipeline payload failure"):
            DeepseekV41GraphParent(make_body(case, pipeline=case.pipeline))(
                case.graph_inputs
            )
        assert get_forward_context() is parent
    assert case.runtime.current is case.runtime.parent
    assert streams.query() is streams.previous
    assert [event for event in case.runtime.trace if event[0] == "parent_stream"] == [
        ("parent_stream", "producer", True, True)
    ]
    assert case.model.shared_attention_state is original
    assert all(layer.self_attn.shared_state is original for layer in case.model.layers)
    assert case.joined == [True]
