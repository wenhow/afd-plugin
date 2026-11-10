# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# ruff: noqa: E402
"""Host dependency protocol and CPU bank arithmetic; no NPU/Graph execution."""

import gc
import weakref
from dataclasses import FrozenInstanceError, replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

import afd_plugin.connectors.npu.hccl_graph_pipeline as module
from afd_plugin.connectors.npu.hccl_overlap import (
    HcclOverlapSettings,
    select_hccl_overlap,
)
from tests.unit.fakes.hccl_graph_pipeline import (
    HostStream,
    make_pipeline_case,
    run_attention_arithmetic,
    run_ffn_arithmetic,
    run_ffn_layer,
)


@pytest.fixture
def case(monkeypatch):
    return make_pipeline_case(monkeypatch, module)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("rows", [(2, 3), (0, 3), (3, 0)])
def test_attention_delayed_continuations_preserve_cpu_outputs(
    monkeypatch, hybrid, rows
):
    case = make_pipeline_case(monkeypatch, module, rows=rows, hybrid=hybrid)
    outputs = run_attention_arithmetic(case)
    expected = tuple(
        torch.full((count, 4), 9 * (stage + 1) + 10, dtype=torch.bfloat16)
        for stage, count in enumerate(rows)
        if count
    )
    for actual, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, reference)


def test_attention_parent_exchange_preserves_original_fp32_ids_and_wire_order(case):
    run_attention_arithmetic(case)
    for stage, exchange in enumerate(case.bank.exchanges):
        messages = exchange.endpoint.messages
        assert [message[:2] for message in messages] == [
            ("send", (case.bank.stage_rows[stage], 4)),
            ("send", (case.bank.stage_rows[stage], 4)),
            ("send", (case.bank.stage_rows[stage],)),
            ("receive", (case.bank.stage_rows[stage], 4)),
        ] * 2
        assert torch.equal(messages[1][2], messages[0][2].float() + 0.00031)
        assert torch.equal(
            messages[2][2], torch.arange(case.bank.stage_rows[stage], dtype=torch.int32)
        )
    wire = [event for event in case.runtime.trace if event[0] == "wire"]
    assert [event[2] for event in wire] == [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4
    assert {event[-1] for event in wire} == {"parent"}


@pytest.mark.parametrize("hybrid", [False, True])
def test_attention_next_compute_uses_its_declared_ready_frontier(monkeypatch, hybrid):
    case = make_pipeline_case(monkeypatch, module, hybrid=hybrid)
    run_attention_arithmetic(case)
    trace = case.runtime.trace
    for stage in range(2):
        ready = f"L0S{stage}.receive" if hybrid else f"L1S{stage}.ready"
        assert ("wait", ready, "compute") in trace
        assert ("wait", f"L0S{stage}.compute", "parent") in trace
        final_ready = f"L1S{stage}.receive" if hybrid else f"S{stage}.final_ready"
        assert ("wait", final_ready, "compute") in trace
        assert (("record", f"S{stage}.final_ready", "parent") in trace) is not hybrid
    assert [
        event for event in trace if event[:1] == ("wait",) and event[1].endswith("tail")
    ] == [("wait", "S0.tail", "parent"), ("wait", "S1.tail", "parent")]


@pytest.mark.parametrize("rows", [(2, 3), (0, 3), (3, 0)])
def test_ffn_side_compute_preserves_cpu_outputs_and_parent_hccl(monkeypatch, rows):
    case = make_pipeline_case(monkeypatch, module, role="ffn", rows=rows)
    outputs = run_ffn_arithmetic(case)
    expected = tuple(
        torch.full((count, 4), layer + stage + 1, dtype=torch.bfloat16)
        for layer in (0, 1)
        for stage, count in enumerate(rows)
        if count
    )
    for actual, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(actual, reference)
    assert {event[-1] for event in case.runtime.trace if event[0] == "wire"} == {
        "parent"
    }


def test_ffn_stage_reuse_and_final_join_follow_matching_send_events(monkeypatch):
    case = make_pipeline_case(monkeypatch, module, role="ffn")
    run_ffn_arithmetic(case)
    trace = case.runtime.trace
    for stage in range(2):
        assert ("wait", f"L0S{stage}.receive", "compute") in trace
        assert ("wait", f"L0S{stage}.send", "parent") in trace
    assert trace[-2:] == [
        ("wait", "L1S0.send", "parent"),
        ("wait", "L1S1.send", "parent"),
    ]


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_static_zero_stage_records_no_events_or_hccl(monkeypatch, role):
    case = make_pipeline_case(monkeypatch, module, role=role, rows=(0, 3))
    if role == "attention":
        run_attention_arithmetic(case)
    else:
        run_ffn_arithmetic(case)
    assert all(event[2] == 1 for event in case.runtime.trace if event[0] == "wire")
    assert all(
        "S0" not in event[1]
        for event in case.runtime.trace
        if event[0] in ("wait", "record")
    )


@pytest.mark.parametrize(
    "failure", ["missing_compute", "wire_order", "missing_receive", "missing_join"]
)
def test_incomplete_dependency_run_fails_without_wire_and_cannot_reenter(case, failure):
    pipeline = case.pipeline
    with pytest.raises(RuntimeError), pipeline.run():
        if failure == "missing_compute":
            with pipeline.attention_exchange(0, 0):
                case.runtime.trace.append(("unexpected",))
        elif failure == "missing_join":
            pass
        else:
            for stage in range(2):
                with pipeline.compute(0, stage):
                    pass
            if failure == "wire_order":
                with pipeline.attention_exchange(0, 1):
                    case.runtime.trace.append(("unexpected",))
            else:
                with pipeline.compute(1, 0):
                    case.runtime.trace.append(("unexpected",))
    assert not any(event[0] in ("unexpected", "wire") for event in case.runtime.trace)
    with pytest.raises(RuntimeError, match="failed"), pipeline.run():
        pass


@pytest.mark.parametrize("operation", ["exchange", "ffn_receive", "ffn_send"])
def test_hccl_scope_rejects_a_foreign_current_stream_before_body(case, operation):
    pipeline, runtime = case.pipeline, case.runtime
    with (
        pytest.raises(RuntimeError, match="parent stream"),
        pipeline.run(),
        runtime.stream(HostStream("foreign", case.bank.device)),
    ):
        context = {
            "exchange": pipeline.attention_exchange,
            "ffn_receive": pipeline.ffn_receive,
            "ffn_send": pipeline.ffn_send,
        }[operation](0, 0)
        with context:
            runtime.trace.append(("unexpected",))
    assert ("unexpected",) not in runtime.trace


def test_nested_run_is_rejected_without_corrupting_the_outer_parent(case):
    with (
        pytest.raises(RuntimeError, match="active"),
        case.pipeline.run(),
        case.pipeline.run(),
    ):
        pass
    assert case.runtime.current is case.runtime.parent


def test_body_failure_restores_current_stream_and_retains_all_crossing_allocations(
    case,
):
    references = []
    with pytest.raises(ValueError, match="compute failure"), case.pipeline.run():
        tensors = (
            torch.ones((2, 4), dtype=torch.bfloat16),
            torch.ones((2, 4), dtype=torch.float32),
            torch.ones(2, dtype=torch.int32),
            torch.zeros((2, 4), dtype=torch.bfloat16),
        )
        references = [weakref.ref(tensor) for tensor in tensors]
        with case.pipeline.compute(0, 0, inputs=tensors):
            raise ValueError("compute failure")
    del tensors
    gc.collect()
    assert case.runtime.current is case.runtime.parent
    assert all(reference() is not None for reference in references)


def test_successful_join_releases_temporary_cpu_references(case):
    outputs = run_attention_arithmetic(case)
    reference = weakref.ref(outputs[0])
    del outputs
    gc.collect()
    assert reference() is None


def test_replaced_stage_frontier_releases_only_its_old_cpu_references(case):
    with pytest.raises(ValueError, match="end trace"), case.pipeline.run():
        first, second = torch.ones(2), torch.ones(2)
        first_ref, second_ref = weakref.ref(first), weakref.ref(second)
        with case.pipeline.compute(0, 0, inputs=(first,)):
            pass
        with case.pipeline.compute(0, 1, inputs=(second,)):
            pass
        del first, second
        gc.collect()
        assert first_ref() is not None and second_ref() is not None
        with case.pipeline.attention_exchange(0, 0):
            pass
        gc.collect()
        assert first_ref() is None
        assert second_ref() is not None
        raise ValueError("end trace")


def test_constructor_rejects_reused_dependency_event_objects(case):
    events = dict(case.pipeline.step_events)
    events[(0, 1)] = replace(events[(0, 1)], receive_done=events[(0, 0)].receive_done)
    with pytest.raises(ValueError, match="must not alias"):
        module.HcclGraphComputePipeline(
            case.bank, case.runtime.compute, events, case.pipeline.tail_events
        )


def test_device_allocation_rejects_cpu_bank_before_creating_streams(case):
    with pytest.raises(ValueError, match="indexed NPU"):
        module.HcclGraphComputePipeline.allocate(case.bank)


def test_completed_body_cannot_reuse_events_for_a_different_phase(case):
    run_attention_arithmetic(case)
    with pytest.raises(RuntimeError, match="fresh phase owner"), case.pipeline.run():
        pass


@pytest.mark.parametrize("frontier", ["first_layer_compute", "previous_layer_wire"])
def test_attention_requires_complete_layer_frontiers_before_advancing(case, frontier):
    pipeline = case.pipeline
    with pytest.raises(RuntimeError, match="every"), pipeline.run():
        with pipeline.compute(0, 0):
            pass
        if frontier == "first_layer_compute":
            with pipeline.attention_exchange(0, 0):
                pass
        else:
            with pipeline.compute(0, 1):
                pass
            with pipeline.attention_exchange(0, 0):
                pass
            with pipeline.compute(1, 0):
                pass


@pytest.mark.parametrize("value", [None, 0, 1, "1"])
@pytest.mark.parametrize(
    "field", ["hybrid", "attention_three_stream", "ffn_recv_stream", "ffn_cross_layer"]
)
def test_graph_pipeline_policy_rejects_non_bool_modes(value, field):
    with pytest.raises(ValueError, match="must be bool"):
        module.HcclGraphPipelinePolicy(**{field: value})


@pytest.mark.parametrize("hybrid", [False, True])
def test_graph_pipeline_policy_is_fixed_for_its_owner_lifetime(hybrid):
    policy = module.HcclGraphPipelinePolicy(hybrid)
    assert policy.hybrid is hybrid
    with pytest.raises(FrozenInstanceError):
        policy.hybrid = not hybrid


def test_graph_pipeline_policy_rejects_cross_layer_without_receive_stream():
    with pytest.raises(ValueError, match="cross-layer.*receive stream"):
        module.HcclGraphPipelinePolicy(ffn_recv_stream=False)


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("phase", ["graph_warmup", "graph_capture", "graph_replay"])
def test_policy_projection_keeps_session_settings_for_each_eligible_role(role, phase):
    settings = HcclOverlapSettings(graph_u2_hybrid_dag=False)
    selection = select_hccl_overlap(settings, role=role, phase=phase, num_stages=2)
    assert module.HcclGraphPipelinePolicy.from_selection(selection) == (
        module.HcclGraphPipelinePolicy(hybrid=False)
    )


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("num_stages,compute", [(1, True), (2, False)])
def test_policy_projection_leaves_graph_u1_and_compute_off_synchronous(
    role, num_stages, compute
):
    selection = select_hccl_overlap(
        HcclOverlapSettings(graph_u2_compute_overlap=compute),
        role=role,
        phase="graph_capture",
        num_stages=num_stages,
    )
    assert module.HcclGraphPipelinePolicy.from_selection(selection) is None
    assert selection.phase == "graph_capture"


def test_policy_projection_rejects_eager_selection():
    selection = select_hccl_overlap(
        HcclOverlapSettings(), role="attention", phase="eager", num_stages=2
    )
    with pytest.raises(ValueError, match="Graph phase"):
        module.HcclGraphPipelinePolicy.from_selection(selection)


@pytest.mark.parametrize("three_stream", [False, True])
@pytest.mark.parametrize("use_selection", [False, True])
def test_attention_logical_plan_resolves_to_parent_exchange_without_changing_outputs(
    monkeypatch, three_stream, use_selection
):
    selection = select_hccl_overlap(
        HcclOverlapSettings(graph_u2_attention_three_stream=three_stream),
        role="attention",
        phase="graph_capture",
        num_stages=2,
    )
    option = (
        {"selection": selection}
        if use_selection
        else {"policy": module.HcclGraphPipelinePolicy.from_selection(selection)}
    )
    case = make_pipeline_case(monkeypatch, module, **option)
    outputs = run_attention_arithmetic(case)
    for stage, output in enumerate(outputs):
        torch.testing.assert_close(
            output, torch.full_like(output, 9 * (stage + 1) + 10)
        )
    assert case.pipeline.communication_plan() == {
        "logical.send_stream": "send" if three_stream else "parent",
        "logical.recv_stream": "recv" if three_stream else "parent",
        "physical.send_stream": "parent",
        "physical.recv_stream": "parent",
        "parent_transport_override": True,
    }
    assert case.pipeline.selection == (selection if use_selection else None)
    assert {event[-1] for event in case.runtime.trace if event[0] == "wire"} == {
        "parent"
    }


@pytest.mark.parametrize(
    "recv_stream,cross_layer", [(True, True), (True, False), (False, False)]
)
@pytest.mark.parametrize("use_selection", [False, True])
def test_ffn_receive_and_cross_layer_axes_consume_independent_send_frontiers(
    monkeypatch, recv_stream, cross_layer, use_selection
):
    selection = select_hccl_overlap(
        HcclOverlapSettings(
            graph_u2_ffn_recv_stream=recv_stream, graph_u2_ffn_cross_layer=cross_layer
        ),
        role="ffn",
        phase="graph_warmup",
        num_stages=2,
    )
    option = (
        {"selection": selection}
        if use_selection
        else {"policy": module.HcclGraphPipelinePolicy.from_selection(selection)}
    )
    case = make_pipeline_case(monkeypatch, module, role="ffn", **option)
    outputs = run_ffn_arithmetic(case)
    for index, output in enumerate(outputs):
        layer, stage = divmod(index, 2)
        torch.testing.assert_close(output, torch.full_like(output, layer + stage + 1))
    waits = [event[1] for event in case.runtime.trace if event[0] == "wait"]
    for stage in range(2):
        assert waits.count(f"L0S{stage}.send") == int(recv_stream) + int(
            not cross_layer
        )
        assert waits.count(f"L1S{stage}.send") == 1 + int(not cross_layer)
        assert ("wait", f"L0S{stage}.receive", "compute") in case.runtime.trace
        assert ("wait", f"L1S{stage}.receive", "compute") in case.runtime.trace
    assert case.pipeline.communication_plan()["logical.recv_stream"] == (
        "recv" if recv_stream else "parent"
    )
    assert {event[-1] for event in case.runtime.trace if event[0] == "wire"} == {
        "parent"
    }


@pytest.mark.parametrize(
    "combination", ["hybrid_policy", "hybrid_selection", "policy_selection", "all"]
)
@pytest.mark.parametrize("allocate", [False, True])
def test_multiple_policy_inputs_are_rejected_before_allocation_or_run(
    case, combination, allocate
):
    options = {}
    if combination in ("hybrid_policy", "hybrid_selection", "all"):
        options["hybrid"] = False
    if combination in ("hybrid_policy", "policy_selection", "all"):
        options["policy"] = module.HcclGraphPipelinePolicy()
    if combination in ("hybrid_selection", "policy_selection", "all"):
        options["selection"] = select_hccl_overlap(
            HcclOverlapSettings(), role="attention", phase="graph_capture", num_stages=2
        )
    with pytest.raises(ValueError, match="pass one"):
        if allocate:
            module.HcclGraphComputePipeline.allocate(case.bank, **options)
        else:
            module.HcclGraphComputePipeline(
                case.bank,
                case.runtime.compute,
                case.pipeline.step_events,
                case.pipeline.tail_events,
                **options,
            )
    assert case.runtime.trace == []


@pytest.mark.parametrize("invalid", ["role", "stages", "eager", "compute_off"])
def test_selection_must_match_the_bank_and_active_graph_compute(case, invalid):
    selection = select_hccl_overlap(
        HcclOverlapSettings(graph_u2_compute_overlap=invalid != "compute_off"),
        role="ffn" if invalid == "role" else "attention",
        phase="eager" if invalid == "eager" else "graph_capture",
        num_stages=1 if invalid == "stages" else 2,
    )
    with pytest.raises(ValueError, match="selection"):
        module.HcclGraphComputePipeline(
            case.bank,
            case.runtime.compute,
            case.pipeline.step_events,
            case.pipeline.tail_events,
            selection=selection,
        )
    assert case.runtime.trace == []


@pytest.mark.parametrize(
    "failure",
    ["incomplete", "duplicate", "wrong_layer", "missing_prior", "missing_final"],
)
def test_ffn_non_cross_layer_requires_each_exact_layer_frontier(monkeypatch, failure):
    case = make_pipeline_case(
        monkeypatch,
        module,
        role="ffn",
        policy=module.HcclGraphPipelinePolicy(ffn_cross_layer=False),
    )
    with pytest.raises(RuntimeError, match="layer join"), case.pipeline.run():
        if failure == "incomplete":
            case.pipeline.join_layer(0)
        elif failure == "wrong_layer":
            case.pipeline.join_layer(99)
        else:
            run_ffn_layer(case, 0)
            if failure == "missing_prior":
                with case.pipeline.ffn_receive(1, 0):
                    pytest.fail("next layer entered without its required frontier")
            case.pipeline.join_layer(0)
            if failure == "duplicate":
                case.pipeline.join_layer(0)
            run_ffn_layer(case, 1)
            case.pipeline.join()


@pytest.mark.parametrize("role,cross_layer", [("attention", False), ("ffn", True)])
def test_layer_join_rejects_wrong_role_or_cross_layer_policy(
    monkeypatch, role, cross_layer
):
    case = make_pipeline_case(
        monkeypatch,
        module,
        role=role,
        policy=module.HcclGraphPipelinePolicy(ffn_cross_layer=cross_layer),
    )
    with pytest.raises(RuntimeError, match="layer join"), case.pipeline.run():
        case.pipeline.join_layer(0)


def test_legacy_explicit_u1_pipeline_remains_available(monkeypatch):
    case = make_pipeline_case(monkeypatch, module, rows=(5,), hybrid=False)
    (output,) = run_attention_arithmetic(case)
    torch.testing.assert_close(output, torch.full_like(output, 19))
    assert case.pipeline.selection is None
    assert case.pipeline.hybrid is False
