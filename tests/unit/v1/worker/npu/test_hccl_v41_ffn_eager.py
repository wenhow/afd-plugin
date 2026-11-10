# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow the CPU dependency gates.
# ruff: noqa: E402
"""CPU ticket/context seams only; no native MoE or NPU progress acceptance."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from afd_plugin.connectors.npu import hccl_eager_pipeline as owner_module
from afd_plugin.connectors.npu import hccl_staged_transport as transport_module
from afd_plugin.v1.worker.npu import hccl_v41_ffn_eager as consumer_module
from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner
from tests.unit.fakes.hccl import ROUTER_DELTA_SCALE
from tests.unit.fakes.hccl_eager_pipeline import make_eager_progress
from tests.unit.fakes.hccl_v41_ffn_eager import make_ffn_eager_case
from tests.unit.fakes.npu_stream import HostStream


@pytest.fixture
def case(monkeypatch):
    return make_ffn_eager_case(
        monkeypatch,
        owner_module,
        transport_module,
        consumer_module,
        AFDNPUFFNModelRunner,
    )


def run(case, *, progress=None):
    return consumer_module.execute_hccl_v41_eager_parent(
        case.runner,
        case.transport,
        case.owner,
        compute_stream=case.pair.runtime.compute,
        progress=progress,
    )


@pytest.mark.parametrize("tp,role_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize(
    "rows,valid", [((2, 3), (2, 1)), ((0, 3), (0, 0)), ((2, 0), (0, 0))]
)
def test_native_context_and_exact_unmasked_outputs_complete_once(
    monkeypatch, tp, role_rank, rows, valid
):
    case = make_ffn_eager_case(
        monkeypatch,
        owner_module,
        transport_module,
        consumer_module,
        AFDNPUFFNModelRunner,
        tp=tp,
        role_rank=role_rank,
        rows=rows,
        valid=valid,
    )
    assert run(case) is None
    expected_steps = list(case.inputs)
    assert [(x["layer"], x["stage"]) for x in case.model.observations] == expected_steps
    for observed in case.model.observations:
        layer, stage = observed["layer"], observed["stage"]
        hidden, fp32, ids = case.inputs[layer, stage]
        torch.testing.assert_close(observed["fp32"], fp32, rtol=0, atol=0)
        torch.testing.assert_close(observed["ids"], ids, rtol=0, atol=0)
        assert not torch.equal(fp32, hidden.float())
        expected = (
            (fp32 - hidden.float()) * ROUTER_DELTA_SCALE + ids[:, None].float() + layer
        ).bfloat16()
        packet = case.pair.backend.packets[stage, role_rank, tp + role_rank].popleft()
        torch.testing.assert_close(packet, expected, rtol=0, atol=0)
        assert observed["stream"] == "compute" and observed["inference"]
    assert case.owner.finished and not case.owner.failed
    assert case.context.current is None
    assert case.pair.runtime.current is case.pair.runtime.parent
    assert case.query_stream() is case.previous_stream
    messages = [
        message.kind
        for direction, _, _, message in case.transport.transcript
        if direction == "send"
    ]
    assert messages.count("COMPLETE") == 1


def test_ffn_diagnostics_order_received_loan_native_compute_and_posted_output(case):
    diagnostics = make_eager_progress(case.owner, case.pair.runtime.compute)
    run(case, progress=diagnostics.progress)
    events = [
        record["event"]
        for record in diagnostics.recorder.records
        if record.get("layer") == 0 and record.get("stage") == 0
    ]
    assert events == [
        "recv_post_begin",
        "recv_post_returned",
        "loan_wait_begin",
        "loan_wait_returned",
        "compute_begin",
        "compute_returned",
        "send_post_begin",
        "send_post_returned",
        "stage_reuse_wait_begin",
        "stage_reuse_wait_returned",
    ]
    assert diagnostics.recorder.events[-1] == "parent_finish_returned"
    assert "device_sync" not in diagnostics.recorder.events
    assert case.owner.finished and len(case.model.observations) == 4


@pytest.mark.parametrize("boundary", ["transaction", "rank", "actual_streams"])
def test_foreign_ffn_progress_is_rejected_without_receiving_or_failing_owner(
    case, boundary
):
    diagnostics = make_eager_progress(case.owner, case.pair.runtime.compute)
    progress = diagnostics.progress
    if boundary == "transaction":
        progress = replace(progress, transaction_id="foreign")
    elif boundary == "rank":
        progress = replace(
            progress, diagnostics=replace(progress.diagnostics, world_rank=5)
        )
    else:
        progress = replace(
            progress,
            actual_execution=replace(
                progress.actual_execution, stream_aliases=(True, True, True)
            ),
        )
    before = list(case.pair.backend.trace)
    with pytest.raises(ValueError, match="parent|executor streams"):
        run(case, progress=progress)
    assert case.pair.backend.trace == before and case.model.observations == []
    assert not case.owner.failed and not case.owner.finished
    assert case.query_stream() is case.previous_stream


@pytest.mark.parametrize("event", ["compute_begin", "compute_returned"])
def test_ffn_compute_sink_failure_keeps_live_owner_and_restores_native_context(
    case, event
):
    diagnostics = make_eager_progress(case.owner, case.pair.runtime.compute)
    diagnostics.recorder.fail_event = event
    with pytest.raises(RuntimeError, match="diagnostic sink failed"):
        run(case, progress=diagnostics.progress)
    assert case.owner.failed and not case.owner.finished
    assert case.transport.eager_pipeline is case.owner and case.transport.terminal
    assert case.context.current is None
    assert case.query_stream() is case.previous_stream
    assert len(case.model.observations) == (event == "compute_returned")


def test_empty_parent_finishes_without_a_loaded_model(monkeypatch):
    case = make_ffn_eager_case(
        monkeypatch,
        owner_module,
        transport_module,
        consumer_module,
        AFDNPUFFNModelRunner,
        rows=(0, 0),
    )
    case.runner.model = None
    run(case)
    assert case.owner.finished and case.model.observations == []
    assert case.transport.plan is None


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("data_parallel_size", 2),
        ("data_parallel_rank", 1),
        ("pipeline_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("use_sequence_parallel_moe", True),
        ("tensor_parallel_size", 2),
        ("rank", 1),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_invalid_native_coordinates_fail_before_receiving(
    monkeypatch, attribute, value, empty
):
    case = make_ffn_eager_case(
        monkeypatch,
        owner_module,
        transport_module,
        consumer_module,
        AFDNPUFFNModelRunner,
        rows=(0, 0) if empty else (2, 3),
    )
    setattr(case.runner.vllm_config.parallel_config, attribute, value)
    with pytest.raises(ValueError, match="paired native TP1/TP2"):
        run(case)
    assert case.model.observations == []
    assert case.owner.failed and not case.owner.finished
    assert case.transport.terminal


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("initialized", False),
        ("closed", True),
        ("terminal", True),
        ("role", "attention"),
    ],
)
def test_invalid_transport_cannot_compute_or_finish(case, attribute, value):
    setattr(case.transport, attribute, value)
    with pytest.raises(RuntimeError, match="active FFN transport"):
        run(case)
    assert case.model.observations == [] and not case.owner.failed


@pytest.mark.parametrize(
    "boundary",
    ["no_model", "layer_layout", "speculation", "parent_stream", "compute_device"],
)
def test_invalid_model_or_stream_boundary_is_terminal_before_compute(case, boundary):
    compute = case.pair.runtime.compute
    if boundary == "no_model":
        case.runner.model = None
    elif boundary == "layer_layout":
        case.model.layers = (0,)
    elif boundary == "speculation":
        case.runner.vllm_config.speculative_config = SimpleNamespace()
    elif boundary == "parent_stream":
        case.pair.runtime.current = compute
    else:
        compute = HostStream("foreign", torch.device("meta"))
    with pytest.raises((ValueError, RuntimeError)):
        consumer_module.execute_hccl_v41_eager_parent(
            case.runner, case.transport, case.owner, compute_stream=compute
        )
    assert case.model.observations == []
    assert case.owner.failed and case.transport.eager_pipeline is case.owner


def test_native_compute_failure_restores_parent_context_and_retains_owner(case):
    case.model.fail = True
    with pytest.raises(ValueError, match="controlled staged FFN compute failure"):
        run(case)
    assert case.context.current is None
    assert case.pair.runtime.current is case.pair.runtime.parent
    assert case.owner.failed and not case.owner.finished
    assert case.transport.eager_pipeline is case.owner and case.transport.terminal
    assert case.query_stream() is case.previous_stream
    assert not case.pair.backend.packets[0, 0, 1]


def test_work_failure_retains_owner_without_entering_native_compute(case):
    case.pair.backend.fail_wait = True
    with pytest.raises(RuntimeError, match="host Work failure"):
        run(case)
    assert case.model.observations == [] and case.context.current is None
    assert case.owner.failed and case.transport.eager_pipeline is case.owner


def test_foreign_owner_is_rejected_without_failing_the_other_session(case):
    foreign_transport = SimpleNamespace()
    with pytest.raises(ValueError, match="exact active transport/plan owner"):
        consumer_module.execute_hccl_v41_eager_parent(
            case.runner,
            foreign_transport,
            case.owner,
            compute_stream=case.pair.runtime.compute,
        )
    assert not case.owner.failed and not case.owner.finished
    assert case.model.observations == []


@pytest.mark.parametrize("boundary", ["finished", "failed", "replaced_plan"])
def test_stale_owner_rejection_preserves_the_original_error_and_state(case, boundary):
    if boundary == "finished":
        run(case)
    elif boundary == "failed":
        case.owner.fail()
    else:
        case.transport.plan = replace(case.transport.plan)
    observations = list(case.model.observations)
    with pytest.raises(ValueError, match="exact active transport/plan owner"):
        run(case)
    assert case.owner.failed == (boundary == "failed")
    assert case.owner.finished == (boundary == "finished")
    assert case.model.observations == observations
    assert case.query_stream() is case.previous_stream


@pytest.mark.parametrize("empty", [False, True])
def test_native_complete_observes_parent_tls_before_restoring_foreign_cache(
    monkeypatch, empty
):
    case = make_ffn_eager_case(
        monkeypatch,
        owner_module,
        transport_module,
        consumer_module,
        AFDNPUFFNModelRunner,
        rows=(0, 0) if empty else (2, 3),
    )
    synchronize = case.pair.runtime.synchronize

    def complete_on_parent(device):
        assert case.query_stream() is case.pair.runtime.parent
        assert case.pair.runtime.current is case.pair.runtime.parent
        synchronize(device)

    monkeypatch.setattr(case.pair.runtime, "synchronize", complete_on_parent)
    run(case)
    assert case.owner.finished
    assert case.query_stream() is case.previous_stream
