# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Actual control/native slicing/model-wrapper seams with CPU arithmetic."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.forward_context import (  # noqa: E402
    get_forward_context,
    override_forward_context,
)

from tests.unit.fakes.hccl_v41_graph_execution import (  # noqa: E402
    make_graph_execution_case,
)


@pytest.fixture
def graph_case(monkeypatch):
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda _: None
    )
    monkeypatch.setattr(
        "afd_plugin.v1.worker.npu.hccl_v41_ubatch_runner.current_platform.set_additional_forward_context",
        lambda *args, **kwargs: {},
    )
    return make_graph_execution_case


@pytest.mark.parametrize("warmup", [True, False])
@pytest.mark.parametrize("engram", [False, True])
def test_native_parent_callback_uses_actual_counts_and_one_unsliced_producer(
    graph_case, warmup, engram
):
    case = graph_case(warmup=warmup, engram=engram)
    prepared = case.runner.prepare(
        case.batch, (), torch.zeros(1, 4, dtype=torch.int32), CUDAGraphMode.FULL, True
    )
    finish = case.runner.begin_capturable_run(
        case.model, case.inputs, prepared, for_capture=True
    )
    parent = SimpleNamespace(ubatch_idx=-1)

    with override_forward_context(parent):
        output, auxiliary = finish()
        assert get_forward_context() is parent

    assert output.shape == (4, 4) and len(auxiliary) == 2
    assert case.transport.plan.graph_execution == case.reservation.decision
    assert [stage.attention_valid_tokens for stage in case.transport.plan.stages] == [
        (0,),
        (0,),
    ]
    assert all(
        peer.messages[2].tolist() == [0, 0] for peer in case.model_case.endpoints
    )
    assert len(case.model_case.producer_parents) == int(engram)
    assert case.model_case.joined == ([True] if engram else [])
    # Completion is deliberately outside this native tensor callback.
    case.transport.complete_graph_forward(case.reservation.decision.key)
    assert case.transport.plan is None and not case.transport.terminal


def test_foreign_reservation_handle_fails_before_counts_submission(graph_case):
    case = graph_case()
    geometry = case.reservation.geometry
    case.execution.reservation = replace(
        case.reservation,
        geometry=replace(
            geometry,
            stages=(
                replace(geometry.stages[0], request_stop=1),
                geometry.stages[1],
            ),
        ),
    )

    with pytest.raises(ValueError, match="current owned reservation"):
        case.runner.prepare(
            case.batch,
            (),
            torch.zeros(1, 4, dtype=torch.int32),
            CUDAGraphMode.FULL,
            True,
        )

    assert case.transport.terminal and case.transport.plan is None
    assert all(peer.messages == [] for peer in case.model_case.endpoints)


def test_actual_native_request_boundary_mismatch_rejects_before_counts(graph_case):
    case = graph_case()
    case.batch.num_tokens = case.batch.num_reqs = 2

    with pytest.raises(ValueError, match="actual native preparation"):
        case.runner.prepare(
            case.batch,
            (),
            torch.zeros(1, 4, dtype=torch.int32),
            CUDAGraphMode.FULL,
            True,
        )

    assert case.transport.terminal and case.transport.plan is None
    assert all(peer.messages == [] for peer in case.model_case.endpoints)


@pytest.mark.parametrize("foreign", ["model", "metadata"])
def test_foreign_native_wrapper_is_rejected_before_parent_payload(graph_case, foreign):
    case = graph_case()
    prepared = case.runner.prepare(
        case.batch, (), torch.zeros(1, 4, dtype=torch.int32), CUDAGraphMode.FULL, True
    )
    if foreign == "model":
        case.model.original_model = torch.nn.Identity()
    else:
        case.model.device_metadata = SimpleNamespace()

    finish = case.runner.begin_capturable_run(
        case.model, case.inputs, prepared, for_capture=True
    )
    with pytest.raises(ValueError, match="native ModelWithContext"):
        finish()

    assert case.transport.terminal
    assert all(peer.messages == [] for peer in case.model_case.endpoints)


def test_retired_reservation_cannot_launch_the_prepared_parent(graph_case):
    case = graph_case()
    prepared = case.runner.prepare(
        case.batch, (), torch.zeros(1, 4, dtype=torch.int32), CUDAGraphMode.FULL, True
    )
    finish = case.runner.begin_capturable_run(
        case.model, case.inputs, prepared, for_capture=True
    )
    case.transport.complete_graph_forward(case.reservation.decision.key)

    with pytest.raises(ValueError, match="current accepted reservation"):
        finish()

    assert case.transport.terminal
    assert all(peer.messages == [] for peer in case.model_case.endpoints)


@pytest.mark.parametrize("changed", ["generation", "metadata"])
def test_storage_change_after_preparation_rejects_before_parent_producer(
    graph_case, changed
):
    case = graph_case(engram=True)
    prepared = case.runner.prepare(
        case.batch, (), torch.zeros(1, 4, dtype=torch.int32), CUDAGraphMode.FULL, True
    )
    finish = case.runner.begin_capturable_run(
        case.model, case.inputs, prepared, for_capture=True
    )
    if changed == "generation":
        case.generation.value += 1
    else:
        case.metadata.shapes = ((2, 4),)

    with pytest.raises(ValueError, match="storage or metadata changed"):
        finish()

    assert case.transport.terminal
    assert case.model_case.producer_parents == []
    assert all(peer.messages == [] for peer in case.model_case.endpoints)


def test_parent_payload_failure_retains_terminal_state_and_restores_native_context(
    graph_case, monkeypatch
):
    case = graph_case(engram=True)
    prepared = case.runner.prepare(
        case.batch, (), torch.zeros(1, 4, dtype=torch.int32), CUDAGraphMode.FULL, True
    )

    def fail(*args):
        raise RuntimeError("parent tensor exchange failed")

    monkeypatch.setattr(case.model_case.endpoints[1], "receive", fail)
    finish = case.runner.begin_capturable_run(
        case.model, case.inputs, prepared, for_capture=True
    )
    parent = SimpleNamespace(ubatch_idx=-1)
    with override_forward_context(parent):
        with pytest.raises(RuntimeError, match="tensor exchange failed"):
            finish()
        assert get_forward_context() is parent

    assert case.transport.terminal
    assert case.model_case.joined == [True]
