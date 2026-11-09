# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Native CPU preparation and host protocol only; no NPU graph acceptance."""

from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode

from afd_plugin.connectors.npu.hccl_graph_cache import HcclGraphCache
from afd_plugin.v1.worker.npu.hccl_v41_graph_eager import HcclV41GraphEager
from tests.unit.fakes.graph_cache import DeviceCompletion, GraphResources, ReplayGraph
from tests.unit.fakes.hccl_v41_graph_coordinator import dispatch_runtime
from tests.unit.fakes.hccl_v41_graph_eager import (
    make_graph_eager_case,
    run_graph_eager_parent,
)


@pytest.fixture
def eager_case(monkeypatch):
    # CPU operators stop at the existing public tensor peer. These two native
    # CPU preparation collaborators exclude NPU rotary/platform metadata work.
    monkeypatch.setattr(torch.npu, "synchronize", lambda *args: None)
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda _: None
    )
    monkeypatch.setattr(
        "afd_plugin.v1.worker.npu.hccl_v41_ubatch_runner.current_platform.set_additional_forward_context",
        lambda *args, **kwargs: {},
    )
    return make_graph_eager_case


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize(
    "mode,dummy", [("target", False), ("dummy", True), ("profile", True)]
)
def test_pre_kv_entry_agrees_eager_before_native_parent_and_completes(
    eager_case, stages, mode, dummy
):
    case = eager_case(stages=stages)
    with case.eager.execution_scope(
        mode=mode, valid_tokens=0 if dummy else 4, dummy_run=dummy
    ):
        descriptor, (output, auxiliary) = run_graph_eager_parent(
            case, dummy=dummy, tokens=4
        )
    assert descriptor.cg_mode == CUDAGraphMode.NONE
    # Native dummy dispatch has its genuine U1 geometry, even with a U2 runner.
    expected_rows = (2, 2) if stages == 2 and not dummy else (4,)
    sent = [
        message
        for direction, _, _, message in case.transport.transcript
        if direction == "send"
    ]
    kinds = [message.kind for message in sent]
    assert [kind for kind in kinds if kind.startswith("GRAPH_")] == [
        "GRAPH_QUERY",
        "GRAPH_DECISION",
        "GRAPH_ACK",
    ]
    assert kinds.index("GRAPH_ACK") < kinds.index("FORWARD") < kinds.index("COMPLETE")
    candidate = next(
        message.graph_candidate for message in sent if message.kind == "GRAPH_QUERY"
    )
    assert candidate.phase == "eager"
    assert candidate.eager_padded_tokens == expected_rows
    plan = next(message.plan for message in sent if message.kind == "FORWARD")
    assert tuple(stage.attention_valid_tokens[0] for stage in plan.stages) == (
        (0,) * len(expected_rows) if dummy else expected_rows
    )
    assert plan.mode == mode and plan.graph_execution.phase == "eager"
    assert output.shape == (4, 4) and len(auxiliary) == 2
    assert case.transport.plan is None and case.transport.graph_reservation is None
    assert not case.transport.terminal and case.cache.is_empty


def test_zero_early_return_retires_fresh_empty_parents_without_native_request_state(
    eager_case,
):
    case = eager_case()
    for _ in range(2):
        with case.eager.execution_scope(mode="target", valid_tokens=0, dummy_run=False):
            pass
    plans = [
        message.plan
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == "FORWARD"
    ]
    assert [plan.transaction_id for plan in plans] == ["native-0", "native-1"]
    assert all(plan.stages[0].attention_padded_tokens == (0,) for plan in plans)
    assert case.transport.plan is None and case.transport.graph_reservation is None
    assert not case.transport.terminal and case.cache.is_empty


@pytest.mark.parametrize("dispatched", [False, True])
def test_nonempty_return_requires_dispatch_and_accepted_parent_completion(
    eager_case, dispatched
):
    case = eager_case()
    message = "did not complete" if dispatched else "before pre-KV dispatch"
    with (
        pytest.raises(RuntimeError, match=message),
        case.eager.execution_scope(mode="target", valid_tokens=4, dummy_run=False),
    ):
        if dispatched:
            dispatch_runtime(case)
    assert case.transport.terminal and case.cache.terminal
    assert not any(
        message.kind == "COMPLETE"
        for direction, _, _, message in case.transport.transcript
        if direction == "send"
    )


@pytest.mark.parametrize("scheduled", [(1, 1, 1, 1), (2, 3)])
def test_actual_dispatch_selects_none_before_engram_input_preparation(
    eager_case, scheduled
):
    case = eager_case(scheduled=scheduled)
    case.owner.model_state._cg_mode = CUDAGraphMode.FULL
    with (
        pytest.raises(RuntimeError, match="body boundary"),
        case.eager.execution_scope(
            mode="target", valid_tokens=sum(scheduled), dummy_run=False
        ),
    ):
        descriptor = dispatch_runtime(case)
        assert descriptor.num_tokens == sum(scheduled)
        assert descriptor.num_reqs == len(scheduled)
        assert descriptor.cg_mode == CUDAGraphMode.NONE
        prepared = case.owner.model_state.prepare_inputs()
        assert prepared["producer_mode"] == CUDAGraphMode.NONE
        raise RuntimeError("body boundary")
    assert case.transport.terminal and case.cache.terminal


@pytest.mark.parametrize(
    "change",
    [
        "manager",
        "control-manager",
        "transport",
        "cache",
        "model",
        "model-state",
        "config",
        "parallel",
        "ubatch",
    ],
)
def test_changed_owner_is_rejected_before_any_query(eager_case, change):
    case = eager_case()
    replacement = SimpleNamespace()
    if change == "control-manager":
        case.control.manager = replacement
    elif change == "transport":
        case.owner.hccl_v41_transport = replacement
    elif change == "cache":
        case.control.cache = HcclGraphCache(DeviceCompletion().synchronize)
    else:
        field = {
            "manager": "cudagraph_manager",
            "model-state": "model_state",
            "config": "vllm_config",
            "parallel": "parallel_config",
            "ubatch": "ubatch_runner",
        }.get(change, change)
        setattr(case.owner, field, replacement)
    with (
        pytest.raises(ValueError, match="manager-free owners"),
        case.eager.execution_scope(mode="target", valid_tokens=0, dummy_run=False),
    ):
        pass
    assert case.transport.transcript == []


def test_pre_kv_entry_rejects_retained_cache_dependencies_before_query(eager_case):
    case = eager_case()
    case.cache.reserve_capture(case.key, 0, ReplayGraph(), GraphResources())
    with (
        pytest.raises(ValueError, match="idle fresh graph session"),
        case.eager.execution_scope(mode="target", valid_tokens=0, dummy_run=False),
    ):
        pass
    assert case.cache.contains(case.key) and case.transport.transcript == []


@pytest.mark.parametrize("valid,dummy", [(True, False), (-1, False), (1, True), (0, 1)])
def test_pre_kv_counts_require_exact_native_types(eager_case, valid, dummy):
    case = eager_case()
    with (
        pytest.raises(ValueError, match="native mode and valid counts"),
        case.eager.execution_scope(mode="target", valid_tokens=valid, dummy_run=dummy),
    ):
        pass
    assert case.transport.transcript == []


def test_constructor_requires_the_actual_bound_transport(eager_case):
    case = eager_case()
    case.owner.hccl_v41_transport = None
    with pytest.raises(ValueError, match="manager-free owners"):
        HcclV41GraphEager(
            case.owner, case.control, next_transaction_id=case.next_transaction_id
        )
    assert case.transport.transcript == []


def test_native_dispatch_binding_is_restored_after_exception(eager_case):
    case = eager_case()
    with (
        pytest.raises(RuntimeError, match="body boundary"),
        case.eager.execution_scope(mode="target", valid_tokens=4, dummy_run=False),
    ):
        dispatch_runtime(case)
        raise RuntimeError("body boundary")
    before = len(case.transport.transcript)
    # The restored native dispatcher requires explicit eager mode when its
    # actual manager is None; only the scoped bridge adds that guarantee.
    descriptor = dispatch_runtime(case, need_eager=True)
    assert descriptor.cg_mode == CUDAGraphMode.NONE
    assert len(case.transport.transcript) == before
