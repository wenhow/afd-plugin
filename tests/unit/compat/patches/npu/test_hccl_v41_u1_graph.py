# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Actual native Engram preparation and U1 context protocol on CPU tensors."""

from concurrent.futures import ThreadPoolExecutor
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode
from vllm.forward_context import create_forward_context
from vllm_ascend.worker.v2.model_states.default import AscendModelState

from afd_plugin.compat.patches.npu.hccl_v41_u1_graph import (
    use_hccl_v41_u1_preparation,
)
from afd_plugin.v1.worker.npu.hccl_v41_ubatch_runner import HcclV41UBatchRunner
from tests.unit.fakes.deepseek_v41 import ParentEngramState
from tests.unit.fakes.hccl_v41_ubatch import GraphUBatchOwner, U1GraphOwner
from tests.unit.fakes.input_batch import (
    make_ascend_decode_batch,
    make_hccl_ubatch_seam,
    snapshot_ascend_child,
)


@pytest.fixture
def u1_case(monkeypatch):
    # Actual EngramModelState.prepare_attn executes. Its backend metadata
    # builder is the CPU boundary; no NPU attention/kernel is emulated.
    monkeypatch.setattr(
        AscendModelState, "prepare_attn", staticmethod(snapshot_ascend_child)
    )
    monkeypatch.setattr(
        "afd_plugin.compat.patches.npu.hccl_v41_u1_graph.current_platform.set_additional_forward_context",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda _: None
    )
    state = ParentEngramState({})
    runner = make_hccl_ubatch_seam(
        HcclV41UBatchRunner, model_state=state, max_num_reqs=8
    )
    peer = SimpleNamespace(terminal=False)
    parent = create_forward_context({}, runner.vllm_config)
    owner = U1GraphOwner(peer, parent)
    return SimpleNamespace(
        state=state,
        runner=runner,
        peer=peer,
        owner=owner,
        batch=make_ascend_decode_batch(),
        slots=torch.arange(8, dtype=torch.int32).reshape(1, 8),
    )


@pytest.mark.parametrize("for_capture", [False, True])
def test_u1_keeps_native_metadata_and_retains_full_parent_producer_mode(
    u1_case, for_capture
):
    case = u1_case
    mode = CUDAGraphMode.NONE if for_capture else CUDAGraphMode.FULL
    with use_hccl_v41_u1_preparation(
        case.state, case.runner.vllm_config, lambda: case.owner
    ):
        metadata = case.state.prepare_attn(
            case.batch,
            mode,
            (),
            case.slots,
            [],
            case.runner.kv_cache_config,
            for_capture=for_capture,
        )
    torch.testing.assert_close(metadata["child"].input_ids, case.batch.input_ids)
    assert case.state.prepare_inputs(case.batch, None) == {
        "producer_mode": CUDAGraphMode.FULL
    }
    assert not case.peer.terminal


def test_no_u1_owner_preserves_native_engram_mode_and_metadata(u1_case):
    case = u1_case
    with use_hccl_v41_u1_preparation(case.state, case.runner.vllm_config, lambda: None):
        metadata = case.state.prepare_attn(
            case.batch,
            CUDAGraphMode.NONE,
            (),
            case.slots,
            [],
            case.runner.kv_cache_config,
        )
    torch.testing.assert_close(metadata["child"].input_ids, case.batch.input_ids)
    assert case.state.prepare_inputs(case.batch, None) == {
        "producer_mode": CUDAGraphMode.NONE
    }


@pytest.mark.parametrize(
    "reject,error", [("before", "reservation"), ("after", "geometry")]
)
def test_failed_u1_phase_restores_mode_and_selected_native_method(
    u1_case, reject, error
):
    case = u1_case
    case.owner.reject = reject
    original = case.state.prepare_attn
    # Preserve a caller's existing bound-method override by identity.
    case.state.prepare_attn = original
    with (
        pytest.raises(RuntimeError, match=error),
        use_hccl_v41_u1_preparation(
            case.state, case.runner.vllm_config, lambda: case.owner
        ),
    ):
        case.state.prepare_attn(
            case.batch,
            CUDAGraphMode.NONE,
            (),
            case.slots,
            [],
            case.runner.kv_cache_config,
            for_capture=True,
        )
    assert case.state.prepare_attn is original
    assert case.state.prepare_inputs(case.batch, None) == {
        "producer_mode": CUDAGraphMode.NONE
    }
    assert case.peer.terminal


def test_native_u2_child_preparation_bypasses_the_outer_u1_scope(u1_case):
    case = u1_case
    case.owner.reject = "before"
    case.runner.graph_owner = GraphUBatchOwner(case.peer, case.owner.parent_context)
    with use_hccl_v41_u1_preparation(
        case.state, case.runner.vllm_config, lambda: case.owner
    ):
        prepared = case.runner.prepare(
            case.batch, (), case.slots, CUDAGraphMode.FULL, True
        )
    assert len(prepared.forward_contexts) == 2
    assert all(
        context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        for context in prepared.forward_contexts
    )
    assert not case.peer.terminal


def test_nested_u1_scope_rejects_and_preserves_the_outer_preparation(u1_case):
    case = u1_case
    with use_hccl_v41_u1_preparation(
        case.state, case.runner.vllm_config, lambda: case.owner
    ):
        with (
            pytest.raises(RuntimeError, match="cannot be nested"),
            use_hccl_v41_u1_preparation(
                case.state, case.runner.vllm_config, lambda: case.owner
            ),
        ):
            pass
        metadata = case.state.prepare_attn(
            case.batch,
            CUDAGraphMode.NONE,
            (),
            case.slots,
            [],
            case.runner.kv_cache_config,
            for_capture=True,
        )
    torch.testing.assert_close(metadata["child"].input_ids, case.batch.input_ids)


def test_u1_scope_rejects_foreign_thread_preparation(u1_case):
    case = u1_case
    with use_hccl_v41_u1_preparation(
        case.state, case.runner.vllm_config, lambda: case.owner
    ), ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(
            case.state.prepare_attn,
            case.batch,
            CUDAGraphMode.NONE,
            (),
            case.slots,
            [],
            case.runner.kv_cache_config,
            True,
        )
        with pytest.raises(RuntimeError, match="owning thread"):
            result.result()
    assert case.peer.terminal


def test_leaked_u1_binding_rejects_a_different_model_state(u1_case):
    case = u1_case
    other = ParentEngramState({})
    with use_hccl_v41_u1_preparation(
        case.state, case.runner.vllm_config, lambda: case.owner
    ):
        foreign = MethodType(case.state.prepare_attn.__func__, other)
        with pytest.raises(ValueError, match="selected ModelState"):
            foreign(
                case.batch,
                CUDAGraphMode.NONE,
                (),
                case.slots,
                [],
                case.runner.kv_cache_config,
                for_capture=True,
            )
    assert other.prepare_inputs(case.batch, None) == {
        "producer_mode": CUDAGraphMode.NONE
    }


@pytest.mark.parametrize(
    "mode,for_capture,index",
    [
        (CUDAGraphMode.NONE, False, 0),
        (CUDAGraphMode.PIECEWISE, False, 0),
        (CUDAGraphMode.PIECEWISE, True, 0),
        (CUDAGraphMode.FULL, False, 1),
    ],
)
def test_u1_owner_rejects_an_inconsistent_native_mode_or_child(
    u1_case, mode, for_capture, index
):
    case = u1_case
    with (
        pytest.raises(ValueError, match="FULL child zero"),
        use_hccl_v41_u1_preparation(
            case.state, case.runner.vllm_config, lambda: case.owner
        ),
    ):
        case.state.prepare_attn(
            case.batch,
            mode,
            (),
            case.slots,
            [],
            case.runner.kv_cache_config,
            for_capture=for_capture,
            ubatch_idx=index,
        )
    assert case.peer.terminal


def test_native_u1_builder_failure_restores_parent_mode(u1_case, monkeypatch):
    case = u1_case

    def fail(*args, **kwargs):
        raise RuntimeError("native metadata builder failed")

    monkeypatch.setattr(AscendModelState, "prepare_attn", staticmethod(fail))
    with (
        pytest.raises(RuntimeError, match="metadata builder failed"),
        use_hccl_v41_u1_preparation(
            case.state, case.runner.vllm_config, lambda: case.owner
        ),
    ):
        case.state.prepare_attn(
            case.batch,
            CUDAGraphMode.NONE,
            (),
            case.slots,
            [],
            case.runner.kv_cache_config,
            for_capture=True,
        )
    assert case.state.prepare_inputs(case.batch, None) == {
        "producer_mode": CUDAGraphMode.NONE
    }
    assert case.peer.terminal
