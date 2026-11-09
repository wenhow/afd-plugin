# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Pre-prepare CPU dispatch seams; no graph/device execution is simulated."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState

from afd_plugin.compat.patches.npu.hccl_graph_dispatch import (
    use_hccl_graph_dispatch,
)
from tests.unit.fakes.batch_req_state import make_batch_req_state
from tests.unit.fakes.cudagraph_manager import make_native_graph_manager
from tests.unit.fakes.graph_dispatch import NativeGraphDispatchOwner
from tests.unit.fakes.vllm_config import npu_parallel_config


@pytest.fixture
def owner():
    descriptor = BatchExecutionDescriptor(
        CUDAGraphMode.FULL, 8, 8, uniform_token_count=1
    )
    manager = make_native_graph_manager((descriptor,))
    return NativeGraphDispatchOwner(
        (make_batch_req_state((1,) * 4), 1), npu_parallel_config(), manager
    )


@pytest.fixture
def dispatch(owner):
    def dispatch(**overrides):
        kwargs = dict(
            cudagraph_manager=owner.cudagraph_manager,
            num_reqs=4,
            num_tokens=4,
            uniform_token_count=1,
            dp_size=1,
            dp_rank=0,
            parallel_config=owner.parallel_config,
        )
        kwargs.update(overrides)
        return model_runner.dispatch_cg_and_sync_dp(**kwargs)

    return dispatch


def test_selector_uses_real_gather_tuple_before_native_preparation(owner, dispatch):
    def select(state, arguments, native):
        assert state is owner.result[0]
        assert arguments.num_tokens == state.num_tokens
        assert native.cg_mode == CUDAGraphMode.FULL
        assert native.num_ubatches == 1
        return BatchExecutionDescriptor(
            CUDAGraphMode.NONE,
            state.num_tokens,
            len(state.req_ids),
            num_ubatches=2,
        )

    with use_hccl_graph_dispatch(owner, select):
        assert owner.gather_batch_req_state(None, False) is owner.result
        descriptor, sync = dispatch()

    assert descriptor == BatchExecutionDescriptor(
        CUDAGraphMode.NONE, 4, 4, num_ubatches=2
    )
    assert sync is None
    assert owner.gather_batch_req_state(None, False) is owner.result


def test_selector_preserves_native_returned_dp_sync(owner, dispatch, monkeypatch):
    native = BatchExecutionDescriptor(CUDAGraphMode.NONE, 4, 4)
    native_sync = DPSyncState(torch.tensor([4]), 1, True, 4)
    monkeypatch.setattr(
        model_runner,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (native, native_sync),
    )

    with use_hccl_graph_dispatch(owner, lambda state, args, desc: desc):
        owner.gather_batch_req_state(None, False)
        descriptor, sync = dispatch()

    assert descriptor is native
    assert sync is native_sync


@pytest.mark.parametrize("num_tokens,num_reqs", [(0, 0), (8, 8)])
def test_no_state_still_selects_with_native_eager_descriptor(
    owner, dispatch, num_tokens, num_reqs
):
    owner.result = (None, None)
    owner.cudagraph_manager = None

    def select(state, arguments, native):
        assert state is None
        assert arguments.need_eager
        assert native.cg_mode == CUDAGraphMode.NONE
        return replace(native, num_ubatches=2)

    with use_hccl_graph_dispatch(owner, select):
        owner.gather_batch_req_state(None, True)
        descriptor, sync = dispatch(
            num_tokens=num_tokens, num_reqs=num_reqs, uniform_token_count=None
        )

    assert descriptor == BatchExecutionDescriptor(
        CUDAGraphMode.NONE, num_tokens, num_reqs, num_ubatches=2
    )
    assert sync is None


@pytest.mark.parametrize("dummy_run", [False, True])
def test_forced_eager_cannot_be_promoted_to_graph(owner, dispatch, dummy_run):
    if dummy_run:
        owner.result = (None, 1)

    def select(state, arguments, native):
        return replace(native, cg_mode=CUDAGraphMode.FULL)

    with use_hccl_graph_dispatch(owner, select):
        owner.gather_batch_req_state(None, dummy_run)
        with pytest.raises(ValueError, match="forced-eager"):
            dispatch(need_eager=not dummy_run)


def test_native_reused_eager_agreement_cannot_be_promoted_to_graph(owner, dispatch):
    sync = DPSyncState(torch.tensor([4]), 1, True, 4)

    def select(state, arguments, native):
        assert arguments.dp_sync is sync
        assert arguments.need_eager
        assert native.cg_mode == CUDAGraphMode.NONE
        return replace(native, cg_mode=CUDAGraphMode.FULL)

    with use_hccl_graph_dispatch(owner, select):
        owner.gather_batch_req_state(None, False)
        with pytest.raises(ValueError, match="forced-eager"):
            dispatch(dp_sync=sync)


@pytest.mark.parametrize("failure", ["native", "selector"])
def test_exception_restores_native_dispatch_and_selected_gather(
    owner, dispatch, monkeypatch, failure
):
    def fail(*args, **kwargs):
        raise RuntimeError(failure)

    if failure == "native":
        monkeypatch.setattr(model_runner, "dispatch_cg_and_sync_dp", fail)
    original_dispatch = model_runner.dispatch_cg_and_sync_dp
    original_gather = owner.gather_batch_req_state
    # An existing selected-instance override must survive the temporary scope.
    owner.gather_batch_req_state = original_gather
    select = fail if failure == "selector" else lambda state, args, desc: desc

    with (
        pytest.raises(RuntimeError, match=failure),
        use_hccl_graph_dispatch(owner, select),
    ):
        owner.gather_batch_req_state(None, False)
        dispatch()

    assert model_runner.dispatch_cg_and_sync_dp is original_dispatch
    assert owner.gather_batch_req_state is original_gather
    assert owner.gather_batch_req_state(None, False) is owner.result


@pytest.mark.parametrize("same_owner", [False, True])
def test_nested_scope_is_rejected_without_replacing_outer_binding(
    owner, dispatch, same_owner
):
    other = (
        owner
        if same_owner
        else NativeGraphDispatchOwner(owner.result, npu_parallel_config())
    )
    with use_hccl_graph_dispatch(owner, lambda state, args, desc: desc):
        with (
            pytest.raises(RuntimeError, match="cannot be nested"),
            use_hccl_graph_dispatch(other, lambda state, args, desc: desc),
        ):
            pytest.fail("nested scope entered")
        owner.gather_batch_req_state(None, False)
        descriptor, _ = dispatch()

    assert descriptor.cg_mode == CUDAGraphMode.FULL


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"dp_size": 2}, "DP1"),
        ({"dp_rank": 1}, "DP1"),
        ({"num_reqs": 5}, "gathered batch"),
        ({"num_tokens": 5}, "gathered batch"),
        ({"uniform_token_count": None}, "gathered batch"),
        ({"cudagraph_manager": None}, "this runner"),
        ({"parallel_config": npu_parallel_config()}, "another runner"),
    ],
)
def test_dispatch_rejects_wrong_runner_or_batch_before_selector(
    owner, dispatch, overrides, match
):
    def select(state, args, desc):
        pytest.fail("invalid ownership reached selector")

    with use_hccl_graph_dispatch(owner, select):
        owner.gather_batch_req_state(None, False)
        with pytest.raises(ValueError, match=match):
            dispatch(**overrides)


def test_dispatch_requires_gather_first(owner, dispatch):
    with (
        use_hccl_graph_dispatch(owner, lambda state, args, desc: desc),
        pytest.raises(RuntimeError, match="owning batch gather"),
    ):
        dispatch()


@pytest.mark.parametrize("repeat", ["gather", "dispatch"])
def test_scope_rejects_multiple_native_batches(owner, dispatch, repeat):
    with use_hccl_graph_dispatch(owner, lambda state, args, desc: desc):
        owner.gather_batch_req_state(None, False)
        dispatch()
        with pytest.raises(RuntimeError, match="one"):
            if repeat == "gather":
                owner.gather_batch_req_state(None, False)
            else:
                dispatch()


def test_dispatch_on_another_execute_thread_is_rejected(owner, dispatch):
    with use_hccl_graph_dispatch(owner, lambda state, args, desc: desc):
        owner.gather_batch_req_state(None, False)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(dispatch)
            with pytest.raises(RuntimeError, match="one execute thread"):
                future.result()


def test_selector_must_return_a_native_descriptor(owner, dispatch):
    with use_hccl_graph_dispatch(owner, lambda state, args, desc: None):
        owner.gather_batch_req_state(None, False)
        with pytest.raises(ValueError, match="native execution descriptor"):
            dispatch()
