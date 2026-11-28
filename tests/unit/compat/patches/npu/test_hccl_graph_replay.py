# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Native replay host contract; NPU replay is an explicit excluded boundary."""

from concurrent.futures import ThreadPoolExecutor
from types import MethodType

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager

from afd_plugin.compat.patches.npu.hccl_graph_replay import use_hccl_graph_replay
from tests.unit.fakes.graph_replay import make_replay_boundary_case


@pytest.fixture
def replay_case():
    def build(*, failure=None):
        case = make_replay_boundary_case(ModelAclGraphManager, failure=failure)
        case.scope = lambda: use_hccl_graph_replay(
            case.manager,
            before_replay=case.notifications.before_replay,
            after_replay=case.notifications.after_replay,
            fail=case.notifications.fail,
        )
        return case

    return build


def test_metadata_handoff_surrounds_native_replay_and_precedes_completion(replay_case):
    case = replay_case()
    original = case.manager.run_fullgraph
    with case.scope():
        result = case.manager.run_fullgraph(case.descriptor)
    torch.testing.assert_close(result, torch.full((4,), case.descriptor.num_tokens))
    assert case.notifications.messages == [
        "accepted metadata",
        "native replay",
        "restored metadata",
        "COMPLETE",
    ]
    assert case.manager.run_fullgraph is original
    assert not case.notifications.terminal


@pytest.mark.parametrize("failure", ["before", "native", "after"])
def test_failed_replay_restores_binding_and_metadata_and_marks_terminal(
    replay_case, failure
):
    case = replay_case(failure=failure)
    original = case.manager.run_fullgraph
    with pytest.raises(RuntimeError, match="failed"), case.scope():
        case.manager.run_fullgraph(case.descriptor)
    assert case.manager.run_fullgraph is original
    assert case.manager.metadata is case.notifications.previous_metadata
    assert case.notifications.terminal
    if failure != "after":
        assert "COMPLETE" not in case.notifications.messages


def test_other_manager_keeps_its_own_native_replay_binding(replay_case):
    case, other = replay_case(), replay_case()
    original = other.manager.run_fullgraph
    with case.scope():
        assert other.manager.run_fullgraph is original
    assert case.notifications.messages == []
    assert not case.notifications.terminal


def test_nested_replay_scope_is_rejected_without_disturbing_outer_binding(replay_case):
    case = replay_case()
    with case.scope():
        with pytest.raises(RuntimeError, match="cannot be nested"), case.scope():
            pass
        result = case.manager.run_fullgraph(case.descriptor)
    assert result.shape == (4,)
    assert not case.notifications.terminal


@pytest.mark.parametrize("boundary", ["repeat", "foreign-thread", "foreign-receiver"])
def test_selected_replay_rejects_another_invocation_thread_or_receiver(
    replay_case, boundary
):
    case = replay_case()
    with case.scope():
        if boundary == "repeat":
            case.manager.run_fullgraph(case.descriptor)
            with pytest.raises(RuntimeError, match="one owning-thread"):
                case.manager.run_fullgraph(case.descriptor)
        elif boundary == "foreign-thread":
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(case.manager.run_fullgraph, case.descriptor)
                with pytest.raises(RuntimeError, match="one owning-thread"):
                    future.result()
        else:
            other = replay_case()
            borrowed = MethodType(case.manager.run_fullgraph.__func__, other.manager)
            with pytest.raises(RuntimeError, match="one owning-thread"):
                borrowed(case.descriptor)
    assert case.notifications.terminal
