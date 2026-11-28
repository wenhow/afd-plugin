# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Capture-factory host seam; graph capture is an explicit CPU boundary."""

from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

from afd_plugin.compat.patches.npu.hccl_graph_capture import use_hccl_graph_capture
from tests.unit.fakes.graph_capture import (
    CaptureNotifications,
    cpu_capture_boundary,
    make_capture_boundary_case,
)


@pytest.fixture
def capture_case(monkeypatch):
    monkeypatch.setattr(CudaGraphManager, "capture", cpu_capture_boundary)

    def build(*, failure=None):
        case = make_capture_boundary_case(CudaGraphManager, failure=failure)
        case.notifications = CaptureNotifications(failure=failure)
        case.scope = lambda: use_hccl_graph_capture(
            case.manager,
            before_factory=case.notifications.before_factory,
            after_capture=case.notifications.after_capture,
            fail=case.notifications.fail,
        )
        return case

    return build


def test_reservation_precedes_each_native_factory_and_flush_follows_graph_storage(
    capture_case,
):
    case = capture_case()
    notifications = case.notifications

    def final_flush():
        assert case.descriptor in case.manager.graphs
        notifications.after_capture()

    original = CudaGraphManager.capture
    with use_hccl_graph_capture(
        case.manager,
        before_factory=notifications.before_factory,
        after_capture=final_flush,
        fail=notifications.fail,
    ):
        case.manager.capture(notifications.create_forward_fn)
    assert notifications.messages == [
        ("reserve", case.descriptor, True),
        ("prepare", case.descriptor, True),
        ("forward", case.descriptor, True, CUDAGraphMode.NONE),
        ("reserve", case.descriptor, False),
        ("prepare", case.descriptor, False),
        ("forward", case.descriptor, False, CUDAGraphMode.NONE),
        ("flush",),
    ]
    assert not notifications.terminal
    assert CudaGraphManager.capture is original


def test_another_manager_keeps_native_capture_without_phase_notifications(capture_case):
    case, other = capture_case(), capture_case()
    with case.scope():
        other.manager.capture(other.notifications.create_forward_fn)
    assert case.notifications.messages == []
    assert not case.notifications.terminal
    assert other.descriptor in other.manager.graphs
    assert [message[0] for message in other.notifications.messages] == [
        "prepare",
        "forward",
        "prepare",
        "forward",
    ]


@pytest.mark.parametrize(
    "failure", ["reserve", "prepare", "forward", "native", "flush"]
)
def test_failed_capture_phase_marks_terminal_and_restores_native_binding(
    capture_case, failure
):
    case = capture_case(failure=failure)
    original = CudaGraphManager.capture
    with pytest.raises(RuntimeError, match="failed"), case.scope():
        case.manager.capture(case.notifications.create_forward_fn)
    assert case.notifications.terminal
    assert CudaGraphManager.capture is original
    if failure != "flush":
        assert ("flush",) not in case.notifications.messages


@pytest.mark.parametrize("same_manager", [True, False])
def test_nested_capture_scope_is_rejected_without_disturbing_outer_binding(
    capture_case, same_manager
):
    case = capture_case()
    other = case if same_manager else capture_case()
    original = CudaGraphManager.capture
    with case.scope():
        with pytest.raises(RuntimeError, match="cannot be nested"), other.scope():
            pass
        case.manager.capture(case.notifications.create_forward_fn)
    assert case.descriptor in case.manager.graphs
    assert CudaGraphManager.capture is original


@pytest.mark.parametrize("boundary", ["repeat", "foreign-thread"])
def test_selected_capture_requires_one_invocation_on_its_owning_thread(
    capture_case, boundary
):
    case = capture_case()
    with case.scope():
        if boundary == "repeat":
            case.manager.capture(case.notifications.create_forward_fn)
            with pytest.raises(RuntimeError, match="one owning-thread capture"):
                case.manager.capture(case.notifications.create_forward_fn)
        else:
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(
                    case.manager.capture, case.notifications.create_forward_fn
                )
                with pytest.raises(RuntimeError, match="one owning-thread capture"):
                    result.result()
    assert case.notifications.terminal
