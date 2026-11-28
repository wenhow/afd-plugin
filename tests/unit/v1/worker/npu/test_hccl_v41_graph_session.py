# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Persistent native phase binding on CPU, without graph/device execution."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    create_forward_context,
    get_forward_context,
    override_forward_context,
)
from vllm.v1.worker.gpu.ubatch_utils import UBatchState

from afd_plugin.v1.worker.npu.hccl_v41_graph_session import HcclV41GraphSession
from tests.unit.fakes.hccl_v41_ubatch import GraphParentModel, GraphUBatchOwner
from tests.unit.fakes.vllm_config import npu_vllm_config


@pytest.fixture
def session_case():
    config = npu_vllm_config()
    config.compilation_config.static_forward_context = {}
    parent = create_forward_context(
        {}, config, cudagraph_runtime_mode=CUDAGraphMode.FULL
    )
    transport = SimpleNamespace(
        role="attention",
        initialized=True,
        closed=False,
        terminal=False,
        plan=None,
        graph_reservation=None,
    )
    return SimpleNamespace(
        session=HcclV41GraphSession(transport),
        transport=transport,
        owner=GraphUBatchOwner(transport, parent),
        batch=SimpleNamespace(num_tokens_after_padding=4, num_reqs_after_padding=4),
        prepared=UBatchState([], [parent]),
        outer=create_forward_context({}, config),
    )


def test_one_binding_can_arm_successive_completed_parent_phases(session_case):
    case = session_case
    for _ in range(2):
        case.session.arm(case.owner)
        case.session.require_full(case.batch, for_capture=True)
        case.session.prepared(case.batch, case.prepared, for_capture=True)
        with override_forward_context(case.outer):
            output = case.session.forward_parent(
                GraphParentModel(), {"input_ids": torch.arange(4)}, case.prepared
            )
            assert get_forward_context() is case.outer
        torch.testing.assert_close(output, torch.arange(4) + 1)
        case.session.clear()
    assert case.session.transport is case.transport
    assert not case.transport.terminal


def test_transport_binding_is_read_only(session_case):
    with pytest.raises(AttributeError):
        session_case.session.transport = SimpleNamespace()
    assert session_case.session.transport is session_case.transport


@pytest.mark.parametrize(
    "boundary",
    ["foreign", "armed", "closed", "terminal", "inactive", "self", "binding"],
)
def test_arm_rejects_an_inactive_or_different_transport_owner(session_case, boundary):
    case = session_case
    if boundary == "foreign":
        case.owner.transport = SimpleNamespace()
    elif boundary == "self":
        case.owner = case.session
    elif boundary == "binding":
        case.owner = HcclV41GraphSession(case.transport)
    elif boundary == "armed":
        case.session.arm(case.owner)
    elif boundary == "inactive":
        case.transport.initialized = False
    else:
        setattr(case.transport, boundary, True)
    with pytest.raises(ValueError, match="one active owning phase"):
        case.session.arm(case.owner)
    assert case.transport.terminal


@pytest.mark.parametrize("boundary", ["unarmed", "plan", "reservation", "terminal"])
def test_clear_requires_the_accepted_parent_to_be_retired(session_case, boundary):
    case = session_case
    if boundary != "unarmed":
        case.session.arm(case.owner)
    if boundary == "plan":
        case.transport.plan = SimpleNamespace()
    elif boundary == "reservation":
        case.transport.graph_reservation = SimpleNamespace()
    elif boundary == "terminal":
        case.transport.terminal = True
    with pytest.raises(ValueError, match="before clearing"):
        case.session.clear()
    assert case.transport.terminal


@pytest.mark.parametrize("entry", ["require", "prepared", "forward"])
def test_unarmed_parent_callback_is_terminal(session_case, entry):
    case = session_case
    with pytest.raises(ValueError, match="armed phase owner"):
        if entry == "require":
            case.session.require_full(case.batch, for_capture=True)
        elif entry == "prepared":
            case.session.prepared(case.batch, case.prepared, for_capture=True)
        else:
            case.session.forward_parent(GraphParentModel(), {}, case.prepared)
    assert case.transport.terminal


@pytest.mark.parametrize("entry", ["require", "prepared", "forward"])
def test_failed_phase_callback_is_terminal_and_restores_native_context(
    session_case, entry
):
    case = session_case
    case.owner.reject = "before" if entry == "require" else "after"
    case.session.arm(case.owner)
    with override_forward_context(case.outer):
        with pytest.raises(RuntimeError):
            if entry == "require":
                case.session.require_full(case.batch, for_capture=True)
            elif entry == "prepared":
                case.session.prepared(case.batch, case.prepared, for_capture=True)
            else:
                case.session.forward_parent(
                    GraphParentModel(fail=True),
                    {"input_ids": torch.arange(4)},
                    case.prepared,
                )
        assert get_forward_context() is case.outer
    assert case.transport.terminal
