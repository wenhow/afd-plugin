# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU role/phase/session contracts, without an NPU runtime."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from tests.unit.fakes.hccl_v41_eager import (  # noqa: E402
    accept_session_parent,
    make_eager_session_case,
)


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("enabled", [False, True])
def test_eager_session_warms_only_enabled_family_and_opens_target_u2(role, enabled):
    case = make_eager_session_case(role=role, enabled=enabled)
    case.session.initialize()
    accept_session_parent(case)

    owner = case.session.open_parent()

    assert case.session.initialized
    assert (owner is not None) == enabled
    assert [kind for kind, _ in case.transport.messages] == (
        ["warmup", "open"] if enabled else []
    )
    if enabled:
        assert owner.transport is case.transport and owner.plan is case.transport.plan
        assert case.transport.messages[-1][1] == {
            key: value for key, value in case.streams.items() if key != "compute_stream"
        }


@pytest.mark.parametrize(
    "mode,stages",
    [
        ("target", 1),
        ("profile", 2),
        ("warmup", 2),
        ("dummy", 2),
    ],
)
def test_native_u1_and_preparation_parents_preserve_serial_execution(mode, stages):
    case = make_eager_session_case()
    case.session.initialize()
    accept_session_parent(case, mode=mode, stages=stages)

    assert case.session.open_parent() is None
    assert [kind for kind, _ in case.transport.messages] == ["warmup"]


@pytest.mark.parametrize("invalid", ["uninitialized", "closed", "terminal", "no_plan"])
def test_eager_selection_rejects_unusable_session_before_data_posting(invalid):
    case = make_eager_session_case()
    if invalid != "uninitialized":
        case.session.initialize()
    if invalid != "no_plan":
        accept_session_parent(case)
    if invalid in ("closed", "terminal"):
        setattr(case.transport, invalid, True)

    with pytest.raises(RuntimeError, match="initialized accepted parent"):
        case.session.open_parent()

    assert "open" not in [kind for kind, _ in case.transport.messages]


def test_eager_session_initialization_requires_an_idle_transport_and_runs_once():
    case = make_eager_session_case()
    accept_session_parent(case)
    with pytest.raises(RuntimeError, match="idle live transport"):
        case.session.initialize()
    assert case.transport.messages == []
    case.transport.plan = None
    case.session.initialize()
    with pytest.raises(RuntimeError, match="idle live transport"):
        case.session.initialize()
