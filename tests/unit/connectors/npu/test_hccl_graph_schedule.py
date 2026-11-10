# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Static paired actions only; no dependency on runtime/device packages."""

import importlib.util
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


@pytest.fixture
def schedule_module(monkeypatch):
    name = "_afd_unit_hccl_graph_schedule"
    path = (
        Path(__file__).resolve().parents[4]
        / "afd_plugin/connectors/npu/hccl_graph_schedule.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_three_layer_common_actions_preserve_both_native_compute_orders(
    schedule_module,
):
    schedule = schedule_module.HcclGraphSchedule((0, 1, 2), (2, 3))
    assert [(a.kind, a.layer_idx, a.stage_idx) for a in schedule.actions] == [
        ("input", 0, 0),
        ("input", 0, 1),
        ("output", 0, 0),
        ("input", 1, 0),
        ("output", 0, 1),
        ("input", 1, 1),
        ("output", 1, 0),
        ("input", 2, 0),
        ("output", 1, 1),
        ("input", 2, 1),
        ("output", 2, 0),
        ("output", 2, 1),
    ]
    assert schedule.compute_steps == ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))


@pytest.mark.parametrize("rows,positive", [((0, 3), 1), ((3, 0), 0)])
def test_one_positive_stage_degenerates_to_matched_serial_actions(
    schedule_module, rows, positive
):
    schedule = schedule_module.HcclGraphSchedule((2, 5, 8), rows)
    assert [(a.kind, a.layer_idx, a.stage_idx) for a in schedule.actions] == [
        (kind, layer, positive) for layer in (2, 5, 8) for kind in ("input", "output")
    ]
    assert schedule.stage_rows == rows


def test_all_static_zero_geometry_has_no_wire_or_compute(schedule_module):
    schedule = schedule_module.HcclGraphSchedule((0, 1), (0, 0))
    assert schedule.actions == schedule.compute_steps == ()


@pytest.mark.parametrize(
    "layers,rows",
    [
        ([], (2, 3)),
        ((), (2, 3)),
        ((1, 0), (2, 3)),
        ((0, 0), (2, 3)),
        ((True,), (2, 3)),
        ((-1,), (2, 3)),
        ((0,), [2, 3]),
        ((0,), (5,)),
        ((0,), (2, 3, 4)),
        ((0,), (-1, 3)),
        ((0,), (True, 3)),
    ],
)
def test_invalid_geometry_is_rejected_before_any_action(schedule_module, layers, rows):
    with pytest.raises(ValueError, match="ordered layers and U2 rows"):
        schedule_module.HcclGraphSchedule(layers, rows)


@pytest.mark.parametrize("field,value", [("stage_rows", (3, 2)), ("actions", ())])
def test_schedule_is_immutable_for_the_session(schedule_module, field, value):
    schedule = schedule_module.HcclGraphSchedule((0, 1), (2, 3))
    with pytest.raises(FrozenInstanceError):
        setattr(schedule, field, value)


def test_action_coordinates_are_immutable(schedule_module):
    action = schedule_module.HcclGraphSchedule((0,), (2, 3)).actions[0]
    with pytest.raises(FrozenInstanceError):
        action.stage_idx = 1
