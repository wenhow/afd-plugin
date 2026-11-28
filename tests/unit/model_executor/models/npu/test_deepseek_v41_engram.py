# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Parent producer/input contracts only; no NPU lookup or graph acceptance."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode  # noqa: E402

from afd_plugin.model_executor.models.npu.deepseek_v41_engram import (  # noqa: E402
    AFDDeepseekV41EngramModelState,
)
from tests.unit.fakes.engram import make_external_engram_state  # noqa: E402


def test_external_parent_joins_lookup_before_publishing_event_free_rows(monkeypatch):
    case = make_external_engram_state(AFDDeepseekV41EngramModelState)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: case.stream)

    inputs = case.state.prepare_engram()

    assert set(inputs) == {"engram_lookups", "engram_mask"}
    assert inputs["engram_lookups"][1] is case.producer.lookups[1]
    assert case.stream.messages == [
        ("produce", CUDAGraphMode.NONE),
        ("wait", "mask-ready"),
        ("wait", "layer-ready"),
    ]
    assert case.state.prepare_engram() == {}


@pytest.mark.parametrize("rows", [0, 3, 8])
def test_dummy_capture_has_empty_history_and_stable_rows_without_events(rows):
    case = make_external_engram_state(AFDDeepseekV41EngramModelState)

    inputs = case.state.prepare_engram_dummy_inputs(2, rows)

    assert set(inputs) == {"lookback_token_ids", "engram_lookups", "engram_mask"}
    assert inputs["lookback_token_ids"].tolist() == [[-1] * 3, [-1] * 3]
    assert inputs["engram_mask"][:rows].tolist() == [False] * rows
    assert inputs["engram_lookups"][1] is case.producer.lookups[1]
    assert case.stream.messages == []


def test_dummy_capture_rejects_capacity_overflow():
    case = make_external_engram_state(AFDDeepseekV41EngramModelState)
    with pytest.raises(ValueError, match="capacity"):
        case.state.prepare_engram_dummy_inputs(2, 9)
