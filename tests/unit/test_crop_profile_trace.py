from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tools.dsv4 import crop_profile_trace


def _write_trace(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(events), encoding="utf-8")


def _read_trace(path: Path) -> list[dict[str, object]]:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def test_streams_events_across_small_input_chunks(tmp_path):
    source = tmp_path / "trace_view.json"
    events = [
        {"name": "process_name", "ph": "M", "pid": 1},
        {"name": "MatMul", "ph": "X", "ts": "100.25", "dur": 2.5},
    ]
    _write_trace(source, events)

    assert (
        list(crop_profile_trace._iter_trace_events_stdlib(source, chunk_size=7))
        == events
    )


def test_auto_crop_uses_longest_receive_and_preserves_metadata(tmp_path):
    source = tmp_path / "trace_view.json"
    output = tmp_path / "ffn-comm-trace.json.gz"
    _write_trace(
        source,
        [
            {"name": "thread_name", "ph": "M", "pid": 8, "tid": 3},
            {"name": "old", "ph": "X", "ts": 10, "dur": 1},
            {
                "name": "HcclReceive",
                "cat": "cpu_op",
                "ph": "X",
                "ts": 100,
                "dur": 5,
                "pid": 8,
                "tid": 3,
            },
            {
                "name": "hcom_receive__kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 200,
                "dur": 40,
                "pid": 88,
                "tid": 9,
            },
            {"name": "MatMul", "ph": "X", "ts": 245, "dur": 10},
            {"name": "future", "ph": "X", "ts": 400, "dur": 1},
        ],
    )

    window, anchor, scanned = crop_profile_trace.select_anchor_window(
        source,
        anchor_regex=crop_profile_trace.DEFAULT_ANCHOR_REGEX,
        context_ms=0.01,
        max_window_ms=1,
    )
    counts = crop_profile_trace.crop_trace(source, output, window)

    assert anchor.name == "hcom_receive__kernel"
    assert scanned == 6
    assert window == crop_profile_trace.TraceWindow(190, 250)
    assert counts == (6, 1, 2)
    assert [event["name"] for event in _read_trace(output)] == [
        "thread_name",
        "hcom_receive__kernel",
        "MatMul",
    ]


def test_reuses_exact_window_from_paired_trace(tmp_path):
    source = tmp_path / "trace_view.json"
    output = tmp_path / "attention-comm-trace.json"
    window_file = tmp_path / "ffn.window.json"
    _write_trace(
        source,
        [
            {"name": "before", "ph": "X", "ts": 999, "dur": 0.5},
            {"name": "send", "ph": "X", "ts": 1005, "dur": 2},
            {"name": "compute", "ph": "X", "ts": 1010, "dur": 20},
            {"name": "after", "ph": "X", "ts": 1101, "dur": 1},
        ],
    )
    window_file.write_text(
        json.dumps({"start_us": 1000, "end_us": 1100}), encoding="utf-8"
    )

    window = crop_profile_trace.load_window(window_file)
    crop_profile_trace.crop_trace(source, output, window)

    assert [event["name"] for event in _read_trace(output)] == ["send", "compute"]


def test_rejects_a_non_array_trace(tmp_path):
    source = tmp_path / "trace_view.json"
    source.write_text('{"traceEvents": []}', encoding="utf-8")

    with pytest.raises(
        crop_profile_trace.TraceCropError,
        match="top-level JSON array",
    ):
        list(crop_profile_trace.iter_trace_events(source, chunk_size=4))
