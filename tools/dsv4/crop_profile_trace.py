#!/usr/bin/env python3
"""Crop a large CANN trace_view.json without loading it into memory."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import BinaryIO, TextIO

try:
    import ijson
except ImportError:  # pragma: no cover - vLLM installs ijson in production.
    ijson = None

DEFAULT_CHUNK_SIZE_BYTES = 4 * 1024 * 1024
DEFAULT_ANCHOR_REGEX = r"(?i)(hccl|hcom).*(recv|receive)|(recv|receive).*(hccl|hcom)"
DEFAULT_CONTEXT_MS = 25.0
DEFAULT_MAX_WINDOW_MS = 0.0
MICROSECONDS_PER_MILLISECOND = 1000.0


class TraceCropError(ValueError):
    """The trace or requested crop window is invalid."""


@dataclass(frozen=True)
class TraceWindow:
    start_us: float
    end_us: float

    def __post_init__(self) -> None:
        if self.end_us <= self.start_us:
            raise TraceCropError("window end must be greater than window start")


@dataclass(frozen=True)
class AnchorEvent:
    name: str
    category: str
    timestamp_us: float
    duration_us: float
    pid: object
    tid: object


def _read_more(stream: TextIO, buffer: str, chunk_size: int) -> tuple[str, bool]:
    chunk = stream.read(chunk_size)
    return buffer + chunk, not chunk


def _discard_leading_whitespace(buffer: str) -> str:
    return buffer.lstrip()


def _iter_trace_events_stdlib(
    path: Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE_BYTES
) -> Iterator[dict[str, object]]:
    """Standard-library streaming fallback for environments without ijson."""
    if chunk_size <= 0:
        raise TraceCropError("chunk size must be positive")

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        buffer = ""
        eof = False
        array_started = False
        expect_value = True

        while True:
            buffer = _discard_leading_whitespace(buffer)
            if not buffer and not eof:
                buffer, eof = _read_more(stream, buffer, chunk_size)
                continue

            if not array_started:
                if not buffer:
                    raise TraceCropError(f"empty trace: {path}")
                if buffer[0] != "[":
                    raise TraceCropError(
                        f"trace must be a top-level JSON array: {path}"
                    )
                buffer = buffer[1:]
                array_started = True
                continue

            buffer = _discard_leading_whitespace(buffer)
            if not buffer and not eof:
                buffer, eof = _read_more(stream, buffer, chunk_size)
                continue
            if not buffer:
                raise TraceCropError(f"unterminated JSON array: {path}")

            if expect_value:
                if buffer[0] == "]":
                    trailing = buffer[1:].strip()
                    while not eof:
                        trailing, eof = _read_more(stream, trailing, chunk_size)
                    if trailing.strip():
                        raise TraceCropError(
                            f"unexpected data after JSON array: {path}"
                        )
                    return
                try:
                    value, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError as error:
                    if eof:
                        raise TraceCropError(
                            f"invalid JSON event in {path}: {error}"
                        ) from error
                    buffer, eof = _read_more(stream, buffer, chunk_size)
                    continue
                if not isinstance(value, dict):
                    raise TraceCropError(
                        f"trace array entries must be JSON objects: {path}"
                    )
                yield value
                buffer = buffer[end:]
                expect_value = False
                continue

            if buffer[0] == ",":
                buffer = buffer[1:]
                expect_value = True
            elif buffer[0] == "]":
                trailing = buffer[1:].strip()
                while not eof:
                    trailing, eof = _read_more(stream, trailing, chunk_size)
                if trailing.strip():
                    raise TraceCropError(f"unexpected data after JSON array: {path}")
                return
            elif eof:
                raise TraceCropError(f"expected ',' or ']' in {path}")
            else:
                buffer, eof = _read_more(stream, buffer, chunk_size)


def _require_top_level_array(stream: BinaryIO, path: Path) -> None:
    while character := stream.read(1):
        if character.isspace():
            continue
        if character != b"[":
            raise TraceCropError(f"trace must be a top-level JSON array: {path}")
        stream.seek(0)
        return
    raise TraceCropError(f"empty trace: {path}")


def iter_trace_events(
    path: Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE_BYTES
) -> Iterator[dict[str, object]]:
    """Yield trace events with bounded memory, preferring ijson's C backend."""
    if ijson is None:
        yield from _iter_trace_events_stdlib(path, chunk_size=chunk_size)
        return

    with path.open("rb") as stream:
        _require_top_level_array(stream, path)
        for value in ijson.items(stream, "item"):
            if not isinstance(value, dict):
                raise TraceCropError(
                    f"trace array entries must be JSON objects: {path}"
                )
            yield value


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _event_interval(event: dict[str, object]) -> tuple[float, float] | None:
    timestamp = _number(event.get("ts"))
    if timestamp is None:
        return None
    duration = _number(event.get("dur")) or 0.0
    return timestamp, timestamp + max(duration, 0.0)


def _event_overlaps_window(event: dict[str, object], window: TraceWindow) -> bool:
    interval = _event_interval(event)
    if interval is None:
        return False
    start, end = interval
    return start <= window.end_us and end >= window.start_us


def select_anchor_window(
    input_path: Path,
    *,
    anchor_regex: str,
    context_ms: float,
    max_window_ms: float,
) -> tuple[TraceWindow, AnchorEvent, int]:
    """Select a window around the longest matching receive event."""
    if context_ms < 0:
        raise TraceCropError("context must not be negative")
    if max_window_ms < 0:
        raise TraceCropError("maximum window must not be negative")

    pattern = re.compile(anchor_regex)
    selected: AnchorEvent | None = None
    event_count = 0
    for event in iter_trace_events(input_path):
        event_count += 1
        interval = _event_interval(event)
        if interval is None:
            continue
        name = str(event.get("name", ""))
        category = str(event.get("cat", ""))
        if not pattern.search(f"{category} {name}"):
            continue
        timestamp, end = interval
        candidate = AnchorEvent(
            name=name,
            category=category,
            timestamp_us=timestamp,
            duration_us=end - timestamp,
            pid=event.get("pid"),
            tid=event.get("tid"),
        )
        if selected is None or candidate.duration_us > selected.duration_us:
            selected = candidate

    if selected is None:
        raise TraceCropError(
            f"no event matches anchor regex {anchor_regex!r} in {input_path}"
        )

    context_us = context_ms * MICROSECONDS_PER_MILLISECOND
    start_us = selected.timestamp_us - context_us
    end_us = selected.timestamp_us + selected.duration_us + context_us
    max_window_us = max_window_ms * MICROSECONDS_PER_MILLISECOND
    if max_window_us and end_us - start_us > max_window_us:
        # Keep the beginning of a long receive plus its preceding producer work.
        end_us = start_us + max_window_us
    return TraceWindow(start_us=start_us, end_us=end_us), selected, event_count


def load_window(path: Path) -> TraceWindow:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TraceCropError(f"cannot read window file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TraceCropError(f"window file must contain a JSON object: {path}")
    start_us = _number(payload.get("start_us"))
    end_us = _number(payload.get("end_us"))
    if start_us is None or end_us is None:
        raise TraceCropError(f"window file lacks numeric start_us/end_us: {path}")
    return TraceWindow(start_us=start_us, end_us=end_us)


def _open_output(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def crop_trace(
    input_path: Path, output_path: Path, window: TraceWindow
) -> tuple[int, int, int]:
    """Write metadata and all events overlapping the selected time window."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_events = 0
    metadata_events = 0
    window_events = 0
    first = True
    with _open_output(output_path) as output:
        output.write("[")
        for event in iter_trace_events(input_path):
            total_events += 1
            is_metadata = event.get("ph") == "M"
            if not is_metadata and not _event_overlaps_window(event, window):
                continue
            if is_metadata:
                metadata_events += 1
            else:
                window_events += 1
            if not first:
                output.write(",")
            json.dump(
                event,
                output,
                ensure_ascii=True,
                separators=(",", ":"),
                default=_json_default,
            )
            first = False
        output.write("]\n")

    if window_events == 0:
        output_path.unlink(missing_ok=True)
        raise TraceCropError(
            "selected window contains no timestamped events; verify that the "
            "Attention and FFN hosts use synchronized clocks"
        )
    return total_events, metadata_events, window_events


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream-crop trace_view.json around the longest receive event, or "
            "reuse an exact window selected on the paired FFN trace."
        )
    )
    parser.add_argument("input", type=Path, help="source trace_view.json")
    parser.add_argument("output", type=Path, help="output .json or .json.gz")
    parser.add_argument(
        "--window-file",
        type=Path,
        help="reuse start_us/end_us from a paired crop's window JSON",
    )
    parser.add_argument("--start-us", type=float, help="explicit window start")
    parser.add_argument("--end-us", type=float, help="explicit window end")
    parser.add_argument(
        "--anchor-regex",
        default=DEFAULT_ANCHOR_REGEX,
        help="regex used to select the longest receive anchor",
    )
    parser.add_argument(
        "--context-ms",
        type=float,
        default=DEFAULT_CONTEXT_MS,
        help="context retained before and after the receive anchor",
    )
    parser.add_argument(
        "--max-window-ms",
        type=float,
        default=DEFAULT_MAX_WINDOW_MS,
        help="cap auto-selected window length; zero keeps the full receive",
    )
    parser.add_argument(
        "--window-output",
        type=Path,
        help="window metadata path; defaults to OUTPUT.window.json",
    )
    return parser


def _resolve_window(
    arguments: argparse.Namespace,
) -> tuple[TraceWindow, AnchorEvent | None, int | None, str]:
    explicit_count = sum(
        (
            arguments.window_file is not None,
            arguments.start_us is not None or arguments.end_us is not None,
        )
    )
    if explicit_count > 1:
        raise TraceCropError("use only one of --window-file or --start-us/--end-us")
    if arguments.window_file is not None:
        return load_window(arguments.window_file), None, None, "window-file"
    if arguments.start_us is not None or arguments.end_us is not None:
        if arguments.start_us is None or arguments.end_us is None:
            raise TraceCropError("--start-us and --end-us must be used together")
        return (
            TraceWindow(arguments.start_us, arguments.end_us),
            None,
            None,
            "explicit",
        )
    window, anchor, scanned_events = select_anchor_window(
        arguments.input,
        anchor_regex=arguments.anchor_regex,
        context_ms=arguments.context_ms,
        max_window_ms=arguments.max_window_ms,
    )
    return window, anchor, scanned_events, "auto-longest-receive"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if not arguments.input.is_file():
            raise TraceCropError(f"input does not exist: {arguments.input}")
        if arguments.input.resolve() == arguments.output.resolve():
            raise TraceCropError("input and output must be different files")

        window, anchor, scanned_events, selection = _resolve_window(arguments)
        total_events, metadata_events, window_events = crop_trace(
            arguments.input, arguments.output, window
        )
        window_output = arguments.window_output or Path(
            f"{arguments.output}.window.json"
        )
        summary: dict[str, object] = {
            "input": str(arguments.input.resolve()),
            "output": str(arguments.output.resolve()),
            "selection": selection,
            "start_us": window.start_us,
            "end_us": window.end_us,
            "duration_ms": (window.end_us - window.start_us)
            / MICROSECONDS_PER_MILLISECOND,
            "anchor": asdict(anchor) if anchor is not None else None,
            "anchor_scan_event_count": scanned_events,
            "total_event_count": total_events,
            "metadata_event_count": metadata_events,
            "window_event_count": window_events,
            "output_size_bytes": arguments.output.stat().st_size,
        }
        window_output.parent.mkdir(parents=True, exist_ok=True)
        window_output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, TraceCropError, re.error) as error:
        parser.exit(1, f"crop_profile_trace.py: ERROR: {error}\n")

    print(f"TRACE={arguments.output}")
    print(f"WINDOW={window_output}")
    print(f"EVENTS={window_events}")
    print(f"DURATION_MS={summary['duration_ms']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
