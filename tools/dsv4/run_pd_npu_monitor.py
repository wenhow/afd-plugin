#!/usr/bin/env python3
"""Collect bounded NPU utilization snapshots for a PD performance window."""

from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType

DEFAULT_INTERVAL_SECONDS = 5.0
MAX_ALLOWED_SAMPLES = 7200
NPU_SMI_TIMEOUT_SECONDS = 30.0
SNAPSHOTS_FILENAME = "npu_samples.jsonl"
SUMMARY_FILENAME = "summary.json"

_DEVICE_HEADER_PATTERN = re.compile(r"^(?P<npu_id>\d+)\s+(?P<name>.+)$")
_DEVICE_ID_PATTERN = re.compile(r"^(?P<chip_id>\d+)\s+(?P<phy_id>\d+)$")
_USAGE_PATTERN = re.compile(
    r"^(?P<aicore>[\d.]+|-)\s+"
    r"(?P<memory_used>[\d.]+|-)\s*/\s*(?P<memory_total>[\d.]+|-)\s+"
    r"(?P<hbm_used>[\d.]+|-)\s*/\s*(?P<hbm_total>[\d.]+|-)$"
)


@dataclass(frozen=True)
class DeviceReading:
    npu_id: int
    chip_id: int
    phy_id: int
    aicore_percent: float
    hbm_used_mb: float
    hbm_total_mb: float

    def as_json(self) -> dict[str, int | float]:
        return {
            # This is the machine-level logical ID, before any visibility remap.
            "logical_device_id": self.phy_id,
            "phy_id": self.phy_id,
            "npu_id": self.npu_id,
            "chip_id": self.chip_id,
            "aicore_percent": self.aicore_percent,
            "hbm_used_mb": self.hbm_used_mb,
            "hbm_total_mb": self.hbm_total_mb,
        }


@dataclass
class DeviceStatistics:
    npu_id: int
    chip_id: int
    phy_id: int
    samples: int = 0
    aicore_sum: float = 0.0
    aicore_peak: float = 0.0
    hbm_used_peak: float = 0.0
    hbm_total_peak: float = 0.0

    def add(self, reading: DeviceReading) -> None:
        self.samples += 1
        self.aicore_sum += reading.aicore_percent
        self.aicore_peak = max(self.aicore_peak, reading.aicore_percent)
        self.hbm_used_peak = max(self.hbm_used_peak, reading.hbm_used_mb)
        self.hbm_total_peak = max(self.hbm_total_peak, reading.hbm_total_mb)

    def as_json(self) -> dict[str, int | float]:
        peak_percent = (
            self.hbm_used_peak / self.hbm_total_peak * 100.0
            if self.hbm_total_peak
            else 0.0
        )
        return {
            "logical_device_id": self.phy_id,
            "phy_id": self.phy_id,
            "npu_id": self.npu_id,
            "chip_id": self.chip_id,
            "samples": self.samples,
            "aicore_average_percent": self.aicore_sum / self.samples,
            "aicore_peak_percent": self.aicore_peak,
            "hbm_peak_used_mb": self.hbm_used_peak,
            "hbm_reported_total_mb": self.hbm_total_peak,
            "hbm_peak_used_percent": peak_percent,
        }


class StatisticsAccumulator:
    def __init__(self) -> None:
        self.devices: dict[int, DeviceStatistics] = {}
        self.successful_samples = 0
        self.failed_samples = 0
        self.device_observations = 0
        self.aicore_sum = 0.0
        self.aicore_sample_average_peak = 0.0
        self.aicore_single_device_peak = 0.0
        self.aggregate_hbm_peak_used_mb = 0.0
        self.aggregate_hbm_total_at_peak_mb = 0.0
        self.single_device_hbm_peak_used_mb = 0.0

    def add_success(self, readings: list[DeviceReading]) -> None:
        if not readings:
            raise ValueError("a successful sample must contain at least one device")
        self.successful_samples += 1
        aggregate_hbm_used = sum(reading.hbm_used_mb for reading in readings)
        if (
            self.successful_samples == 1
            or aggregate_hbm_used > self.aggregate_hbm_peak_used_mb
        ):
            self.aggregate_hbm_peak_used_mb = aggregate_hbm_used
            self.aggregate_hbm_total_at_peak_mb = sum(
                reading.hbm_total_mb for reading in readings
            )
        sample_aicore_average = sum(
            reading.aicore_percent for reading in readings
        ) / len(readings)
        self.aicore_sample_average_peak = max(
            self.aicore_sample_average_peak, sample_aicore_average
        )

        for reading in readings:
            statistics = self.devices.setdefault(
                reading.phy_id,
                DeviceStatistics(
                    npu_id=reading.npu_id,
                    chip_id=reading.chip_id,
                    phy_id=reading.phy_id,
                ),
            )
            statistics.add(reading)
            self.device_observations += 1
            self.aicore_sum += reading.aicore_percent
            self.aicore_single_device_peak = max(
                self.aicore_single_device_peak, reading.aicore_percent
            )
            self.single_device_hbm_peak_used_mb = max(
                self.single_device_hbm_peak_used_mb, reading.hbm_used_mb
            )

    def add_failure(self) -> None:
        self.failed_samples += 1

    def as_json(self) -> dict[str, object]:
        aggregate_hbm_percent = (
            self.aggregate_hbm_peak_used_mb
            / self.aggregate_hbm_total_at_peak_mb
            * 100.0
            if self.aggregate_hbm_total_at_peak_mb
            else 0.0
        )
        return {
            "samples": {
                "written": self.successful_samples + self.failed_samples,
                "successful": self.successful_samples,
                "failed": self.failed_samples,
                "device_observations": self.device_observations,
            },
            "overall": {
                "aicore_average_percent": (
                    self.aicore_sum / self.device_observations
                    if self.device_observations
                    else 0.0
                ),
                "aicore_peak_percent": self.aicore_sample_average_peak,
                "aicore_peak_single_device_percent": (self.aicore_single_device_peak),
                "hbm_peak_aggregate_used_mb": self.aggregate_hbm_peak_used_mb,
                "hbm_aggregate_total_at_peak_mb": (self.aggregate_hbm_total_at_peak_mb),
                "hbm_peak_aggregate_used_percent": aggregate_hbm_percent,
                "hbm_peak_single_device_used_mb": (self.single_device_hbm_peak_used_mb),
            },
            "devices": [
                self.devices[phy_id].as_json() for phy_id in sorted(self.devices)
            ],
        }


class StopController:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.reason = "max_samples"

    def request(self, signum: int, _frame: FrameType | None = None) -> None:
        try:
            self.reason = signal.Signals(signum).name.lower()
        except ValueError:
            self.reason = f"signal_{signum}"
        self.event.set()


def _number(value: str, label: str) -> float:
    if value == "-":
        raise ValueError(f"npu-smi reported unavailable {label}")
    return float(value)


def parse_npu_smi_info(output: str) -> list[DeviceReading]:
    """Parse the two-line device records emitted by ``npu-smi info``."""
    readings: list[DeviceReading] = []
    current_npu_id: int | None = None
    seen_phy_ids: set[int] = set()

    for raw_line in output.splitlines():
        if not raw_line.startswith("|"):
            continue
        columns = [column.strip() for column in raw_line.split("|")[1:-1]]
        if len(columns) != 3:
            continue

        header_match = _DEVICE_HEADER_PATTERN.fullmatch(columns[0])
        if header_match and not header_match.group("name").isdigit():
            current_npu_id = int(header_match.group("npu_id"))
            continue

        device_match = _DEVICE_ID_PATTERN.fullmatch(columns[0])
        usage_match = _USAGE_PATTERN.fullmatch(columns[2])
        if current_npu_id is None or device_match is None or usage_match is None:
            continue

        phy_id = int(device_match.group("phy_id"))
        if phy_id in seen_phy_ids:
            raise ValueError(f"duplicate Phy-ID in npu-smi output: {phy_id}")
        seen_phy_ids.add(phy_id)
        readings.append(
            DeviceReading(
                npu_id=current_npu_id,
                chip_id=int(device_match.group("chip_id")),
                phy_id=phy_id,
                aicore_percent=_number(
                    usage_match.group("aicore"), "AICore utilization"
                ),
                hbm_used_mb=_number(usage_match.group("hbm_used"), "HBM usage"),
                hbm_total_mb=_number(usage_match.group("hbm_total"), "HBM capacity"),
            )
        )

    if not readings:
        raise ValueError("no NPU device readings found in npu-smi output")
    return sorted(readings, key=lambda reading: reading.phy_id)


def _sample_npus() -> list[DeviceReading]:
    try:
        completed = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=NPU_SMI_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"npu-smi invocation failed: {error}") from error
    if completed.returncode != 0:
        details = completed.stderr.strip()[-500:]
        raise RuntimeError(f"npu-smi exited with {completed.returncode}: {details}")
    return parse_npu_smi_info(completed.stdout)


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _select_devices(
    readings: list[DeviceReading], requested_devices: tuple[int, ...] | None
) -> list[DeviceReading]:
    if requested_devices is None:
        return readings
    by_phy_id = {reading.phy_id: reading for reading in readings}
    missing = [phy_id for phy_id in requested_devices if phy_id not in by_phy_id]
    if missing:
        raise ValueError(
            "npu-smi sample is missing requested Phy-ID(s): "
            + ",".join(str(phy_id) for phy_id in missing)
        )
    return [by_phy_id[phy_id] for phy_id in requested_devices]


def run_monitor(
    output_dir: Path,
    interval: float,
    max_samples: int,
    stop: StopController,
    *,
    requested_devices: tuple[int, ...] | None = None,
    sample_npus: Callable[[], list[DeviceReading]] = _sample_npus,
) -> dict[str, object]:
    if interval <= 0:
        raise ValueError("interval must be positive")
    if not 1 <= max_samples <= MAX_ALLOWED_SAMPLES:
        raise ValueError(f"max_samples must be between 1 and {MAX_ALLOWED_SAMPLES}")
    if requested_devices is not None:
        if not requested_devices:
            raise ValueError("requested_devices must not be empty")
        if any(device < 0 for device in requested_devices):
            raise ValueError("requested_devices must be non-negative")
        if len(set(requested_devices)) != len(requested_devices):
            raise ValueError("requested_devices must not contain duplicates")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_path = output_dir / SNAPSHOTS_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    if snapshots_path.exists() or summary_path.exists():
        raise FileExistsError(f"refusing to overwrite monitor output in {output_dir}")

    started_at = _timestamp()
    started_monotonic = time.monotonic()
    statistics = StatisticsAccumulator()

    with snapshots_path.open("x", encoding="utf-8", buffering=1) as snapshots:
        for sample_index in range(max_samples):
            if stop.event.is_set():
                break
            elapsed_seconds = time.monotonic() - started_monotonic
            snapshot: dict[str, object] = {
                "sample_index": sample_index,
                "timestamp": _timestamp(),
                "elapsed_seconds": elapsed_seconds,
            }
            try:
                readings = _select_devices(sample_npus(), requested_devices)
                snapshot["devices"] = [reading.as_json() for reading in readings]
                statistics.add_success(readings)
            except (OSError, RuntimeError, ValueError) as error:
                snapshot["devices"] = []
                snapshot["error"] = f"{type(error).__name__}: {error}"
                statistics.add_failure()
            snapshots.write(
                json.dumps(snapshot, ensure_ascii=True, separators=(",", ":")) + "\n"
            )

            if sample_index + 1 == max_samples:
                break
            # Schedule from the end of the previous sample.  A slow or timed-out
            # npu-smi call must not trigger a burst of catch-up samples during
            # the benchmark window.
            if stop.event.wait(interval):
                break

    finished_monotonic = time.monotonic()
    summary: dict[str, object] = {
        "schema_version": 1,
        "passed": (
            statistics.successful_samples > 0 and statistics.failed_samples == 0
        ),
        "command": ["npu-smi", "info"],
        "identity": (
            "logical_device_id is npu-smi Phy-ID before "
            "ASCEND_RT_VISIBLE_DEVICES remapping"
        ),
        "started_at": started_at,
        "finished_at": _timestamp(),
        "duration_seconds": finished_monotonic - started_monotonic,
        "configured_interval_seconds": interval,
        "configured_max_samples": max_samples,
        "requested_devices": (
            list(requested_devices) if requested_devices is not None else None
        ),
        "stop_reason": stop.reason,
        "snapshots_path": snapshots_path.name,
        **statistics.as_json(),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _positive_interval(value: str) -> float:
    interval = float(value)
    if interval <= 0:
        raise argparse.ArgumentTypeError("interval must be positive")
    return interval


def _bounded_samples(value: str) -> int:
    max_samples = int(value)
    if not 1 <= max_samples <= MAX_ALLOWED_SAMPLES:
        raise argparse.ArgumentTypeError(
            f"max-samples must be between 1 and {MAX_ALLOWED_SAMPLES}"
        )
    return max_samples


def _parse_devices(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "devices must be a comma-separated list of Phy-IDs"
        )
    try:
        devices = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "devices must be a comma-separated list of integer Phy-IDs"
        ) from error
    if any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError("device Phy-IDs must be non-negative")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("device Phy-IDs must not contain duplicates")
    return devices


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect compact npu-smi samples for a PD benchmark window."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument(
        "--interval", type=_positive_interval, default=DEFAULT_INTERVAL_SECONDS
    )
    run_parser.add_argument(
        "--max-samples", type=_bounded_samples, default=MAX_ALLOWED_SAMPLES
    )
    run_parser.add_argument(
        "--devices",
        type=_parse_devices,
        help="comma-separated logical device/Phy-IDs to include",
    )
    preflight_parser = subparsers.add_parser(
        "preflight", help="take one sample and validate the requested devices"
    )
    preflight_parser.add_argument(
        "--devices",
        type=_parse_devices,
        required=True,
        help="comma-separated logical device/Phy-IDs to validate",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "preflight":
        try:
            readings = _select_devices(_sample_npus(), args.devices)
        except (OSError, RuntimeError, ValueError) as error:
            print(
                json.dumps(
                    {
                        "passed": False,
                        "timestamp": _timestamp(),
                        "requested_devices": list(args.devices),
                        "error": f"{type(error).__name__}: {error}",
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            return 1
        print(
            json.dumps(
                {
                    "passed": True,
                    "timestamp": _timestamp(),
                    "requested_devices": list(args.devices),
                    "devices": [reading.as_json() for reading in readings],
                },
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 0

    stop = StopController()
    previous_handlers = {
        signum: signal.signal(signum, stop.request)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        summary = run_monitor(
            args.output_dir,
            args.interval,
            args.max_samples,
            stop,
            requested_devices=args.devices,
        )
    except (FileExistsError, OSError, ValueError) as error:
        print(f"ERROR: {error}", flush=True)
        return 1
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)

    sample_counts = summary["samples"]
    if not isinstance(sample_counts, dict):
        raise TypeError("monitor summary has invalid sample counts")
    print(
        f"passed={summary['passed']} stop_reason={summary['stop_reason']} "
        f"samples={sample_counts['written']} "
        f"output={args.output_dir}",
        flush=True,
    )
    return 0 if summary["passed"] or stop.event.is_set() else 1


if __name__ == "__main__":
    raise SystemExit(main())
