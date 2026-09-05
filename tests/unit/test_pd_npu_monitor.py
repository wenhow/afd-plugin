from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from tools.dsv4 import run_pd_npu_monitor

NPU_SMI_OUTPUT = """\
| 0 Ascend910 | OK | 172.7 45 0 / 0 |
| 0 0 | 0000:18:00.0 | 12 0 / 0 3122 / 65536 |
| 0 Ascend910 | OK | - 47 0 / 0 |
| 1 1 | 0000:19:00.0 | 35.5 0 / 0 4096 / 65536 |
"""


def _reading(
    phy_id: int,
    *,
    aicore_percent: float,
    hbm_used_mb: float,
) -> run_pd_npu_monitor.DeviceReading:
    return run_pd_npu_monitor.DeviceReading(
        npu_id=0,
        chip_id=phy_id,
        phy_id=phy_id,
        aicore_percent=aicore_percent,
        hbm_used_mb=hbm_used_mb,
        hbm_total_mb=65536,
    )


def test_parse_npu_smi_info_maps_phy_ids_to_logical_devices():
    readings = run_pd_npu_monitor.parse_npu_smi_info(NPU_SMI_OUTPUT)

    assert [reading.phy_id for reading in readings] == [0, 1]
    assert [reading.npu_id for reading in readings] == [0, 0]
    assert readings[0].as_json()["logical_device_id"] == 0
    assert readings[0].aicore_percent == 12
    assert readings[1].aicore_percent == 35.5
    assert readings[1].hbm_used_mb == 4096
    assert readings[1].hbm_total_mb == 65536


def test_parse_npu_smi_info_rejects_missing_device_rows():
    with pytest.raises(ValueError, match="no NPU device readings"):
        run_pd_npu_monitor.parse_npu_smi_info("npu-smi has no devices")


def test_sample_npus_invokes_npu_smi_info(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, NPU_SMI_OUTPUT, "")

    monkeypatch.setattr(run_pd_npu_monitor.subprocess, "run", run)

    readings = run_pd_npu_monitor._sample_npus()

    assert len(readings) == 2
    assert calls[0][0] == ["npu-smi", "info"]
    assert calls[0][1]["timeout"] == run_pd_npu_monitor.NPU_SMI_TIMEOUT_SECONDS


def test_monitor_writes_compact_samples_and_summary(tmp_path):
    samples = iter(
        [
            [
                _reading(0, aicore_percent=20, hbm_used_mb=1000),
                _reading(1, aicore_percent=40, hbm_used_mb=2000),
            ],
            [
                _reading(0, aicore_percent=60, hbm_used_mb=3000),
                _reading(1, aicore_percent=80, hbm_used_mb=4000),
            ],
        ]
    )
    stop = run_pd_npu_monitor.StopController()

    summary = run_pd_npu_monitor.run_monitor(
        tmp_path,
        interval=0.001,
        max_samples=2,
        stop=stop,
        sample_npus=lambda: next(samples),
    )

    lines = (tmp_path / "npu_samples.jsonl").read_text().splitlines()
    snapshots = [json.loads(line) for line in lines]
    persisted_summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(snapshots) == 2
    assert " " not in lines[0]
    assert snapshots[0]["devices"][1]["phy_id"] == 1
    assert summary == persisted_summary
    assert summary["passed"] is True
    assert summary["samples"] == {
        "written": 2,
        "successful": 2,
        "failed": 0,
        "device_observations": 4,
    }
    assert summary["overall"]["aicore_average_percent"] == 50
    assert summary["overall"]["aicore_peak_percent"] == 70
    assert summary["overall"]["aicore_peak_single_device_percent"] == 80
    assert summary["overall"]["hbm_peak_aggregate_used_mb"] == 7000
    assert summary["devices"][0]["aicore_average_percent"] == 40
    assert summary["devices"][1]["hbm_peak_used_mb"] == 4000


def test_monitor_waits_full_interval_after_a_slow_sample(tmp_path):
    waits = []

    class FakeEvent:
        def is_set(self):
            return False

        def wait(self, timeout):
            waits.append(timeout)
            return False

    class FakeStop:
        event = FakeEvent()
        reason = "max_samples"

    run_pd_npu_monitor.run_monitor(
        tmp_path,
        interval=5.0,
        max_samples=2,
        stop=FakeStop(),
        sample_npus=lambda: [_reading(0, aicore_percent=20, hbm_used_mb=1000)],
    )

    assert waits == [5.0]


def test_monitor_records_sample_failure_and_continues(tmp_path):
    attempts = 0

    def sample_npus() -> list[run_pd_npu_monitor.DeviceReading]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary npu-smi failure")
        return [_reading(0, aicore_percent=25, hbm_used_mb=2048)]

    summary = run_pd_npu_monitor.run_monitor(
        tmp_path,
        interval=0.001,
        max_samples=2,
        stop=run_pd_npu_monitor.StopController(),
        sample_npus=sample_npus,
    )

    lines = (tmp_path / "npu_samples.jsonl").read_text().splitlines()
    first_snapshot = json.loads(lines[0])
    assert "temporary npu-smi failure" in first_snapshot["error"]
    assert summary["samples"]["failed"] == 1
    assert summary["samples"]["successful"] == 1
    assert summary["passed"] is False


def test_monitor_filters_and_records_requested_devices(tmp_path):
    summary = run_pd_npu_monitor.run_monitor(
        tmp_path,
        interval=0.001,
        max_samples=1,
        stop=run_pd_npu_monitor.StopController(),
        requested_devices=(1,),
        sample_npus=lambda: [
            _reading(0, aicore_percent=20, hbm_used_mb=1000),
            _reading(1, aicore_percent=40, hbm_used_mb=2000),
        ],
    )

    snapshot = json.loads((tmp_path / "npu_samples.jsonl").read_text().splitlines()[0])
    assert summary["requested_devices"] == [1]
    assert [device["phy_id"] for device in snapshot["devices"]] == [1]
    assert [device["phy_id"] for device in summary["devices"]] == [1]


def test_missing_requested_device_marks_sample_failed(tmp_path):
    summary = run_pd_npu_monitor.run_monitor(
        tmp_path,
        interval=0.001,
        max_samples=1,
        stop=run_pd_npu_monitor.StopController(),
        requested_devices=(0, 7),
        sample_npus=lambda: [_reading(0, aicore_percent=20, hbm_used_mb=1000)],
    )

    snapshot = json.loads((tmp_path / "npu_samples.jsonl").read_text().splitlines()[0])
    assert summary["passed"] is False
    assert summary["samples"]["successful"] == 0
    assert summary["samples"]["failed"] == 1
    assert "missing requested Phy-ID(s): 7" in snapshot["error"]


def test_sigterm_requests_a_normal_stop(tmp_path):
    stop = run_pd_npu_monitor.StopController()
    stop.request(signal.SIGTERM)

    summary = run_pd_npu_monitor.run_monitor(
        tmp_path,
        interval=1,
        max_samples=10,
        stop=stop,
        sample_npus=lambda: pytest.fail("sampling should not start"),
    )

    assert summary["stop_reason"] == "sigterm"
    assert summary["samples"]["written"] == 0
    assert (tmp_path / "summary.json").is_file()


def test_cli_sigterm_writes_summary_and_exits_zero(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_npu_smi = fake_bin / "npu-smi"
    fake_npu_smi.write_text(
        f"#!/usr/bin/env python3\nprint({NPU_SMI_OUTPUT!r})\n",
        encoding="utf-8",
    )
    fake_npu_smi.chmod(0o755)
    output_dir = tmp_path / "output"
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
    process = subprocess.Popen(
        [
            sys.executable,
            str(run_pd_npu_monitor.__file__),
            "run",
            "--output-dir",
            str(output_dir),
            "--interval",
            "30",
            "--max-samples",
            "10",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    snapshots_path = output_dir / "npu_samples.jsonl"
    deadline = time.monotonic() + 5
    while (
        not snapshots_path.exists() or snapshots_path.stat().st_size == 0
    ) and time.monotonic() < deadline:
        time.sleep(0.01)
    process.terminate()
    output, _ = process.communicate(timeout=5)

    assert process.returncode == 0, output
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["stop_reason"] == "sigterm"
    assert summary["samples"]["written"] == 1


@pytest.mark.parametrize("value", ["0", "7201", "-1"])
def test_max_samples_is_bounded(value):
    with pytest.raises(argparse.ArgumentTypeError):
        run_pd_npu_monitor._bounded_samples(value)


@pytest.mark.parametrize("value", ["", "0,", "-1", "0,0", "one"])
def test_devices_reject_invalid_lists(value):
    with pytest.raises(argparse.ArgumentTypeError):
        run_pd_npu_monitor._parse_devices(value)


def test_devices_parse_comma_separated_phy_ids():
    assert run_pd_npu_monitor._parse_devices("0, 2,15") == (0, 2, 15)


def test_cli_preflight_validates_and_reports_requested_devices(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_npu_smi = fake_bin / "npu-smi"
    fake_npu_smi.write_text(
        f"#!/usr/bin/env python3\nprint({NPU_SMI_OUTPUT!r})\n",
        encoding="utf-8",
    )
    fake_npu_smi.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"

    completed = subprocess.run(
        [
            sys.executable,
            str(run_pd_npu_monitor.__file__),
            "preflight",
            "--devices",
            "0,1",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["passed"] is True
    assert payload["requested_devices"] == [0, 1]


def test_cli_preflight_rejects_a_missing_requested_device(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_npu_smi = fake_bin / "npu-smi"
    fake_npu_smi.write_text(
        f"#!/usr/bin/env python3\nprint({NPU_SMI_OUTPUT!r})\n",
        encoding="utf-8",
    )
    fake_npu_smi.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}:{environment['PATH']}"

    completed = subprocess.run(
        [
            sys.executable,
            str(run_pd_npu_monitor.__file__),
            "preflight",
            "--devices",
            "0,7",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["passed"] is False
    assert "missing requested Phy-ID(s): 7" in payload["error"]
