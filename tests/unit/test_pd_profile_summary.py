from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from tools.dsv4 import summarize_pd_profiles

STARTED_EPOCH = 1_800_000_000.0
SCHEDULE = summarize_pd_profiles.ExpectedSchedule(
    skip_first=1500,
    wait=2,
    warmup=1,
    active=20,
)


def _write(path: Path, content: str, *, mtime: float = STARTED_EPOCH + 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _make_role(
    base: Path,
    role: str,
    *,
    cann_version: str | None = "9.0.0",
    with_stack: bool = False,
    schedule: dict[str, int] | None = None,
    manual_window: bool = False,
    suffix: str = "capture_ascend_pt",
    mtime: float = STARTED_EPOCH + 1,
) -> Path:
    role_dir = base / role
    root = role_dir / suffix
    payload = {
        "config": {
            "common_config": {
                "schedule": ({} if manual_window else (schedule or SCHEDULE.as_json())),
                "with_stack": with_stack,
            }
        },
        "torch_npu_version": "2.10.0.post2",
        "cann_version": cann_version,
        "rank_id": 0,
    }
    _write(
        root / "profiler_info_0.json",
        json.dumps(payload),
        mtime=mtime,
    )
    _write(root / "profiler_metadata.json", '{"ENV_VARIABLES":{}}', mtime=mtime)
    output_dir = root / "ASCEND_PROFILER_OUTPUT"
    _write(output_dir / "kernel_details.csv", "Name,Duration\nop,1\n", mtime=mtime)
    _write(output_dir / "trace_view.json", '[{"name":"op"}]', mtime=mtime)
    _write(output_dir / "communication.json", '{"hccl":[]}', mtime=mtime)
    _write(output_dir / "analyse.done", "", mtime=mtime)
    return role_dir


def _summarize(tmp_path: Path) -> dict:
    return summarize_pd_profiles.summarize_profiles(
        tmp_path / "attention",
        tmp_path / "ffn",
        started_epoch=STARTED_EPOCH,
        expected_cann_version="9.0.0",
        expected_schedule=SCHEDULE,
    )


def test_summarizes_one_current_profile_per_role_without_hashing_traces(tmp_path):
    for role in ("attention", "ffn"):
        _make_role(tmp_path, role)

    summary = _summarize(tmp_path)

    assert summary["passed"] is True
    assert summary["expectations"]["with_stack"] is False
    for role in ("attention", "ffn"):
        result = summary["roles"][role]
        assert result["root"] == "capture_ascend_pt"
        assert result["metadata"]["cann_version"] == "9.0.0"
        assert result["metadata"]["cann_version_cross_check"] == "matched"
        assert result["metadata"]["torch_npu_version"] == "2.10.0.post2"
        assert result["metadata"]["schedule"] == SCHEDULE.as_json()
        info_record = result["metadata_files"]["profiler_info_0.json"]
        info_path = tmp_path / role / info_record["path"]
        assert (
            info_record["sha256"] == hashlib.sha256(info_path.read_bytes()).hexdigest()
        )
        assert "sha256" not in result["required_outputs"]["trace_view.json"]
        assert result["analysis_marker"]["size_bytes"] == 0
        assert result["inventory"]["file_count"] == 6
        assert result["inventory"]["total_bytes"] > 0
        assert result["inventory"]["mtime_range"]["earliest_epoch"] >= STARTED_EPOCH


def test_summarizes_manual_window_without_step_schedule(tmp_path):
    for role in ("attention", "ffn"):
        _make_role(tmp_path, role, manual_window=True)

    summary = summarize_pd_profiles.summarize_profiles(
        tmp_path / "attention",
        tmp_path / "ffn",
        started_epoch=STARTED_EPOCH,
        expected_cann_version="9.0.0",
        expected_schedule=None,
    )

    assert summary["expectations"]["profile_mode"] == "manual-window"
    assert summary["expectations"]["schedule"] is None
    for role in ("attention", "ffn"):
        assert summary["roles"][role]["metadata"]["schedule"] == {}


def test_manual_window_rejects_a_step_schedule(tmp_path):
    for role in ("attention", "ffn"):
        _make_role(tmp_path, role)

    with pytest.raises(
        summarize_pd_profiles.ProfileValidationError,
        match="manual profile window requires no step schedule",
    ):
        summarize_pd_profiles.summarize_profiles(
            tmp_path / "attention",
            tmp_path / "ffn",
            started_epoch=STARTED_EPOCH,
            expected_cann_version="9.0.0",
            expected_schedule=None,
        )


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_summarizes_a_single_split_role(tmp_path, role):
    role_dir = _make_role(tmp_path, role)

    summary = summarize_pd_profiles.summarize_role_profile(
        role,
        role_dir,
        started_epoch=STARTED_EPOCH,
        expected_cann_version="9.0.0",
        expected_schedule=SCHEDULE,
    )

    assert summary["passed"] is True
    assert list(summary["roles"]) == [role]
    assert summary["roles"][role]["root"] == "capture_ascend_pt"


def test_rejects_an_unknown_split_role(tmp_path):
    with pytest.raises(
        summarize_pd_profiles.ProfileValidationError,
        match="unsupported profiler role",
    ):
        summarize_pd_profiles.summarize_role_profile(
            "prefill",
            tmp_path / "prefill",
            started_epoch=STARTED_EPOCH,
            expected_cann_version="9.0.0",
            expected_schedule=SCHEDULE,
        )


def test_rejects_a_stale_profiler_root_mixed_with_current_run(tmp_path):
    for role in ("attention", "ffn"):
        _make_role(tmp_path, role)
    _make_role(
        tmp_path,
        "attention",
        suffix="old_ascend_pt",
        mtime=STARTED_EPOCH - 1,
    )

    with pytest.raises(
        summarize_pd_profiles.ProfileValidationError, match="mixes stale roots"
    ):
        _summarize(tmp_path)


def test_rejects_multiple_current_roots(tmp_path):
    for role in ("attention", "ffn"):
        _make_role(tmp_path, role)
    _make_role(tmp_path, "ffn", suffix="second_ascend_pt")

    with pytest.raises(
        summarize_pd_profiles.ProfileValidationError,
        match="exactly one current profiler root",
    ):
        _summarize(tmp_path)


def test_rejects_missing_or_stale_derived_output(tmp_path):
    for role in ("attention", "ffn"):
        _make_role(tmp_path, role)
    missing = (
        tmp_path
        / "attention/capture_ascend_pt/ASCEND_PROFILER_OUTPUT/communication.json"
    )
    missing.unlink()

    with pytest.raises(
        summarize_pd_profiles.ProfileValidationError, match="profile is incomplete"
    ):
        _summarize(tmp_path)

    _write(missing, "{}", mtime=STARTED_EPOCH - 1)
    with pytest.raises(summarize_pd_profiles.ProfileValidationError, match="stale"):
        _summarize(tmp_path)


def test_reports_an_incomplete_raw_cann_capture_before_offline_analysis(tmp_path):
    for role in ("attention", "ffn"):
        _make_role(tmp_path, role)
    attention_root = tmp_path / "attention/capture_ascend_pt"
    (attention_root / "ASCEND_PROFILER_OUTPUT/communication.json").unlink()
    (attention_root / "PROF_000001/device_0/data").mkdir(parents=True)

    with pytest.raises(
        summarize_pd_profiles.ProfileValidationError,
        match="raw CANN capture is incomplete.*offline analyse cannot repair",
    ):
        _summarize(tmp_path)

    _write(
        attention_root / "PROF_000001/device_0/end_info.0.done",
        "",
    )
    _write(attention_root / "PROF_000001/host/end_info.done", "")
    with pytest.raises(
        summarize_pd_profiles.ProfileValidationError, match="profile is incomplete"
    ):
        _summarize(tmp_path)


def test_requires_current_analysis_done_marker(tmp_path):
    for role in ("attention", "ffn"):
        _make_role(tmp_path, role)
    marker = (
        tmp_path / "attention/capture_ascend_pt/ASCEND_PROFILER_OUTPUT/analyse.done"
    )
    marker.unlink()

    with pytest.raises(
        summarize_pd_profiles.ProfileValidationError, match="analyse.done"
    ):
        _summarize(tmp_path)

    _write(marker, "", mtime=STARTED_EPOCH - 1)
    with pytest.raises(summarize_pd_profiles.ProfileValidationError, match="stale"):
        _summarize(tmp_path)


def test_rejects_artifacts_that_change_during_stability_check(monkeypatch, tmp_path):
    for role in ("attention", "ffn"):
        _make_role(tmp_path, role)
    trace = (
        tmp_path / "attention/capture_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json"
    )

    def mutate_during_sleep(_seconds):
        trace.write_text('[{"name":"still-writing"}]', encoding="utf-8")

    monkeypatch.setattr(summarize_pd_profiles.time, "sleep", mutate_during_sleep)

    with pytest.raises(
        summarize_pd_profiles.ProfileValidationError, match="still changing"
    ):
        summarize_pd_profiles.summarize_profiles(
            tmp_path / "attention",
            tmp_path / "ffn",
            started_epoch=STARTED_EPOCH,
            expected_cann_version="9.0.0",
            expected_schedule=SCHEDULE,
            stability_interval_seconds=0.1,
        )


@pytest.mark.parametrize(
    "change,error",
    [
        ({"cann_version": "9.0.1"}, "CANN version mismatch"),
        ({"with_stack": True}, "with_stack must be false"),
        (
            {"schedule": {**SCHEDULE.as_json(), "active": 19}},
            "profile schedule mismatch",
        ),
    ],
)
def test_rejects_metadata_contract_mismatch(tmp_path, change, error):
    _make_role(tmp_path, "attention", **change)
    _make_role(tmp_path, "ffn")

    with pytest.raises(summarize_pd_profiles.ProfileValidationError, match=error):
        _summarize(tmp_path)


@pytest.mark.parametrize("reported", ["not known", "NOT KNOWN", "", None])
def test_unknown_cann_version_defers_to_runtime_fingerprint(tmp_path, reported):
    _make_role(tmp_path, "attention", cann_version=reported)
    _make_role(tmp_path, "ffn", cann_version=reported)

    summary = _summarize(tmp_path)

    for role in ("attention", "ffn"):
        metadata = summary["roles"][role]["metadata"]
        assert metadata["cann_version_cross_check"] == "unavailable"


def test_cli_writes_compact_failure_summary(monkeypatch, tmp_path):
    _make_role(tmp_path, "attention")
    _make_role(tmp_path, "ffn", cann_version="9.0.1")
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_pd_profiles.py",
            "--attention-dir",
            str(tmp_path / "attention"),
            "--ffn-dir",
            str(tmp_path / "ffn"),
            "--started-epoch",
            str(STARTED_EPOCH),
            "--expected-cann-version",
            "9.0.0",
            "--expected-skip-first",
            "1500",
            "--expected-wait",
            "2",
            "--expected-warmup",
            "1",
            "--expected-active",
            "20",
            "--artifact-timeout-seconds",
            "0",
            "--stability-interval-seconds",
            "0",
            "--output",
            str(output),
        ],
    )

    assert summarize_pd_profiles.main() == 1
    line = output.read_text(encoding="utf-8")
    assert '": ' not in line
    assert '", ' not in line
    assert json.loads(line)["passed"] is False
