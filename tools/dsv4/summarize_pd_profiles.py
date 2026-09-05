#!/usr/bin/env python3
"""Validate and summarize one Attention/FFN Ascend profiler capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_OUTPUT_FILES = (
    "kernel_details.csv",
    "trace_view.json",
    "communication.json",
)
MAX_METADATA_HASH_BYTES = 1024 * 1024
DEFAULT_ARTIFACT_TIMEOUT_SECONDS = 300.0
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_STABILITY_INTERVAL_SECONDS = 1.0


class ProfileValidationError(ValueError):
    """The profiler directories do not contain one coherent capture."""


class _PendingProfileArtifactsError(ProfileValidationError):
    """A capture may still be exporting or analysing its artifacts."""


@dataclass(frozen=True)
class ExpectedSchedule:
    skip_first: int
    wait: int
    warmup: int
    active: int

    def as_json(self) -> dict[str, int]:
        return {
            "skip_first": self.skip_first,
            "wait": self.wait,
            "warmup": self.warmup,
            "active": self.active,
        }


def _utc_timestamp(epoch: float) -> str:
    return (
        datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _file_record(path: Path, base: Path, *, include_sha256: bool) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": _relative(path, base),
        "size_bytes": stat.st_size,
        "mtime_epoch": stat.st_mtime,
        "mtime_utc": _utc_timestamp(stat.st_mtime),
    }
    if include_sha256:
        if stat.st_size > MAX_METADATA_HASH_BYTES:
            raise ProfileValidationError(
                f"metadata file exceeds {MAX_METADATA_HASH_BYTES} bytes: {path}"
            )
        record["sha256"] = _sha256(path)
    return record


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProfileValidationError(f"{label} must be a JSON object")
    return value


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileValidationError(f"{label} must be a non-empty string")
    return value


def _require_schedule_value(schedule: dict[str, Any], name: str) -> int:
    value = schedule.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileValidationError(f"profile schedule {name} must be an integer")
    return value


def _validate_metadata(
    profiler_info: Path,
    *,
    expected_cann_version: str,
    expected_schedule: ExpectedSchedule | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        payload = json.loads(profiler_info.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _PendingProfileArtifactsError(
            f"cannot parse current profiler metadata {profiler_info}: {error}"
        ) from error

    payload = _require_mapping(payload, "profiler metadata")
    raw_cann_version = payload.get("cann_version")
    if raw_cann_version is not None and not isinstance(raw_cann_version, str):
        raise ProfileValidationError("cann_version must be a string or null")
    cann_version = raw_cann_version.strip() if raw_cann_version else None
    cann_version_known = (
        cann_version is not None and cann_version.casefold() != "not known"
    )
    if cann_version_known and cann_version != expected_cann_version:
        raise ProfileValidationError(
            "CANN version mismatch in "
            f"{profiler_info}: expected {expected_cann_version!r}, got "
            f"{cann_version!r}"
        )
    torch_npu_version = _require_text(
        payload.get("torch_npu_version"), "torch_npu_version"
    )

    config = _require_mapping(payload.get("config"), "config")
    common_config = _require_mapping(
        config.get("common_config"), "config.common_config"
    )
    raw_schedule = common_config.get("schedule")
    if expected_schedule is None:
        # torch_npu substitutes its always-RECORD default function when no
        # schedule is passed. Depending on the release, metadata serializes
        # that function as either an empty object or null.
        if raw_schedule not in ({}, None):
            raise ProfileValidationError(
                "manual profile window requires no step schedule in "
                f"{profiler_info}, got {raw_schedule!r}"
            )
        actual_schedule = raw_schedule
    else:
        schedule = _require_mapping(raw_schedule, "config.common_config.schedule")
        actual_schedule = {
            name: _require_schedule_value(schedule, name)
            for name in ("skip_first", "wait", "warmup", "active")
        }
        if actual_schedule != expected_schedule.as_json():
            raise ProfileValidationError(
                "profile schedule mismatch in "
                f"{profiler_info}: expected {expected_schedule.as_json()}, got "
                f"{actual_schedule}"
            )
    if common_config.get("with_stack") is not False:
        raise ProfileValidationError(
            f"config.common_config.with_stack must be false in {profiler_info}"
        )

    selected = {
        "cann_version": cann_version,
        # torch_npu commonly records "not known" here. The outer profile
        # session/runtime fingerprint remains the authoritative version gate.
        "cann_version_cross_check": "matched" if cann_version_known else "unavailable",
        "torch_npu_version": torch_npu_version,
        "schedule": actual_schedule,
        "with_stack": False,
    }
    return payload, selected


def _inventory(root: Path) -> dict[str, Any]:
    stats = [path.stat() for path in root.rglob("*") if path.is_file()]
    if not stats:
        raise ProfileValidationError(f"profiler root contains no files: {root}")
    earliest = min(stat.st_mtime for stat in stats)
    latest = max(stat.st_mtime for stat in stats)
    return {
        "file_count": len(stats),
        "total_bytes": sum(stat.st_size for stat in stats),
        "mtime_range": {
            "earliest_epoch": earliest,
            "earliest_utc": _utc_timestamp(earliest),
            "latest_epoch": latest,
            "latest_utc": _utc_timestamp(latest),
        },
    }


def _artifact_state(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _require_stable_artifacts(
    role: str,
    artifacts: tuple[Path, ...],
    *,
    stability_interval_seconds: float,
) -> None:
    before = {path: _artifact_state(path) for path in artifacts}
    if stability_interval_seconds:
        time.sleep(stability_interval_seconds)
    try:
        after = {path: _artifact_state(path) for path in artifacts}
    except OSError as error:
        raise _PendingProfileArtifactsError(
            f"{role} profile artifacts changed during stability check: {error}"
        ) from error
    changed = [path for path in artifacts if before[path] != after[path]]
    if changed:
        paths = ", ".join(str(path) for path in changed)
        raise _PendingProfileArtifactsError(
            f"{role} profile artifacts are still changing: {paths}"
        )


def _summarize_role(
    role: str,
    base_dir: Path,
    *,
    started_epoch: float,
    expected_cann_version: str,
    expected_schedule: ExpectedSchedule | None,
    stability_interval_seconds: float,
) -> dict[str, Any]:
    if not base_dir.is_dir():
        raise ProfileValidationError(
            f"{role} profiler directory does not exist: {base_dir}"
        )

    all_info_files = sorted(
        path for path in base_dir.rglob("profiler_info_*.json") if path.is_file()
    )
    current_info_files = [
        path for path in all_info_files if path.stat().st_mtime >= started_epoch
    ]
    stale_info_files = [
        path for path in all_info_files if path.stat().st_mtime < started_epoch
    ]
    if stale_info_files:
        paths = ", ".join(_relative(path, base_dir) for path in stale_info_files)
        raise ProfileValidationError(
            f"{role} profiler directory mixes stale roots with this run: {paths}"
        )
    if not current_info_files:
        raise _PendingProfileArtifactsError(
            f"{role} has no profiler_info_*.json newer than started epoch "
            f"{started_epoch}"
        )
    if len(current_info_files) != 1:
        paths = ", ".join(_relative(path, base_dir) for path in current_info_files)
        raise ProfileValidationError(
            f"{role} must contain exactly one current profiler root; found "
            f"{len(current_info_files)}: {paths}"
        )

    profiler_info = current_info_files[0]
    root = profiler_info.parent
    _, selected_metadata = _validate_metadata(
        profiler_info,
        expected_cann_version=expected_cann_version,
        expected_schedule=expected_schedule,
    )

    output_dir = root / "ASCEND_PROFILER_OUTPUT"
    required_artifacts: list[Path] = []
    pending: list[str] = []
    for name in REQUIRED_OUTPUT_FILES:
        artifact = output_dir / name
        if not artifact.is_file():
            pending.append(f"missing {artifact}")
            continue
        stat = artifact.stat()
        if stat.st_size <= 0:
            pending.append(f"empty {artifact}")
            continue
        if stat.st_mtime < started_epoch:
            pending.append(
                f"stale {artifact} (mtime {stat.st_mtime} < {started_epoch})"
            )
            continue
        required_artifacts.append(artifact)

    analysis_marker = output_dir / "analyse.done"
    if not analysis_marker.is_file():
        pending.append(f"missing {analysis_marker}")
    elif analysis_marker.stat().st_mtime < started_epoch:
        pending.append(
            f"stale {analysis_marker} "
            f"(mtime {analysis_marker.stat().st_mtime} < {started_epoch})"
        )
    if pending:
        raw_roots = sorted(path for path in root.glob("PROF_*") if path.is_dir())
        device_end_markers = sorted(
            marker
            for raw_root in raw_roots
            for marker in raw_root.glob("device_*/end_info*.done")
            if marker.is_file()
        )
        host_end_markers = sorted(
            marker
            for raw_root in raw_roots
            for marker in raw_root.glob("host/end_info.done")
            if marker.is_file()
        )
        if raw_roots and (not device_end_markers or not host_end_markers):
            paths = ", ".join(_relative(path, base_dir) for path in raw_roots)
            raise ProfileValidationError(
                f"{role} raw CANN capture is incomplete: expected "
                f"device_*/end_info*.done and host/end_info.done under {paths}; "
                "offline analyse cannot repair an unfinalized capture, so rerun "
                "the Profile round after checking graceful profiler shutdown "
                "and free disk space"
            )
        raise _PendingProfileArtifactsError(
            f"{role} profile is incomplete: {'; '.join(pending)}"
        )

    stable_artifacts = (*required_artifacts, analysis_marker)
    _require_stable_artifacts(
        role,
        stable_artifacts,
        stability_interval_seconds=stability_interval_seconds,
    )
    # The derived CSV/trace files can be hundreds of MiB. Record their
    # identity and size without reading them solely to calculate a digest.
    required_records = {
        artifact.name: _file_record(artifact, base_dir, include_sha256=False)
        for artifact in required_artifacts
    }

    metadata_records = {
        profiler_info.name: _file_record(profiler_info, base_dir, include_sha256=True)
    }
    optional_metadata = root / "profiler_metadata.json"
    if optional_metadata.is_file():
        if optional_metadata.stat().st_mtime < started_epoch:
            raise ProfileValidationError(
                f"{role} profiler_metadata.json predates this run: {optional_metadata}"
            )
        metadata_records[optional_metadata.name] = _file_record(
            optional_metadata, base_dir, include_sha256=True
        )

    return {
        "root": _relative(root, base_dir),
        "metadata": selected_metadata,
        "metadata_files": metadata_records,
        "required_outputs": required_records,
        # CANN writes an empty analyse.done sentinel after post-processing.
        "analysis_marker": _file_record(
            analysis_marker, base_dir, include_sha256=False
        ),
        "inventory": _inventory(root),
    }


def summarize_profiles(
    attention_dir: Path,
    ffn_dir: Path,
    *,
    started_epoch: float,
    expected_cann_version: str,
    expected_schedule: ExpectedSchedule | None,
    artifact_timeout_seconds: float = 0.0,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    stability_interval_seconds: float = 0.0,
) -> dict[str, Any]:
    _validate_summary_arguments(
        started_epoch=started_epoch,
        expected_cann_version=expected_cann_version,
        expected_schedule=expected_schedule,
        artifact_timeout_seconds=artifact_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        stability_interval_seconds=stability_interval_seconds,
    )

    deadline = time.monotonic() + artifact_timeout_seconds
    while True:
        try:
            roles = {
                "attention": _summarize_role(
                    "attention",
                    attention_dir,
                    started_epoch=started_epoch,
                    expected_cann_version=expected_cann_version,
                    expected_schedule=expected_schedule,
                    stability_interval_seconds=stability_interval_seconds,
                ),
                "ffn": _summarize_role(
                    "ffn",
                    ffn_dir,
                    started_epoch=started_epoch,
                    expected_cann_version=expected_cann_version,
                    expected_schedule=expected_schedule,
                    stability_interval_seconds=stability_interval_seconds,
                ),
            }
            break
        except _PendingProfileArtifactsError as error:
            if time.monotonic() >= deadline:
                raise ProfileValidationError(str(error)) from error
            time.sleep(
                min(poll_interval_seconds, max(0.0, deadline - time.monotonic()))
            )

    return _summary_payload(
        roles,
        started_epoch=started_epoch,
        expected_cann_version=expected_cann_version,
        expected_schedule=expected_schedule,
    )


def summarize_role_profile(
    role: str,
    role_dir: Path,
    *,
    started_epoch: float,
    expected_cann_version: str,
    expected_schedule: ExpectedSchedule | None,
    artifact_timeout_seconds: float = 0.0,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    stability_interval_seconds: float = 0.0,
) -> dict[str, Any]:
    if role not in {"attention", "ffn"}:
        raise ProfileValidationError(f"unsupported profiler role: {role}")
    _validate_summary_arguments(
        started_epoch=started_epoch,
        expected_cann_version=expected_cann_version,
        expected_schedule=expected_schedule,
        artifact_timeout_seconds=artifact_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        stability_interval_seconds=stability_interval_seconds,
    )

    deadline = time.monotonic() + artifact_timeout_seconds
    while True:
        try:
            role_summary = _summarize_role(
                role,
                role_dir,
                started_epoch=started_epoch,
                expected_cann_version=expected_cann_version,
                expected_schedule=expected_schedule,
                stability_interval_seconds=stability_interval_seconds,
            )
            break
        except _PendingProfileArtifactsError as error:
            if time.monotonic() >= deadline:
                raise ProfileValidationError(str(error)) from error
            time.sleep(
                min(poll_interval_seconds, max(0.0, deadline - time.monotonic()))
            )

    return _summary_payload(
        {role: role_summary},
        started_epoch=started_epoch,
        expected_cann_version=expected_cann_version,
        expected_schedule=expected_schedule,
    )


def _validate_summary_arguments(
    *,
    started_epoch: float,
    expected_cann_version: str,
    expected_schedule: ExpectedSchedule | None,
    artifact_timeout_seconds: float,
    poll_interval_seconds: float,
    stability_interval_seconds: float,
) -> None:
    if started_epoch < 0:
        raise ProfileValidationError("started_epoch must be non-negative")
    if not expected_cann_version.strip():
        raise ProfileValidationError("expected_cann_version must not be empty")
    if expected_schedule is not None:
        for name, value in expected_schedule.as_json().items():
            minimum = 1 if name == "active" else 0
            if value < minimum:
                raise ProfileValidationError(f"expected {name} must be >= {minimum}")
    if artifact_timeout_seconds < 0:
        raise ProfileValidationError("artifact_timeout_seconds must be non-negative")
    if poll_interval_seconds <= 0:
        raise ProfileValidationError("poll_interval_seconds must be positive")
    if stability_interval_seconds < 0:
        raise ProfileValidationError("stability_interval_seconds must be non-negative")


def _summary_payload(
    roles: dict[str, Any],
    *,
    started_epoch: float,
    expected_cann_version: str,
    expected_schedule: ExpectedSchedule | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "passed": True,
        "started_epoch": started_epoch,
        "started_utc": _utc_timestamp(started_epoch),
        "expectations": {
            "cann_version": expected_cann_version,
            "profile_mode": (
                "manual-window" if expected_schedule is None else "scheduled"
            ),
            "schedule": (
                None if expected_schedule is None else expected_schedule.as_json()
            ),
            "with_stack": False,
        },
        "roles": roles,
    }


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-dir", type=Path)
    parser.add_argument("--ffn-dir", type=Path)
    parser.add_argument("--role", choices=("attention", "ffn"))
    parser.add_argument("--role-dir", type=Path)
    parser.add_argument("--started-epoch", required=True, type=_non_negative_float)
    parser.add_argument("--expected-cann-version", required=True)
    parser.add_argument(
        "--expect-manual-window",
        action="store_true",
        help="require the profiler metadata to contain no step schedule",
    )
    parser.add_argument("--expected-skip-first", type=_non_negative_int)
    parser.add_argument("--expected-wait", type=_non_negative_int)
    parser.add_argument("--expected-warmup", type=_non_negative_int)
    parser.add_argument("--expected-active", type=_positive_int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--artifact-timeout-seconds",
        type=_non_negative_float,
        default=DEFAULT_ARTIFACT_TIMEOUT_SECONDS,
        help="wait for torch_npu/CANN post-processing (default: 300 seconds)",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--stability-interval-seconds",
        type=_non_negative_float,
        default=DEFAULT_STABILITY_INTERVAL_SECONDS,
        help="recheck derived output sizes/mtimes after this delay (default: 1 second)",
    )
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _build_parser().parse_args()
    if args.role is None:
        if (
            args.attention_dir is None
            or args.ffn_dir is None
            or args.role_dir is not None
        ):
            raise SystemExit(
                "dual-role mode requires --attention-dir and --ffn-dir only"
            )
    elif (
        args.role_dir is None
        or args.attention_dir is not None
        or args.ffn_dir is not None
    ):
        raise SystemExit("single-role mode requires --role and --role-dir only")
    schedule_values = (
        args.expected_skip_first,
        args.expected_wait,
        args.expected_warmup,
        args.expected_active,
    )
    if args.expect_manual_window:
        if any(value is not None for value in schedule_values):
            raise SystemExit(
                "--expect-manual-window cannot be combined with expected "
                "schedule values"
            )
        expected_schedule = None
    else:
        if any(value is None for value in schedule_values):
            raise SystemExit(
                "use --expect-manual-window or provide all four expected "
                "schedule values"
            )
        expected_schedule = ExpectedSchedule(
            skip_first=args.expected_skip_first,
            wait=args.expected_wait,
            warmup=args.expected_warmup,
            active=args.expected_active,
        )
    try:
        kwargs = {
            "started_epoch": args.started_epoch,
            "expected_cann_version": args.expected_cann_version,
            "expected_schedule": expected_schedule,
            "artifact_timeout_seconds": args.artifact_timeout_seconds,
            "poll_interval_seconds": args.poll_interval_seconds,
            "stability_interval_seconds": args.stability_interval_seconds,
        }
        if args.role is None:
            summary = summarize_profiles(args.attention_dir, args.ffn_dir, **kwargs)
        else:
            summary = summarize_role_profile(args.role, args.role_dir, **kwargs)
    except ProfileValidationError as error:
        failure = {
            "schema_version": 1,
            "passed": False,
            "error": str(error),
        }
        _write_json(args.output, failure)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    _write_json(args.output, summary)
    print(f"passed=True output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
