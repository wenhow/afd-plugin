#!/usr/bin/env python3
"""Run and compare the fixed Mooncake PD Graph performance points."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "p50_ttft_ms",
    "p90_ttft_ms",
    "p99_ttft_ms",
    "p50_tpot_ms",
    "p90_tpot_ms",
    "p99_tpot_ms",
)
POINT_ORDER = (
    "control_graph_u1",
    "afd_graph_u1",
    "afd_graph_u2",
    "afd_graph_u2_mtp1",
)
RATIO_POINT_ORDER = (
    "afd_graph_u2",
    "afd_graph_u2_split_a8f8",
    "afd_graph_u2_split_a16f8",
)
RUN_POINT_ORDER = (*POINT_ORDER, *RATIO_POINT_ORDER[1:])
EXPECTED_EXECUTION = {
    "control_graph_u1": {
        "mode": "full-decode-only",
        "u_batches": 1,
        "mtp": 0,
        "mtp_num_speculative_tokens": 1,
        "batch_invariant": False,
        "golden_checked": False,
    },
    "afd_graph_u1": {
        "mode": "full-decode-only",
        "u_batches": 1,
        "mtp": 0,
        "mtp_num_speculative_tokens": 1,
        "batch_invariant": False,
        "golden_checked": False,
    },
    "afd_graph_u2": {
        "mode": "full-decode-only",
        "u_batches": 2,
        "mtp": 0,
        "mtp_num_speculative_tokens": 1,
        "batch_invariant": False,
        "golden_checked": False,
    },
    "afd_graph_u2_mtp1": {
        "mode": "full-decode-only",
        "u_batches": 2,
        "mtp": 1,
        "mtp_num_speculative_tokens": 1,
        "batch_invariant": False,
        "golden_checked": False,
    },
    "afd_graph_u2_split_a8f8": {
        "mode": "full-decode-only",
        "u_batches": 2,
        "mtp": 0,
        "mtp_num_speculative_tokens": 1,
        "batch_invariant": False,
        "golden_checked": False,
    },
    "afd_graph_u2_split_a16f8": {
        "mode": "full-decode-only",
        "u_batches": 2,
        "mtp": 0,
        "mtp_num_speculative_tokens": 1,
        "batch_invariant": False,
        "golden_checked": False,
    },
}
P2_MIN_U2_THROUGHPUT_GAIN_PCT = 10.0
P2_MAX_P99_TPOT_REGRESSION_PCT = 5.0
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _benchmark_command(
    args: argparse.Namespace,
    result_path: Path,
    *,
    input_len: int,
    output_len: int,
    num_prompts: int,
    concurrency: int,
    run_kind: str,
    repeat: int,
) -> list[str]:
    return [
        str(args.python_bin),
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--endpoint",
        "/v1/completions",
        "--model",
        args.served_model,
        "--tokenizer",
        str(args.tokenizer),
        "--tokenizer-mode",
        "deepseek_v4",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        "0.0",
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--seed",
        "1024",
        "--temperature",
        "0",
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--disable-tqdm",
        "--save-result",
        "--result-dir",
        str(result_path.parent),
        "--result-filename",
        result_path.name,
        "--metadata",
        f"matrix_point={args.point}",
        f"phase={args.phase}",
        f"repeat={repeat}",
        f"run_kind={run_kind}",
    ]


def _healthcheck(url: str, timeout: float) -> None:
    with _NO_PROXY_OPENER.open(url, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"health check returned HTTP {response.status}: {url}")


def _fetch_spec_metrics(url: str, timeout: float) -> dict[str, int] | None:
    try:
        with _NO_PROXY_OPENER.open(url, timeout=timeout) as response:
            if response.status != 200:
                return None
            text = response.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    values = {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0}
    found = False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("vllm:spec_decode"):
            continue
        parts = line.split(None, 1)
        metric_name = parts[0].split("{")[0]
        if len(parts) != 2 or not metric_name.endswith("_total"):
            continue
        try:
            value = int(float(parts[1]))
        except ValueError:
            continue
        found = True
        if "num_drafts" in metric_name:
            values["drafts"] += value
        elif "num_draft_tokens" in metric_name:
            values["draft_tokens"] += value
        elif "num_accepted_tokens_per_pos" not in metric_name and (
            "num_accepted_tokens" in metric_name
        ):
            values["accepted_tokens"] += value
    return values if found else None


def _spec_metric_delta(
    before: dict[str, int] | None, after: dict[str, int] | None
) -> dict[str, Any]:
    if before is None or after is None:
        return {
            "available": False,
            "valid": False,
            "drafts": None,
            "draft_tokens": None,
            "accepted_tokens": None,
            "acceptance_rate": None,
            "validation_errors": ["speculative decode metrics are unavailable"],
        }
    delta = {key: after[key] - before[key] for key in before}
    errors: list[str] = []
    for key, value in delta.items():
        if value < 0:
            errors.append(f"{key} delta is negative: {value}")
    if delta["drafts"] <= 0:
        errors.append(f"drafts delta must be positive: {delta['drafts']}")
    if delta["draft_tokens"] <= 0:
        errors.append(f"draft_tokens delta must be positive: {delta['draft_tokens']}")
    if delta["draft_tokens"] < delta["drafts"]:
        errors.append(
            "draft_tokens delta must be greater than or equal to drafts delta"
        )
    if delta["accepted_tokens"] > delta["draft_tokens"]:
        errors.append("accepted_tokens delta exceeds draft_tokens delta")
    return {
        "available": True,
        "valid": not errors,
        **delta,
        "acceptance_rate": (
            delta["accepted_tokens"] / delta["draft_tokens"]
            if delta["draft_tokens"] > 0
            else None
        ),
        "validation_errors": errors,
    }


def _spec_metric_validation_errors(values: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    counts: dict[str, int] = {}
    for key in ("drafts", "draft_tokens", "accepted_tokens"):
        value = values.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not float(value).is_integer()
        ):
            errors.append(f"{key} must be a finite integer: {value!r}")
            continue
        counts[key] = int(value)
    if len(counts) != 3:
        return errors
    if counts["drafts"] <= 0:
        errors.append(f"drafts must be positive: {counts['drafts']}")
    if counts["draft_tokens"] <= 0:
        errors.append(f"draft_tokens must be positive: {counts['draft_tokens']}")
    if counts["accepted_tokens"] < 0:
        errors.append(
            f"accepted_tokens must be nonnegative: {counts['accepted_tokens']}"
        )
    if counts["draft_tokens"] < counts["drafts"]:
        errors.append("draft_tokens must be greater than or equal to drafts")
    if counts["accepted_tokens"] > counts["draft_tokens"]:
        errors.append("accepted_tokens exceeds draft_tokens")
    if counts["draft_tokens"] <= 0:
        return errors
    acceptance_rate = values.get("acceptance_rate")
    expected_rate = counts["accepted_tokens"] / counts["draft_tokens"]
    if (
        not isinstance(acceptance_rate, (int, float))
        or isinstance(acceptance_rate, bool)
        or not math.isfinite(float(acceptance_rate))
        or not math.isclose(float(acceptance_rate), expected_rate, abs_tol=1e-12)
    ):
        errors.append("acceptance_rate does not match accepted_tokens / draft_tokens")
    return errors


def _run_command(command: list[str], result_path: Path, timeout: float) -> dict:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = result_path.with_suffix(".log")
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(
            f"benchmark exited with {completed.returncode}; log={log_path}\n{tail}"
        )
    if not result_path.is_file():
        raise RuntimeError(f"benchmark did not create {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def _validate_result(
    result: dict[str, Any],
    *,
    input_len: int,
    output_len: int,
    num_prompts: int,
) -> dict[str, Any]:
    errors = [error for error in result.get("errors", []) if error]
    metric_values: list[float] = []
    metrics_present = all(name in result for name in METRICS)
    if metrics_present:
        try:
            metric_values = [float(result[name]) for name in METRICS]
        except (TypeError, ValueError):
            metric_values = []
    checks = {
        "completed": int(result.get("completed", -1)) == num_prompts,
        "failed": int(result.get("failed", -1)) == 0,
        "input_tokens": int(result.get("total_input_tokens", -1))
        == input_len * num_prompts,
        "output_tokens": int(result.get("total_output_tokens", -1))
        == output_len * num_prompts,
        "errors": not errors,
        "metrics_present": metrics_present,
        "metrics_numeric": len(metric_values) == len(METRICS),
        "metrics_finite": bool(metric_values)
        and all(math.isfinite(value) for value in metric_values),
        "metrics_positive": bool(metric_values)
        and all(value > 0 for value in metric_values),
    }
    return {"passed": all(checks.values()), "checks": checks, "errors": errors}


def _metric_stats(values: list[float]) -> dict[str, float | list[float]]:
    mean = statistics.fmean(values)
    stddev = statistics.pstdev(values)
    return {
        "values": values,
        "mean": mean,
        "min": min(values),
        "max": max(values),
        "stddev": stddev,
        "cv": stddev / mean if mean else 0.0,
    }


def _aggregate_results(
    records: list[dict[str, Any]], total_npus: int
) -> dict[str, Any]:
    metrics = {
        name: _metric_stats([float(record["result"][name]) for record in records])
        for name in METRICS
    }
    accepted = sum(
        int(record["result"].get("spec_decode_accepted_tokens", 0))
        for record in records
    )
    drafted = sum(
        int(record["result"].get("spec_decode_draft_tokens", 0)) for record in records
    )
    return {
        "metrics": metrics,
        "output_tokens_per_second_per_npu": metrics["output_throughput"]["mean"]
        / total_npus,
        "spec_decode": {
            "accepted_tokens": accepted,
            "draft_tokens": drafted,
            "acceptance_rate": accepted / drafted if drafted else None,
        },
    }


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "completed",
        "failed",
        "duration",
        "total_input_tokens",
        "total_output_tokens",
        *METRICS,
        "spec_decode_num_drafts",
        "spec_decode_draft_tokens",
        "spec_decode_accepted_tokens",
        "spec_decode_acceptance_rate",
    )
    return {key: result[key] for key in keys if key in result}


def _resource_manifest(args: argparse.Namespace) -> dict[str, Any]:
    reserved_npus = getattr(args, "reserved_npus", None)
    control_reserved_npus = getattr(args, "control_reserved_npus", None)
    server_count = getattr(args, "server_count", None)
    control_server_count = getattr(args, "control_server_count", None)
    prefill_server_count = getattr(args, "prefill_server_count", None)
    decode_server_count = getattr(args, "decode_server_count", None)
    if control_reserved_npus is None:
        control_reserved_npus = reserved_npus
    if control_server_count is None:
        control_server_count = server_count
    return {
        "prefill_npus": args.prefill_npus,
        "decode_npus": args.decode_npus,
        "attention_npus": args.attention_npus,
        "ffn_npus": args.ffn_npus,
        "active_npus": args.total_npus,
        "total_npus": args.total_npus,
        "reserved_npus": reserved_npus,
        "server_count": server_count,
        "prefill_server_count": prefill_server_count,
        "decode_server_count": decode_server_count,
        "control_active_npus": args.control_total_npus,
        "control_total_npus": args.control_total_npus,
        "control_reserved_npus": control_reserved_npus,
        "control_server_count": control_server_count,
        "extra_npus_vs_control": args.total_npus - args.control_total_npus,
        "extra_npus_pct_vs_control": (
            (args.total_npus / args.control_total_npus - 1.0) * 100.0
        ),
        "extra_active_npus_vs_control": args.total_npus - args.control_total_npus,
        "extra_active_npus_pct_vs_control": (
            (args.total_npus / args.control_total_npus - 1.0) * 100.0
        ),
        "extra_reserved_npus_vs_control": (
            reserved_npus - control_reserved_npus
            if reserved_npus is not None and control_reserved_npus is not None
            else None
        ),
        "extra_reserved_npus_pct_vs_control": (
            (reserved_npus / control_reserved_npus - 1.0) * 100.0
            if reserved_npus is not None
            and control_reserved_npus is not None
            and control_reserved_npus > 0
            else None
        ),
        "extra_servers_vs_control": (
            server_count - control_server_count
            if server_count is not None and control_server_count is not None
            else None
        ),
        "extra_servers_pct_vs_control": (
            (server_count / control_server_count - 1.0) * 100.0
            if server_count is not None
            and control_server_count is not None
            and control_server_count > 0
            else None
        ),
        "ffn_share_pct_of_deployment": args.ffn_npus / args.total_npus * 100.0,
        "ffn_share_pct_of_active_npus": args.ffn_npus / args.total_npus * 100.0,
        "ffn_share_pct_of_reserved_npus": (
            args.ffn_npus / reserved_npus * 100.0
            if reserved_npus is not None and reserved_npus > 0
            else None
        ),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.phase == "p1" and args.repeats != 1:
        raise ValueError("P1 must use exactly one measurement repeat")
    if args.phase == "p2" and args.repeats != 3:
        raise ValueError("P2 must use exactly three measurement repeats")
    expected_total = args.prefill_npus + args.decode_npus
    if args.total_npus != expected_total:
        raise ValueError(
            f"total_npus must equal prefill_npus + decode_npus ({expected_total})"
        )
    if args.control_total_npus <= 0:
        raise ValueError("control_total_npus must be positive")
    if (
        min(
            args.prefill_npus,
            args.decode_npus,
            args.attention_npus,
            args.ffn_npus,
            args.total_npus,
        )
        < 0
    ):
        raise ValueError("NPU resource counts must be nonnegative")
    if args.prefill_npus == 0 or args.decode_npus == 0 or args.total_npus == 0:
        raise ValueError("prefill_npus, decode_npus, and total_npus must be positive")
    reserved_npus = getattr(args, "reserved_npus", None)
    control_reserved_npus = getattr(args, "control_reserved_npus", None)
    server_count = getattr(args, "server_count", None)
    control_server_count = getattr(args, "control_server_count", None)
    prefill_server_count = getattr(args, "prefill_server_count", None)
    decode_server_count = getattr(args, "decode_server_count", None)
    if reserved_npus is not None and reserved_npus < args.total_npus:
        raise ValueError("reserved_npus must be greater than or equal to total_npus")
    if (
        control_reserved_npus is not None
        and control_reserved_npus < args.control_total_npus
    ):
        raise ValueError(
            "control_reserved_npus must be greater than or equal to control_total_npus"
        )
    if server_count is not None and server_count <= 0:
        raise ValueError("server_count must be positive")
    if control_server_count is not None and control_server_count <= 0:
        raise ValueError("control_server_count must be positive")
    role_server_counts = (prefill_server_count, decode_server_count)
    if any(value is not None for value in role_server_counts):
        if server_count is None or any(value is None for value in role_server_counts):
            raise ValueError(
                "prefill_server_count, decode_server_count, and server_count "
                "must be declared together"
            )
        if prefill_server_count <= 0 or decode_server_count <= 0:
            raise ValueError(
                "prefill_server_count and decode_server_count must be positive"
            )
        if prefill_server_count + decode_server_count != server_count:
            raise ValueError(
                "prefill_server_count + decode_server_count must equal server_count"
            )
    if args.point == "control_graph_u1":
        if args.attention_npus != 0 or args.ffn_npus != 0:
            raise ValueError(
                "the control point must not declare separate Attention/FFN NPUs"
            )
    elif (
        args.attention_npus <= 0
        or args.ffn_npus <= 0
        or args.attention_npus + args.ffn_npus != args.decode_npus
    ):
        raise ValueError(
            "AFD attention_npus + ffn_npus must equal decode_npus and both be positive"
        )

    execution = {
        "mode": args.execution_mode,
        "u_batches": args.u_batches,
        "mtp": args.mtp,
        "mtp_num_speculative_tokens": args.mtp_num_speculative_tokens,
        "batch_invariant": False,
        "golden_checked": False,
    }
    if execution != EXPECTED_EXECUTION[args.point]:
        raise ValueError(
            f"execution signature mismatch for {args.point}: "
            f"expected={EXPECTED_EXECUTION[args.point]}, actual={execution}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema_version": 2,
        "passed": False,
        "measurement_passed": False,
        "acceptance_passed": None,
        "phase": args.phase,
        "point": args.point,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "execution": execution,
        "resource": _resource_manifest(args),
        "workload": {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "num_prompts": args.num_prompts,
            "concurrency": args.concurrency,
            "request_rate": "inf",
            "repeats": args.repeats,
            "seed": 1024,
            "temperature": 0,
            "ignore_eos": True,
        },
        "stack": {
            "cann_version": args.cann_version,
            "vllm_commit": args.vllm_commit,
            "vllm_ascend_commit": args.vllm_ascend_commit,
            "afd_commit": args.afd_commit,
            "model_path": str(args.tokenizer),
        },
        "records": [],
    }
    records: list[dict[str, Any]] = []
    try:
        _healthcheck(f"http://{args.host}:{args.port}/healthcheck", args.health_timeout)
        warmup_path = args.output_dir / "warmup.json"
        warmup_command = _benchmark_command(
            args,
            warmup_path,
            input_len=args.warmup_input_len,
            output_len=args.warmup_output_len,
            num_prompts=args.warmup_prompts,
            concurrency=args.warmup_concurrency,
            run_kind="warmup",
            repeat=0,
        )
        warmup_result = _run_command(
            warmup_command, warmup_path, args.benchmark_timeout
        )
        summary["warmup_gate"] = _validate_result(
            warmup_result,
            input_len=args.warmup_input_len,
            output_len=args.warmup_output_len,
            num_prompts=args.warmup_prompts,
        )
        if not summary["warmup_gate"]["passed"]:
            raise RuntimeError("warmup result gate failed")

        spec_metrics_before = _fetch_spec_metrics(args.metrics_url, args.health_timeout)

        for repeat in range(1, args.repeats + 1):
            result_path = args.output_dir / f"c{args.concurrency}_r{repeat}.json"
            command = _benchmark_command(
                args,
                result_path,
                input_len=args.input_len,
                output_len=args.output_len,
                num_prompts=args.num_prompts,
                concurrency=args.concurrency,
                run_kind="measurement",
                repeat=repeat,
            )
            result = _run_command(command, result_path, args.benchmark_timeout)
            gate = _validate_result(
                result,
                input_len=args.input_len,
                output_len=args.output_len,
                num_prompts=args.num_prompts,
            )
            record = {
                "repeat": repeat,
                "path": str(result_path),
                "gate": gate,
                "result": _compact_result(result),
            }
            records.append(record)
            if not gate["passed"]:
                raise RuntimeError(f"measurement result gate failed: {result_path}")
        summary["aggregate"] = _aggregate_results(records, args.total_npus)
        spec_metrics_after = _fetch_spec_metrics(args.metrics_url, args.health_timeout)
        summary["aggregate"]["spec_decode"] = _spec_metric_delta(
            spec_metrics_before, spec_metrics_after
        )
        if args.mtp and not summary["aggregate"]["spec_decode"]["valid"]:
            validation_errors = summary["aggregate"]["spec_decode"]["validation_errors"]
            raise RuntimeError(
                "MTP metrics did not prove active speculative decoding: "
                f"{validation_errors}; source={args.metrics_url}"
            )
        summary["measurement_passed"] = True
        summary["passed"] = True
    except BaseException as error:
        summary["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    finally:
        summary["records"] = records
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (args.output_dir / "performance_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _pct_change(current: float, baseline: float) -> float | None:
    return (current / baseline - 1.0) * 100.0 if baseline else None


def _point_metrics(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary["aggregate"]["metrics"]
    return {
        "request_throughput": float(metrics["request_throughput"]["mean"]),
        "output_throughput": float(metrics["output_throughput"]["mean"]),
        "output_tokens_per_second_per_npu": float(
            summary["aggregate"]["output_tokens_per_second_per_npu"]
        ),
        "p50_ttft_ms": float(metrics["p50_ttft_ms"]["mean"]),
        "p90_ttft_ms": float(metrics["p90_ttft_ms"]["mean"]),
        "p99_ttft_ms": float(metrics["p99_ttft_ms"]["mean"]),
        "p50_tpot_ms": float(metrics["p50_tpot_ms"]["mean"]),
        "p90_tpot_ms": float(metrics["p90_tpot_ms"]["mean"]),
        "p99_tpot_ms": float(metrics["p99_tpot_ms"]["mean"]),
    }


def _point_metric_cvs(summary: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(stats["cv"])
        for name, stats in summary["aggregate"]["metrics"].items()
    }


def _optional_pct_change(
    current: int | float | None, baseline: int | float | None
) -> float | None:
    if current is None or baseline is None:
        return None
    return _pct_change(float(current), float(baseline))


def _resource_delta(
    current: int | float | None, baseline: int | float | None
) -> dict[str, int | float | None]:
    return {
        "current": current,
        "baseline": baseline,
        "absolute_change": (
            current - baseline if current is not None and baseline is not None else None
        ),
        "change_pct": _optional_pct_change(current, baseline),
    }


def _comparison(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    current_metrics = _point_metrics(current)
    baseline_metrics = _point_metrics(baseline)
    current_active_npus = current["resource"].get(
        "active_npus", current["resource"]["total_npus"]
    )
    baseline_active_npus = baseline["resource"].get(
        "active_npus", baseline["resource"]["total_npus"]
    )
    current_reserved_npus = current["resource"].get("reserved_npus")
    baseline_reserved_npus = baseline["resource"].get("reserved_npus")
    current_server_count = current["resource"].get("server_count")
    baseline_server_count = baseline["resource"].get("server_count")
    result: dict[str, Any] = {
        "current": current["point"],
        "baseline": baseline["point"],
        "current_total_npus": current["resource"]["total_npus"],
        "baseline_total_npus": baseline["resource"]["total_npus"],
        "resource_increase_pct": _pct_change(
            float(current_active_npus),
            float(baseline_active_npus),
        ),
        "current_active_npus": current_active_npus,
        "baseline_active_npus": baseline_active_npus,
        "active_npu_increase_pct": _pct_change(
            float(current_active_npus), float(baseline_active_npus)
        ),
        "current_reserved_npus": current_reserved_npus,
        "baseline_reserved_npus": baseline_reserved_npus,
        "reserved_npu_increase_pct": _optional_pct_change(
            current_reserved_npus, baseline_reserved_npus
        ),
        "current_server_count": current_server_count,
        "baseline_server_count": baseline_server_count,
        "server_count_increase_pct": _optional_pct_change(
            current_server_count, baseline_server_count
        ),
        "resource_change": {
            "active_npus": _resource_delta(current_active_npus, baseline_active_npus),
            "reserved_npus": _resource_delta(
                current_reserved_npus, baseline_reserved_npus
            ),
            "server_count": _resource_delta(
                current_server_count, baseline_server_count
            ),
        },
        "metric_change_pct": {
            key: _pct_change(value, baseline_metrics[key])
            for key, value in current_metrics.items()
        },
    }
    added_npus = int(current_active_npus) - int(baseline_active_npus)
    result["incremental_output_tokens_per_second_per_added_npu"] = (
        (current_metrics["output_throughput"] - baseline_metrics["output_throughput"])
        / added_npus
        if added_npus > 0
        else None
    )
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_execution_signatures(
    by_point: dict[str, dict[str, Any]],
) -> None:
    for point in by_point:
        expected = EXPECTED_EXECUTION[point]
        actual = by_point[point].get("execution")
        if actual != expected:
            raise ValueError(
                f"execution signature mismatch for {point}: "
                f"expected={expected}, actual={actual}"
            )


def _gate(
    *,
    actual: float,
    threshold: float,
    operator: str,
    unit: str = "percent",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if operator == "<=":
        passed = actual <= threshold
    elif operator == ">=":
        passed = actual >= threshold
    elif operator == ">":
        passed = actual > threshold
    else:
        raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "passed": passed,
        "actual": actual,
        "operator": operator,
        "threshold": threshold,
        "unit": unit,
        **(details or {}),
    }


def _build_p2_acceptance_gates(
    by_point: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    control = by_point["control_graph_u1"]
    afd_u1 = by_point["afd_graph_u1"]
    afd_u2 = by_point["afd_graph_u2"]
    afd_u2_mtp1 = by_point["afd_graph_u2_mtp1"]
    point_cv_pct = {
        point: _point_metric_cvs(summary)["output_throughput"] * 100.0
        for point, summary in by_point.items()
    }
    u2_gain_pct = comparisons["u2_vs_u1"]["metric_change_pct"]["output_throughput"]
    u2_tpot_regression_pct = comparisons["u2_vs_u1"]["metric_change_pct"]["p99_tpot_ms"]
    u2_vs_control_gain_pct = comparisons["afd_u2_vs_control"]["metric_change_pct"][
        "output_throughput"
    ]
    u2_vs_control_tpot_regression_pct = comparisons["afd_u2_vs_control"][
        "metric_change_pct"
    ]["p99_tpot_ms"]
    mtp_gain_pct = comparisons["mtp1_vs_u2"]["metric_change_pct"]["output_throughput"]
    mtp_tpot_regression_pct = comparisons["mtp1_vs_u2"]["metric_change_pct"][
        "p99_tpot_ms"
    ]
    mtp_vs_control_gain_pct = comparisons["mtp1_vs_control"]["metric_change_pct"][
        "output_throughput"
    ]
    mtp_vs_control_tpot_regression_pct = comparisons["mtp1_vs_control"][
        "metric_change_pct"
    ]["p99_tpot_ms"]
    u2_variability_budget_pct = (
        point_cv_pct["afd_graph_u1"] + point_cv_pct["afd_graph_u2"]
    )
    control_variability_budget_pct = (
        point_cv_pct["control_graph_u1"] + point_cv_pct["afd_graph_u2"]
    )
    mtp_variability_budget_pct = (
        point_cv_pct["afd_graph_u2"] + point_cv_pct["afd_graph_u2_mtp1"]
    )
    mtp_control_variability_budget_pct = (
        point_cv_pct["control_graph_u1"] + point_cv_pct["afd_graph_u2_mtp1"]
    )
    return {
        "u2_throughput_gain_vs_u1": _gate(
            actual=u2_gain_pct,
            threshold=P2_MIN_U2_THROUGHPUT_GAIN_PCT,
            operator=">=",
        ),
        "u2_gain_exceeds_combined_variability": _gate(
            actual=u2_gain_pct,
            threshold=u2_variability_budget_pct,
            operator=">",
            details={
                "method": "gain_pct > baseline_cv_pct + current_cv_pct",
                "baseline": afd_u1["point"],
                "current": afd_u2["point"],
            },
        ),
        "u2_p99_tpot_regression_vs_u1": _gate(
            actual=u2_tpot_regression_pct,
            threshold=P2_MAX_P99_TPOT_REGRESSION_PCT,
            operator="<=",
        ),
        "afd_u2_gain_vs_control_exceeds_combined_variability": _gate(
            actual=u2_vs_control_gain_pct,
            threshold=control_variability_budget_pct,
            operator=">",
            details={
                "method": "gain_pct > baseline_cv_pct + current_cv_pct",
                "baseline": control["point"],
                "current": afd_u2["point"],
            },
        ),
        "afd_u2_p99_tpot_regression_vs_control": _gate(
            actual=u2_vs_control_tpot_regression_pct,
            threshold=P2_MAX_P99_TPOT_REGRESSION_PCT,
            operator="<=",
        ),
        "mtp1_gain_vs_u2_exceeds_combined_variability": _gate(
            actual=mtp_gain_pct,
            threshold=mtp_variability_budget_pct,
            operator=">",
            details={
                "method": "gain_pct > baseline_cv_pct + current_cv_pct",
                "baseline": afd_u2["point"],
                "current": afd_u2_mtp1["point"],
            },
        ),
        "mtp1_p99_tpot_regression_vs_u2": _gate(
            actual=mtp_tpot_regression_pct,
            threshold=P2_MAX_P99_TPOT_REGRESSION_PCT,
            operator="<=",
        ),
        "mtp1_gain_vs_control_exceeds_combined_variability": _gate(
            actual=mtp_vs_control_gain_pct,
            threshold=mtp_control_variability_budget_pct,
            operator=">",
            details={
                "method": "gain_pct > baseline_cv_pct + current_cv_pct",
                "baseline": control["point"],
                "current": afd_u2_mtp1["point"],
            },
        ),
        "mtp1_p99_tpot_regression_vs_control": _gate(
            actual=mtp_vs_control_tpot_regression_pct,
            threshold=P2_MAX_P99_TPOT_REGRESSION_PCT,
            operator="<=",
        ),
    }


def _build_comparison(
    summaries: list[dict[str, Any]],
    source_paths: list[Path] | None = None,
) -> dict[str, Any]:
    by_point = {summary.get("point"): summary for summary in summaries}
    missing = [point for point in POINT_ORDER if point not in by_point]
    if missing:
        raise ValueError(f"missing matrix points: {missing}")
    if len(by_point) != len(summaries):
        raise ValueError("duplicate matrix point")
    if not all(
        summary.get("measurement_passed", summary.get("passed")) is True
        for summary in summaries
    ):
        raise ValueError("all source performance measurements must pass")
    _validate_execution_signatures(by_point)
    mtp_spec_decode = (
        by_point["afd_graph_u2_mtp1"].get("aggregate", {}).get("spec_decode", {})
    )
    mtp_spec_errors = _spec_metric_validation_errors(mtp_spec_decode)
    if mtp_spec_decode.get("valid") is not True or mtp_spec_errors:
        raise ValueError(
            "afd_graph_u2_mtp1 does not contain valid speculative decode metrics: "
            f"{mtp_spec_errors or mtp_spec_decode.get('validation_errors')}"
        )
    phases = {summary.get("phase") for summary in summaries}
    workloads = {
        json.dumps(summary.get("workload"), sort_keys=True) for summary in summaries
    }
    required_stack_keys = (
        "cann_version",
        "vllm_commit",
        "vllm_ascend_commit",
        "afd_commit",
        "model_path",
    )
    stacks = {json.dumps(summary.get("stack"), sort_keys=True) for summary in summaries}
    if len(phases) != 1:
        raise ValueError(f"matrix phases differ: {sorted(phases)}")
    if len(workloads) != 1:
        raise ValueError("matrix workloads differ")
    if len(stacks) != 1:
        raise ValueError("matrix software stacks differ")
    stack = summaries[0].get("stack", {})
    missing_stack_values = [key for key in required_stack_keys if not stack.get(key)]
    if missing_stack_values:
        raise ValueError(f"matrix stack values are missing: {missing_stack_values}")

    phase = next(iter(phases))
    if phase not in ("p1", "p2"):
        raise ValueError(f"unsupported matrix phase: {phase}")
    expected_repeats = 1 if phase == "p1" else 3
    workload = summaries[0].get("workload", {})
    if workload.get("repeats") != expected_repeats:
        raise ValueError(
            f"{phase} comparison requires repeats={expected_repeats}, "
            f"got {workload.get('repeats')}"
        )

    if source_paths is not None and len(source_paths) != len(summaries):
        raise ValueError("source path count must match summary count")
    sources_by_point: dict[str, dict[str, str | None]] = {}
    for index, summary in enumerate(summaries):
        point = summary["point"]
        if source_paths is None:
            sources_by_point[point] = {"path": None, "sha256": None}
            continue
        source_path = source_paths[index]
        sources_by_point[point] = {
            "path": str(source_path.resolve()),
            "sha256": _file_sha256(source_path),
        }

    control = by_point["control_graph_u1"]
    vs_control = {
        point: _comparison(by_point[point], control) for point in POINT_ORDER[1:]
    }
    incremental = {
        "u2_vs_u1": _comparison(by_point["afd_graph_u2"], by_point["afd_graph_u1"]),
        "mtp1_vs_u2": _comparison(
            by_point["afd_graph_u2_mtp1"], by_point["afd_graph_u2"]
        ),
    }
    gate_comparisons = {
        "u2_vs_u1": incremental["u2_vs_u1"],
        "mtp1_vs_u2": incremental["mtp1_vs_u2"],
        "afd_u2_vs_control": vs_control["afd_graph_u2"],
        "mtp1_vs_control": vs_control["afd_graph_u2_mtp1"],
    }
    acceptance_gates = (
        _build_p2_acceptance_gates(by_point, gate_comparisons) if phase == "p2" else {}
    )
    acceptance_passed = (
        all(gate["passed"] for gate in acceptance_gates.values())
        if phase == "p2"
        else None
    )
    passed = True if acceptance_passed is None else acceptance_passed
    return {
        "schema_version": 2,
        "passed": passed,
        "measurement_passed": True,
        "acceptance_passed": acceptance_passed,
        "claim_status": (
            "p2_acceptance_passed"
            if acceptance_passed is True
            else "p2_acceptance_failed"
            if acceptance_passed is False
            else "p1_measurement_only"
        ),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": phase,
        "workload": control["workload"],
        "stack": dict(control["stack"]),
        "metric_semantics": {
            "capacity_metric": "output_throughput",
            "efficiency_metric": "output_tokens_per_second_per_npu",
            "efficiency_denominator": "active_npus",
            "efficiency_is_acceptance_gate": False,
        },
        "sources": {point: sources_by_point[point] for point in POINT_ORDER},
        "points": {
            point: {
                "execution": by_point[point]["execution"],
                "resource": by_point[point]["resource"],
                "metrics": _point_metrics(by_point[point]),
                "metric_cv": _point_metric_cvs(by_point[point]),
                "spec_decode": by_point[point]["aggregate"]["spec_decode"],
            }
            for point in POINT_ORDER
        },
        "vs_control": vs_control,
        "incremental": incremental,
        "acceptance_gates": acceptance_gates,
    }


def _build_ratio_comparison(
    summaries: list[dict[str, Any]],
    source_paths: list[Path] | None = None,
) -> dict[str, Any]:
    by_point = {summary.get("point"): summary for summary in summaries}
    missing = [point for point in RATIO_POINT_ORDER if point not in by_point]
    unexpected = [point for point in by_point if point not in RATIO_POINT_ORDER]
    if missing or unexpected:
        raise ValueError(
            "AF ratio comparison point mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if len(by_point) != len(summaries):
        raise ValueError("duplicate AF ratio point")
    if not all(
        summary.get("measurement_passed", summary.get("passed")) is True
        for summary in summaries
    ):
        raise ValueError("all AF ratio performance measurements must pass")
    _validate_execution_signatures(by_point)

    phases = {summary.get("phase") for summary in summaries}
    workloads = {
        json.dumps(summary.get("workload"), sort_keys=True) for summary in summaries
    }
    stacks = {json.dumps(summary.get("stack"), sort_keys=True) for summary in summaries}
    if len(phases) != 1 or len(workloads) != 1 or len(stacks) != 1:
        raise ValueError("AF ratio points must use the same phase, workload, and stack")
    phase = next(iter(phases))
    if phase not in {"p1", "p2"}:
        raise ValueError(f"unsupported AF ratio phase: {phase}")
    expected_repeats = 1 if phase == "p1" else 3
    workload = summaries[0].get("workload", {})
    if workload.get("repeats") != expected_repeats:
        raise ValueError(
            f"{phase} comparison requires repeats={expected_repeats}, "
            f"got {workload.get('repeats')}"
        )
    if source_paths is not None and len(source_paths) != len(summaries):
        raise ValueError("source path count must match summary count")

    sources: dict[str, dict[str, str | None]] = {}
    for index, summary in enumerate(summaries):
        point = summary["point"]
        if source_paths is None:
            sources[point] = {"path": None, "sha256": None}
        else:
            source_path = source_paths[index]
            sources[point] = {
                "path": str(source_path.resolve()),
                "sha256": _file_sha256(source_path),
            }

    colocated = by_point["afd_graph_u2"]
    split_a8f8 = by_point["afd_graph_u2_split_a8f8"]
    split_a16f8 = by_point["afd_graph_u2_split_a16f8"]
    comparisons = {
        "placement_penalty_split_a8f8_vs_colocated_a8f8": _comparison(
            split_a8f8, colocated
        ),
        "ratio_gain_a16f8_vs_split_a8f8": _comparison(split_a16f8, split_a8f8),
        "end_to_end_a16f8_vs_colocated_a8f8": _comparison(split_a16f8, colocated),
    }
    return {
        "schema_version": 2,
        "passed": True,
        "measurement_passed": True,
        "acceptance_passed": None,
        "claim_status": "measurement_only_no_fixed_gain_threshold",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": phase,
        "workload": workload,
        "stack": dict(summaries[0]["stack"]),
        "sources": {point: sources[point] for point in RATIO_POINT_ORDER},
        "points": {
            point: {
                "execution": by_point[point]["execution"],
                "resource": by_point[point]["resource"],
                "metrics": _point_metrics(by_point[point]),
                "metric_cv": _point_metric_cvs(by_point[point]),
            }
            for point in RATIO_POINT_ORDER
        },
        "comparisons": comparisons,
    }


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point", choices=RUN_POINT_ORDER, required=True)
    parser.add_argument("--phase", choices=("p1", "p2"), required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--served-model", default="dsv4-afd")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--execution-mode", default="full-decode-only")
    parser.add_argument("--u-batches", type=int, choices=(1, 2), required=True)
    parser.add_argument("--mtp", type=int, choices=(0, 1), required=True)
    parser.add_argument("--mtp-num-speculative-tokens", type=int, default=1)
    parser.add_argument("--repeats", type=int, choices=(1, 3), required=True)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--warmup-input-len", type=int, default=256)
    parser.add_argument("--warmup-output-len", type=int, default=16)
    parser.add_argument("--warmup-prompts", type=int, default=16)
    parser.add_argument("--warmup-concurrency", type=int, default=8)
    parser.add_argument("--health-timeout", type=float, default=10)
    parser.add_argument("--benchmark-timeout", type=float, default=3600)
    parser.add_argument("--prefill-npus", type=int, required=True)
    parser.add_argument("--decode-npus", type=int, required=True)
    parser.add_argument("--attention-npus", type=int, required=True)
    parser.add_argument("--ffn-npus", type=int, required=True)
    parser.add_argument("--total-npus", type=int, required=True)
    parser.add_argument("--control-total-npus", type=int, required=True)
    parser.add_argument("--reserved-npus", type=int)
    parser.add_argument("--control-reserved-npus", type=int)
    parser.add_argument("--server-count", type=int)
    parser.add_argument("--control-server-count", type=int)
    parser.add_argument("--prefill-server-count", type=int)
    parser.add_argument("--decode-server-count", type=int)
    parser.add_argument("--cann-version", required=True)
    parser.add_argument("--vllm-commit", required=True)
    parser.add_argument("--vllm-ascend-commit", required=True)
    parser.add_argument("--afd-commit", required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    _add_run_args(run_parser)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--input", type=Path, action="append", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    ratio_parser = subparsers.add_parser("compare-ratio")
    ratio_parser.add_argument("--input", type=Path, action="append", required=True)
    ratio_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "run":
        summary = _run(args)
        output_path = args.output_dir / "performance_summary.json"
        print(
            f"passed={summary['passed']} output={output_path}",
            flush=True,
        )
        return 0 if summary["passed"] else 1

    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    if args.command == "compare-ratio":
        comparison = _build_ratio_comparison(summaries, args.input)
    else:
        comparison = _build_comparison(summaries, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"measurement_passed={comparison['measurement_passed']} "
        f"acceptance_passed={comparison['acceptance_passed']} "
        f"output={args.output}",
        flush=True,
    )
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
