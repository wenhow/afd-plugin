#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Collect sequential requests from an already running native DSpark service.

Uses only the standard library. It never starts or stops a model service.
Response metrics follow vLLM ced6857a RequestSpecDecodeMetrics.to_dict;
missing metrics are an incomplete result, never an acceptance count of zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from datetime import UTC, datetime
from http.client import HTTPException, IncompleteRead
from pathlib import Path
from typing import TypeAlias
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT = 300.0
DEFAULT_METADATA_TIMEOUT = 10.0
DEFAULT_PROMPT = "Explain how to keep a simple daily work log."


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_timeout(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive timeout")
    return number


def write_json(path: Path, payload: JSONValue) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def fetch(
    url: str, path: Path, *, timeout: float, payload: bytes | None = None
) -> dict[str, JSONValue]:
    request = Request(url, data=payload)
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    status = None
    error = None
    body = b""
    try:
        try:
            response = urlopen(request, timeout=timeout)
        except HTTPError as exc:
            response = exc
        with response:
            status = response.status
            if status != 200:
                error = f"HTTP {status}"
            body = response.read(MAX_RESPONSE_BYTES + 1)
            declared_length = response.headers.get("Content-Length")
            if declared_length is not None:
                try:
                    expected_length = int(declared_length)
                except ValueError:
                    error = "invalid HTTP Content-Length"
                else:
                    if expected_length < 0:
                        error = "invalid HTTP Content-Length"
                    elif len(body) < expected_length:
                        error = "incomplete HTTP response body"
    except IncompleteRead as exc:
        body = exc.partial
        error = f"incomplete HTTP response: {exc}"
    except (URLError, TimeoutError, OSError, HTTPException) as exc:
        error = str(exc)
    truncated = len(body) > MAX_RESPONSE_BYTES
    if truncated:
        error = f"response exceeded {MAX_RESPONSE_BYTES} bytes"
    path.write_bytes(body[:MAX_RESPONSE_BYTES])
    return {"status": status, "error": error, "truncated": truncated}


def nonnegative_int(value: JSONValue, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def integer_list(value: JSONValue, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an integer array")
    return [nonnegative_int(item, name) for item in value]


def validate_response(
    payload: JSONValue, *, spec_tokens: int, max_tokens: int
) -> dict[str, JSONValue]:
    """Validate the actual response, including zero accepted draft tokens."""
    if not isinstance(payload, dict):
        raise ValueError("completion response must be a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("completion response requires exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("completion choice must be an object")
    token_ids = integer_list(choice.get("token_ids"), "choices[0].token_ids")
    if not token_ids or len(token_ids) > max_tokens:
        raise ValueError("output token count must be between 1 and max_tokens")
    finish_reason = choice.get("finish_reason")
    if finish_reason not in ("stop", "length"):
        raise ValueError("completion response requires finish_reason stop or length")
    usage = payload.get("usage")
    if not isinstance(usage, dict) or nonnegative_int(
        usage.get("completion_tokens"), "usage.completion_tokens"
    ) != len(token_ids):
        raise ValueError("usage.completion_tokens disagrees with output token IDs")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(
        metrics.get("speculative_decoding"), dict
    ):
        raise ValueError("metrics.speculative_decoding is missing")
    spec = metrics["speculative_decoding"]
    configured = nonnegative_int(spec.get("num_spec_tokens"), "num_spec_tokens")
    steps = nonnegative_int(spec.get("num_spec_steps"), "num_spec_steps")
    drafted = nonnegative_int(spec.get("num_draft_tokens"), "num_draft_tokens")
    accepted = nonnegative_int(
        spec.get("num_accepted_draft_tokens"), "num_accepted_draft_tokens"
    )
    histogram = integer_list(spec.get("acceptance_histogram"), "acceptance_histogram")
    per_accepted = integer_list(spec.get("per_step_accepted"), "per_step_accepted")
    per_drafted = integer_list(spec.get("per_step_drafted"), "per_step_drafted")
    if configured != spec_tokens:
        raise ValueError("response num_spec_tokens differs from --spec-tokens")
    if steps == 0 or drafted == 0:
        raise ValueError("response reports no effective speculative proposal")
    if len(histogram) != configured + 1 or sum(histogram) != steps:
        raise ValueError("acceptance_histogram disagrees with num_spec_steps")
    if len(per_accepted) != steps or len(per_drafted) != steps:
        raise ValueError("detailed per-step arrays are missing or incomplete")
    observed_histogram = [0] * (configured + 1)
    for count, proposed in zip(per_accepted, per_drafted, strict=True):
        if not 0 <= count <= proposed <= configured:
            raise ValueError("per-step counts violate accepted <= drafted <= N")
        observed_histogram[count] += 1
    if (
        sum(per_accepted) != accepted
        or sum(per_drafted) != drafted
        or observed_histogram != histogram
    ):
        raise ValueError("detailed per-step counts disagree with aggregate metrics")
    rate = accepted / drafted
    mean_length = 1 + accepted / steps
    for name, expected in (
        ("draft_acceptance_rate", rate),
        ("mean_acceptance_length", mean_length),
    ):
        value = spec.get(name)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise ValueError(f"{name} disagrees with actual counts")
    return {
        "response_id": payload.get("id"),
        "finish_reason": finish_reason,
        "token_ids": token_ids,
        "completion_tokens": len(token_ids),
        "speculative_decoding": spec,
    }


def collect_request(
    directory: Path, args: argparse.Namespace, max_tokens: int
) -> dict[str, JSONValue]:
    directory.mkdir()
    request: dict[str, JSONValue] = {
        "model": args.model,
        "prompt": args.prompt,
        "temperature": 0,
        "seed": args.seed,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "n": 1,
        "return_token_ids": True,
        "stream": False,
    }
    write_json(directory / "request.json", request)
    before = fetch(
        args.base_url + "/metrics",
        directory / "metrics-before.txt",
        timeout=DEFAULT_METADATA_TIMEOUT,
    )
    response = fetch(
        args.base_url + "/v1/completions",
        directory / "response.json",
        timeout=args.timeout,
        payload=json.dumps(request).encode("utf-8"),
    )
    after = fetch(
        args.base_url + "/metrics",
        directory / "metrics-after.txt",
        timeout=DEFAULT_METADATA_TIMEOUT,
    )
    report: dict[str, JSONValue] = {
        "directory": directory.name,
        "max_tokens": max_tokens,
        "http": {"before": before, "completion": response, "after": after},
        "valid": False,
    }
    try:
        if response["status"] != 200 or response["error"] is not None:
            raise ValueError("completion HTTP request failed; see response.json/http")
        decoded: JSONValue = json.loads((directory / "response.json").read_bytes())
        report["response"] = validate_response(
            decoded, spec_tokens=args.spec_tokens, max_tokens=max_tokens
        )
        if any(item["status"] != 200 or item["error"] for item in (before, after)):
            raise ValueError("Prometheus snapshot failed; see http and metrics files")
        report["valid"] = True
    except (ValueError, UnicodeError) as exc:
        report["error"] = str(exc)
    write_json(directory / "summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18030")
    parser.add_argument("--model", default="dsv41-native")
    parser.add_argument("--spec-tokens", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--max-tokens", type=positive_int, nargs="+", default=[64])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--timeout", type=positive_timeout, default=DEFAULT_REQUEST_TIMEOUT
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        parsed_url = urlsplit(args.base_url)
        if parsed_url.port == 0:
            parser.error("--base-url port must be positive")
    except ValueError as exc:
        parser.error(f"invalid --base-url: {exc}")
    if (
        parsed_url.scheme not in ("http", "https")
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.path not in ("", "/")
        or parsed_url.query
        or parsed_url.fragment
    ):
        parser.error("--base-url requires an HTTP(S) origin without credentials/query")
    args.base_url = args.base_url.rstrip("/")
    args.output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="dspark-requests-", dir=args.output_root))
    print(directory, flush=True)
    repo = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
        )
        head = revision.stdout.strip() if revision.returncode == 0 else None
    except OSError:
        head = None
    summary: dict[str, JSONValue] = {
        "scope": "Native HTTP DSpark checks; not AFD or hardware qualification",
        "started_utc": datetime.now(UTC).isoformat(),
        "git_head": head,
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_url": args.base_url,
        "model": args.model,
        "spec_tokens": args.spec_tokens,
        "requested_lengths": args.max_tokens,
        "requests": [],
        "success": False,
    }
    health = fetch(
        args.base_url + "/health",
        directory / "health.txt",
        timeout=DEFAULT_METADATA_TIMEOUT,
    )
    summary["health"] = health
    reports = []
    if health["status"] == 200 and health["error"] is None:
        for index, max_tokens in enumerate(args.max_tokens, start=1):
            report = collect_request(
                directory / f"request-{index:03d}", args, max_tokens
            )
            reports.append(report)
            if not report["valid"]:
                break
    summary["requests"] = reports
    summary["success"] = len(reports) == len(args.max_tokens) and all(
        report["valid"] for report in reports
    )
    write_json(directory / "summary.json", summary)
    result = 0 if summary["success"] else 1
    (directory / "exit-code.txt").write_text(f"{result}\n", encoding="utf-8")
    print(f"requests={len(reports)}/{len(args.max_tokens)}; exit={result}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
