#!/usr/bin/env python3
"""Run DeepSeek-V4 functional requests without a golden comparison."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _request_batch(
    endpoint: str,
    model: str,
    batch_size: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    prompts = [
        f"DeepSeek-V4 functional smoke batch {batch_size}, request {index}."
        for index in range(batch_size)
    ]
    payload = {
        "model": model,
        "prompt": prompts,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": 1024,
        "stream": False,
        "return_token_ids": True,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {details}") from error


def _validate_response(response: dict[str, Any], batch_size: int) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != batch_size:
        raise ValueError(
            f"expected {batch_size} completion choices, got "
            f"{len(choices) if isinstance(choices, list) else type(choices).__name__}"
        )

    token_counts: list[int] = []
    for choice_index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            raise ValueError(f"choice {choice_index} is not an object")
        token_ids = choice.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"choice {choice_index} has no output token IDs")
        if not all(
            isinstance(token_id, int) and token_id >= 0 for token_id in token_ids
        ):
            raise ValueError(f"choice {choice_index} has invalid output token IDs")
        prompt_token_ids = choice.get("prompt_token_ids")
        if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
            raise ValueError(f"choice {choice_index} has no prompt token IDs")
        token_counts.append(len(token_ids))

    return {
        "batch_size": batch_size,
        "choice_count": len(choices),
        "output_token_counts": token_counts,
        "usage": response.get("usage", {}),
    }


def _parse_batch_sizes(value: str) -> list[int]:
    sizes = [int(item) for item in value.split()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("batch sizes must contain positive integers")
    if len(set(sizes)) != len(sizes):
        raise ValueError("batch sizes must not contain duplicates")
    return sizes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-sizes", default="1 8 32")
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    report: dict[str, Any] = {
        "started_at": started_at,
        "endpoint": args.endpoint,
        "model": args.model,
        "golden_checked": False,
        "results": [],
    }
    try:
        batch_sizes = _parse_batch_sizes(args.batch_sizes)
        for batch_size in batch_sizes:
            started = time.monotonic()
            response = _request_batch(
                args.endpoint,
                args.model,
                batch_size,
                args.max_tokens,
                args.timeout,
            )
            result = _validate_response(response, batch_size)
            result["elapsed_seconds"] = time.monotonic() - started
            report["results"].append(result)
        report["passed"] = True
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"

    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"passed={report['passed']} output={args.output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
