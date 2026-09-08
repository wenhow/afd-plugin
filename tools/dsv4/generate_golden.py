#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _request_completion(
    endpoint: str,
    model: str,
    prompt: str,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 16,
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
            body = json.load(response)
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {details}") from error

    choice = body["choices"][0]
    return {
        "prompt_token_ids": choice["prompt_token_ids"],
        "token_ids": choice["token_ids"],
        "text": choice["text"],
        "finish_reason": choice["finish_reason"],
        "usage": body["usage"],
    }


def _load_prompt_source(path: Path) -> tuple[list[str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts")
    if isinstance(prompts, list) and prompts and all(
        isinstance(prompt, str) and prompt for prompt in prompts
    ):
        return prompts, {}
    golden = payload.get("golden")
    if not isinstance(golden, dict) or not golden:
        raise ValueError(f"prompt source has no golden records: {path}")
    prompts = [golden[str(index)]["prompt"] for index in range(len(golden))]
    return prompts, golden


def _parse_metadata(entries: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for entry in entries:
        key, separator, value = entry.partition("=")
        if not separator or not key or key in metadata:
            raise ValueError(f"invalid or duplicate metadata entry: {entry}")
        metadata[key] = value
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--metadata", action="append", default=[])
    args = parser.parse_args()
    if args.rounds <= 0:
        parser.error("--rounds must be positive")

    prompts, reference_golden = _load_prompt_source(args.prompt_source)
    metadata = _parse_metadata(args.metadata)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    records: list[dict[str, Any]] = []
    for round_index in range(args.rounds):
        for prompt_index, prompt in enumerate(prompts):
            result = _request_completion(
                args.endpoint,
                args.model,
                prompt,
                args.timeout,
            )
            records.append(
                {
                    "round": round_index + 1,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "matched_reference": bool(reference_golden)
                    and (
                        result["prompt_token_ids"]
                        == reference_golden[str(prompt_index)]["prompt_token_ids"]
                        and result["token_ids"]
                        == reference_golden[str(prompt_index)]["token_ids"]
                    ),
                    **result,
                }
            )
            print(
                f"round={round_index + 1} prompt={prompt_index:02d} "
                f"tokens={result['token_ids']}",
                flush=True,
            )

    golden: dict[str, Any] = {}
    mismatches: list[int] = []
    for prompt_index, prompt in enumerate(prompts):
        runs = [record for record in records if record["prompt_index"] == prompt_index]
        expected = runs[0]["token_ids"]
        expected_prompt = runs[0]["prompt_token_ids"]
        stable = all(
            run["prompt_token_ids"] == expected_prompt and run["token_ids"] == expected
            for run in runs[1:]
        )
        if not stable:
            mismatches.append(prompt_index)
        golden[str(prompt_index)] = {
            "prompt": prompt,
            "prompt_token_ids": runs[0]["prompt_token_ids"],
            "token_ids": expected,
            "text": runs[0]["text"],
            "stable_across_rounds": stable,
        }

    report = {
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "endpoint": args.endpoint,
        "model": args.model,
        "prompt_source": str(args.prompt_source),
        "metadata": metadata,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 16,
            "seed": 1024,
        },
        "rounds": args.rounds,
        "prompt_count": len(prompts),
        "passed": not mismatches,
        "mismatched_prompt_indices": mismatches,
        "reference_comparison": {
            "exact_match_count": sum(
                bool(record["matched_reference"]) for record in records
            ),
            "request_count": len(records),
        },
        "golden": golden,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"passed={report['passed']} output={args.output}", flush=True)
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
