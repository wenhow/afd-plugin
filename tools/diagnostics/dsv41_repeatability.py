# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Serial HTTP repeatability diagnostics; no model imports or runtime patches."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import run_dsv41_a5_afd as launch

DEFAULT_WARMUP_ROUNDS = 1
DEFAULT_REPEAT_ROUNDS = 3
MIN_WARMUP_ROUNDS = 1
MIN_REPEAT_ROUNDS = 2
MAX_ACTIVE_REQUESTS = 1
LOGPROB_CANDIDATES = 5
PROTOCOL_VERSION = 1


def validate_logprobs(response: dict) -> None:
    """Require usable token-ID top-k scores, rather than silently losing evidence."""
    choice = response["choices"][0]
    scores = choice["logprobs"]
    tokens = choice["token_ids"]
    if (
        not scores
        or scores["tokens"] != [f"token_id:{token}" for token in tokens]
        or len(scores["token_logprobs"]) != len(tokens)
        or len(scores["top_logprobs"]) != len(tokens)
    ):
        raise ValueError("repeatability requires token-ID logprobs for every output")
    for selected, candidates in zip(
        scores["token_logprobs"], scores["top_logprobs"], strict=True
    ):
        if (
            not isinstance(candidates, dict)
            or len(candidates) < LOGPROB_CANDIDATES
            or any(not name.startswith("token_id:") for name in candidates)
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in (selected, *candidates.values())
            )
        ):
            raise ValueError("repeatability requires finite top-5 logprobs")


def token_logprobs_at(response: dict, index: int) -> dict | None:
    scores = response["choices"][0].get("logprobs")
    if scores is None:
        return None
    candidates = scores["top_logprobs"][index]
    ranked = sorted(candidates.items(), key=lambda pair: pair[1], reverse=True)
    return {
        "selected_logprob": scores["token_logprobs"][index],
        "ranked_candidates": ranked,
        "top1_minus_top2": ranked[0][1] - ranked[1][1],
    }


def compare_responses(actual: Path, reference: Path) -> dict:
    response = json.loads((actual / "response.json").read_bytes())
    baseline = json.loads((reference / "response.json").read_bytes())
    comparison = launch.compare_response_tokens(response, baseline)
    comparison.update(actual_directory=str(actual), reference_directory=str(reference))
    mismatch = comparison["first_output_mismatch"]
    if mismatch is not None:
        index = mismatch["index"]
        divergence = {
            name: scores
            for name, body in (("actual", response), ("reference", baseline))
            if (scores := token_logprobs_at(body, index)) is not None
        }
        comparison["first_divergence_logprobs"] = divergence or None
    return comparison


def run_repeatability(
    directory: Path,
    reference: Path | None,
    *,
    api_url: str,
    protocol: dict,
    cancel: Callable[[Path], None],
) -> dict:
    """Collect all token mismatches, then fail strictly after saving the matrix.

    HTTP/schema failures stop immediately. Token differences remain failures but
    do not hide post-cancel evidence. Warmup differences are informational; only
    measured before/after rounds determine repeatability. A serial client and
    max-num-seqs=1 constrain active requests without claiming a two-request
    HTTP wave has a deterministic internal scheduler batch.
    """
    report = {
        "protocol": protocol,
        "diagnostic_only": True,
        "hardware_acceptance": False,
        "model_equivalence_accepted": False,
        "completed": False,
        "passed": False,
        "requests": [],
        "comparisons": [],
        "cancellation": "not_started",
    }
    report_path = directory / "repeatability.json"
    launch.save_json(report_path, report)
    try:
        phases = (
            ("warmup", protocol["warmup_rounds"]),
            ("before-cancel", protocol["repeat_rounds"]),
            ("after-cancel", protocol["repeat_rounds"]),
        )
        for phase, rounds in phases:
            if phase == "after-cancel":
                print("repeatability: cancelling a stream", flush=True)
                cancel(directory)
                report["cancellation"] = "client_closed_after_first_text_chunk"
                # Closing the client is not proof of engine-side abort completion.
                launch.save_json(report_path, report)
            for round_index in range(rounds):
                for prompt_index, prompt in enumerate(protocol["prompts"]):
                    relative = Path(f"{phase}-{round_index:02d}") / (
                        f"request-{prompt_index:02d}"
                    )
                    target = directory / relative
                    target.mkdir(parents=True)
                    print(f"repeatability: {relative.as_posix()}", flush=True)
                    payload = {
                        "model": "dsv41-afd",
                        "prompt": prompt,
                        "temperature": 0,
                        "seed": 0,
                        "max_tokens": launch.REQUEST_TOKENS,
                        "ignore_eos": True,
                        "n": 1,
                        "return_token_ids": True,
                        "stream": False,
                    }
                    if protocol["logprobs"]:
                        payload.update(
                            logprobs=protocol["logprobs"],
                            return_tokens_as_token_ids=True,
                        )
                    # Keep send_request's qualification gate unchanged. Exact
                    # comparisons are aggregated below after all HTTP responses.
                    result = launch.send_request(
                        target, None, payload=payload, api_url=api_url
                    )
                    response = json.loads((target / "response.json").read_bytes())
                    if protocol["logprobs"]:
                        validate_logprobs(response)
                    choice = response["choices"][0]
                    if not choice["prompt_token_ids"] or any(
                        type(token) is not int or token < 0
                        for token in choice["prompt_token_ids"]
                    ):
                        raise ValueError(
                            "repeatability requires valid prompt token IDs"
                        )
                    report["requests"].append(
                        {
                            "directory": relative.as_posix(),
                            "phase": phase,
                            "round": round_index,
                            "prompt": prompt_index,
                            **result,
                        }
                    )
                    launch.save_json(report_path, report)
        for request in report["requests"]:
            target = directory / request["directory"]
            baseline = (
                directory / "before-cancel-00" / f"request-{request['prompt']:02d}"
            )
            checks = [("within_run", baseline)]
            if reference is not None:
                checks.append(("native_reference", reference / request["directory"]))
            for kind, baseline in checks:
                comparison = compare_responses(target, baseline)
                comparison.update(kind=kind, phase=request["phase"])
                report["comparisons"].append(comparison)
                launch.save_json(target / f"{kind}-comparison.json", comparison)

        def matched(kind: str, phases: tuple[str, ...]) -> bool:
            checks = [
                item
                for item in report["comparisons"]
                if item["kind"] == kind and item["phase"] in phases
            ]
            return bool(checks) and all(
                item["status"] == "tokens_match" for item in checks
            )

        report.update(
            completed=True,
            warmup_matches_measured_baseline=matched("within_run", ("warmup",)),
            before_cancel_stable=matched("within_run", ("before-cancel",)),
            after_cancel_matches_before=matched("within_run", ("after-cancel",)),
            native_tokens_match=(
                matched("native_reference", ("before-cancel", "after-cancel"))
                if reference is not None
                else None
            ),
        )
        report["passed"] = (
            report["before_cancel_stable"]
            and report["after_cancel_matches_before"]
            and report["native_tokens_match"] is not False
        )
        launch.save_json(report_path, report)
        if not report["passed"]:
            raise RuntimeError(
                "repeatability token comparison failed; see repeatability.json"
            )
        return report
    except Exception as error:
        report.update(passed=False, error=f"{type(error).__name__}: {error}")
        launch.save_json(report_path, report)
        raise
