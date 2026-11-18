#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the local frozen-harness checks without importing a device runtime."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
DIRECTORY = Path(__file__).resolve().parent


def run(command: list[str]) -> dict:
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main() -> None:
    results = []
    with tempfile.TemporaryDirectory(
        prefix="a5-graph-rebuild-", dir=ROOT / ".tmp"
    ) as name:
        output = Path(name) / "bundle"
        results.append(
            run(
                [
                    sys.executable,
                    "-B",
                    "tools/diagnostics/prepare_dsv41_a5_functional_bundle.py",
                    "--output-dir",
                    str(output),
                ]
            )
        )
        artifacts = ("runtime-source.tar.gz", "bundle-lock.json", "preparation.json")
        deterministic = {
            artifact: (
                (output / artifact).read_bytes()
                == (ROOT / "reports/dsv41-a5-functional" / artifact).read_bytes()
            )
            for artifact in artifacts
        }
    commands = (
        [
            sys.executable,
            "-B",
            "-S",
            "-m",
            "unittest",
            "tests.e2e.diagnostics.test_dsv41_a5_graph_observer",
            "tests.e2e.diagnostics.test_dsv41_a5_functional_validation",
        ],
        [
            sys.executable,
            "-B",
            "-m",
            "ruff",
            "check",
            "tools/diagnostics/prepare_dsv41_a5_functional_bundle.py",
            "tests/e2e/diagnostics/test_dsv41_a5_graph_observer.py",
            "tests/npu/hccl_native_attention_pd_dspark_graph_probe.py",
            str(DIRECTORY / "assess_report.py"),
            str(Path(__file__).resolve()),
        ],
        [sys.executable, "-B", ".agents/dsv41/harness.py", "check"],
        [sys.executable, "-B", ".agents/dsv41/harness.py", "guard"],
    )
    for command in commands:
        result = run(command)
        results.append(result)
        print(result["returncode"], " ".join(command), flush=True)
    result = {
        "status": "local_host_checks_not_NPU_acceptance",
        "checks": results,
        "deterministic_rebuild": deterministic,
        "runtime_source_sha256": hashlib.sha256(
            (ROOT / "reports/dsv41-a5-functional/runtime-source.tar.gz").read_bytes()
        ).hexdigest(),
        "A5_replay_cancel_replace_clean_exit": "pending",
    }
    (DIRECTORY / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    assert all(row["returncode"] == 0 for row in results)
    assert all(deterministic.values())


if __name__ == "__main__":
    main()
