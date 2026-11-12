#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Snapshot installed quantization and DSpark sources without runtime imports.

Uses only stdlib metadata, filesystem reads and read-only Git commands. It never
imports vLLM/torch, loads operator libraries, runs npu-smi or starts a service.
The fresh output directory and archive are local evidence, not source delivery.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
SOURCE_PATHS = (
    "quantization/methods/w4a8/w4a8_mxfp4.py",
    "worker/v2/spec_decode/dspark/speculator.py",
    "ascend_forward_context.py",
    "ops/fused_moe/dataclass/fused_experts.py",
)
GIT_TIMEOUT_SECONDS = 10


def git_result(directory: Path, *arguments: str) -> dict[str, JSONValue]:
    try:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(directory), *arguments],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="parent for a fresh dsv41-a5-sources-* directory and .tar.gz archive",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="dsv41-a5-sources-", dir=args.output_root))
    report: dict[str, JSONValue] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Installed source bytes only; no runtime or device acceptance",
        "python_executable": sys.executable,
        "python_version": sys.version,
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": {},
        "errors": [],
    }
    errors: list[JSONValue] = []
    sources: dict[str, JSONValue] = {}
    try:
        report["distribution_version"] = importlib.metadata.version("vllm-ascend")
    except importlib.metadata.PackageNotFoundError:
        report["distribution_version"] = None
    try:
        # Looking up the top-level module does not execute its __init__.py.
        spec = importlib.util.find_spec("vllm_ascend")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("Installed vllm_ascend package not found")
        roots = list(spec.submodule_search_locations)
        if len(roots) != 1:
            raise ImportError(f"Expected one installed vllm_ascend root, got {roots}")
        root = Path(roots[0]).resolve()
        report["package_origin"] = spec.origin
        report["package_root"] = str(root)
        report["package_git"] = {
            "root": git_result(root, "rev-parse", "--show-toplevel"),
            "head": git_result(root, "rev-parse", "HEAD"),
            "source_status": git_result(
                root, "status", "--short", "--untracked-files=all", "--", *SOURCE_PATHS
            ),
        }
        for relative in SOURCE_PATHS:
            source = root / relative
            try:
                content = source.read_bytes()
                destination = output / "sources" / "vllm_ascend" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                sources[relative] = {
                    "installed_path": str(source),
                    "resolved_path": str(source.resolve()),
                    "snapshot_path": destination.relative_to(output).as_posix(),
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            except OSError as error:
                errors.append(f"{source}: {error}")
    except (ImportError, ValueError, OSError) as error:
        errors.append(str(error))
    report["sources"] = sources
    report["errors"] = errors
    report["passed"] = not errors and len(sources) == len(SOURCE_PATHS)
    exit_code = 0 if report["passed"] else 1
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "collector-exit-code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
    archive = output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(output, arcname=output.name)
    print(f"results: {output.resolve()}")
    print(f"archive: {archive.resolve()}")
    print(f"archive_sha256: {hashlib.sha256(archive.read_bytes()).hexdigest()}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
