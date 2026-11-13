#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Collect one bounded A5 single-layer TP1/TP4 MoE diagnostic archive.

The default only inspects checkpoint metadata and source files. --run explicitly
loads one layer in sequential native TP1/TP4 phases; --setup-only with --run
checks imports/configuration without loading weights or executing kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import tempfile
import traceback
from contextlib import suppress
from pathlib import Path

from run_dsv41_a5_afd import (
    Supervision,
    handle_termination,
    pack_run,
)

PROBE = "tests/npu/dsv41_moe_parity_probe.py"
PROBE_TIMEOUT_SECONDS = 600
OUTER_TIMEOUT_SECONDS = 720
INTERRUPT_CLEANUP_SECONDS = 60
HASH_CHUNK_BYTES = 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--transformer-vendor", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()
    if args.setup_only and not args.run:
        parser.error("setup-only requires explicit --run")
    repo = Path(__file__).resolve().parents[2]
    args.output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(prefix="dsv41-a5-moe-parity-", dir=args.output_root)
    ).resolve()
    supervision = Supervision(directory)
    previous_handler = signal.signal(signal.SIGTERM, handle_termination)
    command = [
        sys.executable,
        "-B",
        str(repo / PROBE),
        "--model",
        str(args.model.resolve()),
        "--output-dir",
        str(directory / "probe"),
        "--devices",
        args.devices,
        "--timeout",
        str(PROBE_TIMEOUT_SECONDS),
    ]
    if args.run:
        command.append("--run")
    if args.setup_only:
        command.append("--setup-only")
    vendor = args.transformer_vendor.resolve()
    environment = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(
            part for part in (str(repo), os.environ.get("PYTHONPATH")) if part
        ),
        "ASCEND_CUSTOM_OPP_PATH": os.pathsep.join(
            part
            for part in (str(vendor), os.environ.get("ASCEND_CUSTOM_OPP_PATH"))
            if part
        ),
        "LD_LIBRARY_PATH": os.pathsep.join(
            part
            for part in (str(vendor / "op_api/lib"), os.environ.get("LD_LIBRARY_PATH"))
            if part
        ),
    }
    summary = {
        "scope": (
            "single-layer native TP diagnostic; "
            "complete model and AFD equivalence unaccepted"
        ),
        "run_requested": args.run,
        "setup_only": args.setup_only,
        "command": command,
        "environment": environment,
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256((repo / PROBE).read_bytes()).hexdigest(),
        "wrapper_dependency_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("run_dsv41_a5_afd.py", "preflight_dsv41_a5_afd.py")
        },
        "probe_dependency_sha256": {
            name: file_sha256(repo / "tests/npu" / name)
            for name in (
                "afd_profiler_probe.py",
                "hccl_eager_backend_probe.py",
                "hccl_staged_probe.py",
            )
        },
        "package_versions": {},
        "model_equivalence_accepted": False,
        "success": False,
    }
    code = 1
    owned = None
    try:
        for name in (
            "torch",
            "torch-npu",
            "vllm",
            "vllm-ascend",
            "cann-ops-transformer",
        ):
            try:
                summary["package_versions"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                summary["package_versions"][name] = None
        revision = subprocess.run(
            ["git", "--no-optional-locks", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        summary["plugin_git_head"] = revision.stdout.strip() or None
        library = vendor / "op_api/lib/libcust_opapi.so"
        summary["operator_library"] = {
            "path": str(library),
            "exists": library.is_file(),
        }
        if args.run and not library.is_file():
            raise FileNotFoundError(
                "transformer vendor requires op_api/lib/libcust_opapi.so"
            )
        if args.run:
            summary["operator_library"].update(
                {"bytes": library.stat().st_size, "sha256": file_sha256(library)}
            )
        bundle_manifest = repo / "source-manifest.json"
        if bundle_manifest.is_file():
            summary["source_bundle_manifest"] = json.loads(
                bundle_manifest.read_text(encoding="utf-8")
            )
        owned = supervision.start("moe-parity", command, environment, repo)
        if not supervision.wait_exit(owned, OUTER_TIMEOUT_SECONDS):
            raise TimeoutError("single-layer diagnostic exceeded its outer deadline")
        if supervision.survivors():
            raise RuntimeError("single-layer diagnostic left owned processes alive")
        code = owned.process.returncode
        summary["success"] = code == 0
    except BaseException:
        code = 1
        summary["success"] = False
        summary["error"] = traceback.format_exc()
        if owned is not None and owned.process.poll() is None:
            with suppress(ProcessLookupError):
                owned.process.send_signal(signal.SIGINT)
            supervision.wait_exit(owned, INTERRUPT_CLEANUP_SECONDS)
    finally:
        supervision.force_owned_cleanup()
        summary["forced_cleanup"] = supervision.forced_cleanup
        summary["survivors"] = supervision.survivors()
        summary["exit_code"] = owned.process.returncode if owned is not None else None
        if summary["forced_cleanup"] or summary["survivors"]:
            code = 1
            summary["success"] = False
        for process in supervision.processes:
            process.stream.close()
        (directory / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        signal.signal(signal.SIGTERM, previous_handler)
        pack_run(directory)
    print(f"results: {directory}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
