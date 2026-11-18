#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze the non-Engram A5 functional suite; no runtime imports or devices."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from prepare_dsv41_a5_dspark_bundle import (
    A5_RUNTIME,
    BASELINE,
    BASELINE_SHA256,
    a5_helper_successors,
    archive_files,
    json_bytes,
    read_locked_sources,
    write_archive,
)

DSPARK_BUNDLE = "reports/dsv41-a5-dspark-functional/runtime-source.tar.gz"
DSPARK_SHA256 = "311cafc72c81dd06f09a67fe903527db3820089e23a39192f971e022cc5f7c3b"
WORKER_BUNDLE = (
    "openspec/changes/afd-hccl-full-parity/evidence/"
    "a3-text-normal-worker-20261116/prepared-execute-005/source.tar.gz"
)
WORKER_SHA256 = "a8bde079dd5126e6cf0f7b6aede1c093e02489aa070f1df88d5d7478df2ece81"
LAUNCHER = "tests/npu/dsv41_a5_functional_validation.py"
PROBE = "tests/npu/dsv41_functional_worker_probe.py"
MODEL_PROBE = "tests/npu/dsv41_normal_model_probe.py"
C2_PROBE = "tests/npu/dsv41_normal_c2_probe.py"
FEATURE = "afd_plugin/compat/npu/feature_validation.py"
PRODUCTION_OVERLAYS = (
    FEATURE,
    "afd_plugin/compat/npu/runtime_config.py",
    "afd_plugin/v1/worker/npu/ffn_worker.py",
    "afd_plugin/compat/patches/engine_core.py",
    "afd_plugin/compat/npu/profiler.py",
    "afd_plugin/compat/npu/profiler_api.py",
    "afd_plugin/v1/worker/npu/ubatch_runner.py",
)
OUTPUT = "reports/dsv41-a5-functional"
PRODUCTION_COUNT = 146


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verified_archive(root: Path, name: str, expected: str) -> dict[str, bytes]:
    path = root / name
    if digest(path.read_bytes()) != expected:
        raise ValueError(f"immutable parent archive differs: {name}")
    files = archive_files(path)
    manifest_name = (
        "bundle-manifest.json" if "bundle-manifest.json" in files else "manifest.json"
    )
    manifest = json.loads(files.pop(manifest_name))
    expected_sources = manifest.get("sources", manifest)
    if {name: digest(raw) for name, raw in files.items()} != expected_sources:
        raise ValueError(f"parent member hashes differ: {name}")
    return files


def add_local_helpers(root: Path, files: dict[str, bytes]) -> None:
    pending = [LAUNCHER, PROBE, MODEL_PROBE, C2_PROBE]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        raw = (root / name).read_bytes()
        files[name] = raw
        for node in ast.walk(ast.parse(raw, filename=name)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                candidates = [
                    "tests/npu/" + module.replace(".", "/") + ".py",
                    module.replace(".", "/") + ".py",
                ]
                for candidate in candidates:
                    # Keep the reviewed A5 DSpark successors frozen.
                    if (
                        candidate.startswith("tests/")
                        and (root / candidate).is_file()
                        and candidate not in files
                    ):
                        pending.append(candidate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("--output-dir must be fresh")
    files = verified_archive(root, DSPARK_BUNDLE, DSPARK_SHA256)
    # Refresh only the A5 indexer declarations from the immutable donor;
    # retain all other reviewed DSpark helper bytes from its parent bundle.
    donor = verified_archive(root, BASELINE, BASELINE_SHA256)
    successors, _ = a5_helper_successors(donor, root)
    for name in (A5_RUNTIME, "tests/npu/dsv41_dspark_speculator_probe.py"):
        files[name] = successors[name]
    original_production = {
        name: digest(raw)
        for name, raw in files.items()
        if name.startswith("afd_plugin/")
    }
    worker = verified_archive(root, WORKER_BUNDLE, WORKER_SHA256)
    production = {
        name: raw for name, raw in worker.items() if name.startswith("afd_plugin/")
    }
    if len(production) != PRODUCTION_COUNT or set(production) != set(
        original_production
    ):
        raise ValueError("normal Worker production closure differs")
    # Retain the reviewed normal-Worker closure, excluding unrelated paused MTP
    # work in the mutable checkout. Include the bounded admission, existing
    # DSpark cross-layer PD eligibility and existing failure ownership fixes,
    # including terminal RPC handling. No profiler device scene is added.
    for name in PRODUCTION_OVERLAYS:
        production[name] = (root / name).read_bytes()
    files.update(production)
    add_local_helpers(root, files)
    identity = json.loads(files["native-source-identity.json"])
    for name, raw in worker.items():
        if name.endswith(".json"):
            for source, expected in json.loads(raw).get("source_expected", {}).items():
                if source.startswith(("vllm/", "vllm_ascend/")):
                    previous = identity["source_expected"].get(source, expected)
                    if previous != expected:
                        raise ValueError(f"conflicting native source pin: {source}")
                    identity["source_expected"][source] = expected
    repositories = json.loads(
        (root / ".agents/dsv41/sources.lock.json").read_text("utf-8")
    )["repositories"]
    for package in ("vllm", "vllm_ascend"):
        names = sorted(
            n for n in identity["source_expected"] if n.startswith(package + "/")
        )
        for name, raw in read_locked_sources(repositories[package], names).items():
            if digest(raw) != identity["source_expected"][name]:
                raise ValueError(f"native Git pin differs: {name}")
    files["native-source-identity.json"] = json_bytes(identity)
    production_hashes = {name: digest(raw) for name, raw in sorted(production.items())}
    delta = {
        name: {"parent_sha256": original_production[name], "sha256": value}
        for name, value in production_hashes.items()
        if original_production[name] != value
    }
    files["functional-source-provenance.json"] = json_bytes(
        {
            "dspark_parent_sha256": DSPARK_SHA256,
            "worker_parent_sha256": WORKER_SHA256,
            "production_delta_from_dspark": delta,
            "production_overlays_from_worker": list(PRODUCTION_OVERLAYS),
            "paused_old_v4_mtp_included": False,
        }
    )
    for name, raw in files.items():
        if not name.endswith((".py", ".json")):
            raise ValueError(f"non-source member: {name}")
        if name.endswith(".py"):
            compile(raw, name, "exec")
    manifest = {
        "baseline_archive_sha256": BASELINE_SHA256,
        "sources": {name: digest(raw) for name, raw in sorted(files.items())},
        "source_count": len(files),
        "production_count": len(production),
    }
    output.mkdir(parents=True)
    archive = output / "runtime-source.tar.gz"
    write_archive(archive, {**files, "bundle-manifest.json": json_bytes(manifest)})
    lock = {
        "baseline_archive_sha256": BASELINE_SHA256,
        "runtime_source_sha256": digest(archive.read_bytes()),
        "launcher_sha256": digest(files[LAUNCHER]),
        "production_sources": production_hashes,
    }
    (output / "bundle-lock.json").write_bytes(json_bytes(lock))
    preparation = {
        "status": "source_prepared_not_device_validated",
        "runtime_source_sha256": lock["runtime_source_sha256"],
        "python_sources_compiled": sum(n.endswith(".py") for n in files),
        "source_count": len(files),
        "production_count": len(production),
        "native_git_pins_verified": len(identity["source_expected"]),
        "production_delta_from_dspark": delta,
        "new_admission_source_sha256": digest(production[FEATURE]),
        "npu_validation": "pending_on_A5",
        "contains_binaries": False,
    }
    (output / "preparation.json").write_bytes(json_bytes(preparation))
    print(json.dumps(preparation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
