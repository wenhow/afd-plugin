#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare the exact frozen audit change, preserving all production bytes."""

from __future__ import annotations

import difflib
import hashlib
import json
import tarfile
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[4]
PRODUCTION_COUNT = 146


def archive_files(path: Path) -> dict[str, bytes]:
    with tarfile.open(path) as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }


def main() -> None:
    parent_path = DIRECTORY / "parent-runtime-source.tar.gz"
    current_path = ROOT / "reports/dsv41-a5-functional/runtime-source.tar.gz"
    original_path = (
        DIRECTORY.parent / "a5-dspark-graph-final-audit-20261119/"
        "parent-runtime-source.tar.gz"
    )
    parent = archive_files(parent_path)
    current = archive_files(current_path)
    original = archive_files(original_path)
    assert set(parent) == set(current)
    provenance = json.loads(current["functional-source-provenance.json"])
    helpers = provenance["graph_device_audit_successors"]
    changed = sorted(name for name in parent if parent[name] != current[name])
    assert set(changed) == set(helpers) | {
        "bundle-manifest.json",
        "functional-source-provenance.json",
    }
    for name, identity in helpers.items():
        assert hashlib.sha256(parent[name]).hexdigest() == identity["parent_sha256"]
        assert hashlib.sha256(current[name]).hexdigest() == identity["sha256"]
        assert parent[name] == original[name]
    production = [name for name in parent if name.startswith("afd_plugin/")]
    assert len(production) == PRODUCTION_COUNT
    assert all(parent[name] == current[name] for name in production)
    assert (
        parent["native-source-identity.json"] == current["native-source-identity.json"]
    )
    sources = json.loads(current["bundle-manifest.json"])["sources"]
    for name, expected in sources.items():
        assert hashlib.sha256(current[name]).hexdigest() == expected
        if name.endswith(".py"):
            compile(current[name], name, "exec")
    rebuild = ROOT / ".tmp/a5-device-successor-repro-001"
    deterministic = {
        name: (current_path.parent / name).read_bytes() == (rebuild / name).read_bytes()
        for name in ("runtime-source.tar.gz", "bundle-lock.json", "preparation.json")
    }
    assert all(deterministic.values())
    diff = "".join(
        line
        for name in sorted(helpers)
        for line in difflib.unified_diff(
            parent[name].decode().splitlines(keepends=True),
            current[name].decode().splitlines(keepends=True),
            fromfile="parent/" + name,
            tofile="successor/" + name,
            n=0,
        )
    )
    (DIRECTORY / "graph-device-audit.patch").write_bytes(diff.encode("utf-8"))
    result = {
        "success": True,
        "original_executed_bundle_sha256": hashlib.sha256(
            original_path.read_bytes()
        ).hexdigest(),
        "parent_bundle_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "successor_bundle_sha256": hashlib.sha256(
            current_path.read_bytes()
        ).hexdigest(),
        "changed_members": changed,
        "six_audit_parents_match_original_executed_snapshot": True,
        "production_byte_identical": len(production),
        "native_identity_byte_identical": True,
        "python_compiled": sum(name.endswith(".py") for name in sources),
        "deterministic_rebuild": deterministic,
    }
    (DIRECTORY / "source-verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
