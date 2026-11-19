#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check full-checkpoint preflight identities against read-only locked Git objects."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "tools/diagnostics"))
    preflight = importlib.import_module("preflight_dsv41_a5_afd")

    repositories = json.loads(
        (ROOT / ".agents/dsv41/sources.lock.json").read_text("utf-8")
    )["repositories"]
    rows = []
    for source in preflight.DEFAULT_PROFILE.sources:
        repo = repositories[source.package]
        name = source.package + "/" + source.path
        raw = subprocess.check_output(
            ["git", "-C", repo["path"], "show", repo["commit"] + ":" + name]
        )
        actual = hashlib.sha256(raw).hexdigest()
        rows.append(
            {
                "path": name,
                "git_revision": repo["commit"],
                "locked_sha256": actual,
                "preflight_sha256": source.sha256,
                "matches": source.sha256 == actual,
            }
        )
    result = {
        "success": all(row["matches"] for row in rows),
        "scope": "read_only_locked_Git_source_check_not_NPU_execution",
        "pins": rows,
        "revision_labels_match_lock": (
            repositories["vllm"]["commit"] == preflight.VLLM_SOURCE_REVISION
            and repositories["vllm_ascend"]["commit"]
            == preflight.ASCEND_SOURCE_REVISION
        ),
        "native_runtime_imports": sorted(
            name
            for name in sys.modules
            if name.split(".")[0] in ("torch", "torch_npu", "vllm", "vllm_ascend")
        ),
    }
    assert not result["native_runtime_imports"]
    result["success"] = result["success"] and result["revision_labels_match_lock"]
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"success": result["success"], "pins_checked": len(rows)}))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
