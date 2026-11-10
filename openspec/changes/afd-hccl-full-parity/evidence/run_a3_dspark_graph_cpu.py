"""Freeze C7 CPU seams in the existing A3 venv; no device.

Defaults retain the DSpark Graph scope. Other C7 seams must provide their own
installed-source identity, evidence directory and complete module targets.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

CPU_HARNESS = (
    "openspec/changes/afd-hccl-v2-graph/evidence/"
    "a3-control-cpu-20261109/run_a3_graph_cpu.py"
)
SOURCE_IDENTITY = (
    "openspec/changes/afd-hccl-full-parity/evidence/"
    "dspark-graph-source-identity-20261110.json"
)
EVIDENCE_DIRECTORY = (
    "openspec/changes/afd-hccl-full-parity/evidence/a3-dspark-graph-cpu-20261110"
)
TARGETS = (
    "tests/unit/compat/npu/test_runtime_config.py",
    "tests/unit/connectors/npu/test_hccl_config.py",
    "tests/unit/v1/worker/npu/test_hccl_v41_ubatch_runner.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--source-identity", default=SOURCE_IDENTITY)
    parser.add_argument("--evidence-dir", default=EVIDENCE_DIRECTORY)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "existing_dspark_graph_cpu_harness", root / CPU_HARNESS
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    harness.SOURCE_IDENTITY = args.source_identity
    sys.argv = [
        str(root / CPU_HARNESS),
        "--label",
        args.label,
        "--evidence-dir",
        args.evidence_dir,
        "--targets",
        *args.targets,
    ]
    return harness.main()


if __name__ == "__main__":
    raise SystemExit(main())
