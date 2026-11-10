"""Freeze target-shape/isolation/native MC2 CPU seams in the existing A3 venv."""

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
    "dspark-shape-source-identity-20261110.json"
)
EVIDENCE_DIRECTORY = (
    "openspec/changes/afd-hccl-full-parity/evidence/a3-dspark-shape-cpu-20261110"
)
TARGETS = (
    "tests/unit/compat/npu/test_runtime_config.py",
    "tests/unit/v1/worker/npu/test_ffn_worker.py",
    "tests/unit/v1/worker/npu/test_ffn_model_runner.py",
    "tests/unit/v1/worker/test_npu_runtime.py",
    "tests/unit/test_async_cam_startup_wiring.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "existing_dspark_shape_cpu_harness", root / CPU_HARNESS
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    harness.SOURCE_IDENTITY = SOURCE_IDENTITY
    sys.argv = [
        str(root / CPU_HARNESS),
        "--label",
        args.label,
        "--evidence-dir",
        EVIDENCE_DIRECTORY,
        "--targets",
        *args.targets,
    ]
    return harness.main()


if __name__ == "__main__":
    raise SystemExit(main())
