"""Run existing CPU harness with endpoint-specific locked upstream identities.

Uses the existing environment and snapshot/timeout/retrieval implementation.
It does not import torch, install dependencies or start a device probe.
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
    "profiler-api-source-identity-20261110.json"
)
EVIDENCE_DIRECTORY = (
    "openspec/changes/afd-hccl-full-parity/evidence/a3-profiler-cpu-20261110"
)
TARGETS = (
    "tests/unit/compat/npu/test_profiler_api.py",
    "tests/unit/compat/npu/test_profiler.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "existing_profiler_cpu_harness", root / CPU_HARNESS
    )
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    # The existing harness embeds this identity's source_expected mapping into
    # its immutable archive and checks it remotely before importing the tests.
    harness.SOURCE_IDENTITY = SOURCE_IDENTITY
    sys.argv = [
        str(root / CPU_HARNESS),
        "--label",
        args.label,
        "--evidence-dir",
        EVIDENCE_DIRECTORY,
        "--targets",
        *TARGETS,
    ]
    return harness.main()


if __name__ == "__main__":
    raise SystemExit(main())
