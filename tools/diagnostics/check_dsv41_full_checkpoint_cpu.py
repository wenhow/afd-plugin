#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run full-checkpoint boundary regressions without initializing an NPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from unittest.mock import patch

TEST_PATHS = (
    "tests/unit/connectors/npu/test_hccl_config.py",
    "tests/unit/connectors/npu/test_hccl_staged_transport.py",
    "tests/unit/v1/worker/npu/test_hccl_v41_ffn_runner.py",
    "tests/unit/entrypoints/test_npu_ffn.py",
    "tests/unit/distributed/test_hccl_plan.py",
    "tests/unit/connectors/npu/test_hccl_eager_pipeline.py",
    "tests/unit/connectors/npu/test_hccl_graph_stage.py",
    "tests/unit/v1/worker/npu/test_hccl_v41_ffn_eager.py",
    "tests/unit/compat/npu/test_engram_history.py",
    "tests/unit/compat/npu/test_feature_validation.py",
    "tests/unit/compat/npu/test_kv_layout_connector.py",
    "tests/unit/compat/patches/npu/test_ubatching_config.py",
    "tests/unit/compat/patches/test_signatures.py::test_scoped_patch_preserves_upstream_signature[npu-dspark-ubatching-config]",
    "tests/unit/compat/patches/test_signatures.py::test_scoped_patch_preserves_upstream_signature[npu-engram-ubatching-config]",
    "tests/unit/model_executor/models/npu/test_deepseek_v41_engram.py",
    "tests/unit/model_executor/models/npu/test_deepseek_v41_graph.py",
    "tests/unit/v1/worker/npu/test_hccl_v41_graph_execution.py",
    "tests/unit/v1/worker/npu/test_hccl_v41_graph_coordinator.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Runtime imports are deliberate only after explicit invocation in the
    # existing CPU-contract environment. This driver installs no packages.
    import pytest
    import torch
    import torch_npu  # noqa: F401

    root = Path(__file__).resolve().parents[2]
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    before = torch.npu.is_initialized()
    if before:
        raise RuntimeError("CPU validation requires an uninitialized NPU runtime")
    with patch.object(
        torch.npu,
        "_lazy_init",
        side_effect=RuntimeError("CPU validation forbids NPU initialization"),
    ):
        exit_code = pytest.main(
            [
                *[str(root / path) for path in TEST_PATHS],
                "-q",
                "-p",
                "no:cacheprovider",
                "--junitxml=" + str(directory / "junit.xml"),
            ]
        )
    after = torch.npu.is_initialized()
    record = {
        "exit_code": int(exit_code),
        "npu_initialized_before": before,
        "npu_initialized_after": after,
        "hardware_acceptance": False,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch-npu", "vllm", "vllm-ascend")
        },
        "sources": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for parent in (root / "afd_plugin", root / "tests/unit")
            for path in sorted(parent.rglob("*.py"))
        },
    }
    (directory / "summary.json").write_text(
        json.dumps(record, indent=2) + "\n", "utf-8"
    )
    return int(exit_code) if not after else 1


if __name__ == "__main__":
    raise SystemExit(main())
