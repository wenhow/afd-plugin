from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tools/dsv4/repro_a5_hccl_reducescatter.py"
RUNNER = ROOT / "tools/dsv4/run_a5_hccl_reducescatter_repro.sh"


def _load_worker_module():
    spec = importlib.util.spec_from_file_location("a5_hccl_rs_repro", WORKER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_defaults_match_a4f2_failure(tmp_path):
    module = _load_worker_module()
    args = module.build_parser().parse_args(["--output-dir", str(tmp_path)])

    assert args.group_count == 1
    assert args.prime_operations == 0
    assert args.output_count == 16_777_216
    assert args.iterations == 1


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--group-count", "0"),
        ("--prime-operations", "-1"),
        ("--output-count", "0"),
        ("--iterations", "0"),
    ],
)
def test_worker_rejects_invalid_counts(tmp_path, flag, value):
    module = _load_worker_module()

    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            ["--output-dir", str(tmp_path), flag, value]
        )


def test_runner_preserves_exact_hccl_failure_shape_and_sequence():
    runner = RUNNER.read_text(encoding="utf-8")

    assert 'run_case world_only 1 0' in runner
    assert 'run_case group19_op136 20 135' in runner
    assert "output_count=%s\\n' '16777216'" in runner
    assert "input_bytes=%s\\n' '67108864'" in runner
    assert "output_bytes=%s\\n' '33554432'" in runner
    assert 'HCCL_OP_EXPANSION_MODE:-AIV' in runner
