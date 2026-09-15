#!/usr/bin/env python3
"""Reproduce the A5 A4F2 BF16 HCCL ReduceScatter failure."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from datetime import timedelta
from pathlib import Path

EXPECTED_WORLD_SIZE = 2
DEFAULT_OUTPUT_COUNT = 16_777_216
DEFAULT_TIMEOUT_SECONDS = 120


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the two-rank BF16 ReduceScatter shape observed in the A5 "
            "A4F2 failure. Launch this worker through torch.distributed.run."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-count", type=_positive_int, default=1)
    parser.add_argument("--prime-operations", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--output-count", type=_positive_int, default=DEFAULT_OUTPUT_COUNT
    )
    parser.add_argument("--iterations", type=_positive_int, default=1)
    parser.add_argument(
        "--timeout-seconds", type=_positive_int, default=DEFAULT_TIMEOUT_SECONDS
    )
    return parser


def _write_result(output_dir: Path, rank: int, result: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"rank-{rank}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()

    # Imports stay inside the worker so CPU-only checks can inspect this tool.
    import torch
    import torch.distributed as dist
    import torch_npu

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    result: dict[str, object] = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "group_count": args.group_count,
        "prime_operations": args.prime_operations,
        "iterations": args.iterations,
        "dtype": "bfloat16",
        "output_count": args.output_count,
        "input_bytes": args.output_count * world_size * 2,
        "output_bytes": args.output_count * 2,
        "hccl_op_expansion_mode": os.environ.get("HCCL_OP_EXPANSION_MODE", ""),
        "torch_version": torch.__version__,
        "torch_npu_version": torch_npu.__version__,
        "passed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_result(args.output_dir, rank, result)

    created_groups = []
    initialized = False
    passed = False
    try:
        if world_size != EXPECTED_WORLD_SIZE:
            raise RuntimeError(
                f"expected {EXPECTED_WORLD_SIZE} ranks, received {world_size}"
            )

        torch.npu.set_device(local_rank)
        dist.init_process_group(
            backend="hccl",
            timeout=timedelta(seconds=args.timeout_seconds),
        )
        initialized = True
        groups = [dist.group.WORLD]
        ranks = list(range(world_size))
        for _ in range(1, args.group_count):
            group = dist.new_group(ranks=ranks, backend="hccl")
            groups.append(group)
            created_groups.append(group)

        # Materialize every communicator, as vLLM does before model execution.
        for group in groups:
            materialization_value = torch.tensor(
                [rank + 1], dtype=torch.bfloat16, device=f"npu:{local_rank}"
            )
            dist.all_reduce(materialization_value, group=group)
            torch.npu.synchronize()

        active_group = groups[-1]
        prime_input = torch.tensor(
            [rank + 1], dtype=torch.bfloat16, device=f"npu:{local_rank}"
        )
        prime_output = torch.empty(
            world_size, dtype=torch.bfloat16, device=f"npu:{local_rank}"
        )
        for _ in range(args.prime_operations):
            dist.all_gather_into_tensor(
                prime_output,
                prime_input,
                group=active_group,
            )
            torch.npu.synchronize()

        input_tensor = torch.full(
            (args.output_count * world_size,),
            rank + 1,
            dtype=torch.bfloat16,
            device=f"npu:{local_rank}",
        )
        output_tensor = torch.empty(
            args.output_count,
            dtype=torch.bfloat16,
            device=f"npu:{local_rank}",
        )
        expected_value = world_size * (world_size + 1) // 2
        sample_indices = torch.tensor(
            [0, args.output_count // 2, args.output_count - 1],
            device=f"npu:{local_rank}",
        )
        samples: list[list[float]] = []
        for _ in range(args.iterations):
            dist.reduce_scatter_tensor(
                output_tensor,
                input_tensor,
                group=active_group,
            )
            torch.npu.synchronize()
            sample = output_tensor.index_select(0, sample_indices).cpu().tolist()
            samples.append(sample)
            if sample != [float(expected_value)] * len(sample_indices):
                raise RuntimeError(
                    f"ReduceScatter value mismatch: {sample}, "
                    f"expected {expected_value}"
                )

        result["samples"] = samples
        result["passed"] = True
        passed = True
    except BaseException as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        _write_result(args.output_dir, rank, result)
        # A failed AICPU task poisons the device context; teardown then only
        # obscures the first HCCL error with secondary synchronization errors.
        if initialized and passed:
            for group in reversed(created_groups):
                dist.destroy_process_group(group)
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
