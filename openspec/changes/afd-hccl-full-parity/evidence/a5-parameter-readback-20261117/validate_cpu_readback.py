#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check logical parameter hashes on CPU; does not exercise NPU format casts."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be fresh")
    root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(root / "tests/npu"))

    # Explicit CPU execution only; metadata inspection needs no runtime imports.
    import torch
    from dsv41_normal_model_probe import parameter_record

    assert sys.byteorder == "little"
    cases = (
        (
            "scalar_with_grad",
            torch.tensor(-2.5, requires_grad=True),
            struct.pack("<f", -2.5),
            True,
        ),
        (
            "strided_view_with_offset",
            torch.arange(12, dtype=torch.float32).reshape(3, 4).t()[1:3],
            struct.pack("<6f", 1, 5, 9, 2, 6, 10),
            True,
        ),
        (
            "bfloat16",
            torch.tensor([1, -2.5], dtype=torch.bfloat16),
            bytes.fromhex("803f20c0"),
            True,
        ),
        (
            "integer",
            torch.tensor([0, 256, -1], dtype=torch.int64),
            struct.pack("<3q", 0, 256, -1),
            None,
        ),
        (
            "infinite",
            torch.tensor([float("inf")]),
            struct.pack("<f", float("inf")),
            False,
        ),
        ("empty", torch.empty((0, 2)), b"", True),
    )
    results = []
    for name, tensor, expected_bytes, expected_finite in cases:
        original = tensor.detach().clone()
        pointer = tensor.data_ptr()
        stride = tuple(tensor.stride())
        offset = tensor.storage_offset()
        requires_grad = tensor.requires_grad
        record = parameter_record(tensor)
        assert record["sha256"] == hashlib.sha256(expected_bytes).hexdigest(), name
        assert record["bytes"] == len(expected_bytes), name
        assert record["finite"] is expected_finite, name
        assert record["shape"] == list(original.shape), name
        assert record["dtype"] == str(original.dtype), name
        assert record["numel"] == original.numel(), name
        assert torch.equal(tensor, original), name
        assert tensor.data_ptr() == record["data_ptr"] == pointer, name
        assert tuple(tensor.stride()) == tuple(record["stride"]) == stride, name
        assert tensor.storage_offset() == record["storage_offset"] == offset, name
        assert tensor.requires_grad == requires_grad, name
        results.append({"case": name, "record": record, "original_unchanged": True})
    assert "torch_npu" not in sys.modules
    report = {
        "status": "passed_cpu_only",
        "torch_version": torch.__version__,
        "cases": results,
        "torch_npu_imported": False,
        "npu_format_conversion": "not_exercised_pending_A5",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"status": report["status"], "cases": len(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
