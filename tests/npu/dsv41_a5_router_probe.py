#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolate the installed A5 hash router; metadata mode imports no runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path

from dsv41_dspark_a5_runtime import native_a5_bootstrap

# name, expert count, selected experts, dtype, vision bias. Each case must
# execute in a fresh process because a vector-core fault poisons its context.
CASES = (
    ("e64-k2-fp32", 64, 2, "float32", False),
    ("e384-k2-fp32", 384, 2, "float32", False),
    ("e384-k6-fp32", 384, 6, "float32", False),
    ("e64-k2-bf16", 64, 2, "bfloat16", False),
    ("e384-k6-bf16", 384, 6, "bfloat16", False),
    ("e384-k6-fp32-vision", 384, 6, "float32", True),
)
ROWS = 8
SCALING_FACTOR = 1.5
SEED = 20260909
BIAS_SCALE = 0.2
ROUTING_EPSILON = 1e-20
FP32_TOLERANCE = 1e-5
BF16_TOLERANCE = 1e-2
IMAGE_SENTINEL_LO = 129257
IMAGE_SENTINEL_COUNT = 5


def loaded_operator_libraries() -> list[dict]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    paths = {
        line.split()[-1]
        for line in maps.read_text().splitlines()
        if any(
            name in line
            for name in (
                "libcust_opapi.so",
                "libopapi.so",
                "libopapi_transformer.so",
                "liboptiling.so",
                "vllm_ascend_C.",
            )
        )
    }
    return [
        {"path": name, "sha256": hashlib.sha256(Path(name).read_bytes()).hexdigest()}
        for name in sorted(paths)
        if Path(name).is_file()
    ]


def run_case(
    case: tuple[str, int, int, str, bool], device: int, output: Path, report: dict
) -> None:
    # Explicit device mode only; bootstrap and registration match the normal
    # A5 probe. No worker, checkpoint, collective or replacement router.
    from dsv41_dspark_speculator_probe import NATIVE_PLUGINS

    os.environ["VLLM_PLUGINS"] = NATIVE_PLUGINS
    import torch
    import torch_npu  # noqa: F401
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm_ascend.utils import adapt_patch

    report["runtime_imported"] = True
    report["phase"] = "bootstrap"
    current_platform.pre_register_and_update()
    load_general_plugins()
    adapt_patch()
    torch.npu.set_device(device)
    report["native_a5_bootstrap"] = native_a5_bootstrap()
    report["schema"] = str(torch.ops._C_ascend.moe_gating_top_k_hash.default._schema)
    _, experts, top_k, dtype_name, vision = case
    dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    logits = torch.randn(ROWS, experts, dtype=dtype, generator=generator)
    bias = torch.randn(experts, dtype=dtype, generator=generator) * BIAS_SCALE
    bias_vl = (
        torch.randn(experts, dtype=dtype, generator=generator) * BIAS_SCALE
        if vision
        else None
    )
    input_ids = (
        torch.tensor(
            [
                11,
                IMAGE_SENTINEL_LO,
                22,
                IMAGE_SENTINEL_LO + 2,
                33,
                IMAGE_SENTINEL_LO + 4,
                44,
                55,
            ],
            dtype=torch.int64,
        )
        if vision
        else None
    )
    # Oracle follows the locked native test_npu_moe_gating_top_k_hash.py
    # at vLLM-Ascend 7e37e29a; bias affects IDs, not gathered weights.
    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    row_bias = bias.float()
    if vision:
        image = (input_ids >= IMAGE_SENTINEL_LO) & (
            input_ids < IMAGE_SENTINEL_LO + IMAGE_SENTINEL_COUNT
        )
        row_bias = torch.where(image[:, None], bias_vl.float(), bias.float())
    expected_ids = torch.topk(scores + row_bias, top_k, dim=-1, sorted=True).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = (
        expected_weights / expected_weights.sum(dim=-1, keepdim=True) * SCALING_FACTOR
    ).to(dtype)
    inputs = {
        "x": logits,
        "k": top_k,
        "bias": bias,
        "input_ids": input_ids,
        "tid2eid": None,
        "k_group": 1,
        "group_count": 1,
        "routed_scaling_factor": SCALING_FACTOR,
        "eps": ROUTING_EPSILON,
        "group_select_mode": 1,
        "renorm": 0,
        "norm_type": 2,
        "out_flag": False,
        "bias_vl": bias_vl,
        "image_sentinel_lo": IMAGE_SENTINEL_LO,
        "image_sentinel_count": IMAGE_SENTINEL_COUNT,
    }
    torch.save(inputs, output / "inputs.pt")
    report["inputs_sha256"] = hashlib.sha256(
        (output / "inputs.pt").read_bytes()
    ).hexdigest()
    npu_inputs = {
        name: value.to(device=f"npu:{device}")
        if isinstance(value, torch.Tensor)
        else value
        for name, value in inputs.items()
    }
    report["phase"] = "operator"
    (output / "result.json").write_bytes((json.dumps(report, indent=2) + "\n").encode())
    actual_weights, actual_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
        **npu_inputs
    )
    torch.npu.synchronize()
    report["operator_returned"] = True
    report["phase"] = "comparison"
    actual_weights, actual_ids = actual_weights.cpu(), actual_ids.cpu()
    torch.save({"weights": actual_weights, "ids": actual_ids}, output / "outputs.pt")
    report["actual_ids"] = actual_ids.tolist()
    report["expected_ids"] = expected_ids.tolist()
    report["max_weight_absolute_error"] = float(
        (actual_weights.float() - expected_weights.float()).abs().max()
    )
    torch.testing.assert_close(actual_ids, expected_ids.int(), rtol=0, atol=0)
    tolerance = FP32_TOLERANCE if dtype == torch.float32 else BF16_TOLERANCE
    torch.testing.assert_close(
        actual_weights, expected_weights, rtol=tolerance, atol=tolerance
    )
    report["reference_matched"] = True
    report["phase"] = "complete"
    report["success"] = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case[0] for case in CASES], required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be nonnegative")
    case = next(case for case in CASES if case[0] == args.case)
    report = {
        "case": args.case,
        "shape": [ROWS, case[1]],
        "top_k": case[2],
        "dtype": case[3],
        "vision_bias": case[4],
        "hash_routing": False,
        "device": args.device,
        "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
        "runtime_imported": False,
        "operator_returned": False,
        "reference_matched": False,
        "model_executed": False,
        "collective_executed": False,
        "functional_worker_component_accepted": False,
        "diagnostic_only": True,
        "success": not args.run,
    }
    if not args.run:
        print(json.dumps(report))
        return 0
    if os.name != "posix" or os.environ.get("ASCEND_LAUNCH_BLOCKING") != "1":
        parser.error("--run requires Linux and ASCEND_LAUNCH_BLOCKING=1")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run_case(case, args.device, args.output_dir, report)
    except BaseException:
        report["traceback"] = traceback.format_exc()
    finally:
        report["runtime_environment"] = {
            name: os.environ.get(name)
            for name in ("ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH")
        }
        try:
            report["loaded_operator_libraries"] = loaded_operator_libraries()
        except BaseException:
            report["library_collection_error"] = traceback.format_exc()
            report["success"] = False
        (args.output_dir / "result.json").write_bytes(
            (json.dumps(report, indent=2) + "\n").encode()
        )
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
