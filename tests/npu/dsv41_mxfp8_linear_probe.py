# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A5 DS linear post-load + real MX quant/matmul probe, without checkpoint.

Runs one device, no model, distributed group, Engram, or service. The method's
configuration fields are supplied explicitly; its VllmConfig constructor is not
exercised. Target-wheel operator execution remains required on A5.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import traceback
from pathlib import Path

OUTPUT_WIDTH = 5120
BLOCK_SIZE = 32
INPUT_WIDTHS = (256, 288, 320)
SCALE_ALGORITHMS = (0, 1)
RELATIVE_TOLERANCE = 0.02
ABSOLUTE_TOLERANCE = 0.05


def run_probe(device_index):
    # Lazy imports allow --help on a CPU host; all execution below is real NPU.
    import torch
    import torch_npu
    from torch import nn
    from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
        AscendW8A8MXFP8DSDynamicLinearMethod,
    )

    from afd_plugin.compat.patches.npu.mxfp8 import apply_dsv41_mxfp8_scale_patch

    torch.npu.set_device(device_index)
    device = torch.device(f"npu:{device_index}")
    source_file = Path(inspect.getfile(AscendW8A8MXFP8DSDynamicLinearMethod))
    report = {
        "scope": "DS post-load packing and real dynamic_mx_quant/quant_matmul only",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": str(device),
        "device_name": torch.npu.get_device_name(device_index),
        "upstream_file": str(source_file),
        "upstream_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "cases": [],
    }
    apply_dsv41_mxfp8_scale_patch()
    # No ModelState or checkpoint loading is involved in this component probe.
    scheme = AscendW8A8MXFP8DSDynamicLinearMethod.__new__(
        AscendW8A8MXFP8DSDynamicLinearMethod
    )
    scheme.block_size = BLOCK_SIZE
    scheme.group_size = BLOCK_SIZE
    scheme.o_lora_rank = 1024
    try:
        for width in INPUT_WIDTHS:
            scales = torch.pow(
                2.0,
                torch.arange(OUTPUT_WIDTH // BLOCK_SIZE * (width // BLOCK_SIZE))
                .reshape(OUTPUT_WIDTH // BLOCK_SIZE, width // BLOCK_SIZE)
                .remainder(3)
                - 1,
            )
            weight = torch.full((OUTPUT_WIDTH, width), 0.25)
            expanded = scales.repeat_interleave(BLOCK_SIZE, 0).repeat_interleave(
                BLOCK_SIZE, 1
            )
            layer = nn.Module()
            layer.prefix = "model.layers.0.mlp.shared_experts.down_proj"
            layer.weight = nn.Parameter(
                weight.to(device=device, dtype=torch.float8_e4m3fn), requires_grad=False
            )
            layer.weight_scale = nn.Parameter(scales.to(device), requires_grad=False)
            scheme.process_weights_after_loading(layer)
            assert tuple(layer.weight.shape) == (width, OUTPUT_WIDTH)
            assert tuple(layer.weight_scale.shape) == (
                (width + 63) // 64,
                OUTPUT_WIDTH,
                2,
            )
            # Row 0 covers all groups, row 1 isolates the last valid group.
            inputs = torch.ones((2, width), dtype=torch.bfloat16)
            inputs[1, : width - BLOCK_SIZE] = 0
            reference = inputs.float() @ (weight * expanded).T
            for algorithm in SCALE_ALGORITHMS:
                scheme.dynamic_mx_quant_scale_alg = algorithm
                result = scheme.apply(layer, inputs.to(device))
                torch.npu.synchronize()
                actual = result.float().cpu()
                case = {
                    "input_width": width,
                    "output_width": OUTPUT_WIDTH,
                    "scale_alg": algorithm,
                    "weight_shape": list(layer.weight.shape),
                    "scale_shape": list(layer.weight_scale.shape),
                    "max_absolute_error": (actual - reference).abs().max().item(),
                    "last_group_nonzero": bool(torch.count_nonzero(actual[1])),
                }
                report["cases"].append(case)
                torch.testing.assert_close(
                    actual,
                    reference,
                    rtol=RELATIVE_TOLERANCE,
                    atol=ABSOLUTE_TOLERANCE,
                )
                assert case["last_group_nonzero"]
                case["passed"] = True
        report["passed"] = True
    except Exception:
        report["passed"] = False
        report["traceback"] = traceback.format_exc()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_probe(args.device)
    except Exception:
        report = {"passed": False, "traceback": traceback.format_exc()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
