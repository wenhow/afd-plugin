# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A5 DS W13 -> fused SwiGLU/MX quant -> W2, without checkpoint or routing."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import traceback
from pathlib import Path

EXPERT_COUNT = 2
TOKENS_PER_EXPERT = 2
HIDDEN_SIZE = 5120
INPUT_DIVISOR = 512
GROUP_SIZE = 32
WIDTHS = (256, 288, 320)
CLAMP_LIMITS = (0.0, 6.0)
GROUP_LIST_TYPES = (0, 1)
FP4_ONE_PAIR = 0x22
FP4_NEGATIVE_ONE_PAIR = 0xAA
E8M0_UNIT = 127
RELATIVE_TOLERANCE = 0.05
ABSOLUTE_TOLERANCE = 0.05


def run_probe(device_index):
    # Lazy imports let --help run on a CPU-only host.
    import torch
    import torch_npu
    from torch import nn
    from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights
    from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
    from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEQuantParams
    from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
        AscendW4A8MXFPDSDynamicFusedMoEMethod,
    )

    from afd_plugin.compat.patches.npu.mxfp4 import apply_dsv41_mxfp4_scale_patch

    torch.npu.set_device(device_index)
    device = torch.device(f"npu:{device_index}")
    apply_dsv41_mxfp4_scale_patch()
    target = AscendW4A8MXFPDSDynamicFusedMoEMethod
    patch_file = Path(inspect.getfile(target.apply_gmm1_act_quant))
    upstream_file = Path(inspect.getfile(target))
    report = {
        "scope": __doc__,
        "device_name": torch.npu.get_device_name(device_index),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "upstream_sha256": hashlib.sha256(upstream_file.read_bytes()).hexdigest(),
        "patch_file": str(patch_file),
        "patch_sha256": hashlib.sha256(patch_file.read_bytes()).hexdigest(),
        "cases": [],
    }
    scheme = target.__new__(target)
    scheme.dynamic_eplb = False
    scheme.group_size = GROUP_SIZE
    rows = EXPERT_COUNT * TOKENS_PER_EXPERT
    try:
        for width in WIDTHS:
            layer = nn.Module()
            for stem, inputs, outputs in (
                ("w13", HIDDEN_SIZE, 2 * width),
                ("w2", width, HIDDEN_SIZE),
            ):
                packed = torch.full(
                    (EXPERT_COUNT, outputs, inputs // 2),
                    FP4_ONE_PAIR,
                    dtype=torch.uint8,
                )
                scales = torch.full(
                    (EXPERT_COUNT, outputs, inputs // GROUP_SIZE),
                    E8M0_UNIT,
                    dtype=torch.uint8,
                )
                if stem == "w13":
                    # Distinct gate/up halves and a different final group expose
                    # accidental concatenated-tail padding or dropped values.
                    packed[1, :width] = FP4_NEGATIVE_ONE_PAIR
                    scales[:, width:] -= 1
                    scales[:, width - GROUP_SIZE : width] += 1
                    scales[:, -GROUP_SIZE:] += 1
                layer.register_parameter(
                    f"{stem}_weight",
                    nn.Parameter(packed.to(device), requires_grad=False),
                )
                layer.register_parameter(
                    f"{stem}_weight_scale",
                    nn.Parameter(
                        scales.to(device).view(torch.float8_e8m0fnu),
                        requires_grad=False,
                    ),
                )
            scheme.process_weights_after_loading(layer)
            for limit in CLAMP_LIMITS:
                gate = torch.full((rows, width), HIDDEN_SIZE / INPUT_DIVISOR)
                gate[TOKENS_PER_EXPERT:] *= -1
                gate[:, -GROUP_SIZE:] *= 2
                up = torch.full((rows, width), HIDDEN_SIZE / INPUT_DIVISOR / 2)
                up[:, -GROUP_SIZE:] *= 2
                if limit:
                    gate = gate.clamp(max=limit)
                    up = up.clamp(min=-limit, max=limit)
                expected_activation = torch.nn.functional.silu(gate) * up
                expected_output = expected_activation.sum(-1, keepdim=True).expand(
                    rows, HIDDEN_SIZE
                )
                for group_type in GROUP_LIST_TYPES:
                    case = {
                        "intermediate_width": width,
                        "swiglu_limit": limit,
                        "group_list_type": group_type,
                        "passed": False,
                    }
                    report["cases"].append(case)
                    counts = torch.full(
                        (EXPERT_COUNT,),
                        TOKENS_PER_EXPERT,
                        dtype=torch.int64,
                        device=device,
                    )
                    mlp_input = MoEMlpComputeInput(
                        hidden_states=torch.full(
                            (rows, HIDDEN_SIZE),
                            1 / INPUT_DIVISOR,
                            dtype=torch.bfloat16,
                            device=device,
                        ),
                        group_list=counts.cumsum(0) if group_type == 0 else counts,
                        group_list_type=group_type,
                        dynamic_scale=None,
                        topk_scales=None,
                        weights=MoEWeights(w1=layer.w13_weight, w2=layer.w2_weight),
                        quant=MoEQuantParams(quant_type=scheme.quant_type),
                        fusion=False,
                        layer=layer,
                        swiglu_limit=limit,
                    )
                    quantized, scales = scheme.apply_gmm1_act_quant(mlp_input)
                    output = scheme.apply_gmm2(mlp_input, quantized, scales)
                    torch.npu.synchronize()
                    scale_values = scales.view(torch.uint8).cpu().flatten(-2)
                    scale_values = scale_values.repeat_interleave(GROUP_SIZE, -1)
                    activation = quantized.float().cpu() * torch.pow(
                        2.0, scale_values.float() - E8M0_UNIT
                    )
                    torch.testing.assert_close(
                        activation[:, :width],
                        expected_activation,
                        rtol=RELATIVE_TOLERANCE,
                        atol=ABSOLUTE_TOLERANCE,
                    )
                    assert torch.isfinite(activation).all()
                    assert torch.count_nonzero(activation[:, width:]) == 0
                    torch.testing.assert_close(
                        output.float().cpu(),
                        expected_output,
                        rtol=RELATIVE_TOLERANCE,
                        atol=ABSOLUTE_TOLERANCE,
                    )
                    case.update(
                        passed=True,
                        activation_shape=list(quantized.shape),
                        activation_scale_shape=list(scales.shape),
                        activation_max_absolute_error=(
                            activation[:, :width] - expected_activation
                        )
                        .abs()
                        .max()
                        .item(),
                        output_max_absolute_error=(
                            output.float().cpu() - expected_output
                        )
                        .abs()
                        .max()
                        .item(),
                    )
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
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
