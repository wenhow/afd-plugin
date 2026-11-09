# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A5 DS MoE post-load plus real FP4 grouped-matmul, without checkpoint.

Two synthetic experts exercise W13/W2 with K=256/288/320 at W2. This does not
exercise routing, SwiGLU, TP collectives, Engram or complete model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import traceback
from pathlib import Path

EXPERT_COUNT = 2
HIDDEN_SIZE = 5120
GROUP_SIZE = 32
INPUT_WIDTHS = (256, 288, 320)
SCALE_ALGORITHMS = (0, 1)
TOKENS_PER_EXPERT = 2
FP4_POSITIVE_ONE_PAIR = 0x22
FP4_NEGATIVE_ONE_PAIR = 0xAA
E8M0_UNIT_EXPONENT = 127
RELATIVE_TOLERANCE = 0.02
ABSOLUTE_TOLERANCE = 0.05


def run_probe(device_index):
    # Lazy imports allow --help on a CPU host; execution below requires A5.
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
    source_file = Path(inspect.getfile(AscendW4A8MXFPDSDynamicFusedMoEMethod))
    report = {
        "scope": "DS MoE post-load, W13 GMM and native apply_gmm2 with K alignment",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": str(device),
        "device_name": torch.npu.get_device_name(device_index),
        "upstream_file": str(source_file),
        "upstream_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "cases": [],
    }
    apply_dsv41_mxfp4_scale_patch()
    gmm2_file = Path(inspect.getfile(AscendW4A8MXFPDSDynamicFusedMoEMethod.apply_gmm2))
    report["gmm2_file"] = str(gmm2_file)
    report["gmm2_sha256"] = hashlib.sha256(gmm2_file.read_bytes()).hexdigest()
    scheme = AscendW4A8MXFPDSDynamicFusedMoEMethod.__new__(
        AscendW4A8MXFPDSDynamicFusedMoEMethod
    )
    scheme.dynamic_eplb = False
    group_list = torch.arange(1, EXPERT_COUNT + 1, device=device, dtype=torch.int64)
    group_list *= TOKENS_PER_EXPERT
    try:
        for width in INPUT_WIDTHS:
            layer = nn.Module()
            references = {}
            for stem, inputs, outputs in (
                ("w13", HIDDEN_SIZE, 2 * width),
                ("w2", width, HIDDEN_SIZE),
            ):
                packed = torch.full(
                    (EXPERT_COUNT, outputs, inputs // 2),
                    FP4_POSITIVE_ONE_PAIR,
                    dtype=torch.uint8,
                )
                packed[1] = FP4_NEGATIVE_ONE_PAIR
                exponents = torch.arange(inputs // GROUP_SIZE).remainder(3) - 1
                scale_bytes = (exponents + E8M0_UNIT_EXPONENT).to(torch.uint8)
                scale_bytes = scale_bytes.expand(EXPERT_COUNT, outputs, -1).clone()
                layer.register_parameter(
                    f"{stem}_weight",
                    nn.Parameter(packed.to(device), requires_grad=False),
                )
                layer.register_parameter(
                    f"{stem}_weight_scale",
                    nn.Parameter(
                        scale_bytes.to(device).view(torch.float8_e8m0fnu),
                        requires_grad=False,
                    ),
                )
                # One row covers all groups; one isolates the last valid group.
                reference = torch.empty((EXPERT_COUNT * TOKENS_PER_EXPERT, outputs))
                scales = torch.pow(2.0, exponents.float())
                for expert in range(EXPERT_COUNT):
                    sign = 1 if expert == 0 else -1
                    reference[expert * TOKENS_PER_EXPERT] = (
                        sign * scales.sum() * GROUP_SIZE
                    )
                    reference[expert * TOKENS_PER_EXPERT + 1] = (
                        sign * scales[-1] * GROUP_SIZE
                    )
                references[stem] = (inputs, reference)
            scheme.process_weights_after_loading(layer)
            for stem, (inputs, reference) in references.items():
                x = torch.ones(
                    (EXPERT_COUNT * TOKENS_PER_EXPERT, inputs), dtype=torch.bfloat16
                )
                x[1::TOKENS_PER_EXPERT, :-GROUP_SIZE] = 0
                for algorithm in SCALE_ALGORITHMS:
                    case = {
                        "intermediate_width": width,
                        "projection": stem,
                        "scale_alg": algorithm,
                        "logical_input_width": inputs,
                        "physical_weight_input_width": getattr(
                            layer, f"{stem}_weight"
                        ).shape[1]
                        * 2,
                        "weight_shape": list(getattr(layer, f"{stem}_weight").shape),
                        "scale_shape": list(
                            getattr(layer, f"{stem}_weight_scale").shape
                        ),
                        "call": "native_scheme.apply_gmm2"
                        if stem == "w2"
                        else "npu_grouped_matmul",
                        "passed": False,
                    }
                    report["cases"].append(case)
                    quantized, x_scale = torch_npu.npu_dynamic_mx_quant(
                        x.to(device),
                        dst_type=torch.float8_e4m3fn,
                        scale_alg=algorithm,
                    )
                    case["activation_scale_shape"] = list(x_scale.shape)
                    if stem == "w2":
                        # Exercise the same post-SwiGLU hook used by native MoE;
                        # only its quantized inputs are synthetic in this probe.
                        mlp_input = MoEMlpComputeInput(
                            hidden_states=x.to(device),
                            group_list=group_list,
                            group_list_type=0,
                            dynamic_scale=None,
                            topk_scales=None,
                            weights=MoEWeights(w1=layer.w13_weight, w2=layer.w2_weight),
                            quant=MoEQuantParams(quant_type=scheme.quant_type),
                            fusion=False,
                            layer=layer,
                        )
                        result = scheme.apply_gmm2(mlp_input, quantized, x_scale)
                    else:
                        result = torch_npu.npu_grouped_matmul(
                            x=[quantized],
                            weight=[layer.w13_weight],
                            scale=None,
                            antiquant_scale=[layer.w13_weight_scale],
                            per_token_scale=[x_scale],
                            split_item=2,
                            group_type=0,
                            group_list=group_list,
                            group_list_type=0,
                            output_dtype=torch.bfloat16,
                            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
                            x_dtype=torch.float8_e4m3fn,
                            weight_dtype=torch_npu.float4_e2m1fn_x2,
                        )[0]
                    torch.npu.synchronize()
                    actual = result.float().cpu()
                    case["max_absolute_error"] = (actual - reference).abs().max().item()
                    case["last_group_nonzero"] = bool(
                        torch.all(actual[1::TOKENS_PER_EXPERT] != 0)
                    )
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
