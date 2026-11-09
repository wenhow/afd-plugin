# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Measure the locked V4.1 RMSNorm outputs and routing precision.

Only native normalization and gate/top-k operators run. Inputs, norm weights,
and router weights are deterministic synthetic tensors; no checkpoint weights
or MoE experts are loaded. The decoder constructor is explicitly bypassed to
call its unchanged rms_norm_cast method with one real RMSNorm module. This is
component evidence, not complete decoder/model or checkpoint MoE acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from importlib.metadata import version
from pathlib import Path

CHECKPOINT = "/mnt/workspace/models/DeepSeek-V4.1-Flash"
MAX_TIMEOUT_SECONDS = 180
REAP_TIMEOUT_SECONDS = 10
SOURCE_CONTROL_HIDDEN_SIZE = 7168
SOURCE_CONTROL_EPSILON = 1e-6
TOKEN_COUNTS = (1, 16, 128)
INPUT_SCALES = (1.0, 1e-4, 1e3)
RANDOM_SEED = 20261107
NORM_RTOL = 2e-2
NORM_ATOL = 2e-2
ROUTER_EPSILON = 1e-20
RESULT_PREFIX = "NORM_PRECISION_RESULT="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path(CHECKPOINT))
    parser.add_argument("--timeout", type=int, default=MAX_TIMEOUT_SECONDS)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--allow-fallback",
        action="store_true",
        help="measure the native decoder fallback when compiled custom ops are absent",
    )
    mode.add_argument(
        "--native-path",
        action="store_true",
        help="use the native decoder wheel/profile selection without a capability gate",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if not REAP_TIMEOUT_SECONDS < args.timeout <= MAX_TIMEOUT_SECONDS:
        parser.error("--timeout must be greater than 10 and at most 180 seconds")
    return args


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def run_child(args: argparse.Namespace, *, native_path: bool = False) -> int:
    started = time.monotonic()
    report = {
        "success": False,
        "pid": os.getpid(),
        "local_npu": args.device,
        "checkpoint_weights_loaded": False,
        "native_moe_executed": False,
        "complete_model_constructed": False,
        "decoder_constructor_bypassed": True,
        "normalization_constructor_bypassed": False,
        "decoder_wheel_ops_forced_off": True,
        "production_guard_modified": False,
        "weights": (
            "deterministic synthetic BF16 norm/gate weights; gate FP32 sidecar "
            "is widened from the BF16 weight, not the original random sample"
        ),
        "hash_lookup_and_vision_routing_tested": False,
        "cases": [],
    }
    result_path = args.output_dir / "rank-result.json"
    try:
        # Runtime imports belong to this isolated NPU child. Ascend ops must be
        # registered before importing the native model to avoid circular imports.
        import torch
        import torch_npu
        import vllm_ascend.ops  # noqa: F401
        from vllm.config import (
            CompilationMode,
            CUDAGraphMode,
            VllmConfig,
            set_current_vllm_config,
        )
        from vllm.config.device import DeviceConfig
        from vllm.platforms import current_platform
        from vllm.plugins import load_general_plugins

        current_platform.pre_register_and_update()
        load_general_plugins()

        from vllm_ascend.device.device_op import DeviceOperator
        from vllm_ascend.device.hardware_profile import (
            HardwareCapability,
            get_current_hardware_profile,
        )
        from vllm_ascend.models.deepseek_v41 import model as native
        from vllm_ascend.utils import enable_custom_op

        checkpoint_config_path = args.checkpoint / "config.json"
        config_bytes = checkpoint_config_path.read_bytes()
        checkpoint_config = json.loads(config_bytes)
        text_config = checkpoint_config["text_config"]
        hidden_size = text_config["hidden_size"]
        norm_epsilon = text_config["rms_norm_eps"]
        expert_count = text_config["n_routed_experts"]
        top_k = text_config["num_experts_per_tok"]
        assert checkpoint_config["model_type"] == "deepseek_v41"
        assert text_config["scoring_func"] == "sqrtsoftplus"
        assert text_config["norm_topk_prob"] is True
        assert text_config.get("n_group", 1) == 1
        assert text_config.get("topk_group", 1) == 1
        assert 0 < top_k <= expert_count
        torch.npu.set_device(args.device)
        torch.npu.matmul.allow_hf32 = False
        profile = get_current_hardware_profile()
        report.update(
            versions={
                name: version(name)
                for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "fastapi")
            },
            device_name=torch.npu.get_device_name(args.device),
            device_adaptor=DeviceOperator.__name__,
            supports_rms_norm_cast=profile.supports(HardwareCapability.RMS_NORM_CAST),
            custom_ops_enabled=enable_custom_op(),
            matmul_allow_hf32=torch.npu.matmul.allow_hf32,
            checkpoint_config={
                "path": str(checkpoint_config_path),
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
                "hidden_size": hidden_size,
                "rms_norm_eps": norm_epsilon,
                "n_routed_experts": expert_count,
                "num_experts_per_tok": top_k,
                "scoring_func": text_config["scoring_func"],
                "quantization_config": checkpoint_config.get("quantization_config"),
            },
            source={
                "native_model_path": native.__file__,
                "native_model_sha256": hashlib.sha256(
                    Path(native.__file__).read_bytes()
                ).hexdigest(),
                "decoder_norm_method_sha256": hashlib.sha256(
                    inspect.getsource(
                        native.DeepseekV41DecoderLayer.rms_norm_cast
                    ).encode()
                ).hexdigest(),
                "device_norm_method_sha256": hashlib.sha256(
                    inspect.getsource(DeviceOperator.rms_norm_cast).encode()
                ).hexdigest(),
            },
        )
        report["fused_norm_available"] = (
            report["supports_rms_norm_cast"] and report["custom_ops_enabled"]
        )
        report["native_router_topk_available"] = report["custom_ops_enabled"]
        if native_path:
            wheel_ops = native._DS41_WHEEL_OPS
            packed_cache = native.uses_a5_packed_cache()
            fused_op_registered = (
                getattr(torch.ops._C_ascend, "npu_rms_norm_cast", None) is not None
            )
            native_branch = (
                "a5_wheel_npu_rms_norm"
                if wheel_ops and packed_cache
                else (
                    "device_rms_norm_cast"
                    if report["fused_norm_available"] and fused_op_registered
                    else "decoder_rmsnorm_fallback"
                )
            )
            report.update(
                native_path=True,
                decoder_wheel_ops_forced_off=False,
                native_norm_selection={
                    "ds41_wheel_ops_imported": wheel_ops,
                    "uses_a5_packed_cache": packed_cache,
                    "decoder_ds41_wheel_ops": wheel_ops and packed_cache,
                    "device_fused_norm_registered": fused_op_registered,
                    "branch": native_branch,
                },
                native_router_topk_available=(
                    getattr(torch.ops._C_ascend, "moe_gating_top_k_hash", None)
                    is not None
                ),
                native_router_topk_success=None,
            )
            cache_source = Path(inspect.getsourcefile(native.uses_a5_packed_cache))
            report["source"].update(
                native_cache_config_path=str(cache_source),
                native_cache_config_sha256=hashlib.sha256(
                    cache_source.read_bytes()
                ).hexdigest(),
                native_wheel_selection_method_sha256=hashlib.sha256(
                    inspect.getsource(native.uses_a5_packed_cache).encode()
                ).hexdigest(),
            )
        if not args.allow_fallback and not native_path:
            assert report["supports_rms_norm_cast"], (
                "target profile lacks RMS_NORM_CAST"
            )
            assert report["custom_ops_enabled"], "native custom ops are unavailable"
        config = VllmConfig(device_config=DeviceConfig(device="cpu"))
        config.device_config = DeviceConfig(device="npu")
        config.compilation_config.mode = CompilationMode.NONE
        config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        config.compilation_config.custom_ops = ["all"]
        shape_specs = [("checkpoint", hidden_size, norm_epsilon)]
        if (hidden_size, norm_epsilon) != (
            SOURCE_CONTROL_HIDDEN_SIZE,
            SOURCE_CONTROL_EPSILON,
        ):
            shape_specs.append(
                (
                    "upstream_operator_test",
                    SOURCE_CONTROL_HIDDEN_SIZE,
                    SOURCE_CONTROL_EPSILON,
                )
            )

        def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
            actual = actual.detach().cpu().double()
            expected = expected.detach().cpu().double()
            difference = (actual - expected).abs()
            return {
                "max_abs": difference.max().item(),
                "mean_abs": difference.mean().item(),
                "rms_abs": difference.square().mean().sqrt().item(),
                "different_elements": torch.count_nonzero(difference).item(),
                "elements": difference.numel(),
                "exact_equal": torch.equal(actual, expected),
            }

        def native_topk(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # This is the exact native sqrtsoftplus operator branch with no
            # token-id table or vision bias, not a replacement MoE/router model.
            weights, indices, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
                x=logits,
                k=top_k,
                bias=torch.zeros(expert_count, dtype=torch.float32, device="npu"),
                input_ids=None,
                tid2eid=None,
                k_group=1,
                group_count=1,
                routed_scaling_factor=text_config["routed_scaling_factor"],
                eps=ROUTER_EPSILON,
                group_select_mode=1,
                renorm=0,
                norm_type=2,
                out_flag=False,
                bias_vl=None,
                image_sentinel_lo=text_config.get("image_sentinel_base_id", 129257),
                image_sentinel_count=5,
            )
            return weights, indices

        with set_current_vllm_config(config), torch.inference_mode():
            for shape_name, width, epsilon in shape_specs:
                generator = torch.Generator(device="cpu").manual_seed(
                    RANDOM_SEED + width
                )
                norm = native.RMSNorm(width, eps=epsilon, dtype=torch.bfloat16).to(
                    "npu"
                )
                gamma_cpu = (torch.randn(width, generator=generator) / 2 + 1).to(
                    torch.bfloat16
                )
                norm.weight.copy_(gamma_cpu.to("npu"))
                # Full decoder construction would allocate attention, MoE and
                # mHC parameters. Only the two fields read by this unchanged
                # native method are installed; real RMSNorm construction ran.
                decoder = object.__new__(native.DeepseekV41DecoderLayer)
                torch.nn.Module.__init__(decoder)
                decoder.post_attention_layernorm = norm
                decoder._ds41_wheel_ops = (
                    wheel_ops and packed_cache if native_path else False
                )
                gate_weight = (
                    torch.randn(expert_count, width, generator=generator) / width**0.5
                ).to(device="npu", dtype=torch.bfloat16)
                gate_weight_fp32 = gate_weight.float()
                for rows in TOKEN_COUNTS:
                    base = torch.randn(rows, width, generator=generator)
                    for scale in INPUT_SCALES:
                        case = {
                            "shape_source": shape_name,
                            "tokens": rows,
                            "hidden_size": width,
                            "epsilon": epsilon,
                            "input_scale": scale,
                        }
                        report["cases"].append(case)
                        x_cpu = (base * scale).to(torch.bfloat16)
                        x = x_cpu.to("npu")
                        if native_path:
                            case["native_norm_success"] = False
                            decoder_rounded, decoder_fp32 = decoder.rms_norm_cast(x)
                            rounded, fp32 = decoder_rounded, decoder_fp32
                            outputs = None
                            case["execution_path"] = native_branch
                        else:
                            outputs = DeviceOperator.rms_norm_cast(
                                x, norm.weight, epsilon
                            )
                            decoder_rounded, decoder_fp32 = decoder.rms_norm_cast(x)
                            if outputs is None:
                                assert args.allow_fallback, (
                                    "native fused norm fell back"
                                )
                                rounded, fp32 = decoder_rounded, decoder_fp32
                                case["execution_path"] = (
                                    "native decoder RMSNorm fallback"
                                )
                            else:
                                rounded, fp32 = outputs
                                case["execution_path"] = "native fused RMSNormCast"
                        npu_reference, _ = torch_npu.npu_rms_norm(
                            x, norm.weight, epsilon
                        )
                        torch.npu.synchronize()
                        assert rounded.dtype == torch.bfloat16
                        assert fp32.dtype == torch.float32
                        assert rounded.shape == fp32.shape == x.shape
                        torch.testing.assert_close(
                            decoder_rounded, rounded, rtol=0, atol=0
                        )
                        torch.testing.assert_close(decoder_fp32, fp32, rtol=0, atol=0)
                        torch.testing.assert_close(
                            rounded, npu_reference, rtol=NORM_RTOL, atol=NORM_ATOL
                        )
                        if native_path:
                            case["native_norm_success"] = True
                        reference_fp64 = x_cpu.double()
                        reference_fp64 = (
                            reference_fp64
                            * torch.rsqrt(
                                reference_fp64.square().mean(-1, keepdim=True) + epsilon
                            )
                            * gamma_cpu.double()
                        )
                        widened = rounded.float()
                        fp32_difference = error_metrics(fp32, widened)
                        case.update(
                            dtypes={
                                "input": str(x.dtype),
                                "rounded": str(rounded.dtype),
                                "fp32": str(fp32.dtype),
                                "norm_weight": str(norm.weight.dtype),
                                "gate_weight": str(gate_weight.dtype),
                                "gate_weight_fp32": str(gate_weight_fp32.dtype),
                            },
                            decoder_matches_device_operator=(
                                None if native_path else outputs is not None
                            ),
                            decoder_fallback_executed=(
                                native_branch == "decoder_rmsnorm_fallback"
                                if native_path
                                else outputs is None
                            ),
                            rounded_vs_npu_rms_norm=error_metrics(
                                rounded, npu_reference
                            ),
                            fp32_vs_bf16_float=fp32_difference,
                            fp32_contains_unrounded_precision=not fp32_difference[
                                "exact_equal"
                            ],
                            fp32_vs_cpu_fp64_rms_norm=error_metrics(
                                fp32, reference_fp64
                            ),
                            bf16_float_vs_cpu_fp64_rms_norm=error_metrics(
                                widened, reference_fp64
                            ),
                        )
                        native_logits = DeviceOperator.compute_gate_logits(
                            rounded,
                            gate_weight,
                            weight_fp32=gate_weight_fp32,
                            prepared_input=fp32,
                        )
                        rebuilt_logits = DeviceOperator.compute_gate_logits(
                            rounded, gate_weight, weight_fp32=gate_weight_fp32
                        )
                        controlled_logits = torch.nn.functional.linear(
                            fp32, gate_weight_fp32
                        )
                        controlled_rebuilt_logits = torch.nn.functional.linear(
                            widened, gate_weight_fp32
                        )
                        case["router"] = {
                            "projection": "native DeviceOperator.compute_gate_logits",
                            "native_topk_executed": False,
                            "synthetic_weights": True,
                            "fp32_weight_source": "gate_weight_bf16.float()",
                            "dynamic_text_route_only": True,
                            "logits_native_fp32_vs_rebuilt": error_metrics(
                                native_logits, rebuilt_logits
                            ),
                            "controlled_fp32_vs_rebuilt": error_metrics(
                                controlled_logits, controlled_rebuilt_logits
                            ),
                            "native_projection_vs_controlled_fp32": error_metrics(
                                native_logits, controlled_logits
                            ),
                            "native_rebuilt_vs_controlled_rebuilt": error_metrics(
                                rebuilt_logits, controlled_rebuilt_logits
                            ),
                        }
                        if report["native_router_topk_available"]:
                            if native_path:
                                report["native_router_topk_success"] = False
                                case["router"]["native_topk_success"] = False
                            native_weights, native_indices = native_topk(native_logits)
                            rebuilt_weights, rebuilt_indices = native_topk(
                                rebuilt_logits
                            )
                            torch.npu.synchronize()
                            index_equal = native_indices.eq(rebuilt_indices)
                            case["router"].update(
                                native_topk_executed=True,
                                topk="native moe_gating_top_k_hash, norm_type=2",
                                topk_weights_native_fp32_vs_rebuilt=error_metrics(
                                    native_weights, rebuilt_weights
                                ),
                                topk_different_positions=torch.count_nonzero(
                                    ~index_equal
                                ).item(),
                                topk_different_rows=torch.count_nonzero(
                                    ~index_equal.all(dim=-1)
                                ).item(),
                                native_topk_ids=native_indices.cpu().tolist(),
                                rebuilt_topk_ids=rebuilt_indices.cpu().tolist(),
                            )
                            if native_path:
                                report["native_router_topk_success"] = True
                                case["router"]["native_topk_success"] = True
                        else:
                            if native_path:
                                case["router"]["native_topk_success"] = None
                            case["router"]["topk_unavailable_reason"] = (
                                "native moe_gating_top_k_hash is not registered"
                                if native_path
                                else "compiled vllm_ascend_C extension is unavailable"
                            )
                        if fp32_difference["exact_equal"]:
                            torch.testing.assert_close(
                                native_logits, rebuilt_logits, rtol=0, atol=0
                            )
                            if report["native_router_topk_available"]:
                                torch.testing.assert_close(
                                    native_weights, rebuilt_weights, rtol=0, atol=0
                                )
                                torch.testing.assert_close(
                                    native_indices, rebuilt_indices, rtol=0, atol=0
                                )
                        if shape_name == "checkpoint" and rows == 16 and scale == 1.0:
                            artifact = args.output_dir / "norm-router-tensors.pt"
                            torch.save(
                                {
                                    "input": x_cpu,
                                    "gamma": gamma_cpu,
                                    "native_rounded": rounded.cpu(),
                                    "native_fp32": fp32.cpu(),
                                    "bf16_float": widened.cpu(),
                                    "npu_rms_norm": npu_reference.cpu(),
                                    "cpu_fp64_reference": reference_fp64,
                                    "gate_weight_bf16": gate_weight.cpu(),
                                    "gate_weight_fp32": gate_weight_fp32.cpu(),
                                    "native_logits": native_logits.cpu(),
                                    "rebuilt_logits": rebuilt_logits.cpu(),
                                    "controlled_fp32_logits": controlled_logits.cpu(),
                                    "controlled_rebuilt_logits": (
                                        controlled_rebuilt_logits.cpu()
                                    ),
                                },
                                artifact,
                            )
                            report["tensor_artifact"] = str(artifact)
                        write_report(result_path, report)
                        print("NORM_PRECISION_CASE=" + json.dumps(case), flush=True)
        all_widened = all(
            case["fp32_vs_bf16_float"]["exact_equal"] for case in report["cases"]
        )
        if native_path:
            assert (
                hashlib.sha256(checkpoint_config_path.read_bytes()).hexdigest()
                == (report["checkpoint_config"]["sha256"])
            ), "checkpoint config changed during the read-only probe"
            report["checkpoint_config_unchanged"] = True
        report.update(
            success=True,
            all_cases_fp32_equal_bf16_float=all_widened,
            matches_locked_upstream_test_expectation=(
                all_widened if report["fused_norm_available"] else None
            ),
            validation_scope=(
                "native fused norm and controlled native routing operators"
                if report["fused_norm_available"]
                else (
                    "isolated non-wheel decoder norm fallback, controlled gate, "
                    "and optional native top-k; not the A5 packed-cache path"
                )
            ),
            measured_transport_conclusion=(
                "BF16.float() reconstructed native FP32 exactly in every tested case; "
                "this conclusion is restricted to the recorded stack, shapes and inputs"
                if all_widened
                else "native FP32 differs from BF16.float(); preserve FP32 payload"
            ),
            production_transport_changed=False,
        )
        if native_path:
            report["validation_scope"] = (
                "native decoder norm selection only: "
                + native_branch
                + "; controlled gate and separately recorded optional native top-k"
            )
            report["matches_locked_upstream_test_expectation"] = None
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_report(result_path, report)
        print(RESULT_PREFIX + json.dumps(report), flush=True)
    return 0 if report["success"] else 1


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path(tempfile.mkdtemp(prefix="afd-norm-precision-"))
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.child:
        return run_child(args, native_path=args.native_path)
    started = time.monotonic()
    result_path = args.output_dir / "result.json"
    rank_path = args.output_dir / "rank-result.json"
    # Always use a fresh destination; stale success must not survive an abort.
    if rank_path.exists() or result_path.exists():
        raise ValueError("--output-dir already contains a precision probe report")
    environment = os.environ.copy()
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--device",
        str(args.device),
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.allow_fallback:
        command.append("--allow-fallback")
    if args.native_path:
        command.append("--native-path")
    timed_out = False
    with (args.output_dir / "probe.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment
        )
        try:
            process.wait(timeout=args.timeout - REAP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait(timeout=REAP_TIMEOUT_SECONDS)
    rank_report = (
        json.loads(rank_path.read_text(encoding="utf-8"))
        if rank_path.exists()
        else None
    )
    report = {
        "success": not timed_out
        and process.returncode == 0
        and bool(rank_report and rank_report["success"]),
        "pid": process.pid,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "supervisor_killed": timed_out,
        "child_reaped": process.poll() is not None,
        "elapsed_seconds": time.monotonic() - started,
        "timeout_seconds": args.timeout,
        "rank_report": rank_report,
    }
    write_report(result_path, report)
    print(RESULT_PREFIX + json.dumps(report), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
