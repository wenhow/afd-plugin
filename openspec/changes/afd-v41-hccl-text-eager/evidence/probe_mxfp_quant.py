# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded A3 MXFP8 API availability measurements with isolated child processes.

The base case matches quantization/methods/base.py's call exactly. A separate
process tests explicit rint/reduce_align_type=0 arguments. Neither case loads
checkpoint weights or constructs a model, gate, MoE or distributed group.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

HIDDEN_SIZE = 5120
TOKENS = 16
CASE_NAMES = ("base", "explicit-rint-align0")
MAX_TIMEOUT_SECONDS = 180
REAP_TIMEOUT_SECONDS = 10


def run_child(args: argparse.Namespace) -> int:
    started = time.monotonic()
    report = {
        "success": False,
        "pid": os.getpid(),
        "case": args.case,
        "local_npu": args.device,
        "shape": [TOKENS, HIDDEN_SIZE],
        "native_moe_executed": False,
        "checkpoint_weights_loaded": False,
        "parameters": {"dst_type": "torch.float8_e4m3fn"},
    }
    try:
        # Keep native runtime imports inside each disposable NPU child.
        import torch
        import torch_npu

        torch.npu.set_device(args.device)
        report["device_name"] = torch.npu.get_device_name(args.device)
        report["versions"] = {
            name: version(name)
            for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "fastapi")
        }
        values = torch.arange(TOKENS * HIDDEN_SIZE, dtype=torch.float32).reshape(
            TOKENS, HIDDEN_SIZE
        )
        hidden_states = (values / 8192 - 5).to(device="npu", dtype=torch.bfloat16)
        if args.case == "base":
            # Exact locked methods/base.py call: optional arguments use the
            # installed torch-npu defaults, not a device adaptor's guard.
            quantized, scales = torch_npu.npu_dynamic_mx_quant(
                hidden_states, dst_type=torch.float8_e4m3fn
            )
        else:
            report["parameters"].update(round_mode="rint", reduce_align_type=0)
            quantized, scales = torch_npu.npu_dynamic_mx_quant(
                hidden_states,
                dst_type=torch.float8_e4m3fn,
                round_mode="rint",
                reduce_align_type=0,
            )
        # A Python return alone does not prove asynchronous device work passed.
        torch.npu.synchronize()
        report["outputs"] = {
            "quantized_shape": list(quantized.shape),
            "quantized_dtype": str(quantized.dtype),
            "scales_shape": list(scales.shape),
            "scales_dtype": str(scales.dtype),
        }
        artifact = args.output_dir / (args.case + "-tensors.pt")
        torch.save(
            {
                "input": hidden_states.cpu(),
                "quantized_bytes": quantized.view(torch.uint8).cpu(),
                "scale_bytes": scales.view(torch.uint8).cpu(),
            },
            artifact,
        )
        report["tensor_artifact"] = str(artifact)
        report["success"] = True
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    report["elapsed_seconds"] = time.monotonic() - started
    (args.output_dir / (args.case + ".json")).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("MXFP_QUANT_CASE=" + json.dumps(report), flush=True)
    return 0 if report["success"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=MAX_TIMEOUT_SECONDS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", choices=CASE_NAMES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        args.device < 0
        or not 2 * REAP_TIMEOUT_SECONDS < args.timeout <= MAX_TIMEOUT_SECONDS
    ):
        parser.error("device must be nonnegative and timeout must be 21..180 seconds")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.case:
        return run_child(args)
    started = time.monotonic()
    per_case_timeout = args.timeout / len(CASE_NAMES) - REAP_TIMEOUT_SECONDS
    results = []
    for case_name in CASE_NAMES:
        report_path = args.output_dir / (case_name + ".json")
        if report_path.exists():
            raise ValueError("output directory contains a previous MXFP report")
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--case",
            case_name,
            "--device",
            str(args.device),
            "--output-dir",
            str(args.output_dir),
        ]
        timed_out = False
        with (args.output_dir / (case_name + ".log")).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                process.wait(timeout=per_case_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait(timeout=REAP_TIMEOUT_SECONDS)
        child_report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.exists()
            else None
        )
        results.append(
            {
                "case": case_name,
                "exit_code": process.returncode,
                "pid": process.pid,
                "timed_out": timed_out,
                "supervisor_killed": timed_out,
                "child_reaped": process.poll() is not None,
                "report": child_report,
            }
        )
    report = {
        "success": all(
            item["exit_code"] == 0
            and not item["timed_out"]
            and item["report"]
            and item["report"]["success"]
            for item in results
        ),
        "cases": results,
        "elapsed_seconds": time.monotonic() - started,
        "timeout_seconds": args.timeout,
        "native_moe_executed": False,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("MXFP_QUANT_RESULT=" + json.dumps(report), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
