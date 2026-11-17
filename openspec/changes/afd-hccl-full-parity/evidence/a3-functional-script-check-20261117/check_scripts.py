#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the frozen A5 suite's shared script paths on the actual A3 runtime.

This does not run the A5 bootstrap or accept A5 inference. The source archive is
unchanged. Child processes isolate native configuration, readback and loading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import traceback
from pathlib import Path

CONFIGURATIONS = (
    ("native", "eager-u1", False),
    ("attention", "eager-u1", False),
    ("ffn", "eager-u1", False),
    ("attention", "eager-u2", False),
    ("ffn", "eager-u2", False),
    ("attention", "graph-u1", False),
    ("ffn", "graph-u1", False),
    ("attention", "graph-u2", False),
    ("ffn", "graph-u2", False),
    ("attention", "eager-u2", True),
    ("ffn", "eager-u2", True),
    ("attention", "graph-u2", True),
    ("ffn", "graph-u2", True),
)
DEVICE_TIMEOUT_SECONDS = 480
CONFIG_TIMEOUT_SECONDS = 600
METADATA_TIMEOUT_SECONDS = 30
SUPERVISION_TIMEOUT_SECONDS = 2


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def readback_check(device: int) -> dict:
    # Explicit device leaf: use real ND/NZ tensors and the frozen observer.
    import torch
    import torch_npu
    from dsv41_normal_model_probe import parameter_record

    torch.npu.set_device(device)
    values = [value / 16 for value in range(17 * 19)]
    # Create owning 2-D storage: a view of a 1-D allocation can retain an
    # ineligible 1-D storage descriptor even when its logical shape is 2-D.
    base = torch.tensor(
        [values[start : start + 19] for start in range(0, len(values), 19)],
        dtype=torch.float16,
        device=f"npu:{device}",
    )
    # The installed _npuConfig exposes only a setter for this option. Its
    # native getter returns the current value, including the registered default.
    original_internal_format = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    assert original_internal_format in (b"enable", b"disable")
    try:
        # A3 defaults to disabling internal formats too. Create a real NZ
        # sample, then restore the setting before exercising the observer.
        torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": "enable"})
        nz = torch_npu.npu_format_cast(base, torch_npu.Format.FRACTAL_NZ)
    finally:
        torch.npu.set_option(
            {"ALLOW_INTERNAL_FORMAT": original_internal_format.decode()}
        )
    cases = (
        ("nd", base, struct.pack("<323e", *values)),
        (
            "nz_with_padding",
            nz,
            struct.pack("<323e", *values),
        ),
        (
            "scalar_bfloat16",
            torch.tensor(-2.5, dtype=torch.bfloat16, device=f"npu:{device}"),
            bytes.fromhex("20c0"),
        ),
    )
    records = []
    for name, tensor, expected_bytes in cases:
        original_format = int(torch_npu.get_npu_format(tensor))
        original_pointer = tensor.data_ptr()
        original_stride = list(tensor.stride())
        first = parameter_record(tensor)
        second = parameter_record(tensor)
        assert first == second, name
        assert first["sha256"] == hashlib.sha256(expected_bytes).hexdigest(), name
        assert first["bytes"] == len(expected_bytes), name
        assert first["finite"] is True, name
        assert tensor.data_ptr() == original_pointer == first["data_ptr"], name
        assert list(tensor.stride()) == original_stride == first["stride"], name
        assert int(torch_npu.get_npu_format(tensor)) == original_format, name
        records.append({"case": name, "original_format": original_format, **first})
        print(
            json.dumps({"readback_case": name, "original_format": original_format}),
            flush=True,
        )
    assert records[0]["original_format"] == int(torch_npu.Format.ND)
    assert records[1]["original_format"] != int(torch_npu.Format.ND)
    torch.npu.synchronize()
    assert (
        torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == original_internal_format
    )
    return {
        "cases": records,
        "actual_device": torch.npu.current_device(),
        "allow_internal_format_restored": True,
    }


def load_check(args: argparse.Namespace, report: dict) -> None:
    # Use the actual native Worker and current functional configuration. Stop
    # after loading/observation; A3's deferred inference diagnosis stays paused.
    import torch
    from dsv41_functional_worker_probe import build_config, execution_environment
    from dsv41_normal_native_probe import record_native_model
    from vllm.config import set_current_vllm_config
    from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel

    os.environ.update(execution_environment(args))
    config = build_config(args, report)
    wrapper = WorkerWrapperBase()
    constructed = False
    try:
        wrapper.init_worker(
            [
                {
                    "vllm_config": config,
                    "local_rank": args.device,
                    "rank": 0,
                    "distributed_init_method": (args.output_dir / "store").as_uri(),
                    "is_driver_worker": True,
                }
            ]
        )
        constructed = True
        wrapper.init_device()
        report["worker_initialized"] = True
        worker = wrapper.worker
        assert worker.device == torch.device("npu", args.device)
        with set_current_vllm_config(config), torch.inference_mode():
            wrapper.load_model()
            report["model_loaded"] = True
            report["loaded_model"] = record_native_model(worker)
            report["readback_repeated_identically"] = (
                record_native_model(worker) == report["loaded_model"]
            )
            assert report["readback_repeated_identically"]
    finally:
        callbacks = [("worker", wrapper.shutdown)] if constructed else []
        callbacks += [
            ("ascend_groups", destroy_ascend_model_parallel),
            ("model_groups", destroy_model_parallel),
            ("distributed_environment", destroy_distributed_environment),
        ]
        failures = []
        for name, callback in callbacks:
            try:
                callback()
            except BaseException:
                failures.append({"phase": name, "traceback": traceback.format_exc()})
        report["cleanup_failures"] = failures
        report["native_groups_destroyed"] = not torch.distributed.is_initialized()
        assert not failures
        assert report["native_groups_destroyed"]


def run_child(args: argparse.Namespace) -> int:
    args.output_dir.mkdir()
    report = {"success": False, "kind": args.child, "inference_executed": False}
    try:
        if args.child == "readback":
            report.update(readback_check(args.device))
        else:
            load_check(args, report)
        report["success"] = True
    except BaseException:
        report["traceback"] = traceback.format_exc()
    finally:
        report["afd_modules_imported"] = sorted(
            name
            for name in sys.modules
            if name == "afd_plugin" or name.startswith("afd_plugin.")
        )
        if args.child == "load" and report["afd_modules_imported"]:
            report["success"] = False
        write_json(args.output_dir / "result.json", report)
    return 0 if report["success"] else 1


def run_checks(args: argparse.Namespace) -> int:
    import dsv41_dspark_a5_validation as shared

    args.output_dir.mkdir()
    snapshot = args.snapshot
    environment = shared.source_environment(snapshot)
    result = {
        "success": False,
        "scope": "A3 shared script checks; no A5 inference acceptance",
        "checks": [],
        "a5_acceptance": False,
    }

    def run(name: str, command: list[str], timeout: int) -> dict:
        process = shared.run_bounded(
            command, environment, snapshot, args.output_dir / (name + ".log"), timeout
        )
        result["checks"].append({"name": name, "process": process})
        write_json(args.output_dir / "summary.json", result)
        print(
            json.dumps({"name": name, "returncode": process["returncode"]}), flush=True
        )
        return process

    launcher = snapshot / "tests/npu/dsv41_a5_functional_validation.py"
    probe = snapshot / "tests/npu/dsv41_functional_worker_probe.py"
    common = [
        sys.executable,
        "-B",
        str(launcher),
        "--model",
        args.model,
        "--source-bundle",
        str(snapshot.parent / "bundle/runtime-source.tar.gz"),
    ]
    identity = {}
    try:
        result["npu_before"] = shared.npu_snapshot(args.output_dir / "npu-before.log")
        assert result["npu_before"]["all_eight_idle"]
        identity = shared.installed_sources(snapshot)
        write_json(args.output_dir / "native-source-identity-before.json", identity)
        assert identity["sources_match"] and identity["source_count"] == 194

        path = args.output_dir / "metadata"
        assert run(
            "metadata", common + ["--output-dir", str(path)], METADATA_TIMEOUT_SECONDS
        )["success"]
        metadata = json.loads((path / "summary.json").read_text())
        assert metadata["success"] and len(metadata["scenes"]) == 7
        assert all(
            not s["process"]["metadata"]["runtime_imported"] for s in metadata["scenes"]
        )
        invalid = (
            ("bad-devices", ["--devices", "0,0,1,2"]),
            ("bad-timeout", ["--timeout", "0"]),
            ("bad-preflight-timeout", ["--preflight-timeout", "0"]),
        )
        for name, options in invalid:
            output = args.output_dir / name
            process = run(
                name,
                common + ["--output-dir", str(output), *options],
                METADATA_TIMEOUT_SECONDS,
            )
            assert process["returncode"] == 2 and not process["timed_out"]
            assert not output.exists()

        for code in (0, 7):
            process = run(
                "process-exit-" + str(code),
                [sys.executable, "-B", "-c", f"raise SystemExit({code})"],
                METADATA_TIMEOUT_SECONDS,
            )
            assert process["returncode"] == code and process["success"] is (code == 0)
            assert (
                not process["forced_killed_pids"]
                and not process["surviving_owned_pids"]
            )
        process = run(
            "process-timeout",
            [
                sys.executable,
                "-B",
                "-c",
                "import signal; print('ready', flush=True); signal.pause()",
            ],
            SUPERVISION_TIMEOUT_SECONDS,
        )
        assert process["timed_out"] and not process["success"]
        assert process["forced_killed_pids"] and not process["surviving_owned_pids"]
        assert "ready" in (args.output_dir / "process-timeout.log").read_text()

        preflight = args.output_dir / "a5-preflight.json"
        command = [
            sys.executable,
            "-B",
            str(snapshot / "tests/npu/dsv41_dspark_a5_validation.py"),
            "--_inspect-runtime",
            str(preflight),
            "--_snapshot",
            str(snapshot),
        ]
        process = run("a5-hardware-guard", command, DEVICE_TIMEOUT_SECONDS)
        guard = json.loads(preflight.read_text())
        assert process["returncode"] == 1 and not process["timed_out"]
        assert guard["built_device_type"] == "A3"
        assert "accepts the real A5 installation only" in guard["traceback"]
        assert guard["model_executed"] is guard["operator_kernel_executed"] is False

        for child in ("readback", "load"):
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--snapshot",
                str(snapshot),
                "--model",
                args.model,
                "--output-dir",
                str(args.output_dir / child),
                "--device",
                str(args.device),
                "--child",
                child,
            ]
            assert run(child, command, DEVICE_TIMEOUT_SECONDS)["success"]
            record = json.loads((args.output_dir / child / "result.json").read_text())
            assert record["success"] and not record["inference_executed"]
        for role, mode, overlap in CONFIGURATIONS:
            name = "config-" + role + "-" + mode + ("-overlap" if overlap else "")
            output = args.output_dir / name
            command = [
                sys.executable,
                "-B",
                str(probe),
                "--model",
                args.model,
                "--role",
                role,
                "--mode",
                mode,
                "--output-dir",
                str(output),
                "--run-config",
            ]
            if overlap:
                command.append("--overlap")
            assert run(name, command, CONFIG_TIMEOUT_SECONDS)["success"]
            record = json.loads((output / "result.json").read_text())
            assert (
                record["success"]
                and record["initial_engine_args_configuration_accepted"]
            )
            assert (
                record["npu_initialized_before_config"]
                is record["npu_initialized_after_config"]
                is False
            )

        result["success"] = True
    except BaseException:
        result["traceback"] = traceback.format_exc()
    finally:
        after = shared.installed_sources(snapshot)
        write_json(args.output_dir / "native-source-identity-after.json", after)
        result["native_sources_unchanged"] = after == identity
        result["npu_after"] = shared.npu_snapshot(args.output_dir / "npu-after.log")
        result["success"] = (
            result["success"]
            and result["native_sources_unchanged"]
            and result["npu_after"]["all_eight_idle"]
        )
        write_json(args.output_dir / "summary.json", result)
    return 0 if result["success"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--child", choices=("readback", "load"))
    args = parser.parse_args()
    args.snapshot = args.snapshot.resolve(strict=True)
    args.output_dir = args.output_dir.absolute()
    if args.output_dir.exists():
        parser.error("--output-dir must be fresh")
    sys.path[:0] = [str(args.snapshot / "tests/npu"), str(args.snapshot)]
    args.run, args.run_config = True, False
    args.role, args.mode, args.overlap = "native", "eager-u1", False
    args.afd_port, args.timeout = 30170, DEVICE_TIMEOUT_SECONDS
    return run_child(args) if args.child else run_checks(args)


if __name__ == "__main__":
    raise SystemExit(main())
