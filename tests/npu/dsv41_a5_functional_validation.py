#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the frozen, non-Engram functional suite on an existing A5 installation.

Default: verify source identities and metadata only. --run: five normal Worker
scenes and two DSpark/PD components, sequentially, with bounded owned cleanup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import dsv41_dspark_a5_validation as shared

FAMILY = "reports/dsv41-a5-functional"
PROBE = "tests/npu/dsv41_functional_worker_probe.py"
NORMAL_SCENES = (
    ("native-eager-u1", "native", "eager-u1", False),
    ("afd-eager-u1", "attention", "eager-u1", False),
    ("afd-eager-u2-streams", "attention", "eager-u2", True),
    ("afd-graph-u1", "attention", "graph-u1", False),
    ("afd-graph-u2-streams", "attention", "graph-u2", True),
)
DEFAULT_TIMEOUT_SECONDS = 480
DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 600
PAIR_EXIT_TIMEOUT_SECONDS = 45
FIRST_AFD_PORT = 31010
PORT_STRIDE = 20
NORMAL_REQUIRED_FLAGS = (
    "success",
    "functional_worker_component_accepted",
    "request_lifecycle_accepted",
    "compile_or_warm_up_returned",
    "loaded_parameters_unchanged",
    "mode_execution_accepted",
)
PARAMETER_IDENTITY_FIELDS = ("shape", "dtype", "numel", "bytes", "sha256", "finite")


def probe_command(
    args, snapshot: Path, output: Path, role: str, device: int
) -> list[str]:
    command = [
        args.python,
        "-B",
        str(snapshot / PROBE),
        "--model",
        args.model,
        "--output-dir",
        str(output),
        "--role",
        role,
        "--mode",
        args._mode,
        "--device",
        str(device),
        "--afd-port",
        str(args._port),
        "--timeout",
        str(args.timeout),
    ]
    if args._overlap:
        command.append("--overlap")
    if args.run:
        command.append("--run")
    return command


def run_pair(args, snapshot: Path, devices: tuple[int, ...]) -> int:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    environment = shared.source_environment(snapshot)
    processes = {}
    report = {"success": False, "roles": {}}
    try:
        # Both init_device calls join the normal AFD group; no fabricated
        # readiness signal, connector or model runner replaces either role.
        for role, device in (("ffn", devices[1]), ("attention", devices[0])):
            with (output / (role + ".log")).open("wb") as log:
                processes[role] = subprocess.Popen(
                    probe_command(args, snapshot, output / role, role, device),
                    env=environment,
                    cwd=snapshot,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        attention_code = processes["attention"].wait(timeout=args.timeout)
        if attention_code:
            raise RuntimeError(f"Attention worker failed: {attention_code}")
        ffn_code = processes["ffn"].wait(timeout=PAIR_EXIT_TIMEOUT_SECONDS)
        if ffn_code:
            raise RuntimeError(f"FFN worker failed: {ffn_code}")
        for role in ("attention", "ffn"):
            result = json.loads((output / role / "result.json").read_text("utf-8"))
            report["roles"][role] = result
            required = (
                NORMAL_REQUIRED_FLAGS
                if role == "attention"
                else (
                    "success",
                    "functional_worker_component_accepted",
                    "compile_or_warm_up_returned",
                    "loaded_parameters_unchanged",
                    "mode_execution_accepted",
                    "ffn_public_poll_observed_peer_stop_and_owner_exit",
                )
            )
            if not all(result.get(name) is True for name in required):
                raise RuntimeError(f"normal {role} functional assertions failed")
            if (
                result.get("cleanup_failures") != []
                or result.get("native_groups_destroyed") is not True
            ):
                raise RuntimeError(f"normal {role} cleanup failed")
        report["success"] = True
    except BaseException:
        report["traceback"] = traceback.format_exc()
    finally:
        for role, process in processes.items():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=PAIR_EXIT_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=PAIR_EXIT_TIMEOUT_SECONDS)
                report.setdefault("terminated_roles", []).append(role)
                report["success"] = False
        report["returncodes"] = {r: p.returncode for r, p in processes.items()}
        shared.write_json(output / "result.json", report)
    return 0 if report["success"] else 1


def normal_command(
    args, snapshot: Path, name: str, role: str, mode: str, overlap: bool, port: int
) -> list[str]:
    output = args.output_dir / name
    if role == "native":
        args._mode, args._port, args._overlap = mode, port, overlap
        return probe_command(args, snapshot, output, role, args._devices[0])
    command = [
        args.python,
        "-B",
        str(snapshot / "tests/npu/dsv41_a5_functional_validation.py"),
        "--_pair",
        "--_snapshot",
        str(snapshot),
        "--_mode",
        mode,
        "--_port",
        str(port),
        "--model",
        args.model,
        "--output-dir",
        str(output),
        "--devices",
        args.devices,
        "--python",
        args.python,
        "--timeout",
        str(args.timeout),
        "--run",
    ]
    if overlap:
        command.append("--_overlap")
    return command


def execute_scene(
    args, command: list[str], environment: dict[str, str], snapshot: Path, name: str
) -> dict:
    log = args.output_dir / (name + ".log")
    if args.run:
        return shared.run_bounded(
            command,
            environment,
            snapshot,
            log,
            args.timeout + shared.OUTER_GRACE_SECONDS,
        )
    completed = subprocess.run(
        command,
        env=environment,
        cwd=snapshot,
        capture_output=True,
        timeout=shared.METADATA_TIMEOUT_SECONDS,
    )
    log.write_bytes(completed.stdout + completed.stderr)
    metadata = json.loads(completed.stdout) if completed.returncode == 0 else {}
    if metadata.get("runtime_imported") is not False:
        raise RuntimeError(f"metadata-only runtime boundary failed: {name}")
    return {
        "command": command,
        "returncode": completed.returncode,
        "success": completed.returncode == 0,
        "metadata": metadata,
    }


def compare_normal_scenes(scenes: list[dict]) -> dict:
    native = scenes[0]["result"]
    expected = native["loaded_model"]["parameters"]
    groups = native["canonical_parameter_groups"]
    outcomes = {}
    for item in scenes[1:]:
        attention = item["result"]["roles"]["attention"]
        ffn = item["result"]["roles"]["ffn"]
        actual = dict(attention["loaded_model"]["parameters"])
        ffn_parameters = ffn["loaded_model"]["parameters"]
        if set(actual) != set(groups["attention_identity_names"]):
            raise ValueError("Attention parameter namespaces differ from native")
        if set(ffn_parameters) != set(groups["ffn_to_native_names"]):
            raise ValueError("FFN parameter namespaces differ from native")
        actual.update(
            {
                groups["ffn_to_native_names"][name]: record
                for name, record in ffn_parameters.items()
            }
        )
        matching_parameters = set(actual) == set(expected) and all(
            all(actual[name][key] == record[key] for key in PARAMETER_IDENTITY_FIELDS)
            for name, record in expected.items()
        )
        lifecycle = attention["request_lifecycle"]
        reference = native["request_lifecycle"]
        matching_samples = (
            lifecycle["prompts"] == reference["prompts"]
            and lifecycle["output_limits"] == reference["output_limits"]
            and lifecycle["outputs"] == reference["outputs"]
        )
        outcomes[item["name"]] = {
            "all_parameter_bytes_match_native": matching_parameters,
            "parameter_count": len(actual),
            "request_samples_match_native": matching_samples,
            "native_outputs": reference["outputs"],
            "afd_outputs": lifecycle["outputs"],
        }
    return {
        "success": all(
            item["all_parameter_bytes_match_native"]
            and item["request_samples_match_native"]
            for item in outcomes.values()
        ),
        "scope": "same bounded BF16 dummy model; no trained checkpoint accuracy claim",
        "scenes": outcomes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--preflight-timeout",
        type=int,
        default=DEFAULT_PREFLIGHT_TIMEOUT_SECONDS,
        help="native runtime import/bootstrap timeout, including cold JIT compilation",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--_pair", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_snapshot", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_mode", default="eager-u1", help=argparse.SUPPRESS)
    parser.add_argument(
        "--_port", type=int, default=FIRST_AFD_PORT, help=argparse.SUPPRESS
    )
    parser.add_argument("--_overlap", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        devices = tuple(map(int, args.devices.split(",")))
    except ValueError:
        parser.error("--devices requires four distinct nonnegative ordinals")
    if len(devices) != 4 or len(set(devices)) != 4 or min(devices) < 0:
        parser.error("--devices requires four distinct nonnegative ordinals")
    args._devices = devices
    if not 0 < args.timeout < shared.SOURCE_RETENTION_SECONDS:
        parser.error("--timeout must be positive and below 720 seconds")
    if args.preflight_timeout <= 0:
        parser.error("--preflight-timeout must be positive")
    executable = shutil.which(args.python)
    if executable is None:
        parser.error("--python must name an existing interpreter")
    # Keep a venv interpreter symlink intact: resolving it loses sys.prefix.
    args.python = os.path.abspath(executable)
    args.output_dir = args.output_dir.absolute()
    if args.run and os.name != "posix":
        parser.error("--run requires the existing A5 Linux environment")
    if args._pair:
        if args._snapshot is None or not args.run:
            parser.error("internal pair requires --_snapshot and --run")
        return run_pair(args, args._snapshot, devices)
    if args.output_dir.exists():
        parser.error("--output-dir must be fresh")
    archive = args.source_bundle or (
        Path(__file__).resolve().parents[2] / FAMILY / "runtime-source.tar.gz"
    )
    archive = archive.resolve(strict=True)
    lock_path = archive.with_name("bundle-lock.json")
    manifest, files = shared.read_bundle(archive, lock_path)
    lock = json.loads(lock_path.read_text("utf-8"))
    if shared.file_sha256(Path(__file__)) != lock["launcher_sha256"]:
        parser.error("launcher differs from the frozen bundle")
    args.output_dir.mkdir(parents=True)
    snapshot = args.output_dir / "snapshot"
    for name, raw in files.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    environment = shared.source_environment(snapshot)
    report = {
        "success": False,
        "run_requested": args.run,
        "preflight_timeout_seconds": args.preflight_timeout,
        "scenes": [],
        "runtime_source_sha256": shared.file_sha256(archive),
        "source_files": len(files),
        "production_files": manifest["production_count"],
        "full_checkpoint_accuracy_accepted": False,
        "engram_accepted": False,
        "physical_overlap_or_performance_accepted": False,
        "npu_acceptance": "pending_on_A5",
    }
    try:
        if args.run:
            before = shared.npu_snapshot(args.output_dir / "npu-before.log")
            report["npu_before"] = before
            if not before["all_eight_idle"]:
                raise RuntimeError("eight idle devices were not confirmed")
            inspect = [
                args.python,
                "-B",
                str(snapshot / "tests/npu/dsv41_dspark_a5_validation.py"),
                "--_inspect-runtime",
                str(args.output_dir / "native-runtime.json"),
                "--_snapshot",
                str(snapshot),
                "--devices",
                args.devices,
            ]
            check = shared.run_bounded(
                inspect,
                environment,
                snapshot,
                args.output_dir / "native-runtime.log",
                args.preflight_timeout,
            )
            report["runtime_preflight"] = check
            if not check["success"]:
                if check["timed_out"]:
                    raise RuntimeError(
                        "A5 native/runtime preflight timed out after "
                        f"{args.preflight_timeout} seconds; "
                        "see native-runtime.log and native-runtime.process.json. "
                        "Increase --preflight-timeout for cold JIT compilation."
                    )
                raise RuntimeError(
                    "A5 native/runtime preflight failed "
                    f"(returncode={check['returncode']}); "
                    "see native-runtime.json and native-runtime.process.json"
                )
        for index, (name, role, mode, overlap) in enumerate(NORMAL_SCENES):
            if args.run:
                command = normal_command(
                    args,
                    snapshot,
                    name,
                    role,
                    mode,
                    overlap,
                    FIRST_AFD_PORT + index * PORT_STRIDE,
                )
            else:
                args._mode, args._port, args._overlap = mode, FIRST_AFD_PORT, overlap
                command = probe_command(
                    args, snapshot, args.output_dir / name, role, devices[0]
                )
            process = execute_scene(args, command, environment, snapshot, name)
            item = {"name": name, "process": process}
            report["scenes"].append(item)
            if not process["success"]:
                raise RuntimeError(f"normal Worker scene failed: {name}")
            if args.run:
                result = json.loads(
                    (args.output_dir / name / "result.json").read_text("utf-8")
                )
                if result.get("success") is not True:
                    raise RuntimeError(f"normal Worker report failed: {name}")
                if role == "native" and not all(
                    result.get(k) is True for k in NORMAL_REQUIRED_FLAGS
                ):
                    raise RuntimeError("native Worker functional assertions failed")
                item["result"] = result
                idle = shared.npu_snapshot(args.output_dir / (name + "-npu.log"))
                item["npu_after"] = idle
                if not idle["all_eight_idle"]:
                    raise RuntimeError(f"devices not idle after {name}")
        if args.run:
            report["native_afd_comparison"] = compare_normal_scenes(report["scenes"])
            if not report["native_afd_comparison"]["success"]:
                raise RuntimeError(
                    "normal native/AFD parameter or sampled-token comparison failed"
                )
        for scene in shared.SCENES:
            name = "dspark-pd-" + scene
            output = args.output_dir / name
            command = shared.scene_command(
                args.python,
                snapshot,
                args.model,
                output,
                args.devices,
                args.timeout,
                scene,
                run=args.run,
            )
            if scene == shared.SCENES[1]:
                command.extend(
                    [
                        "--graph-compute-overlap",
                        "--graph-all-axes",
                        "--graph-exchange-order",
                        "cross_layer_v1",
                    ]
                )
            process = execute_scene(args, command, environment, snapshot, name)
            item = {"name": name, "process": process}
            report["scenes"].append(item)
            if not process["success"]:
                raise RuntimeError(f"DSpark/PD scene failed: {scene}")
            if args.run:
                item["functional_verification"] = shared.verify_scene(scene, output)
                if scene == shared.SCENES[1]:
                    original = json.loads((output / "summary.json").read_text("utf-8"))
                    if (
                        original["pair_protocols"]["consumer"].get(
                            "native_dspark_graph_cross_layer_lifecycle_component_accepted"
                        )
                        is not True
                    ):
                        raise RuntimeError(
                            "DSpark Graph/U2 cross-layer lifecycle failed"
                        )
                idle = shared.npu_snapshot(args.output_dir / (name + "-npu.log"))
                item["npu_after"] = idle
                if not idle["all_eight_idle"]:
                    raise RuntimeError(f"devices not idle after {name}")
        report["success"] = True
        report["npu_acceptance"] = (
            "component_suite_passed" if args.run else "pending_on_A5"
        )
    except BaseException:
        report["traceback"] = traceback.format_exc()
    finally:
        if args.run:
            try:
                report["npu_after"] = shared.npu_snapshot(
                    args.output_dir / "npu-after.log"
                )
                if not report["npu_after"]["all_eight_idle"]:
                    report["success"] = False
            except BaseException:
                report["cleanup_collection_error"] = traceback.format_exc()
                report["success"] = False
        if args.run and not report["success"]:
            report["npu_acceptance"] = "failed"
        shared.write_json(args.output_dir / "summary.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "scenes"}, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
