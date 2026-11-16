#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run two frozen DSpark functional components on the existing A5 installation.

Without --run, validate the source bundle and execute only the two original
probes' stdlib metadata paths. --run additionally checks the installed A5
runtime and runs the original four-role probes sequentially. No package,
operator vendor, loader path, model checkpoint or shared environment is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path, PurePosixPath

BASELINE_SHA256 = "af0c18647dc8f25bb42a901ae40a74bb0f796f75ceef23dc6259ca50964cebdd"
FAMILY = "reports/dsv41-a5-dspark-functional"
PROBE = "tests/npu/hccl_native_attention_pd_dspark_decode_probe.py"
SCENES = ("reject-continue-ready", "graph-u2-cancel-replace")
ROLES = ("kv_producer", "kv_consumer", "ffn_producer", "ffn_consumer")
DEFAULT_TIMEOUT_SECONDS = 360
SOURCE_RETENTION_SECONDS = 720
OUTER_GRACE_SECONDS = 60
INSPECTION_TIMEOUT_SECONDS = 120
METADATA_TIMEOUT_SECONDS = 20
REAP_TIMEOUT_SECONDS = 30
POLL_SECONDS = 0.2
HASH_CHUNK_BYTES = 1024 * 1024
PACKAGE_NAMES = (
    "torch",
    "torch-npu",
    "vllm",
    "vllm-ascend",
    "vllm-afd-plugin",
    "custom-ops",
    "cann-ops-transformer",
)
RUNTIME_ENVIRONMENT = (
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_CUSTOM_OPP_PATH",
    "LD_LIBRARY_PATH",
    "ASCEND_RT_VISIBLE_DEVICES",
    "PYTHONPATH",
)
LIBRARY_NAMES = (
    "libcust_opapi.so",
    "libascendcl.so",
    "libopapi.so",
    "libhccl.so",
    "libtorch_npu.so",
    "libatb.so",
    "libtransformer_aicpu_kernels.so",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, report: dict) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def bundle_paths(argument: Path | None) -> tuple[Path, Path]:
    if argument is not None:
        archive = argument.resolve(strict=True)
    else:
        sibling = Path(__file__).resolve().with_name("runtime-source.tar.gz")
        archive = (
            sibling
            if sibling.is_file()
            else Path(__file__).resolve().parents[2] / FAMILY / "runtime-source.tar.gz"
        )
        archive = archive.resolve(strict=True)
    return archive, archive.with_name("bundle-lock.json")


def read_bundle(archive: Path, lock_path: Path) -> tuple[dict, dict[str, bytes]]:
    lock = json.loads(lock_path.read_text("utf-8"))
    if file_sha256(archive) != lock["runtime_source_sha256"]:
        raise ValueError("runtime source archive differs from the bundle lock")
    if lock["baseline_archive_sha256"] != BASELINE_SHA256:
        raise ValueError("bundle does not derive from the reviewed 447-file baseline")
    with tarfile.open(archive, "r:gz") as stream:
        files = {}
        for entry in stream.getmembers():
            path = PurePosixPath(entry.name)
            if (
                not entry.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in entry.name
                or ":" in entry.name
                or entry.name in files
            ):
                raise ValueError(f"unsafe or duplicate source member: {entry.name}")
            files[entry.name] = stream.extractfile(entry).read()
    manifest = json.loads(files.pop("bundle-manifest.json"))
    if set(files) != set(manifest["sources"]):
        raise ValueError("source members differ from the bundle manifest")
    for name, raw in files.items():
        if hashlib.sha256(raw).hexdigest() != manifest["sources"][name]:
            raise ValueError(f"source hash mismatch: {name}")
        if name.endswith(".py"):
            compile(raw, name, "exec")
        if not name.endswith((".py", ".json")):
            raise ValueError(f"source-only bundle contains unexpected member: {name}")
    if manifest["baseline_archive_sha256"] != BASELINE_SHA256:
        raise ValueError("manifest baseline differs")
    production = {
        name: value
        for name, value in manifest["sources"].items()
        if name.startswith("afd_plugin/")
    }
    if len(production) != 146 or production != lock["production_sources"]:
        raise ValueError("frozen 146-file production closure differs")
    return manifest, files


def scene_command(
    python: str,
    snapshot: Path,
    model: str,
    output: Path,
    devices: str,
    timeout: int,
    scene: str,
    *,
    run: bool,
) -> list[str]:
    command = [
        python,
        "-B",
        str(snapshot / PROBE),
        "--model",
        model,
        "--output-dir",
        str(output),
        "--devices",
        devices,
        "--timeout",
        str(timeout),
    ]
    if scene == SCENES[0]:
        command.extend(
            [
                "--target-logits-policy",
                "reject-second-verification-row",
                "--continue-after-rejection",
                "--observe-ready-frontier",
            ]
        )
    else:
        command.extend(
            [
                "--target-logits-policy",
                "constant7",
                "--target-graph",
                "--target-u2",
                "--target-cancel-replace",
            ]
        )
    if run:
        command.append("--run")
    return command


def source_environment(snapshot: Path) -> dict[str, str]:
    # Native workers retain the user's already configured A5 CANN/ATB/OPP paths.
    # Only source lookup and explicit V2 are selected here; no binary is supplied.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(snapshot), str(snapshot / "tests/npu")]
        + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    environment["PYTHONUNBUFFERED"] = "1"
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    return environment


def installed_sources(snapshot: Path) -> dict:
    expected = json.loads((snapshot / "native-source-identity.json").read_text("utf-8"))
    roots = {}
    for name in ("vllm", "vllm_ascend", "torch_npu"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError(f"installed native package is required: {name}")
        roots[name] = Path(spec.origin).resolve().parent
    sources = {}
    for name, digest in expected["source_expected"].items():
        package, relative = name.split("/", 1)
        path = roots[package] / relative
        actual = file_sha256(path)
        sources[name] = {
            "path": str(path),
            "sha256": actual,
            "expected_sha256": digest,
            "matches": actual == digest,
        }
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "roots": {key: str(value) for key, value in roots.items()},
        "package_versions": versions,
        "sources": sources,
        "source_count": len(sources),
        "sources_match": all(item["matches"] for item in sources.values()),
    }


def loaded_native_libraries() -> list[dict]:
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5])
            if (
                any(
                    path.name == name or path.name.startswith(name + ".")
                    for name in LIBRARY_NAMES
                )
                or (
                    any(
                        name in str(path)
                        for name in (
                            "vllm_ascend",
                            "custom_ops",
                            "cann_ops_transformer",
                        )
                    )
                    and ".so" in path.name
                )
            ) and path.is_file():
                paths.add(path.resolve())
    # A small actual boundary inventory; never recursively hash the vendor tree.
    for root in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep):
        if root:
            library = Path(root) / "op_api/lib/libcust_opapi.so"
            if library.is_file():
                paths.add(library.resolve())
    return [
        {"path": str(path), "size": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(paths)
    ]


def inspect_runtime(snapshot: Path, devices: tuple[int, ...], output: Path) -> int:
    """Explicit run-only native registration; no model, tensor or collective."""
    report = {
        "status": "failed",
        "model_executed": False,
        "collective_executed": False,
        "operator_kernel_executed": False,
    }
    runtime_imported = False
    try:
        report.update(installed_sources(snapshot))
        write_json(output, report)
        expected = json.loads(
            (snapshot / "native-source-identity.json").read_text("utf-8")
        )
        if (
            report["source_count"] != len(expected["source_expected"])
            or not report["sources_match"]
        ):
            raise ValueError("installed native Python differs from the locked Git pins")
        from hccl_native_attention_pd_probe import inspect_mooncake_dependencies

        report["mooncake"] = inspect_mooncake_dependencies()
        if not report["mooncake"]["single_distribution_owns_resolved_extension"]:
            raise ValueError("one installed Mooncake distribution must own engine.so")
        from dsv41_dspark_speculator_probe import NATIVE_PLUGINS

        os.environ["VLLM_PLUGINS"] = NATIVE_PLUGINS
        import torch
        import torch_npu
        from vllm.platforms import current_platform
        from vllm.plugins import load_general_plugins

        runtime_imported = True
        current_platform.pre_register_and_update()
        load_general_plugins()
        from vllm_ascend.utils import (
            adapt_patch,
            check_ascend_device_type,
            get_ascend_device_type,
        )

        adapt_patch()
        report["visible_device_count"] = torch.npu.device_count()
        if max(devices) >= report["visible_device_count"]:
            raise ValueError("--devices must refer to four process-visible devices")
        report["soc_version"] = torch_npu.npu.get_soc_version()
        report["built_device_type"] = get_ascend_device_type().name
        check_ascend_device_type()
        if report["built_device_type"] != "A5":
            raise ValueError(
                "this validation entry accepts the real A5 installation only"
            )
        torch.npu.set_device(devices[0])
        from dsv41_dspark_a5_runtime import native_a5_bootstrap

        report["native_a5_bootstrap"] = native_a5_bootstrap()
        report["custom_ops_enabled"] = False
        report["registered_custom_schemas"] = sorted(
            str(schema)
            for schema in torch._C._jit_get_all_schemas()
            if str(schema).startswith("_C_ascend::")
        )
        report["torch_npu_profiler_source"] = {
            "path": str(Path(report["roots"]["torch_npu"]) / "profiler/profiler.py"),
            "sha256": file_sha256(
                Path(report["roots"]["torch_npu"]) / "profiler/profiler.py"
            ),
            "scope": "A5 actual identity only; no A3 expected hash",
        }
        report["status"] = "passed"
    except BaseException:
        report["traceback"] = traceback.format_exc()
    finally:
        report["inherited_runtime_environment"] = {
            name: os.environ.get(name) for name in RUNTIME_ENVIRONMENT
        }
        if runtime_imported:
            try:
                report["native_libraries"] = loaded_native_libraries()
            except BaseException:
                report["library_collection_error"] = traceback.format_exc()
                report["status"] = "failed"
        write_json(output, report)
    return 0 if report["status"] == "passed" else 1


def live_owned(owned: dict[int, str]) -> dict[int, str]:
    live = {}
    for pid, ticks in owned.items():
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if fields[19] == ticks and fields[0] != "Z":
            live[pid] = ticks
    return live


def run_bounded(
    command: list[str],
    environment: dict[str, str],
    snapshot: Path,
    log: Path,
    timeout: int,
) -> dict:
    # Reuse the same frozen PID/start-tick helpers as the original four-role probe.
    sys.path.insert(0, str(snapshot / "tests/npu"))
    from hccl_eager_backend_probe import kill_owned, observe_owned

    owned, killed = {}, []
    started = time.monotonic()
    timed_out = False
    with log.open("wb") as stream:
        child = subprocess.Popen(
            command,
            env=environment,
            cwd=snapshot,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while child.poll() is None:
                observe_owned(child.pid, owned)
                if time.monotonic() - started >= timeout:
                    timed_out = True
                    break
                time.sleep(POLL_SECONDS)
        finally:
            killed = kill_owned(live_owned(owned))
            child.wait(timeout=REAP_TIMEOUT_SECONDS)
            write_json(
                log.with_suffix(".process.json"),
                {
                    "command": command,
                    "returncode": child.returncode,
                    "timed_out": timed_out,
                    "forced_killed_pids": killed,
                    "owned_pid_start_ticks": {
                        str(pid): ticks for pid, ticks in owned.items()
                    },
                    "surviving_owned_pids": sorted(live_owned(owned)),
                },
            )
    survivors = live_owned(owned)
    return {
        "command": command,
        "returncode": child.returncode,
        "timed_out": timed_out,
        "forced_killed_pids": killed,
        "owned_pid_start_ticks": {str(pid): ticks for pid, ticks in owned.items()},
        "surviving_owned_pids": sorted(survivors),
        "elapsed_seconds": time.monotonic() - started,
        "success": child.returncode == 0
        and not timed_out
        and not killed
        and not survivors,
    }


def npu_snapshot(output: Path) -> dict:
    result = subprocess.run(
        ["npu-smi", "info"], capture_output=True, timeout=REAP_TIMEOUT_SECONDS
    )
    raw = result.stdout + result.stderr
    output.write_bytes(raw)
    idle = sorted(
        set(map(int, re.findall(rb"No running processes found in NPU\s+(\d+)", raw)))
    )
    return {
        "returncode": result.returncode,
        "idle_devices": idle,
        "all_eight_idle": result.returncode == 0 and idle == list(range(8)),
    }


def verify_scene(scene: str, output: Path) -> dict:
    report = json.loads((output / "summary.json").read_text("utf-8"))
    assert report["success"] is True and report["timed_out"] is False
    assert report["forced_killed_pids"] == []
    assert set(report["returncodes"]) == set(ROLES)
    assert all(
        type(code) is int and code == 0 for code in report["returncodes"].values()
    )
    workers = report["workers"]
    assert set(workers) == set(ROLES)
    assert all(worker["success"] is True for worker in workers.values())
    consumer, producer = workers["kv_consumer"], workers["kv_producer"]
    assert producer["typed_done_observed"] is True
    assert consumer["native_decode_and_verification_accepted"] is True
    if scene == SCENES[0]:
        assert report["native_post_rejection_continuation_accepted"] is True
        assert producer["ready_frontier"]["status"] == "passed"
        assert [step["sampled_token_ids"] for step in consumer["decode_steps"]] == [
            [[7]],
            [[7, 8]],
            [[7, 8]],
        ]
    else:
        verified = report["pair_protocols"]["consumer"]
        assert verified["native_target_graph_u2_component_accepted"] is True
        assert verified["native_dspark_request_lifecycle_component_accepted"] is True
        assert consumer["native_dspark_request_lifecycle"]["accepted"] is True
        assert len(producer["sources"]) == 3
    return {
        "accepted": True,
        "role_exit_codes": report["returncodes"],
        "original_summary_sha256": file_sha256(output / "summary.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--devices", default="0,1,2,3", help="four process-visible ordinals: P,D,F1,F2"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="inner budget for each scene; outer adds 60 seconds",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--_inspect-runtime", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_snapshot", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        devices = tuple(map(int, args.devices.split(",")))
    except ValueError:
        parser.error("--devices requires four distinct nonnegative integers")
    if len(devices) != 4 or len(set(devices)) != 4 or min(devices) < 0:
        parser.error("--devices requires four distinct nonnegative integers")
    if not 0 < args.timeout < SOURCE_RETENTION_SECONDS:
        parser.error("--timeout must be positive and below source retention (720s)")
    if args._inspect_runtime is not None:
        if args._snapshot is None or os.name != "posix":
            parser.error("internal runtime inspection requires a Linux source snapshot")
        return inspect_runtime(args._snapshot, devices, args._inspect_runtime)
    if args.model is None or args.output_dir is None:
        parser.error("--model and --output-dir are required")
    if args.run and os.name != "posix":
        parser.error("--run requires the existing A5 Linux environment")
    if args.output_dir.exists():
        parser.error("--output-dir must be a fresh directory")
    executable = shutil.which(args.python)
    if executable is None:
        parser.error("--python must name an existing Python executable")
    args.python = str(Path(executable).resolve(strict=True))
    archive, lock_path = bundle_paths(args.source_bundle)
    manifest, files = read_bundle(archive, lock_path)
    lock = json.loads(lock_path.read_text("utf-8"))
    if file_sha256(Path(__file__)) != lock["launcher_sha256"]:
        parser.error("launcher differs from the frozen bundle lock")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True)
    snapshot = args.output_dir / "snapshot"
    for name, raw in files.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    environment = source_environment(snapshot)
    summary = {
        "status": "failed",
        "success": False,
        "run_requested": args.run,
        "npu_validation_status": "pending",
        "source_files": len(files),
        "production_files": 146,
        "native_python_pins": len(
            json.loads(files["native-source-identity.json"])["source_expected"]
        ),
        "baseline_archive_sha256": BASELINE_SHA256,
        "runtime_source_sha256": file_sha256(archive),
        "launcher_sha256": file_sha256(Path(__file__)),
        "devices": list(devices),
        "scenes": [],
        "checkpoint_target_loaded": False,
        "full_model_accepted": False,
        "engram_accepted": False,
        "performance_accepted": False,
        "physical_overlap_accepted": False,
    }
    try:
        if args.run:
            summary["npu_before"] = npu_snapshot(args.output_dir / "npu-before.log")
            if not summary["npu_before"]["all_eight_idle"]:
                raise RuntimeError(
                    "eight idle devices were not confirmed before the run"
                )
            inspect_command = [
                args.python,
                "-B",
                str(Path(__file__).resolve()),
                "--_inspect-runtime",
                str(args.output_dir / "native-before.json"),
                "--_snapshot",
                str(snapshot),
                "--devices",
                args.devices,
            ]
            check = run_bounded(
                inspect_command,
                environment,
                snapshot,
                args.output_dir / "native-before.log",
                INSPECTION_TIMEOUT_SECONDS,
            )
            summary["native_before_process"] = check
            if not check["success"]:
                raise RuntimeError("A5 installed/native registration preflight failed")
        for scene in SCENES:
            output = args.output_dir / scene
            command = scene_command(
                args.python,
                snapshot,
                args.model,
                output,
                args.devices,
                args.timeout,
                scene,
                run=args.run,
            )
            if args.run:
                item = run_bounded(
                    command,
                    environment,
                    snapshot,
                    args.output_dir / (scene + ".log"),
                    args.timeout + OUTER_GRACE_SECONDS,
                )
                summary["scenes"].append({"name": scene, **item})
                if not item["success"]:
                    raise RuntimeError(f"original scene failed: {scene}")
                item["functional_verification"] = verify_scene(scene, output)
                summary["scenes"][-1] = {"name": scene, **item}
                idle = npu_snapshot(args.output_dir / (scene + "-npu.log"))
                summary["scenes"][-1]["npu_after"] = idle
                if not idle["all_eight_idle"]:
                    raise RuntimeError(f"devices not idle after {scene}")
            else:
                result = subprocess.run(
                    command,
                    env=environment,
                    cwd=snapshot,
                    text=True,
                    capture_output=True,
                    timeout=METADATA_TIMEOUT_SECONDS,
                )
                (args.output_dir / (scene + ".log")).write_text(
                    result.stdout + result.stderr, "utf-8"
                )
                if result.returncode != 0:
                    raise RuntimeError(f"frozen probe metadata path failed: {scene}")
                metadata = json.loads(result.stdout)
                assert (
                    metadata["run"] is False and metadata["runtime_imported"] is False
                )
                graph_scene = scene == SCENES[1]
                assert metadata["target_graph_requested"] is graph_scene
                assert metadata["target_u2_requested"] is graph_scene
                assert metadata["target_cancel_replace_requested"] is graph_scene
                assert metadata["continue_after_rejection_requested"] is not graph_scene
                assert metadata["observe_ready_frontier_requested"] is not graph_scene
                assert metadata["target_logits_policy"] == (
                    "constant7" if graph_scene else "reject-second-verification-row"
                )
                for name in (
                    "graph_compute_overlap_requested",
                    "graph_all_axes_requested",
                    "eager_stream_overlap_requested",
                ):
                    assert metadata[name] is False
                assert metadata["graph_exchange_order_requested"] == "stage_serial"
                summary["scenes"].append(
                    {
                        "name": scene,
                        "command": command,
                        "metadata": metadata,
                        "accepted": False,
                    }
                )
        if args.run:
            inspect_command[inspect_command.index("--_inspect-runtime") + 1] = str(
                args.output_dir / "native-after.json"
            )
            check = run_bounded(
                inspect_command,
                environment,
                snapshot,
                args.output_dir / "native-after.log",
                INSPECTION_TIMEOUT_SECONDS,
            )
            summary["native_after_process"] = check
            if not check["success"]:
                raise RuntimeError("A5 final native identity collection failed")
            before = json.loads(
                (args.output_dir / "native-before.json").read_text("utf-8")
            )
            after = json.loads(
                (args.output_dir / "native-after.json").read_text("utf-8")
            )
            for key in (
                "package_versions",
                "sources",
                "mooncake",
                "native_libraries",
                "native_a5_bootstrap",
            ):
                if before[key] != after[key]:
                    raise RuntimeError(f"installed native identity changed: {key}")
            summary["native_installation_unchanged"] = True
            summary["npu_validation_status"] = "two_components_passed"
        for name, digest in manifest["sources"].items():
            if file_sha256(snapshot / name) != digest:
                raise RuntimeError(f"frozen source changed during the run: {name}")
        summary["status"] = "passed" if args.run else "dry_run_passed"
        summary["success"] = True
    except BaseException:
        summary["traceback"] = traceback.format_exc()
    finally:
        if args.run:
            try:
                summary["npu_after"] = npu_snapshot(args.output_dir / "npu-after.log")
                idle_after = summary["npu_after"]["all_eight_idle"]
            except BaseException:
                summary["npu_after_error"] = traceback.format_exc()
                idle_after = False
            if not idle_after:
                summary["success"] = False
                summary["status"] = "failed"
                summary["npu_validation_status"] = "failed"
        write_json(args.output_dir / "summary.json", summary)
    print(f"results: {args.output_dir}", flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
