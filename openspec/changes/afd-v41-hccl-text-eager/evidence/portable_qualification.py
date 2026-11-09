# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Collect bounded, portable DSV4.1 operator/component evidence on Linux NPUs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from contextlib import suppress
from pathlib import Path

MAX_TIMEOUT_SECONDS = 600
DEFAULT_TIMEOUT_SECONDS = 240
REAP_TIMEOUT_SECONDS = 10
METADATA_TIMEOUT_SECONDS = 30
PACKAGE_NAMES = ("torch", "torch-npu", "vllm", "vllm-ascend", "fastapi")
ALLOWED_CHECKS = frozenset(
    {"preflight", "norm", "norm-fallback", "norm-native", "mxfp", "construction", "tp"}
)
# Generated from the pinned, read-only Git objects named in sources.lock.json.
# This tuple is immutable; no local checkout path is needed on the target host.
PINNED_SOURCES = (
    (
        "vllm/config/model.py",
        "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
        "f4168b67cc93ce72ff2a6d4c66974223e32fd0f1fab4c05e44bd68a275e18d69",
    ),
    (
        "vllm/config/vllm.py",
        "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
        "956b812e5a719bcbfa3a3958801361b38b9c928c96b8c073311bbb96376dfb7a",
    ),
    (
        "vllm/model_executor/model_loader/weight_utils.py",
        "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
        "7443e55184777dcc99f621b0d97d9149d52c2ea8b4304c6621eb4902e1c8ba5b",
    ),
    (
        "vllm/model_executor/layers/layernorm.py",
        "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
        "4126258bf85aa3af54c78cfbf2a0f32000491e42be37661f2d1ac9f0beea00f3",
    ),
    (
        "vllm/models/deepseek_v4/quant_config.py",
        "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
        "6fdd20b617640741a19643f893b9b8a31d390b0ccb04d322ece29f57ff8290c9",
    ),
    (
        "vllm/distributed/parallel_state.py",
        "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
        "d9b10ad56833ebc08a753e1a050ef0989c30d4134021b7b7b00150b7bd01adc6",
    ),
    (
        "vllm_ascend/models/deepseek_v41/model.py",
        "7e37e29a7803d6897a7113088e2b771f83205bc0",
        "989aec6db1e344dca76159dcbef0d5876874facedaa36b47e993146d5b8a2b9f",
    ),
    (
        "vllm_ascend/models/deepseek_v41/vl_model.py",
        "7e37e29a7803d6897a7113088e2b771f83205bc0",
        "bb98a2e2a0e24c18aafe17229eb593937aaed2e37c1fd7cc2003c45ce872d282",
    ),
    (
        "vllm_ascend/models/deepseek_v41/engram/model_state.py",
        "7e37e29a7803d6897a7113088e2b771f83205bc0",
        "677cb13bb86ad80a154e0ff9a43620aa3344e7e93170a05e27aba0cc1f10aa8b",
    ),
    (
        "vllm_ascend/worker/v2/model_runner.py",
        "7e37e29a7803d6897a7113088e2b771f83205bc0",
        "a3471a9f181b45742b8f985c86137cdaafc2871d2ef6f96aa6a45236a242969b",
    ),
    (
        "vllm_ascend/device/device_op.py",
        "7e37e29a7803d6897a7113088e2b771f83205bc0",
        "a603e21fff2dabc06877fe6a1e52fe3637dec316b1a4cae7228a54037ec9766a",
    ),
    (
        "vllm_ascend/device/hardware_profile.py",
        "7e37e29a7803d6897a7113088e2b771f83205bc0",
        "efc6d0f3856be400c62b20c6a39229ccc1a2771ec4351a03328ec535824691f5",
    ),
    (
        "vllm_ascend/ops/layernorm.py",
        "7e37e29a7803d6897a7113088e2b771f83205bc0",
        "d2fda42ad99935725c98b9505a8d2e94696d9cdb2906e5cb2f1e37bf21fb27bc",
    ),
    (
        "vllm_ascend/ops/fused_moe/router/fused_topk_router.py",
        "7e37e29a7803d6897a7113088e2b771f83205bc0",
        "06c47fd91e8c36f6f5465cea0b32a184280f842833c2bff27a18fc6d17fcb838",
    ),
    (
        "vllm_ascend/quantization/methods/base.py",
        "7e37e29a7803d6897a7113088e2b771f83205bc0",
        "51a7aafcc49b64a53902eeb3619023e3d79f9c8f4aaf846205dc83dfd35b0d24",
    ),
    (
        "vllm_ascend/quantization/configs/fp8_config.py",
        "7e37e29a7803d6897a7113088e2b771f83205bc0",
        "59647e01fe838a64d1bfd6e97dceb9defd104681c419b493cbd0f81417eba5f3",
    ),
)


def save(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def qualified_type(value):
    return type(value).__module__ + "." + type(value).__qualname__


def preflight(checkpoint, output):
    report = {
        "scope": "read-only configuration; no model/weights/device allocation",
        "python": sys.version,
        "platform": platform.platform(),
        "success": False,
    }
    try:
        report["packages"] = {}
        for name in PACKAGE_NAMES:
            try:
                report["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                report["packages"][name] = None
        source_checks = {}
        for name, revision, expected in PINNED_SOURCES:
            package, relative = name.split("/", 1)
            spec = importlib.util.find_spec(package)
            path = Path(spec.origin).parent / relative if spec and spec.origin else None
            actual = (
                hashlib.sha256(path.read_bytes()).hexdigest()
                if path and path.is_file()
                else None
            )
            source_checks[name] = {
                "commit": revision,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "path": str(path),
                "matches": actual == expected,
            }
        report["source_checks"] = source_checks
        config_bytes = (checkpoint / "config.json").read_bytes()
        report["checkpoint"] = {
            "path": str(checkpoint),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "config": json.loads(config_bytes),
        }
        import vllm_ascend.ops  # noqa: F401
        from vllm.config import ModelConfig, VllmConfig
        from vllm.config.device import DeviceConfig
        from vllm.config.load import LoadConfig
        from vllm.platforms import current_platform
        from vllm.plugins import load_general_plugins

        current_platform.pre_register_and_update()
        load_general_plugins()
        cells = []
        for requested_quantization in (None, "deepseek_v4_fp8"):
            cell = {"requested_quantization": requested_quantization}
            try:
                config = VllmConfig(
                    model_config=ModelConfig(
                        model=str(checkpoint),
                        trust_remote_code=True,
                        dtype="bfloat16",
                        enforce_eager=True,
                        max_model_len=128,
                        skip_tokenizer_init=True,
                        language_model_only=True,
                        quantization=requested_quantization,
                    ),
                    device_config=DeviceConfig(device="cpu"),
                    load_config=LoadConfig(load_format="dummy"),
                )
                cell.update(
                    success=True,
                    resolved_quantization=config.model_config.quantization,
                    quant_config_class=qualified_type(config.quant_config),
                    architectures=config.model_config.hf_config.architectures,
                    text_model_type=config.model_config.hf_text_config.model_type,
                )
            except Exception as exc:
                cell.update(
                    success=False, error_type=qualified_type(exc), error=str(exc)
                )
            cells.append(cell)
        report["quantization_cells"] = cells
        report["success"] = (
            bool(source_checks)
            and all(item["matches"] for item in source_checks.values())
            and all(item["success"] for item in cells)
        )
    except BaseException as exc:
        report["error"] = {"type": qualified_type(exc), "message": str(exc)}
        traceback.print_exc()
    save(output / "preflight.json", report)
    return 0 if report["success"] else 1


def prepare_environment(args):
    env = os.environ.copy()
    scripts = [path.resolve() for path in (args.cann_env, args.atb_env) if path]
    if scripts:
        for path in scripts:
            if not path.is_file():
                raise FileNotFoundError(path)
        # Script paths are positional shell arguments, never interpolated code.
        command = [
            "bash",
            "-c",
            'set -e; for script in "$@"; do source "$script" >/dev/null; done; env -0',
            "qualification-env",
        ]
        command += list(map(str, scripts))
        completed = subprocess.run(
            command,
            env=env,
            capture_output=True,
            check=True,
            timeout=METADATA_TIMEOUT_SECONDS,
        )
        env = {
            key.decode(): value.decode()
            for field in completed.stdout.split(b"\0")
            if field
            for key, value in [field.split(b"=", 1)]
        }
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONIOENCODING="utf-8",
        PYTHONPATH=os.pathsep.join([str(args.repo_root), env.get("PYTHONPATH", "")]),
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_PLUGINS="ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling",
    )
    return env


def run_bounded(name, command, args, env):
    started = time.monotonic()
    timed_out = False
    owned_kills = []
    process = None
    with (args.output_dir / (name + ".log")).open("wb") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=args.repo_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            owned_kills = stop_owned_processes(process)
        except OSError as exc:
            log.write(str(exc).encode("utf-8"))
        except BaseException:
            if process is not None and process.poll() is None:
                stop_owned_processes(process)
            raise
    return {
        "command": command,
        "pid": process.pid if process else None,
        "exit_code": process.returncode if process else None,
        "timed_out": timed_out,
        "own_processes_killed": owned_kills,
        "elapsed_seconds": time.monotonic() - started,
        "success": process is not None and process.returncode == 0 and not timed_out,
    }


def stop_owned_processes(process):
    """Kill only verified descendants, including a nested probe's new sessions."""
    # pidfd pins a process identity while /proc supplies its parent identity;
    # unlike a later kill(pid), it cannot signal a PID that has been reused.
    owned = {process.pid}
    handles = []
    try:
        for _ in range(3):
            found = False
            for path in Path("/proc").iterdir():
                if not path.name.isdecimal() or int(path.name) in owned:
                    continue
                handle = None
                try:
                    handle = os.pidfd_open(int(path.name))
                    stat = (path / "stat").read_text().rsplit(")", 1)[1].split()
                    if int(stat[1]) in owned:
                        owned.add(int(path.name))
                        handles.append(handle)
                        handle = None
                        found = True
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass
                finally:
                    if handle is not None:
                        os.close(handle)
            if not found:
                break
        for handle in reversed(handles):
            with suppress(ProcessLookupError):
                signal.pidfd_send_signal(handle, signal.SIGKILL)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=REAP_TIMEOUT_SECONDS)
        return sorted(owned)
    finally:
        for handle in handles:
            os.close(handle)


def npu_snapshot(name, args, env):
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            env=env,
            capture_output=True,
            timeout=METADATA_TIMEOUT_SECONDS,
        )
        (args.output_dir / (name + ".log")).write_bytes(result.stdout + result.stderr)
        return {"exit_code": result.returncode}
    except (OSError, subprocess.TimeoutExpired) as exc:
        (args.output_dir / (name + ".log")).write_text(str(exc), encoding="utf-8")
        return {"error": str(exc)}


def commands(args):
    evidence = Path(__file__).resolve().parent
    python = str(args.python)
    cases = []
    if "preflight" in args.checks:
        cases.append(
            (
                "preflight",
                [
                    python,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--preflight-child",
                    "--checkpoint",
                    str(args.checkpoint),
                    "--output-dir",
                    str(args.output_dir),
                ],
            )
        )
    for check in ("norm", "norm-fallback", "norm-native"):
        if check in args.checks:
            command = [
                python,
                "-B",
                str(evidence / "probe_norm_precision.py"),
                "--device",
                str(args.device),
                "--checkpoint",
                str(args.checkpoint),
                "--output-dir",
                str(args.output_dir / check),
            ]
            if check == "norm-fallback":
                command.append("--allow-fallback")
            if check == "norm-native":
                command.append("--native-path")
            cases.append((check, command))
    if "mxfp" in args.checks:
        cases.append(
            (
                "mxfp",
                [
                    python,
                    "-B",
                    str(evidence / "probe_mxfp_quant.py"),
                    "--case",
                    "base",
                    "--device",
                    str(args.device),
                    "--output-dir",
                    str(args.output_dir / "mxfp"),
                ],
            )
        )
    if "construction" in args.checks:
        for cell in ("decoder", "wrapper"):
            cases.append(
                (
                    "construction-" + cell,
                    [
                        python,
                        "-B",
                        str(evidence / "probe_attention_construction.py"),
                        "--device",
                        str(args.device),
                        "--checkpoint",
                        str(args.checkpoint),
                        "--cell",
                        cell,
                        "--output-dir",
                        str(args.output_dir / ("construction-" + cell)),
                    ],
                )
            )
    if "tp" in args.checks:
        launcher = args.output_dir / "tp-configured-launcher.py"
        # Existing TP probe's children rerun its __file__. Point that field at
        # this launcher so every child receives the same checkpoint parameter.
        launcher.write_text(
            "import sys\n"
            "sys.path.insert(0, "
            + repr(str(evidence))
            + ")\n"
            + "import probe_hccl_tp as probe\n"
            + "probe.CHECKPOINT = "
            + repr(str(args.checkpoint))
            + "\n"
            + "probe.__file__ = __file__\n"
            + "args = probe.parse_args()\n"
            + "raise SystemExit(probe.run_rank(args) if args.role "
            "else probe.run_parent(args))\n",
            encoding="utf-8",
        )
        cases.append(
            (
                "tp",
                [
                    python,
                    "-B",
                    str(launcher),
                    "--devices",
                    args.tp_devices,
                    "--output-dir",
                    str(args.output_dir / "tp"),
                ],
            )
        )
    return cases


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--python", type=Path)
    target.add_argument("--venv", type=Path)
    parser.add_argument("--cann-env", type=Path)
    parser.add_argument("--atb-env", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[4]
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--tp-devices", default="0,2,4,6", help="A0,A1,F0,F1 logical indices"
    )
    parser.add_argument("--checks", default="preflight,norm,mxfp")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="per check outer timeout, in seconds",
    )
    parser.add_argument(
        "--preflight-child", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    args.checks = frozenset(args.checks.split(","))
    if not args.checks or args.checks - ALLOWED_CHECKS:
        parser.error("--checks must use: " + ",".join(sorted(ALLOWED_CHECKS)))
    if (
        args.device < 0
        or not DEFAULT_TIMEOUT_SECONDS <= args.timeout <= MAX_TIMEOUT_SECONDS
    ):
        parser.error("device must be nonnegative; timeout must be 240..600 seconds")
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    args.repo_root = args.repo_root.resolve()
    args.python = (
        args.venv.resolve() / "bin/python"
        if args.venv
        else (args.python or Path(sys.executable)).resolve()
    )
    return args


def main():
    args = parse_args()
    if args.preflight_child:
        return preflight(args.checkpoint, args.output_dir)
    if os.name != "posix":
        raise SystemExit(
            "Run this collector on the target Linux host; no SSH is required."
        )
    archive_path = args.output_dir.with_name(args.output_dir.name + ".tar.gz")
    if archive_path.exists():
        raise FileExistsError(archive_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "success": False,
        "full_model_qualification": False,
        "production_guard_modified": False,
        "checkpoint_weights_loaded": False,
        "checks": {},
        "checkpoint": str(args.checkpoint),
        "python": str(args.python),
        "checks_requested": sorted(args.checks),
    }
    try:
        handle = os.pidfd_open(os.getpid())
        try:
            signal.pidfd_send_signal(handle, 0)
        finally:
            os.close(handle)
        env = prepare_environment(args)
        report["environment_scripts"] = {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.cann_env, args.atb_env)
            if path
        }
        version_roots = {
            path.resolve().parent for path in (args.cann_env, args.atb_env) if path
        }
        if env.get("ASCEND_HOME_PATH"):
            version_roots.add(Path(env["ASCEND_HOME_PATH"]).resolve())
        version_files = {
            folder / filename
            for folder in version_roots
            for filename in (
                "version.info",
                "version.cfg",
                "ascend_toolkit_install.info",
            )
        }
        report["runtime_version_files"] = {
            str(path): path.read_text(errors="replace")[:4096]
            for path in sorted(version_files)
            if path.is_file()
        }
        report["selected_environment"] = {
            name: env.get(name)
            for name in (
                "ASCEND_HOME_PATH",
                "ASCEND_OPP_PATH",
                "ATB_HOME_PATH",
                "ASCEND_RT_VISIBLE_DEVICES",
                "VLLM_PLUGINS",
                "VLLM_USE_V2_MODEL_RUNNER",
            )
        }
        evidence = Path(__file__).resolve().parent
        probe_names = (
            "portable_qualification.py",
            "probe_norm_precision.py",
            "probe_mxfp_quant.py",
            "probe_attention_construction.py",
            "probe_hccl_tp.py",
        )
        save(
            args.output_dir / "probe-sources.json",
            {
                name: hashlib.sha256((evidence / name).read_bytes()).hexdigest()
                for name in probe_names
            },
        )
        report["npu_before"] = npu_snapshot("npu-before", args, env)
        for name, command in commands(args):
            report["checks"][name] = run_bounded(name, command, args, env)
            save(args.output_dir / "summary.json", report)
        report["npu_after"] = npu_snapshot("npu-after", args, env)
        report["success"] = all(item["success"] for item in report["checks"].values())
    except BaseException as exc:
        report["error"] = {"type": qualified_type(exc), "message": str(exc)}
        traceback.print_exc()
    finally:
        save(args.output_dir / "summary.json", report)
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(args.output_dir, arcname=args.output_dir.name)
        print("Evidence directory: " + str(args.output_dir), flush=True)
        print("Evidence archive: " + str(archive_path), flush=True)
        print("Component checks passed: " + str(report["success"]), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
