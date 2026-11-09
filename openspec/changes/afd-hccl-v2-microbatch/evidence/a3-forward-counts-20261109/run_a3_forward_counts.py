"""Snapshot forward-count and FFN dispatch seams in the existing A3 venv."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shlex
import signal
import subprocess
import tarfile
from pathlib import Path

SSH_HOST = "devenvc_rissvg.4ff038c993bf40219c03182f786a9def.atomgit.0"
SSH_CONFIG = "C:/Users/zq/.atomgitdevenv/.ssh/config"
REMOTE_PARENT = "/mnt/workspace/tmp/afd-c0-20261104.hVgsIf"
VENV = "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0"
PROBE = "tests/npu/hccl_staged_probe.py"
UNIT_TEST = "tests/unit/model_executor/models/npu/test_deepseek_v41_stage_state.py"
CPU_SEAM_TESTS = (
    UNIT_TEST,
    "tests/unit/model_executor/models/npu/test_deepseek_v41_staged.py",
    "tests/unit/v1/worker/npu/test_hccl_v41_ubatch_runner.py",
)
CPU_REGRESSION_PATHS = (
    "tests/unit/model_executor",
    "tests/unit/v1/worker",
    "tests/unit/connectors",
    "tests/unit/distributed",
    "tests/unit/compat/npu/test_feature_validation.py",
)
CPU_BASELINE_TESTS = (
    "tests/unit/v1/worker/test_ffn_model_runner.py",
    "tests/unit/v1/worker/test_model_runner_v2.py",
    "tests/unit/model_executor/models/test_deepseek_v4_shared_experts.py",
    "tests/unit/model_executor/test_async_cam_w4a8.py",
)
CPU_FFN_TEST = "tests/unit/v1/worker/npu/test_hccl_v41_ffn_runner.py"
CPU_QUANT_TESTS = (
    "tests/unit/compat/patches/npu/test_mxfp8.py",
    "tests/unit/compat/patches/test_signatures.py",
    "tests/unit/package/test_package.py",
)
CPU_FORWARD_COUNTS_TESTS = (
    "tests/unit/connectors/npu/test_hccl_staged_transport.py",
    "tests/unit/distributed/test_hccl_plan.py",
    "tests/unit/v1/worker/npu/test_ffn_worker.py",
    "tests/unit/v1/worker/npu/test_ffn_model_runner.py",
    "tests/unit/v1/worker/test_ffn_model_runner.py",
    "tests/unit/v1/worker/test_npu_runtime.py",
    "tests/unit/v1/worker/test_async_ffn_runtime.py",
    "tests/unit/v1/worker/npu/test_hccl_v41_ffn_runner.py",
)
CPU_SCOPES = (
    "none",
    "state",
    "seams",
    "ffn",
    "quant",
    "baseline",
    "regression",
    "forward-counts",
    "binding",
    "attention-entry",
    "bootstrap",
    "identity",
    "integration",
)
SOURCE_IDENTITY = (
    "openspec/changes/afd-hccl-v2-microbatch/evidence/"
    "a3-staged-20261109/reject-digest-001/identity.json"
)
PROBE_TIMEOUT_SECONDS = 280
SSH_TIMEOUT_SECONDS = 420
REAP_TIMEOUT_SECONDS = 10
EXPECTED_IDLE_BOARDS = 8


def ssh(command, data=None):
    return subprocess.run(
        [
            "ssh",
            "-F",
            SSH_CONFIG,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            SSH_HOST,
            command,
        ],
        input=data,
        capture_output=True,
        timeout=SSH_TIMEOUT_SECONDS,
        check=True,
    ).stdout


def prepare_env(snapshot):
    script = """set -e
source /mnt/workspace/code/.ascend/cann-9.0.1/ascend-toolkit/set_env.sh >/dev/null
export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:\
/usr/local/Ascend/driver/lib64/driver:\
/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH}
export PATH=/mnt/workspace/code/.venvs/afd-v030-dsv41-c0/bin:${PATH}
source /mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1/nnal/atb/set_env.sh >/dev/null
env -0
"""
    prepared = subprocess.run(
        ["bash", "-c", script], capture_output=True, timeout=30, check=True
    )
    env = {
        key.decode(): value.decode()
        for field in prepared.stdout.split(b"\0")
        if field
        for key, value in [field.split(b"=", 1)]
    }
    env["PYTHONPATH"] = str(snapshot) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    env["VLLM_PLUGINS"] = (
        "ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling"
    )
    return env


def run_remote(remote, device, cpu_scope, cpu_only, topology):
    remote = Path(remote)
    snapshot, results = remote / "snapshot", remote / "results"
    snapshot.mkdir()
    results.mkdir()
    with tarfile.open(remote / "source.tar.gz") as archive:
        archive.extractall(snapshot, filter="data")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    actual = {
        name: hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
        for name in manifest
    }
    if actual != manifest:
        raise RuntimeError("uploaded source manifest differs from the snapshot")
    expected = json.loads((snapshot / "source-expected.json").read_text())
    site = Path(VENV) / "lib/python3.12/site-packages"
    checks = {
        name: hashlib.sha256((site / name).read_bytes()).hexdigest() == value
        for name, value in expected.items()
    }
    (results / "source-check.json").write_text(json.dumps(checks, indent=2) + "\n")
    if not all(checks.values()):
        raise RuntimeError(
            "installed upstream files differ from the locked source hashes"
        )
    env = prepare_env(snapshot)
    (results / "effective-env.json").write_text(
        json.dumps(
            {
                key: env.get(key)
                for key in (
                    "PYTHONPATH",
                    "LD_LIBRARY_PATH",
                    "ASCEND_HOME_PATH",
                    "ATB_HOME_PATH",
                    "VLLM_PLUGINS",
                )
            },
            indent=2,
        )
        + "\n"
    )
    versions = subprocess.run(
        [
            VENV + "/bin/python",
            "-B",
            "-c",
            "import json,platform,importlib.metadata as m;print(json.dumps("
            '{"python":platform.python_version(),"packages":{k:m.version(k) for k in '
            '["torch","torch-npu","vllm","vllm-ascend","fastapi"]}},indent=2))',
        ],
        env=env,
        capture_output=True,
        timeout=30,
        check=True,
    )
    (results / "versions.json").write_bytes(versions.stdout)
    unit_record = None
    if cpu_scope != "none":
        unit_command = [
            VENV + "/bin/python",
            "-B",
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "-rs",
            "--tb=short",
            "-p",
            "no:cacheprovider",
            "--junitxml=" + str(results / ("cpu-" + cpu_scope + ".xml")),
        ]
        if cpu_scope == "integration":
            unit_command += [
                "-m",
                "not npu and not gpu",
                *CPU_REGRESSION_PATHS,
                "tests/unit/config/test_validation.py",
                "tests/unit/compat/npu/test_runtime_config.py",
                "tests/unit/compat/patches/test_signatures.py",
            ]
        elif cpu_scope in ("baseline", "regression"):
            paths = (
                CPU_BASELINE_TESTS if cpu_scope == "baseline" else CPU_REGRESSION_PATHS
            )
            unit_command += ["-m", "not npu and not gpu", *paths]
        elif cpu_scope == "ffn":
            unit_command.append(CPU_FFN_TEST)
        elif cpu_scope == "identity":
            unit_command.extend(
                (
                    "tests/unit/v1/worker/npu/test_hccl_v41_execution.py",
                    "tests/unit/v1/worker/npu/test_ffn_worker.py",
                )
            )
        elif cpu_scope == "bootstrap":
            unit_command.extend(
                (
                    "tests/unit/connectors/npu/test_hccl_staged.py",
                    "tests/unit/config/test_validation.py",
                    "tests/unit/compat/npu/test_runtime_config.py",
                )
            )
        elif cpu_scope == "attention-entry":
            unit_command.extend(
                (
                    "tests/unit/v1/worker/npu/test_hccl_v41_execution.py",
                    "tests/unit/v1/worker/npu/test_hccl_v41_ubatch_runner.py",
                    "tests/unit/model_executor/models/npu/test_deepseek_v41.py",
                    "tests/unit/compat/patches/test_signatures.py",
                )
            )
        elif cpu_scope == "binding":
            unit_command.extend(
                (
                    "tests/unit/connectors/npu/test_hccl_config.py",
                    "tests/unit/v1/worker/npu/test_ffn_worker.py",
                )
            )
        elif cpu_scope == "forward-counts":
            unit_command.extend(CPU_FORWARD_COUNTS_TESTS)
        elif cpu_scope == "quant":
            unit_command.extend(CPU_QUANT_TESTS)
        else:
            unit_command += (
                list(CPU_SEAM_TESTS) if cpu_scope == "seams" else [UNIT_TEST]
            )
        unit = subprocess.run(
            unit_command, env=env, cwd=snapshot, capture_output=True, timeout=90
        )
        (results / ("cpu-" + cpu_scope + ".log")).write_bytes(unit.stdout + unit.stderr)
        unit_record = {
            "scope": cpu_scope,
            "command": unit_command,
            "returncode": unit.returncode,
        }
        if unit.returncode or cpu_only:
            (results / "summary.json").write_text(
                json.dumps(
                    {
                        "scope": "CPU seams with actual installed dependencies",
                        "success": unit.returncode == 0,
                        "cpu": unit_record,
                        "remote": str(remote),
                        "source_files_match": all(checks.values()),
                        "device_probe_started": False,
                    },
                    indent=2,
                )
                + "\n"
            )
            print((results / "summary.json").read_text(), flush=True)
            return unit.returncode
    before = subprocess.run(
        ["npu-smi", "info"], env=env, capture_output=True, timeout=30, check=True
    )
    (results / "npu-before.log").write_bytes(before.stdout + before.stderr)
    if (
        before.stdout.count(b"No running processes found in NPU")
        != EXPECTED_IDLE_BOARDS
    ):
        raise RuntimeError(
            "expected all eight boards idle before the single-chip probe"
        )
    command = [
        VENV + "/bin/python",
        "-B",
        str(snapshot / PROBE),
        "--devices",
        "0,2,4,6",
        "--topology",
        topology,
        "--local-counts-service",
        "--output-dir",
        str(results / "probe"),
    ]
    process = subprocess.Popen(
        command,
        env=env,
        cwd=snapshot,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        output, _ = process.communicate(timeout=PROBE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate(timeout=REAP_TIMEOUT_SECONDS)
    (results / "probe.log").write_bytes(output)
    after = subprocess.run(
        ["npu-smi", "info"], env=env, capture_output=True, timeout=30
    )
    (results / "npu-after.log").write_bytes(after.stdout + after.stderr)
    result_path = results / "probe/result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    idle_after = after.stdout.count(b"No running processes found in NPU")
    summary = {
        "scope": "real local-count HCCL and FFN service; controlled arithmetic",
        "remote": str(remote),
        "command": command,
        "device": device,
        "topology": topology,
        "source_files_match": all(checks.values()),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "killed_owned_process": timed_out,
        "cpu": unit_record,
        "device_probe_started": True,
        "idle_boards_before": EXPECTED_IDLE_BOARDS,
        "idle_boards_after": idle_after,
        "success": process.returncode == 0
        and not timed_out
        and result is not None
        and result["success"]
        and after.returncode == 0
        and idle_after == EXPECTED_IDLE_BOARDS,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["success"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--topology", choices=("1a1f", "2a2f-tp2"), default="1a1f")
    parser.add_argument("--remote-root")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cpu-scope", choices=CPU_SCOPES, default="none")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--baseline-ref")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be nonnegative")
    if args.cpu_only and args.cpu_scope == "none":
        parser.error("--cpu-only requires a CPU scope")
    if args.baseline_ref and not args.cpu_only:
        parser.error("--baseline-ref requires --cpu-only")
    if args.remote_root:
        return run_remote(
            args.remote_root, args.device, args.cpu_scope, args.cpu_only, args.topology
        )
    if not args.label or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
        for character in args.label
    ):
        parser.error("--label requires lowercase letters, digits and hyphens")
    evidence = Path(__file__).resolve().parent
    root = evidence.parents[4]
    destination = evidence / args.label
    destination.mkdir(exist_ok=False)
    source_identity = root / SOURCE_IDENTITY
    source_record = json.loads(source_identity.read_text(encoding="utf-8"))
    expected = source_record["source_expected"]
    baseline_commit = None
    if args.baseline_ref:
        baseline_commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", args.baseline_ref + "^{commit}"],
            cwd=root,
            text=True,
        ).strip()
        archived = subprocess.check_output(
            ["git", "archive", "--format=tar", baseline_commit], cwd=root
        )
        with tarfile.open(fileobj=io.BytesIO(archived)) as archive:
            contents = {
                member.name: archive.extractfile(member).read()
                for member in archive.getmembers()
                if member.isfile()
                and (
                    member.name.endswith(".py")
                    or member.name
                    in (
                        "pyproject.toml",
                        ".agents/dsv41/sources.lock.json",
                    )
                )
            }
        # The operational uploader is the only current-worktree overlay; all
        # production and test Python bytes come from the immutable Git archive.
        contents[Path(__file__).relative_to(root).as_posix()] = Path(
            __file__
        ).read_bytes()
    else:
        files = sorted((root / "afd_plugin").rglob("*.py")) + [
            root / PROBE,
            Path(__file__),
            root / ".agents/dsv41/sources.lock.json",
        ]
        if args.cpu_scope != "none":
            files += sorted((root / "tests/unit").rglob("*.py"))
            files += [
                path
                for path in (
                    root / "tests/__init__.py",
                    root / "tests/conftest.py",
                    root / "pyproject.toml",
                )
                if path.exists()
            ]
        contents = {
            path.relative_to(root).as_posix(): path.read_bytes() for path in files
        }
    lock = json.loads(contents[".agents/dsv41/sources.lock.json"])
    manifest = {
        name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        packed = contents | {
            "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
            "source-expected.json": (json.dumps(expected, indent=2) + "\n").encode(),
        }
        for name, data in packed.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    remote = (
        ssh(
            "python3 -B -c "
            + shlex.quote(
                "import tempfile;print(tempfile.mkdtemp("
                "prefix='c4-forward-counts-20261109-',dir=" + repr(REMOTE_PARENT) + "))"
            )
        )
        .decode()
        .strip()
    )
    ssh("cat > " + shlex.quote(remote + "/source.tar.gz"), stream.getvalue())
    ssh(
        "cat > " + shlex.quote(remote + "/run_a3_forward_counts.py"),
        Path(__file__).read_bytes(),
    )
    identity = {
        "scope": "immutable Git archive"
        if baseline_commit
        else "working-tree bytes snapshot",
        "baseline_commit": baseline_commit,
        "head_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "snapshot_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "remote": remote,
        "manifest": manifest,
        "source_expected": expected,
        "source_identity_origin": SOURCE_IDENTITY,
        "source_identity_sha256": hashlib.sha256(
            source_identity.read_bytes()
        ).hexdigest(),
        "source_lock_repositories": {
            name: lock["repositories"][name]["commit"]
            for name in ("vllm", "vllm_ascend")
        },
        "archive_sha256": hashlib.sha256(stream.getvalue()).hexdigest(),
        "device": args.device,
        "topology": args.topology,
        "cpu_scope": args.cpu_scope,
        "device_probe_requested": not args.cpu_only,
    }
    (destination / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps({"remote": remote, "files": len(manifest), "prepared": True}))
        return 0
    command = [
        "ssh",
        "-F",
        SSH_CONFIG,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        SSH_HOST,
        "python3 -B "
        + shlex.quote(remote + "/run_a3_forward_counts.py")
        + " --remote-root "
        + shlex.quote(remote)
        + " --device "
        + str(args.device),
    ]
    command[-1] += " --cpu-scope " + shlex.quote(args.cpu_scope)
    command[-1] += " --topology " + shlex.quote(args.topology)
    if args.cpu_only:
        command[-1] += " --cpu-only"
    run = subprocess.run(command, capture_output=True, timeout=SSH_TIMEOUT_SECONDS)
    (destination / "remote-run.log").write_bytes(run.stdout + run.stderr)
    raw = ssh("tar -czf - -C " + shlex.quote(remote) + " results")
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(destination, filter="data")
    (destination / "delivery.json").write_text(
        json.dumps(
            {
                "command": command,
                "returncode": run.returncode,
            },
            indent=2,
        )
        + "\n"
    )
    print(run.stdout.decode(errors="replace"), flush=True)
    print(run.stderr.decode(errors="replace"), flush=True)
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
