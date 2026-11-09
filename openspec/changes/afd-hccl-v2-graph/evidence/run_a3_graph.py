"""Snapshot one C5 increment and run bounded CPU/static-NPU graph probes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shlex
import subprocess
import tarfile
from pathlib import Path

SSH_HOST = "devenvc_rissvg.4ff038c993bf40219c03182f786a9def.atomgit.0"
SSH_CONFIG = "C:/Users/zq/.atomgitdevenv/.ssh/config"
REMOTE_PARENT = "/mnt/workspace/tmp/afd-c0-20261104.hVgsIf"
VENV = "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0"
PROBE = "tests/npu/hccl_graph_probe.py"
UNIT_TEST = "tests/unit/connectors/npu/test_hccl_graph.py"
EVIDENCE = "openspec/changes/afd-hccl-v2-graph/evidence"
SOURCE_IDENTITY = (
    "openspec/changes/afd-hccl-v2-microbatch/evidence/"
    "a3-staged-20261109/reject-digest-001/identity.json"
)
SSH_TIMEOUT_SECONDS = 480
CPU_TIMEOUT_SECONDS = 120
NPU_TIMEOUT_SECONDS = 280
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
source /mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1/nnal/atb/set_env.sh >/dev/null
export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:\
/usr/local/Ascend/driver/lib64/driver:\
/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH}
env -0
"""
    prepared = subprocess.run(
        ["bash", "-c", script], capture_output=True, check=True, timeout=30
    )
    env = {
        key.decode(): value.decode()
        for field in prepared.stdout.split(b"\0")
        if field
        for key, value in [field.split(b"=", 1)]
    }
    env["PATH"] = VENV + "/bin:" + env["PATH"]
    env["PYTHONPATH"] = str(snapshot) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["VLLM_PLUGINS"] = (
        "ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling"
    )
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    return env


def run_remote(remote, devices, cpu_only):
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
        raise RuntimeError("uploaded source snapshot failed integrity verification")
    expected = json.loads((snapshot / "source-expected.json").read_text())
    site = Path(VENV) / "lib/python3.12/site-packages"
    installed = {
        name: hashlib.sha256((site / name).read_bytes()).hexdigest()
        for name in expected
    }
    (results / "upstream-identity.json").write_text(
        json.dumps(installed, indent=2) + "\n"
    )
    if installed != expected:
        raise RuntimeError(
            "installed source differs from locked/collected source identity"
        )
    env = prepare_env(snapshot)
    (results / "effective-env.json").write_text(
        json.dumps(
            {
                key: env.get(key)
                for key in (
                    "PYTHONPATH",
                    "PATH",
                    "LD_LIBRARY_PATH",
                    "ASCEND_HOME_PATH",
                    "VLLM_PLUGINS",
                )
            },
            indent=2,
        )
        + "\n"
    )
    command = [
        VENV + "/bin/python",
        "-B",
        "-m",
        "pytest",
        UNIT_TEST,
        "-q",
        "-rs",
        "--junitxml=" + str(results / "cpu.xml"),
    ]
    cpu = subprocess.run(
        command, cwd=snapshot, env=env, capture_output=True, timeout=CPU_TIMEOUT_SECONDS
    )
    (results / "cpu.log").write_bytes(cpu.stdout + cpu.stderr)
    summary = {
        "cpu_command": command,
        "cpu_exit_code": cpu.returncode,
        "device_probe_requested": not cpu_only,
    }
    if cpu.returncode or cpu_only:
        (results / "run-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        return cpu.returncode
    before = subprocess.run(
        ["npu-smi", "info"], env=env, capture_output=True, timeout=30, check=True
    )
    (results / "npu-before.log").write_bytes(before.stdout + before.stderr)
    if (
        before.stdout.count(b"No running processes found in NPU")
        != EXPECTED_IDLE_BOARDS
    ):
        raise RuntimeError("expected all eight boards idle before the paired probe")
    command = [
        VENV + "/bin/python",
        "-B",
        PROBE,
        "--devices",
        devices,
        "--output-dir",
        str(results / "native"),
    ]
    try:
        device = subprocess.run(
            command,
            cwd=snapshot,
            env=env,
            capture_output=True,
            timeout=NPU_TIMEOUT_SECONDS,
        )
        (results / "probe.log").write_bytes(device.stdout + device.stderr)
        summary.update(device_command=command, device_exit_code=device.returncode)
    except subprocess.TimeoutExpired as exc:
        (results / "probe.log").write_bytes((exc.stdout or b"") + (exc.stderr or b""))
        summary.update(device_command=command, device_exit_code=124)
    finally:
        after = subprocess.run(
            ["npu-smi", "info"], env=env, capture_output=True, timeout=30
        )
        (results / "npu-after.log").write_bytes(after.stdout + after.stderr)
        summary["after_idle_boards"] = after.stdout.count(
            b"No running processes found in NPU"
        )
        (results / "run-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary["device_exit_code"] or (
        0 if summary["after_idle_boards"] == EXPECTED_IDLE_BOARDS else 1
    )


def run_local(args):
    root = Path(__file__).resolve().parents[4]
    destination = root / EVIDENCE / "a3-static-graph-20261109" / args.label
    destination.mkdir(parents=True, exist_ok=False)
    paths = list(root.glob("afd_plugin/**/*.py")) + list(
        root.glob("tests/**/__init__.py")
    )
    paths.extend(
        root / name
        for name in (
            "pyproject.toml",
            "tests/conftest.py",
            "tests/unit/fakes/hccl_graph.py",
            UNIT_TEST,
            PROBE,
        )
    )
    contents = {path.relative_to(root).as_posix(): path.read_bytes() for path in paths}
    manifest = {
        name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
    }
    expected = json.loads((root / SOURCE_IDENTITY).read_text())["source_expected"]
    collected = json.loads(
        (root / EVIDENCE / "torch-npu-lowering-source-20261109.json").read_text()
    )
    expected.update({entry["path"]: entry["sha256"] for entry in collected["sources"]})
    contents["manifest.json"] = json.dumps(manifest, indent=2).encode()
    contents["source-expected.json"] = json.dumps(expected, indent=2).encode()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, data in sorted(contents.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    remote = (
        ssh(
            "python3 -B -c "
            + shlex.quote(
                "import tempfile;print(tempfile.mkdtemp("
                "prefix='c5-graph-20261109-',dir="
                + repr(REMOTE_PARENT)
                + "))"
            )
        )
        .decode()
        .strip()
    )
    ssh("cat > " + shlex.quote(remote + "/source.tar.gz"), stream.getvalue())
    ssh(
        "cat > " + shlex.quote(remote + "/run_a3_graph.py"), Path(__file__).read_bytes()
    )
    identity = {
        "scope": "immutable working-tree bytes snapshot",
        "remote": remote,
        "manifest": manifest,
        "source_expected": expected,
        "archive_sha256": hashlib.sha256(stream.getvalue()).hexdigest(),
        "devices": args.devices,
        "device_probe_requested": not args.cpu_only,
    }
    (destination / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    command = (
        "python3 -B "
        + shlex.quote(remote + "/run_a3_graph.py")
        + " --remote-root "
        + shlex.quote(remote)
    )
    command += " --devices " + shlex.quote(args.devices)
    if args.cpu_only:
        command += " --cpu-only"
    failure = None
    try:
        ssh(command)
    except subprocess.CalledProcessError as exc:
        failure = exc
        (destination / "remote-run.log").write_bytes(exc.stdout + exc.stderr)
    archived = ssh("tar -czf - -C " + shlex.quote(remote) + " results")
    with tarfile.open(fileobj=io.BytesIO(archived)) as archive:
        archive.extractall(destination, filter="data")
    print(
        json.dumps(
            {
                "evidence": str(destination),
                "remote": remote,
                "returncode": failure.returncode if failure else 0,
            }
        )
    )
    return failure.returncode if failure else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label")
    parser.add_argument("--devices", default="0,2")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--remote-root")
    options = parser.parse_args()
    if options.remote_root:
        raise SystemExit(
            run_remote(options.remote_root, options.devices, options.cpu_only)
        )
    if not options.label or Path(options.label).name != options.label:
        parser.error("provide a new single-component --label")
    raise SystemExit(run_local(options))
