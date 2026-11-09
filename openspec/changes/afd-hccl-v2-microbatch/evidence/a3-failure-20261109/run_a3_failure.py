"""Immutable A3 uploader for quiescent-boundary negative HCCL component probes."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shlex
import signal
import subprocess
import tarfile
from pathlib import Path

PROBE = "tests/npu/hccl_staged_failure_probe.py"
ORACLE = "tests/npu/hccl_staged_probe.py"
BASE_HARNESS = (
    "openspec/changes/afd-hccl-v2-microbatch/evidence/"
    "a3-forward-counts-20261109/run_a3_forward_counts.py"
)
SOURCE_IDENTITY = (
    "openspec/changes/afd-hccl-v2-microbatch/evidence/"
    "a3-staged-20261109/reject-digest-001/identity.json"
)
REMOTE_PARENT = "/mnt/workspace/tmp/afd-c0-20261104.hVgsIf"
VENV = "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0"
PROBE_TIMEOUT_SECONDS = 280
EXPECTED_IDLE_BOARDS = 8


def load_base(root):
    spec = importlib.util.spec_from_file_location(
        "existing_a3_harness", root / BASE_HARNESS
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remote_run(args):
    remote = Path(args.remote_root)
    snapshot, results = remote / "snapshot", remote / "results"
    snapshot.mkdir()
    results.mkdir()
    with tarfile.open(remote / "source.tar.gz") as archive:
        archive.extractall(snapshot, filter="data")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if any(
        hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest
        for name, digest in manifest.items()
    ):
        raise RuntimeError("uploaded snapshot differs from source manifest")
    base = load_base(snapshot)
    expected = json.loads((snapshot / "source-expected.json").read_text())
    site = Path(VENV) / "lib/python3.12/site-packages"
    checks = {
        name: hashlib.sha256((site / name).read_bytes()).hexdigest() == digest
        for name, digest in expected.items()
    }
    (results / "source-check.json").write_text(json.dumps(checks, indent=2) + "\n")
    if not all(checks.values()):
        raise RuntimeError("installed upstream source hashes differ")
    env = base.prepare_env(snapshot)
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
            "import platform,importlib.metadata as m,json;print(json.dumps("
            "{'python':platform.python_version(),'packages':{k:m.version(k) for k in "
            "['torch','torch-npu','vllm','vllm-ascend','fastapi']}},indent=2))",
        ],
        env=env,
        capture_output=True,
        timeout=30,
        check=True,
    )
    (results / "versions.json").write_bytes(versions.stdout)
    before = subprocess.run(
        ["npu-smi", "info"], env=env, capture_output=True, timeout=30, check=True
    )
    (results / "npu-before.log").write_bytes(before.stdout + before.stderr)
    if (
        before.stdout.count(b"No running processes found in NPU")
        != EXPECTED_IDLE_BOARDS
    ):
        raise RuntimeError("all eight boards must be idle before this probe")
    command = [
        VENV + "/bin/python",
        "-B",
        str(snapshot / PROBE),
        "--case",
        args.case,
        "--devices",
        "0,2",
        "--model",
        args.model,
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
        # The inner supervisor owns separate worker process groups; killing the
        # supervisor alone would leak them. Read only its explicitly owned PIDs.
        ownership = results / "probe/ownership.json"
        if ownership.exists():
            for pid in json.loads(ownership.read_text())["worker_pids"].values():
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGKILL)
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate(timeout=10)
    (results / "probe.log").write_bytes(output)
    after = subprocess.run(
        ["npu-smi", "info"], env=env, capture_output=True, timeout=30
    )
    (results / "npu-after.log").write_bytes(after.stdout + after.stderr)
    result_path = results / "probe/result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    idle_after = after.stdout.count(b"No running processes found in NPU")
    summary = {
        "scope": "quiescent component failures; excludes HCCL abort/device recovery",
        "remote": str(remote),
        "case": args.case,
        "command": command,
        "source_files_match": all(checks.values()),
        "returncode": process.returncode,
        "timed_out": timed_out,
        "killed_owned_processes": timed_out,
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
    parser.add_argument(
        "--case", choices=("compute-cancel", "peer-exit"), required=True
    )
    parser.add_argument("--label")
    parser.add_argument("--model", default="/mnt/workspace/models/DeepSeek-V4.1-Flash")
    parser.add_argument("--remote-root")
    args = parser.parse_args()
    if args.remote_root:
        return remote_run(args)
    if not args.label or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
        for character in args.label
    ):
        parser.error("label requires lowercase letters, digits and hyphens")
    evidence = Path(__file__).resolve().parent
    root = evidence.parents[4]
    base = load_base(root)
    destination = evidence / args.label
    destination.mkdir(exist_ok=False)
    expected = json.loads((root / SOURCE_IDENTITY).read_text())["source_expected"]
    files = sorted((root / "afd_plugin").rglob("*.py")) + [
        root / PROBE,
        root / ORACLE,
        root / BASE_HARNESS,
        Path(__file__),
        root / ".agents/dsv41/sources.lock.json",
    ]
    contents = {path.relative_to(root).as_posix(): path.read_bytes() for path in files}
    manifest = {
        name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, data in (
            contents
            | {
                "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
                "source-expected.json": (
                    json.dumps(expected, indent=2) + "\n"
                ).encode(),
            }
        ).items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    remote = (
        base.ssh(
            "python3 -B -c "
            + shlex.quote(
                "import tempfile;print(tempfile.mkdtemp("
                "prefix='c4-failure-20261109-',dir=" + repr(REMOTE_PARENT) + "))"
            )
        )
        .decode()
        .strip()
    )
    base.ssh("cat > " + shlex.quote(remote + "/source.tar.gz"), stream.getvalue())
    base.ssh(
        "cat > " + shlex.quote(remote + "/run_a3_failure.py"),
        Path(__file__).read_bytes(),
    )
    identity = {
        "scope": "working-tree bytes immutable snapshot",
        "head_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "remote": remote,
        "case": args.case,
        "manifest": manifest,
        "snapshot_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "archive_sha256": hashlib.sha256(stream.getvalue()).hexdigest(),
        "source_expected": expected,
        "source_identity_origin": SOURCE_IDENTITY,
    }
    (destination / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    remote_command = "python3 -B " + shlex.quote(remote + "/run_a3_failure.py")
    remote_command += (
        " --remote-root " + shlex.quote(remote) + " --case " + shlex.quote(args.case)
    )
    remote_command += " --model " + shlex.quote(args.model)
    command = [
        "ssh",
        "-F",
        base.SSH_CONFIG,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        base.SSH_HOST,
        remote_command,
    ]
    run = subprocess.run(command, capture_output=True, timeout=base.SSH_TIMEOUT_SECONDS)
    (destination / "remote-run.log").write_bytes(run.stdout + run.stderr)
    raw = base.ssh("tar -czf - -C " + shlex.quote(remote) + " results")
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
