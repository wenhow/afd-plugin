"""Immutable opt-in actual eager-owner component; no package or model changes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shlex
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

PROBE = "tests/npu/hccl_eager_pipeline_probe.py"
SUPERVISION = "tests/npu/hccl_eager_backend_probe.py"
ORACLE = "tests/npu/hccl_staged_probe.py"
EXISTING_HARNESS = (
    "openspec/changes/afd-hccl-v2-graph/evidence/"
    "a3-native-manager-20261109/run_a3_graph_manager.py"
)
SOURCE_IDENTITY = (
    "openspec/changes/afd-hccl-v2-multistream/evidence/"
    "a3-pipeline-device-20261110/hybrid1-001/identity.json"
)
ENDPOINT_SOURCE = (
    "openspec/changes/afd-hccl-v2-multistream/evidence/"
    "eager-endpoint-source-20261110.json"
)
STREAM_SOURCE = (
    "openspec/changes/afd-hccl-v2-multistream/evidence/"
    "a3-pipeline-cpu-20261110/torch-npu-stream-api-20261110.json"
)
EVIDENCE_DIRECTORY = (
    "openspec/changes/afd-hccl-v2-multistream/evidence/a3-eager-pipeline-20261110"
)
REMOTE_PARENT = "/mnt/workspace/tmp/afd-c0-20261104.hVgsIf"
VENV = "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0"
INNER_TIMEOUT_SECONDS = 240
OUTER_TIMEOUT_SECONDS = 280
SSH_TIMEOUT_SECONDS = 420
EXPECTED_IDLE_BOARDS = 8


def load_existing(root):
    spec = importlib.util.spec_from_file_location(
        "existing_manager_supervision", root / EXISTING_HARNESS
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
        raise RuntimeError("uploaded source differs from immutable manifest")
    expected = json.loads((snapshot / "source-expected.json").read_text())
    site = Path(VENV) / "lib/python3.12/site-packages"
    installed = {
        name: hashlib.sha256((site / name).read_bytes()).hexdigest()
        for name in expected
    }
    checks = {name: installed[name] == digest for name, digest in expected.items()}
    (results / "source-check.json").write_text(json.dumps(checks, indent=2) + "\n")
    (results / "upstream-identity.json").write_text(
        json.dumps(installed, indent=2) + "\n"
    )
    if not all(checks.values()):
        raise RuntimeError("installed source hashes changed; no device launch")
    if len(checks) != 33:
        raise RuntimeError("the reviewed installed-source gate requires all 33 files")
    utilities = load_existing(snapshot)
    env = utilities.load_base(snapshot).prepare_env(snapshot)
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
    # First executable preflight is stdlib metadata only. Neither this harness
    # nor this subprocess imports torch or opens an NPU context.
    preflight = subprocess.run(
        [
            VENV + "/bin/python",
            "-B",
            "-c",
            "import sys,platform,importlib.metadata as m,json;print(json.dumps("
            "{'python':platform.python_version(),'packages':{k:m.version(k) for k in "
            "['torch','torch-npu','vllm','vllm-ascend','fastapi']},"
            "'torch_imported':'torch' in sys.modules,"
            "'torch_npu_imported':'torch_npu' in sys.modules},indent=2))",
        ],
        env=env,
        capture_output=True,
        timeout=30,
        check=True,
    )
    versions = json.loads(preflight.stdout)
    (results / "standalone-no-torch-preflight.json").write_bytes(preflight.stdout)
    if versions["torch_imported"] or versions["torch_npu_imported"]:
        raise RuntimeError("preflight unexpectedly imported torch")
    before = subprocess.run(
        ["npu-smi", "info"], env=env, capture_output=True, timeout=30, check=True
    )
    (results / "npu-before.log").write_bytes(before.stdout + before.stderr)
    if (
        before.stdout.count(b"No running processes found in NPU")
        != EXPECTED_IDLE_BOARDS
    ):
        raise RuntimeError("all eight boards must be idle before this gate")
    command = [
        VENV + "/bin/python",
        "-B",
        str(snapshot / PROBE),
        "--tp-size",
        str(args.tp_size),
        "--devices",
        args.devices,
        "--model",
        args.model,
        "--timeout",
        str(INNER_TIMEOUT_SECONDS),
        "--output-dir",
        str(results / "probe"),
    ]
    started = time.monotonic()
    timed_out, owned, killed = False, {}, []
    with (results / "probe.log").open("wb") as output:
        process = subprocess.Popen(
            command,
            env=env,
            cwd=snapshot,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            utilities.collect_owned_processes(process.pid, owned)
            if time.monotonic() - started > OUTER_TIMEOUT_SECONDS:
                timed_out = True
                killed = utilities.kill_owned(owned)
                break
            time.sleep(0.25)
        process.wait(timeout=10)
    (results / "owned-processes.json").write_text(
        json.dumps({"processes": owned, "killed_owned_pids": killed}, indent=2) + "\n"
    )
    after = subprocess.run(
        ["npu-smi", "info"], env=env, capture_output=True, timeout=30
    )
    (results / "npu-after.log").write_bytes(after.stdout + after.stderr)
    result_path = results / "probe/result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    idle_after = after.stdout.count(b"No running processes found in NPU")
    summary = {
        "scope": "actual ticket-owner component; no native model/MoE/overlap claim",
        "remote": str(remote),
        "tp_size": args.tp_size,
        "devices": args.devices,
        "command": command,
        "source_files_match": all(checks.values()),
        "upstream_files_checked": len(checks),
        "returncode": process.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "outer_timed_out": timed_out,
        "outer_killed_owned_pids": killed,
        "idle_boards_before": EXPECTED_IDLE_BOARDS,
        "idle_boards_after": idle_after,
        "owner_component_passed": result is not None and result["success"],
        "success": (
            process.returncode == 0
            and not timed_out
            and not killed
            and result is not None
            and result["success"]
            and after.returncode == 0
            and idle_after == EXPECTED_IDLE_BOARDS
        ),
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["success"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--devices", required=True, help="FFN ranks then Attention ranks"
    )
    parser.add_argument("--label")
    parser.add_argument("--model", default="/mnt/workspace/models/DeepSeek-V4.1-Flash")
    parser.add_argument("--prepare-only", action="store_true", help="no SSH or devices")
    parser.add_argument("--remote-root", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        devices = tuple(map(int, args.devices.split(",")))
    except ValueError:
        parser.error("devices must be comma-separated NPU indices")
    if (
        len(devices) != 2 * args.tp_size
        or len(set(devices)) != len(devices)
        or min(devices) < 0
    ):
        parser.error("provide exactly two distinct NPU devices per TP coordinate")
    if args.remote_root:
        return remote_run(args)
    if not args.label or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
        for character in args.label
    ):
        parser.error("label requires lowercase letters, digits and hyphens")
    script = Path(__file__).resolve()
    root = script.parents[4]
    evidence = root / EVIDENCE_DIRECTORY
    evidence.mkdir(parents=True, exist_ok=True)
    destination = evidence / args.label
    destination.mkdir(exist_ok=False)
    utilities = load_existing(root)
    base = utilities.load_base(root)
    expected = json.loads((root / SOURCE_IDENTITY).read_text())["source_expected"]
    for name, source in json.loads((root / ENDPOINT_SOURCE).read_text())[
        "sources"
    ].items():
        if source["exists"]:
            expected[name] = source["sha256"]
    files = sorted((root / "afd_plugin").rglob("*.py")) + [
        root / PROBE,
        root / SUPERVISION,
        root / ORACLE,
        root / EXISTING_HARNESS,
        root / utilities.BASE_HARNESS,
        root / ENDPOINT_SOURCE,
        root / STREAM_SOURCE,
        root / ".agents/dsv41/sources.lock.json",
        root / "tests/__init__.py",
        script,
    ]
    # Shared dependency paths can be referenced twice; the archive contains
    # each canonical source exactly once and then verifies every source hash.
    contents = {path.relative_to(root).as_posix(): path.read_bytes() for path in files}
    contents["source-expected.json"] = (json.dumps(expected, indent=2) + "\n").encode()
    manifest = {
        name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, data in (
            contents
            | {"manifest.json": (json.dumps(manifest, indent=2) + "\n").encode()}
        ).items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    with tarfile.open(fileobj=io.BytesIO(stream.getvalue())) as archive:
        names = archive.getnames()
        assert len(names) == len(set(names)) == len(manifest) + 1
        assert all(
            hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
            for name, digest in manifest.items()
        )
    identity = {
        "scope": "immutable working-tree bytes; preparation is not execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "head_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "tp_size": args.tp_size,
        "devices": args.devices,
        "prepare_only": args.prepare_only,
        "manifest": manifest,
        "snapshot_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "archive_sha256": hashlib.sha256(stream.getvalue()).hexdigest(),
        "source_expected": expected,
        "source_identity_origin": SOURCE_IDENTITY,
        "endpoint_source_origin": ENDPOINT_SOURCE,
        "archive_members": len(names),
    }
    if args.prepare_only:
        (destination / "identity.json").write_text(
            json.dumps(identity, indent=2) + "\n"
        )
        print(json.dumps(identity | {"manifest": "see identity.json"}, indent=2))
        return 0
    remote = (
        base.ssh(
            "python3 -B -c "
            + shlex.quote(
                "import tempfile;print(tempfile.mkdtemp("
                "prefix='c6-eager-pipeline-20261110-',dir=" + repr(REMOTE_PARENT) + "))"
            )
        )
        .decode()
        .strip()
    )
    base.ssh("cat > " + shlex.quote(remote + "/source.tar.gz"), stream.getvalue())
    base.ssh(
        "cat > " + shlex.quote(remote + "/run_a3_eager_pipeline.py"),
        script.read_bytes(),
    )
    identity["remote"] = remote
    (destination / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    remote_command = (
        "python3 -B "
        + shlex.quote(remote + "/run_a3_eager_pipeline.py")
        + " --remote-root "
        + shlex.quote(remote)
        + " --tp-size "
        + str(args.tp_size)
        + " --devices "
        + shlex.quote(args.devices)
        + " --model "
        + shlex.quote(args.model)
    )
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
    run = subprocess.run(command, capture_output=True, timeout=SSH_TIMEOUT_SECONDS)
    (destination / "remote-run.log").write_bytes(run.stdout + run.stderr)
    raw = base.ssh("tar -czf - -C " + shlex.quote(remote) + " results")
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(destination, filter="data")
    (destination / "delivery.json").write_text(
        json.dumps({"command": command, "returncode": run.returncode}, indent=2) + "\n"
    )
    print(run.stdout.decode(errors="replace"), flush=True)
    print(run.stderr.decode(errors="replace"), flush=True)
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
