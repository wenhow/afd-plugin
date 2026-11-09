"""Immutable, source-pinned A3 native-manager/HCCL phase/cache component run."""

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
import time
from datetime import datetime, timezone
from pathlib import Path

PROBE = "tests/npu/hccl_graph_manager_probe.py"
ORACLE = "tests/npu/hccl_staged_probe.py"
BASE_HARNESS = (
    "openspec/changes/afd-hccl-v2-microbatch/evidence/"
    "a3-forward-counts-20261109/run_a3_forward_counts.py"
)
SOURCE_IDENTITY = (
    "openspec/changes/afd-hccl-v2-graph/evidence/"
    "a3-static-graph-20261109/native-002/identity.json"
)
STREAM_API_IDENTITY = (
    "openspec/changes/afd-hccl-v2-multistream/evidence/"
    "a3-pipeline-cpu-20261110/torch-npu-stream-api-20261110.json"
)
LOCKED_NATIVE_FILES = (
    ("vllm", "vllm/v1/worker/gpu/cudagraph_utils.py"),
    ("vllm", "vllm/v1/worker/gpu/ubatch_utils.py"),
    ("vllm", "vllm/v1/worker/gpu/model_runner.py"),
    ("vllm", "vllm/v1/worker/gpu/input_batch.py"),
    ("vllm", "vllm/v1/worker/ubatch_utils.py"),
    ("vllm", "vllm/v1/worker/ubatching.py"),
    ("vllm_ascend", "vllm_ascend/worker/v2/aclgraph_utils.py"),
    ("vllm_ascend", "vllm_ascend/worker/v2/utils.py"),
    ("vllm_ascend", "vllm_ascend/compilation/acl_graph.py"),
)
REMOTE_PARENT = "/mnt/workspace/tmp/afd-c0-20261104.hVgsIf"
VENV = "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0"
INNER_TIMEOUT_SECONDS = 300
OUTER_TIMEOUT_SECONDS = 340
SSH_TIMEOUT_SECONDS = 480
EXPECTED_IDLE_BOARDS = 8


def load_base(root):
    spec = importlib.util.spec_from_file_location(
        "existing_a3_harness", root / BASE_HARNESS
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_owned_processes(root_pid, owned):
    """Observe only descendants of this launch, retaining PID start identities."""
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
            status = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            owned[pid] = {"start_ticks": status[19], "pgid": os.getpgid(pid)}
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            pending.extend(map(int, children))


def kill_owned(owned):
    killed = []
    for pid, identity in reversed(tuple(owned.items())):
        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
            current = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            if current[19] != identity["start_ticks"]:
                continue
            if identity["pgid"] == pid and os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
            killed.append(pid)
    return killed


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
        raise RuntimeError("uploaded source differs from manifest")
    expected = json.loads((snapshot / "source-expected.json").read_text())
    site = Path(VENV) / "lib/python3.12/site-packages"
    installed = {
        name: hashlib.sha256((site / name).read_bytes()).hexdigest()
        for name in expected
    }
    checks = {name: digest == installed[name] for name, digest in expected.items()}
    (results / "upstream-identity.json").write_text(
        json.dumps(installed, indent=2) + "\n"
    )
    (results / "source-check.json").write_text(json.dumps(checks, indent=2) + "\n")
    if not all(checks.values()):
        raise RuntimeError("installed native sources differ from locked/collected pins")
    env = load_base(snapshot).prepare_env(snapshot)
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
        raise RuntimeError("all eight boards must be idle before the manager probe")
    command = [
        VENV + "/bin/python",
        "-B",
        str(snapshot / PROBE),
        "--devices",
        "0,2",
        "--model",
        args.model,
        "--timeout",
        str(INNER_TIMEOUT_SECONDS),
        "--output-dir",
        str(results / "probe"),
    ]
    if args.stage_bank:
        command.append("--stage-bank")
    if args.compute_pipeline:
        command.extend(
            ["--compute-pipeline", "--pipeline-hybrid", str(args.pipeline_hybrid)]
        )
    timed_out, owned, killed = False, {}, []
    started = time.monotonic()
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
            collect_owned_processes(process.pid, owned)
            if time.monotonic() - started > OUTER_TIMEOUT_SECONDS:
                timed_out = True
                killed = kill_owned(owned)
                break
            time.sleep(0.25)
        process.wait(timeout=10)
    (results / "owned-processes.json").write_text(
        json.dumps(
            {
                "scope": "owned descendants only; start ticks verified before kill",
                "processes": owned,
                "killed_pids": killed,
            },
            indent=2,
        )
        + "\n"
    )
    after = subprocess.run(
        ["npu-smi", "info"], env=env, capture_output=True, timeout=30
    )
    (results / "npu-after.log").write_bytes(after.stdout + after.stderr)
    result_path = results / "probe/summary.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    idle_after = after.stdout.count(b"No running processes found in NPU")
    summary = {
        "scope": "native base manager/UBR + HCCL phase/cache; controlled arithmetic",
        "stage_bank": args.stage_bank,
        "compute_pipeline": args.compute_pipeline,
        "pipeline_hybrid": args.pipeline_hybrid,
        "remote": str(remote),
        "command": command,
        "source_files_match": all(checks.values()),
        "upstream_files_checked": len(checks),
        "returncode": process.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "timed_out": timed_out,
        "outer_killed_owned_pids": killed,
        "cpu_repeated": False,
        "idle_boards_before": EXPECTED_IDLE_BOARDS,
        "idle_boards_after": idle_after,
        "success": process.returncode == 0
        and not timed_out
        and not killed
        and result is not None
        and result["status"] == "passed"
        and not result["supervisor_cleanup"]
        and after.returncode == 0
        and idle_after == EXPECTED_IDLE_BOARDS,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["success"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label")
    parser.add_argument("--model", default="/mnt/workspace/models/DeepSeek-V4.1-Flash")
    parser.add_argument("--remote-root")
    parser.add_argument("--stage-bank", action="store_true")
    parser.add_argument("--compute-pipeline", action="store_true")
    parser.add_argument("--pipeline-hybrid", type=int, choices=(0, 1))
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    if args.compute_pipeline and not args.stage_bank:
        parser.error("--compute-pipeline requires --stage-bank")
    if args.pipeline_hybrid is not None and not args.compute_pipeline:
        parser.error("--pipeline-hybrid requires --compute-pipeline")
    if args.compute_pipeline and args.pipeline_hybrid is None:
        args.pipeline_hybrid = 1
    if args.remote_root:
        return remote_run(args)
    if not args.label or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
        for character in args.label
    ):
        parser.error("label requires lowercase letters, digits and hyphens")
    evidence = Path(__file__).resolve().parent
    root = evidence.parents[4]
    if args.evidence_dir is not None:
        evidence = (root / args.evidence_dir).resolve()
        if not evidence.is_relative_to(root / "openspec/changes"):
            parser.error("evidence directory must stay under openspec/changes")
        evidence.mkdir(parents=True, exist_ok=True)
    base = load_base(root)
    destination = evidence / args.label
    destination.mkdir(exist_ok=False)
    expected = json.loads((root / SOURCE_IDENTITY).read_text())["source_expected"]
    lock = json.loads(
        (root / ".agents/dsv41/sources.lock.json").read_text(encoding="utf-8")
    )
    locked_objects = {}
    for repo, name in LOCKED_NATIVE_FILES:
        source = lock["repositories"][repo]
        data = subprocess.check_output(
            ["git", "-C", source["path"], "show", source["commit"] + ":" + name]
        )
        digest = hashlib.sha256(data).hexdigest()
        expected[name] = digest
        locked_objects[name] = {
            "repo": repo,
            "commit": source["commit"],
            "sha256": digest,
        }
    if args.compute_pipeline:
        source = lock["repositories"]["vllm"]
        name = "vllm/utils/torch_utils.py"
        data = subprocess.check_output(
            ["git", "-C", source["path"], "show", source["commit"] + ":" + name]
        )
        digest = hashlib.sha256(data).hexdigest()
        expected[name] = digest
        locked_objects[name] = {
            "repo": "vllm",
            "commit": source["commit"],
            "sha256": digest,
        }
        installed_api = json.loads(
            (root / STREAM_API_IDENTITY).read_text(encoding="utf-8")
        )
        for name in ("torch_npu/npu/streams.py", "torch_npu/npu/utils.py"):
            expected[name] = installed_api["sources"][name]["sha256"]
    files = sorted((root / "afd_plugin").rglob("*.py")) + [
        root / PROBE,
        root / ORACLE,
        root / BASE_HARNESS,
        Path(__file__),
        root / "tests/__init__.py",
        root / ".agents/dsv41/sources.lock.json",
    ]
    if args.compute_pipeline:
        files.append(root / STREAM_API_IDENTITY)
    contents = {path.relative_to(root).as_posix(): path.read_bytes() for path in files}
    contents["source-expected.json"] = (json.dumps(expected, indent=2) + "\n").encode()
    manifest = {
        name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, data in (
            contents
            | {
                "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
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
                "prefix='c5-native-manager-20261109-',dir=" + repr(REMOTE_PARENT) + "))"
            )
        )
        .decode()
        .strip()
    )
    base.ssh("cat > " + shlex.quote(remote + "/source.tar.gz"), stream.getvalue())
    base.ssh(
        "cat > " + shlex.quote(remote + "/run_a3_graph_manager.py"),
        Path(__file__).read_bytes(),
    )
    identity = {
        "scope": "working-tree bytes immutable snapshot",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "head_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "remote": remote,
        "stage_bank": args.stage_bank,
        "compute_pipeline": args.compute_pipeline,
        "pipeline_hybrid": args.pipeline_hybrid,
        "evidence_directory": destination.relative_to(root).as_posix(),
        "manifest": manifest,
        "snapshot_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "archive_sha256": hashlib.sha256(stream.getvalue()).hexdigest(),
        "source_expected": expected,
        "source_identity_origin": SOURCE_IDENTITY,
        "locked_git_object_sources": locked_objects,
        "installed_stream_source_origin": (
            STREAM_API_IDENTITY if args.compute_pipeline else None
        ),
    }
    (destination / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    remote_command = "python3 -B " + shlex.quote(remote + "/run_a3_graph_manager.py")
    remote_command += (
        " --remote-root " + shlex.quote(remote) + " --model " + shlex.quote(args.model)
    )
    if args.stage_bank:
        remote_command += " --stage-bank"
    if args.compute_pipeline:
        remote_command += " --compute-pipeline --pipeline-hybrid " + str(
            args.pipeline_hybrid
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
