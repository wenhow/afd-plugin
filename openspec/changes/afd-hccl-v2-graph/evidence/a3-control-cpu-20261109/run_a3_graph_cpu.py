"""Freeze C5 graph protocol/cache/geometry CPU seams in the existing A3 venv."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

BASE_HARNESS = (
    "openspec/changes/afd-hccl-v2-microbatch/evidence/"
    "a3-forward-counts-20261109/run_a3_forward_counts.py"
)
SOURCE_IDENTITY = (
    "openspec/changes/afd-hccl-v2-microbatch/evidence/"
    "a3-staged-20261109/reject-digest-001/identity.json"
)
TARGETS = (
    "tests/unit/distributed/test_hccl_graph.py",
    "tests/unit/distributed/test_hccl_plan.py",
    "tests/unit/connectors/npu/test_hccl_staged_transport.py",
    "tests/unit/connectors/npu/test_hccl_graph_cache.py",
    "tests/unit/v1/worker/npu/test_hccl_v41_graph.py",
)
FACTORY_TARGETS = (
    "tests/unit/v1/worker/test_npu_model_runner_v2.py",
    "tests/unit/v1/worker/npu/test_hccl_v41_ubatch_runner.py",
)
ENDPOINT_TARGETS = (
    "tests/unit/connectors/npu/test_hccl_graph.py",
    "tests/unit/connectors/npu/test_hccl_graph_cache.py",
)
CONSUMER_TARGETS = (
    "tests/unit/v1/worker/npu/test_hccl_v41_graph_control.py",
    "tests/unit/v1/worker/npu/test_hccl_v41_graph.py",
    "tests/unit/compat/patches/npu/test_hccl_graph_dispatch.py",
    "tests/unit/compat/patches/test_signatures.py",
    "tests/unit/connectors/npu/test_hccl_graph_cache.py",
)
VENV = "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0"
REMOTE_PARENT = "/mnt/workspace/tmp/afd-c0-20261104.hVgsIf"
CPU_TIMEOUT_SECONDS = 240


def load_base(root):
    spec = importlib.util.spec_from_file_location(
        "existing_a3_harness", root / BASE_HARNESS
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remote_run(remote, targets, exclusions):
    remote = Path(remote)
    snapshot, results = remote / "snapshot", remote / "results"
    snapshot.mkdir()
    results.mkdir()
    with tarfile.open(remote / "source.tar.gz") as archive:
        archive.extractall(snapshot, filter="data")
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if any((snapshot / name).exists() for name in exclusions):
        raise RuntimeError("excluded source exists in uploaded snapshot")
    if any(
        hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest
        for name, digest in manifest.items()
    ):
        raise RuntimeError("uploaded source does not match manifest")
    expected = json.loads((snapshot / "source-expected.json").read_text())
    site = Path(VENV) / "lib/python3.12/site-packages"
    checks = {
        name: hashlib.sha256((site / name).read_bytes()).hexdigest() == digest
        for name, digest in expected.items()
    }
    (results / "source-check.json").write_text(json.dumps(checks, indent=2) + "\n")
    if not all(checks.values()):
        raise RuntimeError("installed upstream source hashes differ")
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
    command = [
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
        "-m",
        "not npu and not gpu",
        "--junitxml=" + str(results / "cpu.xml"),
        *("--ignore=" + name for name in exclusions if name.startswith("tests/unit/")),
        *targets,
    ]
    process = subprocess.run(
        command, cwd=snapshot, env=env, capture_output=True, timeout=CPU_TIMEOUT_SECONDS
    )
    (results / "cpu.stdout.log").write_bytes(process.stdout)
    (results / "cpu.stderr.log").write_bytes(process.stderr)
    summary = {
        "scope": "CPU module seams with actual installed dependencies; no device probe",
        "remote": str(remote),
        "command": command,
        "returncode": process.returncode,
        "source_files_match": all(checks.values()),
        "snapshot_exclusions": exclusions,
        "device_probe_started": False,
        "success": process.returncode == 0,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(process.stdout.decode(errors="replace"), flush=True)
    print(process.stderr.decode(errors="replace"), flush=True)
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label")
    parser.add_argument("--remote-root")
    parser.add_argument("--factory-seams", action="store_true")
    parser.add_argument("--endpoint-seams", action="store_true")
    parser.add_argument("--consumer-seams", action="store_true")
    parser.add_argument("--targets", nargs="+", help="explicit tests/unit paths")
    parser.add_argument(
        "--exclude-paths",
        nargs="+",
        default=[],
        help="repo-relative source/test paths omitted from snapshot and collection",
    )
    args = parser.parse_args()
    if (
        sum(
            (
                args.factory_seams,
                args.endpoint_seams,
                args.consumer_seams,
                bool(args.targets),
            )
        )
        > 1
    ):
        parser.error("seam modes and explicit target selections are exclusive")
    if args.targets and any(
        not name.startswith("tests/unit/") or ".." in Path(name).parts
        for name in args.targets
    ):
        parser.error("explicit targets must stay under tests/unit")
    if any(
        not name.startswith(("afd_plugin/", "tests/unit/"))
        or ".." in PurePosixPath(name).parts
        or "\\" in name
        or name.endswith("/")
        for name in args.exclude_paths
    ):
        parser.error("excluded paths must stay under afd_plugin or tests/unit")
    exclusions = tuple(dict.fromkeys(args.exclude_paths))
    if args.consumer_seams:
        targets = CONSUMER_TARGETS
    else:
        targets = tuple(
            args.targets or (ENDPOINT_TARGETS if args.endpoint_seams else TARGETS)
        )
    targets += FACTORY_TARGETS if args.factory_seams else ()
    if any(name in exclusions for name in targets):
        parser.error("an explicit target cannot also be excluded")
    if args.remote_root:
        return remote_run(args.remote_root, targets, exclusions)
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
    files = sorted((root / "afd_plugin").rglob("*.py"))
    files += sorted((root / "tests/unit").rglob("*.py"))
    files += [
        root / BASE_HARNESS,
        Path(__file__),
        root / "pyproject.toml",
        root / ".agents/dsv41/sources.lock.json",
    ]
    files += [
        path
        for path in (root / "tests/__init__.py", root / "tests/conftest.py")
        if path.exists()
    ]
    contents = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in files
        if not any(
            path.relative_to(root).as_posix() == name
            or path.relative_to(root).as_posix().startswith(name + "/")
            for name in exclusions
        )
    }
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
                "prefix='c5-control-cpu-20261109-',dir=" + repr(REMOTE_PARENT) + "))"
            )
        )
        .decode()
        .strip()
    )
    base.ssh("cat > " + shlex.quote(remote + "/source.tar.gz"), stream.getvalue())
    base.ssh(
        "cat > " + shlex.quote(remote + "/run_a3_graph_cpu.py"),
        Path(__file__).read_bytes(),
    )
    identity = {
        "scope": "working-tree bytes immutable snapshot; CPU only",
        "head_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "remote": remote,
        "manifest": manifest,
        "targets": targets,
        "snapshot_exclusions": exclusions,
        "snapshot_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "archive_sha256": hashlib.sha256(stream.getvalue()).hexdigest(),
        "source_expected": expected,
        "source_identity_origin": SOURCE_IDENTITY,
    }
    (destination / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    remote_command = "python3 -B " + shlex.quote(remote + "/run_a3_graph_cpu.py")
    remote_command += " --remote-root " + shlex.quote(remote)
    remote_command += " --targets " + " ".join(map(shlex.quote, targets))
    if exclusions:
        remote_command += " --exclude-paths " + " ".join(map(shlex.quote, exclusions))
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
