"""Freeze the real A/F role-profiler probe in the existing A3 venv; no installs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shlex
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

PROBE = "tests/npu/afd_profiler_probe.py"
SUPERVISION = "tests/npu/hccl_eager_backend_probe.py"
ORACLE_DEPENDENCY = "tests/npu/hccl_staged_probe.py"
PLUGIN_PROFILER = "afd_plugin/compat/npu/profiler.py"
PLUGIN_PROFILER_SHA256 = (
    "4dcd664757a788f26efbdf1a3dfaf8acbf7a0a007f26bd2e53c1b6a896d1c117"
)
INSTALLED_PROFILER = "torch_npu/profiler/profiler.py"
INSTALLED_PROFILER_SHA256 = (
    "1a04902572383b74a401449e73929b2c4ce728d6c81bb59341876b03c7a82168"
)
BASE_HARNESS = (
    "openspec/changes/afd-hccl-v2-microbatch/evidence/"
    "a3-forward-counts-20261109/run_a3_forward_counts.py"
)
PROFILER_SOURCE = (
    "openspec/changes/afd-hccl-v2-multistream/evidence/"
    "torch-npu-profiler-source-20261110.json"
)
DESIGN = (
    "openspec/changes/afd-hccl-full-parity/evidence/"
    "afd-profiler-probe-design-20261110.json"
)
EVIDENCE_DIRECTORY = (
    "openspec/changes/afd-hccl-full-parity/evidence/a3-profiler-device-20261110"
)
REMOTE_PARENT = "/mnt/workspace/tmp/afd-c0-20261104.hVgsIf"
VENV = "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0"
INNER_TIMEOUT_SECONDS = 300
OUTER_TIMEOUT_SECONDS = 340
SSH_TIMEOUT_SECONDS = 420
REAP_TIMEOUT_SECONDS = 10
EXPECTED_IDLE_BOARDS = 8


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def remote_run(args):
    remote = Path(args.remote_root)
    snapshot, results = remote / "snapshot", remote / "results"
    snapshot.mkdir()
    results.mkdir()
    summary = {
        "scope": "actual profiler session / controlled NPU matmul; no native runner",
        "remote": str(remote),
        "success": False,
        "device_probe_started": False,
        "device": args.device,
        "role_rank": args.role_rank,
        "outer_timed_out": False,
        "outer_killed_owned_pids": [],
    }
    started, process, owned = time.monotonic(), None, {}
    env = None
    supervision = None
    try:
        with tarfile.open(remote / "source.tar.gz") as archive:
            archive.extractall(snapshot, filter="data")
        manifest = json.loads((snapshot / "manifest.json").read_text())
        summary["snapshot_files_match"] = all(
            hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == digest
            for name, digest in manifest.items()
        )
        if not summary["snapshot_files_match"]:
            raise RuntimeError("uploaded source differs from immutable manifest")
        if manifest[PLUGIN_PROFILER] != PLUGIN_PROFILER_SHA256:
            raise RuntimeError("plugin profiler changed; review before device launch")
        own_name = Path(__file__).name
        uploaded_name = "openspec/changes/afd-hccl-full-parity/evidence/" + own_name
        if (
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            != manifest[uploaded_name]
        ):
            raise RuntimeError("remote runner differs from immutable manifest")
        expected = json.loads((snapshot / "source-expected.json").read_text())
        if expected != {INSTALLED_PROFILER: INSTALLED_PROFILER_SHA256}:
            raise RuntimeError("unexpected installed profiler source gate")
        site = Path(VENV) / "lib/python3.12/site-packages"
        installed = {
            name: hashlib.sha256((site / name).read_bytes()).hexdigest()
            for name in expected
        }
        checks = {name: installed[name] == digest for name, digest in expected.items()}
        write_json(results / "source-check.json", checks)
        write_json(results / "installed-source-identity.json", installed)
        summary["installed_source_files_match"] = all(checks.values())
        if not all(checks.values()):
            raise RuntimeError("installed profiler changed; no device launch")
        env = load_module(
            snapshot / BASE_HARNESS, "existing_a3_environment"
        ).prepare_env(snapshot)
        write_json(
            results / "effective-env.json",
            {
                name: env.get(name)
                for name in (
                    "PYTHONPATH",
                    "LD_LIBRARY_PATH",
                    "ASCEND_HOME_PATH",
                    "ATB_HOME_PATH",
                    "VLLM_PLUGINS",
                )
            },
        )
        preflight = subprocess.run(
            [
                VENV + "/bin/python",
                "-B",
                "-c",
                "import sys,platform,importlib.metadata as m,json;print(json.dumps("
                "{'python':platform.python_version(),"
                "'packages':{k:m.version(k) for k in "
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
            raise RuntimeError("metadata preflight unexpectedly imported torch")
        before = subprocess.run(
            ["npu-smi", "info"], env=env, capture_output=True, timeout=30, check=True
        )
        (results / "npu-before.log").write_bytes(before.stdout + before.stderr)
        summary["idle_boards_before"] = before.stdout.count(
            b"No running processes found in NPU"
        )
        if summary["idle_boards_before"] != EXPECTED_IDLE_BOARDS:
            raise RuntimeError("eight idle boards required before profiler probe")
        sys.path.insert(0, str(snapshot / "tests/npu"))
        supervision = load_module(snapshot / SUPERVISION, "profiler_owned_processes")
        command = [
            VENV + "/bin/python",
            "-B",
            str(snapshot / PROBE),
            "--device",
            str(args.device),
            "--role-rank",
            str(args.role_rank),
            "--timeout",
            str(INNER_TIMEOUT_SECONDS),
            "--output-dir",
            str(results / "probe"),
        ]
        summary["command"] = command
        with (results / "probe.log").open("wb") as output:
            process = subprocess.Popen(
                command,
                cwd=snapshot,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            summary["device_probe_started"] = True
            while process.poll() is None:
                supervision.observe_owned(process.pid, owned)
                if time.monotonic() - started > OUTER_TIMEOUT_SECONDS:
                    summary["outer_timed_out"] = True
                    summary["outer_killed_owned_pids"] = supervision.kill_owned(owned)
                    break
                time.sleep(0.25)
            process.wait(timeout=REAP_TIMEOUT_SECONDS)
        summary["returncode"] = process.returncode
        result_path = results / "probe/result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        summary["profiler_component_passed"] = result is not None and result["success"]
        summary["success"] = (
            process.returncode == 0
            and not summary["outer_timed_out"]
            and not summary["outer_killed_owned_pids"]
            and summary["profiler_component_passed"]
        )
    except BaseException:
        summary["error"] = traceback.format_exc()
    finally:
        if process is not None and process.poll() is None and supervision is not None:
            supervision.observe_owned(process.pid, owned)
            summary["outer_killed_owned_pids"] += supervision.kill_owned(owned)
            process.wait(timeout=REAP_TIMEOUT_SECONDS)
        write_json(results / "owned-processes.json", {"processes": owned})
        if env is None:
            summary["postcheck_not_started"] = "retained CANN environment unavailable"
            summary["success"] = False
        else:
            try:
                after = subprocess.run(
                    ["npu-smi", "info"],
                    env=env,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                (results / "npu-after.log").write_bytes(after.stdout + after.stderr)
                summary["idle_boards_after"] = after.stdout.count(
                    b"No running processes found in NPU"
                )
                if (
                    after.returncode != 0
                    or summary["idle_boards_after"] != EXPECTED_IDLE_BOARDS
                ):
                    summary["success"] = False
            except BaseException:
                summary["postcheck_error"] = traceback.format_exc()
                summary["success"] = False
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(results / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["success"] else 1


def prepare_snapshot(args):
    script = Path(__file__).resolve()
    root = script.parents[4]
    destination = root / EVIDENCE_DIRECTORY / args.label
    destination.mkdir(parents=True, exist_ok=False)
    files = sorted((root / "afd_plugin").rglob("*.py")) + [
        root / PROBE,
        root / SUPERVISION,
        root / ORACLE_DEPENDENCY,
        root / BASE_HARNESS,
        root / PROFILER_SOURCE,
        root / DESIGN,
        root / ".agents/dsv41/sources.lock.json",
        script,
    ]
    contents = {path.relative_to(root).as_posix(): path.read_bytes() for path in files}
    if hashlib.sha256(contents[PLUGIN_PROFILER]).hexdigest() != PLUGIN_PROFILER_SHA256:
        raise RuntimeError("plugin profiler changed; refresh reviewed source pin")
    expected = {INSTALLED_PROFILER: INSTALLED_PROFILER_SHA256}
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
        "scope": "immutable A3-only profiler snapshot; prepare-only is not execution",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "head_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "device": args.device,
        "role_rank": args.role_rank,
        "prepare_only": args.prepare_only,
        "manifest": manifest,
        "snapshot_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "archive_sha256": hashlib.sha256(stream.getvalue()).hexdigest(),
        "source_expected": expected,
        "plugin_profiler_expected_sha256": PLUGIN_PROFILER_SHA256,
        "archive_members": len(names),
        "source_members_unique": True,
        "archive_source_hashes_match": True,
        "other_python_sources_in_snapshot_are_not_device_acceptance": True,
        "a5_portable": False,
        "a5_requirement": "collect/review actual installed profiler source before use",
    }
    return root, destination, identity, stream.getvalue()


def recover_results(raw, root, destination, label):
    """Keep the full archive and map selected trace files below MAX_PATH."""
    archive_path = root / ".tmp" / f"profiler-{label}-results.tar.gz"
    archive_path.parent.mkdir(exist_ok=True)
    if archive_path.exists():
        raise FileExistsError("raw archive already exists; retain the previous attempt")
    archive_path.write_bytes(raw)
    hashes, mapping = {}, {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        names = archive.getnames()
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate result archive members")
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("unsafe result archive path")
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError("only regular result files are supported")
            data = archive.extractfile(member).read()
            digest = hashlib.sha256(data).hexdigest()
            hashes[member.name] = {"sha256": digest, "bytes": len(data)}
            if "traces" not in relative.parts:
                target = destination.joinpath(*relative.parts)
            elif relative.name in (
                "trace_view.json",
                "kernel_details.csv",
                "profiler_info.json",
                "profiler_metadata.json",
            ):
                role = relative.parts[2]
                if role not in ("attention", "ffn"):
                    raise RuntimeError("unexpected trace role")
                phase = (
                    "restart" if relative.parts[4].startswith("restart_") else "initial"
                )
                target = (
                    destination / "trace-evidence" / f"{role}-{phase}-{relative.name}"
                )
            else:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise RuntimeError("recovery would overwrite different raw bytes")
            else:
                target.write_bytes(data)
            mapping[member.name] = target.relative_to(destination).as_posix()
    write_json(
        destination / "retrieval.json",
        {
            "full_archive_path": archive_path.relative_to(root).as_posix(),
            "full_archive_sha256": hashlib.sha256(raw).hexdigest(),
            "full_archive_bytes": len(raw),
            "regular_file_hashes": hashes,
            "member_to_local_path": mapping,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--role-rank", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true", help="no SSH/devices")
    parser.add_argument("--remote-root", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device < 0 or args.role_rank < 0:
        parser.error("device and role-rank must be nonnegative")
    if args.remote_root:
        return remote_run(args)
    if not args.label or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
        for character in args.label
    ):
        parser.error("label requires lowercase letters, digits and hyphens")
    root, destination, identity, archive = prepare_snapshot(args)
    if args.prepare_only:
        write_json(destination / "identity.json", identity)
        print(json.dumps(identity | {"manifest": "see identity.json"}, indent=2))
        return 0
    base = load_module(root / BASE_HARNESS, "existing_a3_delivery")
    remote = (
        base.ssh(
            "python3 -B -c "
            + shlex.quote(
                "import tempfile;print(tempfile.mkdtemp("
                "prefix='c7-profiler-20261110-',dir=" + repr(REMOTE_PARENT) + "))"
            )
        )
        .decode()
        .strip()
    )
    base.ssh("cat > " + shlex.quote(remote + "/source.tar.gz"), archive)
    script = Path(__file__).resolve()
    base.ssh(
        "cat > " + shlex.quote(remote + "/run_a3_profiler.py"), script.read_bytes()
    )
    identity["remote"] = remote
    write_json(destination / "identity.json", identity)
    command = [
        "ssh",
        "-F",
        base.SSH_CONFIG,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        base.SSH_HOST,
        "python3 -B "
        + shlex.quote(remote + "/run_a3_profiler.py")
        + " --remote-root "
        + shlex.quote(remote)
        + " --device "
        + str(args.device)
        + " --role-rank "
        + str(args.role_rank),
    ]
    run = subprocess.run(command, capture_output=True, timeout=SSH_TIMEOUT_SECONDS)
    (destination / "remote-run.log").write_bytes(run.stdout + run.stderr)
    raw = base.ssh("tar -czf - -C " + shlex.quote(remote) + " results")
    recover_results(raw, root, destination, args.label)
    write_json(
        destination / "delivery.json",
        {"command": command, "returncode": run.returncode},
    )
    print(run.stdout.decode(errors="replace"), flush=True)
    print(run.stderr.decode(errors="replace"), flush=True)
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
