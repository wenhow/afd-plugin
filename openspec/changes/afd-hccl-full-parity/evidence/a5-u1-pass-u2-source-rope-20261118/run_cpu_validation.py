"""Freeze reviewed sources for isolated A3 CPU checks; never use NPU devices."""

import argparse
import hashlib
import importlib.util
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

TRANSFER_CHUNK_BYTES = 128 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("red", "green"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[5]
    output = args.output_dir.resolve()
    output.mkdir(exist_ok=False)
    transport_path = root / (
        "openspec/changes/afd-hccl-v2-microbatch/evidence/"
        "a3-forward-counts-20261109/run_a3_forward_counts.py"
    )
    spec = importlib.util.spec_from_file_location("cpu_transport", transport_path)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    with tarfile.open(
        root
        / (
            "openspec/changes/afd-hccl-full-parity/evidence/"
            "a3-text-normal-worker-20261116/prepared-execute-005/source.tar.gz"
        )
    ) as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    with tarfile.open(root / "reports/dsv41-a5-functional/runtime-source.tar.gz") as a:
        files.update({m.name: a.extractfile(m).read() for m in a if m.isfile()})
    # HEAD tests/fakes exclude unrelated local work. Overlay only this fix's
    # two modified test files and production constructor.
    raw = subprocess.check_output(
        [
            "git",
            "archive",
            "HEAD",
            "tests/unit/fakes",
            "tests/unit/v1/worker/npu",
            "tests/unit/compat/patches/test_signatures.py",
        ],
        cwd=root,
    )
    with tarfile.open(fileobj=io.BytesIO(raw)) as a:
        files.update({m.name: a.extractfile(m).read() for m in a if m.isfile()})
    for name in (
        "afd_plugin/v1/worker/npu/ubatch_runner.py",
        "tests/unit/fakes/hccl_v41_ubatch.py",
        "tests/unit/v1/worker/npu/test_hccl_v41_ubatch_runner.py",
    ):
        files[name] = (root / name).read_bytes()
    files["cpu-manifest.json"] = json.dumps(
        {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    ).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as a:
        for name, data in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            a.addfile(item, io.BytesIO(data))
    raw = buffer.getvalue()
    (output / "source.tar.gz").write_bytes(raw)
    remote = (
        transport.ssh(
            "python3 -c 'import tempfile;print(tempfile.mkdtemp("
            'prefix="a5-u2-rope-cpu-",'
            'dir="/mnt/workspace/tmp/afd-c0-20261104.hVgsIf"))\''
        )
        .decode()
        .strip()
    )
    (output / "remote.json").write_bytes(
        (
            json.dumps(
                {
                    "remote": remote,
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )
    for offset in range(0, len(raw), TRANSFER_CHUNK_BYTES):
        transport.ssh(
            "cat >> " + shlex.quote(remote + "/source.tar.gz"),
            raw[offset : offset + TRANSFER_CHUNK_BYTES],
        )
    log = transport.ssh(
        "python3 -B - " + shlex.quote(remote) + " " + args.phase,
        Path(__file__).with_name("cpu_native_runner.py").read_bytes(),
    )
    (output / "remote.log").write_bytes(log)
    (output / "results.tar.gz").write_bytes(
        transport.ssh(
            "tar -czf - -C " + shlex.quote(remote) + " results",
        )
    )
    summary = json.loads(
        transport.ssh(
            "cat " + shlex.quote(remote + "/results/summary.json"),
        )
    )
    (output / "summary.json").write_bytes(
        (json.dumps(summary, indent=2) + "\n").encode()
    )
    print(log.decode(errors="replace"))
    raise SystemExit(summary["returncode"])


if __name__ == "__main__":
    main()
