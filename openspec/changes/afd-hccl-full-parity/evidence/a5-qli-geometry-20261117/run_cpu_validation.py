"""Run isolated native-dependency CPU checks; never execute an NPU model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shlex
import subprocess
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("red", "green"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[5]
    output = args.output_dir or Path(__file__).resolve().parent / ("cpu-" + args.phase)
    output.mkdir(exist_ok=True)
    if any(output.iterdir()):
        parser.error("CPU evidence output must be empty")
    base_path = root / (
        "openspec/changes/afd-hccl-v2-microbatch/evidence/"
        "a3-forward-counts-20261109/run_a3_forward_counts.py"
    )
    spec = importlib.util.spec_from_file_location("cpu_transport", base_path)
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
    files.pop("manifest.json", None)
    with tarfile.open(
        root / "reports/dsv41-a5-functional/runtime-source.tar.gz"
    ) as archive:
        files.update(
            {
                m.name: archive.extractfile(m).read()
                for m in archive
                if m.isfile() and m.name.endswith(".py")
            }
        )
    for name in (
        "afd_plugin/compat/npu/feature_validation.py",
        "tests/unit/compat/npu/test_feature_validation.py",
        "tests/unit/fakes/vllm_config.py",
    ):
        files[name] = (root / name).read_bytes()
    files["cpu-manifest.json"] = json.dumps(
        {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}
    ).encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, raw in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    raw = buffer.getvalue()
    (output / "source.tar.gz").write_bytes(raw)
    remote = (
        transport.ssh(
            'python3 -c \'import tempfile;print(tempfile.mkdtemp('
            'prefix="a5-qli-cpu-",'
            'dir="/mnt/workspace/tmp/afd-c0-20261104.hVgsIf"))\''
        )
        .decode()
        .strip()
    )
    (output / "remote.json").write_text(
        json.dumps(
            {"remote": remote, "source_sha256": hashlib.sha256(raw).hexdigest()},
            indent=2,
        )
        + "\n"
    )
    transport.ssh("cat > " + shlex.quote(remote + "/source.tar.gz"), raw)
    command = [
        "ssh",
        "-F",
        transport.SSH_CONFIG,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        transport.SSH_HOST,
        "python3 -B - " + shlex.quote(remote) + " " + args.phase,
    ]
    run = subprocess.run(
        command,
        input=Path(__file__).with_name("cpu_native_runner.py").read_bytes(),
        capture_output=True,
        timeout=1100,
    )
    (output / "remote.log").write_bytes(run.stdout + run.stderr)
    print(run.stdout.decode(errors="replace"))
    print(run.stderr.decode(errors="replace"))
    (output / "results.tar.gz").write_bytes(
        transport.ssh("tar -czf - -C " + shlex.quote(remote) + " results")
    )
    raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
