"""Deliver frozen diagnostic sources to an isolated A3 CPU-only scratch run."""

import argparse
import hashlib
import importlib.util
import io
import json
import shlex
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    bundle = root / "reports/dsv41-a5-functional/runtime-source.tar.gz"
    with tarfile.open(bundle) as archive:
        files = {
            item.name: archive.extractfile(item).read()
            for item in archive
            if item.isfile()
        }
    files["cpu_capture_runner.py"] = (
        Path(__file__).with_name("cpu_capture_runner.py").read_bytes()
    )
    manifest = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
    files["source-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, raw in files.items():
            item = tarfile.TarInfo(name)
            item.size = len(raw)
            archive.addfile(item, io.BytesIO(raw))
    raw = buffer.getvalue()
    (output / "source.tar.gz").write_bytes(raw)
    remote = (
        transport.ssh(
            "python3 -c 'import tempfile;print(tempfile.mkdtemp("
            'prefix="a5-router-capture-cpu-",'
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
                    "runtime_source_sha256": hashlib.sha256(
                        bundle.read_bytes()
                    ).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )
    transport.ssh("cat > " + shlex.quote(remote + "/source.tar.gz"), raw)
    transport.ssh(
        "tar -xzf "
        + shlex.quote(remote + "/source.tar.gz")
        + " -C "
        + shlex.quote(remote)
    )
    command = "\n".join(
        (
            "set -e",
            "cd " + shlex.quote(remote),
            "source /mnt/workspace/code/.ascend/cann-9.2.0/ascend-toolkit/"
            "set_env.sh >/dev/null",
            "source /mnt/workspace/code/.ascend/cann-9.2.0/nnal/atb/"
            "set_env.sh >/dev/null",
            "export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:"
            "/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:"
            "$LD_LIBRARY_PATH",
            'export PYTHONPATH="$PWD/tests/npu"',
            "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0/bin/python -B "
            "cpu_capture_runner.py results > cpu.log 2>&1 || true",
            "cat cpu.log",
            "cat results/summary.json",
        )
    )
    log = transport.ssh("bash -lc " + shlex.quote(command))
    (output / "cpu.log").write_bytes(log)
    (output / "results.tar.gz").write_bytes(
        transport.ssh(
            "tar -czf - -C "
            + shlex.quote(remote)
            + " results cpu.log source-manifest.json"
        )
    )
    summary = json.loads(
        transport.ssh("cat " + shlex.quote(remote + "/results/summary.json"))
    )
    (output / "summary.json").write_bytes(
        (json.dumps(summary, indent=2) + "\n").encode()
    )
    print(json.dumps(summary))
    raise SystemExit(int(not summary["success"]))


if __name__ == "__main__":
    main()
