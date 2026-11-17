"""Check the frozen CLI on the existing A3 interpreter without runtime imports."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shlex
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    family = "reports/dsv41-a5-functional/"
    with tarfile.open(root / family / "runtime-source.tar.gz") as archive:
        files = {
            item.name: archive.extractfile(item).read()
            for item in archive
            if item.isfile()
        }
    for name in (
        "tests/e2e/diagnostics/test_dsv41_a5_functional_validation.py",
        family + "runtime-source.tar.gz",
        family + "bundle-lock.json",
        family + "preparation.json",
    ):
        files[name] = (root / name).read_bytes()
    files["cpu-cli-source-manifest.json"] = json.dumps(
        {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
    ).encode()
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
            'prefix="a5-sync-cli-",'
            'dir="/mnt/workspace/tmp/afd-c0-20261104.hVgsIf"))\''
        )
        .decode()
        .strip()
    )
    (output / "remote.json").write_bytes(
        (
            json.dumps(
                {"remote": remote, "source_sha256": hashlib.sha256(raw).hexdigest()}
            )
            + "\n"
        ).encode()
    )
    transport.ssh("cat > " + shlex.quote(remote + "/source.tar.gz"), raw)
    result = transport.ssh(
        "python3 -B - " + shlex.quote(remote),
        Path(__file__).with_name("cpu_cli_runner.py").read_bytes(),
    )
    (output / "summary.json").write_bytes(result)
    (output / "results.tar.gz").write_bytes(
        transport.ssh("tar -czf - -C " + shlex.quote(remote) + " results")
    )
    summary = json.loads(result)
    print(json.dumps(summary))
    raise SystemExit(summary["returncode"])


if __name__ == "__main__":
    main()
