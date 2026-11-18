"""Verify frozen diagnostic sources, CPU snapshots and read-only Git pins."""

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[4]
REPORTS = ROOT / "reports/dsv41-a5-functional"
EXPECTED_CHANGES = {
    "bundle-manifest.json",
    "tests/npu/dsv41_a5_router_probe.py",
    "tests/npu/dsv41_functional_worker_probe.py",
}


def archive_files(path: Path) -> dict[str, bytes]:
    with tarfile.open(path) as archive:
        return {
            item.name: archive.extractfile(item).read()
            for item in archive.getmembers()
            if item.isfile()
        }


def verify_native_pins(identity: dict) -> int:
    verified = 0
    for package, directory in (
        ("vllm", ROOT.parent / "vllm"),
        ("vllm_ascend", ROOT.parent / "vllm-ascend"),
    ):
        expected = {
            path: digest
            for path, digest in identity["source_expected"].items()
            if path.startswith(package + "/")
        }
        commit = identity["commits"][package]
        requests = "".join(f"{commit}:{path}\n" for path in expected)
        result = subprocess.run(
            ["git", "-C", str(directory), "cat-file", "--batch"],
            input=requests.encode(), capture_output=True, check=True,
        )
        stream = io.BytesIO(result.stdout)
        for path, digest in expected.items():
            header = stream.readline().split()
            assert header[1] == b"blob", (path, header)
            data = stream.read(int(header[2]))
            assert stream.read(1) == b"\n"
            assert hashlib.sha256(data).hexdigest() == digest, path
            verified += 1
        assert not stream.read()
    assert verified == len(identity["source_expected"])
    return verified


def main() -> None:
    frozen = archive_files(REPORTS / "runtime-source.tar.gz")
    parent = archive_files(EVIDENCE / "parent-bundle/runtime-source.tar.gz")
    changed = {
        name for name in frozen.keys() | parent.keys()
        if frozen.get(name) != parent.get(name)
    }
    assert changed == EXPECTED_CHANGES, changed
    manifest = json.loads(frozen["bundle-manifest.json"])
    for name, digest in manifest["sources"].items():
        assert hashlib.sha256(frozen[name]).hexdigest() == digest, name
    python_count = 0
    for name, data in frozen.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
            python_count += 1
    for name in EXPECTED_CHANGES - {"bundle-manifest.json"}:
        assert (ROOT / name).read_bytes() == frozen[name], name
    snapshots = []
    for name in ("cpu-cli-final", "cpu-capture-final"):
        snapshot = archive_files(EVIDENCE / name / "source.tar.gz")
        assert all(snapshot[path] == data for path, data in frozen.items())
        summary = json.loads((EVIDENCE / name / "summary.json").read_bytes())
        assert summary["source_unchanged"]
        snapshots.append(name)
    lock = json.loads((REPORTS / "bundle-lock.json").read_bytes())
    archive = REPORTS / "runtime-source.tar.gz"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert lock["runtime_source_sha256"] == digest
    for name, expected in lock["production_sources"].items():
        assert frozen[name] == parent[name], name
        assert hashlib.sha256(frozen[name]).hexdigest() == expected, name
    launcher = ROOT / "tests/npu/dsv41_a5_functional_validation.py"
    assert hashlib.sha256(launcher.read_bytes()).hexdigest() == lock["launcher_sha256"]
    native_count = verify_native_pins(json.loads(frozen["native-source-identity.json"]))
    report = {
        "runtime_source_sha256": digest,
        "launcher_sha256": lock["launcher_sha256"],
        "changed_from_parent": sorted(changed),
        "production_sources_preserved": len(lock["production_sources"]),
        "python_sources_compiled": python_count,
        "native_git_pins_verified": native_count,
        "cpu_snapshots_match": snapshots,
        "scope": "Static source identity and CPU snapshots; A5 fix effect pending",
    }
    (EVIDENCE / "source-verification.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
