"""Run actual stdlib CLI tests in an isolated frozen source directory."""

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import textwrap


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    snapshot, results = root / "snapshot", root / "results"
    snapshot.mkdir()
    results.mkdir()
    with tarfile.open(root / "source.tar.gz") as archive:
        archive.extractall(snapshot, filter="data")
    (snapshot / ".tmp").mkdir()
    manifest = json.loads((snapshot / "cpu-cli-source-manifest.json").read_text())
    assert all(
        hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == expected
        for name, expected in manifest.items()
    )
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        ("/opt/buildtools/python-3.12.9/lib", env.get("LD_LIBRARY_PATH", ""))
    )
    env["PYTHONPATH"] = str(snapshot) + ":" + str(snapshot / "tests/npu")
    child = textwrap.dedent("""
        import json,pathlib,sys,unittest
        suite=unittest.defaultTestLoader.loadTestsFromName(
            'tests.e2e.diagnostics.test_dsv41_a5_functional_validation')
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        runtime=[name for name in sys.modules if name.split('.')[0] in (
            'torch','torch_npu','vllm','vllm_ascend','afd_plugin','cann_ops_transformer')]
        summary={'tests_run':result.testsRun,'failures':len(result.failures),
                 'errors':len(result.errors),'runtime_modules_imported':runtime}
        pathlib.Path('../results/tests.json').write_text(json.dumps(summary))
        raise SystemExit(int(not result.wasSuccessful() or bool(runtime)))
    """)
    command = [
        "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0/bin/python",
        "-B",
        "-S",
        "-c",
        child,
    ]
    run = subprocess.run(
        command, env=env, cwd=snapshot, capture_output=True, timeout=180
    )
    (results / "cli.log").write_bytes(run.stdout + run.stderr)
    summary = {
        "returncode": run.returncode,
        "command": command,
        "source_unchanged": all(
            hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == expected
            for name, expected in manifest.items()
        ),
        "scope": "CPU CLI and metadata only, no A5 operator acceptance",
    }
    if (results / "tests.json").exists():
        summary["tests"] = json.loads((results / "tests.json").read_text())
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
