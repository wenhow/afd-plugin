"""Run source-RoPE construction and related CPU contracts without NPU init."""

import hashlib
import json
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    snapshot, results = root / "snapshot", root / "results"
    snapshot.mkdir()
    results.mkdir()
    with tarfile.open(root / "source.tar.gz") as archive:
        archive.extractall(snapshot, filter="data")
    manifest = json.loads((snapshot / "cpu-manifest.json").read_bytes())
    assert all(
        hashlib.sha256((snapshot / n).read_bytes()).hexdigest() == h
        for n, h in manifest.items()
    )
    environment_script = (
        "source /mnt/workspace/code/.ascend/cann-9.2.0/"
        "ascend-toolkit/set_env.sh >/dev/null; "
        "source /mnt/workspace/code/.ascend/cann-9.2.0/nnal/atb/set_env.sh >/dev/null; "
        "export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:"
        "/usr/local/Ascend/driver/lib64/driver:"
        "/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH; "
        '/usr/bin/python3 -c "import json,os;print(json.dumps(dict(os.environ)))"'
    )
    env = json.loads(
        subprocess.check_output(
            ["bash", "-c", environment_script],
            text=True,
        )
    )
    env.update(
        PYTHONPATH=(
            str(snapshot)
            + ":"
            + str(snapshot / "tests/npu")
            + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        ),
        VLLM_USE_V2_MODEL_RUNNER="1",
        AFD_ASYNC_CAM_LAYERED_GMM="0",
    )
    tests = ["tests/unit/v1/worker/npu/test_hccl_v41_ubatch_runner.py"]
    if sys.argv[2] == "red":
        tests += ["-k", "source_rope or missing_source"]
    else:
        tests += [
            "tests/unit/v1/worker/npu/test_hccl_v41_graph_coordinator.py",
            "tests/unit/v1/worker/npu/test_hccl_v41_execution.py",
            "tests/unit/compat/patches/test_signatures.py",
        ]
    tests += [
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "--tb=short",
        "--junitxml=../results/junit.xml",
    ]
    child = (
        "import json,pathlib,pytest,torch,torch_npu;"
        "before=torch.npu.is_initialized();"
        f"code=int(pytest.main({tests!r}));"
        "pathlib.Path('../results/native-state.json').write_text(json.dumps("
        "{'before':before,'after':torch.npu.is_initialized(),'returncode':code}));"
        "raise SystemExit(code or int(before or torch.npu.is_initialized()))"
    )
    command = [
        "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0/bin/python",
        "-B",
        "-c",
        child,
    ]
    run = subprocess.run(
        command, cwd=snapshot, env=env, capture_output=True, timeout=240
    )
    (results / "pytest.log").write_bytes(run.stdout + run.stderr)
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    if (results / "junit.xml").is_file():
        for case in ET.parse(results / "junit.xml").iter("testcase"):
            counts["tests"] += 1
            for name, tag in (
                ("failures", "failure"),
                ("errors", "error"),
                ("skipped", "skipped"),
            ):
                counts[name] += case.find(tag) is not None
    summary = {
        "returncode": run.returncode,
        "counts": counts,
        "native_state": json.loads((results / "native-state.json").read_bytes()),
        "source_unchanged": all(
            hashlib.sha256((snapshot / n).read_bytes()).hexdigest() == h
            for n, h in manifest.items()
        ),
        "scope": "CPU source-table preparation and protocol only; no NPU execution",
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    print((run.stdout + run.stderr).decode(errors="replace")[-5000:])


if __name__ == "__main__":
    main()
