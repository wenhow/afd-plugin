"""Isolated native CPU checks and first EngineArgs declarations."""

import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile
import textwrap


def main():
    root = pathlib.Path(sys.argv[1])
    snapshot = root / "snapshot"
    snapshot.mkdir()
    results = root / "results"
    results.mkdir()
    with tarfile.open(root / "source.tar.gz") as archive:
        archive.extractall(snapshot, filter="data")
    manifest = json.loads((snapshot / "cpu-manifest.json").read_text())
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
        subprocess.check_output(["bash", "-c", environment_script], text=True)
    )
    env.update(
        PYTHONPATH=str(snapshot) + ":" + str(snapshot / "tests/npu"),
        VLLM_USE_V2_MODEL_RUNNER="1",
        AFD_ASYNC_CAM_LAYERED_GMM="0",
    )
    python = "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0/bin/python"
    child = textwrap.dedent("""
    import json,pathlib,pytest,torch,torch_npu
    before=torch.npu.is_initialized()
    code=int(pytest.main(['tests/unit/compat/npu/test_feature_validation.py','-q','-o','addopts=','-p','no:cacheprovider','--tb=short','--junitxml=../results/junit.xml']))
    pathlib.Path('../results/native-state.json').write_text(json.dumps({'before':before,'after':torch.npu.is_initialized(),'returncode':code}))
    raise SystemExit(code or int(before or torch.npu.is_initialized()))
    """)
    run = subprocess.run(
        [python, "-B", "-c", child],
        cwd=snapshot,
        env=env,
        capture_output=True,
        timeout=180,
    )
    (results / "pytest.log").write_bytes(run.stdout + run.stderr)
    summary = {"unit_returncode": run.returncode, "config_runs": []}
    if (results / "native-state.json").exists():
        summary["native_state"] = json.loads(
            (results / "native-state.json").read_text()
        )
    if sys.argv[2] == "green":
        for role in ("native", "attention", "ffn"):
            for mode in ("eager-u1", "eager-u2", "graph-u1", "graph-u2"):
                if role == "native" and mode != "eager-u1":
                    continue
                output = results / ("config-" + role + "-" + mode)
                command = [
                    python,
                    "-B",
                    str(snapshot / "tests/npu/dsv41_functional_worker_probe.py"),
                    "--run-config",
                    "--role",
                    role,
                    "--mode",
                    mode,
                    "--model",
                    "/mnt/workspace/models/DeepSeek-V4.1-Flash",
                    "--output-dir",
                    str(output),
                ]
                if mode.endswith("u2"):
                    command.append("--overlap")
                check = subprocess.run(
                    command, cwd=snapshot, env=env, capture_output=True, timeout=90
                )
                (results / (output.name + ".log")).write_bytes(
                    check.stdout + check.stderr
                )
                record = {"role": role, "mode": mode, "returncode": check.returncode}
                if (output / "result.json").exists():
                    report = json.loads((output / "result.json").read_text())
                    record.update(
                        success=report["success"],
                        npu_before=report["npu_initialized_before_config"],
                        npu_after=report.get("npu_initialized_after_config"),
                        geometry=report.get("configuration", {}).get("native_geometry"),
                        traceback=report.get("traceback"),
                    )
                summary["config_runs"].append(record)
    summary["source_unchanged"] = all(
        hashlib.sha256((snapshot / n).read_bytes()).hexdigest() == h
        for n, h in manifest.items()
    )
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
