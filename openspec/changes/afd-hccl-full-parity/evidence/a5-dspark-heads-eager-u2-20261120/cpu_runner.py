"""Run an uploaded snapshot inside the existing isolated A3 CPU environment."""

import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from unittest.mock import patch


def run_tests():
    # Runtime dependencies are imported only in the isolated CPU test child.
    import pytest
    import torch
    import torch_npu  # noqa: F401

    identity = json.loads(Path("identity.json").read_text())
    native = Path(importlib.util.find_spec("vllm_ascend").origin).parent
    matches = {
        name: hashlib.sha256((native / name).read_bytes()).hexdigest() == digest
        for name, digest in identity["pins"].items()
    }
    if not all(matches.values()):
        raise RuntimeError(matches)
    before = torch.npu.is_initialized()
    if before:
        raise RuntimeError("NPU was already initialized")
    with (
        patch.object(
            torch.npu,
            "_lazy_init",
            side_effect=RuntimeError("CPU tests forbid NPU initialization"),
        ),
        patch.object(torch.npu, "synchronize", return_value=None),
    ):
        code = int(
            pytest.main(
                identity["targets"]
                + [
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--tb=short",
                    "--junitxml=../results/junit.xml",
                ]
            )
        )
    after = torch.npu.is_initialized()
    Path("../results/state.json").write_text(
        json.dumps(
            {
                "exit_code": code,
                "npu_initialized_before": before,
                "npu_initialized_after": after,
                "device_synchronize_stubbed": True,
                "native_pins_match": matches,
            },
            indent=2,
        )
    )
    return code or int(after)


def main():
    root = Path(sys.argv[1]).resolve()
    allowed = Path("/mnt/workspace/tmp/afd-c0-20261104.hVgsIf").resolve()
    if not root.is_relative_to(allowed):
        raise ValueError("CPU snapshot must stay within the authorized workspace")
    snapshot = root / "snapshot"
    snapshot.mkdir()
    results = root / "results"
    results.mkdir()
    with tarfile.open(root / "source.tar.gz") as archive:
        for member in archive:
            name = Path(member.name)
            if not member.isfile() or name.is_absolute() or ".." in name.parts:
                raise ValueError(member.name)
            archive.extract(member, snapshot, filter="data")
    identity = json.loads((snapshot / "identity.json").read_text())
    if not all(
        hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == digest
        for name, digest in identity["manifest"].items()
    ):
        raise ValueError("Uploaded source differs from the manifest")
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
    env["PYTHONPATH"] = str(snapshot) + (
        ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    env["AFD_ASYNC_CAM_LAYERED_GMM"] = "0"
    run = subprocess.run(
        [
            "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0/bin/python",
            "-B",
            str(Path(__file__).resolve()),
            "--tests",
        ],
        cwd=snapshot,
        env=env,
        capture_output=True,
        timeout=300,
    )
    (results / "pytest.log").write_bytes(run.stdout + run.stderr)
    unchanged = all(
        hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == digest
        for name, digest in identity["manifest"].items()
    )
    (results / "summary.json").write_text(
        json.dumps(
            {
                "exit_code": run.returncode,
                "source_unchanged": unchanged,
                "hardware_acceptance": False,
            },
            indent=2,
        )
    )
    print((run.stdout + run.stderr).decode(errors="replace")[-6500:])
    return run.returncode or int(not unchanged)


if __name__ == "__main__":
    raise SystemExit(run_tests() if sys.argv[1] == "--tests" else main())
