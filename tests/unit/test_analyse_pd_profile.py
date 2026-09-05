from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYZER = ROOT / "tools/dsv4/analyse_pd_profile.sh"


def _make_runtime(tmp_path: Path) -> Path:
    cann_root = tmp_path / "cann-9.0.0"
    atb_root = tmp_path / "atb"
    venv_root = tmp_path / "venv"
    torch_root = tmp_path / "fake-torch"
    for path in (cann_root, atb_root, venv_root / "bin", torch_root / "lib"):
        path.mkdir(parents=True)
    (cann_root / "set_env.sh").write_text("true\n", encoding="utf-8")
    (atb_root / "set_env.sh").write_text("true\n", encoding="utf-8")
    fake_python = venv_root / "bin/python"
    fake_python.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "${{1:-}}" == "-" && $# == 2 ]]; then
  printf '9.0.0\\n'
elif [[ "${{1:-}}" == "-" ]]; then
  printf '{{"python":"%s","torch_npu":"2.10.0.post2"}}\\n' "$0"
elif [[ "${{2:-}}" == *'importlib.util.find_spec'* ]]; then
  printf '%s\\n' '{torch_root}'
elif [[ "${{2:-}}" == *'torch_npu.profiler.profiler'* ]]; then
  mkdir -p ASCEND_PROFILER_OUTPUT
  printf 'Name,Duration\\nop,1\\n' >ASCEND_PROFILER_OUTPUT/kernel_details.csv
  printf '[{{"name":"op"}}]\\n' >ASCEND_PROFILER_OUTPUT/trace_view.json
  printf '{{"hccl":[]}}\\n' >ASCEND_PROFILER_OUTPUT/communication.json
  : >ASCEND_PROFILER_OUTPUT/analyse.done
else
  printf 'unexpected fake python invocation: %s\\n' "$*" >&2
  exit 1
fi
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    config = tmp_path / "role.env"
    config.write_text(
        f"""AFD_PLUGIN_ROOT="{ROOT}"
CANN_ROOT="{cann_root}"
CANN_VERSION="9.0.0"
ATB_ROOT="{atb_root}"
VENV_ROOT="{venv_root}"
VLLM_ROOT="{tmp_path / "vllm"}"
VLLM_ASCEND_ROOT="{tmp_path / "vllm-ascend"}"
""",
        encoding="utf-8",
    )
    return config


def _make_profile(tmp_path: Path, *, complete: bool) -> Path:
    root = tmp_path / "attention_ascend_pt"
    data = root / "PROF_000001/device_0/data"
    data.mkdir(parents=True)
    (root / "profiler_info_0.json").write_text(
        '{"cann_version":"9.0.0"}\n', encoding="utf-8"
    )
    if complete:
        (data.parent / "end_info.0.done").write_text("", encoding="utf-8")
        host = root / "PROF_000001/host"
        host.mkdir()
        (host / "end_info.done").write_text("", encoding="utf-8")
    return root


def test_analyzer_rejects_raw_capture_without_completion_marker(tmp_path):
    config = _make_runtime(tmp_path)
    profile = _make_profile(tmp_path, complete=False)

    result = subprocess.run(
        ["bash", str(ANALYZER), str(config), str(profile)],
        capture_output=True,
        text=True,
        env=os.environ,
    )

    assert result.returncode == 2
    assert "Raw CANN capture is incomplete" in result.stderr
    assert "offline analyse cannot repair" in result.stderr


def test_analyzer_uses_pinned_runtime_and_validates_all_outputs(tmp_path):
    config = _make_runtime(tmp_path)
    profile = _make_profile(tmp_path, complete=True)

    result = subprocess.run(
        ["bash", str(ANALYZER), str(config), str(profile)],
        capture_output=True,
        text=True,
        env=os.environ,
    )

    assert result.returncode == 0, result.stderr
    output = profile / "ASCEND_PROFILER_OUTPUT"
    for name in (
        "kernel_details.csv",
        "trace_view.json",
        "communication.json",
        "analyse.done",
    ):
        assert (output / name).is_file()
    assert "Captured CANN: 9.0.0" in result.stdout
    assert "Analysis outputs validated" in result.stdout
