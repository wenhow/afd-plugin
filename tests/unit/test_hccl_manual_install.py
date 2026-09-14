from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "tools/dsv4/hccl_manual_install"
MODEL_ARGS = INSTALLER / "bin/model_launch_args.py"


def test_port_check_falls_back_to_proc_without_ss_or_netstat(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "awk").symlink_to("/usr/bin/awk")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        script = f"""
CONFIG_FILE={INSTALLER / "config.env.example"}
source {INSTALLER / "lib/common.sh"}
PATH={bin_dir}
port_is_listening {port}
"""
        result = subprocess.run(
            ["bash", "-c", script],
            env={**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )

    assert result.returncode == 0, result.stderr


def test_preflight_does_not_require_ss():
    preflight = (INSTALLER / "bin/01_preflight.sh").read_text()
    assert "require_command ss" not in preflight
    assert "find ss curl" not in preflight


def test_npu_chip_count_supports_ascend_950_table():
    npu_smi_output = """\
+--------+------------------+---------------+------------------+
| NPU ID | Name             | Health        | Power(W)         |
+========+==================+===============+==================+
| 0      | Ascend950DT      | OK            | 409.7            |
|        |                  | NA            | 0                |
| 1      | Ascend950DT      | OK            | 411.0            |
|        |                  | NA            | 0                |
| 2      | Ascend950DT      | OK            | 402.2            |
| 3      | Ascend950DT      | OK            | 412.3            |
| 4      | Ascend950DT      | OK            | 401.0            |
| 5      | Ascend950DT      | OK            | 409.6            |
| 6      | Ascend950DT      | OK            | 405.8            |
| 7      | Ascend950DT      | OK            | 401.7            |
+========+==================+===============+==================+
"""
    script = f"""
CONFIG_FILE={INSTALLER / "config.env.example"}
source {INSTALLER / "lib/common.sh"}
parse_npu_chip_count
"""
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "PATH": "/usr/local/bin:/usr/bin:/bin"},
        input=npu_smi_output,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "8"


def test_start_uses_runtime_preflight_scope():
    start = (INSTALLER / "bin/07_start.sh").read_text()
    assert 'bash "${SCRIPT_DIR}/01_preflight.sh" runtime' in start


def test_runtime_preflight_does_not_require_install_tools(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in (
        "awk",
        "bash",
        "curl",
        "dirname",
        "grep",
        "nohup",
        "ps",
        "readlink",
        "setsid",
        "uname",
    ):
        (bin_dir / command).symlink_to(Path("/usr/bin") / command)
    npu_smi = bin_dir / "npu-smi"
    npu_smi.write_text("#!/bin/sh\nprintf 'Chip Count : 2\\n'\n")
    npu_smi.chmod(0o755)

    cann_root = tmp_path / "site-cann-a5"
    cann_root.mkdir()
    (cann_root / "set_env.sh").touch()
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4"}), encoding="utf-8"
    )
    (model_root / "quant_model_description.json").write_text("{}\n")
    venv_root = tmp_path / "venv"
    (venv_root / "bin").mkdir(parents=True)
    (venv_root / "bin/python").symlink_to(sys.executable)
    config = tmp_path / "config.env"
    config.write_text(
        f'''SYSTEM_PATH="{bin_dir}"
CANN_ROOT="{cann_root}"
EXPECTED_CANN_VERSION=""
MODEL_PATH="{model_root}"
VENV_ROOT="{venv_root}"
NIC_NAME="lo"
HCCL_IF_IP="127.0.0.1"
PYTHON_BIN="missing-install-python"
ATTENTION_RANKS="1"
FFN_RANKS="1"
ATTENTION_DEVICES="0"
FFN_DEVICES="1"
ATTENTION_MAX_NUM_BATCHED_TOKENS="16"
FFN_MAX_NUM_BATCHED_TOKENS="16"
EXECUTION_MODE="eager"
U_BATCHES="1"
ENABLE_MTP="0"
MTP_NUM_SPECULATIVE_TOKENS="1"
MTP_DRAFT_EXECUTION="eager"
OFFLINE="1"
WHEELHOUSE="{tmp_path / "missing-wheelhouse"}"
ALLOW_NON_AARCH64="0"
ALLOW_CANN_VERSION_MISMATCH="0"
'''
    )

    result = subprocess.run(
        ["bash", str(INSTALLER / "bin/01_preflight.sh"), "runtime"],
        env={
            **os.environ,
            "CONFIG_FILE": str(config),
            "PATH": str(bin_dir),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Runtime preflight passed" in result.stdout
    assert "expected-version=not-enforced" in result.stdout


def test_dirty_afd_seed_is_preserved_and_audited():
    prepare_sources = (INSTALLER / "bin/02_prepare_sources.sh").read_text()
    seed_section = prepare_sources.split("prepare_afd_from_seed_bundle()", 1)[1].split(
        "apply_afd_patch()", 1
    )[0]
    assert 'require_clean_git_tree "${AFD_SEED_ROOT}"' not in seed_section
    assert 'git -C "${AFD_SEED_ROOT}" diff --binary HEAD' in seed_section
    assert "afd-seed-local-changes.patch" in seed_section
    assert 'clone --no-checkout --no-hardlinks "${AFD_SEED_ROOT}"' in seed_section


def test_reused_afd_target_only_allows_clean_lineage_upgrade():
    prepare_sources = (INSTALLER / "bin/02_prepare_sources.sh").read_text()
    seed_section = prepare_sources.split("prepare_afd_from_seed_bundle()", 1)[1].split(
        "apply_afd_patch()", 1
    )[0]
    assert 'require_clean_git_tree "${AFD_PLUGIN_ROOT}"' in seed_section
    assert '"${AFD_SEED_COMMIT}" "${current_commit}"' in seed_section
    assert '"${current_commit}" "${AFD_TARGET_COMMIT}"' in seed_section
    assert 'checkout --detach "${AFD_TARGET_COMMIT}"' in seed_section


def test_release_bundle_allows_clean_a5_tree_upgrade():
    builder = (INSTALLER / "build_bundle.sh").read_text(encoding="utf-8")
    prepare_sources = (INSTALLER / "bin/02_prepare_sources.sh").read_text(
        encoding="utf-8"
    )

    assert "afd-plugin-release.bundle" in builder
    assert "AFD_RELEASE_BUNDLE_SHA256" in builder
    assert "afd-plugin-release.bundle" in prepare_sources
    assert '"${candidate_tree}" == "${current_tree}"' in prepare_sources
    assert "Upgrading clean afd-plugin release tree" in prepare_sources
    assert 'checkout -q --detach "${AFD_TARGET_COMMIT}"' in prepare_sources


def test_a5_bundle_uses_dedicated_validation_guide():
    builder = (INSTALLER / "build_bundle.sh").read_text(encoding="utf-8")
    assert "DEEPSEEK_V4_AFD_PHASE1_A5_VALIDATION_GUIDE_ZH.md" in builder
    assert "PHASE1_A5_VALIDATION_GUIDE_ZH.md" in builder


def test_reused_vllm_ascend_accepts_scm_prefix_with_pinned_commit():
    install_deps = (INSTALLER / "bin/04_install_python_deps.sh").read_text()
    verify_install = (INSTALLER / "bin/06_verify_install.sh").read_text()
    assert '"vllm-ascend": "0.1.dev1+g3da28f941"' not in install_deps
    assert "ascend_version.endswith(ascend_suffix)" in install_deps
    assert "EXPECTED_ASCEND_COMMIT" in install_deps
    assert "endswith(expected_ascend_suffix)" in verify_install


def test_vendor_env_temporarily_disables_and_restores_nounset(tmp_path):
    vendor_env = tmp_path / "set_env.bash"
    vendor_env.write_text(
        'export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH}:/custom/opp"\n'
    )
    script = r"""
source "$1"
set -u
unset ASCEND_CUSTOM_OPP_PATH
source_vendor_env "$2"
[[ "${ASCEND_CUSTOM_OPP_PATH}" == ":/custom/opp" ]]
case $- in
  *u*) ;;
  *) exit 3 ;;
esac
"""
    result = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "bash",
            str(INSTALLER / "lib/common.sh"),
            str(vendor_env),
        ],
        env={
            **os.environ,
            "CONFIG_FILE": str(INSTALLER / "config.env.example"),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_uses_vendor_env_wrapper_for_all_environment_scripts():
    runtime = (INSTALLER / "bin/activate_runtime.sh").read_text()
    assert runtime.count("source_vendor_env") == 3
    assert 'source "${CANN_ROOT}/set_env.sh"' not in runtime
    assert 'source "${CANN_ROOT}/nnal/atb/set_env.sh"' not in runtime
    assert 'source "${ops_env}"' not in runtime


def test_a5_profile_uses_path_only_cann_validation():
    builder = (INSTALLER / "build_bundle.sh").read_text()
    preflight = (INSTALLER / "bin/01_preflight.sh").read_text()
    runtime = (INSTALLER / "bin/activate_runtime.sh").read_text()
    assert "a5-new-install)" in builder
    assert "a5-new-install|a5-reuse)" in builder
    assert 'EXPECTED_CANN_VERSION=""' in builder
    assert 'ATTENTION_RANKS="4"' in builder
    assert 'FFN_RANKS="4"' in builder
    assert 'ATTENTION_DEVICES="0,1,2,3"' in builder
    assert 'FFN_DEVICES="4,5,6,7"' in builder
    assert '[[ -n "${EXPECTED_CANN_VERSION}" ]]' in preflight
    assert '[[ "${EXPECTED_CANN_VERSION}" == "9.0.0"' in preflight
    assert 'if [[ "${EXPECTED_CANN_VERSION}" == "9.0.0" ]]' in runtime
    assert 'export DSV4_CANN_VERSION="${EXPECTED_CANN_VERSION}"' in runtime


def test_model_launch_args_detects_a3_ascend_checkpoint(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4"}), encoding="utf-8"
    )
    (tmp_path / "quant_model_description.json").write_text("{}\n")

    result = subprocess.run(
        [sys.executable, str(MODEL_ARGS), "--model-path", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    args = result.stdout.splitlines()
    assert args[args.index("--quantization") + 1] == "ascend"
    assert args[args.index("--block-size") + 1] == "128"
    assert args[args.index("--safetensors-load-strategy") + 1] == "lazy"


def test_model_launch_args_accepts_official_a5_config_without_quantization_flag(
    tmp_path,
):
    config = INSTALLER / "model/DeepSeek-V4-Flash-config.json"
    (tmp_path / "config.json").write_bytes(config.read_bytes())

    result = subprocess.run(
        [
            sys.executable,
            str(MODEL_ARGS),
            "--model-path",
            str(tmp_path),
            "--quantization",
            "deepseek-v4-native",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    args = result.stdout.splitlines()
    assert "--quantization" not in args
    assert "--hf-overrides" not in args
    assert args[args.index("--block-size") + 1] == "32"
    assert args[args.index("--safetensors-load-strategy") + 1] == "prefetch"
    assert args[args.index("--kv-cache-dtype") + 1] == "auto"

    method = subprocess.run(
        [
            sys.executable,
            str(MODEL_ARGS),
            "--model-path",
            str(tmp_path),
            "--get",
            "speculative_method",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert method.stdout.strip() == "deepseek_mtp"


def test_model_launch_args_rejects_non_official_mxfp8_alias(tmp_path):
    config = json.loads((INSTALLER / "model/DeepSeek-V4-Flash-config.json").read_text())
    config["expert_dtype"] = "fp4"
    config["quantization_config"]["quant_method"] = "mxfp8"
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(MODEL_ARGS), "--model-path", str(tmp_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "restore the official config.json" in result.stderr


def test_install_a5_model_config_backs_up_and_restores_official_config(tmp_path):
    model_root = tmp_path / "model"
    state_root = tmp_path / "state"
    model_root.mkdir()
    official = json.loads(
        (INSTALLER / "model/DeepSeek-V4-Flash-config.json").read_text()
    )
    old_config = dict(official)
    old_config["quantization_config"] = dict(official["quantization_config"])
    old_config["quantization_config"].pop("weight_block_size")
    old_config["quantization_config"]["quant_method"] = "mxfp8"
    (model_root / "config.json").write_text(json.dumps(old_config))
    (model_root / "model.safetensors.index.json").write_text("{}\n")
    config = tmp_path / "config.env"
    config.write_text(
        f'''SYSTEM_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
MODEL_PATH="{model_root}"
STATE_ROOT="{state_root}"
PYTHON_BIN="{sys.executable}"
'''
    )

    result = subprocess.run(
        ["bash", str(INSTALLER / "bin/install_a5_model_config.sh")],
        env={**os.environ, "CONFIG_FILE": str(config)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads((model_root / "config.json").read_text()) == official
    backups = list((state_root / "model-config-backup").glob("config.json.*"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text()) == old_config
    description = json.loads(
        (state_root / "model-config-after-restore.json").read_text()
    )
    assert description["profile"] == "deepseek-v4-native"
    assert description["speculative_method"] == "deepseek_mtp"


def test_launchers_resolve_model_format_and_preserve_soc_version():
    launchers = [
        ROOT / "tools/dsv4/run_v023_native_baseline.sh",
        ROOT / "recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_attention.sh",
        ROOT / "recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_ffn.sh",
        INSTALLER / "bin/run_role.sh",
    ]
    for launcher in launchers:
        script = launcher.read_text(encoding="utf-8")
        assert "model_launch_args.py" in script
        assert "MODEL_SPECULATIVE_METHOD" in script
        assert "--quantization ascend" not in script
        assert "--block-size 128" not in script

    runtime = (
        ROOT / "recipe/npu/deepseek_v4/common/activate_role_runtime.sh"
    ).read_text(encoding="utf-8")
    assert 'export SOC_VERSION="${SOC_VERSION:-ascend910_9362}"' in runtime


def test_a5_reuse_profile_reuses_installed_stack_and_uses_new_plugin_root():
    builder = (INSTALLER / "build_bundle.sh").read_text(encoding="utf-8")
    assert "-name '*.pyc' -o -name '*.pyo'" in builder
    assert "-name __pycache__ -empty -delete" in builder
    section = builder.split("  a5-reuse)", 1)[1].split("  dual-a3-reuse)", 1)[0]
    assert 'CANN_ROOT="/usr/local/Ascend/cann-9.2.0"' in section
    assert 'MODEL_PATH="/home/models/DeepSeek-V4-Flash-MXFP8"' in section
    assert 'MODEL_QUANTIZATION="deepseek-v4-native"' in section
    assert 'ATTENTION_RANKS="4"' in section
    assert 'FFN_RANKS="4"' in section
    assert 'ENABLE_MTP="0"' in section
    assert 'MTP_NUM_SPECULATIVE_TOKENS="1"' in section
    assert 'MTP_DRAFT_EXECUTION="eager"' in section
    assert 'REUSE_VENV="1"' in section
    assert 'INSTALL_PYTHON_DEPS="0"' in section
    assert 'INSTALL_UPSTREAM_STACK="0"' in section
    assert "afd-plugin-phase1-a5-native" in section
