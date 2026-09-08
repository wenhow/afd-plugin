from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "tools/dsv4/hccl_manual_install"


def test_port_check_falls_back_to_proc_without_ss_or_netstat(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "awk").symlink_to("/usr/bin/awk")

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        script = f"""
CONFIG_FILE={INSTALLER / 'config.env.example'}
source {INSTALLER / 'lib/common.sh'}
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

    cann_root = tmp_path / "cann-9.0.0"
    cann_root.mkdir()
    (cann_root / "set_env.sh").touch()
    config = tmp_path / "config.env"
    config.write_text(
        f'''SYSTEM_PATH="{bin_dir}"
CANN_ROOT="{cann_root}"
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
WHEELHOUSE="{tmp_path / 'missing-wheelhouse'}"
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


def test_dirty_afd_seed_is_preserved_and_audited():
    prepare_sources = (INSTALLER / "bin/02_prepare_sources.sh").read_text()
    seed_section = prepare_sources.split("prepare_afd_from_seed_bundle()", 1)[1].split(
        "apply_afd_patch()", 1
    )[0]
    assert 'require_clean_git_tree "${AFD_SEED_ROOT}"' not in seed_section
    assert 'git -C "${AFD_SEED_ROOT}" diff --binary HEAD' in seed_section
    assert 'afd-seed-local-changes.patch' in seed_section
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


def test_reused_vllm_ascend_accepts_scm_prefix_with_pinned_commit():
    install_deps = (INSTALLER / "bin/04_install_python_deps.sh").read_text()
    verify_install = (INSTALLER / "bin/06_verify_install.sh").read_text()
    assert '"vllm-ascend": "0.1.dev1+g3da28f941"' not in install_deps
    assert 'ascend_version.endswith(ascend_suffix)' in install_deps
    assert "EXPECTED_ASCEND_COMMIT" in install_deps
    assert 'endswith(expected_ascend_suffix)' in verify_install


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
