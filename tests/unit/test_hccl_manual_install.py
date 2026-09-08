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
