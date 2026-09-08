from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.dsv4 import generate_golden
from tools.dsv4.generate_golden import _load_prompt_source, _parse_metadata
from tools.dsv4.mooncake_pd_config import build_mooncake_pd_config

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_TOOL = ROOT_DIR / "tools/dsv4/mooncake_pd_config.py"
MANUAL_PD_DIR = ROOT_DIR / "tools/dsv4/mooncake_pd_manual"


def test_build_mooncake_pd_config_is_role_and_topology_explicit():
    config = build_mooncake_pd_config(
        role="kv_consumer",
        engine_id="decode-0",
        kv_port=30100,
        prefill_dp_size=2,
        prefill_tp_size=4,
        decode_dp_size=8,
        decode_tp_size=1,
    )

    assert config == {
        "kv_connector": "MooncakeHybridConnector",
        "kv_role": "kv_consumer",
        "kv_port": 30100,
        "engine_id": "decode-0",
        "kv_parallel_size": 1,
        "kv_connector_extra_config": {
            "prefill": {"dp_size": 2, "tp_size": 4},
            "decode": {"dp_size": 8, "tp_size": 1},
        },
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"role": "kv_both"}, "role must be"),
        ({"engine_id": ""}, "engine_id"),
        ({"kv_port": 0}, "kv_port"),
        ({"decode_tp_size": 0}, "decode_tp_size"),
    ],
)
def test_build_mooncake_pd_config_rejects_invalid_values(overrides, message):
    values = {
        "role": "kv_producer",
        "engine_id": "prefill-0",
        "kv_port": 30000,
        "prefill_dp_size": 2,
        "prefill_tp_size": 4,
        "decode_dp_size": 8,
        "decode_tp_size": 1,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_mooncake_pd_config(**values)


def test_mooncake_pd_config_cli_outputs_compact_json():
    output = subprocess.check_output(
        [
            sys.executable,
            str(CONFIG_TOOL),
            "--role",
            "kv_consumer",
            "--engine-id",
            "decode-0",
            "--kv-port",
            "30100",
            "--prefill-dp-size",
            "2",
            "--prefill-tp-size",
            "4",
            "--decode-dp-size",
            "8",
            "--decode-tp-size",
            "1",
        ],
        text=True,
    )

    assert " " not in output.strip()
    assert json.loads(output)["kv_role"] == "kv_consumer"


def test_mooncake_pd_recipe_keeps_kv_transfer_off_ffn():
    hccl_recipe_dir = ROOT_DIR / "recipe/npu/P2pHcclAFDConnector/deepseek_v4"
    attention = (hccl_recipe_dir / "afd_attention.sh").read_text()
    ffn = (hccl_recipe_dir / "afd_ffn.sh").read_text()
    prefill = (hccl_recipe_dir / "mooncake_pd/prefill.sh").read_text()

    assert "--role kv_consumer" in attention
    assert "--kv-transfer-config" in attention
    assert 'source "${ROOT_DIR}/tools/dsv4/check_mooncake_runtime.sh"' in attention
    assert 'export VLLM_HOST_IP="${VLLM_HOST_IP:-${HCCL_IF_IP}}"' in attention
    assert "--kv-transfer-config" not in ffn
    for script in (attention, ffn):
        assert '--shutdown-timeout "$VLLM_SHUTDOWN_TIMEOUT_SECONDS"' in script
        assert (
            "VLLM_SHUTDOWN_TIMEOUT_SECONDS="
            '"${VLLM_SHUTDOWN_TIMEOUT_SECONDS:-0}"' in script
        )
    assert "--middleware" in attention
    assert (
        "afd_plugin.compat.npu.profile_api.afd_profile_control_middleware" in attention
    )
    assert "--role kv_producer" in prefill
    assert 'source "${ROOT_DIR}/tools/dsv4/check_mooncake_runtime.sh"' in prefill
    assert 'export VLLM_HOST_IP="${VLLM_HOST_IP:-${HCCL_IF_IP}}"' in prefill


def test_mooncake_pd_proxy_bypasses_environment_http_proxies():
    proxy = (
        ROOT_DIR / "recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/proxy.sh"
    ).read_text()

    assert (
        "unset HTTP_PROXY HTTPS_PROXY ALL_PROXY "
        "http_proxy https_proxy all_proxy" in proxy
    )


def test_mooncake_pd_manual_entry_is_safe_and_size_capped():
    script_path = MANUAL_PD_DIR / "pd.sh"
    config_path = MANUAL_PD_DIR / "config.env.example"
    runtime_path = ROOT_DIR / "tools/dsv4/check_mooncake_runtime.sh"
    roundtrip_path = ROOT_DIR / "tools/dsv4/check_mooncake_npu_roundtrip.py"
    activation_path = ROOT_DIR / "tools/dsv4/activate_runtime.sh"
    script = script_path.read_text()
    config = config_path.read_text()
    runtime = runtime_path.read_text()
    roundtrip = roundtrip_path.read_text()
    activation = activation_path.read_text()

    subprocess.run(["bash", "-n", str(script_path)], check=True)
    help_output = subprocess.check_output(
        ["bash", str(script_path), "--help"], text=True
    )

    for action in (
        "init",
        "install",
        "check",
        "start",
        "status",
        "smoke",
        "record-control",
        "validate",
        "stop",
        "collect",
    ):
        assert action in help_output

    assert "pkill" not in script
    assert "stop_name attention" in script
    assert "stop_name ffn" in script
    assert "stop_name decode-control" in script
    assert "ENABLE_PD=1" in script
    assert "RUN_LOCAL_ROUNDTRIP=0\n  check_action" in script
    assert '"${MODEL_NAME}"' in script
    assert 'tail -c "${ARTIFACT_LOG_TAIL_BYTES}"' in script
    assert 'timeout "${PORT_SNAPSHOT_TIMEOUT_SECONDS}" netstat -lntp' in script
    assert 'write_raw_tcp_tables "${output_path}"' in script
    assert "batches.json recovery.json" in script
    assert 'tail -n 50 >"${temp_dir}/kv-transfer-evidence.txt"' in script
    assert 'tail -n 200 >"${temp_dir}/fatal-markers.txt"' in script
    assert 'grep -EHn "${FATAL_PATTERN}"' in script
    assert "recipe/npu/deepseek_v4/common/validate_golden.py" in script
    assert "MOONCAKE_INSTALL_MODE:=wheel" in script
    assert 'MOONCAKE_INSTALL_MODE="existing"' in config
    assert (
        'write_mooncake_fingerprint "${STATE_ROOT}/mooncake-libraries.sha256"' in script
    )
    assert 'CODE_ROOT="/data/z00569729/code"' in config
    assert "CODE_ROOT:=/data/z00569729/code" in script
    assert 'AFD_PD_COMMIT="CHANGE_ME"' in config
    assert "validate_afd_worktree" in script
    assert "afd-plugin worktree must be clean" in script
    assert "tools/dsv4/check_mooncake_runtime.sh" in script
    assert "contains changes outside the delivered overlay" not in script
    assert "run_as_root()" in script
    assert "run_as_root dnf install -y iproute" in script
    assert "run_as_root yum install -y iproute" in script
    assert "run_as_root apt-get install -y" in script
    assert "require_command ip" not in script
    assert "require_command ss" not in script
    assert "fcntl.ioctl(sock.fileno(), 0x8915, request)" in script
    assert "/proc/net/tcp /proc/net/tcp6" in script
    assert "netstat -lnt" in script
    assert "netstat -ie" in script
    assert 'ifconfig "${NIC_NAME}"' in script
    assert "resolve_hostname()" in script
    assert "read -r node_name </proc/sys/kernel/hostname" in script
    assert "printf 'hostname=%s\\n' \"$(resolve_hostname)\"" in script
    assert "printf 'hostname=%s\\n' \"$(hostname)\"" not in script
    assert 'if port_is_listening "${port}"; then' in script
    assert 'port_is_listening "${port}" && die' not in script
    assert 'die "Mooncake local NPU round-trip failed;' in script
    assert "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2" in runtime
    assert "/usr/lib64/libjemalloc.so.2" in runtime
    assert "MOONCAKE_JEMALLOC" in runtime
    assert "ldconfig -p" in runtime
    assert "MOONCAKE_VENV_SITE" in runtime
    assert 'MOONCAKE_LIBRARY_DIR="${MOONCAKE_LIBRARY_DIR:-}"' in runtime
    assert "/usr/local/lib /usr/local/lib64" in runtime
    assert "DSV4_CANN_VERSION:-" in runtime
    assert 'CANN_VERSION="9.0.0"' in config
    assert 'MOONCAKE_LIBRARY_DIR=""' in config
    assert 'ATB_ROOT=""' in config
    assert "validate_cann_version" in script
    assert 'kill -TERM "${pid}"' in script
    assert "wait_for_profile_raw_completion" in script
    assert "wait_for_profile_raw_started" in script
    assert "profile_start_action" in script
    assert "profile_check_action" in script
    assert "profile_stop_action" in script
    assert "The running service was not started with AFD_PROFILE_ENABLE=1" in script
    assert '"${STATE_ROOT}/profile-session.env"' in script
    assert '"${STATE_ROOT}/profile-started.env"' in script
    assert '"${STATE_ROOT}/collect-final-gates.txt"' in script
    profile_start = script.split("profile_start_action()", 1)[1].split(
        "profile_check_action()", 1
    )[0]
    assert '--max-time "${AFD_PROFILE_START_TIMEOUT_SECONDS}"' in profile_start
    assert "AFD_PROFILE_FINALIZE_TIMEOUT_SECONDS" not in profile_start
    assert "profile-started.env.tmp.$$" in profile_start
    assert profile_start.index("wait_for_profile_raw_started attention") < (
        profile_start.index('mv "${started_tmp}"')
    )
    assert "Profile is not finalized; run profile-stop before stop" in script
    assert '"http://127.0.0.1:${DECODE_API_PORT}/afd/profile/start"' in script
    assert '"http://127.0.0.1:${DECODE_API_PORT}/afd/profile/stop"' in script
    assert "end_info*.done" in script
    assert "*/PROF_*/device_*/data/*" in script
    assert "*/PROF_*/host/end_info.done" in script
    assert "resolve_atb_root" in script
    assert "CANN_ROOT is not 9.0.1" not in script
    assert "unset DSV4_CANN_ROOT" not in activation
    assert 'export DSV4_CANN_VERSION="${CANN_VERSION}"' in script
    assert "MOONCAKE_ENGINE_ID MOONCAKE_KV_PORT MOONCAKE_LIBRARY_DIR" in script
    assert 'export DSV4_ATB_ROOT="${ATB_ROOT}"' in script
    assert "/usr/local/Ascend/nnal/atb/set_env.sh" in activation
    assert "source_vendor_env()" in activation
    assert "had_nounset" in activation
    assert "set +u" in activation
    assert "set -u" in activation
    assert 'importlib.util.find_spec("torch")' in activation
    assert 'DSV4_TORCH_LIB="${DSV4_TORCH_PACKAGE}/lib"' in activation
    assert 'DSV4_EXTRA_OPP_ENV="${DSV4_EXTRA_OPP_ENV:-}"' in activation
    assert 'source_vendor_env "${DSV4_EXTRA_OPP_ENV}"' in activation
    assert "NNAL/ATB runtime check failed" in runtime
    assert "libatb.so =>" in runtime
    assert 'ARTIFACT_LOG_TAIL_BYTES="262144"' in config
    assert 'AFD_PROFILE_FINALIZE_TIMEOUT_SECONDS="300"' in config
    assert 'AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS="240"' in config
    assert (
        "export VLLM_SHUTDOWN_TIMEOUT_SECONDS="
        '"${AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS}"' in script
    )
    assert 'ARTIFACT_MAX_BYTES="2097152"' in config
    assert 'PORT_SNAPSHOT_TIMEOUT_SECONDS="10"' in config
    assert "http://127.0.0.1:${FFN_PROCESS_PORT}" not in script
    assert 'DEPLOYMENT_VARIANT="CHANGE_ME"' in config
    assert 'PD_CONTROL_GOLDEN_PATH="/data/z00569729/validation/' in config
    assert "pd_control|pd_afd" in script
    assert "record_control_action" in script
    assert "functional_smoke_action" in script
    assert "run_pd_functional_smoke.py" in script
    assert "golden_checked=0" in script
    assert "status=f0_functional_smoke_passed_no_golden" in script
    assert 'ALLOW_COLOCATED_PD_CONTROL="0"' in config
    assert "device_lists_are_disjoint" in script
    assert "validate_colocated_control_processes" in script
    assert "process_is_descendant_of" in script
    assert "validate_control_golden" in script
    assert '--golden "${PD_CONTROL_GOLDEN_PATH}"' in script
    assert "PD control golden metadata mismatch" in script
    assert '--host "$5" --interface "$6"' in script
    assert 'parser.add_argument("--host", default="127.0.0.1")' in roundtrip
    assert 'parser.add_argument("--interface", default="lo")' in roundtrip
    assert "session_id = f\"{host}:{remote['rpc_port']}\"" in roundtrip
    assert "VLLM_ASCEND_WORKTREE_MODE" in script
    assert "batch_invariant_patch" in script
    vllm_ascend_worktree = script.split("validate_vllm_ascend_worktree()", 1)[1].split(
        "validate_afd_worktree()", 1
    )[0]
    assert "--untracked-files=all" in vllm_ascend_worktree
    assert "--untracked-files=no" not in vllm_ascend_worktree
    assert "cf97a0b6e509fbb128e847babbf8f01cc953f06cb3126936cc4111bbab60b897" in script
    assert "batch-invariant backend: OK" in script


def test_mooncake_pd_control_recipe_does_not_load_afd():
    control_path = (
        ROOT_DIR
        / "recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/decode_control.sh"
    )
    prefill_path = (
        ROOT_DIR / "recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/prefill.sh"
    )
    control = control_path.read_text()
    prefill = prefill_path.read_text()

    subprocess.run(["bash", "-n", str(control_path)], check=True)
    subprocess.run(["bash", "-n", str(prefill_path)], check=True)
    assert "ascend_kv_connector,afd" not in control
    assert "--additional-config" not in control
    assert "--kv-transfer-config" in control
    assert (
        "VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector"
        in control
    )
    assert 'case "${ENABLE_AFD_PLUGIN:-1}"' in prefill


def test_mooncake_pd_recipes_accept_dp4_tp2_without_relaxing_other_modes():
    attention_path = (
        ROOT_DIR / "recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_attention.sh"
    )
    control_path = (
        ROOT_DIR
        / "recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/decode_control.sh"
    )
    manual_path = MANUAL_PD_DIR / "pd.sh"
    attention = attention_path.read_text()
    control = control_path.read_text()
    manual = manual_path.read_text()

    assert "Mooncake PD M9 baseline requires eager/U1" not in attention
    assert "Mooncake PD M9 baseline requires MTP off" not in attention
    assert "8:1|4:1|4:2)" in control
    assert "8:1|4:1|4:2)" in manual
    assert 'export TENSOR_PARALLEL_SIZE="${DECODE_TP_SIZE}"' in manual
    assert 'export EXECUTION_MODE="${DECODE_EXECUTION_MODE}"' in manual
    assert 'export U_BATCHES="${DECODE_U_BATCHES}"' in manual
    assert 'export ENABLE_MTP="${DECODE_ENABLE_MTP}"' in manual
    assert "--enable-dbo" in control
    assert "FULL_DECODE_ONLY" in control
    assert "--speculative-config" in control
    assert "TP2 full-draft Graph U2 + MTP is not validated" in control


@pytest.mark.parametrize(
    ("variant", "role"),
    [
        ("pd_control", "prefill"),
        ("pd_control", "decode"),
        ("pd_control", "proxy"),
        ("pd_afd", "prefill"),
        ("pd_afd", "decode"),
        ("pd_afd", "proxy"),
    ],
)
def test_mooncake_pd_manual_print_config_preserves_variant_and_role(
    tmp_path, variant, role
):
    config_path = tmp_path / "config.env"
    config_path.write_text(f'DEPLOYMENT_VARIANT="{variant}"\nNODE_ROLE="{role}"\n')

    output = subprocess.check_output(
        ["bash", str(MANUAL_PD_DIR / "pd.sh"), "print-config", str(config_path)],
        text=True,
    )

    assert f"DEPLOYMENT_VARIANT={variant}" in output
    assert f"NODE_ROLE={role}" in output


def test_generate_golden_metadata_rejects_duplicates():
    assert _parse_metadata(["baseline_kind=mooncake_pd_no_afd"]) == {
        "baseline_kind": "mooncake_pd_no_afd"
    }
    with pytest.raises(ValueError, match="duplicate"):
        _parse_metadata(["kind=first", "kind=second"])


def test_generate_golden_accepts_prompt_only_source(tmp_path):
    source_path = tmp_path / "prompts.json"
    source_path.write_text(json.dumps({"prompts": ["first", "second"]}))

    prompts, reference = _load_prompt_source(source_path)

    assert prompts == ["first", "second"]
    assert reference == {}


def test_generate_golden_records_stability_metadata_and_reference(
    monkeypatch, tmp_path
):
    source_path = tmp_path / "native.json"
    output_path = tmp_path / "control.json"
    source_path.write_text(
        json.dumps(
            {
                "golden": {
                    "0": {
                        "prompt": "prompt",
                        "prompt_token_ids": [1],
                        "token_ids": [2],
                    }
                }
            }
        )
    )

    monkeypatch.setattr(
        generate_golden,
        "_request_completion",
        lambda endpoint, model, prompt, timeout: {
            "prompt_token_ids": [1],
            "token_ids": [3],
            "text": "control",
            "finish_reason": "length",
            "usage": {},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_golden.py",
            "--endpoint",
            "http://control/v1/completions",
            "--model",
            "dsv4-afd",
            "--prompt-source",
            str(source_path),
            "--output",
            str(output_path),
            "--rounds",
            "3",
            "--metadata",
            "baseline_kind=mooncake_pd_no_afd",
        ],
    )

    generate_golden.main()

    report = json.loads(output_path.read_text())
    assert report["passed"] is True
    assert report["metadata"] == {"baseline_kind": "mooncake_pd_no_afd"}
    assert report["reference_comparison"] == {
        "exact_match_count": 0,
        "request_count": 3,
    }
    assert report["golden"]["0"]["stable_across_rounds"] is True
