from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.dsv4 import run_pd_performance

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "tools/dsv4/mooncake_pd_manual/pd_graph_matrix.sh"
GUIDE = (
    ROOT / "docs/npu/DEEPSEEK_V4_AFD_P8_A16F8_DUAL_A3_GRAPH_U2_VALIDATION_GUIDE_ZH.md"
)


def _source_config(path: Path) -> dict[str, str]:
    command = [
        "bash",
        "-c",
        'set -a; source "$1"; env -0',
        "bash",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    return {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in result.stdout.decode().split("\0")
        if "=" in item
    }


def test_matrix_generates_split_a16f8_contract(tmp_path):
    subprocess.run(["bash", str(MATRIX), "init", str(tmp_path)], check=True)

    generated = sorted(tmp_path.glob("*.env"))
    assert len(generated) == 40  # common.env plus thirteen points x three roles
    repo_head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert _source_config(tmp_path / "common.env")["AFD_PD_COMMIT"] == repo_head

    for role in ("prefill_ffn", "attention", "proxy"):
        config = _source_config(tmp_path / f"afd_graph_u2_split_a16f8-{role}.env")
        assert config["NODE_ROLE"] == role
        assert config["AFD_PLACEMENT"] == "split"
        assert config["DECODE_DP_SIZE"] == "16"
        assert config["DECODE_TP_SIZE"] == "1"
        assert config["ATTENTION_RANKS"] == "16"
        assert config["FFN_RANKS"] == "8"
        assert config["ATTENTION_DEVICES"] == ",".join(map(str, range(16)))
        assert config["FFN_DEVICES"] == ",".join(map(str, range(8, 16)))
        assert config["FFN_MAX_NUM_BATCHED_TOKENS"] == "8192"
        assert config["DECODE_CUDAGRAPH_CAPTURE_SIZES"] == "1 2 4 8 16"
        assert config["MATRIX_PROFILE_BASE"].startswith("/tmp/dsv4-pd-profile/")
        assert config["AFD_PROFILE_ATTENTION_DIR"].endswith(
            "/afd_graph_u2_split_a16f8/attention"
        )
        assert config["AFD_PROFILE_FFN_DIR"].endswith("/afd_graph_u2_split_a16f8/ffn")
        for flag in (
            "AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP",
            "AFD_HCCL_GRAPH_U2_HYBRID_DAG",
            "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM",
            "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM",
            "AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER",
        ):
            assert config[flag] == "1"


def test_matrix_init_accepts_site_common_template(tmp_path):
    template = tmp_path / "site.env"
    config_dir = tmp_path / "matrix"
    template.write_text(
        (MATRIX.parent / "config.env.example")
        .read_text()
        .replace(
            'AFD_PLUGIN_ROOT="${CODE_ROOT}/afd-plugin"',
            'AFD_PLUGIN_ROOT="${CODE_ROOT}/afd-plugin-phase1-a5"',
        )
        .replace(
            'NATIVE_GOLDEN_PATH="/data/z00569729/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json"',
            'NATIVE_GOLDEN_PATH="/data/z00569729/validation/dsv4-phase1-a5-native-controls/eager_mtp_off/golden_results.json"',
        )
    )
    env = os.environ.copy()
    env["PD_GRAPH_MATRIX_COMMON_TEMPLATE"] = str(template)

    subprocess.run(
        ["bash", str(MATRIX), "init", str(config_dir)], env=env, check=True
    )

    config = _source_config(config_dir / "common.env")
    assert config["AFD_PLUGIN_ROOT"].endswith("/afd-plugin-phase1-a5")
    assert config["NATIVE_GOLDEN_PATH"].endswith(
        "/dsv4-phase1-a5-native-controls/eager_mtp_off/golden_results.json"
    )
    assert re.fullmatch(r"[0-9a-f]{40}", config["AFD_PD_COMMIT"])


@pytest.mark.parametrize(
    (
        "point",
        "control_point",
        "attention_ranks",
        "ffn_ranks",
        "attention_devices",
        "ffn_devices",
        "ffn_capacity",
        "tokens",
        "draft",
    ),
    [
        (
            "afd_graph_u2_mtp2",
            "control_graph_u2_mtp2_a8",
            8,
            8,
            range(8),
            range(8, 16),
            4096,
            2,
            "eager",
        ),
        (
            "afd_graph_u2_mtp3",
            "control_graph_u2_mtp3_a8",
            8,
            8,
            range(8),
            range(8, 16),
            4096,
            3,
            "graph",
        ),
        (
            "afd_graph_u2_split_a4f8_mtp3",
            "control_graph_u2_mtp3_a4",
            4,
            8,
            range(4),
            range(8, 16),
            2048,
            3,
            "graph",
        ),
        (
            "afd_graph_u2_split_a8f4_mtp3",
            "control_graph_u2_mtp3_a8",
            8,
            4,
            range(8),
            range(8, 12),
            8192,
            3,
            "graph",
        ),
    ],
)
def test_phase1_points_match_control_and_topology_contract(
    tmp_path,
    point,
    control_point,
    attention_ranks,
    ffn_ranks,
    attention_devices,
    ffn_devices,
    ffn_capacity,
    tokens,
    draft,
):
    subprocess.run(["bash", str(MATRIX), "init", str(tmp_path)], check=True)
    point_role = "attention" if "split" in point else "decode"
    config = _source_config(tmp_path / f"{point}-{point_role}.env")
    control = _source_config(tmp_path / f"{control_point}-decode.env")

    assert config["ATTENTION_RANKS"] == str(attention_ranks)
    assert config["FFN_RANKS"] == str(ffn_ranks)
    assert config["ATTENTION_DEVICES"] == ",".join(map(str, attention_devices))
    assert config["FFN_DEVICES"] == ",".join(map(str, ffn_devices))
    assert config["FFN_MAX_NUM_BATCHED_TOKENS"] == str(ffn_capacity)
    assert config["DECODE_MTP_NUM_SPECULATIVE_TOKENS"] == str(tokens)
    assert config["DECODE_MTP_DRAFT_EXECUTION"] == draft
    assert config["PD_CONTROL_GOLDEN_PATH"] == control["PD_CONTROL_GOLDEN_PATH"]
    assert config["DECODE_DP_SIZE"] == control["DECODE_DP_SIZE"]
    assert config["DECODE_EXECUTION_MODE"] == control["DECODE_EXECUTION_MODE"]
    assert config["DECODE_U_BATCHES"] == control["DECODE_U_BATCHES"]
    assert config["DECODE_ENABLE_MTP"] == control["DECODE_ENABLE_MTP"]
    assert config["DECODE_MTP_NUM_SPECULATIVE_TOKENS"] == control[
        "DECODE_MTP_NUM_SPECULATIVE_TOKENS"
    ]
    assert config["DECODE_MTP_DRAFT_EXECUTION"] == control[
        "DECODE_MTP_DRAFT_EXECUTION"
    ]


def test_a5_matrix_lists_the_blocking_phase1_cases():
    matrix = ROOT / "tools/dsv4/run_phase1_a5_matrix.sh"
    script = matrix.read_text(encoding="utf-8")
    output = subprocess.check_output(["bash", str(matrix), "list"], text=True)

    assert 'export PYTHONPATH="${REPO_ROOT}:${DSV4_VLLM_ROOT}' in script
    assert "imported_roots" in script
    assert "libcust_opapi.so" in script
    assert "PHASE1_GOLDEN_ROOT" in script
    assert "PHASE1_GOLDEN=" not in script
    assert "control_key_for_case" in script
    assert "graph_target_draft_graph_mtp_n3" in script
    assert output.splitlines() == [
        "a8f8_eager_u1_mtp_off",
        "a8f8_eager_u1_n1",
        "a8f8_eager_u2_n2",
        "a8f8_graph_u1_n2",
        "a8f8_graph_u2_n3",
        "a4f8_eager_u1_n2",
        "a4f8_graph_u2_n3",
        "a8f4_eager_u1_n2",
        "a8f4_graph_u2_n3",
    ]


def test_phase1_native_control_generator_lists_path_matched_controls():
    runner = ROOT / "tools/dsv4/run_phase1_native_controls.sh"
    script = runner.read_text(encoding="utf-8")
    output = subprocess.check_output(["bash", str(runner), "list"], text=True)

    assert output.splitlines() == [
        "eager_mtp_off",
        "eager_mtp_n1",
        "eager_mtp_n2",
        "graph_target_draft_eager_mtp_n2",
        "graph_target_draft_graph_mtp_n3",
    ]
    assert "run_v023_native_baseline.sh" in script
    assert "baseline_kind=native_path_control" in script
    assert "--rounds 3" in script
    assert "preflight-native" in script


def test_phase1_collector_caps_logs_and_excludes_profiler_data(tmp_path):
    collector = ROOT / "tools/dsv4/collect_phase1_validation.sh"
    run_root = tmp_path / "run"
    (run_root / "profiles/raw").mkdir(parents=True)
    (run_root / "cycle_1").mkdir(parents=True)
    (run_root / "validation_summary.json").write_text('{"passed": true}\n')
    (run_root / "cycle_1/attention.log").write_text("x" * 1024)
    (run_root / "profiles/raw/large.bin").write_bytes(b"p" * 1024)
    output = tmp_path / "evidence.tar.gz"

    subprocess.run(
        ["bash", str(collector), str(output), str(run_root)],
        check=True,
        env=os.environ | {"PHASE1_EVIDENCE_LOG_TAIL_BYTES": "128"},
    )

    listing = subprocess.check_output(["tar", "-tzf", str(output)], text=True)
    assert "validation_summary.json" in listing
    assert "attention.log.tail" in listing
    assert "profiles/raw/large.bin" not in listing
    assert output.with_suffix(output.suffix + ".sha256").is_file()


def _write_stage_log(path: Path, ranks: range, stage_count: int = 2) -> None:
    records = [
        "AFD NPU Attention send_dp_metadata decision; "
        f"world_rank={rank} stage_count={stage_count} "
        "is_graph_capturing=False is_warmup=False"
        for rank in ranks
    ]
    path.write_text("\n".join(records) + "\n", encoding="utf-8")


def test_evidence_ignores_stale_fatal_logs(tmp_path):
    config_dir = tmp_path / "config"
    run_base = tmp_path / "run"
    subprocess.run(["bash", str(MATRIX), "init", str(config_dir)], check=True)
    log_dir = run_base / "afd_graph_u2/logs/decode"
    log_dir.mkdir(parents=True)
    _write_stage_log(log_dir / "attention.log", range(8))
    (log_dir / "ffn.log").write_text("current run is healthy\n", encoding="utf-8")
    (log_dir / "attention-previous.log").write_text(
        "Traceback from an earlier run\n", encoding="utf-8"
    )
    env = os.environ | {"MATRIX_RUN_BASE": str(run_base)}

    result = subprocess.run(
        ["bash", str(MATRIX), "evidence", str(config_dir), "afd_graph_u2", "decode"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    evidence = run_base / "afd_graph_u2/state/decode/stage-evidence.env"
    assert "observed_attention_ranks=8" in evidence.read_text(encoding="utf-8")


def test_evidence_rejects_current_fatal_log(tmp_path):
    config_dir = tmp_path / "config"
    run_base = tmp_path / "run"
    subprocess.run(["bash", str(MATRIX), "init", str(config_dir)], check=True)
    log_dir = run_base / "afd_graph_u2/logs/decode"
    log_dir.mkdir(parents=True)
    _write_stage_log(log_dir / "attention.log", range(8))
    (log_dir / "ffn.log").write_text(
        "AFD NPU FFN worker loop failed\n", encoding="utf-8"
    )
    env = os.environ | {"MATRIX_RUN_BASE": str(run_base)}

    result = subprocess.run(
        ["bash", str(MATRIX), "evidence", str(config_dir), "afd_graph_u2", "decode"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Fatal marker found while collecting stage evidence" in result.stderr


def test_evidence_without_online_u2_points_to_concurrent_workload(tmp_path):
    config_dir = tmp_path / "config"
    run_base = tmp_path / "run"
    subprocess.run(["bash", str(MATRIX), "init", str(config_dir)], check=True)
    log_dir = run_base / "afd_graph_u2_split_a16f8/logs/attention"
    log_dir.mkdir(parents=True)
    _write_stage_log(log_dir / "attention.log", range(8, 24), stage_count=1)
    env = os.environ | {"MATRIX_RUN_BASE": str(run_base)}

    result = subprocess.run(
        [
            "bash",
            str(MATRIX),
            "evidence",
            str(config_dir),
            "afd_graph_u2_split_a16f8",
            "attention",
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "reached 0/16 Attention ranks" in result.stderr
    assert "after the concurrent benchmark/profile workload" in result.stderr


def test_profile_summary_uses_artifacts_when_enable_markers_are_missing(tmp_path):
    config_dir = tmp_path / "config"
    run_base = tmp_path / "run"
    script_dir = tmp_path / "scripts/mooncake_pd_manual"
    script_dir.mkdir(parents=True)
    matrix = script_dir / "pd_graph_matrix.sh"
    shutil.copy2(MATRIX, matrix)
    summarizer = script_dir.parent / "summarize_pd_profiles.py"
    summarizer.write_text(
        """\
import json
import sys
from pathlib import Path

output = Path(sys.argv[sys.argv.index("--output") + 1])
output.write_text(json.dumps({"passed": True}) + "\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )

    subprocess.run(["bash", str(MATRIX), "init", str(config_dir)], check=True)
    venv_root = tmp_path / "venv"
    (venv_root / "bin").mkdir(parents=True)
    (venv_root / "bin/python").symlink_to(sys.executable)
    common = config_dir / "common.env"
    common.write_text(
        common.read_text(encoding="utf-8")
        .replace('AFD_PROFILE_ENABLE="0"', 'AFD_PROFILE_ENABLE="1"')
        .replace(
            'VENV_ROOT="${CODE_ROOT}/.venvs/afd-v023-vllm-cann"',
            f'VENV_ROOT="{venv_root}"',
        ),
        encoding="utf-8",
    )
    state_dir = run_base / "afd_graph_u2/state/decode"
    state_dir.mkdir(parents=True)
    (state_dir / "profile-session.env").write_text(
        "started_epoch=1800000000\ncann_version=9.0.0\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(matrix),
            "profile-summary",
            str(config_dir),
            "afd_graph_u2",
            "decode",
        ],
        env=os.environ | {"MATRIX_RUN_BASE": str(run_base)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "attention=0" in result.stderr
    assert "ffn=0" in result.stderr
    assert "authoritative current-session artifact validation" in result.stderr
    assert (state_dir / "profile-summary.json").is_file()


def test_collect_final_preserves_failed_gates_and_still_collects(tmp_path):
    config_dir = tmp_path / "config"
    run_base = tmp_path / "run"
    script_dir = tmp_path / "scripts/mooncake_pd_manual"
    script_dir.mkdir(parents=True)
    matrix = script_dir / "pd_graph_matrix.sh"
    shutil.copy2(MATRIX, matrix)
    (script_dir / "pd.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        '[[ "$1" == "collect" ]]\n'
        'printf "collection invoked\\n"\n',
        encoding="utf-8",
    )

    subprocess.run(["bash", str(MATRIX), "init", str(config_dir)], check=True)
    venv_root = tmp_path / "venv"
    (venv_root / "bin").mkdir(parents=True)
    (venv_root / "bin/python").symlink_to(sys.executable)
    common = config_dir / "common.env"
    common.write_text(
        common.read_text(encoding="utf-8").replace(
            'VENV_ROOT="${CODE_ROOT}/.venvs/afd-v023-vllm-cann"',
            f'VENV_ROOT="{venv_root}"',
        ),
        encoding="utf-8",
    )

    point_root = run_base / "afd_graph_u2_split_a8f8"
    log_dir = point_root / "logs/attention"
    state_dir = point_root / "state/attention"
    monitor_dir = point_root / "monitor/attention-test"
    log_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    monitor_dir.mkdir(parents=True)
    (log_dir / "attention.log").write_text(
        "Traceback retained for offline analysis\n", encoding="utf-8"
    )
    (state_dir / "last-monitor-dir").write_text(f"{monitor_dir}\n", encoding="utf-8")
    (monitor_dir / "summary.json").write_text('{"passed": false}\n', encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            str(matrix),
            "collect-final",
            str(config_dir),
            "afd_graph_u2_split_a8f8",
            "attention",
        ],
        env=os.environ | {"MATRIX_RUN_BASE": str(run_base)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "collection invoked" in result.stdout
    assert "Fatal markers found" in result.stderr
    assert "preserving them in the collection artifact" in result.stderr
    assert "missing AFD stage evidence" in result.stderr
    assert "NPU monitor summary is missing or not passed" in result.stderr
    gate_result = (state_dir / "collect-final-gates.txt").read_text(encoding="utf-8")
    assert "status=failed" in gate_result
    assert "failure=fatal markers found in attention logs" in gate_result


def test_performance_http_checks_use_no_proxy_opener(monkeypatch):
    calls: list[tuple[str, float]] = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b""

    def open_url(url, *, timeout):
        calls.append((url, timeout))
        return Response()

    monkeypatch.setattr(run_pd_performance._NO_PROXY_OPENER, "open", open_url)

    run_pd_performance._healthcheck("http://127.0.0.1/healthcheck", 3.0)
    assert (
        run_pd_performance._fetch_spec_metrics("http://127.0.0.1/metrics", 4.0) is None
    )

    assert calls == [
        ("http://127.0.0.1/healthcheck", 3.0),
        ("http://127.0.0.1/metrics", 4.0),
    ]


def test_step_by_step_guide_uses_generated_point_role_contracts(tmp_path):
    subprocess.run(["bash", str(MATRIX), "init", str(tmp_path)], check=True)
    guide = GUIDE.read_text(encoding="utf-8")
    assert "$POINT" not in guide
    command_pattern = re.compile(
        r'bash "\$MATRIX" '
        r"(check|start|status|smoke|evidence|monitor-start|monitor-stop|"
        r"stop|collect|collect-final|profile-start|profile-check|profile-stop|"
        r"profile-finalize|profile-analyse|profile-summary) "
        r'"\$CFG" ([a-z0-9_]+) ([a-z0-9_]+)'
    )
    commands = command_pattern.findall(guide)
    assert commands
    for _action, point, role in commands:
        assert (tmp_path / f"{point}-{role}.env").is_file(), (point, role)

    benchmark_pattern = re.compile(
        r'bash "\$MATRIX" benchmark "\$CFG" ([a-z0-9_]+) (p2|profile)'
    )
    benchmarks = benchmark_pattern.findall(guide)
    assert len(benchmarks) == 6
    for point, _phase in benchmarks:
        assert (tmp_path / f"{point}-proxy.env").is_file(), point

    for point, role in (
        ("afd_graph_u2", "decode"),
        ("afd_graph_u2_split_a8f8", "attention"),
        ("afd_graph_u2_split_a16f8", "attention"),
    ):
        benchmark = f'bash "$MATRIX" benchmark "$CFG" {point} p2'
        evidence = f'bash "$MATRIX" evidence "$CFG" {point} {role}'
        assert guide.index(benchmark) < guide.index(evidence)


def _summary(point: str, *, output_throughput: float, active_npus: int) -> dict:
    metrics = {
        metric: {"mean": output_throughput, "cv": 0.01}
        for metric in run_pd_performance.METRICS
    }
    return {
        "point": point,
        "phase": "p2",
        "passed": True,
        "measurement_passed": True,
        "execution": run_pd_performance.EXPECTED_EXECUTION[point],
        "workload": {"repeats": 3, "input_len": 1024, "output_len": 128},
        "stack": {
            "cann_version": "9.0.0",
            "vllm_commit": "vllm",
            "vllm_ascend_commit": "vllm-ascend",
            "afd_commit": "afd",
            "model_path": "/model",
        },
        "resource": {
            "active_npus": active_npus,
            "total_npus": active_npus,
            "reserved_npus": 32,
            "server_count": 2,
        },
        "aggregate": {
            "metrics": metrics,
            "output_tokens_per_second_per_npu": output_throughput / active_npus,
        },
    }


def test_ratio_comparison_separates_placement_and_attention_scale():
    summaries = [
        _summary("afd_graph_u2", output_throughput=100.0, active_npus=24),
        _summary("afd_graph_u2_split_a8f8", output_throughput=90.0, active_npus=24),
        _summary("afd_graph_u2_split_a16f8", output_throughput=120.0, active_npus=32),
    ]

    result = run_pd_performance._build_ratio_comparison(summaries)

    comparisons = result["comparisons"]
    assert result["passed"] is True
    assert comparisons["placement_penalty_split_a8f8_vs_colocated_a8f8"][
        "metric_change_pct"
    ]["output_throughput"] == pytest.approx(-10.0)
    assert comparisons["ratio_gain_a16f8_vs_split_a8f8"]["metric_change_pct"][
        "output_throughput"
    ] == pytest.approx(100 * (120 / 90 - 1))
    assert comparisons["end_to_end_a16f8_vs_colocated_a8f8"]["metric_change_pct"][
        "output_throughput"
    ] == pytest.approx(20.0)
