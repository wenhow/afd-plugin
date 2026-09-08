from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMP_RECIPE_DIR = REPO_ROOT / "recipe/npu/CAMP2pAFDConnector/deepseek_v4"
HCCL_RECIPE_DIR = REPO_ROOT / "recipe/npu/P2pHcclAFDConnector/deepseek_v4"
COMMON_RECIPE_DIR = REPO_ROOT / "recipe/npu/deepseek_v4/common"
RUNNER_PATH = COMMON_RECIPE_DIR / "run_validation.py"
VALIDATOR_PATH = COMMON_RECIPE_DIR / "validate_golden.py"
PERFORMANCE_RUNNER_PATH = HCCL_RECIPE_DIR / "run_performance.py"
NATIVE_PERFORMANCE_RUNNER_PATH = HCCL_RECIPE_DIR / "run_native_performance.py"
MTP_AUDIT_PATH = REPO_ROOT / "tools/dsv4/audit_mtp_contract.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "dsv4_validate_golden",
        VALIDATOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runner():
    spec = importlib.util.spec_from_file_location("dsv4_run_validation", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_performance_runner():
    spec = importlib.util.spec_from_file_location(
        "dsv4_run_performance",
        PERFORMANCE_RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_native_performance_runner():
    spec = importlib.util.spec_from_file_location(
        "dsv4_run_native_performance",
        NATIVE_PERFORMANCE_RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_mtp_audit():
    spec = importlib.util.spec_from_file_location("dsv4_mtp_audit", MTP_AUDIT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_gate_checks_structure_without_requiring_single_request_tokens():
    validator = _load_validator()
    expected = {"prompt_token_ids": [1, 2], "token_ids": [3, 4]}
    result = {"prompt_token_ids": [1, 2], "token_ids": [9, 10]}

    assert validator._batch_result_valid(result, expected)


def test_batch_gate_rejects_bad_prompt_ids_or_completion_shape():
    validator = _load_validator()
    expected = {"prompt_token_ids": [1, 2], "token_ids": [3, 4]}

    assert not validator._batch_result_valid(
        {"prompt_token_ids": [1], "token_ids": [9, 10]}, expected
    )
    assert not validator._batch_result_valid(
        {"prompt_token_ids": [1, 2], "token_ids": [9]}, expected
    )


def test_dsv4_validation_defaults_to_pinned_v023_native_golden():
    runner = _load_runner()

    assert (
        Path(
            "/mnt/workspace/validation/dsv4_v023_vllm_cann_native_baseline/"
            "golden_results.json"
        )
        == runner.DEFAULT_GOLDEN
    )


def test_dsv4_camp_role_scripts_are_connector_isolated():
    for role in ("attention", "ffn"):
        script = (CAMP_RECIPE_DIR / f"afd_{role}.sh").read_text(encoding="utf-8")
        assert "recipe/npu/deepseek_v4/common/activate_role_runtime.sh" in script
        assert '"connector":"CAMP2pAFDConnector"' in script
        assert "P2pHcclAFDConnector" not in script
        assert "ENABLE_MTP" not in script
        assert "ENABLE_PD" not in script
        assert "MOONCAKE" not in script
        assert 'EXECUTION_MODE="${EXECUTION_MODE:-eager}"' in script
        assert 'U_BATCHES="${U_BATCHES:-1}"' in script
        assert 'TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"' in script
        assert 'MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"' in script
        assert 'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"' in script
        assert '"cudagraph_mode":"FULL_DECODE_ONLY"' in script
        assert '--cudagraph-capture-sizes "${CAPTURE_SIZE_ARGS[@]}"' in script
        assert "--enable-dbo" in script
        assert '--dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"' in script
        assert '--dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD"' in script
        assert '"${UBATCH_ARGS[@]}"' in script
        assert '--max-model-len "$MAX_MODEL_LEN"' in script
        assert '--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"' in script
        assert 'AFD_ASYNC_SCHEDULING="${AFD_ASYNC_SCHEDULING:-auto}"' in script
        assert "SCHEDULING_ARGS=(--async-scheduling)" in script
        assert "SCHEDULING_ARGS=(--no-async-scheduling)" in script
        assert '"${SCHEDULING_ARGS[@]}"' in script
        assert "CAMP2p U2 supports only eager execution" in script
        assert "--tensor-parallel-size 1" in script
        assert "--all2all-backend flashinfer_all2allv" in script

    attention_script = (CAMP_RECIPE_DIR / "afd_attention.sh").read_text(
        encoding="utf-8"
    )
    assert 'HCCL_IF_BASE_PORT="${ATTENTION_HCCL_IF_BASE_PORT:-51000}"' in (
        attention_script
    )
    assert "ATTENTION_MAX_NUM_BATCHED_TOKENS" in attention_script
    assert "ATTENTION_DEVICES" in attention_script

    ffn_script = (CAMP_RECIPE_DIR / "afd_ffn.sh").read_text(encoding="utf-8")
    assert 'HCCL_IF_BASE_PORT="${FFN_HCCL_IF_BASE_PORT:-52000}"' in ffn_script
    assert "FFN_MAX_NUM_BATCHED_TOKENS" in ffn_script
    assert "FFN_DEVICES" in ffn_script
    assert "trap forward_shutdown TERM INT" in ffn_script
    assert "if ((shutdown_requested)); then" in ffn_script


def test_dsv4_hccl_recipe_owns_connector_specific_launchers():
    for role in ("attention", "ffn"):
        script = (HCCL_RECIPE_DIR / f"afd_{role}.sh").read_text(encoding="utf-8")
        assert "readonly AFD_CONNECTOR=P2pHcclAFDConnector" in script
        assert "recipe/npu/deepseek_v4/common/activate_role_runtime.sh" in script
        assert "CAMP2pAFDConnector" not in script
        assert 'ENABLE_MTP="${ENABLE_MTP:-0}"' in script
        assert 'MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-1}"' in (
            script
        )
        assert 'MTP_DRAFT_EXECUTION="${MTP_DRAFT_EXECUTION:-eager}"' in script
        assert '"method":"mtp"' in script
        assert '"num_speculative_tokens":%s' in script
        assert '"${MTP_ARGS[@]}"' in script
        assert '--tensor-parallel-size "$TENSOR_PARALLEL_SIZE"' in script

    runner = (HCCL_RECIPE_DIR / "run_validation.py").read_text(encoding="utf-8")
    assert 'sys.argv.extend(["--connector", "P2pHcclAFDConnector"])' in runner
    assert "runpy.run_path" in runner
    assert "recipe/npu/deepseek_v4/common/run_validation.py" in runner
    assert "CAMP2pAFDConnector/deepseek_v4" not in runner

    performance_runner = PERFORMANCE_RUNNER_PATH.read_text(encoding="utf-8")
    assert "P2pHcclAFDConnector" in performance_runner
    assert "recipe/npu/deepseek_v4/common/run_validation.py" in performance_runner
    assert "CAMP2pAFDConnector/deepseek_v4" not in performance_runner
    assert '"bench",' in performance_runner
    assert '"serve",' in performance_runner

    attention = (HCCL_RECIPE_DIR / "afd_attention.sh").read_text(encoding="utf-8")
    assert 'ENABLE_PD="${ENABLE_PD:-0}"' in attention
    assert "Mooncake PD M9 baseline" in attention


def test_dsv4_shared_recipe_is_connector_neutral():
    common_runner = RUNNER_PATH.read_text(encoding="utf-8")
    role_runtime = (COMMON_RECIPE_DIR / "activate_role_runtime.sh").read_text(
        encoding="utf-8"
    )
    assert "CONNECTOR_RECIPE_DIRS" in common_runner
    assert "COMMON_RECIPE_DIR" in common_runner
    assert (COMMON_RECIPE_DIR / "validate_golden.py").is_file()
    assert (COMMON_RECIPE_DIR / "activate_role_runtime.sh").is_file()
    assert 'export PYTHONPATH="${ROOT_DIR}:${DSV4_VLLM_ROOT}' in role_runtime

    camp_runner = (CAMP_RECIPE_DIR / "run_validation.py").read_text(encoding="utf-8")
    assert 'sys.argv.extend(["--connector", "CAMP2pAFDConnector"])' in camp_runner
    assert "recipe/npu/deepseek_v4/common/run_validation.py" in camp_runner
    assert "P2pHcclAFDConnector" not in camp_runner


def test_dsv4_v023_native_baseline_has_explicit_mtp_switch():
    script = (REPO_ROOT / "tools/dsv4/run_v023_native_baseline.sh").read_text(
        encoding="utf-8"
    )

    assert 'ENABLE_MTP="${ENABLE_MTP:-0}"' in script
    assert 'MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-1}"' in script
    assert 'MTP_DRAFT_EXECUTION="${MTP_DRAFT_EXECUTION:-eager}"' in script
    assert 'EXECUTION_MODE="${EXECUTION_MODE:-eager}"' in script
    assert '\\"method\\":\\"mtp\\"' in script
    assert '\\"num_speculative_tokens\\":${MTP_NUM_SPECULATIVE_TOKENS}' in script
    assert '\\"enforce_eager\\":${mtp_draft_enforce_eager}' in script
    assert "--speculative-config" in script
    assert '"cudagraph_mode":"FULL_DECODE_ONLY"' in script
    assert '"${EXECUTION_ARGS[@]}"' in script
    assert (
        "VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector"
        in script
    )
    assert "ascend_kv_connector,afd" not in script
    assert 'TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"' in script
    assert '--data-parallel-size "${DATA_PARALLEL_SIZE}"' in script
    assert '--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"' in script


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("mtp.0.ffn.experts.0.w1.weight", "ffn"),
        ("mtp.0.ffn.experts.0.w1.weight_scale", "ffn"),
        ("mtp.0.ffn.experts.0.w1.weight_offset", "ffn"),
        ("mtp.0.ffn.gate.weight", "ffn"),
        ("mtp.0.ffn_norm.weight", "attention"),
        ("mtp.0.hc_ffn_fn", "attention"),
        ("mtp.0.attn.wq_a.weight", "attention"),
        ("mtp.0.emb.tok_emb.weight", "attention"),
        ("model.mtp.0.head.weight", "attention"),
    ],
)
def test_dsv4_mtp_contract_classifies_raw_checkpoint_keys(name, role):
    assert _load_mtp_audit().classify_mtp_key(name) == role


def test_dsv4_mtp_contract_audits_quantized_tensor_families(tmp_path):
    audit = _load_mtp_audit()
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.0.ffn.experts.0.w1.weight": "part-1.safetensors",
                    "mtp.0.ffn.experts.0.w1.weight_scale": "part-1.safetensors",
                    "mtp.0.ffn.experts.0.w1.weight_offset": "part-1.safetensors",
                    "mtp.0.attn.wq_a.weight": "part-2.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    report = audit.build_report(index_path)

    assert report["passed"]
    assert report["mtp"]["role_counts"] == {"attention": 1, "ffn": 3}
    assert report["contract"]["weight_scale_offset_same_role"]


def test_dsv4_hccl_a8f4_topology_derives_ffn_capacity_and_unused_devices():
    runner = _load_runner()

    topology = runner._resolve_topology(
        connector="P2pHcclAFDConnector",
        attention_devices=list(range(8)),
        ffn_devices=list(range(8, 12)),
        attention_max_num_batched_tokens=1024,
        ffn_max_num_batched_tokens=None,
    )

    assert topology == {
        "attention_ranks": 8,
        "ffn_ranks": 4,
        "tensor_parallel_size": 1,
        "attention_data_parallel_size": 8,
        "ffn_data_parallel_size": 4,
        "ratio": 2,
        "direction": "fan_in",
        "attention_devices": list(range(8)),
        "ffn_devices": list(range(8, 12)),
        "unused_devices": [12, 13, 14, 15],
        "attention_max_num_batched_tokens": 1024,
        "ffn_max_num_batched_tokens": 2048,
    }


def test_dsv4_hccl_a4f8_topology_derives_fanout_capacity():
    runner = _load_runner()

    topology = runner._resolve_topology(
        connector="P2pHcclAFDConnector",
        attention_devices=list(range(4)),
        ffn_devices=list(range(4, 12)),
        attention_max_num_batched_tokens=1024,
        ffn_max_num_batched_tokens=None,
    )

    assert topology["attention_ranks"] == 4
    assert topology["ffn_ranks"] == 8
    assert topology["ratio"] == 2
    assert topology["direction"] == "fan_out"
    assert topology["ffn_max_num_batched_tokens"] == 512
    assert topology["unused_devices"] == [12, 13, 14, 15]


@pytest.mark.parametrize(
    ("attention_devices", "ffn_devices", "ffn_tokens", "message"),
    [
        ([0, 1], [1], None, "must not overlap"),
        ([0, 1, 2], [8, 9], None, "integer multiple"),
        ([0, 1], [8], 1024, "cover one Attention subgroup"),
    ],
)
def test_dsv4_hccl_topology_rejects_invalid_deployment(
    attention_devices,
    ffn_devices,
    ffn_tokens,
    message,
):
    runner = _load_runner()

    with pytest.raises(ValueError, match=message):
        runner._resolve_topology(
            connector="P2pHcclAFDConnector",
            attention_devices=attention_devices,
            ffn_devices=ffn_devices,
            attention_max_num_batched_tokens=1024,
            ffn_max_num_batched_tokens=ffn_tokens,
        )


def test_dsv4_hccl_graph_topology_accepts_integer_multiple_roles():
    runner = _load_runner()

    runner._validate_execution_topology(
        connector="P2pHcclAFDConnector",
        execution_mode="full-decode-only",
        u_batches=2,
        topology={"attention_ranks": 8, "ffn_ranks": 8},
    )
    runner._validate_execution_topology(
        connector="P2pHcclAFDConnector",
        execution_mode="full-decode-only",
        topology={"attention_ranks": 8, "ffn_ranks": 4},
    )

    runner._validate_execution_topology(
        connector="P2pHcclAFDConnector",
        execution_mode="eager",
        topology={"attention_ranks": 8, "ffn_ranks": 4},
    )

    with pytest.raises(ValueError, match="graph U2 requires P2pHccl"):
        runner._validate_execution_topology(
            connector="CAMP2pAFDConnector",
            execution_mode="full-decode-only",
            u_batches=2,
            topology={"attention_ranks": 8, "ffn_ranks": 8},
        )


def test_dsv4_hccl_tp2_topology_and_environment(monkeypatch):
    runner = _load_runner()
    topology = runner._resolve_topology(
        connector="P2pHcclAFDConnector",
        attention_devices=list(range(8)),
        ffn_devices=list(range(8, 16)),
        attention_max_num_batched_tokens=1024,
        ffn_max_num_batched_tokens=None,
        tensor_parallel_size=2,
    )

    assert topology["tensor_parallel_size"] == 2
    assert topology["attention_data_parallel_size"] == 4
    assert topology["ffn_data_parallel_size"] == 4
    runner._validate_execution_topology(
        connector="P2pHcclAFDConnector",
        execution_mode="full-decode-only",
        u_batches=2,
        topology=topology,
    )
    runner._set_topology_environment(topology)
    assert runner.os.environ["TENSOR_PARALLEL_SIZE"] == "2"

    with pytest.raises(ValueError, match="requires equal Attention and FFN"):
        runner._resolve_topology(
            connector="P2pHcclAFDConnector",
            attention_devices=list(range(8)),
            ffn_devices=list(range(8, 12)),
            attention_max_num_batched_tokens=1024,
            ffn_max_num_batched_tokens=None,
            tensor_parallel_size=2,
        )


def test_dsv4_hccl_mtp_m4_topology_gate_and_environment(monkeypatch):
    runner = _load_runner()
    topology = {"attention_ranks": 8, "ffn_ranks": 8}
    monkeypatch.setenv("ENABLE_MTP", "0")
    monkeypatch.setenv("MTP_NUM_SPECULATIVE_TOKENS", "9")

    runner._validate_execution_topology(
        connector="P2pHcclAFDConnector",
        execution_mode="eager",
        u_batches=1,
        enable_mtp=True,
        mtp_num_speculative_tokens=1,
        topology=topology,
    )
    runner._validate_execution_topology(
        connector="P2pHcclAFDConnector",
        execution_mode="eager",
        u_batches=2,
        enable_mtp=True,
        mtp_num_speculative_tokens=1,
        topology={"attention_ranks": 8, "ffn_ranks": 4},
    )
    runner._validate_execution_topology(
        connector="P2pHcclAFDConnector",
        execution_mode="eager",
        u_batches=2,
        enable_mtp=True,
        mtp_num_speculative_tokens=1,
        topology=topology,
    )
    runner._validate_execution_topology(
        connector="P2pHcclAFDConnector",
        execution_mode="full-decode-only",
        u_batches=1,
        enable_mtp=True,
        mtp_num_speculative_tokens=1,
        topology=topology,
    )
    runner._validate_execution_topology(
        connector="P2pHcclAFDConnector",
        execution_mode="full-decode-only",
        u_batches=2,
        enable_mtp=True,
        mtp_num_speculative_tokens=1,
        topology=topology,
    )
    runner._set_mtp_environment(
        enable_mtp=True,
        mtp_num_speculative_tokens=1,
    )
    assert runner.os.environ["ENABLE_MTP"] == "1"
    assert runner.os.environ["MTP_NUM_SPECULATIVE_TOKENS"] == "1"

    invalid_cases = [
        ({"connector": "CAMP2pAFDConnector"}, "P2pHcclAFDConnector"),
        ({"mtp_num_speculative_tokens": 4}, "num_speculative_tokens in"),
    ]
    defaults = {
        "connector": "P2pHcclAFDConnector",
        "execution_mode": "eager",
        "u_batches": 1,
        "enable_mtp": True,
        "mtp_num_speculative_tokens": 1,
        "topology": topology,
    }
    for overrides, message in invalid_cases:
        kwargs = {**defaults, **overrides}
        with pytest.raises(ValueError, match=message):
            runner._validate_execution_topology(**kwargs)

    for speculative_tokens in (2, 3):
        runner._validate_execution_topology(
            **{**defaults, "mtp_num_speculative_tokens": speculative_tokens}
        )


def test_dsv4_performance_command_locks_workload_and_fixed_python(tmp_path):
    runner = _load_performance_runner()
    result_path = tmp_path / "result.json"

    command = runner._benchmark_command(
        api_port=8910,
        model_path=Path("/models/dsv4"),
        result_path=result_path,
        input_len=1024,
        output_len=128,
        num_prompts=32,
        concurrency=8,
        u_batches=2,
        run_kind="measurement",
        repeat=3,
    )

    assert command[:4] == [
        runner.sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
    ]
    assert command[command.index("--random-range-ratio") + 1] == "0.0"
    assert command[command.index("--temperature") + 1] == "0"
    assert command[command.index("--metric-percentiles") + 1] == "50,90,99"
    assert "--ignore-eos" in command
    assert "--save-detailed" in command
    assert "u_batches=2" in command


def test_dsv4_native_pair_splits_total_load_without_changing_budget():
    runner = _load_native_performance_runner()

    assert runner._split_load(1, 8) == [(1, 8), (0, 0)]
    assert runner._split_load(8, 32) == [(4, 16), (4, 16)]
    assert runner._split_load(32, 128) == [(16, 64), (16, 64)]
    with pytest.raises(ValueError, match="cover total concurrency"):
        runner._split_load(8, 7)


def test_dsv4_native_pair_uses_disjoint_devices_ports_and_no_afd_plugin(tmp_path):
    runner = _load_native_performance_runner()
    args = SimpleNamespace(
        model=tmp_path / "model",
        api_ports=[8920, 8921],
        dp_rpc_ports=[29350, 29450],
        master_ports=[29351, 29451],
        hccl_base_ports=[53000, 54000],
        max_model_len=4096,
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.9,
        enable_mtp=False,
        mtp_num_speculative_tokens=1,
    )

    env0 = runner._service_environment(args, 0)
    env1 = runner._service_environment(args, 1)

    assert env0["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert env1["ASCEND_RT_VISIBLE_DEVICES"] == "8,9,10,11,12,13,14,15"
    assert env0["DATA_PARALLEL_RPC_PORT"] != env1["DATA_PARALLEL_RPC_PORT"]
    assert env0["MASTER_PORT"] != env1["MASTER_PORT"]
    assert env0["HCCL_IF_BASE_PORT"] != env1["HCCL_IF_BASE_PORT"]
    assert env0["VLLM_PLUGINS"] == runner.NATIVE_PLUGINS
    assert ",afd" not in env0["VLLM_PLUGINS"]
    assert env0["TENSOR_PARALLEL_SIZE"] == "1"

    service_script = runner.NATIVE_SERVICE_SCRIPT.read_text(encoding="utf-8")
    assert '--data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT}"' in service_script
    assert '--master-port "${MASTER_PORT}"' in service_script


def test_dsv4_native_pair_supports_dp4_tp2(tmp_path):
    runner = _load_native_performance_runner()
    args = SimpleNamespace(
        model=tmp_path / "model",
        api_ports=[8920, 8921],
        dp_rpc_ports=[29350, 29450],
        master_ports=[29351, 29451],
        hccl_base_ports=[53000, 54000],
        max_model_len=4096,
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.9,
        enable_mtp=False,
        mtp_num_speculative_tokens=1,
        tensor_parallel_size=2,
        input_len=1024,
        output_len=128,
        concurrencies=[32],
        repeats=1,
        prompts_per_concurrency=4,
        min_prompts=128,
        warmup_input_len=256,
        warmup_output_len=16,
        warmup_prompts=16,
        warmup_concurrency=8,
        max_throughput_cv=0.1,
    )

    assert runner._service_environment(args, 0)["TENSOR_PARALLEL_SIZE"] == "2"
    manifest = runner._runtime_manifest(args)
    assert manifest["topology"]["instances"][0]["data_parallel_size"] == 4
    assert manifest["topology"]["instances"][0]["tensor_parallel_size"] == 2


def test_dsv4_native_pair_merges_shared_request_window_and_latencies():
    runner = _load_native_performance_runner()
    common = {
        "completed": 1,
        "failed": 0,
        "total_input_tokens": 4,
        "total_output_tokens": 2,
        "input_lens": [4],
        "output_lens": [2],
        "errors": [""],
        "duration": 2.5,
    }
    first = {
        **common,
        "ttfts": [1.0],
        "itls": [[1.0]],
        "start_times": [10.0],
    }
    second = {
        **common,
        "ttfts": [0.5],
        "itls": [[2.0]],
        "start_times": [10.5],
    }

    merged = runner._merge_results([first, second])

    assert merged["duration"] == 3.0
    assert merged["completed"] == 2
    assert merged["total_input_tokens"] == 8
    assert merged["total_output_tokens"] == 4
    assert merged["output_throughput"] == pytest.approx(4 / 3)
    assert merged["p50_ttft_ms"] == 750.0
    assert merged["p50_tpot_ms"] == 1500.0
    assert merged["p99_e2el_ms"] == pytest.approx(2495.0)


def test_dsv4_native_pair_ignores_thread_exception_only_after_shutdown(tmp_path):
    runner = _load_native_performance_runner()
    for index in range(2):
        (tmp_path / f"native{index}.log").write_text(
            "service ready\n[shutdown] stopping\nException in thread Thread-2:\n",
            encoding="utf-8",
        )

    ignored = runner._service_log_gate(tmp_path)
    assert ignored["passed"] is True
    assert ignored["roles"]["native0"]["ignored_shutdown_thread_exceptions"] == 1

    (tmp_path / "native1.log").write_text(
        "Exception in thread Thread-2:\n[shutdown] stopping\n",
        encoding="utf-8",
    )
    fatal = runner._service_log_gate(tmp_path)
    assert fatal["passed"] is False
    assert fatal["roles"]["native1"]["fatal_markers"] == ["Exception in thread"]


def test_dsv4_performance_reproducibility_files_are_hashed():
    runner = _load_performance_runner()

    for path in runner.REPRODUCIBILITY_FILES:
        digest = runner._file_sha256(path)
        assert len(digest) == 64
        assert digest == runner.hashlib.sha256(path.read_bytes()).hexdigest()


def test_dsv4_performance_defaults_to_validated_sync_scheduler(monkeypatch, tmp_path):
    runner = _load_performance_runner()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PERFORMANCE_RUNNER_PATH),
            "--output-dir",
            str(tmp_path),
            "--u-batches",
            "1",
        ],
    )

    args = runner._parse_args()

    assert args.async_scheduling == "off"
    assert args.tensor_parallel_size == 1
    assert args.graph_u2_compute_overlap == "on"
    assert args.graph_u2_hybrid_dag == "on"
    assert args.graph_u2_attention_three_stream == "on"
    assert args.graph_u2_ffn_recv_stream == "on"
    assert args.graph_u2_ffn_cross_layer == "on"


def test_dsv4_performance_records_graph_u2_overlap_mode(monkeypatch):
    runner = _load_performance_runner()
    args = SimpleNamespace(
        max_model_len=4096,
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.9,
        attention_hccl_base_port=51000,
        ffn_hccl_base_port=52000,
        async_scheduling="off",
        tensor_parallel_size=1,
        graph_u2_compute_overlap="off",
        graph_u2_hybrid_dag="off",
        graph_u2_attention_three_stream="off",
        graph_u2_ffn_recv_stream="off",
        graph_u2_ffn_cross_layer="off",
        enable_mtp=False,
        mtp_num_speculative_tokens=1,
        profile=False,
    )

    runner._set_service_environment(args)

    assert runner.os.environ["AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"] == "0"
    assert runner.os.environ["AFD_HCCL_GRAPH_U2_HYBRID_DAG"] == "0"
    assert runner.os.environ["AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM"] == "0"
    assert runner.os.environ["AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM"] == "0"
    assert runner.os.environ["AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER"] == "0"


def test_dsv4_performance_requires_ffn_recv_stream_for_cross_layer():
    runner = _load_performance_runner()
    args = SimpleNamespace(
        execution_mode="full-decode-only",
        u_batches=2,
        tensor_parallel_size=1,
        enable_mtp=False,
        mtp_num_speculative_tokens=1,
        max_num_batched_tokens=1024,
        graph_u2_ffn_recv_stream="off",
        graph_u2_ffn_cross_layer="on",
    )

    with pytest.raises(ValueError, match="requires the FFN receive stream"):
        runner._validate_execution_args(args)


def test_dsv4_performance_tp2_uses_dp4_topology_and_environment(monkeypatch):
    runner = _load_performance_runner()
    topology = runner._a8f8_topology(1024, tensor_parallel_size=2)

    assert topology["attention_ranks"] == topology["ffn_ranks"] == 8
    assert topology["tensor_parallel_size"] == 2
    assert topology["attention_data_parallel_size"] == 4
    assert topology["ffn_data_parallel_size"] == 4

    args = SimpleNamespace(
        execution_mode="eager",
        u_batches=1,
        tensor_parallel_size=2,
        enable_mtp=False,
        mtp_num_speculative_tokens=1,
        max_num_batched_tokens=1024,
        max_model_len=4096,
        max_num_seqs=8,
        gpu_memory_utilization=0.9,
        attention_hccl_base_port=51000,
        ffn_hccl_base_port=52000,
        async_scheduling="off",
        profile=False,
    )
    runner._validate_execution_args(args)
    runner._set_service_environment(args)

    assert runner.os.environ["TENSOR_PARALLEL_SIZE"] == "2"


def test_dsv4_performance_mtp_uses_functional_gate_and_environment(monkeypatch):
    runner = _load_performance_runner()
    args = SimpleNamespace(
        execution_mode="eager",
        u_batches=1,
        enable_mtp=True,
        mtp_num_speculative_tokens=1,
        max_num_batched_tokens=1024,
    )

    runner._validate_execution_args(args)
    topology = runner._a8f8_topology(args.max_num_batched_tokens)
    assert topology["attention_ranks"] == topology["ffn_ranks"] == 8
    assert topology["attention_devices"] == list(range(8))
    assert topology["ffn_devices"] == list(range(8, 16))

    args.u_batches = 2
    runner._validate_execution_args(args)

    args.execution_mode = "full-decode-only"
    runner._validate_execution_args(args)

    args.execution_mode = "eager"
    args.u_batches = 1
    for speculative_tokens in (2, 3):
        args.mtp_num_speculative_tokens = speculative_tokens
        runner._validate_execution_args(args)

    args.mtp_num_speculative_tokens = 4
    with pytest.raises(ValueError, match="num_speculative_tokens in"):
        runner._validate_execution_args(args)

    args.mtp_num_speculative_tokens = 1
    monkeypatch.setenv("ENABLE_MTP", "0")
    monkeypatch.setenv("MTP_NUM_SPECULATIVE_TOKENS", "9")
    runner._set_service_environment(
        SimpleNamespace(
            **vars(args),
            max_model_len=4096,
            max_num_seqs=8,
            gpu_memory_utilization=0.9,
            attention_hccl_base_port=51000,
            ffn_hccl_base_port=52000,
            async_scheduling="off",
            profile=False,
        )
    )
    assert runner.os.environ["ENABLE_MTP"] == "1"
    assert runner.os.environ["MTP_NUM_SPECULATIVE_TOKENS"] == "1"
    assert runner.os.environ["AFD_ASYNC_SCHEDULING"] == "off"


def test_dsv4_performance_mtp_manifest_preserves_structured_topology(monkeypatch):
    runner = _load_performance_runner()
    monkeypatch.setattr(
        runner.SHARED,
        "_runtime_manifest",
        lambda **kwargs: dict(kwargs),
    )
    monkeypatch.setattr(runner, "_file_sha256", lambda _path: "a" * 64)
    args = SimpleNamespace(
        execution_mode="eager",
        u_batches=1,
        dbo_decode_token_threshold=2,
        dbo_prefill_token_threshold=12,
        profile=False,
        enable_mtp=True,
        mtp_num_speculative_tokens=1,
        max_model_len=4096,
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.9,
        attention_hccl_base_port=51000,
        ffn_hccl_base_port=52000,
        async_scheduling="off",
        graph_u2_compute_overlap="on",
        graph_u2_hybrid_dag="off",
        graph_u2_attention_three_stream="on",
        graph_u2_ffn_recv_stream="on",
        graph_u2_ffn_cross_layer="on",
        input_len=1024,
        output_len=128,
        concurrencies=[32],
        repeats=1,
        prompts_per_concurrency=4,
        min_prompts=128,
        warmup_input_len=256,
        warmup_output_len=16,
        warmup_prompts=16,
        warmup_concurrency=8,
        max_throughput_cv=0.1,
        benchmark_timeout=1800,
    )

    manifest = runner._runtime_manifest(args)

    assert manifest["stage"] == "A3-P7M1-P1"
    assert manifest["topology_label"] == "A8F8"
    assert manifest["enable_mtp"] is True
    assert manifest["mtp_num_speculative_tokens"] == 1
    assert manifest["mtp_draft_execution"] == "eager"
    assert manifest["service"]["async_scheduling"] == "off"
    assert manifest["topology"] == runner._a8f8_topology(1024)

    args.execution_mode = "full-decode-only"
    graph_manifest = runner._runtime_manifest(args)
    assert graph_manifest["stage"] == "A3-P7M2-P1"
    assert graph_manifest["execution_mode"] == "full-decode-only"

    args.u_batches = 2
    graph_u2_manifest = runner._runtime_manifest(args)
    assert graph_u2_manifest["stage"] == "A3-P7M4-P1"
    assert graph_u2_manifest["mtp_phase_u_batches"] == 1
    assert graph_u2_manifest["service"]["graph_u2_compute_overlap"] == "on"
    assert graph_u2_manifest["service"]["graph_u2_hybrid_dag"] == "off"
    assert graph_u2_manifest["service"]["graph_u2_attention_three_stream"] == "on"
    assert graph_u2_manifest["service"]["graph_u2_ffn_recv_stream"] == "on"
    assert graph_u2_manifest["service"]["graph_u2_ffn_cross_layer"] == "on"


def test_dsv4_performance_detects_exited_service():
    runner = _load_performance_runner()
    running = SimpleNamespace(poll=lambda: None)
    exited = SimpleNamespace(poll=lambda: 7)

    assert runner._exited_services({"attention": running, "ffn": exited}) == {"ffn": 7}


def test_dsv4_performance_detects_hidden_worker_fatal_incrementally(tmp_path):
    runner = _load_performance_runner()
    attention_log = tmp_path / "attention.log"
    ffn_log = tmp_path / "ffn.log"
    attention_log.write_text("service ready\n", encoding="utf-8")
    ffn_log.write_text("worker ready\n", encoding="utf-8")
    watcher = runner._FatalLogWatcher(tmp_path)

    assert watcher.poll() == {}
    with ffn_log.open("a", encoding="utf-8") as handle:
        handle.write("AFD NPU FFN worker loop failed\n")

    assert watcher.poll() == {"ffn": ["AFD NPU FFN worker loop failed"]}
    assert watcher.poll() == {}


def test_dsv4_performance_fatal_watcher_handles_split_marker(tmp_path):
    runner = _load_performance_runner()
    ffn_log = tmp_path / "ffn.log"
    ffn_log.write_text("error code is 507", encoding="utf-8")
    watcher = runner._FatalLogWatcher(tmp_path)

    assert watcher.poll() == {}
    with ffn_log.open("a", encoding="utf-8") as handle:
        handle.write("015\n")

    assert watcher.poll() == {"ffn": ["error code is 507015"]}


def test_dsv4_performance_result_gate_requires_exact_tokens():
    runner = _load_performance_runner()
    result = {
        "completed": 2,
        "failed": 0,
        "total_input_tokens": 8,
        "total_output_tokens": 6,
        "errors": ["", ""],
    }

    passed = runner._validate_benchmark_result(
        result,
        input_len=4,
        output_len=3,
        num_prompts=2,
    )
    result["total_output_tokens"] = 5
    failed = runner._validate_benchmark_result(
        result,
        input_len=4,
        output_len=3,
        num_prompts=2,
    )

    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["checks"]["output_tokens"] is False


def test_dsv4_performance_aggregate_reports_variance_and_per_npu():
    runner = _load_performance_runner()
    metric_names = (
        "request_throughput",
        "output_throughput",
        "p50_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
        "p50_tpot_ms",
        "p90_tpot_ms",
        "p99_tpot_ms",
    )
    records = []
    for repeat, throughput in enumerate((160.0, 176.0, 144.0), start=1):
        result = {name: 1.0 for name in metric_names}
        result["output_throughput"] = throughput
        records.append(
            {
                "concurrency": 8,
                "repeat": repeat,
                "gate": {"passed": True},
                "result": result,
            }
        )

    aggregate = runner._aggregate_results(records, max_throughput_cv=0.10)
    point = aggregate["points"]["8"]

    assert aggregate["passed"] is True
    assert point["runs"] == 3
    assert point["metrics"]["output_throughput"]["mean"] == 160.0
    assert point["metrics"]["output_throughput"]["cv"] > 0
    assert point["output_tokens_per_second_per_npu"] == 10.0
    assert point["stability_gate"]["passed"] is True


def test_dsv4_performance_aggregate_rejects_unstable_throughput():
    runner = _load_performance_runner()
    metric_names = (
        "request_throughput",
        "output_throughput",
        "p50_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
        "p50_tpot_ms",
        "p90_tpot_ms",
        "p99_tpot_ms",
    )
    records = []
    for value in (10.0, 20.0, 30.0):
        result = {name: 1.0 for name in metric_names}
        result["output_throughput"] = value
        records.append(
            {
                "concurrency": 8,
                "gate": {"passed": True},
                "result": result,
            }
        )

    point = runner._aggregate_results(
        records,
        max_throughput_cv=0.10,
    )["points"]["8"]

    assert point["passed"] is False
    assert point["stability_gate"]["passed"] is False


def test_dsv4_performance_npu_snapshot_parser_uses_bus_rows():
    runner = _load_performance_runner()
    output = """\
| 0     Ascend910           | OK            | 169.9       43                |
| 0     0                   | 0000:18:00.0  | 71          0 / 0  30114 / 65536 |
| 0     Ascend910           | -             | -           43                |
| 1     1                   | 0000:19:00.0  | 22          0 / 0  28840 / 65536 |
"""

    assert runner._parse_npu_snapshot(output) == [
        {
            "device": 0,
            "aicore_percent": 71,
            "hbm_used_mb": 30114,
            "hbm_total_mb": 65536,
        },
        {
            "device": 1,
            "aicore_percent": 22,
            "hbm_used_mb": 28840,
            "hbm_total_mb": 65536,
        },
    ]


def test_dsv4_runtime_manifest_records_graph_u1(monkeypatch):
    runner = _load_runner()
    monkeypatch.setenv("DSV4_CANN_ROOT", "/opt/cann-9.0.0")
    monkeypatch.setenv("DSV4_CANN_VERSION", "9.0.0")
    monkeypatch.setenv("AFD_NPU_FFN_PROFILER_RANKS", "all")

    def check_output(command, **kwargs):
        if "status" in command:
            return " M tracked.py\n"
        if "diff" in command:
            return b"diff data"
        return "head\n"

    monkeypatch.setattr(runner.subprocess, "check_output", check_output)

    manifest = runner._runtime_manifest(
        connector="P2pHcclAFDConnector",
        execution_mode="full-decode-only",
        u_batches=1,
        dbo_decode_token_threshold=2,
        dbo_prefill_token_threshold=12,
        profile=True,
    )

    assert manifest["execution_mode"] == "full-decode-only"
    assert manifest["connector"] == "P2pHcclAFDConnector"
    assert manifest["u_batches"] == 1
    assert manifest["profile"] is True
    assert manifest["profile_role_ranks"] == [0]
    assert manifest["profile_role_rank_selection"] == {
        "attention": "0",
        "ffn": "all",
    }
    assert manifest["torch_profiler_with_stack"] is False
    assert manifest["cann"] == "/opt/cann-9.0.0"
    assert manifest["cann_version"] == "9.0.0"
    assert manifest["afd_plugin_worktree"]["tracked_dirty"] is True
    assert manifest["afd_plugin_worktree"]["tracked_status"] == [" M tracked.py"]
    assert len(manifest["afd_plugin_worktree"]["tracked_diff_sha256"]) == 64


def test_dsv4_runtime_manifest_records_eager_u2(monkeypatch):
    runner = _load_runner()

    def check_output(command, **kwargs):
        if "status" in command:
            return ""
        if "diff" in command:
            return b""
        return "head\n"

    monkeypatch.setattr(runner.subprocess, "check_output", check_output)

    manifest = runner._runtime_manifest(
        connector="P2pHcclAFDConnector",
        execution_mode="eager",
        u_batches=2,
        dbo_decode_token_threshold=2,
        dbo_prefill_token_threshold=12,
        profile=False,
    )

    assert manifest["execution_mode"] == "eager"
    assert manifest["connector"] == "P2pHcclAFDConnector"
    assert manifest["u_batches"] == 2
    assert manifest["dbo_decode_token_threshold"] == 2
    assert manifest["dbo_prefill_token_threshold"] == 12
    assert manifest["afd_plugin_worktree"]["tracked_dirty"] is False


def test_dsv4_shutdown_gate_requires_both_roles_to_exit_cleanly(monkeypatch):
    runner = _load_runner()
    stop_calls = []

    class FakeProcess:
        def __init__(self, returncode):
            self.returncode = returncode
            self.pid = 1

        def poll(self):
            return self.returncode

    monkeypatch.setattr(
        runner,
        "_stop_process",
        lambda process, **kwargs: stop_calls.append((process, kwargs)),
    )

    clean = runner._shutdown_roles({"attention": FakeProcess(0), "ffn": FakeProcess(0)})
    failed = runner._shutdown_roles(
        {"attention": FakeProcess(0), "ffn": FakeProcess(1)}
    )

    assert clean["passed"] is True
    assert failed["passed"] is False
    assert [kwargs for _process, kwargs in stop_calls] == [
        {},
        {"signal_group": False},
        {},
        {"signal_group": False},
    ]


def test_dsv4_signal_group_reaps_descendants_after_leader_exit(monkeypatch):
    runner = _load_runner()
    signals = []

    class ExitedProcess:
        pid = 1234

        @staticmethod
        def poll():
            return 0

    monkeypatch.setattr(
        runner.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    runner._signal_group(ExitedProcess(), runner.signal.SIGKILL)

    assert signals == [(1234, runner.signal.SIGKILL)]


def test_dsv4_stop_process_drains_owned_group_after_clean_exit(monkeypatch):
    runner = _load_runner()
    signals = []

    class RunningProcess:
        pid = 1234
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 0
            return self.returncode

    monkeypatch.setattr(
        runner.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    runner._stop_process(RunningProcess())

    assert signals == [
        (1234, runner.signal.SIGTERM),
        (1234, runner.signal.SIGKILL),
    ]


@pytest.mark.parametrize(
    "fatal_marker",
    ["AFD NPU FFN worker loop failed", "Exception in thread"],
)
def test_dsv4_log_gate_rejects_hidden_worker_fatal(tmp_path, fatal_marker):
    runner = _load_runner()
    (tmp_path / "attention.log").write_text("clean shutdown\n", encoding="utf-8")
    (tmp_path / "ffn.log").write_text(
        f"{fatal_marker}\n",
        encoding="utf-8",
    )

    result = runner._role_log_gate(tmp_path)

    assert result["passed"] is False
    assert result["roles"]["attention"]["passed"] is True
    assert result["roles"]["ffn"]["fatal_markers"] == [fatal_marker]


def test_dsv4_log_gate_ignores_tbe_queue_eof_after_shutdown(tmp_path):
    runner = _load_runner()
    (tmp_path / "attention.log").write_text(
        """INFO [shutdown] API server: shutdown triggered
Exception in thread Thread-1:
Traceback (most recent call last):
  File \"tbe/common/repository_manager/utils/multiprocess_util.py\", line 68
    item = self.task_q.get()
EOFError
""",
        encoding="utf-8",
    )
    (tmp_path / "ffn.log").write_text("clean shutdown\n", encoding="utf-8")

    result = runner._role_log_gate(tmp_path)

    assert result["passed"] is True
    assert result["roles"]["attention"]["fatal_markers"] == []


def test_dsv4_log_gate_rejects_tbe_queue_eof_before_shutdown(tmp_path):
    runner = _load_runner()
    (tmp_path / "attention.log").write_text(
        """Exception in thread Thread-1:
Traceback (most recent call last):
  File \"tbe/common/repository_manager/utils/multiprocess_util.py\", line 68
    item = self.task_q.get()
EOFError
""",
        encoding="utf-8",
    )
    (tmp_path / "ffn.log").write_text("clean shutdown\n", encoding="utf-8")

    result = runner._role_log_gate(tmp_path)

    assert result["passed"] is False
    assert result["roles"]["attention"]["fatal_markers"] == ["Exception in thread"]


def test_dsv4_log_gate_reports_missing_role_log(tmp_path):
    runner = _load_runner()
    (tmp_path / "ffn.log").write_text("clean shutdown\n", encoding="utf-8")

    result = runner._role_log_gate(tmp_path)

    assert result["passed"] is False
    assert result["roles"]["attention"]["fatal_markers"] == ["<log missing>"]


@pytest.mark.parametrize(
    ("u_batches", "attention_log", "kwargs", "expected"),
    [
        (1, "key=((0, (8,)),)\n", {}, True),
        (2, "key=((0, (8,)),)\n", {}, False),
        (2, "key=((0, (4,)), (1, (4,)))\n", {}, True),
        (
            2,
            "key=((0, (16,)),)\n",
            {"enable_mtp": True, "batch_sizes": [1], "data_parallel_size": 8},
            True,
        ),
        (
            2,
            "key=((0, (8,)),)\n",
            {"enable_mtp": True, "batch_sizes": [16], "data_parallel_size": 8},
            False,
        ),
    ],
)
def test_dsv4_ubatch_gate_requires_two_stage_runtime_evidence(
    tmp_path,
    u_batches,
    attention_log,
    kwargs,
    expected,
):
    runner = _load_runner()
    (tmp_path / "attention.log").write_text(attention_log, encoding="utf-8")

    result = runner._ubatch_execution_gate(tmp_path, u_batches, **kwargs)

    assert result["passed"] is expected


def test_dsv4_profile_gate_requires_every_rank_trace_to_be_nonempty(tmp_path):
    runner = _load_runner()
    for role in ("attention", "ffn"):
        trace_dir = tmp_path / role / f"{role}_dp0_ascend_pt"
        (trace_dir / "FRAMEWORK").mkdir(parents=True)
        (trace_dir / "profiler_info_0.json").write_text("{}\n", encoding="utf-8")
        (trace_dir / "FRAMEWORK/torch.op_range").write_bytes(b"torch-ops")
        raw_dir = trace_dir / "PROF_000001/device_0/data"
        raw_dir.mkdir(parents=True)
        (raw_dir / "stars.data").write_bytes(b"cann-data")

    result = runner._profile_output_gate(tmp_path)

    assert result["passed"] is True
    assert result["roles"]["attention"]["cann_raw_file_count"] == 1

    extra_trace = tmp_path / "attention/attention_dp1_ascend_pt"
    extra_trace.mkdir()

    result = runner._profile_output_gate(tmp_path)

    assert result["passed"] is False
    assert result["roles"]["attention"]["passed"] is False

    (extra_trace / "FRAMEWORK").mkdir()
    (extra_trace / "profiler_info_1.json").write_text("{}\n", encoding="utf-8")
    (extra_trace / "FRAMEWORK/torch.op_range").write_bytes(b"torch-ops")
    raw_dir = extra_trace / "PROF_000001/device_1/data"
    raw_dir.mkdir(parents=True)
    (raw_dir / "stars.data").write_bytes(b"cann-data")

    result = runner._profile_output_gate(tmp_path)

    assert result["passed"] is True
    assert result["roles"]["attention"]["cann_raw_file_count"] == 2
    assert len(result["roles"]["attention"]["traces"]) == 2


def test_dsv4_npu_process_parser_ignores_device_rows():
    runner = _load_runner()
    output = """\
| NPU   Name                | Health        | Power(W) |
| 0     0                   | 0000:18:00.0  | 0        |
| NPU     Chip              | Process id    | Process name |
| 0       0                 | 12345         | VLLM::EngineCore |
| 7       1                 | 67890         | python |
"""

    assert runner._npu_process_ids(output) == [12345, 67890]


def test_dsv4_npu_cleanup_gate_rejects_residual_processes(monkeypatch, tmp_path):
    runner = _load_runner()
    output = """\
| NPU     Chip              | Process id    | Process name |
| 0       0                 | 12345         | VLLM::EngineCore |
"""
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
    )

    result = runner._wait_for_npu_cleanup(tmp_path / "npu.txt", timeout=0)

    assert result["passed"] is False
    assert result["process_ids"] == [12345]


def test_dsv4_npu_cleanup_gate_accepts_clean_npus(monkeypatch, tmp_path):
    runner = _load_runner()
    output = """\
| NPU     Chip              | Process id    | Process name |
| No running processes found in NPU 0 |
"""
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
    )

    result = runner._wait_for_npu_cleanup(tmp_path / "npu.txt", timeout=0)

    assert result["passed"] is True
    assert result["process_ids"] == []


def test_dsv4_npu_cleanup_gate_rejects_truncated_output(monkeypatch, tmp_path):
    runner = _load_runner()
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            "npu-smi version only\n",
            "",
        ),
    )

    result = runner._wait_for_npu_cleanup(tmp_path / "npu.txt", timeout=0)

    assert result["passed"] is False
    assert result["process_table_present"] is False
