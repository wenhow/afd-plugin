#!/usr/bin/env python3
"""Run reproducible DeepSeek-V4 A8F8 HCCL P2P performance baselines."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RECIPE_DIR = Path(__file__).resolve().parent
REPO_ROOT = RECIPE_DIR.parents[3]
SHARED_RUNNER_PATH = REPO_ROOT / "recipe/npu/deepseek_v4/common/run_validation.py"
DEFAULT_MODEL = Path("/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp")
TOTAL_NPUS = 16
REPRODUCIBILITY_FILES = (
    Path(__file__).resolve(),
    SHARED_RUNNER_PATH,
    REPO_ROOT / "recipe/npu/deepseek_v4/common/activate_role_runtime.sh",
    RECIPE_DIR / "afd_attention.sh",
    RECIPE_DIR / "afd_ffn.sh",
)
_DEVICE_ROW = re.compile(
    r"^\|\s*\d+\s+\d+\s+\|\s*[0-9A-Fa-f:.]+\s+\|"
    r"\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)\s*\|"
)


def _load_shared_runner():
    spec = importlib.util.spec_from_file_location(
        "dsv4_shared_validation_runner",
        SHARED_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared runner: {SHARED_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SHARED = _load_shared_runner()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _benchmark_command(
    *,
    api_port: int,
    model_path: Path,
    result_path: Path,
    input_len: int,
    output_len: int,
    num_prompts: int,
    concurrency: int,
    u_batches: int,
    run_kind: str,
    repeat: int,
    served_model: str = "dsv4-afd",
    connector: str = "P2pHcclAFDConnector",
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--host",
        "127.0.0.1",
        "--port",
        str(api_port),
        "--endpoint",
        "/v1/completions",
        "--model",
        served_model,
        "--tokenizer",
        str(model_path),
        "--tokenizer-mode",
        "deepseek_v4",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        "0.0",
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--seed",
        "1024",
        "--temperature",
        "0",
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--disable-tqdm",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_path.parent),
        "--result-filename",
        result_path.name,
        "--metadata",
        f"connector={connector}",
        f"u_batches={u_batches}",
        f"run_kind={run_kind}",
        f"repeat={repeat}",
    ]


def _exited_services(
    processes: Mapping[str, subprocess.Popen[bytes]],
) -> dict[str, int]:
    return {
        role: int(returncode)
        for role, process in processes.items()
        if (returncode := process.poll()) is not None
    }


class _FatalLogWatcher:
    """Incrementally detect worker failures hidden behind live API parents."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self._offsets = {role: 0 for role in ("attention", "ffn")}
        self._tails = {role: b"" for role in self._offsets}
        self._carry_size = max(len(marker) for marker in SHARED.FATAL_LOG_MARKERS)

    def poll(self) -> dict[str, list[str]]:
        failures: dict[str, list[str]] = {}
        for role, offset in self._offsets.items():
            path = self.log_dir / f"{role}.log"
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size < offset:
                offset = 0
                self._tails[role] = b""
            with path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
            self._offsets[role] = size
            if not chunk:
                continue
            content = self._tails[role] + chunk
            text = content.decode("utf-8", errors="replace")
            markers = SHARED._fatal_log_markers(text)
            self._tails[role] = content[-self._carry_size :]
            if markers:
                failures[role] = markers
        return failures


def _stop_benchmark_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _run_benchmark(
    command: list[str],
    result_path: Path,
    log_path: Path,
    *,
    service_processes: Mapping[str, subprocess.Popen[bytes]],
    service_log_dir: Path,
    timeout: float,
) -> dict:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + timeout
        fatal_log_watcher = _FatalLogWatcher(service_log_dir)
        try:
            while process.poll() is None:
                exited = _exited_services(service_processes)
                if exited:
                    raise RuntimeError(
                        f"AFD service exited during benchmark: {exited}",
                    )
                fatal_logs = fatal_log_watcher.poll()
                if fatal_logs:
                    raise RuntimeError(
                        f"AFD service logged a fatal error during benchmark: "
                        f"{fatal_logs}",
                    )
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"benchmark timed out after {timeout:.0f}s")
                time.sleep(1)
        except BaseException:
            _stop_benchmark_process(process)
            raise
    if process.returncode != 0:
        tail = SHARED._log_tail(log_path)
        raise RuntimeError(f"benchmark failed with {process.returncode}\n{tail}")
    if not result_path.is_file():
        raise RuntimeError(f"benchmark result is missing: {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def _validate_benchmark_result(
    result: dict[str, Any],
    *,
    input_len: int,
    output_len: int,
    num_prompts: int,
) -> dict[str, Any]:
    expected_input = input_len * num_prompts
    expected_output = output_len * num_prompts
    errors = [error for error in result.get("errors", []) if error]
    checks = {
        "completed": int(result.get("completed", -1)) == num_prompts,
        "failed": int(result.get("failed", -1)) == 0,
        "input_tokens": int(result.get("total_input_tokens", -1)) == expected_input,
        "output_tokens": int(result.get("total_output_tokens", -1)) == expected_output,
        "errors": not errors,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected_input_tokens": expected_input,
        "expected_output_tokens": expected_output,
        "errors": errors,
    }


def _aggregate_results(
    records: list[dict[str, Any]],
    *,
    max_throughput_cv: float,
) -> dict[str, Any]:
    by_concurrency: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_concurrency.setdefault(int(record["concurrency"]), []).append(record)

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
    points: dict[str, Any] = {}
    for concurrency, point_records in sorted(by_concurrency.items()):
        metrics: dict[str, Any] = {}
        for name in metric_names:
            values = [float(record["result"][name]) for record in point_records]
            mean = statistics.fmean(values)
            stddev = statistics.pstdev(values)
            metrics[name] = {
                "values": values,
                "mean": mean,
                "min": min(values),
                "max": max(values),
                "stddev": stddev,
                "cv": stddev / mean if mean else 0.0,
            }
        output_mean = metrics["output_throughput"]["mean"]
        throughput_cv = metrics["output_throughput"]["cv"]
        stability_gate = {
            "passed": throughput_cv <= max_throughput_cv,
            "output_throughput_cv": throughput_cv,
            "max_output_throughput_cv": max_throughput_cv,
        }
        points[str(concurrency)] = {
            "runs": len(point_records),
            "passed": all(record["gate"]["passed"] for record in point_records)
            and stability_gate["passed"],
            "stability_gate": stability_gate,
            "metrics": metrics,
            "output_tokens_per_second_per_npu": output_mean / TOTAL_NPUS,
        }
    return {
        "passed": bool(points) and all(point["passed"] for point in points.values()),
        "points": points,
    }


def _parse_npu_snapshot(output: str) -> list[dict[str, int]]:
    devices = []
    for line in output.splitlines():
        match = _DEVICE_ROW.match(line)
        if match is None:
            continue
        aicore, hbm_used, hbm_total = (int(value) for value in match.groups())
        devices.append(
            {
                "device": len(devices),
                "aicore_percent": aicore,
                "hbm_used_mb": hbm_used,
                "hbm_total_mb": hbm_total,
            }
        )
    return devices


class _NPUMonitor:
    def __init__(self, output: Path, interval: float) -> None:
        self.output = output
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=max(10.0, self.interval * 2))
        lines = (json.dumps(sample, sort_keys=True) + "\n" for sample in self.samples)
        self.output.write_text(
            "".join(lines),
            encoding="utf-8",
        )
        max_hbm: dict[str, int] = {}
        max_aicore: dict[str, int] = {}
        valid_samples = 0
        for sample in self.samples:
            if sample.get("returncode") != 0 or not sample.get("devices"):
                continue
            valid_samples += 1
            for device in sample["devices"]:
                key = str(device["device"])
                max_hbm[key] = max(max_hbm.get(key, 0), device["hbm_used_mb"])
                max_aicore[key] = max(
                    max_aicore.get(key, 0),
                    device["aicore_percent"],
                )
        return {
            "sample_count": len(self.samples),
            "valid_sample_count": valid_samples,
            "max_hbm_mb": max_hbm,
            "max_aicore_percent": max_aicore,
            "passed": valid_samples > 0 and len(max_hbm) == TOTAL_NPUS,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.samples.append(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "returncode": result.returncode,
                    "devices": _parse_npu_snapshot(result.stdout),
                    "stderr": result.stderr.strip(),
                }
            )
            self._stop.wait(self.interval)


def _runtime_manifest(args: argparse.Namespace) -> dict[str, Any]:
    tensor_parallel_size = int(getattr(args, "tensor_parallel_size", 1))
    manifest = SHARED._runtime_manifest(
        connector="P2pHcclAFDConnector",
        execution_mode=args.execution_mode,
        u_batches=args.u_batches,
        dbo_decode_token_threshold=args.dbo_decode_token_threshold,
        dbo_prefill_token_threshold=args.dbo_prefill_token_threshold,
        profile=args.profile,
        async_scheduling=getattr(args, "async_scheduling", "off"),
        enable_mtp=args.enable_mtp,
        mtp_num_speculative_tokens=args.mtp_num_speculative_tokens,
        mtp_draft_execution=getattr(args, "mtp_draft_execution", "eager"),
        topology=_a8f8_topology(
            args.max_num_batched_tokens,
            tensor_parallel_size,
        ),
    )
    manifest.update(
        {
            "stage": (
                (
                    ("A3-P7M4-P1" if args.u_batches == 2 else "A3-P7M2-P1")
                    if args.execution_mode == "full-decode-only"
                    else ("A3-P7M3-P1" if args.u_batches == 2 else "A3-P7M1-P1")
                )
                if args.enable_mtp
                else "A3-P8"
            ),
            "topology_label": ("A8F8-DP4TP2" if tensor_parallel_size == 2 else "A8F8"),
            "npu_count": TOTAL_NPUS,
            "mtp_draft_execution": (
                getattr(args, "mtp_draft_execution", "eager")
                if args.enable_mtp
                else None
            ),
            "mtp_phase_u_batches": 1 if args.enable_mtp else None,
            "reproducibility_files_sha256": {
                str(path.relative_to(REPO_ROOT)): _file_sha256(path)
                for path in REPRODUCIBILITY_FILES
            },
            "service": {
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "max_num_seqs": args.max_num_seqs,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "attention_hccl_buffsize": 1024,
                "ffn_hccl_buffsize": 2048,
                "attention_hccl_if_base_port": args.attention_hccl_base_port,
                "ffn_hccl_if_base_port": args.ffn_hccl_base_port,
                "async_scheduling": args.async_scheduling,
                "tensor_parallel_size": tensor_parallel_size,
                "graph_u2_compute_overlap": getattr(
                    args,
                    "graph_u2_compute_overlap",
                    "on",
                ),
                "graph_u2_hybrid_dag": getattr(
                    args,
                    "graph_u2_hybrid_dag",
                    "on",
                ),
                "graph_u2_attention_three_stream": getattr(
                    args,
                    "graph_u2_attention_three_stream",
                    "on",
                ),
                "graph_u2_ffn_recv_stream": getattr(
                    args,
                    "graph_u2_ffn_recv_stream",
                    "on",
                ),
                "graph_u2_ffn_cross_layer": getattr(
                    args,
                    "graph_u2_ffn_cross_layer",
                    "on",
                ),
            },
            "workload": {
                "input_len": args.input_len,
                "output_len": args.output_len,
                "concurrencies": args.concurrencies,
                "repeats": args.repeats,
                "prompts_per_concurrency": args.prompts_per_concurrency,
                "min_prompts": args.min_prompts,
                "warmup_input_len": args.warmup_input_len,
                "warmup_output_len": args.warmup_output_len,
                "warmup_prompts": args.warmup_prompts,
                "warmup_concurrency": args.warmup_concurrency,
                "temperature": 0,
                "ignore_eos": True,
                "request_rate": "inf",
                "seed": 1024,
                "max_output_throughput_cv": args.max_throughput_cv,
                "benchmark_timeout_seconds": args.benchmark_timeout,
            },
        }
    )
    if args.profile:
        manifest["profile_schedule"] = {
            "skip_first": args.profile_skip_first,
            "wait": args.profile_wait,
            "warmup": args.profile_warmup,
            "active": args.profile_active,
            "repeat": 1,
            "with_stack": False,
        }
    return manifest


def _set_service_environment(args: argparse.Namespace) -> None:
    tensor_parallel_size = int(getattr(args, "tensor_parallel_size", 1))
    os.environ.update(
        {
            "MAX_MODEL_LEN": str(args.max_model_len),
            "MAX_NUM_BATCHED_TOKENS": str(args.max_num_batched_tokens),
            "MAX_NUM_SEQS": str(args.max_num_seqs),
            "GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization),
            "ATTENTION_HCCL_IF_BASE_PORT": str(args.attention_hccl_base_port),
            "FFN_HCCL_IF_BASE_PORT": str(args.ffn_hccl_base_port),
            "TORCH_PROFILER_WITH_STACK": "0",
            "AFD_ASYNC_SCHEDULING": args.async_scheduling,
            "TENSOR_PARALLEL_SIZE": str(tensor_parallel_size),
            "AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP": (
                "1" if getattr(args, "graph_u2_compute_overlap", "on") == "on" else "0"
            ),
            "AFD_HCCL_GRAPH_U2_HYBRID_DAG": (
                "1" if getattr(args, "graph_u2_hybrid_dag", "on") == "on" else "0"
            ),
            "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM": (
                "1"
                if getattr(args, "graph_u2_attention_three_stream", "on") == "on"
                else "0"
            ),
            "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM": (
                "1" if getattr(args, "graph_u2_ffn_recv_stream", "on") == "on" else "0"
            ),
            "AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER": (
                "1" if getattr(args, "graph_u2_ffn_cross_layer", "on") == "on" else "0"
            ),
        }
    )
    SHARED._set_mtp_environment(
        enable_mtp=args.enable_mtp,
        mtp_num_speculative_tokens=args.mtp_num_speculative_tokens,
        mtp_draft_execution=getattr(args, "mtp_draft_execution", "eager"),
    )
    if not args.profile:
        return
    for role in ("ATTENTION", "FFN"):
        prefix = f"AFD_NPU_{role}_PROFILER"
        os.environ.update(
            {
                f"{prefix}_ENABLE": "1",
                f"{prefix}_SKIP_FIRST": str(args.profile_skip_first),
                f"{prefix}_WAIT": str(args.profile_wait),
                f"{prefix}_WARMUP": str(args.profile_warmup),
                f"{prefix}_ACTIVE": str(args.profile_active),
                f"{prefix}_REPEAT": "1",
                f"{prefix}_WITH_STACK": "0",
            }
        )


def _a8f8_topology(
    max_num_batched_tokens: int,
    tensor_parallel_size: int = 1,
) -> dict[str, Any]:
    if tensor_parallel_size not in (1, 2):
        raise ValueError("A8F8 supports tensor_parallel_size=1 or 2")
    data_parallel_size = 8 // tensor_parallel_size
    return {
        "attention_ranks": 8,
        "ffn_ranks": 8,
        "tensor_parallel_size": tensor_parallel_size,
        "attention_data_parallel_size": data_parallel_size,
        "ffn_data_parallel_size": data_parallel_size,
        "ratio": 1,
        "attention_devices": list(range(8)),
        "ffn_devices": list(range(8, 16)),
        "unused_devices": [],
        "attention_max_num_batched_tokens": max_num_batched_tokens,
        "ffn_max_num_batched_tokens": max_num_batched_tokens,
    }


def _validate_execution_args(args: argparse.Namespace) -> None:
    tensor_parallel_size = int(getattr(args, "tensor_parallel_size", 1))
    SHARED._validate_execution_topology(
        connector="P2pHcclAFDConnector",
        execution_mode=args.execution_mode,
        u_batches=args.u_batches,
        async_scheduling=getattr(args, "async_scheduling", "off"),
        enable_mtp=args.enable_mtp,
        mtp_num_speculative_tokens=args.mtp_num_speculative_tokens,
        mtp_draft_execution=getattr(args, "mtp_draft_execution", "eager"),
        topology=_a8f8_topology(
            args.max_num_batched_tokens,
            tensor_parallel_size,
        ),
    )
    if (
        getattr(args, "graph_u2_ffn_cross_layer", "on") == "on"
        and getattr(args, "graph_u2_ffn_recv_stream", "on") != "on"
    ):
        raise ValueError(
            "Graph/U2 FFN cross-layer overlap requires the FFN receive stream"
        )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    for port in (args.attention_port, args.ffn_port, args.afd_port):
        if not SHARED._port_is_free(port):
            raise RuntimeError(f"port {port} is already in use")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "runtime.json").write_text(
        json.dumps(_runtime_manifest(args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _set_service_environment(args)

    processes: dict[str, Any] = {}
    handles = []
    records: list[dict[str, Any]] = []
    monitor: _NPUMonitor | None = None
    summary: dict[str, Any] = {
        "passed": False,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "execution_mode": args.execution_mode,
        "u_batches": args.u_batches,
        "tensor_parallel_size": int(getattr(args, "tensor_parallel_size", 1)),
        "enable_mtp": args.enable_mtp,
        "mtp_num_speculative_tokens": args.mtp_num_speculative_tokens,
        "profile": args.profile,
    }
    startup_started = time.monotonic()
    profile_dir = args.output_dir / "profiles" if args.profile else None
    try:
        ffn_process, ffn_handle = SHARED._start_role(
            "ffn",
            output_dir=args.output_dir,
            api_port=args.ffn_port,
            afd_port=args.afd_port,
            connector="P2pHcclAFDConnector",
            execution_mode=args.execution_mode,
            u_batches=args.u_batches,
            dbo_decode_token_threshold=args.dbo_decode_token_threshold,
            dbo_prefill_token_threshold=args.dbo_prefill_token_threshold,
            profile_dir=profile_dir,
        )
        processes["ffn"] = ffn_process
        handles.append(ffn_handle)
        time.sleep(2)
        attention_process, attention_handle = SHARED._start_role(
            "attention",
            output_dir=args.output_dir,
            api_port=args.attention_port,
            afd_port=args.afd_port,
            connector="P2pHcclAFDConnector",
            execution_mode=args.execution_mode,
            u_batches=args.u_batches,
            dbo_decode_token_threshold=args.dbo_decode_token_threshold,
            dbo_prefill_token_threshold=args.dbo_prefill_token_threshold,
            profile_dir=profile_dir,
        )
        processes["attention"] = attention_process
        handles.append(attention_handle)
        SHARED._wait_for_api(
            f"http://127.0.0.1:{args.attention_port}/v1/models",
            processes,
            args.output_dir,
            args.startup_timeout,
        )
        summary["startup_seconds"] = round(time.monotonic() - startup_started, 3)
        SHARED._capture_command(["npu-smi", "info"], args.output_dir / "npu_ready.txt")

        monitor = _NPUMonitor(
            args.output_dir / "npu_samples.jsonl",
            args.npu_sample_interval,
        )
        monitor.start()

        warmup_path = args.output_dir / "warmup.json"
        warmup_command = _benchmark_command(
            api_port=args.attention_port,
            model_path=args.model,
            result_path=warmup_path,
            input_len=args.warmup_input_len,
            output_len=args.warmup_output_len,
            num_prompts=args.warmup_prompts,
            concurrency=args.warmup_concurrency,
            u_batches=args.u_batches,
            run_kind="warmup",
            repeat=0,
        )
        warmup_result = _run_benchmark(
            warmup_command,
            warmup_path,
            args.output_dir / "warmup.log",
            service_processes=processes,
            service_log_dir=args.output_dir,
            timeout=args.benchmark_timeout,
        )
        summary["warmup_gate"] = _validate_benchmark_result(
            warmup_result,
            input_len=args.warmup_input_len,
            output_len=args.warmup_output_len,
            num_prompts=args.warmup_prompts,
        )
        if not summary["warmup_gate"]["passed"]:
            raise RuntimeError("warmup benchmark result gate failed")

        if args.profile:
            run_points = [(args.profile_concurrency, 1, args.profile_prompts)]
        else:
            run_points = [
                (
                    concurrency,
                    repeat,
                    max(
                        args.min_prompts,
                        concurrency * args.prompts_per_concurrency,
                    ),
                )
                for concurrency in args.concurrencies
                for repeat in range(1, args.repeats + 1)
            ]

        for concurrency, repeat, num_prompts in run_points:
            stem = (
                f"profile_c{concurrency}"
                if args.profile
                else f"c{concurrency}_r{repeat}"
            )
            result_path = args.output_dir / "benchmarks" / f"{stem}.json"
            command = _benchmark_command(
                api_port=args.attention_port,
                model_path=args.model,
                result_path=result_path,
                input_len=args.input_len,
                output_len=args.output_len,
                num_prompts=num_prompts,
                concurrency=concurrency,
                u_batches=args.u_batches,
                run_kind="profile" if args.profile else "measurement",
                repeat=repeat,
            )
            result = _run_benchmark(
                command,
                result_path,
                result_path.with_suffix(".log"),
                service_processes=processes,
                service_log_dir=args.output_dir,
                timeout=args.benchmark_timeout,
            )
            gate = _validate_benchmark_result(
                result,
                input_len=args.input_len,
                output_len=args.output_len,
                num_prompts=num_prompts,
            )
            record = {
                "concurrency": concurrency,
                "repeat": repeat,
                "num_prompts": num_prompts,
                "result_path": str(result_path),
                "command": command,
                "gate": gate,
                "result": result,
            }
            records.append(record)
            if not gate["passed"]:
                raise RuntimeError(f"benchmark result gate failed: {result_path}")
        summary["aggregate"] = _aggregate_results(
            records,
            max_throughput_cv=args.max_throughput_cv,
        )
    except BaseException as error:
        summary["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    finally:
        if monitor is not None:
            summary["npu_monitor"] = monitor.stop()
        summary["shutdown"] = SHARED._shutdown_roles(processes)
        for handle in handles:
            handle.close()
        summary["log_gate"] = SHARED._role_log_gate(args.output_dir)
        summary["ubatch_gate"] = SHARED._ubatch_execution_gate(
            args.output_dir,
            args.u_batches,
        )
        if profile_dir is not None:
            summary["profile_gate"] = SHARED._profile_output_gate(profile_dir)
        summary["npu_cleanup_gate"] = SHARED._wait_for_npu_cleanup(
            args.output_dir / "npu_after_cleanup.txt"
        )
        gates = [
            summary.get("warmup_gate", {}).get("passed", False),
            summary.get("aggregate", {}).get("passed", False),
            summary.get("shutdown", {}).get("passed", False),
            summary.get("log_gate", {}).get("passed", False),
            summary.get("ubatch_gate", {}).get("passed", False),
            summary.get("npu_monitor", {}).get("passed", False),
            summary.get("npu_cleanup_gate", {}).get("passed", False),
        ]
        if args.profile:
            gates.append(summary.get("profile_gate", {}).get("passed", False))
        summary["passed"] = all(gates)
        summary["records"] = records
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (args.output_dir / "performance_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--attention-port", type=int, default=8910)
    parser.add_argument("--ffn-port", type=int, default=8911)
    parser.add_argument("--afd-port", type=int, default=29761)
    parser.add_argument("--startup-timeout", type=float, default=3600)
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "full-decode-only"),
        default="eager",
    )
    parser.add_argument("--u-batches", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--graph-u2-compute-overlap",
        choices=("on", "off"),
        default="on",
        help=(
            "Enable or disable the Graph/U2 side-compute overlap while keeping "
            "the same source snapshot. This has no effect on U1."
        ),
    )
    parser.add_argument(
        "--graph-u2-hybrid-dag",
        choices=("on", "off"),
        default="on",
        help=(
            "Release each next-layer Attention stage from its own prior-layer "
            "F2A receive. This has no effect when Graph/U2 overlap is off."
        ),
    )
    parser.add_argument(
        "--graph-u2-attention-three-stream",
        choices=("on", "off"),
        default="on",
        help=(
            "Experimentally map Attention Graph compute/send/recv to three "
            "physical streams. Enabled by default on this validation branch."
        ),
    )
    parser.add_argument(
        "--graph-u2-ffn-recv-stream",
        choices=("on", "off"),
        default="on",
        help=(
            "Experimentally run FFN Graph A2F receives on the dedicated receive "
            "stream. Enabled by default on this validation branch."
        ),
    )
    parser.add_argument(
        "--graph-u2-ffn-cross-layer",
        choices=("on", "off"),
        default="on",
        help=(
            "Release next-layer FFN receives per stage instead of joining both "
            "stage sends at every layer. Enabled by default on this validation branch."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        choices=(1, 2),
        default=1,
        help="Use A8F8 as DP8/TP1 or DP4/TP2 for both AFD roles.",
    )
    parser.add_argument(
        "--enable-mtp",
        action="store_true",
        default=os.environ.get("ENABLE_MTP", "0") == "1",
    )
    parser.add_argument(
        "--mtp-num-speculative-tokens",
        type=int,
        default=int(os.environ.get("MTP_NUM_SPECULATIVE_TOKENS", "1")),
    )
    parser.add_argument(
        "--mtp-draft-execution",
        choices=("eager", "graph"),
        default=os.environ.get("MTP_DRAFT_EXECUTION", "eager"),
    )
    parser.add_argument("--dbo-decode-token-threshold", type=int, default=2)
    parser.add_argument("--dbo-prefill-token-threshold", type=int, default=12)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompts-per-concurrency", type=int, default=4)
    parser.add_argument("--min-prompts", type=int, default=8)
    parser.add_argument("--warmup-input-len", type=int, default=256)
    parser.add_argument("--warmup-output-len", type=int, default=16)
    parser.add_argument("--warmup-prompts", type=int, default=16)
    parser.add_argument("--warmup-concurrency", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--attention-hccl-base-port", type=int, default=51000)
    parser.add_argument("--ffn-hccl-base-port", type=int, default=52000)
    parser.add_argument("--npu-sample-interval", type=float, default=2.0)
    parser.add_argument("--max-throughput-cv", type=float, default=0.10)
    parser.add_argument("--benchmark-timeout", type=float, default=1800)
    parser.add_argument(
        "--async-scheduling",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Control vLLM host async scheduling for both AFD roles. "
            "The A3-P8 performance default is off."
        ),
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-concurrency", type=int, default=32)
    parser.add_argument("--profile-prompts", type=int, default=128)
    parser.add_argument("--profile-skip-first", type=int, default=64)
    parser.add_argument("--profile-wait", type=int, default=2)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-active", type=int, default=20)
    args = parser.parse_args()

    positive_fields = (
        "input_len",
        "output_len",
        "repeats",
        "prompts_per_concurrency",
        "min_prompts",
        "warmup_input_len",
        "warmup_output_len",
        "warmup_prompts",
        "warmup_concurrency",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "npu_sample_interval",
        "benchmark_timeout",
        "profile_concurrency",
        "profile_prompts",
        "profile_active",
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if any(concurrency <= 0 for concurrency in args.concurrencies):
        parser.error("--concurrencies values must be positive")
    if args.input_len + args.output_len > args.max_model_len:
        parser.error("input length plus output length exceeds --max-model-len")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if not 0 < args.max_throughput_cv <= 1:
        parser.error("--max-throughput-cv must be in (0, 1]")
    for field in ("attention_hccl_base_port", "ffn_hccl_base_port"):
        port = getattr(args, field)
        if port < 1024 or port > 60000:
            parser.error(f"--{field.replace('_', '-')} must be in [1024, 60000]")
    if abs(args.attention_hccl_base_port - args.ffn_hccl_base_port) < 1000:
        parser.error("Attention and FFN HCCL base ports must be at least 1000 apart")
    try:
        _validate_execution_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> None:
    args = _parse_args()
    summary = _run(args)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
