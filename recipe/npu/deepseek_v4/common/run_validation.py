#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path
from typing import Any

COMMON_RECIPE_DIR = Path(__file__).resolve().parent
REPO_ROOT = COMMON_RECIPE_DIR.parents[3]
DEFAULT_GOLDEN = Path(
    "/mnt/workspace/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json"
)
FUNCTIONAL_SMOKE_TOOL = REPO_ROOT / "tools/dsv4/run_pd_functional_smoke.py"
CANCELLATION_MAX_TIME_SECONDS = 1
CANCELLATION_MAX_TOKENS = 512
REQUEST_QUIESCENCE_TIMEOUT_SECONDS = 30
REQUEST_QUIESCENCE_POLL_SECONDS = 0.5
REQUEST_QUIESCENCE_STABLE_SAMPLES = 2
SHUTDOWN_HANDOFF_MIN_TIMEOUT_SECONDS = 15
SHUTDOWN_HANDOFF_GRACE_SECONDS = 15
SHUTDOWN_HANDOFF_POLL_SECONDS = 0.1
FFN_SHUTDOWN_RECEIPT_MARKER = "AFD NPU FFN received Attention shutdown payload"
DSPARK_DRAFTER_MARKER = (
    "DeepSeek-V4 AFD keeps the complete DSpark draft model on the Attention worker"
)
FATAL_LOG_MARKERS = (
    "AFD NPU FFN worker loop failed",
    "EngineCore encountered a fatal error",
    "RuntimeError: Worker failed with error",
    "Exception in thread",
    "Communication_Error_Bind_IP_Port",
    "error code is 507014",
    "error code is 507015",
    "error code is 507034",
    "error code is 507035",
)

CONNECTOR_RECIPE_DIRS = {
    "CAMP2pAFDConnector": (REPO_ROOT / "recipe/npu/CAMP2pAFDConnector/deepseek_v4"),
    "P2pHcclAFDConnector": (REPO_ROOT / "recipe/npu/P2pHcclAFDConnector/deepseek_v4"),
}


def _port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        # Match the API server's bind behavior so a prior cycle's TIME_WAIT
        # sockets are not mistaken for an active listener.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _start_role(
    role: str,
    *,
    output_dir: Path,
    api_port: int,
    afd_port: int,
    connector: str,
    execution_mode: str,
    u_batches: int,
    dbo_decode_token_threshold: int,
    dbo_prefill_token_threshold: int,
    profile_dir: Path | None,
) -> tuple[subprocess.Popen[bytes], Any]:
    log_handle = (output_dir / f"{role}.log").open("wb")
    env = os.environ.copy()
    env.update(
        {
            "API_PORT": str(api_port),
            "AFD_PORT": str(afd_port),
            "AFD_HOST": "127.0.0.1",
            "AFD_CONNECTOR": connector,
            "HCCL_IF_IP": env.get("HCCL_IF_IP", "192.169.91.106"),
            "PYTHONUNBUFFERED": "1",
            "EXECUTION_MODE": execution_mode,
            "U_BATCHES": str(u_batches),
            "DBO_DECODE_TOKEN_THRESHOLD": str(dbo_decode_token_threshold),
            "DBO_PREFILL_TOKEN_THRESHOLD": str(dbo_prefill_token_threshold),
        }
    )
    if profile_dir is not None:
        role_prefix = f"AFD_NPU_{role.upper()}_PROFILER"
        role_dir = profile_dir / role
        role_dir.mkdir(parents=True, exist_ok=True)
        env.update(
            {
                f"{role_prefix}_ENABLE": env.get(
                    f"{role_prefix}_ENABLE",
                    "1",
                ),
                f"{role_prefix}_WAIT": env.get(f"{role_prefix}_WAIT", "2"),
                f"{role_prefix}_WARMUP": env.get(
                    f"{role_prefix}_WARMUP",
                    "1",
                ),
                f"{role_prefix}_ACTIVE": env.get(
                    f"{role_prefix}_ACTIVE",
                    "10",
                ),
                f"{role_prefix}_REPEAT": env.get(
                    f"{role_prefix}_REPEAT",
                    "1",
                ),
                f"{role_prefix}_SKIP_FIRST": env.get(
                    f"{role_prefix}_SKIP_FIRST",
                    "0",
                ),
                f"{role_prefix}_DIR": str(role_dir),
                f"{role_prefix}_WITH_STACK": "0",
                "TORCH_PROFILER_WITH_STACK": "0",
            }
        )
    script = CONNECTOR_RECIPE_DIRS[connector] / f"afd_{role}.sh"
    process = subprocess.Popen(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_handle


def _log_tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return "<log missing>"
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def _fatal_log_markers(text: str) -> list[str]:
    markers = [
        marker
        for marker in FATAL_LOG_MARKERS
        if marker != "Exception in thread" and marker in text
    ]
    for match in re.finditer(r"Exception in thread", text):
        before = text[max(0, match.start() - 8192) : match.start()]
        traceback = text[match.start() : match.start() + 4096]
        known_tbe_shutdown_eof = (
            "[shutdown]" in before
            and "tbe/common/repository_manager/utils/multiprocess_util.py" in traceback
            and re.search(r"(?:^|\n).*EOFError(?:\n|$)", traceback) is not None
        )
        if not known_tbe_shutdown_eof:
            markers.append("Exception in thread")
            break
    return markers


def _wait_for_api(
    endpoint: str,
    processes: dict[str, subprocess.Popen[bytes]],
    log_dir: Path,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        for role, process in processes.items():
            log_path = log_dir / f"{role}.log"
            if process.poll() is not None:
                tail = _log_tail(log_path)
                raise RuntimeError(
                    f"{role} exited during startup with {process.returncode}\n{tail}"
                )
            tail = _log_tail(log_path)
            markers = _fatal_log_markers(tail)
            if markers:
                raise RuntimeError(
                    f"{role} reported a fatal startup error: {markers}\n{tail}"
                )
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(2)
    raise TimeoutError(f"API did not become ready: {last_error!r}")


def _run_validator(
    *,
    api_port: int,
    golden: Path,
    output: Path,
    rounds: int,
    batch_sizes: list[int],
    prompt_indices: list[int] | None,
) -> None:
    command = [
        sys.executable,
        str(COMMON_RECIPE_DIR / "validate_golden.py"),
        "--endpoint",
        f"http://127.0.0.1:{api_port}/v1/completions",
        "--model",
        "dsv4-afd",
        "--golden",
        str(golden),
        "--output",
        str(output),
        "--rounds",
        str(rounds),
        "--batch-sizes",
        *(str(size) for size in batch_sizes),
    ]
    if prompt_indices is not None:
        command.extend(["--prompt-indices", *(str(index) for index in prompt_indices)])
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _run_functional_validator(
    *,
    api_port: int,
    output: Path,
    batch_sizes: list[int],
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(FUNCTIONAL_SMOKE_TOOL),
            "--endpoint",
            f"http://127.0.0.1:{api_port}/v1/completions",
            "--model",
            "dsv4-afd",
            "--batch-sizes",
            " ".join(str(size) for size in batch_sizes),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def _run_cancellation_gate(*, api_port: int, output_dir: Path) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": "dsv4-afd",
            "prompt": "Write a detailed deterministic systems validation checklist.",
            "temperature": 0,
            "seed": 1024,
            "max_tokens": CANCELLATION_MAX_TOKENS,
            "stream": True,
        }
    )
    result = subprocess.run(
        [
            "curl",
            "-fsS",
            "--max-time",
            str(CANCELLATION_MAX_TIME_SECONDS),
            f"http://127.0.0.1:{api_port}/v1/completions",
            "-H",
            "Content-Type: application/json",
            "-d",
            payload,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    (output_dir / "cancellation-response.json").write_bytes(result.stdout)
    (output_dir / "cancellation.stderr").write_bytes(result.stderr)
    (output_dir / "cancellation.exitcode").write_text(
        f"{result.returncode}\n",
        encoding="utf-8",
    )
    gate = {
        "passed": result.returncode == 28,
        "expected_exitcode": 28,
        "actual_exitcode": result.returncode,
        "max_time_seconds": CANCELLATION_MAX_TIME_SECONDS,
    }
    (output_dir / "cancellation_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return gate


def _request_counts_from_metrics(text: str) -> tuple[float, float]:
    counts: dict[str, list[float]] = {"running": [], "waiting": []}
    pattern = re.compile(
        r"^vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if match is not None:
            counts[match.group(1)].append(float(match.group(2)))
    if not counts["running"] or not counts["waiting"]:
        raise ValueError("request count metrics are missing")
    return sum(counts["running"]), sum(counts["waiting"])


def _wait_for_request_quiescence(
    *,
    api_port: int,
    output_dir: Path,
    artifact_stem: str,
) -> dict[str, Any]:
    endpoint = f"http://127.0.0.1:{api_port}/metrics"
    deadline = time.monotonic() + REQUEST_QUIESCENCE_TIMEOUT_SECONDS
    started = time.monotonic()
    attempts = 0
    stable_samples = 0
    running: float | None = None
    waiting: float | None = None
    last_metrics = ""
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        attempts += 1
        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                last_metrics = response.read().decode("utf-8", errors="replace")
            running, waiting = _request_counts_from_metrics(last_metrics)
            last_error = None
        except (OSError, ValueError, urllib.error.URLError) as error:
            stable_samples = 0
            last_error = error
        else:
            if running == 0 and waiting == 0:
                stable_samples += 1
                if stable_samples >= REQUEST_QUIESCENCE_STABLE_SAMPLES:
                    break
            else:
                stable_samples = 0
        time.sleep(REQUEST_QUIESCENCE_POLL_SECONDS)

    (output_dir / f"{artifact_stem}.metrics").write_text(
        last_metrics,
        encoding="utf-8",
    )
    passed = stable_samples >= REQUEST_QUIESCENCE_STABLE_SAMPLES
    gate: dict[str, Any] = {
        "passed": passed,
        "attempts": attempts,
        "waited_seconds": round(time.monotonic() - started, 3),
        "stable_samples": stable_samples,
        "required_stable_samples": REQUEST_QUIESCENCE_STABLE_SAMPLES,
        "running": running,
        "waiting": waiting,
    }
    if last_error is not None:
        gate["last_error"] = f"{type(last_error).__name__}: {last_error}"
    (output_dir / f"{artifact_stem}_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return gate


def _spec_decode_token_counts(text: str) -> tuple[float, float]:
    counts: dict[str, list[float]] = {"draft": [], "accepted": []}
    pattern = re.compile(
        r"^vllm:spec_decode_num_(draft|accepted)_tokens(?:_total)?"
        r"(?:\{[^}]*\})?\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if match is not None:
            counts[match.group(1)].append(float(match.group(2)))
    if not counts["draft"] or not counts["accepted"]:
        raise ValueError("speculative decode token metrics are missing")
    return sum(counts["draft"]), sum(counts["accepted"])


def _dspark_execution_gate(
    *,
    api_port: int,
    output_dir: Path,
    expected_attention_ranks: int,
) -> dict[str, Any]:
    endpoint = f"http://127.0.0.1:{api_port}/metrics"
    metrics = ""
    drafted: float | None = None
    accepted: float | None = None
    error: str | None = None
    try:
        with urllib.request.urlopen(endpoint, timeout=10) as response:
            metrics = response.read().decode("utf-8", errors="replace")
        drafted, accepted = _spec_decode_token_counts(metrics)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    (output_dir / "dspark.metrics").write_text(metrics, encoding="utf-8")

    attention_log = output_dir / "attention.log"
    attention_text = (
        attention_log.read_text(encoding="utf-8", errors="replace")
        if attention_log.is_file()
        else ""
    )
    drafter_markers = attention_text.count(DSPARK_DRAFTER_MARKER)
    gate: dict[str, Any] = {
        "passed": bool(
            error is None
            and drafted is not None
            and drafted > 0
            and accepted is not None
            and accepted > 0
            and drafter_markers >= expected_attention_ranks
        ),
        "draft_tokens": drafted,
        "accepted_tokens": accepted,
        "drafter_marker": DSPARK_DRAFTER_MARKER,
        "drafter_markers": drafter_markers,
        "expected_attention_ranks": expected_attention_ranks,
    }
    if error is not None:
        gate["error"] = error
    (output_dir / "dspark_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return gate


def _signal_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    # The group can outlive its leader when a vLLM worker is reparented to PID 1.
    # Always address the group that this runner created, even if Popen.poll()
    # already reports that the role's shell or API parent has exited.
    with suppress(ProcessLookupError):
        os.killpg(process.pid, sig)


def _signal_process(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    if process.poll() is None:
        with suppress(ProcessLookupError):
            os.kill(process.pid, sig)


def _stop_process(
    process: subprocess.Popen[bytes],
    timeout: float = 30,
    *,
    signal_group: bool = True,
) -> None:
    _request_process_stop(process, signal_group=signal_group)
    _wait_for_process_stop(process, timeout=timeout)


def _request_process_stop(
    process: subprocess.Popen[bytes],
    *,
    signal_group: bool,
) -> None:
    if signal_group:
        _signal_group(process, signal.SIGTERM)
    else:
        _signal_process(process, signal.SIGTERM)


def _wait_for_process_stop(
    process: subprocess.Popen[bytes],
    *,
    timeout: float = 30,
) -> None:
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        process.wait(timeout=30)
    finally:
        # A clean role-parent exit does not prove that every NPU worker in its
        # process group has exited. Drain any remaining owned descendants.
        _signal_group(process, signal.SIGKILL)


def _wait_for_log_occurrences(
    *,
    log_path: Path,
    marker: str,
    expected: int,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    observed = 0
    while time.monotonic() < deadline:
        if log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            observed = text.count(marker)
        if observed >= expected:
            break
        time.sleep(SHUTDOWN_HANDOFF_POLL_SECONDS)
    return {
        "passed": observed >= expected,
        "marker": marker,
        "expected": expected,
        "observed": observed,
        "waited_seconds": round(time.monotonic() - started, 3),
        "timeout_seconds": timeout,
    }


def _shutdown_handoff_timeout_seconds() -> int:
    """Keep FFN alive for Attention's configured drain plus teardown."""
    raw_timeout = os.environ.get("VLLM_SHUTDOWN_TIMEOUT_SECONDS", "0")
    try:
        drain_timeout = max(int(raw_timeout), 0)
    except ValueError:
        drain_timeout = 0
    return max(
        SHUTDOWN_HANDOFF_MIN_TIMEOUT_SECONDS,
        drain_timeout + SHUTDOWN_HANDOFF_GRACE_SECONDS,
    )


def _shutdown_roles(
    processes: dict[str, subprocess.Popen[bytes]],
    *,
    ffn_log_path: Path,
    expected_ffn_shutdown_receipts: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "order": [
            "attention_request",
            "ffn_handoff_wait",
            "ffn_request",
            "attention_wait",
            "ffn_wait",
        ],
        "coordinated": True,
    }
    attention = processes.get("attention")
    ffn = processes.get("ffn")
    # Attention may execute one final DP dummy batch while entering drain. Keep
    # FFN alive until every FFN worker has consumed Attention's explicit
    # shutdown payload, then let both roles release their process groups.
    if attention is not None:
        _request_process_stop(attention, signal_group=False)
    if attention is not None and ffn is not None:
        result["ffn_handoff_gate"] = _wait_for_log_occurrences(
            log_path=ffn_log_path,
            marker=FFN_SHUTDOWN_RECEIPT_MARKER,
            expected=expected_ffn_shutdown_receipts,
            timeout=_shutdown_handoff_timeout_seconds(),
        )
    if ffn is not None:
        _request_process_stop(ffn, signal_group=False)
    if attention is not None:
        _wait_for_process_stop(attention)
        result["attention_returncode"] = attention.returncode
    if ffn is not None:
        ffn_exited_after_attention = ffn.poll() is not None
        result["ffn_exited_after_attention"] = ffn_exited_after_attention
        _wait_for_process_stop(ffn)
        result["ffn_returncode"] = ffn.returncode
    result["passed"] = all(
        result.get(f"{role}_returncode") == 0
        for role in ("attention", "ffn")
        if role in processes
    ) and result.get("ffn_handoff_gate", {}).get("passed", True)
    return result


def _role_log_gate(log_dir: Path) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in ("attention", "ffn"):
        log_path = log_dir / f"{role}.log"
        if not log_path.is_file():
            roles[role] = {
                "passed": False,
                "fatal_markers": ["<log missing>"],
            }
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        markers = _fatal_log_markers(text)
        roles[role] = {"passed": not markers, "fatal_markers": markers}
    return {
        "roles": roles,
        "passed": all(result["passed"] for result in roles.values()),
    }


def _ubatch_execution_gate(
    log_dir: Path,
    u_batches: int,
    *,
    enable_mtp: bool = False,
    batch_sizes: list[int] | None = None,
    data_parallel_size: int = 1,
) -> dict[str, Any]:
    """Validate U2 execution or the intentional low-concurrency MTP fallback."""
    batch_sizes = batch_sizes or []
    mtp_u2_min_batch = 2 * data_parallel_size
    fallback_expected = bool(
        u_batches == 2 and enable_mtp and max(batch_sizes, default=0) < mtp_u2_min_batch
    )
    required = u_batches == 2 and not fallback_expected
    log_path = log_dir / "attention.log"
    text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )
    observed = any(
        "key=((0," in line and "), (1," in line for line in text.splitlines()
    )
    return {
        "required": required,
        "observed_two_stages": observed,
        "mtp_request_boundary_fallback_expected": fallback_expected,
        "mtp_u2_min_batch": mtp_u2_min_batch if enable_mtp else None,
        "passed": (not observed) if fallback_expected else (not required or observed),
    }


def _profile_output_gate(profile_dir: Path) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in ("attention", "ffn"):
        trace_dirs = sorted((profile_dir / role).glob("*_ascend_pt"))
        traces: list[dict[str, Any]] = []
        required_sizes: dict[str, int] = {}
        cann_raw_file_count = 0
        for trace_dir in trace_dirs:
            profiler_info_files = sorted(trace_dir.glob("profiler_info_*.json"))
            trace_required_sizes = {
                "profiler_info_*.json": (
                    profiler_info_files[0].stat().st_size
                    if len(profiler_info_files) == 1
                    else 0
                ),
                "FRAMEWORK/torch.op_range": (
                    (trace_dir / "FRAMEWORK/torch.op_range").stat().st_size
                    if (trace_dir / "FRAMEWORK/torch.op_range").is_file()
                    else 0
                ),
            }
            trace_raw_file_count = sum(
                1
                for prof_dir in trace_dir.glob("PROF_*")
                if prof_dir.is_dir()
                for path in prof_dir.rglob("*")
                if path.is_file() and path.stat().st_size > 0
            )
            trace_passed = bool(
                len(profiler_info_files) == 1
                and all(size > 0 for size in trace_required_sizes.values())
                and trace_raw_file_count > 0
            )
            traces.append(
                {
                    "passed": trace_passed,
                    "trace_dir": str(trace_dir),
                    "profiler_info_files": [str(path) for path in profiler_info_files],
                    "required_sizes": trace_required_sizes,
                    "cann_raw_file_count": trace_raw_file_count,
                }
            )
            required_sizes.update(
                {
                    f"{trace_dir.name}/{path}": size
                    for path, size in trace_required_sizes.items()
                }
            )
            cann_raw_file_count += trace_raw_file_count
        role_passed = bool(traces) and all(trace["passed"] for trace in traces)
        roles[role] = {
            "passed": role_passed,
            "trace_dirs": [str(path) for path in trace_dirs],
            "traces": traces,
            "required_sizes": required_sizes,
            "cann_raw_file_count": cann_raw_file_count,
        }
    return {
        "roles": roles,
        "passed": all(result["passed"] for result in roles.values()),
    }


def _capture_command(command: list[str], output: Path) -> None:
    with output.open("wb") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)


def _npu_process_ids(output: str) -> list[int]:
    """Return process IDs from the process table in ``npu-smi info`` output."""
    in_process_table = False
    process_ids: list[int] = []
    for line in output.splitlines():
        if "| NPU" in line and "| Process id" in line:
            in_process_table = True
            continue
        if not in_process_table:
            continue
        match = re.match(r"^\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|", line)
        if match is not None:
            process_ids.append(int(match.group(1)))
    return sorted(set(process_ids))


def _has_npu_process_table(output: str) -> bool:
    return any(
        "| NPU" in line and "| Process id" in line for line in output.splitlines()
    )


def _wait_for_npu_cleanup(
    output: Path,
    *,
    timeout: float = 60,
    poll_interval: float = 2,
) -> dict[str, Any]:
    """Wait for role processes to leave the NPUs without killing other workloads."""
    started = time.monotonic()
    attempts = 0
    returncode = -1
    process_ids: list[int] = []
    process_table_present = False
    while True:
        attempts += 1
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            check=False,
        )
        returncode = result.returncode
        combined_output = result.stdout
        if result.stderr:
            combined_output += result.stderr
        output.write_text(combined_output, encoding="utf-8")
        process_table_present = _has_npu_process_table(result.stdout)
        process_ids = _npu_process_ids(result.stdout)
        if returncode == 0 and process_table_present and not process_ids:
            break
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            break
        time.sleep(min(poll_interval, timeout - elapsed))
    return {
        "passed": returncode == 0 and process_table_present and not process_ids,
        "process_ids": process_ids,
        "process_table_present": process_table_present,
        "npu_smi_returncode": returncode,
        "attempts": attempts,
        "waited_seconds": round(time.monotonic() - started, 3),
    }


def _runtime_manifest(
    *,
    connector: str,
    execution_mode: str,
    u_batches: int,
    dbo_decode_token_threshold: int,
    dbo_prefill_token_threshold: int,
    profile: bool,
    async_scheduling: str = "auto",
    eager_u2_stream_overlap: str = "on",
    graph_u2_compute_overlap: str = "on",
    graph_u2_hybrid_dag: str = "on",
    graph_u2_attention_three_stream: str = "on",
    graph_u2_ffn_recv_stream: str = "on",
    graph_u2_ffn_cross_layer: str = "on",
    stage_diagnostics: str = "off",
    enable_mtp: bool = False,
    mtp_num_speculative_tokens: int = 1,
    mtp_draft_execution: str = "eager",
    enable_dspark: bool = False,
    dspark_num_speculative_tokens: int = 0,
    dspark_draft_execution: str = "eager",
    topology: dict[str, Any] | None = None,
) -> dict[str, Any]:
    venv_path = os.environ.get(
        "DSV4_RUNTIME_VENV",
        "/mnt/workspace/code/.venvs/afd-v023-vllm-cann",
    )
    vllm_root = os.environ.get(
        "DSV4_VLLM_ROOT",
        "/mnt/workspace/code/vllm-release-v0.23.0",
    )
    vllm_ascend_root = os.environ.get(
        "DSV4_VLLM_ASCEND_ROOT",
        "/mnt/workspace/code/vllm-ascend-rfc-vllm-cann",
    )
    cann_root = os.environ.get(
        "DSV4_CANN_ROOT",
        "/mnt/workspace/code/.ascend/cann-9.0.0/cann-9.0.0",
    )
    cann_version = os.environ.get("DSV4_CANN_VERSION")
    if not cann_version:
        cann_version = Path(cann_root).name.removeprefix("cann-")
    profile_role_rank_selection = (
        {
            role: os.environ.get(
                f"AFD_NPU_{role.upper()}_PROFILER_RANKS",
                "0",
            )
            for role in ("attention", "ffn")
        }
        if profile
        else {}
    )

    def git_head(path: str) -> str:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True
        ).strip()

    afd_status = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "status",
            "--short",
            "--untracked-files=no",
        ],
        text=True,
    ).splitlines()
    afd_diff = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "diff", "--binary", "HEAD"],
    )
    manifest = {
        "python": sys.version,
        "plugins": "ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd",
        "cann": str(Path(cann_root).resolve()),
        "cann_version": cann_version,
        "venv": venv_path,
        "model": os.environ.get(
            "MODEL_PATH",
            "/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp",
        ),
        "connector": connector,
        "execution_mode": execution_mode,
        "u_batches": u_batches,
        "dbo_decode_token_threshold": dbo_decode_token_threshold,
        "dbo_prefill_token_threshold": dbo_prefill_token_threshold,
        "async_scheduling": async_scheduling,
        "eager_u2_stream_overlap": eager_u2_stream_overlap,
        "graph_u2_compute_overlap": graph_u2_compute_overlap,
        "graph_u2_hybrid_dag": graph_u2_hybrid_dag,
        "graph_u2_attention_three_stream": graph_u2_attention_three_stream,
        "graph_u2_ffn_recv_stream": graph_u2_ffn_recv_stream,
        "graph_u2_ffn_cross_layer": graph_u2_ffn_cross_layer,
        "stage_diagnostics": stage_diagnostics,
        "enable_mtp": enable_mtp,
        "mtp_num_speculative_tokens": mtp_num_speculative_tokens,
        "mtp_draft_execution": mtp_draft_execution if enable_mtp else None,
        "enable_dspark": enable_dspark,
        "dspark_num_speculative_tokens": (
            dspark_num_speculative_tokens if enable_dspark else None
        ),
        "dspark_draft_execution": (dspark_draft_execution if enable_dspark else None),
        "profile": profile,
        "profile_role_ranks": [0] if profile else [],
        "profile_role_rank_selection": profile_role_rank_selection,
        "torch_profiler_with_stack": False,
        "commits": {
            "afd_plugin": git_head(str(REPO_ROOT)),
            "vllm": git_head(vllm_root),
            "vllm_ascend": git_head(vllm_ascend_root),
        },
        "afd_plugin_worktree": {
            "tracked_dirty": bool(afd_status),
            "tracked_status": afd_status,
            "tracked_diff_sha256": hashlib.sha256(afd_diff).hexdigest(),
        },
    }
    if topology is not None:
        manifest["topology"] = topology
    return manifest


def _parse_device_list(raw: str) -> list[int]:
    try:
        devices = [int(value) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "devices must be a comma-separated integer list",
        ) from exc
    if not devices:
        raise argparse.ArgumentTypeError("device list cannot be empty")
    return devices


def _resolve_topology(
    *,
    connector: str,
    attention_devices: list[int],
    ffn_devices: list[int],
    attention_max_num_batched_tokens: int,
    ffn_max_num_batched_tokens: int | None,
    tensor_parallel_size: int = 1,
) -> dict[str, Any]:
    attention_ranks = len(attention_devices)
    ffn_ranks = len(ffn_devices)
    all_devices = [*attention_devices, *ffn_devices]
    if any(device < 0 or device >= 16 for device in all_devices):
        raise ValueError("DeepSeek-V4 recipe devices must be in [0, 15]")
    if len(set(all_devices)) != len(all_devices):
        raise ValueError("Attention and FFN device lists must not overlap")
    if connector == "CAMP2pAFDConnector" and attention_ranks != ffn_ranks:
        raise ValueError("CAMP2pAFDConnector requires equal Attention and FFN ranks")
    larger_ranks = max(attention_ranks, ffn_ranks)
    smaller_ranks = min(attention_ranks, ffn_ranks)
    if larger_ranks % smaller_ranks != 0:
        raise ValueError(
            "P2pHcclAFDConnector requires one role count to be an integer "
            "multiple of the other",
        )
    if attention_max_num_batched_tokens <= 0:
        raise ValueError("Attention max_num_batched_tokens must be positive")
    if tensor_parallel_size not in (1, 2):
        raise ValueError("DeepSeek-V4 AFD supports tensor parallel size 1 or 2")
    if (
        attention_ranks % tensor_parallel_size != 0
        or ffn_ranks % tensor_parallel_size != 0
    ):
        raise ValueError("Attention and FFN ranks must be divisible by TP size")
    if tensor_parallel_size == 2:
        if connector != "P2pHcclAFDConnector":
            raise ValueError("DeepSeek-V4 AFD TP2 requires P2pHcclAFDConnector")
        if attention_ranks != ffn_ranks:
            raise ValueError(
                "DeepSeek-V4 AFD TP2 requires equal Attention and FFN ranks"
            )

    ratio = larger_ranks // smaller_ranks
    attention_fans_out = ffn_ranks > attention_ranks
    required_ffn_tokens = (
        (attention_max_num_batched_tokens + ratio - 1) // ratio
        if attention_fans_out
        else attention_max_num_batched_tokens * ratio
    )
    resolved_ffn_tokens = (
        required_ffn_tokens
        if ffn_max_num_batched_tokens is None
        else ffn_max_num_batched_tokens
    )
    if resolved_ffn_tokens < required_ffn_tokens:
        raise ValueError(
            "FFN max_num_batched_tokens must cover one Attention subgroup: "
            f"required at least {required_ffn_tokens}, got {resolved_ffn_tokens}",
        )

    return {
        "attention_ranks": attention_ranks,
        "ffn_ranks": ffn_ranks,
        "tensor_parallel_size": tensor_parallel_size,
        "attention_data_parallel_size": (attention_ranks // tensor_parallel_size),
        "ffn_data_parallel_size": ffn_ranks // tensor_parallel_size,
        "ratio": ratio,
        "direction": "fan_out" if attention_fans_out else "fan_in",
        "attention_devices": attention_devices,
        "ffn_devices": ffn_devices,
        "unused_devices": sorted(set(range(16)) - set(all_devices)),
        "attention_max_num_batched_tokens": attention_max_num_batched_tokens,
        "ffn_max_num_batched_tokens": resolved_ffn_tokens,
    }


def _set_topology_environment(topology: dict[str, Any]) -> None:
    os.environ.update(
        {
            "ATTENTION_RANKS": str(topology["attention_ranks"]),
            "FFN_RANKS": str(topology["ffn_ranks"]),
            "TENSOR_PARALLEL_SIZE": str(topology["tensor_parallel_size"]),
            "ATTENTION_DEVICES": ",".join(
                str(device) for device in topology["attention_devices"]
            ),
            "FFN_DEVICES": ",".join(str(device) for device in topology["ffn_devices"]),
            "ATTENTION_MAX_NUM_BATCHED_TOKENS": str(
                topology["attention_max_num_batched_tokens"],
            ),
            "FFN_MAX_NUM_BATCHED_TOKENS": str(
                topology["ffn_max_num_batched_tokens"],
            ),
        },
    )


def _set_mtp_environment(
    *,
    enable_mtp: bool,
    mtp_num_speculative_tokens: int,
    mtp_draft_execution: str = "eager",
) -> None:
    os.environ.update(
        {
            "ENABLE_MTP": "1" if enable_mtp else "0",
            "MTP_NUM_SPECULATIVE_TOKENS": str(mtp_num_speculative_tokens),
            "MTP_DRAFT_EXECUTION": mtp_draft_execution,
        }
    )


def _set_dspark_environment(
    *,
    enable_dspark: bool,
    dspark_num_speculative_tokens: int,
    dspark_draft_execution: str = "eager",
) -> None:
    os.environ.update(
        {
            "ENABLE_DSPARK": "1" if enable_dspark else "0",
            "DSPARK_NUM_SPECULATIVE_TOKENS": str(dspark_num_speculative_tokens),
            "DSPARK_DRAFT_EXECUTION": dspark_draft_execution,
        }
    )


def _resolve_dspark_num_speculative_tokens(
    model_path: Path,
    requested: str,
) -> int:
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"DSpark model config does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"DSpark model config is invalid JSON: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError("DSpark model config must contain a JSON object")
    block_size = config.get("dspark_block_size")
    target_layer_ids = config.get("dspark_target_layer_ids")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError(
            "DSpark checkpoint requires a positive integer dspark_block_size"
        )
    if (
        not isinstance(target_layer_ids, list)
        or not target_layer_ids
        or any(
            not isinstance(layer_id, int) or isinstance(layer_id, bool)
            for layer_id in target_layer_ids
        )
    ):
        raise ValueError(
            "DSpark checkpoint requires non-empty integer dspark_target_layer_ids"
        )
    num_hidden_layers = config.get("num_hidden_layers")
    if isinstance(num_hidden_layers, int) and any(
        layer_id < 0 or layer_id >= num_hidden_layers for layer_id in target_layer_ids
    ):
        raise ValueError(
            "DSpark checkpoint target layers must be within num_hidden_layers"
        )
    if requested != "auto":
        try:
            requested_tokens = int(requested)
        except ValueError as exc:
            raise ValueError(
                "DSpark num_speculative_tokens must be auto or a positive integer"
            ) from exc
        if requested_tokens <= 0:
            raise ValueError(
                "DSpark num_speculative_tokens must be auto or a positive integer"
            )
        if requested_tokens != block_size:
            raise ValueError(
                "DSpark num_speculative_tokens must match checkpoint "
                f"dspark_block_size={block_size}"
            )
    return block_size


def _set_execution_environment(
    *,
    async_scheduling: str,
    eager_u2_stream_overlap: str = "on",
    graph_u2_compute_overlap: str = "on",
    graph_u2_hybrid_dag: str = "on",
    graph_u2_attention_three_stream: str = "on",
    graph_u2_ffn_recv_stream: str = "on",
    graph_u2_ffn_cross_layer: str = "on",
    stage_diagnostics: str = "off",
) -> None:
    os.environ.update(
        {
            "AFD_ASYNC_SCHEDULING": async_scheduling,
            "AFD_HCCL_EAGER_U2_STREAM_OVERLAP": (
                "1" if eager_u2_stream_overlap == "on" else "0"
            ),
            "AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP": (
                "1" if graph_u2_compute_overlap == "on" else "0"
            ),
            "AFD_HCCL_GRAPH_U2_HYBRID_DAG": (
                "1" if graph_u2_hybrid_dag == "on" else "0"
            ),
            "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM": (
                "1" if graph_u2_attention_three_stream == "on" else "0"
            ),
            "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM": (
                "1" if graph_u2_ffn_recv_stream == "on" else "0"
            ),
            "AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER": (
                "1" if graph_u2_ffn_cross_layer == "on" else "0"
            ),
            "AFD_HCCL_STAGE_DIAGNOSTICS": ("1" if stage_diagnostics == "on" else "0"),
        }
    )


def _toggle_default(environment_name: str) -> str:
    raw = os.environ.get(environment_name, "1")
    if raw == "1":
        return "on"
    if raw == "0":
        return "off"
    raise ValueError(f"{environment_name} must be 0 or 1, got {raw!r}")


def _validate_execution_topology(
    *,
    connector: str,
    execution_mode: str,
    u_batches: int = 1,
    async_scheduling: str = "off",
    enable_mtp: bool = False,
    mtp_num_speculative_tokens: int = 1,
    mtp_draft_execution: str = "eager",
    enable_dspark: bool = False,
    dspark_num_speculative_tokens: int = 0,
    dspark_draft_execution: str = "eager",
    topology: dict[str, Any],
) -> None:
    tensor_parallel_size = int(topology.get("tensor_parallel_size", 1))
    if tensor_parallel_size == 2 and connector != "P2pHcclAFDConnector":
        raise ValueError("DeepSeek-V4 AFD TP2 requires P2pHcclAFDConnector")
    if tensor_parallel_size == 2 and (
        int(topology["attention_ranks"]) != int(topology["ffn_ranks"])
    ):
        raise ValueError("DeepSeek-V4 AFD TP2 requires equal Attention and FFN ranks")
    if (
        u_batches == 2
        and execution_mode == "full-decode-only"
        and connector != "P2pHcclAFDConnector"
    ):
        raise ValueError(
            "DeepSeek-V4 graph U2 requires P2pHcclAFDConnector",
        )
    if (
        u_batches == 2
        and execution_mode == "full-decode-only"
        and async_scheduling != "off"
    ):
        raise ValueError(
            "DeepSeek-V4 graph U2 requires async scheduling off on the pinned stack",
        )
    if enable_mtp and enable_dspark:
        raise ValueError("DeepSeek-V4 MTP and DSpark cannot both be enabled")
    if not enable_mtp and not enable_dspark:
        return
    if connector != "P2pHcclAFDConnector":
        mode = "DSpark" if enable_dspark else "MTP"
        raise ValueError(f"DeepSeek-V4 {mode} requires P2pHcclAFDConnector")
    if execution_mode not in {"eager", "full-decode-only"}:
        raise ValueError(
            "DeepSeek-V4 speculative decode requires eager or "
            "full-decode-only execution"
        )
    if enable_mtp:
        if not 1 <= mtp_num_speculative_tokens <= 3:
            raise ValueError(
                "DeepSeek-V4 MTP supports num_speculative_tokens in [1, 3]"
            )
        if mtp_draft_execution not in {"eager", "graph"}:
            raise ValueError("DeepSeek-V4 MTP draft execution must be eager or graph")
        if mtp_draft_execution == "graph" and execution_mode != "full-decode-only":
            raise ValueError(
                "DeepSeek-V4 MTP draft Graph requires target full-decode-only"
            )
    if enable_dspark:
        if dspark_num_speculative_tokens <= 0:
            raise ValueError("DeepSeek-V4 DSpark requires positive speculative tokens")
        if dspark_draft_execution not in {"eager", "graph"}:
            raise ValueError(
                "DeepSeek-V4 DSpark draft execution must be eager or graph"
            )
        if dspark_draft_execution == "graph" and execution_mode != "full-decode-only":
            raise ValueError(
                "DeepSeek-V4 DSpark draft Graph requires target full-decode-only"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN,
    )
    parser.add_argument("--attention-port", type=int, default=8910)
    parser.add_argument("--ffn-port", type=int, default=8911)
    parser.add_argument("--afd-port", type=int, default=29761)
    parser.add_argument(
        "--connector",
        choices=("CAMP2pAFDConnector", "P2pHcclAFDConnector"),
        required=True,
    )
    parser.add_argument("--startup-timeout", type=float, default=3600)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--idle-seconds", type=int, default=1800)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 8, 32])
    parser.add_argument("--prompt-indices", type=int, nargs="*")
    parser.add_argument(
        "--functional-smoke",
        action="store_true",
        help="Check HTTP batches and cancellation recovery without golden tokens.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "full-decode-only"),
        default="eager",
    )
    parser.add_argument("--u-batches", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--async-scheduling",
        choices=("auto", "on", "off"),
        default=os.environ.get("AFD_ASYNC_SCHEDULING", "auto"),
    )
    parser.add_argument(
        "--eager-u2-stream-overlap",
        choices=("on", "off"),
        default=_toggle_default("AFD_HCCL_EAGER_U2_STREAM_OVERLAP"),
        help="Toggle eager U2 HCCL communication streams without disabling U2.",
    )
    parser.add_argument(
        "--graph-u2-compute-overlap",
        choices=("on", "off"),
        default=_toggle_default("AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"),
        help="Toggle Graph U2 side-compute streams without disabling U2.",
    )
    parser.add_argument(
        "--graph-u2-hybrid-dag",
        choices=("on", "off"),
        default=_toggle_default("AFD_HCCL_GRAPH_U2_HYBRID_DAG"),
        help="Toggle per-stage Graph U2 receive dependencies.",
    )
    parser.add_argument(
        "--graph-u2-attention-three-stream",
        choices=("on", "off"),
        default=_toggle_default("AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM"),
        help="Toggle dedicated Attention Graph send/receive streams.",
    )
    parser.add_argument(
        "--graph-u2-ffn-recv-stream",
        choices=("on", "off"),
        default=_toggle_default("AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM"),
        help="Toggle the FFN Graph receive stream.",
    )
    parser.add_argument(
        "--graph-u2-ffn-cross-layer",
        choices=("on", "off"),
        default=_toggle_default("AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER"),
        help="Toggle FFN cross-layer receive overlap.",
    )
    parser.add_argument(
        "--stage-diagnostics",
        choices=("on", "off"),
        default="off",
        help="Emit first/last-layer U2 stage progress markers.",
    )
    parser.add_argument("--dbo-decode-token-threshold", type=int, default=2)
    parser.add_argument("--dbo-prefill-token-threshold", type=int, default=12)
    parser.add_argument(
        "--attention-devices",
        type=_parse_device_list,
        default=list(range(8)),
    )
    parser.add_argument(
        "--ffn-devices",
        type=_parse_device_list,
        default=list(range(8, 16)),
    )
    parser.add_argument(
        "--attention-max-num-batched-tokens",
        type=int,
        default=1024,
    )
    parser.add_argument("--ffn-max-num-batched-tokens", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect plugin-owned Attention and FFN torch-npu traces.",
    )
    parser.add_argument(
        "--enable-mtp",
        action="store_true",
        default=os.environ.get("ENABLE_MTP", "0") == "1",
    )
    parser.add_argument(
        "--mtp-num-speculative-tokens",
        type=int,
        choices=(1, 2, 3),
        default=int(os.environ.get("MTP_NUM_SPECULATIVE_TOKENS", "1")),
    )
    parser.add_argument(
        "--mtp-draft-execution",
        choices=("eager", "graph"),
        default=os.environ.get("MTP_DRAFT_EXECUTION", "eager"),
    )
    parser.add_argument(
        "--enable-dspark",
        action="store_true",
        default=os.environ.get("ENABLE_DSPARK", "0") == "1",
    )
    parser.add_argument(
        "--dspark-num-speculative-tokens",
        default=os.environ.get("DSPARK_NUM_SPECULATIVE_TOKENS", "auto"),
    )
    parser.add_argument(
        "--dspark-draft-execution",
        choices=("eager", "graph"),
        default=os.environ.get("DSPARK_DRAFT_EXECUTION", "eager"),
    )
    args = parser.parse_args()

    if args.dbo_decode_token_threshold < 0:
        parser.error("--dbo-decode-token-threshold must be non-negative")
    if args.dbo_prefill_token_threshold < 0:
        parser.error("--dbo-prefill-token-threshold must be non-negative")
    dspark_num_speculative_tokens = 0
    try:
        if args.enable_dspark:
            model_path = Path(os.environ.get("MODEL_PATH", ""))
            dspark_num_speculative_tokens = _resolve_dspark_num_speculative_tokens(
                model_path,
                args.dspark_num_speculative_tokens,
            )
        topology = _resolve_topology(
            connector=args.connector,
            attention_devices=args.attention_devices,
            ffn_devices=args.ffn_devices,
            attention_max_num_batched_tokens=(args.attention_max_num_batched_tokens),
            ffn_max_num_batched_tokens=args.ffn_max_num_batched_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        _validate_execution_topology(
            connector=args.connector,
            execution_mode=args.execution_mode,
            u_batches=args.u_batches,
            async_scheduling=args.async_scheduling,
            enable_mtp=args.enable_mtp,
            mtp_num_speculative_tokens=args.mtp_num_speculative_tokens,
            mtp_draft_execution=args.mtp_draft_execution,
            enable_dspark=args.enable_dspark,
            dspark_num_speculative_tokens=dspark_num_speculative_tokens,
            dspark_draft_execution=args.dspark_draft_execution,
            topology=topology,
        )
    except ValueError as exc:
        parser.error(str(exc))
    _set_topology_environment(topology)
    _set_execution_environment(
        async_scheduling=args.async_scheduling,
        eager_u2_stream_overlap=args.eager_u2_stream_overlap,
        graph_u2_compute_overlap=args.graph_u2_compute_overlap,
        graph_u2_hybrid_dag=args.graph_u2_hybrid_dag,
        graph_u2_attention_three_stream=args.graph_u2_attention_three_stream,
        graph_u2_ffn_recv_stream=args.graph_u2_ffn_recv_stream,
        graph_u2_ffn_cross_layer=args.graph_u2_ffn_cross_layer,
        stage_diagnostics=args.stage_diagnostics,
    )
    _set_mtp_environment(
        enable_mtp=args.enable_mtp,
        mtp_num_speculative_tokens=args.mtp_num_speculative_tokens,
        mtp_draft_execution=args.mtp_draft_execution,
    )
    _set_dspark_environment(
        enable_dspark=args.enable_dspark,
        dspark_num_speculative_tokens=dspark_num_speculative_tokens,
        dspark_draft_execution=args.dspark_draft_execution,
    )

    for port in (args.attention_port, args.ffn_port, args.afd_port):
        if not _port_is_free(port):
            raise RuntimeError(f"port {port} is already in use")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runtime.json").write_text(
        json.dumps(
            _runtime_manifest(
                connector=args.connector,
                execution_mode=args.execution_mode,
                u_batches=args.u_batches,
                dbo_decode_token_threshold=args.dbo_decode_token_threshold,
                dbo_prefill_token_threshold=args.dbo_prefill_token_threshold,
                profile=args.profile,
                async_scheduling=args.async_scheduling,
                eager_u2_stream_overlap=args.eager_u2_stream_overlap,
                graph_u2_compute_overlap=args.graph_u2_compute_overlap,
                graph_u2_hybrid_dag=args.graph_u2_hybrid_dag,
                graph_u2_attention_three_stream=args.graph_u2_attention_three_stream,
                graph_u2_ffn_recv_stream=args.graph_u2_ffn_recv_stream,
                graph_u2_ffn_cross_layer=args.graph_u2_ffn_cross_layer,
                stage_diagnostics=args.stage_diagnostics,
                enable_mtp=args.enable_mtp,
                mtp_num_speculative_tokens=args.mtp_num_speculative_tokens,
                mtp_draft_execution=args.mtp_draft_execution,
                enable_dspark=args.enable_dspark,
                dspark_num_speculative_tokens=dspark_num_speculative_tokens,
                dspark_draft_execution=args.dspark_draft_execution,
                topology=topology,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    cycles = []
    overall_passed = False
    try:
        for cycle_idx in range(1, args.cycles + 1):
            cycle_dir = args.output_dir / f"cycle_{cycle_idx}"
            cycle_dir.mkdir(parents=True, exist_ok=True)
            processes: dict[str, subprocess.Popen[bytes]] = {}
            handles = []
            cycle_result: dict[str, Any] = {
                "cycle": cycle_idx,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            startup_started = time.monotonic()
            profile_dir = cycle_dir / "profiles" if args.profile else None
            if profile_dir is not None:
                cycle_result["profile"] = {
                    "enabled": True,
                    "root": str(profile_dir),
                    "wait": 2,
                    "warmup": 1,
                    "active": 10,
                    "role_ranks": [0],
                    "with_stack": False,
                }
            try:
                ffn_process, ffn_handle = _start_role(
                    "ffn",
                    output_dir=cycle_dir,
                    api_port=args.ffn_port,
                    afd_port=args.afd_port,
                    connector=args.connector,
                    execution_mode=args.execution_mode,
                    u_batches=args.u_batches,
                    dbo_decode_token_threshold=args.dbo_decode_token_threshold,
                    dbo_prefill_token_threshold=args.dbo_prefill_token_threshold,
                    profile_dir=profile_dir,
                )
                processes["ffn"] = ffn_process
                handles.append(ffn_handle)
                time.sleep(2)
                attention_process, attention_handle = _start_role(
                    "attention",
                    output_dir=cycle_dir,
                    api_port=args.attention_port,
                    afd_port=args.afd_port,
                    connector=args.connector,
                    execution_mode=args.execution_mode,
                    u_batches=args.u_batches,
                    dbo_decode_token_threshold=args.dbo_decode_token_threshold,
                    dbo_prefill_token_threshold=args.dbo_prefill_token_threshold,
                    profile_dir=profile_dir,
                )
                processes["attention"] = attention_process
                handles.append(attention_handle)
                _wait_for_api(
                    f"http://127.0.0.1:{args.attention_port}/v1/models",
                    processes,
                    cycle_dir,
                    args.startup_timeout,
                )
                cycle_result["startup_seconds"] = round(
                    time.monotonic() - startup_started, 3
                )
                _capture_command(["npu-smi", "info"], cycle_dir / "npu_ready.txt")
                if args.functional_smoke:
                    _run_functional_validator(
                        api_port=args.attention_port,
                        output=cycle_dir / "functional_smoke.json",
                        batch_sizes=args.batch_sizes,
                    )
                    cancellation_gate = _run_cancellation_gate(
                        api_port=args.attention_port,
                        output_dir=cycle_dir,
                    )
                    cycle_result["cancellation_gate"] = cancellation_gate
                    if not cancellation_gate["passed"]:
                        raise RuntimeError(
                            "cancellation request did not time out with curl exit 28"
                        )
                    cancellation_quiescence_gate = _wait_for_request_quiescence(
                        api_port=args.attention_port,
                        output_dir=cycle_dir,
                        artifact_stem="cancellation_quiescence",
                    )
                    cycle_result["cancellation_quiescence_gate"] = (
                        cancellation_quiescence_gate
                    )
                    if not cancellation_quiescence_gate["passed"]:
                        raise RuntimeError(
                            "request queues did not become stably idle after "
                            "cancellation"
                        )
                    _run_functional_validator(
                        api_port=args.attention_port,
                        output=cycle_dir / "recovery.json",
                        batch_sizes=[1],
                    )
                    quiescence_gate = _wait_for_request_quiescence(
                        api_port=args.attention_port,
                        output_dir=cycle_dir,
                        artifact_stem="request_quiescence",
                    )
                    cycle_result["request_quiescence_gate"] = quiescence_gate
                    if not quiescence_gate["passed"]:
                        raise RuntimeError(
                            "request queues did not become stably idle after recovery"
                        )
                else:
                    _run_validator(
                        api_port=args.attention_port,
                        golden=args.golden,
                        output=cycle_dir / "golden.json",
                        rounds=args.rounds,
                        batch_sizes=args.batch_sizes,
                        prompt_indices=args.prompt_indices,
                    )
                    if cycle_idx == 1 and args.idle_seconds > 0:
                        cycle_result["idle_seconds"] = args.idle_seconds
                        time.sleep(args.idle_seconds)
                        _run_validator(
                            api_port=args.attention_port,
                            golden=args.golden,
                            output=cycle_dir / "idle_resume.json",
                            rounds=1,
                            batch_sizes=[1],
                            prompt_indices=args.prompt_indices,
                        )
                if args.enable_dspark:
                    dspark_gate = _dspark_execution_gate(
                        api_port=args.attention_port,
                        output_dir=cycle_dir,
                        expected_attention_ranks=topology["attention_ranks"],
                    )
                    cycle_result["dspark_gate"] = dspark_gate
                    if not dspark_gate["passed"]:
                        raise RuntimeError(
                            "DSpark drafter or speculative token metric gate failed"
                        )
                cycle_result["passed"] = True
            finally:
                cycle_result["shutdown"] = _shutdown_roles(
                    processes,
                    ffn_log_path=cycle_dir / "ffn.log",
                    expected_ffn_shutdown_receipts=topology["ffn_ranks"],
                )
                for handle in handles:
                    handle.close()
                cycle_result["log_gate"] = _role_log_gate(cycle_dir)
                cycle_result["ubatch_gate"] = _ubatch_execution_gate(
                    cycle_dir,
                    args.u_batches,
                    enable_mtp=args.enable_mtp,
                    batch_sizes=args.batch_sizes,
                    data_parallel_size=topology["attention_data_parallel_size"],
                )
                profile_passed = True
                if profile_dir is not None:
                    profile_gate = _profile_output_gate(profile_dir)
                    cycle_result["profile"]["output_gate"] = profile_gate
                    profile_passed = profile_gate["passed"]
                cycle_result["passed"] = bool(
                    cycle_result.get("passed", False)
                    and cycle_result["shutdown"]["passed"]
                    and cycle_result["log_gate"]["passed"]
                    and cycle_result["ubatch_gate"]["passed"]
                    and cycle_result.get("request_quiescence_gate", {}).get(
                        "passed", True
                    )
                    and cycle_result.get("cancellation_quiescence_gate", {}).get(
                        "passed", True
                    )
                    and cycle_result.get("dspark_gate", {}).get("passed", True)
                    and profile_passed
                )
                cycle_result["npu_cleanup_gate"] = _wait_for_npu_cleanup(
                    cycle_dir / "npu_after_cleanup.txt"
                )
                cycle_result["passed"] = bool(
                    cycle_result["passed"]
                    and cycle_result["npu_cleanup_gate"]["passed"]
                )
                cycle_result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                cycles.append(cycle_result)
                (cycle_dir / "cycle_summary.json").write_text(
                    json.dumps(cycle_result, indent=2, sort_keys=True) + "\n"
                )
        overall_passed = all(cycle.get("passed", False) for cycle in cycles)
    finally:
        summary = {
            "passed": overall_passed,
            "cycles": cycles,
            "validation_mode": (
                "functional_smoke" if args.functional_smoke else "golden_exact"
            ),
            "golden_checked": not args.functional_smoke,
            "golden": None if args.functional_smoke else str(args.golden),
            "execution_mode": args.execution_mode,
            "connector": args.connector,
            "u_batches": args.u_batches,
            "dbo_decode_token_threshold": args.dbo_decode_token_threshold,
            "dbo_prefill_token_threshold": args.dbo_prefill_token_threshold,
            "async_scheduling": args.async_scheduling,
            "eager_u2_stream_overlap": args.eager_u2_stream_overlap,
            "graph_u2_compute_overlap": args.graph_u2_compute_overlap,
            "graph_u2_hybrid_dag": args.graph_u2_hybrid_dag,
            "graph_u2_attention_three_stream": args.graph_u2_attention_three_stream,
            "graph_u2_ffn_recv_stream": args.graph_u2_ffn_recv_stream,
            "graph_u2_ffn_cross_layer": args.graph_u2_ffn_cross_layer,
            "stage_diagnostics": args.stage_diagnostics,
            "profile": args.profile,
            "enable_mtp": args.enable_mtp,
            "mtp_num_speculative_tokens": args.mtp_num_speculative_tokens,
            "mtp_draft_execution": (
                args.mtp_draft_execution if args.enable_mtp else None
            ),
            "enable_dspark": args.enable_dspark,
            "dspark_num_speculative_tokens": (
                dspark_num_speculative_tokens if args.enable_dspark else None
            ),
            "dspark_draft_execution": (
                args.dspark_draft_execution if args.enable_dspark else None
            ),
            "topology": topology,
        }
        (args.output_dir / "validation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
