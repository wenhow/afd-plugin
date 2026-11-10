# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded real role-profiler lifecycle probe, without model weights or HCCL.

The CPU supervisor runs Attention-main and FFN-background-owner cases serially.
Only the role worker imports torch. This validates the plugin profiler session,
controlled NPU matmul and exported traces; it does not start a native runner,
HTTP/RPC server, model, graph, or inference benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import hccl_eager_backend_probe as supervision

EXPECTED_IDLE_BOARDS = 8
TOTAL_TIMEOUT_SECONDS = 300
OWNER_QUEUE_TIMEOUT_SECONDS = 90
REAP_TIMEOUT_SECONDS = 10
SUPERVISOR_POLL_SECONDS = 0.1
NPU_SMI_TIMEOUT_SECONDS = 30
PROFILE_STEPS = 3
ACTIVE_PROFILE_STEPS = 2
MATRIX_SIZE = 64
INPUT_PERIOD = 7
WEIGHT_PERIOD = 5
READ_CHUNK_BYTES = 1_048_576
INSTALLED_PROFILER_SHA256 = (
    "1a04902572383b74a401449e73929b2c4ce728d6c81bb59341876b03c7a82168"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--role-rank", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=TOTAL_TIMEOUT_SECONDS)
    parser.add_argument("--role", choices=("attention", "ffn"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device < 0 or args.role_rank < 0:
        parser.error("device and role-rank must be nonnegative")
    if not REAP_TIMEOUT_SECONDS < args.timeout <= TOTAL_TIMEOUT_SECONDS:
        parser.error("timeout must fit the bounded supervisor interval")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profiler_environment(args, role):
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, env.get("PYTHONPATH")) if part
    )
    prefix = f"AFD_NPU_{role.upper()}_PROFILER"
    env.update(
        {
            f"{prefix}_ENABLE": "1",
            f"{prefix}_WAIT": "0",
            f"{prefix}_WARMUP": "1",
            f"{prefix}_ACTIVE": str(ACTIVE_PROFILE_STEPS),
            f"{prefix}_REPEAT": "1",
            f"{prefix}_SKIP_FIRST": "0",
            f"{prefix}_WITH_STACK": "0",
            f"{prefix}_RANKS": str(args.role_rank),
            f"{prefix}_DIR": str(args.output_dir / role / "traces"),
        }
    )
    return env


def source_preflight():
    if "torch" in sys.modules or "torch_npu" in sys.modules:
        raise RuntimeError("CPU supervisor must not import torch or torch_npu")
    distribution = importlib.metadata.distribution("torch-npu")
    installed = Path(distribution.locate_file("torch_npu/profiler/profiler.py"))
    actual = file_sha256(installed)
    if actual != INSTALLED_PROFILER_SHA256:
        raise RuntimeError("installed profiler source changed; review export API first")
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        root / "afd_plugin/compat/npu/profiler.py",
        Path(supervision.__file__).resolve(),
        Path(supervision.helpers.__file__).resolve(),
    )
    return {
        "torch_imported": False,
        "torch_npu_imported": False,
        "torch_npu_version": distribution.version,
        "installed_profiler_path": str(installed),
        "installed_profiler_sha256": actual,
        "trace_handler_default_async_mode": False,
        "trace_export_boundary": (
            "installed handler calls analyse(async_mode=False); only actual parsed "
            "trace files qualify export, absence remains pending"
        ),
        "source_sha256": {
            str(path.relative_to(root)): file_sha256(path) for path in paths
        },
    }


def sample_idle(output_dir, name):
    result = subprocess.run(
        ["npu-smi", "info"],
        capture_output=True,
        text=True,
        timeout=NPU_SMI_TIMEOUT_SECONDS,
        check=False,
    )
    (output_dir / f"npu-{name}.txt").write_text(
        result.stdout + result.stderr, encoding="utf-8"
    )
    idle = result.stdout.count("No running processes found in NPU")
    if result.returncode != 0 or idle != EXPECTED_IDLE_BOARDS:
        raise RuntimeError(f"eight idle boards required at {name}; observed {idle}")
    return idle


def record_status(report, event, session):
    snapshot = session.status()
    report["status_records"].append(
        {"event": event, "caller_thread_id": threading.get_ident(), **snapshot}
    )
    return snapshot


def matmul_state(device):
    # Runtime imports are deliberately confined to the isolated device owner.
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device)
    values = torch.arange(MATRIX_SIZE * MATRIX_SIZE, dtype=torch.int64)
    left = ((values % INPUT_PERIOD) - INPUT_PERIOD // 2).float()
    right = ((values % WEIGHT_PERIOD) - WEIGHT_PERIOD // 2).float()
    left, right = (
        left.reshape(MATRIX_SIZE, MATRIX_SIZE),
        right.reshape(MATRIX_SIZE, MATRIX_SIZE),
    )
    expected = left @ right
    tensors = (left.to(f"npu:{device}"), right.to(f"npu:{device}"), expected)
    torch.testing.assert_close(tensors[0] @ tensors[1], expected.npu(), rtol=0, atol=0)
    torch.npu.synchronize()
    return torch, tensors


def controlled_step(torch, tensors, report, phase):
    left, right, expected = tensors
    with torch.autograd.profiler.record_function(f"afd_profiler_probe/{phase}"):
        actual = left @ right
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    report["matmul_checks"].append(
        {"phase": phase, "exact_cpu_oracle": True, "shape": list(actual.shape)}
    )


def run_attention(args, session, report):
    from afd_plugin.compat.npu.profiler import (
        afd_npu_profiler_execute,
        afd_npu_profiler_owner,
    )

    torch, tensors = matmul_state(args.device)
    with torch.inference_mode(), afd_npu_profiler_owner(session):
        for index in range(PROFILE_STEPS):
            with afd_npu_profiler_execute(session):
                controlled_step(torch, tensors, report, f"initial-{index}")
        initial = record_status(report, "initial-running", session)
        assert initial["applied"] and initial["steps"] == PROFILE_STEPS
        session.request(False)
        pending = record_status(report, "stop-requested", session)
        assert pending["pending"] and pending["applied"]
        with afd_npu_profiler_execute(session):
            controlled_step(torch, tensors, report, "stop-boundary")
        stopped = record_status(report, "stop-applied", session)
        assert not stopped["pending"] and not stopped["applied"]
        session.request(True, "restart")
        pending = record_status(report, "restart-requested", session)
        assert pending["pending"] and not pending["applied"]
        for index in range(PROFILE_STEPS):
            with afd_npu_profiler_execute(session):
                controlled_step(torch, tensors, report, f"restart-{index}")
        record_status(report, "restart-running", session)
    closed = record_status(report, "owner-finished", session)
    assert closed["closed"] and not closed["failed"] and not closed["pending"]
    assert closed["steps"] == PROFILE_STEPS * 2
    assert closed["owner_thread_id"] == threading.get_ident()


def run_ffn(args, session, report):
    from afd_plugin.compat.npu.profiler import afd_npu_profiler_owner

    commands, replies = queue.Queue(), queue.Queue()

    def owner():
        try:
            torch, tensors = matmul_state(args.device)
            with torch.inference_mode(), afd_npu_profiler_owner(session):
                replies.put(("ready", session.status()))
                while True:
                    phase, steps = commands.get(timeout=OWNER_QUEUE_TIMEOUT_SECONDS)
                    if phase == "finish":
                        break
                    for index in range(steps):
                        session.before_step()
                        controlled_step(torch, tensors, report, f"{phase}-{index}")
                        session.step()
                    replies.put(("ready", session.status()))
            replies.put(("closed", session.status()))
        except BaseException:
            replies.put(("error", traceback.format_exc()))

    def receive(expected):
        event, value = replies.get(timeout=OWNER_QUEUE_TIMEOUT_SECONDS)
        if event == "error":
            raise RuntimeError(value)
        if event != expected:
            raise AssertionError(f"expected {expected}, received {event}")
        return value

    thread = threading.Thread(
        target=owner, name="afd-profiler-probe-owner", daemon=True
    )
    thread.start()
    receive("ready")
    commands.put(("initial", PROFILE_STEPS))
    initial = receive("ready")
    assert initial["applied"] and initial["steps"] == PROFILE_STEPS
    assert initial["owner_thread_id"] == thread.ident != threading.get_ident()
    record_status(report, "owner-awaiting-next-command", session)
    # The owner cannot reach before_step again until management releases a
    # command. No sleep or elapsed-time assertion stands in for this handshake.
    session.request(False)
    pending = record_status(report, "queued-stop-pending", session)
    assert pending["pending"] and pending["applied"]
    assert pending["steps"] == initial["steps"]
    commands.put(("stop-boundary", 1))
    stopped = receive("ready")
    assert not stopped["pending"] and not stopped["applied"]
    assert stopped["steps"] == PROFILE_STEPS
    record_status(report, "queued-stop-applied", session)
    session.request(True, "restart")
    pending = record_status(report, "queued-restart-pending", session)
    assert pending["pending"] and not pending["applied"]
    assert pending["steps"] == PROFILE_STEPS
    commands.put(("restart", PROFILE_STEPS))
    restarted = receive("ready")
    assert restarted["applied"] and not restarted["pending"]
    assert restarted["steps"] == PROFILE_STEPS * 2
    record_status(report, "queued-restart-applied", session)
    commands.put(("finish", 0))
    closed = receive("closed")
    thread.join(timeout=REAP_TIMEOUT_SECONDS)
    assert not thread.is_alive()
    assert closed["closed"] and not closed["failed"] and not closed["pending"]
    assert closed["owner_thread_id"] == thread.ident
    record_status(report, "owner-finished", session)


def inspect_trace(trace_path, kernel_csv, device):
    report = {"npu_matmul_kernel_events": 0, "csv_matches_trace": False}
    try:
        content = json.loads(trace_path.read_text(encoding="utf-8"))
        events = content["traceEvents"] if isinstance(content, dict) else content
        report["events"] = len(events)
        hardware_names = {
            event["pid"]
            for event in events
            if event.get("name") == "process_name"
            and event.get("args", {}).get("name") == "Ascend Hardware"
        }
        hardware_labels = {
            event["pid"]
            for event in events
            if event.get("name") == "process_labels"
            and event.get("args", {}).get("labels") == f"NPU {device}"
        }
        hardware_ids = hardware_names & hardware_labels
        kernels = [
            event
            for event in events
            if event.get("pid") in hardware_ids
            and event.get("ph") == "X"
            and event.get("args", {}).get("Task Type") == "AI_CORE"
            and "matmul" in event.get("name", "").casefold()
        ]
        rows = []
        if kernel_csv.exists():
            with kernel_csv.open(encoding="utf-8-sig", newline="") as handle:
                rows = [
                    row
                    for row in csv.DictReader(handle)
                    if row["Device_id"] == str(device)
                    and row["Accelerator Core"] == "AI_CORE"
                    and "matmul" in row["Name"].casefold()
                ]
        trace_tasks = Counter(
            (event["name"], int(event["args"]["Task Id"]), int(event["tid"]))
            for event in kernels
        )
        csv_tasks = Counter(
            (row["Name"], int(row["Task ID"]), int(row["Stream ID"])) for row in rows
        )
        report.update(
            hardware_process_ids=sorted(hardware_ids),
            npu_matmul_kernel_events=len(kernels),
            kernel_examples=kernels[:2],
            csv_kernel_rows=rows,
            csv_matches_trace=bool(trace_tasks) and trace_tasks == csv_tasks,
        )
    except (
        OSError,
        csv.Error,
        UnicodeError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ) as error:
        report["parse_error"] = f"{type(error).__name__}: {error}"
    return report


def collect_traces(args):
    trace_dir = args.output_dir / "traces"
    files = {
        str(path.relative_to(trace_dir)): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(trace_dir.rglob("*"))
        if path.is_file()
    }
    traces = []
    for path in sorted(trace_dir.rglob("trace_view.json")):
        traces.append(
            {
                "path": str(path.relative_to(trace_dir)),
                **inspect_trace(
                    path, path.with_name("kernel_details.csv"), args.device
                ),
            }
        )
    expected_names = (
        f"{args.role}_rank{args.role_rank}_",
        f"restart_{args.role}_rank{args.role_rank}_",
    )
    matching = {
        name: [
            trace["path"]
            for trace in traces
            if Path(trace["path"]).parts[0].startswith(name)
            and trace.get("events", 0) > 0
            and trace["npu_matmul_kernel_events"] == ACTIVE_PROFILE_STEPS
            and trace["csv_matches_trace"]
            and "parse_error" not in trace
        ]
        for name in expected_names
    }
    return {
        "trace_root": str(trace_dir.resolve()),
        "files": files,
        "parsed_traces": traces,
        "role_rank_prefix_matches": matching,
        "state": "complete" if all(matching.values()) else "pending-or-unproven",
    }


def worker(args):
    from afd_plugin.compat.npu.profiler import create_deferred_afd_npu_profiler

    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "real role profiler session and controlled matmul; no native runner",
        "role": args.role,
        "role_rank": args.role_rank,
        "device": args.device,
        "management_thread_id": threading.get_ident(),
        "status_records": [],
        "matmul_checks": [],
        "success": False,
    }
    session = None
    try:
        session = create_deferred_afd_npu_profiler(args.role, role_rank=args.role_rank)
        if session is None:
            raise RuntimeError("explicit role profiler was not enabled")
        record_status(report, "deferred-before-owner", session)
        run = run_attention if args.role == "attention" else run_ffn
        run(args, session, report)
        report["session_status"] = session.status()
        report["traces"] = collect_traces(args)
        report["success"] = report["traces"]["state"] == "complete"
    except BaseException:
        report["error"] = traceback.format_exc()
        if session is not None:
            report["session_status"] = session.status()
        raise
    finally:
        write_json(args.output_dir / "role.json", report)
    return 0 if report["success"] else 2


def supervise(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "scope": "real profiler session / controlled NPU matmul, no checkpoint or HCCL",
        "success": False,
        "device": args.device,
        "role_rank": args.role_rank,
        "roles": {},
        "owned_pid_start_ticks": {},
        "killed_owned_pids": [],
    }
    started = time.monotonic()
    children = []
    try:
        summary["source_preflight"] = source_preflight()
        write_json(
            args.output_dir / "source-preflight.json", summary["source_preflight"]
        )
        summary["idle_boards_before"] = sample_idle(args.output_dir, "before")
        deadline = started + args.timeout - REAP_TIMEOUT_SECONDS
        for role in ("attention", "ffn"):
            with (args.output_dir / f"{role}.log").open("w") as log:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        str(Path(__file__).resolve()),
                        "--output-dir",
                        str(args.output_dir / role),
                        "--role",
                        role,
                        "--role-rank",
                        str(args.role_rank),
                        "--device",
                        str(args.device),
                    ],
                    env=profiler_environment(args, role),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                children.append(child)
                while child.poll() is None:
                    supervision.observe_owned(
                        child.pid, summary["owned_pid_start_ticks"]
                    )
                    if time.monotonic() >= deadline:
                        summary["timed_out"] = True
                        raise TimeoutError(
                            "role profiler exceeded the bounded deadline"
                        )
                    time.sleep(SUPERVISOR_POLL_SECONDS)
            report_path = args.output_dir / role / "role.json"
            summary["roles"][role] = {
                "returncode": child.returncode,
                "report": json.loads(report_path.read_text())
                if report_path.exists()
                else None,
            }
            summary[f"idle_boards_after_{role}"] = sample_idle(args.output_dir, role)
            if child.returncode != 0:
                raise RuntimeError(f"{role} failed or its trace remains unproven")
        summary["success"] = all(
            data["report"] is not None and data["report"]["success"]
            for data in summary["roles"].values()
        )
    except BaseException:
        summary["error"] = traceback.format_exc()
    finally:
        if any(child.poll() is None for child in children):
            summary["killed_owned_pids"] = supervision.kill_owned(
                summary["owned_pid_start_ticks"]
            )
        for child in children:
            child.wait(timeout=REAP_TIMEOUT_SECONDS)
        try:
            summary["idle_boards_after"] = sample_idle(args.output_dir, "after")
        except BaseException:
            summary["postcheck_error"] = traceback.format_exc()
            summary["success"] = False
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output_dir / "result.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(worker(parsed) if parsed.role else supervise(parsed))
