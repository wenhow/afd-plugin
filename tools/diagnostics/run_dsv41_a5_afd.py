#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Preflight, or explicitly run one bounded A5 AFD eager/U1 request.

The default only inspects source/checkpoint files. --run launches independent
A4/F4 TP4 worlds, sends one request, then stops A before waiting for peer STOP
and F teardown. This is a qualification candidate, not a support declaration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import IO

from preflight_dsv41_a5_afd import JSONValue, collect_preflight

API_PORT = 18031
AFD_PORT = 29600
F_READY = "AFD FFN EngineCore started; workers run connector loop."
PEER_STOP = "AFD HCCL FFN peer STOP complete."
ENGINE_CLEAN = "AFD HCCL FFN engine cleanup complete."
F_EXIT = "AFD FFN launcher exited with all engine exit codes zero."
WORKER_CLEAN = r"AFD HCCL FFN worker cleanup complete; rank=(\d+)\."
FORCED_NATIVE = (
    "workers still running after grace period",
    "Process manager: force killing",
    "after SIGTERM",
)
REQUEST_TOKENS = 32
SHUTDOWN_SECONDS = 180
OUTER_SHUTDOWN_SECONDS = 240
REQUEST_TIMEOUT_SECONDS = 300
PACKED_ATTENTION_PROBE_SECONDS = 180
POLL_SECONDS = 1
TERM_GRACE_SECONDS = 10
KILL_GRACE_SECONDS = 10
MIN_HOST_BYTES = int(188.833133 * 1024**3)
PLUGINS = (
    "ascend,ascend_model,ascend_model_loader,ascend_kv_connector,"
    "ascend_service_profiling,afd"
)


class TerminationRequestedError(Exception):
    """Preserve SIGTERM's shell exit code while releasing owned processes."""


def handle_termination(signum: int, _frame: FrameType | None) -> None:
    raise TerminationRequestedError(signum)


def save_json(path: Path, value: JSONValue) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_plan(model: Path, vendor: Path, repo: Path) -> dict[str, JSONValue]:
    common = [
        "--tensor-parallel-size",
        "4",
        "--pipeline-parallel-size",
        "1",
        "--data-parallel-size",
        "1",
        "--data-parallel-size-local",
        "1",
        "--prefill-context-parallel-size",
        "1",
        "--decode-context-parallel-size",
        "1",
        "--distributed-executor-backend",
        "mp",
        "--no-enable-expert-parallel",
        "--no-enable-dbo",
        "--ubatch-size",
        "0",
        "--language-model-only",
        "--dtype",
        "bfloat16",
        "--quantization",
        "deepseek_v4_fp8",
        "--load-format",
        "safetensors",
        "--tokenizer-mode",
        "deepseek_v41",
        "--trust-remote-code",
        "--block-size",
        "128",
        "--engram-config.cpu_offload",
        "true",
        "--engram-config.embedding_across_dp",
        "false",
        "--engram-config.dp_shared_memory",
        "false",
        "--enforce-eager",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "1024",
        "--gpu-memory-utilization",
        "0.90",
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--generation-config",
        "vllm",
        "--compilation-config",
        json.dumps(
            {"mode": 0, "cudagraph_mode": "NONE", "pass_config": {"enable_sp": False}}
        ),
        "--shutdown-timeout",
        str(SHUTDOWN_SECONDS),
    ]
    environment = {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "AFD_DSV41_EAGER_QUALIFICATION": "1",
        "AFD_DSV41_MXFP8_SCALE_PATCH": "0",
        "VLLM_PLUGINS": PLUGINS,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": str(SHUTDOWN_SECONDS),
        "HCCL_OP_EXPANSION_MODE": "CCU_SCHED",
        "HCCL_BUFFSIZE": "1600",
        "HCCL_CONNECT_TIMEOUT": "1200",
        "HCCL_EXEC_TIMEOUT": "2040",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "3000",
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": "10",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "PYTHONUNBUFFERED": "1",
        "ASCEND_CUSTOM_OPP_PATH": str(vendor)
        + (
            ":" + os.environ["ASCEND_CUSTOM_OPP_PATH"]
            if os.environ.get("ASCEND_CUSTOM_OPP_PATH")
            else ""
        ),
        "LD_LIBRARY_PATH": str(vendor / "op_api/lib")
        + (
            ":" + os.environ["LD_LIBRARY_PATH"]
            if os.environ.get("LD_LIBRARY_PATH")
            else ""
        ),
    }
    roles: dict[str, JSONValue] = {}
    for role, devices, master, rpc, native_port in (
        ("attention", "0,1,2,3", 29501, 29550, 29700),
        ("ffn", "4,5,6,7", 29511, 29560, 29800),
    ):
        additional = {
            "afd": {
                "role": role,
                "connector": "P2pHcclAFDConnector",
                "host": "127.0.0.1",
                "port": AFD_PORT,
                "num_attention_ranks": 4,
                "num_ffn_ranks": 4,
                "compute_gate_on_attention": False,
                "afd_process_group_timeout_s": 1200,
                "connector_extra_config": {"max_tokens": 1024},
            },
            "enable_cpu_binding": False,
            "multistream_engram_overlap": False,
            "multistream_overlap_shared_expert": False,
            "multistream_dsv4_dsa_overlap": False,
            "combine_quant_mode": 4,
            "enable_flashcomm1": False,
            "enable_dsa_cp": False,
        }
        entry = (
            [
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                str(model),
                "--host",
                "127.0.0.1",
                "--port",
                str(API_PORT),
                "--served-model-name",
                "dsv41-afd",
            ]
            if role == "attention"
            else ["-m", "afd_plugin.entrypoints.npu_ffn", "--model", str(model)]
        )
        roles[role] = {
            "argv": [
                sys.executable,
                "-B",
                *entry,
                *common,
                "--master-port",
                str(master),
                "--data-parallel-rpc-port",
                str(rpc),
                "--additional-config",
                json.dumps(additional),
            ],
            "environment": {
                **environment,
                "ASCEND_RT_VISIBLE_DEVICES": devices,
                "VLLM_PORT": str(native_port),
            },
        }
    return {
        "created_utc": utc_now(),
        "repo_root": str(repo),
        "roles": roles,
        "scope": "A5 A4/F4 TP4 eager/U1, DSpark off, one 32-token request",
        "physical_devices": list(range(8)),
        "full_model_qualification": False,
        "ports": [
            API_PORT,
            *range(AFD_PORT, AFD_PORT + 4),
            29501,
            29511,
            29550,
            29560,
            29700,
            29800,
        ],
    }


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    parent: int
    group: int
    session: int
    started: int
    state: str


def process_table() -> dict[int, ProcessIdentity]:
    table = {}
    for folder in Path("/proc").iterdir():
        if not folder.name.isdecimal():
            continue
        try:
            raw = (folder / "stat").read_text()
            # The comm field may contain spaces and parentheses.
            fields = raw[raw.rindex(")") + 2 :].split()
            pid = int(folder.name)
            table[pid] = ProcessIdentity(
                pid,
                int(fields[1]),
                int(fields[2]),
                int(fields[3]),
                int(fields[19]),
                fields[0],
            )
        except (OSError, ValueError, IndexError):
            continue  # A process may finish during this read-only snapshot.
    return table


@dataclass
class OwnedProcess:
    role: str
    process: subprocess.Popen
    log: Path
    stream: IO[bytes]
    identities: dict[int, int] = field(default_factory=dict)

    def observe(self, table: dict[int, ProcessIdentity]) -> list[ProcessIdentity]:
        # A child keeps the parent's session unless it explicitly detaches. Also
        # remember observed descendants, including those with a new session.
        parents = {
            pid
            for pid, started in self.identities.items()
            if pid in table and table[pid].started == started
        }
        root = table.get(self.process.pid)
        session_owned = root is None or (
            self.identities.get(root.pid, root.started) == root.started
        )
        changed = True
        while changed:
            changed = False
            for item in table.values():
                if (
                    (session_owned and item.session == self.process.pid)
                    or item.parent in parents
                ) and (item.pid not in self.identities):
                    self.identities[item.pid] = item.started
                    parents.add(item.pid)
                    changed = True
        return [
            item
            for item in table.values()
            if self.identities.get(item.pid) == item.started and item.state != "Z"
        ]

    def text(self) -> str:
        return self.log.read_text(errors="replace")


class Supervision:
    def __init__(self, directory: Path):
        self.directory = directory
        self.processes: list[OwnedProcess] = []
        self.events: list[JSONValue] = []
        self.forced_cleanup = False

    def record(self, event: str, **details: JSONValue) -> None:
        self.events.append({"utc": utc_now(), "event": event, **details})
        save_json(self.directory / "lifecycle.json", self.events)

    def survivors(self) -> list[JSONValue]:
        table = process_table()
        return [
            {
                "role": owned.role,
                "pid": item.pid,
                "start_ticks": item.started,
                "parent": item.parent,
                "session": item.session,
            }
            for owned in self.processes
            for item in owned.observe(table)
        ]

    def start(
        self, role: str, command: list[str], environment: dict[str, str], repo: Path
    ) -> OwnedProcess:
        log = self.directory / f"{role}-server.log"
        stream = log.open("wb")
        try:
            child = subprocess.Popen(
                command,
                cwd=repo,
                env={**os.environ, **environment},
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            stream.close()
            raise
        owned = OwnedProcess(role, child, log, stream)
        root = process_table().get(child.pid)
        if root is not None:
            owned.identities[root.pid] = root.started
        self.processes.append(owned)
        self.survivors()
        self.record("spawned", role=role, pid=child.pid)
        return owned

    def wait_exit(self, owned: OwnedProcess, seconds: int) -> bool:
        deadline = time.monotonic() + seconds
        while owned.process.poll() is None:
            self.survivors()
            if time.monotonic() >= deadline:
                self.record("exit_timeout", role=owned.role, seconds=seconds)
                return False
            time.sleep(POLL_SECONDS)
        self.record("exited", role=owned.role, exit_code=owned.process.returncode)
        return True

    def stop_attention(self, attention: OwnedProcess) -> None:
        if attention.process.poll() is None:
            self.record("signal", role="attention", signal="SIGINT", target="root")
            attention.process.send_signal(signal.SIGINT)
            self.wait_exit(attention, OUTER_SHUTDOWN_SECONDS)

    def force_owned_cleanup(self) -> None:
        # Never signal a name, an arbitrary PID from npu-smi, or a reused PID.
        # Re-check start ticks immediately before each signal.
        if not self.survivors():
            return
        self.forced_cleanup = True
        for sig, grace in (
            (signal.SIGTERM, TERM_GRACE_SECONDS),
            (signal.SIGKILL, KILL_GRACE_SECONDS),
        ):
            live = self.survivors()
            self.record("forced_cleanup", signal=sig.name, processes=live)
            for identity in reversed(live):
                pid = identity["pid"]
                current = process_table().get(pid)
                if current is not None and current.started == identity["start_ticks"]:
                    with suppress(ProcessLookupError):
                        os.kill(pid, sig)
            deadline = time.monotonic() + grace
            while self.survivors() and time.monotonic() < deadline:
                for owned in self.processes:
                    owned.process.poll()
                time.sleep(POLL_SECONDS)
            if not self.survivors():
                break


def npu_snapshot(directory: Path, name: str) -> dict[str, JSONValue]:
    result = subprocess.run(["npu-smi", "info"], capture_output=True, timeout=30)
    data = result.stdout + result.stderr
    (directory / name).write_bytes(data)
    empty = sorted(
        set(map(int, re.findall(rb"No running processes found in NPU\s+(\d+)", data)))
    )
    # Fail closed on unfamiliar output rather than interpreting an empty match
    # as an idle device. Health/topology are retained for separate assessment.
    return {
        "exit_code": result.returncode,
        "idle_devices": empty,
        "all_eight_idle": result.returncode == 0 and empty == list(range(8)),
    }


def check_resources(directory: Path, plan: dict[str, JSONValue]) -> None:
    snapshot = npu_snapshot(directory, "npu-before.log")
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable"):
            memory[key] = int(value.split()[0]) * 1024
    save_json(
        directory / "resources.json",
        {
            "npu": snapshot,
            "memory_bytes": memory,
            "engram_host_table_lower_bound_bytes": MIN_HOST_BYTES,
        },
    )
    if not snapshot["all_eight_idle"]:
        raise RuntimeError("eight idle devices were not confirmed; see npu-before.log")
    if memory["MemAvailable"] < MIN_HOST_BYTES:
        raise RuntimeError(
            "host memory available is below the Engram table lower bound"
        )
    # This is an availability check, not a reservation. A later bind failure is
    # still a failed run and must not attach to an already running service.
    for port in plan["ports"]:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", port))


def probe_library(directory: Path, environment: dict[str, str], vendor: Path) -> None:
    library = vendor / "op_api/lib/libcust_opapi.so"
    # A fresh process sees the new loader path before loading dependent DSOs.
    code = (
        "import ctypes,json,sys; lib=ctypes.CDLL(sys.argv[1]); "
        "symbols=('aclnnCompressorV2','aclnnCompressorV2GetWorkspaceSize'); "
        "print(json.dumps({name:bool(getattr(lib,name)) for name in symbols}))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(library)],
        env={**os.environ, **environment},
        capture_output=True,
        timeout=30,
    )
    (directory / "operator-probe.log").write_bytes(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("CompressorV2 library probe failed; see operator-probe.log")


def health() -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{API_PORT}/health", timeout=2
        ) as reply:
            return reply.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_ready(
    owner: Supervision, attention: OwnedProcess, ffn: OwnedProcess, timeout: int
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        owner.survivors()
        for child in (attention, ffn):
            if child.process.poll() is not None:
                raise RuntimeError(
                    f"{child.role} exited during startup: {child.process.returncode}"
                )
        if F_READY in ffn.text() and health():
            owner.record("ready", attention_health=True, ffn_service_loop=True)
            return
        time.sleep(POLL_SECONDS)
    raise RuntimeError("startup timed out; server logs retain the first failure")


def send_request(directory: Path, reference: Path | None) -> dict[str, JSONValue]:
    payload = {
        "model": "dsv41-afd",
        "prompt": "Explain how to keep a simple daily work log.",
        "temperature": 0,
        "seed": 0,
        "max_tokens": REQUEST_TOKENS,
        "ignore_eos": True,
        "n": 1,
        "return_token_ids": True,
        "stream": False,
    }
    save_json(directory / "request.json", payload)
    request = urllib.request.Request(
        f"http://127.0.0.1:{API_PORT}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as reply:
            body = reply.read()
            status = reply.status
    except urllib.error.HTTPError as error:
        (directory / "response.json").write_bytes(error.read())
        raise RuntimeError(f"completion HTTP {error.code}") from error
    (directory / "response.json").write_bytes(body)
    response = json.loads(body)
    if len(response["choices"]) != 1:
        raise RuntimeError("expected exactly one completion")
    choice = response["choices"][0]
    tokens = choice["token_ids"]
    if (
        choice["finish_reason"] != "length"
        or len(tokens) != REQUEST_TOKENS
        or not all(type(token) is int for token in tokens)
        or response["usage"]["completion_tokens"] != REQUEST_TOKENS
    ):
        raise RuntimeError("completion did not return 32 token IDs with length finish")
    matches = None
    if reference is not None:
        baseline = json.loads(reference.read_bytes())
        save_json(directory / "reference-response.json", baseline)
        matches = tokens == baseline["choices"][0]["token_ids"]
        if not matches:
            raise RuntimeError("AFD token IDs differ from supplied native reference")
    return {
        "http_status": status,
        "tokens": len(tokens),
        "elapsed_seconds": time.monotonic() - started,
        "reference_token_ids_match": matches,
        "passed": True,
    }


def run_candidate(
    plan: dict[str, JSONValue],
    directory: Path,
    vendor: Path,
    startup_timeout: int,
    reference: Path | None,
) -> int:
    owner = Supervision(directory)
    result: dict[str, JSONValue] = {
        "request_passed": False,
        "clean_shutdown": False,
        "hardware_acceptance": False,
    }
    exit_code = 1
    attention = ffn = None
    try:
        check_resources(directory, plan)
        roles = plan["roles"]
        probe_library(directory, roles["attention"]["environment"], vendor)
        repo = Path(plan["repo_root"])
        probe = owner.start(
            "packed-attention-probe",
            [
                sys.executable,
                "-B",
                str(repo / "tests/npu/dsv41_packed_attention_probe.py"),
                "--run",
                "--device",
                "0",
                "--output",
                str(directory / "packed-attention-probe.json"),
            ],
            roles["attention"]["environment"],
            repo,
        )
        if not owner.wait_exit(probe, PACKED_ATTENTION_PROBE_SECONDS):
            raise RuntimeError("packed attention probe timed out; services not started")
        if probe.process.returncode:
            raise RuntimeError(
                "packed attention probe failed; see packed-attention-probe.json "
                "and packed-attention-probe-server.log"
            )
        result["packed_attention_probe"] = json.loads(
            (directory / "packed-attention-probe.json").read_bytes()
        )
        if not result["packed_attention_probe"]["passed"] or owner.survivors():
            raise RuntimeError("packed attention probe did not finish successfully")
        if not npu_snapshot(directory, "npu-after-packed-probe.log")["all_eight_idle"]:
            raise RuntimeError("NPU processes remain after packed attention probe")
        owner.record("packed_attention_probe_passed")
        # A and F must enter connector construction concurrently. F readiness
        # before starting A would deadlock the paired HCCL rendezvous.
        ffn = owner.start(
            "ffn", roles["ffn"]["argv"], roles["ffn"]["environment"], repo
        )
        attention = owner.start(
            "attention",
            roles["attention"]["argv"],
            roles["attention"]["environment"],
            repo,
        )
        wait_ready(owner, attention, ffn, startup_timeout)
        result["request"] = send_request(directory, reference)
        result["request_passed"] = True
        owner.record("request_completed")
    except (Exception, KeyboardInterrupt) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        owner.record("failed", error=result["error"])
        if isinstance(error, KeyboardInterrupt):
            exit_code = 130
        elif isinstance(error, TerminationRequestedError):
            exit_code = 128 + error.args[0]
    finally:
        # Even a failed request attempts A-driven STOP. F is never signalled
        # during the normal shutdown path.
        try:
            if attention is not None:
                owner.stop_attention(attention)
            if ffn is not None:
                owner.wait_exit(ffn, OUTER_SHUTDOWN_SECONDS)
        except (Exception, KeyboardInterrupt) as error:
            result["shutdown_error"] = f"{type(error).__name__}: {error}"
        owner.force_owned_cleanup()
        result["forced_cleanup"] = owner.forced_cleanup
        result["survivors"] = owner.survivors()
        result["exit_codes"] = {
            child.role: child.process.poll() for child in owner.processes
        }
        for child in owner.processes:
            child.stream.close()
        if ffn is not None:
            text = ffn.text()
            marks = {
                "peer_stop": PEER_STOP in text,
                "engine_cleanup": ENGINE_CLEAN in text,
                "launcher_exit": F_EXIT in text,
                "worker_ranks": sorted(set(map(int, re.findall(WORKER_CLEAN, text)))),
            }
            result["ffn_markers"] = marks
            forced = [
                pattern
                for pattern in FORCED_NATIVE
                if any(pattern in child.text() for child in owner.processes)
            ]
            result["native_forced_cleanup_markers"] = forced
            result["clean_shutdown"] = (
                attention is not None
                and attention.process.returncode == 0
                and ffn.process.returncode == 0
                and all(
                    marks[key]
                    for key in ("peer_stop", "engine_cleanup", "launcher_exit")
                )
                and marks["worker_ranks"] == [0, 1, 2, 3]
                and not forced
                and not owner.forced_cleanup
                and not result["survivors"]
                and "shutdown_error" not in result
            )
        try:
            result["npu_after"] = after = npu_snapshot(directory, "npu-after.log")
            result["clean_shutdown"] = (
                result["clean_shutdown"] and after["all_eight_idle"]
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            result["npu_after_error"] = str(error)
            result["clean_shutdown"] = False
        result["finished_utc"] = utc_now()
        if (
            result["request_passed"]
            and result["clean_shutdown"]
            and "error" not in result
        ):
            exit_code = 0
        result["exit_code"] = exit_code
        # A successful smoke is still subject to review of topology, values and
        # memory. It never enables DSpark or marks full-parity tasks complete.
        save_json(directory / "summary.json", result)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("/home/models/DeepSeek-V4.1-Flash")
    )
    parser.add_argument("--transformer-vendor", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--reference-response", type=Path)
    parser.add_argument(
        "--run", action="store_true", help="explicitly launch the A5 candidate"
    )
    args = parser.parse_args()
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive")
    if args.run and sys.platform != "linux":
        parser.error("--run requires the A5 Linux container")
    if args.reference_response is not None and not args.reference_response.is_file():
        parser.error("--reference-response must name an existing native response JSON")
    repo = Path(__file__).resolve().parents[2]
    output_root = (args.output_root or repo / "results").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="dsv41-a5-afd-eager-", dir=output_root))
    print(f"results: {directory}", flush=True)
    model, vendor = args.model.resolve(), args.transformer_vendor.resolve()
    plan = make_plan(model, vendor, repo)
    save_json(directory / "launch-plan.json", plan)
    report = collect_preflight(model, repo)
    save_json(directory / "preflight.json", report)
    if not report["passed"]:
        print("preflight failed; see preflight.json", flush=True)
        if args.run:
            pack_run(directory)
        return 1
    if not args.run:
        print(
            "preflight passed; services were not started (use --run for the candidate)"
        )
        return 0
    previous_handler = signal.signal(signal.SIGTERM, handle_termination)
    try:
        code = run_candidate(
            plan, directory, vendor, args.startup_timeout, args.reference_response
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        pack_run(directory)
    return code


def pack_run(directory: Path) -> None:
    # Archive only this run's diagnostics; never traverse the model path.
    archive = directory.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(directory, arcname=directory.name, recursive=True)
    print(f"diagnostics: {archive}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
