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
import ast
import hashlib
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
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.client import HTTPResponse
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
HEALTH_TIMEOUT_SECONDS = 2
CURL_EXIT_GRACE_SECONDS = 1
READINESS_PROGRESS_SECONDS = 30
TERM_GRACE_SECONDS = 10
KILL_GRACE_SECONDS = 10
EP_STACK_DIR_ENV = "AFD_DSV41_EP_STACK_DIR"
EP_STACK_DIR = "shutdown-stacks"
EP_GRAPH_ROLES = ("attention", "ffn")
HCCL_CONFIGURATION_PREFIX = "AFD HCCL session configuration: "
HCCL_STAGE_PREFIX = "AFD HCCL stage diagnostic: "
# Match the logger's flat settings snapshot at startup and each parent.
HCCL_GRAPH_CONFIGURED_FIELDS = (
    "configured.AFD_HCCL_EAGER_U2_STREAM_OVERLAP",
    "configured.AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP",
    "configured.AFD_HCCL_GRAPH_U2_HYBRID_DAG",
    "configured.AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM",
    "configured.AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM",
    "configured.AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER",
    "configured.AFD_HCCL_STAGE_DIAGNOSTICS",
    "configured.AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS",
)
MIN_HOST_BYTES = int(188.833133 * 1024**3)
PLUGINS = (
    "ascend,ascend_model,ascend_model_loader,ascend_kv_connector,"
    "ascend_service_profiling,afd"
)


class TerminationRequestedError(Exception):
    """Preserve SIGTERM's shell exit code while releasing owned processes."""


GraphLogRecord = dict[str, str | int | bool]


def _hccl_log_records(
    raw: bytes, path: Path, *, include_eager_stages: bool = False
) -> list[GraphLogRecord]:
    records = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        for prefix in (HCCL_CONFIGURATION_PREFIX, HCCL_STAGE_PREFIX):
            _, found, payload = line.partition(prefix)
            if not found:
                continue
            try:
                node = ast.parse(payload.strip(), mode="eval").body
                if not isinstance(node, ast.Dict):
                    raise ValueError("expected a dict record")
                keys = [ast.literal_eval(key) for key in node.keys]
                if len(set(keys)) != len(keys):
                    raise ValueError("duplicate record field")
                record = ast.literal_eval(node)
                if (
                    any(
                        type(key) is not str or type(value) not in (str, int, bool)
                        for key, value in record.items()
                    )
                    or type(record.get("event")) is not str
                ):
                    raise ValueError("expected scalar record fields and an event")
                configuration = record["event"] == "session_configuration"
                if configuration != (prefix == HCCL_CONFIGURATION_PREFIX):
                    raise ValueError("event does not match its log prefix")
            except (SyntaxError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if (
                include_eager_stages
                or configuration
                or record["event"].startswith("graph_parent_")
            ):
                records.append(record)
            break
    return records


def _check_graph_fields(
    record: GraphLogRecord, expected: GraphLogRecord, name: str
) -> None:
    for key, value in expected.items():
        actual = record.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"{name}: {key} expected {value!r}, got {actual!r}")


def _verify_graph_parent(
    records: list[GraphLogRecord], decision: GraphLogRecord, startup: GraphLogRecord
) -> None:
    role, phase = startup["role"], decision["phase"]
    name = f"{role}/{startup['role_rank']}/{decision['transaction_id']}"
    boundaries = [("graph_parent_reserved", "graph_reservation", "returned")]
    if phase == "replay":
        replay = "run_fullgraph" if role == "attention" else "graph.replay"
        boundaries.extend(
            [
                ("graph_parent_replay_begin", replay, "begin"),
                ("graph_parent_replay_returned", replay, "returned"),
            ]
        )
    finish = "complete_graph_forward"
    if phase == "eager":
        if role == "ffn":
            finish = "execute_hccl_v41_parent"
        else:
            finish = records[-1].get("boundary_api")
            if finish not in ("native_eager_scope", "finish_forward"):
                raise ValueError(f"{name}: invalid Attention eager finish API")
    if finish in ("native_eager_scope", "execute_hccl_v41_parent"):
        boundaries.append(("graph_parent_scope_returned", finish, "returned"))
    else:
        boundaries.extend(
            [
                ("graph_parent_finish_begin", finish, "begin"),
                ("graph_parent_finish_returned", finish, "returned"),
            ]
        )
    if [record["event"] for record in records] != [event for event, _, _ in boundaries]:
        raise ValueError(f"{name}: incorrect Graph host boundary sequence")
    shared = {
        **decision,
        **{
            key: startup[key]
            for key in ("role", "role_rank", "world_rank", "local_device")
        },
        **{key: startup[key] for key in HCCL_GRAPH_CONFIGURED_FIELDS},
    }
    for record, (event, api, outcome) in zip(records, boundaries, strict=True):
        expected = {
            **shared,
            "event": event,
            "boundary": "host_control",
            "boundary_api": api,
            "outcome": outcome,
        }
        if set(record) != set(expected):
            raise ValueError(f"{name}/{event}: incorrect Graph host record fields")
        _check_graph_fields(record, expected, f"{name}/{event}")


def verify_ep_graph_run(
    directory: Path, role_size: JSONValue, max_stages: int
) -> dict[str, JSONValue]:
    """Check all-rank capture, replay and runtime eager host boundaries."""
    if type(role_size) is not int or role_size < 1:
        raise ValueError("Graph role size must be a positive integer")
    start: dict[tuple[str, int], GraphLogRecord] = {}
    transactions: dict[str, dict[tuple[str, int], list[GraphLogRecord]]] = {}
    log_hashes: dict[str, JSONValue] = {}
    for role in EP_GRAPH_ROLES:
        path = directory / f"{role}-server.log"
        raw = path.read_bytes()
        log_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        for record in _hccl_log_records(raw, path):
            rank = record.get("role_rank")
            if (
                type(rank) is not int
                or not 0 <= rank < role_size
                or record.get("role") != role
            ):
                raise ValueError(f"{path.name}: invalid {role} rank {rank!r}")
            coordinate = (role, rank)
            if record["event"] == "session_configuration":
                if coordinate in start:
                    raise ValueError(f"{coordinate}: duplicate Graph startup")
                _check_graph_fields(
                    record,
                    {
                        "world_rank": rank + role_size * int(role == "attention"),
                        "negotiated.expert_parallel": True,
                        "negotiated.native_tp_size": 1,
                        "negotiated.attention_size": role_size,
                        "negotiated.ffn_size": role_size,
                        "negotiated.graph_execution_version": 1,
                        "negotiated.max_stages": max_stages,
                        "configured.AFD_HCCL_STAGE_DIAGNOSTICS": True,
                    },
                    str(coordinate),
                )
                if any(
                    type(record.get(key)) is not bool
                    for key in HCCL_GRAPH_CONFIGURED_FIELDS
                ):
                    raise ValueError(f"{coordinate}: missing Graph startup settings")
                start[coordinate] = record
                continue
            if coordinate not in start:
                raise ValueError(f"{coordinate}: missing Graph startup")
            if record.get("phase") == "eager" and record.get("mode") in (
                "profile",
                "warmup",
            ):
                # Skip pre-KV profiling and the zero-row capture eligibility probe.
                continue
            transaction = record.get("transaction_id")
            if type(transaction) is not str or not transaction.strip():
                raise ValueError(f"{coordinate}: missing Graph transaction ID")
            transactions.setdefault(transaction, {}).setdefault(coordinate, []).append(
                record
            )

    coordinates = {(role, rank) for role in EP_GRAPH_ROLES for rank in range(role_size)}
    if set(start) != coordinates:
        raise ValueError(
            f"missing EP Graph startup ranks: {sorted(coordinates - start.keys())}"
        )

    completed: dict[str, list[str]] = {}
    captures = set()
    for transaction, ranks in transactions.items():
        if set(ranks) != coordinates:
            missing = sorted(coordinates - ranks.keys())
            raise ValueError(
                f"{transaction}: missing participant coordinates {missing}"
            )
        first = ranks[("attention", 0)][0]
        phase, mode, stages = first["phase"], first["mode"], first["num_stages"]
        if type(phase) is not str or phase not in (
            "capture",
            "replay",
            "eager",
            "warmup",
        ):
            raise ValueError(f"{transaction}: unknown Graph phase {phase!r}")
        if type(stages) is not int or not 1 <= stages <= max_stages:
            raise ValueError(f"{transaction}: invalid Graph stage count {stages!r}")
        key = first["key_digest"]
        digests = (
            (first["decision_digest"],)
            if phase == "eager"
            else (first["decision_digest"], key)
        )
        if any(
            type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in digests
        ):
            raise ValueError(f"{transaction}: invalid Graph decision or key digest")
        if phase == "eager" and key != "":
            raise ValueError(f"{transaction}: eager phase carries a Graph key")
        if phase == "replay" and key not in captures:
            raise ValueError(f"{transaction}: replay key was not captured")
        decision = {
            "transaction_id": transaction,
            "phase": phase,
            "mode": mode,
            "reason": first["reason"],
            "key_present": phase != "eager",
            "key_digest": key,
            "decision_digest": first["decision_digest"],
            "num_stages": stages,
        }
        for coordinate, records in ranks.items():
            _verify_graph_parent(records, decision, start[coordinate])
        if phase == "capture":
            captures.add(key)
        if (
            phase == "eager"
            and mode in ("target", "dummy")
            or phase != "eager"
            and stages == max_stages
        ):
            completed.setdefault(phase, []).append(transaction)

    for phase in ("capture", "replay", "eager"):
        if phase not in completed:
            raise ValueError(f"no completed EP {phase} phase")
    return {
        "passed": True,
        "transactions": completed,
        "source_sha256": log_hashes,
        "device_completion_verified_by_logs_alone": False,
    }


def verify_ep_eager_u2_run(
    directory: Path,
    role_size: JSONValue,
    request_windows: dict[str, tuple[int, int]],
) -> dict[str, JSONValue]:
    """Require one serial two-stage model parent for the HTTP requests on every rank."""
    if type(role_size) is not int or role_size < 1:
        raise ValueError("eager U2 role size must be a positive integer")
    stages = (0, 1)
    role_events = {
        "attention": ("compute_returned", "send_returned", "recv_returned"),
        "ffn": ("recv_returned", "compute_returned", "send_returned"),
    }
    transactions: dict[str, dict[tuple[str, int], list[GraphLogRecord]]] = {}
    sources: dict[str, JSONValue] = {}
    for role, events in role_events.items():
        path = directory / f"{role}-server.log"
        start, end = request_windows[role]
        if not 0 <= start < end:
            raise ValueError(f"{path.name}: empty or invalid request log window")
        with path.open("rb") as log:
            log.seek(start)
            raw = log.read(end - start)
        if len(raw) != end - start:
            raise ValueError(f"{path.name}: truncated request log window")
        sources[path.name] = {
            "start_byte": start,
            "end_byte": end,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for record in _hccl_log_records(raw, path, include_eager_stages=True):
            if (
                record["event"] not in (*events, "collapse_returned")
                or record.get("phase") != "eager"
                or record.get("num_stages") != len(stages)
                or record.get("in_profile_run") is not False
                or record.get("is_warmup") is not False
            ):
                continue
            rank = record.get("role_rank")
            transaction = record.get("transaction_id")
            if (
                type(rank) is not int
                or not 0 <= rank < role_size
                or type(transaction) is not str
                or not transaction.strip()
                or type(record.get("layer")) is not int
                or type(record.get("stage")) is not int
                or record["stage"] not in stages
            ):
                raise ValueError(f"{path.name}: invalid eager U2 stage coordinates")
            _check_graph_fields(
                record,
                {
                    "role": role,
                    "world_rank": rank + role_size * int(role == "attention"),
                    "num_stages": len(stages),
                    "boundary": "host_enqueue",
                    "actual.mode": "serial",
                    "actual.compute_stream": "parent",
                    "actual.send_stream": "parent",
                    "actual.recv_stream": "parent",
                },
                f"{transaction}/{role}/{rank}",
            )
            transactions.setdefault(transaction, {}).setdefault(
                (role, rank), []
            ).append(record)

    coordinates = {(role, rank) for role in role_events for rank in range(role_size)}
    completed: dict[str, JSONValue] = {}
    for transaction, ranks in transactions.items():
        if ranks.keys() != coordinates:
            continue
        final_layers = [
            record["layer"]
            for record in ranks[("attention", 0)]
            if record["event"] == "collapse_returned"
        ]
        if len(final_layers) != 1:
            continue
        final_layer = final_layers[0]
        for (role, _rank), records in ranks.items():
            expected = [
                (stage, event) for stage in stages for event in role_events[role]
            ]
            if role == "attention":
                expected.append((stages[-1], "collapse_returned"))
            observed = [
                (record["stage"], record["event"])
                for record in records
                if record["layer"] == final_layer
            ]
            if observed != expected:
                break
        else:
            completed[transaction] = final_layer
    if not completed:
        raise ValueError(
            "no live serial eager U2 parent completed both stages on every rank"
        )
    return {
        "passed": True,
        "world_ranks": list(range(len(coordinates))),
        "transactions": completed,
        "request_log_windows": sources,
        "device_completion_verified_by_logs_alone": False,
    }


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

    def stop_attention(
        self,
        attention: OwnedProcess,
        *,
        stop_signal: signal.Signals = signal.SIGINT,
        trace_signal: bool = False,
    ) -> None:
        if attention.process.poll() is None:
            signal_phase = stop_signal.name.lower()
            if trace_signal:
                self._record_attention_signal_state(attention, f"before_{signal_phase}")
            self.record(
                "signal", role="attention", signal=stop_signal.name, target="root"
            )
            attention.process.send_signal(stop_signal)
            if trace_signal:
                self._record_attention_signal_state(attention, f"after_{signal_phase}")
            exited = self.wait_exit(attention, OUTER_SHUTDOWN_SECONDS)
            if trace_signal and not exited:
                self._record_attention_signal_state(
                    attention, f"{signal_phase}_timeout"
                )
                self._request_attention_stacks(attention)

    def _request_attention_stacks(self, attention: OwnedProcess) -> None:
        stack_directory = self.directory / EP_STACK_DIR
        if not stack_directory.is_dir():
            return
        armed = {
            int(path.stem)
            for path in stack_directory.glob("*.log")
            if path.stem.isdecimal()
        }
        for process in attention.observe(process_table()):
            if process.pid not in armed:
                continue
            try:
                with self._open_process_fd(process) as fd:
                    signal.pidfd_send_signal(fd, signal.SIGUSR1)
            except (AttributeError, OSError, ValueError) as error:
                self.record(
                    "attention_stack_unavailable", pid=process.pid, error=str(error)
                )
            else:
                self.record("attention_stack_requested", pid=process.pid)

    @contextmanager
    def _open_process_fd(self, process: ProcessIdentity) -> Iterator[int]:
        fd = os.pidfd_open(process.pid)
        try:
            current = process_table().get(process.pid)
            if current is None or current.started != process.started:
                raise ValueError("process exited or was replaced")
            with Path(f"/proc/{process.pid}/status").open() as status:
                caught = next(
                    (line.split()[1] for line in status if line.startswith("SigCgt:")),
                    "0",
                )
            if not int(caught, 16) & (1 << (signal.SIGUSR1 - 1)):
                raise ValueError("stack dump signal handler is not active")
            yield fd
        finally:
            os.close(fd)

    def _record_attention_signal_state(
        self, attention: OwnedProcess, phase: str
    ) -> None:
        pid = attention.process.pid
        status_path = Path(f"/proc/{pid}/status")
        fields = {"State", "SigPnd", "ShdPnd", "SigBlk", "SigIgn", "SigCgt"}
        try:
            status = {
                name: value.strip()
                for name, _, value in (
                    line.partition(":") for line in status_path.read_text().splitlines()
                )
                if name in fields
            }
        except OSError as error:
            status = {"error": str(error)}
        self.record("attention_signal_state", phase=phase, pid=pid, status=status)

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
    if not selected_devices_idle(snapshot, plan):
        raise RuntimeError(
            "selected idle devices were not confirmed; see npu-before.log"
        )
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


def selected_devices_idle(
    snapshot: dict[str, JSONValue], plan: dict[str, JSONValue]
) -> bool:
    return snapshot["exit_code"] == 0 and set(plan["physical_devices"]).issubset(
        snapshot["idle_devices"]
    )


def open_service_request(
    request: urllib.request.Request | str, *, timeout: float
) -> HTTPResponse:
    # Validation URLs identify the exact A/P/D service. Inherited HTTP proxies
    # must not receive health checks, completions or original PD prompt history.
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        request, timeout=timeout
    )


def health(
    api_url: str = f"http://127.0.0.1:{API_PORT}", *, directory: Path | None = None
) -> bool:
    url = api_url.rstrip("/") + "/health"
    passed = False
    report: dict[str, JSONValue] = {
        "utc": utc_now(),
        "url": url,
        "connection": "direct",
        "client": "curl",
        "exit_code": None,
        "status_code": None,
        "error": None,
        "passed": False,
    }
    try:
        reply = subprocess.run(
            [
                "curl",
                "--disable",
                "--noproxy",
                "*",
                "--silent",
                "--show-error",
                "--max-time",
                str(HEALTH_TIMEOUT_SECONDS),
                "--output",
                os.devnull,
                "--write-out",
                "%{http_code}",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=HEALTH_TIMEOUT_SECONDS + CURL_EXIT_GRACE_SECONDS,
        )
        status = reply.stdout.strip()
        status_code = int(status) if status.isdecimal() and int(status) else None
        passed = reply.returncode == 0 and status_code == 200
        report.update(
            exit_code=reply.returncode, status_code=status_code, passed=passed
        )
        if not passed:
            report["error"] = (
                f"curl exit {reply.returncode}, HTTP {status or 'unknown'}: "
                f"{reply.stderr.strip()}"
            )
    except subprocess.TimeoutExpired as error:
        report["error"] = f"{type(error).__name__}: {error}"
    except OSError as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise RuntimeError(f"cannot execute curl for readiness: {error}") from error
    finally:
        if directory is not None:
            save_json(directory / "health.json", report)
    return passed


def wait_ready(
    owner: Supervision,
    attention: OwnedProcess,
    ffn: OwnedProcess | None,
    timeout: int,
    *,
    api_url: str = f"http://127.0.0.1:{API_PORT}",
) -> None:
    deadline = time.monotonic() + timeout
    next_progress = 0.0
    previous_state = None
    while time.monotonic() < deadline:
        owner.survivors()
        for child in (attention, ffn):
            if child is None:
                continue
            if child.process.poll() is not None:
                raise RuntimeError(
                    f"{child.role} exited during startup: {child.process.returncode}"
                )
        attention_ready = health(api_url, directory=owner.directory)
        ffn_ready = ffn is None or F_READY in ffn.text()
        if ffn_ready and attention_ready:
            owner.record(
                "ready", attention_health=True, ffn_service_loop=ffn is not None
            )
            return
        state = (attention_ready, ffn_ready)
        now = time.monotonic()
        if state != previous_state or now >= next_progress:
            owner.record(
                "waiting_for_ready",
                attention_health=attention_ready,
                ffn_service_loop=ffn is not None and ffn_ready,
            )
            print(
                f"waiting for readiness: attention_health={attention_ready}, "
                f"ffn_ready={ffn_ready}; latest HTTP result: "
                f"{owner.directory / 'health.json'}",
                flush=True,
            )
            previous_state = state
            next_progress = now + READINESS_PROGRESS_SECONDS
        time.sleep(POLL_SECONDS)
    raise RuntimeError(
        "startup timed out; see health.json for the latest HTTP result and "
        "server logs for startup failures"
    )


def compare_response_tokens(
    response: dict[str, JSONValue], baseline: dict[str, JSONValue] | None
) -> dict[str, JSONValue]:
    """Report exact token differences without claiming topology/model equivalence."""
    choice = response["choices"][0]
    tokens = choice["token_ids"]
    prompt = choice.get("prompt_token_ids")
    result: dict[str, JSONValue] = {
        "status": "not_checked",
        "model_equivalence_accepted": False,
        "actual_token_ids": tokens,
        "actual_prompt_token_ids": prompt,
        "reference_token_ids_match": None,
        "prompt_token_ids_match": None,
        "first_output_mismatch": None,
    }
    if baseline is None:
        return result
    try:
        assert len(baseline["choices"]) == 1
        native = baseline["choices"][0]
        expected = native["token_ids"]
        expected_prompt = native["prompt_token_ids"]
        assert native["finish_reason"] == "length"
        assert isinstance(expected, list) and len(expected) == REQUEST_TOKENS
        assert isinstance(expected_prompt, list) and expected_prompt
        assert all(type(token) is int and token >= 0 for token in expected)
        assert all(type(token) is int and token >= 0 for token in expected_prompt)
    except (AssertionError, KeyError, TypeError) as error:
        result.update(
            status="invalid_reference", error=f"{type(error).__name__}: {error}"
        )
        return result
    result.update(
        reference_token_ids=expected,
        reference_prompt_token_ids=expected_prompt,
        reference_token_ids_match=tokens == expected,
        prompt_token_ids_match=prompt == expected_prompt,
        first_output_mismatch=next(
            (
                {"index": index, "reference_token_id": before, "actual_token_id": after}
                for index, (before, after) in enumerate(
                    zip(expected, tokens, strict=True)
                )
                if before != after
            ),
            None,
        ),
        status="tokens_match"
        if tokens == expected and prompt == expected_prompt
        else "tokens_differ",
    )
    return result


def send_request(
    directory: Path,
    reference: Path | None,
    *,
    payload: dict[str, JSONValue] | None = None,
    api_url: str = f"http://127.0.0.1:{API_PORT}",
    data_parallel_rank: int | None = None,
    collect_token_mismatch: bool = False,
) -> dict[str, JSONValue]:
    payload = payload or {
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
    headers = {"Content-Type": "application/json"}
    if data_parallel_rank is not None:
        headers["X-data-parallel-rank"] = str(data_parallel_rank)
        save_json(directory / "request-routing.json", {"dp_rank": data_parallel_rank})
    request = urllib.request.Request(
        api_url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    started = time.monotonic()
    try:
        with open_service_request(request, timeout=REQUEST_TIMEOUT_SECONDS) as reply:
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
    comparison = compare_response_tokens(response, None)
    if reference is not None:
        try:
            raw = reference.read_bytes()
            (directory / "reference-response.json").write_bytes(raw)
            baseline = json.loads(raw)
            if not isinstance(baseline, dict):
                raise ValueError("native reference must be a completion JSON object")
            comparison = compare_response_tokens(response, baseline)
        except (OSError, ValueError) as error:
            comparison.update(
                status="invalid_reference", error=f"{type(error).__name__}: {error}"
            )
    save_json(directory / "token-comparison.json", comparison)
    if comparison["status"] == "invalid_reference":
        raise RuntimeError("invalid native reference; see token-comparison.json")
    tokens_differ = comparison["status"] == "tokens_differ"
    if tokens_differ and not collect_token_mismatch:
        raise RuntimeError(
            "AFD prompt/output tokens differ from native reference; "
            "see token-comparison.json"
        )
    return {
        "http_status": status,
        "tokens": len(tokens),
        "elapsed_seconds": time.monotonic() - started,
        "reference_token_ids_match": comparison["reference_token_ids_match"],
        "prompt_token_ids_match": comparison["prompt_token_ids_match"],
        "passed": not tokens_differ,
    }


def run_candidate(
    plan: dict[str, JSONValue],
    directory: Path,
    vendor: Path,
    startup_timeout: int,
    reference: Path | None,
    *,
    request_runner: Callable[[Path, Path | None], dict[str, JSONValue]] | None = None,
) -> int:
    owner = Supervision(directory)
    result: dict[str, JSONValue] = {
        "request_passed": False,
        "clean_shutdown": False,
        "hardware_acceptance": False,
    }
    if "native_reference" in plan:
        result["native_reference"] = plan["native_reference"]
    exit_code = 1
    attention = ffn = None
    verify_eager_u2 = (
        plan.get("expert_parallel") is True and plan.get("mode") == "eager-u2"
    )
    try:
        check_resources(directory, plan)
        roles = plan["roles"]
        probe_library(directory, roles["attention"]["environment"], vendor)
        repo = Path(plan["repo_root"])
        # The dedicated 16-to-64 head adapter probe applies to TP4 only.
        if not plan.get("expert_parallel", False):
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
                raise RuntimeError(
                    "packed attention probe timed out; services not started"
                )
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
            if not selected_devices_idle(
                npu_snapshot(directory, "npu-after-packed-probe.log"), plan
            ):
                raise RuntimeError("NPU processes remain after packed attention probe")
            owner.record("packed_attention_probe_passed")
        # A and F must enter connector construction concurrently. F readiness
        # before starting A would deadlock the paired HCCL rendezvous.
        if "ffn" in roles:
            ffn = owner.start(
                "ffn", roles["ffn"]["argv"], roles["ffn"]["environment"], repo
            )
        attention_environment = roles["attention"]["environment"]
        if plan.get("ep_diagnostics") is True:
            stack_directory = directory.resolve() / EP_STACK_DIR
            stack_directory.mkdir()
            attention_environment = {
                **attention_environment,
                EP_STACK_DIR_ENV: str(stack_directory),
            }
        attention = owner.start(
            "attention",
            roles["attention"]["argv"],
            attention_environment,
            repo,
        )
        wait_ready(
            owner,
            attention,
            ffn,
            startup_timeout,
            api_url=plan.get("api_url", f"http://127.0.0.1:{API_PORT}"),
        )
        request_logs = (
            {child.role: child.log.stat().st_size for child in (attention, ffn)}
            if verify_eager_u2 and ffn is not None
            else {}
        )
        result["request"] = (request_runner or send_request)(directory, reference)
        result["request_passed"] = True
        owner.record("request_completed")
        if request_logs and ffn is not None:
            try:
                request_windows = {
                    child.role: (request_logs[child.role], child.log.stat().st_size)
                    for child in (attention, ffn)
                }
                result["eager_u2_verification"] = verify_ep_eager_u2_run(
                    directory, plan["native_dp_size"], request_windows
                )
            except (OSError, ValueError, KeyError) as error:
                result["eager_u2_verification"] = {"passed": False, "error": str(error)}
                raise RuntimeError(
                    f"eager U2 host evidence incomplete: {error}"
                ) from error
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
                owner.stop_attention(
                    attention,
                    stop_signal=signal.Signals[
                        plan.get("attention_stop_signal", "SIGINT")
                    ],
                    trace_signal=plan.get("ep_diagnostics") is True,
                )
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
        elif attention is not None:
            result["clean_shutdown"] = (
                attention.process.returncode == 0
                and not owner.forced_cleanup
                and not result["survivors"]
                and "shutdown_error" not in result
            )
        try:
            result["npu_after"] = after = npu_snapshot(directory, "npu-after.log")
            result["clean_shutdown"] = result[
                "clean_shutdown"
            ] and selected_devices_idle(after, plan)
        except (OSError, subprocess.TimeoutExpired) as error:
            result["npu_after_error"] = str(error)
            result["clean_shutdown"] = False
        result["finished_utc"] = utc_now()
        comparison_path = directory / "token-comparison.json"
        if comparison_path.is_file():
            result["token_comparison"] = json.loads(comparison_path.read_bytes())
        if (
            ffn is not None
            and result["request_passed"]
            and plan.get("mode") in ("graph-u1", "graph-u2")
            and plan.get("expert_parallel", False)
            and plan.get("request_suite") != "repeatability"
        ):
            try:
                result["graph_verification"] = verify_ep_graph_run(
                    directory,
                    plan["native_dp_size"],
                    2 if plan["mode"] == "graph-u2" else 1,
                )
            except (OSError, ValueError, KeyError) as error:
                result["graph_verification"] = {"passed": False, "error": str(error)}
                result["error"] = f"Graph host evidence incomplete: {error}"
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
