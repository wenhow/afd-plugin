# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in host-posting/Work dependency gate on the existing HCCL data group.

The supervisor withholds the matcher's CPU permission to enter its data op.
Posting return, Work.wait return, pending consumer event and exact payload are
separate observations. This is an endpoint capability gate, without a parent
plan, model weights, eager tickets, HCCL abort or device recovery acceptance.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import selectors
import signal
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import hccl_staged_probe as helpers

EVENT_PREFIX = "EAGER_BACKEND_EVENT "
PAYLOAD_KINDS = ("bf16", "fp32", "ids")
PAYLOAD_ROWS = 8
CONSUMER_DELTA = 7
GATE_WINDOW_SECONDS = 5
TOTAL_TIMEOUT_SECONDS = 240
READ_CHUNK_BYTES = 65_536


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=("isend", "irecv"), required=True)
    parser.add_argument("--devices", default="0,2", help="FFN then Attention")
    parser.add_argument("--model", default=helpers.CHECKPOINT, help="config only")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-window", type=int, default=GATE_WINDOW_SECONDS)
    parser.add_argument("--timeout", type=int, default=TOTAL_TIMEOUT_SECONDS)
    parser.add_argument("--role", choices=("ffn", "attention"), help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        args.devices = tuple(map(int, args.devices.split(",")))
    except ValueError:
        parser.error("devices must be comma-separated integer NPU indices")
    if len(args.devices) != 2 or len(set(args.devices)) != 2 or min(args.devices) < 0:
        parser.error("exactly two distinct nonnegative NPU indices are required")
    if not 0 < args.gate_window <= GATE_WINDOW_SECONDS:
        parser.error("gate window must be between one and five seconds")
    if not args.gate_window < args.timeout <= TOTAL_TIMEOUT_SECONDS:
        parser.error("timeout must exceed the gate window and fit the outer bound")
    if args.role and args.port is None:
        parser.error("worker invocation needs a port")
    return args


def emit(event, **fields):
    print(
        EVENT_PREFIX + json.dumps({"event": event, "pid": os.getpid()} | fields),
        flush=True,
    )


def await_command(expected):
    command = sys.stdin.readline().strip()
    if command != expected:
        raise RuntimeError(f"expected CPU command {expected!r}, got {command!r}")


def make_payload(torch, kind):
    if kind == "ids":
        return torch.arange(PAYLOAD_ROWS, dtype=torch.int32) * 3
    values = torch.arange(
        PAYLOAD_ROWS * helpers.HIDDEN_SIZE, dtype=torch.float32
    ).reshape(PAYLOAD_ROWS, helpers.HIDDEN_SIZE)
    base = values / 8 + 1
    # Original FP32 retains values that BF16 cannot represent; it is never
    # reconstructed by converting the BF16 payload back to FP32.
    return base.to(torch.bfloat16) if kind == "bf16" else base + helpers.ROUTER_DELTA


def worker(args):
    # Lazy imports keep the CPU supervisor and its preflight free of torch/NPU.
    import torch
    import torch.distributed as dist
    import vllm_ascend.ops  # noqa: F401
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.utils import adapt_patch

    from afd_plugin.compat.npu.stream import use_npu_current_stream
    from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
    from afd_plugin.distributed.hccl_plan import HcclPlanCapabilities

    is_poster = args.role == "attention"
    device_idx = args.devices[int(is_poster)]
    device = torch.device("npu", device_idx)
    report = {
        "scope": "existing transport HCCL data-group primitive gate, idle session",
        "operation": args.operation,
        "role": args.role,
        "device": str(device),
        "pid": os.getpid(),
        "model_weights_loaded": False,
        "accepted_parent_or_cursor_modified": False,
        "abort_or_device_recovery_exercised": False,
        "payloads": {},
        "success": False,
    }

    def native_check():
        value = torch.tensor([1.0], device=device)
        dist.all_reduce(value, group=get_tp_group().device_group)
        assert value.item() == 1.0
        return value.cpu().tolist()

    try:
        current_platform.pre_register_and_update()
        load_general_plugins()
        torch.npu.set_device(device_idx)
        adapt_patch()
        config = helpers.build_config(1, args.model)
        init_ascend_config(config)
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=device_idx,
                distributed_init_method=(
                    f"tcp://127.0.0.1:{args.port + 1 + int(is_poster)}"
                ),
                backend="hccl",
                timeout=timedelta(seconds=helpers.GROUP_TIMEOUT_SECONDS),
            )
            initialize_model_parallel(tensor_model_parallel_size=1, backend="hccl")
            report["native_tp_before"] = native_check()
            transport = HcclStagedTransport(
                args.role,
                0,
                HcclPlanCapabilities(
                    helpers.HIDDEN_SIZE,
                    helpers.VOCAB_SIZE,
                    helpers.CAPACITY,
                    helpers.LAYERS,
                    1,
                    1,
                    1,
                    local_counts_version=1,
                ),
                init_method=f"tcp://127.0.0.1:{args.port}",
                local_device=device_idx,
                timeout_seconds=helpers.GROUP_TIMEOUT_SECONDS,
            )
            transport.initialize()
            group = transport.data_groups[0]
            peer = 0 if is_poster else 1
            report["endpoint"] = {
                "group_type": f"{type(group).__module__}.{type(group).__qualname__}",
                "backend": dist.get_backend(group),
                "afd_group_rank": group.rank(),
                "native_default_rank": dist.get_rank(),
                "group_peer": peer,
                "stage_group": 0,
                "functions": {
                    name: {
                        "module": function.__module__,
                        "signature": str(inspect.signature(function)),
                        "source_file": inspect.getsourcefile(function),
                        "source_sha256": hashlib.sha256(
                            Path(inspect.getsourcefile(function)).read_bytes()
                        ).hexdigest(),
                    }
                    for name, function in (("isend", dist.isend), ("irecv", dist.irecv))
                },
            }
            posting_stream = torch.npu.Stream(device=device)
            consumer_stream = torch.npu.Stream(device=device)
            sending = is_poster == (args.operation == "isend")

            def post(tensor):
                if sending:
                    return dist.isend(tensor, group=group, group_dst=peer)
                return dist.irecv(tensor, group=group, group_src=peer)

            # Match every dtype on the exact same group and both side streams
            # before any gated operation; first-use rendezvous is out of scope.
            for kind in PAYLOAD_KINDS:
                reference = make_payload(torch, kind)
                tensor = (
                    reference.to(device)
                    if sending
                    else torch.full_like(reference, -101, device=device)
                )
                torch.npu.synchronize()
                with use_npu_current_stream(posting_stream):
                    warm_work = post(tensor)
                    assert warm_work is not None
                with use_npu_current_stream(consumer_stream):
                    warm_work.wait()
                    warm_consumer = (
                        tensor.add_(CONSUMER_DELTA)
                        if sending
                        else tensor + CONSUMER_DELTA
                    )
                torch.npu.synchronize()
                torch.testing.assert_close(
                    tensor.cpu(),
                    reference + CONSUMER_DELTA if sending else reference,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    warm_consumer.cpu(), reference + CONSUMER_DELTA, rtol=0, atol=0
                )
            emit("warmup-complete")

            for kind in PAYLOAD_KINDS:
                reference = make_payload(torch, kind)
                tensor = (
                    reference.to(device)
                    if sending
                    else torch.full_like(reference, -101, device=device)
                )
                # No asynchronous allocation or producer is left ahead of post.
                torch.npu.synchronize()
                emit("gate-ready", payload=kind)
                await_command(("post:" if is_poster else "match:") + kind)
                emit("post-enter" if is_poster else "match-enter", payload=kind)
                with use_npu_current_stream(posting_stream):
                    work = post(tensor)
                    assert work is not None
                emit("post-return" if is_poster else "match-return", payload=kind)
                report["payloads"][kind] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "work_type": f"{type(work).__module__}.{type(work).__qualname__}",
                    "work_wait_documentation": type(work).wait.__doc__,
                }
                if is_poster:
                    await_command("wait:" + kind)
                    emit("wait-enter", payload=kind)
                with use_npu_current_stream(consumer_stream):
                    wait_result = work.wait()
                    if is_poster:
                        emit("wait-return", payload=kind, result=wait_result)
                    # Reuse the actual send allocation only after its Work
                    # dependency. The receiver must still obtain the original
                    # payload, so a missing dependency can corrupt the oracle.
                    consumer = (
                        tensor.add_(CONSUMER_DELTA)
                        if sending
                        else tensor + CONSUMER_DELTA
                    )
                    consumed = torch.npu.Event()
                    consumed.record(consumer_stream)
                if is_poster:
                    pending = not consumed.query()
                    emit("consumer-enqueued", payload=kind, pending=pending)
                    await_command("query:" + kind)
                    emit(
                        "consumer-observed", payload=kind, pending=not consumed.query()
                    )
                    await_command("drain:" + kind)
                consumed.synchronize()
                torch.testing.assert_close(
                    tensor.cpu(),
                    reference + CONSUMER_DELTA if sending else reference,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    consumer.cpu(), reference + CONSUMER_DELTA, rtol=0, atol=0
                )
                report["payloads"][kind].update(
                    wait_result=wait_result,
                    received_payload_exact=not sending,
                    source_reuse_exact=sending,
                    consumer_exact=True,
                    consumer_event_complete=consumed.query(),
                )
                emit("gate-drained", payload=kind)
            await_command("stop")
            if is_poster:
                transport.stop_attention()
            else:
                assert transport.receive_forward() is None
            report["stop_acknowledged"] = True
            transport.close()
            report["plugin_closed"] = transport.closed
            report["native_tp_after_plugin_close"] = native_check()
            destroy_model_parallel()
            destroy_distributed_environment()
            report["success"] = True
            emit("closed-native-tp-preserved")
    except BaseException:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
        raise
    finally:
        (args.output_dir / f"{args.role}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


def observe_owned(root_pid, owned):
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            if pid in owned and owned[pid] != fields[19]:
                continue
            owned[pid] = fields[19]
            pending.extend(
                map(int, Path(f"/proc/{pid}/task/{pid}/children").read_text().split())
            )


def kill_owned(owned):
    killed = []
    for pid, start_ticks in reversed(tuple(owned.items())):
        with contextlib.suppress(FileNotFoundError, ProcessLookupError):
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            if fields[19] == start_ticks:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
    return killed


def supervise(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    port = helpers.select_port_block(3)
    children, handles, buffers, events, owned = {}, {}, {}, [], {}
    selector = selectors.DefaultSelector()
    gates = {kind: {"ready": [], "drained": []} for kind in PAYLOAD_KINDS}
    active = None
    started = time.monotonic()
    error = None
    timed_out = False

    def command(role, value):
        children[role].stdin.write((value + "\n").encode())
        children[role].stdin.flush()

    def release_match(kind, reason):
        gate = gates[kind]
        if "match_released_at" not in gate:
            gate["match_released_at"] = time.monotonic() - started
            gate["match_release_reason"] = reason
            command("ffn", "match:" + kind)

    try:
        for role in ("ffn", "attention"):
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    __file__,
                    "--operation",
                    args.operation,
                    "--devices",
                    ",".join(map(str, args.devices)),
                    "--model",
                    args.model,
                    "--output-dir",
                    str(args.output_dir),
                    "--role",
                    role,
                    "--port",
                    str(port),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                bufsize=0,
            )
            children[role] = child
            buffers[role] = b""
            handles[role] = (args.output_dir / f"{role}.log").open("wb")
            selector.register(child.stdout, selectors.EVENT_READ, role)
        while selector.get_map() or any(c.poll() is None for c in children.values()):
            now = time.monotonic()
            for child in children.values():
                if child.poll() is None:
                    observe_owned(child.pid, owned)
            if now - started >= args.timeout:
                timed_out = True
                break
            if active is not None:
                gate = gates[active]
                if "match_released_at" not in gate and now >= gate["deadline"]:
                    if gate["phase"] == "consumer-observe":
                        command("attention", "query:" + active)
                        gate["phase"] = "query-return"
                        gate["deadline"] = time.monotonic() + args.gate_window
                    else:
                        release_match(
                            active, gate["phase"] + "-deadline-release-to-drain"
                        )
            for key, _ in selector.select(timeout=0.1):
                role = key.data
                chunk = os.read(key.fd, READ_CHUNK_BYTES)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                handles[role].write(chunk)
                handles[role].flush()
                buffers[role] += chunk
                while b"\n" in buffers[role]:
                    line, buffers[role] = buffers[role].split(b"\n", 1)
                    if not line.startswith(EVENT_PREFIX.encode()):
                        continue
                    event = json.loads(line[len(EVENT_PREFIX) :])
                    event |= {
                        "role": role,
                        "observed_seconds": time.monotonic() - started,
                    }
                    events.append(event)
                    print(json.dumps(event), flush=True)
                    name, kind = event["event"], event.get("payload")
                    if kind is None:
                        continue
                    gate = gates[kind]
                    held = "match_released_at" not in gate
                    if name == "gate-ready":
                        gate["ready"].append(role)
                        if len(gate["ready"]) == 2:
                            active = kind
                            gate["phase"] = "post"
                            gate["deadline"] = time.monotonic() + args.gate_window
                            command("attention", "post:" + kind)
                    elif name == "post-return":
                        gate["post_returned_while_matcher_held"] = held
                        gate["phase"] = "wait"
                        gate["deadline"] = time.monotonic() + args.gate_window
                        command("attention", "wait:" + kind)
                    elif name == "wait-return":
                        gate["wait_returned_while_matcher_held"] = held
                        gate["phase"] = "consumer-enqueue"
                    elif name == "consumer-enqueued":
                        gate["early_consumer_event_pending"] = event["pending"]
                        if held:
                            gate["phase"] = "consumer-observe"
                            gate["deadline"] = time.monotonic() + args.gate_window
                        else:
                            command("attention", "query:" + kind)
                    elif name == "consumer-observed":
                        gate["late_query_while_matcher_held"] = held
                        gate["late_consumer_event_pending"] = event["pending"]
                        release_match(kind, "late-consumer-event-observed")
                        command("attention", "drain:" + kind)
                    elif name == "gate-drained":
                        gate["drained"].append(role)
                        if len(gate["drained"]) == 2:
                            active = None
                            if kind == PAYLOAD_KINDS[-1]:
                                command("ffn", "stop")
                                command("attention", "stop")
            for role, child in children.items():
                if child.poll() not in (None, 0):
                    raise RuntimeError(
                        f"{role} exited {child.returncode}; retain raw log"
                    )
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
    finally:
        for child in children.values():
            if child.poll() is None:
                observe_owned(child.pid, owned)
        killed = (
            kill_owned(owned)
            if any(c.poll() is None for c in children.values())
            else []
        )
        for child in children.values():
            child.wait(timeout=helpers.REAP_TIMEOUT_SECONDS)
        for handle in handles.values():
            handle.close()
        selector.close()
    workers = {
        role: json.loads(path.read_text()) if path.exists() else None
        for role in children
        for path in [args.output_dir / f"{role}.json"]
    }
    progress_checks = (
        "post_returned_while_matcher_held",
        "wait_returned_while_matcher_held",
        "late_query_while_matcher_held",
    )
    capability = all(
        all(gate.get(check) for check in progress_checks)
        and (args.operation == "isend" or gate.get("late_consumer_event_pending"))
        for gate in gates.values()
    )
    drained = (
        not error
        and not timed_out
        and not killed
        and len(workers) == 2
        and all(worker is not None and worker["success"] for worker in workers.values())
        and all(child.returncode == 0 for child in children.values())
    )
    result = {
        "scope": (
            "endpoint capability gate; not C4 regression or eager ticket acceptance"
        ),
        "operation": args.operation,
        "observed_inflight_at_late_query": {
            kind: gate.get("late_query_while_matcher_held", False)
            and gate.get("late_consumer_event_pending", False)
            for kind, gate in gates.items()
        },
        "capability_gate_passed": capability and drained,
        "all_payloads_drained_and_session_closed": bool(drained),
        "timed_out": timed_out,
        "supervisor_error": error,
        "owned_pid_start_ticks": owned,
        "killed_owned_pids": killed,
        "exit_codes": {role: child.returncode for role, child in children.items()},
        "gates": gates,
        "events": events,
        "workers": workers,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["capability_gate_passed"] else 2 if drained else 1


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.role:
        worker(parsed)
    else:
        raise SystemExit(supervise(parsed))
