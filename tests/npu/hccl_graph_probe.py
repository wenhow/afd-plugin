# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded paired static HCCL graph capture/replay with controlled arithmetic.

No model, native graph manager, AFD parent protocol or production Graph selector
participates. Every peer agrees the entire probe schedule on Gloo before HCCL.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from datetime import timedelta
from pathlib import Path

HIDDEN_SIZE = 16
ROUTER_DELTA = 0.00031
ROUTER_SCALE = 4096
REPLAYS = 3
GROUP_TIMEOUT_SECONDS = 45
TOTAL_TIMEOUT_SECONDS = 240
REAP_TIMEOUT_SECONDS = 10
CASES = (("u1", (4,)), ("u2", (3, 2)), ("u2-empty-stage", (0, 2)), ("empty", (0,)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,2", help="FFN then Attention NPU")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=TOTAL_TIMEOUT_SECONDS)
    parser.add_argument("--rank", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        args.devices = tuple(int(value) for value in args.devices.split(","))
    except ValueError:
        parser.error("--devices requires two comma-separated NPU indices")
    if len(args.devices) != 2 or len(set(args.devices)) != 2 or min(args.devices) < 0:
        parser.error("--devices requires two distinct nonnegative NPU indices")
    if not 0 < args.timeout <= TOTAL_TIMEOUT_SECONDS:
        parser.error("invalid probe timeout")
    if args.rank is not None and args.port is None:
        parser.error("worker requires port")
    return args


def payload(torch, rows, stage, iteration):
    hidden = (
        torch.arange(rows * HIDDEN_SIZE, dtype=torch.float32).reshape(rows, HIDDEN_SIZE)
        / 8
        + stage
        + iteration
    ).to(torch.bfloat16)
    fp32 = hidden.float() + ROUTER_DELTA * (iteration + 1)
    ids = torch.arange(rows, dtype=torch.int32) + 11 * stage + iteration
    valid = max(0, rows - iteration)
    return hidden, fp32, ids, valid


def expected_output(torch, hidden, fp32, ids, valid):
    output = (
        hidden.float() + (fp32 - hidden.float()) * ROUTER_SCALE + ids[:, None].float()
    ).to(torch.bfloat16)
    output[valid:] = 0
    return output


def run_worker(args):
    # The supervisor imports no device stack. Device initialization is isolated
    # to these bounded subprocesses; no environment package is installed/changed.
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    import vllm_ascend.ops  # noqa: F401

    from afd_plugin.connectors.npu.hccl_graph import (
        HcclGraphBuffers,
        HcclGraphEndpoint,
        HcclGraphExchange,
    )

    device = torch.device("npu", args.devices[args.rank])
    torch.npu.set_device(device)
    report = {
        "rank": args.rank,
        "role": "ffn" if args.rank == 0 else "attention",
        "device": str(device),
        "mode": "native-torch-npu-graph",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torch-npu", "vllm", "vllm-ascend")
        },
        "component_sha256": hashlib.sha256(
            Path(sys.modules[HcclGraphExchange.__module__].__file__).read_bytes()
        ).hexdigest(),
        "cases": [],
    }
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{args.port}",
        rank=args.rank,
        world_size=2,
        timeout=timedelta(seconds=GROUP_TIMEOUT_SECONDS),
    )
    schedules = [None, None]
    dist.all_gather_object(schedules, (CASES, HIDDEN_SIZE, REPLAYS))
    if schedules[0] != schedules[1]:
        raise RuntimeError("both peers must agree every stage/empty/replay coordinate")
    groups = [
        dist.new_group(
            ranks=[0, 1],
            backend="hccl",
            timeout=timedelta(seconds=GROUP_TIMEOUT_SECONDS),
        )
        for _ in range(2)
    ]
    endpoints = [
        HcclGraphEndpoint.from_process_group(group, 1 - args.rank) for group in groups
    ]
    captures = []
    for name, row_counts in CASES:
        record = {"name": name, "rows": row_counts, "replays": []}
        print(
            f"phase=prepare rank={args.rank} case={name} rows={row_counts}", flush=True
        )
        if not any(row_counts):
            # Same frozen schedule on both peers. No unilateral empty-stage skip.
            dist.barrier()
            record["status"] = "agreed-empty-skip"
            report["cases"].append(record)
            continue
        exchanges = tuple(
            HcclGraphExchange(
                endpoints[stage],
                HcclGraphBuffers.allocate(rows, HIDDEN_SIZE, device),
                role=report["role"],
            )
            if rows
            else None
            for stage, rows in enumerate(row_counts)
        )
        valid_counts = tuple(
            torch.zeros(1, dtype=torch.int32, device=device) for _ in row_counts
        )
        row_indices = tuple(torch.arange(rows, device=device) for rows in row_counts)
        addresses = [
            [
                tensor.data_ptr()
                for tensor in (
                    exchange.buffers.hidden_states,
                    exchange.buffers.hidden_states_fp32,
                    exchange.buffers.input_ids,
                    exchange.buffers.output,
                )
            ]
            if exchange
            else []
            for exchange in exchanges
        ]
        record["buffer_addresses"] = addresses

        def refill(
            iteration,
            exchanges=exchanges,
            row_counts=row_counts,
            valid_counts=valid_counts,
        ):
            for stage, exchange in enumerate(exchanges):
                if exchange is None:
                    continue
                hidden, fp32, ids, valid = payload(
                    torch, row_counts[stage], stage, iteration
                )
                valid_counts[stage].fill_(valid)
                if args.rank == 1:
                    exchange.update_inputs(
                        hidden.to(device), fp32.to(device), ids.to(device)
                    )

        def body(
            exchanges=exchanges, row_indices=row_indices, valid_counts=valid_counts
        ):
            for stage, exchange in enumerate(exchanges):
                if exchange is None:
                    continue
                if args.rank == 1:
                    exchange.send_inputs()
                    exchange.receive_output()
                else:
                    exchange.receive_inputs()
                    buffers = exchange.buffers
                    computed = (
                        buffers.hidden_states.float()
                        + (buffers.hidden_states_fp32 - buffers.hidden_states.float())
                        * ROUTER_SCALE
                        + buffers.input_ids[:, None].float()
                    ).to(torch.bfloat16)
                    buffers.output.copy_(
                        computed.masked_fill(
                            row_indices[stage][:, None] >= valid_counts[stage], 0
                        )
                    )
                    exchange.send_output()

        refill(0)
        dist.barrier()
        print(f"phase=warmup rank={args.rank} case={name}", flush=True)
        body()
        torch.npu.synchronize(device)
        dist.barrier()
        graph = torch.npu.NPUGraph()
        print(f"phase=capture rank={args.rank} case={name}", flush=True)
        with torch.npu.graph(graph):
            body()
        # A cached graph and every fixed-address input must have one lifetime.
        # Retain masks/counts as well as the exchange's four payload buffers.
        captures.append((graph, exchanges, valid_counts, row_indices))
        torch.npu.synchronize(device)
        dist.barrier()
        for iteration in range(REPLAYS):
            refill(iteration)
            dist.barrier()
            print(
                f"phase=replay rank={args.rank} case={name} iteration={iteration}",
                flush=True,
            )
            graph.replay()
            torch.npu.synchronize(device)
            values = []
            for stage, exchange in enumerate(exchanges):
                if exchange is None:
                    continue
                buffers = exchange.buffers
                hidden, fp32, ids, valid = payload(
                    torch, row_counts[stage], stage, iteration
                )
                expected = expected_output(torch, hidden, fp32, ids, valid)
                torch.testing.assert_close(
                    buffers.output.cpu(), expected, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    buffers.hidden_states_fp32.cpu(), fp32, rtol=0, atol=0
                )
                torch.testing.assert_close(buffers.input_ids.cpu(), ids, rtol=0, atol=0)
                current = [
                    tensor.data_ptr()
                    for tensor in (
                        buffers.hidden_states,
                        buffers.hidden_states_fp32,
                        buffers.input_ids,
                        buffers.output,
                    )
                ]
                if current != addresses[stage]:
                    raise RuntimeError("graph exchange replaced a persistent buffer")
                values.append(
                    {
                        "stage": stage,
                        "valid": valid,
                        "output": buffers.output.cpu().tolist(),
                    }
                )
            record["replays"].append({"iteration": iteration, "values": values})
            dist.barrier()
        record["status"] = "passed"
        report["cases"].append(record)
    torch.npu.synchronize(device)
    dist.barrier()
    for capture in captures:
        capture[0].reset()
    captures.clear()
    gc.collect()
    for group in reversed(groups):
        dist.destroy_process_group(group)
    # Endpoint/graph destruction must not destroy the caller's default domain.
    dist.barrier()
    dist.destroy_process_group()
    report["status"] = "passed"
    (args.output_dir / f"rank-{args.rank}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def run_supervisor(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    started = time.monotonic()
    processes, handles = [], []
    summary = {"scope": "controlled-arithmetic HCCL graph component", "cases": CASES}
    try:
        for rank in range(2):
            handle = (args.output_dir / f"rank-{rank}.log").open("wb")
            handles.append(handle)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--devices",
                ",".join(map(str, args.devices)),
                "--output-dir",
                str(args.output_dir),
                "--rank",
                str(rank),
                "--port",
                str(port),
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        while any(process.poll() is None for process in processes):
            failed = any(process.poll() not in (None, 0) for process in processes)
            if failed or time.monotonic() - started > args.timeout:
                summary["failure"] = (
                    "worker failure" if failed else "bounded probe timeout"
                )
                break
            time.sleep(0.1)
    finally:
        for process in processes:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=REAP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=REAP_TIMEOUT_SECONDS)
        for handle in handles:
            handle.close()
        summary["exit_codes"] = [process.returncode for process in processes]
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["status"] = (
            "passed"
            if summary["exit_codes"] == [0, 0] and "failure" not in summary
            else "failed"
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    options = parse_args()
    if options.rank is None:
        raise SystemExit(run_supervisor(options))
    try:
        run_worker(options)
    except BaseException:
        traceback.print_exc()
        raise SystemExit(1) from None
