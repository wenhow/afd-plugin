# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded real-NPU HCCL stage/peer probe using controlled arithmetic, no MoE.

Run on idle devices after activating the pinned CANN and Python environment.
Each process retains a separate native vLLM TP group; only the transport-owned
groups are closed before the final native collective is checked.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import timedelta
from functools import partial
from pathlib import Path

HIDDEN_SIZE = 16
VOCAB_SIZE = 4_096
CAPACITY = 32
LAYERS = (0, 2)
CHECKPOINT = "/mnt/workspace/models/DeepSeek-V4.1-Flash"
ROUTER_DELTA = 0.00031
ROUTER_DELTA_SCALE = 4_096
GROUP_TIMEOUT_SECONDS = 45
TOTAL_TIMEOUT_SECONDS = 240
REAP_TIMEOUT_SECONDS = 10
PORT_ATTEMPTS = 20
MAX_TCP_PORT = 65_535
TOPOLOGIES = ("1a1f", "2a1f", "1a2f", "2a2f-tp1", "2a2f-tp2")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=TOPOLOGIES, default="1a1f")
    parser.add_argument(
        "--devices", default="0,2,4,6", help="FFN ranks, then Attention ranks"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=TOTAL_TIMEOUT_SECONDS)
    rejection = parser.add_mutually_exclusive_group()
    rejection.add_argument(
        "--reject-hidden-size",
        action="store_true",
        help="expect a mismatched FFN0 HELLO to reject every rank",
    )
    rejection.add_argument(
        "--reject-plan-digest",
        action="store_true",
        help="inject a bad FFN0 ACCEPT digest on the real Gloo domain (2a1f)",
    )
    parser.add_argument("--role", choices=("attention", "ffn"), help=argparse.SUPPRESS)
    parser.add_argument("--role-rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    attention, ffn, tp = {
        "1a1f": (1, 1, 1),
        "2a1f": (2, 1, 1),
        "1a2f": (1, 2, 1),
        "2a2f-tp1": (2, 2, 1),
        "2a2f-tp2": (2, 2, 2),
    }[args.topology]
    args.attention_size, args.ffn_size, args.tp_size = attention, ffn, tp
    if args.reject_plan_digest and args.topology != "2a1f":
        parser.error("--reject-plan-digest is scoped to the 2a1f drain case")
    try:
        devices = tuple(map(int, args.devices.split(",")))
    except ValueError:
        parser.error("--devices requires comma-separated integer NPU IDs")
    if len(devices) < attention + ffn or len(set(devices)) != len(devices):
        parser.error("--devices requires enough distinct NPUs for the topology")
    args.devices = devices[: attention + ffn]
    if min(args.devices) < 0 or not 0 < args.timeout <= TOTAL_TIMEOUT_SECONDS:
        parser.error("invalid device index or timeout")
    if args.role and (
        args.role_rank is None or args.port is None or args.output_dir is None
    ):
        parser.error("worker invocation requires role rank, port and output directory")
    return args


def build_config(tp_size):
    # Worker isolation: the supervisor does not import torch/vLLM or touch NPU.
    from vllm.config import CompilationMode, CUDAGraphMode, ModelConfig, VllmConfig
    from vllm.config.device import DeviceConfig
    from vllm.config.load import LoadConfig
    from vllm.config.parallel import ParallelConfig

    config = VllmConfig(
        model_config=ModelConfig(
            model=CHECKPOINT,
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=128,
            skip_tokenizer_init=True,
            language_model_only=True,
        ),
        parallel_config=ParallelConfig(tensor_parallel_size=tp_size),
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(load_format="dummy"),
    )
    config.device_config = DeviceConfig(device="npu")
    config.quant_config = config.engram_config = config.speculative_config = None
    config.compilation_config.mode = CompilationMode.NONE
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    config.compilation_config.use_inductor_graph_partition = False
    return config


def build_plans(args):
    from afd_plugin.distributed.hccl import HcclStageLayout
    from afd_plugin.distributed.hccl_plan import HcclParentPlan

    def counts(first, following):
        return tuple(
            first if rank // args.tp_size == 0 else following
            for rank in range(args.attention_size)
        )

    zero = (0,) * args.attention_size

    def stage(index, valid, padded):
        return HcclStageLayout(index, valid, padded, args.ffn_size, args.tp_size)

    return (
        HcclParentPlan("u1-one", LAYERS, (stage(0, counts(1, 1), counts(1, 1)),)),
        HcclParentPlan(
            "u2-mixed",
            LAYERS,
            (
                stage(0, counts(0, 3), counts(0, 4)),
                stage(1, counts(1, 0), counts(2, 2)),
            ),
        ),
        HcclParentPlan(
            "u2-empty-stage",
            LAYERS,
            (stage(0, zero, zero), stage(1, counts(5, 1), counts(6, 2))),
        ),
        HcclParentPlan(
            "u2-empty", LAYERS, (stage(0, zero, zero), stage(1, zero, zero))
        ),
        HcclParentPlan(
            "u2-dummy",
            LAYERS,
            (stage(0, zero, counts(4, 2)), stage(1, zero, counts(2, 2))),
            "dummy",
        ),
        HcclParentPlan(
            "u2-repeat",
            LAYERS,
            (
                stage(0, counts(3, 1), counts(4, 2)),
                stage(1, counts(1, 3), counts(2, 4)),
            ),
        ),
    )


def make_attention_inputs(torch, args, stage, attention_rank, layer_idx):
    padded = stage.attention_padded_tokens[attention_rank]
    valid = stage.attention_valid_tokens[attention_rank]
    dp_rank = attention_rank // args.tp_size
    values = torch.arange(padded * HIDDEN_SIZE, dtype=torch.float32)
    hidden = (values.reshape(padded, HIDDEN_SIZE) / 64 + dp_rank).bfloat16()
    fp32 = hidden.float() + ROUTER_DELTA
    ids = (
        torch.arange(padded, dtype=torch.int32)
        + 100
        + dp_rank * 100
        + stage.stage_idx * 20
        + layer_idx * 3
    )
    ids[valid:] = 0
    return hidden, fp32, ids


def local_output(hidden, fp32, ids, ffn_rank, tp_size, layer_idx, stage_idx):
    return (
        (fp32 - hidden.float()) * ROUTER_DELTA_SCALE
        + ids[:, None].float()
        + (ffn_rank // tp_size) / 8
        + layer_idx / 8
        + stage_idx / 16
    ).bfloat16()


def expected_ffn_inputs(torch, args, stage, ffn_rank, layer_idx):
    pieces = [[], [], []]
    for peer in stage.for_ffn(ffn_rank):
        attention_rank = peer.attention_world_rank - args.ffn_size
        source = make_attention_inputs(torch, args, stage, attention_rank, layer_idx)
        for index, (tensor, shape) in enumerate(
            zip(
                source,
                (
                    (peer.padded_tokens, HIDDEN_SIZE),
                    (peer.padded_tokens, HIDDEN_SIZE),
                    (peer.padded_tokens,),
                ),
                strict=True,
            )
        ):
            chunk = torch.zeros(shape, dtype=tensor.dtype)
            available = min(
                peer.padded_tokens, max(0, tensor.shape[0] - peer.attention_offset)
            )
            chunk[:available] = tensor[
                peer.attention_offset : peer.attention_offset + available
            ]
            pieces[index].append(chunk)
    return tuple(torch.cat(parts, dim=0) for parts in pieces)


def expected_attention_output(torch, args, stage, attention_rank, layer_idx):
    source = make_attention_inputs(torch, args, stage, attention_rank, layer_idx)
    padded = stage.attention_padded_tokens[attention_rank]
    expected = torch.zeros((padded, HIDDEN_SIZE), dtype=torch.bfloat16)
    for peer in stage.for_attention(attention_rank):
        start = peer.attention_offset
        stop = min(start + peer.padded_tokens, padded)
        if start >= stop:
            continue
        output = local_output(
            *(tensor[start:stop] for tensor in source),
            peer.ffn_world_rank,
            args.tp_size,
            layer_idx,
            stage.stage_idx,
        )
        expected[start:stop] = (output.float() * args.tp_size).bfloat16()
    return expected


def expect_rejection(report, label, error_type, operation):
    """Exercise a public pre-I/O boundary and retain the actual rejection."""
    try:
        operation()
    except error_type as exc:
        report["rejections"].append(
            {"label": label, "type": type(exc).__name__, "message": str(exc)}
        )
    else:
        raise AssertionError(f"{label} unexpectedly succeeded")


def run_rank(args):
    # Runtime imports are intentionally local to the isolated worker process.
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
    from vllm_ascend.utils import adapt_patch

    from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
    from afd_plugin.distributed.hccl_plan import (
        MAX_CONTROL_MESSAGE_BYTES,
        HcclPlanCapabilities,
        HcclPlanMessage,
        decode_hccl_plan_message,
        encode_hccl_plan_message,
    )

    started = time.monotonic()
    world_rank = args.role_rank + (args.ffn_size if args.role == "attention" else 0)
    device = args.devices[world_rank]
    tp_rank = args.role_rank % args.tp_size
    native_group_idx = args.role_rank // args.tp_size
    if args.role == "attention":
        native_group_idx += args.ffn_size // args.tp_size
    report = {
        "role": args.role,
        "role_rank": args.role_rank,
        "world_rank": world_rank,
        "pid": os.getpid(),
        "device": device,
        "topology": args.topology,
        "native_tp_size": args.tp_size,
        "native_tp_rank": tp_rank,
        "native_moe_executed": False,
        "checkpoint_weights_loaded": False,
        "scope": "real multi-stage HCCL transport with controlled arithmetic",
        "transactions": [],
        "rejections": [],
        "success": False,
    }
    result_path = args.output_dir / f"{args.role}-{args.role_rank}.json"
    transport = None

    def native_group_check():
        assert dist.get_world_size() == args.tp_size
        assert dist.get_rank() == tp_rank
        value = torch.tensor([tp_rank + 1], dtype=torch.float32, device=f"npu:{device}")
        dist.all_reduce(value, group=get_tp_group().device_group)
        assert value.item() == sum(range(1, args.tp_size + 1))
        return value.cpu().tolist()

    def receive_injected_control(rank):
        size = torch.empty(1, dtype=torch.int64, device="cpu")
        dist.recv(size, group_src=rank, group=transport.control_group)
        length = int(size.item())
        if not 0 < length <= MAX_CONTROL_MESSAGE_BYTES:
            raise ValueError("injected peer received an invalid control frame size")
        body = torch.empty(length, dtype=torch.uint8, device="cpu")
        dist.recv(body, group_src=rank, group=transport.control_group)
        return decode_hccl_plan_message(bytes(body.tolist()))

    def send_injected_control(rank, message):
        data = encode_hccl_plan_message(message)
        size = torch.tensor([len(data)], dtype=torch.int64, device="cpu")
        body = torch.tensor(list(data), dtype=torch.uint8, device="cpu")
        dist.send(size, group_dst=rank, group=transport.control_group)
        dist.send(body, group_dst=rank, group=transport.control_group)

    try:
        current_platform.pre_register_and_update()
        load_general_plugins()
        torch.npu.set_device(device)
        adapt_patch()
        config = build_config(args.tp_size)
        with set_current_vllm_config(config):
            native_port = args.port + 1 + native_group_idx
            init_distributed_environment(
                world_size=args.tp_size,
                rank=tp_rank,
                local_rank=device,
                distributed_init_method=f"tcp://127.0.0.1:{native_port}",
                backend="hccl",
                timeout=timedelta(seconds=GROUP_TIMEOUT_SECONDS),
            )
            initialize_model_parallel(
                tensor_model_parallel_size=args.tp_size, backend="hccl"
            )
            report["native_tp_before_transport"] = native_group_check()
            transport = HcclStagedTransport(
                args.role,
                args.role_rank,
                HcclPlanCapabilities(
                    HIDDEN_SIZE
                    + int(
                        args.reject_hidden_size
                        and args.role == "ffn"
                        and args.role_rank == 0
                    ),
                    VOCAB_SIZE,
                    CAPACITY,
                    LAYERS,
                    args.attention_size,
                    args.ffn_size,
                    args.tp_size,
                ),
                init_method=f"tcp://127.0.0.1:{args.port}",
                local_device=device,
                timeout_seconds=GROUP_TIMEOUT_SECONDS,
            )
            if args.reject_hidden_size:
                try:
                    transport.initialize()
                except (RuntimeError, ValueError) as exc:
                    if "hidden_size" not in str(exc):
                        raise
                    report["expected_rejection"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                else:
                    raise AssertionError("mismatched HELLO unexpectedly initialized")
            else:
                transport.initialize()
                report["native_tp_after_initialize"] = native_group_check()
                if args.reject_plan_digest:
                    first_plan = build_plans(args)[0]
                    if args.role == "ffn" and args.role_rank == 0:
                        forward = receive_injected_control(args.ffn_size)
                        if forward.kind != "FORWARD" or forward.plan != first_plan:
                            raise AssertionError(
                                "injected peer did not receive expected FORWARD"
                            )
                        send_injected_control(
                            args.ffn_size,
                            HcclPlanMessage(
                                "ACCEPT",
                                world_rank=0,
                                transaction_id=first_plan.transaction_id,
                                plan_digest="0" * 64,
                            ),
                        )
                        rejected = receive_injected_control(args.ffn_size)
                        if rejected.kind != "ERROR" or "digest" not in rejected.error:
                            raise AssertionError(
                                "injected peer did not receive digest ERROR"
                            )
                        report["expected_rejection"] = {
                            "type": "ERROR",
                            "message": rejected.error,
                        }
                    else:
                        try:
                            if transport.is_coordinator:
                                transport.begin_forward(first_plan)
                            else:
                                transport.receive_forward()
                        except (RuntimeError, ValueError) as exc:
                            if "digest" not in str(exc):
                                raise
                            report["expected_rejection"] = {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            }
                        else:
                            raise AssertionError(
                                "mismatched ACCEPT unexpectedly succeeded"
                            )
                else:
                    for plan in build_plans(args):
                        if args.role == "attention" and args.role_rank == 0:
                            transport.begin_forward(plan)
                        else:
                            assert transport.receive_forward() == plan
                        if any(
                            any(stage.attention_padded_tokens) for stage in plan.stages
                        ):
                            expect_rejection(
                                report,
                                "early_complete",
                                RuntimeError,
                                transport.finish_forward,
                            )
                        if transport.is_coordinator:
                            expect_rejection(
                                report, "active_stop", RuntimeError, transport.stop
                            )
                        exchanges = []
                        for layer_idx in LAYERS:
                            for stage in plan.stages:
                                if not any(stage.attention_padded_tokens):
                                    continue
                                if args.role == "attention":
                                    source = make_attention_inputs(
                                        torch, args, stage, args.role_rank, layer_idx
                                    )
                                    expect_rejection(
                                        report,
                                        "receive_before_inputs",
                                        RuntimeError,
                                        partial(
                                            transport.receive_layer_output,
                                            layer_idx,
                                            stage.stage_idx,
                                        ),
                                    )
                                    native = tuple(
                                        tensor.to(f"npu:{device}") for tensor in source
                                    )
                                    expect_rejection(
                                        report,
                                        "wrong_stage",
                                        ValueError,
                                        partial(
                                            transport.send_layer_inputs,
                                            layer_idx,
                                            stage.stage_idx + 1,
                                            *native,
                                        ),
                                    )
                                    transport.send_layer_inputs(
                                        layer_idx,
                                        stage.stage_idx,
                                        *native,
                                    )
                                    output = transport.receive_layer_output(
                                        layer_idx, stage.stage_idx
                                    )
                                    expected = expected_attention_output(
                                        torch, args, stage, args.role_rank, layer_idx
                                    )
                                    torch.testing.assert_close(
                                        output.cpu(), expected, rtol=0, atol=0
                                    )
                                else:
                                    expect_rejection(
                                        report,
                                        "output_before_inputs",
                                        RuntimeError,
                                        partial(
                                            transport.send_layer_output,
                                            layer_idx,
                                            stage.stage_idx,
                                            torch.empty(
                                                (0, HIDDEN_SIZE),
                                                dtype=torch.bfloat16,
                                                device=f"npu:{device}",
                                            ),
                                        ),
                                    )
                                    received = transport.receive_layer(
                                        layer_idx, stage.stage_idx
                                    )
                                    expected = expected_ffn_inputs(
                                        torch, args, stage, args.role_rank, layer_idx
                                    )
                                    for actual, reference in zip(
                                        (
                                            received.hidden_states,
                                            received.hidden_states_fp32,
                                            received.input_ids,
                                        ),
                                        expected,
                                        strict=True,
                                    ):
                                        torch.testing.assert_close(
                                            actual.cpu(), reference, rtol=0, atol=0
                                        )
                                    output = local_output(
                                        received.hidden_states,
                                        received.hidden_states_fp32,
                                        received.input_ids,
                                        args.role_rank,
                                        args.tp_size,
                                        layer_idx,
                                        stage.stage_idx,
                                    )
                                    dist.all_reduce(
                                        output, group=get_tp_group().device_group
                                    )
                                    transport.send_layer_output(
                                        layer_idx, stage.stage_idx, output
                                    )
                                exchanges.append([layer_idx, stage.stage_idx])
                        transport.finish_forward()
                        if transport.is_coordinator:
                            expect_rejection(
                                report,
                                "completed_transaction_replay",
                                ValueError,
                                partial(transport.begin_forward, plan),
                            )
                        report["transactions"].append(
                            {
                                "transaction_id": plan.transaction_id,
                                "mode": plan.mode,
                                "plan_digest": plan.digest,
                                "verified_layer_stages": exchanges,
                                "stages": [
                                    {
                                        "stage_idx": stage.stage_idx,
                                        "valid": stage.attention_valid_tokens,
                                        "padded": stage.attention_padded_tokens,
                                    }
                                    for stage in plan.stages
                                ],
                            }
                        )
                    if args.role == "attention" and args.role_rank == 0:
                        transport.stop()
                    else:
                        assert transport.receive_forward() is None
                    report["stop_acknowledged"] = True
            transport.close()
            transport.close()
            report["native_tp_after_close"] = native_group_check()
            report["success"] = True
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    finally:
        result_path.write_text(json.dumps(report, indent=2) + "\n")
        try:
            if transport is not None:
                transport.close()
            destroy_model_parallel()
            destroy_distributed_environment()
            report["native_groups_destroyed"] = True
        except BaseException as exc:
            report["success"] = False
            report["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
            traceback.print_exc()
        report["elapsed_seconds"] = time.monotonic() - started
        result_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["success"] else 1


def select_port_block(count):
    for _ in range(PORT_ATTEMPTS):
        listeners = []
        try:
            first = socket.socket()
            listeners.append(first)
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            if port + count - 1 > MAX_TCP_PORT:
                continue
            for offset in range(1, count):
                listener = socket.socket()
                listeners.append(listener)
                listener.bind(("127.0.0.1", port + offset))
            return port
        except OSError:
            continue
        finally:
            for listener in listeners:
                listener.close()
    raise RuntimeError("could not reserve a contiguous probe port block")


def run_parent(args):
    started = time.monotonic()
    args.output_dir = args.output_dir or Path(
        tempfile.mkdtemp(prefix="afd-stage-probe-")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    port = select_port_block(1 + (args.attention_size + args.ffn_size) // args.tp_size)
    processes, logs = [], []
    summary = {
        "scope": "multi-stage transport component; no native MoE/model",
        "success": False,
    }
    deadline = started + args.timeout - REAP_TIMEOUT_SECONDS
    try:
        for role, count in (("ffn", args.ffn_size), ("attention", args.attention_size)):
            for rank in range(count):
                key = f"{role}-{rank}"
                log = (args.output_dir / (key + ".log")).open("w")
                logs.append(log)
                command = [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--topology",
                    args.topology,
                    "--role",
                    role,
                    "--role-rank",
                    str(rank),
                    "--port",
                    str(port),
                    "--devices",
                    ",".join(map(str, args.devices)),
                    "--output-dir",
                    str(args.output_dir),
                ]
                if args.reject_hidden_size:
                    command.append("--reject-hidden-size")
                if args.reject_plan_digest:
                    command.append("--reject-plan-digest")
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append((key, process))
        for _, process in processes:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        summary["timed_out"] = True
    finally:
        summary["external_kills"] = []
        for key, process in processes:
            if process.poll() is None:
                summary["external_kills"].append(key)
                os.killpg(process.pid, signal.SIGKILL)
        for _, process in processes:
            process.wait(timeout=REAP_TIMEOUT_SECONDS)
        for log in logs:
            log.close()
        summary["ranks"] = {
            key: {
                "pid": process.pid,
                "exit_code": process.returncode,
                "result": json.loads((args.output_dir / (key + ".json")).read_text())
                if (args.output_dir / (key + ".json")).exists()
                else None,
            }
            for key, process in processes
        }
        summary["success"] = (
            bool(summary["ranks"])
            and not summary["external_kills"]
            and all(
                value["exit_code"] == 0
                and value["result"] is not None
                and value["result"]["success"]
                for value in summary["ranks"].values()
            )
        )
        summary["topology"] = args.topology
        summary["reject_hidden_size"] = args.reject_hidden_size
        summary["reject_plan_digest"] = args.reject_plan_digest
        summary["devices"] = args.devices
        summary["elapsed_seconds"] = time.monotonic() - started
        (args.output_dir / "result.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(run_rank(parsed) if parsed.role else run_parent(parsed))
