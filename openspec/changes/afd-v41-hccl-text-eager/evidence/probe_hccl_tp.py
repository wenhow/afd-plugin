# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded four-process native TP2 + HCCL pair component probe, without MoE."""

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
from pathlib import Path

CHECKPOINT = "/mnt/workspace/models/DeepSeek-V4.1-Flash"
TP_SIZE = 2
HIDDEN_SIZE = 16
VOCAB_SIZE = 1024
CAPACITY = 8
LAYER_INDICES = (0, 1)
GROUP_TIMEOUT_SECONDS = 45
TOTAL_TIMEOUT_SECONDS = 210
REAP_TIMEOUT_SECONDS = 10
PORT_ATTEMPTS = 20
PORT_COUNT = 4
MAX_TCP_PORT = 65535
ROUTER_DELTA = 0.00031
ROUTER_DELTA_SCALE = 4096
TRANSACTIONS = (
    ("target-1", 3, 4, "target"),
    ("empty-2", 0, 0, "target"),
    ("dummy-3", 0, 3, "dummy"),
    ("profile-4", 0, 4, "profile"),
    ("target-5", 1, 2, "target"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,2,4,6", help="A0,A1,F0,F1 local NPUs")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=TOTAL_TIMEOUT_SECONDS)
    parser.add_argument("--role", choices=("attention", "ffn"), help=argparse.SUPPRESS)
    parser.add_argument("--tp-rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.devices = tuple(map(int, args.devices.split(",")))
    if len(args.devices) != 2 * TP_SIZE or len(set(args.devices)) != 2 * TP_SIZE:
        parser.error("--devices requires four distinct local NPUs")
    if min(args.devices) < 0 or not 0 < args.timeout <= TOTAL_TIMEOUT_SECONDS:
        parser.error("invalid device index or timeout")
    return args


def build_config(role, pair_port):
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
        parallel_config=ParallelConfig(tensor_parallel_size=TP_SIZE),
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(load_format="dummy"),
    )
    config.device_config = DeviceConfig(device="npu")
    config.quant_config = config.engram_config = None
    config.compilation_config.mode = CompilationMode.NONE
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    config.compilation_config.use_inductor_graph_partition = False
    config.scheduler_config.max_num_batched_tokens = CAPACITY
    config.scheduler_config.max_num_seqs = CAPACITY
    for hf_config in (
        config.model_config.hf_config,
        config.model_config.hf_text_config,
    ):
        hf_config.hidden_size = HIDDEN_SIZE
        hf_config.vocab_size = VOCAB_SIZE
        hf_config.num_hidden_layers = len(LAYER_INDICES)
    config.additional_config = {
        "afd": {
            "role": role,
            "connector": "P2pHcclAFDConnector",
            "host": "127.0.0.1",
            "port": pair_port,
            "num_attention_ranks": TP_SIZE,
            "num_ffn_ranks": TP_SIZE,
            "afd_process_group_timeout_s": GROUP_TIMEOUT_SECONDS,
        }
    }
    return config


def run_rank(args):
    import torch
    import torch.distributed as dist
    import vllm_ascend.ops  # noqa: F401
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tensor_model_parallel_rank,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm_ascend.utils import adapt_patch

    from afd_plugin.compat.npu.feature_validation import (
        fail_if_unsupported_npu_afd_features,
    )
    from afd_plugin.compat.patches.npu.ascend_config import init_ascend_config
    from afd_plugin.connectors.factory import AFDConnectorFactory
    from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
    from afd_plugin.connectors.npu.hccl_protocol import HcclForwardPlan

    started = time.monotonic()
    index = args.tp_rank + (0 if args.role == "attention" else TP_SIZE)
    device = args.devices[index]
    report = {
        "role": args.role,
        "tp_rank": args.tp_rank,
        "pid": os.getpid(),
        "device": device,
        "native_world_size": TP_SIZE,
        "native_moe_executed": False,
        "checkpoint_weights_loaded": False,
        "scope": "real HCCL tensors plus controlled arithmetic on native TP groups",
        "transactions": [],
        "success": False,
    }
    result_path = args.output_dir / f"{args.role}-{args.tp_rank}.json"
    connector = None

    def native_tp_sum(value):
        dist.all_reduce(value, group=get_tp_group().device_group)
        return value

    def native_group_check():
        assert dist.get_world_size() == TP_SIZE
        assert dist.get_rank() == args.tp_rank
        assert get_tensor_model_parallel_rank() == args.tp_rank
        value = torch.tensor(
            [args.tp_rank + 1], dtype=torch.float32, device=f"npu:{device}"
        )
        assert native_tp_sum(value).item() == sum(range(1, TP_SIZE + 1))
        return value.cpu().tolist()

    def inputs(plan, layer_idx):
        values = torch.arange(plan.padded_tokens * HIDDEN_SIZE, dtype=torch.float32)
        hidden = (values.reshape(plan.padded_tokens, HIDDEN_SIZE) / 64).bfloat16()
        fp32 = hidden.float() + ROUTER_DELTA
        ids = torch.arange(plan.padded_tokens, dtype=torch.int32) + layer_idx + 7
        ids[plan.valid_tokens :] = 0
        return hidden, fp32, ids

    def local_output(hidden, fp32, ids, rank, layer_idx):
        return (
            (fp32 - hidden.float()) * ROUTER_DELTA_SCALE
            + ids[:, None].float()
            + rank
            + layer_idx
        ).bfloat16()

    try:
        current_platform.pre_register_and_update()
        load_general_plugins()
        torch.npu.set_device(device)
        adapt_patch()
        config = build_config(args.role, args.port + 2)
        try:
            fail_if_unsupported_npu_afd_features(config)
        except RuntimeError as exc:
            assert "V4.1" in str(exc) and "not available" in str(exc)
            report["full_model_guard_retained"] = True
        else:
            raise AssertionError("full model guard was unexpectedly open")
        report["config_delta"] = {
            "hidden_size": HIDDEN_SIZE,
            "vocab_size": VOCAB_SIZE,
            "num_hidden_layers": len(LAYER_INDICES),
            "quant_config": None,
            "engram_config": None,
            "load_format": "dummy",
        }
        native_port = args.port + (0 if args.role == "attention" else 1)
        with set_current_vllm_config(config):
            init_ascend_config(config)
            init_distributed_environment(
                world_size=TP_SIZE,
                rank=args.tp_rank,
                local_rank=device,
                distributed_init_method=f"tcp://127.0.0.1:{native_port}",
                backend="hccl",
                timeout=timedelta(seconds=GROUP_TIMEOUT_SECONDS),
            )
            initialize_model_parallel(
                tensor_model_parallel_size=TP_SIZE, backend="hccl"
            )
            report["native_tp_before_connector"] = native_group_check()
            connector = AFDConnectorFactory.create_connector(
                args.tp_rank, device, config
            )
            connector.init_afd_connector()
            report["native_tp_after_connector_init"] = native_group_check()
            for transaction, valid, padded, mode in TRANSACTIONS:
                plan = HcclForwardPlan(transaction, LAYER_INDICES, valid, padded, mode)
                if args.role == "attention":
                    connector.begin_forward_plan(plan)
                else:
                    assert connector.receive_forward_plan() == plan
                if padded:
                    for layer_idx in LAYER_INDICES:
                        if args.role == "attention":
                            hidden, fp32, ids = inputs(plan, layer_idx)
                            context = AFDTransferContext(
                                AFDTransferMetadata(layer_idx, 0, [padded])
                            )
                            ref = hidden.to(f"npu:{device}")
                            connector.send_attn_output(
                                ref,
                                context,
                                hidden_states_fp32=fp32.to(f"npu:{device}"),
                                input_ids=ids.to(f"npu:{device}"),
                            )
                            output = connector.recv_ffn_output(ref, context=context)
                            expected = (
                                torch.stack(
                                    [
                                        local_output(
                                            hidden, fp32, ids, rank, layer_idx
                                        ).float()
                                        for rank in range(TP_SIZE)
                                    ]
                                )
                                .sum(0)
                                .bfloat16()
                            )
                            torch.testing.assert_close(
                                output.cpu(), expected, rtol=0, atol=0
                            )
                        else:
                            payload = connector.recv_attn_output(layer_idx=layer_idx)
                            hidden, fp32, ids = inputs(plan, layer_idx)
                            torch.testing.assert_close(
                                payload.hidden_states.cpu(), hidden, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                payload.hidden_states_fp32.cpu(), fp32, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                payload.input_ids.cpu(), ids, rtol=0, atol=0
                            )
                            output = local_output(
                                payload.hidden_states,
                                payload.hidden_states_fp32,
                                payload.input_ids,
                                args.tp_rank,
                                layer_idx,
                            )
                            connector.send_ffn_output(
                                native_tp_sum(output), payload.context
                            )
                    torch.npu.synchronize(device)
                report["transactions"].append(
                    {
                        "transaction_id": transaction,
                        "valid_tokens": valid,
                        "padded_tokens": padded,
                        "mode": mode,
                        "verified_layers": list(LAYER_INDICES) if padded else [],
                    }
                )
            if args.role == "attention":
                connector.request_stop()
            else:
                assert connector.receive_forward_plan() is None
            report["stop_acknowledged"] = True
            connector.close()
            connector.close()
            report["native_tp_after_connector_close"] = native_group_check()
            report["success"] = True
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    finally:
        result_path.write_text(json.dumps(report, indent=2) + "\n")
        try:
            if connector is not None:
                connector.close()
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


def select_port_block():
    for _ in range(PORT_ATTEMPTS):
        listeners = []
        try:
            first = socket.socket()
            listeners.append(first)
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            if port + PORT_COUNT - 1 > MAX_TCP_PORT:
                continue
            for offset in range(1, PORT_COUNT):
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
    args.output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="afd-tp-probe-"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    port = select_port_block()
    processes, logs = [], []
    summary = {
        "scope": "TP2 component; controlled arithmetic, no native MoE/model",
        "success": False,
    }
    deadline = started + args.timeout - REAP_TIMEOUT_SECONDS
    try:
        for role in ("ffn", "attention"):
            for rank in range(TP_SIZE):
                key = f"{role}-{rank}"
                log = (args.output_dir / (key + ".log")).open("w")
                logs.append(log)
                command = [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--role",
                    role,
                    "--tp-rank",
                    str(rank),
                    "--port",
                    str(port),
                    "--devices",
                    ",".join(map(str, args.devices)),
                    "--output-dir",
                    str(args.output_dir),
                ]
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
        summary["success"] = not summary["external_kills"] and all(
            value["exit_code"] == 0 and value["result"]["success"]
            for value in summary["ranks"].values()
        )
        summary["elapsed_seconds"] = time.monotonic() - started
        (args.output_dir / "result.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(run_rank(parsed) if parsed.role else run_parent(parsed))
