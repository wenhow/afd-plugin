# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded actual eager-ticket component probe; no model, MoE or overlap claim.

Every payload operation goes through the public matched-warm transport and its
owner-issued loans. The CPU supervisor imports no torch. Launch only after a
separate review and explicit device allocation; this file is also reviewable
with --help on a CPU host.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import hccl_eager_backend_probe as supervision
import hccl_staged_probe as helpers

PARENT_EXTENTS = (
    ((3, 5), (2, 5)),
    ((1, 4), (1, 2)),
    ((0, 3), (0, 1)),
    ((2, 0), (0, 0)),
    ((0, 0), (0, 0)),
    ((5, 2), (4, 0)),
)
FP32_DELTA = 1 / 1_024
FP32_DELTA_SCALE = 1_024
SUPERVISOR_POLL_SECONDS = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--devices", default="0,2", help="FFN ranks then Attention ranks"
    )
    parser.add_argument("--model", default=helpers.CHECKPOINT, help="config only")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=helpers.TOTAL_TIMEOUT_SECONDS)
    parser.add_argument("--role", choices=("attention", "ffn"), help=argparse.SUPPRESS)
    parser.add_argument("--role-rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        args.devices = tuple(map(int, args.devices.split(",")))
    except ValueError:
        parser.error("devices must be comma-separated NPU indices")
    if (
        len(args.devices) != args.tp_size * 2
        or len(set(args.devices)) != len(args.devices)
        or min(args.devices) < 0
    ):
        parser.error("provide exactly two distinct NPUs per native TP coordinate")
    if not 0 < args.timeout <= helpers.TOTAL_TIMEOUT_SECONDS:
        parser.error("timeout is outside the bounded supervisor limit")
    if args.role and (
        args.port is None
        or args.role_rank is None
        or not 0 <= args.role_rank < args.tp_size
    ):
        parser.error("worker invocation requires a valid role rank and port")
    return args


def payload(torch, parent_index, stage_index, padded, valid, previous=None):
    if previous is None:
        values = torch.arange(
            padded * helpers.HIDDEN_SIZE, dtype=torch.float32
        ).reshape(padded, helpers.HIDDEN_SIZE)
        hidden = (values / 64 + parent_index / 8 + stage_index / 4).bfloat16()
    else:
        hidden = previous.clone()
    fp32 = hidden.float() + FP32_DELTA
    ids = (
        torch.arange(padded, dtype=torch.int32)
        + 1
        + parent_index * 16
        + stage_index * 8
    )
    ids[valid:] = 0
    return hidden, fp32, ids


def arithmetic(hidden, fp32, ids, layer, stage):
    return (
        hidden.float() / 4
        + (fp32 - hidden.float()) * FP32_DELTA_SCALE
        + ids[:, None].float() / 16
        + layer / 8
        + stage / 16
    ).bfloat16()


def cpu_expected(torch, parent_index, layer_position, stage_index, padded, valid):
    previous = None
    for layer in helpers.LAYERS[:layer_position]:
        previous = arithmetic(
            *payload(torch, parent_index, stage_index, padded, valid, previous),
            layer,
            stage_index,
        )
    return payload(torch, parent_index, stage_index, padded, valid, previous)


def run_attention(torch, transport, owner, plan, parent_index, compute, report):
    from afd_plugin.compat.npu.stream import use_npu_current_stream

    previous = {}
    for position, layer in enumerate(plan.layer_indices):
        tickets, originals = {}, {}
        for stage in plan.stages:
            rows, valid = (
                stage.attention_padded_tokens[transport.role_rank],
                stage.attention_valid_tokens[transport.role_rank],
            )
            if not rows:
                continue
            with use_npu_current_stream(compute):
                if stage.stage_idx in previous:
                    hidden = previous[stage.stage_idx]
                    fp32 = hidden.float() + FP32_DELTA
                    ids = payload(torch, parent_index, stage.stage_idx, rows, valid)[
                        2
                    ].to(hidden.device)
                    values = hidden, fp32, ids
                else:
                    values = tuple(
                        value.to(owner.parent_stream.device)
                        for value in payload(
                            torch, parent_index, stage.stage_idx, rows, valid
                        )
                    )
            originals[stage.stage_idx] = values
            tickets[stage.stage_idx] = owner.post_attention(
                layer,
                stage.stage_idx,
                *values,
                producer_stream=compute,
                retain=tuple(previous.values()),
            )
        report["posting_frontiers"].append(
            {
                "parent": plan.transaction_id,
                "layer": layer,
                "occupied_stages_before_output_loans": list(owner.occupied_stages),
                "ticket_stages": list(tickets),
            }
        )
        if len(owner.occupied_stages) > len(plan.stages):
            raise AssertionError("owner exceeded its U-stage occupancy bound")
        for stage in plan.stages:
            if stage.stage_idx not in tickets:
                continue
            rows, valid = (
                stage.attention_padded_tokens[transport.role_rank],
                stage.attention_valid_tokens[transport.role_rank],
            )
            expected_inputs = cpu_expected(
                torch, parent_index, position, stage.stage_idx, rows, valid
            )
            with use_npu_current_stream(compute):
                for actual, expected in zip(
                    originals[stage.stage_idx], expected_inputs, strict=True
                ):
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
            with owner.attention_output(
                tickets[stage.stage_idx],
                consumer_stream=compute,
                retain=originals[stage.stage_idx],
            ) as output:
                # This clone is the actual delayed next-layer dependency. No
                # borrowed output escapes its loan for asynchronous later use.
                previous[stage.stage_idx] = output.clone()
            expected_output = arithmetic(*expected_inputs, layer, stage.stage_idx)
            with use_npu_current_stream(compute):
                torch.testing.assert_close(
                    previous[stage.stage_idx].cpu(), expected_output, rtol=0, atol=0
                )
            report["checked_steps"].append(
                {
                    "parent": plan.transaction_id,
                    "layer": layer,
                    "stage": stage.stage_idx,
                    "rows": rows,
                    "valid": valid,
                    "independent_fp32": True,
                    "raw_ids_exact": True,
                    "full_output_exact": True,
                }
            )


def run_ffn(torch, transport, owner, plan, parent_index, compute, report):
    for position, layer in enumerate(plan.layer_indices):
        for stage in plan.stages:
            rows = stage.attention_padded_tokens[transport.role_rank]
            valid = stage.attention_valid_tokens[transport.role_rank]
            if not rows:
                continue
            expected = cpu_expected(
                torch, parent_index, position, stage.stage_idx, rows, valid
            )
            ticket = owner.post_ffn_receive(layer, stage.stage_idx)
            with owner.ffn_inputs(ticket, consumer_stream=compute) as inputs:
                for actual, reference in zip(
                    (inputs.hidden_states, inputs.hidden_states_fp32, inputs.input_ids),
                    expected,
                    strict=True,
                ):
                    torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
                output = arithmetic(
                    inputs.hidden_states,
                    inputs.hidden_states_fp32,
                    inputs.input_ids,
                    layer,
                    stage.stage_idx,
                )
            owner.post_ffn_output(ticket, output, producer_stream=compute)
            report["checked_steps"].append(
                {
                    "parent": plan.transaction_id,
                    "layer": layer,
                    "stage": stage.stage_idx,
                    "rows": rows,
                    "valid": valid,
                    "bf16_exact": True,
                    "original_fp32_exact": True,
                    "raw_ids_exact": True,
                }
            )


def worker(args):
    # Runtime isolation: supervisor and --help do not import device packages.
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

    from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
    from afd_plugin.distributed.hccl_plan import HcclLocalForward, HcclPlanCapabilities

    role_offset = args.tp_size if args.role == "attention" else 0
    device_index = args.devices[role_offset + args.role_rank]
    device = torch.device("npu", device_index)
    report = {
        "scope": "actual eager ticket/loan component; controlled arithmetic only",
        "role": args.role,
        "role_rank": args.role_rank,
        "device": str(device),
        "pid": os.getpid(),
        "success": False,
        "model_weights_loaded": False,
        "native_moe_exercised": False,
        "overlap_or_performance_claim": False,
        "checked_steps": [],
        "posting_frontiers": [],
        "parents": [],
    }

    def native_check():
        value = torch.tensor([1.0], device=device)
        dist.all_reduce(value, group=get_tp_group().device_group)
        torch.testing.assert_close(
            value.cpu(), torch.tensor([float(args.tp_size)]), rtol=0, atol=0
        )
        return value.cpu().tolist()

    try:
        current_platform.pre_register_and_update()
        load_general_plugins()
        torch.npu.set_device(device_index)
        adapt_patch()
        config = helpers.build_config(args.tp_size, args.model)
        init_ascend_config(config)
        with set_current_vllm_config(config):
            native_port = args.port + 1 + int(args.role == "attention")
            init_distributed_environment(
                world_size=args.tp_size,
                rank=args.role_rank,
                local_rank=device_index,
                distributed_init_method=f"tcp://127.0.0.1:{native_port}",
                backend="hccl",
                timeout=timedelta(seconds=helpers.GROUP_TIMEOUT_SECONDS),
            )
            initialize_model_parallel(
                tensor_model_parallel_size=args.tp_size, backend="hccl"
            )
            report["native_tp_before"] = native_check()
            transport = HcclStagedTransport(
                args.role,
                args.role_rank,
                HcclPlanCapabilities(
                    helpers.HIDDEN_SIZE,
                    helpers.VOCAB_SIZE,
                    helpers.CAPACITY,
                    helpers.LAYERS,
                    args.tp_size,
                    args.tp_size,
                    args.tp_size,
                    local_counts_version=1,
                ),
                init_method=f"tcp://127.0.0.1:{args.port}",
                local_device=device_index,
                timeout_seconds=helpers.GROUP_TIMEOUT_SECONDS,
            )
            transport.initialize()
            parent, send, receive, compute = (
                torch.npu.current_stream(device),
                *(torch.npu.Stream(device=device) for _ in range(3)),
            )
            transport.warmup_eager_pipeline(parent_stream=parent)
            report["matched_stage_groups_warmed"] = transport.eager_pipeline_warmed
            for index, (padded, valid) in enumerate(PARENT_EXTENTS):
                transaction = f"eager-owner-{index}"
                if args.role == "attention":
                    plan = transport.submit_forward(
                        HcclLocalForward("target", valid, padded),
                        transaction_id=transaction
                        if transport.is_coordinator
                        else None,
                    )
                else:
                    plan = transport.receive_forward()
                if plan is None or plan.transaction_id != transaction:
                    raise AssertionError("accepted unexpected parent")
                owner = transport.open_eager_pipeline(
                    parent_stream=parent, send_stream=send, receive_stream=receive
                )
                run = run_attention if args.role == "attention" else run_ffn
                run(torch, transport, owner, plan, index, compute, report)
                owner.finish()
                if (
                    not owner.finished
                    or transport.eager_pipeline is not None
                    or transport.plan is not None
                ):
                    raise AssertionError("parent failed to retire its eager owner")
                report["parents"].append(
                    {
                        "transaction_id": transaction,
                        "padded": list(padded),
                        "valid": list(valid),
                        "finished": True,
                    }
                )
            if args.role == "attention":
                transport.stop_attention()
            elif transport.receive_forward() is not None:
                raise AssertionError("expected normal STOP")
            report["normal_stop"] = transport.terminal
            transport.close()
            report["native_tp_after_plugin_close"] = native_check()
            destroy_model_parallel()
            destroy_distributed_environment()
            report["success"] = True
    except BaseException:
        # Keep live communication/storage in place; the supervisor owns failed
        # processes. There is no opportunistic close/abort after a tensor fault.
        report["error"] = traceback.format_exc()
        raise
    finally:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"{args.role}-{args.role_rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )


def supervise(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    port = helpers.select_port_block(3)
    children, logs, owned = {}, [], {}
    summary = {
        "scope": "actual eager owner component; no MoE/model/overlap acceptance",
        "success": False,
        "tp_size": args.tp_size,
        "devices": args.devices,
    }
    try:
        for role in ("ffn", "attention"):
            for rank in range(args.tp_size):
                name = f"{role}-{rank}"
                log = (args.output_dir / f"{name}.log").open("w")
                logs.append(log)
                children[name] = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        str(Path(__file__).resolve()),
                        "--tp-size",
                        str(args.tp_size),
                        "--devices",
                        ",".join(map(str, args.devices)),
                        "--model",
                        args.model,
                        "--output-dir",
                        str(args.output_dir),
                        "--role",
                        role,
                        "--role-rank",
                        str(rank),
                        "--port",
                        str(port),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        deadline = started + args.timeout - helpers.REAP_TIMEOUT_SECONDS
        while any(child.poll() is None for child in children.values()):
            for child in children.values():
                if child.poll() is None:
                    supervision.observe_owned(child.pid, owned)
            if any(child.poll() not in (None, 0) for child in children.values()):
                raise RuntimeError("a rank failed; retain per-rank logs")
            if time.monotonic() >= deadline:
                summary["timed_out"] = True
                break
            time.sleep(SUPERVISOR_POLL_SECONDS)
    except BaseException:
        summary["supervisor_error"] = traceback.format_exc()
    finally:
        for child in children.values():
            if child.poll() is None:
                supervision.observe_owned(child.pid, owned)
        summary["killed_owned_pids"] = (
            supervision.kill_owned(owned)
            if any(child.poll() is None for child in children.values())
            else []
        )
        for child in children.values():
            child.wait(timeout=helpers.REAP_TIMEOUT_SECONDS)
        for log in logs:
            log.close()
        summary["exit_codes"] = {
            name: child.returncode for name, child in children.items()
        }
        summary["workers"] = {
            name: json.loads(path.read_text()) if path.exists() else None
            for name in children
            for path in [args.output_dir / f"{name}.json"]
        }
        summary["owned_pid_start_ticks"] = owned
        summary["success"] = (
            not summary.get("supervisor_error")
            and not summary.get("timed_out")
            and not summary["killed_owned_pids"]
            and len(children) == args.tp_size * 2
            and all(child.returncode == 0 for child in children.values())
            and all(
                report is not None and report["success"]
                for report in summary["workers"].values()
            )
        )
        summary["elapsed_seconds"] = time.monotonic() - started
        (args.output_dir / "result.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.role:
        worker(parsed)
    else:
        raise SystemExit(supervise(parsed))
