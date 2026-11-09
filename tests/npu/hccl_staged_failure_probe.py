# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded quiescent-boundary failures on real NPU HCCL, without model weights.

These cases exercise external cooperative cancellation and a peer exit between
parents. They do not exercise abort of blocked HCCL, device faults, or a native
model/worker constructor. No cancellation message is added to the wire protocol.
"""

from __future__ import annotations

import argparse
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

CASES = ("compute-cancel", "peer-exit")
EXPECTED_PEER_EXIT = 17
EVENT_PREFIX = "FAILURE_PROBE_EVENT "
EXPECTED_COMPUTE_ERROR = "controlled FFN computation failure after real input receive"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--model", default=helpers.CHECKPOINT)
    parser.add_argument("--devices", default="0,2", help="FFN then Attention")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=helpers.TOTAL_TIMEOUT_SECONDS)
    parser.add_argument("--role", choices=("attention", "ffn"), help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.devices = tuple(map(int, args.devices.split(",")))
    if len(args.devices) != 2 or len(set(args.devices)) != 2 or min(args.devices) < 0:
        parser.error("exactly two distinct NPU device indices are required")
    if not 0 < args.timeout <= helpers.TOTAL_TIMEOUT_SECONDS:
        parser.error("timeout must fit the bounded supervisor limit")
    if args.role and args.port is None:
        parser.error("worker needs a port")
    args.tp_size = args.attention_size = args.ffn_size = 1
    return args


def emit(event):
    print(EVENT_PREFIX + json.dumps({"event": event, "pid": os.getpid()}), flush=True)


def await_command(expected):
    command = sys.stdin.readline().strip()
    if command != expected:
        raise RuntimeError(f"expected supervisor command {expected!r}, got {command!r}")


def worker(args):
    # Lazy imports isolate the actual runtime/device work from the supervisor.
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
    from vllm.forward_context import get_forward_context
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.ascend_forward_context import set_mc2_mask, set_mc2_tokens_capacity
    from vllm_ascend.utils import adapt_patch

    from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
    from afd_plugin.distributed.hccl_plan import HcclLocalForward, HcclPlanCapabilities
    from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner

    device_idx = args.devices[int(args.role == "attention")]
    device = f"npu:{device_idx}"
    result_path = args.output_dir / f"{args.role}.json"
    report = {
        "case": args.case,
        "role": args.role,
        "pid": os.getpid(),
        "device": device_idx,
        "native_tp_size": 1,
        "checkpoint_weights_loaded": False,
        "native_moe_executed": False,
        "ffn_runner_constructor_called": False,
        "blocked_hccl_abort_exercised": False,
        "device_fault_recovery_exercised": False,
        "success": False,
    }
    transport = None

    def save():
        result_path.write_text(json.dumps(report, indent=2) + "\n")

    def native_check():
        value = torch.tensor([1.0], device=device)
        dist.all_reduce(value, group=get_tp_group().device_group)
        assert value.item() == 1.0
        return value.cpu().tolist()

    def reject_reuse():
        errors = {}
        for label, operation in (
            (
                "resubmit",
                lambda: transport.submit_forward(
                    HcclLocalForward("target", (1,), (1,)),
                    transaction_id="after-failure",
                ),
            ),
            ("stop", transport.stop_attention),
        ):
            try:
                operation()
            except RuntimeError as exc:
                errors[label] = str(exc)
            else:
                raise AssertionError(f"terminal transport accepted {label}")
        report["public_reuse_rejections"] = errors

    try:
        current_platform.pre_register_and_update()
        load_general_plugins()
        torch.npu.set_device(device_idx)
        adapt_patch()
        config = helpers.build_config(1, args.model)
        init_ascend_config(config)
        with set_current_vllm_config(config):
            native_port = args.port + 1 + int(args.role == "attention")
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=device_idx,
                distributed_init_method=f"tcp://127.0.0.1:{native_port}",
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
            emit("initialized")
            if args.role == "ffn":

                class ControlledArithmetic(torch.nn.Module):
                    def compute_ffn_output(
                        self,
                        hidden_states,
                        layer_idx,
                        *,
                        input_ids,
                        hidden_states_fp32,
                        already_sequence_parallel,
                    ):
                        plan = transport.plan
                        context = get_forward_context()
                        metadata = context.additional_kwargs["afd_metadata"]
                        stage = plan.stages[metadata.stage_idx]
                        expected = helpers.expected_ffn_inputs(
                            torch, args, stage, 0, layer_idx
                        )
                        for actual, reference in zip(
                            (hidden_states, hidden_states_fp32, input_ids),
                            expected,
                            strict=True,
                        ):
                            torch.testing.assert_close(
                                actual.cpu(), reference, rtol=0, atol=0
                            )
                        assert context.additional_kwargs["model_instance"] is self
                        assert torch.equal(context.input_ids, input_ids)
                        assert not already_sequence_parallel
                        report.setdefault("received_layer_stages", []).append(
                            [layer_idx, stage.stage_idx]
                        )
                        if args.case == "compute-cancel":
                            raise ValueError(EXPECTED_COMPUTE_ERROR)
                        output = helpers.local_output(
                            hidden_states,
                            hidden_states_fp32,
                            input_ids,
                            0,
                            1,
                            layer_idx,
                            stage.stage_idx,
                        )
                        dist.all_reduce(output, group=get_tp_group().device_group)
                        return output

                runner = AFDNPUFFNModelRunner.__new__(AFDNPUFFNModelRunner)
                runner.vllm_config = config
                runner.connector = transport
                runner.model = ControlledArithmetic()
                set_mc2_tokens_capacity(config, config.scheduler_config.max_num_seqs, 1)
                set_mc2_mask(config, device)
                if args.case == "compute-cancel":
                    try:
                        runner.execute_hccl_v41_step(transport)
                    except ValueError as exc:
                        assert str(exc) == EXPECTED_COMPUTE_ERROR
                        assert transport.terminal
                        report["expected_compute_error"] = str(exc)
                        report["terminal_after_failure"] = transport.terminal
                        emit("compute-failed")
                    else:
                        raise AssertionError("controlled computation did not fail")
                    await_command("close")
                else:
                    assert runner.execute_hccl_v41_step(transport) is True
                    torch.npu.synchronize()
                    report["first_parent_completed"] = True
                    report["intentional_exit_code"] = EXPECTED_PEER_EXIT
                    report["success"] = True
                    save()
                    emit("parent-complete-exiting")
                    os._exit(EXPECTED_PEER_EXIT)
            else:
                local = HcclLocalForward("target", (3, 1), (4, 2))
                plan = transport.submit_forward(local, transaction_id="failure-first")
                assert plan is not None
                report["accepted_plan_digest"] = plan.digest
                for layer_idx in helpers.LAYERS:
                    for stage in plan.stages:
                        payload = helpers.make_attention_inputs(
                            torch, args, stage, 0, layer_idx
                        )
                        transport.send_layer_inputs(
                            layer_idx,
                            stage.stage_idx,
                            *(tensor.to(device) for tensor in payload),
                        )
                        if args.case == "compute-cancel":
                            torch.npu.synchronize()
                            report["inputs_sent_before_external_cancel"] = [
                                layer_idx,
                                stage.stage_idx,
                            ]
                            emit("inputs-sent-awaiting-cancel")
                            await_command("cancel")
                            report["external_cooperative_cancel_received"] = True
                            break
                        output = transport.receive_layer_output(
                            layer_idx, stage.stage_idx
                        )
                        torch.testing.assert_close(
                            output.cpu(),
                            helpers.expected_attention_output(
                                torch, args, stage, 0, layer_idx
                            ),
                            rtol=0,
                            atol=0,
                        )
                    if args.case == "compute-cancel":
                        break
                if args.case == "peer-exit":
                    transport.finish_forward()
                    report["first_parent_completed"] = True
                    emit("parent-complete-awaiting-peer-exit")
                    await_command("peer-exited")
                    try:
                        transport.submit_forward(local, transaction_id="failure-next")
                    except RuntimeError as exc:
                        report["expected_peer_failure"] = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                        assert transport.terminal
                        report["terminal_after_failure"] = transport.terminal
                    else:
                        raise AssertionError(
                            "next parent succeeded after peer process exited"
                        )
                    reject_reuse()
            transport.close()
            report["plugin_closed"] = transport.closed
            report["native_tp_after_plugin_close"] = native_check()
            report["success"] = True
            emit("closed-native-tp-preserved")
            destroy_model_parallel()
            destroy_distributed_environment()
    except BaseException:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
        raise
    finally:
        save()


def supervise(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    port = helpers.select_port_block(3)
    children, logs, events, forced = {}, {}, [], []
    selector = selectors.DefaultSelector()
    deadline = time.monotonic() + args.timeout
    started = time.monotonic()
    timed_out = False
    notified = False
    try:
        for role in ("ffn", "attention"):
            command = [
                sys.executable,
                "-B",
                __file__,
                "--case",
                args.case,
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
            ]
            children[role] = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                bufsize=0,
            )
            (args.output_dir / "ownership.json").write_text(
                json.dumps(
                    {
                        "supervisor_pid": os.getpid(),
                        "worker_pids": {
                            name: child.pid for name, child in children.items()
                        },
                    },
                    indent=2,
                )
                + "\n"
            )
            logs[role] = (args.output_dir / f"{role}.log").open("wb")
            selector.register(children[role].stdout, selectors.EVENT_READ, role)
        buffers = {role: b"" for role in children}
        while selector.get_map() or any(
            child.poll() is None for child in children.values()
        ):
            if time.monotonic() >= deadline:
                timed_out = True
                break
            for key, _ in selector.select(timeout=0.25):
                role = key.data
                chunk = os.read(key.fd, 65_536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                logs[role].write(chunk)
                logs[role].flush()
                buffers[role] += chunk
                while b"\n" in buffers[role]:
                    line, buffers[role] = buffers[role].split(b"\n", 1)
                    if line.startswith(EVENT_PREFIX.encode()):
                        event = json.loads(line[len(EVENT_PREFIX) :])
                        events.append(event | {"role": role})
                        print(json.dumps(events[-1]), flush=True)
                        if event["event"] == "compute-failed" and not notified:
                            for target, command in (
                                ("attention", b"cancel\n"),
                                ("ffn", b"close\n"),
                            ):
                                children[target].stdin.write(command)
                                children[target].stdin.flush()
                            notified = True
            if (
                args.case == "peer-exit"
                and children["ffn"].poll() == EXPECTED_PEER_EXIT
                and not notified
            ):
                children["attention"].stdin.write(b"peer-exited\n")
                children["attention"].stdin.flush()
                notified = True
            for role, child in children.items():
                code = child.poll()
                expected = (
                    EXPECTED_PEER_EXIT
                    if args.case == "peer-exit" and role == "ffn"
                    else 0
                )
                if code is not None and code != expected:
                    raise RuntimeError(
                        f"unexpected {role} exit {code}; see original worker log"
                    )
    finally:
        for role, child in children.items():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                forced.append({"role": role, "pid": child.pid})
            child.wait(timeout=helpers.REAP_TIMEOUT_SECONDS)
        for handle in logs.values():
            handle.close()
        selector.close()
        ranks = {
            role: json.loads(path.read_text()) if path.exists() else None
            for role in children
            for path in [args.output_dir / f"{role}.json"]
        }
        result = {
            "case": args.case,
            "scope": "real HCCL quiescent cancellation / between-parent peer exit",
            "blocked_hccl_abort_exercised": False,
            "device_fault_recovery_exercised": False,
            "elapsed_seconds": time.monotonic() - started,
            "timed_out": timed_out,
            "forced_owned_processes": forced,
            "exit_codes": {role: child.returncode for role, child in children.items()},
            "events": events,
            "ranks": ranks,
            "success": not timed_out
            and not forced
            and notified
            and all(rank is not None and rank["success"] for rank in ranks.values()),
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    return 0 if result["success"] else 1


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.role:
        worker(arguments)
    else:
        raise SystemExit(supervise(arguments))
