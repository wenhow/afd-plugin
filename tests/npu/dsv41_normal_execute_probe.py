# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Normal C2 Worker: real Scheduler prefill, decode, and empty retirement.

Metadata/help use stdlib only. Explicit paired --run preserves native model,
Attention, HC, compressed cache, remote MoE and sampler execution. Output IDs
are observed, never fixed. This component does not establish native parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from dsv41_normal_c2_probe import (
    C2_COMPRESS_RATIO,
    C2_SOURCE_LAYER,
    CANDIDATE_SOURCE_LAYER,
    DEFAULT_AFD_PORT,
    DEFAULT_TIMEOUT_SECONDS,
    NATIVE_PLUGINS,
    POLL_SECONDS,
    QUALIFICATION_ENV,
    VOCAB_SIZE,
    build_config,
    config_record,
    initialize_attention_kv,
    observe_controls,
    parameter_record,
    record_loaded_model,
    runtime_record,
    source_record,
    write_json,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

PROMPT_TOKEN_IDS = (11, 12, 13, 14)
GENERATED_TOKENS = 2
REQUEST_ID = "normal-c2-first-request"
METADATA_QUERY_LENGTH_DIAGNOSTIC_ENV = "AFD_DSV41_METADATA_QUERY_LENGTH_DIAGNOSTIC"


def tensor_record(tensor, report: dict, label: str) -> dict:
    # Explicit eager diagnostics: blocking readback is observational overhead,
    # not a replacement for native completion or a performance measurement.
    import torch

    host = tensor.detach().cpu().contiguous()
    raw = host.reshape(-1).view(torch.uint8).numpy().tobytes()
    name = label + ".bin"
    (Path(report["output_dir"]) / name).write_bytes(raw)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr": tensor.data_ptr(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "raw_file": name,
        "finite": bool(torch.isfinite(host).all()),
    }


@contextmanager
def observe_smla_metadata(report: dict):
    # Runtime-only diagnosis: reading the small input tensors back can synchronize
    # NPU work. The default preserves original inputs, result and exception. The
    # explicit query-length experiment changes only an absent/None seqused_q;
    # this isolates optional-empty handling and is not a production fix.
    import torch
    import torch_npu

    namespace = torch.ops._C_ascend
    original = namespace.npu_sparse_flash_mla_metadata
    schema = original.default._schema
    query_length_diagnostic = (
        os.environ.get(METADATA_QUERY_LENGTH_DIAGNOSTIC_ENV) == "1"
    )
    records = report["smla_metadata_calls"] = []
    path = Path(report["output_dir"]) / "metadata-calls.json"

    def persist():
        write_json(
            path,
            {
                "operator": "torch.ops._C_ascend.npu_sparse_flash_mla_metadata",
                "schema": str(schema),
                "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
                "input_readback_is_diagnostic_overhead": True,
                "query_length_diagnostic_requested": query_length_diagnostic,
                "diagnostic_inputs_changed": any(
                    call["diagnostic_inputs_changed"] for call in records
                ),
                "inputs_or_result_replaced": any(
                    call["diagnostic_inputs_changed"] for call in records
                ),
                "operator_outputs_replaced": False,
                "calls": records,
            },
        )

    def diagnostic_read(call, location, read):
        try:
            return read()
        except Exception:
            call["diagnostic_errors"].append(
                {"location": location, "traceback": traceback.format_exc()}
            )
            return None

    def describe(value, call, location):
        if isinstance(value, torch.Tensor):
            description = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "device": str(value.device),
            }
            reads = {
                "data_ptr": value.data_ptr,
                "stride": lambda: list(value.stride()),
                "storage_offset": value.storage_offset,
                "is_contiguous": value.is_contiguous,
                "storage_nbytes": lambda: value.untyped_storage().nbytes(),
                "storage_data_ptr": lambda: value.untyped_storage().data_ptr(),
                "npu_format": (
                    lambda: (
                        torch_npu.get_npu_format(value)
                        if value.device.type == "npu"
                        else None
                    )
                ),
            }
            for name, read in reads.items():
                description[name] = diagnostic_read(call, f"{location}.{name}", read)
            return description
        return {"value": value}

    def current_stream_record():
        stream = torch.npu.current_stream()
        # The pinned Ascend PyHCCL communicator uses this public stream handle.
        return {"npu_stream": stream.npu_stream, "device": str(stream.device)}

    def observed(*args, **kwargs):
        bound = {
            schema.arguments[index].name: value for index, value in enumerate(args)
        }
        bound.update(kwargs)
        call = {
            "ordinal": len(records),
            "phase": "input_snapshot",
            "positional_arguments": [
                schema.arguments[index].name for index in range(len(args))
            ],
            "keyword_arguments": list(kwargs),
            "inputs": {},
            "diagnostic_errors": [],
            "original_seqused_q_present": "seqused_q" in bound,
            "diagnostic_inputs_changed": False,
        }
        thread = threading.current_thread()
        call["runtime_context"] = {
            "thread_name": thread.name,
            "thread_ident": thread.ident,
        }
        context_reads = {
            "current_device": torch.npu.current_device,
            "current_stream": current_stream_record,
            "inference_mode_enabled": torch.is_inference_mode_enabled,
            "grad_enabled": torch.is_grad_enabled,
            "default_dtype": lambda: str(torch.get_default_dtype()),
        }
        for name, read in context_reads.items():
            call["runtime_context"][name] = diagnostic_read(
                call, f"runtime_context.{name}", read
            )
        call["inputs"] = {
            name: describe(value, call, f"inputs.{name}")
            for name, value in bound.items()
        }
        records.append(call)
        persist()
        try:
            for name, value in bound.items():
                if isinstance(value, torch.Tensor):
                    call["inputs"][name]["values"] = diagnostic_read(
                        call,
                        f"inputs.{name}.values",
                        lambda value=value: value.detach().cpu().tolist(),
                    )
                    persist()
            forwarded_kwargs = kwargs
            if query_length_diagnostic and (
                "seqused_q" not in bound or bound["seqused_q"] is None
            ):
                # 7e CheckConsistencySmla accepts 1D int32 seqused_q with batch
                # length len(cuq)-1 and values no larger than each cuq segment.
                # Subtraction retains the real input tensor's dtype and device.
                call["phase"] = "query_length_diagnostic_input"
                cuq = bound["cu_seqlens_q"]
                query_lengths = cuq[1:] - cuq[:-1]
                forwarded_kwargs = {**kwargs, "seqused_q": query_lengths}
                call["diagnostic_inputs_changed"] = True
                call["forwarded_seqused_q"] = describe(
                    query_lengths, call, "forwarded_seqused_q"
                )
                persist()
                call["forwarded_seqused_q"]["values"] = diagnostic_read(
                    call,
                    "forwarded_seqused_q.values",
                    lambda: query_lengths.detach().cpu().tolist(),
                )
                persist()
            else:
                call["forwarded_seqused_q"] = (
                    dict(call["inputs"]["seqused_q"])
                    if "seqused_q" in bound
                    else {"value": None}
                )
            call["phase"] = "operator_enter"
            persist()
            result = original(*args, **forwarded_kwargs)
        except BaseException:
            call["exception_phase"] = call["phase"]
            call["phase"] = "raised"
            call["traceback"] = traceback.format_exc()
            try:
                persist()
            except BaseException:
                report["smla_metadata_record_write_error"] = traceback.format_exc()
            raise
        call["phase"] = "operator_returned"
        # Metadata contents are not used as expected output or read back here.
        call["result"] = describe(result, call, "result")
        persist()
        return result

    persist()
    try:
        namespace.npu_sparse_flash_mla_metadata = observed
        yield
    finally:
        namespace.npu_sparse_flash_mla_metadata = original


@contextmanager
def observe_payloads(report: dict):
    # Wrap the actual legacy transport methods once and preserve their results.
    # Each process owns its own class bindings, including F's daemon thread.
    from afd_plugin.connectors.npu.hccl_transport import HcclEagerTransport

    records = report["layer_payloads"] = []
    send_inputs = HcclEagerTransport.send_layer_inputs
    receive_inputs = HcclEagerTransport.receive_layer
    send_output = HcclEagerTransport.send_layer_output
    receive_output = HcclEagerTransport.receive_layer_output

    def record(transport, transaction, layer, kind, tensors):
        ordinal = len(records)
        records.append(
            {
                "ordinal": ordinal,
                "role": transport.role.value,
                "transaction_id": transaction,
                "layer": layer,
                "kind": kind,
                "tensors": {
                    key: tensor_record(value, report, f"payload-{ordinal}-{key}")
                    for key, value in tensors.items()
                },
            }
        )
        write_json(
            Path(report["output_dir"]) / "payloads.json", {"layer_payloads": records}
        )

    def observed_send(transport, layer, hidden, router, ids):
        transaction = transport.plan.transaction_id
        result = send_inputs(transport, layer, hidden, router, ids)
        record(
            transport,
            transaction,
            layer,
            "inputs",
            {"hidden": hidden, "router": router, "ids": ids},
        )
        return result

    def observed_receive(transport, layer):
        transaction = transport.plan.transaction_id
        result = receive_inputs(transport, layer)
        record(
            transport,
            transaction,
            layer,
            "inputs",
            {
                "hidden": result.hidden_states,
                "router": result.hidden_states_fp32,
                "ids": result.input_ids,
            },
        )
        return result

    def observed_send_output(transport, layer, output):
        transaction = transport.plan.transaction_id
        result = send_output(transport, layer, output)
        record(transport, transaction, layer, "output", {"output": output})
        return result

    def observed_receive_output(transport, layer):
        transaction = transport.plan.transaction_id
        result = receive_output(transport, layer)
        record(transport, transaction, layer, "output", {"output": result})
        return result

    try:
        HcclEagerTransport.send_layer_inputs = observed_send
        HcclEagerTransport.receive_layer = observed_receive
        HcclEagerTransport.send_layer_output = observed_send_output
        HcclEagerTransport.receive_layer_output = observed_receive_output
        yield
    finally:
        HcclEagerTransport.send_layer_inputs = send_inputs
        HcclEagerTransport.receive_layer = receive_inputs
        HcclEagerTransport.send_layer_output = send_output
        HcclEagerTransport.receive_layer_output = receive_output


def schedule_record(step) -> dict:
    return {
        "total_num_scheduled_tokens": step.total_num_scheduled_tokens,
        "num_scheduled_tokens": dict(step.num_scheduled_tokens),
        "finished_req_ids": sorted(step.finished_req_ids),
        "new_request_ids": [item.req_id for item in step.scheduled_new_reqs],
        "cached_request_ids": list(step.scheduled_cached_reqs.req_ids),
        "new_block_ids_to_zero": step.new_block_ids_to_zero,
    }


def run_request(wrapper, report: dict) -> None:
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        resolve_kv_cache_block_sizes,
    )
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager

    worker, runner = wrapper.worker, wrapper.worker.model_runner
    config = worker.vllm_config
    assert config.model_config.skip_tokenizer_init is True
    assert config.model_config.is_multimodal_model is True
    assert config.model_config.multimodal_config.language_model_only is True
    manager = StructuredOutputManager(config)
    scheduler_kv_config = generate_scheduler_kv_cache_config([runner.kv_cache_config])
    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        scheduler_kv_config, config
    )
    scheduler = Scheduler(
        config,
        scheduler_kv_config,
        manager,
        block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
    )
    params = SamplingParams(
        temperature=0, max_tokens=GENERATED_TOKENS, ignore_eos=True, detokenize=False
    )
    params.verify(
        config.model_config,
        config.speculative_config,
        config.structured_outputs_config,
        None,
    )
    request = Request(REQUEST_ID, list(PROMPT_TOKEN_IDS), params, None)
    assert not request.mm_features and not request.use_structured_output
    assert request.kv_transfer_params is None
    report["scheduler"] = {
        "class": source_record(Scheduler),
        "request_class": source_record(Request),
        "structured_manager_class": source_record(StructuredOutputManager),
        "native_block_size_resolver": source_record(resolve_kv_cache_block_sizes),
        "scheduler_block_size": scheduler_block_size,
        "hash_block_size": hash_block_size,
        "wrapper_execute_method": source_record(type(wrapper).execute_model),
        "wrapper_mm_cache_is_none": wrapper.mm_receiver_cache is None,
        "prompt": list(PROMPT_TOKEN_IDS),
        "max_tokens": GENERATED_TOKENS,
        "skip_tokenizer_init": True,
        "model_is_multimodal": True,
        "language_model_only": True,
        "steps": [],
    }
    detail = report["scheduler"]
    actual_outputs = []
    scheduler.add_request(request)
    try:
        for ordinal in range(GENERATED_TOKENS):
            before = request.num_computed_tokens
            step = scheduler.schedule()
            expected_ids = (
                list(PROMPT_TOKEN_IDS) if ordinal == 0 else [actual_outputs[-1]]
            )
            expected_positions = (
                list(range(len(PROMPT_TOKEN_IDS)))
                if ordinal == 0
                else [len(PROMPT_TOKEN_IDS)]
            )
            assert step.num_scheduled_tokens == {REQUEST_ID: len(expected_ids)}
            assert step.total_num_scheduled_tokens == len(expected_ids)
            assert request.num_computed_tokens == before + len(expected_ids)
            output = wrapper.execute_model(step)
            report["inference_executed"] = True
            assert output is None
            state = runner.execute_model_state
            assert state is not None
            batch = state.input_batch
            assert batch.req_ids == [REQUEST_ID]
            assert batch.num_tokens == len(expected_ids)
            ids = batch.input_ids[: batch.num_tokens].cpu().tolist()
            positions = batch.positions[: batch.num_tokens].cpu().tolist()
            assert ids == expected_ids and positions == expected_positions
            hidden = tensor_record(
                state.hidden_states, report, f"target-{ordinal}-hidden"
            )
            assert hidden["finite"]
            grammar = scheduler.get_grammar_bitmask(step)
            assert grammar is None
            sampled = wrapper.sample_tokens(grammar)
            assert isinstance(sampled, AsyncModelRunnerOutput)
            returned = sampled.get_output()
            assert isinstance(returned, ModelRunnerOutput)
            assert returned.req_ids == [REQUEST_ID]
            assert len(returned.sampled_token_ids) == 1
            assert len(returned.sampled_token_ids[0]) == 1
            token = returned.sampled_token_ids[0][0]
            assert type(token) is int and 0 <= token < VOCAB_SIZE
            assert runner.execute_model_state is None
            actual_outputs.append(token)
            index = runner.req_states.req_id_to_index[REQUEST_ID]
            total = int(runner.req_states.total_len.gpu[index].cpu())
            history = runner.req_states.all_token_ids.gpu[index, :total].cpu().tolist()
            assert history == list(PROMPT_TOKEN_IDS) + actual_outputs
            engine_outputs = scheduler.update_from_output(step, returned)
            assert list(request.output_token_ids) == actual_outputs
            assert request.num_computed_tokens == len(PROMPT_TOKEN_IDS) + ordinal
            detail["steps"].append(
                {
                    "schedule": schedule_record(step),
                    "input_ids": ids,
                    "positions": positions,
                    "computed_before_schedule": before,
                    "computed_after_update": request.num_computed_tokens,
                    "request_index": index,
                    "worker_history": history,
                    "sampled_token_ids": returned.sampled_token_ids,
                    "request_status": request.status.name,
                    "engine_output_client_ids": sorted(engine_outputs),
                    "hidden": hidden,
                    "native_state_identity": id(state),
                }
            )
            write_json(Path(report["output_dir"]) / "execute-progress.json", detail)
        assert request.status is RequestStatus.FINISHED_LENGTH_CAPPED
        assert REQUEST_ID not in scheduler.requests
        retirement = scheduler.schedule()
        assert retirement.total_num_scheduled_tokens == 0
        assert retirement.finished_req_ids == {REQUEST_ID}
        empty = wrapper.execute_model(retirement)
        assert isinstance(empty, ModelRunnerOutput)
        assert runner.execute_model_state is None
        scheduler.update_from_output(retirement, empty)
        assert REQUEST_ID not in runner.req_states.req_id_to_index
        assert not scheduler.has_requests()
        assert not scheduler.requests and not scheduler.finished_req_ids
        detail.update(
            {
                "retirement": schedule_record(retirement),
                "empty_output_executed": True,
                "request_status": request.status.name,
                "actual_output_ids": actual_outputs,
                "worker_request_retired": True,
                "scheduler_empty": True,
            }
        )
    finally:
        scheduler.shutdown()
    report["native_scheduler_prefill_decode_retirement_accepted"] = True


def run_model(args: argparse.Namespace, config: VllmConfig, report: dict) -> None:
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel

    wrapper = WorkerWrapperBase()
    constructed = False
    with observe_controls(report), observe_payloads(report):
        try:
            wrapper.init_worker(
                [
                    {
                        "vllm_config": config,
                        "local_rank": args.device,
                        "rank": 0,
                        "distributed_init_method": (args.output_dir / "store").as_uri(),
                        "is_driver_worker": True,
                    }
                ]
            )
            constructed = True
            worker = wrapper.worker
            report["worker_class"] = source_record(type(worker))
            wrapper.init_device()
            report["normal_role_init_device_completed"] = True
            report["worker_configuration"] = config_record(worker.vllm_config)
            report["runner_class"] = source_record(type(worker.model_runner))
            report["actual_device"] = str(worker.device)
            assert worker.device == torch.device("npu", args.device)
            with set_current_vllm_config(worker.vllm_config), torch.inference_mode():
                wrapper.load_model()
                report["normal_load_model_returned"] = True
                report["model_loaded"] = True
                # A load includes HELLO. F must enter its normal empty-cache
                # startup now; waiting for A's load return first would deadlock.
                if args.role == "ffn":
                    spec = worker.get_kv_cache_spec()
                    assert spec == {}
                    configs = get_kv_cache_configs(worker.vllm_config, [spec], [0])
                    assert len(configs) == 1
                    assert configs[0].num_blocks == 1
                    assert configs[0].kv_cache_tensors == []
                    assert configs[0].kv_cache_groups == []
                    report["ffn_native_empty_kv_plan"] = {
                        "num_blocks": configs[0].num_blocks,
                        "tensors": len(configs[0].kv_cache_tensors),
                        "groups": len(configs[0].kv_cache_groups),
                        "planner": source_record(get_kv_cache_configs),
                    }
                    wrapper.initialize_from_config(configs)
                    report["ffn_empty_cache_initialized"] = True
                # F's daemon may already consume A's legitimate STOP before
                # management resumes; completed HELLO codec calls are stable.
                assert len(report["pair_control_messages"]) >= 2
                assert all(
                    item["encoded_message"]["kind"] == "HELLO"
                    for item in report["pair_control_messages"][:2]
                )
                report["normal_connector_hello_completed"] = True
                report["loaded_model"] = record_loaded_model(worker, args.role)
                if args.initialize_attention_kv:
                    initialize_attention_kv(wrapper, report)
                    with observe_smla_metadata(report):
                        run_request(wrapper, report)
                    report["inference_executed"] = True
                write_json(args.output_dir / "progress.json", report)
                if args.role == "ffn":
                    deadline = time.monotonic() + args.timeout
                    while worker.poll_hccl_ffn_service():
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "FFN normal service did not receive peer STOP"
                            )
                        time.sleep(POLL_SECONDS)
                    report["ffn_public_poll_observed_peer_stop_and_owner_exit"] = True
                    report["inference_executed"] = len(report["layer_payloads"]) > 0
                model = worker.model_runner.get_model()
                after = {
                    name: parameter_record(parameter)
                    for name, parameter in model.named_parameters()
                }
                assert after == report["loaded_model"]["parameters"]
                report["loaded_parameters_after_execution"] = after
                report["loaded_parameters_unchanged"] = True
        except BaseException:
            report["model_failure_traceback"] = traceback.format_exc()
            raise
        finally:
            callbacks = [("worker_shutdown", wrapper.shutdown)] if constructed else []
            callbacks.extend(
                (
                    ("ascend_groups", destroy_ascend_model_parallel),
                    ("model_groups", destroy_model_parallel),
                    ("distributed_environment", destroy_distributed_environment),
                )
            )
            failures = []
            for name, callback in callbacks:
                try:
                    callback()
                    report[name + "_completed"] = True
                except BaseException:
                    failures.append(
                        {"phase": name, "traceback": traceback.format_exc()}
                    )
            report["cleanup_failures"] = failures
            report["native_groups_destroyed"] = not torch.distributed.is_initialized()
            if failures and sys.exc_info()[0] is None:
                raise RuntimeError("normal model cleanup failed; see cleanup_failures")
    report["native_model_load_component_accepted"] = True
    report["native_c2_kv_component_accepted"] = bool(
        args.initialize_attention_kv and report["attention_kv_initialized"]
    )
    report["native_attention_kv_component_accepted"] = bool(
        args.initialize_attention_kv and report["attention_kv_initialized"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--role", choices=("attention", "ffn"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--afd-port", type=int, default=DEFAULT_AFD_PORT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    if args.device < 0 or not 0 < args.afd_port < 65536 or args.timeout <= 0:
        parser.error("device >=0, valid port and positive timeout required")
    if args.run and (args.model is None or args.output_dir is None):
        parser.error("--run requires --model and fresh --output-dir")
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
        if args.output_dir.exists():
            parser.error("--output-dir must not exist")
    args.initialize_attention_kv = args.role == "attention"
    report = {
        "role": args.role,
        "run": args.run,
        "runtime_imported": False,
        "compress_ratios": [0, 0, C2_COMPRESS_RATIO],
        "c2_source_layer": C2_SOURCE_LAYER,
        "candidate_source_layer": CANDIDATE_SOURCE_LAYER,
        "model_loaded": False,
        "attention_kv_initialized": False,
        "initialize_attention_kv_requested": args.initialize_attention_kv,
        "ffn_empty_cache_initialized": False,
        "pure_swa_patch_installed": False,
        "native_attention_kv_component_accepted": False,
        "native_c2_kv_component_accepted": False,
        "native_model_load_component_accepted": False,
        "native_scheduler_prefill_decode_retirement_accepted": False,
        "inference_executed": False,
        "numerical_equivalence_accepted": False,
        "full_checkpoint_accepted": False,
        "success": False,
    }
    if args.run:
        args.output_dir.mkdir(parents=True)
        report["output_dir"] = str(args.output_dir)
        os.environ.setdefault("VLLM_PLUGINS", NATIVE_PLUGINS)
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
        os.environ.setdefault(QUALIFICATION_ENV, "1")
        try:
            report["runtime_imported"] = True
            config = build_config(args, report)
            run_model(args, config, report)
            report["success"] = True
        except BaseException:
            report["traceback"] = traceback.format_exc()
        finally:
            try:
                report["runtime"] = runtime_record()
            except BaseException:
                report["runtime_record_error"] = traceback.format_exc()
                report["success"] = False
            write_json(args.output_dir / "result.json", report)
    else:
        report["success"] = True
        report["scope"] = "stdlib metadata; normal C2 prefill/decode not executed"
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
