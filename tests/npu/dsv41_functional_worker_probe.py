# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A5 normal Worker functional candidate, with explicit bounded dummy geometry.

Default/help are stdlib-only. --run-config constructs the real initial config
without initializing NPU. --run uses the normal Worker factory, loader, native
KV planner, warmup/capture, Scheduler, sampler and retirement. A/F roles must
run concurrently under an external bounded supervisor. Native is an eager-U1
baseline. Readbacks and host observers are diagnostic overhead, not benchmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path

from dsv41_dspark_a5_runtime import NATIVE_INDEX_HEADS, native_a5_bootstrap
from dsv41_normal_c2_probe import (
    ATTENTION_KV_BUDGET_BYTES,
    BLOCK_SIZE,
    DEFAULT_AFD_PORT,
    DEFAULT_TIMEOUT_SECONDS,
    GROUP_TIMEOUT_SECONDS,
    HC_MULTIPLIER,
    HEAD_SIZE,
    HIDDEN_SIZE,
    LAYER_COUNT,
    MAX_BATCHED_TOKENS,
    MAX_MODEL_LEN,
    NATIVE_PLUGINS,
    POLL_SECONDS,
    QUERY_HEADS,
    VOCAB_SIZE,
    bounded_c2_hf_config,
    observe_controls,
    parameter_record,
    record_loaded_model,
    source_record,
    write_json,
)
from dsv41_normal_execute_probe import schedule_record, tensor_record
from dsv41_normal_native_probe import (
    canonical_parameter_groups,
    record_native_model,
)
from dsv41_normal_worker_probe import INDEX_HEAD_SIZE

FUNCTIONAL_ENV = "AFD_DSV41_FUNCTIONAL_QUALIFICATION"
OLDER_QUALIFICATIONS = (
    "AFD_DSV41_EAGER_QUALIFICATION",
    "AFD_DSV41_DUMMY_QUALIFICATION",
)
MAX_REQUESTS = 2
UBATCH_THRESHOLD = 2
GRAPH_CAPTURE_SIZES = (1, 2)
MAX_SCENE_STEPS = 16
PROMPTS = {
    "functional-A": (11, 12, 13, 14),
    "functional-B": (21, 22, 23, 24),
    "functional-C": (31, 32, 33, 34),
}
OUTPUT_LIMITS = {"functional-A": 4, "functional-B": 4, "functional-C": 2}
POLICY_ENVIRONMENTS = (
    "AFD_HCCL_EAGER_U2_STREAM_OVERLAP",
    "AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP",
    "AFD_HCCL_GRAPH_U2_HYBRID_DAG",
    "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM",
    "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM",
    "AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER",
)


def execution_environment(args: argparse.Namespace) -> dict[str, str]:
    graph = args.mode.startswith("graph")
    policies = {name: "0" for name in POLICY_ENVIRONMENTS}
    if args.overlap:
        selected = POLICY_ENVIRONMENTS[1:] if graph else POLICY_ENVIRONMENTS[:1]
        policies.update({name: "1" for name in selected})
    return {
        **policies,
        "AFD_HCCL_STAGE_DIAGNOSTICS": "0" if args.role == "native" else "1",
        "AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_PLUGINS": NATIVE_PLUGINS,
        FUNCTIONAL_ENV: "0" if args.role == "native" else "1",
        **{name: "0" for name in OLDER_QUALIFICATIONS},
    }


def bounded_a5_c2_hf_config(hf_config):
    """Declare A5's packed QLI geometry before native model construction."""
    bounded = bounded_c2_hf_config(hf_config)
    raw = bounded.to_dict()
    raw["model_type"] = bounded.model_type
    raw["index_n_heads"] = NATIVE_INDEX_HEADS
    if "text_config" in raw:
        raw["text_config"] = {
            **raw["text_config"],
            "index_n_heads": NATIVE_INDEX_HEADS,
        }
    return type(bounded)(**raw)


def configuration_record(config) -> dict:
    model, parallel = config.model_config, config.parallel_config
    text = model.hf_text_config
    return {
        "config_identity": id(config),
        "worker_cls": parallel.worker_cls,
        "native_geometry": {
            "hidden_size": model.get_hidden_size(),
            "head_size": model.get_head_size(),
            "vocab_size": model.get_vocab_size(),
            "num_layers": model.get_num_layers(parallel),
            "num_attention_heads": text.num_attention_heads,
            "index_n_heads": text.index_n_heads,
            "index_head_dim": text.index_head_dim,
            "hc_mult": text.hc_mult,
            "compress_ratios": list(text.compress_ratios),
        },
        "max_num_seqs": config.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        "max_model_len": model.max_model_len,
        "enable_chunked_prefill": config.scheduler_config.enable_chunked_prefill,
        "enforce_eager": model.enforce_eager,
        "use_ubatching": parallel.use_ubatching,
        "num_ubatches": parallel.num_ubatches,
        "dbo_prefill_token_threshold": parallel.dbo_prefill_token_threshold,
        "dbo_decode_token_threshold": parallel.dbo_decode_token_threshold,
        "compile_mode": config.compilation_config.mode.name,
        "graph_mode": config.compilation_config.cudagraph_mode.name,
        "capture_sizes": config.compilation_config.cudagraph_capture_sizes,
        "language_model_only": model.multimodal_config.language_model_only,
        "model_is_multimodal": model.is_multimodal_model,
        "skip_tokenizer_init": model.skip_tokenizer_init,
        "load_format": config.load_config.load_format,
        "speculation_is_none": config.speculative_config is None,
        "pd_is_none": config.kv_transfer_config is None,
        "engram_is_none": config.engram_config is None,
        "quantization_is_none": model.quantization is config.quant_config is None,
        "additional_config": config.additional_config,
    }


def build_config(args: argparse.Namespace, report: dict):
    # Isolation leaf: no runtime package is imported by metadata/help paths.
    import torch
    import torch_npu  # noqa: F401
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins

    report["npu_initialized_before_config"] = torch.npu.is_initialized()
    if args.run:
        # Native config queries may initialize NPU on A5. Bind the role's
        # visible ordinal first, before those queries can select a default.
        torch.npu.set_device(args.device)
        report["configuration_device_before"] = torch.npu.current_device()
        assert report["configuration_device_before"] == args.device
    else:
        assert report["npu_initialized_before_config"] is False
    assert current_platform.device_type == "npu"
    current_platform.pre_register_and_update()
    load_general_plugins()
    if args.role != "native":
        from afd_plugin.compat.npu import apply_afd_ascend_config_patch_if_needed

        apply_afd_ascend_config_patch_if_needed()
        import afd_plugin.compat.patches.config_validation  # noqa: F401

    original_path = Path(args.model) / "config.json"
    original = original_path.read_bytes()
    declaration = json.loads(original)
    for container in (declaration, declaration.get("text_config", {})):
        container.pop("quantization_config", None)
        container.pop("compression_config", None)
        container["expert_dtype"] = "bf16"
    controlled = args.output_dir / "controlled-model"
    controlled.mkdir()
    (controlled / "config.json").write_text(json.dumps(declaration) + "\n", "utf-8")
    graph, u2 = args.mode.startswith("graph"), args.mode.endswith("u2")
    additional = {"enable_cpu_binding": False}
    if args.role != "native":
        additional["afd"] = {
            "connector": "P2pHcclAFDConnector",
            "role": args.role,
            "host": "127.0.0.1",
            "num_attention_ranks": 1,
            "num_ffn_ranks": 1,
            "port": args.afd_port,
            "afd_process_group_timeout_s": GROUP_TIMEOUT_SECONDS,
            "connector_extra_config": {
                "max_tokens": MAX_BATCHED_TOKENS,
                "graph_exchange_order": "stage_serial",
            },
        }
    report["checkpoint_config_sha256"] = hashlib.sha256(original).hexdigest()
    report["first_declaration"] = {
        "role": args.role,
        "mode": args.mode,
        "hf_override": "bounded_a5_c2_hf_config",
        "hidden_size": HIDDEN_SIZE,
        "head_size": HEAD_SIZE,
        "num_attention_heads": QUERY_HEADS,
        "index_n_heads": NATIVE_INDEX_HEADS,
        "index_head_dim": INDEX_HEAD_SIZE,
        "hc_mult": HC_MULTIPLIER,
        "num_hidden_layers": LAYER_COUNT,
        "compress_ratios": [0, 0, 2],
        "max_num_seqs": MAX_REQUESTS,
        "ubatch_size": 2 if u2 else 0,
        "enforce_eager": not graph,
        "capture_sizes": list(GRAPH_CAPTURE_SIZES) if graph else [],
        "kv_budget_bytes": ATTENTION_KV_BUDGET_BYTES,
        "additional_config": additional,
        "geometry_or_mode_mutated_after_construction": False,
    }
    config = EngineArgs(
        model=str(controlled),
        tokenizer=args.model,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        language_model_only=True,
        dtype="bfloat16",
        hf_overrides=bounded_a5_c2_hf_config,
        load_format="dummy",
        enforce_eager=not graph,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_REQUESTS,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        disable_hybrid_kv_cache_manager=False,
        block_size=BLOCK_SIZE,
        kv_cache_memory_bytes=ATTENTION_KV_BUDGET_BYTES,
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        enable_dbo=False,
        ubatch_size=2 if u2 else 0,
        dbo_prefill_token_threshold=UBATCH_THRESHOLD,
        dbo_decode_token_threshold=UBATCH_THRESHOLD,
        async_scheduling=False,
        seed=0,
        speculative_config=None,
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=(
                CUDAGraphMode.FULL_DECODE_ONLY if graph else CUDAGraphMode.NONE
            ),
            cudagraph_capture_sizes=list(GRAPH_CAPTURE_SIZES) if graph else [],
            max_cudagraph_capture_size=max(GRAPH_CAPTURE_SIZES) if graph else None,
            cudagraph_num_of_warmups=1,
        ),
        additional_config=additional,
    ).create_engine_config()
    if args.role != "native":
        from afd_plugin.compat.npu.feature_validation import (
            fail_if_unsupported_npu_afd_features,
        )

        fail_if_unsupported_npu_afd_features(config)
    actual = report["configuration"] = configuration_record(config)
    assert actual["native_geometry"] == {
        "hidden_size": HIDDEN_SIZE,
        "head_size": HEAD_SIZE,
        "vocab_size": VOCAB_SIZE,
        "num_layers": LAYER_COUNT,
        "num_attention_heads": QUERY_HEADS,
        "index_n_heads": NATIVE_INDEX_HEADS,
        "index_head_dim": INDEX_HEAD_SIZE,
        "hc_mult": HC_MULTIPLIER,
        "compress_ratios": [0, 0, 2],
    }
    assert actual["max_num_seqs"] == MAX_REQUESTS
    assert actual["use_ubatching"] is u2
    assert actual["enforce_eager"] is (not graph)
    assert actual["compile_mode"] == "NONE"
    assert actual["graph_mode"] == ("FULL_DECODE_ONLY" if graph else "NONE")
    assert actual["language_model_only"] is True
    assert actual["model_is_multimodal"] is True
    assert all(
        actual[key]
        for key in (
            "speculation_is_none",
            "pd_is_none",
            "engram_is_none",
            "quantization_is_none",
        )
    )
    assert original_path.read_bytes() == original
    report["npu_initialized_after_config"] = torch.npu.is_initialized()
    if args.run:
        report["configuration_device_after"] = torch.npu.current_device()
        assert report["configuration_device_after"] == args.device
    else:
        assert report["npu_initialized_after_config"] is False
    report["initial_engine_args_configuration_accepted"] = True
    return config


@contextmanager
def observe_runtime(report: dict):
    """Delegate unchanged control calls and retain actual host diagnostic fields."""
    from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
    from afd_plugin.distributed.hccl_plan import encode_hccl_plan_message

    records = report["staged_control_messages"] = []
    host_records = report["host_diagnostic_records"] = []
    lock = threading.Lock()
    send, receive = HcclStagedTransport._send_control, HcclStagedTransport._recv_control
    path = Path(report["output_dir"])

    def record(direction, transport, rank, message):
        encoded = encode_hccl_plan_message(message)
        with lock:
            records.append(
                {
                    "ordinal": len(records),
                    "direction": direction,
                    "role": transport.role,
                    "world_rank": transport.world_rank,
                    "peer_world_rank": rank,
                    "encoded_message": json.loads(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
            write_json(path / "staged-controls.json", {"messages": list(records)})

    def observed_send(transport, rank, message):
        result = send(transport, rank, message)
        record("send_returned", transport, rank, message)
        return result

    def observed_receive(transport, rank):
        message = receive(transport, rank)
        record("receive_returned", transport, rank, message)
        return message

    class DiagnosticCapture(logging.Handler):
        def emit(self, log_record):
            # LogRecord unwraps a single Mapping argument into args itself.
            if isinstance(log_record.args, Mapping):
                fields = dict(log_record.args)
                if fields.get("boundary") in ("host_control", "host_enqueue"):
                    with lock:
                        host_records.append(fields)
                        write_json(
                            path / "host-diagnostics.json",
                            {"records": list(host_records)},
                        )

    handler = DiagnosticCapture()
    logger = logging.getLogger("afd_plugin.connectors.npu.hccl_diagnostics")
    logger.addHandler(handler)
    try:
        HcclStagedTransport._send_control = observed_send
        HcclStagedTransport._recv_control = observed_receive
        with observe_controls(report):
            yield
    finally:
        HcclStagedTransport._send_control = send
        HcclStagedTransport._recv_control = receive
        logger.removeHandler(handler)
        handler.close()


def tensor_components(value):
    import torch

    if isinstance(value, torch.Tensor):
        return [value]
    assert isinstance(value, (list, tuple))
    return [tensor for child in value for tensor in tensor_components(child)]


def initialize_attention_kv(wrapper, report: dict) -> None:
    """Follow EngineCore's actual planner/capacity/initializer sequence."""
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_configs,
        update_kv_cache_capacity,
    )
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs

    worker = wrapper.worker
    config, runner = worker.vllm_config, worker.model_runner
    register_all_kvcache_specs(config)
    specs = worker.get_kv_cache_spec()
    assert specs
    supported = worker.get_supported_kv_cache_layouts()
    layout = resolve_kv_cache_layout(config, [supported], specs.values())
    worker.set_kv_cache_layout(layout.name)
    configs = get_kv_cache_configs(config, [specs], [ATTENTION_KV_BUDGET_BYTES])
    assert len(configs) == 1
    native_config = configs[0]
    native_config.kv_cache_layout = config.cache_config.kv_cache_layout
    scheduler_config = generate_scheduler_kv_cache_config(configs)
    config.cache_config.num_gpu_blocks = scheduler_config.num_blocks
    groups = scheduler_config.kv_cache_groups
    participating = [
        g.kv_cache_spec.block_size for g in groups if g.kv_cache_spec.prefix_cacheable
    ]
    config.cache_config.block_size = min(
        participating or [g.kv_cache_spec.block_size for g in groups]
    )
    update_kv_cache_capacity(config, scheduler_config)
    config.validate_block_size()
    assert (
        0
        < sum(item.size for item in native_config.kv_cache_tensors)
        <= ATTENTION_KV_BUDGET_BYTES
    )
    detail = report["attention_kv"] = {
        "budget_bytes": ATTENTION_KV_BUDGET_BYTES,
        "budget_scope": "native KV backing only, excluding model/graphs/scratch",
        "planner": source_record(get_kv_cache_configs),
        "resolved_layout": layout.name,
        "specs": {
            name: {
                "class": source_record(type(spec)),
                "fields": json.loads(json.dumps(asdict(spec), default=str)),
            }
            for name, spec in specs.items()
        },
        "num_blocks": native_config.num_blocks,
        "descriptors": [asdict(item) for item in native_config.kv_cache_tensors],
        "groups": [
            {
                "layer_names": g.layer_names,
                "spec_class": source_record(type(g.kv_cache_spec)),
                "block_size": g.kv_cache_spec.block_size,
            }
            for g in groups
        ],
    }
    wrapper.initialize_from_config(configs)
    allocated = tensor_components(runner.kv_caches)
    bindings = {}
    for name in specs:
        carrier = config.compilation_config.static_forward_context[name]
        assert len(carrier.kv_cache) == 1
        components = tensor_components(carrier.kv_cache[0])
        assert components
        entries = []
        for tensor in components:
            assert tensor.device == worker.device
            assert any(tensor is item for item in allocated)
            storage = tensor.untyped_storage()
            end = tensor.storage_offset() + sum(
                (size - 1) * stride
                for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
            )
            assert 0 <= tensor.storage_offset() <= end
            assert (end + 1) * tensor.element_size() <= storage.nbytes()
            entries.append(
                {
                    "shape": list(tensor.shape),
                    "stride": list(tensor.stride()),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "data_ptr": tensor.data_ptr(),
                    "storage_offset": tensor.storage_offset(),
                    "storage_ptr": storage.data_ptr(),
                    "storage_bytes": storage.nbytes(),
                }
            )
        bindings[name] = {
            "carrier_class": source_record(type(carrier)),
            "components": entries,
        }
    detail["bindings"] = bindings
    assert runner.req_states.req_id_to_index == {}
    assert runner.speculator is None
    report["attention_kv_initialized"] = True


def graph_inventory(runner) -> list[dict]:
    manager = runner.cudagraph_manager
    if manager is None:
        return []
    return [
        {
            "descriptor": json.loads(json.dumps(asdict(desc), default=str)),
            "graph_identity": id(graph),
        }
        for desc, graph in manager.graphs.items()
    ]


def run_lifecycle(wrapper, report: dict) -> None:
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
    kv_config = generate_scheduler_kv_cache_config([runner.kv_cache_config])
    block_size, hash_size = resolve_kv_cache_block_sizes(kv_config, config)
    scheduler = Scheduler(
        config,
        kv_config,
        StructuredOutputManager(config),
        block_size=block_size,
        hash_block_size=hash_size,
    )
    detail = report["request_lifecycle"] = {
        "scheduler_class": source_record(Scheduler),
        "steps": [],
        "retirements": [],
        "prompts": {name: list(ids) for name, ids in PROMPTS.items()},
        "output_limits": dict(OUTPUT_LIMITS),
        "outputs": {name: [] for name in PROMPTS},
        "slots": {},
        "same_owner_identity": id(runner),
        "physical_block_reuse_required": False,
    }
    requests = {}
    progress = Path(report["output_dir"]) / "lifecycle-progress.json"

    def add_request(name):
        params = SamplingParams(
            temperature=0,
            max_tokens=OUTPUT_LIMITS[name],
            ignore_eos=True,
            detokenize=False,
        )
        params.verify(
            config.model_config,
            config.speculative_config,
            config.structured_outputs_config,
            None,
        )
        request = Request(name, list(PROMPTS[name]), params, None)
        assert not request.mm_features and request.kv_transfer_params is None
        requests[name] = request
        scheduler.add_request(request)

    def execute_step():
        ordinal = len(detail["steps"])
        assert ordinal < MAX_SCENE_STEPS
        frontiers = {
            name: request.num_computed_tokens
            for name, request in scheduler.requests.items()
        }
        step = scheduler.schedule()
        assert step.total_num_scheduled_tokens > 0
        controls_begin = len(report.get("staged_control_messages", []))
        host_begin = len(report.get("host_diagnostic_records", []))
        result = wrapper.execute_model(step)
        report["inference_executed"] = True
        assert result is None
        state = runner.execute_model_state
        assert state is not None
        batch = state.input_batch
        order = list(batch.req_ids)
        assert set(order) == set(step.num_scheduled_tokens)
        assert 0 < len(order) <= MAX_REQUESTS
        valid_tokens = step.total_num_scheduled_tokens
        ids = batch.input_ids[:valid_tokens].cpu().tolist()
        positions = batch.positions[:valid_tokens].cpu().tolist()
        expected_ids, expected_positions = [], []
        for name in order:
            count, start = step.num_scheduled_tokens[name], frontiers[name]
            history = list(PROMPTS[name]) + detail["outputs"][name]
            expected_ids.extend(history[start : start + count])
            expected_positions.extend(range(start, start + count))
        assert ids == expected_ids and positions == expected_positions
        hidden = tensor_record(
            state.hidden_states[:valid_tokens], report, f"step-{ordinal}-hidden-valid"
        )
        assert hidden["finite"] is True
        grammar = scheduler.get_grammar_bitmask(step)
        assert grammar is None
        sampled = wrapper.sample_tokens(grammar)
        returned = (
            sampled.get_output()
            if isinstance(sampled, AsyncModelRunnerOutput)
            else sampled
        )
        assert isinstance(returned, ModelRunnerOutput)
        assert returned.req_ids == order
        assert len(returned.sampled_token_ids) == len(order)
        assert runner.execute_model_state is None
        worker_states = {}
        for name, tokens in zip(order, returned.sampled_token_ids, strict=True):
            assert len(tokens) == 1
            assert type(tokens[0]) is int and 0 <= tokens[0] < VOCAB_SIZE
            detail["outputs"][name].extend(tokens)
            index = runner.req_states.req_id_to_index[name]
            detail["slots"][name] = index
            total = int(runner.req_states.total_len.gpu[index].cpu())
            history = runner.req_states.all_token_ids.gpu[index, :total].cpu().tolist()
            assert history == list(PROMPTS[name]) + detail["outputs"][name]
            worker_states[name] = {
                "request_index": index,
                "history": history,
                "total_len": total,
            }
        engine_outputs = scheduler.update_from_output(step, returned)
        for name in order:
            assert list(requests[name].output_token_ids) == detail["outputs"][name]
        detail["steps"].append(
            {
                "ordinal": ordinal,
                "schedule": schedule_record(step),
                "request_order": order,
                "input_ids": ids,
                "positions": positions,
                "frontiers_before_schedule": frontiers,
                "sampled_token_ids": returned.sampled_token_ids,
                "worker_states": worker_states,
                "request_status": {name: requests[name].status.name for name in order},
                "computed_after_update": {
                    name: requests[name].num_computed_tokens for name in order
                },
                "hidden": hidden,
                "parent_hidden_shape": list(state.hidden_states.shape),
                "engine_output_client_ids": sorted(engine_outputs),
                "staged_controls_begin": controls_begin,
                "staged_controls_end": len(report.get("staged_control_messages", [])),
                "host_diagnostics_begin": host_begin,
                "host_diagnostics_end": len(report.get("host_diagnostic_records", [])),
            }
        )
        write_json(progress, detail)
        return order

    def retire(name):
        step = scheduler.schedule()
        assert step.total_num_scheduled_tokens == 0
        assert step.finished_req_ids == {name}
        output = wrapper.execute_model(step)
        assert isinstance(output, ModelRunnerOutput)
        scheduler.update_from_output(step, output)
        assert not scheduler.has_requests()
        assert runner.req_states.req_id_to_index == {}
        detail["retirements"].append(schedule_record(step))
        write_json(progress, detail)

    try:
        add_request("functional-A")
        add_request("functional-B")
        assert set(execute_step()) == {"functional-A", "functional-B"}
        # Keep both requests for the first real decode: Graph/U2 may replay its
        # captured two-request key. Phase acceptance is checked from real logs.
        assert set(execute_step()) == {"functional-A", "functional-B"}
        scheduler.finish_requests("functional-A", RequestStatus.FINISHED_ABORTED)
        assert requests["functional-A"].status is RequestStatus.FINISHED_ABORTED
        detail["cancellation_after_sample_round"] = 2
        while (
            requests["functional-B"].status is not RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            assert execute_step() == ["functional-B"]
        assert requests["functional-B"].status is RequestStatus.FINISHED_LENGTH_CAPPED
        assert len(detail["outputs"]["functional-B"]) == OUTPUT_LIMITS["functional-B"]
        retire("functional-B")
        retired_slots = {
            detail["slots"]["functional-A"],
            detail["slots"]["functional-B"],
        }
        assert len(retired_slots) == MAX_REQUESTS
        add_request("functional-C")
        while (
            requests["functional-C"].status is not RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            assert execute_step() == ["functional-C"]
        assert requests["functional-C"].status is RequestStatus.FINISHED_LENGTH_CAPPED
        assert len(detail["outputs"]["functional-C"]) == OUTPUT_LIMITS["functional-C"]
        retire("functional-C")
        detail.update(
            {
                "scheduler_empty": True,
                "worker_requests_retired": True,
                "retired_slots": sorted(retired_slots),
                "replacement_slot_matches_retired": detail["slots"]["functional-C"]
                in retired_slots,
                "replacement_slot_previous_request_ids": [
                    name
                    for name in ("functional-A", "functional-B")
                    if detail["slots"][name] == detail["slots"]["functional-C"]
                ],
            }
        )
        assert id(runner) == detail["same_owner_identity"]
        assert len(detail["outputs"]["functional-A"]) == 2
        assert len(detail["steps"]) == 6
        report["request_lifecycle_accepted"] = True
    finally:
        scheduler.shutdown()
        write_json(progress, detail)


def record_execution_evidence(args: argparse.Namespace, report: dict) -> None:
    if args.role == "native":
        report["mode_execution_accepted"] = report["request_lifecycle_accepted"]
        return
    host = report["host_diagnostic_records"]
    controls = report["staged_control_messages"]
    decisions = [
        item["encoded_message"]["graph_decision"]
        for item in controls
        if item["encoded_message"]["kind"] == "GRAPH_DECISION"
    ]
    plans = [
        item["encoded_message"]["plan"]
        for item in controls
        if item["encoded_message"]["kind"] == "FORWARD"
    ]
    capture = [
        item
        for item in host
        if item["boundary"] == "host_control"
        and item["phase"] == "capture"
        and item["event"] == "graph_parent_finish_returned"
    ]
    replay = [
        item
        for item in host
        if item["boundary"] == "host_control"
        and item["phase"] == "replay"
        and item["event"] == "graph_parent_replay_returned"
    ]
    body = [item for item in host if item["boundary"] == "host_enqueue"]
    report["execution_evidence"] = {
        "actual_graph_decisions": decisions,
        "actual_staged_plans": plans,
        "capture_returned": capture,
        "replay_returned": replay,
        "actual_body_modes": sorted({item["actual.mode"] for item in body}),
        "actual_body_stage_counts": sorted({item["num_stages"] for item in body}),
        "graph_u2_replay_observed": any(item["num_stages"] == 2 for item in replay),
        "physical_overlap_proven": False,
    }
    graph, u2 = args.mode.startswith("graph"), args.mode.endswith("u2")
    if graph:
        expected_stages = 2 if u2 else 1
        assert any(item["num_stages"] == expected_stages for item in capture)
        assert any(item["num_stages"] == expected_stages for item in replay)
        if args.overlap:
            assert any(item["actual.mode"] == "graph-streamed" for item in body)
    elif u2:
        assert any(item["num_stages"] == 2 for item in body)
        expected = "eager-streamed" if args.overlap else "serial"
        assert any(
            item["actual.mode"] == expected and item["num_stages"] == 2 for item in body
        )
    else:
        assert report["pair_control_messages"]
    if args.role == "ffn":
        # Matched control completion and normal public poll establish service
        # execution; this does not claim full-model numerical equivalence.
        report["inference_executed"] = True
    report["mode_execution_accepted"] = True


def run_worker(args: argparse.Namespace, config, report: dict) -> None:
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel

    wrapper = WorkerWrapperBase()
    constructed = False
    observation = observe_runtime(report) if args.role != "native" else nullcontext()
    with observation:
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
            runner = worker.model_runner
            report["normal_init_device_returned"] = True
            report["actual_device"] = str(worker.device)
            report["runner_class"] = source_record(type(runner))
            report["worker_configuration"] = configuration_record(worker.vllm_config)
            assert worker.device == torch.device("npu", args.device)
            with set_current_vllm_config(worker.vllm_config), torch.inference_mode():
                report["native_a5_bootstrap"] = native_a5_bootstrap()
                wrapper.load_model()
                report["model_loaded"] = True
                if args.role == "ffn":
                    specs = worker.get_kv_cache_spec()
                    assert specs == {}
                    configs = get_kv_cache_configs(worker.vllm_config, [specs], [0])
                    assert len(configs) == 1 and not configs[0].kv_cache_tensors
                    assert not configs[0].kv_cache_groups
                    wrapper.initialize_from_config(configs)
                    report["ffn_empty_kv_initialized"] = True
                report["loaded_model"] = (
                    record_native_model(worker)
                    if args.role == "native"
                    else record_loaded_model(worker, args.role)
                )
                write_json(
                    args.output_dir / "loaded-model.json", report["loaded_model"]
                )
                if args.role != "ffn":
                    initialize_attention_kv(wrapper, report)
                times = wrapper.compile_or_warm_up_model()
                report["compile_or_warm_up_returned"] = True
                report["compilation_times"] = dict(times._asdict())
                if args.role != "ffn":
                    report["captured_graphs_before_requests"] = graph_inventory(runner)
                    if args.mode.startswith("graph"):
                        assert report["captured_graphs_before_requests"]
                    run_lifecycle(wrapper, report)
                    report["captured_graphs_after_requests"] = graph_inventory(runner)
                    assert (
                        report["captured_graphs_after_requests"]
                        == report["captured_graphs_before_requests"]
                    )
                else:
                    deadline = time.monotonic() + args.timeout
                    while worker.poll_hccl_ffn_service():
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "FFN normal service did not receive peer STOP"
                            )
                        time.sleep(POLL_SECONDS)
                    report["ffn_public_poll_observed_peer_stop_and_owner_exit"] = True
                after = {
                    name: parameter_record(p)
                    for name, p in runner.get_model().named_parameters()
                }
                assert after == report["loaded_model"]["parameters"]
                report["loaded_parameters_after_execution"] = after
                report["loaded_parameters_unchanged"] = True
                record_execution_evidence(args, report)
        except BaseException:
            report["worker_failure_traceback"] = traceback.format_exc()
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
                raise RuntimeError("normal Worker cleanup failed; see cleanup_failures")
    report["functional_worker_component_accepted"] = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--run-config", action="store_true")
    execution.add_argument("--run", action="store_true")
    parser.add_argument(
        "--role", choices=("native", "attention", "ffn"), default="attention"
    )
    parser.add_argument(
        "--mode",
        choices=("eager-u1", "eager-u2", "graph-u1", "graph-u2"),
        default="eager-u1",
    )
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--afd-port", type=int, default=DEFAULT_AFD_PORT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    if args.role == "native" and (args.mode != "eager-u1" or args.overlap):
        parser.error("native baseline supports eager-u1 without --overlap only")
    if args.overlap and not args.mode.endswith("u2"):
        parser.error("--overlap requires eager-u2 or graph-u2")
    if args.device < 0 or not 0 < args.afd_port < 65536 or args.timeout <= 0:
        parser.error("device >=0, valid port and positive timeout required")
    if (args.run or args.run_config) and (
        args.model is None or args.output_dir is None
    ):
        parser.error("execution requires --model and fresh --output-dir")
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
        if args.output_dir.exists():
            parser.error("--output-dir must not exist")
    environment = execution_environment(args)
    report = {
        "schema_version": 1,
        "role": args.role,
        "mode": args.mode,
        "run": args.run,
        "run_config": args.run_config,
        "overlap_requested": args.overlap,
        "environment_declaration": environment,
        "runtime_imported": False,
        "model_loaded": False,
        "attention_kv_initialized": False,
        "ffn_empty_kv_initialized": False,
        "inference_executed": False,
        "compile_or_warm_up_returned": False,
        "mode_execution_accepted": False,
        "request_lifecycle_accepted": False,
        "functional_worker_component_accepted": False,
        "cross_role_weight_equivalence_verified": False,
        "numerical_equivalence_accepted": False,
        "physical_overlap_accepted": False,
        "performance_accepted": False,
        "full_checkpoint_accepted": False,
        "pd_accepted": False,
        "dspark_accepted": False,
        "engram_accepted": False,
        "canonical_parameter_groups": canonical_parameter_groups(),
        "scope": "bounded BF16 dummy C2 normal Worker functional component",
        "success": False,
    }
    if args.run or args.run_config:
        args.output_dir.mkdir(parents=True)
        report["output_dir"] = str(args.output_dir)
        os.environ.update(environment)
        try:
            report["runtime_imported"] = True
            config = build_config(args, report)
            if args.run:
                run_worker(args, config, report)
            report["success"] = True
        except BaseException:
            report["traceback"] = traceback.format_exc()
        finally:
            report["afd_modules_imported"] = sorted(
                name
                for name in sys.modules
                if name == "afd_plugin" or name.startswith("afd_plugin.")
            )
            if args.role == "native" and report["afd_modules_imported"]:
                report["success"] = False
                report["native_isolation_error"] = "native baseline imported afd_plugin"
            write_json(args.output_dir / "result.json", report)
    else:
        report["success"] = True
        report["scope"] = "stdlib metadata only; no runtime, Worker or device executed"
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
