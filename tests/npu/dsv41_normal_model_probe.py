# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Normal dummy A/F model loading with optional native Attention KV setup.

Default/help only emit stdlib metadata. The paired driver starts both explicit
--run roles concurrently: A's normal load waits for F's normal empty-cache
initialization to perform HELLO. Only A's normal shutdown sends STOP.
--initialize-attention-kv adds bounded native planning/allocation on A only;
neither mode schedules requests, warms up the model, or executes inference.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from dsv41_normal_worker_probe import (
    DEFAULT_AFD_PORT,
    HC_MULTIPLIER,
    HIDDEN_SIZE,
    LAYER_COUNT,
    NATIVE_PLUGINS,
    QUALIFICATION_ENV,
    VOCAB_SIZE,
    build_config,
    config_record,
)

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig

DEFAULT_TIMEOUT_SECONDS = 240
ATTENTION_KV_BUDGET_BYTES = 8 * 1024 * 1024
POLL_SECONDS = 0.1
NATIVE_RUNTIME_MODULES = (
    "vllm",
    "torch_npu",
    "vllm_ascend",
    "vllm_ascend.vllm_ascend_C",
    "vllm.model_executor.model_loader.base_loader",
    "vllm.model_executor.model_loader.dummy_loader",
    "vllm.model_executor.model_loader.weight_utils",
    "vllm.v1.worker.worker_base",
    "vllm.v1.worker.utils",
    "vllm.v1.core.kv_cache_utils",
    "vllm.v1.core.single_type_kv_cache_manager",
    "vllm.v1.attention.backends.utils",
    "vllm_ascend.worker.worker",
    "vllm_ascend.worker.v2.model_runner",
    "vllm_ascend.models.deepseek_v41.model",
)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", "utf-8")
    temporary.replace(path)


def source_record(value) -> dict:
    path = Path(inspect.getfile(value)).resolve(strict=True)
    return {
        "qualified_name": value.__module__ + "." + value.__qualname__,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@contextmanager
def observe_controls(report: dict):
    # Worker isolation: import and wrap only inside explicit runtime execution.
    from afd_plugin.connectors.npu.hccl_protocol import encode_hccl_control_message
    from afd_plugin.connectors.npu.hccl_transport import HcclEagerTransport

    records = report["pair_control_messages"] = []
    original_send = HcclEagerTransport._send_control
    original_receive = HcclEagerTransport._recv_control

    def record(direction, transport, message):
        encoded = encode_hccl_control_message(message)
        records.append(
            {
                "ordinal": len(records),
                "direction": direction,
                "role": transport.role.value,
                "peer_pair_rank": transport.peer_rank,
                "encoded_message": json.loads(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
        write_json(
            Path(report["output_dir"]) / "controls.json",
            {"pair_control_messages": list(records)},
        )

    def send(transport, message):
        result = original_send(transport, message)
        record("send_returned", transport, message)
        return result

    def receive(transport):
        message = original_receive(transport)
        record("receive_returned", transport, message)
        return message

    try:
        HcclEagerTransport._send_control = send
        HcclEagerTransport._recv_control = receive
        yield
    finally:
        HcclEagerTransport._send_control = original_send
        HcclEagerTransport._recv_control = original_receive


def parameter_record(tensor: torch.Tensor) -> dict:
    import torch

    # Observation only: blocking readback records post-loader bytes, including
    # scalars. It is not a timed load, kernel check or cross-role value oracle.
    cpu = tensor.detach().cpu().contiguous()
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "storage_offset": tensor.storage_offset(),
        "data_ptr": tensor.data_ptr(),
        "numel": tensor.numel(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "finite": bool(torch.isfinite(cpu).all()) if cpu.is_floating_point() else None,
    }


def record_loaded_model(worker, role: str) -> dict:
    import torch
    from vllm_ascend.models.deepseek_v41 import model as native

    from afd_plugin.model_executor.models.npu.deepseek_v41 import (
        AFDDeepseekV41Attention,
        AFDDeepseekV41AttentionForCausalLM,
        AFDDeepseekV41FFNForCausalLM,
        AFDDeepseekV41RemoteMoE,
    )

    runner = worker.model_runner
    model = runner.get_model()
    assert model.training is False
    parameters = {name: parameter_record(p) for name, p in model.named_parameters()}
    assert parameters
    assert all(item["device"] == str(worker.device) for item in parameters.values())
    assert all(item["finite"] is not False for item in parameters.values())
    record = {
        "root_class": source_record(type(model)),
        "normal_worker_load_method": source_record(type(worker).load_model),
        "normal_runner_load_method": source_record(type(runner).load_model),
        "model_load_weights_method": source_record(type(model).load_weights),
        "training": model.training,
        "parameters": parameters,
        "parameter_count": len(parameters),
        "parameter_bytes": sum(item["bytes"] for item in parameters.values()),
        "layers": [],
        "cross_role_weight_equivalence_verified": False,
        "parameter_readback_is_observation": True,
    }
    if role == "attention":
        assert type(model) is AFDDeepseekV41AttentionForCausalLM
        inner = model.language_model.model
        assert inner.has_engram is False
        assert len(inner.layers) == LAYER_COUNT
        record["model_state_class"] = source_record(type(runner.model_state))
        record["embedding"] = parameter_record(inner.embed_tokens.weight)
        record["lm_head"] = parameter_record(model.language_model.lm_head.weight)
        assert record["embedding"]["shape"] == [VOCAB_SIZE, HIDDEN_SIZE]
        assert record["lm_head"]["shape"] == [VOCAB_SIZE, HIDDEN_SIZE]
        for index, layer in enumerate(inner.layers):
            assert type(layer.self_attn) is AFDDeepseekV41Attention
            assert isinstance(layer.self_attn, native.DeepseekV41Attention)
            assert type(layer.mlp) is AFDDeepseekV41RemoteMoE
            assert not tuple(layer.mlp.parameters())
            hc = {
                name: parameter_record(parameter)
                for name, parameter in layer.named_parameters(recurse=False)
                if name.startswith("hc_")
            }
            assert set(hc) == {
                "hc_attn_fn",
                "hc_ffn_fn",
                "hc_attn_base",
                "hc_ffn_base",
                "hc_attn_scale",
                "hc_ffn_scale",
            }
            mix_size = (2 + HC_MULTIPLIER) * HC_MULTIPLIER
            for name, item in hc.items():
                assert item["dtype"] == str(torch.float32)
                expected = (
                    [mix_size, HC_MULTIPLIER * HIDDEN_SIZE]
                    if name.endswith("_fn")
                    else [3]
                    if name.endswith("_scale")
                    else [mix_size]
                )
                assert item["shape"] == expected
            record["layers"].append(
                {
                    "index": index,
                    "decoder_class": source_record(type(layer)),
                    "attention_class": source_record(type(layer.self_attn)),
                    "attention_native_constructor": source_record(
                        native.DeepseekV41Attention.__init__
                    ),
                    "moe_class": source_record(type(layer.mlp)),
                    "hc_parameters": hc,
                }
            )
    else:
        assert type(model) is AFDDeepseekV41FFNForCausalLM
        assert model.model.has_engram is False
        assert len(model.model.layers) == LAYER_COUNT
        assert model.get_experts_layer_indices() == tuple(range(LAYER_COUNT))
        for index, layer in enumerate(model.model.layers):
            assert type(layer.mlp) is native.DeepseekV41MoE
            record["layers"].append(
                {
                    "index": index,
                    "moe_class": source_record(type(layer.mlp)),
                    "gate_class": source_record(type(layer.mlp.gate)),
                    "experts_class": source_record(type(layer.mlp.experts)),
                    "shared_experts_class": source_record(
                        type(layer.mlp.shared_experts)
                    ),
                }
            )
    return record


def runtime_record() -> dict:
    modules = {}
    for name in NATIVE_RUNTIME_MODULES:
        if name in sys.modules:
            module = sys.modules[name]
            path = Path(module.__file__).resolve(strict=True)
            modules[name] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    libraries = {}
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text().splitlines():
            path = line.split()[-1]
            if ".so" in path and any(
                token in path
                for token in ("/vllm_ascend/", "/cann-9.2.0/", "/torch_npu/")
            ):
                resolved = Path(path).resolve(strict=True)
                libraries[str(resolved)] = hashlib.sha256(
                    resolved.read_bytes()
                ).hexdigest()
    return {"modules": modules, "mapped_native_libraries": libraries}


def initialize_attention_kv(wrapper, report: dict) -> None:
    # Explicit device execution only. Follow EngineCore's native spec/layout/
    # planner/Worker sequence, using a declared component budget instead of
    # profiling model execution to discover all free device memory.
    import torch
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_configs,
        update_kv_cache_capacity,
    )
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs

    worker = wrapper.worker
    config = worker.vllm_config
    runner = worker.model_runner
    original_owners = {
        "worker": id(worker),
        "runner": id(runner),
        "model": id(runner.get_model()),
        "model_state": id(runner.model_state),
        "request_state": id(runner.req_states),
        "input_buffers": id(runner.input_buffers),
        "sampler": id(runner.sampler),
    }
    register_all_kvcache_specs(config)
    specs = worker.get_kv_cache_spec()
    assert len(specs) == LAYER_COUNT
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
    assert groups
    participating = [
        group.kv_cache_spec.block_size
        for group in groups
        if group.kv_cache_spec.prefix_cacheable
    ]
    config.cache_config.block_size = min(
        participating or [group.kv_cache_spec.block_size for group in groups]
    )
    update_kv_cache_capacity(config, scheduler_config)
    config.validate_block_size()
    assert native_config.num_blocks > 1
    assert native_config.kv_cache_tensors
    assert all(
        0 < tensor.size <= ATTENTION_KV_BUDGET_BYTES
        for tensor in native_config.kv_cache_tensors
    )
    detail = {
        "budget_bytes": ATTENTION_KV_BUDGET_BYTES,
        "budget_scope": "KV backing storage only; excludes model, RoPE and scratch",
        "owner_identity_before": original_owners,
        "budget_is_explicit_component_limit": True,
        "memory_profile_executed": False,
        "model_warmup_executed": False,
        "scheduler_created": False,
        "fresh_block_zeroer_executed": False,
        "supported_layouts": supported,
        "resolved_layout": layout.name,
        "specs": {
            name: {
                "class": source_record(type(spec)),
                "fields": json.loads(json.dumps(asdict(spec), default=str)),
                "page_size_bytes": spec.page_size_bytes,
            }
            for name, spec in specs.items()
        },
        "planner": source_record(get_kv_cache_configs),
        "layout_resolver": source_record(resolve_kv_cache_layout),
        "normal_worker_initializer": source_record(type(worker).initialize_from_config),
        "normal_runner_initializer": source_record(
            type(worker.model_runner).initialize_kv_cache
        ),
        "logical_plan": {
            "num_blocks": native_config.num_blocks,
            "tensors": [asdict(tensor) for tensor in native_config.kv_cache_tensors],
            "groups": [
                {
                    "layer_names": group.layer_names,
                    "spec_class": source_record(type(group.kv_cache_spec)),
                    "block_size": group.kv_cache_spec.block_size,
                    "page_size_bytes": group.kv_cache_spec.page_size_bytes,
                }
                for group in native_config.kv_cache_groups
            ],
        },
    }
    report["attention_kv"] = detail
    # The public Worker path constructs actual backends, BlockTables, caches,
    # source RoPE and (when required) the native zeroing metadata itself.
    wrapper.initialize_from_config(configs)
    runner = worker.model_runner
    assert runner.kv_cache_config.num_blocks == native_config.num_blocks
    assert runner.kv_caches
    assert runner.block_tables is not None
    assert runner.speculator is None
    assert runner.execute_model_state is None
    assert runner.req_states.req_id_to_index == {}
    storages = {}
    bindings = {}
    for name in specs:
        layer = config.compilation_config.static_forward_context[name]
        assert len(layer.kv_cache) == 1
        cache = layer.kv_cache[0]
        assert isinstance(cache, torch.Tensor)
        assert cache.device == worker.device
        assert cache.dtype == torch.bfloat16
        assert any(cache is allocated for allocated in runner.kv_caches)
        storage = cache.untyped_storage()
        storage_pointer = storage.data_ptr()
        assert storage_pointer > 0 and storage.nbytes() > 0
        assert storage.nbytes() <= ATTENTION_KV_BUDGET_BYTES
        end = cache.storage_offset() + sum(
            (size - 1) * stride
            for size, stride in zip(cache.shape, cache.stride(), strict=True)
        )
        assert (end + 1) * cache.element_size() <= storage.nbytes()
        bindings[name] = {
            "layer_class": source_record(type(layer)),
            "shape": list(cache.shape),
            "stride": list(cache.stride()),
            "dtype": str(cache.dtype),
            "device": str(cache.device),
            "storage_offset": cache.storage_offset(),
            "data_ptr": cache.data_ptr(),
            "storage_pointer": storage_pointer,
            "storage_bytes": storage.nbytes(),
        }
        if str(storage_pointer) not in storages:
            raw_view = torch.empty(0, dtype=torch.uint8, device=worker.device)
            raw_view.set_(storage, 0, (storage.nbytes(),), (1,))
            # Full backing readback is an observation after native allocation.
            # Do not require initial zero/finite contents or clear any bytes.
            raw = raw_view.cpu().numpy().tobytes()
            raw_file = f"kv-storage-{len(storages)}.bin"
            (Path(report["output_dir"]) / raw_file).write_bytes(raw)
            storages[str(storage_pointer)] = {
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "raw_file": raw_file,
            }
    actual_bytes = sum(item["bytes"] for item in storages.values())
    assert 0 < actual_bytes <= ATTENTION_KV_BUDGET_BYTES
    assert {item["storage_pointer"] for item in bindings.values()} == {
        int(pointer) for pointer in storages
    }
    owners_after = {
        "worker": id(worker),
        "runner": id(runner),
        "model": id(runner.get_model()),
        "model_state": id(runner.model_state),
        "request_state": id(runner.req_states),
        "input_buffers": id(runner.input_buffers),
        "sampler": id(runner.sampler),
    }
    assert owners_after == original_owners
    parameters_after = {
        name: parameter_record(parameter)
        for name, parameter in runner.get_model().named_parameters()
    }
    assert parameters_after == report["loaded_model"]["parameters"]
    detail.update(
        {
            "physical_plan": {
                "num_blocks": runner.kv_cache_config.num_blocks,
                "tensors": [
                    asdict(tensor) for tensor in runner.kv_cache_config.kv_cache_tensors
                ],
            },
            "bindings": bindings,
            "underlying_storages": storages,
            "actual_storage_bytes": actual_bytes,
            "owner_identity_after": owners_after,
            "whole_storage_readback_is_observation": True,
            "initial_kv_contents_verified": False,
            "loaded_parameters_after": parameters_after,
            "loaded_parameters_unchanged": True,
            "block_tables_class": source_record(type(runner.block_tables)),
            "model_state_class": source_record(type(runner.model_state)),
            "attention_backends": [
                {
                    "backend": source_record(group.backend),
                    "layer_names": group.layer_names,
                    "builders": [
                        source_record(type(builder))
                        for builder in group.metadata_builders
                    ],
                }
                for groups_for_cache in runner.attn_groups
                for group in groups_for_cache
            ],
            "zeroing_metadata_initialized": runner.kv_block_zeroer is not None,
            "native_initialize_from_config_returned": True,
        }
    )
    if native_config.needs_kv_cache_zeroing:
        assert detail["zeroing_metadata_initialized"]
    report["attention_kv_initialized"] = True


def run_model(args: argparse.Namespace, config: VllmConfig, report: dict) -> None:
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel

    wrapper = WorkerWrapperBase()
    constructed = False
    with observe_controls(report):
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
    report["native_attention_kv_component_accepted"] = bool(
        args.initialize_attention_kv and report["attention_kv_initialized"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--initialize-attention-kv", action="store_true")
    parser.add_argument("--role", choices=("attention", "ffn"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--afd-port", type=int, default=DEFAULT_AFD_PORT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    if args.device < 0 or not 0 < args.afd_port < 65536 or args.timeout <= 0:
        parser.error("device >= 0, port in 1..65535 and positive timeout required")
    if args.run and (args.model is None or args.output_dir is None):
        parser.error("--run requires --model and a fresh --output-dir")
    if args.initialize_attention_kv and args.role != "attention":
        parser.error("--initialize-attention-kv requires --role attention")
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
        if args.output_dir.exists():
            parser.error("--output-dir must not exist")
    report = {
        "role": args.role,
        "run": args.run,
        "runtime_imported": False,
        "model_loaded": False,
        "attention_kv_initialized": False,
        "initialize_attention_kv_requested": args.initialize_attention_kv,
        "native_attention_kv_component_accepted": False,
        "ffn_empty_cache_initialized": False,
        "inference_executed": False,
        "numerical_equivalence_accepted": False,
        "full_checkpoint_accepted": False,
        "native_model_load_component_accepted": False,
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
        report["scope"] = "metadata only; use the paired driver for actual A/F loading"
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
