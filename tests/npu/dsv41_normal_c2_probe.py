# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Normal three-layer C2 A/F loading and native compressed KV initialization.

Default/help only emit stdlib metadata. The paired driver starts both explicit
--run roles concurrently: A's normal load waits for F's normal empty-cache
initialization to perform HELLO. Only A's normal shutdown sends STOP.
--initialize-attention-kv adds bounded native planning/allocation on A only;
neither mode schedules requests, warms up the model, or executes inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from dsv41_normal_model_probe import (
    observe_controls,
    parameter_record,
    runtime_record,
    source_record,
    write_json,
)
from dsv41_normal_worker_probe import (
    BLOCK_SIZE,
    DEFAULT_AFD_PORT,
    GROUP_TIMEOUT_SECONDS,
    HC_MULTIPLIER,
    HEAD_SIZE,
    HIDDEN_SIZE,
    MAX_BATCHED_TOKENS,
    MAX_MODEL_LEN,
    NATIVE_PLUGINS,
    QUALIFICATION_ENV,
    QUERY_HEADS,
    VOCAB_SIZE,
    config_record,
)
from dsv41_normal_worker_probe import (
    bounded_hf_config as swa_hf_config,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

LAYER_COUNT = 3
C2_SOURCE_LAYER = 2
C2_COMPRESS_RATIO = 2
CANDIDATE_SOURCE_LAYER = 20
CANDIDATE_BLOCK_SIZE = 8
CANDIDATE_TOPK_BLOCKS = 8
DEFAULT_TIMEOUT_SECONDS = 240
ATTENTION_KV_BUDGET_BYTES = 8 * 1024 * 1024
POLL_SECONDS = 0.1


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
    hidden_size = worker.vllm_config.model_config.get_hidden_size()
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
        assert record["embedding"]["shape"] == [VOCAB_SIZE, hidden_size]
        assert record["lm_head"]["shape"] == [VOCAB_SIZE, hidden_size]
        for index, layer in enumerate(inner.layers):
            assert type(layer.self_attn) is AFDDeepseekV41Attention
            assert isinstance(layer.self_attn, native.DeepseekV41Attention)
            attention = layer.self_attn
            expected_source = C2_SOURCE_LAYER if index == C2_SOURCE_LAYER else None
            assert attention.role.layer_idx == index
            assert attention.role.compress_ratio == (
                C2_COMPRESS_RATIO if index == C2_SOURCE_LAYER else 0
            )
            assert attention.role.kv_source_layer == expected_source
            assert attention.role.index_source_layer == expected_source
            assert attention.role.is_kv_source is (index == C2_SOURCE_LAYER)
            assert attention.role.is_index_source is (index == C2_SOURCE_LAYER)
            assert attention.role.is_candidate_source is False
            assert attention.role.uses_candidate_filter is False
            source_modules = {}
            if index == C2_SOURCE_LAYER:
                assert type(attention.compressor) is native.DeepseekV41Compressor
                assert type(attention.indexer) is native.DeepseekV41Indexer
                assert attention.long_kv_source_prefix.endswith(
                    ".layers.2.self_attn.long_kv_cache"
                )
                assert attention.index_k_source_prefix.endswith(
                    ".layers.2.self_attn.indexer.k_cache"
                )
                source_modules = {
                    "compressor": source_record(type(attention.compressor)),
                    "indexer": source_record(type(attention.indexer)),
                    "long_kv_source_prefix": attention.long_kv_source_prefix,
                    "index_k_source_prefix": attention.index_k_source_prefix,
                }
            else:
                assert attention.compressor is None and attention.indexer is None
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
                    [mix_size, HC_MULTIPLIER * hidden_size]
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
                    "native_attention_role": asdict(attention.role),
                    "native_source_modules": source_modules,
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


def bounded_c2_hf_config(hf_config):
    """Declare the complete C2 topology before ModelConfig caches getters."""
    bounded = swa_hf_config(hf_config)
    raw = bounded.to_dict()
    raw["model_type"] = bounded.model_type
    topology = {
        "num_hidden_layers": LAYER_COUNT,
        "compress_ratios": [0, 0, C2_COMPRESS_RATIO],
        "kv_source_layer_ids": [C2_SOURCE_LAYER],
        "index_source_layer_ids": [C2_SOURCE_LAYER],
        "candidate_source_layer_id": CANDIDATE_SOURCE_LAYER,
        "candidate_block_size": CANDIDATE_BLOCK_SIZE,
        "candidate_topk_blocks": CANDIDATE_TOPK_BLOCKS,
    }
    raw.update(topology)
    if "text_config" in raw:
        raw["text_config"] = {**raw["text_config"], **topology}
    return type(bounded)(**raw)


def build_config(args: argparse.Namespace, report: dict) -> VllmConfig:
    # Explicit host/device leaves only: importing metadata/help stays stdlib.
    import torch
    import torch_npu  # noqa: F401
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins

    from afd_plugin.compat.npu import apply_afd_ascend_config_patch_if_needed
    from afd_plugin.compat.npu.feature_validation import (
        fail_if_unsupported_npu_afd_features,
    )

    assert os.environ[QUALIFICATION_ENV] == "1"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert current_platform.device_type == "npu"
    report["npu_initialized_before_config"] = torch.npu.is_initialized()
    current_platform.pre_register_and_update()
    load_general_plugins()
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
    report["checkpoint_config_sha256"] = hashlib.sha256(original).hexdigest()
    report["first_declaration"] = {
        "role": args.role,
        "hf_override": "bounded_c2_hf_config",
        "num_hidden_layers": LAYER_COUNT,
        "compress_ratios": [0, 0, C2_COMPRESS_RATIO],
        "kv_source_layer_ids": [C2_SOURCE_LAYER],
        "index_source_layer_ids": [C2_SOURCE_LAYER],
        "candidate_source_layer_id": CANDIDATE_SOURCE_LAYER,
        "candidate_block_size": CANDIDATE_BLOCK_SIZE,
        "candidate_topk_blocks": CANDIDATE_TOPK_BLOCKS,
        "load_format": "dummy",
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "speculation": False,
        "pd": False,
        "engram": False,
        "graph": False,
        "u2": False,
    }
    config = EngineArgs(
        model=str(controlled),
        tokenizer=args.model,
        trust_remote_code=True,
        skip_tokenizer_init=True,
        language_model_only=True,
        dtype="bfloat16",
        hf_overrides=bounded_c2_hf_config,
        load_format="dummy",
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        disable_hybrid_kv_cache_manager=False,
        block_size=BLOCK_SIZE,
        tensor_parallel_size=1,
        enable_expert_parallel=False,
        enable_dbo=False,
        ubatch_size=0,
        async_scheduling=False,
        seed=0,
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE
        ),
        speculative_config=None,
        additional_config={
            "enable_cpu_binding": False,
            "afd": {
                "connector": "P2pHcclAFDConnector",
                "role": args.role,
                "host": "127.0.0.1",
                "num_attention_ranks": 1,
                "num_ffn_ranks": 1,
                "port": args.afd_port,
                "afd_process_group_timeout_s": GROUP_TIMEOUT_SECONDS,
                "connector_extra_config": {"max_tokens": MAX_BATCHED_TOKENS},
            },
        },
    ).create_engine_config()
    fail_if_unsupported_npu_afd_features(config)
    report["configuration"] = config_record(config)
    assert report["configuration"]["native_geometry"] == {
        "hidden_size": HIDDEN_SIZE,
        "head_size": HEAD_SIZE,
        "vocab_size": VOCAB_SIZE,
        "num_layers": LAYER_COUNT,
        "num_attention_heads": QUERY_HEADS,
        "hc_mult": HC_MULTIPLIER,
        "compress_ratios": [0, 0, C2_COMPRESS_RATIO],
    }
    text = config.model_config.hf_text_config
    assert text.kv_source_layer_ids == text.index_source_layer_ids == [C2_SOURCE_LAYER]
    assert text.candidate_source_layer_id == CANDIDATE_SOURCE_LAYER
    assert text.candidate_block_size == CANDIDATE_BLOCK_SIZE
    assert text.candidate_topk_blocks == CANDIDATE_TOPK_BLOCKS
    assert original_path.read_bytes() == original
    report["dummy_qualification_accepted"] = True
    report["npu_initialized_after_config"] = torch.npu.is_initialized()
    assert report["npu_initialized_after_config"] is False
    return config


def initialize_attention_kv(wrapper, report: dict) -> None:
    """Use the unchanged native mixed V4.1 planner, allocator and binder."""
    import torch
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_configs,
        update_kv_cache_capacity,
    )
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
    from vllm_ascend.models.deepseek_v41.cache_config import (
        get_layer_tuples,
        is_deepseek_v41_cache,
    )

    worker = wrapper.worker
    config, runner = worker.vllm_config, worker.model_runner
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
    assert len(specs) == LAYER_COUNT + 3
    assert is_deepseek_v41_cache(specs)
    pages, tuples = get_layer_tuples(specs)
    assert len(pages) == len(tuples) == 1
    long_name, index_name, *aliases = tuples[0]
    state_names = [name for name in aliases if name.endswith(".compressor.state_cache")]
    swa_names = [name for name in aliases if name.endswith(".swa_cache")]
    assert len(state_names) == 1 and len(swa_names) == LAYER_COUNT
    assert f".layers.{C2_SOURCE_LAYER}." in long_name
    assert f".layers.{C2_SOURCE_LAYER}." in index_name
    assert specs[long_name].storage_block_size == BLOCK_SIZE // C2_COMPRESS_RATIO
    assert specs[index_name].head_size == 128
    assert specs[index_name].dtype == torch.int8
    assert specs[index_name].scale_dtype == torch.float16
    state = specs[state_names[0]]
    assert state.block_size == 32 and state.head_size == 2 * HEAD_SIZE
    assert state.dtype == torch.float32
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
    assert len(native_config.kv_cache_tensors) == 1
    descriptor = native_config.kv_cache_tensors[0]
    assert descriptor.layers == list(tuples[0])
    assert descriptor.block_stride == pages[0]
    assert descriptor.layer_stride == 0 and descriptor.offset == 0
    assert descriptor.size == pages[0] * native_config.num_blocks
    assert 0 < descriptor.size <= ATTENTION_KV_BUDGET_BYTES
    detail = report["attention_kv"] = {
        "budget_bytes": ATTENTION_KV_BUDGET_BYTES,
        "budget_scope": "KV backing storage only; excludes model, RoPE and scratch",
        "budget_is_explicit_component_limit": True,
        "owner_identity_before": original_owners,
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
        "logical_plan": {
            "num_blocks": native_config.num_blocks,
            "tensors": [asdict(item) for item in native_config.kv_cache_tensors],
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
        "source_tuple": {
            "long_kv": long_name,
            "index_k": index_name,
            "state": state_names,
            "swa": swa_names,
        },
        "native_tuple_page_bytes": pages,
    }
    wrapper.initialize_from_config(configs)
    assert runner.kv_cache_config.num_blocks == native_config.num_blocks
    assert runner.block_tables is not None and runner.kv_caches
    assert runner.speculator is None and runner.execute_model_state is None
    assert runner.req_states.req_id_to_index == {}
    storages, bindings = {}, {}
    for name, spec in specs.items():
        carrier = config.compilation_config.static_forward_context[name]
        assert len(carrier.kv_cache) == 1
        cache = carrier.kv_cache[0]
        components = cache if isinstance(cache, tuple) else (cache,)
        expected_dtypes = (
            (spec.dtype, spec.scale_dtype)
            if isinstance(spec, AscendMLAAttentionSpec) and spec.scale_dim
            else (spec.dtype,)
        )
        assert len(components) == len(expected_dtypes)
        entries = []
        for component, dtype in zip(components, expected_dtypes, strict=True):
            assert isinstance(component, torch.Tensor)
            assert component.device == worker.device and component.dtype == dtype
            assert any(component is allocated for allocated in runner.kv_caches)
            storage = component.untyped_storage()
            end = component.storage_offset() + sum(
                (size - 1) * stride
                for size, stride in zip(
                    component.shape, component.stride(), strict=True
                )
            )
            assert (end + 1) * component.element_size() <= storage.nbytes()
            assert 0 < storage.nbytes() <= ATTENTION_KV_BUDGET_BYTES
            entries.append(
                {
                    "shape": list(component.shape),
                    "stride": list(component.stride()),
                    "dtype": str(component.dtype),
                    "device": str(component.device),
                    "storage_offset": component.storage_offset(),
                    "data_ptr": component.data_ptr(),
                    "storage_pointer": storage.data_ptr(),
                    "storage_bytes": storage.nbytes(),
                }
            )
            if str(storage.data_ptr()) not in storages:
                raw_view = torch.empty(0, dtype=torch.uint8, device=worker.device)
                raw_view.set_(storage, 0, (storage.nbytes(),), (1,))
                raw = raw_view.cpu().numpy().tobytes()
                filename = f"kv-storage-{len(storages)}.bin"
                (Path(report["output_dir"]) / filename).write_bytes(raw)
                storages[str(storage.data_ptr())] = {
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "raw_file": filename,
                }
        bindings[name] = {
            "layer_class": source_record(type(carrier)),
            "carrier_is_one_element_list": True,
            "component_container": "tuple" if isinstance(cache, tuple) else "Tensor",
            "components": entries,
        }
    assert len(storages) == 1
    base = bindings[long_name]["components"][0]["data_ptr"]
    assert all(bindings[name]["components"][0]["data_ptr"] == base for name in aliases)
    index = bindings[index_name]["components"]
    assert index[0]["data_ptr"] - base == specs[long_name].unpadded_page_size_bytes
    index_data_bytes = (
        specs[index_name].storage_block_size
        * specs[index_name].num_kv_heads
        * specs[index_name].head_size
    )
    assert index[1]["data_ptr"] - index[0]["data_ptr"] == index_data_bytes
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
                    asdict(item) for item in runner.kv_cache_config.kv_cache_tensors
                ],
            },
            "bindings": bindings,
            "underlying_storages": storages,
            "actual_storage_bytes": sum(item["bytes"] for item in storages.values()),
            "owner_identity_after": owners_after,
            "whole_storage_readback_is_observation": True,
            "initial_kv_contents_verified": False,
            "loaded_parameters_after": parameters_after,
            "loaded_parameters_unchanged": True,
            "native_tuple_aliases_verified": True,
            "native_initialize_from_config_returned": True,
            "zeroing_metadata_initialized": runner.kv_block_zeroer is not None,
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
    report["native_c2_kv_component_accepted"] = bool(
        args.initialize_attention_kv and report["attention_kv_initialized"]
    )
    report["native_attention_kv_component_accepted"] = bool(
        args.initialize_attention_kv and report["attention_kv_initialized"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--run-config", action="store_true")
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
    if (args.run or args.run_config) and (
        args.model is None or args.output_dir is None
    ):
        parser.error("--run requires --model and a fresh --output-dir")
    if args.initialize_attention_kv and args.role != "attention":
        parser.error("--initialize-attention-kv requires --role attention")
    if args.initialize_attention_kv and args.run_config:
        parser.error("--initialize-attention-kv cannot run with --run-config")
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
        if args.output_dir.exists():
            parser.error("--output-dir must not exist")
    report = {
        "role": args.role,
        "run": args.run,
        "run_config": args.run_config,
        "c2_source_layer": C2_SOURCE_LAYER,
        "candidate_source_layer": CANDIDATE_SOURCE_LAYER,
        "compress_ratios": [0, 0, C2_COMPRESS_RATIO],
        "pure_swa_patch_installed": False,
        "runtime_imported": False,
        "model_loaded": False,
        "attention_kv_initialized": False,
        "initialize_attention_kv_requested": args.initialize_attention_kv,
        "native_attention_kv_component_accepted": False,
        "ffn_empty_cache_initialized": False,
        "native_c2_kv_component_accepted": False,
        "inference_executed": False,
        "numerical_equivalence_accepted": False,
        "full_checkpoint_accepted": False,
        "native_model_load_component_accepted": False,
        "success": False,
    }
    if args.run or args.run_config:
        args.output_dir.mkdir(parents=True)
        report["output_dir"] = str(args.output_dir)
        os.environ.setdefault("VLLM_PLUGINS", NATIVE_PLUGINS)
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
        os.environ.setdefault(QUALIFICATION_ENV, "1")
        try:
            report["runtime_imported"] = True
            config = build_config(args, report)
            if args.run:
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
        report["scope"] = (
            "metadata only; C2 normal load/KV and numerical inference remain pending"
        )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
