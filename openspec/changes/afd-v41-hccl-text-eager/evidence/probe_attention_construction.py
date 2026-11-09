# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Construct reduced native Ascend Attention roles; never load checkpoint weights.

Run one cell per process. The decoder cell invokes the real A-role constructor,
including native Attention/norm/HC/Engram projections. The wrapper cell also
uses the real native DummyModelLoader and EngramModelState constructor. Engram
tables are explicitly reduced before construction; no forward kernels run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import tempfile
import time
import traceback
from copy import copy, deepcopy
from importlib.metadata import version
from pathlib import Path

CHECKPOINT = "/mnt/workspace/models/DeepSeek-V4.1-Flash"
REDUCED_TEXT_VALUES = (
    ("hidden_size", 128),
    ("moe_intermediate_size", 128),
    ("num_hidden_layers", 2),
    ("num_attention_heads", 8),
    ("q_lora_rank", 128),
    ("o_lora_rank", 128),
    ("o_groups", 1),
    ("vocab_size", 128),
    ("n_routed_experts", 2),
    ("num_experts_per_tok", 1),
    ("compress_ratios", (0, 2)),
    ("kv_source_layer_ids", (1,)),
    ("index_source_layer_ids", (1,)),
    ("candidate_source_layer_id", 1),
    ("candidate_topk_blocks", 8),
    ("candidate_block_size", 8),
    ("index_topk", 8),
    ("index_n_heads", 8),
    ("engram_layer_ids", (1,)),
    ("engram_num_embeddings", (1024,)),
    ("engram_max_ngram_size", 3),
    ("engram_vocab_size", 31),
    ("engram_n_heads", 2),
    ("engram_head_dim", 32),
    ("num_nextn_predict_layers", 0),
)
MAX_PARAMETER_BYTES = 256 * 1024 * 1024


def qualified_type(value) -> str:
    return type(value).__module__ + "." + type(value).__qualname__


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def config_deltas(before: dict, after: dict, prefix: str = "") -> list[dict]:
    differences = []
    for key in sorted(before.keys() | after.keys()):
        name = f"{prefix}.{key}" if prefix else key
        old, new = before.get(key), after.get(key)
        if isinstance(old, dict) and isinstance(new, dict):
            differences.extend(config_deltas(old, new, name))
        elif old != new:
            differences.append({"path": name, "before": old, "after": new})
    return differences


def tensor_manifest(module) -> dict:
    def describe(tensor):
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "bytes": tensor.numel() * tensor.element_size(),
        }

    return {
        "parameters": {
            name: describe(tensor) for name, tensor in module.named_parameters()
        },
        "buffers": {name: describe(tensor) for name, tensor in module.named_buffers()},
        "module_types": {
            name: qualified_type(child) for name, child in module.named_modules()
        },
    }


def quantization_description(runtime) -> dict:
    quant = runtime.quant_config
    return {
        "model_config_quantization": runtime.model_config.quantization,
        "quant_config_type": qualified_type(quant) if quant is not None else None,
        "quant_config_name": quant.get_name() if quant is not None else None,
        "hf_quantization_config": runtime.model_config.hf_config.quantization_config,
    }


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    report = {
        "success": False,
        "cell": args.cell,
        "device": args.device,
        "checkpoint_weights_loaded": False,
        "checkpoint_files_modified": False,
        "inference_executed": False,
        "model_constructor_bypassed": False,
        "production_guard_modified": False,
        "connector_initialized": False,
        "topology": "single process TP1/PP1/DP1, one A3 card, no A/F transport",
        "quantization_for_construction": None,
    }
    distributed_started = False
    try:
        # Worker isolation: initialize Ascend before importing its model classes.
        import torch
        import torch_npu  # noqa: F401
        import vllm_ascend.ops  # noqa: F401
        from vllm.config import (
            CompilationMode,
            CUDAGraphMode,
            ModelConfig,
            VllmConfig,
            set_current_vllm_config,
        )
        from vllm.config.device import DeviceConfig
        from vllm.config.engram import EngramConfig
        from vllm.config.load import LoadConfig
        from vllm.config.scheduler import SchedulerConfig
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
        from vllm.model_executor.model_loader.utils import get_model_architecture
        from vllm.model_executor.models.interfaces import get_mixture_of_experts_model
        from vllm.platforms import current_platform
        from vllm.plugins import load_general_plugins

        current_platform.pre_register_and_update()
        load_general_plugins()

        from vllm_ascend.models.deepseek_v41 import model as native
        from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
        from vllm_ascend.utils import adapt_patch, register_ascend_customop

        from afd_plugin.compat.patches.npu.ascend_config import init_ascend_config
        from afd_plugin.compat.patches.npu.model_loader import (
            apply_afd_model_loader_patch,
        )
        from afd_plugin.model_executor.models.npu.deepseek_v41 import (
            AFDDeepseekV41AttentionDecoderLayer,
            AFDDeepseekV41AttentionForCausalLM,
            AFDDeepseekV41RemoteMoE,
        )

        torch.npu.set_device(args.device)
        torch.set_default_dtype(torch.bfloat16)
        raw_config_path = args.checkpoint / "config.json"
        raw_config = raw_config_path.read_bytes()
        report.update(
            versions={
                name: version(name)
                for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "fastapi")
            },
            checkpoint_config={
                "path": str(raw_config_path),
                "sha256": hashlib.sha256(raw_config).hexdigest(),
            },
            device_name=torch.npu.get_device_name(args.device),
            source={
                "native_decoder_file": native.__file__,
                "native_decoder_file_sha256": hashlib.sha256(
                    Path(native.__file__).read_bytes()
                ).hexdigest(),
                "attention_decoder_source_sha256": hashlib.sha256(
                    inspect.getsource(AFDDeepseekV41AttentionDecoderLayer).encode()
                ).hexdigest(),
            },
        )
        quant_cells = []
        for explicit_quantization in (None, "deepseek_v4_fp8"):
            quant_cell = {"requested_quantization": explicit_quantization}
            try:
                model_config = ModelConfig(
                    model=str(args.checkpoint),
                    trust_remote_code=True,
                    dtype="bfloat16",
                    enforce_eager=True,
                    max_model_len=128,
                    skip_tokenizer_init=True,
                    language_model_only=True,
                    quantization=explicit_quantization,
                )
                actual_runtime = VllmConfig(
                    model_config=model_config,
                    device_config=DeviceConfig(device="cpu"),
                    load_config=LoadConfig(load_format="dummy"),
                )
                quant_cell.update(quantization_description(actual_runtime))
                quant_cell["success"] = True
                if explicit_quantization is None:
                    original_runtime = actual_runtime
            except Exception as error:
                quant_cell.update(
                    success=False, error_type=qualified_type(error), error=str(error)
                )
            quant_cells.append(quant_cell)
        report["original_checkpoint_quantization_config_cells"] = quant_cells
        if not quant_cells[0]["success"]:
            raise RuntimeError(
                "native checkpoint configuration could not be constructed"
            )
        runtime = copy(original_runtime)
        runtime.model_config = deepcopy(original_runtime.model_config)
        identity_before = {
            "hf_architectures": list(runtime.model_config.hf_config.architectures),
            "architectures": list(runtime.model_config.architectures),
            "architecture": runtime.model_config.architecture,
        }
        hf_before = deepcopy(runtime.model_config.hf_config.to_dict())
        for config in (
            runtime.model_config.hf_config,
            runtime.model_config.hf_text_config,
        ):
            for name, value in REDUCED_TEXT_VALUES:
                setattr(
                    config, name, list(value) if isinstance(value, tuple) else value
                )
        runtime.device_config = DeviceConfig(device="npu")
        runtime.cache_config.block_size = 128
        runtime.quant_config = None
        runtime.engram_config = EngramConfig(cpu_offload=False)
        runtime.scheduler_config = SchedulerConfig(
            max_model_len=128,
            max_num_batched_tokens=128,
            max_num_seqs=2,
            is_encoder_decoder=runtime.model_config.is_encoder_decoder,
        )
        runtime.compilation_config.mode = CompilationMode.NONE
        runtime.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        assert not runtime.parallel_config.use_sequence_parallel_moe
        runtime.additional_config = {
            "afd": {
                "role": "attention",
                "connector": "P2pHcclAFDConnector",
                "compute_gate_on_attention": False,
            }
        }
        report["configuration_deltas_before_constructor"] = config_deltas(
            hf_before, runtime.model_config.hf_config.to_dict()
        )
        report["runtime_overrides"] = {
            "device_config": "npu",
            "cache_block_size": 128,
            "quant_config": None,
            "load_format": "dummy",
            "language_model_only": True,
            "engram_cpu_offload": False,
            "max_model_len": 128,
            "max_num_batched_tokens": 128,
            "max_num_seqs": 2,
            "compilation_mode": "NONE",
            "cudagraph_mode": "NONE",
            "use_sequence_parallel_moe": False,
            "additional_config": runtime.additional_config,
        }
        report["preserved_compressed_vocab_size"] = (
            runtime.model_config.hf_text_config.engram_compressed_vocab_size
        )
        assert runtime.model_config.hf_text_config.engram_num_embeddings == [1024]
        assert runtime.model_config.hf_text_config.num_hidden_layers == 2
        assert runtime.use_v2_model_runner
        with (
            tempfile.TemporaryDirectory(prefix="attention-hccl-") as temporary,
            set_current_vllm_config(runtime),
        ):
            adapt_patch()
            register_ascend_customop(runtime)
            init_ascend_config(runtime)
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=args.device,
                distributed_init_method=Path(temporary, "store").as_uri(),
                backend="hccl",
            )
            distributed_started = True
            initialize_model_parallel(backend="hccl")
            registry_before = get_model_architecture(runtime.model_config)
            report["npu_bytes_before"] = torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            if args.cell == "decoder":
                with torch.device(f"npu:{args.device}"):
                    model = torch.nn.ModuleList(
                        AFDDeepseekV41AttentionDecoderLayer(
                            runtime, f"model.layers.{layer_idx}"
                        )
                        for layer_idx in range(2)
                    )
                layers = model
            else:
                apply_afd_model_loader_patch()
                model = DummyModelLoader(runtime.load_config).load_model(
                    vllm_config=runtime, model_config=runtime.model_config
                )
                assert isinstance(model, AFDDeepseekV41AttentionForCausalLM)
                assert not model.training
                layers = model.language_model.model.layers
                state_type = model.get_model_state_cls()
                assert state_type is EngramModelState
                state = state_type(
                    runtime, model, None, torch.device(f"npu:{args.device}")
                )
                report["model_state"] = {
                    "type": qualified_type(state),
                    "requires_raw_input_tokens": model.requires_raw_input_tokens,
                    "lookback_depth": model.token_lookback_depth,
                    "lookback_shape": list(state.lookback_token_ids.shape),
                    "lookback_dtype": str(state.lookback_token_ids.dtype),
                    "generation_tasks": list(state.get_supported_generation_tasks()),
                }
                assert model.requires_raw_input_tokens
                assert model.token_lookback_depth == 2
                assert state.lookback_token_ids.shape == (2, 2)
                assert get_mixture_of_experts_model(model) is None
                assert all(
                    layer.self_attn.shared_state
                    is model.language_model.model.shared_attention_state
                    for layer in layers
                )
                report["native_shared_attention_state_bound"] = True
                report["a_local_moe_metadata"] = {
                    name: getattr(model.language_model, name)
                    for name in (
                        "num_moe_layers",
                        "num_logical_experts",
                        "num_physical_experts",
                        "num_local_physical_experts",
                    )
                }
                assert all(
                    value == 0 for value in report["a_local_moe_metadata"].values()
                )
            manifest = tensor_manifest(model)
            save(args.output_dir / "parameter-manifest.json", manifest)
            names = tuple(manifest["parameters"])
            assert names and not any(".mlp." in name for name in names)
            assert not any(
                isinstance(child, native.DeepseekV41MoE) for child in model.modules()
            )
            assert all(
                isinstance(layer.mlp, AFDDeepseekV41RemoteMoE) for layer in layers
            )
            assert all(
                isinstance(layer.self_attn, native.DeepseekV41Attention)
                for layer in layers
            )
            assert all(
                isinstance(layer.input_layernorm, native.RMSNorm) for layer in layers
            )
            assert all(layer.hc_attn_fn.dtype == torch.float32 for layer in layers)
            assert layers[0].engram is None
            assert isinstance(layers[1].engram, native.AscendEngram)
            parameter_bytes = sum(
                value["bytes"] for value in manifest["parameters"].values()
            )
            assert parameter_bytes < MAX_PARAMETER_BYTES
            assert get_model_architecture(runtime.model_config) == registry_before
            assert identity_before == {
                "hf_architectures": list(runtime.model_config.hf_config.architectures),
                "architectures": list(runtime.model_config.architectures),
                "architecture": runtime.model_config.architecture,
            }
            torch.npu.synchronize()
            report.update(
                success=True,
                model_type=qualified_type(model),
                parameter_count=len(names),
                parameter_bytes=parameter_bytes,
                buffer_bytes=sum(
                    value["bytes"] for value in manifest["buffers"].values()
                ),
                no_local_moe_parameters=True,
                native_attention_norm_hc_engram_preserved=True,
                registry_and_hf_identity_preserved=True,
                npu_bytes_after=torch.npu.memory_allocated(),
                npu_peak_bytes=torch.npu.max_memory_allocated(),
                hf_config_deltas_after_constructor=config_deltas(
                    hf_before, runtime.model_config.hf_config.to_dict()
                ),
            )
            del model
            gc.collect()
            destroy_model_parallel()
            destroy_distributed_environment()
            distributed_started = False
        assert (
            hashlib.sha256(raw_config_path.read_bytes()).hexdigest()
            == (report["checkpoint_config"]["sha256"])
        )
    except Exception as error:
        report.update(
            success=False,
            error_type=qualified_type(error),
            error=str(error),
            traceback=traceback.format_exc(),
            distributed_cleanup_pending=distributed_started,
        )
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        save(args.output_dir / "result.json", report)
        print("ATTENTION_CONSTRUCTION_RESULT=" + json.dumps(report), flush=True)
    return 0 if report["success"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path(CHECKPOINT))
    parser.add_argument("--cell", choices=("decoder", "wrapper"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
