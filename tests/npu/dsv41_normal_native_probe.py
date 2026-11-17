# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native three-layer BF16 C2 dummy control, without an AFD configuration.

Default/help use stdlib only. --run creates a normal native Worker and uses the
unchanged native loader, KV planner, Scheduler and sampler. Actual tokens and
hidden states are recorded; this probe alone does not establish AFD parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from dsv41_normal_c2_probe import (
    C2_COMPRESS_RATIO,
    C2_SOURCE_LAYER,
    CANDIDATE_SOURCE_LAYER,
    DEFAULT_TIMEOUT_SECONDS,
    LAYER_COUNT,
    bounded_c2_hf_config,
    initialize_attention_kv,
)
from dsv41_normal_execute_probe import (
    GENERATED_TOKENS,
    PROMPT_TOKEN_IDS,
    observe_smla_metadata,
    run_request,
)
from dsv41_normal_model_probe import (
    parameter_record,
    runtime_record,
    source_record,
    write_json,
)
from dsv41_normal_worker_probe import (
    BLOCK_SIZE,
    HC_MULTIPLIER,
    HEAD_SIZE,
    HIDDEN_SIZE,
    MAX_BATCHED_TOKENS,
    MAX_MODEL_LEN,
    NATIVE_PLUGINS,
    QUERY_HEADS,
    VOCAB_SIZE,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

EXPECTED_ATTENTION_PARAMETERS = 58
EXPECTED_FFN_PARAMETERS = 21
ATTENTION_LAYER_PARAMETERS = (
    "hc_attn_fn",
    "hc_ffn_fn",
    "hc_attn_base",
    "hc_ffn_base",
    "hc_attn_scale",
    "hc_ffn_scale",
    "self_attn.attn_sink",
    "self_attn.wq_a.weight",
    "self_attn.q_norm.weight",
    "self_attn.wq_b.weight",
    "self_attn.wkv.weight",
    "self_attn.kv_norm.weight",
    "self_attn.wo_a.weight",
    "self_attn.wo_b.weight",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)
C2_PARAMETERS = (
    "compressor.wkv.weight",
    "compressor.norm.weight",
    "compressor.wgate.weight",
    "indexer.wq_b.weight",
    "indexer.weights_proj.weight",
    "indexer.wk.weight",
    "indexer.k_norm.weight",
)
FFN_LAYER_PARAMETERS = (
    "gate.weight",
    "gate.e_score_correction_bias",
    "gate.weight_fp32",
    "shared_experts.gate_up_proj.weight",
    "shared_experts.down_proj.weight",
    "experts.routed_experts.w13_weight",
    "experts.routed_experts.w2_weight",
)


def canonical_parameter_groups() -> dict:
    attention = [
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
    ]
    for index in range(LAYER_COUNT):
        prefix = f"language_model.model.layers.{index}."
        attention.extend(prefix + name for name in ATTENTION_LAYER_PARAMETERS)
    attention.extend(
        f"language_model.model.layers.{C2_SOURCE_LAYER}.self_attn." + name
        for name in C2_PARAMETERS
    )
    ffn = {
        f"model.layers.{index}.mlp.{name}": (
            f"language_model.model.layers.{index}.mlp.{name}"
        )
        for index in range(LAYER_COUNT)
        for name in FFN_LAYER_PARAMETERS
    }
    assert len(attention) == len(set(attention)) == EXPECTED_ATTENTION_PARAMETERS
    assert len(ffn) == len(set(ffn.values())) == EXPECTED_FFN_PARAMETERS
    assert not set(attention).intersection(ffn.values())
    return {"attention_identity_names": attention, "ffn_to_native_names": ffn}


def build_native_config(args: argparse.Namespace, report: dict) -> VllmConfig:
    # Native host/device leaf: do not install AFD shims, qualification or roles.
    import torch
    import torch_npu  # noqa: F401
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins

    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert os.environ["VLLM_PLUGINS"] == NATIVE_PLUGINS
    assert current_platform.device_type == "npu"
    report["npu_initialized_before_config"] = torch.npu.is_initialized()
    current_platform.pre_register_and_update()
    load_general_plugins()
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
        additional_config={"enable_cpu_binding": False},
    ).create_engine_config()
    text = config.model_config.hf_text_config
    geometry = {
        "hidden_size": config.model_config.get_hidden_size(),
        "head_size": config.model_config.get_head_size(),
        "vocab_size": config.model_config.get_vocab_size(),
        "num_layers": config.model_config.get_num_layers(config.parallel_config),
        "num_attention_heads": text.num_attention_heads,
        "hc_mult": text.hc_mult,
        "compress_ratios": list(text.compress_ratios),
    }
    assert geometry == {
        "hidden_size": HIDDEN_SIZE,
        "head_size": HEAD_SIZE,
        "vocab_size": VOCAB_SIZE,
        "num_layers": LAYER_COUNT,
        "num_attention_heads": QUERY_HEADS,
        "hc_mult": HC_MULTIPLIER,
        "compress_ratios": [0, 0, C2_COMPRESS_RATIO],
    }
    assert text.kv_source_layer_ids == text.index_source_layer_ids == [C2_SOURCE_LAYER]
    assert text.candidate_source_layer_id == CANDIDATE_SOURCE_LAYER
    assert config.model_config.is_multimodal_model is True
    assert config.model_config.multimodal_config.language_model_only is True
    assert config.model_config.skip_tokenizer_init is True
    assert config.model_config.quantization is config.quant_config is None
    assert config.engram_config is config.speculative_config is None
    assert config.kv_transfer_config is None
    assert config.parallel_config.use_ubatching is False
    assert config.parallel_config.num_ubatches in (0, 1)
    assert config.model_config.enforce_eager is True
    assert config.compilation_config.mode is CompilationMode.NONE
    assert config.compilation_config.cudagraph_mode is CUDAGraphMode.NONE
    assert "afd" not in config.additional_config
    assert original_path.read_bytes() == original
    report["configuration"] = {
        "native_geometry": geometry,
        "worker_cls": config.parallel_config.worker_cls,
        "architectures": list(config.model_config.architectures),
        "additional_config": config.additional_config,
        "model_is_multimodal": config.model_config.is_multimodal_model,
        "language_model_only": True,
        "skip_tokenizer_init": True,
        "load_format": config.load_config.load_format,
        "quantization": config.model_config.quantization,
        "engram_config_is_none": True,
        "speculative_config_is_none": True,
        "kv_transfer_config_is_none": True,
        "enforce_eager": True,
        "compile_mode": config.compilation_config.mode.name,
        "graph_mode": config.compilation_config.cudagraph_mode.name,
        "enable_chunked_prefill": config.scheduler_config.enable_chunked_prefill,
        "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        "max_num_seqs": config.scheduler_config.max_num_seqs,
        "max_model_len": config.model_config.max_model_len,
        "use_ubatching": config.parallel_config.use_ubatching,
        "num_ubatches": config.parallel_config.num_ubatches,
    }
    report["npu_initialized_after_config"] = torch.npu.is_initialized()
    assert report["npu_initialized_after_config"] is False
    return config


def record_native_model(worker) -> dict:
    from vllm_ascend.models.deepseek_v41 import model as native
    from vllm_ascend.models.deepseek_v41.vl_model import AscendDeepseekV41ForCausalLM

    runner = worker.model_runner
    model = runner.get_model()
    assert type(model) is AscendDeepseekV41ForCausalLM
    assert type(model.language_model) is native.AscendDeepseekV41LLMForCausalLM
    inner = model.language_model.model
    assert type(inner) is native.DeepseekV41Model
    assert inner.has_engram is False
    assert len(inner.layers) == LAYER_COUNT
    assert model.training is False
    parameters = {name: parameter_record(p) for name, p in model.named_parameters()}
    groups = canonical_parameter_groups()
    expected = set(groups["attention_identity_names"]) | set(
        groups["ffn_to_native_names"].values()
    )
    assert set(parameters) == expected
    assert len(parameters) == EXPECTED_ATTENTION_PARAMETERS + EXPECTED_FFN_PARAMETERS
    assert all(item["device"] == str(worker.device) for item in parameters.values())
    assert all(item["finite"] is not False for item in parameters.values())
    layers = []
    for index, layer in enumerate(inner.layers):
        assert type(layer) is native.DeepseekV41DecoderLayer
        assert type(layer.self_attn) is native.DeepseekV41Attention
        assert type(layer.mlp) is native.DeepseekV41MoE
        layers.append(
            {
                "index": index,
                "decoder_class": source_record(type(layer)),
                "attention_class": source_record(type(layer.self_attn)),
                "moe_class": source_record(type(layer.mlp)),
                "gate_class": source_record(type(layer.mlp.gate)),
                "experts_class": source_record(type(layer.mlp.experts)),
            }
        )
    return {
        "root_class": source_record(type(model)),
        "language_model_class": source_record(type(model.language_model)),
        "inner_class": source_record(type(inner)),
        "model_state_class": source_record(type(runner.model_state)),
        "normal_worker_load_method": source_record(type(worker).load_model),
        "normal_runner_load_method": source_record(type(runner).load_model),
        "model_load_weights_method_definition": source_record(type(model).load_weights),
        "dummy_loader_does_not_call_model_load_weights": True,
        "training": model.training,
        "parameters": parameters,
        "parameter_count": len(parameters),
        "parameter_bytes": sum(item["bytes"] for item in parameters.values()),
        "canonical_parameter_groups": groups,
        "layers": layers,
        "parameter_readback_is_observation": True,
        "cross_role_weight_equivalence_verified": False,
    }


def run_native(args: argparse.Namespace, config: VllmConfig, report: dict) -> None:
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner
    from vllm_ascend.worker.worker import NPUWorker

    wrapper = WorkerWrapperBase()
    constructed = False
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
        assert type(worker) is NPUWorker
        report["worker_class"] = source_record(type(worker))
        wrapper.init_device()
        assert type(worker.model_runner) is NPUModelRunner
        report["runner_class"] = source_record(type(worker.model_runner))
        report["normal_native_init_device_completed"] = True
        report["actual_device"] = str(worker.device)
        assert worker.device == torch.device("npu", args.device)
        assert "afd" not in worker.vllm_config.additional_config
        with set_current_vllm_config(worker.vllm_config), torch.inference_mode():
            wrapper.load_model()
            report["normal_native_load_model_returned"] = True
            report["model_loaded"] = True
            report["loaded_model"] = record_native_model(worker)
            write_json(args.output_dir / "loaded-model.json", report["loaded_model"])
            initialize_attention_kv(wrapper, report)
            report["attention_kv"]["recorded_subphase"] = "before_first_request"
            runner = worker.model_runner
            report["native_zeroer_class"] = (
                source_record(type(runner.kv_block_zeroer))
                if runner.kv_block_zeroer is not None
                else None
            )
            with observe_smla_metadata(report):
                run_request(wrapper, report)
            after = {
                name: parameter_record(parameter)
                for name, parameter in runner.get_model().named_parameters()
            }
            assert after == report["loaded_model"]["parameters"]
            report["loaded_parameters_after_execution"] = after
            report["loaded_parameters_unchanged"] = True
    except BaseException:
        report["native_failure_traceback"] = traceback.format_exc()
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
                failures.append({"phase": name, "traceback": traceback.format_exc()})
        report["cleanup_failures"] = failures
        report["native_groups_destroyed"] = not torch.distributed.is_initialized()
        if failures and sys.exc_info()[0] is None:
            raise RuntimeError("native control cleanup failed; see cleanup_failures")
    report["native_control_component_accepted"] = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    if args.device < 0 or args.timeout <= 0:
        parser.error("device >=0 and positive timeout required")
    if args.run and (args.model is None or args.output_dir is None):
        parser.error("--run requires --model and fresh --output-dir")
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
        if args.output_dir.exists():
            parser.error("--output-dir must not exist")
    report = {
        "role": "native",
        "run": args.run,
        "runtime_imported": False,
        "afd_configuration_created": False,
        "afd_runtime_bootstrap_called": False,
        "native_plugins": NATIVE_PLUGINS,
        "compress_ratios": [0, 0, C2_COMPRESS_RATIO],
        "c2_source_layer": C2_SOURCE_LAYER,
        "candidate_source_layer": CANDIDATE_SOURCE_LAYER,
        "expected_native_parameters": (
            EXPECTED_ATTENTION_PARAMETERS + EXPECTED_FFN_PARAMETERS
        ),
        "canonical_parameter_groups": canonical_parameter_groups(),
        "prompt": list(PROMPT_TOKEN_IDS),
        "max_tokens": GENERATED_TOKENS,
        "model_loaded": False,
        "attention_kv_initialized": False,
        "inference_executed": False,
        "native_scheduler_prefill_decode_retirement_accepted": False,
        "native_control_component_accepted": False,
        "cross_role_weight_equivalence_verified": False,
        "numerical_equivalence_accepted": False,
        "full_checkpoint_accepted": False,
        "success": False,
    }
    if args.run:
        args.output_dir.mkdir(parents=True)
        report["output_dir"] = str(args.output_dir)
        os.environ.setdefault("VLLM_PLUGINS", NATIVE_PLUGINS)
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
        try:
            report["runtime_imported"] = True
            config = build_native_config(args, report)
            run_native(args, config, report)
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
        report["scope"] = "stdlib metadata; native control not executed"
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
