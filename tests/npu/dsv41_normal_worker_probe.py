# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Construct the explicit no-DSpark dummy candidate through the normal worker.

Default/--help is stdlib metadata only. --run-config validates a real first
EngineArgs declaration without initializing NPU. --run-worker additionally uses
the native WorkerWrapperBase factory and actual role init_device/shutdown.
This initial probe does not load a model, allocate KV, or execute inference.
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

if TYPE_CHECKING:
    from vllm.config import VllmConfig

QUALIFICATION_ENV = "AFD_DSV41_DUMMY_QUALIFICATION"
HIDDEN_SIZE = 512
HC_MULTIPLIER = 4
QUERY_HEADS = 64
HEAD_SIZE = 512
ROPE_SIZE = 64
LORA_RANK = 64
VOCAB_SIZE = 4096
LAYER_COUNT = 2
ROUTED_EXPERTS = 8
EXPERTS_PER_TOKEN = 2
MAX_MODEL_LEN = 128
MAX_BATCHED_TOKENS = 16
BLOCK_SIZE = 128
INDEX_HEAD_SIZE = 128
INDEX_HEADS = 64
INDEX_TOPK = 512
DEFAULT_AFD_PORT = 30010
GROUP_TIMEOUT_SECONDS = 45
NATIVE_PLUGINS = (
    "ascend,ascend_model,ascend_model_loader,ascend_kv_connector,"
    "ascend_service_profiling"
)


def bounded_hf_config(hf_config):
    """Declare native SWA geometry before native cached model getters exist."""
    values = {
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": LAYER_COUNT,
        "num_attention_heads": QUERY_HEADS,
        "num_key_value_heads": 1,
        "head_dim": HEAD_SIZE,
        "qk_rope_head_dim": ROPE_SIZE,
        "q_lora_rank": LORA_RANK,
        "kv_lora_rank": HEAD_SIZE,
        "o_lora_rank": LORA_RANK,
        "o_groups": 1,
        "hc_mult": HC_MULTIPLIER,
        "vocab_size": VOCAB_SIZE,
        "intermediate_size": HIDDEN_SIZE,
        "moe_intermediate_size": HIDDEN_SIZE,
        "n_routed_experts": ROUTED_EXPERTS,
        "n_shared_experts": 1,
        "num_experts_per_tok": EXPERTS_PER_TOKEN,
        "n_group": 1,
        "topk_group": 1,
        "num_hash_layers": 0,
        "max_position_embeddings": MAX_MODEL_LEN,
        "sliding_window": BLOCK_SIZE,
        "compress_ratios": [0] * LAYER_COUNT,
        "index_head_dim": INDEX_HEAD_SIZE,
        "index_n_heads": INDEX_HEADS,
        "index_topk": INDEX_TOPK,
        "engram_layer_ids": [],
        "vision_n_layers": 0,
        "quantization_config": None,
        "expert_dtype": "bf16",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    raw = hf_config.to_dict()
    raw["model_type"] = hf_config.model_type
    raw.update(values)
    if "text_config" in raw:
        raw["text_config"] = {**raw["text_config"], **values}
    raw["vision_config"] = {"num_hidden_layers": 0}
    return type(hf_config)(**raw)


def config_record(config: VllmConfig) -> dict:
    text = config.model_config.hf_text_config
    return {
        "worker_cls": config.parallel_config.worker_cls,
        "device_type": config.device_config.device_type,
        "native_geometry": {
            "hidden_size": config.model_config.get_hidden_size(),
            "head_size": config.model_config.get_head_size(),
            "vocab_size": config.model_config.get_vocab_size(),
            "num_layers": config.model_config.get_num_layers(config.parallel_config),
            "num_attention_heads": text.num_attention_heads,
            "hc_mult": text.hc_mult,
            "compress_ratios": list(text.compress_ratios),
        },
        "use_ubatching": config.parallel_config.use_ubatching,
        "num_ubatches": config.parallel_config.num_ubatches,
        "enable_dbo": config.parallel_config.enable_dbo,
        "enforce_eager": config.model_config.enforce_eager,
        "compile_mode": config.compilation_config.mode.name,
        "graph_mode": config.compilation_config.cudagraph_mode.name,
        "load_format": config.load_config.load_format,
        "quantization": config.model_config.quantization,
        "quant_config_is_none": config.quant_config is None,
        "engram_config_is_none": config.engram_config is None,
        "speculative_config_is_none": config.speculative_config is None,
        "kv_transfer_config_is_none": config.kv_transfer_config is None,
        "model_is_multimodal": config.model_config.is_multimodal_model,
        "language_model_only": (
            config.model_config.multimodal_config.language_model_only
        ),
        "max_model_len": config.model_config.max_model_len,
        "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        "max_num_seqs": config.scheduler_config.max_num_seqs,
        "enable_chunked_prefill": config.scheduler_config.enable_chunked_prefill,
        "async_scheduling": config.scheduler_config.async_scheduling,
        "afd": config.additional_config["afd"],
    }


def build_config(args: argparse.Namespace, report: dict) -> VllmConfig:
    # Runtime imports belong to explicit host/device probe leaves.
    import torch
    import torch_npu  # noqa: F401
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins

    assert os.environ[QUALIFICATION_ENV] == "1"
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"
    assert current_platform.device_type == "npu"
    report["npu_initialized_before_config"] = torch.npu.is_initialized()
    current_platform.pre_register_and_update()
    load_general_plugins()
    from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
    from vllm.engine.arg_utils import EngineArgs

    from afd_plugin.compat.npu import apply_afd_ascend_config_patch_if_needed
    from afd_plugin.compat.npu.feature_validation import (
        fail_if_unsupported_npu_afd_features,
    )

    apply_afd_ascend_config_patch_if_needed()
    import afd_plugin.compat.patches.config_validation  # noqa: F401

    original_path = Path(args.model) / "config.json"
    original = original_path.read_bytes()
    source = json.loads(original)
    for container in (source, source.get("text_config", {})):
        container.pop("quantization_config", None)
        container.pop("compression_config", None)
        container["expert_dtype"] = "bf16"
    controlled = args.output_dir / "controlled-model"
    controlled.mkdir()
    (controlled / "config.json").write_text(json.dumps(source) + "\n", "utf-8")
    report["checkpoint_config_sha256"] = hashlib.sha256(original).hexdigest()
    report["first_declaration"] = {
        "role": args.role,
        "hf_override": "bounded_hf_config",
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
        hf_overrides=bounded_hf_config,
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
    report["configuration"] = config_record(config)
    fail_if_unsupported_npu_afd_features(config)
    report["dummy_qualification_accepted"] = True
    assert report["configuration"]["native_geometry"] == {
        "hidden_size": HIDDEN_SIZE,
        "head_size": HEAD_SIZE,
        "vocab_size": VOCAB_SIZE,
        "num_layers": LAYER_COUNT,
        "num_attention_heads": QUERY_HEADS,
        "hc_mult": HC_MULTIPLIER,
        "compress_ratios": [0] * LAYER_COUNT,
    }
    assert original_path.read_bytes() == original
    report["npu_initialized_after_config"] = torch.npu.is_initialized()
    assert report["npu_initialized_after_config"] is False
    return config


def run_worker(args: argparse.Namespace, config: VllmConfig, report: dict) -> None:
    import torch
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from vllm_ascend.distributed.parallel_state import destroy_ascend_model_parallel

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
        report["worker_class"] = type(worker).__module__ + "." + type(worker).__name__
        report["native_wrapper_factory_constructed_worker"] = True
        wrapper.init_device()
        report["normal_role_init_device_completed"] = True
        report["worker_configuration"] = config_record(worker.vllm_config)
        report["runner_class"] = (
            type(worker.model_runner).__module__
            + "."
            + type(worker.model_runner).__name__
        )
        report["actual_device"] = str(worker.device)
        assert worker.device == torch.device("npu", args.device)
        assert worker.vllm_config.engram_config is None
        assert worker.vllm_config.speculative_config is None
        assert worker.vllm_config.kv_transfer_config is None
        assert config.engram_config is config.speculative_config is None
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
                failures.append({"phase": name, "traceback": traceback.format_exc()})
        report["cleanup_failures"] = failures
        report["native_groups_destroyed"] = not torch.distributed.is_initialized()
        if failures and sys.exc_info()[0] is None:
            raise RuntimeError("normal Worker cleanup failed; see cleanup_failures")
    report["native_worker_component_accepted"] = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("attention", "ffn"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--afd-port", type=int, default=DEFAULT_AFD_PORT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--run-config", action="store_true")
    modes.add_argument("--run-worker", action="store_true")
    args = parser.parse_args()
    if args.device < 0 or not 0 < args.afd_port < 65536:
        parser.error("device must be nonnegative and port must be in 1..65535")
    if (args.run_config or args.run_worker) and (
        args.model is None or args.output_dir is None
    ):
        parser.error("runtime probes require --model and a fresh --output-dir")
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
        if args.output_dir.exists():
            parser.error("--output-dir must not exist")
    report = {
        "role": args.role,
        "run_config": args.run_config,
        "run_worker": args.run_worker,
        "runtime_imported": False,
        "model_loaded": False,
        "kv_allocated": False,
        "inference_executed": False,
        "full_checkpoint_accepted": False,
        "u2_graph_pd_accepted": False,
        "success": False,
    }
    if args.run_config or args.run_worker:
        args.output_dir.mkdir(parents=True)
        try:
            report["runtime_imported"] = True
            config = build_config(args, report)
            if args.run_worker:
                run_worker(args, config, report)
            report["success"] = True
        except BaseException:
            report["traceback"] = traceback.format_exc()
        (args.output_dir / "result.json").write_text(
            json.dumps(report, indent=2) + "\n", "utf-8"
        )
    else:
        report["success"] = True
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
