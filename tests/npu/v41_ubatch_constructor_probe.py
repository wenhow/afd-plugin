# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Construct the explicit V2 U2 runner with native NPU streams and ModelState.

This isolated process loads configuration only, creates weightless native model
shells, and calls the actual runner constructors through the scoped factory.
Attention groups are empty. It does not prepare metadata, initialize HCCL, run
model operators, load weights, or validate inference/overlap.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

MAX_REQUESTS = 2
MAX_TOKENS = 16
HIDDEN_SIZE = 128
VOCAB_SIZE = 256
NUM_UBATCHES = 2


@dataclass(frozen=True)
class FactoryOwner:
    kernel_block_sizes: tuple[int, ...] = ()


def create_config(output_dir):
    # Runtime imports are confined to this standalone worker process.
    from vllm.config import (
        CompilationConfig,
        CUDAGraphMode,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm_ascend.ascend_config import init_ascend_config

    config_dir = output_dir / "model-config"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "architectures": ["DeepseekV41ForCausalLM"],
                "text_config": {
                    "hidden_size": HIDDEN_SIZE,
                    "intermediate_size": HIDDEN_SIZE * 2,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "vocab_size": VOCAB_SIZE,
                    "max_position_embeddings": MAX_TOKENS,
                    "qk_rope_head_dim": 64,
                    "qk_nope_head_dim": 64,
                    "v_head_dim": 64,
                    "q_lora_rank": 128,
                    "kv_lora_rank": 64,
                    "index_topk": 4,
                    "compress_ratios": [1],
                    "n_routed_experts": 2,
                    "num_experts_per_tok": 1,
                    "n_shared_experts": 1,
                    "moe_intermediate_size": HIDDEN_SIZE,
                },
            },
            indent=2,
        )
        + "\n"
    )
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(config_dir),
            skip_tokenizer_init=True,
            enforce_eager=True,
            dtype="bfloat16",
            max_model_len=MAX_TOKENS,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=MAX_TOKENS,
            is_encoder_decoder=False,
            max_num_seqs=MAX_REQUESTS,
            max_num_batched_tokens=MAX_TOKENS,
        ),
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
    )
    init_ascend_config(config)
    # Configure this explicit component after native platform initialization;
    # the platform resets CLI ubatch_size and production admission stays closed.
    config.parallel_config.ubatch_size = NUM_UBATCHES
    config.parallel_config.rank = 0
    config.additional_config["afd"] = {
        "role": "attention",
        "connector": "P2pHcclAFDConnector",
        "num_attention_ranks": 1,
        "num_ffn_ranks": 1,
    }
    return config


def run_probe(args):
    import torch
    import vllm_ascend.ops  # noqa: F401
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu import model_runner, ubatch_utils
    from vllm_ascend.models.deepseek_v41 import model as native
    from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
    from vllm_ascend.utils import adapt_patch
    from vllm_ascend.worker.v2.utils import torch_cuda_wrapper

    from afd_plugin.compat.patches.npu.mrv2_ubatching import use_npu_ubatch_factory
    from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
    from afd_plugin.distributed.hccl_plan import HcclPlanCapabilities
    from afd_plugin.v1.worker.npu.hccl_v41_ubatch_runner import HcclV41UBatchRunner

    current_platform.pre_register_and_update()
    load_general_plugins()
    torch.npu.set_device(args.device)
    adapt_patch()
    device = torch.device("npu", args.device)
    config = create_config(args.output_dir)
    inner = native.DeepseekV41Model.__new__(native.DeepseekV41Model)
    torch.nn.Module.__init__(inner)
    inner.engram_hash = None
    model = native.AscendDeepseekV41LLMForCausalLM.__new__(
        native.AscendDeepseekV41LLMForCausalLM
    )
    torch.nn.Module.__init__(model)
    model.model = inner
    state = EngramModelState(config, model, None, device)
    assert state.lookback_token_ids is None and model.token_lookback_depth == 0
    transport = HcclStagedTransport(
        "attention",
        0,
        HcclPlanCapabilities(HIDDEN_SIZE, VOCAB_SIZE, MAX_TOKENS, (0,), 1, 1),
        init_method="tcp://127.0.0.1:1",
        local_device=args.device,
    )
    original_factory = model_runner.maybe_build_ubatch_runner
    original_sm_control = ubatch_utils.create_sm_control_context
    with torch_cuda_wrapper():
        assert torch.cuda.Stream is torch.npu.Stream
        with use_npu_ubatch_factory(FactoryOwner(), hccl_transport=transport):
            runner = model_runner.maybe_build_ubatch_runner(
                config, device, state, [], KVCacheConfig(0, [], []), MAX_REQUESTS
            )
            assert isinstance(runner, HcclV41UBatchRunner)
            assert isinstance(runner.comm_stream, torch.npu.Stream)
            assert isinstance(runner.capture_stream, torch.npu.Stream)
            assert runner.comm_stream != runner.capture_stream
            assert runner.num_ubatches == NUM_UBATCHES
        assert model_runner.maybe_build_ubatch_runner is original_factory
        assert ubatch_utils.create_sm_control_context is original_sm_control
        buffers = []
        for kind, tensors, expected_size in (
            ("query_start_loc", runner.ubatch_query_start_loc, MAX_REQUESTS + 1),
            ("seq_lens", runner.ubatch_seq_lens, MAX_REQUESTS),
        ):
            assert len(tensors) == NUM_UBATCHES
            assert tensors[0].data_ptr() != tensors[1].data_ptr()
            for stage_idx, tensor in enumerate(tensors):
                assert tensor.device == device and tensor.dtype == torch.int32
                assert tuple(tensor.shape) == (expected_size,)
                assert torch.equal(
                    tensor.cpu(), torch.zeros(expected_size, dtype=torch.int32)
                )
                buffers.append(
                    {
                        "kind": kind,
                        "stage": stage_idx,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "device": str(tensor.device),
                    }
                )
        try:
            runner.run(model, {}, ubatch_utils.UBatchState([], []))
        except RuntimeError as exc:
            assert "requires an active HCCL transport" in str(exc)
            admission_error = str(exc)
        else:
            raise AssertionError("native run accepted an uninitialized real transport")
        assert not transport.initialized and not transport.closed
        torch.npu.synchronize()
    transport.close()
    return {
        "runner_class": type(runner).__module__ + "." + type(runner).__qualname__,
        "model_state_class": type(state).__module__ + "." + type(state).__qualname__,
        "native_engram_model_state_constructor_called": True,
        "native_ubatch_runner_constructor_called": True,
        "native_torch_cuda_wrapper_used": True,
        "factory_and_sm_control_restored": True,
        "native_npu_streams": [str(runner.comm_stream), str(runner.capture_stream)],
        "buffers": buffers,
        "uninitialized_transport_run_rejection": admission_error,
        "caller_owned_transport_remained_uninitialized": True,
        "attention_groups": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "scope": "real native V2 U2 constructor and inactive-transport run admission",
        "device": args.device,
        "success": False,
        "native_model_weights_loaded": False,
        "native_model_constructor_called": False,
        "attention_metadata_or_model_operators_executed": False,
        "distributed_groups_created": False,
        "accepted_parent_or_successful_model_run": False,
    }
    try:
        report.update(run_probe(args))
        report["success"] = True
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    report["elapsed_seconds"] = time.monotonic() - started
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({key: report[key] for key in ("scope", "success", "device")}),
        flush=True,
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
