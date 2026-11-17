"""Construct real native MoE in reduced CPU/NPU cells; no inference claim."""

import argparse
import json
import tempfile
from copy import copy, deepcopy
from pathlib import Path

import torch
import vllm_ascend.ops  # noqa: F401
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.config.device import DeviceConfig
from vllm.config.load import LoadConfig
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.utils import get_model_architecture
from vllm.platforms import current_platform
from vllm.plugins import load_general_plugins
from vllm_ascend.models.deepseek_v41 import model as native
from vllm_ascend.utils import adapt_patch, register_ascend_customop

from afd_plugin.compat.npu.runtime import apply_afd_ascend_patches_if_needed
from afd_plugin.compat.patches.npu.ascend_config import init_ascend_config
from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    AFDDeepseekV41FFNForCausalLM,
)

CHECKPOINT = "/mnt/workspace/models/DeepSeek-V4.1-Flash"
HIDDEN_SIZE = 128
INTERMEDIATE_SIZE = 128
VOCAB_SIZE = 32
LAYER_COUNT = 2
EXPERT_COUNT = 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    parser.add_argument("--dummy-loader", action="store_true")
    parser.add_argument("--ffn-runner", choices=("sync", "async"))
    args = parser.parse_args()
    if args.dummy_loader and args.device != "npu":
        parser.error("the role loader cell requires --device npu")
    if args.ffn_runner and not args.dummy_loader:
        parser.error("the runner cell requires --dummy-loader --device npu")
    backend = "hccl" if args.device == "npu" else "gloo"
    current_platform.pre_register_and_update()
    load_general_plugins()
    model_config = ModelConfig(
        model=CHECKPOINT,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=128,
        skip_tokenizer_init=True,
    )
    runtime = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(load_format="dummy"),
    )
    # Separate construction cell: reduced dimensions, no checkpoint quantization.
    runtime = copy(runtime)
    runtime.model_config = deepcopy(model_config)
    runtime.device_config = DeviceConfig(device=args.device)
    runtime.quant_config = None
    runtime.engram_config = None
    runtime.additional_config = {
        "afd": {"role": "ffn", "connector": "P2pHcclAFDConnector"}
    }
    for config in (
        runtime.model_config.hf_config,
        runtime.model_config.hf_text_config,
    ):
        for name, value in {
            "hidden_size": HIDDEN_SIZE,
            "moe_intermediate_size": INTERMEDIATE_SIZE,
            "num_hidden_layers": LAYER_COUNT,
            "n_routed_experts": EXPERT_COUNT,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "num_hash_layers": 1,
            "vocab_size": VOCAB_SIZE,
        }.items():
            setattr(config, name, value)
    report = {
        "scope": "real native MoE reduced construction and weight loader",
        "device": args.device,
        "device_config": runtime.device_config.device_type,
        "checkpoint": CHECKPOINT,
        "checkpoint_weights_loaded": False,
        "quantization": None,
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "layer_count": LAYER_COUNT,
        "expert_count": EXPERT_COUNT,
        "backend": backend,
        "world_size": 1,
        "inference_executed": False,
        "native_dummy_loader": args.dummy_loader,
        "ffn_runner_load_entry": args.ffn_runner,
        "runner_constructor_executed": False,
        "connector_initialized": False,
        "use_v2_model_runner": runtime.use_v2_model_runner,
    }
    print("FFN_PROBE_CELL " + json.dumps(report), flush=True)
    with tempfile.TemporaryDirectory(prefix="ffn-gloo-") as temp:
        try:
            with set_current_vllm_config(runtime):
                if args.device == "npu":
                    torch.npu.set_device(0)
                    adapt_patch()
                    register_ascend_customop(runtime)
                init_ascend_config(runtime)
                init_distributed_environment(
                    world_size=1,
                    rank=0,
                    local_rank=0,
                    distributed_init_method=Path(temp, "store").as_uri(),
                    backend=backend,
                )
                initialize_model_parallel(backend=backend)
                class_before, arch_before = get_model_architecture(runtime.model_config)
                identity_before = (
                    list(runtime.model_config.hf_config.architectures),
                    list(runtime.model_config.architectures),
                    runtime.model_config.architecture,
                )
                if args.device == "npu":
                    report["npu_bytes_before"] = torch.npu.memory_allocated()
                    torch.npu.reset_peak_memory_stats()
                if args.dummy_loader:
                    assert (
                        get_afd_model_config(runtime.model_config, device_type="npu")
                        is runtime.model_config
                    )
                    apply_afd_ascend_patches_if_needed()
                    if args.ffn_runner:
                        # Exercise the real loading seam only; connector setup
                        # remains guarded until the HCCL implementation exists.
                        from afd_plugin.v1.worker.npu.async_ffn_model_runner import (
                            AsyncNPUFFNModelRunner,
                        )
                        from afd_plugin.v1.worker.npu.ffn_model_runner import (
                            AFDNPUFFNModelRunner,
                        )

                        runner_class = (
                            AFDNPUFFNModelRunner
                            if args.ffn_runner == "sync"
                            else AsyncNPUFFNModelRunner
                        )
                        runner = runner_class.__new__(runner_class)
                        runner.vllm_config = runtime
                        runner.model_config = runtime.model_config
                        runner.device = torch.device("npu:0")
                        runner.model = None
                        runner.load_model()
                        model = runner.get_model()
                        report["runner_model_memory_usage"] = runner.model_memory_usage
                        try:
                            runner.load_model()
                        except RuntimeError as error:
                            assert "already loaded" in str(error)
                            report["duplicate_load_rejected"] = True
                        else:
                            raise AssertionError("duplicate loading was accepted")
                    else:
                        model = DummyModelLoader(runtime.load_config).load_model(
                            vllm_config=runtime, model_config=runtime.model_config
                        )
                    assert isinstance(model, AFDDeepseekV41FFNForCausalLM)
                    assert not model.training
                else:
                    with torch.device(args.device):
                        model = AFDDeepseekV41FFNForCausalLM(vllm_config=runtime)
                class_after, arch_after = get_model_architecture(runtime.model_config)
                assert class_before is class_after and arch_before == arch_after
                assert identity_before == (
                    list(runtime.model_config.hf_config.architectures),
                    list(runtime.model_config.architectures),
                    runtime.model_config.architecture,
                )
                report["identity_preserved"] = True
                report["registry_class"] = (
                    class_after.__module__ + "." + class_after.__name__
                )
                report["eval_applied"] = not model.training
                parameters = dict(model.named_parameters())
                buffers = dict(model.named_buffers())
                report["parameters"] = {
                    name: {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "device": str(tensor.device),
                        "bytes": tensor.numel() * tensor.element_size(),
                    }
                    for name, tensor in parameters.items()
                }
                report["buffers"] = {
                    name: {"shape": list(tensor.shape), "device": str(tensor.device)}
                    for name, tensor in buffers.items()
                }
                report["module_types"] = {
                    name: type(module).__module__ + "." + type(module).__name__
                    for name, module in model.named_modules()
                }
                assert all(".mlp." in name for name in parameters)
                assert all(".mlp." in name for name in buffers)
                assert all(
                    tensor.device.type == args.device for tensor in parameters.values()
                )
                assert all(
                    isinstance(layer.mlp, native.DeepseekV41MoE)
                    for layer in model.model.layers
                )
                assert not model.model.has_engram
                if args.dummy_loader:
                    for layer in model.model.layers:
                        gate = layer.mlp.gate
                        assert gate.weight_fp32.dtype == torch.float32
                        assert torch.isfinite(gate.weight_fp32).all()
                        torch.testing.assert_close(
                            gate.weight_fp32, gate.weight.float()
                        )
                    report["gate_fp32_postload_matches"] = True
                    report["gate_load_kind"] = "native dummy initialization/postload"
                else:
                    expected = (
                        torch.arange(EXPERT_COUNT * HIDDEN_SIZE)
                        .reshape(EXPERT_COUNT, HIDDEN_SIZE)
                        .float()
                    )
                    loaded = model.load_weights(
                        iter(
                            [
                                ("layers.0.ffn.gate.weight", expected),
                                (
                                    "layers.0.attn.indexer.compressor.wgate.weight",
                                    expected,
                                ),
                                ("layers.0.engram.embed.weight", torch.empty(1)),
                                ("head.weight", torch.empty(1)),
                            ]
                        )
                    )
                    assert loaded == {"model.layers.0.mlp.gate.weight"}
                    torch.testing.assert_close(
                        model.model.layers[0].mlp.gate.weight.float().cpu(), expected
                    )
                    report["gate_load_kind"] = "synthetic gate checkpoint path"
                report["native_gate_loaded"] = True
                report["native_expert_mapping_entries"] = len(
                    model.get_expert_mapping()
                )
                report["parameter_bytes"] = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in parameters.values()
                )
                if args.device == "npu":
                    torch.npu.synchronize()
                    report["npu_bytes_after"] = torch.npu.memory_allocated()
                    report["npu_peak_bytes"] = torch.npu.max_memory_allocated()
                    assert all(
                        type(layer.mlp.experts).__module__.startswith("vllm_ascend.")
                        for layer in model.model.layers
                    )
                print("FFN_CONSTRUCTION_RESULT " + json.dumps(report), flush=True)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
