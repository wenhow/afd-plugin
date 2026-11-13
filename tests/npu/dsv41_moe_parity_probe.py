# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compare one real DSV4.1 checkpoint MoE at native TP1 and TP4.

The default inspects checkpoint metadata and installed sources only. --run
explicitly starts two sequential, bounded NPU phases. This is a numerical
diagnostic with controlled inputs, not a model-output equivalence acceptance.
The full checkpoint config and layer-0 parameter namespace remain unchanged;
Attention, Engram, AFD transport, services and native worker construction are
not exercised. No complete-model TP4 allocation is attempted.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import importlib.util
import inspect
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import afd_profiler_probe as profiler_helpers
import hccl_eager_backend_probe as supervision
import hccl_staged_probe as hccl_helpers

LAYER_INDEX = 0
TP_SIZES = (1, 4)
PROMPT_TOKEN_IDS = (65106, 1192, 304, 3144, 260, 4654, 7274, 1116, 2912, 16)
INPUT_SEED = 0
NATIVE_PLUGINS = (
    "ascend,ascend_model,ascend_model_loader,ascend_kv_connector,"
    "ascend_service_profiling"
)
RECORDED_ENVIRONMENT_KEYS = (
    "VLLM_PLUGINS",
    "VLLM_USE_V2_MODEL_RUNNER",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_CUSTOM_OPP_PATH",
    "LD_LIBRARY_PATH",
    "HCCL_OP_EXPANSION_MODE",
    "HCCL_BUFFSIZE",
    "HCCL_CONNECT_TIMEOUT",
    "HCCL_EXEC_TIMEOUT",
    "PYTORCH_NPU_ALLOC_CONF",
    "OMP_NUM_THREADS",
)
TOTAL_TIMEOUT_SECONDS = 600
GROUP_TIMEOUT_SECONDS = 120
POLL_SECONDS = 0.1
MAX_HEADER_BYTES = 128 * 1024 * 1024
MAX_LAYER_PAYLOAD_BYTES = 16 * 1024**3
REQUIRED_ROUTER_ARGUMENTS = frozenset(
    {
        "x",
        "k",
        "bias",
        "input_ids",
        "tid2eid",
        "k_group",
        "group_count",
        "routed_scaling_factor",
        "eps",
        "group_select_mode",
        "renorm",
        "norm_type",
        "out_flag",
        "bias_vl",
        "image_sentinel_lo",
        "image_sentinel_count",
    }
)
SOURCE_PATHS = (
    "vllm/model_executor/custom_op.py",
    "vllm/model_executor/model_loader/utils.py",
    "vllm/model_executor/layers/fused_moe/layer.py",
    "vllm/model_executor/layers/fused_moe/routed_experts.py",
    "vllm/model_executor/layers/fused_moe/runner/moe_runner.py",
    "vllm/model_executor/layers/linear.py",
    "vllm/model_executor/parameter.py",
    "vllm/transformers_utils/configs/deepseek_v41.py",
    "vllm_ascend/models/deepseek_v41/model.py",
    "vllm_ascend/device/device_op.py",
    "vllm_ascend/ascend_forward_context.py",
    "vllm_ascend/ops/linear.py",
    "vllm_ascend/ops/fused_moe/fused_moe.py",
    "vllm_ascend/ops/fused_moe/routed_experts.py",
    "vllm_ascend/ops/fused_moe/prepare_finalize.py",
    "vllm_ascend/ops/fused_moe/shared_experts.py",
    "vllm_ascend/ops/fused_moe/moe_comm_method.py",
    "vllm_ascend/ops/fused_moe/router/fused_topk_router.py",
    "vllm_ascend/quantization/configs/fp8_config.py",
    "vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py",
    "vllm_ascend/patch/platform/patch_fused_moe.py",
    "vllm_ascend/platform.py",
    "vllm_ascend/utils.py",
    "vllm_ascend/worker/worker.py",
    "afd_plugin/__init__.py",
    "afd_plugin/compat/patches/npu/mxfp4.py",
    "afd_plugin/compat/patches/npu/mxfp_scales.py",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help=(
            "with --run, check actual runtime imports/config only; "
            "no NPU groups or kernels"
        ),
    )
    parser.add_argument("--timeout", type=int, default=TOTAL_TIMEOUT_SECONDS)
    parser.add_argument("--tp-size", type=int, choices=TP_SIZES, help=argparse.SUPPRESS)
    parser.add_argument("--rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        args.devices = tuple(map(int, args.devices.split(",")))
    except ValueError:
        parser.error("devices must be comma-separated NPU indices")
    if len(args.devices) != TP_SIZES[-1] or len(set(args.devices)) != len(args.devices):
        parser.error("exactly four distinct NPU indices are required")
    if min(args.devices) < 0:
        parser.error("NPU indices must be nonnegative")
    if args.setup_only and not args.run:
        parser.error("setup-only requires explicit --run")
    cleanup = (
        hccl_helpers.REAP_TIMEOUT_SECONDS + profiler_helpers.NPU_SMI_TIMEOUT_SECONDS
    )
    if not cleanup < args.timeout <= TOTAL_TIMEOUT_SECONDS:
        parser.error("timeout must fit the bounded supervisor limit")
    if args.tp_size is not None and (
        not args.run
        or args.port is None
        or args.rank is None
        or not 0 <= args.rank < args.tp_size
    ):
        parser.error("internal rank invocation requires run/port and a valid rank")
    args.model = args.model.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    profiler_helpers.write_json(temporary, value)
    temporary.replace(path)


def selected_weight(name):
    parts = name.removeprefix("model.").split(".")
    return (
        len(parts) >= 4
        and parts[:2] == ["layers", str(LAYER_INDEX)]
        and parts[2] in ("ffn", "mlp")
    )


def locate_source(relative):
    # Reading package paths must not import torch, vLLM or initialize a device.
    package_name, package_relative = relative.split("/", 1)
    if package_name in ("vllm", "vllm_ascend"):
        # Editable installs can expose /opt sources through a meta-path finder
        # without adding their checkout to sys.path or the distribution root.
        # Resolve only the top-level package, so its __init__ is never executed.
        spec = importlib.util.find_spec(package_name)
        candidates = (
            [
                Path(root) / package_relative
                for root in (spec.submodule_search_locations or ())
            ]
            if spec is not None
            else []
        )
    else:
        candidates = [Path(__file__).resolve().parents[2] / relative]
        candidates.extend(Path(entry) / relative for entry in sys.path if entry)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    searched = ", ".join(map(str, candidates)) or "package not found"
    raise FileNotFoundError(
        f"installed source is required: {relative}; searched: {searched}"
    )


def source_identity(output_dir):
    manifest = {}
    for relative in (*SOURCE_PATHS, "tests/npu/dsv41_moe_parity_probe.py"):
        path = locate_source(relative)
        raw = path.read_bytes()
        target = output_dir / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        manifest[relative] = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return manifest


def worker_environment():
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AFD_", "VLLM_", "MOONCAKE_", "ENGRAM_"))
    }
    environment["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    environment["VLLM_PLUGINS"] = NATIVE_PLUGINS
    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return environment


def checkpoint_identity(model):
    config_path, index_path = (
        model / "config.json",
        model / "model.safetensors.index.json",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # This checkpoint is a multimodal wrapper; the native normalizer selects
    # text_config without changing the root checkpoint bytes or layer count.
    text_config = config["text_config"]
    index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    selected = {name: shard for name, shard in index.items() if selected_weight(name)}
    if not selected:
        raise ValueError("checkpoint has no layer-0 FFN weights")
    tensors, shards = {}, {}
    for shard in sorted(set(selected.values())):
        path = (model / shard).resolve()
        if path.parent != model or not path.is_file():
            raise ValueError(
                "index shard must be a file directly inside the checkpoint"
            )
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < header_size <= MAX_HEADER_BYTES:
                raise ValueError("safetensors header exceeds the metadata budget")
            header_raw = stream.read(header_size)
        header = json.loads(header_raw)
        for name in sorted(name for name, owner in selected.items() if owner == shard):
            item = header[name]
            start, stop = item["data_offsets"]
            if not 0 <= start <= stop <= path.stat().st_size - 8 - header_size:
                raise ValueError(f"invalid tensor range: {name}")
            tensors[name] = {"shard": shard, **item, "bytes": stop - start}
        shards[shard] = {
            "bytes": path.stat().st_size,
            "header_sha256": hashlib.sha256(header_raw).hexdigest(),
        }
    for namespace in ("gate", "experts", "shared_experts"):
        if not any(f".{namespace}." in name for name in tensors):
            raise ValueError(f"checkpoint is missing layer-0 {namespace}")
    canonical = set()
    for name in tensors:
        name = name.removeprefix("model.").replace(".ffn.", ".mlp.")
        for old, new in (("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")):
            name = name.replace(f".{old}.", f".{new}.")
        if name.endswith(".scale"):
            name = name.removesuffix(".scale") + ".weight_scale"
        canonical.add(name)
    required = {"layers.0.mlp.gate.weight"}
    # normalize_deepseek_v41_config defaults omitted num_hash_layers to zero.
    # The released V4.1 checkpoint uses an ordinary gate, unlike old V4.
    num_hash_layers = text_config.get("num_hash_layers", 0)
    if num_hash_layers > LAYER_INDEX:
        required.add("layers.0.mlp.gate.tid2eid")
    elif not (
        "layers.0.mlp.gate.bias" in canonical
        or "layers.0.mlp.gate.e_score_correction_bias" in canonical
    ):
        raise ValueError("ordinary layer-0 router requires its checkpoint bias")
    for owner in (
        *(f"experts.{index}" for index in range(text_config["n_routed_experts"])),
        "shared_experts",
    ):
        for projection in ("gate_proj", "down_proj", "up_proj"):
            for suffix in ("weight", "weight_scale"):
                required.add(f"layers.0.mlp.{owner}.{projection}.{suffix}")
    missing = sorted(required - canonical)
    if missing:
        raise ValueError(f"checkpoint lacks required layer-0 weights/scales: {missing}")
    payload_bytes = sum(item["bytes"] for item in tensors.values())
    if payload_bytes > MAX_LAYER_PAYLOAD_BYTES:
        raise ValueError("layer-0 tensor payload exceeds the explicit loading budget")
    return {
        "config_sha256": profiler_helpers.file_sha256(config_path),
        "index_sha256": profiler_helpers.file_sha256(index_path),
        "config_num_hidden_layers": text_config["num_hidden_layers"],
        "config_hidden_size": text_config["hidden_size"],
        "config_num_hash_layers_raw": text_config.get("num_hash_layers"),
        "config_num_hash_layers": num_hash_layers,
        "config_n_shared_experts": text_config["n_shared_experts"],
        "required_tensor_namespace_count": len(required),
        "tensor_payload_bytes_read": 0,
        "selected_payload_bytes": payload_bytes,
        "max_layer_payload_bytes": MAX_LAYER_PAYLOAD_BYTES,
        "selected_tensors": tensors,
        "selected_shards": shards,
        "scope": (
            "config/index and selected-shard headers; "
            "payload hashes only exist after --run"
        ),
    }


def tensor_bytes(tensor):
    # Called only by explicit --run workers, after runtime imports.
    import torch

    return tensor.detach().cpu().contiguous().view(dtype=torch.uint8).numpy().tobytes()


def save_tensor(path, tensor):
    import torch

    raw = tensor_bytes(tensor)
    path.write_bytes(raw)
    values = tensor.detach().cpu().double()
    finite = bool(torch.isfinite(values).all().item())
    return {
        "path": path.name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "statistics": {
            "finite": finite,
            "max_absolute": values.abs().max().item(),
            "mean": values.mean().item(),
            "mean_absolute": values.abs().mean().item(),
            "l2_norm": values.square().sum().sqrt().item(),
        },
    }


def build_runtime_config(args):
    # Device dependencies are confined to --run child processes.
    from vllm.config import CompilationMode, CUDAGraphMode, ModelConfig, VllmConfig
    from vllm.config.device import DeviceConfig
    from vllm.config.load import LoadConfig
    from vllm.config.parallel import ParallelConfig

    config = VllmConfig(
        model_config=ModelConfig(
            model=str(args.model),
            dtype="bfloat16",
            quantization="deepseek_v4_fp8",
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=1024,
            skip_tokenizer_init=True,
            language_model_only=True,
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=args.tp_size, enable_expert_parallel=False
        ),
        device_config=DeviceConfig(device="npu"),
        load_config=LoadConfig(load_format="safetensors"),
        additional_config={
            "enable_cpu_binding": False,
            "enable_flashcomm1": False,
            "enable_dsa_cp": False,
            "multistream_overlap_shared_expert": False,
            "multistream_engram_overlap": False,
            "multistream_dsv4_dsa_overlap": False,
            "combine_quant_mode": 4,
        },
    )
    config.compilation_config.mode = CompilationMode.NONE
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    config.compilation_config.pass_config.enable_sp = False
    if config.quant_config.get_name() != "deepseek_v4_fp8":
        raise ValueError("the actual DS FP4 expert quantization config is required")
    if config.parallel_config.use_sequence_parallel_moe:
        raise ValueError("the TP diagnostic requires non-SP execution")
    return config


def load_layer(torch, args, config, report):
    from safetensors import safe_open
    from torch import nn
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.utils.torch_utils import set_default_torch_dtype
    from vllm_ascend.ascend_forward_context import MoECommType
    from vllm_ascend.models.deepseek_v41 import model as native
    from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

    class SingleMoELayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = native.DeepseekV41MoE(
                config=carrier.config,
                parallel_config=config.parallel_config,
                quant_config=config.quant_config,
                prefix="model.layers.0.mlp",
            )

    # Native loader mapping uses model.layers.0.mlp. No full native constructor
    # or edited HF layer count is needed to retain that exact namespace.
    carrier = native.AscendDeepseekV41LLMForCausalLM.__new__(
        native.AscendDeepseekV41LLMForCausalLM
    )
    nn.Module.__init__(carrier)
    carrier.config = native.normalize_deepseek_v41_config(config.model_config.hf_config)
    carrier.quant_config = config.quant_config
    carrier.model = nn.Module()
    carrier.model.has_engram = False
    carrier.model.start_layer, carrier.model.end_layer = 0, 1
    device = torch.device("npu", args.devices[args.rank])
    with set_default_torch_dtype(config.model_config.dtype), device:
        carrier.model.layers = nn.ModuleList([SingleMoELayer()])
    carrier.num_moe_layers = 1
    carrier.set_moe_parameters()
    carrier.moe_comm_methods = {kind: get_moe_comm_method(kind) for kind in MoECommType}
    identity = checkpoint_identity(args.model)
    report["checkpoint_before"] = identity
    payload_hashes = {}

    def weights():
        names = identity["selected_tensors"]
        for shard in identity["selected_shards"]:
            with safe_open(args.model / shard, framework="pt", device="cpu") as stream:
                for name in sorted(
                    name for name, item in names.items() if item["shard"] == shard
                ):
                    value = stream.get_tensor(name)
                    raw = tensor_bytes(value)
                    if len(raw) != names[name]["bytes"]:
                        raise ValueError(
                            f"loaded tensor size differs from indexed metadata: {name}"
                        )
                    payload_hashes[name] = hashlib.sha256(raw).hexdigest()
                    yield name, value

    loaded = carrier.load_weights(weights())
    report["checkpoint_payload_sha256"] = payload_hashes
    report["loaded_parameter_names"] = sorted(loaded)
    report["parameter_names_before_postload"] = sorted(
        name for name, _ in carrier.named_parameters()
    )
    missing_parameters = sorted(set(report["parameter_names_before_postload"]) - loaded)
    report["unloaded_parameter_names"] = missing_parameters
    # Be stricter than the native quantized default: a packed parameter must
    # have received checkpoint values before post-load can transform it. The
    # separate header schema above also checks every expert/projection/scale.
    if missing_parameters:
        raise ValueError(f"uninitialized layer-0 parameters: {missing_parameters}")
    loader = DefaultModelLoader(config.load_config)
    loader.track_weights_loading(carrier, set(loaded))
    process_weights_after_loading(carrier, config.model_config, device)
    report["routing_parameters"] = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(tensor_bytes(value)).hexdigest(),
        }
        for name, value in carrier.named_parameters()
        if ".mlp.gate." in name
    }
    report["checkpoint_after"] = checkpoint_identity(args.model)
    if report["checkpoint_before"] != report["checkpoint_after"]:
        raise ValueError("checkpoint metadata changed during loading")
    report["runtime_parameters"] = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in carrier.named_parameters()
    }
    report["component_layer_count"] = len(carrier.model.layers)
    report["runtime_config_num_hidden_layers"] = carrier.config.num_hidden_layers
    if carrier.config.num_hidden_layers != identity["config_num_hidden_layers"]:
        raise ValueError("the full checkpoint configuration must remain unchanged")
    return carrier


def load_native_router_operator(report):
    # Call only after the child binds its device. Native physical-device lookup
    # also loads this extension independently of optional custom-op flags.
    import torch
    from vllm.platforms import current_platform

    current_platform.import_kernels()
    from vllm_ascend import vllm_ascend_C

    extension_path = Path(vllm_ascend_C.__file__).resolve()
    identity = {
        "path": str(extension_path),
        "bytes": extension_path.stat().st_size,
        "sha256": profiler_helpers.file_sha256(extension_path),
    }
    report["compiled_router_operator"] = identity
    schema = torch.ops._C_ascend.moe_gating_top_k_hash.default._schema
    identity["schema"] = str(schema)
    missing = sorted(
        REQUIRED_ROUTER_ARGUMENTS - {argument.name for argument in schema.arguments}
    )
    if missing:
        raise TypeError(f"native MoE router operator lacks arguments: {missing}")
    identity["privateuse1_kernel"] = torch._C._dispatch_has_kernel_for_dispatch_key(
        "_C_ascend::moe_gating_top_k_hash", "PrivateUse1"
    )
    if not identity["privateuse1_kernel"]:
        raise RuntimeError("native MoE router operator lacks its PrivateUse1 kernel")


def run_worker(args):
    report = {
        "tp_size": args.tp_size,
        "rank": args.rank,
        "device": args.devices[args.rank],
        "pid": os.getpid(),
        "execution_success": False,
        "setup_only": args.setup_only,
        "cases": [],
    }
    path = args.output_dir / f"tp{args.tp_size}-rank{args.rank}.json"
    try:
        # Match native serving's ops registration before importing device_op;
        # importing DeviceOperator first leaves its native ops cycle partial.
        import torch
        import torch.distributed as dist
        import torch_npu
        import vllm_ascend.ops  # noqa: F401
        from vllm.config import set_current_vllm_config
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
            get_tp_group,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.forward_context import get_forward_context, set_forward_context
        from vllm.model_executor.custom_op import maybe_get_oot_by_class
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
        from vllm.platforms import current_platform
        from vllm.plugins import load_general_plugins
        from vllm_ascend.ascend_config import init_ascend_config
        from vllm_ascend.ascend_forward_context import (
            _EXTRA_CTX,
            override_mrv2_in_profile_run,
        )
        from vllm_ascend.device.device_op import DeviceOperator
        from vllm_ascend.utils import adapt_patch, register_ascend_customop

        report["sources_before"] = source_identity(
            args.output_dir / f"tp{args.tp_size}-rank{args.rank}"
        )
        report["versions"] = {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
        }
        report["environment"] = {
            name: os.environ.get(name) for name in RECORDED_ENVIRONMENT_KEYS
        }
        report["plugin_policy"] = (
            "Explicit native Ascend plugins only; native deepseek_v4_fp8 registration "
            "and native FP4 post-load, no AFD quantization patches"
        )
        current_platform.pre_register_and_update()
        load_general_plugins()
        adapt_patch()
        config = build_runtime_config(args)
        # Native NPUWorker registers the OOT layer replacements before loading
        # a model. Importing ops alone leaves the generic RoutedExperts active.
        with set_current_vllm_config(config):
            vllm_ascend.ops.register_dummy_fusion_op()
            register_ascend_customop(config)
        init_ascend_config(config)
        routed_experts_cls = maybe_get_oot_by_class(RoutedExperts)
        moe_runner_cls = maybe_get_oot_by_class(MoERunner)
        report["configuration"] = {
            "checkpoint_hf_config": config.model_config.hf_config.to_dict(),
            "dtype": str(config.model_config.dtype),
            "quantization": config.model_config.quantization,
            "quant_config": json.loads(
                json.dumps(config.quant_config.__dict__, default=str)
            ),
            "parallel_config": json.loads(
                json.dumps(config.parallel_config.__dict__, default=str)
            ),
            "additional_config": config.additional_config,
            "scope": (
                "one layer, no EP; excludes the earlier full-model DP8/TP1/EP8 baseline"
            ),
        }
        report["runtime_contracts"] = {
            "set_forward_context": str(inspect.signature(set_forward_context)),
            "override_mrv2_in_profile_run": str(
                inspect.signature(override_mrv2_in_profile_run)
            ),
            "use_v2_model_runner": config.use_v2_model_runner,
            "routed_experts_class": (
                f"{routed_experts_cls.__module__}.{routed_experts_cls.__name__}"
            ),
            "moe_runner_class": ".".join(
                (moe_runner_cls.__module__, moe_runner_cls.__name__)
            ),
        }
        if routed_experts_cls is RoutedExperts or moe_runner_cls is MoERunner:
            raise ValueError("native Ascend MoE implementations must be registered")
        if not config.use_v2_model_runner:
            raise ValueError("the diagnostic requires the actual native V2 context")
        if args.setup_only:
            report["setup_success"] = True
            return
        torch.npu.set_device(args.devices[args.rank])
        load_native_router_operator(report)
        device = torch.device("npu", args.devices[args.rank])
        with set_current_vllm_config(config):
            init_distributed_environment(
                world_size=args.tp_size,
                rank=args.rank,
                local_rank=args.devices[args.rank],
                distributed_init_method=f"tcp://127.0.0.1:{args.port}",
                backend="hccl",
                timeout=timedelta(seconds=GROUP_TIMEOUT_SECONDS),
            )
            initialize_model_parallel(
                tensor_model_parallel_size=args.tp_size, backend="hccl"
            )
            carrier = load_layer(torch, args, config, report)
            moe = carrier.model.layers[LAYER_INDEX].mlp
            if (
                moe.hash != (carrier.config.num_hash_layers > LAYER_INDEX)
                or moe.shared_experts is None
            ):
                raise ValueError(
                    "layer 0 must retain its actual router kind and shared experts"
                )
            report["moe"] = {
                "tp_size": moe.tp_size,
                "ep_size": moe.ep_size,
                "sequence_parallel": moe.is_sequence_parallel,
                "hash": moe.hash,
                "shared_experts": type(moe.shared_experts).__name__,
                "quant_config": type(config.quant_config).__name__,
                "gate_dtype": str(moe.gate.weight.dtype),
                "runner_class": type(moe.experts).__name__,
                "routed_experts_class": type(moe.experts.routed_experts).__name__,
            }
            generator = torch.Generator(device="cpu").manual_seed(INPUT_SEED)
            fp32 = torch.randn(
                (len(PROMPT_TOKEN_IDS), carrier.config.hidden_size), generator=generator
            )
            fp32 /= fp32.square().mean(dim=-1, keepdim=True).sqrt()
            # Actual A5 wheel rms_norm_cast returns x_bf16 and x_bf16.float().
            hidden = fp32.bfloat16()
            fp32 = hidden.float()
            ids = torch.tensor(PROMPT_TOKEN_IDS, dtype=torch.int32)
            for label, count in (
                ("prefill-rows", len(PROMPT_TOKEN_IDS)),
                ("decode-row", 1),
            ):
                case = {"label": label, "rows": count}
                report["cases"].append(case)
                stem = f"tp{args.tp_size}-rank{args.rank}-{label}"
                inputs = (
                    hidden[:count].to(device),
                    fp32[:count].to(device),
                    ids[:count].to(device),
                )
                case["inputs"] = {
                    name: save_tensor(args.output_dir / f"{stem}-{name}.bin", value)
                    for name, value in zip(
                        ("hidden", "router", "ids"), inputs, strict=True
                    )
                }
                with (
                    torch.inference_mode(),
                    override_mrv2_in_profile_run(False),
                    set_forward_context(
                        None,
                        config,
                        num_tokens=count,
                        num_tokens_across_dp=torch.tensor([count]),
                    ),
                ):
                    context = get_forward_context()
                    context.additional_kwargs["model_instance"] = carrier
                    context.input_ids = inputs[2]
                    case["moe_comm_type"] = _EXTRA_CTX.moe_comm_type.name
                    output = moe(
                        inputs[0],
                        input_ids=inputs[2],
                        hidden_states_fp32=inputs[1],
                        already_sequence_parallel=False,
                    )
                    torch.npu.synchronize()
                    if _EXTRA_CTX.moe_comm_type.name != "ALLGATHER":
                        raise ValueError(
                            "EP-off component must select the native ALLGATHER path"
                        )
                    case["moe_comm_method"] = type(_EXTRA_CTX.moe_comm_method).__name__
                    gate = DeviceOperator.compute_gate_logits(
                        inputs[0],
                        moe.gate.weight,
                        moe.gate.weight_fp32,
                        prepared_input=inputs[1],
                    )
                    case["gate"] = save_tensor(
                        args.output_dir / f"{stem}-gate.bin", gate.float()
                    )
                if not bool(torch.isfinite(output).all().item()):
                    raise ValueError("native MoE returned nonfinite output")
                case["output"] = save_tensor(
                    args.output_dir / f"{stem}-output.bin", output.float()
                )
                native_value = torch.tensor([1.0], device=device)
                dist.all_reduce(native_value, group=get_tp_group().device_group)
                if native_value.item() != args.tp_size:
                    raise ValueError(
                        "native TP group no longer computes the expected reduction"
                    )
                case["native_tp_check"] = native_value.item()
            report["peak_allocated_device_bytes"] = torch.npu.max_memory_allocated(
                device
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        report["sources_after"] = source_identity(
            args.output_dir / f"tp{args.tp_size}-rank{args.rank}-after"
        )
        if report["sources_before"] != report["sources_after"]:
            raise ValueError("installed sources changed during the component")
        report["execution_success"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
        raise
    finally:
        write_json(path, report)


def compare_outputs(output_dir, left, right):
    a, b = array.array("f"), array.array("f")
    for item, target in ((left, a), (right, b)):
        raw = (output_dir / item["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("saved output hash does not match its report")
        target.frombytes(raw)
    if len(a) != len(b) or left["shape"] != right["shape"]:
        raise ValueError("TP output shapes differ")
    if not all(math.isfinite(value) for value in (*a, *b)):
        raise ValueError("saved output contains nonfinite values")
    errors = [float(x) - float(y) for x, y in zip(a, b, strict=True)]
    squared = sum(value * value for value in errors)
    reference_squared = sum(float(value) ** 2 for value in a)
    return {
        "exact_equal": a == b,
        "max_absolute_error": max(map(abs, errors)),
        "rmse": math.sqrt(squared / len(a)),
        "relative_l2_error": math.sqrt(squared / reference_squared)
        if reference_squared
        else None,
        "scope": (
            "observed numerical difference; no model-equivalence tolerance asserted"
        ),
    }


def compare_phases(args, report):
    reference = report["phases"][0]["ranks"][0]
    compared = report["phases"][1]["ranks"]
    if any(
        rank["checkpoint_payload_sha256"] != reference["checkpoint_payload_sha256"]
        for rank in compared
    ):
        raise ValueError("the TP phases did not load identical checkpoint tensor bytes")
    if any(rank["sources_before"] != reference["sources_before"] for rank in compared):
        raise ValueError("the TP phases did not run the same installed sources")
    report["comparisons"] = []
    for index, case in enumerate(reference["cases"]):
        for rank in compared:
            other = rank["cases"][index]
            if any(
                other["inputs"][name]["sha256"] != value["sha256"]
                for name, value in case["inputs"].items()
            ):
                raise ValueError("the TP phases did not use identical input bytes")
            report["comparisons"].append(
                {
                    "case": case["label"],
                    "rank": rank["rank"],
                    "output": compare_outputs(
                        args.output_dir, case["output"], other["output"]
                    ),
                    "gate": compare_outputs(
                        args.output_dir, case["gate"], other["gate"]
                    ),
                }
            )
            peer = compared[0]["cases"][index]
            if other["output"]["sha256"] != peer["output"]["sha256"]:
                raise ValueError(
                    "native TP4 ranks did not return identical reduced outputs"
                )


def interrupt_supervisor(signum, _frame):
    raise SystemExit(f"supervisor received {signal.Signals(signum).name}")


def supervise(args, preflight):
    started = time.monotonic()
    deadline = (
        started
        + args.timeout
        - hccl_helpers.REAP_TIMEOUT_SECONDS
        - profiler_helpers.NPU_SMI_TIMEOUT_SECONDS
    )
    children, handles, owned = [], [], {}
    report = {
        "scope": __doc__,
        "preflight": preflight,
        "execution_success": False,
        "model_equivalence_accepted": False,
        "killed_owned_pids": [],
        "phases": [],
        "timeout_seconds": args.timeout,
        "setup_only": args.setup_only,
    }
    previous_sigterm_handler = signal.signal(signal.SIGTERM, interrupt_supervisor)
    try:
        report["idle_boards_before"] = profiler_helpers.sample_idle(
            args.output_dir, "before"
        )
        ports = hccl_helpers.select_port_block(len(TP_SIZES))
        for phase, tp_size in enumerate(TP_SIZES):
            phase_children = []
            for rank in range(1 if args.setup_only else tp_size):
                handle = (args.output_dir / f"tp{tp_size}-rank{rank}.log").open("wb")
                handles.append(handle)
                command = [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--model",
                    str(args.model),
                    "--output-dir",
                    str(args.output_dir),
                    "--devices",
                    ",".join(map(str, args.devices)),
                    "--run",
                    "--tp-size",
                    str(tp_size),
                    "--rank",
                    str(rank),
                    "--port",
                    str(ports + phase),
                ]
                if args.setup_only:
                    command.append("--setup-only")
                environment = worker_environment()
                child = subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=environment,
                )
                children.append(child)
                phase_children.append(child)
            while any(child.poll() is None for child in phase_children):
                for child in phase_children:
                    if child.poll() is None:
                        supervision.observe_owned(child.pid, owned)
                    elif child.returncode != 0:
                        raise RuntimeError(
                            f"TP{tp_size} rank failed; retain its original log"
                        )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "single-layer MoE comparison exceeded its supervisor budget"
                    )
                time.sleep(POLL_SECONDS)
            if any(child.returncode != 0 for child in phase_children):
                raise RuntimeError(f"TP{tp_size} rank exited with an error")
            ranks = [
                json.loads(
                    (args.output_dir / f"tp{tp_size}-rank{rank}.json").read_text()
                )
                for rank in range(1 if args.setup_only else tp_size)
            ]
            success_key = "setup_success" if args.setup_only else "execution_success"
            if not all(rank[success_key] for rank in ranks):
                raise RuntimeError("a rank did not complete its numerical diagnostic")
            report["phases"].append({"tp_size": tp_size, "ranks": ranks})
            profiler_helpers.sample_idle(args.output_dir, f"after-tp{tp_size}")
        if args.setup_only:
            report["setup_success"] = True
        else:
            compare_phases(args, report)
            report["execution_success"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        report["execution_success"] = False
        report["setup_success"] = False
        print(report["error"], flush=True)
    finally:
        # Preserve the cleanup budget if the caller repeats SIGTERM while
        # owned ranks are being reaped. Rank processes retain native signals.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for child in children:
            if child.poll() is None:
                supervision.observe_owned(child.pid, owned)
        if any(child.poll() is None for child in children):
            report["killed_owned_pids"] = supervision.kill_owned(owned)
        reap_deadline = time.monotonic() + hccl_helpers.REAP_TIMEOUT_SECONDS
        for child in children:
            try:
                child.wait(timeout=max(0, reap_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                report["reap_timeout"] = True
                report["execution_success"] = False
                report["setup_success"] = False
        for handle in handles:
            handle.close()
        report["exit_codes"] = [child.returncode for child in children]
        report["owned_pid_start_ticks"] = owned
        try:
            report["idle_boards_after"] = profiler_helpers.sample_idle(
                args.output_dir, "after"
            )
        except BaseException:
            report["postcheck_error"] = traceback.format_exc()
            report["execution_success"] = False
            report["setup_success"] = False
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output_dir / "result.json", report)
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
    return 0 if report.get("setup_success") or report["execution_success"] else 1


def main():
    args = parse_args()
    if args.tp_size is not None:
        run_worker(args)
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        preflight = {
            "scope": (
                "metadata/source inspection only; "
                "no device imports, allocation or payload reads"
            ),
            "checkpoint": checkpoint_identity(args.model),
            "sources": source_identity(args.output_dir),
            "run_requested": args.run,
            "devices": args.devices,
        }
        write_json(args.output_dir / "preflight.json", preflight)
        if not args.run:
            return 0
        return supervise(args, preflight)
    except BaseException:
        write_json(args.output_dir / "failure.json", {"error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
