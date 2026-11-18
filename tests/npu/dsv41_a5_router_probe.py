#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolate the installed A5 hash router; metadata mode imports no runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

from dsv41_dspark_a5_runtime import native_a5_bootstrap

# name, expert count, selected experts, dtype, vision bias. Each case must
# execute in a fresh process because a vector-core fault poisons its context.
CASES = (
    ("e64-k2-fp32", 64, 2, "float32", False),
    ("e384-k2-fp32", 384, 2, "float32", False),
    ("e384-k6-fp32", 384, 6, "float32", False),
    ("e64-k2-bf16", 64, 2, "bfloat16", False),
    ("e384-k6-bf16", 384, 6, "bfloat16", False),
    ("e384-k6-fp32-vision", 384, 6, "float32", True),
)
ROWS = 8
SCALING_FACTOR = 1.5
SEED = 20260909
BIAS_SCALE = 0.2
ROUTING_EPSILON = 1e-20
FP32_TOLERANCE = 1e-5
BF16_TOLERANCE = 1e-2
IMAGE_SENTINEL_LO = 129257
IMAGE_SENTINEL_COUNT = 5
ROUTER_OPERATOR = "_C_ascend::moe_gating_top_k_hash"
MAX_CAPTURED_CALLS = 64
MAX_REPLAY_STORAGE_ELEMENTS = 1024 * 1024
ROUTER_SYMBOLS = (
    "aclnnMoeGatingTopKHashGetWorkspaceSize",
    "aclnnMoeGatingTopKHash",
)


def prepare_native_operator_environment() -> dict:
    """Establish installed vendor precedence before any runtime import.

    The pinned extension snapshots ASCEND_CUSTOM_OPP_PATH at dlopen, before
    native_a5_bootstrap currently runs in the normal Worker. Match the native
    bootstrap's installed vendor first, preserving external transformer roots.
    """
    before = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    record = {
        "before": before,
        "extension_already_imported": "vllm_ascend.vllm_ascend_C" in sys.modules,
        "available": False,
    }
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None:
        return record
    vendor = Path(spec.origin).parent / "_cann_ops_custom/vendors/custom_transformer"
    library = vendor / "op_api/lib/libcust_opapi.so"
    record["vendor"] = str(vendor)
    if not library.is_file():
        return record
    entries = [
        part for part in before.split(os.pathsep) if part and part != str(vendor)
    ]
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join((str(vendor), *entries))
    record.update(
        available=True,
        after=os.environ["ASCEND_CUSTOM_OPP_PATH"],
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    )
    return record


def operator_symbol_providers(
    paths: list[str], symbols: tuple[str, ...] = ROUTER_SYMBOLS
) -> list[dict]:
    """Inspect already loaded ELF handles; never load or invoke an operator."""
    # Runtime-only ELF inspection; metadata children must not load _ctypes.
    import ctypes

    class LibraryInfo(ctypes.Structure):
        _fields_ = (
            ("filename", ctypes.c_char_p),
            ("base", ctypes.c_void_p),
            ("symbol", ctypes.c_char_p),
            ("address", ctypes.c_void_p),
        )

    loader = ctypes.CDLL(None)
    loader.dlopen.argtypes = (ctypes.c_char_p, ctypes.c_int)
    loader.dlopen.restype = ctypes.c_void_p
    loader.dlsym.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
    loader.dlsym.restype = ctypes.c_void_p
    loader.dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(LibraryInfo))
    loader.dladdr.restype = ctypes.c_int
    loader.dlclose.argtypes = (ctypes.c_void_p,)
    loader.dlclose.restype = ctypes.c_int
    providers = []
    for path in paths:
        handle = loader.dlopen(os.fsencode(path), os.RTLD_NOLOAD | os.RTLD_LAZY)
        if not handle:
            continue
        try:
            for symbol in symbols:
                address = loader.dlsym(handle, symbol.encode())
                if address:
                    info = LibraryInfo()
                    if loader.dladdr(address, ctypes.byref(info)):
                        providers.append(
                            {
                                "lookup_library": path,
                                "symbol": symbol,
                                "provider": os.fsdecode(info.filename),
                            }
                        )
        finally:
            loader.dlclose(handle)
    return providers


@contextmanager
def capture_router_inputs(output: Path, operator_name: str = ROUTER_OPERATOR):
    """Observe Python op calls, including calls nested inside opaque MoE ops."""
    import torch
    from torch._ops import OpOverload, OpOverloadPacket

    output.mkdir(parents=True, exist_ok=False)
    capture = {"calls": [], "limit": MAX_CAPTURED_CALLS}
    pending = {}
    operator_codes = (OpOverload.__call__.__code__, OpOverloadPacket.__call__.__code__)

    def observe(frame, event, result):
        if event == "return" and frame in pending:
            record = pending.pop(frame)
            # The native router returns a tuple, never None. Python profile
            # reports None when an exception unwinds this call frame.
            record["operator_returned"] = result is not None
            Path(record["metadata_file"]).write_bytes(
                (json.dumps(record, indent=2) + "\n").encode()
            )
            return
        if event != "call" or frame.f_code not in operator_codes:
            return
        operator = frame.f_locals["self"]
        name = (
            operator._qualified_op_name
            if isinstance(operator, OpOverloadPacket)
            else operator._schema.name
        )
        if name != operator_name:
            return
        if len(capture["calls"]) >= MAX_CAPTURED_CALLS:
            capture["limit_reached"] = True
            return
        schema = (
            operator.default._schema
            if isinstance(operator, OpOverloadPacket)
            else operator._schema
        )
        index = len(capture["calls"]) + 1
        stem = output / f"call-{index:03d}"
        values = dict(
            zip(
                (argument.name for argument in schema.arguments),
                frame.f_locals["args"],
                strict=False,
            )
        )
        values.update(frame.f_locals["kwargs"])
        record = {
            "index": index,
            "schema": str(schema),
            "inputs_file": str(stem.with_suffix(".pt")),
            "metadata_file": str(stem.with_suffix(".json")),
            "capture_complete": False,
            "operator_called": False,
            "operator_returned": False,
            "tensors": {},
            "scalars": {},
            "runtime_context": {
                "inference_mode": torch.is_inference_mode_enabled(),
                "grad_enabled": torch.is_grad_enabled(),
                "deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
            },
        }
        if index == 1:
            libraries = loaded_operator_libraries()
            record["libraries_before_call"] = libraries
            record["symbol_providers_before_call"] = operator_symbol_providers(
                [item["path"] for item in libraries]
            )
            record["custom_opp_path_before_call"] = os.environ.get(
                "ASCEND_CUSTOM_OPP_PATH"
            )
        capture["calls"].append(record)
        cpu_values = {}
        try:
            for name, value in values.items():
                if isinstance(value, torch.Tensor):
                    tensor = {
                        "shape": list(value.shape),
                        "stride": list(value.stride()),
                        "storage_offset": value.storage_offset(),
                        "dtype": str(value.dtype),
                        "device": str(value.device),
                    }
                    record["tensors"][name] = tensor
                    cpu = value.detach().cpu().clone()
                    finite = torch.isfinite(cpu)
                    tensor["finite"] = bool(finite.all())
                    tensor["nonfinite_count"] = cpu.numel() - int(finite.sum())
                    if cpu.numel() and bool(finite.any()):
                        tensor["finite_min"] = float(cpu[finite].min())
                        tensor["finite_max"] = float(cpu[finite].max())
                    cpu_values[name] = cpu
                else:
                    record["scalars"][name] = value
                    cpu_values[name] = value
            torch.save(cpu_values, stem.with_suffix(".pt"))
            record["inputs_sha256"] = hashlib.sha256(
                stem.with_suffix(".pt").read_bytes()
            ).hexdigest()
            record["capture_complete"] = True
            record["operator_called"] = True
            pending[frame] = record
        except BaseException:
            record["capture_traceback"] = traceback.format_exc()
            raise
        finally:
            stem.with_suffix(".json").write_bytes(
                (json.dumps(record, indent=2) + "\n").encode()
            )

    # The locked vllm::moe_forward_shared Python kernel calls the native
    # router inside an opaque dispatcher op. A dispatch mode cannot observe
    # that nested call. A scoped thread-local Python profile observes the
    # OpOverloadPacket/OpOverload call frame without replacing either kernel.
    previous_profile = sys.getprofile()
    capture["observer"] = "thread_local_python_profile"
    try:
        sys.setprofile(observe)
        yield capture
    finally:
        sys.setprofile(previous_profile)
        pending.clear()
        try:
            capture["loaded_operator_libraries"] = loaded_operator_libraries()
        except BaseException:
            capture["library_collection_error"] = traceback.format_exc()


def restore_tensor_layout(value, layout: dict, device: str):
    """Recreate shape, strides and offset without serializing source storage."""
    import torch

    shape, stride = layout["shape"], layout["stride"]
    offset = layout["storage_offset"]
    elements = (
        offset
        + 1
        + sum((size - 1) * step for size, step in zip(shape, stride, strict=True))
    )
    if not 0 < elements <= MAX_REPLAY_STORAGE_ELEMENTS:
        raise ValueError("captured layout exceeds the bounded replay allocation")
    storage = torch.empty(elements, dtype=value.dtype, device=device)
    restored = storage.as_strided(shape, stride, offset)
    restored.copy_(value)
    return restored


def loaded_operator_libraries() -> list[dict]:
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    paths = {
        line.split()[-1]
        for line in maps.read_text().splitlines()
        if any(
            name in line
            for name in (
                "libcust_opapi.so",
                "libopapi.so",
                "libopapi_transformer.so",
                "liboptiling.so",
                "vllm_ascend_C.",
            )
        )
    }
    return [
        {"path": name, "sha256": hashlib.sha256(Path(name).read_bytes()).hexdigest()}
        for name in sorted(paths)
        if Path(name).is_file()
    ]


def bootstrap_router(device: int, report: dict) -> None:
    # Explicit device mode only; bootstrap and registration match the normal
    # A5 probe. No worker, checkpoint, collective or replacement router.
    from dsv41_dspark_speculator_probe import NATIVE_PLUGINS

    os.environ["VLLM_PLUGINS"] = NATIVE_PLUGINS
    import torch
    import torch_npu  # noqa: F401
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm_ascend.utils import adapt_patch

    report["runtime_imported"] = True
    report["phase"] = "bootstrap"
    current_platform.pre_register_and_update()
    load_general_plugins()
    adapt_patch()
    torch.npu.set_device(device)
    report["native_a5_bootstrap"] = native_a5_bootstrap()
    report["schema"] = str(torch.ops._C_ascend.moe_gating_top_k_hash.default._schema)


def replay_inputs(path: Path, device: int, output: Path, report: dict) -> None:
    import torch

    record = json.loads(path.with_suffix(".json").read_text("utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["inputs_sha256"]:
        raise ValueError("captured router inputs differ from their SHA")
    values = torch.load(path, map_location="cpu", weights_only=True)
    report["source_capture"] = record
    report["phase"] = "restore_inputs"
    if not all(tensor["finite"] for tensor in record["tensors"].values()):
        raise ValueError("captured router inputs contain nonfinite values")
    bootstrap_router(device, report)
    npu_values = {
        name: restore_tensor_layout(value, record["tensors"][name], f"npu:{device}")
        if isinstance(value, torch.Tensor)
        else value
        for name, value in values.items()
    }
    report["restored_layouts"] = {
        name: {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "storage_offset": value.storage_offset(),
        }
        for name, value in npu_values.items()
        if isinstance(value, torch.Tensor)
    }
    report["phase"] = "operator"
    report["runtime_context"] = {
        "inference_mode": torch.is_inference_mode_enabled(),
        "grad_enabled": torch.is_grad_enabled(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    libraries = loaded_operator_libraries()
    report["libraries_before_call"] = libraries
    report["symbol_providers_before_call"] = operator_symbol_providers(
        [item["path"] for item in libraries]
    )
    (output / "result.json").write_bytes((json.dumps(report, indent=2) + "\n").encode())
    weights, ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(**npu_values)
    torch.npu.synchronize()
    report["operator_returned"] = True
    weights, ids = weights.cpu(), ids.cpu()
    torch.save({"weights": weights, "ids": ids}, output / "outputs.pt")
    report["output_finite"] = bool(torch.isfinite(weights).all())
    report["output_ids"] = ids.tolist()
    report["output_weights"] = weights.tolist()
    report["output_ids_in_range"] = bool(
        ((ids >= 0) & (ids < values["x"].shape[1])).all()
    )
    report["success"] = report["output_finite"] and report["output_ids_in_range"]
    report["phase"] = "complete"


def run_case(
    case: tuple[str, int, int, str, bool], device: int, output: Path, report: dict
) -> None:
    import torch

    bootstrap_router(device, report)
    _, experts, top_k, dtype_name, vision = case
    dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    logits = torch.randn(ROWS, experts, dtype=dtype, generator=generator)
    bias = torch.randn(experts, dtype=dtype, generator=generator) * BIAS_SCALE
    bias_vl = (
        torch.randn(experts, dtype=dtype, generator=generator) * BIAS_SCALE
        if vision
        else None
    )
    input_ids = (
        torch.tensor(
            [
                11,
                IMAGE_SENTINEL_LO,
                22,
                IMAGE_SENTINEL_LO + 2,
                33,
                IMAGE_SENTINEL_LO + 4,
                44,
                55,
            ],
            dtype=torch.int64,
        )
        if vision
        else None
    )
    # Oracle follows the locked native test_npu_moe_gating_top_k_hash.py
    # at vLLM-Ascend 7e37e29a; bias affects IDs, not gathered weights.
    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    row_bias = bias.float()
    if vision:
        image = (input_ids >= IMAGE_SENTINEL_LO) & (
            input_ids < IMAGE_SENTINEL_LO + IMAGE_SENTINEL_COUNT
        )
        row_bias = torch.where(image[:, None], bias_vl.float(), bias.float())
    expected_ids = torch.topk(scores + row_bias, top_k, dim=-1, sorted=True).indices
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = (
        expected_weights / expected_weights.sum(dim=-1, keepdim=True) * SCALING_FACTOR
    ).to(dtype)
    inputs = {
        "x": logits,
        "k": top_k,
        "bias": bias,
        "input_ids": input_ids,
        "tid2eid": None,
        "k_group": 1,
        "group_count": 1,
        "routed_scaling_factor": SCALING_FACTOR,
        "eps": ROUTING_EPSILON,
        "group_select_mode": 1,
        "renorm": 0,
        "norm_type": 2,
        "out_flag": False,
        "bias_vl": bias_vl,
        "image_sentinel_lo": IMAGE_SENTINEL_LO,
        "image_sentinel_count": IMAGE_SENTINEL_COUNT,
    }
    torch.save(inputs, output / "inputs.pt")
    report["inputs_sha256"] = hashlib.sha256(
        (output / "inputs.pt").read_bytes()
    ).hexdigest()
    npu_inputs = {
        name: value.to(device=f"npu:{device}")
        if isinstance(value, torch.Tensor)
        else value
        for name, value in inputs.items()
    }
    report["phase"] = "operator"
    (output / "result.json").write_bytes((json.dumps(report, indent=2) + "\n").encode())
    actual_weights, actual_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
        **npu_inputs
    )
    torch.npu.synchronize()
    report["operator_returned"] = True
    report["phase"] = "comparison"
    actual_weights, actual_ids = actual_weights.cpu(), actual_ids.cpu()
    torch.save({"weights": actual_weights, "ids": actual_ids}, output / "outputs.pt")
    report["actual_ids"] = actual_ids.tolist()
    report["expected_ids"] = expected_ids.tolist()
    report["max_weight_absolute_error"] = float(
        (actual_weights.float() - expected_weights.float()).abs().max()
    )
    torch.testing.assert_close(actual_ids, expected_ids.int(), rtol=0, atol=0)
    tolerance = FP32_TOLERANCE if dtype == torch.float32 else BF16_TOLERANCE
    torch.testing.assert_close(
        actual_weights, expected_weights, rtol=tolerance, atol=tolerance
    )
    report["reference_matched"] = True
    report["phase"] = "complete"
    report["success"] = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--case", choices=[case[0] for case in CASES])
    operation.add_argument("--replay-inputs", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be nonnegative")
    case = next((case for case in CASES if case[0] == args.case), None)
    report = {
        "case": args.case,
        "shape": [ROWS, case[1]] if case else None,
        "top_k": case[2] if case else None,
        "dtype": case[3] if case else None,
        "vision_bias": case[4] if case else None,
        "replay_inputs": str(args.replay_inputs) if args.replay_inputs else None,
        "hash_routing": False,
        "device": args.device,
        "ascend_launch_blocking": os.environ.get("ASCEND_LAUNCH_BLOCKING"),
        "runtime_imported": False,
        "operator_returned": False,
        "reference_matched": False,
        "model_executed": False,
        "collective_executed": False,
        "functional_worker_component_accepted": False,
        "diagnostic_only": True,
        "success": not args.run,
    }
    if not args.run:
        print(json.dumps(report))
        return 0
    if os.name != "posix" or os.environ.get("ASCEND_LAUNCH_BLOCKING") != "1":
        parser.error("--run requires Linux and ASCEND_LAUNCH_BLOCKING=1")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        report["operator_environment_preparation"] = (
            prepare_native_operator_environment()
        )
        if args.replay_inputs:
            replay_inputs(args.replay_inputs, args.device, args.output_dir, report)
        else:
            run_case(case, args.device, args.output_dir, report)
    except BaseException:
        report["traceback"] = traceback.format_exc()
    finally:
        report["runtime_environment"] = {
            name: os.environ.get(name)
            for name in ("ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH")
        }
        try:
            report["loaded_operator_libraries"] = loaded_operator_libraries()
        except BaseException:
            report["library_collection_error"] = traceback.format_exc()
            report["success"] = False
        (args.output_dir / "result.json").write_bytes(
            (json.dumps(report, indent=2) + "\n").encode()
        )
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
