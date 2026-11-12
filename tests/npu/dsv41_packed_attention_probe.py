# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real A5 packed-cache/qsmla head-independence probe, without a model.

Four small cases cover three-query prefill and one-query decode with original
KV alone or original plus compressed KV. Only --run imports device packages.
The launcher supplies the existing container environment before this process.
This component result does not qualify HCCL, full-model inference or performance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

HEAD_DIM = 512
LOCAL_QUERY_HEADS = 16
NATIVE_QUERY_HEADS = 64
NUM_KV_HEADS = 1
BLOCK_SIZE = 128
WINDOW_SIZE = 128
COMPRESSED_TOPK = 128
CACHE_ROWS = 8
SWA_PACKED_WIDTH = HEAD_DIM + (HEAD_DIM // 32) * 2
COMPRESSED_PACKED_WIDTH = HEAD_DIM // 2 + (HEAD_DIM // 16) * 2
RELATIVE_TOLERANCE = 0.01
ABSOLUTE_TOLERANCE = 0.002
CASES = (("prefill", 3, 0), ("decode", 1, 5))
DISTRIBUTIONS = (
    "torch",
    "torch-npu",
    "vllm",
    "vllm-ascend",
    "cann-ops-transformer",
)


def source_identity(module):
    path = Path(inspect.getfile(module))
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def distribution_versions():
    versions = {"python": sys.version}
    for name in DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_case(torch, native, device, tokens, start_position, compressed):
    """Use the native write, causal-index and fixed-64 metadata producers."""
    source = torch.arange(CACHE_ROWS * HEAD_DIM).reshape(CACHE_ROWS, HEAD_DIM)
    ori_values = ((source.float() / 31).sin() * 0.25).to(torch.bfloat16).to(device)
    cmp_values = ((source.float() / 47).cos() * 0.375).to(torch.bfloat16).to(device)
    slots = torch.arange(CACHE_ROWS, dtype=torch.int64, device=device)
    ori_kv = torch.zeros(
        (1, BLOCK_SIZE, NUM_KV_HEADS, SWA_PACKED_WIDTH),
        dtype=torch.uint8,
        device=device,
    )
    native.write_attention_cache(ori_kv, slots, ori_values, kind="win")
    cmp_kv = None
    if compressed:
        cmp_kv = torch.zeros(
            (1, BLOCK_SIZE, NUM_KV_HEADS, COMPRESSED_PACKED_WIDTH),
            dtype=torch.uint8,
            device=device,
        )
        native.write_attention_cache(cmp_kv, slots, cmp_values, kind="cmp")

    positions = torch.arange(
        start_position, start_position + tokens, dtype=torch.int64, device=device
    )
    ori_indices, ori_lengths = native.MixedQuantPackedCacheOps.build_window_indices(
        positions, WINDOW_SIZE
    )
    query_start_loc = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    # Native build_smla_metadata uses one common length carrier and always sets
    # has_cmp_kv=True, including SWA-only layers. Keep that production behavior.
    length_rows = torch.full_like(ori_lengths, WINDOW_SIZE)
    task_metadata = native.build_smla_metadata(length_rows, query_start_loc)
    block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    seq_lens = torch.tensor([start_position + tokens], dtype=torch.int32, device=device)
    compressed_indices = torch.full(
        (tokens, COMPRESSED_TOPK), -1, dtype=torch.int32, device=device
    )
    compressed_indices[:, :2] = torch.tensor([0, 1], dtype=torch.int32, device=device)
    compressed_lengths = torch.full((tokens, 1), 2, dtype=torch.int32, device=device)
    compressed_seq_lens = torch.tensor([CACHE_ROWS], dtype=torch.int32, device=device)
    metadata = SimpleNamespace(
        positions=positions,
        swa=SimpleNamespace(
            ori_sparse_indices=ori_indices,
            ori_topk_length=ori_lengths,
            smla_metadata=task_metadata,
            block_table=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
        ),
        attention=SimpleNamespace(
            block_table=block_table,
            cache_seq_lens=compressed_seq_lens,
        ),
    )
    query_source = torch.arange(tokens * LOCAL_QUERY_HEADS * HEAD_DIM).reshape(
        tokens, LOCAL_QUERY_HEADS, HEAD_DIM
    )
    q = ((query_source.float() / 67).sin() * 0.5).to(torch.bfloat16).to(device)
    sinks = torch.linspace(-0.75, 0.75, LOCAL_QUERY_HEADS, device=device)
    padding_heads = NATIVE_QUERY_HEADS - LOCAL_QUERY_HEADS
    extra_source = torch.arange(tokens * padding_heads * HEAD_DIM).reshape(
        tokens, padding_heads, HEAD_DIM
    )
    extra_q = ((extra_source.float() / 53).cos() * 0.75).to(torch.bfloat16).to(device)
    extra_sinks = torch.linspace(0.25, 1.25, padding_heads, device=device)
    nonzero_q64 = torch.cat((q, extra_q), dim=1)
    nonzero_sinks64 = torch.cat((sinks, extra_sinks))
    zero_q64 = torch.nn.functional.pad(q, (0, 0, 0, padding_heads))
    zero_sinks64 = torch.nn.functional.pad(sinks, (0, padding_heads))
    stable_inputs = {
        "q16": q,
        "sinks16": sinks,
        "q64_nonzero_tail": nonzero_q64,
        "sinks64_nonzero_tail": nonzero_sinks64,
        "q64_zero_tail": zero_q64,
        "sinks64_zero_tail": zero_sinks64,
        "ori_kv": ori_kv,
        "ori_values": ori_values,
        "cmp_values": cmp_values,
        "slots": slots,
        "positions": positions,
        "ori_indices": ori_indices,
        "ori_lengths": ori_lengths,
        "length_rows": length_rows,
        "compressed_indices": compressed_indices,
        "compressed_lengths": compressed_lengths,
        "block_table": block_table,
        "query_start_loc": query_start_loc,
        "seq_lens": seq_lens,
        "compressed_seq_lens": compressed_seq_lens,
    }
    if cmp_kv is not None:
        stable_inputs["cmp_kv"] = cmp_kv
    return {
        "q": q,
        "sinks": sinks,
        "ori_kv": ori_kv,
        "cmp_kv": cmp_kv,
        "metadata": metadata,
        "compressed_indices": compressed_indices,
        "compressed_lengths": compressed_lengths if compressed else None,
        "q64_nonzero": nonzero_q64,
        "sinks64_nonzero": nonzero_sinks64,
        "q64_zero": zero_q64,
        "sinks64_zero": zero_sinks64,
        "stable_inputs": stable_inputs,
    }


def execute_case(torch, native, backend, inputs, case):
    snapshots = {
        name: tensor.cpu().clone() for name, tensor in inputs["stable_inputs"].items()
    }
    common = (
        inputs["ori_kv"],
        inputs["cmp_kv"],
        inputs["metadata"],
        inputs["compressed_indices"],
    )
    options = {
        "window_size": WINDOW_SIZE,
        "softmax_scale": HEAD_DIM**-0.5,
        "compressed_lengths": inputs["compressed_lengths"],
    }
    case["stage"] = "native64zero"
    native_zero = native.qsmla(
        inputs["q64_zero"], *common, sinks=inputs["sinks64_zero"], **options
    )
    torch.npu.synchronize()
    # Keep each task table alive until its kernel completes, then regenerate
    # it. The ABI does not promise reusable consumed scheduling workspace.
    case["stage"] = "native64nonzero"
    inputs["metadata"].swa.smla_metadata = native.build_smla_metadata(
        inputs["stable_inputs"]["length_rows"],
        inputs["metadata"].swa.query_start_loc,
    )
    native_nonzero = native.qsmla(
        inputs["q64_nonzero"], *common, sinks=inputs["sinks64_nonzero"], **options
    )
    torch.npu.synchronize()
    case["stage"] = "adapter16"
    inputs["metadata"].swa.smla_metadata = native.build_smla_metadata(
        inputs["stable_inputs"]["length_rows"],
        inputs["metadata"].swa.query_start_loc,
    )
    task_metadata = inputs["metadata"].swa.smla_metadata
    metadata_before = task_metadata.cpu().clone()
    actual = backend.qsmla(inputs["q"], *common, sinks=inputs["sinks"], **options)
    torch.npu.synchronize()
    case["stage"] = "compare"
    scheduling_metadata_unchanged = torch.equal(task_metadata.cpu(), metadata_before)
    actual_cpu = actual.float().cpu()
    zero_cpu = native_zero[:, :LOCAL_QUERY_HEADS, :].float().cpu()
    nonzero_cpu = native_nonzero[:, :LOCAL_QUERY_HEADS, :].float().cpu()
    assert actual.shape == inputs["q"].shape
    assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
    assert bool(torch.isfinite(actual_cpu).all()), "nonfinite adapted output"
    assert bool(torch.isfinite(zero_cpu).all()), "nonfinite native zero-tail output"
    assert bool(torch.isfinite(nonzero_cpu).all()), (
        "nonfinite native nonzero-tail output"
    )
    assert bool(torch.count_nonzero(actual_cpu)), "all-zero output is inconclusive"
    case.update(
        output_shape=list(actual.shape),
        output_dtype=str(actual.dtype),
        output_contiguous=actual.is_contiguous(),
        max_abs_diff_native_zero=(actual_cpu - zero_cpu).abs().max().item(),
        max_abs_diff_native_nonzero=(actual_cpu - nonzero_cpu).abs().max().item(),
        max_abs_diff_extra_heads=(zero_cpu - nonzero_cpu).abs().max().item(),
        output_max_absolute_value=actual_cpu.abs().max().item(),
        # Scheduling metadata may include kernel-owned workspace state. Observe
        # byte changes without imposing an undocumented immutability contract.
        scheduling_metadata_unchanged=scheduling_metadata_unchanged,
    )
    for expected in (zero_cpu, nonzero_cpu):
        torch.testing.assert_close(
            actual_cpu,
            expected,
            rtol=RELATIVE_TOLERANCE,
            atol=ABSOLUTE_TOLERANCE,
        )
    for name, tensor in inputs["stable_inputs"].items():
        assert torch.equal(tensor.cpu(), snapshots[name]), f"input changed: {name}"
    case["inputs_and_caches_unchanged"] = True
    case["passed"] = True
    case["stage"] = "completed"


def run_probe(device_index, report):
    # Deferred device imports keep ordinary import and --help CPU-safe. The
    # installed kernels below execute only after the explicit --run CLI gate.
    import torch
    import torch_npu

    torch.npu.set_device(device_index)
    # Match the native serving import order and avoid its device/ops cycle.
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.models.deepseek_v41 import mixed_quant_attention as native

    from afd_plugin.model_executor.models.npu import dsv41_packed_cache as adapter

    device = torch.device(f"npu:{device_index}")
    report.update(
        device=str(device),
        device_name=torch.npu.get_device_name(device_index),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        native_source=source_identity(native),
        adapter_source=source_identity(adapter),
    )
    for phase, tokens, start_position in CASES:
        for compressed in (False, True):
            case = {
                "phase": phase,
                "tokens": tokens,
                "start_position": start_position,
                "compressed": compressed,
                "passed": False,
                "stage": "setup",
                "q_shape": [tokens, LOCAL_QUERY_HEADS, HEAD_DIM],
                "native_q_shape": [tokens, NATIVE_QUERY_HEADS, HEAD_DIM],
                "ori_cache_shape": [1, BLOCK_SIZE, NUM_KV_HEADS, SWA_PACKED_WIDTH],
                "cmp_cache_shape": (
                    [1, BLOCK_SIZE, NUM_KV_HEADS, COMPRESSED_PACKED_WIDTH]
                    if compressed
                    else None
                ),
            }
            report["cases"].append(case)
            try:
                with torch.inference_mode():
                    inputs = build_case(
                        torch, native, device, tokens, start_position, compressed
                    )
                    execute_case(
                        torch, native, adapter.AFDMixedQuantPackedCacheOps, inputs, case
                    )
            except Exception:
                case["traceback"] = traceback.format_exc()
                # A device/kernel error can poison the context; preserve this
                # first failure instead of submitting further kernels.
                raise
    report["passed"] = all(case["passed"] for case in report["cases"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "scope": "real A5 packed-cache writes and TP4 qsmla head independence only",
        "passed": False,
        "versions": distribution_versions(),
        "relative_tolerance": RELATIVE_TOLERANCE,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "cases": [],
    }
    try:
        run_probe(args.device, report)
    except Exception:
        report["traceback"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
