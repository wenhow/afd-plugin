# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Check real NPU stage views and binding cleanup without model construction.

The native model, decoder and Attention shells bypass their constructors. Only
ordinary tensors and readiness events are allocated; no native HC, indexer,
Engram producer, MoE or checkpoint weights are executed.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

CAPACITY = 12
ACTIVE_ROWS = 7
SPLIT_ROW = 3
LAYER_COUNT = 3
TOPK_WIDTH = 4
CANDIDATE_WIDTH = 6
INITIAL_VALUE = -11
BIND_WRITE_BASE = 100
STAGE_VALUE_STRIDE = 10
ENGRAM_CHILD_WRITE_BASE = 101
ENGRAM_PARENT_WRITE_BASE = 201
BUFFER_NAMES = ("topk_indices", "candidates", "candidate_lengths", "topk_lengths")
LAYOUTS = (
    ("u1", ((0, ACTIVE_ROWS),)),
    ("u2-offset", ((0, SPLIT_ROW), (SPLIT_ROW, ACTIVE_ROWS))),
    ("empty-first", ((0, 0), (0, ACTIVE_ROWS))),
    ("empty-last", ((0, ACTIVE_ROWS), (ACTIVE_ROWS, ACTIVE_ROWS))),
    ("empty-parent", ((0, 0), (0, 0))),
)


def tensor_identity(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "storage_offset": tensor.storage_offset(),
        "storage_pointer": tensor.untyped_storage().data_ptr(),
        "data_pointer": tensor.data_ptr(),
        "contiguous": tensor.is_contiguous(),
    }


def verify_alias(parent, child, token_slice):
    assert child.shape == parent[token_slice].shape
    assert child.stride() == parent.stride()
    assert child.dtype == parent.dtype and child.device == parent.device
    assert child.untyped_storage().data_ptr() == parent.untyped_storage().data_ptr()
    assert child.storage_offset() == (
        parent.storage_offset() + token_slice.start * parent.stride(0)
    )
    assert child.is_contiguous()


def native_model_shell(torch, native, parent):
    model = native.DeepseekV41Model.__new__(native.DeepseekV41Model)
    torch.nn.Module.__init__(model)
    model.shared_attention_state = parent
    layers = []
    for _ in range(LAYER_COUNT):
        layer = native.DeepseekV41DecoderLayer.__new__(native.DeepseekV41DecoderLayer)
        torch.nn.Module.__init__(layer)
        attention = native.DeepseekV41Attention.__new__(native.DeepseekV41Attention)
        torch.nn.Module.__init__(attention)
        attention.shared_state = parent
        layer.self_attn = attention
        layers.append(layer)
    model.layers = torch.nn.ModuleList(layers)
    return model


def verify_bindings(model, expected):
    assert model.shared_attention_state is expected
    assert all(layer.self_attn.shared_state is expected for layer in model.layers)


def verify_shared_states(torch, native, stage_states_class, device):
    results = []
    for optional_lengths in (False, True):
        for label, bounds in LAYOUTS:
            tensors = {
                "topk_indices": torch.full(
                    (CAPACITY, TOPK_WIDTH),
                    INITIAL_VALUE,
                    dtype=torch.int32,
                    device=device,
                ),
                "candidates": torch.full(
                    (CAPACITY, 1, CANDIDATE_WIDTH),
                    INITIAL_VALUE,
                    dtype=torch.int32,
                    device=device,
                ),
                "candidate_lengths": None
                if optional_lengths
                else torch.full(
                    (CAPACITY, 1), INITIAL_VALUE, dtype=torch.int32, device=device
                ),
                "topk_lengths": None
                if optional_lengths
                else torch.full(
                    (CAPACITY, 1), INITIAL_VALUE, dtype=torch.int32, device=device
                ),
            }
            parent = native.DeepseekV41SharedAttentionState(**tensors)
            model = native_model_shell(torch, native, parent)
            slices = tuple(slice(start, stop) for start, stop in bounds)
            bindings = stage_states_class(model, slices)
            expected = {
                name: None if tensor is None else tensor.cpu().clone()
                for name, tensor in tensors.items()
            }
            observed = {
                "layout": label,
                "optional_lengths_absent": optional_lengths,
                "slices": list(bounds),
                "stages": [],
            }
            child_states = []
            for stage_idx, token_slice in enumerate(slices):
                fields = {}
                with bindings.bind(stage_idx) as stage:
                    child_states.append(stage)
                    assert stage is not parent
                    verify_bindings(model, stage)
                    children = (
                        stage.topk_indices,
                        stage.candidates,
                        stage.candidate_lengths,
                        stage.topk_lengths,
                    )
                    for field_idx, (name, child) in enumerate(
                        zip(BUFFER_NAMES, children, strict=True)
                    ):
                        source = tensors[name]
                        if source is None:
                            assert child is None
                            continue
                        verify_alias(source, child, token_slice)
                        value = (
                            BIND_WRITE_BASE + stage_idx * STAGE_VALUE_STRIDE + field_idx
                        )
                        child.fill_(value)
                        expected[name][token_slice] = value
                        torch.testing.assert_close(
                            source.cpu(), expected[name], rtol=0, atol=0
                        )
                        fields[name] = tensor_identity(child)
                    try:
                        with bindings.bind(stage_idx):
                            raise AssertionError(
                                "nested stage binding unexpectedly entered"
                            )
                    except RuntimeError as exc:
                        assert "reentrant" in str(exc)
                    verify_bindings(model, stage)
                verify_bindings(model, parent)
                try:
                    with bindings.bind(stage_idx) as stage:
                        verify_bindings(model, stage)
                        raise ValueError("intentional stage body failure")
                except ValueError as exc:
                    assert str(exc) == "intentional stage body failure"
                verify_bindings(model, parent)
                observed["stages"].append(
                    {
                        "stage_idx": stage_idx,
                        "fields": fields,
                        "normal_restore": True,
                        "exception_restore": True,
                        "nested_rejection_preserved_binding": True,
                    }
                )
            assert len({id(state) for state in child_states}) == len(child_states)
            for name, source in tensors.items():
                if source is not None:
                    torch.testing.assert_close(
                        source.cpu(), expected[name], rtol=0, atol=0
                    )
            observed["parent_bindings_restored"] = True
            observed["outside_rows_unchanged"] = True
            results.append(observed)
    return results


def verify_engram(torch, inputs_class, device):
    lookups = {
        0: torch.arange(CAPACITY * TOPK_WIDTH, dtype=torch.int64, device=device)
        .reshape(CAPACITY, TOPK_WIDTH)
        .to(torch.bfloat16),
        1: torch.arange(
            CAPACITY * TOPK_WIDTH, dtype=torch.int64, device=device
        ).reshape(CAPACITY, TOPK_WIDTH),
        2: torch.arange(
            CAPACITY * CANDIDATE_WIDTH, dtype=torch.float32, device=device
        ).reshape(CAPACITY, CANDIDATE_WIDTH),
    }
    mask = torch.arange(CAPACITY, dtype=torch.int32, device=device).remainder(2).bool()
    pending = {layer: torch.npu.Event() for layer in lookups}
    mask_ready = torch.npu.Event()
    for event in (*pending.values(), mask_ready):
        event.record(torch.npu.current_stream())
    parent = inputs_class(lookups, mask, pending, mask_ready)
    original = {layer: tensor.cpu().clone() for layer, tensor in lookups.items()}
    original_mask = mask.cpu().clone()
    slices = (
        slice(0, SPLIT_ROW),
        slice(SPLIT_ROW, ACTIVE_ROWS),
        slice(ACTIVE_ROWS, ACTIVE_ROWS),
    )
    children = tuple(parent.slice(token_slice) for token_slice in slices)
    for layer, tensor in lookups.items():
        torch.testing.assert_close(tensor.cpu(), original[layer], rtol=0, atol=0)
    torch.testing.assert_close(mask.cpu(), original_mask, rtol=0, atol=0)
    reports = []
    for stage_idx, (token_slice, child) in enumerate(
        zip(slices, children, strict=True)
    ):
        assert child.pending is pending and child.mask_ready_event is mask_ready
        assert child.lookups.keys() == lookups.keys()
        identities = {}
        for layer, source in lookups.items():
            view = child.lookups[layer]
            verify_alias(source, view, token_slice)
            torch.testing.assert_close(
                view.cpu(), original[layer][token_slice], rtol=0, atol=0
            )
            # Both directions must alias: child writes reach the parent; parent
            # writes subsequently reach the already-created child view.
            view.fill_(stage_idx + ENGRAM_CHILD_WRITE_BASE)
            original[layer][token_slice] = stage_idx + ENGRAM_CHILD_WRITE_BASE
            torch.testing.assert_close(source.cpu(), original[layer], rtol=0, atol=0)
            source[token_slice].fill_(stage_idx + ENGRAM_PARENT_WRITE_BASE)
            original[layer][token_slice] = stage_idx + ENGRAM_PARENT_WRITE_BASE
            torch.testing.assert_close(
                view.cpu(), original[layer][token_slice], rtol=0, atol=0
            )
            identities[str(layer)] = tensor_identity(view)
        verify_alias(mask, child.mask, token_slice)
        child.mask.fill_(True)
        original_mask[token_slice] = True
        torch.testing.assert_close(mask.cpu(), original_mask, rtol=0, atol=0)
        reports.append(
            {
                "stage_idx": stage_idx,
                "slice": [token_slice.start, token_slice.stop],
                "lookups": identities,
                "mask": tensor_identity(child.mask),
                "events_preserved": True,
                "bidirectional_alias": True,
            }
        )
    torch.npu.synchronize()
    assert all(event.query() for event in (*pending.values(), mask_ready))
    empty = inputs_class({}, torch.empty(0, dtype=torch.bool, device=device))
    for token_slice in slices:
        assert empty.slice(token_slice) is empty
    try:
        parent.slice(slice(0, CAPACITY + 1))
    except ValueError as exc:
        assert "parent token storage" in str(exc)
    else:
        raise AssertionError(
            "Engram slice beyond parent capacity unexpectedly succeeded"
        )
    return {
        "stages": reports,
        "slice_creation_preserved_parent_values": True,
        "readiness_events_completed": True,
        "disabled_engram_preserved_empty_input": True,
        "parent_capacity_rejected": True,
        "request_history_or_producer_executed": False,
    }


def run_probe(args):
    # This standalone process owns all runtime imports and its single device.
    import torch
    import vllm_ascend.ops  # noqa: F401
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm_ascend.models.deepseek_v41 import model as native
    from vllm_ascend.utils import adapt_patch

    from afd_plugin.model_executor.models.npu.deepseek_v41_stage_state import (
        DeepseekV41StageAttentionStates,
    )
    from afd_plugin.model_executor.models.npu.deepseek_v41_staged import (
        DeepseekV41EngramInputs,
    )

    current_platform.pre_register_and_update()
    load_general_plugins()
    torch.npu.set_device(args.device)
    adapt_patch()
    device = f"npu:{args.device}"
    return {
        "attention": verify_shared_states(
            torch, native, DeepseekV41StageAttentionStates, device
        ),
        "engram": verify_engram(torch, DeepseekV41EngramInputs, device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("--device must be a nonnegative NPU chip index")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "scope": "real NPU stage views, binding recovery and Engram input slicing",
        "device": args.device,
        "success": False,
        "native_model_constructors_called": False,
        "native_model_weights_loaded": False,
        "native_hc_indexer_moe_or_engram_producer_executed": False,
        "distributed_groups_created": False,
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
