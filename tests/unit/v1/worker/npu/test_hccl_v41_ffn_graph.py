# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# ruff: noqa: E402
"""Real FFN compute boundary with CPU MoE/wire seams, not native NPU evidence."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)

from afd_plugin.v1.worker.npu.hccl_v41_ffn_graph import HcclV41FFNGraph
from tests.unit.fakes.hccl import staged_ffn_expected_output
from tests.unit.fakes.hccl_graph_stage import make_graph_stage_plan
from tests.unit.fakes.hccl_v41_ffn_graph import (
    make_ffn_graph_case,
    queue_ffn_graph_inputs,
)


@pytest.mark.parametrize(
    "rows,valid", [((3,), (2,)), ((3, 2), (2, 0)), ((0, 2), (0, 1))]
)
@pytest.mark.parametrize("native_tp_size,role_rank", [(1, 0), (2, 0), (2, 1)])
def test_layer_major_graph_body_sends_native_complete_moe_once_with_padding_zeroed(
    rows,
    valid,
    native_tp_size,
    role_rank,
):
    case = make_ffn_graph_case(
        rows=rows, valid=valid, native_tp_size=native_tp_size, role_rank=role_rank
    )
    payloads = queue_ffn_graph_inputs(case)
    assert case.driver.run() is None

    assert case.events == [
        (kind, stage, shape)
        for layer in case.bank.layer_indices
        for stage, count in enumerate(rows)
        if count
        for kind, shape in (
            ("receive", (count, 4)),
            ("receive", (count, 4)),
            ("receive", (count,)),
            ("send", (count, 4)),
        )
    ]
    for stage, count in enumerate(rows):
        if not count:
            continue
        outputs = [
            message[2]
            for message in case.bank.exchanges[stage].endpoint.messages
            if message[0] == "send"
        ]
        for layer, output in zip(case.bank.layer_indices, outputs, strict=True):
            expected = staged_ffn_expected_output(payloads[layer, stage], layer)
            expected[valid[stage] :] = 0
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_new_parent_counts_and_payloads_refresh_without_rebuilding_graph_storage():
    case = make_ffn_graph_case(rows=(3,), valid=(2,))
    storage = case.bank.buffers(0)
    for valid, offset in ((2, 0), (0, 10), (3, 20)):
        case.bank.update_counts(make_graph_stage_plan(case.bank, valid=(valid,)))
        payloads = queue_ffn_graph_inputs(case, offset=offset)
        case.driver.run()
        output = case.bank.exchanges[0].endpoint.messages[-1][2]
        expected = staged_ffn_expected_output(payloads[1, 0], 1)
        expected[valid:] = 0
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert case.bank.buffers(0) is storage


@pytest.mark.parametrize(
    "field,value",
    [
        ("data_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("use_sequence_parallel_moe", True),
        ("tensor_parallel_size", 2),
        ("rank", 1),
    ],
)
def test_ffn_graph_rejects_invalid_native_topology_before_any_wire(field, value):
    case = make_ffn_graph_case()
    setattr(case.parallel, field, value)
    with pytest.raises(ValueError, match="paired native TP"):
        HcclV41FFNGraph(case.model, case.contexts, case.bank, case.parallel)
    assert case.events == []


@pytest.mark.parametrize("change", ["mode", "stage", "ids", "count", "layers"])
def test_ffn_graph_rejects_context_or_layer_ownership_before_any_wire(change):
    case = make_ffn_graph_case()
    contexts = case.contexts
    if change == "mode":
        contexts[0].cudagraph_runtime_mode = CUDAGraphMode.NONE
    elif change == "stage":
        contexts[0].ubatch_idx = 1
    elif change == "ids":
        contexts[0].input_ids = contexts[0].input_ids.clone()
    elif change == "count":
        contexts = contexts[:1]
    else:
        case.model.model.end_layer = 1
    with pytest.raises(ValueError, match="static layer/stage layout|FULL contexts"):
        HcclV41FFNGraph(case.model, contexts, case.bank, case.parallel)
    assert case.events == []


def test_native_moe_error_restores_callers_context_and_sends_no_output():
    case = make_ffn_graph_case()
    queue_ffn_graph_inputs(case)
    case.model.model.layers[0].mlp.fail = True
    outer = ForwardContext({}, {}, {})
    with override_forward_context(outer):
        with pytest.raises(RuntimeError, match="controlled graph MoE failure"):
            case.driver.run()
        assert get_forward_context() is outer
    assert case.events == [
        ("receive", 0, (3, 4)),
        ("receive", 0, (3, 4)),
        ("receive", 0, (3,)),
    ]
