# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# ruff: noqa: E402
"""Static storage/payload seams; CPU tensors do not validate NPU graph work."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank
from tests.unit.fakes.hccl_graph_control import make_graph_key
from tests.unit.fakes.hccl_graph_stage import (
    make_graph_stage_bank,
    make_graph_stage_plan,
)


@pytest.mark.parametrize("rows", [(2,), (2, 3)])
@pytest.mark.parametrize("order", ["bulk_v1", "cross_layer_v1"])
def test_bulk_bank_uses_split_packets_only_for_u2_and_preserves_storage(rows, order):
    bank = make_graph_stage_bank(rows=rows, exchange_order=order)
    assert bank.exchange_order == (order if len(rows) == 2 else "stage_serial")
    inputs = tuple(
        torch.full((count, 4), index + 1, dtype=torch.bfloat16)
        for index, count in enumerate(rows)
    )
    for stage, hidden in enumerate(inputs):
        bank.exchanges[stage].endpoint.incoming.append(hidden * 3)
        bank.attention_send(
            stage,
            hidden,
            hidden.float() + 0.00031,
            torch.arange(rows[stage], dtype=torch.int32),
        )
    for stage, hidden in enumerate(inputs):
        output = bank.attention_receive(stage)
        assert output is bank.buffers(stage).output
        torch.testing.assert_close(output, hidden * 3)
    if len(rows) == 2:
        with pytest.raises(
            RuntimeError,
            match="all stage sends" if order == "bulk_v1" else "paired split",
        ):
            bank.attention_exchange(
                0,
                inputs[0],
                inputs[0].float(),
                torch.arange(rows[0], dtype=torch.int32),
            )


@pytest.mark.parametrize("valid", [0, 2, 3])
def test_attention_preserves_router_fp32_and_masks_only_wire_padding_ids(valid):
    bank = make_graph_stage_bank(rows=(3,), valid=(valid,))
    endpoint = bank.exchanges[0].endpoint
    hidden = torch.ones((3, 4), dtype=torch.bfloat16)
    original = hidden.float() + 0.00031
    ids = torch.tensor([7, 9, 11], dtype=torch.int64)
    expected_ids = ids.int()
    expected_ids[valid:] = 0
    expected_output = torch.full_like(hidden, 5)
    expected_output[valid:] = 0
    endpoint.incoming.append(expected_output)

    output = bank.attention_exchange(0, hidden, original, ids)

    assert [message[:2] for message in endpoint.messages] == [
        ("send", (3, 4)),
        ("send", (3, 4)),
        ("send", (3,)),
        ("receive", (3, 4)),
    ]
    assert torch.equal(endpoint.messages[0][2], hidden)
    assert torch.equal(endpoint.messages[1][2], original)
    assert torch.equal(endpoint.messages[2][2], expected_ids)
    assert torch.equal(ids, torch.tensor([7, 9, 11], dtype=torch.int64))
    assert output is bank.buffers(0).output
    assert torch.equal(output, expected_output)


def test_count_refresh_preserves_storage_and_changes_next_tensor_payload():
    bank = make_graph_stage_bank(rows=(2,), valid=(2,))
    endpoint = bank.exchanges[0].endpoint
    buffers = bank.buffers(0)
    tensors = (
        buffers.hidden_states,
        buffers.hidden_states_fp32,
        buffers.input_ids,
        buffers.output,
        *bank.valid_counts,
        *bank.row_indices,
        *bank.padding_masks,
    )
    addresses = tuple(tensor.data_ptr() for tensor in tensors)
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)
    ids = torch.tensor([7, 9])
    for count in (2, 0, 1):
        bank.update_counts(make_graph_stage_plan(bank, valid=(count,)))
        endpoint.incoming.append(torch.zeros_like(hidden))
        bank.attention_exchange(0, hidden, hidden.float() + 0.00031, ids)
    assert [
        message[2].tolist()
        for message in endpoint.messages
        if message[:2] == ("send", (2,))
    ] == [
        [7, 9],
        [0, 0],
        [7, 0],
    ]
    assert tuple(tensor.data_ptr() for tensor in tensors) == addresses


@pytest.mark.parametrize("valid", [0, 2, 3])
def test_ffn_masks_native_output_after_receive_without_mutating_native_result(valid):
    bank = make_graph_stage_bank(role="ffn", rows=(3,), valid=(valid,))
    endpoint = bank.exchanges[0].endpoint
    hidden = torch.ones((3, 4), dtype=torch.bfloat16)
    fp32 = hidden.float() + 0.00031
    ids = torch.tensor([3, 5, 0], dtype=torch.int32)
    endpoint.incoming.extend((hidden, fp32, ids))

    received = bank.receive_inputs(0)
    native_output = torch.full_like(hidden, 7)
    bank.send_output(0, native_output)

    assert torch.equal(received.hidden_states_fp32, fp32)
    assert torch.equal(received.input_ids, ids)
    expected = native_output.clone()
    expected[valid:] = 0
    assert torch.equal(endpoint.messages[-1][2], expected)
    assert torch.equal(native_output, torch.full_like(hidden, 7))
    assert [message[:2] for message in endpoint.messages] == [
        ("receive", (3, 4)),
        ("receive", (3, 4)),
        ("receive", (3,)),
        ("send", (3, 4)),
    ]


def test_prepared_stage_output_survives_native_workspace_reuse_before_send():
    bank = make_graph_stage_bank(role="ffn", rows=(3, 2), valid=(2, 0))
    workspace = torch.full((3, 4), 7, dtype=torch.bfloat16)
    first = bank.prepare_output(0, workspace)
    workspace.fill_(11)
    second = bank.prepare_output(1, workspace[:2])
    assert first is bank.buffers(0).output
    assert second is bank.buffers(1).output
    torch.testing.assert_close(first, torch.full((3, 4), 7, dtype=torch.bfloat16))
    assert all(exchange.endpoint.messages == [] for exchange in bank.exchanges)
    bank.send_output(0, first)
    bank.send_output(1, second)
    expected = torch.full_like(first, 7)
    expected[2:] = 0
    torch.testing.assert_close(bank.exchanges[0].endpoint.messages[-1][2], expected)
    torch.testing.assert_close(
        bank.exchanges[1].endpoint.messages[-1][2], torch.zeros_like(second)
    )


@pytest.mark.parametrize("defect", ["shape", "dtype", "layout", "device"])
@pytest.mark.parametrize("operation", ["prepare_output", "send_output"])
def test_ffn_rejects_bad_native_output_before_copy_or_wire(defect, operation):
    bank = make_graph_stage_bank(role="ffn", rows=(3,), valid=(2,))
    output = torch.ones((3, 4), dtype=torch.bfloat16)
    if defect == "shape":
        output = output[:1]
    elif defect == "dtype":
        output = output.float()
    elif defect == "layout":
        output = output.repeat_interleave(2, dim=1)[:, ::2]
    else:
        output = torch.empty_like(output, device="meta")
    execute = bank.prepare_output if operation == "prepare_output" else bank.send_output
    with pytest.raises(ValueError, match="static BF16 shape/device/layout"):
        execute(0, output)
    assert bank.exchanges[0].endpoint.messages == []
    torch.testing.assert_close(
        bank.buffers(0).output, torch.zeros((3, 4), dtype=torch.bfloat16)
    )


def test_attention_cannot_prepare_ffn_output_storage():
    bank = make_graph_stage_bank(rows=(3,))
    with pytest.raises(RuntimeError, match="requires FFN role"):
        bank.prepare_output(0, torch.ones((3, 4), dtype=torch.bfloat16))
    assert bank.exchanges[0].endpoint.messages == []


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_static_zero_stage_has_no_exchange_and_rejects_accidental_body_call(role):
    bank = make_graph_stage_bank(role=role, rows=(0, 2))
    assert bank.stage_rows == (0, 2)
    assert bank.exchanges[0] is None
    with pytest.raises(ValueError, match="skip zero-row"):
        bank.buffers(0)


@pytest.mark.parametrize("stage_idx", [True, -1, 2, 0.0])
def test_stage_indices_require_explicit_in_range_integers(stage_idx):
    bank = make_graph_stage_bank()
    with pytest.raises(ValueError, match="stage index"):
        bank.buffers(stage_idx)


@pytest.mark.parametrize("field", ["hidden", "fp32", "ids", "noncontiguous"])
def test_attention_rejects_all_bad_sources_before_copy_or_wire(field):
    bank = make_graph_stage_bank(rows=(2,))
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)
    fp32 = hidden.float()
    ids = torch.tensor([3, 5], dtype=torch.int32)
    if field == "hidden":
        hidden = hidden.float()
    elif field == "fp32":
        fp32 = fp32.bfloat16()
    elif field == "ids":
        ids = ids.float()
    else:
        fp32 = torch.ones((4, 2)).t()
    with pytest.raises(ValueError, match="static shape/dtype/device/layout"):
        bank.attention_exchange(0, hidden, fp32, ids)
    assert bank.exchanges[0].endpoint.messages == []
    assert torch.equal(
        bank.buffers(0).hidden_states, torch.zeros((2, 4), dtype=torch.bfloat16)
    )


@pytest.mark.parametrize(
    "change", ["capability", "role_rank", "peer_rank", "local_rank", "overlap"]
)
def test_constructor_rejects_incompatible_session_or_stage_storage(change):
    bank = make_graph_stage_bank()
    capabilities, role_rank, exchanges = (
        bank.capabilities,
        bank.role_rank,
        bank.exchanges,
    )
    if change == "capability":
        capabilities = replace(capabilities, graph_execution_version=0)
    elif change == "role_rank":
        role_rank = True
    elif change == "peer_rank":
        exchanges[0].endpoint.peer_group_rank = 1
    elif change == "local_rank":
        exchanges[0].endpoint.group.rank = lambda: 0
    else:
        exchanges = (exchanges[0], exchanges[0])
    with pytest.raises(ValueError):
        HcclGraphStageBank(
            role="attention",
            role_rank=role_rank,
            capabilities=capabilities,
            key=bank.key,
            exchanges=exchanges,
        )


@pytest.mark.parametrize("change", ["key", "layers", "eager"])
def test_rejected_parent_cannot_partially_update_count_storage(change):
    bank = make_graph_stage_bank(valid=(1, 1))
    plan = make_graph_stage_plan(bank, valid=(2, 0))
    if change == "key":
        decision = replace(
            plan.graph_execution, key=make_graph_key(metadata_shapes=((2,),))
        )
        plan = replace(plan, graph_execution=decision)
    elif change == "layers":
        plan = replace(plan, layer_indices=(0, 2))
    else:
        decision = replace(plan.graph_execution, phase="eager", key=None)
        plan = replace(plan, graph_execution=decision)
    with pytest.raises(ValueError):
        bank.update_counts(plan)
    assert [count.tolist() for count in bank.valid_counts] == [[1], [1]]


@pytest.mark.parametrize("native_tp_size,role_rank", [(2, 0), (2, 1)])
@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_paired_native_tp_binds_distinct_role_coordinates(
    native_tp_size, role_rank, role
):
    bank = make_graph_stage_bank(
        native_tp_size=native_tp_size, role_rank=role_rank, role=role
    )
    endpoint = bank.exchanges[0].endpoint
    assert endpoint.group.rank() == role_rank + (
        native_tp_size if role == "attention" else 0
    )
    assert endpoint.peer_group_rank == role_rank + (
        native_tp_size if role == "ffn" else 0
    )
