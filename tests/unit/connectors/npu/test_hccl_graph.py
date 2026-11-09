# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Static tensor and protocol seams; actual graph behavior needs the NPU probe."""

from dataclasses import replace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")

import torch

from afd_plugin.connectors.npu import hccl_graph
from afd_plugin.connectors.npu.hccl_graph import HcclGraphBuffers, HcclGraphEndpoint
from tests.unit.fakes.hccl_graph import graph_process_group, make_graph_exchange


@pytest.fixture
def attention(monkeypatch):
    return make_graph_exchange(monkeypatch, hccl_graph)


def test_attention_sends_original_payloads_with_group_relative_peer(attention):
    exchange, recorder = attention
    hidden = torch.ones((3, 2), dtype=torch.bfloat16)
    fp32 = hidden.float() + 0.00031
    ids = torch.tensor([3, 8, 19], dtype=torch.int32)
    exchange.update_inputs(hidden, fp32, ids)
    exchange.send_inputs()
    assert [message[:7] for message in recorder.messages] == [
        ("send", None, (10, 20), "graph-test", 0, 1, shape)
        for shape in ((3, 2), (3, 2), (3,))
    ]
    for message, expected in zip(recorder.messages, (hidden, fp32, ids), strict=True):
        assert torch.equal(message[7], expected)


def test_endpoint_uses_its_independent_process_group_rank(monkeypatch):
    # Native default-world translation reports zero, but this AFD domain's
    # actual local coordinate is one. Peer zero remains a valid remote peer.
    group = graph_process_group(monkeypatch, hccl_graph, ranks=(0, 1), group_rank=1)
    endpoint = HcclGraphEndpoint.from_process_group(group, 0)
    assert endpoint.peer_group_rank == 0
    with pytest.raises(ValueError, match="different peer"):
        HcclGraphEndpoint.from_process_group(group, 1)


def test_received_output_updates_the_existing_buffer(attention):
    exchange, recorder = attention
    output = exchange.buffers.output
    recorder.incoming.extend((torch.full_like(output, 7), torch.full_like(output, 11)))
    exchange.receive_output()
    assert torch.equal(output, torch.full_like(output, 7))
    exchange.receive_output()
    assert torch.equal(output, torch.full_like(output, 11))
    assert all(
        message[:7] == ("receive", None, (10, 20), "graph-test", 0, 1, (3, 2))
        for message in recorder.messages
    )


def test_ffn_receives_all_payloads_and_sends_native_output(monkeypatch):
    exchange, recorder = make_graph_exchange(monkeypatch, hccl_graph, role="ffn")
    hidden = torch.full((3, 2), 2, dtype=torch.bfloat16)
    fp32 = hidden.float() + 0.00031
    ids = torch.tensor([4, 0, 2], dtype=torch.int32)
    recorder.incoming.extend((hidden, fp32, ids))
    exchange.receive_inputs()
    assert torch.equal(exchange.buffers.hidden_states, hidden)
    assert torch.equal(exchange.buffers.hidden_states_fp32, fp32)
    assert torch.equal(exchange.buffers.input_ids, ids)
    expected = torch.full_like(hidden, 27)
    exchange.buffers.output.copy_(expected)
    exchange.send_output()
    assert [message[0] for message in recorder.messages] == [
        "receive",
        "receive",
        "receive",
        "send",
    ]
    assert torch.equal(recorder.messages[-1][7], expected)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "layout"])
def test_invalid_input_update_preserves_all_previous_inputs(attention, invalid):
    exchange, recorder = attention
    hidden = torch.ones((3, 2), dtype=torch.bfloat16)
    fp32 = hidden.float()
    ids = torch.tensor([1, 2, 3], dtype=torch.int32)
    exchange.update_inputs(hidden, fp32, ids)
    replacement = {
        "shape": torch.ones((2, 2), dtype=torch.float32),
        "dtype": torch.ones((3, 2), dtype=torch.bfloat16),
        "layout": torch.ones((2, 3), dtype=torch.float32).t(),
    }[invalid]
    with pytest.raises(ValueError, match="static shape, dtype, device and layout"):
        exchange.update_inputs(hidden + 4, replacement, ids + 4)
    exchange.send_inputs()
    for message, expected in zip(recorder.messages, (hidden, fp32, ids), strict=True):
        assert torch.equal(message[7], expected)


@pytest.mark.parametrize(
    "role,method",
    [
        ("ffn", "send_inputs"),
        ("ffn", "receive_output"),
        ("attention", "receive_inputs"),
        ("attention", "send_output"),
    ],
)
def test_wrong_role_rejects_before_sending_a_payload(monkeypatch, role, method):
    exchange, recorder = make_graph_exchange(monkeypatch, hccl_graph, role=role)
    with pytest.raises(RuntimeError, match="requires"):
        getattr(exchange, method)()
    assert recorder.messages == []


@pytest.mark.parametrize("peer", [True, -1, 99, 10])
def test_endpoint_rejects_invalid_peer_before_capture(monkeypatch, peer):
    group = graph_process_group(monkeypatch, hccl_graph)
    with pytest.raises(ValueError, match="graph"):
        HcclGraphEndpoint.from_process_group(group, peer)


def test_endpoint_rejects_non_hccl_group(monkeypatch):
    group = graph_process_group(monkeypatch, hccl_graph, backend="gloo")
    with pytest.raises(ValueError, match="HCCL"):
        HcclGraphEndpoint.from_process_group(group, 20)


@pytest.mark.parametrize(
    "rows,hidden", [(0, 2), (-1, 2), (True, 2), (2, 0), (2, False)]
)
def test_allocation_rejects_empty_or_invalid_dimensions(rows, hidden):
    with pytest.raises(ValueError, match="positive"):
        HcclGraphBuffers.allocate(rows, hidden, torch.device("cpu"))


def test_allocation_requires_an_indexed_npu_device():
    with pytest.raises(ValueError, match="indexed NPU"):
        HcclGraphBuffers.allocate(3, 2, torch.device("cpu"))


def test_malformed_output_buffer_rejects_before_any_payload(attention):
    exchange, recorder = attention
    buffers = replace(exchange.buffers, output=torch.zeros((3, 2), dtype=torch.float32))
    with pytest.raises(ValueError, match="static shape, dtype, device and layout"):
        hccl_graph.HcclGraphExchange(exchange.endpoint, buffers, role="attention")
    assert recorder.messages == []


def test_overlapping_payload_buffers_reject_before_any_payload(attention):
    exchange, recorder = attention
    buffers = replace(exchange.buffers, output=exchange.buffers.hidden_states)
    with pytest.raises(ValueError, match="must not overlap"):
        hccl_graph.HcclGraphExchange(exchange.endpoint, buffers, role="attention")
    assert recorder.messages == []
