# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow the CPU dependency gates.
# ruff: noqa: E402
"""Connector seam behavior over CPU queues; these tests are not NPU evidence."""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu.hccl_protocol import HcclForwardPlan
from afd_plugin.connectors.npu.p2p_hccl import HCCLTransferState, P2pHcclAFDConnector
from tests.unit.fakes.hccl import (
    HcclEndpointRegistry,
    hccl_layer_tensors,
    make_hccl_connector_pair,
)


@pytest.fixture
def pair(monkeypatch):
    connectors = make_hccl_connector_pair(monkeypatch)
    yield connectors
    for connector in connectors:
        connector.close()


@pytest.fixture
def active_pair(pair):
    attention, ffn = pair
    plan = HcclForwardPlan("existing-forward", (0, 1), 2, 3)
    attention.begin_forward_plan(plan)
    assert ffn.receive_forward_plan() == plan
    return attention, ffn, plan


@pytest.fixture
def layer_inputs():
    return hccl_layer_tensors()


@pytest.fixture
def context():
    return AFDTransferContext(AFDTransferMetadata(0, 0, [3]))


def test_paired_forward_preserves_fp32_ids_and_output(active_pair, layer_inputs):
    attention, ffn, plan = active_pair
    hidden, fp32, ids = layer_inputs
    for layer_idx in plan.layer_indices:
        context = AFDTransferContext(AFDTransferMetadata(layer_idx, 0, [3]))
        attention.send_attn_output(
            hidden, context, hidden_states_fp32=fp32, input_ids=ids
        )
        payload = ffn.recv_attn_output(layer_idx=layer_idx)
        assert torch.equal(payload.hidden_states, hidden)
        assert torch.equal(payload.hidden_states_fp32, fp32)
        assert torch.equal(payload.input_ids, ids)
        assert payload.context.states == HCCLTransferState(
            plan.transaction_id, layer_idx
        )
        output = payload.hidden_states + payload.hidden_states_fp32.to(torch.bfloat16)
        ffn.send_ffn_output(output, payload.context)
        ref = torch.empty_like(hidden)
        assert attention.recv_ffn_output(ref, context=context) is ref
        assert torch.equal(ref, output)
    assert attention.active_forward_plan is None
    assert ffn.active_forward_plan is None
    attention.request_stop()
    assert ffn.receive_forward_plan() is None
    assert not attention.is_initialized
    assert not ffn.is_initialized


@pytest.mark.parametrize("missing", ["hidden_states_fp32", "input_ids"])
def test_attention_requires_complete_input_contract(
    active_pair, layer_inputs, context, missing
):
    attention, _, _ = active_pair
    hidden, fp32, ids = layer_inputs
    kwargs = {"hidden_states_fp32": fp32, "input_ids": ids}
    del kwargs[missing]
    with pytest.raises(ValueError, match=missing):
        attention.send_attn_output(hidden, context, **kwargs)


@pytest.mark.parametrize("field,value", [("stage_idx", 1), ("seq_lens", [2])])
def test_attention_rejects_context_layout_mismatch(
    active_pair, layer_inputs, context, field, value
):
    attention, _, _ = active_pair
    hidden, fp32, ids = layer_inputs
    setattr(context.metadata, field, value)
    with pytest.raises(ValueError, match="stage 0.*padded token"):
        attention.send_attn_output(
            hidden, context, hidden_states_fp32=fp32, input_ids=ids
        )


def test_attention_rejects_state_from_another_transaction(
    active_pair, layer_inputs, context
):
    attention, _, _ = active_pair
    hidden, fp32, ids = layer_inputs
    context.states = HCCLTransferState("old-forward", 0)
    with pytest.raises(ValueError, match="transaction and layer"):
        attention.send_attn_output(
            hidden, context, hidden_states_fp32=fp32, input_ids=ids
        )


@pytest.mark.parametrize("layer_idx", [1, True, None])
def test_ffn_receive_requires_the_next_target_layer(active_pair, layer_idx):
    _, ffn, _ = active_pair
    with pytest.raises((ValueError, RuntimeError), match="layer_idx|active HCCL"):
        ffn.recv_attn_output(layer_idx=layer_idx)


def test_each_send_requires_the_previous_output_to_be_consumed(
    active_pair, layer_inputs, context
):
    attention, ffn, plan = active_pair
    hidden, fp32, ids = layer_inputs
    attention.send_attn_output(hidden, context, hidden_states_fp32=fp32, input_ids=ids)
    with pytest.raises(RuntimeError, match="consume the previous FFN output"):
        attention.send_attn_output(
            hidden, context, hidden_states_fp32=fp32, input_ids=ids
        )
    with pytest.raises(RuntimeError, match="not been consumed"):
        attention.begin_forward_plan(replace(plan, transaction_id="another"))
    with pytest.raises(RuntimeError, match="idle transaction"):
        attention.request_stop()
    payload = ffn.recv_attn_output(layer_idx=0)
    with pytest.raises(RuntimeError, match="send the previous FFN output"):
        ffn.recv_attn_output(layer_idx=1)
    ffn.send_ffn_output(hidden, payload.context)
    attention.recv_ffn_output(torch.empty_like(hidden), context=context)
    with pytest.raises(RuntimeError, match="no HCCL FFN output"):
        attention.recv_ffn_output(torch.empty_like(hidden), context=context)


def test_output_exchange_requires_the_original_context(
    active_pair, layer_inputs, context
):
    attention, ffn, _ = active_pair
    hidden, fp32, ids = layer_inputs
    attention.send_attn_output(hidden, context, hidden_states_fp32=fp32, input_ids=ids)
    payload = ffn.recv_attn_output(layer_idx=0)
    with pytest.raises(ValueError, match="original transfer context"):
        ffn.send_ffn_output(hidden, replace(payload.context))
    ffn.send_ffn_output(hidden, payload.context)
    with pytest.raises(ValueError, match="original transfer context"):
        attention.recv_ffn_output(torch.empty_like(hidden), context=replace(context))
    assert torch.equal(
        attention.recv_ffn_output(torch.empty_like(hidden), context=context), hidden
    )


@pytest.mark.parametrize("layout", ["shape", "dtype", "ubatch"])
def test_output_buffer_layout_must_match_the_exchange(
    active_pair, layer_inputs, context, layout
):
    attention, ffn, _ = active_pair
    hidden, fp32, ids = layer_inputs
    attention.send_attn_output(hidden, context, hidden_states_fp32=fp32, input_ids=ids)
    payload = ffn.recv_attn_output(layer_idx=0)
    ffn.send_ffn_output(hidden, payload.context)
    ref = torch.empty_like(hidden)
    ubatch_idx = 0
    if layout == "shape":
        ref = torch.empty((2, 16), dtype=torch.bfloat16)
    elif layout == "dtype":
        ref = ref.float()
    else:
        ubatch_idx = 1
    with pytest.raises(ValueError, match="layout|ubatch_idx"):
        attention.recv_ffn_output(ref, ubatch_idx, context=context)
    assert torch.equal(
        attention.recv_ffn_output(torch.empty_like(hidden), context=context), hidden
    )


def test_empty_plan_completes_before_the_next_transaction(pair):
    attention, ffn = pair
    empty = HcclForwardPlan("empty", (0, 1), 0, 0)
    attention.begin_forward_plan(empty)
    assert ffn.receive_forward_plan() == empty
    assert attention.active_forward_plan is None
    assert ffn.active_forward_plan is None
    with pytest.raises(RuntimeError, match="active HCCL target plan"):
        ffn.recv_attn_output(layer_idx=0)
    following = HcclForwardPlan("next", (0, 1), 2, 3)
    attention.begin_forward_plan(following)
    assert ffn.receive_forward_plan() == following


def test_context_cannot_be_modified_between_receive_and_reply(
    active_pair, layer_inputs, context
):
    attention, ffn, _ = active_pair
    hidden, fp32, ids = layer_inputs
    attention.send_attn_output(hidden, context, hidden_states_fp32=fp32, input_ids=ids)
    payload = ffn.recv_attn_output(layer_idx=0)
    payload.context.states = HCCLTransferState("wrong", 0)
    with pytest.raises(ValueError, match="transaction and layer"):
        ffn.send_ffn_output(hidden, payload.context)


def test_close_is_idempotent_and_disables_the_session(pair):
    attention, ffn = pair
    for connector in (attention, ffn):
        connector.close()
        connector.close()
        assert not connector.is_initialized
    with pytest.raises(RuntimeError, match="not initialized"):
        attention.begin_forward_plan(HcclForwardPlan("closed", (0, 1), 2, 3))


@pytest.mark.parametrize("rank", [-1, 1, True])
def test_connector_requires_matching_native_caller_and_role_rank(pair, rank):
    attention, _ = pair
    with pytest.raises(ValueError, match="caller rank equal"):
        P2pHcclAFDConnector(rank, 0, attention.vllm_config, attention.afd_config, 0)


@pytest.fixture
def tp_pairs(monkeypatch):
    wire = HcclEndpointRegistry()
    pairs = tuple(
        make_hccl_connector_pair(
            monkeypatch, tensor_parallel_size=2, role_rank=rank, wire=wire
        )
        for rank in range(2)
    )
    yield pairs
    for pair in pairs:
        for connector in pair:
            connector.close()


def test_tp_rank_pairs_exchange_only_with_their_matching_endpoint(
    tp_pairs, layer_inputs
):
    hidden, fp32, ids = layer_inputs
    plans = [HcclForwardPlan(f"rank-{rank}", (0, 1), 2, 3) for rank in range(2)]
    for rank, (attention, _) in enumerate(tp_pairs):
        attention.begin_forward_plan(plans[rank])
    for rank in reversed(range(2)):
        assert tp_pairs[rank][1].receive_forward_plan() == plans[rank]

    for layer_idx in (0, 1):
        contexts = []
        for rank, (attention, _) in enumerate(tp_pairs):
            context = AFDTransferContext(AFDTransferMetadata(layer_idx, 0, [3]))
            contexts.append(context)
            attention.send_attn_output(
                hidden + rank, context, hidden_states_fp32=fp32, input_ids=ids
            )
        for rank in reversed(range(2)):
            ffn = tp_pairs[rank][1]
            payload = ffn.recv_attn_output(layer_idx=layer_idx)
            torch.testing.assert_close(payload.hidden_states, hidden + rank)
            ffn.send_ffn_output(payload.hidden_states + layer_idx, payload.context)
        for rank, (attention, _) in enumerate(tp_pairs):
            output = attention.recv_ffn_output(
                torch.empty_like(hidden), context=contexts[rank]
            )
            torch.testing.assert_close(output, hidden + rank + layer_idx)
    for attention, ffn in tp_pairs:
        attention.request_stop()
        assert ffn.receive_forward_plan() is None
