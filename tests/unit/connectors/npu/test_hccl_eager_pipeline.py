# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Actual CPU tensor/protocol seams; no NPU progress or overlap acceptance."""

import gc
import weakref

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")

import torch

from afd_plugin.connectors.npu import hccl_eager_pipeline as owner_module
from afd_plugin.connectors.npu import hccl_staged_transport as transport_module
from afd_plugin.connectors.npu.hccl_eager_pipeline import HcclAttentionTicket
from afd_plugin.distributed.hccl_graph import HcclGraphKey, HcclGraphStage
from tests.unit.fakes.hccl_eager_pipeline import (
    EagerHostGroup,
    eager_payload,
    make_eager_pair,
    open_eager_pair,
    run_eager_layer,
)
from tests.unit.fakes.npu_stream import HostStream


@pytest.fixture
def case(monkeypatch):
    return make_eager_pair(monkeypatch, owner_module, transport_module)


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize(
    "rows,valid",
    [((2, 3), (2, 3)), ((0, 3), (0, 1)), ((2, 0), (0, 0)), ((0, 0), (0, 0))],
)
def test_two_stage_parent_returns_independent_fp32_and_raw_ids_without_padding_rewrite(
    monkeypatch, tp, rows, valid
):
    case = make_eager_pair(
        monkeypatch, owner_module, transport_module, rows=rows, valid=valid, tp=tp
    )
    attention, ffn = open_eager_pair(case)
    for layer in case.plan.layer_indices:
        outputs, expected = run_eager_layer(case, attention, ffn, layer)
        for output, result in zip(outputs, expected, strict=True):
            torch.testing.assert_close(output, result)
    attention.finish()
    ffn.finish()
    assert attention.finished and ffn.finished
    assert case.attention.plan is case.ffn.plan is None
    assert case.attention.eager_pipeline is case.ffn.eager_pipeline is None
    assert [
        message.kind
        for direction, _, _, message in case.attention.transcript
        if direction == "send"
    ][-1] == "COMPLETE"


def test_both_attention_stages_are_posted_before_any_output_wait(case):
    attention, _ = open_eager_pair(case)
    for stage in case.plan.stages:
        attention.post_attention(
            0,
            stage.stage_idx,
            *eager_payload(stage.attention_padded_tokens[0]),
            producer_stream=case.runtime.parent,
        )
    assert attention.occupied_stages == (0, 1)
    tensor_ops = [
        entry[0]
        for entry in case.backend.trace
        if entry[0] in ("isend", "irecv", "work_wait")
    ]
    assert tensor_ops == ["isend", "isend", "isend", "irecv"] * 2


def test_ffn_loan_preserves_bf16_original_fp32_and_raw_padding_ids(case):
    attention, ffn = open_eager_pair(case)
    payload = eager_payload(2)
    attention.post_attention(0, 0, *payload, producer_stream=case.runtime.parent)
    ticket = ffn.post_ffn_receive(0, 0)
    with ffn.ffn_inputs(ticket, consumer_stream=case.runtime.compute) as inputs:
        torch.testing.assert_close(inputs.hidden_states, payload[0])
        torch.testing.assert_close(inputs.hidden_states_fp32, payload[1])
        torch.testing.assert_close(inputs.input_ids, payload[2])
        assert inputs.hidden_states_fp32.dtype == torch.float32
        assert not torch.equal(inputs.hidden_states_fp32, inputs.hidden_states.float())


def test_stage_reuse_waits_old_native_consumer_and_works_before_next_posts(case):
    attention, ffn = open_eager_pair(case)
    run_eager_layer(case, attention, ffn, 0)
    start = len(case.runtime.trace)
    attention.post_attention(
        1, 0, *eager_payload(2), producer_stream=case.runtime.parent
    )
    trace = case.runtime.trace[start:]
    assert ("wait_event", "parent") in trace
    assert ("record_event", "parent") in trace
    assert ("wait_event", "send") in trace
    assert ("wait_event", "receive") in trace
    assert attention.occupied_stages == (0, 1)


@pytest.mark.parametrize("kind", ["forged", "foreign", "stale", "duplicate"])
def test_output_loan_rejects_unissued_or_consumed_ticket(case, kind):
    attention, ffn = open_eager_pair(case)
    ticket = attention.post_attention(
        0, 0, *eager_payload(2), producer_stream=case.runtime.parent
    )
    if kind == "forged":
        rejected = HcclAttentionTicket(ticket.layer_idx, ticket.stage_idx)
    elif kind == "foreign":
        rejected = ffn.post_ffn_receive(0, 0)
    else:
        received = ffn.post_ffn_receive(0, 0)
        with ffn.ffn_inputs(received, consumer_stream=case.runtime.compute) as inputs:
            output = inputs.hidden_states.clone()
        ffn.post_ffn_output(received, output, producer_stream=case.runtime.compute)
        with attention.attention_output(ticket, consumer_stream=case.runtime.compute):
            pass
        rejected = ticket
        if kind == "stale":
            attention.post_attention(
                0, 1, *eager_payload(3), producer_stream=case.runtime.parent
            )
            attention.post_attention(
                1, 0, *eager_payload(2), producer_stream=case.runtime.parent
            )
    with (
        pytest.raises((ValueError, RuntimeError), match="forged|foreign|stale|once"),
        attention.attention_output(rejected, consumer_stream=case.runtime.compute),
    ):
        pytest.fail("invalid ticket yielded an output")


@pytest.mark.parametrize("step", [(0, 1), (1, 0), (True, 0), (0, False)])
def test_post_rejects_wrong_layer_stage_without_network_traffic(case, step):
    attention, _ = open_eager_pair(case)
    before = len(case.backend.trace)
    with pytest.raises(ValueError, match="posting"):
        attention.post_attention(
            *step, *eager_payload(2), producer_stream=case.runtime.parent
        )
    assert len(case.backend.trace) == before


@pytest.mark.parametrize(
    "operation",
    ["serial_send", "serial_finish", "graph_finish", "close", "second_open", "stop"],
)
def test_active_ticket_owner_excludes_other_parent_lifecycles(case, operation):
    open_eager_pair(case)
    peer = case.attention
    key = HcclGraphKey(2, 2, 1, 1, 0, 1, (HcclGraphStage(0, 2, 0, 2),))
    operations = {
        "serial_send": lambda: peer.send_layer_inputs(0, 0, *eager_payload(2)),
        "serial_finish": peer.finish_forward,
        "graph_finish": lambda: peer.complete_graph_forward(key),
        "close": peer.close,
        "second_open": lambda: peer.open_eager_pipeline(
            parent_stream=case.runtime.parent,
            send_stream=case.runtime.send,
            receive_stream=case.runtime.receive,
        ),
        "stop": peer.stop,
    }
    with pytest.raises(RuntimeError, match="exclusively|already active|idle"):
        operations[operation]()
    assert peer.plan == case.plan


@pytest.mark.parametrize("kind", ["missing_outputs", "open_loan", "ffn_output_missing"])
def test_finish_requires_every_native_consumer_and_output(case, kind):
    attention, ffn = open_eager_pair(case)
    if kind == "missing_outputs":
        owner = attention
    else:
        attention.post_attention(
            0, 0, *eager_payload(2), producer_stream=case.runtime.parent
        )
        ticket = ffn.post_ffn_receive(0, 0)
        with ffn.ffn_inputs(ticket, consumer_stream=case.runtime.compute):
            if kind == "open_loan":
                with pytest.raises(RuntimeError, match="actual parent|finish"):
                    ffn.finish()
                return
        owner = ffn
    with pytest.raises(RuntimeError, match="finish"):
        owner.finish()
    assert case.ffn.plan == case.plan


def test_work_failure_is_terminal_and_retains_owner_for_external_supervision(case):
    attention, ffn = open_eager_pair(case)
    attention.post_attention(
        0, 0, *eager_payload(2), producer_stream=case.runtime.parent
    )
    ticket = ffn.post_ffn_receive(0, 0)
    case.backend.fail_wait = True
    with (
        pytest.raises(RuntimeError, match="Work failure"),
        ffn.ffn_inputs(ticket, consumer_stream=case.runtime.compute),
    ):
        pytest.fail("failed Work yielded tensors")
    assert ffn.failed and case.ffn.terminal
    assert case.ffn.eager_pipeline is ffn
    assert ffn.occupied_stages == (0,)
    with pytest.raises(RuntimeError, match="exclusively"):
        case.ffn.close()


def test_terminal_owner_keeps_original_inputs_and_continuation_alive(case):
    attention, _ = open_eager_pair(case)
    payload = eager_payload(2)
    continuation = payload[1].clone()
    refs = tuple(weakref.ref(tensor) for tensor in (*payload, continuation))
    attention.post_attention(
        0, 0, *payload, producer_stream=case.runtime.parent, retain=(continuation,)
    )
    del payload, continuation
    attention.fail()
    gc.collect()
    assert all(reference() is not None for reference in refs)
    assert case.attention.eager_pipeline is attention


def test_ffn_output_cannot_be_sent_twice_or_before_native_input_loan(case):
    attention, ffn = open_eager_pair(case)
    attention.post_attention(
        0, 0, *eager_payload(2), producer_stream=case.runtime.parent
    )
    ticket = ffn.post_ffn_receive(0, 0)
    output = eager_payload(2)[0]
    with pytest.raises(RuntimeError, match="closed input loan"):
        ffn.post_ffn_output(ticket, output, producer_stream=case.runtime.compute)
    with ffn.ffn_inputs(ticket, consumer_stream=case.runtime.compute) as inputs:
        output = inputs.hidden_states.clone()
    ffn.post_ffn_output(ticket, output, producer_stream=case.runtime.compute)
    with pytest.raises(RuntimeError, match="one send"):
        ffn.post_ffn_output(ticket, output, producer_stream=case.runtime.compute)


def test_consumer_exception_preserves_original_error_and_stream_context(case):
    attention, ffn = open_eager_pair(case)
    attention.post_attention(
        0, 0, *eager_payload(2), producer_stream=case.runtime.parent
    )
    ticket = ffn.post_ffn_receive(0, 0)
    with (
        pytest.raises(ValueError, match="native MoE failed"),
        ffn.ffn_inputs(ticket, consumer_stream=case.runtime.compute),
    ):
        raise ValueError("native MoE failed")
    assert case.runtime.current == case.runtime.parent
    assert ffn.failed and case.ffn.eager_pipeline is ffn


@pytest.mark.parametrize("kind", ["shape", "dtype", "ids_range", "ids_padding"])
def test_attention_validation_preserves_shape_precision_and_ids(monkeypatch, kind):
    case = make_eager_pair(
        monkeypatch,
        owner_module,
        transport_module,
        valid=(0, 3) if kind == "ids_padding" else None,
    )
    attention, _ = open_eager_pair(case)
    payload = list(eager_payload(2))
    if kind == "shape":
        payload[0] = payload[0][:1]
    elif kind == "dtype":
        payload[1] = payload[1].to(torch.bfloat16)
    elif kind == "ids_range":
        payload[2][0] = case.caps.vocab_size
    with pytest.raises(ValueError, match="expected contiguous|vocabulary|padding"):
        attention.post_attention(0, 0, *payload, producer_stream=case.runtime.parent)
    assert not any(entry[0] == "isend" for entry in case.backend.trace)


@pytest.mark.parametrize(
    "kind", ["not_warmed", "duplicate_warmup", "active_plan", "foreign_stream"]
)
def test_warmup_gate_is_explicit_and_only_idle_once(case, kind):
    peer = case.attention
    if kind == "not_warmed":
        peer.begin_forward(case.plan)
        with pytest.raises(RuntimeError, match="warmup"):
            peer.open_eager_pipeline(
                parent_stream=case.runtime.parent,
                send_stream=case.runtime.send,
                receive_stream=case.runtime.receive,
            )
        return
    if kind == "duplicate_warmup":
        peer.warmup_eager_pipeline(parent_stream=case.runtime.parent)
    elif kind == "active_plan":
        peer.begin_forward(case.plan)
    stream = (
        HostStream("foreign", torch.device("cpu"))
        if kind == "foreign_stream"
        else case.runtime.parent
    )
    with pytest.raises((RuntimeError, ValueError), match="idle|actual"):
        peer.warmup_eager_pipeline(parent_stream=stream)


def test_warmup_uses_all_stage_groups_and_ends_with_control_barrier(case):
    case.attention.warmup_eager_pipeline(parent_stream=case.runtime.parent)
    assert case.attention.eager_pipeline_warmed
    assert [entry[0] for entry in case.backend.trace] == [
        "warm_send",
        "warm_send",
        "warm_send",
        "warm_recv",
    ] * 2 + ["warm_barrier"]
    assert {entry[3] for entry in case.backend.trace if entry[0] != "warm_barrier"} == {
        0,
        1,
    }


def test_failed_warmup_never_marks_warm_or_allows_open(case):
    case.backend.fail_warmup = True
    with pytest.raises(RuntimeError, match="warmup failed"):
        case.ffn.warmup_eager_pipeline(parent_stream=case.runtime.parent)
    assert case.ffn.terminal and not case.ffn.eager_pipeline_warmed
    with pytest.raises(RuntimeError, match="retains its groups"):
        case.ffn.close()
    with pytest.raises(RuntimeError, match="not active"):
        case.ffn.open_eager_pipeline(
            parent_stream=case.runtime.parent,
            send_stream=case.runtime.send,
            receive_stream=case.runtime.receive,
        )


def test_warmup_does_not_authorize_a_replacement_stage_group(case):
    case.attention.warmup_eager_pipeline(parent_stream=case.runtime.parent)
    case.attention.begin_forward(case.plan)
    case.attention.data_groups[0] = EagerHostGroup(0, case.attention.world_rank)
    with pytest.raises(RuntimeError, match="exact matched-warm"):
        case.attention.open_eager_pipeline(
            parent_stream=case.runtime.parent,
            send_stream=case.runtime.send,
            receive_stream=case.runtime.receive,
        )


def test_eager_owner_rejects_foreign_current_stream_before_posting(case):
    attention, _ = open_eager_pair(case)
    with (
        case.runtime.stream(case.runtime.compute),
        pytest.raises(RuntimeError, match="actual parent"),
    ):
        attention.post_attention(
            0, 0, *eager_payload(2), producer_stream=case.runtime.compute
        )
    assert attention.occupied_stages == ()


@pytest.mark.parametrize("mode", ["profile", "dummy", "warmup"])
def test_only_actual_target_parent_may_open_eager_owner(monkeypatch, mode):
    case = make_eager_pair(monkeypatch, owner_module, transport_module, mode=mode)
    with pytest.raises(RuntimeError, match="target U2"):
        open_eager_pair(case)
    assert case.attention.eager_pipeline is None


def test_native_u1_parent_keeps_existing_serial_lifecycle(monkeypatch):
    case = make_eager_pair(monkeypatch, owner_module, transport_module, rows=(2,))
    with pytest.raises(RuntimeError, match="target U2"):
        open_eager_pair(case)
    assert case.attention.plan == case.plan


def test_finished_owner_cannot_terminate_a_later_parent(case):
    attention, ffn = open_eager_pair(case)
    for layer in case.plan.layer_indices:
        run_eager_layer(case, attention, ffn, layer)
    attention.finish()
    ffn.finish()
    with pytest.raises(RuntimeError, match="stale eager owner"):
        attention.fail()
    assert not case.attention.terminal and case.attention.plan is None


def test_backend_gate_does_not_authorize_other_native_tp_sizes(monkeypatch):
    case = make_eager_pair(monkeypatch, owner_module, transport_module, tp=3)
    with pytest.raises(ValueError, match="paired native TP1/TP2"):
        case.attention.warmup_eager_pipeline(parent_stream=case.runtime.parent)
    assert case.backend.trace == []
