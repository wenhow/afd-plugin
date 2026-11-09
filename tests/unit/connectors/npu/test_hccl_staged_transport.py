# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Control/lifecycle seams only; tensors still need the real NPU probe."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")

from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_graph import HcclGraphReport
from afd_plugin.distributed.hccl_plan import (
    HcclLocalForward,
    HcclParentPlan,
    HcclPlanCapabilities,
    HcclPlanMessage,
)
from tests.unit.fakes.hccl import ScriptedStagedControlPeer
from tests.unit.fakes.hccl_graph_control import (
    make_graph_control_round,
    make_graph_key,
)


@pytest.fixture
def graph_round(monkeypatch):
    monkeypatch.setattr("torch.npu.synchronize", lambda device: None)
    return make_graph_control_round()


def test_graph_reservation_commits_every_ack_before_native_counts(graph_round):
    case = graph_round
    peer = ScriptedStagedControlPeer("attention", 0, case.capabilities, case.incoming)
    assert (
        peer.reserve_attention_forward(
            lambda candidate: case.reports[2],
            candidate=case.candidate,
            transaction_id="graph-1",
        )
        == case.decision
    )
    assert peer.submit_forward(case.local, transaction_id="graph-1") == case.plan
    kinds = [message.kind for _, _, _, message in peer.transcript]
    assert kinds[:15] == [
        *(["GRAPH_QUERY"] * 3),
        *(["GRAPH_REPORT"] * 3),
        *(["GRAPH_DECISION"] * 3),
        *(["GRAPH_ACK"] * 6),
    ]
    assert kinds[15] == "COUNTS_REQUEST"


@pytest.mark.parametrize("role,role_rank", [("attention", 1), ("ffn", 0)])
def test_graph_peer_waits_for_query_and_global_commit_before_return(
    graph_round, role, role_rank
):
    case = graph_round
    peer = ScriptedStagedControlPeer(
        role, role_rank, case.capabilities, {2: case.peer_incoming}
    )

    def resolve(candidate):
        return case.reports[peer.world_rank]

    decision = (
        peer.reserve_attention_forward(resolve)
        if role == "attention"
        else peer.receive_graph_reservation(resolve)
    )
    assert decision == case.decision
    assert [
        (direction, message.kind) for direction, _, _, message in peer.transcript
    ] == [
        ("recv", "GRAPH_QUERY"),
        ("send", "GRAPH_REPORT"),
        ("recv", "GRAPH_DECISION"),
        ("send", "GRAPH_ACK"),
        ("recv", "GRAPH_ACK"),
    ]


@pytest.mark.parametrize("role,rank", [("attention", 1), ("ffn", 0)])
def test_graph_idle_peer_acknowledges_stop_without_unsolicited_report(
    graph_round, role, rank
):
    peer = ScriptedStagedControlPeer(
        role,
        rank,
        graph_round.capabilities,
        {2: [HcclPlanMessage("STOP"), HcclPlanMessage("STOP_ACK")]},
    )

    def resolve(candidate):
        return HcclGraphReport("missing")

    decision = (
        peer.reserve_attention_forward(resolve)
        if role == "attention"
        else peer.receive_graph_reservation(resolve)
    )
    assert decision is None
    assert [
        message.kind
        for direction, _, _, message in peer.transcript
        if direction == "send"
    ] == ["STOP_ACK"]
    assert peer.terminal


@pytest.mark.parametrize("bad_kind", ["report", "ack"])
def test_graph_coordinator_drains_bad_complete_replies_before_global_error(
    graph_round, bad_kind
):
    case = graph_round
    index = 0 if bad_kind == "report" else 1
    case.incoming[0][index] = ValueError("invalid complete graph JSON")
    peer = ScriptedStagedControlPeer("attention", 0, case.capabilities, case.incoming)
    with pytest.raises(RuntimeError, match="invalid complete graph JSON"):
        peer.reserve_attention_forward(
            lambda candidate: case.reports[2],
            candidate=case.candidate,
            transaction_id="graph-1",
        )
    events = [
        (
            direction,
            source,
            message.kind if isinstance(message, HcclPlanMessage) else "bad",
        )
        for direction, source, _, message in peer.transcript
    ]
    first_error = next(
        index for index, event in enumerate(events) if event[2] == "ERROR"
    )
    assert events[first_error - 3 : first_error] == [
        (
            "recv",
            rank,
            "bad"
            if rank == 0
            else f"GRAPH_{'REPORT' if bad_kind == 'report' else 'ACK'}",
        )
        for rank in (0, 1, 3)
    ]
    assert peer.terminal and peer.plan is None


@pytest.mark.parametrize(
    "field,value",
    [("world_rank", 2), ("transaction_id", "stale"), ("graph_digest", "0" * 64)],
)
def test_graph_report_cannot_impersonate_another_query_or_sender(
    graph_round, field, value
):
    case = graph_round
    case.incoming[0][0] = replace(case.report_messages[0], **{field: value})
    peer = ScriptedStagedControlPeer("attention", 0, case.capabilities, case.incoming)
    with pytest.raises(RuntimeError, match="query or sender"):
        peer.reserve_attention_forward(
            lambda candidate: case.reports[2],
            candidate=case.candidate,
            transaction_id="graph-1",
        )
    assert peer.terminal


def test_graph_miss_selects_eager_geometry_before_input_preparation():
    case = make_graph_control_round(
        states=("missing", "ready", "ready", "ready"), eager_rows=(1, 1)
    )
    peer = ScriptedStagedControlPeer("attention", 0, case.capabilities, case.incoming)
    decision = peer.reserve_attention_forward(
        lambda candidate: case.reports[2],
        candidate=case.candidate,
        transaction_id="graph-1",
    )
    assert decision.phase == "eager" and decision.padded_tokens == (1, 1)
    assert (
        peer.submit_forward(case.local, transaction_id="graph-1").graph_execution
        == decision
    )


def test_graph_completion_retains_parent_complete_digest_and_releases_reservation(
    graph_round,
):
    case = graph_round
    peer = ScriptedStagedControlPeer("attention", 0, case.capabilities, case.incoming)
    peer.reserve_attention_forward(
        lambda candidate: case.reports[2],
        candidate=case.candidate,
        transaction_id="graph-1",
    )
    peer.submit_forward(case.local, transaction_id="graph-1")
    peer.complete_graph_forward(case.candidate.key)
    sent = [
        message for direction, _, _, message in peer.transcript if direction == "send"
    ]
    assert sent[-3:] == [case.complete] * 3
    assert peer.plan is None
    with pytest.raises(RuntimeError, match="graph completion requires"):
        peer.complete_graph_forward(case.candidate.key)


@pytest.mark.parametrize(
    "operation", ["stop", "second-reservation", "different-id", "actual-rows"]
)
def test_reserved_graph_rejects_stop_reentry_or_changed_postprepare_geometry(
    graph_round, operation
):
    case = graph_round
    peer = ScriptedStagedControlPeer("attention", 0, case.capabilities, case.incoming)
    peer.reserve_attention_forward(
        lambda candidate: case.reports[2],
        candidate=case.candidate,
        transaction_id="graph-1",
    )
    with pytest.raises((ValueError, RuntimeError)):
        if operation == "stop":
            peer.stop_attention()
        elif operation == "second-reservation":
            peer.reserve_attention_forward(
                lambda candidate: case.reports[2],
                candidate=case.candidate,
                transaction_id="graph-2",
            )
        else:
            peer.submit_forward(
                case.local
                if operation == "different-id"
                else HcclLocalForward("target", (1, 1), (1, 1)),
                transaction_id="other" if operation == "different-id" else "graph-1",
            )
    assert peer.plan is None


@pytest.mark.parametrize(
    "operation", ["wrong-key", "eager-send", "eager-recv", "premature-finish"]
)
def test_graph_parent_rejects_partial_eager_execution_and_wrong_completion_key(
    graph_round, operation
):
    case = graph_round
    peer = ScriptedStagedControlPeer("attention", 0, case.capabilities, case.incoming)
    peer.reserve_attention_forward(
        lambda candidate: case.reports[2],
        candidate=case.candidate,
        transaction_id="graph-1",
    )
    peer.submit_forward(case.local, transaction_id="graph-1")
    with pytest.raises((ValueError, RuntimeError)):
        if operation == "wrong-key":
            peer.complete_graph_forward(make_graph_key(rows=(4,)))
        elif operation == "eager-send":
            peer.send_layer_inputs(0, 0, None, None, None)
        elif operation == "eager-recv":
            peer.receive_layer_output(0, 0)
        else:
            peer.finish_forward()
    assert peer.plan == case.plan


@pytest.mark.parametrize("stage", ["query", "decision"])
def test_graph_peer_rejects_complete_bad_frame_with_coordinated_error(
    graph_round, stage
):
    case = graph_round
    incoming = [] if stage == "query" else [case.query]
    incoming += [
        ValueError("invalid complete graph JSON"),
        HcclPlanMessage("ERROR", error="graph rejected"),
    ]
    peer = ScriptedStagedControlPeer("ffn", 0, case.capabilities, {2: incoming})
    with pytest.raises((ValueError, RuntimeError)):
        peer.receive_graph_reservation(lambda candidate: case.reports[0])
    sent = [
        message.kind
        for direction, _, _, message in peer.transcript
        if direction == "send"
    ]
    assert sent == (["ERROR"] if stage == "query" else ["GRAPH_REPORT", "ERROR"])
    assert peer.terminal


def test_graph_peer_receiving_global_error_does_not_send_an_unsolicited_ack(
    graph_round,
):
    case = graph_round
    peer = ScriptedStagedControlPeer(
        "ffn",
        0,
        case.capabilities,
        {2: [case.query, HcclPlanMessage("ERROR", error="peer miss")]},
    )
    with pytest.raises(RuntimeError, match="peer miss"):
        peer.receive_graph_reservation(lambda candidate: case.reports[0])
    assert [
        message.kind
        for direction, _, _, message in peer.transcript
        if direction == "send"
    ] == ["GRAPH_REPORT"]
    assert peer.terminal


@pytest.mark.parametrize("boundary", ["report", "ack"])
def test_graph_broken_io_is_terminal_without_claiming_coordinated_abort(
    graph_round, boundary
):
    case = graph_round
    case.incoming[0][0 if boundary == "report" else 1] = RuntimeError(
        "broken graph frame"
    )
    peer = ScriptedStagedControlPeer("attention", 0, case.capabilities, case.incoming)
    with pytest.raises(RuntimeError, match="broken graph frame"):
        peer.reserve_attention_forward(
            lambda candidate: case.reports[2],
            candidate=case.candidate,
            transaction_id="graph-1",
        )
    assert "ERROR" not in [
        message.kind
        for direction, _, _, message in peer.transcript
        if direction == "send"
    ]
    assert peer.terminal


@pytest.mark.parametrize("changed", ["transaction_id", "graph_digest", "world_rank"])
def test_graph_peer_rejects_incorrect_global_commit(graph_round, changed):
    case = graph_round
    value = {"transaction_id": "other", "graph_digest": "0" * 64, "world_rank": 3}[
        changed
    ]
    case.peer_incoming[-1] = replace(case.acks[2], **{changed: value})
    peer = ScriptedStagedControlPeer(
        "ffn", 0, case.capabilities, {2: case.peer_incoming}
    )
    with pytest.raises(ValueError, match="global GRAPH_ACK"):
        peer.receive_graph_reservation(lambda candidate: case.reports[0])
    assert peer.terminal and peer.plan is None


def test_graph_peer_cannot_accept_replay_when_its_cache_report_was_missing(graph_round):
    case = graph_round
    incoming = [
        *case.peer_incoming[:2],
        HcclPlanMessage("ERROR", error="bad replay decision"),
    ]
    peer = ScriptedStagedControlPeer("ffn", 0, case.capabilities, {2: incoming})
    with pytest.raises(RuntimeError, match="bad replay decision"):
        peer.receive_graph_reservation(lambda candidate: HcclGraphReport("missing"))
    assert [
        message.kind
        for direction, _, _, message in peer.transcript
        if direction == "send"
    ] == ["GRAPH_REPORT", "ERROR"]
    assert peer.terminal


def test_graph_ffn_rejects_stop_between_reservation_and_actual_forward(graph_round):
    case = graph_round
    peer = ScriptedStagedControlPeer(
        "ffn", 0, case.capabilities, {2: [*case.peer_incoming, HcclPlanMessage("STOP")]}
    )
    peer.receive_graph_reservation(lambda candidate: case.reports[0])
    with pytest.raises(ValueError, match="idle count/reservation"):
        peer.receive_forward()
    assert [
        message.kind
        for direction, _, _, message in peer.transcript
        if direction == "send"
    ] == ["GRAPH_REPORT", "GRAPH_ACK"]
    assert peer.terminal


def test_graph_attention_rejects_stop_between_reservation_and_actual_counts(
    graph_round,
):
    case = graph_round
    peer = ScriptedStagedControlPeer(
        "attention",
        1,
        case.capabilities,
        {2: [*case.peer_incoming, HcclPlanMessage("STOP")]},
    )
    peer.reserve_attention_forward(lambda candidate: case.reports[3])
    with pytest.raises(ValueError, match="idle graph reservation"):
        peer.submit_forward(case.local)
    assert [
        message.kind
        for direction, _, _, message in peer.transcript
        if direction == "send"
    ] == ["GRAPH_REPORT", "GRAPH_ACK"]
    assert peer.terminal


def test_graph_attention_rejects_changed_counts_request_before_sending_local_counts(
    graph_round,
):
    case = graph_round
    peer = ScriptedStagedControlPeer(
        "attention",
        1,
        case.capabilities,
        {
            2: [
                *case.peer_incoming,
                HcclPlanMessage("COUNTS_REQUEST", transaction_id="other"),
                HcclPlanMessage("ERROR", error="wrong reserved ID"),
            ]
        },
    )
    peer.reserve_attention_forward(lambda candidate: case.reports[3])
    with pytest.raises(RuntimeError, match="wrong reserved ID"):
        peer.submit_forward(case.local)
    assert [
        message.kind
        for direction, _, _, message in peer.transcript
        if direction == "send"
    ] == ["GRAPH_REPORT", "GRAPH_ACK", "ERROR"]
    assert peer.terminal


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_graph_enabled_peers_require_reservation_before_actual_forward(
    graph_round, role
):
    peer = ScriptedStagedControlPeer(role, 0, graph_round.capabilities, {})
    with pytest.raises(RuntimeError, match="reserv"):
        if role == "attention":
            peer.submit_forward(graph_round.local, transaction_id="graph-1")
        else:
            peer.receive_forward()
    assert peer.transcript == []


@pytest.fixture
def capabilities():
    return HcclPlanCapabilities(16, 128, 8, (0, 2), 2, 2, native_tp_size=2)


@pytest.mark.parametrize(
    "role,rank,world_rank,coordinator",
    [
        ("ffn", 0, 0, False),
        ("ffn", 1, 1, False),
        ("attention", 0, 2, True),
        ("attention", 1, 3, False),
    ],
)
def test_rank_identity_preserves_ffn_first_world_order(
    capabilities, role, rank, world_rank, coordinator
):
    transport = HcclStagedTransport(
        role, rank, capabilities, init_method="tcp://localhost:1", local_device=0
    )
    assert transport.world_rank == world_rank
    assert transport.is_coordinator is coordinator
    transport.close()
    transport.close()
    with pytest.raises(RuntimeError, match="not active"):
        transport.finish_forward()


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "draft"),
        ("role_rank", True),
        ("role_rank", -1),
        ("role_rank", 2),
        ("local_device", False),
        ("local_device", -1),
        ("timeout_seconds", 0),
        ("timeout_seconds", True),
    ],
)
def test_construction_rejects_invalid_endpoint_configuration(
    capabilities, field, value
):
    options = dict(
        role="attention",
        role_rank=0,
        capabilities=capabilities,
        init_method="tcp://localhost:1",
        local_device=0,
        timeout_seconds=60,
    )
    with pytest.raises(ValueError):
        HcclStagedTransport(**{**options, field: value})


def test_closed_endpoint_cannot_be_initialized(capabilities):
    transport = HcclStagedTransport(
        "attention", 0, capabilities, init_method="tcp://localhost:1", local_device=0
    )
    transport.close()
    with pytest.raises(RuntimeError, match="cannot be initialized again"):
        transport.initialize()


@pytest.fixture
def count_round(capabilities):
    capabilities = replace(capabilities, local_counts_version=1)
    local = HcclLocalForward("target", (4, 1), (4, 4))
    plan = HcclParentPlan(
        "counted-1",
        capabilities.layer_indices,
        tuple(
            HcclStageLayout(i, (v, v), (p, p), 2, 2)
            for i, (v, p) in enumerate(
                zip(local.valid_tokens, local.padded_tokens, strict=True)
            )
        ),
    )
    request = HcclPlanMessage("COUNTS_REQUEST", transaction_id=plan.transaction_id)
    report = HcclPlanMessage(
        "COUNTS_REPORT",
        transaction_id=plan.transaction_id,
        world_rank=3,
        local_forward=local,
    )
    accepts = {
        rank: HcclPlanMessage(
            "ACCEPT",
            transaction_id=plan.transaction_id,
            plan_digest=plan.digest,
            world_rank=rank,
        )
        for rank in (0, 1, 2, 3)
    }
    incoming = {0: [accepts[0]], 1: [accepts[1]], 3: [report, accepts[3]]}
    return SimpleNamespace(
        capabilities=capabilities,
        local=local,
        plan=plan,
        request=request,
        report=report,
        accepts=accepts,
        incoming=incoming,
    )


def test_coordinator_collects_actual_tp_reports_before_global_accept(count_round):
    case = count_round
    peer = ScriptedStagedControlPeer("attention", 0, case.capabilities, case.incoming)
    assert peer.submit_forward(case.local, transaction_id="counted-1") == case.plan
    assert peer.plan == case.plan
    assert [
        (direction, source, dest, message.kind)
        for direction, source, dest, message in peer.transcript
    ] == [
        ("send", 2, 3, "COUNTS_REQUEST"),
        ("recv", 3, 2, "COUNTS_REPORT"),
        ("send", 2, 0, "FORWARD"),
        ("send", 2, 1, "FORWARD"),
        ("send", 2, 3, "FORWARD"),
        ("recv", 0, 2, "ACCEPT"),
        ("recv", 1, 2, "ACCEPT"),
        ("recv", 3, 2, "ACCEPT"),
        ("send", 2, 0, "ACCEPT"),
        ("send", 2, 1, "ACCEPT"),
        ("send", 2, 3, "ACCEPT"),
    ]


def test_attention_peer_verifies_its_report_before_accepting_parent(count_round):
    case = count_round
    peer = ScriptedStagedControlPeer(
        "attention",
        1,
        case.capabilities,
        {
            2: [
                case.request,
                HcclPlanMessage("FORWARD", plan=case.plan),
                case.accepts[2],
            ],
        },
    )
    assert peer.submit_forward(case.local) == case.plan
    sent = [
        message for direction, _, _, message in peer.transcript if direction == "send"
    ]
    assert sent == [case.report, case.accepts[3]]
    assert peer.plan == case.plan


@pytest.mark.parametrize(
    "local,error",
    [
        (HcclLocalForward("target", (4, 2), (4, 4)), "native TP coordinates"),
        (HcclLocalForward("target", (4, 1), (4, 5)), "native TP coordinates"),
        (HcclLocalForward("profile", (4, 1), (4, 4)), "mode or stage count"),
        (HcclLocalForward("target", (4,), (4,)), "mode or stage count"),
    ],
)
def test_disagreeing_actual_tp_reports_reject_all_peers_before_forward(
    count_round,
    local,
    error,
):
    case = count_round
    incoming = {3: [replace(case.report, local_forward=local)]}
    peer = ScriptedStagedControlPeer("attention", 0, case.capabilities, incoming)
    with pytest.raises(RuntimeError, match=error):
        peer.submit_forward(case.local, transaction_id="counted-1")
    sent = [
        (dest, message.kind)
        for direction, _, dest, message in peer.transcript
        if direction == "send"
    ]
    assert sent == [(3, "COUNTS_REQUEST"), (0, "ERROR"), (1, "ERROR"), (3, "ERROR")]
    assert peer.terminal and peer.plan is None


def test_complete_bad_report_frame_is_drained_before_global_rejection(count_round):
    case = count_round
    capabilities = replace(
        case.capabilities, attention_size=4, ffn_size=4, native_tp_size=4
    )
    incoming = {
        5: [ValueError("invalid complete JSON frame")],
        6: [replace(case.report, world_rank=6)],
        7: [replace(case.report, world_rank=7)],
    }
    peer = ScriptedStagedControlPeer("attention", 0, capabilities, incoming)
    with pytest.raises(RuntimeError, match="invalid complete JSON"):
        peer.submit_forward(case.local, transaction_id="counted-1")
    transcript = [
        (
            direction,
            source,
            dest,
            message.kind if isinstance(message, HcclPlanMessage) else "invalid-frame",
        )
        for direction, source, dest, message in peer.transcript
    ]
    assert transcript[:6] == [
        ("send", 4, 5, "COUNTS_REQUEST"),
        ("send", 4, 6, "COUNTS_REQUEST"),
        ("send", 4, 7, "COUNTS_REQUEST"),
        ("recv", 5, 4, "invalid-frame"),
        ("recv", 6, 4, "COUNTS_REPORT"),
        ("recv", 7, 4, "COUNTS_REPORT"),
    ]
    assert transcript[6:] == [
        ("send", 4, rank, "ERROR") for rank in (0, 1, 2, 3, 5, 6, 7)
    ]
    assert peer.terminal and peer.plan is None


def test_broken_count_io_is_terminal_without_an_unsafe_error_broadcast(count_round):
    case = count_round
    peer = ScriptedStagedControlPeer(
        "attention",
        0,
        case.capabilities,
        {
            3: [RuntimeError("injected broken control frame")],
        },
    )
    with pytest.raises(RuntimeError, match="broken control frame"):
        peer.submit_forward(case.local, transaction_id="counted-1")
    assert [
        (dest, message.kind)
        for direction, _, dest, message in peer.transcript
        if direction == "send"
    ] == [(3, "COUNTS_REQUEST")]
    assert peer.terminal


@pytest.mark.parametrize("frame", ["request", "forward"])
def test_peer_rejects_a_complete_bad_coordinator_frame_without_stranding_ack(
    count_round,
    frame,
):
    case = count_round
    incoming = [] if frame == "request" else [case.request]
    incoming.extend(
        [
            ValueError("invalid complete coordinator JSON"),
            HcclPlanMessage("ERROR", error="invalid complete coordinator JSON"),
        ]
    )
    peer = ScriptedStagedControlPeer("attention", 1, case.capabilities, {2: incoming})
    with pytest.raises(RuntimeError, match="invalid complete coordinator JSON"):
        peer.submit_forward(case.local)
    sent = [
        message.kind
        for direction, _, _, message in peer.transcript
        if direction == "send"
    ]
    assert sent == (["COUNTS_REPORT", "ERROR"] if frame == "forward" else ["ERROR"])
    assert peer.terminal and peer.plan is None


@pytest.mark.parametrize("change", ["id", "rank"])
def test_count_report_cannot_impersonate_another_request_or_sender(count_round, change):
    case = count_round
    report = replace(
        case.report,
        **({"transaction_id": "stale"} if change == "id" else {"world_rank": 2}),
    )
    peer = ScriptedStagedControlPeer("attention", 0, case.capabilities, {3: [report]})
    with pytest.raises(RuntimeError, match="identity does not match"):
        peer.submit_forward(case.local, transaction_id="counted-1")
    assert peer.terminal and peer.plan is None


@pytest.mark.parametrize("boundary", ["id", "mode", "valid", "padded", "stages"])
def test_peer_rejects_plan_that_changes_its_report_before_accept(count_round, boundary):
    case = count_round
    plan = case.plan
    if boundary == "id":
        plan = replace(plan, transaction_id="other-parent")
    elif boundary == "mode":
        plan = replace(plan, mode="profile")
    elif boundary == "valid":
        plan = replace(
            plan, stages=(plan.stages[0], HcclStageLayout(1, (0, 0), (4, 4), 2, 2))
        )
    elif boundary == "padded":
        plan = replace(
            plan, stages=(plan.stages[0], HcclStageLayout(1, (1, 1), (3, 3), 2, 2))
        )
    else:
        plan = replace(plan, stages=plan.stages[:1])
    peer = ScriptedStagedControlPeer(
        "attention",
        1,
        case.capabilities,
        {
            2: [
                case.request,
                HcclPlanMessage("FORWARD", plan=plan),
                HcclPlanMessage("ERROR", error="parent changed local report"),
            ],
        },
    )
    with pytest.raises(RuntimeError, match="local report"):
        peer.submit_forward(case.local)
    assert [
        message.kind
        for direction, _, _, message in peer.transcript
        if direction == "send"
    ] == ["COUNTS_REPORT", "ERROR"]
    assert peer.terminal and peer.plan is None


@pytest.mark.parametrize("boundary", ["invalid", "capacity", "caller-id"])
def test_peer_reports_local_preflight_failure_after_request(count_round, boundary):
    case = count_round
    local, options = case.local, {}
    if boundary == "invalid":
        local = HcclLocalForward("target", (True,), (4,))
    elif boundary == "capacity":
        local = HcclLocalForward("target", (1,), (9,))
    else:
        options["transaction_id"] = "wrong-request"
    peer = ScriptedStagedControlPeer(
        "attention",
        1,
        case.capabilities,
        {
            2: [case.request, HcclPlanMessage("ERROR", error="local report rejected")],
        },
    )
    with pytest.raises(RuntimeError, match="local report rejected"):
        peer.submit_forward(local, **options)
    assert [
        message.kind
        for direction, _, _, message in peer.transcript
        if direction == "send"
    ] == ["ERROR"]
    assert peer.terminal and peer.plan is None


def test_peer_waiting_to_submit_can_receive_idle_stop_without_sending_counts(
    count_round,
):
    case = count_round
    peer = ScriptedStagedControlPeer(
        "attention",
        1,
        case.capabilities,
        {
            2: [HcclPlanMessage("STOP"), HcclPlanMessage("STOP_ACK")],
        },
    )
    assert peer.submit_forward(case.local) is None
    assert [
        message.kind
        for direction, _, _, message in peer.transcript
        if direction == "send"
    ] == ["STOP_ACK"]
    assert peer.terminal


@pytest.mark.parametrize("boundary", ["ffn", "disabled", "no-id", "local-invalid"])
def test_submit_rejects_unavailable_protocol_or_coordinator_preflight(
    count_round, boundary
):
    case = count_round
    capabilities, role, local = case.capabilities, "attention", case.local
    if boundary == "ffn":
        role = "ffn"
    elif boundary == "disabled":
        capabilities = replace(capabilities, local_counts_version=0)
    elif boundary == "local-invalid":
        local = HcclLocalForward("target", (2,), (1,))
    peer = ScriptedStagedControlPeer(role, 0, capabilities, {})
    with pytest.raises((ValueError, RuntimeError)):
        peer.submit_forward(
            local, transaction_id=None if boundary == "no-id" else "counted-1"
        )
    sent = [
        message.kind
        for direction, _, _, message in peer.transcript
        if direction == "send"
    ]
    assert sent == (["ERROR"] * 3 if boundary in {"no-id", "local-invalid"} else [])
    assert peer.plan is None


@pytest.fixture
def empty_count_sequence(count_round, monkeypatch):
    monkeypatch.setattr("torch.npu.synchronize", lambda device: None)
    case = count_round
    locals_ = [
        HcclLocalForward("profile", (0,), (0,)),
        HcclLocalForward("target", (0, 0), (0, 0)),
    ]
    plans, incoming = [], {0: [], 1: [], 3: []}
    for index, local in enumerate(locals_):
        plan = HcclParentPlan(
            f"sequence-{index}",
            case.capabilities.layer_indices,
            tuple(
                HcclStageLayout(i, (0, 0), (0, 0), 2, 2)
                for i in range(len(local.valid_tokens))
            ),
            local.mode,
        )
        plans.append(plan)
        for rank in incoming:
            if rank == 3:
                incoming[rank].append(
                    HcclPlanMessage(
                        "COUNTS_REPORT",
                        world_rank=rank,
                        transaction_id=plan.transaction_id,
                        local_forward=local,
                    )
                )
            incoming[rank].extend(
                [
                    HcclPlanMessage(
                        "ACCEPT",
                        world_rank=rank,
                        transaction_id=plan.transaction_id,
                        plan_digest=plan.digest,
                    ),
                    HcclPlanMessage(
                        "COMPLETE",
                        transaction_id=plan.transaction_id,
                        plan_digest=plan.digest,
                    ),
                ]
            )
    for messages in incoming.values():
        messages.append(HcclPlanMessage("STOP_ACK"))
    return SimpleNamespace(
        peer=ScriptedStagedControlPeer("attention", 0, case.capabilities, incoming),
        locals=locals_,
        plans=plans,
    )


def test_u1_and_u2_counts_share_a_session_until_complete_then_idle_stop(
    empty_count_sequence,
):
    case = empty_count_sequence
    for local, plan in zip(case.locals, case.plans, strict=True):
        assert (
            case.peer.submit_forward(local, transaction_id=plan.transaction_id) == plan
        )
        case.peer.finish_forward()
        assert case.peer.plan is None
        assert not case.peer.terminal
    case.peer.stop()
    assert case.peer.terminal
    published = [
        message.plan
        for direction, _, dest, message in case.peer.transcript
        if direction == "send" and dest == 0 and message.kind == "FORWARD"
    ]
    assert published == case.plans
    assert [len(plan.stages) for plan in published] == [1, 2]


@pytest.mark.parametrize("rank", [0, 1])
def test_counts_enabled_attention_cannot_accept_an_unverified_manual_plan(
    count_round, rank
):
    case = count_round
    peer = ScriptedStagedControlPeer("attention", rank, case.capabilities, {})
    with pytest.raises(RuntimeError, match="must use submit_forward"):
        if rank == 0:
            peer.begin_forward(case.plan)
        else:
            peer.receive_forward()
    assert peer.plan is None
    assert peer.transcript == []


@pytest.mark.parametrize("version", [0, 1])
@pytest.mark.parametrize("rank", [0, 1])
def test_all_attention_shutdown_uses_stop_ack_without_submitting_counts(
    count_round,
    version,
    rank,
):
    case = count_round
    capabilities = replace(case.capabilities, local_counts_version=version)
    incoming = (
        {source: [HcclPlanMessage("STOP_ACK")] for source in (0, 1, 3)}
        if rank == 0
        else {2: [HcclPlanMessage("STOP"), HcclPlanMessage("STOP_ACK")]}
    )
    peer = ScriptedStagedControlPeer("attention", rank, capabilities, incoming)
    peer.stop_attention()
    sent = [
        (dest, message.kind)
        for direction, _, dest, message in peer.transcript
        if direction == "send"
    ]
    assert sent == (
        [
            (0, "STOP"),
            (1, "STOP"),
            (3, "STOP"),
            (0, "STOP_ACK"),
            (1, "STOP_ACK"),
            (3, "STOP_ACK"),
        ]
        if rank == 0
        else [(2, "STOP_ACK")]
    )
    assert peer.terminal and peer.plan is None


@pytest.mark.parametrize(
    "kind", ["COUNTS_REQUEST", "FORWARD", "ERROR", "invalid-frame"]
)
def test_attention_shutdown_rejects_other_control_phases_without_a_stop_ack(
    count_round,
    kind,
):
    case = count_round
    messages = {
        "COUNTS_REQUEST": case.request,
        "FORWARD": HcclPlanMessage("FORWARD", plan=case.plan),
        "ERROR": HcclPlanMessage("ERROR", error="shutdown rejected"),
        "invalid-frame": ValueError("invalid complete shutdown frame"),
    }
    incoming = [messages[kind]]
    if kind != "ERROR":
        incoming.append(HcclPlanMessage("ERROR", error="shutdown rejected"))
    peer = ScriptedStagedControlPeer("attention", 1, case.capabilities, {2: incoming})
    with pytest.raises(RuntimeError, match="shutdown rejected"):
        peer.stop_attention()
    sent = [
        message.kind
        for direction, _, _, message in peer.transcript
        if direction == "send"
    ]
    assert sent == ([] if kind == "ERROR" else ["ERROR"])
    assert peer.terminal and peer.plan is None


def test_ffn_cannot_initiate_attention_shutdown(count_round):
    peer = ScriptedStagedControlPeer("ffn", 0, count_round.capabilities, {})
    with pytest.raises(RuntimeError, match="requires attention"):
        peer.stop_attention()
    assert peer.transcript == []
