# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU graph-control geometry builders, independent of torch and vLLM."""

from types import SimpleNamespace

from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_graph import (
    HcclGraphCandidate,
    HcclGraphKey,
    HcclGraphReport,
    HcclGraphStage,
    decide_graph_execution,
)
from afd_plugin.distributed.hccl_plan import (
    HcclLocalForward,
    HcclParentPlan,
    HcclPlanCapabilities,
    HcclPlanMessage,
)


def make_graph_key(*, rows=(2, 2), metadata_shapes=(), target_query_len=1):
    start = 0
    stages = []
    for count in rows:
        stop = start + count
        stages.append(
            HcclGraphStage(
                start // target_query_len,
                (stop + target_query_len - 1) // target_query_len,
                start,
                stop,
            )
        )
        start += count
    return HcclGraphKey(
        start,
        start // target_query_len,
        target_query_len,
        target_query_len,
        0,
        len(rows),
        tuple(stages),
        metadata_shapes,
    )


def make_graph_candidate(
    *, phase="replay", mode="target", rows=(2, 2), eager_rows=None, target_query_len=1
):
    return HcclGraphCandidate(
        phase,
        mode,
        None
        if phase == "eager"
        else make_graph_key(rows=rows, target_query_len=target_query_len),
        rows if eager_rows is None else eager_rows,
    )


def make_graph_control_round(*, phase="replay", states=None, eager_rows=(2, 2)):
    capabilities = HcclPlanCapabilities(
        16,
        128,
        8,
        (0, 2),
        2,
        2,
        2,
        local_counts_version=1,
        graph_execution_version=1,
    )
    candidate = make_graph_candidate(phase=phase, eager_rows=eager_rows)
    states = states or ("ready" if phase == "replay" else "missing",) * 4
    reports = {rank: HcclGraphReport(state) for rank, state in enumerate(states)}
    decision = decide_graph_execution("graph-1", candidate, tuple(reports.values()))
    local = HcclLocalForward("target", decision.padded_tokens, decision.padded_tokens)
    plan = HcclParentPlan(
        decision.transaction_id,
        capabilities.layer_indices,
        tuple(
            HcclStageLayout(index, (rows, rows), (rows, rows), 2, 2)
            for index, rows in enumerate(decision.padded_tokens)
        ),
        graph_execution=decision,
    )
    query = HcclPlanMessage(
        "GRAPH_QUERY", transaction_id="graph-1", graph_candidate=candidate
    )
    report_messages = {
        rank: HcclPlanMessage(
            "GRAPH_REPORT",
            transaction_id="graph-1",
            world_rank=rank,
            graph_digest=candidate.digest,
            graph_report=report,
        )
        for rank, report in reports.items()
    }
    acks = {
        rank: HcclPlanMessage(
            "GRAPH_ACK",
            transaction_id="graph-1",
            world_rank=rank,
            graph_digest=decision.digest,
        )
        for rank in range(4)
    }
    accepts = {
        rank: HcclPlanMessage(
            "ACCEPT", transaction_id="graph-1", world_rank=rank, plan_digest=plan.digest
        )
        for rank in range(4)
    }
    complete = HcclPlanMessage(
        "COMPLETE", transaction_id="graph-1", plan_digest=plan.digest
    )
    count_report = HcclPlanMessage(
        "COUNTS_REPORT", transaction_id="graph-1", world_rank=3, local_forward=local
    )
    incoming = {
        rank: [
            report_messages[rank],
            acks[rank],
            *([count_report] if rank == 3 else []),
            accepts[rank],
            complete,
        ]
        for rank in (0, 1, 3)
    }
    peer_incoming = [
        query,
        HcclPlanMessage("GRAPH_DECISION", graph_decision=decision),
        acks[2],
    ]
    return SimpleNamespace(
        capabilities=capabilities,
        candidate=candidate,
        decision=decision,
        reports=reports,
        report_messages=report_messages,
        acks=acks,
        query=query,
        plan=plan,
        local=local,
        accepts=accepts,
        complete=complete,
        incoming=incoming,
        peer_incoming=peer_incoming,
    )
