# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# ruff: noqa: E402
"""Loaded native context bodies and request-ready codecs, with CPU device leaves."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from afd_plugin.compat.npu.dspark_pd import (
    DsparkPdReadLease,
    DsparkPdReadQuery,
    draft_schema_digest,
    prompt_digest,
    read_lease_from_payload,
    read_lease_payload,
    read_query_from_payload,
    read_query_payload,
    ready_from_payload,
    ready_payload,
    source_request_from_payload,
    source_request_payload,
)
from tests.unit.fakes.dspark_pd import make_dspark_pd_prefix_case


@pytest.fixture
def prefix_case(monkeypatch):
    return make_dspark_pd_prefix_case(monkeypatch)


@pytest.mark.parametrize("record", ["source", "ready", "query", "lease"])
def test_request_ready_records_round_trip_with_all_groups(prefix_case, record):
    query = DsparkPdReadQuery(
        prefix_case.expected_ready,
        "destination",
        "destination-generation",
        prefix_case.source.transfer_block_ids,
        "query-1",
    )
    records = {
        "source": (
            prefix_case.source,
            source_request_payload,
            source_request_from_payload,
        ),
        "ready": (prefix_case.expected_ready, ready_payload, ready_from_payload),
        "query": (query, read_query_payload, read_query_from_payload),
        "lease": (
            DsparkPdReadLease(
                query.expected_ready,
                query.query_id,
                query.destination_engine_id,
                query.destination_generation,
                query.destination_block_ids,
                "lease-1",
            ),
            read_lease_payload,
            read_lease_from_payload,
        ),
    }
    typed, encode, decode = records[record]
    assert decode(encode(typed)) == typed


@pytest.mark.parametrize("destination_mode", ["tail", "empty-group"])
def test_read_lease_round_trip_preserves_source_superset(prefix_case, destination_mode):
    destination = list(prefix_case.source.transfer_block_ids)
    if destination_mode == "tail":
        destination[-1] = destination[-1][-1:]
    else:
        destination[0] = ()
    query = DsparkPdReadQuery(
        prefix_case.expected_ready,
        "destination",
        "destination-generation",
        tuple(destination),
        "query-tail",
    )
    assert read_query_from_payload(read_query_payload(query)) == query
    lease = DsparkPdReadLease(
        query.expected_ready,
        query.query_id,
        query.destination_engine_id,
        query.destination_generation,
        query.destination_block_ids,
        "lease-tail",
    )
    decoded = read_lease_from_payload(read_lease_payload(lease))
    assert decoded == lease
    assert decoded.ready == prefix_case.expected_ready
    assert decoded.destination_block_ids == tuple(destination)


@pytest.mark.parametrize("invalid", ["missing-group", "extra-group", "overrun"])
def test_read_query_rejects_destinations_outside_source_groups(prefix_case, invalid):
    destination = prefix_case.source.transfer_block_ids
    if invalid == "missing-group":
        destination = destination[:-1]
    elif invalid == "extra-group":
        destination += ((),)
    else:
        destination = destination[:-1] + (
            destination[-1] + (max(destination[-1]) + 1,),
        )
    payload = {
        "expected_ready": ready_payload(prefix_case.expected_ready),
        "destination_engine_id": "destination",
        "destination_generation": "destination-generation",
        "destination_block_ids": [list(group) for group in destination],
        "query_id": "query-invalid",
    }
    with pytest.raises(ValueError, match="every published source transfer group"):
        read_query_from_payload(payload)


@pytest.mark.parametrize("invalid", ["extra", "missing", "bool", "tuple-wire"])
def test_source_codec_rejects_incomplete_or_untyped_payload(prefix_case, invalid):
    payload = source_request_payload(prefix_case.source)
    if invalid == "extra":
        payload["generation_fallback"] = True
    elif invalid == "missing":
        del payload["prompt_digest"]
    elif invalid == "bool":
        payload["computed_prefix_tokens"] = True
    else:
        payload["allocation_block_ids"] = prefix_case.source.allocation_block_ids
    with pytest.raises(ValueError):
        source_request_from_payload(payload)


@pytest.mark.parametrize(
    "invalid", ["bool", "negative", "empty", "digest", "group", "rewrite"]
)
def test_source_declaration_rejects_invalid_boundaries(prefix_case, invalid):
    source = prefix_case.source
    changes = {
        "bool": {"original_prompt_tokens": True},
        "negative": {"computed_prefix_tokens": -1},
        "empty": {"source_generation": ""},
        "digest": {"prompt_digest": "A" * 64},
        "group": {"transfer_block_ids": source.transfer_block_ids[:-1]},
        "rewrite": {
            "transfer_block_ids": tuple(
                tuple(reversed(group)) for group in source.transfer_block_ids
            )
        },
    }
    with pytest.raises(ValueError):
        replace(source, **changes[invalid])


def test_prompt_digest_preserves_original_token_identity():
    assert prompt_digest((1, 2, 3)) != prompt_digest((1, 3, 2))
    with pytest.raises(ValueError, match="integer"):
        prompt_digest((1, True))


def test_schema_binds_native_group_order_without_local_capacity(prefix_case):
    case = prefix_case
    larger = replace(case.kv_config, num_blocks=case.kv_config.num_blocks + 5)
    assert draft_schema_digest(case.config, larger) == case.source.draft_schema_digest
    reordered = replace(
        case.kv_config, kv_cache_groups=list(reversed(case.kv_config.kv_cache_groups))
    )
    assert (
        draft_schema_digest(case.config, reordered) != case.source.draft_schema_digest
    )


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1), (4, 3), (8, 7)])
def test_native_chunk_projection_publishes_only_complete_prefix(
    monkeypatch, tp_size, tp_rank
):
    case = make_dspark_pd_prefix_case(monkeypatch, tp_size=tp_size, tp_rank=tp_rank)
    first, state = case.make_step(0, 2)
    assert case.consumer.consume_prefill(first, state, (case.source,)) == ()
    second, state = case.make_step(2, 3)
    assert case.consumer.consume_prefill(second, state, (case.source,)) == (
        case.expected_ready,
    )
    for group_id, name in zip(case.draft_group_ids, case.draft_names, strict=True):
        cache = case.caches[name]
        size = case.kv_config.kv_cache_groups[group_id].kv_cache_spec.block_size
        actual = torch.stack(
            [
                cache[
                    case.source.allocation_block_ids[group_id][position // size],
                    position % size,
                    0,
                ]
                for position in range(case.source.computed_prefix_tokens)
            ]
        )
        expected = torch.tensor(
            [
                [240 + 60 * position, 246 + 60 * position]
                for position in range(case.source.computed_prefix_tokens)
            ],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not case.consumer.terminal


def test_length_one_prompt_is_projected_without_truncation(monkeypatch):
    case = make_dspark_pd_prefix_case(monkeypatch, original_tokens=1)
    scheduler, state = case.make_step(0, 1)
    assert case.consumer.consume_prefill(scheduler, state, (case.source,)) == (
        case.expected_ready,
    )
    assert (
        case.expected_ready.original_prompt_tokens
        == case.expected_ready.computed_prefix_tokens
        == 1
    )


def test_chunked_allocation_may_append_blocks_without_rewriting(prefix_case):
    case = prefix_case
    initial = replace(
        case.source,
        allocation_block_ids=tuple(
            group[:1] for group in case.source.allocation_block_ids
        ),
        transfer_block_ids=tuple(
            group[:1] for group in case.source.allocation_block_ids
        ),
    )
    scheduler, state = case.make_step(0, 2)
    assert case.consumer.consume_prefill(scheduler, state, (initial,)) == ()
    scheduler, state = case.make_step(2, 3)
    assert case.consumer.consume_prefill(scheduler, state, (case.source,)) == (
        case.expected_ready,
    )


@pytest.mark.parametrize("invalid", ["gap", "duplicate", "generation", "allocation"])
def test_noncontiguous_or_changed_source_becomes_terminal(prefix_case, invalid):
    case = prefix_case
    scheduler, state = case.make_step(0, 2)
    assert case.consumer.consume_prefill(scheduler, state, (case.source,)) == ()
    offset = {"gap": 3, "duplicate": 0}.get(invalid, 2)
    scheduler, state = case.make_step(offset, 1)
    source = case.source
    if invalid == "generation":
        source = replace(source, source_generation="replacement-generation")
    elif invalid == "allocation":
        blocks = tuple(
            tuple(reversed(group)) if len(group) > 1 else (2,)
            for group in source.allocation_block_ids
        )
        source = replace(source, allocation_block_ids=blocks, transfer_block_ids=blocks)
    with pytest.raises(RuntimeError, match="TP agreement"):
        case.consumer.consume_prefill(scheduler, state, (source,))
    assert case.consumer.terminal
    assert case.agreement.reports[-1].error is not None
    assert case.agreement.reports[-1].ready == ()


@pytest.mark.parametrize("invalid", ["schema", "capacity", "view", "no-state"])
def test_validation_failure_reports_outcome_before_publication(prefix_case, invalid):
    case = prefix_case
    scheduler, state = case.make_step(0, 5)
    source = case.source
    if invalid == "schema":
        source = replace(source, draft_schema_digest="0" * 64)
    elif invalid == "capacity":
        blocks = tuple(
            (*group[:-1], case.layout.num_blocks)
            for group in source.allocation_block_ids
        )
        source = replace(source, allocation_block_ids=blocks, transfer_block_ids=blocks)
    elif invalid == "view":
        layer = case.config.compilation_config.static_forward_context[
            case.draft_names[0]
        ]
        layer.kv_cache = [case.caches[case.draft_names[0]].clone()]
    else:
        state = None
    with pytest.raises(RuntimeError, match="TP agreement"):
        case.consumer.consume_prefill(scheduler, state, (source,))
    assert case.consumer.terminal
    assert case.agreement.reports[-1].error is not None
    assert case.agreement.reports[-1].ready == ()
    assert all(torch.count_nonzero(case.caches[name]) == 0 for name in case.draft_names)


@pytest.mark.parametrize("failure", ["local", "peer"])
def test_projection_failure_cannot_publish_ready_or_resume(monkeypatch, failure):
    case = make_dspark_pd_prefix_case(
        monkeypatch,
        tp_size=2,
        projection_error=failure == "local",
        peer_error=failure == "peer",
    )
    scheduler, state = case.make_step(0, 5)
    with pytest.raises(RuntimeError, match="TP agreement"):
        case.consumer.consume_prefill(scheduler, state, (case.source,))
    assert case.consumer.terminal
    with pytest.raises(RuntimeError, match="terminal"):
        case.consumer.consume_prefill(scheduler, state, (case.source,))
    assert case.consumer.binding.draft_model is case.model
    assert case.consumer.binding.target_model is case.target_model


def test_finished_cleanup_runs_for_zero_work_and_allows_new_generation(prefix_case):
    case = prefix_case
    scheduler, state = case.make_step(0, 2)
    assert case.consumer.consume_prefill(scheduler, state, (case.source,)) == ()
    scheduler.finished_req_ids = {case.source.request_id}
    assert case.consumer.consume_prefill(scheduler, None, ()) == ()
    source = replace(case.source, source_generation="generation-after-retirement")
    scheduler, state = case.make_step(0, 5)
    ready = case.consumer.consume_prefill(scheduler, state, (source,))
    assert ready == (
        replace(case.expected_ready, source_generation=source.source_generation),
    )


def test_storage_binding_rejects_mismatched_layout_before_projection(monkeypatch):
    case = make_dspark_pd_prefix_case(monkeypatch, bind=False)
    with pytest.raises(ValueError, match="materialized storage"):
        case.consumer.bind_registered_layout(
            replace(case.layout, device_type="npu"),
            case.kv_config,
            case.caches,
        )
    assert all(torch.count_nonzero(case.caches[name]) == 0 for name in case.draft_names)


def test_source_extraction_failure_participates_in_tp_outcome(prefix_case):
    scheduler, _ = prefix_case.make_step(0, 5)
    with pytest.raises(RuntimeError, match="TP agreement"):
        prefix_case.consumer.consume_prefill(
            scheduler,
            None,
            (),
            source_error=ValueError("invalid source metadata"),
        )
    report = prefix_case.agreement.reports[-1]
    assert report.error == "ValueError: invalid source metadata"
    assert report.ready == ()
    assert prefix_case.consumer.terminal


def test_all_empty_tp_outcomes_are_a_noop(prefix_case):
    scheduler, _ = prefix_case.make_step(0, 5)
    assert prefix_case.consumer.consume_prefill(scheduler, None, ()) == ()
    assert not prefix_case.consumer.terminal
    assert prefix_case.agreement.reports[-1].prefixes == ()


def test_sliding_window_retires_and_reuses_blocks(monkeypatch):  # long-test: 3 chunks
    case = make_dspark_pd_prefix_case(monkeypatch, original_tokens=13, num_blocks=8)
    mappings = (
        (((1,), (1, 2)), ((1,), (1, 2))),
        (((1, 2), (0, 2, 1, 3)), ((1, 2), (1, 3))),
        (((0, 2, 1), (0, 0, 0, 3, 1, 2)), ((2, 1), (1, 2))),
    )
    final_source = case.source
    for offset, (allocated, transfer) in zip((0, 4, 8), mappings, strict=True):
        allocations, transfers = (
            list(case.source.allocation_block_ids),
            list(case.source.transfer_block_ids),
        )
        for group_id, blocks, exported in zip(
            case.draft_group_ids, allocated, transfer, strict=True
        ):
            allocations[group_id], transfers[group_id] = blocks, exported
        source = replace(
            case.source,
            allocation_block_ids=tuple(allocations),
            transfer_block_ids=tuple(transfers),
        )
        scheduler, state = case.make_step(offset, 4)
        result = case.consumer.consume_prefill(scheduler, state, (source,))
        if offset < 8:
            assert result == ()
        else:
            final_source = source
            assert result == (
                replace(
                    case.expected_ready, transfer_block_ids=source.transfer_block_ids
                ),
            )
    for group_id, name in zip(case.draft_group_ids, case.draft_names, strict=True):
        size = case.kv_config.kv_cache_groups[group_id].kv_cache_spec.block_size
        actual = torch.stack(
            [
                case.caches[name][
                    final_source.allocation_block_ids[group_id][position // size],
                    position % size,
                    0,
                ]
                for position in range(8, 12)
            ]
        )
        expected = torch.tensor(
            [[240 + 60 * position, 246 + 60 * position] for position in range(8, 12)],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_current_chunk_cannot_project_into_native_null_block(prefix_case):
    case = prefix_case
    allocated = list(case.source.allocation_block_ids)
    allocated[case.draft_group_ids[0]] = (0, 2)
    transfers = list(case.source.transfer_block_ids)
    transfers[case.draft_group_ids[0]] = (2,)
    source = replace(
        case.source,
        allocation_block_ids=tuple(allocated),
        transfer_block_ids=tuple(transfers),
    )
    scheduler, state = case.make_step(0, 5)
    with pytest.raises(RuntimeError, match="TP agreement"):
        case.consumer.consume_prefill(scheduler, state, (source,))
    assert "cannot write a null block" in case.agreement.reports[-1].error
    assert all(torch.count_nonzero(case.caches[name]) == 0 for name in case.draft_names)
