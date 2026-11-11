# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# ruff: noqa: E402
"""Actual codec/native metadata and CPU thread seams; no engine/NPU execution."""

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")
msgspec = pytest.importorskip("msgspec")

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.v1.request import RequestStatus

from afd_plugin.compat.npu import kv_layout_connector as adapter
from afd_plugin.compat.npu.dspark_pd import (
    read_lease_from_payload,
    source_request_payload,
)
from afd_plugin.compat.npu.kv_layout import layout_from_json
from afd_plugin.compat.npu.kv_layout_native import collect_deepseek_v41_kv_layout
from tests.unit.fakes.dspark_pd import make_dspark_pd_prefix_case
from tests.unit.fakes.kv_layout_connector import (
    DESTINATION_SENTINEL,
    REMOTE_BASE_ADDRESS,
    REMOTE_HANDSHAKE_PORT,
    THREAD_TEST_TIMEOUT,
    TRANSFER_PORT,
    RegistrationBoundary,
    attach_consumer_worker,
    connector_config,
    dspark_query,
    hold_receiver_admission,
    make_dspark_read_case,
    make_dspark_registration_case,
    make_dspark_scheduler_case,
    make_receiver_case,
    make_registration_case,
    make_retiring_dspark_read_case,
    make_retiring_receiver_case,
    peer_is_published,
    publish_dspark_prefix,
    request_map,
    sender_request,
)

RETIREMENT_TEST_DEADLINE = 0.01
BYTE_VALUE_COUNT = 256


@pytest.fixture
def dspark_case(monkeypatch):
    prefix = make_dspark_pd_prefix_case(monkeypatch, bind=False)
    return make_dspark_registration_case(monkeypatch, prefix)


@pytest.fixture
def receiver_case(monkeypatch):
    case = make_receiver_case(monkeypatch)
    try:
        yield case
    finally:
        case.close()


def test_external_factory_selects_only_the_configured_attesting_connector():
    transfer = connector_config().kv_transfer_config
    assert (
        KVConnectorFactory.get_connector_class(transfer)
        is adapter.MooncakeHybridConnector
    )


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_registration_publishes_required_layout_before_starting_threads(
    monkeypatch, role
):
    case = make_registration_case(monkeypatch, role=role)
    case.worker.register_kv_caches(case.cache.caches)
    metadata = case.worker.xfer_handshake_metadata
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(metadata), type=adapter.AttestedMooncakeAgentMetadata
    )
    layout = layout_from_json(decoded.afd_kv_layout.decode())
    assert layout.num_blocks == case.cache.config.num_blocks
    assert decoded.block_lens == [layout.pages[0].block_stride_bytes]
    assert case.boundary.events == ["registered", "started"]
    assert len(case.boundary.regions) == 1
    with pytest.raises(RuntimeError, match="single-use"):
        case.worker.register_kv_caches(case.cache.caches)


@pytest.mark.parametrize("failure", ["registration", "start"])
def test_registration_failure_retains_memory_dependencies_and_forbids_retry(
    monkeypatch, failure
):
    case = make_registration_case(monkeypatch)
    setattr(case.boundary, "fail_" + failure, True)
    with pytest.raises(RuntimeError, match="controlled"):
        case.worker.register_kv_caches(case.cache.caches)
    assert case.boundary.regions
    assert case.worker.kv_caches is case.cache.caches
    with pytest.raises(RuntimeError, match="single-use"):
        case.worker.register_kv_caches(case.cache.caches)


@pytest.mark.parametrize("unsupported", ["cpu", "hybrid", "mamba", "transfer", "host"])
def test_unsupported_materialized_registration_never_reaches_the_engine(
    monkeypatch, unsupported
):
    case = make_registration_case(monkeypatch)
    if unsupported == "cpu":
        monkeypatch.setattr(
            adapter, "collect_deepseek_v41_kv_layout", collect_deepseek_v41_kv_layout
        )
    elif unsupported == "hybrid":
        case.worker.use_hybrid = False
    elif unsupported == "mamba":
        case.worker.use_mamba = True
    elif unsupported == "transfer":
        case.cache.config.kv_cache_groups[-1].enable_kv_transfer = False
    else:
        case.cache.config.kv_cache_groups[-1].host_resident = True
    with pytest.raises(ValueError):
        case.worker.register_kv_caches(case.cache.caches)
    assert case.boundary.events == [] and case.boundary.regions == []


def test_existing_global_registration_cannot_publish_a_new_worker_layout(monkeypatch):
    case = make_registration_case(monkeypatch)
    case.boundary.is_register_buffer = True
    with pytest.raises(RuntimeError, match="already registered"):
        case.worker.register_kv_caches(case.cache.caches)
    assert case.worker.xfer_handshake_metadata is None
    assert case.boundary.regions == []


def test_a_second_native_engine_owner_is_rejected_before_constructor_side_effects(
    monkeypatch,
):
    boundary = RegistrationBoundary()
    boundary.transfer_engine = boundary
    monkeypatch.setattr(adapter, "global_te", boundary)
    with pytest.raises(RuntimeError, match="fresh native engine"):
        adapter.AFDMooncakeConnectorWorker(connector_config(), "local", None)
    assert boundary.regions == []


def test_attested_metadata_accepts_different_capacities_before_publishing_addresses(
    receiver_case,
):
    case = receiver_case
    case.receiver._get_remote_metadata("remote", "127.0.0.1", REMOTE_HANDSHAKE_PORT)
    assert peer_is_published(case)
    assert case.receiver.kv_caches_base_addr["remote"][REMOTE_HANDSHAKE_PORT] == [
        REMOTE_BASE_ADDRESS
    ]
    assert case.request_sent == [msgspec.msgpack.encode((adapter.GET_META_MSG, ""))]


@pytest.mark.parametrize(
    "invalid",
    [
        "legacy",
        "capacity",
        "stride",
        "addresses",
        "zero",
        "bool",
        "overflow",
        "block-size",
        "layout",
        "schema",
        "utf8",
        "engine",
        "runtime",
        "endpoint",
        "frame-limit",
        "layout-limit",
    ],
)
def test_bad_peer_metadata_never_publishes_native_addresses(receiver_case, invalid):
    case = receiver_case
    metadata = case.metadata
    if invalid == "legacy":
        case.metadata = adapter.MooncakeAgentMetadata(
            "remote", TRANSFER_PORT, 4, [REMOTE_BASE_ADDRESS], 5, [24], (0, 0)
        )
    elif invalid == "capacity":
        metadata.num_blocks += 1
    elif invalid == "stride":
        metadata.block_lens[0] += 1
    elif invalid == "addresses":
        metadata.kv_caches_base_addr = []
    elif invalid in ("zero", "bool", "overflow"):
        metadata.kv_caches_base_addr[0] = {
            "zero": 0,
            "bool": True,
            "overflow": adapter.MAX_TRANSFER_ADDRESS - 1,
        }[invalid]
    elif invalid == "block-size":
        metadata.block_size += 1
    elif invalid == "layout":
        metadata.afd_kv_layout = (
            replace(case.remote_layout, cache_layout="different").to_json().encode()
        )
    elif invalid == "schema":
        payload = json.loads(metadata.afd_kv_layout)
        payload["schema_version"] = 2
        metadata.afd_kv_layout = json.dumps(payload).encode()
    elif invalid == "utf8":
        metadata.afd_kv_layout = b"\xff"
    elif invalid == "engine":
        metadata.engine_id = "foreign"
    elif invalid == "runtime":
        metadata.runtime_id = "foreign-runtime"
    elif invalid == "endpoint":
        metadata.te_rpc_port = 0
    elif invalid == "frame-limit":
        case.frame = b"x" * (adapter.MAX_METADATA_BYTES + 1)
    else:
        metadata.afd_kv_layout = b" " * (adapter.MAX_LAYOUT_BYTES + 1)
    with pytest.raises((ValueError, RuntimeError, msgspec.ValidationError)):
        case.receiver._get_remote_metadata("remote", "127.0.0.1", REMOTE_HANDSHAKE_PORT)
    assert not peer_is_published(case)
    assert case.engine.reads == []


def test_a_changed_attested_peer_cannot_replace_its_published_storage(receiver_case):
    case = receiver_case
    case.receiver._get_remote_metadata("remote", "127.0.0.1", REMOTE_HANDSHAKE_PORT)
    case.metadata.kv_caches_base_addr[0] += 128
    with pytest.raises(RuntimeError, match="storage changed"):
        case.receiver._get_remote_metadata("remote", "127.0.0.1", REMOTE_HANDSHAKE_PORT)
    assert case.receiver.kv_caches_base_addr["remote"][REMOTE_HANDSHAKE_PORT] == [
        REMOTE_BASE_ADDRESS
    ]


def test_native_tail_projection_is_validated_then_read_as_one_complete_batch(
    receiver_case,
):
    case = receiver_case
    groups = len(case.layout.groups)
    req = request_map(case, source=[[0, 2, 3]] * groups, destination=[[0, 1]] * groups)
    case.receiver._transfer_kv_cache_all_groups(req)
    stride = case.layout.pages[0].block_stride_bytes
    assert case.engine.reads == [
        (
            f"127.0.0.1:{TRANSFER_PORT}",
            (case.address,) * groups,
            (REMOTE_BASE_ADDRESS + 2 * stride,) * groups,
            (2 * stride,) * groups,
        )
    ]
    assert req["remote_block_ids"] == [[0, 2, 3]] * groups


def test_published_local_arrays_cannot_change_the_proof_bound_read(receiver_case):
    case = receiver_case
    case.receiver.block_len_per_addr[0] += 128
    case.receiver.block_stride_per_addr[0] += 128
    case.receiver.addr_group_idx[0].clear()
    case.receiver.hma_group_size = 0
    case.receiver._transfer_kv_cache_all_groups(request_map(case))
    stride = case.layout.pages[0].block_stride_bytes
    groups = len(case.layout.groups)
    assert case.engine.reads == [
        (
            f"127.0.0.1:{TRANSFER_PORT}",
            (case.address + stride,) * groups,
            (REMOTE_BASE_ADDRESS + 3 * stride,) * groups,
            (stride,) * groups,
        )
    ]


@pytest.mark.parametrize(
    "invalid", ["late", "conflict", "tail-prefix", "short", "boolean", "missing"]
)
def test_any_invalid_request_group_rejects_the_entire_read(receiver_case, invalid):
    case = receiver_case
    req = request_map(case)
    if invalid == "late":
        req["local_block_ids"][-1] = [case.layout.num_blocks]
    elif invalid == "conflict":
        req["remote_block_ids"][-1] = [2]
    elif invalid == "tail-prefix":
        req["remote_block_ids"][-1] = [case.remote_layout.num_blocks, 3]
    elif invalid == "short":
        req["remote_block_ids"][-1] = []
    elif invalid == "boolean":
        req["remote_block_ids"][-1] = [True]
    else:
        req["local_block_ids"].pop()
    with pytest.raises(ValueError):
        case.receiver._transfer_kv_cache_all_groups(req)
    assert case.engine.reads == []


def test_empty_groups_do_not_turn_the_native_tail_slice_into_a_read(receiver_case):
    case = receiver_case
    req = request_map(case)
    req["local_block_ids"][0] = []
    case.receiver._transfer_kv_cache_all_groups(req)
    assert len(case.engine.reads[0][1]) == len(case.layout.groups) - 1


@pytest.mark.parametrize("empty", [False, True])
def test_native_failure_and_empty_retirement_preserve_done_protocol(
    receiver_case, empty
):
    case = receiver_case
    req = request_map(case)
    if empty:
        req["remote_block_ids"] = [[] for _ in case.layout.groups]
        req["local_block_ids"] = [[] for _ in case.layout.groups]
    else:
        req["local_block_ids"][-1] = [case.layout.num_blocks]
    receiver = case.receiver
    receiver.task_tracker.add_req_to_process("request")
    receiver.request_queue.put(req)
    receiver._mark_request_task_submitted(req)
    receiver._handle_request(receiver.request_queue.get())
    assert receiver.get_and_clear_finished_requests() == {"request"}
    assert receiver.done == [("free", "remote-request"), ("done", "remote-request")]
    assert case.engine.reads == []
    assert receiver.get_and_clear_invalid_block_ids() == (set() if empty else {1})


def test_mirrored_address_mutation_retires_the_proof_before_read(receiver_case):
    case = receiver_case
    case.receiver._get_remote_metadata("remote", "127.0.0.1", REMOTE_HANDSHAKE_PORT)
    case.receiver.kv_caches_base_addr["remote"][REMOTE_HANDSHAKE_PORT][0] += 128
    with pytest.raises(RuntimeError, match="outside its attestation"):
        case.receiver._transfer_kv_cache_all_groups(request_map(case))
    assert not peer_is_published(case)
    assert case.engine.reads == []


def test_evicted_native_mirror_refetches_the_whole_attestation(receiver_case):
    case = receiver_case
    case.receiver._get_remote_metadata("remote", "127.0.0.1", REMOTE_HANDSHAKE_PORT)
    case.receiver.remote_te_host.pop("remote")
    case.receiver._transfer_kv_cache_all_groups(request_map(case))
    assert case.request_sent == [msgspec.msgpack.encode((adapter.GET_META_MSG, ""))] * 2
    assert len(case.engine.reads) == 1


@pytest.fixture
def retiring_receiver_case(monkeypatch):
    case = make_retiring_receiver_case(monkeypatch)
    try:
        yield case
    finally:
        case.close()


def test_receiver_retirement_joins_input_after_real_native_done_ack(
    retiring_receiver_case,
):
    case = retiring_receiver_case
    case.submit()
    case.allow_ack.set()
    case.receiver.retire(timeout=THREAD_TEST_TIMEOUT)
    assert case.receiver.retired
    assert not case.receiver.is_alive()
    assert case.receiver.get_and_clear_finished_requests() == {"request"}
    assert case.engine.reads
    frames = [msgspec.msgpack.decode(data) for data in case.request_sent]
    assert frames[-1][0:2] == [adapter.DONE_RECVING_MSG, "remote-request"]
    case.receiver.retire(timeout=THREAD_TEST_TIMEOUT)
    with pytest.raises(RuntimeError, match="admission is closed"):
        case.receiver.add_request(**request_map(case))


def test_local_queue_done_cannot_replace_remote_ack_for_retirement(
    retiring_receiver_case,
):
    case = retiring_receiver_case
    case.submit()
    # Native task_done and local completion precede the blocked final ACK.
    case.receiver.request_queue.join()
    assert case.receiver.get_and_clear_finished_requests() == {"request"}
    with pytest.raises(TimeoutError, match="finish DONE/ACK"):
        case.receiver.retire(timeout=RETIREMENT_TEST_DEADLINE)
    assert not case.receiver.retired
    assert case.receiver.is_alive()
    with pytest.raises(RuntimeError, match="admission is closed"):
        case.receiver.add_request(**request_map(case))


def test_invalid_remote_ack_is_retained_as_retirement_failure(
    monkeypatch,
):
    case = make_retiring_receiver_case(monkeypatch, ack=b"NOT_ACK")
    try:
        case.submit()
        case.allow_ack.set()
        with pytest.raises(RuntimeError, match="DONE/ACK|retained failure"):
            case.receiver.retire(timeout=THREAD_TEST_TIMEOUT)
        assert not case.receiver.retired
        assert case.receiver.is_alive()
        assert case.receiver.engine is case.engine
        assert case.receiver.kv_caches is case.local.caches
    finally:
        case.close()


def test_native_none_queue_item_does_not_stop_the_receiver(
    retiring_receiver_case,
):
    receiver = retiring_receiver_case.receiver
    receiver.request_queue.put(None)
    receiver.retire(timeout=THREAD_TEST_TIMEOUT)
    assert receiver.retired
    assert not receiver.is_alive()


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_retirement_rejects_invalid_deadline_before_thread_work(
    retiring_receiver_case, timeout
):
    receiver = retiring_receiver_case.receiver
    with pytest.raises(ValueError, match="finite and positive"):
        receiver.retire(timeout=timeout)
    assert not receiver.retired
    receiver.retire(timeout=THREAD_TEST_TIMEOUT)


def test_receiver_must_be_started_before_retirement(monkeypatch):
    case = make_retiring_receiver_case(monkeypatch, start=False)
    try:
        with pytest.raises(RuntimeError, match="started external receiver"):
            case.receiver.retire(timeout=THREAD_TEST_TIMEOUT)
        assert not case.receiver.retired
    finally:
        case.close()


def test_worker_batch_finishes_native_done_before_closing_admission(
    retiring_receiver_case,
):
    case = attach_consumer_worker(retiring_receiver_case)
    case.allow_ack.set()
    case.worker.start_load_kv(case.batch)
    case.worker.retire_receiver(timeout=THREAD_TEST_TIMEOUT)
    assert case.receiver.retired
    assert case.receiver.get_and_clear_finished_requests() == {"first", "second"}
    assert case.engine.reads == []
    with pytest.raises(RuntimeError, match="admission is closed"):
        case.worker.start_load_kv(adapter.MooncakeConnectorMetadata())


def test_failed_native_batch_cannot_report_successful_retirement(
    retiring_receiver_case,
):
    case = attach_consumer_worker(retiring_receiver_case)
    case.allow_ack.set()
    case.batch.requests["second"].remote_port = "invalid"
    with pytest.raises(TypeError):
        case.worker.start_load_kv(case.batch)
    with pytest.raises(RuntimeError, match="retained failure"):
        case.worker.retire_receiver(timeout=THREAD_TEST_TIMEOUT)
    assert not case.receiver.retired
    assert case.receiver.engine is case.engine


def test_retirement_cannot_close_an_in_progress_batch(
    retiring_receiver_case,
):
    case = retiring_receiver_case
    gate = hold_receiver_admission(case)
    try:
        with pytest.raises(TimeoutError, match="batch admission"):
            case.receiver.retire(timeout=RETIREMENT_TEST_DEADLINE)
        assert case.receiver.is_alive()
        assert case.receiver.engine is case.engine
    finally:
        gate.release.set()
        gate.thread.join(timeout=THREAD_TEST_TIMEOUT)
    assert not gate.thread.is_alive()
    assert gate.errors == []
    assert not case.receiver.retired
    case.receiver.retire(timeout=THREAD_TEST_TIMEOUT)
    assert case.receiver.retired


def test_worker_cannot_retire_the_producer_receiver(retiring_receiver_case):
    case = attach_consumer_worker(retiring_receiver_case)
    case.worker.kv_role = "kv_producer"
    with pytest.raises(RuntimeError, match="registered attested consumer"):
        case.worker.retire_receiver(timeout=THREAD_TEST_TIMEOUT)
    assert not case.receiver.retired


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
def test_request_ready_wire_requires_complete_native_draft_projection(
    monkeypatch, tp_size, tp_rank
):
    prefix = make_dspark_pd_prefix_case(
        monkeypatch, tp_size=tp_size, tp_rank=tp_rank, bind=False
    )
    case = make_dspark_registration_case(monkeypatch, prefix)
    query = dspark_query(prefix)
    assert case.request_ready(query)["ok"] is False
    case.admit()
    first, state = prefix.make_step(0, 2)
    case.connector.publish_dspark_ready(
        prefix.consumer.consume_prefill(first, state, (prefix.source,))
    )
    assert case.request_ready(query)["ok"] is False
    second, state = prefix.make_step(2, 3)
    case.connector.publish_dspark_ready(
        prefix.consumer.consume_prefill(second, state, (prefix.source,))
    )
    lease = read_lease_from_payload(case.request_ready(query)["lease"])
    assert lease.ready == prefix.expected_ready
    assert lease.destination_block_ids == query.destination_block_ids
    for group_id, name in zip(prefix.draft_group_ids, prefix.draft_names, strict=True):
        size = prefix.kv_config.kv_cache_groups[group_id].kv_cache_spec.block_size
        actual = torch.stack(
            [
                prefix.caches[name][
                    prefix.source.allocation_block_ids[group_id][pos // size],
                    pos % size,
                    0,
                ]
                for pos in range(prefix.source.computed_prefix_tokens)
            ]
        )
        expected = torch.tensor(
            [[240 + 60 * pos, 246 + 60 * pos] for pos in range(5)], dtype=torch.bfloat16
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["prompt", "generation", "layout", "rank"])
def test_changed_ready_contract_is_rejected_without_poisoning_valid_query(
    dspark_case, invalid
):
    case, prefix = dspark_case, dspark_case.prefix_case
    publish_dspark_prefix(case)
    query = dspark_query(prefix)
    changes = {
        "prompt": {"prompt_digest": "0" * 64},
        "generation": {"source_generation": "other-generation"},
        "layout": {"layout_digest": "0" * 64},
        "rank": {"source_tp_rank": 1, "tp_size": 2},
    }
    wrong = replace(
        query, expected_ready=replace(query.expected_ready, **changes[invalid])
    )
    assert case.request_ready(wrong)["ok"] is False
    assert (
        read_lease_from_payload(case.request_ready(query)["lease"]).ready
        == prefix.expected_ready
    )


def test_finished_generation_cannot_release_a_reused_request_id(dspark_case):
    prefix = dspark_case.prefix_case
    next_source = replace(prefix.source, source_generation="source-generation-2")
    registry = adapter.DsparkPdReadyRegistry()
    registry.admit((prefix.source,))
    registry.publish((prefix.expected_ready,))
    lease = registry.acquire(dspark_query(prefix))
    registry.complete(lease)
    with pytest.raises(RuntimeError, match="not yet delivered"):
        registry.admit((next_source,))
    assert registry.take_finished() == {prefix.source.request_id}
    registry.admit((next_source,))
    registry.complete(lease)
    assert registry.take_finished() == set()
    next_ready = replace(prefix.expected_ready, source_generation="source-generation-2")
    registry.publish((next_ready,))
    query = replace(dspark_query(prefix), expected_ready=next_ready)
    assert registry.acquire(query).ready == next_ready


@pytest.mark.parametrize("with_lease", [False, True])
def test_cancelled_source_is_released_only_after_its_existing_lease(
    dspark_case, with_lease
):
    case, prefix = dspark_case, dspark_case.prefix_case
    lease = publish_dspark_prefix(case) if with_lease else None
    if not with_lease:
        case.admit()
    metadata = adapter.DsparkMooncakeConnectorMetadata()
    metadata.dspark_cancelled = (
        (prefix.source.request_id, prefix.source.source_generation),
    )
    case.worker.start_load_kv(metadata)
    assert case.request_ready(dspark_query(prefix))["ok"] is False
    sender = case.worker.kv_send_thread
    assert sender.get_and_clear_finished_requests() == (
        set() if with_lease else {prefix.source.request_id}
    )
    if with_lease:
        assert case.complete_read(lease) == b"ACK"
        assert sender.get_and_clear_finished_requests() == {prefix.source.request_id}
        assert case.complete_read(lease) == b"ACK"
        assert sender.get_and_clear_finished_requests() == set()


@pytest.mark.parametrize("known_source", [False, True])
def test_legacy_done_cannot_retire_a_dspark_source(dspark_case, known_source):
    case, prefix = dspark_case, dspark_case.prefix_case
    lease = publish_dspark_prefix(case)
    reply = sender_request(
        case.worker.kv_send_thread,
        adapter.DONE_RECVING_MSG,
        prefix.source.request_id if known_source else "unknown-or-evicted-source",
    )
    assert reply["ok"] is False
    assert case.worker.kv_send_thread.get_and_clear_finished_requests() == set()
    assert case.complete_read(lease) == b"ACK"
    assert case.worker.kv_send_thread.get_and_clear_finished_requests() == {
        prefix.source.request_id
    }


def test_source_batch_rejection_does_not_admit_an_earlier_source(dspark_case):
    prefix = dspark_case.prefix_case
    registry = adapter.DsparkPdReadyRegistry()
    registry.admit((prefix.source,))
    other = replace(prefix.source, request_id="other-request")
    changed = replace(prefix.source, original_prompt_tokens=7)
    with pytest.raises(ValueError, match="changed its contract"):
        registry.admit((other, changed))
    other_ready = replace(prefix.expected_ready, source_request_id="other-request")
    with pytest.raises(KeyError):
        registry.publish((other_ready,))
    registry.publish((prefix.expected_ready,))
    assert registry.acquire(dspark_query(prefix)).ready == prefix.expected_ready


@pytest.mark.parametrize("read_result", [0, -1])
def test_destination_completion_requires_a_drained_read_and_typed_done(
    monkeypatch, read_result
):
    prefix = make_dspark_pd_prefix_case(monkeypatch, bind=False)
    case = make_dspark_read_case(monkeypatch, prefix)
    publish_dspark_prefix(case.producer, acquire=False)
    case.engine.result = read_result
    try:
        if read_result < 0:
            with pytest.raises(RuntimeError):
                case.process()
            assert case.receiver.get_and_clear_finished_requests() == set()
            with (
                pytest.raises(RuntimeError, match="closed"),
                case.receiver.admission_scope(),
            ):
                pass
            assert adapter.DONE_DSPARK_RECV_MSG not in [
                msgspec.msgpack.decode(frame)[0] for frame in case.sent
            ]
            assert (
                case.producer.worker.kv_send_thread.get_and_clear_finished_requests()
                == set()
            )
        else:
            case.process()
            assert case.receiver.get_and_clear_finished_requests() == {"request"}
            assert [msgspec.msgpack.decode(frame)[0] for frame in case.sent] == [
                adapter.GET_META_MSG,
                adapter.GET_DSPARK_READY_MSG,
                adapter.DONE_DSPARK_RECV_MSG,
            ]
            assert (
                case.producer.worker.kv_send_thread.get_and_clear_finished_requests()
                == {prefix.source.request_id}
            )
    finally:
        case.close()


@pytest.mark.parametrize(
    "destination_blocks,destination_offset,reverse_destination",
    [(4, 0, False), (8, 2, True)],
)
def test_pd_read_copies_independent_pages(  # long-test: whole-page transfer oracle
    monkeypatch, destination_blocks, destination_offset, reverse_destination
):
    prefix = make_dspark_pd_prefix_case(monkeypatch, bind=False)
    case = make_dspark_read_case(
        monkeypatch,
        prefix,
        destination_blocks=destination_blocks,
        destination_offset=destination_offset,
        reverse_destination=reverse_destination,
        copy_payload=True,
    )
    source_pages = case.destination_cache.source_pages
    for index, page in enumerate(source_pages):
        positions = torch.arange(page.numel())
        physical_blocks = positions // prefix.layout.pages[index].block_stride_bytes
        page.copy_(
            ((positions + physical_blocks + index) % BYTE_VALUE_COUNT).to(torch.uint8)
        )
    publish_dspark_prefix(case.producer, acquire=False)
    before = tuple(page.clone() for page in source_pages)
    expected = [
        torch.full_like(page, DESTINATION_SENTINEL)
        for page in case.destination_cache.pages
    ]
    for page_index, page in enumerate(prefix.layout.pages):
        stride = page.block_stride_bytes
        for group_index, group in enumerate(prefix.layout.groups):
            if not set(page.layers) & set(group.layer_names):
                continue
            for source_id, destination_id in zip(
                prefix.source.transfer_block_ids[group_index],
                case.destination.block_ids[group_index],
                strict=True,
            ):
                expected[page_index][
                    destination_id * stride : (destination_id + 1) * stride
                ] = before[page_index][source_id * stride : (source_id + 1) * stride]
    try:
        assert {page.data_ptr() for page in source_pages}.isdisjoint(
            page.data_ptr() for page in case.destination_cache.pages
        )
        case.process()
        for source, unchanged in zip(source_pages, before, strict=True):
            torch.testing.assert_close(source, unchanged, rtol=0, atol=0)
        for actual, target in zip(case.destination_cache.pages, expected, strict=True):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
        assert [msgspec.msgpack.decode(frame)[0] for frame in case.sent] == [
            adapter.GET_META_MSG,
            adapter.GET_DSPARK_READY_MSG,
            adapter.DONE_DSPARK_RECV_MSG,
        ]
        assert case.receiver.get_and_clear_finished_requests() == {"request"}
        assert (
            case.producer.worker.kv_send_thread.get_and_clear_finished_requests()
            == {prefix.source.request_id}
        )
    finally:
        case.close()


@pytest.mark.parametrize("invalid", ["rank", "source", "empty"])
def test_invalid_destination_contract_is_rejected_before_read_lease(
    monkeypatch, invalid
):
    prefix = make_dspark_pd_prefix_case(monkeypatch, bind=False)
    case = make_dspark_read_case(monkeypatch, prefix)
    publish_dspark_prefix(case.producer, acquire=False)
    if invalid == "rank":
        metadata = case.producer.worker.xfer_handshake_metadata
        metadata.afd_tp_rank, metadata.afd_tp_size = 1, 2
        case.receiver.tp_size = 2
    elif invalid == "source":
        case.request["remote_engine_id"] = "other-engine"
    else:
        case.request["remote_block_ids"] = [[] for _ in prefix.layout.groups]
        case.request["local_block_ids"] = [[] for _ in prefix.layout.groups]
    try:
        if invalid == "empty":
            with pytest.raises(ValueError, match="mapping changed"):
                case.process()
        else:
            case.process()
        assert case.engine.reads == []
        assert adapter.GET_DSPARK_READY_MSG not in [
            msgspec.msgpack.decode(frame)[0] for frame in case.sent
        ]
        assert (
            case.producer.worker.kv_send_thread.get_and_clear_finished_requests()
            == set()
        )
    finally:
        case.close()


def test_running_chunks_snapshot_swa_nulls(monkeypatch):  # long-test: native KV
    prefix = make_dspark_pd_prefix_case(
        monkeypatch, original_tokens=18, num_blocks=64, bind=False
    )
    case = make_dspark_scheduler_case(monkeypatch, prefix)
    request, manager, connector = case.request, case.manager, case.scheduler
    snapshots = []
    for offset, count in ((0, 5), (5, 4), (9, 8)):
        request.num_computed_tokens = offset
        assert manager.allocate_slots(request, count) is not None
        if offset == 0:
            connector.update_state_after_alloc(
                request, manager.get_blocks(request.request_id), 0
            )
        scheduled, _ = prefix.make_step(offset, count)
        source = connector.build_connector_meta(scheduled).dspark_sources[0]
        snapshots.append(source)
        assert source.allocation_block_ids == tuple(
            tuple(ids) for ids in manager.get_block_ids(request.request_id)
        )
        assert source.prompt_digest == prefix.source.prompt_digest
    assert snapshots[0].allocation_block_ids != snapshots[-1].allocation_block_ids
    assert any(
        0 in snapshots[-1].allocation_block_ids[group]
        for group in prefix.draft_group_ids
    )
    assert all(0 not in group for group in snapshots[-1].transfer_block_ids)
    assert (
        snapshots[0].computed_prefix_tokens
        == snapshots[-1].computed_prefix_tokens
        == 17
    )


def test_preempted_source_cannot_publish_metadata_or_resume_old_allocation(monkeypatch):
    prefix = make_dspark_pd_prefix_case(monkeypatch, bind=False, num_blocks=32)
    case = make_dspark_scheduler_case(monkeypatch, prefix)
    assert case.manager.allocate_slots(case.request, 2) is not None
    case.scheduler.update_state_after_alloc(
        case.request, case.manager.get_blocks(case.request.request_id), 0
    )
    scheduled, _ = prefix.make_step(0, 2)
    case.scheduler.build_connector_meta(scheduled)
    case.manager.free(case.request)
    scheduled.preempted_req_ids = {case.request.request_id}
    with pytest.raises(RuntimeError, match="preempted"):
        case.scheduler.build_connector_meta(scheduled)
    scheduled.preempted_req_ids = set()
    with pytest.raises(RuntimeError, match="terminal"):
        case.scheduler.build_connector_meta(scheduled)


@pytest.mark.parametrize("cancelled", [False, True])
def test_finished_request_id_reuse_has_a_fresh_prompt_and_allocation(
    monkeypatch, cancelled
):
    prefix = make_dspark_pd_prefix_case(monkeypatch, bind=False, num_blocks=32)
    case = make_dspark_scheduler_case(monkeypatch, prefix)
    request, manager, connector = case.request, case.manager, case.scheduler
    assert manager.allocate_slots(request, request.num_prompt_tokens) is not None
    connector.update_state_after_alloc(
        request, manager.get_blocks(request.request_id), 0
    )
    scheduled, _ = prefix.make_step(0, request.num_prompt_tokens)
    old_source = connector.build_connector_meta(scheduled).dspark_sources[0]
    request.append_output_token_ids(19)
    request.status = (
        RequestStatus.FINISHED_ABORTED
        if cancelled
        else RequestStatus.FINISHED_LENGTH_CAPPED
    )
    delayed, params = connector.request_finished_all_groups(
        request, manager.get_block_ids(request.request_id)
    )
    assert delayed
    if cancelled:
        metadata = connector.build_connector_meta(scheduled)
        assert metadata.dspark_cancelled == (
            (request.request_id, old_source.source_generation),
        )
    else:
        assert params[adapter.DSPARK_SOURCE_PARAM] == source_request_payload(old_source)
    manager.free(request)
    replacement = case.new_request(list(range(100, 108)))
    assert (
        manager.allocate_slots(replacement, replacement.num_prompt_tokens) is not None
    )
    connector.update_state_after_alloc(
        replacement, manager.get_blocks(replacement.request_id), 0
    )
    source = connector.build_connector_meta(scheduled).dspark_sources[0]
    assert source.source_generation != old_source.source_generation
    assert source.prompt_digest != old_source.prompt_digest
    assert source.original_prompt_tokens == 8
    assert source.computed_prefix_tokens == 7
    assert source.allocation_block_ids == tuple(
        tuple(ids) for ids in manager.get_block_ids(replacement.request_id)
    )


@pytest.mark.parametrize("invalid", ["prompt", "prefix", "blocks"])
def test_destination_scheduler_rejects_a_wrong_source_before_queue_admission(
    monkeypatch, invalid
):
    prefix = make_dspark_pd_prefix_case(monkeypatch, bind=False)
    case = make_dspark_scheduler_case(monkeypatch, prefix, role="kv_consumer")
    source = prefix.source
    source = replace(
        source,
        **{
            "prompt": {"prompt_digest": "0" * 64},
            "prefix": {"computed_prefix_tokens": 4},
            "blocks": {"source_engine_id": "other-engine"},
        }[invalid],
    )
    params = {
        "do_remote_prefill": True,
        "remote_request_id": source.request_id,
        "remote_engine_id": prefix.source.source_engine_id,
        "remote_block_ids": prefix.source.transfer_block_ids,
        adapter.DSPARK_SOURCE_PARAM: source_request_payload(source),
    }
    with pytest.raises(ValueError, match="does not match this prompt"):
        case.new_request(
            list(range(prefix.source.original_prompt_tokens)), params=params
        )


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
def test_target_only_connector_keeps_native_noop_shutdown(monkeypatch, role):
    case = make_registration_case(monkeypatch, role=role)
    case.worker.register_kv_caches(case.cache.caches)
    assert case.connector.shutdown() is None


def test_producer_shutdown_closes_ready_admission_after_real_lease_done(dspark_case):
    case = dspark_case
    lease = publish_dspark_prefix(case)
    assert case.complete_read(lease) == b"ACK"
    assert case.connector.shutdown() is None
    assert case.connector.shutdown() is None
    assert case.request_ready(dspark_query(case.prefix_case))["ok"] is False
    with pytest.raises(RuntimeError, match="closed"):
        case.admit()


@pytest.mark.parametrize("with_lease", [False, True])
def test_producer_shutdown_cannot_release_an_active_source(dspark_case, with_lease):
    case = dspark_case
    lease = publish_dspark_prefix(case, acquire=with_lease)
    tensor = case.prefix_case.caches[case.prefix_case.draft_names[0]]
    before = tensor.clone()
    with pytest.raises(RuntimeError, match="not drained"):
        case.connector.shutdown()
    assert case.request_ready(dspark_query(case.prefix_case))["ok"] is False
    with pytest.raises(RuntimeError, match="closed"):
        case.admit()
    if with_lease:
        assert case.complete_read(lease) == b"ACK"
        assert case.worker.kv_send_thread.get_and_clear_finished_requests() == {
            case.prefix_case.source.request_id
        }
    with pytest.raises(RuntimeError, match="not drained"):
        case.connector.shutdown()
    torch.testing.assert_close(tensor, before, rtol=0, atol=0)


@pytest.mark.parametrize("acknowledged", [True, False])
def test_consumer_shutdown_requires_the_actual_typed_done_ack(
    monkeypatch, acknowledged
):
    prefix = make_dspark_pd_prefix_case(monkeypatch, bind=False)
    case = make_retiring_dspark_read_case(
        monkeypatch, prefix, done_ack=b"ACK" if acknowledged else b"NACK"
    )
    publish_dspark_prefix(case.producer, acquire=False)
    try:
        case.submit()
        if acknowledged:
            assert case.connector.shutdown() is None
            assert case.receiver.retired
            assert not case.receiver.is_alive()
            assert case.receiver.get_and_clear_finished_requests() == {"request"}
            assert (
                case.producer.worker.kv_send_thread.get_and_clear_finished_requests()
                == {prefix.source.request_id}
            )
        else:
            with pytest.raises(RuntimeError, match="DONE/ACK|retained failure"):
                case.connector.shutdown()
            assert not case.receiver.retired
            assert case.receiver.is_alive()
            assert case.receiver.get_and_clear_finished_requests() == set()
            assert (
                case.producer.worker.kv_send_thread.get_and_clear_finished_requests()
                == set()
            )
        with (
            pytest.raises(RuntimeError, match="closed"),
            case.receiver.admission_scope(),
        ):
            pass
    finally:
        case.close()
