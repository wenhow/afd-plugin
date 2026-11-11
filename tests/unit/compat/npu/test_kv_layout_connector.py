# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# ruff: noqa: E402
"""Actual codec/native metadata and CPU thread seams; no engine/NPU execution."""

import json
from dataclasses import replace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")
msgspec = pytest.importorskip("msgspec")

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

from afd_plugin.compat.npu import kv_layout_connector as adapter
from afd_plugin.compat.npu.kv_layout import layout_from_json
from afd_plugin.compat.npu.kv_layout_native import collect_deepseek_v41_kv_layout
from tests.unit.fakes.kv_layout_connector import (
    REMOTE_BASE_ADDRESS,
    REMOTE_HANDSHAKE_PORT,
    THREAD_TEST_TIMEOUT,
    TRANSFER_PORT,
    RegistrationBoundary,
    attach_consumer_worker,
    connector_config,
    hold_receiver_admission,
    make_receiver_case,
    make_registration_case,
    make_retiring_receiver_case,
    peer_is_published,
    request_map,
)

RETIREMENT_TEST_DEADLINE = 0.01


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
