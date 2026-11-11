# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU protocol boundaries only: no Mooncake read or device registration.

Native typed specs and real CPU views come from kv_cache. The logical wire
record is labeled NPU explicitly for schema tests; this never verifies NPU
storage, transfer-engine registration or execution.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import msgspec
import zmq
from transformers import PretrainedConfig

from afd_plugin.compat.npu import kv_layout_connector as adapter
from afd_plugin.compat.npu.kv_layout_native import collect_deepseek_v41_kv_layout
from tests.unit.fakes.kv_cache import make_native_kv_cache_case
from tests.unit.fakes.vllm_config import npu_vllm_config, pd_kv_transfer_config

REMOTE_BASE_ADDRESS = 1_048_576
REMOTE_HANDSHAKE_PORT = 9001
LOCAL_HANDSHAKE_PORT = 9000
TRANSFER_PORT = 9876
THREAD_TEST_TIMEOUT = 5.0


def connector_config(*, role="kv_consumer"):
    transfer = pd_kv_transfer_config()
    transfer.kv_connector_module_path = "afd_plugin.compat.npu.kv_layout_connector"
    transfer.kv_role = role
    transfer.engine_id = "local"
    config = npu_vllm_config(
        model_type="deepseek_v41",
        architectures=["DeepseekV41ForCausalLM"],
        cudagraph_mode="NONE",
        kv_transfer_config=transfer,
    )
    config.model_config.is_deepseek_mla = True
    config.model_config.hf_config.compress_ratios = [2]
    config.model_config.hf_config.num_hidden_layers = 1
    text_config = PretrainedConfig(**vars(config.model_config.hf_config))
    config.model_config.hf_config = config.model_config.hf_text_config = text_config
    config.cache_config.block_size = 4
    return config


def wire_layout(case):
    return replace(
        collect_deepseek_v41_kv_layout(case.config, case.caches), device_type="npu"
    )


class ReadBoundary:
    def __init__(self):
        self.reads = []
        self.result = 0

    def batch_transfer_sync_read(self, session, local, remote, lengths):
        self.reads.append((session, tuple(local), tuple(remote), tuple(lengths)))
        return self.result


class RegistrationBoundary:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer = False
        self.regions = []
        self.events = []
        self.fail_registration = False
        self.fail_start = False

    def register_buffer(self, pointers, lengths):
        self.regions = list(zip(pointers, lengths, strict=True))
        self.events.append("registered")
        if self.fail_registration:
            raise RuntimeError("controlled memory registration failure")
        self.is_register_buffer = True

    def sender(self, *args):
        return _StartBoundary(self, args[7], args[6])

    def receiver(self, *args, local_layout):
        return _StartBoundary(self, args[15], local_layout)


class _StartBoundary:
    def __init__(self, owner, ready_event, metadata):
        self.owner = owner
        self.ready_event = ready_event
        self.metadata = metadata

    def start(self):
        self.owner.events.append("started")
        if self.owner.fail_start:
            raise RuntimeError("controlled thread start failure")
        self.ready_event.set()


def make_registration_case(monkeypatch, *, role="kv_producer"):
    cache = make_native_kv_cache_case()
    config = connector_config(role=role)
    worker = adapter.AFDMooncakeConnectorWorker.__new__(
        adapter.AFDMooncakeConnectorWorker
    )
    values = {
        "vllm_config": config,
        "kv_cache_config": cache.config,
        "kv_role": role,
        "use_hybrid": True,
        "use_mamba": False,
        "use_compress": True,
        "block_size": 4,
        "engine_id": "local",
        "te_rpc_port": TRANSFER_PORT,
        "mooncake_runtime_id": "same-runtime",
        "tp_rank": 0,
        "tp_size": 1,
        "pcp_rank": 0,
        "side_channel_host": "127.0.0.1",
        "side_channel_port": LOCAL_HANDSHAKE_PORT,
        "handshake_port": LOCAL_HANDSHAKE_PORT,
        "hma_group_size": len(cache.config.kv_cache_groups),
        "engine": ReadBoundary(),
        "kv_send_thread": None,
        "kv_recv_thread": None,
        "xfer_handshake_metadata": None,
        "_prefill_tp_size": 1,
        "_prefill_pp_size": 1,
        "_prefill_pp_layer_partition": None,
        "_mamba_ssm_size": (0, 0),
        "_registration_started": False,
        "_registered_layout": None,
    }
    vars(worker).update(values)
    boundary = RegistrationBoundary()
    monkeypatch.setattr(adapter, "global_te", boundary)
    monkeypatch.setattr(adapter, "KVCacheSendingThread", boundary.sender)
    monkeypatch.setattr(adapter, "AFDMooncakeRecvingThread", boundary.receiver)
    monkeypatch.setattr(
        adapter,
        "collect_deepseek_v41_kv_layout",
        lambda cfg, tensors: wire_layout(cache),
    )
    return SimpleNamespace(worker=worker, cache=cache, boundary=boundary)


class _ReceiverNetworkBoundary(adapter.AFDMooncakeRecvingThread):
    def _get_remote_socket(self, host, port):
        return self.socket

    def _return_remote_socket(self, socket, host, port):
        pass

    def _send_done_signal_to_free_remote_port(self, request_id, host, ports):
        self.done.append(("free", request_id))

    def _send_done_recv_signal(self, request_id, host, port, ports):
        self.done.append(("done", request_id))


def make_receiver_case(
    monkeypatch,
    *,
    source_blocks=5,
    destination_blocks=3,
    receiver_type=_ReceiverNetworkBoundary,
):
    local = make_native_kv_cache_case(num_blocks=destination_blocks)
    remote = make_native_kv_cache_case(num_blocks=source_blocks)
    layout = wire_layout(local)
    remote_layout = wire_layout(remote)
    address = min(
        t.data_ptr()
        for v in local.caches.values()
        for t in (v if isinstance(v, tuple) else (v,))
    )
    config = connector_config()
    engine = ReadBoundary()
    receiver = receiver_type(
        0,
        1,
        1,
        engine,
        "local",
        LOCAL_HANDSHAKE_PORT,
        LOCAL_HANDSHAKE_PORT,
        [address],
        [layout.pages[0].block_stride_bytes],
        [layout.pages[0].block_stride_bytes],
        [list(range(len(layout.groups)))],
        (0, 0),
        True,
        False,
        len(layout.groups),
        threading.Event(),
        config,
        local.config,
        local.caches,
        mooncake_runtime_id="same-runtime",
        local_layout=layout,
    )
    context = zmq.Context()
    receiver.socket = context.socket(zmq.REQ)
    receiver.done = []
    metadata = adapter.AttestedMooncakeAgentMetadata(
        engine_id="remote",
        te_rpc_port=TRANSFER_PORT,
        block_size=4,
        kv_caches_base_addr=[REMOTE_BASE_ADDRESS],
        num_blocks=source_blocks,
        block_lens=[remote_layout.pages[0].block_stride_bytes],
        ssm_sizes=(0, 0),
        local_ip="127.0.0.1",
        runtime_id="same-runtime",
        afd_kv_layout=remote_layout.to_json().encode(),
    )
    case = SimpleNamespace(
        receiver=receiver,
        local=local,
        remote=remote,
        layout=layout,
        remote_layout=remote_layout,
        metadata=metadata,
        engine=engine,
        address=address,
        request_sent=[],
        frame=None,
    )
    monkeypatch.setattr(
        adapter,
        "ensure_zmq_send",
        lambda sock, data, path: case.request_sent.append(data),
    )
    monkeypatch.setattr(
        adapter,
        "ensure_zmq_recv",
        lambda sock, path: (
            case.frame
            if case.frame is not None
            else msgspec.msgpack.encode(case.metadata)
        ),
    )
    case.close = lambda: (
        receiver.executor.shutdown(wait=True),
        context.destroy(linger=0),
    )
    return case


class _ReceiverSocketBoundary(adapter.AFDMooncakeRecvingThread):
    """Keep native run/handle/DONE; replace only the CPU transport socket."""

    def _get_remote_socket(self, host, port):
        return self.socket

    def _return_remote_socket(self, socket, host, port):
        pass


def make_retiring_receiver_case(monkeypatch, *, ack=b"ACK", start=True):
    case = make_receiver_case(monkeypatch, receiver_type=_ReceiverSocketBoundary)
    receiver = case.receiver
    receiver.executor.shutdown(wait=True)
    # This is a real CPU thread seam, not a simulated NPU executor.
    receiver.executor = ThreadPoolExecutor(max_workers=1)
    case.ack_entered = threading.Event()
    case.allow_ack = threading.Event()
    case.ack = ack

    def receive(socket, path):
        kind = msgspec.msgpack.decode(case.request_sent[-1])[0]
        if kind == adapter.GET_META_MSG:
            return msgspec.msgpack.encode(case.metadata)
        if kind != adapter.DONE_RECVING_MSG:
            raise AssertionError("unexpected native CPU control frame")
        case.ack_entered.set()
        if not case.allow_ack.wait(THREAD_TEST_TIMEOUT):
            raise RuntimeError("CPU test did not release the ACK gate")
        return case.ack

    monkeypatch.setattr(adapter, "ensure_zmq_recv", receive)
    original_close = case.close

    def close():
        case.allow_ack.set()
        if receiver.is_alive():
            # Test-only cleanup of a deliberately failed retirement. Production
            # failure retains this owner for the outer process supervisor.
            receiver.request_queue.put(adapter._ReceiverControl.RETIRE)
            receiver.join(THREAD_TEST_TIMEOUT)
            if receiver.is_alive():
                raise AssertionError("CPU test receiver did not exit")
        original_close()

    case.close = close
    case.submit = lambda: submit_receiver_request(case)
    if start:
        receiver.start()
        if not receiver.ready_event.wait(THREAD_TEST_TIMEOUT):
            raise AssertionError("CPU test receiver did not become ready")
    return case


def submit_receiver_request(case):
    request = request_map(case)
    # The native worker publishes the parent tracker within the same admission
    # scope as queue publication. Reproduce that public protocol here.
    with case.receiver.admission_scope():
        case.receiver.task_tracker.add_req_to_process(request["request_id"])
        case.receiver.add_request(**request)
    if not case.ack_entered.wait(THREAD_TEST_TIMEOUT):
        raise AssertionError("native DONE did not reach the CPU ACK gate")


def attach_consumer_worker(case):
    """Native TP1 batch construction using the actual started CPU receiver."""
    worker = adapter.AFDMooncakeConnectorWorker.__new__(
        adapter.AFDMooncakeConnectorWorker
    )
    vars(worker).update(
        kv_role="kv_consumer",
        kv_recv_thread=case.receiver,
        kv_send_thread=None,
        vllm_config=case.receiver.vllm_config,
        use_mamba=False,
        tp_rank=0,
        tp_size=1,
        _decode_tp_size=1,
        _prefill_tp_size=1,
        _prefill_pp_size=1,
        tp_num_need_pulls=1,
    )
    metadata = adapter.MooncakeConnectorMetadata()
    for request_id in ("first", "second"):
        metadata.add_new_req(
            request_id,
            [[] for _ in case.layout.groups],
            0,
            {
                "remote_block_ids": [[] for _ in case.layout.groups],
                "remote_engine_id": "remote",
                "remote_request_id": "remote-" + request_id,
                "remote_host": "127.0.0.1",
                "remote_port": REMOTE_HANDSHAKE_PORT,
                "remote_ptp_size": 1,
                "remote_pcp_size": 1,
            },
        )
        metadata.reqs_in_batch.add(request_id)
    case.worker = worker
    case.batch = metadata
    return case


def hold_receiver_admission(case):
    """A caller-owned batch gate; all test waits are explicit Event handshakes."""
    entered = threading.Event()
    release = threading.Event()
    errors = []

    def hold():
        try:
            with case.receiver.admission_scope():
                entered.set()
                if not release.wait(THREAD_TEST_TIMEOUT):
                    raise AssertionError("CPU test did not release batch admission")
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=hold)
    thread.start()
    if not entered.wait(THREAD_TEST_TIMEOUT):
        raise AssertionError("CPU test batch did not acquire admission")
    return SimpleNamespace(thread=thread, release=release, errors=errors)


def request_map(case, *, source=None, destination=None):
    count = len(case.layout.groups)
    return {
        "request_id": "request",
        "remote_request_id": "remote-request",
        "local_block_ids": [[1] for _ in range(count)]
        if destination is None
        else destination,
        "remote_block_ids": [[3] for _ in range(count)] if source is None else source,
        "remote_engine_id": "remote",
        "remote_host": "127.0.0.1",
        "remote_handshake_port": REMOTE_HANDSHAKE_PORT,
        "remote_port_send_num": {},
        "all_task_done": True,
        "offset": 0,
        "tp_num_need_pulls": 1,
    }


def peer_is_published(case):
    receiver = case.receiver
    return (
        "remote" in receiver.kv_caches_base_addr
        and REMOTE_HANDSHAKE_PORT in receiver.kv_caches_base_addr["remote"]
    )
