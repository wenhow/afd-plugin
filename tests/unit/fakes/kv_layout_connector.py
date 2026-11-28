# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU protocol boundaries only: no Mooncake read or device registration.

Native typed specs and real CPU views come from kv_cache. The logical wire
record is labeled NPU explicitly for schema tests; this never verifies NPU
storage, transfer-engine registration or execution.
"""

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import msgspec
import torch
import zmq
from transformers import PretrainedConfig
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p import (
    mooncake_hybrid_connector as native_mooncake,
)

from afd_plugin.compat.npu import kv_layout_connector as adapter
from afd_plugin.compat.npu.dspark_pd import (
    DsparkPdReadQuery,
    read_lease_from_payload,
    read_lease_payload,
    read_query_payload,
    source_request_payload,
)
from afd_plugin.compat.npu.kv_layout_connector import (
    AFDMooncakeRecvingThread,
    AFDMooncakeSendingThread,
)
from afd_plugin.compat.npu.kv_layout_native import collect_deepseek_v41_kv_layout
from tests.unit.fakes.dspark_pd import make_dspark_pd_loaded_case
from tests.unit.fakes.kv_cache import make_native_kv_cache_case
from tests.unit.fakes.vllm_config import (
    configure_hccl_ep,
    npu_vllm_config,
    pd_kv_transfer_config,
)

REMOTE_BASE_ADDRESS = 1_048_576
REMOTE_HANDSHAKE_PORT = 9001
LOCAL_HANDSHAKE_PORT = 9000
TRANSFER_PORT = 9876
THREAD_TEST_TIMEOUT = 5.0
DSPARK_CPU_COMPRESS_RATIO = 2
DESTINATION_SENTINEL = 165
SCHEDULED_DESTINATION_BLOCKS = 64
SCHEDULED_DESTINATION_REQUEST = "destination-request"
REUSED_PREFIX_CONTEXT_OFFSET = 128
SOURCE_COMPLETION_TOKEN = 19


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


class CopyReadBoundary(ReadBoundary):
    """CPU copy leaf for real registered-address batches, with strong storage refs.

    This is neither a Mooncake engine nor a device DMA simulation. Address
    coalescing and request validation remain in the actual receiver body.
    """

    def __init__(self, source_pages, destination_pages):
        super().__init__()
        self.source_pages, self.destination_pages = source_pages, destination_pages

    @staticmethod
    def _bytes_at(pages, address, length):
        for page in pages:
            start = address - page.data_ptr()
            if start >= 0 and start + length <= page.numel():
                return page[start : start + length]
        raise ValueError("CPU copy span exceeds its registered physical page")

    def batch_transfer_sync_read(self, session, local, remote, lengths):
        result = super().batch_transfer_sync_read(session, local, remote, lengths)
        if result < 0:
            return result
        spans = [
            (
                self._bytes_at(self.destination_pages, local_address, length),
                self._bytes_at(self.source_pages, remote_address, length),
            )
            for local_address, remote_address, length in zip(
                local, remote, lengths, strict=True
            )
        ]
        for destination, source in spans:
            destination.copy_(source)
        return result


def _independent_destination_cache(prefix, num_blocks):
    """Re-materialize the existing typed CPU fixture, preserving page aliases.

    Each descriptor in this shared fixture owns one storage. This helper reuses
    its typed specs and materialized per-page view ABI; it does not reproduce
    the upstream allocator or construct a second loaded model/speculator.
    """
    layout = prefix.layout
    if len({page.storage_alias for page in layout.pages}) != len(layout.pages):
        raise ValueError("CPU destination fixture requires independent page storage")
    source_pages, destination_pages = [], []
    for page in layout.pages:
        if page.offset_bytes or page.storage_offset_bytes:
            raise ValueError("CPU destination fixture requires zero-offset pages")
        value = prefix.caches[page.layers[0]]
        tensor = value[0] if isinstance(value, (tuple, list)) else value
        source_pages.append(
            torch.empty(0, dtype=torch.uint8).set_(
                tensor.untyped_storage(), 0, (page.size_bytes,), (1,)
            )
        )
        destination_pages.append(
            torch.full(
                (num_blocks * page.block_stride_bytes,),
                DESTINATION_SENTINEL,
                dtype=torch.uint8,
            )
        )
    components = {}
    for view in layout.views:
        value = prefix.caches[view.layer_name]
        original = (
            value[view.tensor_index] if isinstance(value, (tuple, list)) else value
        )
        components[(view.layer_name, view.tensor_index)] = original.new_empty(0).set_(
            destination_pages[view.page_index].untyped_storage(),
            view.page_offset_bytes // view.element_bytes,
            (num_blocks, *view.shape[1:]),
            tuple(stride // view.element_bytes for stride in view.strides_bytes),
        )
    caches = {
        name: (
            tuple(components[(name, index)] for index in range(len(value)))
            if isinstance(value, (tuple, list))
            else components[(name, 0)]
        )
        for name, value in prefix.caches.items()
    }
    config = replace(
        prefix.kv_config,
        num_blocks=num_blocks,
        kv_cache_tensors=[
            replace(descriptor, size=num_blocks * page.block_stride_bytes)
            for descriptor, page in zip(
                prefix.kv_config.kv_cache_tensors, layout.pages, strict=True
            )
        ],
    )
    return SimpleNamespace(
        config=config,
        caches=caches,
        layout=collect_deepseek_v41_kv_layout(config, caches),
        source_pages=tuple(source_pages),
        pages=tuple(destination_pages),
    )


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

    def receiver(self, *args, local_layout, draft_layer_names=()):
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


def make_registration_case(monkeypatch, *, role="kv_producer", receiver_boundary=True):
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
        "pcp_size": config.parallel_config.prefill_context_parallel_size,
        "side_channel_host": "127.0.0.1",
        "side_channel_port": LOCAL_HANDSHAKE_PORT,
        "handshake_port": LOCAL_HANDSHAKE_PORT,
        "hma_group_size": len(cache.config.kv_cache_groups),
        "engine": ReadBoundary(),
        "kv_send_thread": None,
        "kv_recv_thread": None,
        "xfer_handshake_metadata": None,
        "_prefill_tp_size": 1,
        "_decode_tp_size": config.kv_transfer_config.kv_connector_extra_config[
            "decode"
        ]["tp_size"],
        "_prefill_pp_size": 1,
        "_prefill_pp_layer_partition": None,
        "_mamba_ssm_size": (0, 0),
        "_registration_started": False,
        "_registered_layout": None,
        "_dspark_enabled": False,
        "_dspark_consumer": None,
        "_dspark_registry": adapter.DsparkPdReadyRegistry(),
        "_dspark_metadata_admission": None,
    }
    vars(worker).update(values)
    boundary = RegistrationBoundary()
    monkeypatch.setattr(adapter, "global_te", boundary)
    monkeypatch.setattr(adapter, "KVCacheSendingThread", boundary.sender)
    if receiver_boundary:
        monkeypatch.setattr(adapter, "AFDMooncakeRecvingThread", boundary.receiver)
    monkeypatch.setattr(
        adapter,
        "collect_deepseek_v41_kv_layout",
        lambda cfg, tensors: wire_layout(cache),
    )
    connector = adapter.MooncakeHybridConnector.__new__(adapter.MooncakeHybridConnector)
    connector.connector_worker, connector.connector_scheduler = worker, None
    return SimpleNamespace(
        worker=worker, cache=cache, boundary=boundary, connector=connector
    )


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


class _QueuedReceiverBoundary(_ReceiverSocketBoundary):
    """Real receiver queue/handler, with no NPU executor or resident thread."""

    def start(self):
        # Construction and registration validation remain real. CPU callers
        # explicitly dispatch an already queued task; no device worker starts.
        self.executor.shutdown(wait=True)
        self.ready_event.set()


class _StartedReceiverSocketBoundary(_ReceiverSocketBoundary):
    """Actual receiver input/peer threads with CPU executor initialization.

    The Event observes the actual handler's return after typed DONE and its
    accepted-work frontier. It supplies no result or retirement state.
    """

    def start(self):
        self.executor.shutdown(wait=True)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.request_finished = threading.Event()
        self.processing_error = None
        super().start()

    def _handle_request(self, request):
        try:
            return super()._handle_request(request)
        except BaseException as error:
            self.processing_error = error
            raise
        finally:
            self.request_finished.set()


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


def attach_consumer_worker(case, *, dspark=False):
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
        _dspark_enabled=dspark,
        _dspark_consumer=None,
        _registration_started=True,
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
    case.connector = adapter.MooncakeHybridConnector.__new__(
        adapter.MooncakeHybridConnector
    )
    case.connector.connector_worker, case.connector.connector_scheduler = worker, None
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


class _SenderRequestComplete(BaseException):
    """The CPU socket leaf ends one iteration without stopping a native loop."""


class SenderSocketBoundary:
    def __init__(self, payload):
        self.payload = payload
        self.response = None

    def recv_multipart(self):
        if self.payload is None:
            raise _SenderRequestComplete()
        payload, self.payload = self.payload, None
        return [b"controlled-cpu-client", b"", payload]

    def send_multipart(self, frames, **kwargs):
        self.response = frames[-1]

    def setsockopt(self, option, value):
        # Synchronous protocol tests do not execute a blocking socket.
        pass


def sender_request(sender, kind, payload):
    """Execute the actual sender dispatch at an explicit socket boundary."""
    socket = SenderSocketBoundary(msgspec.msgpack.encode((kind, payload)))
    with suppress(_SenderRequestComplete):
        sender.run_busy_loop(socket)
    return (
        socket.response
        if socket.response == b"ACK"
        else msgspec.msgpack.decode(socket.response)
    )


class _ControlledSendingThread(adapter.AFDMooncakeSendingThread):
    def start(self):
        # Socket/thread startup is excluded. Constructor and actual native
        # dispatch are retained, and all protocol calls are synchronous CPU.
        self.ready_event.set()

    def retire(self, *, timeout):
        # This fixture excludes thread startup; live sender retirement is
        # exercised separately with the real native run and ZMQ socket owner.
        self.dspark_registry.close()


def make_dspark_registration_case(
    monkeypatch, prefix_case, *, bind=True, live_sender=False
):
    """Real CPU layout/model binding; only NPU admission and TE/socket are leaves.

    ``prefix_case`` is the existing dspark_pd loaded-model builder with bind=False.
    ``bind=False`` leaves connector binding/registration to an outer runner test.
    No second model graph or forged NPU layout is constructed here.
    """
    case = make_registration_case(monkeypatch)
    # The producer-only registration leaf replaces an unused receiver factory.
    # Restore its actual class before this component can compose a real receiver
    # and execute production instance/type guards during shutdown.
    monkeypatch.setattr(adapter, "AFDMooncakeRecvingThread", AFDMooncakeRecvingThread)
    worker, prefix = case.worker, prefix_case
    prefix.config.model_config.is_deepseek_mla = True
    prefix.config.cache_config.block_size = 4
    text = PretrainedConfig(**vars(prefix.config.model_config.hf_text_config))
    if "compress_ratios" not in vars(text):
        text.compress_ratios = [DSPARK_CPU_COMPRESS_RATIO]
    prefix.config.model_config.hf_config = prefix.config.model_config.hf_text_config = (
        text
    )
    worker.vllm_config = prefix.config
    worker.kv_cache_config = prefix.kv_config
    worker.engine_id = prefix.source.source_engine_id
    worker.tp_rank = prefix.expected_ready.source_tp_rank
    worker.tp_size = prefix.expected_ready.tp_size
    worker._prefill_tp_size = worker.tp_size
    # Native requests_to_send walks PCP selection and both P/D TP sizes before
    # invoking the sender's real delayed/unused-source lifecycle methods.
    worker._decode_tp_size = prefix.config.kv_transfer_config.kv_connector_extra_config[
        "decode"
    ]["tp_size"]
    worker._dspark_enabled = True
    worker.hma_group_size = len(prefix.kv_config.kv_cache_groups)
    monkeypatch.setattr(
        adapter, "collect_deepseek_v41_kv_layout", collect_deepseek_v41_kv_layout
    )
    native_role_validation = adapter._validate_layout_role

    def validate_cpu_layout(layout, *, draft_layer_names=()):
        # Preserve the real CPU snapshot in model binding and on the wire;
        # only the hardware-admission condition is excluded from this seam.
        native_role_validation(
            replace(layout, device_type="npu"), draft_layer_names=draft_layer_names
        )

    monkeypatch.setattr(adapter, "_validate_layout_role", validate_cpu_layout)
    monkeypatch.setattr(
        adapter,
        "AFDMooncakeSendingThread",
        AFDMooncakeSendingThread if live_sender else _ControlledSendingThread,
    )
    monkeypatch.setattr(
        native_mooncake, "get_pp_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    monkeypatch.setattr(
        native_mooncake, "get_pcp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(
        native_mooncake, "get_tensor_model_parallel_world_size", lambda: worker.tp_size
    )
    connector = case.connector
    metadata = adapter.DsparkMooncakeConnectorMetadata()
    metadata.dspark_sources = (prefix.source,)
    metadata.reqs_in_batch = {prefix.source.request_id}
    case.prefix_case, case.connector, case.metadata = prefix, connector, metadata
    case.layout, case.cache = prefix.layout, prefix

    def register():
        worker.register_kv_caches(prefix.caches)

    def bind_and_register():
        connector.bind_dspark_model(prefix.consumer)
        register()

    case.register = register
    case.bind_and_register = bind_and_register
    case.admit = lambda: worker.start_load_kv(metadata)
    case.request_ready = lambda query: sender_request(
        worker.kv_send_thread, adapter.GET_DSPARK_READY_MSG, read_query_payload(query)
    )
    case.complete_read = lambda lease: sender_request(
        worker.kv_send_thread, adapter.DONE_DSPARK_RECV_MSG, read_lease_payload(lease)
    )
    if bind:
        bind_and_register()
    return case


def make_retiring_sender_case(monkeypatch, prefix_case):
    """Real CPU sender thread/socket lifecycle; no TE or NPU execution."""
    retire_sender = AFDMooncakeSendingThread.retire
    context = zmq.Context()
    endpoint = "inproc://dspark-sender-" + uuid4().hex

    @contextmanager
    def socket_owner(socket_type, path):
        with context.socket(socket_type) as socket:
            socket.bind(endpoint)
            yield socket

    monkeypatch.setattr(native_mooncake, "zmq_ctx", socket_owner)
    case = make_dspark_registration_case(monkeypatch, prefix_case, live_sender=True)
    case.sender = case.worker.kv_send_thread

    def exchange(kind, payload):
        with context.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.RCVTIMEO, int(THREAD_TEST_TIMEOUT * 1000))
            socket.setsockopt(zmq.SNDTIMEO, int(THREAD_TEST_TIMEOUT * 1000))
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(endpoint)
            socket.send(msgspec.msgpack.encode((kind, payload)))
            response = socket.recv()
            return response if response == b"ACK" else msgspec.msgpack.decode(response)

    def close():
        try:
            retire_sender(case.sender, timeout=THREAD_TEST_TIMEOUT)
        except RuntimeError:
            # A failed registry close is intentionally terminal. Test teardown
            # stops only this host thread, retaining the production failure.
            case.sender._retirement_requested.set()
            case.sender.join(timeout=THREAD_TEST_TIMEOUT)
        finally:
            context.term()

    case.exchange, case.close = exchange, close
    case.request_ready = lambda query: exchange(
        adapter.GET_DSPARK_READY_MSG, read_query_payload(query)
    )
    case.complete_read = lambda lease: exchange(
        adapter.DONE_DSPARK_RECV_MSG, read_lease_payload(lease)
    )
    return case


def dspark_query(prefix, *, generation="destination-generation"):
    return DsparkPdReadQuery(
        prefix.expected_ready,
        "destination-engine",
        generation,
        prefix.source.transfer_block_ids,
        "query-" + generation,
    )


def publish_dspark_prefix(case, *, acquire=True, execution=None):
    case.admit()
    if execution is None:
        scheduler, state = case.prefix_case.make_step(
            0, case.prefix_case.source.computed_prefix_tokens
        )
        sources = (case.prefix_case.source,)
    else:
        scheduler, state, sources = execution
    ready = case.prefix_case.consumer.consume_prefill(scheduler, state, sources)
    case.connector.publish_dspark_ready(ready)
    if not acquire:
        return None
    return read_lease_from_payload(
        case.request_ready(dspark_query(case.prefix_case))["lease"]
    )


def make_connector_scheduler(monkeypatch, config, kv_config, engine_id):
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    monkeypatch.setattr(native_mooncake, "init_ascend_config", lambda config: None)
    monkeypatch.setattr(native_mooncake, "get_ascend_config", lambda: None)
    return adapter.AFDMooncakeConnectorScheduler(config, engine_id, kv_config)


def make_ep_pd_connector_schedulers(monkeypatch, *, producer_rank, consumer_rank):
    """Separate native scheduler instances on independently selected DP ranks."""
    dp_size = 4
    cache = make_native_kv_cache_case()

    def create(role, dp_rank):
        config = configure_hccl_ep(
            connector_config(role=role), dp_size=dp_size, dp_rank=dp_rank
        )
        config.parallel_config.nnodes = 1
        config.engram_config = SimpleNamespace(cpu_offload=True, dp_shared_memory=True)
        engine_id = f"ep-{role}"
        config.kv_transfer_config = KVTransferConfig(
            kv_connector="MooncakeHybridConnector",
            kv_connector_module_path="afd_plugin.compat.npu.kv_layout_connector",
            kv_role=role,
            engine_id=engine_id,
            kv_port=LOCAL_HANDSHAKE_PORT,
            kv_connector_extra_config={
                side: {"tp_size": 1, "dp_size": dp_size, "pp_size": 1}
                for side in ("prefill", "decode")
            },
        )
        return make_connector_scheduler(monkeypatch, config, cache.config, engine_id)

    return SimpleNamespace(
        producer=create("kv_producer", producer_rank),
        consumer=create("kv_consumer", consumer_rank),
        block_ids=tuple([0] for _ in cache.config.kv_cache_groups),
    )


def make_dspark_scheduler_case(
    monkeypatch, prefix, *, role="kv_producer", sampling_params=None
):
    """Native scheduler metadata and KV manager allocation; no model/device work."""
    config = prefix.config
    original = config.kv_transfer_config
    config.kv_transfer_config = KVTransferConfig(
        kv_connector=original.kv_connector,
        kv_connector_module_path=original.kv_connector_module_path,
        kv_role=role,
        engine_id=prefix.source.source_engine_id,
        kv_buffer_device="cpu",
        kv_port=LOCAL_HANDSHAKE_PORT,
        kv_connector_extra_config=original.kv_connector_extra_config,
    )
    config.cache_config.block_size = 4
    if "compress_ratios" not in vars(config.model_config.hf_config):
        config.model_config.hf_config.compress_ratios = [DSPARK_CPU_COMPRESS_RATIO]
    scheduler = make_connector_scheduler(
        monkeypatch, config, prefix.kv_config, prefix.source.source_engine_id
    )

    def new_request(tokens, *, params=None, request_id=None):
        request = Request(
            prefix.source.request_id if request_id is None else request_id,
            tokens,
            (
                SamplingParams(max_tokens=1)
                if sampling_params is None
                else deepcopy(sampling_params)
            ),
            None,
        )
        request.kv_transfer_params = (
            {"do_remote_decode": True} if params is None else params
        )
        scheduler.on_new_request(request)
        return request

    request = new_request(list(range(prefix.source.original_prompt_tokens)))
    manager = KVCacheManager(
        prefix.kv_config,
        max_model_len=64,
        scheduler_block_size=4,
        hash_block_size=2,
        enable_caching=False,
    )
    return SimpleNamespace(
        scheduler=scheduler, request=request, manager=manager, new_request=new_request
    )


def make_dspark_read_case(
    monkeypatch,
    prefix,
    *,
    done_ack=b"ACK",
    destination_blocks=None,
    destination_offset=0,
    reverse_destination=False,
    copy_payload=False,
):
    """Actual ready/GETMETA/DONE dispatch; TE read is an explicit CPU leaf.

    Queue dispatch is driven synchronously, without the native NPU thread
    initializer. The public admission/queue envelope and actual handler remain.
    """
    producer = make_dspark_registration_case(monkeypatch, prefix)
    destination_cache = _independent_destination_cache(
        prefix,
        prefix.kv_config.num_blocks
        if destination_blocks is None
        else destination_blocks,
    )
    layout = destination_cache.layout
    worker = producer.worker
    engine = (
        CopyReadBoundary(destination_cache.source_pages, destination_cache.pages)
        if copy_payload
        else worker.engine
    )
    receiver = _ReceiverSocketBoundary(
        worker.tp_rank,
        worker.tp_size,
        1,
        engine,
        "destination-engine",
        LOCAL_HANDSHAKE_PORT,
        LOCAL_HANDSHAKE_PORT,
        [page.data_ptr() for page in destination_cache.pages],
        worker.block_len_per_addr,
        worker.block_stride_per_addr,
        worker.addr_group_idx,
        (0, 0),
        True,
        False,
        len(layout.groups),
        threading.Event(),
        prefix.config,
        destination_cache.config,
        destination_cache.caches,
        mooncake_runtime_id="same-runtime",
        local_layout=layout,
        draft_layer_names=prefix.draft_names,
    )
    receiver.executor.shutdown(wait=True)
    context = zmq.Context()
    receiver.socket = context.socket(zmq.REQ)
    sent = []
    done_entered = threading.Event()

    def receive(sock, path):
        kind, payload = msgspec.msgpack.decode(sent[-1])
        if kind == adapter.DONE_DSPARK_RECV_MSG:
            done_entered.set()
            if done_ack != b"ACK":
                return done_ack
        return (
            msgspec.msgpack.encode(worker.xfer_handshake_metadata)
            if kind == adapter.GET_META_MSG
            else _encoded_sender_reply(worker.kv_send_thread, kind, payload)
        )

    monkeypatch.setattr(
        adapter, "ensure_zmq_send", lambda sock, data, path: sent.append(data)
    )
    monkeypatch.setattr(adapter, "ensure_zmq_recv", receive)
    source_ids = sorted(
        {block_id for group in prefix.source.transfer_block_ids for block_id in group}
    )
    mapped_ids = reversed(source_ids) if reverse_destination else source_ids
    # Shared logical groups must use the same physical-page permutation.
    destination_ids = {
        source_id: destination_id + destination_offset
        for source_id, destination_id in zip(source_ids, mapped_ids, strict=True)
    }
    destination = adapter.DsparkPdDestination(
        prefix.source,
        "destination-generation",
        tuple(
            tuple(destination_ids[block_id] for block_id in ids)
            for ids in prefix.source.transfer_block_ids
        ),
    )
    request = request_map(
        producer,
        source=[list(ids) for ids in prefix.source.transfer_block_ids],
        destination=[list(ids) for ids in destination.block_ids],
    )
    request.update(
        remote_request_id=prefix.source.request_id,
        remote_engine_id=worker.engine_id,
        remote_handshake_port=worker.handshake_port,
    )

    def admit_request():
        with receiver.admission_scope(
            destinations={request["request_id"]: destination}
        ):
            receiver.task_tracker.add_req_to_process(request["request_id"])
            receiver.add_request(**request)

    def process():
        admit_request()
        queued = receiver.request_queue.get_nowait()
        receiver._mark_request_task_submitted(queued)
        return receiver._handle_request(queued)

    return SimpleNamespace(
        producer=producer,
        receiver=receiver,
        engine=engine,
        layout=layout,
        destination=destination,
        destination_cache=destination_cache,
        sent=sent,
        done_entered=done_entered,
        admit_request=admit_request,
        process=process,
        request=request,
        close=lambda: context.destroy(linger=0),
    )


def make_retiring_dspark_read_case(monkeypatch, prefix, *, done_ack=b"ACK"):
    """Real CPU receiver thread with typed lease-DONE at the socket leaf."""
    case = attach_consumer_worker(
        make_dspark_read_case(monkeypatch, prefix, done_ack=done_ack), dspark=True
    )
    receiver = case.receiver
    receiver.executor = ThreadPoolExecutor(max_workers=1)
    original_close = case.close

    def submit():
        case.admit_request()
        if not case.done_entered.wait(THREAD_TEST_TIMEOUT):
            raise AssertionError("typed DONE did not reach the CPU socket boundary")

    def close():
        if receiver.is_alive():
            # Test-only cleanup of failed ownership. Production retains the
            # receiver and registration for the outer process supervisor.
            receiver.request_queue.put(adapter._ReceiverControl.RETIRE)
            receiver.join(THREAD_TEST_TIMEOUT)
            if receiver.is_alive():
                raise AssertionError("CPU test receiver did not exit")
        receiver.executor.shutdown(wait=True)
        original_close()

    case.submit, case.close = submit, close
    receiver.start()
    if not receiver.ready_event.wait(THREAD_TEST_TIMEOUT):
        raise AssertionError("CPU test receiver did not become ready")
    return case


class _DeferredReceiverExecutor:
    """Defer host scheduling so native peer dispatch sees both queued requests."""

    def __init__(self):
        self.pending = deque()

    def submit(self, function, *args):
        self.pending.append((function, args))

    def run_pending(self):
        while self.pending:
            function, args = self.pending.popleft()
            function(*args)


def make_queued_dspark_read_case(monkeypatch, prefix, *, first_failure):
    """Two real projected publications and queued reads on the same owners.

    Native queue admission, per-peer dispatch and typed lease/DONE remain real.
    Only executor scheduling, socket exchange and CPU copying are host leaves;
    this does not prove DMA drain, device execution or process supervision.
    """
    case = make_dspark_read_case(monkeypatch, prefix, copy_payload=True)
    publish_dspark_prefix(case.producer, acquire=False)
    source_offset = 1 + max(
        block for group in prefix.source.allocation_block_ids for block in group
    )
    source = replace(
        prefix.source,
        request_id="queued-source",
        source_generation="queued-source-generation",
        allocation_block_ids=tuple(
            tuple(block + source_offset for block in group)
            for group in prefix.source.allocation_block_ids
        ),
        transfer_block_ids=tuple(
            tuple(block + source_offset for block in group)
            for group in prefix.source.transfer_block_ids
        ),
    )
    metadata = adapter.DsparkMooncakeConnectorMetadata()
    metadata.dspark_sources = (source,)
    metadata.reqs_in_batch = {source.request_id}
    case.producer.worker.start_load_kv(metadata)
    scheduled, state = prefix.make_step(0, source.computed_prefix_tokens)
    scheduled.num_scheduled_tokens = {source.request_id: source.computed_prefix_tokens}
    state.input_batch.req_ids[:] = [source.request_id]
    case.producer.connector.publish_dspark_ready(
        prefix.consumer.consume_prefill(scheduled, state, (source,))
    )
    destination_offset = 1 + max(
        block for group in case.destination.block_ids for block in group
    )
    destination = adapter.DsparkPdDestination(
        source,
        "queued-destination-generation",
        tuple(
            tuple(block + destination_offset for block in group)
            for group in case.destination.block_ids
        ),
    )
    request = {
        **case.request,
        "request_id": "queued-destination",
        "remote_request_id": source.request_id,
        "remote_block_ids": [list(group) for group in source.transfer_block_ids],
        "local_block_ids": [list(group) for group in destination.block_ids],
    }
    receive = adapter.ensure_zmq_recv

    def receive_first_failure(sock, path):
        kind, payload = msgspec.msgpack.decode(case.sent[-1])
        if (
            first_failure == "done-nack"
            and kind == adapter.DONE_DSPARK_RECV_MSG
            and read_lease_from_payload(payload).ready.source_request_id
            == prefix.source.request_id
        ):
            return b"NACK"
        return receive(sock, path)

    monkeypatch.setattr(adapter, "ensure_zmq_recv", receive_first_failure)
    read = case.engine.batch_transfer_sync_read

    def read_first_failure(*args):
        case.engine.result = (
            -1 if first_failure == "negative-read" and not case.engine.reads else 0
        )
        return read(*args)

    monkeypatch.setattr(case.engine, "batch_transfer_sync_read", read_first_failure)
    executor = _DeferredReceiverExecutor()
    case.receiver.executor = executor
    case.admit_request()
    with case.receiver.admission_scope(
        destinations={request["request_id"]: destination}
    ):
        case.receiver.task_tracker.add_req_to_process(request["request_id"])
        case.receiver.add_request(**request)

    def process_queued():
        # Actual native submit/per-peer methods own counters, exception handling
        # and the decision whether the second request may enter its read.
        while not case.receiver.request_queue.empty():
            case.receiver._submit_request(case.receiver.request_queue.get_nowait())
        executor.run_pending()

    spans = []
    for index, page in enumerate(case.layout.pages):
        group_index = next(
            group_index
            for group_index, group in enumerate(case.layout.groups)
            if set(page.layers) & set(group.layer_names)
        )
        stride = page.block_stride_bytes
        for remote, local in zip(
            source.transfer_block_ids[group_index],
            destination.block_ids[group_index],
            strict=True,
        ):
            spans.append(
                (
                    case.destination_cache.pages[index][
                        local * stride : (local + 1) * stride
                    ],
                    case.destination_cache.source_pages[index][
                        remote * stride : (remote + 1) * stride
                    ].clone(),
                )
            )
    case.process_queued = process_queued
    case.queued_source, case.queued_request, case.queued_spans = source, request, spans
    return case


def make_scheduled_dspark_read_case(
    monkeypatch,
    prefix,
    *,
    destination_blocks=SCHEDULED_DESTINATION_BLOCKS,
    request_id=SCHEDULED_DESTINATION_REQUEST,
    done_ack=b"ACK",
    initialize_source_pages=None,
    sampling_params: SamplingParams | None = None,
    scheduler_source=False,
    producer=None,
    producer_execution=None,
    started_receiver=False,
    destination_config=None,
):
    """Real scheduler allocation -> ActiveKV registration -> typed queued read.

    The producer computes and publishes its actual CPU draft prefix first.
    Destination storage, native target/draft identities, consumer and config
    are independent and fixed before binding. The returned connector is bound
    but unregistered: create_active_connector performs the real single-use
    registration and the controlled receiver socket exactly once.

    Native model/worker constructors do not load checkpoints or initialize a
    device here. TE registration, socket exchange, CPU byte copy and receiver
    startup are explicit leaves. This does not execute the destination model.
    """
    # Circular: scheduler reuses this module's checkpoint-metadata constant.
    from tests.unit.fakes.scheduler import make_native_pd_scheduler_case

    source_scheduler_case = None
    if scheduler_source:
        source_scheduler_case = make_dspark_scheduler_case(monkeypatch, prefix)
        source_metadata, _ = _allocate_source_prefix(
            prefix, source_scheduler_case, source_scheduler_case.request
        )
        prefix.source = source_metadata.dspark_sources[0]
        prefix.expected_ready = replace(
            prefix.expected_ready,
            source_generation=prefix.source.source_generation,
            transfer_block_ids=prefix.source.transfer_block_ids,
        )
    if producer is None:
        producer = make_dspark_registration_case(monkeypatch, prefix)
    destination_cache = _independent_destination_cache(prefix, destination_blocks)
    if initialize_source_pages is not None:
        # The optional caller's page/block byte pattern precedes projection
        # and Ready publication. Published source storage is never rewritten.
        initialize_source_pages(destination_cache.source_pages, prefix.layout)
    publish_dspark_prefix(producer, acquire=False, execution=producer_execution)
    scheduler_case = make_native_pd_scheduler_case(
        monkeypatch,
        config=prefix.config if destination_config is None else destination_config,
        kv_cache_config=destination_cache.config,
        request_id=request_id,
        prompt_token_ids=list(range(prefix.source.original_prompt_tokens)),
        sampling_params=sampling_params,
        kv_transfer_params=_remote_prefill_params(
            prefix.source, producer.worker.handshake_port
        ),
    )
    config = scheduler_case.config
    loaded = make_dspark_pd_loaded_case(
        monkeypatch,
        config=config,
        kv_cache_config=destination_cache.config,
        caches=destination_cache.caches,
        draft_names=prefix.draft_names,
        source_engine_id=config.kv_transfer_config.engine_id,
        tp_rank=prefix.expected_ready.source_tp_rank,
        tp_size=prefix.expected_ready.tp_size,
        complete_cpu_models=prefix.complete_cpu_models,
    )
    registration = make_registration_case(monkeypatch, role="kv_consumer")
    worker, connector = registration.worker, registration.connector
    vars(worker).update(
        vllm_config=config,
        kv_cache_config=destination_cache.config,
        engine_id=config.kv_transfer_config.engine_id,
        engine=CopyReadBoundary(
            destination_cache.source_pages, destination_cache.pages
        ),
        tp_rank=prefix.expected_ready.source_tp_rank,
        tp_size=prefix.expected_ready.tp_size,
        _decode_tp_size=prefix.expected_ready.tp_size,
        _prefill_tp_size=prefix.expected_ready.tp_size,
        tp_num_need_pulls=1,
        hma_group_size=len(destination_cache.layout.groups),
        _dspark_enabled=True,
    )
    connector.engine_id = worker.engine_id
    connector._kv_transfer_config = config.kv_transfer_config
    connector.bind_dspark_model(loaded.consumer)
    # Keep the actual CPU materialization proof. The inherited validator was
    # already narrowed to NPU-admission-only by producer registration above.
    monkeypatch.setattr(
        adapter, "collect_deepseek_v41_kv_layout", collect_deepseek_v41_kv_layout
    )
    monkeypatch.setattr(
        adapter,
        "AFDMooncakeRecvingThread",
        _StartedReceiverSocketBoundary if started_receiver else _QueuedReceiverBoundary,
    )
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", connector)
    context = zmq.Context()
    sent = []

    def receive(sock, path):
        kind, payload = msgspec.msgpack.decode(sent[-1])
        if kind == adapter.DONE_DSPARK_RECV_MSG and done_ack != b"ACK":
            return done_ack
        return (
            msgspec.msgpack.encode(producer.worker.xfer_handshake_metadata)
            if kind == adapter.GET_META_MSG
            else _encoded_sender_reply(producer.worker.kv_send_thread, kind, payload)
        )

    monkeypatch.setattr(
        adapter, "ensure_zmq_send", lambda sock, data, path: sent.append(data)
    )
    monkeypatch.setattr(adapter, "ensure_zmq_recv", receive)

    def create_active_connector():
        active = ActiveKVConnector(config, destination_cache.caches)
        worker.kv_recv_thread.socket = context.socket(zmq.REQ)
        return active

    def process_next_queued_request():
        receiver = worker.kv_recv_thread
        if started_receiver:
            if not receiver.request_finished.wait(THREAD_TEST_TIMEOUT):
                raise AssertionError(
                    "native typed read did not finish on its CPU thread"
                )
            if receiver.processing_error is not None:
                raise receiver.processing_error
            return
        # Only process a task admitted by the real native worker. Do not build
        # a second envelope or call add_request again at this boundary.
        queued = receiver.request_queue.get_nowait()
        receiver._mark_request_task_submitted(queued)
        return receiver._handle_request(queued)

    def close():
        if worker.kv_recv_thread is not None:
            if started_receiver and worker.kv_recv_thread.is_alive():
                # Test-only finalization of retained failure; production owns
                # no cooperative cleanup after the fail-closed worker gate.
                worker.kv_recv_thread.request_queue.put(adapter._ReceiverControl.RETIRE)
                worker.kv_recv_thread.join(THREAD_TEST_TIMEOUT)
                if worker.kv_recv_thread.is_alive():
                    raise AssertionError("CPU test receiver did not exit")
            worker.kv_recv_thread.executor.shutdown(wait=True)
        context.destroy(linger=0)
        scheduler_case.close()

    return SimpleNamespace(
        producer=producer,
        prefix_case=prefix,
        source_scheduler_case=source_scheduler_case,
        scheduler_case=scheduler_case,
        scheduler=scheduler_case.scheduler,
        config=config,
        request=scheduler_case.request,
        connector=connector,
        worker=worker,
        loaded_destination=loaded,
        destination_cache=destination_cache,
        engine=worker.engine,
        sent=sent,
        create_active_connector=create_active_connector,
        process_next_queued_request=process_next_queued_request,
        close=close,
    )


def _allocate_source_prefix(prefix, case, request):
    if case.manager.allocate_slots(request, request.num_prompt_tokens) is None:
        raise AssertionError("CPU source prefix allocation failed")
    case.scheduler.update_state_after_alloc(
        request, case.manager.get_blocks(request.request_id), 0
    )
    scheduled, state = prefix.make_step(0, request.num_prompt_tokens)
    return case.scheduler.build_connector_meta(scheduled), state


def _remote_prefill_params(source, handshake_port):
    return {
        "do_remote_prefill": True,
        "remote_request_id": source.request_id,
        "remote_engine_id": source.source_engine_id,
        "remote_host": "127.0.0.1",
        "remote_port": handshake_port,
        "remote_block_ids": source.transfer_block_ids,
        adapter.DSPARK_SOURCE_PARAM: source_request_payload(source),
    }


def reuse_scheduled_remote_prefix(case, prompt_token_ids):
    """Reuse registered P/D owners after the first native D request retires.

    P generation/allocation and D admission come from their existing native
    schedulers. The caller has already observed the first typed DONE/ACK and
    delivered P's finished notification before these source pages are reused.
    The same CPU projection model writes a distinct second context; no second
    source/destination fixture, registration, worker or consumer is constructed.
    """
    source_case, prefix = case.source_scheduler_case, case.prefix_case
    previous = source_case.request
    previous.append_output_token_ids(SOURCE_COMPLETION_TOKEN)
    previous.status = RequestStatus.FINISHED_LENGTH_CAPPED
    delayed, _ = source_case.scheduler.request_finished_all_groups(
        previous, source_case.manager.get_block_ids(previous.request_id)
    )
    if not delayed:
        raise AssertionError("CPU source did not retain its transfer allocation")
    source_case.manager.free(previous)
    replacement = source_case.new_request(list(prompt_token_ids))
    metadata, state = _allocate_source_prefix(prefix, source_case, replacement)
    source = metadata.dspark_sources[0]
    scheduled, _ = prefix.make_step(0, source.computed_prefix_tokens)
    scheduled.finished_req_ids = {source.request_id}
    for hidden_states in state.aux_hidden_states:
        hidden_states.add_(REUSED_PREFIX_CONTEXT_OFFSET)
    case.producer.worker.start_load_kv(metadata)
    ready = prefix.consumer.consume_prefill(scheduled, state, (source,))
    case.producer.connector.publish_dspark_ready(ready)
    destination = Request(
        case.request.request_id,
        list(prompt_token_ids),
        SamplingParams(max_tokens=1),
        None,
    )
    destination.kv_transfer_params = _remote_prefill_params(
        source, case.producer.worker.handshake_port
    )
    case.scheduler.add_request(destination)
    source_case.request = replacement
    return destination, source


def _encoded_sender_reply(sender, kind, payload):
    reply = sender_request(sender, kind, payload)
    return reply if isinstance(reply, bytes) else msgspec.msgpack.encode(reply)
