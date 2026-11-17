# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native runner lifecycle boundaries using shared host graph builders."""

from collections import deque
from types import SimpleNamespace

import torch
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from afd_plugin.connectors.npu import p2p_hccl as p2p_module
from afd_plugin.connectors.npu.hccl_graph_cache import HcclGraphCache
from afd_plugin.connectors.npu.hccl_staged import HcclStagedAFDConnector
from afd_plugin.distributed.hccl_plan import HcclPlanMessage
from afd_plugin.v1.worker.npu import attention_model_runner_v2 as runner_module
from afd_plugin.v1.worker.npu.hccl_v41_graph_control import HcclV41GraphControl
from afd_plugin.v1.worker.npu.hccl_v41_graph_coordinator import HcclV41GraphCoordinator
from afd_plugin.v1.worker.npu.hccl_v41_graph_eager import HcclV41GraphEager
from tests.unit.fakes.connector import FakeRoutedConnector
from tests.unit.fakes.dspark_pd import make_dspark_pd_prefix_case
from tests.unit.fakes.hccl import HcclLoopbackSession
from tests.unit.fakes.hccl_v41_graph_coordinator import make_coordinator_case
from tests.unit.fakes.hccl_v41_graph_eager import make_graph_eager_case
from tests.unit.fakes.kv_layout_connector import make_dspark_registration_case
from tests.unit.fakes.vllm_config import hccl_vllm_config


def make_legacy_attention_startup_case(
    monkeypatch, *, other_connector=False, other_model=True, fail_load=False
):
    """Actual A constructor/load with native allocation and queue-wire leaves.

    Admission is a separate boundary; no native model, groups or HELLO is run.
    The runtime factory and its registered P2p adapter remain the real path.
    """
    config = hccl_vllm_config(
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
    )
    config.parallel_config.rank = 0
    fallback = FakeRoutedConnector()
    if other_connector:
        config.additional_config["afd"]["connector"] = "CAMP2pAFDConnector"
        config.parallel_config.enable_expert_parallel = True
        # Exercise the model_type selector independently of the generic
        # registration guard; this fixture never constructs either model.
        config.model_config.hf_config.architectures = ["DeepseekV2ForCausalLM"]
        if other_model:
            config.model_config.hf_config.model_type = "deepseek_v2"
        monkeypatch.setattr(
            runner_module, "create_npu_runtime_connector", lambda *a, **k: fallback
        )
    events = []

    def native_init(owner, native_config, device):
        owner.vllm_config = native_config
        owner.parallel_config = native_config.parallel_config
        owner.model_config = native_config.model_config
        owner.device = device

    def native_load(owner, load_dummy_weights=False, *args, **kwargs):
        if fail_load:
            raise RuntimeError("controlled native load failure")
        events.append("native-load")

    monkeypatch.setattr(NPUModelRunner, "__init__", native_init)
    monkeypatch.setattr(NPUModelRunner, "load_model", native_load)
    monkeypatch.setattr(NPUModelRunner, "shutdown", lambda owner: None)
    monkeypatch.setattr(
        runner_module, "fail_if_unsupported_npu_afd_features", lambda *a, **k: None
    )
    monkeypatch.setattr(runner_module, "_resolve_world_ranks", lambda: (0, 0))
    monkeypatch.setenv("AFD_NPU_ATTENTION_PROFILER_ENABLE", "0")
    wire = HcclLoopbackSession()
    monkeypatch.setattr(p2p_module, "HcclEagerTransport", wire.endpoint)
    owner = runner_module.AFDNPUAttentionModelRunnerV2(config, torch.device("npu", 0))
    return SimpleNamespace(owner=owner, config=config, events=events, wire=wire)


def make_graph_runner_case(*, fail_close=False, stages=1, target_query_len=1):
    """Use the actual AFD runner with CPU-only model/control collaborators."""
    case = make_graph_eager_case(stages=stages, target_query_len=target_query_len)
    # Binding precedes real KV/native runner construction, even for U2 config.
    case.owner.ubatch_runner = None
    case.events = []

    def synchronize():
        case.events.append("graph-sync")
        if fail_close:
            raise RuntimeError("graph synchronization failed")

    case.cache = HcclGraphCache(synchronize)
    case.control = HcclV41GraphControl(
        case.transport,
        None,
        case.cache,
        case.generation,
        metadata_shapes=case.metadata,
    )
    case.eager = HcclV41GraphEager(
        case.owner, case.control, next_transaction_id=case.next_transaction_id
    )
    # This builder owns explicit binder tests, so its replacement control/eager
    # starts unbound even when the lower-level execution builder preselects one.
    case.owner.hccl_v41_graph_eager = None
    connector = object.__new__(HcclStagedAFDConnector)
    connector.transport = case.transport
    connector.overlap_settings = None
    connector._pending_context = connector._pending_tensor_spec = None
    case.owner.connector = connector
    case.owner._afd_pending_metadata = None

    def close_groups():
        case.events.append("groups-close")
        case.transport.closed = True

    case.transport.close = close_groups
    return case


def install_graph_runner_coordinator(case, *, overlap_settings=None):
    """Install host manager/storage identities; no real KV or NPU capture."""
    native = make_coordinator_case(
        stages=case.owner.parallel_config.num_ubatches,
        target_query_len=case.owner.decode_query_len,
    ).manager
    native.model_runner = case.owner
    native.ubatch_runner = case.owner.ubatch_runner
    case.owner.cudagraph_manager = native
    control = HcclV41GraphControl(
        case.transport,
        native,
        case.cache,
        case.generation,
        metadata_shapes=case.metadata,
    )
    coordinator = HcclV41GraphCoordinator(
        case.owner,
        control,
        case.session,
        next_transaction_id=case.next_transaction_id,
        overlap_settings=overlap_settings,
    )
    return coordinator


def queue_graph_runner_stop(case):
    """Script the external FFN acknowledgement to the real STOP handshake."""
    case.transport.incoming.setdefault(0, deque()).append(HcclPlanMessage("STOP_ACK"))


def make_dspark_pd_runner_case(monkeypatch, *, failure=None):
    """Actual AFD load/KV/execute seams, with native model/device leaves on CPU.

    The parent load/allocate/execute/sample are explicit boundaries. Actual native
    draft projection, the plugin's model/storage binding and sender dispatch
    remain intact; this does not execute a target model, native proposal or
    transfer engine.
    """
    case = make_graph_runner_case()
    prefix = make_dspark_pd_prefix_case(
        monkeypatch, bind=False, projection_error=failure == "projection"
    )
    prefix.config.kv_transfer_config.kv_role = "kv_producer"
    prefix.config.kv_transfer_config.engine_id = prefix.source.source_engine_id
    registration = make_dspark_registration_case(monkeypatch, prefix, bind=False)
    owner = case.owner
    owner.vllm_config = prefix.config
    owner.parallel_config = prefix.config.parallel_config
    owner.afd_config = owner.parse_config(prefix.config)
    owner.device = torch.device("cpu")
    owner.hccl_v41_transport = None
    owner.hccl_v41_graph_eager = owner.hccl_v41_graph_session = None
    owner.hccl_v41_graph_coordinator = owner.hccl_v41_graph_bootstrap = None
    owner.hccl_v41_diagnostics = None
    owner.dspark_pd_consumer = None
    owner._dspark_pd_pending_scheduler = None
    owner._afd_suppress_metadata_send = False
    owner._is_warmup = owner._afd_is_graph_capturing = False
    owner._afd_is_graph_replaying = owner._afd_is_profile = False
    owner.execute_model_state = None
    case.next_state = None

    def load_native(self, load_dummy_weights=False, *args, **kwargs):
        self.model, self.speculator = prefix.target_model, prefix.speculator

    def initialize_native(self, kv_cache_config, kv_cache_allocation_context=None):
        # This is the allocation leaf; registration itself is the real worker
        # implementation and binds the actual CPU tensor views to the consumer.
        registration.register()

    def execute_native(self, scheduler_output, intermediate_tensors=None, **kwargs):
        registration.admit()
        if failure == "target":
            raise RuntimeError("controlled target compute failure")
        self.execute_model_state = case.next_state
        return None

    def sample_native(self, grammar_output):
        # Exclude native sampling/proposal execution at this existing lifecycle
        # leaf. Production still saves the original parent before clearing it.
        self.execute_model_state = None
        return None

    monkeypatch.setattr(NPUModelRunner, "load_model", load_native)
    monkeypatch.setattr(NPUModelRunner, "initialize_kv_cache", initialize_native)
    monkeypatch.setattr(NPUModelRunner, "execute_model", execute_native)
    monkeypatch.setattr(NPUModelRunner, "sample_tokens", sample_native)
    monkeypatch.setattr(
        runner_module,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    monkeypatch.setattr(
        runner_module, "get_kv_transfer_group", lambda: registration.connector
    )
    owner.load_model()
    prefix.consumer = owner.dspark_pd_consumer
    owner.initialize_kv_cache(prefix.kv_config)

    def step(offset, count):
        scheduler, case.next_state = prefix.make_step(offset, count)
        scheduler.kv_connector_metadata = (
            object() if failure == "metadata" else registration.metadata
        )
        output = owner.execute_model(scheduler)
        return owner.sample_tokens(None) if output is None else output

    case.prefix, case.registration, case.step = prefix, registration, step
    return case
