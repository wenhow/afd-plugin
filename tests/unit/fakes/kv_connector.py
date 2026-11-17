# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native KV lifecycle with CPU storage and explicit device boundaries.

The AFD, Ascend and GPU execute methods, ActiveKVConnector and EngramModelState
are real. Only transfer registration/load and auxiliary device retirement are
host boundaries. FULL replay additionally supplies prepared inputs, metadata
and an inert graph device boundary. This does not run PD or Graph device work.
"""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import partial
from itertools import count
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from vllm import SamplingParams
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    ObservabilityConfig,
    SchedulerConfig,
)
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorTransferResults,
)
from vllm.forward_context import get_forward_context
from vllm.utils import torch_utils
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.gpu import cudagraph_utils as native_cudagraph
from vllm.v1.worker.gpu import model_runner as gpu_runner
from vllm.v1.worker.gpu.async_utils import (
    AsyncOutput,
    StepTimingCollector,
    async_copy_to_np,
)
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.ec_connector import ECConnector
from vllm.v1.worker.gpu.eplb_utils import EPLBController
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector, KVConnector
from vllm.v1.worker.gpu.sample import sampler as sampler_module
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rejection_module
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
from vllm.v1.worker.utils import clear_layer_kv_caches
from vllm_ascend.ascend_config import SparseKVOffloadConfig
from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAImpl
from vllm_ascend.compilation import acl_graph as native_acl_graph
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41LLMForCausalLM
from vllm_ascend.ops.triton import engram_lookback
from vllm_ascend.worker.v2 import model_runner as native_runner
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.model_states import default as ascend_model_state

from afd_plugin.compat.npu.runtime_config import (
    validate_deepseek_v41_pd,
    validate_deepseek_v41_speculation,
)
from afd_plugin.connectors.npu import hccl_graph_pipeline as pipeline_module
from afd_plugin.connectors.npu.hccl_config import validate_staged_hccl_runtime
from afd_plugin.connectors.npu.hccl_graph_cache import HcclGraphCache
from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank
from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.connectors.npu.hccl_staged import HcclStagedAFDConnector
from afd_plugin.distributed.hccl_plan import HcclLocalForward
from afd_plugin.model_executor.models.npu.deepseek_v41_graph import (
    DeepseekV41GraphParent,
    DeepseekV41GraphStages,
)
from afd_plugin.v1.worker.npu import attention_model_runner_v2 as attention_module
from afd_plugin.v1.worker.npu import hccl_v41_graph_coordinator as coordinator_module
from afd_plugin.v1.worker.npu import ubatch_runner as ubatch_module
from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
    AFDNPUAttentionModelRunnerV2,
)
from afd_plugin.v1.worker.npu.attention_worker import AFDNPUAttentionWorker
from afd_plugin.v1.worker.npu.hccl_v41_graph_control import HcclV41GraphControl
from afd_plugin.v1.worker.npu.hccl_v41_graph_coordinator import HcclV41GraphResources
from afd_plugin.v1.worker.npu.hccl_v41_graph_eager import HcclV41GraphEager
from afd_plugin.v1.worker.npu.hccl_v41_graph_execution import HcclV41GraphExecution
from afd_plugin.v1.worker.npu.hccl_v41_graph_session import HcclV41GraphSession
from afd_plugin.v1.worker.npu.hccl_v41_ubatch_runner import HcclV41UBatchRunner
from tests.unit.fakes.attention_model_runner_v2 import (
    install_graph_runner_coordinator,
    make_graph_runner_case,
    queue_graph_runner_stop,
)
from tests.unit.fakes.block_table import install_native_block_table_kernels_cpu
from tests.unit.fakes.connector import FakeRoutedConnector
from tests.unit.fakes.deepseek_v41 import native_pd_target_reference
from tests.unit.fakes.deepseek_v41_graph import GraphParentNorm, GraphStreamAttention
from tests.unit.fakes.device_metadata import (
    install_native_target_metadata_cpu,
    make_cpu_target_metadata,
)
from tests.unit.fakes.dspark_pd import (
    PD_U2_PREFILL_LENGTHS,
    PD_U2_SOURCE_BLOCKS,
    initialize_pd_cpu_source_pages,
    make_dspark_pd_prefill_bundle,
    make_dspark_pd_prefix_case,
)
from tests.unit.fakes.dspark_speculator import initialize_existing_dspark_proposal
from tests.unit.fakes.graph_cache import DeviceCompletion, ReplayGraph
from tests.unit.fakes.hccl_graph_pipeline import attach_pipeline
from tests.unit.fakes.hccl_graph_stage import make_graph_stage_bank
from tests.unit.fakes.hccl_v41_graph_control import (
    NativeGraphGeneration,
    NativeGraphMetadata,
)
from tests.unit.fakes.hccl_v41_graph_coordinator import (
    cpu_allocate_bank,
    cpu_retained_capture,
    install_runtime_eager_tensor_boundary,
    reference_runtime_eager_output,
)
from tests.unit.fakes.hccl_v41_graph_eager import PreKVAttentionPeer
from tests.unit.fakes.input_batch import (
    install_native_input_preparation_cpu,
    make_ascend_decode_batch,
    make_hccl_ubatch_seam,
    snapshot_ascend_child,
)
from tests.unit.fakes.npu_stream import HostStreamRecorder
from tests.unit.fakes.platform import stub_platform_forward_context
from tests.unit.fakes.rejection_sampler import (
    CPULookbackGather,
    GreedySamplingState,
    greedy_rejection_cpu,
    greedy_sample_cpu,
    sampled_and_rejected_cpu,
    update_speculative_history_cpu,
)
from tests.unit.fakes.request_state import (
    make_native_request_admission_case,
    native_zeroed_cache_reference,
)
from tests.unit.fakes.scheduler import (
    SCHEDULER_MAX_MODEL_LEN,
    SchedulerSpeculativeConfig,
    SchedulerVllmConfig,
)
from tests.unit.fakes.vllm_config import (
    hccl_vllm_config,
    v41_dspark_pd_config,
    v41_dspark_pd_graph_config,
)

FULL_REPLAY_ROWS = 4
FULL_REPLAY_HISTORY_TOKENS = 12
PREFILL_CPU_NUM_STAGES = 2
PREFILL_CPU_DRAFT_STEPS = 2


class TransferMetadata(KVConnectorMetadata):
    """A real native metadata subtype, independent of a specific PD backend."""


class KVTransferBoundary:
    """Public transfer lifecycle; no remote address or NPU operation."""

    def __init__(self, events, *, fail_load):
        self.events = events
        self.load_error = (
            RuntimeError("controlled KV load failure") if fail_load else None
        )
        self.registered_views = None
        self.bound_metadata = None
        self.copy_blocks = None

    def register_kv_caches(self, views):
        self.registered_views = views

    def set_host_xfer_buffer_ops(self, copy_blocks):
        self.copy_blocks = copy_blocks

    def handle_preemptions(self, metadata):
        # Native Ascend also invokes this before GPU request-state updates.
        # Repeating preemption handling is native behavior, not a model step.
        pass

    def bind_connector_metadata(self, metadata):
        if self.bound_metadata is not None:
            raise AssertionError("a previous KV request still owns metadata")
        self.bound_metadata = metadata
        self.events.append("kv-bind")

    def start_load_kv(self, context, **kwargs):
        if context.attn_metadata is not None or kwargs:
            raise AssertionError("zero-work load must use the native empty context")
        self.events.append("kv-load")
        if self.load_error is not None:
            raise self.load_error

    def finish_forward(self):
        self.events.append("kv-finish")

    def wait_for_save(self):
        self.events.append("kv-wait-save")

    def get_transfer_results(self, finished_req_ids):
        self.events.append("kv-results")
        return KVConnectorTransferResults(
            finished_sending={"sent"},
            finished_recving={"received", "failed"},
            failed_recving={"failed"},
        )

    def get_block_ids_with_load_errors(self):
        return {3, 7}

    def get_kv_connector_stats(self):
        return None

    def get_kv_connector_kv_cache_events(self):
        return None

    def build_connector_worker_meta(self):
        return None

    def clear_connector_metadata(self):
        self.bound_metadata = None
        self.events.append("kv-clear")

    def reset_capture_state(self):
        self.events.append("kv-capture-reset")


class AuxiliaryDeviceBoundary:
    """One live lookup/metadata lease; a second retirement is invalid."""

    def __init__(self, events):
        self.events = events
        self.lookup_live = self.metadata_live = True
        self.reset_events = None

    def retire_engram_lookups(self, *, reset_events):
        if not self.lookup_live:
            raise AssertionError("lookup lease was retired twice")
        self.lookup_live = False
        self.reset_events = reset_events
        self.events.append("engram-failed" if reset_events else "engram-finish")

    def finish(self):
        if not self.metadata_live:
            raise AssertionError("device metadata lease was retired twice")
        self.metadata_live = False
        self.events.append("metadata-finish")

    def complete_forward(self):
        # KVPPRuntime's execution boundary is independent of ModelState.finish.
        pass

    def apply_staged_writes(self):
        # Empty block tables have no device writes to flush.
        pass


class IdleAuxiliaryDeviceBoundary(AuxiliaryDeviceBoundary):
    """No prepare, lookup or device task exists during a zero-token PD poll.

    Native ModelState still retires idle resources on each execute. Those
    operations are idempotent; this leaf never opens a pretend device lease.
    """

    def __init__(self, events):
        self.events = events
        self.lookup_live = self.metadata_live = False
        self.reset_events = None

    def retire_engram_lookups(self, *, reset_events):
        self.reset_events = reset_events

    def finish(self):
        pass


def make_native_kv_runner_case(
    monkeypatch,
    *,
    post_kv,
    sync_load,
    fail_load=False,
    target_query_len=1,
    stages=1,
    pd_read_case=None,
):
    """Keep the complete zero-token native execute/no_forward control flow.

    Config admission remains unchanged: an explicit CPU component owns the KV
    singleton below. It is not a serving Graph/PD configuration or handshake.
    All singleton/DP/backend substitutions are restored by pytest monkeypatch.
    """
    if pd_read_case is None:
        case = make_graph_runner_case(target_query_len=target_query_len, stages=stages)
    else:
        if post_kv or fail_load:
            raise ValueError("external PD zero-work case requires eager polling")
        parallel = pd_read_case.config.parallel_config
        configured_stages = parallel.num_ubatches if parallel.use_ubatching else 1
        if stages != configured_stages:
            raise ValueError("PD poll stages must match the already constructed owner")
        # Construct this native owner from the complete D bundle before any
        # model/state/KV binding. No Graph-capable transport or model execution
        # is constructed and no old Graph configuration is replaced in place.
        owner = AFDNPUAttentionModelRunnerV2.__new__(AFDNPUAttentionModelRunnerV2)
        owner.vllm_config = pd_read_case.config
        owner.parallel_config = owner.vllm_config.parallel_config
        owner.afd_config = owner.parse_config(owner.vllm_config)
        owner.model = pd_read_case.loaded_destination.target_model
        owner.speculator = pd_read_case.loaded_destination.speculator
        owner.dspark_pd_consumer = pd_read_case.loaded_destination.consumer
        owner._dspark_pd_pending_scheduler = None
        owner.model_state = SimpleNamespace(model=owner.model)
        owner.block_tables = SimpleNamespace()
        owner.device = torch.device("cpu")
        owner.prof = owner.connector = owner.cudagraph_manager = None
        owner.ubatch_runner = owner.hccl_v41_transport = None
        owner.hccl_v41_graph_eager = owner.hccl_v41_graph_session = None
        owner.hccl_v41_graph_coordinator = owner.hccl_v41_graph_bootstrap = None
        owner.hccl_v41_diagnostics = None
        owner._afd_pending_metadata = None
        owner._afd_suppress_metadata_send = False
        owner._is_warmup = owner._afd_is_graph_capturing = False
        owner._afd_is_profile = owner._afd_is_graph_replaying = False
        case = SimpleNamespace(owner=owner, pd_read_case=pd_read_case)
    owner = case.owner
    events = []
    device = (
        AuxiliaryDeviceBoundary(events)
        if pd_read_case is None
        else IdleAuxiliaryDeviceBoundary(events)
    )
    transfer = (
        KVTransferBoundary(events, fail_load=fail_load)
        if pd_read_case is None
        else pd_read_case.connector
    )
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", transfer)
    monkeypatch.setattr(
        native_runner, "should_skip_allreduce_across_dp_group", lambda _: False
    )
    monkeypatch.setattr(
        coordinator_module, "_get_graph_update_backend", lambda _: AscendSFABackend
    )
    # Keep the real context manager while skipping Ascend's TP/MC2 device
    # context augmentation; there is no initialized device group in this seam.
    stub_platform_forward_context(monkeypatch)

    # Real native retirement, replacing the existing model fixture's sentinel.
    state = EngramModelState.__new__(EngramModelState)
    vars(state).update(vars(owner.model_state))
    state.device_metadata = device
    owner.model_state = state
    if pd_read_case is None or not pd_read_case.prefix_case.complete_cpu_models:
        owner.model.retire_engram_lookups = device.retire_engram_lookups
    owner.ascend_config = (
        SimpleNamespace(
            scheduler_config=SimpleNamespace(
                profiling_chunk_config=SimpleNamespace(need_timing=False)
            )
        )
        if pd_read_case is None
        else pd_read_case.scheduler_case.ascend_config
    )
    owner.pooling_runner = owner.encoder_cache = owner.pp_handler = owner.sampler = None
    owner.req_states = SimpleNamespace(
        num_computed_tokens_np=np.zeros(0, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.zeros(0, dtype=np.int32)),
        num_computed_prefill_tokens=np.zeros(0, dtype=np.int32),
    )
    owner.block_tables.apply_staged_writes = device.apply_staged_writes
    owner.ec_connector, owner.kvpp = ECConnector(), device
    owner.execute_model_state = None
    views = (
        {"layer": torch.arange(16, dtype=torch.int32).reshape(2, 2, 4)}
        if pd_read_case is None
        else pd_read_case.destination_cache.caches
    )
    owner.kv_connector = (
        ActiveKVConnector(owner.vllm_config, views)
        if pd_read_case is None
        else pd_read_case.create_active_connector()
    )
    owner.kv_caches = list(views.values())
    if pd_read_case is not None:
        owner.kv_cache_config = pd_read_case.destination_cache.config
        case.kv_transfer = transfer
        case.kv_views, case.auxiliary_device, case.native_events = views, device, events
        return case
    case.eager = HcclV41GraphEager(
        owner, case.control, next_transaction_id=case.next_transaction_id
    )
    owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    if post_kv:
        owner.bind_hccl_v41_graph_coordinator(install_graph_runner_coordinator(case))

    scheduler = SchedulerOutput.make_empty()
    scheduler.kv_connector_metadata = TransferMetadata()
    scheduler.has_sync_kv_loads = sync_load
    case.scheduler, case.kv_transfer = scheduler, transfer
    case.kv_views, case.auxiliary_device, case.native_events = views, device, events
    return case


def make_scheduled_pd_runner_case(
    monkeypatch,
    *,
    prompt_length,
    stages=1,
    started_receiver=False,
    complete_hccl=False,
    target_graph=False,
):
    """Freeze P/D metadata before construction, then retain the real handoff.

    U2 changes the target execution component, not the receiver or sampler.
    Source projection, typed Ready/read/DONE and native scheduling remain the
    existing CPU contract. Checkpoint, TE and device constructors stay leaves.
    """
    if type(stages) is not int or stages not in (1, PREFILL_CPU_NUM_STAGES):
        raise ValueError("CPU PD component requires U1 or U2")
    config = v41_dspark_pd_config()
    if stages == PREFILL_CPU_NUM_STAGES or complete_hccl:
        eager = hccl_vllm_config(
            dtype=torch.bfloat16,
            compilation_mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
            num_ubatches=stages,
        )
        config.parallel_config = eager.parallel_config
        config.compilation_config = eager.compilation_config
        config.parallel_config.rank = 0
        config.parallel_config.dbo_prefill_token_threshold = stages
        config.parallel_config.dbo_decode_token_threshold = stages
        config.model_config.is_encoder_decoder = eager.model_config.is_encoder_decoder
        config.model_config.is_multimodal_model = eager.model_config.is_multimodal_model
        config.model_config.multimodal_config = eager.model_config.multimodal_config
        config.lora_config = None
    prefix = make_dspark_pd_prefix_case(
        monkeypatch,
        config=config,
        original_tokens=prompt_length,
        bind=False,
        complete_cpu_models=True,
        num_speculative_tokens=PREFILL_CPU_DRAFT_STEPS
        if stages == 2 or target_graph
        else None,
    )
    destination_config = None
    if target_graph:
        destination_config = v41_dspark_pd_graph_config(
            dtype=torch.bfloat16,
            compilation_mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            num_ubatches=stages,
        )
        # The existing eager P supplies the actual three-layer CPU checkpoint
        # metadata. Freeze the independent D config before its native scheduler,
        # model and consumer exist; never copy P's compilation/static contexts.
        destination_config.model_config = deepcopy(prefix.config.model_config)
        destination_config.model_config.enforce_eager = False
        # The normal registration builder completes this native checkpoint
        # flag on P before copying its D config. This independent D config is
        # frozen earlier and must declare the same MLA metadata itself.
        destination_config.model_config.is_deepseek_mla = True
        destination_config.speculative_config = deepcopy(
            prefix.config.speculative_config
        )
        destination_config.compilation_config.cudagraph_capture_sizes = [
            PREFILL_CPU_DRAFT_STEPS + 1
        ]
        destination_config.compilation_config.max_cudagraph_capture_size = (
            PREFILL_CPU_DRAFT_STEPS + 1
        )
        destination_config.parallel_config.dbo_prefill_token_threshold = stages
        destination_config.parallel_config.dbo_decode_token_threshold = stages
    # Circular: the protocol factory consumes this module's native D helpers.
    from tests.unit.fakes.kv_layout_connector import make_scheduled_dspark_read_case

    transfer = make_scheduled_dspark_read_case(
        monkeypatch,
        prefix,
        initialize_source_pages=initialize_pd_cpu_source_pages,
        sampling_params=SamplingParams(temperature=0),
        started_receiver=started_receiver,
        destination_config=destination_config,
    )
    runner = make_native_kv_runner_case(
        monkeypatch,
        post_kv=False,
        sync_load=False,
        stages=stages,
        pd_read_case=transfer,
    )
    return transfer, runner


def complete_native_pd_handoff(case, *, schedule_target=True):
    """Run zero-work polling; optionally leave target admission to EngineCore."""
    transfer, owner = case.pd_read_case, case.owner
    first = transfer.scheduler.schedule()
    transfer.scheduler.update_from_output(first, owner.execute_model(first))
    transfer.process_next_queued_request()
    pending = transfer.scheduler.schedule()
    transfer.scheduler.update_from_output(pending, owner.execute_model(pending))
    return transfer.scheduler.schedule() if schedule_target else None


def initialize_native_pd_target_execution(
    monkeypatch, case, *, scheduler_output=None, graph_target=False
):
    """Complete the existing registered D owner for native eager CPU execute.

    The bound target/draft/config/cache and ActiveKVConnector stay identical.
    Real EngramModelState construction, native NONE manager dispatch, request
    admission, input preparation, metadata lifecycle and target forward run.
    Only CPU configuration/allocation and declared device operator leaves are
    supplied. This has no remote FFN, checkpoint, device or Graph execution.
    """
    owner, transfer = case.owner, case.pd_read_case
    if not transfer.prefix_case.complete_cpu_models:
        raise ValueError("native target requires models completed before PD binding")
    case.target_metadata = initialize_loaded_cpu_target(
        monkeypatch,
        owner,
        transfer.loaded_destination.model_bundle,
        case.kv_views,
        create_graph_manager=not graph_target,
    )
    if owner.parallel_config.use_ubatching and not graph_target:
        case.transport = install_native_u2_cpu_runner(
            monkeypatch,
            owner,
            kv_cache_config=transfer.destination_cache.config,
            model_bundle=transfer.loaded_destination.model_bundle,
            graph=graph_target,
        )
        observe_native_pd_target_children(case)
    destination = transfer.destination_cache
    case.source_pages_before = tuple(page.clone() for page in destination.source_pages)
    if scheduler_output is None:
        # EngineCore contract cases schedule the first real target themselves;
        # they do not need this fixture's independent target-arithmetic oracle.
        return
    request = scheduler_output.scheduled_new_reqs[0]
    width = scheduler_output.num_scheduled_tokens[request.req_id]
    computed = request.num_computed_tokens
    case.expected_input_ids = torch.tensor(
        [request.prompt_token_ids[-1]] + [0] * (width - 1)
    )
    case.expected_positions = torch.arange(computed, computed + width)
    case.expected_target = native_pd_target_step_reference(
        case,
        scheduler_output,
        input_ids=case.expected_input_ids,
        computed=computed,
        block_ids=request.block_ids,
        stages=owner.parallel_config.num_ubatches
        if owner.parallel_config.use_ubatching
        and width >= owner.parallel_config.num_ubatches
        else 1,
    )


def observe_native_pd_target_children(case):
    """Observe actual CPU Attention operands after native child preparation."""
    case.target_children = []

    def observe_target_attention(layer, inputs):
        context = get_forward_context()
        cache_name = layer.dsa_attn.swa_cache_layer.prefix
        metadata = context.attn_metadata[cache_name.replace(".swa_cache", ".v41_attn")]
        case.target_children.append(
            SimpleNamespace(
                stage=context.additional_kwargs.get("ubatch_idx"),
                descriptor_tokens=context.batch_descriptor.num_tokens
                if context.batch_descriptor is not None
                else None,
                positions=inputs[0].clone(),
                query_start_loc=metadata.query_start_loc.clone(),
                seq_lens=metadata.seq_lens[: metadata.num_reqs].clone(),
                slot_mapping=context.slot_mapping[cache_name].clone(),
            )
        )

    case.target_hook = case.owner.model.language_model.model.layers[
        0
    ].self_attn.register_forward_pre_hook(observe_target_attention)


def initialize_loaded_cpu_target(
    monkeypatch,
    owner,
    model_bundle,
    kv_views,
    *,
    preserve_connector=False,
    create_graph_manager=True,
):
    """Attach real request/input/ModelState owners to prebound CPU models.

    Both the existing legal U1/PD D owner and the independent no-PD P owner
    reuse this assembly. It neither constructs a prefix consumer nor changes
    any bound parallel/transfer configuration or target/draft model identity.
    """
    ascend_config = SimpleNamespace(sparse_kv_offload_config=SparseKVOffloadConfig())
    monkeypatch.setattr(ascend_model_state, "get_ascend_config", lambda: ascend_config)
    owner.model_state = EngramModelState(
        owner.vllm_config, owner.model, None, owner.device
    )
    owner.model_state.kv_cache_config = owner.kv_cache_config
    metadata = install_native_target_metadata_cpu(
        monkeypatch, owner.model_state, model_bundle
    )
    make_native_request_admission_case(monkeypatch, owner, kv_cache_views=kv_views)
    install_native_input_preparation_cpu(monkeypatch, owner)
    install_native_block_table_kernels_cpu(monkeypatch, owner.block_tables)
    owner.observability_config = owner.vllm_config.observability_config
    owner.lora_config = None
    owner.is_encoder_decoder = owner.uses_inputs_embeds = False
    owner.use_aux_hidden_state_outputs = True
    owner.routed_experts_capturer = None
    owner.use_spec_pp = owner.use_pp = False
    owner.dp_size, owner.dp_rank = 1, 0
    owner.dcp_size, owner.dcp_rank, owner.cp_interleave = 1, 0, 1
    owner.step_timing = StepTimingCollector()
    owner.attn_groups = []
    owner.update_stream = None
    # The CPU model executes local FFN arithmetic, without an A/F transport.
    # Keep the real eager metadata provider and its normal constructor state.
    owner._afd_transaction_counter = 0
    if not preserve_connector:
        owner.connector = SimpleNamespace(control_plane=None)
    pp = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(native_cudagraph, "get_pp_group", lambda: pp)
    if create_graph_manager:
        # Eager's actual NONE manager reads the resolved capture-size list.
        owner.vllm_config.compilation_config.cudagraph_capture_sizes = []
        owner.cudagraph_manager = ModelAclGraphManager(
            owner.vllm_config,
            owner.device,
            CUDAGraphMode.NONE,
            owner.decode_query_len,
            owner,
        )
    return metadata


def install_native_u2_cpu_runner(
    monkeypatch, owner, *, kv_cache_config, model_bundle, graph=False, transport=None
):
    """Install the existing constructor-free U2/transport allocation leaf.

    Configuration validation, the control codec, native child preparation and
    the HCCL runner's execution stay real. Device runner construction, native
    HCCL/NPU binding and automatic stream bootstrap are outside this CPU seam.
    """
    config = owner.vllm_config
    # Eager negotiates Graph's default key width one independently of native
    # DSpark's N+1 target decode width; child row counts remain actual payloads.
    caps = validate_staged_hccl_runtime(config, owner.afd_config, 0, graph=graph)
    if transport is None:
        transport = PreKVAttentionPeer(caps)
    elif transport.negotiated_capabilities != caps:
        raise ValueError("native U2 runner must retain this session's capabilities")
    runner = make_hccl_ubatch_seam(
        HcclV41UBatchRunner,
        model_state=owner.model_state,
        max_num_reqs=owner.max_num_reqs,
    )
    runner.vllm_config, runner.parallel_config = config, config.parallel_config
    runner.kv_cache_config = kv_cache_config
    runner.transport = transport
    runner._stage_block_tables = [None] * PREFILL_CPU_NUM_STAGES
    runner._stage_rope_buffers = [{} for _ in range(PREFILL_CPU_NUM_STAGES)]
    # Keep actual Hccl -> NPU -> native UBatchRunner preparation and slicing.
    # Only its backend metadata builder consumes the CPU cache operands.
    del runner._prepare_attn

    def build_cpu_metadata(**kwargs):
        names = (
            "num_reqs",
            "query_start_loc_gpu",
            "positions",
            "seq_lens",
            "is_prefilling",
            "block_tables",
        )
        return model_bundle.prepare_target_attention(
            **{name: kwargs[name] for name in names}
        )

    monkeypatch.setattr(ubatch_module, "build_attn_metadata", build_cpu_metadata)
    owner.ubatch_runner, owner.hccl_v41_transport = runner, transport
    return transport


def make_native_pd_prefill_owner_case(
    monkeypatch,
    *,
    stages=1,
    tp_size=1,
    tp_rank=0,
    tp_group=None,
    share_scheduler=None,
):
    """One immutable P owner through actual load/KV/execute/sample frontiers.

    Native load and allocation have explicit CPU leaves. The AFD public entry
    methods, loaded consumer construction, real connector registration,
    ActiveKVConnector, native target execution and native proposal stay real.
    This does not create a second P owner or pre-admit a source. The observer
    queries the actual Ready protocol at declared CPU cache-write boundaries;
    it does not replace projection/proposal or simulate NPU concurrency.
    U2 uses the existing constructor-free runner allocation leaf and the real
    PD validator; it cannot pass a closed production configuration guard.
    """
    if type(stages) is not int or stages not in (1, PREFILL_CPU_NUM_STAGES):
        raise ValueError("CPU P component requires U1 or U2")
    if (
        type(tp_size) is not int
        or type(tp_rank) is not int
        or not 0 <= tp_rank < tp_size
    ):
        raise ValueError("CPU P component requires a valid TP rank and size")
    if tp_group is None and (tp_size, tp_rank) != (1, 0):
        raise ValueError("multi-rank CPU P component requires its actual TP group")
    if tp_group is not None and (tp_group.world_size, tp_group.rank_in_group) != (
        tp_size,
        tp_rank,
    ):
        raise ValueError("CPU P component TP group differs from its frozen config")
    config = v41_dspark_pd_config(tensor_parallel_size=tp_size)
    config.kv_transfer_config.kv_role = "kv_producer"
    config.kv_transfer_config.engine_id = "source-engine"
    eager = hccl_vllm_config(
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        num_ubatches=stages,
        tensor_parallel_size=tp_size,
    )
    config.parallel_config = eager.parallel_config
    config.compilation_config = eager.compilation_config
    config.parallel_config.rank = tp_rank
    if stages == PREFILL_CPU_NUM_STAGES:
        config.parallel_config.dbo_prefill_token_threshold = PREFILL_CPU_NUM_STAGES
        config.parallel_config.dbo_decode_token_threshold = PREFILL_CPU_NUM_STAGES
    config.model_config.is_encoder_decoder = eager.model_config.is_encoder_decoder
    config.model_config.is_multimodal_model = eager.model_config.is_multimodal_model
    config.model_config.multimodal_config = eager.model_config.multimodal_config
    config.model_config.max_model_len = SCHEDULER_MAX_MODEL_LEN
    config.lora_config = None
    config.scheduler_config = SchedulerConfig(
        max_model_len=SCHEDULER_MAX_MODEL_LEN,
        is_encoder_decoder=False,
        max_num_seqs=len(PD_U2_PREFILL_LENGTHS),
        max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
        enable_chunked_prefill=True,
        disable_hybrid_kv_cache_manager=False,
        async_scheduling=False,
    )
    config.observability_config = ObservabilityConfig()
    observations = SimpleNamespace(enabled=False, replies=[])

    def observe_store():
        if observations.enabled:
            observations.replies.append(bundle.producer.request_ready(bundle.query))

    prefix = make_dspark_pd_prefix_case(
        monkeypatch,
        config=config,
        tp_size=tp_size,
        tp_rank=tp_rank,
        bind=False,
        create_consumer=False,
        complete_cpu_models=True,
        num_blocks=PD_U2_SOURCE_BLOCKS,
        num_speculative_tokens=PREFILL_CPU_DRAFT_STEPS,
        before_draft_store=observe_store,
    )
    # Match the native scheduler's actual reservation for proposal writes,
    # rather than allocating only the target's shortened P prompt. Reuse its
    # pinned sizing property over the same already selected DSpark metadata.
    sizing = SchedulerVllmConfig(
        speculative_config=SchedulerSpeculativeConfig(**vars(config.speculative_config))
    )
    bundle = make_dspark_pd_prefill_bundle(
        monkeypatch,
        prefix=prefix,
        bind=False,
        sampling_params=SamplingParams(max_tokens=1, temperature=0),
        num_lookahead_tokens=sizing.num_lookahead_tokens,
        share_scheduler=share_scheduler,
    )
    owner = AFDNPUAttentionModelRunnerV2.__new__(AFDNPUAttentionModelRunnerV2)
    owner.vllm_config, owner.parallel_config = config, config.parallel_config
    owner.afd_config = owner.parse_config(config)
    owner.device = torch.device("cpu")
    owner.model = owner.speculator = owner.dspark_pd_consumer = None
    owner.hccl_v41_transport = owner.hccl_v41_diagnostics = None
    owner.hccl_v41_graph_eager = owner.hccl_v41_graph_session = None
    owner.hccl_v41_graph_coordinator = owner.hccl_v41_graph_bootstrap = None
    owner.hccl_v41_eager_bootstrap = None
    owner._dspark_pd_load_started = False
    owner._dspark_pd_pending_scheduler = None
    owner._hccl_configuration_logged = False
    owner.connector = FakeRoutedConnector()
    owner.connector.control_plane = None
    owner.prof = owner.cudagraph_manager = owner.ubatch_runner = None
    owner.execute_model_state = None
    owner.pooling_runner = owner.encoder_cache = owner.pp_handler = None
    owner._afd_pending_metadata = None
    owner._afd_suppress_metadata_send = False
    owner._is_warmup = owner._afd_is_graph_capturing = False
    owner._afd_is_graph_replaying = owner._afd_is_profile = False
    owner.ec_connector = ECConnector()
    owner.kvpp = IdleAuxiliaryDeviceBoundary([])
    owner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            profiling_chunk_config=SimpleNamespace(need_timing=False)
        )
    )
    monkeypatch.setattr(
        kv_transfer_state, "_KV_CONNECTOR_AGENT", bundle.producer.connector
    )
    monkeypatch.setattr(
        attention_module,
        "get_tp_group",
        lambda: (
            SimpleNamespace(world_size=1, rank_in_group=0)
            if tp_group is None
            else tp_group
        ),
    )
    monkeypatch.setattr(
        native_runner, "should_skip_allreduce_across_dp_group", lambda _: False
    )
    stub_platform_forward_context(monkeypatch)

    def load_cpu(self, load_dummy_weights=False, *args, **kwargs):
        self.model, self.speculator = prefix.target_model, prefix.speculator

    def allocate_cpu(self, kv_cache_config, kv_cache_allocation_context=None):
        self.kv_cache_config = kv_cache_config
        initialize_loaded_cpu_target(
            monkeypatch,
            self,
            prefix.model_bundle,
            prefix.caches,
            preserve_connector=True,
        )
        self.kv_connector = ActiveKVConnector(config, prefix.caches)
        self.kv_caches = list(prefix.caches.values())
        if stages == PREFILL_CPU_NUM_STAGES:
            bundle.transport = install_native_u2_cpu_runner(
                monkeypatch,
                self,
                kv_cache_config=prefix.kv_config,
                model_bundle=prefix.model_bundle,
            )

    monkeypatch.setattr(native_runner.NPUModelRunner, "load_model", load_cpu)
    monkeypatch.setattr(
        native_runner.NPUModelRunner, "initialize_kv_cache", allocate_cpu
    )
    owner.load_model()
    prefix.consumer = owner.dspark_pd_consumer
    owner.initialize_kv_cache(prefix.kv_config)
    initialize_native_pd_sampling(monkeypatch, SimpleNamespace(owner=owner))
    bundle.scheduler.kv_connector_metadata = bundle.producer.metadata
    # Circular: this protocol factory already consumes the loaded prefix helper.
    from tests.unit.fakes.kv_layout_connector import dspark_query

    bundle.query = dspark_query(prefix)
    bundle.owner, bundle.observations = owner, observations
    return bundle


def make_native_u2_prefill_prefix_case(monkeypatch):
    """Actual no-PD P-prefill U2 execute -> independent legal U1/PD consumer.

    One target instance has one active native ModelState owner. The consumer
    binds that producing target as identity only and owns its separate legal
    U1/PD config/speculator. No guard is patched and no ExecuteModelState or
    auxiliary tensor is manufactured. CPU kernels/cache arithmetic/remote FFN
    leaves validate coordinates and ownership, not device/model equivalence.
    """
    bundle = make_dspark_pd_prefill_bundle(monkeypatch)
    prefix = bundle.prefix
    config = deepcopy(prefix.config)
    eager = hccl_vllm_config(
        dtype=config.model_config.dtype,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        num_ubatches=PREFILL_CPU_NUM_STAGES,
    )
    config.parallel_config = eager.parallel_config
    config.compilation_config = eager.compilation_config
    config.model_config.is_encoder_decoder = eager.model_config.is_encoder_decoder
    config.model_config.is_multimodal_model = eager.model_config.is_multimodal_model
    config.model_config.multimodal_config = eager.model_config.multimodal_config
    config.lora_config = None
    # Freeze both independent configs before target/state/transport binding.
    # The loaded cache-layer identities are shared intentionally; deepcopy of
    # the config must not create a second target writer or registered storage.
    config.compilation_config.static_forward_context = dict(
        prefix.config.compilation_config.static_forward_context
    )
    config.kv_transfer_config = None
    config.parallel_config.use_ubatching = True
    config.parallel_config.enable_dbo = True
    config.parallel_config.num_ubatches = PREFILL_CPU_NUM_STAGES
    config.parallel_config.ubatch_size = PREFILL_CPU_NUM_STAGES
    config.parallel_config.dbo_prefill_token_threshold = PREFILL_CPU_NUM_STAGES
    config.parallel_config.dbo_decode_token_threshold = PREFILL_CPU_NUM_STAGES
    config.parallel_config.rank = 0
    # Native ModelState/request pools consume the same finite CPU context
    # budget as the independently prepared D scheduler/owner.
    config.model_config.max_model_len = SCHEDULER_MAX_MODEL_LEN
    config.scheduler_config = SchedulerConfig(
        max_model_len=SCHEDULER_MAX_MODEL_LEN,
        is_encoder_decoder=False,
        max_num_seqs=len(bundle.requests),
        max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
        enable_chunked_prefill=True,
        disable_hybrid_kv_cache_manager=False,
        async_scheduling=False,
    )
    config.observability_config = ObservabilityConfig()
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    config.compilation_config.cudagraph_capture_sizes = []
    validate_deepseek_v41_pd(config)
    validate_deepseek_v41_speculation(config)
    validate_deepseek_v41_pd(prefix.config)

    owner = AFDNPUAttentionModelRunnerV2.__new__(AFDNPUAttentionModelRunnerV2)
    owner.vllm_config, owner.parallel_config = config, config.parallel_config
    owner.afd_config = owner.parse_config(config)
    owner.model, owner.speculator = prefix.target_model, None
    owner.dspark_pd_consumer = None
    owner.device = torch.device("cpu")
    owner.prof = owner.cudagraph_manager = owner.ubatch_runner = None
    owner.hccl_v41_transport = None
    owner.hccl_v41_graph_eager = owner.hccl_v41_graph_session = None
    owner.hccl_v41_graph_coordinator = owner.hccl_v41_graph_bootstrap = None
    owner.hccl_v41_diagnostics = None
    owner._afd_pending_metadata = None
    owner._afd_suppress_metadata_send = False
    owner._is_warmup = owner._afd_is_graph_capturing = False
    owner._afd_is_graph_replaying = owner._afd_is_profile = False
    owner.pooling_runner = owner.encoder_cache = owner.pp_handler = None
    owner.kv_cache_config = prefix.kv_config
    owner.kv_caches = list(prefix.caches.values())
    owner.kv_connector = KVConnector()
    owner.ec_connector = ECConnector()
    owner.kvpp = IdleAuxiliaryDeviceBoundary([])
    owner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            profiling_chunk_config=SimpleNamespace(need_timing=False)
        )
    )
    owner.execute_model_state = None
    stub_platform_forward_context(monkeypatch)
    monkeypatch.setattr(
        native_runner, "should_skip_allreduce_across_dp_group", lambda _: False
    )
    metadata = initialize_loaded_cpu_target(
        monkeypatch, owner, prefix.model_bundle, prefix.caches
    )
    transport = install_native_u2_cpu_runner(
        monkeypatch,
        owner,
        kv_cache_config=prefix.kv_config,
        model_bundle=prefix.model_bundle,
    )
    pages = tuple(
        torch.empty(0, dtype=torch.uint8).set_(
            prefix.caches[page.layers[0]].untyped_storage(),
            0,
            (page.size_bytes,),
            (1,),
        )
        if isinstance(prefix.caches[page.layers[0]], torch.Tensor)
        else torch.empty(0, dtype=torch.uint8).set_(
            prefix.caches[page.layers[0]][0].untyped_storage(),
            0,
            (page.size_bytes,),
            (1,),
        )
        for page in prefix.layout.pages
    )
    initialize_pd_cpu_source_pages(pages, prefix.layout)
    expected = native_u2_prefill_reference(
        bundle.scheduler, prefix, pages, decode_query_len=owner.decode_query_len
    )
    return SimpleNamespace(
        owner=owner,
        prefix=prefix,
        producer=bundle.producer,
        scheduler=bundle.scheduler,
        sources=bundle.sources,
        requests=bundle.requests,
        transport=transport,
        source_scheduler=bundle.source_scheduler,
        target_metadata=metadata,
        pages=pages,
        expected_target=expected,
    )


def native_u2_prefill_reference(scheduler, prefix, pages, *, decode_query_len):
    """Independent split-coordinate oracle for the declared CPU Attention leaf.

    That leaf reads only the prepared prefix, not intra-chunk causal context.
    A request crossing U2 sees the first child's writes in its second segment.
    This oracle models that explicit CPU contract, not U1 mathematical parity
    or device Attention. It never consumes executed tensors or child metadata.
    Only SWA is read/written by this leaf; distinct target SWA pages make the
    request/chunk-major arithmetic commute with the actual layer-major order.
    Existing long/index/compressor aliases in layer zero's page are preserved.
    """
    width = scheduler.total_num_scheduled_tokens
    split = width // PREFILL_CPU_NUM_STAGES
    expected_pages = tuple(page.clone() for page in pages)
    target_swa_names = {
        f"model.layers.{index}.self_attn.swa_cache"
        for index in range(prefix.config.model_config.hf_text_config.num_hidden_layers)
    }
    target_swa_aliases = {
        view.storage_alias
        for view in prefix.layout.views
        if view.layer_name in target_swa_names and view.tensor_index == 0
    }
    if len(target_swa_aliases) != len(target_swa_names):
        raise ValueError(
            "chunk reference requires distinct registered target SWA pages"
        )
    outputs, auxiliary = [], [[], [], []]
    start = 0
    # Derive the pinned native sort from original scheduler declarations, not
    # the executed InputBatch. All requests here are actual P prefills.
    requests = sorted(
        scheduler.scheduled_new_reqs,
        key=lambda request: (
            not scheduler.scheduled_spec_decode_tokens.get(request.req_id),
            scheduler.num_scheduled_tokens[request.req_id] != decode_query_len,
            scheduler.num_scheduled_tokens[request.req_id],
        ),
    )
    for request in requests:
        count = scheduler.num_scheduled_tokens[request.req_id]
        chunks = [count]
        if start < split < start + count:
            chunks = [split - start, start + count - split]
        offset = 0
        for chunk in chunks:
            reference = native_pd_target_reference(
                torch.tensor(request.prefill_token_ids[offset : offset + chunk]),
                torch.arange(offset, offset + chunk),
                computed=offset,
                block_ids=tuple(tuple(group) for group in request.block_ids),
                layout=prefix.layout,
                pages=expected_pages,
                caches=prefix.caches,
            )
            expected_pages = reference.pages
            outputs.append(reference.hidden_states)
            for actual, expected in zip(
                auxiliary, reference.aux_hidden_states, strict=True
            ):
                actual.append(expected)
            offset += chunk
        start += count
    return SimpleNamespace(
        req_ids=[request.req_id for request in requests],
        hidden_states=torch.cat(outputs),
        aux_hidden_states=tuple(torch.cat(layer) for layer in auxiliary),
        pages=expected_pages,
    )


def native_pd_target_step_reference(
    case,
    scheduler_output: SchedulerOutput,
    *,
    input_ids: torch.Tensor,
    computed: int,
    block_ids: tuple[list[int], ...],
    stages: int = 1,
) -> SimpleNamespace:
    """Snapshot D's current pages before one actual scheduled target step.

    Callers supply independent token/frontier expectations and the scheduler's
    complete absolute block IDs, never incremental CachedRequestData blocks.
    The byte oracle applies only this step's native zeroing and CPU arithmetic;
    it neither observes prepared runner inputs nor initializes a new owner.
    """
    destination = case.pd_read_case.destination_cache
    zeroed = native_zeroed_cache_reference(
        layout=destination.layout,
        config=destination.config,
        caches=destination.caches,
        pages=destination.pages,
        block_ids=scheduler_output.new_block_ids_to_zero,
    )
    if stages == 1:
        return native_pd_target_reference(
            input_ids,
            torch.arange(computed, computed + input_ids.numel()),
            computed=computed,
            block_ids=tuple(tuple(group) for group in block_ids),
            layout=destination.layout,
            pages=zeroed,
            caches=destination.caches,
        )
    if stages != PREFILL_CPU_NUM_STAGES or input_ids.numel() < stages:
        raise ValueError("CPU target reference requires a nonempty U1/U2 step")
    # The declared Attention leaf reads the prepared child prefix. The next
    # child sees earlier-child writes, unlike this leaf's U1 full-query read.
    # Derive this split from the declared scheduler width, never actual child
    # metadata. Separate target SWA pages make chunk/layer order commute.
    target_swa_names = {
        f"model.layers.{index}.self_attn.swa_cache"
        for index in range(
            case.owner.vllm_config.model_config.hf_text_config.num_hidden_layers
        )
    }
    if len(
        {
            view.storage_alias
            for view in destination.layout.views
            if view.layer_name in target_swa_names and view.tensor_index == 0
        }
    ) != len(target_swa_names):
        raise ValueError("U2 reference requires distinct target SWA pages")
    split = input_ids.numel() // stages
    references = []
    pages = zeroed
    for begin, end in ((0, split), (split, input_ids.numel())):
        reference = native_pd_target_reference(
            input_ids[begin:end],
            torch.arange(computed + begin, computed + end),
            computed=computed + begin,
            block_ids=tuple(tuple(group) for group in block_ids),
            layout=destination.layout,
            pages=pages,
            caches=destination.caches,
        )
        references.append(reference)
        pages = reference.pages
    return SimpleNamespace(
        hidden_states=torch.cat([reference.hidden_states for reference in references]),
        aux_hidden_states=tuple(
            torch.cat([reference.aux_hidden_states[index] for reference in references])
            for index in range(len(references[0].aux_hidden_states))
        ),
        mtp_hidden_states=torch.cat(
            [reference.mtp_hidden_states for reference in references]
        ),
        pages=pages,
    )


def initialize_native_pd_sampling(monkeypatch, case):
    """Continue the same native target owner through greedy sample/proposal.

    Keep the registered Sampler and request state. Actual sampler selection,
    RejectionSampler, history bookkeeping and DSpark projection/backbone/Markov
    execute; counters, history stores, target greedy kernel and output copy
    have CPU leaves. No accepted counts or replacement target logits are fed.
    """
    owner = case.owner
    monkeypatch.setattr(native_runner, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(gpu_runner, "post_update", update_speculative_history_cpu)
    monkeypatch.setattr(gpu_runner, "AsyncOutput", CPUCopyAsyncOutput)
    monkeypatch.setattr(sampler_module, "gumbel_sample", greedy_sample_cpu)
    monkeypatch.setattr(
        sampler_module, "get_num_sampled_and_rejected", sampled_and_rejected_cpu
    )
    monkeypatch.setattr(rejection_module, "rejection_sample", greedy_rejection_cpu)
    monkeypatch.setattr(
        rejection_module, "get_num_sampled_and_rejected", sampled_and_rejected_cpu
    )
    owner.rejection_sampler = RejectionSampler(
        owner.sampler, owner.vllm_config.speculative_config, owner.device
    )
    owner.batch_sharder = None
    owner.main_stream = owner.output_copy_stream = None
    owner.check_ep_fault = False
    owner._draft_workspace_lane = 1
    owner.draft_tokens_handler = CPUDraftTokensHandler()
    initialize_existing_dspark_proposal(
        monkeypatch,
        owner.speculator,
        block_tables=owner.block_tables,
        max_num_reqs=owner.max_num_reqs,
        max_num_tokens=owner.max_num_tokens,
        max_model_len=owner.max_model_len,
    )


def make_native_pd_shutdown_case(monkeypatch, *, stages=1, fail_sampling=False):
    """Actual D handoff/sample and Worker shutdown with one CPU release leaf.

    Native Worker.shutdown and ensure_kv_transfer_shutdown stay intact, as do
    the receiver's real thread, typed DONE, retirement and executor join. Only
    the final NPU runner release is a CPU leaf using native cache detachment.
    Its observations describe that release boundary, not device cleanup.
    """
    transfer, runner = make_scheduled_pd_runner_case(
        monkeypatch,
        prompt_length=6,
        stages=stages,
        started_receiver=True,
        complete_hccl=True,
    )
    try:
        scheduled = complete_native_pd_handoff(runner)
        initialize_native_pd_target_execution(
            monkeypatch, runner, scheduler_output=scheduled
        )
        initialize_native_pd_sampling(monkeypatch, runner)
        owner = runner.owner
        if stages == 1:
            runner.transport = PreKVAttentionPeer(
                validate_staged_hccl_runtime(owner.vllm_config, owner.afd_config, 0)
            )
            owner.hccl_v41_transport = runner.transport
        else:
            owner.ubatch_runner._pending_finish = None
        connector = HcclStagedAFDConnector.__new__(HcclStagedAFDConnector)
        connector.transport = runner.transport
        connector.overlap_settings = None
        connector._pending_context = connector._pending_tensor_spec = None
        owner.connector = connector
        owner._afd_shutdown_prepared = owner._afd_staged_stop_completed = False
        owner.hccl_v41_eager_bootstrap = None
        assert owner.execute_model(scheduled) is None
        cause = RuntimeError("controlled native D shutdown proposal failure")

        def fail_draft(layer, inputs):
            raise cause

        hook = None
        if fail_sampling:
            draft_layer = next(iter(owner.speculator.model.model.layers.values()))
            hook = draft_layer.self_attn.register_forward_pre_hook(fail_draft)
        try:
            output = owner.sample_tokens(None).get_output()
        except RuntimeError as error:
            if not fail_sampling or error is not cause:
                raise
            output = None
        finally:
            if hook is not None:
                hook.remove()
            if stages == 2:
                runner.target_hook.remove()
        if fail_sampling and output is not None:
            raise AssertionError(
                "native proposal did not reach the failing CPU Attention"
            )
        case = SimpleNamespace(
            transfer=transfer,
            runner=runner,
            worker=AFDNPUAttentionWorker.__new__(AFDNPUAttentionWorker),
            owner=owner,
            receiver=transfer.worker.kv_recv_thread,
            transport=runner.transport,
            target=owner.model,
            draft=owner.speculator.model,
            target_views=tuple(
                layer.self_attn.dsa_attn.swa_cache_layer.kv_cache
                for layer in owner.model.language_model.model.layers
            ),
            draft_views=tuple(
                layer.self_attn.dsa_attn.swa_cache_layer.kv_cache
                for layer in owner.speculator.model.model.layers.values()
            ),
            pages=tuple(page.clone() for page in transfer.destination_cache.pages),
            cause=cause if fail_sampling else None,
            output=output,
            release_observation=None,
            close=transfer.close,
        )
        case.worker.model_runner = owner
        case.worker.profiler = case.worker.profiler_config = None
        case.worker.weight_transfer_engine = None
        queue_graph_runner_stop(case)

        def release_cpu_storage(native_owner):
            case.release_observation = {
                "kv_owner_cleared": kv_transfer_state._KV_CONNECTOR_AGENT is None,
                "receiver_retired": case.receiver.retired,
                "receiver_alive": case.receiver.is_alive(),
                "stop_acknowledged": any(
                    direction == "recv" and message.kind == "STOP_ACK"
                    for direction, _, _, message in case.transport.transcript
                ),
            }
            clear_layer_kv_caches(case.target.modules())
            clear_layer_kv_caches(case.draft.modules())
            native_owner.kv_caches.clear()

        monkeypatch.setattr(
            native_runner.NPUModelRunner, "shutdown", release_cpu_storage
        )
        return case
    except BaseException:
        transfer.close()
        raise


class ReplayDeviceMetadata(AuxiliaryDeviceBoundary):
    """Attention metadata and replay device work are explicit CPU boundaries."""

    def __init__(self, events, *, native_metadata=None):
        super().__init__(events)
        self.attn_metadata = None
        self.replay_live = False
        self.prepared_once = False
        self.native_metadata = native_metadata
        self.executor = None if native_metadata is None else native_metadata.executor

    def begin_preparation(self, *, full_graph=True):
        if self.replay_live or (
            self.prepared_once and (self.lookup_live or self.metadata_live)
        ):
            raise AssertionError(
                "previous target metadata/lookup lease was not retired"
            )
        self.lookup_live = self.metadata_live = True
        self.replay_live = full_graph
        self.reset_events = None
        self.prepared_once = True

    @contextmanager
    def build(self, attn_groups, full_graph):
        # U2 retains the real native single-parent build frontier. Providers
        # and its task executor are the existing shared CPU device boundary.
        self.begin_preparation(full_graph=full_graph)
        with self.native_metadata.build(attn_groups, full_graph):
            yield
        self.events.append("attn-ready")

    def run_build(self, build_metadata, **kwargs):
        # Keep native EngramModelState.prepare_attn, including its real FULL
        # mode and buffer ownership. Do not invoke a device metadata kernel.
        self.begin_preparation(full_graph=kwargs["full_graph_mode"])
        self.attn_metadata = {
            "child": SimpleNamespace(
                padded_tokens=kwargs["num_tokens"],
                valid_tokens=kwargs["num_actual_tokens"],
                query_start_loc=kwargs["query_start_loc_gpu"],
                block_tables=kwargs["block_tables"],
                slot_mappings=kwargs["slot_mappings"],
            )
        }
        self.events.append("attn-ready")
        return self.attn_metadata

    def prepare_child(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
    ):
        # The native HCCL runner builds one FULL metadata parent, but NONE
        # reaches this existing CPU child-operator leaf directly. Open its
        # parent lease at the first actual child preparation, never in a test.
        # No eager SFA metadata task execution is claimed by this snapshot.
        if cudagraph_mode == CUDAGraphMode.NONE and ubatch_idx == 0:
            self.begin_preparation(full_graph=False)
        return snapshot_ascend_child(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            ubatch_idx,
        )

    def finish_replay(self):
        if not self.replay_live:
            raise AssertionError("replay metadata must have been prepared")
        if self.native_metadata is not None:
            self.native_metadata.finish_replay()
        self.replay_live = False
        self.events.append("metadata-replay-finish")

    def finish(self):
        if self.native_metadata is not None:
            self.native_metadata.finish()
        super().finish()


class FactoryDeviceMetadata(ReplayDeviceMetadata):
    """Keep native capture/forward retirement over the existing CPU task leaf.

    Capture has no live Engram lookup in this no-Engram model. Its metadata
    submission retires in ModelWithContext, followed by the manager's idempotent
    finish. Runtime replay instead retires lookup and replay leases in the
    unchanged native execute finally. No test manually reopens either lease.
    """

    def __init__(self, events, *, native_metadata):
        super().__init__(events, native_metadata=native_metadata)
        self.lookup_live = self.metadata_live = False

    def begin_preparation(self, *, full_graph=True):
        super().begin_preparation(full_graph=full_graph)
        reservation = self.transport.graph_reservation
        if reservation is not None and reservation.phase in ("warmup", "capture"):
            self.lookup_live = self.replay_live = False

    def begin_forward(self):
        self.native_metadata.begin_forward()

    def finish(self):
        self.native_metadata.finish()
        if self.metadata_live:
            AuxiliaryDeviceBoundary.finish(self)


class ReplayKVTransferBoundary(KVTransferBoundary):
    """Validate the real positive native pre_forward metadata and coordinates."""

    def __init__(self, events, batch, metadata, *, stages=1):
        super().__init__(events, fail_load=False)
        self.batch, self.metadata = batch, metadata
        self.stages = stages

    def start_load_kv(self, context, **kwargs):
        expected_metadata = None if self.stages == 2 else self.metadata.attn_metadata
        # Native FULL invokes pre_forward before its model context, passing
        # explicit metadata. NONE invokes it inside its real parent context
        # without that keyword. U2 retains native parent metadata=None in both.
        full_graph = "attn_metadata" in kwargs
        actual_metadata = (
            kwargs["attn_metadata"] if full_graph else context.attn_metadata
        )
        if (
            full_graph
            and context.attn_metadata is not None
            or actual_metadata is not expected_metadata
        ):
            raise AssertionError("KV load lost the native FULL/NONE metadata boundary")
        if (
            kwargs["request_ids"] != self.batch.req_ids
            or kwargs["request_state_indices"] is not self.batch.idx_mapping
            or kwargs["num_tokens"] != self.batch.num_tokens
        ):
            raise AssertionError("positive KV load lost native prepared coordinates")
        self.events.append("kv-load")


class PreparedReplayInputs:
    """Replace only input-materialization kernels, preserving native gather."""

    def __init__(self, batch, events, *, stages=1, allow_eager=False):
        self.batch, self.events = batch, events
        self.stages = stages
        self.allow_eager = allow_eager

    def __call__(self, scheduler, state, descriptor):
        if (
            state.req_ids != self.batch.req_ids
            or tuple(state.num_scheduled_tokens)
            != tuple(self.batch.num_scheduled_tokens)
            or state.num_tokens != self.batch.num_tokens
            or descriptor.cg_mode
            not in (
                (CUDAGraphMode.FULL, CUDAGraphMode.NONE)
                if self.allow_eager
                else (CUDAGraphMode.FULL,)
            )
            or descriptor.num_tokens != self.batch.num_tokens_after_padding
            or descriptor.num_ubatches != self.stages
        ):
            raise AssertionError(
                "native dispatch must precede CPU input materialization"
            )
        self.events.append("inputs-ready")
        return self.batch


class ReplayBlockTables:
    """Persistent CPU views consumed by the unchanged GPU prepare_attn method."""

    def __init__(self, rows):
        self.input_block_tables = (torch.zeros(rows, 2, dtype=torch.int32),)
        self.slot_mappings = torch.zeros(1, rows, dtype=torch.int64)

    def apply_staged_writes(self):
        pass

    def gather_block_tables(self, idx_mapping, num_reqs_padded):
        return tuple(table[:num_reqs_padded] for table in self.input_block_tables)

    def compute_slot_mappings(
        self, idx_mapping, query_start_loc, positions, num_tokens_padded
    ):
        return self.slot_mappings[:, :num_tokens_padded]

    def get_dummy_block_tables(self, num_reqs):
        return tuple(table[:num_reqs] for table in self.input_block_tables)

    def get_dummy_slot_mappings(self, num_tokens):
        return self.slot_mappings[:, :num_tokens]


class CPUGraphAllocationDevice(str):
    """CPU allocation string with explicit NPU-labelled constructor metadata.

    Torch parses the immutable value ``cpu``. Only the component's graph-owner
    metadata reads the NPU label/index; neither that admission nor actual NPU
    device binding is validated by this facade. All retained tensors stay CPU.
    """

    def __new__(cls):
        return super().__new__(cls, "cpu")

    @property
    def type(self):
        return "npu"

    @property
    def index(self):
        return 0


def make_native_pd_graph_case(monkeypatch, *, stages=1, fail_replay=False):
    """Same D owner receives PD, executes eager, then consumes a static bundle.

    Native preparation, ActiveKV, graph selection, FULL manager host lifecycle
    and global-parent sampling stay real. A supplied inert handle/output is an
    input fixture, not a capture result or new-input Graph arithmetic. The CPU
    allocation facade does not establish the real device-binding guard. P is
    independently eager and both role configurations precede model construction.
    """
    # Native graph-parameter allocation is a process/device leaf here. Retain
    # pytest's original binding for restoration and isolate each CPU case; do
    # not claim these host parameters represent a live NPU graph pool.
    monkeypatch.setattr(native_acl_graph, "_graph_params", None)
    transfer, case = make_scheduled_pd_runner_case(
        monkeypatch,
        prompt_length=6,
        stages=stages,
        complete_hccl=True,
        target_graph=True,
    )
    owner = case.owner
    owner.compilation_config = owner.vllm_config.compilation_config
    initialize_native_pd_target_execution(monkeypatch, case, graph_target=True)
    initialize_native_pd_sampling(monkeypatch, case)
    # The constructor-free CPU owner declares the native model-state default.
    # U2 prepares child metadata and leaves its parent at this initial None.
    owner.model_state.attn_metadata = None
    caps = validate_staged_hccl_runtime(
        owner.vllm_config, owner.afd_config, 0, graph=True
    )
    case.transport = PreKVAttentionPeer(caps)
    owner.hccl_v41_transport = case.transport
    case.session = HcclV41GraphSession(case.transport)
    # The native sampler/request/input constructors above keep an actual CPU
    # device. Later tensor allocations parse this facade as CPU as well.
    owner.device = CPUGraphAllocationDevice()
    owner.pcp_manager = None
    owner.intermediate_tensors = None
    case.cache = HcclGraphCache(DeviceCompletion().synchronize)
    case.generation, case.metadata = NativeGraphGeneration(), NativeGraphMetadata()
    ids = count()
    case.next_transaction_id = lambda: f"pd-graph-{next(ids)}"
    control = HcclV41GraphControl(
        case.transport,
        None,
        case.cache,
        case.generation,
        metadata_shapes=case.metadata,
    )
    eager = HcclV41GraphEager(
        owner, control, next_transaction_id=case.next_transaction_id
    )
    owner.bind_hccl_v41_graph_eager(eager, case.session)
    if owner.parallel_config.use_ubatching:
        install_native_u2_cpu_runner(
            monkeypatch,
            owner,
            kv_cache_config=transfer.destination_cache.config,
            model_bundle=transfer.loaded_destination.model_bundle,
            graph=True,
            transport=case.transport,
        )
        owner.ubatch_runner.graph_owner = case.session
        observe_native_pd_target_children(case)
    owner.update_stream = SimpleNamespace(label="cpu-inert-replay")
    manager = ModelAclGraphManager(
        owner.vllm_config,
        owner.device,
        CUDAGraphMode.FULL_DECODE_ONLY,
        owner.decode_query_len,
        owner,
        ubatch_runner=owner.ubatch_runner,
    )
    owner.cudagraph_manager, control.manager = manager, manager
    manager.hidden_states, manager.aux_hidden_states = None, []
    manager.intermediate_tensors = None
    monkeypatch.setattr(
        AscendSFABackend, "get_impl_cls", staticmethod(lambda: AscendSFAImpl)
    )
    # Native metadata sees a backend with no device-producer tasks. Its actual
    # TargetDeviceMetadata/run_build/finish state machine remains unchanged.
    owner.attn_groups = [
        [SimpleNamespace(backend=AscendSFABackend, get_metadata_builder=lambda _: None)]
    ]
    if owner.ubatch_runner is not None:
        owner.ubatch_runner.attn_groups = owner.attn_groups
    coordinator = coordinator_module.HcclV41GraphCoordinator(
        owner,
        control,
        case.session,
        next_transaction_id=case.next_transaction_id,
    )
    owner.bind_hccl_v41_graph_coordinator(coordinator)
    case.manager, case.control = manager, control
    descriptor = next(
        descriptor
        for descriptor in manager._capture_descs[CUDAGraphMode.FULL]
        if descriptor.num_ubatches == stages
    )
    manager.host_descriptors = (descriptor,)
    geometry = next(
        geometry
        for geometry in coordinator.coverage.geometries
        if geometry.descriptor == descriptor
    )
    rows = tuple(stage.token_stop - stage.token_start for stage in geometry.stages)
    original = make_graph_stage_bank(
        rows=rows,
        layers=case.transport.negotiated_capabilities.layer_indices,
        target_query_len=owner.decode_query_len,
    )
    case.model_case = SimpleNamespace(
        bank=original,
        contexts=tuple(
            SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL, ubatch_idx=index)
            for index in range(stages)
        ),
    )
    case.transport.data_groups = [
        exchange.endpoint.group for exchange in original.exchanges
    ]
    # These independent declared values have three auxiliary layers, matching
    # the real PD consumer/model schema. Replay must return these exact tensors.
    hidden = torch.arange(descriptor.num_tokens * 4, dtype=torch.bfloat16).reshape(
        -1, 4
    )
    auxiliary = tuple(hidden + ordinal for ordinal in (2, 4, 6))
    _publish_cpu_replay_fixture(case, coordinator, outputs=(hidden, auxiliary))
    # This is part of the supplied recorded-manager state, not a completed
    # capture assertion. Native dispatch must still select its own descriptor.
    manager._graphs_captured = True
    case.replay_error = (
        RuntimeError("controlled D PD FULL replay failure") if fail_replay else None
    )
    case.replayed = []

    def replay_device(manager, backend, descriptor, num_tokens, attn_metadata):
        if owner.model_state.attn_metadata is not attn_metadata:
            raise AssertionError(
                "FULL replay must retain this parent's prepared metadata"
            )
        case.replayed.append((descriptor, num_tokens))
        if case.replay_error is not None:
            raise case.replay_error
        manager.graphs[descriptor].replay()
        return manager.hidden_states, manager.aux_hidden_states

    manager._graph_relay = MethodType(replay_device, manager)
    case.recorded_outputs = hidden, auxiliary
    case.startup_pd_messages = tuple(transfer.sent)
    return transfer, case


def _publish_cpu_replay_fixture(case, coordinator, *, outputs=None):
    """Publish a pre-recorded CPU bundle as input, without claiming capture.

    The real control/cache/resource types enforce its ownership. No native
    capture factory or graph body is invoked by this fixture; the later native
    run_fullgraph keeps its Python lifecycle around an explicit device boundary.
    """
    owner, manager = case.owner, coordinator.manager
    descriptor = manager.host_descriptors[0]
    reservation = coordinator.control.reserve_capture(
        descriptor, warmup=False, transaction_id="recorded-cpu-fixture"
    )
    key = reservation.decision.key
    original = case.model_case.bank
    bank = HcclGraphStageBank(
        role="attention",
        role_rank=case.transport.role_rank,
        capabilities=case.transport.negotiated_capabilities,
        key=key,
        exchanges=original.exchanges,
    )
    execution = HcclV41GraphExecution(
        owner.vllm_config,
        owner.model,
        owner.model_state,
        coordinator.control,
        reservation,
        bank,
    )
    execution.parent_body = DeepseekV41GraphParent(
        DeepseekV41GraphStages(
            owner.model.language_model.model,
            tuple(case.model_case.contexts),
            tuple(
                slice(stage.token_start, stage.token_stop)
                for stage in reservation.geometry.stages
            ),
            bank,
        )
    )
    graph = ReplayGraph()
    manager.graphs[descriptor] = graph
    if outputs is None:
        manager.hidden_states = torch.arange(
            descriptor.num_tokens * bank.capabilities.hidden_size, dtype=torch.float32
        ).reshape(descriptor.num_tokens, bank.capabilities.hidden_size)
        manager.aux_hidden_states = (
            [manager.hidden_states + 2, manager.hidden_states + 6]
            if owner.use_aux_hidden_state_outputs
            else []
        )
    else:
        manager.hidden_states, auxiliary = outputs
        manager.aux_hidden_states = list(auxiliary)
    resources = HcclV41GraphResources(
        bank=bank,
        execution=execution,
        model_state=owner.model_state,
        input_buffers=owner.input_buffers,
        ubatch_runner=owner.ubatch_runner,
        attn_groups=tuple(tuple(group) for group in owner.attn_groups),
        block_tables=owner.block_tables,
        kv_cache_config=owner.kv_cache_config,
        kv_caches=tuple(owner.kv_caches),
        manager=manager,
        hidden_states=manager.hidden_states,
        auxiliary=tuple(manager.aux_hidden_states),
        intermediate_tensors=manager.intermediate_tensors,
    )
    case.cache.reserve_capture(key, reservation.generation, graph, resources)
    case.transport.submit_forward(
        HcclLocalForward("warmup", (0,) * key.num_ubatches, key.padded_tokens),
        transaction_id=reservation.decision.transaction_id,
    )
    case.transport.complete_graph_forward(key)
    case.cache.publish_capture(key, reservation.generation)
    case.transport.states[0] = "ready"
    case.recorded_graph, case.recorded_resources = graph, resources
    case.replay_baseline = tuple(case.transport.transcript)


def make_native_full_kv_replay_case(
    monkeypatch,
    *,
    fail_replay=False,
    target_query_len=1,
    auxiliary=False,
    stages=1,
    eager_fallback=False,
    capture_settings=None,
):
    """Real AFD -> Ascend -> GPU FULL execution over explicit CPU inputs.

    prepare_inputs and _graph_relay are the two instance method boundaries.
    Native execute, gather, prepare_attn, run_fullgraph and Engram retirement
    are unchanged. The static cache is a supplied fixture, not a capture result.
    """
    case = make_native_kv_runner_case(
        monkeypatch,
        post_kv=False,
        sync_load=True,
        target_query_len=target_query_len,
        stages=stages,
    )
    owner = case.owner
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda _: None
    )
    # Select the real non-offload SFA implementation without global device
    # configuration. Native graph-backend discovery and replay stay intact.
    monkeypatch.setattr(
        AscendSFABackend, "get_impl_cls", staticmethod(lambda: AscendSFAImpl)
    )
    rows = case.native_state.num_tokens
    batch = make_ascend_decode_batch(rows, target_query_len=target_query_len)
    batch.req_ids = [f"decode-{index}" for index in range(batch.num_reqs)]
    batch.is_dummy = False
    batch.is_padding.fill_(False)
    batch.num_computed_tokens_np.fill(FULL_REPLAY_HISTORY_TOKENS)
    batch.prefill_len_np.fill(FULL_REPLAY_HISTORY_TOKENS)
    batch.num_computed_prefill_tokens_np.fill(FULL_REPLAY_HISTORY_TOKENS)
    batch.seq_lens.fill_(FULL_REPLAY_HISTORY_TOKENS + target_query_len)
    batch.seq_lens_np.fill(FULL_REPLAY_HISTORY_TOKENS + target_query_len)
    case.batch = batch
    case.eager_fallback = eager_fallback
    owner.req_states = SimpleNamespace(
        req_id_to_index=dict(zip(batch.req_ids, range(batch.num_reqs), strict=True)),
        num_computed_tokens_np=batch.num_computed_tokens_np,
        prefill_len=SimpleNamespace(np=batch.prefill_len_np),
        num_computed_prefill_tokens=batch.num_computed_prefill_tokens_np,
        num_computed_tokens=SimpleNamespace(
            gpu=torch.full((batch.num_reqs,), FULL_REPLAY_HISTORY_TOKENS)
        ),
    )
    # Remove the lower-level builder's gather lambda: execute must select and
    # preserve the actual native request ordering before the prepared input seam.
    del owner.gather_batch_req_state
    owner.prepare_inputs = PreparedReplayInputs(
        batch, case.native_events, stages=stages, allow_eager=eager_fallback
    )
    owner.block_tables = ReplayBlockTables(rows)
    owner.input_buffers = (
        SimpleNamespace(dcp_local_seq_lens=None)
        if capture_settings is None
        else AscendInputBuffers(batch.num_reqs, rows, torch.device("cpu"))
    )
    owner.model_config = owner.vllm_config.model_config
    owner.observability_config = SimpleNamespace(cudagraph_metrics=False)
    owner.lora_config = owner.adaptive_verification = None
    owner.is_encoder_decoder = owner.uses_inputs_embeds = False
    owner.is_first_pp_rank = owner.is_last_pp_rank = True
    owner.use_aux_hidden_state_outputs = auxiliary
    owner.routed_experts_capturer = owner.speculator = None
    owner.prompt_logprobs_worker = None
    owner.use_spec_pp = False
    owner.dp_size, owner.dp_rank = 1, 0
    owner.dcp_size, owner.dcp_rank, owner.cp_interleave = 1, 0, 1
    owner.parallel_config.enable_eplb = False
    owner.eplb = EPLBController(owner.parallel_config, torch.device("cpu"))
    owner.step_timing = StepTimingCollector()
    owner.attn_groups = [[SimpleNamespace(backend=AscendSFABackend)]]
    native_metadata = None
    if stages == 2:
        native_metadata, _, case.runner.stage_attn_groups = make_cpu_target_metadata()
        # The existing CPU metadata leaf consumes the original native sliced
        # InputBatch. HCCL/NPU/UBatchRunner prepare and FULL contexts remain real.
        case.runner.model_state = owner.model_state
        case.runner.attn_groups = owner.attn_groups
        owner.ubatch_runner = case.runner
    metadata_type = (
        ReplayDeviceMetadata if capture_settings is None else FactoryDeviceMetadata
    )
    metadata = metadata_type(case.native_events, native_metadata=native_metadata)
    if capture_settings is not None:
        metadata.transport = case.transport
    if eager_fallback and stages == 2:
        case.runner._prepare_attn = metadata.prepare_child
    owner.model_state.device_metadata = metadata
    owner.model_state.vllm_config = owner.vllm_config
    owner.model_state.max_model_len = FULL_REPLAY_HISTORY_TOKENS + rows
    owner.model_state.rope_state = owner.model_state.lookback_token_ids = None
    owner.model.retire_engram_lookups = metadata.retire_engram_lookups
    case.auxiliary_device = metadata
    transfer = ReplayKVTransferBoundary(
        case.native_events, batch, metadata, stages=stages
    )
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", transfer)
    owner.kv_connector = ActiveKVConnector(owner.vllm_config, case.kv_views)
    case.kv_transfer = transfer
    if eager_fallback or capture_settings is not None:
        # Freeze the deterministic CPU weights before the recorded bundle.
        # Actual next anchors can exceed the old toy model's 32-row vocabulary.
        native = owner.model.language_model.model
        capabilities = case.transport.negotiated_capabilities
        native.embed_tokens = torch.nn.Embedding.from_pretrained(
            torch.arange(capabilities.vocab_size * capabilities.hidden_size)
            .reshape(capabilities.vocab_size, capabilities.hidden_size)
            .bfloat16(),
            freeze=True,
        )
        native.engram_rotation = torch.empty(0, dtype=torch.bfloat16)
        # These are the actual native no-Engram public branches. Their empty
        # CPU mask storage does not validate lookup/hash or NPU producer work.
        language_model = owner.model.language_model
        language_model.prepare_engram_inputs = MethodType(
            AscendDeepseekV41LLMForCausalLM.prepare_engram_inputs, language_model
        )
        language_model.prepare_engram_graph_inputs = MethodType(
            AscendDeepseekV41LLMForCausalLM.prepare_engram_graph_inputs, language_model
        )
        case.engram_hook.remove()
        # Installs the actual replacement ModelState's pre-hook exactly once;
        # control/negotiation/cursors stay real, only tensor backend is bounded.
        install_runtime_eager_tensor_boundary(case)
    case.scheduler.num_scheduled_tokens = dict.fromkeys(batch.req_ids, target_query_len)
    case.scheduler.total_num_scheduled_tokens = rows
    case.transport.data_groups = [
        endpoint.group for endpoint in case.model_case.endpoints
    ]
    coordinator = install_graph_runner_coordinator(
        case, overlap_settings=capture_settings
    )
    case.manager = coordinator.manager
    case.manager.vllm_config = owner.vllm_config
    case.manager.update_stream = SimpleNamespace(label="device-replay-boundary")
    owner.bind_hccl_v41_graph_coordinator(coordinator)
    if capture_settings is None:
        _publish_cpu_replay_fixture(case, coordinator)
    case.replay_error = (
        RuntimeError("controlled FULL replay failure") if fail_replay else None
    )

    def replay_device(manager, backend, descriptor, num_tokens, attn_metadata):
        if (
            owner.model_state.attn_metadata is not attn_metadata
            or num_tokens != batch.num_tokens
            or attn_metadata["child"].padded_tokens
            != descriptor.num_tokens // descriptor.num_ubatches
        ):
            raise AssertionError("native replay received another batch's metadata")
        case.native_events.append("replay-device-boundary")
        if case.replay_error is not None:
            raise case.replay_error
        manager.graphs[descriptor].replay()
        if owner.use_aux_hidden_state_outputs:
            return manager.hidden_states, manager.aux_hidden_states
        return manager.hidden_states

    case.manager._graph_relay = MethodType(replay_device, case.manager)
    return case


class DecodeLogitsBoundary:
    """Deterministic CPU head, reached through the real model wrapper."""

    def __call__(self, hidden_states):
        centers = torch.arange(FULL_REPLAY_ROWS, dtype=hidden_states.dtype)
        centers *= hidden_states.shape[1]
        return -(hidden_states[:, :1] - centers).square()


class DecodeSamplerBoundary:
    """A q1/no-prefill CPU operator returning the real native output type."""

    def __init__(self, events, *, fail):
        self.events = events
        self.error = RuntimeError("controlled sampler failure") if fail else None
        self.penalties_state = SimpleNamespace(output_bin_counts=None)

    def apply_staged_writes(self):
        # There are no new requests or pending sampling state in this fixture.
        pass

    def __call__(self, logits, batch):
        if (
            logits.device.type != "cpu"
            or batch.num_draft_tokens != 0
            or not np.all(batch.num_scheduled_tokens == 1)
            or not np.all(batch.num_computed_prefill_tokens_np == batch.prefill_len_np)
        ):
            raise AssertionError("sampling boundary is resident decode q1 only")
        self.events.append("sample-boundary")
        if self.error is not None:
            raise self.error
        return SamplerOutput(
            sampled_token_ids=logits.argmax(dim=-1, keepdim=True),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.ones(batch.num_reqs, dtype=torch.int32),
            num_rejected=torch.zeros(batch.num_reqs, dtype=torch.int32),
        )


def update_decode_history_cpu(
    idx_mapping,
    num_computed_tokens,
    last_sampled_tokens,
    output_bin_counts,
    sampled_tokens,
    num_sampled,
    num_rejected,
    query_start_loc,
    all_token_ids,
    total_len,
):
    """Explicit replacement for input_batch.post_update's Triton operator.

    This implements only unmasked q1, one sampled token and zero rejections.
    Native postprocess_sampled and ModelState.postprocess_state stay unchanged;
    it does not establish the device kernel or DSpark history semantics.
    """
    if (
        output_bin_counts is not None
        or sampled_tokens.shape != (idx_mapping.numel(), 1)
        or not torch.all(num_sampled == 1)
        or not torch.all(num_rejected == 0)
        or not torch.all(torch.diff(query_start_loc) == 1)
        or not torch.all(idx_mapping >= 0)
    ):
        raise AssertionError("history boundary supports unmasked decode q1 only")
    for row, request_index in enumerate(idx_mapping.tolist()):
        position = int(total_len[request_index])
        token = sampled_tokens[row, 0]
        all_token_ids[request_index, position] = token
        last_sampled_tokens[request_index] = token
        total_len[request_index] += 1
        num_computed_tokens[request_index] += 1


class CPUCopyAsyncOutput(AsyncOutput):
    """Supply a CPU copy owner; inherit the actual native get_output body.

    Only constructor event/stream/D2H work is outside this seam. The native
    tensor-to-NumPy helper and final output assembly remain real. No device
    synchronization or async-copy behavior is simulated as accepted.
    """

    def __init__(
        self,
        model_runner_output,
        sampler_output,
        num_sampled_tokens,
        main_stream,
        copy_stream,
        check_ep_fault=False,
        routed_experts=None,
    ):
        if (
            check_ep_fault
            or routed_experts is not None
            or sampler_output.logprobs_tensors is not None
            or sampler_output.num_nans is not None
            or sampler_output.sampling_mask_tensors is not None
            or model_runner_output.prompt_logprobs_dict
            or sampler_output.sampled_token_ids.device.type != "cpu"
        ):
            raise AssertionError("copy boundary supports CPU token-only output")
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.routed_experts = None
        self.copy_event = DeviceCompletion()
        self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
        self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
        self.logprobs_tensors = self.num_nans = self.sampling_mask_tensors = None
        self.routed_experts_cpu = None
        self.prompt_logprobs_dict = {}
        self._has_fault = None


def make_native_full_kv_sampling_case(monkeypatch, *, fail_sample=False):
    """Continue the actual FULL execute(None) on the same sampling owner.

    There is no new request/prefill, TP head, grammar, rejection or drafter.
    The only new native module substitutions are head-TP configuration, the
    post_update device operator and AsyncOutput's device-copy constructor.
    """
    case = make_native_full_kv_replay_case(monkeypatch)
    owner = case.owner
    monkeypatch.setattr(native_runner, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(gpu_runner, "post_update", update_decode_history_cpu)
    monkeypatch.setattr(gpu_runner, "AsyncOutput", CPUCopyAsyncOutput)
    owner.model.language_model.compute_logits = DecodeLogitsBoundary()
    owner.sampler = DecodeSamplerBoundary(case.native_events, fail=fail_sample)
    owner.prompt_logprobs_worker = PromptLogprobsWorker(FULL_REPLAY_ROWS)
    owner.batch_sharder = owner.rejection_sampler = owner.pcp_manager = None
    owner.main_stream = owner.output_copy_stream = None
    owner.check_ep_fault = owner.use_pp = owner.sync_spec_pp_cpu_counts = False
    owner.num_speculative_steps = 0
    state = owner.req_states
    state.prompt_len = SimpleNamespace(np=case.batch.prefill_len_np.copy())
    state.last_sampled_tokens = torch.full((FULL_REPLAY_ROWS,), -1, dtype=torch.int64)
    state.total_len = SimpleNamespace(
        gpu=torch.full((FULL_REPLAY_ROWS,), FULL_REPLAY_HISTORY_TOKENS)
    )
    history = torch.full(
        (FULL_REPLAY_ROWS, owner.model_state.max_model_len), -1, dtype=torch.int64
    )
    history[:, :FULL_REPLAY_HISTORY_TOKENS] = torch.arange(FULL_REPLAY_HISTORY_TOKENS)
    state.all_token_ids = SimpleNamespace(gpu=history)
    case.initial_history = history.clone()
    case.sampling_error = owner.sampler.error
    return case


class SpeculativeLogitsBoundary:
    """CPU head whose actual logits independently drive three rejection cases."""

    def __init__(
        self, hidden_size, vocab_size, *, ordinal_rows=False, predictions=None
    ):
        self.hidden_size, self.vocab_size = hidden_size, vocab_size
        self.ordinal_rows = ordinal_rows
        # Request 0: both drafts match; request 1: first draft mismatches;
        # request 2: first matches and second mismatches. Final rows are bonus.
        self.predictions = torch.tensor(
            [11, 12, 13, 29, 22, 23, 31, 39, 33] if predictions is None else predictions
        )

    def __call__(self, hidden_states):
        rows = (
            torch.arange(hidden_states.shape[0])
            if self.ordinal_rows
            else (hidden_states[:, 0] / self.hidden_size).to(torch.int64)
        )
        logits = hidden_states.new_full((rows.numel(), self.vocab_size), -20)
        return logits.scatter_(1, self.predictions[rows, None], 20)


class CPUDraftTokensHandler(DraftTokensHandler):
    """Native no-grammar token-handler methods without device copy allocation."""

    def __init__(self):
        self.req_ids, self.draft_tokens_np, self.num_draft_tokens = [], None, 0


def make_native_dspark_sampling_case(
    monkeypatch,
    proposal_factory,
    *,
    fail_backbone=False,
    stages=1,
    eager_fallback=False,
    capture_settings=None,
):
    """Actual target FULL -> sample/reject -> local DSpark -> KV post-forward.

    The target has three requests, q3 (anchor plus two draft tokens). Its
    persistent CPU graph output is supplied, not captured. Rejection/counts,
    post_update and lookback gather are explicit CPU operator boundaries;
    native runner, RejectionSampler and DSpark proposal bodies remain real.
    """
    case = make_native_full_kv_replay_case(
        monkeypatch,
        target_query_len=3,
        auxiliary=True,
        stages=stages,
        eager_fallback=eager_fallback,
        capture_settings=capture_settings,
    )
    owner, batch = case.owner, case.batch
    if (batch.num_reqs, batch.num_tokens, owner.decode_query_len) != (3, 9, 3):
        raise AssertionError("verification fixture requires three actual q3 requests")
    monkeypatch.setattr(native_runner, "lmhead_tp_enable", lambda: False)
    monkeypatch.setattr(gpu_runner, "post_update", update_speculative_history_cpu)
    monkeypatch.setattr(gpu_runner, "AsyncOutput", CPUCopyAsyncOutput)
    monkeypatch.setattr(rejection_module, "rejection_sample", greedy_rejection_cpu)
    monkeypatch.setattr(
        rejection_module, "get_num_sampled_and_rejected", sampled_and_rejected_cpu
    )
    monkeypatch.setattr(engram_lookback, "_gather_lookback_kernel", CPULookbackGather())
    batch.input_ids.copy_(torch.tensor([10, 11, 12, 20, 21, 22, 30, 31, 32]))
    batch.positions.copy_(torch.arange(12, 15).repeat(3))
    batch.logits_indices = torch.arange(batch.num_tokens)
    batch.cu_num_logits = batch.query_start_loc[: batch.num_reqs + 1].clone()
    batch.cu_num_logits_np = batch.query_start_loc_np[: batch.num_reqs + 1].copy()
    batch.num_draft_tokens_per_req = np.full(batch.num_reqs, 2, dtype=np.int32)
    batch.num_draft_tokens = int(batch.num_draft_tokens_per_req.sum())
    batch.expanded_idx_mapping = torch.repeat_interleave(batch.idx_mapping, 3)
    batch.expanded_local_pos = torch.arange(3, dtype=torch.int32).repeat(3)
    state = owner.req_states
    state.prefill_len.gpu = torch.full((batch.num_reqs,), FULL_REPLAY_HISTORY_TOKENS)
    state.prompt_len = SimpleNamespace(np=batch.prefill_len_np.copy())
    state.last_sampled_tokens = torch.tensor([10, 20, 30])
    state.next_prefill_tokens = torch.full((batch.num_reqs,), -1, dtype=torch.int64)
    state.total_len = SimpleNamespace(
        gpu=torch.full((batch.num_reqs,), FULL_REPLAY_HISTORY_TOKENS + 1)
    )
    history = torch.full((batch.num_reqs, owner.model_state.max_model_len), -1)
    history[:, :FULL_REPLAY_HISTORY_TOKENS] = torch.arange(FULL_REPLAY_HISTORY_TOKENS)
    history[:, FULL_REPLAY_HISTORY_TOKENS] = state.last_sampled_tokens
    state.all_token_ids = SimpleNamespace(gpu=history)
    state.draft_tokens = torch.full((batch.num_reqs, 2), -1, dtype=torch.int64)
    owner.sampler = GreedySamplingState(state, batch.num_reqs)
    owner.rejection_sampler = RejectionSampler(
        owner.sampler, owner.vllm_config.speculative_config, torch.device("cpu")
    )
    owner.model.language_model.compute_logits = SpeculativeLogitsBoundary(
        4, 128, ordinal_rows=eager_fallback or capture_settings is not None
    )
    # Preserve the actual native public getter and its persistent target buffer.
    # DSpark still consumes auxiliary layers; no ordinary-MTP proposal is added.
    owner.model.language_model.get_mtp_target_hidden_states = MethodType(
        AscendDeepseekV41LLMForCausalLM.get_mtp_target_hidden_states,
        owner.model.language_model,
    )
    owner.prompt_logprobs_worker = PromptLogprobsWorker(batch.num_reqs)
    owner.batch_sharder = owner.pcp_manager = None
    owner.main_stream = owner.output_copy_stream = None
    owner.check_ep_fault = owner.use_pp = owner.sync_spec_pp_cpu_counts = False
    owner.num_speculative_steps, owner._draft_workspace_lane = 2, 1
    owner.draft_tokens_handler = CPUDraftTokensHandler()
    owner.model_state.supports_mm_inputs = False
    proposal = proposal_factory(
        monkeypatch,
        vllm_config=owner.vllm_config,
        num_speculative_tokens=2,
        num_reqs=batch.num_reqs,
        fail_backbone=fail_backbone,
    )
    owner.speculator = proposal.speculator
    # These are actual persistent ModelState inputs. Next preparation invokes
    # its unchanged native history selection over the post-rejection state.
    owner.model_state.lookback_token_ids = torch.full(
        (batch.num_reqs + 1, 4), -1, dtype=torch.int32
    )
    owner.model_state._engram_graph_inputs = {
        "engram_query_start_loc": torch.zeros(batch.num_reqs + 1, dtype=torch.int32),
        "engram_valid_token_count": torch.zeros(1, dtype=torch.int32),
    }
    owner.model_state.kvpp_is_dummy_run = False
    owner.model_state.pcp_context = None
    case.initial_history, case.proposal = history.clone(), proposal
    return case


def make_native_dspark_factory_case(
    monkeypatch, proposal_factory, *, fail_backbone=False
):
    """Actual GPU -> ACL -> model factory and persistent output, on CPU only.

    Only base CudaGraphManager.capture is a host notification boundary: it
    executes each fresh warmup/capture callable once and retains inert graph
    handles. Replay never reruns that Python callable. Memory accounting,
    workspace locking, stream/events and the remote FFN are explicit CPU leaves;
    native preparation, ModelWithContext and store_capture_output remain real.
    """
    settings = HcclOverlapSettings(graph_u2_hybrid_dag=False)
    case = make_native_dspark_sampling_case(
        monkeypatch,
        proposal_factory,
        fail_backbone=fail_backbone,
        stages=2,
        capture_settings=settings,
    )
    owner, manager = case.owner, case.manager
    owner.model_state.prompt_embeds_state = None
    owner.maybe_setup_dummy_loras = lambda _: nullcontext()
    manager.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    manager.use_breakable_cg = False
    manager.is_first_pp_rank = manager.is_last_pp_rank = True
    manager.max_num_reqs, manager.dp_size = case.batch.num_reqs, 1
    # DSpark's native NONE capture has no forward factory invocation. Keep its
    # real capture method and manager, rather than inventing a draft graph.
    draft = owner.speculator
    draft_manager = DFlashCudaGraphManager.__new__(DFlashCudaGraphManager)
    vars(draft_manager).update(vars(draft.query_cudagraph_manager))
    draft_manager.cudagraph_mode = CUDAGraphMode.NONE
    draft_manager._capture_descs = {}
    draft_manager.host_descriptors, draft_manager.graphs = (), {}
    draft_manager.capture_failure = None
    draft.query_cudagraph_manager = draft_manager

    runtime = HostStreamRecorder(torch.device("cpu"))
    case.capture_runtime = runtime
    monkeypatch.setattr(
        torch_utils, "_current_stream_tls", SimpleNamespace(value=runtime.parent)
    )
    monkeypatch.setattr(torch.npu, "current_stream", runtime.current_stream)
    monkeypatch.setattr(torch.npu, "stream", runtime.stream)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    native_model = owner.model.language_model.model
    for layer in native_model.layers:
        layer.self_attn = GraphStreamAttention(layer.self_attn, runtime)
    native_model.norm = GraphParentNorm(runtime)

    def observe_draft(model, args):
        runtime.trace.append(
            (
                "draft_parent",
                torch.npu.current_stream() is runtime.parent,
                torch_utils.current_stream() is runtime.parent,
            )
        )

    case.proposal.model.register_forward_pre_hook(observe_draft)

    def allocate_pipeline(bank, *, policy=None, selection=None):
        return attach_pipeline(
            monkeypatch,
            pipeline_module,
            bank,
            policy=policy,
            selection=selection,
            runtime=runtime,
        ).pipeline

    monkeypatch.setattr(
        HcclGraphStageBank,
        "allocate",
        partial(cpu_allocate_bank, case, mask_padding=True, runtime=runtime),
    )
    monkeypatch.setattr(
        pipeline_module.HcclGraphComputePipeline,
        "allocate",
        staticmethod(allocate_pipeline),
    )

    def capture_boundary(
        manager, create_forward_fn, progress_bar_desc="Capturing CUDA graphs"
    ):
        cpu_retained_capture(manager, create_forward_fn, progress_bar_desc)
        if manager is case.manager:
            # The scripted FFN peer now has the matching captured component.
            # This does not replace the real reservation or replay decision.
            case.transport.states[0] = "ready"

    monkeypatch.setattr(CudaGraphManager, "capture", capture_boundary)
    # Match Ascend's pinned patch_v2/patch_input_batch alias. Construct the real
    # Ascend subtype with CPU buffers; do not fabricate the native dummy batch.
    monkeypatch.setattr(native_cudagraph, "InputBatch", AscendInputBatch)
    monkeypatch.setattr(gpu_runner, "freeze_gc_for_cudagraph_capture", nullcontext)
    monkeypatch.setattr(gpu_runner, "lock_workspace", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda: (1024, 2048))
    # The reference is an independent unsplit native CPU model. Capture's
    # actual dummy query is all zero and both accepted stage valid counts are
    # zero; its remote FFN therefore returns zero after the real bank mask.
    dummy = make_ascend_decode_batch(case.batch.num_tokens, target_query_len=3)
    dummy.input_ids.zero_()
    dummy.positions.zero_()
    case.capture_reference = reference_runtime_eager_output(
        dummy, embed_tokens=native_model.embed_tokens, padding_only=True
    )
    return case


def advance_native_dspark_sampling_case(
    case, *, fail_backbone=False, predictions=(41, 15, 16, 13, 42, 43, 6, 7, 44)
):
    """Prepare a fresh scheduled query from the first native sampling result.

    RequestState GPU buffers retain their owner. A real CachedRequestData lets
    native update_requests refresh the same CPU mirrors on the next execute.
    Input-materialization kernels remain a CPU leaf: it derives the new input,
    positions and lengths from actual last/draft/computed state, not an oracle.
    Metadata and lookup leases reopen only when native metadata preparation
    enters run_build/build; this helper does not reset retirement flags.
    """
    owner, previous = case.owner, case.batch
    state = owner.req_states
    if case.kv_transfer.bound_metadata is not None:
        raise AssertionError(
            "finish the first native proposal/KV step before rescheduling"
        )
    width = owner.decode_query_len
    computed = state.num_computed_tokens.gpu[previous.idx_mapping].clone()
    drafts = state.draft_tokens[previous.idx_mapping].clone()
    anchors = state.last_sampled_tokens[previous.idx_mapping].clone()
    batch = make_ascend_decode_batch(previous.num_tokens, target_query_len=width)
    batch.req_ids = previous.req_ids.copy()
    batch.is_dummy = False
    batch.is_padding.fill_(False)
    batch.input_ids.copy_(torch.cat((anchors[:, None], drafts), dim=1).flatten())
    batch.positions.copy_((computed[:, None] + torch.arange(width)).flatten())
    batch.seq_lens.copy_(computed + width)
    batch.seq_lens_cpu_upper_bound.copy_(computed + width)
    batch.seq_lens_np = (computed + width).numpy().astype(np.int32)
    batch.num_computed_tokens_np = computed.numpy().astype(np.int32)
    batch.prefill_len_np = state.prefill_len.np[batch.idx_mapping_np].copy()
    batch.num_computed_prefill_tokens_np = np.minimum(
        batch.num_computed_tokens_np, batch.prefill_len_np
    )
    batch.logits_indices = torch.arange(batch.num_tokens)
    batch.cu_num_logits = batch.query_start_loc[: batch.num_reqs + 1].clone()
    batch.cu_num_logits_np = batch.query_start_loc_np[: batch.num_reqs + 1].copy()
    batch.num_draft_tokens_per_req = np.full(batch.num_reqs, width - 1, dtype=np.int32)
    batch.num_draft_tokens = int(batch.num_draft_tokens_per_req.sum())
    batch.expanded_idx_mapping = torch.repeat_interleave(batch.idx_mapping, width)
    batch.expanded_local_pos = torch.arange(width, dtype=torch.int32).repeat(
        batch.num_reqs
    )
    scheduler = SchedulerOutput.make_empty()
    scheduler.num_scheduled_tokens = dict.fromkeys(batch.req_ids, width)
    scheduler.total_num_scheduled_tokens = batch.num_tokens
    scheduler.scheduled_spec_decode_tokens = dict(
        zip(batch.req_ids, drafts.tolist(), strict=True)
    )
    scheduler.scheduled_cached_reqs = CachedRequestData(
        req_ids=batch.req_ids.copy(),
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None] * batch.num_reqs,
        num_computed_tokens=computed.tolist(),
        num_output_tokens=(
            state.total_len.gpu[batch.idx_mapping]
            - state.prefill_len.gpu[batch.idx_mapping]
        ).tolist(),
    )
    scheduler.kv_connector_metadata = TransferMetadata()
    scheduler.has_sync_kv_loads = True
    owner.prepare_inputs = PreparedReplayInputs(
        batch,
        case.native_events,
        stages=owner.parallel_config.num_ubatches,
        allow_eager=case.eager_fallback,
    )
    owner.model.language_model.compute_logits.predictions.copy_(
        torch.tensor(predictions)
    )
    if fail_backbone:
        case.proposal.model.error = RuntimeError(
            "controlled second DSpark backbone failure"
        )
    expected_history = case.initial_history.clone()
    expected_history[0, 13:17] = torch.tensor([11, 12, 13, 41])
    expected_history[1, 13:16] = torch.tensor([29, 13, 42])
    expected_history[2, 13:18] = torch.tensor([31, 39, 6, 7, 44])
    case.expected_second_history = expected_history
    case.batch, case.scheduler = batch, scheduler
    case.kv_transfer.batch = batch
    return batch


def native_dspark_eager_reference(case):
    """Independent unsplit CPU body, sharing only frozen read-only weights."""
    return reference_runtime_eager_output(
        case.batch, embed_tokens=case.owner.model.language_model.model.embed_tokens
    )
