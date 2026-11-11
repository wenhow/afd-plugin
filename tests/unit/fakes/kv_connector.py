# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native KV lifecycle with CPU storage and explicit device boundaries.

The AFD, Ascend and GPU execute methods, ActiveKVConnector and EngramModelState
are real. Only transfer registration/load and auxiliary device retirement are
host boundaries. FULL replay additionally supplies prepared inputs, metadata
and an inert graph device boundary. This does not run PD or Graph device work.
"""

from contextlib import contextmanager, nullcontext
from functools import partial
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorTransferResults,
)
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
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.spec_decode import rejection_sampler as rejection_module
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAImpl
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.models.deepseek_v41.model import AscendDeepseekV41LLMForCausalLM
from vllm_ascend.ops.triton import engram_lookback
from vllm_ascend.worker.v2 import model_runner as native_runner
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers

from afd_plugin.connectors.npu import hccl_graph_pipeline as pipeline_module
from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank
from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.distributed.hccl_plan import HcclLocalForward
from afd_plugin.model_executor.models.npu.deepseek_v41_graph import (
    DeepseekV41GraphParent,
    DeepseekV41GraphStages,
)
from afd_plugin.v1.worker.npu import hccl_v41_graph_coordinator as coordinator_module
from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
    AFDNPUAttentionModelRunnerV2,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph_coordinator import HcclV41GraphResources
from afd_plugin.v1.worker.npu.hccl_v41_graph_eager import HcclV41GraphEager
from afd_plugin.v1.worker.npu.hccl_v41_graph_execution import HcclV41GraphExecution
from tests.unit.fakes.attention_model_runner_v2 import (
    install_graph_runner_coordinator,
    make_graph_runner_case,
)
from tests.unit.fakes.deepseek_v41_graph import GraphParentNorm, GraphStreamAttention
from tests.unit.fakes.device_metadata import make_cpu_target_metadata
from tests.unit.fakes.graph_cache import DeviceCompletion, ReplayGraph
from tests.unit.fakes.hccl_graph_pipeline import attach_pipeline
from tests.unit.fakes.hccl_v41_graph_coordinator import (
    cpu_allocate_bank,
    cpu_retained_capture,
    install_runtime_eager_tensor_boundary,
    reference_runtime_eager_output,
)
from tests.unit.fakes.input_batch import (
    make_ascend_decode_batch,
    snapshot_ascend_child,
)
from tests.unit.fakes.npu_stream import HostStreamRecorder
from tests.unit.fakes.platform import stub_platform_forward_context
from tests.unit.fakes.rejection_sampler import (
    CPULookbackGather,
    GreedySamplingState,
    greedy_rejection_cpu,
    sampled_and_rejected_cpu,
    update_speculative_history_cpu,
)

FULL_REPLAY_ROWS = 4
FULL_REPLAY_HISTORY_TOKENS = 12


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
        if post_kv or fail_load or stages != 1:
            raise ValueError("external PD zero-work case requires eager/U1 polling")
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


def _publish_cpu_replay_fixture(case, coordinator):
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
    manager.hidden_states = torch.arange(
        descriptor.num_tokens * bank.capabilities.hidden_size, dtype=torch.float32
    ).reshape(descriptor.num_tokens, bank.capabilities.hidden_size)
    manager.aux_hidden_states = (
        [manager.hidden_states + 2, manager.hidden_states + 6]
        if owner.use_aux_hidden_state_outputs
        else []
    )
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

    def __init__(self, hidden_size, vocab_size, *, ordinal_rows=False):
        self.hidden_size, self.vocab_size = hidden_size, vocab_size
        self.ordinal_rows = ordinal_rows
        # Request 0: both drafts match; request 1: first draft mismatches;
        # request 2: first matches and second mismatches. Final rows are bonus.
        self.predictions = torch.tensor([11, 12, 13, 29, 22, 23, 31, 39, 33])

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
