# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native KV lifecycle with CPU storage and explicit device boundaries.

The AFD, Ascend and GPU execute methods, ActiveKVConnector and EngramModelState
are real. Only transfer registration/load and auxiliary device retirement are
host boundaries. FULL replay additionally supplies prepared inputs, metadata
and an inert graph device boundary. This does not run PD or Graph device work.
"""

from types import MethodType, SimpleNamespace

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorTransferResults,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.async_utils import StepTimingCollector
from vllm.v1.worker.gpu.ec_connector import ECConnector
from vllm.v1.worker.gpu.eplb_utils import EPLBController
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
from vllm_ascend.attention.sfa_v1 import AscendSFABackend, AscendSFAImpl
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.worker.v2 import model_runner as native_runner

from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank
from afd_plugin.distributed.hccl_plan import HcclLocalForward
from afd_plugin.model_executor.models.npu.deepseek_v41_graph import (
    DeepseekV41GraphParent,
    DeepseekV41GraphStages,
)
from afd_plugin.v1.worker.npu import hccl_v41_graph_coordinator as coordinator_module
from afd_plugin.v1.worker.npu.hccl_v41_graph_coordinator import HcclV41GraphResources
from afd_plugin.v1.worker.npu.hccl_v41_graph_eager import HcclV41GraphEager
from afd_plugin.v1.worker.npu.hccl_v41_graph_execution import HcclV41GraphExecution
from tests.unit.fakes.attention_model_runner_v2 import (
    install_graph_runner_coordinator,
    make_graph_runner_case,
)
from tests.unit.fakes.graph_cache import ReplayGraph
from tests.unit.fakes.input_batch import make_ascend_decode_batch
from tests.unit.fakes.platform import stub_platform_forward_context

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


def make_native_kv_runner_case(monkeypatch, *, post_kv, sync_load, fail_load=False):
    """Keep the complete zero-token native execute/no_forward control flow.

    Config admission remains unchanged: an explicit CPU component owns the KV
    singleton below. It is not a serving Graph/PD configuration or handshake.
    All singleton/DP/backend substitutions are restored by pytest monkeypatch.
    """
    case = make_graph_runner_case()
    owner = case.owner
    events = []
    device = AuxiliaryDeviceBoundary(events)
    transfer = KVTransferBoundary(events, fail_load=fail_load)
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
    owner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            profiling_chunk_config=SimpleNamespace(need_timing=False)
        )
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
    views = {"layer": torch.arange(16, dtype=torch.int32).reshape(2, 2, 4)}
    owner.kv_connector = ActiveKVConnector(owner.vllm_config, views)
    owner.kv_caches = list(views.values())
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

    def __init__(self, events):
        super().__init__(events)
        self.attn_metadata = None
        self.replay_live = False

    def run_build(self, build_metadata, **kwargs):
        # Keep native EngramModelState.prepare_attn, including its real FULL
        # mode and buffer ownership. Do not invoke a device metadata kernel.
        self.attn_metadata = {
            "child": SimpleNamespace(
                padded_tokens=kwargs["num_tokens"],
                valid_tokens=kwargs["num_actual_tokens"],
                query_start_loc=kwargs["query_start_loc_gpu"],
                block_tables=kwargs["block_tables"],
                slot_mappings=kwargs["slot_mappings"],
            )
        }
        self.replay_live = True
        self.events.append("attn-ready")
        return self.attn_metadata

    def finish_replay(self):
        if not self.replay_live:
            raise AssertionError("replay metadata must have been prepared")
        self.replay_live = False
        self.events.append("metadata-replay-finish")


class ReplayKVTransferBoundary(KVTransferBoundary):
    """Validate the real positive native pre_forward metadata and coordinates."""

    def __init__(self, events, batch, metadata):
        super().__init__(events, fail_load=False)
        self.batch, self.metadata = batch, metadata

    def start_load_kv(self, context, **kwargs):
        if (
            context.attn_metadata is not None
            or kwargs["attn_metadata"] is not self.metadata.attn_metadata
            or kwargs["request_ids"] != self.batch.req_ids
            or kwargs["request_state_indices"] is not self.batch.idx_mapping
            or kwargs["num_tokens"] != self.batch.num_tokens
        ):
            raise AssertionError("positive KV load lost native prepared coordinates")
        # Native FULL pre_forward precedes run_fullgraph's context. It passes
        # the real metadata explicitly; ActiveKVConnector creates an empty
        # maintenance context rather than pretending replay has already begun.
        self.events.append("kv-load")


class PreparedReplayInputs:
    """Replace only input-materialization kernels, preserving native gather."""

    def __init__(self, batch, events):
        self.batch, self.events = batch, events

    def __call__(self, scheduler, state, descriptor):
        if (
            state.req_ids != self.batch.req_ids
            or tuple(state.num_scheduled_tokens)
            != tuple(self.batch.num_scheduled_tokens)
            or state.num_tokens != self.batch.num_tokens
            or descriptor.cg_mode != CUDAGraphMode.FULL
            or descriptor.num_tokens != self.batch.num_tokens_after_padding
            or descriptor.num_ubatches != 1
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
            (slice(0, descriptor.num_tokens),),
            bank,
        )
    )
    graph = ReplayGraph()
    manager.graphs[descriptor] = graph
    manager.hidden_states = torch.arange(
        FULL_REPLAY_ROWS * bank.capabilities.hidden_size, dtype=torch.float32
    ).reshape(FULL_REPLAY_ROWS, bank.capabilities.hidden_size)
    manager.aux_hidden_states = []
    resources = HcclV41GraphResources(
        bank=bank,
        execution=execution,
        model_state=owner.model_state,
        input_buffers=owner.input_buffers,
        ubatch_runner=None,
        attn_groups=tuple(tuple(group) for group in owner.attn_groups),
        block_tables=owner.block_tables,
        kv_cache_config=owner.kv_cache_config,
        kv_caches=tuple(owner.kv_caches),
        manager=manager,
        hidden_states=manager.hidden_states,
        auxiliary=(),
        intermediate_tensors=manager.intermediate_tensors,
    )
    case.cache.reserve_capture(key, reservation.generation, graph, resources)
    case.transport.submit_forward(
        HcclLocalForward("warmup", (0,), key.padded_tokens),
        transaction_id=reservation.decision.transaction_id,
    )
    case.transport.complete_graph_forward(key)
    case.cache.publish_capture(key, reservation.generation)
    case.transport.states[0] = "ready"
    case.recorded_graph, case.recorded_resources = graph, resources
    case.replay_baseline = tuple(case.transport.transcript)


def make_native_full_kv_replay_case(monkeypatch, *, fail_replay=False):
    """Real AFD -> Ascend -> GPU FULL execution over explicit CPU inputs.

    prepare_inputs and _graph_relay are the two instance method boundaries.
    Native execute, gather, prepare_attn, run_fullgraph and Engram retirement
    are unchanged. The static cache is a supplied fixture, not a capture result.
    """
    case = make_native_kv_runner_case(monkeypatch, post_kv=False, sync_load=True)
    owner = case.owner
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda _: None
    )
    # Select the real non-offload SFA implementation without global device
    # configuration. Native graph-backend discovery and replay stay intact.
    monkeypatch.setattr(
        AscendSFABackend, "get_impl_cls", staticmethod(lambda: AscendSFAImpl)
    )
    batch = make_ascend_decode_batch(FULL_REPLAY_ROWS)
    batch.req_ids = [f"decode-{index}" for index in range(FULL_REPLAY_ROWS)]
    batch.is_dummy = False
    batch.is_padding.fill_(False)
    batch.num_computed_tokens_np.fill(FULL_REPLAY_HISTORY_TOKENS)
    batch.prefill_len_np.fill(FULL_REPLAY_HISTORY_TOKENS)
    batch.num_computed_prefill_tokens_np.fill(FULL_REPLAY_HISTORY_TOKENS)
    batch.seq_lens.fill_(FULL_REPLAY_HISTORY_TOKENS + 1)
    batch.seq_lens_np.fill(FULL_REPLAY_HISTORY_TOKENS + 1)
    case.batch = batch
    owner.req_states = SimpleNamespace(
        req_id_to_index=dict(zip(batch.req_ids, range(FULL_REPLAY_ROWS), strict=True)),
        num_computed_tokens_np=batch.num_computed_tokens_np,
        prefill_len=SimpleNamespace(np=batch.prefill_len_np),
        num_computed_prefill_tokens=batch.num_computed_prefill_tokens_np,
        num_computed_tokens=SimpleNamespace(
            gpu=torch.full((FULL_REPLAY_ROWS,), FULL_REPLAY_HISTORY_TOKENS)
        ),
    )
    # Remove the lower-level builder's gather lambda: execute must select and
    # preserve the actual native request ordering before the prepared input seam.
    del owner.gather_batch_req_state
    owner.prepare_inputs = PreparedReplayInputs(batch, case.native_events)
    owner.block_tables = ReplayBlockTables(FULL_REPLAY_ROWS)
    owner.input_buffers = SimpleNamespace(dcp_local_seq_lens=None)
    owner.model_config = owner.vllm_config.model_config
    owner.observability_config = SimpleNamespace(cudagraph_metrics=False)
    owner.lora_config = owner.adaptive_verification = None
    owner.is_encoder_decoder = owner.uses_inputs_embeds = False
    owner.is_first_pp_rank = owner.is_last_pp_rank = True
    owner.use_aux_hidden_state_outputs = False
    owner.routed_experts_capturer = owner.speculator = None
    owner.prompt_logprobs_worker = None
    owner.use_spec_pp = False
    owner.dp_size, owner.dp_rank = 1, 0
    owner.dcp_size, owner.dcp_rank, owner.cp_interleave = 1, 0, 1
    owner.parallel_config.enable_eplb = False
    owner.eplb = EPLBController(owner.parallel_config, torch.device("cpu"))
    owner.step_timing = StepTimingCollector()
    owner.attn_groups = [[SimpleNamespace(backend=AscendSFABackend)]]
    metadata = ReplayDeviceMetadata(case.native_events)
    owner.model_state.device_metadata = metadata
    owner.model_state.vllm_config = owner.vllm_config
    owner.model_state.max_model_len = FULL_REPLAY_HISTORY_TOKENS + FULL_REPLAY_ROWS
    owner.model_state.rope_state = owner.model_state.lookback_token_ids = None
    owner.model.retire_engram_lookups = metadata.retire_engram_lookups
    case.auxiliary_device = metadata
    transfer = ReplayKVTransferBoundary(case.native_events, batch, metadata)
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", transfer)
    owner.kv_connector = ActiveKVConnector(owner.vllm_config, case.kv_views)
    case.kv_transfer = transfer
    case.scheduler.num_scheduled_tokens = dict.fromkeys(batch.req_ids, 1)
    case.scheduler.total_num_scheduled_tokens = FULL_REPLAY_ROWS
    case.transport.data_groups = [
        endpoint.group for endpoint in case.model_case.endpoints
    ]
    coordinator = install_graph_runner_coordinator(case)
    case.manager = coordinator.manager
    case.manager.vllm_config = owner.vllm_config
    case.manager.update_stream = SimpleNamespace(label="device-replay-boundary")
    owner.bind_hccl_v41_graph_coordinator(coordinator)
    _publish_cpu_replay_fixture(case, coordinator)
    case.replay_error = (
        RuntimeError("controlled FULL replay failure") if fail_replay else None
    )

    def replay_device(manager, backend, descriptor, num_tokens, attn_metadata):
        if (
            attn_metadata is not metadata.attn_metadata
            or num_tokens != batch.num_tokens
        ):
            raise AssertionError("native replay received another batch's metadata")
        case.native_events.append("replay-device-boundary")
        if case.replay_error is not None:
            raise case.replay_error
        manager.graphs[descriptor].replay()
        return manager.hidden_states

    case.manager._graph_relay = MethodType(replay_device, case.manager)
    return case
