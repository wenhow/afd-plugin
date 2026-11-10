# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native zero-work KV lifecycle with CPU storage and transfer boundaries.

The AFD, Ascend and GPU execute methods, ActiveKVConnector and EngramModelState
are real. Only transfer registration/load and auxiliary device retirement are
host boundaries. This fixture does not run PD, Graph or Engram device work.
"""

from types import SimpleNamespace

import numpy as np
import torch
from vllm.distributed.kv_transfer import kv_transfer_state
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorTransferResults,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.ec_connector import ECConnector
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.worker.v2 import model_runner as native_runner

from afd_plugin.v1.worker.npu import hccl_v41_graph_coordinator as coordinator_module
from afd_plugin.v1.worker.npu.hccl_v41_graph_eager import HcclV41GraphEager
from tests.unit.fakes.attention_model_runner_v2 import (
    install_graph_runner_coordinator,
    make_graph_runner_case,
)
from tests.unit.fakes.platform import stub_platform_forward_context


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
