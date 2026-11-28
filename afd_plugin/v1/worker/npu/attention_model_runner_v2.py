# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU Attention-side model runner for AFD ModelRunnerV2 execution."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from dataclasses import replace
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING

import torch
from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.distributed import get_tp_group
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.dp_utils import skip_dp_coordination
from vllm.v1.worker.gpu import cudagraph_utils as v2_cudagraph_utils
from vllm.v1.worker.gpu import model_runner as v2_model_runner
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.model_runner import BatchReqState
from vllm_ascend.models.deepseek_v41.dspark import DeepseekV41DSparkAttention
from vllm_ascend.models.deepseek_v41.mixed_quant_attention import (
    MixedQuantPackedCacheOps,
)
from vllm_ascend.worker.v2 import model_runner as ascend_model_runner
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2

from afd_plugin.compat.npu import fail_if_unsupported_npu_afd_features
from afd_plugin.compat.npu.dspark_pd import (
    DsparkPdPrefixConsumer,
    DsparkPdSourceRequest,
    DsparkPdWriteReport,
)
from afd_plugin.compat.npu.profiler import (
    AFDNPUProfilerSession,
    afd_npu_profiler_execute,
    create_deferred_afd_npu_profiler,
    stop_afd_npu_profiler,
)
from afd_plugin.compat.npu.runtime_config import (
    configure_async_cam_dbo,
    is_deepseek_v41_model,
    npu_afd_num_ubatches,
    uses_deepseek_v41_dspark_pd,
)
from afd_plugin.compat.patches.npu.mrv2_ubatching import (
    apply_ep_pd_dp_coordination_patch,
    use_async_cam_dbo_dispatch,
    use_npu_ubatch_factory,
)
from afd_plugin.config import (
    HCCL_P2P_CONNECTOR,
    WINDOW_CONNECTOR,
    AFDConfig,
    parse_afd_config,
    use_afd_dspark,
)
from afd_plugin.connectors import (
    AFDConnectorBase,
    AFDForwardContextMetadata,
)
from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.connectors.npu.hccl_protocol import HcclForwardPlan
from afd_plugin.connectors.npu.hccl_staged import (
    HcclStagedAFDConnector,
    create_npu_runtime_connector,
)
from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector
from afd_plugin.model_executor.models.forward_context import use_afd_metadata_provider
from afd_plugin.model_executor.models.npu.dsv41_packed_cache import (
    AFDMixedQuantPackedCacheOps,
)
from afd_plugin.v1.worker.attention_metadata import (
    AFDMetadataProviderMixin,
    _resolve_world_ranks,
)
from afd_plugin.v1.worker.attention_model_runner_v2 import (
    _AFDCaptureEventTracker,
    _use_afd_capture_input_preparation,
    _use_afd_ubatch_preparation,
)
from afd_plugin.v1.worker.npu.hccl_v41_diagnostics import HcclV41DiagnosticBinding
from afd_plugin.v1.worker.npu.hccl_v41_eager import HcclV41EagerSession
from afd_plugin.v1.worker.npu.hccl_v41_eager_bootstrap import HcclV41EagerBootstrap
from afd_plugin.v1.worker.npu.hccl_v41_execution import (
    HcclV41Execution,
    validate_hccl_v41_binding,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph_bootstrap import (
    HcclV41AttentionGraphBootstrap,
)
from afd_plugin.v1.worker.npu.hccl_v41_ubatch_runner import HcclV41UBatchRunner
from afd_plugin.v1.worker.npu.ubatch_runner import NPUUBatchRunner
from afd_plugin.validation import validate_npu_model_runner_v2_config

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager

    from afd_plugin.compat.npu.kv_layout_connector import MooncakeHybridConnector
    from afd_plugin.v1.worker.npu.hccl_v41_graph_coordinator import (
        HcclV41GraphCoordinator,
    )
    from afd_plugin.v1.worker.npu.hccl_v41_graph_eager import HcclV41GraphEager
    from afd_plugin.v1.worker.npu.hccl_v41_graph_session import HcclV41GraphSession

logger = init_logger(__name__)


def window_dspark_capture_descs(
    descs: list[BatchExecutionDescriptor],
    width: int,
    max_reqs: int,
    num_ubatches: int,
) -> list[BatchExecutionDescriptor]:
    """Align whole verification rows to the target's chosen stage count."""
    if width < 1 or num_ubatches < 1:
        raise ValueError("Verification width and target stage count must be positive")
    buckets = set()
    for desc in descs:
        rows = min(desc.num_tokens // width, max_reqs)
        rows -= rows % num_ubatches
        if rows:
            buckets.add(
                replace(
                    desc,
                    num_reqs=rows,
                    num_tokens=rows * width,
                    uniform_token_count=width,
                    max_query_len=None,
                    num_ubatches=num_ubatches,
                )
            )
    if not buckets:
        raise ValueError(
            f"Window DSpark needs max_num_seqs >= {num_ubatches} and a target "
            f"graph bucket of at least {num_ubatches * width} tokens"
        )
    return sorted(buckets, key=lambda desc: desc.num_tokens, reverse=True)


class _WindowDSparkTargetGraphManager(ascend_model_runner.ModelAclGraphManager):
    def _init_candidates(self) -> None:
        super()._init_candidates()
        if self.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
            return
        descs = window_dspark_capture_descs(
            self._capture_descs.get(CUDAGraphMode.FULL, []),
            self.decode_query_len,
            self.max_num_reqs,
            npu_afd_num_ubatches(self.vllm_config),
        )
        self._capture_descs[CUDAGraphMode.FULL] = descs
        self._candidates.clear()
        next_token_count: dict[int, int] = {}
        for desc in reversed(descs):
            loras = desc.num_active_loras
            for tokens in range(next_token_count.get(loras, 1), desc.num_tokens + 1):
                self._candidates[tokens, loras] = [desc]
            next_token_count[loras] = desc.num_tokens + 1


@contextmanager
def _use_window_dspark_target_graph_manager() -> Iterator[None]:
    original = ascend_model_runner.ModelAclGraphManager
    try:
        ascend_model_runner.ModelAclGraphManager = _WindowDSparkTargetGraphManager
        yield
    finally:
        ascend_model_runner.ModelAclGraphManager = original


def install_window_dspark_replay(manager: DFlashAclGraphManager) -> None:
    """Bind draft metadata replay only on the current worker's draft manager."""
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm_ascend.compilation.acl_graph import update_full_graph_params
    from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager

    def _graph_replay(self, desc, attn_backend, num_tokens, draft_attn_metadatas):
        self.update_stream.wait_stream(torch.npu.current_stream())
        result = super(DFlashAclGraphManager, self).run_fullgraph(desc)
        num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens)
        with set_forward_context(
            draft_attn_metadatas[0],
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=desc.cg_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            batch_descriptor=None,
            slot_mapping=None,
        ):
            _EXTRA_CTX.is_draft_model = True
            _EXTRA_CTX.is_draft_model_prefill = False
            update_full_graph_params(
                attn_backend,
                self.update_stream,
                get_forward_context(),
                num_tokens,
                self.vllm_config,
                self.speculator.speculative_config,
                draft_attn_metadatas=draft_attn_metadatas,
            )
        return result

    manager._graph_replay = MethodType(_graph_replay, manager)


def _normalize_window_u2_capture_descs(
    capture_descs: list[v2_cudagraph_utils.BatchExecutionDescriptor],
) -> list[v2_cudagraph_utils.BatchExecutionDescriptor]:
    """Keep Window graph capture structurally U2, including one-token decode."""
    normalized: list[v2_cudagraph_utils.BatchExecutionDescriptor] = []
    for desc in capture_descs:
        fixed = replace(
            desc,
            num_tokens=max(int(desc.num_tokens), 2),
            num_reqs=(
                max(int(desc.num_reqs or 1), 2)
                if int(desc.num_tokens) < 2
                else desc.num_reqs
            ),
            num_ubatches=2,
        )
        if fixed not in normalized:
            normalized.append(fixed)
    return normalized


_AFD_FULLGRAPH_HOOK_MARKER = "_afd_fullgraph_replay_hook_active"


@contextmanager
def _use_afd_fullgraph_replay_hook(
    runner: AFDNPUAttentionModelRunnerV2,
    real_tokens: int,
    *,
    verifying_drafts: bool = False,
) -> Iterator[None]:
    """Scope AFD full ACL-graph replay control to one graph manager instance."""

    manager = runner.cudagraph_manager
    if manager is None:
        raise RuntimeError(
            "AFD ACL graph replay hook requires an initialized graph manager",
        )
    manager_state = manager.__dict__
    if _AFD_FULLGRAPH_HOOK_MARKER in manager_state:
        raise RuntimeError("AFD ACL graph replay hook is already active")

    had_instance_override = "run_fullgraph" in manager_state
    previous_instance_override = manager_state.get("run_fullgraph")
    original_run_fullgraph = manager.run_fullgraph
    manager_state[_AFD_FULLGRAPH_HOOK_MARKER] = True

    # Patch reason: native FULL replay bypasses ForwardContext creation, so the
    # execute-scoped AFD provider cannot publish runtime control.
    # Patch functionality: wrap only this manager instance and publish one
    # ordinary padded control payload immediately before each native replay.
    # Signature: matches vLLM v0.30.0 ModelCudaGraphManager.run_fullgraph exactly.
    # Upstream source: vllm/v1/worker/gpu/cudagraph_utils.py,
    # ModelCudaGraphManager.run_fullgraph; commit
    # ced6857afa0ea7b2e3f0846a62e1394e90f15607.
    # Delegation exception: native replay remains wholly in the saved bound
    # method; this scope owns only the AFD pre-replay control seam.
    # Removal/upstream plan: delete this hook when vLLM exposes a per-manager
    # callback immediately before FULL graph replay.
    def run_fullgraph(
        self: v2_cudagraph_utils.ModelCudaGraphManager,
        desc: v2_cudagraph_utils.BatchExecutionDescriptor,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]] | IntermediateTensors:
        # ### PATCH START: publish one AFD pre-replay payload.
        if desc.num_ubatches > 1:
            # Native execute already prepared live stages and published their
            # complete A/F transaction. Update their captured MLA records at
            # this graph-manager seam without publishing duplicate control.
            assert runner.ubatch_runner is not None
            result = runner.ubatch_runner.run_fullgraph(self, desc)
            if verifying_drafts:
                logger.info_once(
                    "Window DSpark target replayed U2 FULL verification graph "
                    "(requests=%d, tokens=%d, query_width=%d)",
                    desc.num_reqs,
                    desc.num_tokens,
                    desc.uniform_token_count,
                )
            return result
        previous_is_graph_replaying = getattr(
            runner,
            "_afd_is_graph_replaying",
            False,
        )
        try:
            padded_tokens = int(desc.num_tokens)
            metadata = runner.build_afd_metadata(None, real_tokens)
            metadata.tokens_lens = [padded_tokens]
            runner._afd_pending_metadata = metadata
            runner._afd_suppress_metadata_send = True
            runner._is_warmup = False
            runner._afd_is_graph_capturing = False
            runner._afd_is_graph_replaying = True
            if runner.connector.control_plane is not None:
                runner.send_dp_metadata(
                    runner.build_capture_dp_metadata(padded_tokens),
                    None,
                )
            result = original_run_fullgraph(desc)
        finally:
            runner._afd_is_graph_replaying = previous_is_graph_replaying
        # ### PATCH END: publish one AFD pre-replay payload.
        return result

    try:
        manager.run_fullgraph = MethodType(run_fullgraph, manager)
        yield
    finally:
        if had_instance_override:
            manager.run_fullgraph = previous_instance_override
        else:
            del manager.run_fullgraph
        del manager_state[_AFD_FULLGRAPH_HOOK_MARKER]


@contextmanager
def _use_camp2p_dp_padding(attention_size: int, ffn_size: int) -> Iterator[None]:
    """Pad each CAMP2P receiver group equally before input preparation (FFN TP=1)."""
    original_dispatch = v2_model_runner.dispatch_cg_and_sync_dp

    # Patch reason: native execute calls this module-level function before input
    # preparation; no runner method exposes that boundary.
    # Patch functionality: apply CAMP2P group padding to the agreed eager batch.
    # Signature: matches vLLM v0.30.0 dispatch_cg_and_sync_dp exactly.
    # Upstream source: vllm/v1/worker/gpu/dp_utils.py;
    # ced6857afa0ea7b2e3f0846a62e1394e90f15607.
    # Delegation exception: preserve native graph selection and DP collectives.
    # Removal plan: replace when native dispatch exposes a padding callback.
    def dispatch_cg_and_sync_dp(
        cudagraph_manager: v2_cudagraph_utils.CudaGraphManager | None,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        dp_size: int,
        dp_rank: int,
        max_query_len: int | None = None,
        need_eager: bool = False,
        num_active_loras: int = 0,
        parallel_config: ParallelConfig | None = None,
        allow_ubatching: bool = False,
        uniform_decode: bool = False,
        dp_sync: DPSyncState | None = None,
    ) -> tuple[v2_cudagraph_utils.BatchExecutionDescriptor, DPSyncState | None]:
        desc, sync = original_dispatch(
            cudagraph_manager,
            num_reqs,
            num_tokens,
            uniform_token_count,
            dp_size,
            dp_rank,
            max_query_len=max_query_len,
            need_eager=need_eager,
            num_active_loras=num_active_loras,
            parallel_config=parallel_config,
            allow_ubatching=allow_ubatching,
            uniform_decode=uniform_decode,
            dp_sync=dp_sync,
        )
        # ### PATCH START: equal CAMP2P strides for eager and graph fallback.
        if desc.cg_mode == CUDAGraphMode.NONE and sync is not None:
            attention_tp = attention_size // dp_size
            # Rank a sends to FFN a % ffn_size. Contiguous Attention TP peers
            # must also pad equally; non-aligned TP blocks connect all groups.
            num_groups = ffn_size // attention_tp if ffn_size % attention_tp == 0 else 1
            padded_counts = (
                sync.num_tokens_across_dp.reshape(-1, num_groups)
                .amax(dim=0)
                .repeat(dp_size // num_groups)
            )
            desc = replace(desc, num_tokens=int(padded_counts[dp_rank].item()))
            sync = replace(sync, num_tokens_across_dp=padded_counts)
        # ### PATCH END: equal CAMP2P strides for eager and graph fallback.
        return desc, sync

    try:
        v2_model_runner.dispatch_cg_and_sync_dp = dispatch_cg_and_sync_dp
        yield
    finally:
        v2_model_runner.dispatch_cg_and_sync_dp = original_dispatch


class AFDNPUAttentionModelRunnerV2(AFDMetadataProviderMixin, NPUModelRunnerV2):
    """Thin AFD seam over native vLLM-Ascend ModelRunnerV2.

    Native V2 retains request state, input preparation, Attention/KV handling,
    sampling, and output ownership. The inherited AFD metadata methods are
    pure connector/context plumbing and are reused without porting V1's
    execution, dummy, or graph lifecycle methods.
    """

    afd_expected_role = "attention"
    ubatch_runner: NPUUBatchRunner | None
    hccl_v41_transport: HcclStagedTransport | None = None
    hccl_v41_graph_eager: HcclV41GraphEager | None = None
    hccl_v41_graph_session: HcclV41GraphSession | None = None
    hccl_v41_graph_coordinator: HcclV41GraphCoordinator | None = None
    hccl_v41_graph_bootstrap: HcclV41AttentionGraphBootstrap | None = None
    hccl_v41_eager_bootstrap: HcclV41EagerBootstrap | None = None
    hccl_v41_diagnostics: HcclV41DiagnosticBinding | None = None
    dspark_pd_consumer: DsparkPdPrefixConsumer | None = None
    _afd_shutdown_prepared = False
    _afd_staged_stop_completed = False
    _hccl_graph_load_started = False
    _hccl_eager_load_started = False
    _hccl_configuration_logged = False
    _dspark_pd_load_started = False
    _dspark_pd_pending_scheduler: SchedulerOutput | None

    # Upstream: NPUModelRunner.__init__, vllm_ascend/worker/v2/model_runner.py
    # at 7e37e29a7803d6897a7113088e2b771f83205bc0.
    # Reason/functionality: freeze DSV4.1 HCCL settings before connector
    # construction and retain its configuration-record lifetime with the runner.
    # Signature matches upstream, including no return annotation; no added args.
    # Delegation exception: the large native constructor owns model/state and
    # device setup. Retain super() and compose the AFD connector around it.
    # Remove when native runner hooks own connector settings/startup records.
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        fail_if_unsupported_npu_afd_features(vllm_config)
        configure_async_cam_dbo(vllm_config)
        validate_npu_model_runner_v2_config(
            vllm_config,
            expected_role="attention",
            device_type=device.type,
        )
        fail_if_unsupported_npu_afd_features(vllm_config)
        super().__init__(vllm_config, device)
        connector: AFDConnectorBase | None = None
        try:
            self.afd_config = self.parse_config(self.vllm_config)
            # ### PATCH START: preserve native DP agreement for EP PD
            if (
                self.afd_config.connector == HCCL_P2P_CONNECTOR
                and vllm_config.parallel_config.enable_expert_parallel
                and vllm_config.kv_transfer_config is not None
            ):
                apply_ep_pd_dp_coordination_patch()
            # ### PATCH END: preserve native DP agreement for EP PD
            rank, _ = _resolve_world_ranks()
            # Freeze role/rank settings now; start/export only on execute's
            # owner thread. Constructor/load traces are intentionally excluded.
            self.prof = create_deferred_afd_npu_profiler("attention", role_rank=rank)
            if (
                self.prof is not None
                and vllm_config.profiler_config is not None
                and vllm_config.profiler_config.profiler is not None
            ):
                raise ValueError(
                    "AFD_NPU_ATTENTION_PROFILER_ENABLE and native --profiler-config "
                    "cannot be enabled together"
                )
            local_rank = int(device.index)
            # ### PATCH START: freeze DSV4.1 HCCL startup settings
            overlap_settings = (
                HcclOverlapSettings.from_environment(
                    dict(os.environ),
                    num_stages=2 if self.parallel_config.use_ubatching else 1,
                )
                if is_deepseek_v41_model(self.vllm_config)
                and self.afd_config.connector == HCCL_P2P_CONNECTOR
                else None
            )
            self._hccl_configuration_logged = False
            # ### PATCH END: freeze DSV4.1 HCCL startup settings
            connector = create_npu_runtime_connector(
                rank,
                local_rank,
                self.vllm_config,
                self.afd_config,
                overlap_settings=overlap_settings,
            )
            self.connector = connector
            # The connector rendezvous is deferred to the end of ``load_model()``
            # so Attention and FFN weight loading overlap, matching the V1
            # Attention runner lifecycle.
            if (
                connector.control_plane is None
                and not self.afd_config.async_dp
                and not isinstance(
                    connector, (P2pHcclAFDConnector, HcclStagedAFDConnector)
                )
            ):
                raise RuntimeError(
                    "AFD ModelRunnerV2 requires a control-plane-driven connector",
                )
            self._is_warmup = False
            self._afd_is_graph_capturing = False
            self._afd_is_profile = False
            self._afd_is_graph_replaying = False
            self._afd_pending_metadata: AFDForwardContextMetadata | None = None
            self._afd_suppress_metadata_send = False
            self._afd_transaction_counter = 0
            self._afd_hccl_execution: tuple[str, int] | None = None
            self.hccl_v41_graph_eager = None
            self.hccl_v41_graph_session = None
            self.hccl_v41_graph_coordinator = None
            self.hccl_v41_graph_bootstrap = None
            self.hccl_v41_eager_bootstrap = None
            self._hccl_graph_load_started = False
            self._hccl_eager_load_started = False
            self.dspark_pd_consumer = None
            self._dspark_pd_load_started = False
            self._dspark_pd_pending_scheduler: SchedulerOutput | None = None
            if isinstance(connector, HcclStagedAFDConnector):
                self.bind_hccl_v41_transport(connector.transport)
        except BaseException:
            try:
                if connector is not None:
                    connector.close()
            finally:
                super().shutdown()
            raise

    def _agree_dspark_pd_batch(
        self, report: DsparkPdWriteReport
    ) -> tuple[DsparkPdWriteReport, ...]:
        """All TP ranks decide publication after their actual local writes."""
        group = get_tp_group()
        if group.world_size == 1:
            return (report,)
        reports = [report for _ in range(group.world_size)]
        torch.distributed.all_gather_object(reports, report, group=group.cpu_group)
        return tuple(reports)

    def _get_dspark_pd_connector(self) -> MooncakeHybridConnector:
        # Lazy worker-local PD dependency: ordinary AFD workers need not import
        # the optional Mooncake connector and its transfer-engine module.
        from afd_plugin.compat.npu.kv_layout_connector import MooncakeHybridConnector

        connector = get_kv_transfer_group()
        if type(connector) is not MooncakeHybridConnector:
            raise RuntimeError(
                "DSpark prefix ownership requires its exact PD connector"
            )
        return connector

    @contextmanager
    def _dspark_pd_execution(
        self,
        scheduler_output: SchedulerOutput,
        *,
        dummy_run: bool,
        is_profile: bool,
    ) -> Iterator[None]:
        """Retain one native P parent until its sampling writes have finished."""
        consumer = self.dspark_pd_consumer
        if consumer is None:
            yield
            return
        if consumer.terminal:
            raise RuntimeError("DSpark prefix consumer is terminal")
        if self._dspark_pd_pending_scheduler is not None:
            consumer.fail()
            raise RuntimeError("DSpark prefix publication is pending native sampling")
        consumer.reap_finished(scheduler_output.finished_req_ids)
        transfer = self.vllm_config.kv_transfer_config
        assert transfer is not None
        publish_step = (
            not dummy_run
            and not is_profile
            and bool(scheduler_output.total_num_scheduled_tokens)
            and transfer.kv_role == "kv_producer"
        )
        try:
            yield
            if publish_step:
                self._dspark_pd_pending_scheduler = scheduler_output
        except BaseException:
            # Successful peers now retain their parent, rather than entering a
            # write-report collective here. Native RPC timeout/process ownership
            # handles a failed execute; do not introduce a mismatched collective.
            consumer.fail()
            raise

    def _afd_num_tokens_for_context(self, forward_context: ForwardContext) -> int:
        # vLLM 0.30.0's token chain is
        # scheduler_output.total_num_scheduled_tokens ->
        # dispatch_cg_and_sync_dp(..., need_eager=is_profile or skip_compiled)
        # -> ordinary ModelCudaGraphManager.dispatch() or the forced-eager
        # descriptor -> BatchExecutionDescriptor.num_tokens ->
        # prepare_inputs(...).num_tokens_after_padding. For eager execution,
        # CAMP2P many-to-one execution pads the descriptor and DP counts before
        # native input preparation, including eager fallback and dummy ranks.
        batch_descriptor = forward_context.batch_descriptor
        if batch_descriptor is None:
            raise RuntimeError(
                "AFD ModelRunnerV2 requires a native eager BatchDescriptor",
            )
        return int(batch_descriptor.num_tokens)

    # Native MRV2 KV maintenance creates a context without a model forward or
    # BatchDescriptor. It must not create/publish an A/F transaction; ordinary
    # compute contexts still require their native eager descriptor.
    def install_afd_metadata_on_forward_context(
        self, forward_context: ForwardContext
    ) -> None:
        if (
            forward_context.batch_descriptor is None
            and forward_context.attn_metadata is None
            and self.vllm_config.kv_transfer_config is not None
        ):
            return
        if isinstance(self.connector, P2pHcclAFDConnector):
            self._install_hccl_metadata(forward_context, self.connector)
            return
        super().install_afd_metadata_on_forward_context(forward_context)

    def _install_hccl_metadata(
        self,
        forward_context: ForwardContext,
        connector: P2pHcclAFDConnector,
    ) -> None:
        """Publish one typed plan at the native model-forward context boundary."""
        if self._afd_hccl_execution is None:
            raise RuntimeError("HCCL forward context requires an execute_model scope")
        if forward_context.additional_kwargs is None:
            forward_context.additional_kwargs = {}
        if "afd_metadata" in forward_context.additional_kwargs:
            return
        mode, valid_tokens = self._afd_hccl_execution
        padded_tokens = self._afd_num_tokens_for_context(forward_context)
        metadata = self.build_afd_metadata(None, valid_tokens)
        metadata.tokens_lens = [padded_tokens]
        assert metadata.transaction_id is not None
        plan = HcclForwardPlan(
            transaction_id=metadata.transaction_id,
            layer_indices=connector.target_layer_indices,
            valid_tokens=valid_tokens,
            padded_tokens=padded_tokens,
            mode=mode,
        )
        connector.begin_forward_plan(plan)
        forward_context.additional_kwargs["afd_metadata"] = metadata
        self._afd_pending_metadata = metadata

    @staticmethod
    def parse_config(vllm_config: VllmConfig) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="attention")

    def bind_hccl_v41_transport(self, transport: HcclStagedTransport) -> None:
        """Select explicit v3 lifecycle before model loading or KV initialization.

        Configuration/model guards retain their ordinary admission. This API
        supplies the runtime dependency; it does not enable guarded features.
        """
        validate_hccl_v41_binding(self.vllm_config, transport, self.device)
        if (
            self.hccl_v41_transport is not None
            or self.ubatch_runner is not None
            or not isinstance(
                self.connector, (P2pHcclAFDConnector, HcclStagedAFDConnector)
            )
            or self.connector.is_initialized
            or (
                isinstance(self.connector, HcclStagedAFDConnector)
                and transport is not self.connector.transport
            )
        ):
            raise RuntimeError("staged transport must bind before connector/KV startup")
        self.hccl_v41_transport = transport

    def bind_hccl_v41_graph_eager(
        self, eager: HcclV41GraphEager, session: HcclV41GraphSession
    ) -> None:
        """Bind the explicit graph protocol before native pre-KV profiling.

        Automatic bootstrap or an explicit caller supplies the cache and fixed
        generation after loading. Native KV initialization later installs this
        same session on its U2 runner. Model qualification remains separate.
        """
        transport = self.hccl_v41_transport
        if (
            self.hccl_v41_graph_eager is not None
            or self.hccl_v41_diagnostics is not None
            or self.hccl_v41_graph_session is not None
            or self.hccl_v41_graph_coordinator is not None
            or eager.owner is not self
            or transport is None
            or eager.control.transport is not transport
            or eager.control.manager is not None
            or eager.control.cache.closed
            or eager.control.cache.terminal
            or session.transport is not transport
            or self.cudagraph_manager is not None
            or self.ubatch_runner is not None
            or not transport.initialized
            or transport.closed
            or transport.terminal
            or transport.plan is not None
            or transport.graph_reservation is not None
            or transport.negotiated_capabilities.graph_execution_version != 1
        ):
            raise RuntimeError("graph eager binding requires its idle pre-KV owners")
        validate_hccl_v41_binding(self.vllm_config, transport, self.device)
        self.hccl_v41_graph_eager = eager
        self.hccl_v41_graph_session = session

    def bind_hccl_v41_eager_session(self, session: HcclV41EagerSession) -> None:
        """Select warmed eager streams after native KV/U2 runner construction.

        Both roles must explicitly bind the same scheduling family. Profiling
        before this call and later native U1 parents use the serial path.
        """
        if (
            session.transport is not self.hccl_v41_transport
            or self.hccl_v41_graph_session is not None
            or not isinstance(self.ubatch_runner, HcclV41UBatchRunner)
        ):
            raise RuntimeError("eager stream binding requires its native U2 runner")
        self.ubatch_runner.bind_eager_session(session)

    def bind_hccl_v41_diagnostics(self, binding: HcclV41DiagnosticBinding) -> None:
        """Bind eager U1/U2 diagnostics independently of a stream session."""
        transport = self.hccl_v41_transport
        if self.hccl_v41_diagnostics is not None or transport is None:
            raise RuntimeError("bind Attention diagnostics once to its staged session")
        binding.require_idle(transport)
        if isinstance(self.ubatch_runner, HcclV41UBatchRunner):
            self.ubatch_runner.bind_diagnostics(binding)
        self.hccl_v41_diagnostics = binding

    def bind_hccl_v41_graph_coordinator(
        self, coordinator: HcclV41GraphCoordinator
    ) -> None:
        """Hand the same graph session to its real-KV native coordinator."""
        eager = self.hccl_v41_graph_eager
        transport = self.hccl_v41_transport
        if (
            self.hccl_v41_graph_coordinator is not None
            or eager is None
            or coordinator.owner is not self
            or coordinator.manager is not self.cudagraph_manager
            or transport is None
            or coordinator.transport is not transport
            or coordinator.control.transport is not transport
            or coordinator.control.manager is not self.cudagraph_manager
            or coordinator.session is not self.hccl_v41_graph_session
            or coordinator.session.transport is not transport
            or coordinator.control.cache is not eager.control.cache
            or coordinator.control.generation is not eager.control.generation
            or coordinator.control.metadata_shapes is not eager.control.metadata_shapes
            or not transport.initialized
            or transport.closed
            or transport.terminal
            or transport.plan is not None
            or transport.graph_reservation is not None
            or coordinator.control.cache.closed
            or coordinator.control.cache.terminal
        ):
            raise RuntimeError(
                "graph coordinator must retain its pre-KV session owners"
            )
        self.hccl_v41_graph_coordinator = coordinator

    # Patch reason: native V2 load_model has no AFD connector lifecycle.
    # Patch functionality: preserve native model loading, then rendezvous the
    # AFD connector after both worker roles have loaded their weights, preparing
    # the selected eager family before native serial memory profiling.
    # Upstream: vllm_ascend/worker/v2/model_runner.py, 7e37e29a, load_model.
    # Signature matches upstream; no added parameters. Delegate the large
    # load/ModelState lifecycle so Engram remains owned by the native runner.
    # The exact DSpark PD path also binds the actual loaded target/draft models;
    # KV storage is attached later by its existing native registration owner.
    # A5 DSpark additionally selects the existing TP4 head adapter on its native
    # draft instances before profiling/capture; native draft/cache/weights stay
    # unchanged. This costs one load-time module walk and the adapter's existing
    # 4x head work; remove this selection when the wheel accepts native 16 heads.
    # Remove when upstream exposes a connector callback after model loading.
    def load_model(
        self,
        load_dummy_weights: bool = False,
        *args,
        **kwargs,
    ) -> None:
        # ### PATCH START: AFD connector rendezvous
        transport = self.hccl_v41_transport
        graph = (
            transport is not None
            and transport.negotiated_capabilities.graph_execution_version == 1
        )
        eager = (
            transport is not None
            and not graph
            and isinstance(self.connector, HcclStagedAFDConnector)
        )
        dspark_pd = uses_deepseek_v41_dspark_pd(self.vllm_config)
        if dspark_pd:
            if self._dspark_pd_load_started:
                raise RuntimeError("DSpark PD model loading is single-use")
            self._dspark_pd_load_started = True
        if graph:
            if self._hccl_graph_load_started or self.hccl_v41_graph_eager is not None:
                raise RuntimeError("native Graph model loading is single-use")
            validate_hccl_v41_binding(self.vllm_config, transport, self.device)
            self._hccl_graph_load_started = True
        elif eager:
            if self._hccl_eager_load_started:
                transport.terminal = True
                if self.hccl_v41_eager_bootstrap is not None:
                    self.hccl_v41_eager_bootstrap.fail()
                raise RuntimeError("native eager model loading is single-use")
            self._hccl_eager_load_started = True
        # ### PATCH END: AFD connector rendezvous
        try:
            super().load_model(load_dummy_weights, *args, **kwargs)
            # ### PATCH START: native DSpark TP4 packed Attention
            speculation = self.vllm_config.speculative_config
            if (
                is_deepseek_v41_model(self.vllm_config)
                and speculation is not None
                and speculation.method == "dspark"
            ):
                assert self.speculator is not None
                adapted_draft_layers = 0
                for module in self.speculator.model.modules():
                    if (
                        isinstance(module, DeepseekV41DSparkAttention)
                        and module.packed_cache_ops is MixedQuantPackedCacheOps
                    ):
                        module.packed_cache_ops = AFDMixedQuantPackedCacheOps
                        adapted_draft_layers += 1
                if adapted_draft_layers:
                    v2_model_runner.logger.info(
                        "AFD DSpark: installed TP4 16-to-64 query-head adapter "
                        "on %d native draft Attention layers.",
                        adapted_draft_layers,
                    )
            # ### PATCH END: native DSpark TP4 packed Attention
            # ### PATCH START: AFD connector rendezvous
            if dspark_pd:
                group = get_tp_group()
                assert self.speculator is not None
                transfer = self.vllm_config.kv_transfer_config
                assert transfer is not None
                self.dspark_pd_consumer = DsparkPdPrefixConsumer(
                    self.vllm_config,
                    self.speculator,
                    target_model=self.model,
                    device=self.device,
                    source_engine_id=str(transfer.engine_id),
                    tp_rank=group.rank_in_group,
                    tp_size=group.world_size,
                    agree_batch=self._agree_dspark_pd_batch,
                )
            if transport is not None:
                if not transport.initialized:
                    transport.initialize()
                validate_hccl_v41_binding(self.vllm_config, transport, self.device)
                settings = (
                    self.connector.overlap_settings
                    if isinstance(self.connector, HcclStagedAFDConnector)
                    else None
                )
                if settings is None:
                    # Explicit legacy bindings keep family 0; load does not
                    # reinterpret process settings after HELLO.
                    settings = HcclOverlapSettings(eager_u2_stream_overlap=False)
                if graph:
                    self.hccl_v41_graph_bootstrap = HcclV41AttentionGraphBootstrap(
                        self,
                        synchronize=partial(torch.npu.synchronize, self.device),
                        overlap_settings=settings,
                    )
                    self.hccl_v41_graph_bootstrap.bind_loaded_model()
                elif eager:
                    self.hccl_v41_eager_bootstrap = HcclV41EagerBootstrap(
                        self.vllm_config,
                        transport,
                        synchronize=partial(torch.npu.synchronize, self.device),
                        overlap_settings=settings,
                    )
                    self.hccl_v41_eager_bootstrap.prepare(
                        torch.npu.current_stream(self.device)
                    )
                    diagnostics = self.hccl_v41_eager_bootstrap.diagnostics
                    if diagnostics is not None:
                        self.bind_hccl_v41_diagnostics(diagnostics)
            elif not self.connector.is_initialized:
                self.connector.init_afd_connector()
            if (
                isinstance(
                    self.connector, (P2pHcclAFDConnector, HcclStagedAFDConnector)
                )
                # Explicit v3 bindings leave the legacy connector unstarted;
                # it cannot attest this distinct live transport's identity.
                and (
                    transport is None
                    or isinstance(self.connector, HcclStagedAFDConnector)
                )
                and not self._hccl_configuration_logged
            ):
                v2_model_runner.logger.info(
                    "AFD HCCL session configuration: %s",
                    self.connector.configuration_record(),
                )
                self._hccl_configuration_logged = True
            # ### PATCH END: AFD connector rendezvous
        except BaseException:
            # ### PATCH START: retain failed HCCL initialization owners
            if self.dspark_pd_consumer is not None:
                self.dspark_pd_consumer.fail()
            if (dspark_pd or eager) and transport is not None:
                transport.terminal = True
                if eager and self.hccl_v41_eager_bootstrap is not None:
                    self.hccl_v41_eager_bootstrap.fail()
            if graph:
                transport.terminal = True
                if self.hccl_v41_graph_bootstrap is not None:
                    self.hccl_v41_graph_bootstrap.fail()
            # ### PATCH END: retain failed HCCL initialization owners
            raise

    # Native MRV2 imports its factory by value; scope that exact binding while
    # preserving Ascend's KV initialization (too large to copy). A controlled
    # Graph binding passes its pre-KV session to that same native U2 factory;
    # eager binds its prepared session only after native allocation succeeds.
    # Signature matches Ascend 7e37e29a; restore the alias even on failure.
    # Delegate native Engram/shared-KV/cache-plan ownership unchanged. Remove
    # the scoped factory override when upstream accepts an ubatch factory.
    # Exact DSpark PD additionally binds its loaded names before native scratch
    # planning; the connector materialization callback seals the actual storage.
    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        # ### PATCH START: create the native NPU ubatch adapter.
        target_graph = (
            use_afd_dspark(self.vllm_config, self.afd_config)
            and not self.model_config.enforce_eager
        )
        draft_graph = (
            self.speculator is not None
            and target_graph
            and not self.speculative_config.enforce_eager
        )
        transport = (
            self.hccl_v41_transport
            if self.vllm_config.parallel_config.use_ubatching
            else None
        )
        binding = self.hccl_v41_transport
        graph_session = self.hccl_v41_graph_session
        bootstrap = self.hccl_v41_graph_bootstrap
        eager_bootstrap = self.hccl_v41_eager_bootstrap
        if eager_bootstrap is not None:
            eager_bootstrap.begin_kv_init()
        try:
            if self.dspark_pd_consumer is not None:
                pd_connector = self._get_dspark_pd_connector()
                worker = pd_connector.connector_worker
                if worker is None or worker.kv_cache_config is not kv_cache_config:
                    raise RuntimeError("DSpark PD must bind the native KV config owner")
                pd_connector.bind_dspark_model(self.dspark_pd_consumer)
                # This is the native optional field consumed before Ascend's
                # deepcopy/apply_layerwise_kv_cache_plan; preserve every KV group.
                kv_cache_config.dspark_draft_layer_names = frozenset(
                    self.dspark_pd_consumer.binding.draft_layer_names
                )
            if (
                binding is not None
                and binding.negotiated_capabilities.graph_execution_version == 1
                and graph_session is None
            ):
                raise RuntimeError(
                    "bind the graph session before native KV initialization"
                )
        except BaseException:
            if eager_bootstrap is not None:
                if self.dspark_pd_consumer is not None:
                    self.dspark_pd_consumer.fail()
                eager_bootstrap.fail()
            raise
        if bootstrap is not None:
            bootstrap.begin_kv_init()
        try:
            with (
                use_npu_ubatch_factory(
                    self,
                    hccl_transport=transport,
                    hccl_graph_owner=graph_session if transport is not None else None,
                ),
                _use_window_dspark_target_graph_manager()
                if target_graph
                else nullcontext(),
            ):
                super().initialize_kv_cache(
                    kv_cache_config, kv_cache_allocation_context
                )
            if bootstrap is not None:
                bootstrap.complete_kv_init()
        except BaseException:
            if self.dspark_pd_consumer is not None:
                self.dspark_pd_consumer.fail()
            if bootstrap is not None:
                bootstrap.fail()
            if eager_bootstrap is not None:
                eager_bootstrap.fail()
            raise
        try:
            if self.hccl_v41_diagnostics is not None and isinstance(
                self.ubatch_runner, HcclV41UBatchRunner
            ):
                self.ubatch_runner.bind_diagnostics(self.hccl_v41_diagnostics)
            if eager_bootstrap is not None:
                eager_bootstrap.complete_kv_init(self.bind_hccl_v41_eager_session)
        except BaseException:
            if eager_bootstrap is not None:
                eager_bootstrap.fail()
            raise
        if draft_graph:
            install_window_dspark_replay(self.speculator.query_cudagraph_manager)
        if (
            self.afd_config.connector == WINDOW_CONNECTOR
            and isinstance(self.ubatch_runner, NPUUBatchRunner)
        ):
            # Native MRV2 prepares only its own Attention builders. U2 owns an
            # independent builder per stage, so initialize their V4.1 source
            # RoPE after KV/cache registration has completed as well.
            self.ubatch_runner.prepare_v41_source_rope()
        # ### PATCH END: create the native NPU ubatch adapter.

    def _uses_fixed_window_u2(self) -> bool:
        return bool(
            getattr(self.connector, "is_window_connector", False)
            and int(getattr(self.connector, "micro_batch_num", 1)) == 2
        )

    # Upstream: Ascend NPUModelRunner.prepare_inputs at 7e37e29a.
    # Fixed Window U2 needs deterministic dummy-child inputs when native MRV2
    # pads a single real child. Keep all native input preparation in super().
    def prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        batch_req_state: BatchReqState,
        batch_desc: BatchExecutionDescriptor,
    ) -> AscendInputBatch:
        input_batch = super().prepare_inputs(
            scheduler_output,
            batch_req_state,
            batch_desc,
        )
        # ### PATCH START: initialize padding for the fixed Window U2 child.
        if not self._uses_fixed_window_u2() or batch_desc.num_ubatches != 2:
            return input_batch

        num_tokens = int(input_batch.num_tokens)
        num_tokens_padded = int(input_batch.num_tokens_after_padding)
        if num_tokens_padded <= num_tokens:
            return input_batch

        # Native MRV2 represents the dummy child as zero real query tokens over
        # one padded model token.  Make that token deterministic and keep its
        # Attention state empty, equivalent to DSV4's fixed-U2 dummy padding.
        padding = slice(num_tokens, num_tokens_padded)
        input_batch.input_ids[padding].copy_(
            input_batch.input_ids[:1].expand(num_tokens_padded - num_tokens)
        )
        input_batch.positions[padding].fill_(127)
        input_batch.is_padding[:num_tokens].fill_(False)
        input_batch.is_padding[padding].fill_(True)
        input_batch.seq_lens[input_batch.num_reqs :].zero_()
        return input_batch
        # ### PATCH END: initialize padding for the fixed Window U2 child.

    # Patch reason: vLLM v0.30.0 prepares FULL graph inputs before each warmup
    # and formal capture forward, outside torch.cuda.graph, but does not expose
    # that lifecycle to AFD's control plane.
    # Patch functionality: temporarily wrap the exact upstream input-preparation
    # symbol so each native FULL descriptor publishes one warmup and one capture
    # payload before its forward, while the provider installs the pending AFD
    # sidecar without sending control from inside torch.cuda.graph. Explicit
    # counts/Graph sessions instead use their owned native coordinator.
    # Signature: matches vLLM v0.30.0 GPUModelRunner.capture_model exactly.
    # Upstream source: vllm/v1/worker/gpu/model_runner.py,
    # GPUModelRunner.capture_model; commit
    # ced6857afa0ea7b2e3f0846a62e1394e90f15607.
    # Delegation exception: native capture remains wholly in super(); this
    # wrapper owns only the temporary AFD lifecycle seam.
    # Removal/upstream plan: delete this wrapper when vLLM exposes graph
    # warmup/capture metadata hooks around prepare_inputs_to_capture.
    def capture_model(self, *, profile_only: bool = False) -> int:
        # ### PATCH START: publish AFD FULL graph warmup/capture control.
        transport = self.hccl_v41_transport
        if (
            transport is not None
            and transport.negotiated_capabilities.graph_execution_version == 1
        ):
            coordinator = self.hccl_v41_graph_coordinator
            if coordinator is None:
                raise RuntimeError("bind the real-KV graph coordinator before capture")
            coordinator.preflight_capture()
            return coordinator.capture_native(profile_only=profile_only)
        manager = self.cudagraph_manager
        capture_descs = manager._capture_descs.get(CUDAGraphMode.FULL, [])
        if self._uses_fixed_window_u2() and not use_afd_dspark(
            self.vllm_config, self.afd_config
        ):
            capture_descs = _normalize_window_u2_capture_descs(capture_descs)
            manager._capture_descs[CUDAGraphMode.FULL] = capture_descs
        # Native graph-memory profiling captures only this descriptor prefix.
        capture_descs = capture_descs[: manager._max_full_descs_to_capture]
        if not capture_descs:
            raise RuntimeError(
                "AFD NPU ModelRunnerV2 ACL graph expected at least one FULL "
                "capture descriptor",
            )
        for desc in capture_descs:
            if desc.cg_mode != CUDAGraphMode.FULL:
                raise RuntimeError(
                    "AFD NPU ModelRunnerV2 ACL graph expected only FULL capture "
                    "descriptors",
                )
        event_tracker = _AFDCaptureEventTracker(capture_descs)
        previous_metadata = self._afd_pending_metadata
        previous_suppress_send = self._afd_suppress_metadata_send
        previous_is_warmup = self._is_warmup
        previous_is_graph_capturing = self._afd_is_graph_capturing
        previous_is_graph_replaying = self._afd_is_graph_replaying
        # Pinned Ascend patch_v2/patch_input_batch.py already supplies
        # AscendInputBatch in both native dummy/capture branches.
        try:
            with (
                _use_afd_capture_input_preparation(self, event_tracker),
                _use_afd_ubatch_preparation(
                    self,
                    event_tracker,
                    self.ubatch_runner.record_capture_graph_params
                    if self.ubatch_runner is not None
                    else None,
                ),
                use_afd_metadata_provider(
                    self.install_afd_metadata_on_forward_context,
                ),
            ):
                result = super().capture_model(profile_only=profile_only)
                event_tracker.assert_complete()
            return result
        finally:
            self._afd_pending_metadata = previous_metadata
            self._afd_suppress_metadata_send = previous_suppress_send
            self._is_warmup = previous_is_warmup
            self._afd_is_graph_capturing = previous_is_graph_capturing
            self._afd_is_graph_replaying = previous_is_graph_replaying
        # ### PATCH END: publish AFD FULL graph warmup/capture control.

    # Patch reason: native V2 creates ForwardContext inside execute_model, so
    # AFD must install its sidecar at that exact context-construction seam.
    # Patch functionality: delegate all request/input/Attention/KV/sampling/
    # output work to native V2 while temporarily installing AFD metadata. An
    # explicit Graph session reserves its mode before native input preparation,
    # using its eager bridge before KV initialization and coordinator after KV.
    # Signature: matches Ascend 7e37e29a NPUModelRunner.execute_model, including
    # its missing return annotation and the forwarded dummy-state parameters.
    # Upstream source: vllm_ascend/worker/v2/model_runner.py,
    # NPUModelRunner.execute_model; commit
    # 7e37e29a7803d6897a7113088e2b771f83205bc0.
    # Delegation exception: the upstream method is intentionally not copied;
    # only this narrow provider/profiler wrapper is AFD-specific.
    # Async CAM uses the v0.30 native DP opt-out for requests and profile/dummy
    # execution. The scope leaves TP collectives and the configured topology intact.
    # Removal/upstream plan: delete this wrapper when vLLM exposes a plugin
    # ForwardContext/set_forward_context sidecar/provider hook.
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
    ):
        # ### PATCH START: scope AFD metadata provider/replay and profiler step.
        with (
            afd_npu_profiler_execute(self.prof),
            self._dspark_pd_execution(
                scheduler_output, dummy_run=dummy_run, is_profile=is_profile
            ),
        ):
            transport = self.hccl_v41_transport
            if transport is not None:
                mode = "profile" if is_profile else "dummy" if dummy_run else "target"
                valid_tokens = (
                    0 if dummy_run else int(scheduler_output.total_num_scheduled_tokens)
                )
                if transport.negotiated_capabilities.graph_execution_version == 1:
                    graph_execution = (
                        self.hccl_v41_graph_eager
                        if self.cudagraph_manager is None
                        else self.hccl_v41_graph_coordinator
                    )
                    if graph_execution is None:
                        raise RuntimeError(
                            "bind the owning graph lifecycle before execute"
                        )
                    with graph_execution.execution_scope(
                        mode=mode, valid_tokens=valid_tokens, dummy_run=dummy_run
                    ):
                        return super().execute_model(
                            scheduler_output,
                            intermediate_tensors,
                            dummy_run=dummy_run,
                            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                            is_profile=is_profile,
                            context_len=context_len,
                            valid_dummy_state_slots=valid_dummy_state_slots,
                        )
                execution = HcclV41Execution(
                    self,
                    transport,
                    mode=mode,
                    valid_tokens=valid_tokens,
                    dummy_run=dummy_run,
                    diagnostics=self.hccl_v41_diagnostics,
                )
                with execution.scope():
                    return super().execute_model(
                        scheduler_output,
                        intermediate_tensors,
                        dummy_run=dummy_run,
                        skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                        is_profile=is_profile,
                        context_len=context_len,
                        valid_dummy_state_slots=valid_dummy_state_slots,
                    )
            use_fullgraph_replay_hook = (
                self.vllm_config.compilation_config.cudagraph_mode
                in (CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY)
                and self.cudagraph_manager is not None
            )
            previous_metadata = self._afd_pending_metadata
            previous_suppress_send = self._afd_suppress_metadata_send
            previous_is_warmup = self._is_warmup
            previous_is_graph_capturing = self._afd_is_graph_capturing
            previous_is_graph_replaying = getattr(
                self, "_afd_is_graph_replaying", False
            )
            previous_is_profile = self._afd_is_profile
            self._afd_is_profile = bool(is_profile)
            self._afd_is_graph_replaying = False
            is_hccl = isinstance(self.connector, P2pHcclAFDConnector)
            if is_hccl:
                previous_hccl_execution = self._afd_hccl_execution
                self._afd_hccl_execution = (
                    "profile" if is_profile else "dummy" if dummy_run else "target",
                    0
                    if dummy_run
                    else int(scheduler_output.total_num_scheduled_tokens),
                )
                self._afd_pending_metadata = None

            window_dspark = use_afd_dspark(self.vllm_config, self.afd_config)
            replay_scope = (
                _use_afd_fullgraph_replay_hook(
                    self,
                    int(scheduler_output.total_num_scheduled_tokens),
                    verifying_drafts=(
                        not dummy_run
                        and window_dspark
                        and bool(scheduler_output.scheduled_spec_decode_tokens)
                    ),
                )
                if use_fullgraph_replay_hook
                else nullcontext()
            )
            try:
                with (
                    skip_dp_coordination()
                    if self.afd_config.async_dp and not window_dspark
                    else nullcontext(),
                    _use_camp2p_dp_padding(
                        self.afd_config.num_attention_ranks,
                        self.afd_config.num_ffn_ranks,
                    )
                    if not self.afd_config.async_dp
                    and self.afd_config.num_attention_ranks
                    > self.afd_config.num_ffn_ranks
                    else nullcontext(),
                    replay_scope,
                    use_async_cam_dbo_dispatch(
                        allow_target_graph=not dummy_run or not window_dspark,
                        force_window_u2=self._uses_fixed_window_u2() and not dummy_run,
                        dspark_query_width=self.decode_query_len if window_dspark else None,
                    )
                    if self.afd_config.async_dp
                    and (self.ubatch_runner is not None or window_dspark)
                    else nullcontext(),
                    _use_afd_ubatch_preparation(self),
                    use_afd_metadata_provider(
                        self.install_afd_metadata_on_forward_context,
                    ),
                ):
                    return super().execute_model(
                        scheduler_output,
                        intermediate_tensors,
                        dummy_run=dummy_run,
                        skip_attn_for_dummy_run=skip_attn_for_dummy_run,
                        is_profile=is_profile,
                        context_len=context_len,
                        valid_dummy_state_slots=valid_dummy_state_slots,
                    )
            finally:
                self._afd_pending_metadata = previous_metadata
                self._afd_suppress_metadata_send = previous_suppress_send
                self._is_warmup = previous_is_warmup
                self._afd_is_graph_capturing = previous_is_graph_capturing
                self._afd_is_graph_replaying = previous_is_graph_replaying
                self._afd_is_profile = previous_is_profile
                if is_hccl:
                    self._afd_hccl_execution = previous_hccl_execution
        # ### PATCH END: scope AFD metadata provider/replay and profiler step.

    # Patch reason: a registered DSpark PD owner's target forward may succeed
    # before native sampling/proposal fails, after history or draft KV writes.
    # Patch functionality: publish P Ready only after native proposal/post writes
    # and final prefix projection; retain failed owners and reject their retries.
    # Upstream: vllm_ascend/worker/v2/model_runner.py, 7e37e29a, sample_tokens.
    # Signature matches upstream, including its missing annotations.
    # Delegation exception: Ascend/GPU own the full sampling, history, proposal
    # and async-output lifecycle; duplicating them would enlarge this failure
    # boundary. The existing projection/report/sync moves from execute to this
    # final frontier; no additional device synchronization is added. Remove when
    # native sampling exposes a quiescent PD publication/failure callback.
    def sample_tokens(self, grammar_output):
        # ### PATCH START: retain failed DSpark PD sampling owners.
        consumer = self.dspark_pd_consumer
        scheduler_output = (
            None if consumer is None else self._dspark_pd_pending_scheduler
        )
        state = self.execute_model_state
        if (
            consumer is not None
            and consumer.terminal
            and (state is not None or scheduler_output is not None)
        ):
            raise RuntimeError("DSpark prefix consumer is terminal")
        report_started = False
        try:
            output = super().sample_tokens(grammar_output)
            if scheduler_output is not None:
                assert consumer is not None
                connector = None
                sources: tuple[DsparkPdSourceRequest, ...] = ()
                source_error: Exception | None = None
                try:
                    connector = self._get_dspark_pd_connector()
                    sources = connector.get_dspark_prefill_sources(
                        scheduler_output.kv_connector_metadata
                    )
                except Exception as exc:
                    source_error = exc
                # Native sample cleared its execute_model_state and advanced
                # req_states; projection must use this original parent batch and
                # auxiliary rows. Its existing stream sync follows proposal/post.
                report_started = True
                ready = consumer.consume_prefill(
                    scheduler_output, state, sources, source_error=source_error
                )
                assert connector is not None
                connector.publish_dspark_ready(ready)
                self._dspark_pd_pending_scheduler = None
            return output
        except BaseException as exc:
            if consumer is not None:
                if (
                    isinstance(exc, Exception)
                    and scheduler_output is not None
                    and not report_started
                ):
                    # Peers that finished native sampling enter this same final
                    # report. Preserve the first error if reporting also fails.
                    # A poisoned/stuck native group still needs its own timeout
                    # and process supervision; this is not an abort protocol.
                    with suppress(BaseException):
                        consumer.consume_prefill(
                            scheduler_output, None, (), source_error=exc
                        )
                consumer.fail()
            raise
        # ### PATCH END: retain failed DSpark PD sampling owners.

    def prepare_shutdown(self) -> None:
        """Complete peer/cache/profiler cleanup before native Worker releases KV.

        Only a wholly successful preparation is idempotent. A failed STOP,
        graph reset or export retains native resources for worker supervision.
        """
        if (
            self.dspark_pd_consumer is not None
            and self._dspark_pd_pending_scheduler is not None
        ):
            self.dspark_pd_consumer.fail()
            raise RuntimeError("DSpark prefix publication is pending native sampling")
        if self.dspark_pd_consumer is not None and self.dspark_pd_consumer.terminal:
            raise RuntimeError("DSpark prefix consumer is terminal")
        if isinstance(self.prof, AFDNPUProfilerSession):
            # Reject foreign/failed owners before changing any peer state.
            self.prof.require_owner()
        if self._afd_shutdown_prepared:
            return
        transport = self.hccl_v41_transport
        staged_stop_completed = False
        if transport is not None and transport.initialized:
            if transport.terminal or transport.closed:
                raise RuntimeError("failed staged session requires worker supervision")
            transport.stop_attention()
            staged_stop_completed = True
            if self.hccl_v41_graph_eager is not None:
                self.hccl_v41_graph_eager.control.cache.close()
        elif transport is None and (
            isinstance(self.connector, P2pHcclAFDConnector)
            and self.connector.is_initialized
        ):
            self.connector.request_stop()
        if isinstance(self.prof, AFDNPUProfilerSession):
            self.prof.finish()
        self._afd_staged_stop_completed = staged_stop_completed
        self._afd_shutdown_prepared = True

    # Patch reason: native V2 shutdown does not know about AFD's profiler,
    # connector, pending metadata sidecar, or stage-local graph inputs.
    # Patch functionality: preserve delegated native cleanup and guarantee all
    # AFD cleanup layers run when any earlier layer raises. Explicit Graph
    # caches are reset only after STOP and before native storage release; a
    # failed STOP, cache reset or export retains native state for supervision.
    # Signature: matches vLLM v0.30.0 GPUModelRunner.shutdown exactly.
    # Upstream source: vllm/v1/worker/gpu/model_runner.py,
    # GPUModelRunner.shutdown; commit
    # ced6857afa0ea7b2e3f0846a62e1394e90f15607.
    # Delegation exception: native resource release remains in super().shutdown;
    # this override contains only AFD-specific finalizers.
    # Removal/upstream plan: delete this wrapper if AFD-owned lifecycle moves
    # to a shared composition/factory owner; it never copies native cleanup.
    def shutdown(self) -> None:
        # ### PATCH START: guarantee profiler/native/connector cleanup.
        self.prepare_shutdown()
        transport = self.hccl_v41_transport
        native_cleanup_complete = False
        try:
            stop_afd_npu_profiler(self.prof)
        finally:
            try:
                if self.ubatch_runner is not None:
                    self.ubatch_runner.abort_pending_run()
            finally:
                try:
                    super().shutdown()
                    # Native shutdown synchronizes and releases its graph
                    # manager. Drop the child owner too: it retains model/KV
                    # state, captured MLA registries and persistent RoPE inputs.
                    # If native completion fails, retain it for process cleanup.
                    self.ubatch_runner = None
                    self.hccl_v41_graph_coordinator = None
                    self.hccl_v41_graph_session = None
                    self.hccl_v41_graph_eager = None
                    self.hccl_v41_graph_bootstrap = None
                    self.hccl_v41_eager_bootstrap = None
                    self.hccl_v41_diagnostics = None
                    native_cleanup_complete = True
                finally:
                    self._afd_pending_metadata = None
                    if transport is None:
                        self.connector.close()
                    elif self._afd_staged_stop_completed and native_cleanup_complete:
                        try:
                            self.connector.close()
                        finally:
                            # The staged adapter already owns this transport;
                            # explicit old-adapter injection has a separate one.
                            if not isinstance(self.connector, HcclStagedAFDConnector):
                                transport.close()
        # ### PATCH END: guarantee profiler/native/connector cleanup.


__all__ = ["AFDNPUAttentionModelRunnerV2"]
