# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit native model capture assembly for one retained HCCL session.

This component surrounds the selected native capture call. It owns all-peer
preflight, reservations and retained resources, without replacing the native
model factory, output store or final ModelState retirement. Production Graph
admission remains separate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_runner import BatchReqState
from vllm.v1.worker.utils import AttentionGroup
from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheBackend
from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.worker.v2.aclgraph_utils import (
    ModelAclGraphManager,
    _get_graph_update_backend,
)
from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2

from afd_plugin.compat.patches.npu.hccl_graph_capture import use_hccl_graph_capture
from afd_plugin.compat.patches.npu.hccl_graph_dispatch import (
    HcclGraphDispatchArguments,
    use_hccl_graph_dispatch,
)
from afd_plugin.compat.patches.npu.hccl_graph_replay import use_hccl_graph_replay
from afd_plugin.compat.patches.npu.hccl_v41_u1_graph import (
    use_hccl_v41_u1_preparation,
)
from afd_plugin.connectors.npu.hccl_graph_cache import (
    HcclGraphCacheEntry,
    HcclReplayGraph,
)
from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank
from afd_plugin.distributed.hccl_graph import (
    HcclGraphCandidate,
    HcclGraphReport,
    graph_capture_sequence_digest,
)
from afd_plugin.distributed.hccl_plan import HcclLocalForward
from afd_plugin.model_executor.models.forward_context import use_afd_metadata_provider
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    HCCL_V41_PARENT_FORWARD_KEY,
)
from afd_plugin.v1.worker.npu.hccl_v41_execution import HcclV41Execution
from afd_plugin.v1.worker.npu.hccl_v41_graph import (
    eager_hccl_v41_geometry,
    validate_hccl_v41_capture_coverage,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph_control import (
    HcclV41GraphControl,
    HcclV41GraphReservation,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph_execution import HcclV41GraphExecution
from afd_plugin.v1.worker.npu.hccl_v41_graph_session import HcclV41GraphSession
from afd_plugin.v1.worker.npu.hccl_v41_ubatch_runner import HcclV41UBatchRunner

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )


@dataclass(frozen=True, slots=True)
class HcclV41GraphResources:
    """Exact references retained across native runner field replacements."""

    bank: HcclGraphStageBank
    execution: HcclV41GraphExecution
    model_state: EngramModelState
    input_buffers: InputBuffers
    ubatch_runner: HcclV41UBatchRunner | None
    attn_groups: tuple[tuple[AttentionGroup, ...], ...]
    block_tables: BlockTables
    kv_cache_config: KVCacheConfig
    kv_caches: tuple[torch.Tensor, ...]
    manager: ModelAclGraphManager
    hidden_states: torch.Tensor | None
    auxiliary: tuple[torch.Tensor, ...]
    intermediate_tensors: IntermediateTensors | None


class HcclV41GraphCoordinator:
    """One native ModelState/input generation and one capture invocation.

    All Attention ranks call preflight_capture before invoking native capture.
    capture_native invokes the selected native Ascend capture only when the
    returned result is true. A false result means all peers agreed to skip
    FULL capture, not that native capture or model execution succeeded.
    Profiling and later real-KV captures require fresh session/cache ownership;
    recapture of an existing key is intentionally unsupported.
    """

    def __init__(
        self,
        owner: AFDNPUAttentionModelRunnerV2,
        control: HcclV41GraphControl[HcclV41GraphResources],
        session: HcclV41GraphSession,
        *,
        next_transaction_id: Callable[[], str],
    ) -> None:
        manager = owner.cudagraph_manager
        transport = control.transport
        capabilities = transport.negotiated_capabilities
        text = owner.vllm_config.model_config.hf_text_config
        if (
            not isinstance(manager, ModelAclGraphManager)
            or control.manager is not manager
            or manager.model_runner is not owner
            or owner.hccl_v41_transport is not transport
            or session.transport is not transport
            or not isinstance(owner.model_state, EngramModelState)
            or owner.vllm_config.model_config.enforce_eager
            or owner.vllm_config.compilation_config.cudagraph_mode
            != CUDAGraphMode.FULL_DECODE_ONLY
            or owner.vllm_config.speculative_config is not None
        ):
            raise ValueError(
                "native graph coordinator requires its FULL_DECODE_ONLY owners"
            )
        if (
            owner.vllm_config.model_config.dtype != torch.bfloat16
            or text.hidden_size != capabilities.hidden_size
            or text.vocab_size != capabilities.vocab_size
            or tuple(range(text.num_hidden_layers)) != capabilities.layer_indices
            or owner.device.type != "npu"
            or owner.device.index != transport.local_device
            or manager.ubatch_runner is not owner.ubatch_runner
            or owner.model_state.model is not owner.model
            or owner.pcp_manager is not None
        ):
            raise ValueError(
                "native graph coordinator requires its model/device/storage layout"
            )
        if owner.ubatch_runner is not None and (
            not isinstance(owner.ubatch_runner, HcclV41UBatchRunner)
            or owner.ubatch_runner.graph_owner is not session
        ):
            raise ValueError("native graph coordinator requires its bound U2 session")
        with set_current_vllm_config(owner.vllm_config):
            backend = _get_graph_update_backend(owner.attn_groups)
        if backend not in (AscendSFABackend, DeepseekV41CacheBackend):
            raise ValueError("native graph coordinator requires V4.1 SFA/DSA metadata")
        descriptors = manager._capture_descs.get(CUDAGraphMode.FULL, [])[
            : manager._max_full_descs_to_capture
        ]
        self.coverage = validate_hccl_v41_capture_coverage(
            descriptors, transport.negotiated_capabilities
        )
        if owner.ubatch_runner is None and any(
            geometry.descriptor.num_ubatches != 1
            for geometry in self.coverage.geometries
        ):
            raise ValueError(
                "native U2 capture requires the selected microbatch runner"
            )
        self.owner, self.manager = owner, manager
        self.model, self.model_state = owner.model, owner.model_state
        self.input_buffers = owner.input_buffers
        self.ubatch_runner = owner.ubatch_runner
        self.attn_groups = owner.attn_groups
        self.attn_group_refs = tuple(tuple(groups) for groups in owner.attn_groups)
        self.block_tables = owner.block_tables
        self.kv_cache_config = owner.kv_cache_config
        self.kv_caches = tuple(owner.kv_caches)
        self.intermediate_tensors = owner.intermediate_tensors
        self.capture_descriptors = tuple(descriptors)
        self.capture_prefix = manager._max_full_descs_to_capture
        self.control, self.session = control, session
        self.transport = transport
        self.next_transaction_id = next_transaction_id
        self.current_execution: HcclV41GraphExecution | None = None
        self._u1_context: ForwardContext | None = None
        self._preflight_done = self._capture_enabled = self._capture_started = False
        self._runtime_active = False

    def _require_storage(self) -> None:
        owner, manager = self.owner, self.manager
        groups = tuple(tuple(group) for group in owner.attn_groups)
        if (
            owner.cudagraph_manager is not manager
            or owner.hccl_v41_transport is not self.transport
            or manager.model_runner is not owner
            or manager.ubatch_runner is not self.ubatch_runner
            or owner.model is not self.model
            or owner.model_state is not self.model_state
            or self.model_state.model is not self.model
            or owner.input_buffers is not self.input_buffers
            or owner.ubatch_runner is not self.ubatch_runner
            or owner.attn_groups is not self.attn_groups
            or len(groups) != len(self.attn_group_refs)
            or any(
                len(actual) != len(retained)
                or any(
                    left is not right
                    for left, right in zip(actual, retained, strict=True)
                )
                for actual, retained in zip(groups, self.attn_group_refs, strict=True)
            )
            or owner.block_tables is not self.block_tables
            or owner.kv_cache_config is not self.kv_cache_config
            or len(owner.kv_caches) != len(self.kv_caches)
            or any(
                left is not right
                for left, right in zip(owner.kv_caches, self.kv_caches, strict=True)
            )
            or owner.intermediate_tensors is not self.intermediate_tensors
            or owner.pcp_manager is not None
            or manager._max_full_descs_to_capture != self.capture_prefix
            or tuple(
                manager._capture_descs.get(CUDAGraphMode.FULL, [])[
                    : self.capture_prefix
                ]
            )
            != self.capture_descriptors
        ):
            raise ValueError(
                "native graph capture storage or descriptor ownership changed"
            )

    def _transaction_id(self) -> str | None:
        return self.next_transaction_id() if self.transport.is_coordinator else None

    def fail(self) -> None:
        self.session.fail()
        self.control.cache.fail()

    def preflight_capture(self) -> bool:
        """Agree capture eligibility with an honest miss and zero-row fallback.

        A replay availability query names a real native descriptor that is
        genuinely uncaptured. Eligible peers report missing; unsupported peers
        report unsupported. No peer advertises ready or executes that graph.
        The common eager result is retired as an empty parent on every role.
        """
        try:
            self._require_storage()
            if (
                self._preflight_done
                or not self.control.cache.is_empty
                or self.manager.graphs
                or self.transport.plan is not None
                or self.transport.graph_reservation is not None
            ):
                raise ValueError("capture eligibility requires a fresh graph session")
            native = self.owner.model.language_model.model
            supported = bool(self.coverage.geometries) and (
                not native.has_engram or native.can_capture_engram_producer()
            )
            keys = tuple(
                self.control.key_for(geometry) for geometry in self.coverage.geometries
            )
            sequence_digest = graph_capture_sequence_digest(
                keys, max_full_descs_to_capture=self.capture_prefix
            )

            def resolve(candidate: HcclGraphCandidate) -> HcclGraphReport:
                if (
                    candidate.phase not in ("replay", "eager")
                    or candidate.mode != "warmup"
                    or candidate.eager_padded_tokens != (0,)
                    or (candidate.key is None) != (candidate.phase == "eager")
                    or candidate.capture_sequence_digest is None
                ):
                    raise ValueError("invalid native capture eligibility query")
                matching = (
                    candidate.capture_sequence_digest == sequence_digest
                    and bool(keys)
                    and candidate.key == keys[0]
                )
                return HcclGraphReport(
                    "missing" if supported and matching else "unsupported"
                )

            candidate = None
            if self.transport.is_coordinator:
                candidate = HcclGraphCandidate(
                    "replay" if keys else "eager",
                    "warmup",
                    keys[0] if keys else None,
                    (0,),
                    capture_sequence_digest=sequence_digest,
                )
            decision = self.transport.reserve_attention_forward(
                resolve, candidate=candidate, transaction_id=self._transaction_id()
            )
            if (
                decision is None
                or decision.phase != "eager"
                or decision.reason not in ("cache_miss", "unsupported", "requested")
            ):
                raise ValueError(
                    "capture eligibility must agree an empty eager fallback"
                )
            plan = self.transport.submit_forward(
                HcclLocalForward("warmup", (0,), (0,)),
                transaction_id=decision.transaction_id
                if self.transport.is_coordinator
                else None,
            )
            if plan is None or any(
                any(stage.attention_padded_tokens) for stage in plan.stages
            ):
                raise ValueError(
                    "capture eligibility requires an empty accepted parent"
                )
            self.transport.finish_forward()
            self._preflight_done = True
            self._capture_enabled = bool(keys) and decision.reason == "cache_miss"
            return self._capture_enabled
        except BaseException:
            self.fail()
            raise

    def resolve_u1_owner(self) -> HcclV41GraphExecution | None:
        execution = self.current_execution
        if (
            execution is None
            or execution.reservation.geometry.descriptor.num_ubatches != 1
        ):
            return None
        return execution

    def _require_current(self) -> HcclV41GraphExecution:
        self._require_storage()
        execution = self.current_execution
        if (
            execution is None
            or self.transport.terminal
            or self.transport.closed
            or self.transport.plan is None
            or self.transport.graph_reservation is not execution.reservation.decision
        ):
            raise ValueError("native graph callback requires its active reservation")
        self.control.verify_prepared(
            execution.reservation, execution.reservation.geometry
        )
        return execution

    def install_context(self, context: ForwardContext) -> None:
        """Install only the accepted U1 tensor callback before native wrapper entry."""
        execution = self.resolve_u1_owner()
        if execution is None or execution.prepared_state is None:
            return
        child = execution.prepared_state.forward_contexts[0]
        if context.attn_metadata is not child.attn_metadata:
            return
        self._require_current()
        self._u1_context = context
        context.additional_kwargs[HCCL_V41_PARENT_FORWARD_KEY] = self._run_u1

    def _run_u1(
        self, model_inputs: dict[str, Any]
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        try:
            execution = self._require_current()
            if (
                get_forward_context() is not self._u1_context
                or execution.parent_body is None
            ):
                raise ValueError(
                    "native U1 graph body requires its owning parent context"
                )
            return execution.parent_body(model_inputs)
        except BaseException:
            self.fail()
            raise

    def _retain(self, execution: HcclV41GraphExecution) -> HcclV41GraphResources:
        manager = self.manager
        return HcclV41GraphResources(
            execution.bank,
            execution,
            execution.model_state,
            self.input_buffers,
            self.ubatch_runner,
            self.attn_group_refs,
            self.block_tables,
            self.kv_cache_config,
            self.kv_caches,
            manager,
            manager.hidden_states,
            tuple(manager.aux_hidden_states),
            manager.intermediate_tensors,
        )

    def _finish_phase(self) -> None:
        execution = self.current_execution
        if execution is None:
            return
        self._require_current()
        reservation = execution.reservation
        key = reservation.decision.key
        assert key is not None
        if reservation.decision.phase == "capture":
            graph = self.manager.graphs[reservation.geometry.descriptor]
            self.control.cache.reserve_capture(
                key,
                reservation.generation,
                cast(HcclReplayGraph, graph),
                self._retain(execution),
            )
        self.transport.complete_graph_forward(key)
        if reservation.decision.phase == "capture":
            self.control.cache.publish_capture(key, reservation.generation)
        self.session.clear()
        self.current_execution, self._u1_context = None, None

    def _before_factory(
        self, descriptor: BatchExecutionDescriptor, warmup: bool
    ) -> None:
        self._finish_phase()
        self._require_storage()
        if descriptor not in tuple(
            geometry.descriptor for geometry in self.coverage.geometries
        ):
            raise ValueError("native factory descriptor is outside validated coverage")
        reservation = self.control.reserve_capture(
            descriptor, warmup=warmup, transaction_id=self._transaction_id()
        )
        if reservation is None:
            raise RuntimeError(
                "native graph session stopped before capture preparation"
            )
        assert reservation.decision.key is not None
        bank = HcclGraphStageBank.allocate(
            self.transport, reservation.decision.key, device=self.owner.device
        )
        execution = HcclV41GraphExecution(
            self.owner.vllm_config,
            self.owner.model,
            self.owner.model_state,
            self.control,
            reservation,
            bank,
        )
        self.current_execution = execution
        self.session.arm(execution)

    @contextmanager
    def _capture_scope(self) -> Iterator[None]:
        if (
            not self._preflight_done
            or not self._capture_enabled
            or self._capture_started
        ):
            raise ValueError(
                "native capture requires one successful all-peer preflight"
            )
        self._capture_started = True
        try:
            with (
                use_hccl_graph_capture(
                    self.manager,
                    before_factory=self._before_factory,
                    after_capture=self._finish_phase,
                    fail=self.fail,
                ),
                use_hccl_v41_u1_preparation(
                    self.owner.model_state,
                    self.owner.vllm_config,
                    self.resolve_u1_owner,
                ),
                use_afd_metadata_provider(self.install_context),
            ):
                yield
        except BaseException:
            self.fail()
            raise

    def capture_native(self, *, profile_only: bool = False) -> int:
        """Call the pinned native lifecycle with this runner's exact storage.

        The Ascend runner inherits GPUModelRunner.capture_model unchanged at
        ced6857afa0ea7b2e3f0846a62e1394e90f15607. That method supplies this
        owner's model, ModelState, inputs, block tables and cache configuration
        to its ModelAclGraphManager; callers cannot inject foreign capture
        parameters. Its ModelWithContext still owns metadata begin/finish and
        its native factory still prepares inputs and stores outputs once.
        Legacy AFD capture scopes must not surround this selected native call.
        """
        if not self._preflight_done:
            raise ValueError("native capture requires all-peer eligibility preflight")
        if not self._capture_enabled:
            return 0
        try:
            self._require_storage()
            with self._capture_scope():
                return NPUModelRunnerV2.capture_model(
                    self.owner, profile_only=profile_only
                )
        except BaseException:
            self.fail()
            raise

    @contextmanager
    def _replay_metadata(self, descriptor: BatchExecutionDescriptor) -> Iterator[None]:
        execution = self._require_current()
        if (
            execution.reservation.decision.phase != "replay"
            or execution.reservation.geometry.descriptor != descriptor
            or execution.prepared_state is None
        ):
            raise ValueError("native replay requires its prepared parent descriptor")
        previous = self.model_state.attn_metadata
        # Ascend 7e37e29a ModelAclGraphManager.run_fullgraph reads this field
        # for its parent prepare_engram/update context. Native U2 preparation
        # instead returns separate child metadata. Only the pinned V4.1 SFA/
        # DSA no-op update contract is admitted by this coordinator: hand the
        # first child's valid metadata to that lifecycle context, preserving
        # each child's graph-recorded metadata and builders without flattening.
        self.model_state.attn_metadata = execution.prepared_state.forward_contexts[
            0
        ].attn_metadata
        try:
            yield
        finally:
            self.model_state.attn_metadata = previous

    def _after_replay(self, descriptor: BatchExecutionDescriptor) -> None:
        execution = self._require_current()
        if execution.reservation.geometry.descriptor != descriptor:
            raise ValueError("native replay completion requires its parent descriptor")
        self._finish_phase()

    def _require_cached_resources(
        self,
        entry: HcclGraphCacheEntry[HcclV41GraphResources],
        reservation: HcclV41GraphReservation,
    ) -> None:
        resources = entry.resources
        captured = resources.execution
        if (
            captured.bank is not resources.bank
            or captured.parent_body is None
            or captured.parent_body.stages.bank is not resources.bank
            or captured.parent_body.stages.model is not self.model.language_model.model
            or captured.model is not self.model
            or captured.model_state is not self.model_state
            or captured.control is not self.control
            or captured.transport is not self.transport
            or captured.reservation.decision.key != reservation.decision.key
            or captured.reservation.generation != reservation.generation
            or resources.model_state is not self.model_state
            or resources.input_buffers is not self.input_buffers
            or resources.ubatch_runner is not self.ubatch_runner
            or resources.block_tables is not self.block_tables
            or resources.kv_cache_config is not self.kv_cache_config
            or len(resources.kv_caches) != len(self.kv_caches)
            or any(
                left is not right
                for left, right in zip(resources.kv_caches, self.kv_caches, strict=True)
            )
            or len(resources.attn_groups) != len(self.attn_group_refs)
            or any(
                len(actual) != len(retained)
                or any(
                    left is not right
                    for left, right in zip(actual, retained, strict=True)
                )
                for actual, retained in zip(
                    resources.attn_groups, self.attn_group_refs, strict=True
                )
            )
            or resources.manager is not self.manager
            or self.manager.graphs.get(reservation.geometry.descriptor)
            is not entry.graph
            or resources.hidden_states is not self.manager.hidden_states
            or len(resources.auxiliary) != len(self.manager.aux_hidden_states)
            or any(
                left is not right
                for left, right in zip(
                    resources.auxiliary, self.manager.aux_hidden_states, strict=True
                )
            )
            or resources.intermediate_tensors is not self.manager.intermediate_tensors
        ):
            raise ValueError("native replay cache bundle does not own captured storage")

    @contextmanager
    def execution_scope(
        self, *, mode: str, valid_tokens: int, dummy_run: bool
    ) -> Iterator[None]:
        """Surround one native execute; native still retires ModelState once.

        Actual gather/dispatch selects and agrees the mode before metadata or
        Engram preparation. Only then is the eager or FULL preparation scope
        installed. The scope does not invoke native execute or sampling itself.
        Runtime use is explicit after KV initialization. The pre-KV native
        profile EAGER bridge and production admission remain separate.
        """
        if (
            self._runtime_active
            or self.current_execution is not None
            or self.transport.plan is not None
            or self.transport.graph_reservation is not None
            or mode not in ("target", "dummy", "profile")
            or type(valid_tokens) is not int
            or valid_tokens < 0
        ):
            raise ValueError("native execution requires one idle owning graph session")
        self._require_storage()
        self._runtime_active = True
        reservation: HcclV41GraphReservation | None = None

        def select(
            batch_state: BatchReqState | None,
            arguments: HcclGraphDispatchArguments,
            native_descriptor: BatchExecutionDescriptor,
        ) -> BatchExecutionDescriptor:
            nonlocal reservation
            self._require_storage()
            native = self.model.language_model.model
            if batch_state is None:
                geometry = eager_hccl_v41_geometry(native_descriptor)
                reservation = self.control.reserve_eager(
                    geometry, mode=mode, transaction_id=self._transaction_id()
                )
            else:
                reservation = self.control.reserve_batch(
                    batch_state,
                    parallel_config=self.owner.parallel_config,
                    allow_ubatching=arguments.allow_ubatching,
                    need_eager=(
                        arguments.need_eager
                        or native.has_engram
                        and not native.can_capture_engram_producer()
                    ),
                    max_query_len=arguments.max_query_len,
                    num_active_loras=arguments.num_active_loras,
                    transaction_id=self._transaction_id(),
                    mode=mode,
                )
            if reservation is None:
                raise RuntimeError("native graph session stopped before preparation")
            if reservation.decision.phase == "eager":
                # In particular do not enter this scope before graph selection:
                # it establishes NONE mode and installs an eager-only provider.
                eager = HcclV41Execution(
                    self.owner,
                    self.transport,
                    mode=mode,
                    valid_tokens=valid_tokens,
                    dummy_run=dummy_run,
                )
                preparation.enter_context(eager.scope(manage_dispatch=False))
            else:
                key = reservation.decision.key
                assert key is not None
                cached = self.control.cache.find(key, reservation.generation)
                if cached is None:
                    raise ValueError(
                        "accepted native replay lost its retained graph bank"
                    )
                self._require_cached_resources(cached, reservation)
                execution = HcclV41GraphExecution(
                    self.owner.vllm_config,
                    self.model,
                    self.model_state,
                    self.control,
                    reservation,
                    cached.resources.bank,
                )
                self.current_execution = execution
                self.session.arm(execution)
                preparation.enter_context(
                    use_hccl_v41_u1_preparation(
                        self.model_state, self.owner.vllm_config, self.resolve_u1_owner
                    )
                )
                preparation.enter_context(
                    use_afd_metadata_provider(self.install_context)
                )
            return reservation.geometry.descriptor

        try:
            with (
                ExitStack() as preparation,
                use_hccl_graph_dispatch(self.owner, select),
                use_hccl_graph_replay(
                    self.manager,
                    before_replay=self._replay_metadata,
                    after_replay=self._after_replay,
                    fail=self.fail,
                ),
            ):
                yield
                if reservation is None:
                    if valid_tokens:
                        raise RuntimeError(
                            "native execution returned before graph dispatch"
                        )
                    # Native zero-work paths may return even before dispatch.
                    # Reserve an honest zero descriptor, without request state.
                    zero = BatchExecutionDescriptor(CUDAGraphMode.NONE, 0, 0)
                    reservation = self.control.reserve_eager(
                        eager_hccl_v41_geometry(zero),
                        mode=mode,
                        transaction_id=self._transaction_id(),
                    )
                    if reservation is None:
                        raise RuntimeError(
                            "native graph session stopped before empty parent"
                        )
                if reservation.geometry.descriptor.num_tokens == 0:
                    plan = self.transport.submit_forward(
                        HcclLocalForward(mode, (0,), (0,)),
                        transaction_id=reservation.decision.transaction_id
                        if self.transport.is_coordinator
                        else None,
                    )
                    if plan is None:
                        raise RuntimeError(
                            "native graph session stopped before empty acceptance"
                        )
                    self.transport.finish_forward()
                if (
                    self.current_execution is not None
                    or self.transport.plan is not None
                    or self.transport.graph_reservation is not None
                ):
                    raise RuntimeError(
                        "native execution did not complete its accepted parent"
                    )
        except BaseException:
            self.fail()
            raise
        finally:
            self._runtime_active = False
