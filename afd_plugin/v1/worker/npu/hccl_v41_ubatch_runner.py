# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit V2 input/run seams for the unadmitted staged HCCL model path.

The scoped factory selects this class only with explicit transport injection.
Production callers retain their existing admission. The caller must accept the
v3 parent before native run; U1 fallback and plan collection are separate seams.
Native Ascend execute_model retains the sole finish_execution/Engram retirement.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from threading import get_ident
from typing import Any, Protocol, cast

import torch
from torch import nn
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    ForwardContext,
    create_forward_context,
)
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.ubatch_utils import UBatchState
from vllm.v1.worker.ubatch_utils import UBatchSlices
from vllm.v1.worker.utils import AttentionGroup
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState

from afd_plugin.config import HCCL_P2P_CONNECTOR, parse_afd_config
from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    AFDDeepseekV41AttentionForCausalLM,
)
from afd_plugin.v1.worker.npu.hccl_v41_diagnostics import HcclV41DiagnosticBinding
from afd_plugin.v1.worker.npu.hccl_v41_eager import HcclV41EagerSession
from afd_plugin.v1.worker.npu.hccl_v41_execution import execute_hccl_v41_stages
from afd_plugin.v1.worker.npu.ubatch_runner import NPUUBatchRunner


class HcclV41GraphOwner(Protocol):
    """One parent reservation and tensor body, outside production admission.

    The owner validates the accepted phase before preparation and compares the
    real prepared geometry afterwards. ``forward_parent`` calls the supplied
    native ModelWithContext exactly once in its owning parent context, routing
    the actual model through its staged tensor body. It owns neither native
    factory output storage nor execute_model's final Engram retirement.
    """

    @property
    def transport(self) -> HcclStagedTransport: ...

    def require_full(self, input_batch: InputBatch, *, for_capture: bool) -> None: ...

    def prepared(
        self,
        input_batch: InputBatch,
        ubatch_state: UBatchState,
        *,
        for_capture: bool,
    ) -> None: ...

    def forward_parent(
        self,
        model: nn.Module,
        model_inputs: dict[str, Any],
        ubatch_state: UBatchState,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]: ...

    def fail(self) -> None: ...


class HcclV41UBatchRunner(NPUUBatchRunner):
    # Upstream: NPUUBatchRunner.__init__, adapting UBatchRunner @ ced6857a.
    # Reason: native run has no connector argument, so keep its v3 dependency
    # on the explicitly constructed runner rather than mutable module state.
    # Functionality: retain native buffers/builders under the NPU factory's
    # SM-control scope. Added transport and optional graph_owner are caller-owned;
    # graph construction requires that explicit owner and negotiated capability.
    # Neither is initialized/closed here. Remove with upstream runner injection.
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        model_state: ModelState,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_num_reqs: int,
        *,
        kernel_block_sizes: list[int],
        timeout_s: float,
        transport: HcclStagedTransport,
        graph_owner: HcclV41GraphOwner | None = None,
    ):
        # ### PATCH START: admit explicit synchronous U2 construction
        afd_config = parse_afd_config(vllm_config)
        parallel = vllm_config.parallel_config
        if (
            afd_config.connector != HCCL_P2P_CONNECTOR
            or afd_config.role != "attention"
            or transport.role != "attention"
            or not parallel.use_ubatching
            or parallel.num_ubatches != 2
        ):
            raise ValueError(
                "explicit staged HCCL construction requires eager Attention U2"
            )
        if graph_owner is None:
            if (
                not vllm_config.model_config.enforce_eager
                or vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            ):
                raise ValueError(
                    "explicit staged HCCL construction requires eager Attention U2"
                )
        elif (
            vllm_config.model_config.enforce_eager
            or vllm_config.compilation_config.cudagraph_mode
            != CUDAGraphMode.FULL_DECODE_ONLY
            or vllm_config.speculative_config is not None
        ):
            raise ValueError(
                "explicit HCCL graph owner requires FULL_DECODE_ONLY "
                "and speculation off"
            )
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.use_sequence_parallel_moe
        ):
            raise ValueError("V4.1 staged HCCL requires PP1, CP1 and non-SP")
        if not isinstance(model_state, EngramModelState):
            raise ValueError("V4.1 staged HCCL requires the native Engram ModelState")
        if parallel.data_parallel_size != 1 or parallel.data_parallel_rank != 0:
            raise ValueError("V4.1 staged HCCL construction requires DP1 rank zero")
        capabilities = transport.negotiated_capabilities
        if capabilities.max_stages < 2:
            raise ValueError("V4.1 staged HCCL construction requires negotiated U2")
        if capabilities.native_tp_size != parallel.tensor_parallel_size:
            raise ValueError("V4.1 staged HCCL transport native TP size must match")
        if capabilities.attention_size != parallel.tensor_parallel_size:
            raise ValueError(
                "V4.1 staged HCCL role size must match its native TP world"
            )
        if graph_owner is not None and (
            capabilities.graph_execution_version != 1
            or capabilities.local_counts_version != 1
            or capabilities.ffn_size != parallel.tensor_parallel_size
        ):
            raise ValueError(
                "explicit HCCL graph owner requires graph protocol v1 "
                "and paired native TP"
            )
        if graph_owner is not None and graph_owner.transport is not transport:
            raise ValueError("explicit HCCL graph owner must own the bound transport")
        # WorkerBase writes the actual native rank to parallel.rank. With
        # DP1/PP1/CP1, one Attention role is exactly that native TP world.
        if parallel.rank != transport.role_rank:
            raise ValueError("V4.1 staged HCCL native rank does not match role_rank")
        self.transport = transport
        # ### PATCH END: admit explicit synchronous U2 construction
        super().__init__(
            vllm_config,
            device,
            model_state,
            attn_groups,
            kv_cache_config,
            max_num_reqs,
            kernel_block_sizes=kernel_block_sizes,
            timeout_s=timeout_s,
        )
        # ### PATCH START: retain the explicit parent graph owner
        self.graph_owner = graph_owner
        self.eager_session: HcclV41EagerSession | None = None
        self.diagnostic_binding: HcclV41DiagnosticBinding | None = None
        self._preparing_cg_mode = CUDAGraphMode.NONE
        # ### PATCH END: retain the explicit parent graph owner

    # Upstream: UBatchRunner.prepare @ ced6857a and EngramModelState.prepare_attn
    # @ 7e37e29a. Patch reason: NPU child metadata preparation temporarily
    # bypasses Engram's override that records the parent's graph mode.
    # Patch functionality: retain the selected eager/FULL parent mode until
    # native prepare_inputs chooses its single history/lookup producer. An
    # explicit graph owner validates FULL geometry and the accepted phase.
    # Signature matches native prepare; no added parameters. Delegation keeps
    # native request/KV slicing and Ascend metadata builders in their owners.
    # Removal: use an upstream ModelState parent-mode preparation hook.
    def prepare(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        for_capture: bool = False,
    ) -> UBatchState:
        # ### PATCH START: explicit eager or accepted FULL parent preparation
        full_graph = cg_mode == CUDAGraphMode.FULL
        if (cg_mode != CUDAGraphMode.NONE or for_capture) and not (
            full_graph and self.graph_owner is not None
        ):
            raise ValueError("V4.1 staged HCCL preparation requires eager execution")
        if not isinstance(self.model_state, EngramModelState):
            raise ValueError("V4.1 staged HCCL requires the native Engram ModelState")
        if (
            self.parallel_config.pipeline_parallel_size != 1
            or self.parallel_config.decode_context_parallel_size != 1
            or self.parallel_config.prefill_context_parallel_size != 1
            or self.parallel_config.use_sequence_parallel_moe
        ):
            raise ValueError("V4.1 staged HCCL requires PP1, CP1 and non-SP execution")
        previous_mode = self.model_state._cg_mode
        previous_context_mode = self._preparing_cg_mode
        try:
            if full_graph:
                assert self.graph_owner is not None
                self.graph_owner.require_full(input_batch, for_capture=for_capture)
            self.model_state._cg_mode = cg_mode
            self._preparing_cg_mode = cg_mode
            metadata = self.model_state.device_metadata if full_graph else None
            # Native TargetDeviceMetadata retains one build's tasks until the
            # owning parent forward. Collect both independent child builders in
            # that single frontier rather than replacing it on the second child.
            with (
                metadata.build(
                    [groups for stage in self.stage_attn_groups for groups in stage],
                    full_graph=True,
                )
                if metadata is not None
                else nullcontext()
            ):
                prepared = super().prepare(
                    input_batch, block_tables, slot_mappings, cg_mode, for_capture
                )
            if metadata is not None:
                for context in prepared.forward_contexts:
                    for resource in context.attn_metadata.values():
                        resource.device_metadata_executor = metadata.executor
            if full_graph:
                self.graph_owner.prepared(
                    input_batch, prepared, for_capture=for_capture
                )
            return prepared
        except BaseException:
            self.model_state._cg_mode = previous_mode
            if full_graph:
                self.graph_owner.fail()
            raise
        finally:
            self._preparing_cg_mode = previous_context_mode
        # Keep the selected mode on success until native parent prepare_inputs
        # consumes it. Restoring here would select a stale producer mode.
        # ### PATCH END: explicit eager or accepted FULL parent preparation

    # Upstream: UBatchRunner.begin_capturable_run @ ced6857a:432-524.
    # Reason: native model capture directly calls this method instead of run,
    # and native child threads each execute the complete model. Functionality:
    # return one synchronous parent callback owned by the bound graph owner,
    # preserving the supplied native ModelWithContext and output factory.
    # Signature matches upstream; no added parameters. The native threaded body
    # is unsuitable: the HCCL graph owner owns a single layer-major tensor body.
    # No native pending-thread callback is installed; abort has no parked thread
    # to join. Remove with upstream synchronous capture callback injection.
    def begin_capturable_run(
        self,
        model: Any,
        model_inputs: dict[str, Any],
        ubatch_state: UBatchState,
        for_capture: bool = False,
    ) -> Callable[[], Any]:
        # ### PATCH START: one owning-thread graph parent callback
        owner = self.graph_owner
        if (
            owner is None
            or not for_capture
            or self.model_state._cg_mode != CUDAGraphMode.FULL
            or self._prepared_state is not ubatch_state
            or not isinstance(model, nn.Module)
        ):
            raise ValueError(
                "HCCL graph callback requires a prepared FULL capture and graph owner"
            )
        thread_id = get_ident()
        consumed = False

        def finish() -> Any:
            nonlocal consumed
            if consumed or get_ident() != thread_id:
                owner.fail()
                raise RuntimeError(
                    "HCCL graph callback must run once on its owning thread"
                )
            consumed = True
            try:
                return owner.forward_parent(model, model_inputs, ubatch_state)
            except BaseException:
                owner.fail()
                raise

        # ### PATCH END: one owning-thread graph parent callback
        return finish

    # Upstream: UBatchRunner._make_forward_contexts @ ced6857a:389-421.
    # Patch reason: native DPMetadata.make asserts for DP1/non-SP and invents
    # uniform child counts for DP>1, whereas HCCL peers use the accepted plan.
    # Patch functionality: A-only contexts have no local-MoE DP sidecar and
    # retain the currently prepared eager/FULL mode; native parent Engram
    # preparation retains its real DP context. F owns MoE counts.
    # Signature matches upstream; no added parameters. Remove when the native
    # context factory supports externally scheduled A-only microbatches.
    def _make_forward_contexts(
        self,
        ubatch_slices: UBatchSlices,
        attn_metadata: list[dict[str, Any]],
        slot_mappings_by_layer: list[dict[str, torch.Tensor]],
        is_padding: list[torch.Tensor | None],
    ) -> list[ForwardContext]:
        # ### PATCH START: preserve the currently prepared native child mode
        mode = self._preparing_cg_mode
        # ### PATCH END: preserve the currently prepared native child mode
        forward_contexts = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            num_tokens = ubatch_slice.num_tokens
            forward_contexts.append(
                create_forward_context(
                    attn_metadata[i],
                    self.vllm_config,
                    # ### PATCH START: A-only child; plan owns actual peer counts
                    dp_metadata=None,
                    # ### PATCH END: A-only child; plan owns actual peer counts
                    # ### PATCH START: preserve the parent's prepared mode
                    cudagraph_runtime_mode=mode,
                    # ### PATCH END: preserve the parent's prepared mode
                    slot_mapping=slot_mappings_by_layer[i],
                    is_padding=is_padding[i],
                )
            )
            # ### PATCH START: Ascend child kwargs without native handoff threads
            context = forward_contexts[-1]
            context.additional_kwargs = current_platform.set_additional_forward_context(
                context.attn_metadata,
                self.vllm_config,
                context.dp_metadata,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=mode,
                ubatch_slices=ubatch_slices,
            )
            context.additional_kwargs.update(
                ubatch_idx=i, num_ubatches=len(ubatch_slices), dbo_enabled=False
            )
            context.ubatch_idx = i
            # ### PATCH END: Ascend child kwargs without native handoff threads
        return forward_contexts

    # Upstream: UBatchRunner.run @ ced6857a:423-430.
    # Reason: native begin_capturable_run starts full-model handoff threads.
    # Functionality: execute the accepted v3 parent on one host thread using
    # the explicitly injected transport; no model/history lifecycle is copied.
    # Signature matches upstream, including its broad model/return annotations.
    # Remove when native runner supports a synchronous stage execution callback.
    def run(
        self,
        model: Any,
        model_inputs: dict[str, Any],
        ubatch_state: UBatchState,
    ) -> Any:
        # ### PATCH START: native entry uses the accepted caller-owned transport
        transport = self.transport
        if not transport.initialized or transport.closed or transport.terminal:
            raise RuntimeError("native staged run requires an active HCCL transport")
        return self.run_staged(
            cast(AFDDeepseekV41AttentionForCausalLM, model),
            model_inputs,
            ubatch_state,
            transport,
        )
        # ### PATCH END: native entry uses the accepted caller-owned transport

    def run_staged(
        self,
        model: AFDDeepseekV41AttentionForCausalLM,
        model_inputs: dict[str, Any],
        ubatch_state: UBatchState,
        transport: HcclStagedTransport,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Run one accepted parent without native microbatch threads.

        Any is confined to the native ModelState's heterogeneous kwargs at this
        boundary. Model execution uses typed stage inputs. This method neither
        submits child history nor retires the native parent's producer.
        """
        session = self.eager_session
        if session is not None and session.transport is not transport:
            raise ValueError("native eager execution requires its bound transport")
        diagnostics = self.diagnostic_binding
        if diagnostics is not None and diagnostics.transport is not transport:
            raise ValueError("native diagnostics require their bound transport")
        pipeline = None if session is None else session.open_parent()
        accepted_plan = transport.plan
        schedule = (
            {"eager_pipeline": pipeline, "compute_stream": session.compute_stream}
            if pipeline is not None
            else {}
        )
        try:
            progress = (
                None
                if diagnostics is None
                else diagnostics.bind_parent(
                    eager_pipeline=pipeline,
                    compute_stream=None if pipeline is None else session.compute_stream,
                )
            )
            return execute_hccl_v41_stages(
                model,
                self.model_state,
                model_inputs,
                tuple(ubatch_state.forward_contexts),
                tuple(stage.token_slice for stage in ubatch_state.slices),
                transport,
                **({"progress": progress} if progress is not None else {}),
                **schedule,
            )
        except BaseException:
            if (
                pipeline is not None
                and not pipeline.finished
                and not pipeline.failed
                and transport.eager_pipeline is pipeline
                and transport.plan is accepted_plan
            ):
                pipeline.fail()
            if transport.plan is None or transport.plan is accepted_plan:
                transport.terminal = True
            raise

    def bind_eager_session(self, session: HcclV41EagerSession) -> None:
        """Inject a warmed caller-owned session before accepting any parent."""
        if (
            self.eager_session is not None
            or self.graph_owner is not None
            or session.transport is not self.transport
            or not session.initialized
            or not self.transport.initialized
            or self.transport.closed
            or self.transport.terminal
            or self.transport.plan is not None
            or self.transport.graph_reservation is not None
        ):
            raise RuntimeError("bind one idle eager session to its exact U2 transport")
        if self.diagnostic_binding is not None:
            self.diagnostic_binding.require_settings(session.settings)
        self.eager_session = session

    def bind_diagnostics(self, binding: HcclV41DiagnosticBinding) -> None:
        """Fix diagnostics before any accepted native parent or producer."""
        if self.diagnostic_binding is not None or self.graph_owner is not None:
            raise RuntimeError("bind eager diagnostics once before native execution")
        binding.require_idle(self.transport)
        if self.eager_session is not None:
            binding.require_settings(self.eager_session.settings)
        self.diagnostic_binding = binding
