# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native parent execution scope for explicitly bound synchronous v3 sessions.

The caller binds transport before loading. Native Ascend retains its execute
and Engram retirement lifecycle; no production feature guard is opened here.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from copy import copy
from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.ubatch_utils import UBatchState, slice_model_inputs
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState

from afd_plugin.compat.npu.stream import use_npu_current_stream
from afd_plugin.compat.patches.npu.mrv2_ubatching import use_async_cam_dbo_dispatch
from afd_plugin.connectors.npu.hccl_config import validate_staged_hccl_binding
from afd_plugin.connectors.npu.hccl_diagnostics import (
    HcclDiagnosticExecution,
    HcclStageDiagnosticRun,
)
from afd_plugin.connectors.npu.hccl_eager_pipeline import HcclEagerPipeline
from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.distributed.hccl_plan import HcclLocalForward
from afd_plugin.model_executor.models.forward_context import use_afd_metadata_provider
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    HCCL_V41_PARENT_FORWARD_KEY,
    AFDDeepseekV41AttentionForCausalLM,
)
from afd_plugin.model_executor.models.npu.deepseek_v41_staged import (
    DeepseekV41EngramInputs,
    DeepseekV41StageInputs,
    run_deepseek_v41_stages,
)

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
        AFDNPUAttentionModelRunnerV2,
    )
    from afd_plugin.v1.worker.npu.hccl_v41_diagnostics import HcclV41DiagnosticBinding


def validate_hccl_v41_binding(
    vllm_config: VllmConfig, transport: HcclStagedTransport, device: torch.device
) -> None:
    """Bind one native TP world without changing production admission."""
    validate_staged_hccl_binding(vllm_config, transport, device)
    parallel = vllm_config.parallel_config
    if parallel.use_ubatching and (
        parallel.num_ubatches != 2 or transport.negotiated_capabilities.max_stages < 2
    ):
        raise ValueError("staged binding requires a negotiated U2 capacity")
    if not parallel.use_ubatching and parallel.num_ubatches not in (0, 1):
        raise ValueError("staged binding requires native U1 or U2 configuration")


class HcclV41Execution:
    """One native execute scope, including U1 before KV/ubatch initialization."""

    def __init__(
        self,
        owner: AFDNPUAttentionModelRunnerV2,
        transport: HcclStagedTransport,
        *,
        mode: str,
        valid_tokens: int,
        dummy_run: bool,
        diagnostics: HcclV41DiagnosticBinding | None = None,
    ) -> None:
        self.owner, self.transport = owner, transport
        self.mode, self.valid_tokens, self.dummy_run = mode, valid_tokens, dummy_run
        self.prepared: UBatchState | None = None
        self.local: HcclLocalForward | None = None
        self.parent_context: ForwardContext | None = None
        self.diagnostics = diagnostics
        self.progress: HcclStageDiagnosticRun | None = None
        if diagnostics is not None and diagnostics.transport is not transport:
            raise ValueError("native diagnostics require the execution transport")

    def install_context(self, context: ForwardContext) -> None:
        # KV maintenance has no model call. Child contexts are constructed by
        # the native runner's imported alias and have no parent descriptor.
        if context.batch_descriptor is None:
            return
        if self.parent_context is not None:
            raise RuntimeError("one native execution must create one parent context")
        if context.cudagraph_runtime_mode != CUDAGraphMode.NONE:
            raise ValueError("staged native execution requires an eager parent")
        self.parent_context = context
        if self.local is None:
            self.local = HcclLocalForward(
                self.mode,
                (self.valid_tokens,),
                (int(context.batch_descriptor.num_tokens),),
            )
        transaction_id = None
        if self.transport.negotiated_capabilities.graph_execution_version:
            decision = self.transport.graph_reservation
            if decision is None or decision.phase != "eager":
                raise ValueError(
                    "native eager execution requires its eager reservation"
                )
            if self.transport.is_coordinator:
                transaction_id = decision.transaction_id
        elif self.transport.is_coordinator:
            self.owner._afd_transaction_counter += 1
            transaction_id = f"native-{self.owner._afd_transaction_counter}"
        plan = self.transport.submit_forward(self.local, transaction_id=transaction_id)
        if plan is None:
            raise RuntimeError("staged session stopped before native forward")
        if self.prepared is None:
            if self.diagnostics is not None:
                self.progress = self.diagnostics.bind_parent()
            if context.additional_kwargs is None:
                context.additional_kwargs = {}
            context.additional_kwargs[HCCL_V41_PARENT_FORWARD_KEY] = self

    def __call__(
        self, model_inputs: dict[str, Any]
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if (
            self.parent_context is None
            or get_forward_context() is not self.parent_context
        ):
            raise RuntimeError("native U1 must run in its accepted parent context")
        if self.prepared is not None:
            raise RuntimeError("native U2 must execute through its microbatch runner")
        child = copy(self.parent_context)
        child.additional_kwargs = dict(self.parent_context.additional_kwargs or {})
        child.additional_kwargs.update(ubatch_idx=0, num_ubatches=1, dbo_enabled=False)
        child.ubatch_idx, child.dp_metadata = 0, None
        return execute_hccl_v41_stages(
            self.owner.model,
            self.owner.model_state,
            model_inputs,
            (child,),
            (slice(0, model_inputs["positions"].shape[0]),),
            self.transport,
            engram_prepared=True,
            progress=self.progress,
        )

    @contextmanager
    def scope(self, *, manage_dispatch: bool = True) -> Iterator[None]:
        """Own eager preparation; an outer graph selector may own dispatch.

        ``manage_dispatch=False`` keeps the already selected native descriptor
        and the outer pre-preparation agreement hook in control of dispatch.
        """
        runner = self.owner.ubatch_runner
        if not isinstance(self.owner.model_state, EngramModelState):
            raise ValueError("staged execution requires the native Engram ModelState")
        # U1/profile may skip prepare_attn; establish the parent producer mode
        # before native prepare_inputs without reproducing its history work.
        self.owner.model_state._cg_mode = CUDAGraphMode.NONE
        original_prepare = runner.prepare if runner is not None else None
        had_override = runner is not None and "prepare" in vars(runner)
        original_diagnostics = (
            runner.diagnostic_binding
            if runner is not None and self.diagnostics is not None
            else None
        )
        if original_diagnostics is not None:
            original_diagnostics.require_settings(self.diagnostics.settings)

        # Upstream UBatchRunner.prepare @ ced6857a. Reason: capture authoritative
        # native child counts before parent preparation. Functionality: retain
        # native metadata/slicing and record one local report; no wire is sent
        # until the parent context. Signature matches upstream. Remove when
        # native execute exposes prepared-state callbacks.
        # Delegation exception: the selected Hccl/NPU prepare override owns
        # backend slicing and eager validation; copying the base body here
        # would bypass that selected instance's preparation contract.
        def prepare(
            self,
            input_batch: InputBatch,
            block_tables: tuple[torch.Tensor, ...],
            slot_mappings: torch.Tensor,
            cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
            for_capture: bool = False,
        ) -> UBatchState:
            # ### PATCH START: retain actual native stage counts
            assert original_prepare is not None
            state = original_prepare(
                input_batch, block_tables, slot_mappings, cg_mode, for_capture
            )
            execution.prepared = state
            valid = 0 if execution.dummy_run else input_batch.num_tokens
            execution.local = HcclLocalForward(
                execution.mode,
                tuple(
                    max(0, min(stage.token_slice.stop, valid) - stage.token_slice.start)
                    for stage in state.slices
                ),
                tuple(
                    stage.token_slice.stop - stage.token_slice.start
                    for stage in state.slices
                ),
            )
            # ### PATCH END: retain actual native stage counts
            return state

        execution = self
        try:
            if runner is not None:
                runner.prepare = MethodType(prepare, runner)
                if self.diagnostics is not None:
                    runner.diagnostic_binding = self.diagnostics
            # This existing scope applies native thresholds locally at DP1;
            # its implementation is independent of connector selection.
            with (
                use_async_cam_dbo_dispatch()
                if runner is not None and manage_dispatch
                else nullcontext(),
                use_afd_metadata_provider(self.install_context),
            ):
                yield
        except BaseException:
            self.transport.terminal = True
            raise
        finally:
            if runner is not None:
                if self.diagnostics is not None:
                    runner.diagnostic_binding = original_diagnostics
                if had_override:
                    runner.prepare = original_prepare
                else:
                    del runner.prepare


def execute_hccl_v41_stages(
    model: AFDDeepseekV41AttentionForCausalLM,
    model_state: EngramModelState,
    model_inputs: dict[str, Any],
    contexts: tuple[ForwardContext, ...],
    token_slices: tuple[slice, ...],
    transport: HcclStagedTransport,
    *,
    engram_prepared: bool = False,
    eager_pipeline: HcclEagerPipeline | None = None,
    compute_stream: torch.npu.Stream | None = None,
    progress: HcclStageDiagnosticRun | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Execute one accepted parent; retire its native producer in Ascend only.

    U1's native model pre-hook has already merged produced kwargs. U2 bypasses
    __call__ and consumes the producer here once in its native parent context.
    """
    if (eager_pipeline is None) != (compute_stream is None):
        raise ValueError("eager execution requires both its owner and compute stream")
    if eager_pipeline is not None and (
        eager_pipeline.transport is not transport
        or eager_pipeline.plan is not transport.plan
        or eager_pipeline.finished
        or eager_pipeline.failed
        or transport.eager_pipeline is not eager_pipeline
    ):
        raise ValueError("eager parent requires its exact active transport/plan owner")
    if (
        eager_pipeline is not None
        and torch.npu.current_stream(eager_pipeline.parent_stream.device)
        != eager_pipeline.parent_stream
    ):
        raise ValueError("eager parent must enter on its actual owning stream")
    if progress is not None:
        plan = transport.plan
        if plan is None:
            raise ValueError("Attention diagnostics require an accepted parent")
        progress.validate_parent(
            role="attention",
            world_rank=transport.world_rank,
            role_rank=transport.role_rank,
            num_stages=len(plan.stages),
            layer_indices=plan.layer_indices,
            transaction_id=plan.transaction_id,
        )
        if eager_pipeline is None:
            progress.validate_execution(
                HcclDiagnosticExecution(mode="serial", compute_stream="parent")
            )
        else:
            eager_pipeline.bind_progress(progress, compute_stream=compute_stream)
    with (
        use_npu_current_stream(eager_pipeline.parent_stream)
        if eager_pipeline is not None
        else nullcontext()
    ):
        return _execute_hccl_v41_parent(
            model,
            model_state,
            model_inputs,
            contexts,
            token_slices,
            transport,
            engram_prepared=engram_prepared,
            eager_pipeline=eager_pipeline,
            compute_stream=compute_stream,
            progress=progress,
        )


def _execute_hccl_v41_parent(
    model: AFDDeepseekV41AttentionForCausalLM,
    model_state: EngramModelState,
    model_inputs: dict[str, Any],
    contexts: tuple[ForwardContext, ...],
    token_slices: tuple[slice, ...],
    transport: HcclStagedTransport,
    *,
    engram_prepared: bool,
    eager_pipeline: HcclEagerPipeline | None,
    compute_stream: torch.npu.Stream | None,
    progress: HcclStageDiagnosticRun | None,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    accepted_plan = transport.plan
    try:
        parent_context = get_forward_context()
        if (
            parent_context.cudagraph_runtime_mode != CUDAGraphMode.NONE
            or model_inputs.get("engram_graph_events", False)
            or model_inputs.get("intermediate_tensors") is not None
        ):
            raise ValueError("V4.1 staged HCCL run requires an eager PP1 parent")
        if not isinstance(model_state, EngramModelState):
            raise ValueError("V4.1 staged HCCL requires the native Engram ModelState")
        if model_state._cg_mode != CUDAGraphMode.NONE:
            raise ValueError("V4.1 staged HCCL requires prepared eager Engram mode")
        if transport.role != "attention" or transport.plan is None:
            raise ValueError("V4.1 staged HCCL requires an accepted Attention plan")
        parent_inputs = dict(model_inputs)
        # Native prepare_inputs already built the full parent's lookback
        # and pending producer. Run it here once, in the parent context;
        # bypassing nn.Module.__call__ avoids its per-model pre-hook.
        if not engram_prepared:
            parent_inputs.update(model_state.prepare_engram())
        if parent_inputs.get("engram_graph_events", False):
            raise ValueError("V4.1 staged HCCL does not consume Graph Engram events")
        if not model.language_model.model.has_engram:
            parent_inputs.setdefault("engram_lookups", {})
            parent_inputs.setdefault(
                "engram_mask",
                parent_inputs["positions"].new_empty(0, dtype=torch.bool),
            )
        engram = DeepseekV41EngramInputs(
            parent_inputs["engram_lookups"],
            parent_inputs["engram_mask"],
            parent_inputs.get("engram_pending"),
            parent_inputs.get("engram_mask_ready_event"),
        )
        slices = token_slices
        if not slices or slices[-1].stop != parent_inputs["positions"].shape[0]:
            raise ValueError("V4.1 child slices must cover the parent padded tokens")
        stages = []
        for token_slice in slices:
            child = slice_model_inputs(parent_inputs, token_slice)
            if child["input_ids"] is None:
                raise ValueError("V4.1 staged HCCL requires native raw token IDs")
            stages.append(
                DeepseekV41StageInputs(
                    child["input_ids"],
                    child["positions"],
                    engram.slice(token_slice),
                    child.get("inputs_embeds"),
                )
            )
        schedule = (
            {"pipeline": eager_pipeline, "compute_stream": compute_stream}
            if eager_pipeline is not None or compute_stream is not None
            else {}
        )
        output = run_deepseek_v41_stages(
            model.language_model.model,
            tuple(stages),
            contexts,
            slices,
            transport,
            **({"progress": progress} if progress is not None else {}),
            **schedule,
        )
        if eager_pipeline is None:
            transport.finish_forward()
        else:
            eager_pipeline.finish()
        return output
    except BaseException:
        # A peer can already be waiting on payload/COMPLETE. Mark this
        # endpoint terminal without blocking on an unsolicited error send;
        # backend timeout/process supervision owns failed peer unwinding.
        if transport.plan is None or transport.plan is accepted_plan:
            transport.terminal = True
        if (
            eager_pipeline is not None
            and not eager_pipeline.finished
            and not eager_pipeline.failed
            and transport.eager_pipeline is eager_pipeline
            and transport.plan is accepted_plan
        ):
            eager_pipeline.fail()
        raise
