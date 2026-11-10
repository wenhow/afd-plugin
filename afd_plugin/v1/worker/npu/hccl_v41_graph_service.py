# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit FFN graph-phase service; no production factory selects this owner.

The caller initializes native TP/model/MC2 state and owns the transport and
cache. After acknowledged STOP, close() retires the cache before the caller
closes groups. A failed service retains its active dependencies and marks the
session/cache terminal; only external supervision can recover peers blocked
in device communication.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import torch
from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.forward_context import ForwardContext
from vllm_ascend.ascend_forward_context import _EXTRA_CTX

from afd_plugin.compat.npu import ascend_forward_context
from afd_plugin.config import HCCL_P2P_CONNECTOR, parse_afd_config
from afd_plugin.connectors.metadata import AFDForwardContextMetadata
from afd_plugin.connectors.npu.hccl_config import DEEPSEEK_V41_ARCHITECTURES
from afd_plugin.connectors.npu.hccl_diagnostics import HcclStageDiagnostics
from afd_plugin.connectors.npu.hccl_graph_cache import HcclGraphCache, HcclReplayGraph
from afd_plugin.connectors.npu.hccl_graph_pipeline import (
    HcclGraphComputePipeline,
    HcclGraphPipelinePolicy,
)
from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank
from afd_plugin.connectors.npu.hccl_overlap import (
    HcclOverlapSelection,
    HcclOverlapSettings,
    select_hccl_overlap,
)
from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.distributed.hccl_graph import HcclGraphCandidate, HcclGraphReport
from afd_plugin.distributed.hccl_plan import HcclParentPlan
from afd_plugin.v1.worker.npu.hccl_v41_ffn_graph import HcclV41FFNGraph
from afd_plugin.v1.worker.npu.hccl_v41_ffn_runner import execute_hccl_v41_parent

if TYPE_CHECKING:
    from afd_plugin.model_executor.models.npu.deepseek_v41 import (
        AFDDeepseekV41FFNForCausalLM,
    )
    from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner


@dataclass(frozen=True, slots=True)
class HcclV41FFNGraphResources:
    bank: HcclGraphStageBank
    contexts: tuple[ForwardContext, ...]
    body: HcclV41FFNGraph
    pipeline: HcclGraphComputePipeline | None = None
    overlap_settings: HcclOverlapSettings | None = None


class HcclV41GraphService:
    """Consume accepted phases before submitting any graph or eager payload.

    ``graph_factory`` and ``graph_capture`` are external graph-API seams. Their
    defaults record a real NPUGraph. CPU substitutes test lifecycle/protocol
    boundaries only and cannot establish device capture/replay correctness.
    The cache must be exclusive to this runner and transport session.
    """

    def __init__(
        self,
        runner: AFDNPUFFNModelRunner,
        transport: HcclStagedTransport,
        cache: HcclGraphCache[HcclV41FFNGraphResources],
        generation: Callable[[], int],
        *,
        pipeline_policy: HcclGraphPipelinePolicy | None = None,
        overlap_settings: HcclOverlapSettings | None = None,
        diagnostic_sink: Callable[[dict[str, str | int | bool]], None] | None = None,
        graph_factory: Callable[[], HcclReplayGraph] | None = None,
        graph_capture: Callable[[HcclReplayGraph], AbstractContextManager[None]]
        | None = None,
    ) -> None:
        if pipeline_policy is not None and overlap_settings is not None:
            raise ValueError("choose graph pipeline policy or overlap settings")
        self.runner, self.transport, self.cache = runner, transport, cache
        self._capabilities = transport.negotiated_capabilities
        self.generation = generation
        self._pipeline_policy = pipeline_policy
        self._overlap_settings = overlap_settings
        self._diagnostic_sink = diagnostic_sink
        self.graph_factory = graph_factory or torch.npu.NPUGraph
        self.graph_capture = graph_capture or (
            lambda graph: torch.npu.graph(graph, pool=runner.graph_pool)
        )
        # Keep warmup or partly built resources too: they have no cache entry
        # to retain them if context construction or the body fails.
        self._active_bank: HcclGraphStageBank | None = None
        self._active_contexts: list[ForwardContext] = []
        self._active_resources: HcclV41FFNGraphResources | None = None
        self._active_graph: HcclReplayGraph | None = None
        self._reserved_generation: int | None = None
        self._session_model: AFDDeepseekV41FFNForCausalLM | None = None
        self._acknowledged_stop = self._failed = False
        self._validate_binding()

    @property
    def pipeline_policy(self) -> HcclGraphPipelinePolicy | None:
        """Capture policy fixed for the lifetime of this exclusive cache."""
        return self._pipeline_policy

    @property
    def overlap_settings(self) -> HcclOverlapSettings | None:
        return self._overlap_settings

    def _diagnostics(
        self, selection: HcclOverlapSelection | None
    ) -> HcclStageDiagnostics | None:
        if selection is None or not (
            selection.settings.stage_diagnostics
            or selection.settings.ffn_compute_sync_diagnostics
        ):
            return None
        return HcclStageDiagnostics(
            selection,
            world_rank=self.transport.world_rank,
            role_rank=self.transport.role_rank,
            emit=self._diagnostic_sink,
            synchronize=partial(torch.npu.synchronize, self.runner.device),
        )

    @property
    def acknowledged_stop(self) -> bool:
        """Whether the transport completed the idle STOP/STOP_ACK handshake."""
        return self._acknowledged_stop

    def close(self) -> None:
        """Retire cache dependencies after STOP; the caller still owns groups.

        An active or failed service cannot claim peer cancellation or release
        storage under device work. External supervision owns that recovery.
        """
        if (
            not self.acknowledged_stop
            or self._failed
            or self.transport.plan is not None
            or self.transport.graph_reservation is not None
            or self._active_bank is not None
            or self._active_contexts
            or self._active_resources is not None
            or self._active_graph is not None
        ):
            raise RuntimeError(
                "FFN graph service close requires acknowledged idle STOP"
            )
        if self.cache.closed:
            return
        if self.cache.terminal:
            raise RuntimeError("FFN graph service close retains a failed graph cache")
        try:
            self.cache.close()
        except BaseException:
            self._failed = True
            raise

    def _validate_binding(self) -> None:
        runner, transport = self.runner, self.transport
        config, caps = runner.vllm_config, transport.negotiated_capabilities
        afd = parse_afd_config(config, expected_role="ffn")
        parallel, text = config.parallel_config, config.model_config.hf_text_config
        model_config = config.model_config
        architectures = model_config.hf_config.architectures or ()
        caps.validate()
        if (
            transport.role != "ffn"
            or caps != self._capabilities
            or not transport.initialized
            or transport.closed
            or transport.terminal
            or transport.plan is not None
            or transport.graph_reservation is not None
            or self.cache.closed
            or self.cache.terminal
            or caps.graph_execution_version != 1
            or caps.local_counts_version != 1
            or afd.connector != HCCL_P2P_CONNECTOR
            or afd.compute_gate_on_attention
            or afd.num_attention_ranks != caps.attention_size
            or afd.num_ffn_ranks != caps.ffn_size
            or not config.use_v2_model_runner
            or model_config.hf_config.model_type != "deepseek_v41"
            or not architectures
            or any(arch not in DEEPSEEK_V41_ARCHITECTURES for arch in architectures)
            or model_config.is_encoder_decoder
            or (
                model_config.is_multimodal_model
                and (
                    model_config.multimodal_config is None
                    or not model_config.multimodal_config.language_model_only
                )
            )
            or config.model_config.dtype != torch.bfloat16
            or config.engram_config is not None
            or config.speculative_config is not None
            or config.kv_transfer_config is not None
            or config.lora_config is not None
            or parallel.data_parallel_size != 1
            or parallel.data_parallel_rank != 0
            or parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.use_sequence_parallel_moe
            or config.compilation_config.pass_config.enable_sp
            or parallel.enable_expert_parallel
            or parallel.enable_eplb
            or parallel.enable_elastic_ep
            or parallel.enable_fault_tolerance
            or parallel.tensor_parallel_size != caps.native_tp_size
            or parallel.rank != transport.role_rank
            or runner.device != torch.device("npu", transport.local_device)
            or caps.hidden_size != text.hidden_size
            or caps.vocab_size != text.vocab_size
            or caps.layer_indices != tuple(range(text.num_hidden_layers))
            or caps.max_tokens > config.scheduler_config.max_num_batched_tokens
        ):
            raise ValueError(
                "FFN graph service requires its idle V4.1 paired-TP session, "
                "native DP1/PP1/CP1 rank/device and isolated FFN model configuration"
            )

    def _model(self) -> AFDDeepseekV41FFNForCausalLM:
        model = cast("AFDDeepseekV41FFNForCausalLM", self.runner.get_model())
        if (
            model.get_experts_layer_indices()
            != self.transport.negotiated_capabilities.layer_indices
        ):
            raise ValueError(
                "FFN graph model does not own this session's complete layers"
            )
        if self._session_model is None:
            self._session_model = model
        elif model is not self._session_model:
            raise ValueError("FFN graph model changed inside its transport session")
        return model

    def _generation(self) -> int:
        generation = self.generation()
        if type(generation) is not int or generation < 0:
            raise ValueError("FFN graph generation must be a nonnegative integer")
        return generation

    def _validate_resources(self, resources: HcclV41FFNGraphResources) -> None:
        bank = resources.bank
        pipeline = resources.body.pipeline
        policy = self.pipeline_policy
        if self.overlap_settings is not None:
            policy = HcclGraphPipelinePolicy.from_selection(
                select_hccl_overlap(
                    self.overlap_settings,
                    role="ffn",
                    phase="graph_replay",
                    num_stages=len(bank.stage_rows),
                )
            )
        if (
            bank.role != "ffn"
            or resources.overlap_settings != self.overlap_settings
            or bank.role_rank != self.transport.role_rank
            or bank.device != self.runner.device
            or bank.capabilities != self.transport.negotiated_capabilities
            or resources.body.bank is not bank
            or resources.body.contexts is not resources.contexts
            or resources.body.model is not self._model()
            or pipeline is not resources.pipeline
            or (pipeline is None) != (policy is None)
            or (
                pipeline is not None
                and (
                    policy is None
                    or pipeline.bank is not bank
                    or pipeline.policy != policy
                )
            )
            or len(self.transport.data_groups) < len(bank.exchanges)
            or any(
                exchange is not None
                and exchange.endpoint.group is not self.transport.data_groups[index]
                for index, exchange in enumerate(bank.exchanges)
            )
        ):
            raise ValueError("FFN graph resources belong to another model/session")

    def _resolve(self, candidate: HcclGraphCandidate) -> HcclGraphReport:
        self._reserved_generation = self._generation()
        caps = self.transport.negotiated_capabilities
        if candidate.key is not None and (
            candidate.key.num_tokens > caps.max_tokens
            or candidate.key.num_ubatches > caps.max_stages
        ):
            raise ValueError("FFN graph key exceeds this bank's session capacity")
        policy = self.pipeline_policy
        if self.overlap_settings is not None:
            policy = HcclGraphPipelinePolicy.from_selection(
                select_hccl_overlap(
                    self.overlap_settings,
                    role="ffn",
                    phase="graph_capture",
                    num_stages=2,
                )
            )
        order_unsupported = (
            caps.graph_exchange_order != "stage_serial"
            and candidate.key is not None
            and (
                candidate.key.num_ubatches == 2
                or candidate.capture_sequence_digest is not None
            )
            and (
                policy is None
                or not policy.supports_exchange_order("ffn", caps.graph_exchange_order)
            )
        )
        if candidate.capture_sequence_digest is not None:
            # Attention peers own the native ordered capture list. FFN has no
            # corresponding manager list: it must report an honest miss from
            # a fresh cache, never make preflight look ready after a stale or
            # pending capture. Generation and model/session ownership were
            # checked before this availability report. A genuinely empty
            # Attention capture list requests eager with no invented key.
            self._model()
            if not self.cache.is_empty:
                raise ValueError("capture preflight requires a fresh FFN graph cache")
            return HcclGraphReport(
                "unsupported"
                if candidate.key is None or order_unsupported
                else "missing"
            )
        if candidate.key is None:
            return HcclGraphReport("unsupported")
        self._model()
        if order_unsupported:
            # Report before GRAPH_DECISION: every role selects eager, without
            # entering an unmatched bulk warmup or partial tensor schedule.
            return HcclGraphReport("unsupported")
        if candidate.phase in ("warmup", "capture"):
            if self.cache.contains(candidate.key):
                raise ValueError(
                    "FFN graph key already reserved; restart before recapture"
                )
            return HcclGraphReport("missing")
        entry = self.cache.find(candidate.key, self._reserved_generation)
        if entry is None:
            return HcclGraphReport("missing")
        self._validate_resources(entry.resources)
        if entry.resources.bank.key != candidate.key:
            raise ValueError("FFN graph cache key does not match its retained bank")
        return HcclGraphReport("ready")

    def _prepare_resources(self, plan: HcclParentPlan) -> HcclV41FFNGraphResources:
        decision = plan.graph_execution
        assert decision is not None and decision.key is not None
        model = self._model()
        bank = HcclGraphStageBank.allocate(
            self.transport, decision.key, device=self.runner.device
        )
        self._active_bank = bank
        bank.update_counts(plan)
        contexts = self._active_contexts
        for index, rows in enumerate(bank.stage_rows):
            metadata = AFDForwardContextMetadata(
                tokens_start_loc=[0],
                requests_start_loc=[0],
                stage_idx=index,
                connector=self.runner.connector,
                tokens_lens=[rows],
                num_stages=len(bank.stage_rows),
                transaction_id=plan.transaction_id,
                tokens_unpadded_lens=[
                    plan.stages[index].attention_valid_tokens[bank.role_rank]
                ],
            )
            ids = (
                bank.buffers(index).input_ids if rows else bank.valid_counts[index][:0]
            )
            with ascend_forward_context(
                vllm_config=self.runner.vllm_config,
                afd_metadata=metadata,
                model_instance=model,
                num_tokens=rows,
                num_tokens_across_dp=torch.tensor([rows], dtype=torch.int32),
                aclgraph_runtime_mode=CUDAGraphMode.FULL,
                input_ids=ids,
            ) as context:
                # Like native ModelWithContext, this flag belongs to the
                # selected context, never a process-global phase variable.
                _EXTRA_CTX.capturing = decision.phase == "capture"
                context.ubatch_idx = index
                context.additional_kwargs.update(
                    ubatch_idx=index,
                    num_ubatches=len(bank.stage_rows),
                    dbo_enabled=False,
                )
                contexts.append(context)
        retained_contexts = tuple(contexts)
        selection = None
        policy = self.pipeline_policy
        if self.overlap_settings is not None:
            selection = select_hccl_overlap(
                self.overlap_settings,
                role="ffn",
                phase="graph_warmup" if decision.phase == "warmup" else "graph_capture",
                num_stages=len(bank.stage_rows),
            )
            policy = HcclGraphPipelinePolicy.from_selection(selection)
        pipeline = None
        if policy is not None:
            pipeline = (
                HcclGraphComputePipeline.allocate(bank, selection=selection)
                if selection is not None
                else HcclGraphComputePipeline.allocate(bank, policy=policy)
            )
        resources = HcclV41FFNGraphResources(
            bank,
            retained_contexts,
            HcclV41FFNGraph(
                model,
                retained_contexts,
                bank,
                self.runner.vllm_config.parallel_config,
                pipeline=pipeline,
                diagnostics=self._diagnostics(selection),
            ),
            pipeline=pipeline,
            overlap_settings=self.overlap_settings,
        )
        self._active_resources = resources
        self._validate_resources(resources)
        return resources

    @torch.inference_mode()
    def execute_step(self) -> bool:
        """Serve one parent; acknowledged idle STOP returns False.

        Recording a capture does not execute or validate its arithmetic. The
        accepted capture phase still retires its host parent after publication;
        later replay performs device work. No device failure retries as eager.
        """
        try:
            self._validate_binding()
            with set_current_vllm_config(self.runner.vllm_config):
                decision = self.transport.receive_graph_reservation(self._resolve)
                if decision is None:
                    self._acknowledged_stop = True
                    return False
                plan = self.transport.receive_forward()
                if plan is None:
                    raise RuntimeError(
                        "FFN graph session stopped inside a reserved parent"
                    )
                if decision.phase == "eager":
                    selection = (
                        select_hccl_overlap(
                            self.overlap_settings,
                            role="ffn",
                            phase="eager",
                            num_stages=len(plan.stages),
                        )
                        if self.overlap_settings is not None
                        else None
                    )
                    diagnostics = self._diagnostics(selection)
                    if diagnostics is None:
                        execute_hccl_v41_parent(self.runner, self.transport)
                    else:
                        execute_hccl_v41_parent(
                            self.runner, self.transport, diagnostics=diagnostics
                        )
                    return True
                generation = self._reserved_generation
                if generation is None or generation != self._generation():
                    raise ValueError("FFN graph storage changed after reservation")
                assert decision.key is not None
                if decision.phase == "replay":
                    entry = self.cache.find(decision.key, generation)
                    if entry is None:
                        raise ValueError(
                            "FFN graph readiness changed after reservation"
                        )
                    self._validate_resources(entry.resources)
                    entry.resources.bank.update_counts(plan)
                    entry.graph.replay()
                else:
                    resources = self._prepare_resources(plan)
                    if generation != self._generation():
                        raise ValueError("FFN graph storage changed during preparation")
                    if decision.phase == "warmup":
                        resources.body.run()
                    else:
                        graph = self.graph_factory()
                        self._active_graph = graph
                        self.cache.reserve_capture(
                            decision.key, generation, graph, resources
                        )
                        with self.graph_capture(graph):
                            resources.body.run()
                        if generation != self._generation():
                            raise ValueError("FFN graph storage changed during capture")
                        self.cache.publish_capture(decision.key, generation)
                if generation != self._generation():
                    raise ValueError("FFN graph storage changed during graph phase")
                self.transport.complete_graph_forward(decision.key)
                self._active_bank = self._active_resources = self._active_graph = None
                self._active_contexts = []
                return True
        except BaseException:
            self._failed = True
            self.transport.terminal = True
            self.cache.fail()
            raise
