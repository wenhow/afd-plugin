# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU FFN-side model runner for AFD execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import DPMetadata, override_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm_ascend.patch.worker.patch_draft_quarot import patch_load_weights
from vllm_ascend.worker.model_runner_v1 import (
    NPUModelRunner,
    get_tp_context,
    graph_capture,
)

from afd_plugin.compat.npu import (
    ascend_forward_context,
    fail_if_unsupported_npu_afd_features,
)
from afd_plugin.compat.npu.profiler import (
    create_afd_npu_profiler,
    step_afd_npu_profiler,
    stop_afd_npu_profiler,
)
from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.connectors import (
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
    AFDF2ATransferPayload,
    AFDForwardContextMetadata,
    AFDTransferContext,
)
from afd_plugin.connectors.npu.async_cam import (
    AFDAsyncTransferState,
    CAMAsyncAFDConnector,
)
from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector
from afd_plugin.v1.worker.attention_model_runner import (
    _resolve_world_ranks,
)
from afd_plugin.v1.worker.cuda_graph import (
    AFDGraphRunMode,
    graph_run_mode,
    make_ffn_graph_key,
    make_mtp_ffn_graph_key,
)
from afd_plugin.v1.worker.ffn_model_runner import _set_moe_layer_index

if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

    from afd_plugin.connectors import AFDConnectorBase

logger = init_logger(__name__)


class AFDNPUFFNModelRunner(NPUModelRunner):
    """Connector-driven NPU FFN runner for AFD execution."""

    afd_expected_role = "ffn"

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        afd_config = self.parse_config(vllm_config)
        super().__init__(vllm_config, device)

        self.afd_config = afd_config
        fail_if_unsupported_npu_afd_features(
            vllm_config,
            afd_config=afd_config,
        )
        rank, _ = _resolve_world_ranks()
        local_rank = int(device.index)
        self.connector = AFDConnectorFactory.create_connector(
            rank,
            local_rank,
            vllm_config,
            self.afd_config,
        )
        self.num_layers = int(self.model_config.hf_config.num_hidden_layers)
        self.ffn_stream_overlap_enabled = bool(
            isinstance(self.connector, P2pHcclAFDConnector)
            and self.connector.stream_overlap_enabled
        )
        self.ffn_recv_stream = None
        self.ffn_compute_stream = None
        self.ffn_send_stream = None
        self.ffn_recv_events: dict[tuple[int, int], Any] = {}
        self.ffn_compute_events: dict[tuple[int, int], Any] = {}
        self.ffn_send_events: dict[tuple[int, int], Any] = {}
        self.ffn_graph_recv_ready_event = None
        if self.ffn_stream_overlap_enabled:
            self._initialize_ffn_stream_pipeline(device)
        self.use_aclgraph = _use_npu_aclgraph(vllm_config, self)
        self._acl_graphs: dict[tuple, dict[str, Any]] = {}
        self._mtp_acl_graphs: dict[tuple, dict[str, Any]] = {}
        self.graph_pool = (
            current_platform.get_global_graph_pool() if self.use_aclgraph else None
        )
        self._afd_profiler_role_rank = rank
        self.prof = None
        self._afd_profiler_window_active = False
        self._is_shutdown = False
        self._ffn_input_ids_cache: dict[int, torch.Tensor] = {}
        self.mtp_ffn_model: torch.nn.Module | None = None

    def _initialize_ffn_stream_pipeline(self, device: torch.device) -> None:
        """Create U2 FFN streams/events shared by eager and Graph."""
        self.ffn_recv_stream = torch.npu.Stream(device=device)
        self.ffn_compute_stream = torch.npu.Stream(device=device)
        self.ffn_send_stream = torch.npu.Stream(device=device)
        stage_ids = range(int(self.vllm_config.parallel_config.num_ubatches))
        event_keys = [
            (layer_idx, stage_idx)
            for layer_idx in range(self.num_layers)
            for stage_idx in stage_ids
        ]
        self.ffn_recv_events = {key: torch.npu.Event() for key in event_keys}
        self.ffn_compute_events = {key: torch.npu.Event() for key in event_keys}
        self.ffn_send_events = {key: torch.npu.Event() for key in event_keys}
        self.ffn_graph_recv_ready_event = torch.npu.Event()

    @staticmethod
    def parse_config(vllm_config: VllmConfig) -> AFDConfig:
        return parse_afd_config(vllm_config, expected_role="ffn")

    def initialize_afd_connector(self) -> None:
        self.connector.init_afd_connector()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {}

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        return None

    def profile_run(self) -> None:
        return None

    def load_model(self) -> None:
        if self.speculative_config is None:
            super().load_model()
            return
        if self.speculative_config.method != "mtp" or self.drafter is None:
            raise RuntimeError("DSV4 AFD FFN requires an initialized MTP drafter")

        # The FFN service owns no scheduler-side proposer or sampler. Load the
        # target model normally, then construct only the role-filtered MTP MoE.
        drafter = self.drafter
        self.drafter = None
        try:
            super().load_model()
        finally:
            self.drafter = drafter
        if self.vllm_config.quant_config is not None:
            patch_load_weights(self.vllm_config)
        with get_tp_context(drafter):
            drafter.model = drafter._get_model()
        self.mtp_ffn_model = drafter.get_model()

    def execute_ffn_step(
        self,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
        connector_state_prepared: bool = False,
    ) -> None:
        if dp_metadata_list is None:
            raise RuntimeError("AFD NPU FFN requires dp_metadata_list")
        if not connector_state_prepared:
            self._prepare_connector_step(
                dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
                is_warmup=is_warmup,
            )
        if bool(self.use_aclgraph) and (is_graph_capturing or is_warmup):
            input_ids_by_stage = self._receive_input_ids_before_model(
                dp_metadata_list,
            )
            logger.debug(
                "AFD NPU FFN execute_ffn_step enters capture_model; "
                "key=%s is_graph_capturing=%s is_warmup=%s",
                self._make_graph_key(dp_metadata_list),
                is_graph_capturing,
                is_warmup,
            )
            self.capture_model(
                dp_metadata_list=dp_metadata_list,
                is_warmup=is_warmup,
                is_attn_graph_capturing=is_graph_capturing,
                input_ids_by_stage=input_ids_by_stage,
                connector_state_prepared=True,
            )
            return None
        input_ids_by_stage = self._receive_input_ids_before_model(
            dp_metadata_list,
        )
        self.execute_model(
            dp_metadata_list=dp_metadata_list,
            input_ids_by_stage=input_ids_by_stage,
            connector_state_prepared=True,
        )
        return None

    def execute_connector_driven_step(self) -> None:
        if self.connector.control_plane is not None:
            raise RuntimeError(
                "execute_connector_driven_step requires a connector-driven "
                "AFD connector",
            )
        step_afd_npu_profiler(self.prof)
        self._ffn_forward_connector_driven()
        return None

    def execute_model(
        self,
        scheduler_output: SchedulerOutput | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] | None = None,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
        input_ids_by_stage: dict[int, torch.Tensor] | None = None,
        connector_state_prepared: bool = False,
    ) -> None:
        step_afd_npu_profiler(self.prof)
        if dp_metadata_list is None:
            raise RuntimeError("AFD NPU FFN is connector-driven")
        if not connector_state_prepared:
            self._prepare_connector_step(
                dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
                is_warmup=is_warmup,
            )
        graph_key = self._make_graph_key(dp_metadata_list)
        acl_graphs = self._acl_graphs
        graph_info = acl_graphs.get(graph_key)
        graph_enabled = bool(self.use_aclgraph)
        run_mode = graph_run_mode(
            is_warmup=is_warmup and graph_enabled,
            is_graph_capturing=is_graph_capturing and graph_enabled,
            graph_enabled=graph_enabled,
            # An idle Attention DP can run eager with the same padded shape
            # as a cached graph. Its selected mode must also govern FFN.
            graph_exists=graph_info is not None
            and getattr(self.connector, "target_graph_replay", None) is not False,
        )
        if run_mode in (AFDGraphRunMode.WARMUP, AFDGraphRunMode.CAPTURE):
            return self.execute_ffn_step(
                dp_metadata_list=dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
                is_warmup=is_warmup,
                connector_state_prepared=True,
            )
        if input_ids_by_stage is None and run_mode is AFDGraphRunMode.REPLAY:
            input_ids_by_stage = self._receive_input_ids_before_model(dp_metadata_list)
        if run_mode is AFDGraphRunMode.REPLAY:
            logger.debug(
                "AFD NPU FFN replaying ACL graph; key=%s cached_graphs=%d",
                graph_key,
                len(acl_graphs),
            )
            graph_info["graph"].replay()
            if not self._recv_mtp_phase_ready():
                return None
            self._execute_mtp_after_target(dp_metadata_list)
            return None

        self._ffn_forward(
            dp_metadata_list=dp_metadata_list,
            input_ids_by_stage=input_ids_by_stage,
            update_connector_state=False,
        )
        if not self._recv_mtp_phase_ready():
            return None
        self._execute_mtp_after_target(dp_metadata_list)
        return None

    def _execute_mtp_after_target(
        self,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
    ) -> None:
        control_enabled = bool(
            getattr(self.connector, "mtp_phase_control_enabled", False),
        )
        speculative_config = self.vllm_config.speculative_config
        if speculative_config is None:
            return
        num_speculative_tokens = int(speculative_config.num_speculative_tokens)
        stage_ids = sorted(int(stage_idx) for stage_idx in dp_metadata_list) or [0]
        graph_replay = bool(
            getattr(self.connector, "mtp_phase_graph_replay", False),
        )
        if self._draft_uses_aclgraph() and (
            graph_replay
            or (
                not control_enabled
                # Ascend's dummy drafter inherits its target runtime mode;
                # only a live proposer can independently select draft Graph.
                and getattr(self.connector, "target_graph_replay", None) is not False
                and self._make_mtp_graph_key(dp_metadata_list) in self._mtp_acl_graphs
            )
        ):
            # One graph contains the entire merged proposal, including all N
            # draft iterations, matching Attention's _run_merged_draft.
            self._replay_mtp_graph(dp_metadata_list)
            return
        for speculative_step in range(num_speculative_tokens):
            self._mtp_ffn_forward(
                stage_ids,
                expected_speculative_step=self._expected_mtp_wire_step(
                    speculative_step,
                ),
            )

    def _recv_mtp_phase_ready(self) -> bool:
        if self.vllm_config.speculative_config is None:
            return False
        if not bool(
            getattr(self.connector, "mtp_phase_control_enabled", False),
        ):
            return True
        control_plane = self.connector.control_plane
        recv_phase_ready = getattr(control_plane, "recv_mtp_phase_ready", None)
        if not callable(recv_phase_ready):
            raise RuntimeError(
                "DSV4 AFD MTP requires an explicit phase-aware control plane",
            )
        return bool(recv_phase_ready())

    def _prepare_connector_step(
        self,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
        *,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
    ) -> None:
        control_plane = self.connector.control_plane
        if control_plane is None:
            raise RuntimeError("AFD NPU FFN requires a DP metadata control plane")
        control_plane.update_state_from_dp_metadata(
            _make_dp_metadata_payload(
                dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
                is_warmup=is_warmup,
            ),
        )

    def _receive_input_ids_before_model(
        self,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
    ) -> dict[int, torch.Tensor]:
        """Receive DSV4 IDs before entering an ACL graph region."""
        if not getattr(self.connector, "requires_input_ids", False):
            return {}
        input_ids_by_stage: dict[int, torch.Tensor] = {}
        for stage_idx in sorted(int(index) for index in dp_metadata_list) or [0]:
            num_tokens_across_dp = _ffn_token_counts_across_ranks(
                self.connector,
                dp_metadata_list,
                stage_idx,
                fallback=self.max_num_tokens,
            )
            num_tokens = _ffn_token_count_for_rank(
                self.connector,
                num_tokens_across_dp,
            )
            input_ids_by_stage[stage_idx] = self.connector.recv_input_ids(
                num_tokens,
                ubatch_idx=stage_idx,
            )
        return input_ids_by_stage

    def _make_graph_key(
        self,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
    ) -> tuple:
        # ### PATCH START: isolate target graphs by speculative configuration.
        decoder_key = make_ffn_graph_key(
            dp_metadata_list,
            attention_size=int(self.connector.attn_size),
            ffn_size=int(self.connector.ffn_size),
            fallback=int(self.max_num_tokens),
        )
        graph_key = ("decoder", *self._speculative_graph_signature(), *decoder_key)
        # ### PATCH END: isolate target graphs by speculative configuration.
        return graph_key

    def _speculative_graph_signature(self) -> tuple[str, int, bool]:
        vllm_config = getattr(self, "vllm_config", None)
        speculative_config = getattr(vllm_config, "speculative_config", None)
        if speculative_config is None:
            return ("off", 0, False)
        return (
            str(getattr(speculative_config, "method", "unknown")),
            int(getattr(speculative_config, "num_speculative_tokens", 0)),
            bool(getattr(speculative_config, "enforce_eager", False)),
        )

    def _draft_uses_aclgraph(self) -> bool:
        speculative_config = getattr(self.vllm_config, "speculative_config", None)
        return bool(
            self.use_aclgraph
            and speculative_config is not None
            and getattr(speculative_config, "method", None) == "mtp"
            and not bool(getattr(speculative_config, "enforce_eager", False))
        )

    def _expected_mtp_wire_step(self, logical_step: int) -> int:
        """Return the step encoded by the active Attention wire protocol."""
        # The Graph-capable merged-draft path uses one stable header buffer for
        # every MTP iteration. Its wire step is therefore always zero; FFN's
        # ordered loop remains the source of truth for the logical step.
        return 0 if self._draft_uses_aclgraph() else logical_step

    def _make_mtp_graph_key(
        self,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
    ) -> tuple:
        peer_layout = make_mtp_ffn_graph_key(
            dp_metadata_list,
            attention_size=int(self.connector.attn_size),
            ffn_size=int(self.connector.ffn_size),
            fallback=int(self.max_num_tokens),
        )
        capture_sizes = tuple(
            int(size) for size in getattr(self, "cudagraph_batch_sizes", ())
        )
        if capture_sizes:
            # Attention dispatches each DP rank to the smallest captured batch
            # descriptor that can hold its live token count. Mirror that
            # per-peer padding here so FFN selects the graph whose HCCL tensor
            # slices have the shapes actually replayed by Attention.
            peer_layout = tuple(
                next(
                    (size for size in capture_sizes if size >= peer_tokens),
                    peer_tokens,
                )
                for peer_tokens in peer_layout
            )
        return ("mtp", *self._speculative_graph_signature(), peer_layout)

    def _replay_mtp_graph(
        self,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
    ) -> None:
        graph_key = self._make_mtp_graph_key(dp_metadata_list)
        graph_info = self._mtp_acl_graphs.get(graph_key)
        if graph_info is None:
            raise RuntimeError(
                "AFD NPU FFN has no captured MTP graph for the current peer "
                f"layout: key={graph_key}",
            )
        logger.debug(
            "AFD NPU FFN replaying MTP ACL graph; key=%s cached_graphs=%d",
            graph_key,
            len(self._mtp_acl_graphs),
        )
        graph_info["graph"].replay()

    def _ffn_forward(
        self,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
        aclgraph_runtime_mode: CUDAGraphMode | None = None,
        is_graph_capturing: bool = False,
        update_connector_state: bool = True,
        input_ids_by_stage: dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        if update_connector_state:
            assert self.connector.control_plane, (
                "Only DP metadata control plane supports ",
                "update_connector_state == True.",
            )
            self.connector.control_plane.update_state_from_dp_metadata(
                _make_dp_metadata_payload(
                    dp_metadata_list,
                    is_graph_capturing=is_graph_capturing,
                ),
            )
        num_stages = max(len(dp_metadata_list), 1)
        afd_metadata = AFDForwardContextMetadata(
            tokens_start_loc=[],
            requests_start_loc=[],
            stage_idx=0,
            connector=self.connector,
            tokens_lens=[],
            num_stages=num_stages,
        )
        stage_ids = tuple(sorted(int(stage_idx) for stage_idx in dp_metadata_list)) or (
            0,
        )
        layer_indices = tuple(_ffn_layer_indices(self))
        stage_runtime: dict[int, tuple[int, torch.Tensor]] = {}
        dp_size = int(self.vllm_config.parallel_config.data_parallel_size)
        for stage_idx in stage_ids:
            num_tokens_across_dp = _ffn_token_counts_across_ranks(
                self.connector,
                dp_metadata_list,
                stage_idx,
                fallback=self.max_num_tokens,
            )
            stage_runtime[stage_idx] = (
                _ffn_token_count_for_rank(
                    self.connector,
                    num_tokens_across_dp,
                ),
                _to_dp_level_token_counts(
                    num_tokens_across_dp,
                    dp_size=dp_size,
                ),
            )
        stage_forward_contexts = {}
        for stage_idx in stage_ids:
            num_tokens, dp_num_tokens_across_dp = stage_runtime[stage_idx]
            initial_input_ids = (
                input_ids_by_stage.get(stage_idx)
                if input_ids_by_stage is not None
                else None
            )
            with ascend_forward_context(
                vllm_config=self.vllm_config,
                afd_metadata=afd_metadata,
                model_instance=self.model,
                input_ids=initial_input_ids,
                num_tokens=num_tokens,
                num_tokens_across_dp=dp_num_tokens_across_dp,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
            ) as forward_context:
                stage_forward_contexts[stage_idx] = forward_context
        rank_ffn_output = None
        model_config = getattr(self, "model_config", self.vllm_config.model_config)
        num_hash_layers = int(getattr(model_config.hf_config, "num_hash_layers", 0))
        input_ids_cache = getattr(self, "_ffn_input_ids_cache", None)
        if input_ids_cache is None:
            input_ids_cache = self._ffn_input_ids_cache = {}
        input_ids_cache.clear()
        graph_stream_overlap = bool(
            getattr(self, "ffn_stream_overlap_enabled", False)
            and getattr(
                self.connector,
                "graph_u2_compute_overlap_enabled",
                True,
            )
            and len(stage_ids) > 1
            and (
                is_graph_capturing
                or (
                    bool(getattr(self, "use_aclgraph", False))
                    and bool(getattr(self.connector, "is_warmup", False))
                )
            )
        )
        eager_stream_overlap = bool(
            getattr(self, "ffn_stream_overlap_enabled", False)
            and len(stage_ids) > 1
            and not graph_stream_overlap
            and aclgraph_runtime_mode in (None, CUDAGraphMode.NONE)
        )
        graph_recv_stream_overlap = bool(
            graph_stream_overlap
            and getattr(
                self.connector,
                "graph_u2_ffn_recv_stream_enabled",
                False,
            )
        )
        graph_cross_layer_overlap = bool(
            graph_recv_stream_overlap
            and getattr(
                self.connector,
                "graph_u2_ffn_cross_layer_enabled",
                False,
            )
        )
        compute_stream_overlap = graph_stream_overlap or eager_stream_overlap
        if compute_stream_overlap:
            assert isinstance(self.connector, P2pHcclAFDConnector)
            assert self.ffn_compute_stream is not None
            assert self.ffn_send_stream is not None
        if eager_stream_overlap or graph_recv_stream_overlap:
            assert self.ffn_recv_stream is not None
        graph_recv_ready_event = None
        if graph_recv_stream_overlap:
            graph_recv_ready_event = self.ffn_graph_recv_ready_event
            assert graph_recv_ready_event is not None
            graph_recv_ready_event.record(torch.npu.current_stream())
        try:
            for layer_idx in layer_indices:
                graph_pending_send_events: list[Any] = []
                for stage_idx in stage_ids:
                    stage_input_ids = (
                        input_ids_by_stage.get(stage_idx)
                        if input_ids_by_stage is not None and layer_idx == 0
                        else None
                    )
                    recv_event = None
                    if eager_stream_overlap or graph_recv_stream_overlap:
                        previous_send_event = (
                            self.ffn_send_events[(layer_idx - 1, stage_idx)]
                            if layer_idx > 0
                            else graph_recv_ready_event
                        )
                        recv_event = self.ffn_recv_events[(layer_idx, stage_idx)]
                        payload, recv_event = self.connector.recv_attn_output_streamed(
                            ubatch_idx=stage_idx,
                            layer_idx=layer_idx,
                            max_num_tokens=self.max_num_tokens,
                            input_ids=stage_input_ids,
                            recv_stream=self.ffn_recv_stream,
                            wait_event=previous_send_event,
                            done_event=recv_event,
                        )
                    else:
                        payload = self.connector.recv_attn_output(
                            ubatch_idx=stage_idx,
                            layer_idx=layer_idx,
                            max_num_tokens=self.max_num_tokens,
                            input_ids=stage_input_ids,
                        )
                        if graph_stream_overlap:
                            recv_event = self.ffn_recv_events[(layer_idx, stage_idx)]
                            recv_event.record(torch.npu.current_stream())
                    if layer_idx == 0 and num_hash_layers > 0:
                        if payload.input_ids is None:
                            raise RuntimeError(
                                "DSV4 FFN layer 0 did not receive input_ids"
                            )
                        input_ids_cache[stage_idx] = payload.input_ids
                    elif payload.input_ids is not None:
                        raise RuntimeError("DSV4 FFN received input_ids after layer 0")
                    hash_input_ids = (
                        input_ids_cache.get(stage_idx)
                        if layer_idx < num_hash_layers
                        else None
                    )
                    forward_context = stage_forward_contexts[stage_idx]
                    with override_forward_context(forward_context):
                        context = payload.context
                        metadata = context.metadata
                        states = context.states
                        hidden_states = payload.hidden_states
                        metadata.layer_idx = layer_idx
                        metadata.stage_idx = stage_idx
                        forward_context.input_ids = hash_input_ids
                        forward_context.additional_kwargs["afd_metadata"] = metadata
                        assert states, "Context.states must not be None"
                        _set_moe_layer_index(forward_context, layer_idx)

                        compute_kwargs = {}
                        if hash_input_ids is not None:
                            compute_kwargs["input_ids"] = hash_input_ids
                        if compute_stream_overlap:
                            assert recv_event is not None
                            compute_event = self.ffn_compute_events[
                                (layer_idx, stage_idx)
                            ]
                            with torch.npu.stream(self.ffn_compute_stream):
                                recv_event.wait(self.ffn_compute_stream)
                                _record_npu_stream(
                                    hidden_states,
                                    self.ffn_compute_stream,
                                )
                                if hash_input_ids is not None:
                                    _record_npu_stream(
                                        hash_input_ids,
                                        self.ffn_compute_stream,
                                    )
                                rank_ffn_output = self.model.compute_ffn_output(
                                    hidden_states=hidden_states,
                                    layer_idx=layer_idx,
                                    **compute_kwargs,
                                )
                                compute_event.record(self.ffn_compute_stream)
                            send_event = self.ffn_send_events[(layer_idx, stage_idx)]
                            self.connector.send_ffn_output_streamed(
                                rank_ffn_output,
                                context,
                                ubatch_idx=stage_idx,
                                send_stream=self.ffn_send_stream,
                                wait_event=compute_event,
                                done_event=send_event,
                            )
                            if graph_stream_overlap:
                                graph_pending_send_events.append(send_event)
                        else:
                            rank_ffn_output = self.model.compute_ffn_output(
                                hidden_states=hidden_states,
                                layer_idx=layer_idx,
                                **compute_kwargs,
                            )
                            _send_ffn_output(
                                self.connector,
                                rank_ffn_output,
                                context,
                                stage_idx=stage_idx,
                            )
                if graph_pending_send_events and not graph_cross_layer_overlap:
                    current_stream = torch.npu.current_stream()
                    for send_event in graph_pending_send_events:
                        send_event.wait(current_stream)
            if eager_stream_overlap or graph_cross_layer_overlap:
                current_stream = torch.npu.current_stream()
                final_layer_idx = layer_indices[-1]
                for stage_idx in stage_ids:
                    self.ffn_send_events[(final_layer_idx, stage_idx)].wait(
                        current_stream
                    )
        finally:
            input_ids_cache.clear()
        return rank_ffn_output

    def _mtp_ffn_forward(
        self,
        stage_ids: list[int],
        *,
        header: Any | None = None,
        expected_speculative_step: int | None = None,
    ) -> torch.Tensor | None:
        if self.vllm_config.speculative_config is None:
            return None
        if stage_ids not in ([0], [0, 1]):
            raise RuntimeError(
                "DSV4 AFD MTP requires target decoder stages [0] or [0, 1]"
            )
        if self.mtp_ffn_model is None:
            raise RuntimeError("DSV4 AFD MTP FFN model is not loaded")

        # The target decoder may run U2, but the upstream proposer consumes the
        # merged target output and executes one full-batch MTP phase.
        stage_idx = 0
        if header is None:
            header = self.connector.recv_mtp_header(stage_idx=stage_idx)
        payload = self.connector.recv_attn_output(
            ubatch_idx=stage_idx,
            layer_idx=0,
            phase="mtp",
            speculative_step=header.speculative_step,
            num_tokens=header.num_tokens,
        )
        metadata = payload.context.metadata
        if metadata.phase != "mtp" or (
            expected_speculative_step is not None
            and metadata.speculative_step != expected_speculative_step
        ):
            raise RuntimeError(
                "DSV4 AFD FFN received an invalid MTP phase: "
                f"phase={metadata.phase!r}, "
                f"speculative_step={metadata.speculative_step}, "
                f"expected_step={expected_speculative_step}"
            )

        afd_metadata = AFDForwardContextMetadata(
            tokens_start_loc=[0],
            requests_start_loc=[0],
            stage_idx=stage_idx,
            connector=self.connector,
            tokens_lens=[header.num_tokens],
            num_stages=1,
            tokens_unpadded_lens=[header.num_tokens],
        )
        with ascend_forward_context(
            vllm_config=self.vllm_config,
            afd_metadata=afd_metadata,
            model_instance=self.mtp_ffn_model,
            num_tokens=header.num_tokens,
            num_tokens_across_dp=header.num_tokens_across_dp,
            is_draft_model=True,
        ) as forward_context:
            forward_context.additional_kwargs["afd_metadata"] = metadata
            _set_moe_layer_index(forward_context, 0)
            output = self.mtp_ffn_model.compute_ffn_output(
                hidden_states=payload.hidden_states,
                layer_idx=0,
            )
        _send_ffn_output(
            self.connector,
            output,
            payload.context,
            stage_idx=stage_idx,
        )
        return output

    def _ffn_forward_connector_driven(
        self,
    ) -> torch.Tensor | AFDF2ATransferPayload | None:
        stage_idx = 0
        rank_ffn_output = None
        connector = cast(CAMAsyncAFDConnector, self.connector)

        for _ in _ffn_layer_indices(self):
            work_item = connector.recv_ffn_work_item(
                stage_idx=stage_idx,
                max_num_tokens=self.max_num_tokens,
            )
            hidden_states = work_item.hidden_states
            metadata = work_item.context.metadata
            states = work_item.context.states
            if not isinstance(states, AFDAsyncTransferState):
                raise RuntimeError(
                    "CAM async FFN work item requires AFDAsyncTransferState",
                )
            layer_idx = work_item.layer_idx
            num_tokens = work_item.num_tokens
            afd_metadata = AFDForwardContextMetadata(
                tokens_start_loc=[0],
                requests_start_loc=[0],
                stage_idx=stage_idx,
                connector=self.connector,
                tokens_lens=[num_tokens],
                num_stages=1,
                tokens_unpadded_lens=[num_tokens],
            )

            with ascend_forward_context(
                vllm_config=self.vllm_config,
                afd_metadata=afd_metadata,
                model_instance=self.model,
                num_tokens=num_tokens,
            ) as forward_context:
                forward_context.dp_metadata = None
                forward_context.additional_kwargs["afd_metadata"] = metadata
                _set_moe_layer_index(forward_context, layer_idx)

                rank_ffn_output = self.model.compute_ffn_output(
                    hidden_states=hidden_states,
                    layer_idx=layer_idx,
                    group_list=states.group_list,
                    dynamic_scales=states.dynamic_scales,
                    expand_x_shared=states.expand_x_shared,
                    dynamic_scales_shared=states.dynamic_scales_shared,
                )
                rank_ffn_output = connector.send_ffn_work_item_output(
                    work_item,
                    rank_ffn_output,
                )
        return rank_ffn_output

    def capture_model(
        self,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] | None = None,
        is_warmup: bool = False,
        is_attn_graph_capturing: bool = True,
        input_ids_by_stage: dict[int, torch.Tensor] | None = None,
        connector_state_prepared: bool = False,
    ) -> int:
        if not self.use_aclgraph:
            return 0
        if dp_metadata_list is None:
            raise RuntimeError("AFD NPU FFN capture requires dp_metadata_list")
        if not connector_state_prepared:
            self._prepare_connector_step(
                dp_metadata_list,
                is_graph_capturing=is_attn_graph_capturing,
                is_warmup=is_warmup,
            )
        if input_ids_by_stage is None:
            input_ids_by_stage = self._receive_input_ids_before_model(dp_metadata_list)

        # ### PATCH START: mirror duplicate target graph replay.
        # MTP request-boundary U2 can fall back to the U1 graph key for the
        # smallest capture shape. The Attention wrapper replays that existing
        # graph instead of capturing it again, so FFN must replay its matching
        # decoder graph as well. Skipping the duplicate key would leave the
        # Attention graph's HCCL sends waiting for receives that never launch.
        graph_key = self._make_graph_key(dp_metadata_list)
        existing_graph_info = getattr(self, "_acl_graphs", {}).get(graph_key)
        if not is_warmup and existing_graph_info is not None:
            logger.debug(
                "AFD NPU FFN replaying existing graph during duplicate capture; key=%s",
                graph_key,
            )
            existing_graph_info["graph"].replay()
            if self._draft_uses_aclgraph():
                self._replay_mtp_graph(dp_metadata_list)
            return 0
        # ### PATCH END: mirror duplicate target graph replay.

        logger.debug(
            "AFD NPU FFN capture_model start; key=%s is_warmup=%s "
            "is_attn_graph_capturing=%s",
            graph_key,
            is_warmup,
            is_attn_graph_capturing,
        )
        start_free_memory = int(torch.npu.mem_get_info()[0])
        set_cudagraph_capturing_enabled(True)
        try:
            stage_ids = sorted(int(stage_idx) for stage_idx in dp_metadata_list) or [0]
            if is_warmup:
                self._ffn_forward(
                    dp_metadata_list=dp_metadata_list,
                    is_graph_capturing=False,
                    input_ids_by_stage=input_ids_by_stage,
                    update_connector_state=False,
                )
                # Warmup remains end-to-end; only the actual target capture
                # omits the eager draft phase.
                speculative_config = getattr(self, "speculative_config", None)
                if speculative_config is not None:
                    for speculative_step in range(
                        int(speculative_config.num_speculative_tokens)
                    ):
                        self._mtp_ffn_forward(
                            stage_ids,
                            expected_speculative_step=self._expected_mtp_wire_step(
                                speculative_step,
                            ),
                        )
            else:
                with graph_capture(device=self.device):
                    self._capture_graphs(
                        aclgraph_runtime_mode=CUDAGraphMode.FULL,
                        dp_metadata_list=dp_metadata_list,
                        is_attn_graph_capturing=is_attn_graph_capturing,
                        input_ids_by_stage=input_ids_by_stage,
                    )
                    if self._draft_uses_aclgraph():
                        self._capture_mtp_graphs(dp_metadata_list)
        finally:
            set_cudagraph_capturing_enabled(False)

        end_free_memory = int(torch.npu.mem_get_info()[0])
        graph_size = max(0, int(start_free_memory - end_free_memory))
        logger.debug(
            "AFD NPU FFN capture_model end; key=%s graph_size=%d",
            graph_key,
            graph_size,
        )
        return graph_size

    def _capture_mtp_graphs(
        self,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
    ) -> None:
        """Capture the merged MTP phase after the target decoder graph."""

        graph_key = self._make_mtp_graph_key(dp_metadata_list)
        existing = self._mtp_acl_graphs.get(graph_key)
        if existing is not None:
            existing["graph"].replay()
            return

        peer_layout = graph_key[-1]
        graph = torch.npu.NPUGraph()
        logger.debug("AFD NPU FFN capturing MTP ACL graph for key=%s", graph_key)
        assert self.speculative_config is not None
        # vLLM-Ascend's _run_merged_draft captures every draft iteration in one
        # graph. HCCL creates a distinct capture resource for each send/recv;
        # replaying a single-step FFN graph cannot match Attention's later
        # capture-time sends. Capture the same N exchanges on both sides.
        with torch.npu.graph(graph, pool=self.graph_pool):
            for _ in range(int(self.speculative_config.num_speculative_tokens)):
                header = self.connector.recv_mtp_header_for_graph(
                    stage_idx=0,
                    attention_peer_counts=peer_layout,
                )
                output = self._mtp_ffn_forward(
                    [0],
                    header=header,
                    expected_speculative_step=0,
                )
        self._mtp_acl_graphs[graph_key] = {
            "graph": graph,
            "output": output,
        }
        logger.debug("AFD NPU FFN captured MTP ACL graph for key=%s", graph_key)

    def _capture_graphs(
        self,
        *,
        aclgraph_runtime_mode: CUDAGraphMode,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
        is_attn_graph_capturing: bool = True,
        input_ids_by_stage: dict[int, torch.Tensor] | None = None,
    ) -> None:
        assert self.connector.control_plane, (
            "Only DP metadata control plane supports graph capturing."
        )
        graph_key = self._make_graph_key(dp_metadata_list)
        if graph_key in self._acl_graphs:
            logger.debug(
                "AFD NPU FFN ACL graph capture skipped for existing key=%s",
                graph_key,
            )
            return

        logger.debug("AFD NPU FFN capturing ACL graph for key=%s", graph_key)
        graph = torch.npu.NPUGraph()
        logger.debug("AFD NPU FFN created NPUGraph for key=%s", graph_key)
        with torch.npu.graph(graph, pool=self.graph_pool):
            logger.debug(
                "AFD NPU FFN entered NPU graph context for key=%s",
                graph_key,
            )
            output = self._ffn_forward(
                dp_metadata_list=dp_metadata_list,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                is_graph_capturing=is_attn_graph_capturing,
                update_connector_state=False,
                input_ids_by_stage=input_ids_by_stage,
            )
            logger.debug("AFD NPU FFN left _ffn_forward for key=%s", graph_key)
        self._acl_graphs[graph_key] = {
            "graph": graph,
            "output": output,
        }
        logger.debug("AFD NPU FFN captured ACL graph for key=%s", graph_key)

    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        raise RuntimeError("AFD NPU FFN runners do not sample tokens")

    def start_afd_profiler(self) -> None:
        if self._afd_profiler_window_active:
            logger.warning("AFD NPU FFN profiler start is already complete")
            return
        self.prof = create_afd_npu_profiler(
            "ffn",
            role_rank=self._afd_profiler_role_rank,
        )
        self._afd_profiler_window_active = True

    def stop_afd_profiler(self) -> None:
        profiler = self.prof
        stop_afd_npu_profiler(profiler)
        self.prof = None
        self._afd_profiler_window_active = False

    def shutdown(self) -> None:
        if self._is_shutdown:
            return
        self.stop_afd_profiler()
        if self.connector.is_initialized:
            self.connector.close()
        super().shutdown()
        self._is_shutdown = True


def _send_ffn_output(
    connector: AFDConnectorBase,
    ffn_output: torch.Tensor | AFDF2ATransferPayload,
    context: AFDTransferContext,
    *,
    stage_idx: int,
) -> None:
    if not isinstance(ffn_output, AFDF2ATransferPayload):
        connector.send_ffn_output(
            ffn_output,
            context,
            ubatch_idx=stage_idx,
        )
        return

    kwargs: dict[str, object] = {"ubatch_idx": stage_idx}
    if ffn_output.shared_output is not None:
        kwargs["expand_x_shared"] = ffn_output.shared_output
    connector.send_ffn_output(
        ffn_output.routed_output,
        context,
        **kwargs,
    )


def _record_npu_stream(tensor: torch.Tensor, stream) -> None:
    if tensor.device.type == "npu":
        tensor.record_stream(stream)


def _ffn_layer_indices(runner: AFDNPUFFNModelRunner) -> range | list[int]:
    num_layers = max(int(runner.num_layers or 0), 1)
    if not runner.afd_config.compute_gate_on_attention:
        return range(num_layers)
    hf_config = runner.model_config.hf_config
    return [
        layer_idx
        for layer_idx in range(num_layers)
        if _is_moe_layer(hf_config, layer_idx)
    ]


def _is_moe_layer(hf_config: object, layer_idx: int) -> bool:
    moe_layer_freq = getattr(hf_config, "moe_layer_freq", 1)
    return (
        hf_config.n_routed_experts is not None
        and layer_idx >= hf_config.first_k_dense_replace
        and layer_idx % moe_layer_freq == 0
    )


def _make_dp_metadata_payload(
    dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
    *,
    is_graph_capturing: bool = False,
    is_warmup: bool = False,
) -> AFDControlPayload:
    return AFDControlPayload(
        dp_metadata_list=dp_metadata_list,
        is_graph_capturing=is_graph_capturing,
        is_warmup=is_warmup,
    )


def _ffn_token_counts_across_ranks(
    connector: AFDConnectorBase,
    dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
    stage_idx: int,
    *,
    fallback: int,
) -> torch.Tensor:
    dp_metadata = dp_metadata_list.get(int(stage_idx))
    if dp_metadata is None:
        values = [max(1, int(fallback))] * int(connector.ffn_size)
    else:
        attention_counts = _to_int_list(dp_metadata.num_tokens_across_dp_cpu)
        # Expand DP-level counts to AFD-level counts when TP > 1.
        # With TP, attn_size = num_attention_ranks includes TP workers
        # but num_tokens_across_dp_cpu only has dp_size entries.
        # Each DP rank's token count is replicated tp_size times because
        # all TP workers within the same DP rank process the same tokens.
        if (
            len(attention_counts) < int(connector.attn_size)
            and int(connector.attn_size) % len(attention_counts) == 0
        ):
            tp_size = int(connector.attn_size) // len(attention_counts)
            attention_counts = [
                attention_counts[i // tp_size] for i in range(int(connector.attn_size))
            ]
        attention_size = int(connector.attn_size)
        ffn_size = int(connector.ffn_size)
        if len(attention_counts) >= attention_size and attention_size >= ffn_size:
            group_size = attention_size // ffn_size
            values = [
                max(1, sum(attention_counts[idx * group_size : (idx + 1) * group_size]))
                for idx in range(ffn_size)
            ]
        elif len(attention_counts) >= attention_size and ffn_size > attention_size:
            fanout = ffn_size // attention_size
            values = []
            for attention_tokens in attention_counts[:attention_size]:
                transport_tokens = max(int(attention_tokens), fanout)
                base, remainder = divmod(transport_tokens, fanout)
                values.extend(base + (offset < remainder) for offset in range(fanout))
        else:
            values = [max(1, int(fallback))] * ffn_size
    return torch.tensor(values, dtype=torch.int32, device="cpu")


def _ffn_token_count_for_rank(
    connector: AFDConnectorBase,
    num_tokens_across_dp: torch.Tensor,
) -> int:
    values = _to_int_list(num_tokens_across_dp)
    role_rank = int(connector.topology.role_rank)
    if role_rank >= len(values):
        return max(1, values[0] if values else 1)
    return max(1, int(values[role_rank]))


def _to_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item) for item in value.tolist()]


def _to_dp_level_token_counts(
    num_tokens_across_dp: torch.Tensor,
    *,
    dp_size: int,
) -> torch.Tensor:
    """Project AFD-level token counts back to DP-level for vLLM's forward context.

    ``num_tokens_across_dp`` has ``ffn_size`` entries (one per AFD role_rank).
    With TP > 1 each DP rank's count is replicated ``tp_size`` times, so the
    layout is ``[dp0, dp0, dp1, dp1, ...]``.  vLLM's ``DPMetadata.make()``
    expects exactly ``dp_size`` entries where ``[dp_rank] == batchsize``.
    """
    numel = len(num_tokens_across_dp)
    if numel == dp_size:
        return num_tokens_across_dp
    if dp_size <= 0 or numel % dp_size != 0:
        return num_tokens_across_dp
    tp_size = numel // dp_size
    # Take the first TP slot of each DP group (all TP slots are identical).
    indices = [dp_idx * tp_size for dp_idx in range(dp_size)]
    return num_tokens_across_dp[indices].contiguous()


def _use_npu_aclgraph(vllm_config: VllmConfig, runner: object) -> bool:
    inherited = bool(runner.use_aclgraph)
    if bool(vllm_config.model_config.enforce_eager):
        return False

    mode_name = vllm_config.compilation_config.cudagraph_mode.name
    return inherited or mode_name in {
        "FULL",
        "FULL_AND_PIECEWISE",
        "FULL_DECODE_ONLY",
        "PIECEWISE",
    }


__all__ = ["AFDNPUFFNModelRunner"]
