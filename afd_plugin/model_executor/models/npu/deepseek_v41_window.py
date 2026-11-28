# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 Window AFD wrapper over the Ascend baseline model."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm_ascend.ascend_forward_context import get_mrv2_in_profile_run

from afd_plugin.config import parse_afd_config
from afd_plugin.connectors import (
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.distributed import build_window_expert_layout, resolve_role_rank
from afd_plugin.model_executor.models import get_afd_metadata_from_forward_context
from afd_plugin.model_executor.models.deepseek_v2 import RemoteFFNProxy
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

try:
    from vllm_ascend.models.deepseek_v41 import model as native
    from vllm_ascend.models.deepseek_v41 import vl_model as native_vl
except ImportError as exc:  # pragma: no cover
    raise ImportError("DSV4.1 AFD requires the Ascend DSV4.1 model") from exc

_ATTENTION_ROLE = "attention"
_FFN_ROLE = "ffn"


class AFDDeepseekV41RoutedMoE(native.DeepseekV41MoE):
    """Routed-only V4.1 MoE with the Window rank's local expert slice."""

    def __init__(
        self,
        *,
        config: Any,
        parallel_config: Any,
        quant_config: Any,
        prefix: str,
        is_draft_layer: bool,
        expert_layout: Any,
    ) -> None:
        nn.Module.__init__(self)
        if is_draft_layer or expert_layout.is_shared:
            raise ValueError("Invalid routed Window FFN layout")
        self.layer_idx = int(prefix.split(".")[-2])
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.5)
        )
        self.swiglu_limit = getattr(config, "swiglu_limit", None)
        self.n_routed_experts = int(config.n_routed_experts)
        self.n_shared_experts = 0
        self.n_redundant_experts = 0
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_routed_experts
        self.n_local_physical_experts = int(expert_layout.local_expert_count)
        self.physical_expert_start = int(expert_layout.local_expert_start)
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )
        self.enable_eplb = False
        self.is_sequence_parallel = False
        self.gate = None
        self.shared_experts = None
        self.hash = False
        self.experts = native.FusedMoEFactory(
            shared_experts=None,
            gate=None,
            num_experts=self.n_local_physical_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            tp_size=1,
            dp_size=1,
            pcp_size=1,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
            routed_scaling_factor=self.routed_scaling_factor,
            swiglu_limit=self.swiglu_limit,
            enable_eplb=False,
            num_redundant_experts=0,
            is_sequence_parallel=False,
            n_shared_experts=0,
        )
        expert_map = torch.full(
            (self.n_routed_experts,), -1, dtype=torch.int32
        )
        start = self.physical_expert_start
        count = self.n_local_physical_experts
        expert_map[start : start + count] = torch.arange(count, dtype=torch.int32)
        # v0.30 keeps expert placement on both the MoE runner and its routed
        # weight owner. Window has an independent EP topology, so all cached
        # global/local metadata must describe the same checkpoint mapping.
        manager = self.experts.expert_map_manager
        manager.global_num_experts = self.n_routed_experts
        manager._expert_map = expert_map
        manager._local_num_experts = count
        owner = self.experts.routed_experts
        owner.global_num_experts = self.n_routed_experts
        owner.moe_config.num_experts = self.n_routed_experts
        owner._expert_map = expert_map
        owner.ascend_expert_map = expert_map


class AFDDeepseekV41SharedMoE(native.DeepseekV41MoE):
    """Shared-only V4.1 MLP placed on the first Window FFN rank."""

    def __init__(
        self,
        *,
        config: Any,
        parallel_config: Any,
        quant_config: Any,
        prefix: str,
        is_draft_layer: bool,
    ) -> None:
        nn.Module.__init__(self)
        if is_draft_layer:
            raise ValueError("DSV4.1 Window AFD does not support draft layers")
        self.layer_idx = int(prefix.split(".")[-2])
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.5)
        )
        self.n_routed_experts = int(config.n_routed_experts)
        self.n_shared_experts = int(config.n_shared_experts)
        self.experts = None
        self.gate = None
        self.shared_experts = native.DeepseekV41MLP(
            hidden_size=config.hidden_size,
            intermediate_size=(
                config.moe_intermediate_size * config.n_shared_experts
            ),
            hidden_act=config.hidden_act,
            swiglu_limit=getattr(config, "swiglu_limit", None),
            quant_config=quant_config,
            is_sequence_parallel=False,
            reduce_results=False,
            prefix=f"{prefix}.shared_experts",
        )

    def forward(self, hidden_states: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.shared_experts(hidden_states)


class AFDDeepseekV41RemoteMoE(RemoteFFNProxy):
    """Attention-owned V4.1 router with experts delegated to Window FFN."""

    def __init__(
        self,
        config: Any,
        parallel_config: Any,
        quant_config: Any = None,
        prefix: str = "",
        is_draft_layer: bool = False,
    ) -> None:
        del quant_config
        super().__init__(layer_idx=int(prefix.split(".")[-2]))
        if is_draft_layer:
            raise ValueError("DSV4.1 Window AFD does not support draft layers yet")
        if parallel_config.use_sequence_parallel_moe:
            raise ValueError("DSV4.1 Window eager execution does not support MoE SP")
        self.top_k = int(config.num_experts_per_tok)
        self.renormalize = bool(config.norm_topk_prob)
        self.scoring_func = getattr(config, "scoring_func", "softmax")
        self.num_expert_group = int(getattr(config, "n_group", 1))
        self.topk_group = int(getattr(config, "topk_group", 1))
        # Window applies routed scaling on the routed FFN rank before F2A;
        # keep the Attention-side combine weights normalized to avoid applying
        # the factor twice.
        self.routed_scaling_factor = 1.0
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.gate.precast_fp32_weight = True
        self.gate.bias_vl = None
        if int(getattr(config, "vision_n_layers", 0)) > 0:
            self.gate.bias_vl = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        if self.layer_idx < int(config.num_hash_layers):
            self.gate.tid2eid = nn.Parameter(
                torch.zeros(
                    config.vocab_size,
                    config.num_experts_per_tok,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            self.gate.e_score_correction_bias = None
        else:
            self.gate.tid2eid = None
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states_fp32: torch.Tensor | None = None,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        if already_sequence_parallel:
            raise RuntimeError("DSV4.1 Window eager execution does not support MoE SP")
        from afd_plugin.model_executor.models.npu.deepseek_v4_attention_gate import (
            compute_attention_gate_topk,
        )

        topk_weights, topk_ids = compute_attention_gate_topk(
            self,
            hidden_states,
            input_ids=input_ids,
            hidden_states_fp32=hidden_states_fp32,
        )
        afd_metadata = get_afd_metadata_from_forward_context()
        if afd_metadata is None:
            raise RuntimeError("DSV4.1 Window routing requires AFD metadata")
        forward_context = get_forward_context()
        # Startup profiling is a local Attention dummy forward.  vLLM disables
        # ubatching for this run, so a Window configured with two transport
        # stages has no matching stage-1 request on FFN.  Keep the local Gate in
        # the memory profile, but do not launch unmatched A2F/F2A communication.
        # This mirrors the verified DSV4 async profiling path.
        if (
            get_mrv2_in_profile_run()
            and int(getattr(afd_metadata.connector, "micro_batch_num", 1)) == 2
        ):
            return torch.zeros_like(hidden_states)
        stage_idx = int(
            getattr(forward_context, "ubatch_idx", afd_metadata.stage_idx)
        )
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=self.layer_idx,
                stage_idx=stage_idx,
                seq_len=int(hidden_states.shape[0]),
            )
        )
        afd_metadata.connector.send_attn_output(
            hidden_states,
            context,
            expert_ids=topk_ids,
            expert_scales=topk_weights,
        )
        # Native DBO starts stage 0 first.  Yield after A2F so stage 1 can
        # publish the same layer before Window batching/Combine waits for it.
        hidden_states = maybe_apply_dbo_yield(
            hidden_states,
            role="attention",
        )
        hidden_states = afd_metadata.connector.recv_ffn_output(
            ref_tensor=hidden_states,
            ubatch_idx=stage_idx,
            layer_idx=self.layer_idx,
        )
        return hidden_states


class AFDDeepseekV41Attention(native.DeepseekV41Attention):
    """Select stage-local index handoff buffers for U2."""

    @property
    def shared_state(self):
        stage_states = self.__dict__.get("_afd_stage_shared_states")
        if stage_states is None:
            return self._afd_shared_state

        context = get_forward_context()
        stage_idx = context.additional_kwargs.get("ubatch_idx", 0)
        return stage_states[stage_idx]

    @shared_state.setter
    def shared_state(self, state):
        # The native constructor and model initialization assign this field.
        self._afd_shared_state = state


class AFDDeepseekV41DecoderLayer(native.DeepseekV41DecoderLayer):
    """Native V4.1 layer with role-local Attention or MoE allocation."""

    attention_cls = AFDDeepseekV41Attention

    # Upstream source: vllm-ascend rfc/vllm_cann,
    # DeepseekV41DecoderLayer.__init__.
    # Patch reason: the baseline allocates Attention and MoE in every process.
    # Patch functionality: retain the baseline Attention/mHC/Engram state on A
    # and allocate only rank-local expert weights on connector-driven F.
    # Signature: matches upstream; no added parameters.
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: Any = None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        # ### PATCH START: select the AFD-owned half of the layer.
        afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = afd_config.role
        self.compute_gate_on_attention = afd_config.compute_gate_on_attention
        if not self.compute_gate_on_attention:
            raise ValueError("DSV4.1 Window AFD requires Attention-side routing")
        nn.Module.__init__(self)
        if config is None:
            config = native.normalize_deepseek_v41_config(
                vllm_config.model_config.hf_config
            )
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        self.layer_idx = int(prefix.split(".")[-1])

        # The FFN worker is connector-driven. It needs only its local expert
        # weights; allocating a fake Attention module still constructs cache,
        # mHC and Engram state and is therefore not a valid role split.
        if self.afd_role == _FFN_ROLE:
            role_rank = resolve_role_rank(vllm_config, afd_config)
            layout = build_window_expert_layout(
                routed_expert_num=int(config.n_routed_experts),
                ffn_size=afd_config.num_ffn_ranks,
                ffn_rank=role_rank,
            )
            self.mlp = (
                AFDDeepseekV41SharedMoE(
                    config=config,
                    parallel_config=parallel_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                    is_draft_layer=is_draft_layer,
                )
                if layout.is_shared
                else AFDDeepseekV41RoutedMoE(
                    config=config,
                    parallel_config=parallel_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                    is_draft_layer=is_draft_layer,
                    expert_layout=layout,
                )
            )
            return
        if self.afd_role != _ATTENTION_ROLE:
            raise ValueError(f"Unsupported AFD role {self.afd_role!r}")
        # ### PATCH END: select the AFD-owned half of the layer.

        # Keep the pinned DSV4.1 baseline state exactly; the sole behavioral
        # replacement is the native MoE module below.
        cache_config = vllm_config.cache_config
        self.hidden_size = config.hidden_size
        max_position_embeddings = config.rope_parameters[
            "original_max_position_embeddings"
        ]
        self.norm_eps = config.rms_norm_eps
        self.use_sequence_parallel_moe = parallel_config.use_sequence_parallel_moe
        self.enable_dsa_cp = native.enable_dsa_cp()
        self.self_attn = self.attention_cls(
            vllm_config=vllm_config,
            config=config,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            reduce_results=not self.use_sequence_parallel_moe,
            need_gather_q_kv=self.use_sequence_parallel_moe and self.enable_dsa_cp,
        )
        # ### PATCH START: replace only the native local MoE call.
        self.mlp = AFDDeepseekV41RemoteMoE(
            config=config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            is_draft_layer=is_draft_layer,
        )
        # ### PATCH END: replace only the native local MoE call.
        self.input_layernorm = native.RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.post_attention_layernorm = native.RMSNorm(
            config.hidden_size, eps=self.norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        )
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self._ds41_wheel_ops = (
            native._DS41_WHEEL_OPS and native.uses_a5_packed_cache()
        )
        self.use_sequence_parallel = parallel_config.use_sequence_parallel_moe
        if self.use_sequence_parallel:
            self.self_attn.wo_b.reduce_results = False
        has_engram = native._engram_enabled_for_runtime(config, vllm_config)
        self.engram = (
            native.AscendEngram(config, quant_config, f"{prefix}.engram")
            if has_engram
            and not is_draft_layer
            and self.layer_idx in config.engram_layer_ids
            else None
        )

class AFDDeepseekV41Model(native.DeepseekV41Model):
    """V4.1 backbone retaining baseline forward and pre_mix propagation."""

    decoder_layer_cls = AFDDeepseekV41DecoderLayer

    # Upstream source: vllm-ascend rfc/vllm_cann, DeepseekV41Model.__init__.
    # Patch reason: Window FFN never executes the full text-model forward.
    # Patch functionality: delegate Attention to the baseline constructor and
    # construct only layer-local expert containers on FFN.
    # Signature: matches upstream; no added parameters.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.vllm_config = vllm_config
        self._window_global_mxfp_weights = None
        if self.afd_config.role == _ATTENTION_ROLE:
            super().__init__(vllm_config=vllm_config, prefix=prefix)

            # ### PATCH START: isolate index handoff buffers by U2 stage.
            num_stages = int(
                self.afd_config.connector_extra_config.get("micro_batch_num", 1)
            )
            if num_stages > 1:
                base = self.shared_attention_state
                stage_states = [base]

                for _ in range(1, num_stages):
                    stage_states.append(
                        native.DeepseekV41SharedAttentionState(
                            topk_indices=torch.full_like(base.topk_indices, -1),
                            candidates=torch.full_like(base.candidates, -1),
                            candidate_lengths=(
                                None
                                if base.candidate_lengths is None
                                else torch.zeros_like(base.candidate_lengths)
                            ),
                            topk_lengths=(
                                None
                                if base.topk_lengths is None
                                else torch.zeros_like(base.topk_lengths)
                            ),
                        )
                    )

                self._afd_stage_shared_states = tuple(stage_states)
                for layer in self.layers:
                    if isinstance(layer, AFDDeepseekV41DecoderLayer):
                        layer.self_attn._afd_stage_shared_states = (
                            self._afd_stage_shared_states
                        )
            # ### PATCH END: isolate index handoff buffers by U2 stage.
            return

        # ### PATCH START: minimal connector-driven FFN model.
        # The FFN service is driven by Window batching and never executes the
        # text-model forward. Do not allocate embeddings, Attention/cache,
        # mHC, Engram, candidate buffers, final norm, or MTP state here.
        nn.Module.__init__(self)
        self.config = native.normalize_deepseek_v41_config(
            vllm_config.model_config.hf_config
        )
        self.has_engram = False
        self.start_layer, self.end_layer, self.layers = native.make_layers(
            self.config.num_hidden_layers,
            lambda prefix: AFDDeepseekV41DecoderLayer(
                vllm_config,
                prefix,
                is_draft_layer=False,
            ),
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = lambda *args, **kwargs: None
        # ### PATCH END: minimal connector-driven FFN model.

    def initialize_window_global_mxfp_weights(self) -> None:
        if self._window_global_mxfp_weights is not None:
            return
        from afd_plugin.model_executor.models.npu.deepseek_v4_window_ffn import (
            build_window_global_mxfp_weights,
        )

        local_layers = [
            layer
            for layer in self.layers
            if not isinstance(layer, native.PPMissingLayer)
        ]
        self._window_global_mxfp_weights = build_window_global_mxfp_weights(
            local_layers
        )

    def compute_window_global_ffn_output(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        dynamic_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._window_global_mxfp_weights is None:
            self.initialize_window_global_mxfp_weights()
        from afd_plugin.model_executor.models.npu.deepseek_v4_window_ffn import (
            compute_window_global_mxfp_ffn,
        )

        return compute_window_global_mxfp_ffn(
            hidden_states=hidden_states,
            compact_group_list=group_list,
            dynamic_scale=dynamic_scale,
            weights=self._window_global_mxfp_weights,
            output_dtype=self.vllm_config.model_config.dtype,
        )


class AFDDeepseekV41LLMForCausalLM(native.AscendDeepseekV41LLMForCausalLM):
    model_cls = AFDDeepseekV41Model
    afd_requires_input_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        self.vllm_config = vllm_config
        if self.afd_role == _ATTENTION_ROLE:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            return

        # Window FFN has no token embedding, LM head, logits processor or
        # native MoE communicator; the runner invokes the model's global GMM.
        nn.Module.__init__(self)
        self.config = native.normalize_deepseek_v41_config(
            vllm_config.model_config.hf_config
        )
        self.quant_config = vllm_config.quant_config
        self.requires_uncompiled_fallback = False
        self.model = self.model_cls(
            vllm_config=vllm_config,
            prefix=native.maybe_prefix(prefix, "model"),
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.num_moe_layers = self.config.num_hidden_layers
        self.num_redundant_experts = 0
        self.set_moe_parameters()
        self.moe_comm_methods = {}

    def set_moe_parameters(self) -> None:
        """Expose only routed Window experts to the native loader metadata."""
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        if self.afd_role == _FFN_ROLE:
            for layer in self.model.layers:
                if isinstance(layer, native.PPMissingLayer):
                    continue
                if isinstance(layer.mlp, AFDDeepseekV41RoutedMoE):
                    example_moe = layer.mlp
                    self.moe_mlp_layers.append(layer.mlp)
                    self.moe_layers.append(layer.mlp.experts)
        # Keep the baseline metadata contract for every role.  Attention and
        # shared-expert FFN ranks have no local routed MoE, in which case the
        # baseline helper initializes all MoE metadata fields to zero.
        self.extract_moe_parameters(example_moe)

    def initialize_window_global_mxfp_weights(self) -> None:
        self.model.initialize_window_global_mxfp_weights()

    def compute_window_global_ffn_output(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        dynamic_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.model.compute_window_global_ffn_output(
            hidden_states, group_list, dynamic_scale
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ffn_kind = None
        if self.afd_role == _FFN_ROLE:
            role_rank = resolve_role_rank(self.vllm_config, self.afd_config)
            ffn_kind = build_window_expert_layout(
                routed_expert_num=int(self.config.n_routed_experts),
                ffn_size=self.afd_config.num_ffn_ranks,
                ffn_rank=role_rank,
            ).kind
        loaded = super().load_weights(
            _iter_role_weights(
                weights,
                self.afd_role,
                ffn_kind=ffn_kind,
                num_hidden_layers=int(self.config.num_hidden_layers),
                routed_expert_num=int(self.config.n_routed_experts),
            )
        )
        if self.afd_role == _FFN_ROLE and ffn_kind == "routed":
            self.initialize_window_global_mxfp_weights()
        return loaded


class AFDDeepseekV41WindowForCausalLM(native_vl.AscendDeepseekV41ForCausalLM):
    language_model_cls = AFDDeepseekV41LLMForCausalLM
    afd_requires_input_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        afd_config = parse_afd_config(vllm_config, validate=False)
        if afd_config.role == _ATTENTION_ROLE:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            return

        # The FFN service never consumes multimodal inputs. Avoid constructing
        # the visual tower merely because the checkpoint supports images.
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.vision = None
        self.aligner = None
        self.image_start = None
        self.image_end = None
        self.image_newline = None
        self.language_model = self.language_model_cls(
            vllm_config=vllm_config,
            prefix=native.maybe_prefix(prefix, "language_model"),
        )
        self.requires_uncompiled_fallback = False
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.moe_comm_methods = {}

    def initialize_window_global_mxfp_weights(self) -> None:
        self.language_model.initialize_window_global_mxfp_weights()

    def compute_window_global_ffn_output(
        self,
        hidden_states: torch.Tensor,
        group_list: torch.Tensor,
        dynamic_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.language_model.compute_window_global_ffn_output(
            hidden_states, group_list, dynamic_scale
        )


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    role: str,
    *,
    ffn_kind: str | None = None,
    num_hidden_layers: int,
    routed_expert_num: int,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield only checkpoint tensors for modules allocated by one role."""
    for name, tensor in weights:
        parts = name.split(".")
        try:
            marker = parts.index("layers")
            layer_idx = int(parts[marker + 1])
        except (ValueError, IndexError):
            if role == _ATTENTION_ROLE:
                yield name, tensor
            continue
        if layer_idx >= num_hidden_layers:
            continue
        stage = parts[marker + 2] if len(parts) > marker + 2 else ""
        remainder = parts[marker + 3 :]
        if (
            stage in {"ffn", "mlp"}
            and len(remainder) > 1
            and remainder[0] == "experts"
            and int(remainder[1]) >= routed_expert_num
        ):
            continue
        if stage in {"attn", "self_attn", "engram"}:
            owners = {_ATTENTION_ROLE}
        elif stage in {"ffn", "mlp"}:
            owners = {_FFN_ROLE}
            if remainder and remainder[0] == "gate":
                owners = {_ATTENTION_ROLE}
        else:
            # mHC, norms and any future layer-level state execute only on the
            # Attention service. The connector-driven FFN model does not
            # allocate these parameters.
            owners = {_ATTENTION_ROLE}
        if role not in owners:
            continue
        if role == _FFN_ROLE and ffn_kind is not None and stage in {"ffn", "mlp"}:
            is_shared = bool(remainder and remainder[0] == "shared_experts")
            if ffn_kind == "shared" and not is_shared:
                continue
            if ffn_kind == "routed" and is_shared:
                continue
        if role in owners:
            if role == _ATTENTION_ROLE and tuple(remainder) in {
                ("gate", "weight"),
                ("gate", "bias"),
                ("gate", "bias_vl"),
            }:
                tensor = tensor[:routed_expert_num]
            yield name, tensor


__all__ = [
    "AFDDeepseekV41DecoderLayer",
    "AFDDeepseekV41WindowForCausalLM",
    "AFDDeepseekV41LLMForCausalLM",
    "AFDDeepseekV41Model",
    "AFDDeepseekV41RemoteMoE",
]
