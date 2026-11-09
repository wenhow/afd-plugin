# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""V4.1 role construction and MoE boundary; production entry remains guarded."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import make_layers, maybe_prefix
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.models.deepseek_v41 import model as native
from vllm_ascend.models.deepseek_v41 import vl_model as native_wrapper
from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

from afd_plugin.compat.npu.runtime_config import validate_deepseek_v41_speculation
from afd_plugin.config import parse_afd_config
from afd_plugin.connectors import AFDTransferContext, AFDTransferMetadata
from afd_plugin.model_executor.models.forward_context import (
    get_afd_metadata_from_forward_context,
)


def _is_ffn_checkpoint_weight(name: str) -> bool:
    path = name.removeprefix("model.").split(".")
    return (
        len(path) >= 4
        and path[0] == "layers"
        and path[1].isdecimal()
        and path[2] in ("ffn", "mlp")
    )


@dataclass(frozen=True, slots=True)
class DeepseekV41FFNContinuation:
    """Attention-owned HC state held until the matching remote MoE returns."""

    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    pre_mix: torch.Tensor


class AFDDeepseekV41RemoteMoE(nn.Module):
    """Transport the native MoE inputs while Attention retains HC state."""

    def __init__(self, *, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

    # Upstream: DeepseekV41MoE.forward @ 7e37e29a.
    # Patch reason: the FFN role owns the complete gate/shared/routed MoE.
    # Patch functionality: preserve all native inputs and the transfer context.
    # Signature: matches upstream; no added parameters.
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        hidden_states_fp32: torch.Tensor | None = None,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        # ### PATCH START: remote complete MoE
        if already_sequence_parallel:
            raise ValueError("V4.1 HCCL proxy requires non-SP input")
        if input_ids is None or hidden_states_fp32 is None:
            raise ValueError("V4.1 HCCL proxy requires raw IDs and independent FP32")
        metadata = get_afd_metadata_from_forward_context()
        if metadata is None:
            raise RuntimeError("V4.1 HCCL proxy requires AFD forward metadata")
        if metadata.stage_idx != 0 or metadata.num_stages != 1:
            raise ValueError("V4.1 HCCL proxy supports only stage 0 and U1")
        valid_tokens = metadata.tokens_unpadded_lens[0]
        input_ids = input_ids[: hidden_states.shape[0]]
        if valid_tokens < hidden_states.shape[0]:
            # Native V2 reuses its raw-token/history buffer. Sanitize only a
            # private transfer view; padded IDs must never modify that state.
            input_ids = input_ids.clone()
            input_ids[valid_tokens:] = 0
        context = AFDTransferContext(
            metadata=AFDTransferMetadata.create_attention_metadata(
                layer_idx=self.layer_idx,
                stage_idx=metadata.stage_idx,
                seq_len=hidden_states.shape[0],
            )
        )
        output = hidden_states.contiguous()
        metadata.connector.send_attn_output(
            output,
            context,
            hidden_states_fp32=hidden_states_fp32.contiguous(),
            input_ids=input_ids.contiguous(),
        )
        return metadata.connector.recv_ffn_output(
            ref_tensor=output,
            ubatch_idx=metadata.stage_idx,
            context=context,
        )
        # ### PATCH END: remote complete MoE


class AFDDeepseekV41AttentionDecoderLayer(native.DeepseekV41DecoderLayer):
    """Retain native Attention, HC and Engram without constructing local MoE."""

    # Upstream: DeepseekV41DecoderLayer.__init__, model.py:783-858 @ 7e37e29a.
    # Patch reason: native construction allocates Attention and complete MoE.
    # Patch functionality: construct the Attention role and parameter-free proxy.
    # Only native same-checkpoint DSpark may accompany the target. Its complete
    # draft uses the native class, never this target-only decoder constructor.
    # Signature: matches upstream; no added parameters.
    # Removal plan: use the native MoE factory when an injection hook exists.
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config=None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        # ### PATCH START: initialize Attention role
        afd_config = parse_afd_config(vllm_config, validate=False)
        if afd_config.role != "attention" or afd_config.compute_gate_on_attention:
            raise ValueError(
                "V4.1 Attention construction requires Attention and F gate"
            )
        if is_draft_layer:
            raise ValueError("V4.1 Attention construction requires target layers")
        validate_deepseek_v41_speculation(vllm_config)
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise ValueError("V4.1 Attention construction requires non-SP execution")
        nn.Module.__init__(self)
        # ### PATCH END: initialize Attention role

        if config is None:
            config = native.normalize_deepseek_v41_config(
                vllm_config.model_config.hf_config
            )
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = config.rope_parameters[
            "original_max_position_embeddings"
        ]
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps
        self.use_sequence_parallel_moe = parallel_config.use_sequence_parallel_moe
        self.enable_dsa_cp = native.enable_dsa_cp()

        attn_cls = self.attention_cls

        self.self_attn = attn_cls(
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

        # ### PATCH START: parameter-free remote MoE
        self.mlp = AFDDeepseekV41RemoteMoE(layer_idx=layer_idx)
        # ### PATCH END: parameter-free remote MoE
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
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self._ds41_wheel_ops = native._DS41_WHEEL_OPS and native.uses_a5_packed_cache()
        self.use_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        if self.use_sequence_parallel:
            self.self_attn.wo_b.reduce_results = False
        has_engram = native._engram_enabled_for_runtime(config, vllm_config)
        self.engram: native.AscendEngram | None
        if (
            has_engram
            and not is_draft_layer
            and self.layer_idx in config.engram_layer_ids
        ):
            self.engram = native.AscendEngram(config, quant_config, f"{prefix}.engram")
        else:
            self.engram = None

    # Upstream: DeepseekV41DecoderLayer.forward, model.py:930-955 @ 7e37e29a.
    # Patch reason: layer-major U2 pauses each decoder at the remote MoE seam.
    # Patch functionality: retain native HC/Attention/norm and return its exact
    # FP32 router input plus the stage-owned delayed HC continuation.
    # New API: positions/hidden_states/pre_mix/llama_4_scaling have the native
    # meanings; no MoE is called here. The inherited forward remains unchanged.
    # Removal: use an upstream decoder continuation hook when one is available.
    def prepare_remote_ffn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        pre_mix: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, DeepseekV41FFNContinuation]:
        # ### PATCH START: synchronous non-SP continuation admission
        if self.use_sequence_parallel:
            raise ValueError("V4.1 staged continuation requires non-SP execution")
        # ### PATCH END: synchronous non-SP continuation admission
        residual = hidden_states
        x, attn_post, attn_comb, attn_pre = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            pre_mix,
        )
        x = self.input_layernorm(x)
        # ### PATCH START: SP branches excluded by admission above
        x = self.self_attn(positions, x, llama_4_scaling)
        # ### PATCH END: SP branches excluded by admission above
        hidden_states = self.hc_post(x, residual, attn_post, attn_comb)

        residual = hidden_states
        x, ffn_post, ffn_comb, ffn_pre = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            attn_pre,
        )
        x, x_fp32 = self.rms_norm_cast(x)
        # ### PATCH START: hold native HC state across remote MoE
        return (
            x,
            x_fp32,
            DeepseekV41FFNContinuation(residual, ffn_post, ffn_comb, ffn_pre),
        )
        # ### PATCH END: hold native HC state across remote MoE

    # Upstream: DeepseekV41DecoderLayer.forward, model.py:962-963 @ 7e37e29a.
    # Patch reason: resume only the stage whose remote MoE output is ready.
    # Patch functionality: finish native HC post and preserve ffn_pre for the
    # next decoder and final collapse. New API accepts the continuation above.
    # Removal: use the same upstream continuation hook as prepare_remote_ffn.
    def complete_remote_ffn(
        self,
        ffn_output: torch.Tensor,
        continuation: DeepseekV41FFNContinuation,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ### PATCH START: resume the matching stage's HC state
        hidden_states = self.hc_post(
            ffn_output, continuation.residual, continuation.post, continuation.comb
        )
        return hidden_states, continuation.pre_mix
        # ### PATCH END: resume the matching stage's HC state


class AFDDeepseekV41AttentionModel(native.DeepseekV41Model):
    decoder_layer_cls = AFDDeepseekV41AttentionDecoderLayer


class AFDDeepseekV41AttentionLLMForCausalLM(native.AscendDeepseekV41LLMForCausalLM):
    model_cls = AFDDeepseekV41AttentionModel

    # Upstream: AscendDeepseekV41LLMForCausalLM.load_weights @ 7e37e29a.
    # Patch reason: parameter-free proxies do not own FFN checkpoint weights.
    # Patch functionality: select complete Attention namespaces before loading.
    # Delegation exception: retain native packed/expert/quantization loading.
    # Signature: matches upstream; no added parameters.
    # Removal plan: use the native role filter when one becomes available.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ### PATCH START: exclude FFN checkpoint paths
        attention_weights = (
            (name, tensor)
            for name, tensor in weights
            if not _is_ffn_checkpoint_weight(name)
        )
        return super().load_weights(attention_weights)
        # ### PATCH END: exclude FFN checkpoint paths


class AFDDeepseekV41AttentionForCausalLM(native_wrapper.AscendDeepseekV41ForCausalLM):
    language_model_cls = AFDDeepseekV41AttentionLLMForCausalLM


class AFDDeepseekV41FFNLayer(nn.Module):
    """Keep the native loader's layer/MoE parameter namespace."""

    def __init__(
        self, vllm_config: VllmConfig, config: PretrainedConfig, prefix: str
    ) -> None:
        super().__init__()
        self.mlp = native.DeepseekV41MoE(
            config=config,
            parallel_config=vllm_config.parallel_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.mlp",
        )


class AFDDeepseekV41FFNModel(nn.Module):
    """Construct local MoE layers without invoking the full model constructor."""

    def __init__(
        self, vllm_config: VllmConfig, config: PretrainedConfig, prefix: str
    ) -> None:
        super().__init__()
        self.has_engram = False
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: AFDDeepseekV41FFNLayer(vllm_config, config, prefix),
            prefix=f"{prefix}.layers",
        )


class AFDDeepseekV41FFNForCausalLM(native.AscendDeepseekV41LLMForCausalLM):
    """Reuse native MoE loading while allocating only the FFN role's modules."""

    # Upstream: vllm_ascend/models/deepseek_v41/model.py @ 7e37e29a.
    # Patch reason: the native LLM constructor allocates A-owned modules/state.
    # Patch functionality: retain native config/MoE metadata, construct only F.
    # Signature: matches the native constructor, including no return annotation.
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: F-only initialization
        afd_config = parse_afd_config(vllm_config, validate=False)
        if afd_config.role != "ffn" or afd_config.compute_gate_on_attention:
            raise ValueError("V4.1 FFN construction requires the FFN role and gate")
        if vllm_config.engram_config is not None:
            raise ValueError("V4.1 FFN construction requires Engram to be disabled")
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "V4.1 FFN construction requires speculation to be disabled"
            )
        nn.Module.__init__(self)
        # ### PATCH END: F-only initialization
        config = native.normalize_deepseek_v41_config(
            vllm_config.model_config.hf_config
        )
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # ### PATCH START: omit Attention resources
        self.model = AFDDeepseekV41FFNModel(
            vllm_config, config, maybe_prefix(prefix, "model")
        )
        # ### PATCH END: omit Attention resources
        self.num_moe_layers = self.config.num_hidden_layers
        self.set_moe_parameters()
        self.moe_comm_methods = {
            kind: get_moe_comm_method(kind) for kind in MoECommType
        }

    # Upstream: AscendDeepseekV41LLMForCausalLM.load_weights @ 7e37e29a.
    # Patch reason: the checkpoint also contains A-owned and draft parameters.
    # Patch functionality: filter complete FFN paths once, retain names/tensors.
    # Delegation exception: preserve the large native packed/expert/quant loader
    # rather than copying its weight mapping and sharding loops into the plugin.
    # Signature: matches upstream; no added parameters.
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ### PATCH START: select FFN checkpoint paths
        def ffn_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if _is_ffn_checkpoint_weight(name):
                    yield name, tensor

        return super().load_weights(ffn_weights())
        # ### PATCH END: select FFN checkpoint paths

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(range(self.model.start_layer, self.model.end_layer))

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        *,
        input_ids: torch.Tensor,
        hidden_states_fp32: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        """Forward native MoE inputs without reconstructing router precision."""
        if not self.model.start_layer <= layer_idx < self.model.end_layer:
            raise ValueError(f"FFN layer {layer_idx} is not owned by this PP rank")
        if hidden_states_fp32.dtype != torch.float32:
            raise ValueError("V4.1 router input must be FP32")
        if hidden_states_fp32.shape != hidden_states.shape:
            raise ValueError("V4.1 router input and activation shapes must match")
        return self.model.layers[layer_idx].mlp(
            hidden_states,
            input_ids=input_ids,
            hidden_states_fp32=hidden_states_fp32,
            already_sequence_parallel=already_sequence_parallel,
        )
