# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek-V4 AFD model wrapper for the pinned Ascend runtime."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, override_forward_context
from vllm_ascend.models import deepseek_v4 as native
from vllm_ascend.models import deepseek_v4_mtp as native_mtp

from afd_plugin.config import parse_afd_config
from afd_plugin.model_executor.models.deepseek_v2 import (
    AFDRemoteFFNTransfer,
    RemoteFFNProxy,
)

_ATTENTION_ROLE = frozenset(("attention",))
_FFN_ROLE = frozenset(("ffn",))
_NO_ROLE = frozenset()


def _checkpoint_weight_roles(name: str) -> frozenset[str]:
    """Return the DSV4 AFD role that owns one raw checkpoint key."""
    normalized = name.removeprefix("model.")
    if normalized.startswith("mtp."):
        return _NO_ROLE

    parts = normalized.split(".")
    if len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit():
        return _FFN_ROLE if parts[2] == "ffn" else _ATTENTION_ROLE
    return _ATTENTION_ROLE


def _mtp_checkpoint_weight_roles(name: str) -> frozenset[str]:
    """Return the strict AFD owner of one raw DSV4 MTP checkpoint key."""
    normalized = name.removeprefix("model.")
    parts = normalized.split(".")
    if len(parts) < 3 or parts[0] != "mtp" or not parts[1].isdigit():
        return _NO_ROLE
    return _FFN_ROLE if parts[2] == "ffn" else _ATTENTION_ROLE


def _iter_mtp_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume one MTP checkpoint iterator once for the active AFD role."""
    for name, loaded_weight in weights:
        if role in _mtp_checkpoint_weight_roles(name):
            yield name, loaded_weight


def _uses_mtp(vllm_config: VllmConfig) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    return (
        speculative_config is not None
        and getattr(speculative_config, "method", None) == "mtp"
    )


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume a checkpoint iterator once and retain the active role's keys."""
    for name, loaded_weight in weights:
        if role in _checkpoint_weight_roles(name):
            yield name, loaded_weight


class AFDDeepseekV4RemoteMoEProxy(RemoteFFNProxy):
    """Parameter-free DSV4 MoE stage executed by the remote FFN role."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._send_and_receive(hidden_states)

    def dispatch_remote_ffn(
        self,
        hidden_states: torch.Tensor,
        **send_kwargs: torch.Tensor,
    ) -> AFDRemoteFFNTransfer:
        if self.phase == "decoder" and self.layer_idx == 0:
            input_ids = getattr(get_forward_context(), "input_ids", None)
            if input_ids is None:
                raise RuntimeError(
                    "DSV4 layer 0 requires input_ids in the forward context"
                )
            send_kwargs["input_ids"] = input_ids
        return super().dispatch_remote_ffn(hidden_states, **send_kwargs)


class AFDDeepseekV4DecoderLayer(native.DeepseekV2DecoderLayer):
    """DSV4 decoder layer that constructs only the active AFD role."""

    # Patch reason: native DSV4 constructs Attention, HC, and MoE for every role.
    # Patch functionality: construct Attention/HC or MoE, never both.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: 80d8c194f7584b17fe08065ea99a130916f6b0e7
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: native.DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
        attn_cls: type[nn.Module] | None = None,
    ) -> None:
        # ### PATCH START: role-selective DSV4 construction.
        nn.Module.__init__(self)
        afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = afd_config.role

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.split(sep=".")[-1])
        self.norm_eps = config.rms_norm_eps
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        if self.afd_role == "attention":
            max_position_embeddings = config.rope_parameters[
                "original_max_position_embeddings"
            ]
            attention_class = attn_cls or native.DeepseekV4Attention
            self.self_attn = attention_class(
                vllm_config=vllm_config,
                config=config,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
            )
            self.mlp = AFDDeepseekV4RemoteMoEProxy(
                layer_idx=self.layer_idx,
                phase="mtp" if is_draft_layer else "decoder",
            )
            self.input_layernorm = native.RMSNorm(
                config.hidden_size,
                eps=self.norm_eps,
            )
            self.post_attention_layernorm = native.RMSNorm(
                config.hidden_size,
                eps=self.norm_eps,
            )
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
        else:
            self.self_attn = native.PPMissingLayer()
            self.mlp = native.DeepseekV4MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                is_draft_layer=is_draft_layer,
            )
        # ### PATCH END

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.npu_hc_pre_v2(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.norm_eps,
            self.hc_eps,
        )

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(dim=0),
            residual.unsqueeze(dim=0),
            post.unsqueeze(dim=0),
            comb.unsqueeze(dim=0),
        )
        return output.squeeze(dim=0)

    # Patch reason: native forward invokes the locally constructed MoE.
    # Patch functionality: the Attention role invokes a parameter-free remote proxy.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: 80d8c194f7584b17fe08065ea99a130916f6b0e7
    # Patch reason: upstream forward always runs locally owned draft MoE.
    # Patch functionality: reject FFN full execution and identify the remote step.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ffn_output, continuation = self.forward_attention_to_remote_ffn(
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        )
        hidden_states = self.complete_remote_ffn(ffn_output, continuation)
        return hidden_states, continuation[0]

    def forward_attention_to_remote_ffn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Run through the remote MoE receive, deferring FFN HC post."""
        hidden_states, continuation = self.forward_attention_to_remote_ffn_input(
            positions,
            hidden_states,
            residual,
            llama_4_scaling,
        )
        if not isinstance(self.mlp, AFDDeepseekV4RemoteMoEProxy):
            return self.mlp(hidden_states), continuation
        transfer = self.dispatch_remote_ffn(hidden_states)
        return self.receive_remote_ffn(transfer), continuation

    def forward_attention_to_remote_ffn_input(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Compute the Attention-side input consumed by the remote MoE."""
        # ### PATCH START: reject accidental FFN full-model execution.
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 FFN layers are connector-driven")
        # ### PATCH END
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
        )
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            llama_4_scaling=llama_4_scaling,
        )
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        return hidden_states, (residual, post, comb)

    def dispatch_remote_ffn(
        self,
        hidden_states: torch.Tensor,
    ) -> AFDRemoteFFNTransfer:
        if not isinstance(self.mlp, AFDDeepseekV4RemoteMoEProxy):
            raise RuntimeError("DSV4 Attention layer requires its remote MoE proxy")
        return self.mlp.dispatch_remote_ffn(hidden_states)

    def receive_remote_ffn(
        self,
        transfer: AFDRemoteFFNTransfer,
    ) -> torch.Tensor:
        if not isinstance(self.mlp, AFDDeepseekV4RemoteMoEProxy):
            raise RuntimeError("DSV4 Attention layer requires its remote MoE proxy")
        return self.mlp.receive_remote_ffn(transfer, layer_idx=self.layer_idx)

    def complete_remote_ffn(
        self,
        ffn_output: torch.Tensor,
        continuation: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Complete the FFN HC post after the remote output becomes visible."""
        residual, post, comb = continuation
        return self.hc_post(ffn_output, residual, post, comb)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.afd_role != "ffn":
            raise RuntimeError("DSV4 Attention role does not own local MoE weights")
        return self.mlp(hidden_states, input_ids=input_ids)


@native.support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class AFDDeepseekV4Model(native.DeepseekV4Model):
    """Role-aware DSV4 model with a remote-MoE Attention forward path."""

    fall_back_to_pt_during_load = False

    # Patch reason: native DSV4 allocates embedding, every full layer, and head HC.
    # Patch functionality: build role-owned modules and omit the disabled MTP buffer.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: 80d8c194f7584b17fe08065ea99a130916f6b0e7
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: initialize role-aware storage without native allocation.
        nn.Module.__init__(self)
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        # ### PATCH END

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = native.current_platform.device_type
        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")
        # vLLM 0.23 does not initialize this field for the Ascend DSV4 model,
        # while the newer forward path checks it even when Eagle/MTP is off.
        self.aux_hidden_state_layers: tuple[int, ...] = ()

        # ### PATCH START: DSA scratch data belongs only to Attention.
        if self.is_v32 and self.afd_role == "attention":
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.topk_indices_buffer = None
        # ### PATCH END

        if self.afd_role == "attention" and native.get_pp_group().is_first_rank:
            self.embed_tokens = native.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = native.PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            lambda prefix: AFDDeepseekV4DecoderLayer(
                vllm_config,
                prefix,
                topk_indices_buffer=self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if self.afd_role == "attention" and native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()

        self.hc_mult = config.hc_mult

        def make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> native.IntermediateTensors:
            return native.IntermediateTensors(
                {
                    "hidden_states": torch.zeros(
                        (batch_size, self.hc_mult, config.hidden_size),
                        dtype=dtype,
                        device=device,
                    )
                }
            )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors

        # ### PATCH START: head HC and normalization belong only to Attention.
        if self.afd_role == "attention":
            self.norm_eps = config.rms_norm_eps
            self.hc_eps = config.hc_eps
            hc_dim = self.hc_mult * config.hidden_size
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, hc_dim, dtype=torch.float32)
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(self.hc_mult, dtype=torch.float32)
            )
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.mtp_enabled = self.afd_role == "attention" and _uses_mtp(vllm_config)
        if self.mtp_enabled:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                hc_dim,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
        # ### PATCH END

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 FFN role does not own token embeddings")
        return self.embed_tokens(input_ids)

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        flattened = x.flatten(1).float()
        rsqrt = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = torch.nn.functional.linear(flattened, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        output = torch.sum(pre.unsqueeze(-1) * flattened.view(shape), dim=1)
        return output.to(dtype)

    # Patch reason: native forward runs native full layers and always updates MTP.
    # Patch functionality: run role-aware layers and omit disabled MTP state.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: 80d8c194f7584b17fe08065ea99a130916f6b0e7
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: native.IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | native.IntermediateTensors:
        # ### PATCH START: only Attention runs the complete DSV4 model.
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 FFN model execution is connector-driven")
        # ### PATCH END
        if native.get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
        else:
            if intermediate_tensors is None:
                raise RuntimeError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        llama_4_scaling = None
        aux_hidden_states: list[torch.Tensor] = []
        if native.get_pp_group().is_first_rank:
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, _ = layer(
                positions,
                hidden_states,
                None,
                llama_4_scaling,
            )
            if layer.layer_idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states.mean(dim=1))

        if self.mtp_enabled:
            mtp_hidden = hidden_states.flatten(1)
            self._mtp_hidden_buffer[: mtp_hidden.shape[0]].copy_(mtp_hidden)

        if not native.get_pp_group().is_last_rank:
            return native.IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
        )
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def forward_ubatches_layer_major(
        self,
        ubatch_metadata: list[Any],
    ) -> list[Any]:
        """Run HCCL U2 in layer-major, stage-minor order on one host thread."""
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 FFN model execution is connector-driven")
        if len(ubatch_metadata) != 2:
            raise RuntimeError(
                "DSV4 layer-major execution requires exactly two stages; "
                f"got {len(ubatch_metadata)}"
            )

        stage_contexts = [item.context.forward_context for item in ubatch_metadata]
        connectors = []
        for stage_idx, (item, forward_context) in enumerate(
            zip(ubatch_metadata, stage_contexts, strict=True)
        ):
            afd_metadata = (forward_context.additional_kwargs or {}).get("afd_metadata")
            connector = getattr(afd_metadata, "connector", None)
            if connector is None:
                raise RuntimeError(
                    "DSV4 layer-major U2 requires AFD connector metadata"
                )
            if int(getattr(forward_context, "ubatch_idx", -1)) != stage_idx:
                raise RuntimeError(
                    "DSV4 layer-major stage context order is invalid: "
                    f"expected={stage_idx} actual="
                    f"{getattr(forward_context, 'ubatch_idx', None)}"
                )
            forward_context.input_ids = item.input_ids
            forward_context.afd_layer_major_u2 = True
            connectors.append(connector)

        connector = connectors[0]
        if any(stage_connector is not connector for stage_connector in connectors[1:]):
            raise RuntimeError("DSV4 layer-major U2 stages must share one connector")
        require_idle = getattr(connector, "require_attention_pipeline_idle", None)
        wait_for_receive = getattr(
            connector,
            "wait_for_attention_stage_receive",
            None,
        )
        reset_pipeline = getattr(connector, "reset_attention_pipeline_state", None)
        if not all(callable(method) for method in (require_idle, wait_for_receive)):
            raise RuntimeError("DSV4 layer-major U2 requires the HCCL stream connector")

        hidden_ubatches: list[torch.Tensor] = []
        pending_layers: list[AFDDeepseekV4DecoderLayer | None] = [None, None]
        pending_continuations: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ] = [None, None]
        aux_hidden_ubatches: list[list[torch.Tensor]] = [[], []]
        require_idle()
        try:
            for item, forward_context in zip(
                ubatch_metadata,
                stage_contexts,
                strict=True,
            ):
                with override_forward_context(forward_context):
                    if native.get_pp_group().is_first_rank:
                        hidden_states = (
                            item.inputs_embeds
                            if item.inputs_embeds is not None
                            else self.embed_input_ids(item.input_ids)
                        )
                        hidden_states = hidden_states.unsqueeze(1).repeat(
                            1,
                            self.hc_mult,
                            1,
                        )
                    else:
                        if item.intermediate_tensors is None:
                            raise RuntimeError(
                                "pipeline stage requires intermediate tensors"
                            )
                        hidden_states = item.intermediate_tensors["hidden_states"]
                    hidden_ubatches.append(hidden_states)

            graph_pipeline_active = getattr(
                connector,
                "attention_graph_compute_pipeline_active",
                None,
            )
            with override_forward_context(stage_contexts[0]):
                use_graph_compute = bool(
                    callable(graph_pipeline_active) and graph_pipeline_active()
                )
            if use_graph_compute:
                self._forward_ubatches_graph_compute_pipeline(
                    ubatch_metadata=ubatch_metadata,
                    stage_contexts=stage_contexts,
                    connector=connector,
                    hidden_ubatches=hidden_ubatches,
                    pending_layers=pending_layers,
                    pending_continuations=pending_continuations,
                    aux_hidden_ubatches=aux_hidden_ubatches,
                )
            else:
                self._forward_ubatches_eager_pipeline(
                    ubatch_metadata=ubatch_metadata,
                    stage_contexts=stage_contexts,
                    wait_for_receive=wait_for_receive,
                    hidden_ubatches=hidden_ubatches,
                    pending_layers=pending_layers,
                    pending_continuations=pending_continuations,
                    aux_hidden_ubatches=aux_hidden_ubatches,
                )
        except BaseException:
            if callable(reset_pipeline):
                reset_pipeline()
            raise

        if not native.get_pp_group().is_last_rank:
            return [
                native.IntermediateTensors({"hidden_states": hidden_states})
                for hidden_states in hidden_ubatches
            ]

        if self.mtp_enabled:
            buffer_offset = 0
            for hidden_states in hidden_ubatches:
                mtp_hidden = hidden_states.flatten(1)
                next_offset = buffer_offset + mtp_hidden.shape[0]
                self._mtp_hidden_buffer[buffer_offset:next_offset].copy_(
                    mtp_hidden,
                )
                buffer_offset = next_offset

        outputs: list[Any] = []
        for stage_idx, forward_context in enumerate(stage_contexts):
            with override_forward_context(forward_context):
                hidden_states = self.hc_head(
                    hidden_ubatches[stage_idx],
                    self.hc_head_fn,
                    self.hc_head_scale,
                    self.hc_head_base,
                )
                hidden_states = self.norm(hidden_states)
                aux_hidden_states = aux_hidden_ubatches[stage_idx]
                outputs.append(
                    (hidden_states, aux_hidden_states)
                    if aux_hidden_states
                    else hidden_states
                )
        return outputs

    def _forward_ubatches_eager_pipeline(
        self,
        *,
        ubatch_metadata: list[Any],
        stage_contexts: list[Any],
        wait_for_receive: Any,
        hidden_ubatches: list[torch.Tensor],
        pending_layers: list[AFDDeepseekV4DecoderLayer | None],
        pending_continuations: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ],
        aux_hidden_ubatches: list[list[torch.Tensor]],
    ) -> None:
        llama_4_scaling = None
        for layer_offset, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            for stage_idx, (item, forward_context) in enumerate(
                zip(ubatch_metadata, stage_contexts, strict=True)
            ):
                with override_forward_context(forward_context):
                    if layer_offset > 0:
                        wait_for_receive(
                            stage_idx=stage_idx,
                            tensor=hidden_ubatches[stage_idx],
                        )
                        self._complete_pending_stage(
                            stage_idx,
                            hidden_ubatches,
                            pending_layers,
                            pending_continuations,
                            aux_hidden_ubatches,
                        )
                    hidden_states, continuation = layer.forward_attention_to_remote_ffn(
                        item.positions,
                        hidden_ubatches[stage_idx],
                        None,
                        llama_4_scaling,
                    )
                    hidden_ubatches[stage_idx] = hidden_states
                    pending_layers[stage_idx] = layer
                    pending_continuations[stage_idx] = continuation

        for stage_idx, forward_context in enumerate(stage_contexts):
            with override_forward_context(forward_context):
                wait_for_receive(
                    stage_idx=stage_idx,
                    tensor=hidden_ubatches[stage_idx],
                )
                self._complete_pending_stage(
                    stage_idx,
                    hidden_ubatches,
                    pending_layers,
                    pending_continuations,
                    aux_hidden_ubatches,
                    final=True,
                )

    def _forward_ubatches_graph_compute_pipeline(
        self,
        *,
        ubatch_metadata: list[Any],
        stage_contexts: list[Any],
        connector: Any,
        hidden_ubatches: list[torch.Tensor],
        pending_layers: list[AFDDeepseekV4DecoderLayer | None],
        pending_continuations: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ],
        aux_hidden_ubatches: list[list[torch.Tensor]],
    ) -> None:
        compute_scope = getattr(connector, "attention_graph_compute", None)
        wait_for_compute = getattr(
            connector,
            "wait_for_attention_graph_compute",
            None,
        )
        join_compute = getattr(
            connector,
            "join_attention_graph_compute",
            None,
        )
        hybrid_dag_active = getattr(
            connector,
            "attention_graph_hybrid_dag_active",
            None,
        )
        wait_for_receive = getattr(
            connector,
            "wait_for_attention_stage_receive",
            None,
        )
        if not all(
            callable(method)
            for method in (
                compute_scope,
                wait_for_compute,
                join_compute,
                wait_for_receive,
                hybrid_dag_active,
            )
        ):
            raise RuntimeError("DSV4 Graph U2 requires the HCCL compute pipeline")

        layers = list(islice(self.layers, self.start_layer, self.end_layer))
        llama_4_scaling = None
        with override_forward_context(stage_contexts[0]):
            use_hybrid_dag = hybrid_dag_active()

        def enqueue_attention_compute(
            layer_offset: int,
            layer: AFDDeepseekV4DecoderLayer,
            stage_idx: int,
        ) -> None:
            item = ubatch_metadata[stage_idx]
            forward_context = stage_contexts[stage_idx]
            with (
                override_forward_context(forward_context),
                compute_scope(
                    layer_idx=layer.layer_idx,
                    stage_idx=stage_idx,
                    tensors=(hidden_ubatches[stage_idx], item.positions),
                    wait_for_receive_layer_idx=(
                        layers[layer_offset - 1].layer_idx
                        if use_hybrid_dag and layer_offset > 0
                        else None
                    ),
                ),
            ):
                if layer_offset > 0:
                    wait_for_receive(
                        stage_idx=stage_idx,
                        tensor=hidden_ubatches[stage_idx],
                    )
                    self._complete_pending_stage(
                        stage_idx,
                        hidden_ubatches,
                        pending_layers,
                        pending_continuations,
                        aux_hidden_ubatches,
                    )
                hidden_states, continuation = (
                    layer.forward_attention_to_remote_ffn_input(
                        item.positions,
                        hidden_ubatches[stage_idx],
                        None,
                        llama_4_scaling,
                    )
                )
                hidden_ubatches[stage_idx] = hidden_states
                pending_continuations[stage_idx] = continuation

        # Host-side Graph construction remains layer-major. With the hybrid DAG,
        # each side-stream stage waits only for its own prior-layer F2A receive,
        # so S0 compute can execute while the parent stream finishes S1 exchange.
        first_layer = layers[0]
        for stage_idx in range(len(stage_contexts)):
            enqueue_attention_compute(0, first_layer, stage_idx)

        final_event_layer = int(self.config.num_hidden_layers)
        for layer_offset, layer in enumerate(layers):
            for stage_idx, forward_context in enumerate(stage_contexts):
                with override_forward_context(forward_context):
                    wait_for_compute(
                        layer_idx=layer.layer_idx,
                        stage_idx=stage_idx,
                        tensors=(hidden_ubatches[stage_idx],),
                    )
                    transfer = layer.dispatch_remote_ffn(hidden_ubatches[stage_idx])
                    hidden_ubatches[stage_idx] = layer.receive_remote_ffn(transfer)
                    pending_layers[stage_idx] = layer

            next_layer_offset = layer_offset + 1
            if next_layer_offset < len(layers):
                for stage_idx in range(len(stage_contexts)):
                    enqueue_attention_compute(
                        next_layer_offset,
                        layers[next_layer_offset],
                        stage_idx,
                    )
            else:
                for stage_idx, forward_context in enumerate(stage_contexts):
                    with (
                        override_forward_context(forward_context),
                        compute_scope(
                            layer_idx=final_event_layer,
                            stage_idx=stage_idx,
                            tensors=(hidden_ubatches[stage_idx],),
                            wait_for_receive_layer_idx=(
                                layer.layer_idx if use_hybrid_dag else None
                            ),
                        ),
                    ):
                        wait_for_receive(
                            stage_idx=stage_idx,
                            tensor=hidden_ubatches[stage_idx],
                        )
                        self._complete_pending_stage(
                            stage_idx,
                            hidden_ubatches,
                            pending_layers,
                            pending_continuations,
                            aux_hidden_ubatches,
                            final=True,
                        )

        for stage_idx, forward_context in enumerate(stage_contexts):
            with override_forward_context(forward_context):
                join_compute(
                    layer_idx=final_event_layer,
                    stage_idx=stage_idx,
                    tensors=(hidden_ubatches[stage_idx],),
                )

    def _complete_pending_stage(
        self,
        stage_idx: int,
        hidden_ubatches: list[torch.Tensor],
        pending_layers: list[AFDDeepseekV4DecoderLayer | None],
        pending_continuations: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ],
        aux_hidden_ubatches: list[list[torch.Tensor]],
        *,
        final: bool = False,
    ) -> None:
        pending_layer = pending_layers[stage_idx]
        continuation = pending_continuations[stage_idx]
        if pending_layer is None or continuation is None:
            qualifier = "final " if final else ""
            raise RuntimeError(
                f"DSV4 layer-major stage has no {qualifier}pending layer: "
                f"stage={stage_idx}"
            )
        hidden_ubatches[stage_idx] = pending_layer.complete_remote_ffn(
            hidden_ubatches[stage_idx],
            continuation,
        )
        if pending_layer.layer_idx + 1 in self.aux_hidden_state_layers:
            aux_hidden_ubatches[stage_idx].append(
                hidden_ubatches[stage_idx].mean(dim=1)
            )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(hidden_states, **kwargs)


class AFDDeepseekV4ForCausalLM(native.AscendDeepseekV4ForCausalLM):
    """DSV4 causal LM wrapper with strict role ownership."""

    model_cls = AFDDeepseekV4Model

    # Patch reason: native construction allocates the LM head for both roles.
    # Patch functionality: build the head only for Attention and register FFN MoE.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: 80d8c194f7584b17fe08065ea99a130916f6b0e7
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: establish the AFD role before allocating modules.
        nn.Module.__init__(self)
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        # ### PATCH END
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = self.model_cls(
            vllm_config=vllm_config,
            prefix=native.maybe_prefix(prefix, "model"),
        )
        if self.afd_role == "attention" and native.get_pp_group().is_last_rank:
            self.lm_head = native.ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=native.maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = native.PPMissingLayer()
        self.logits_processor = native.LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.num_moe_layers = config.num_hidden_layers
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        if self.afd_role == "ffn":
            for layer in self.model.layers:
                if isinstance(layer, native.PPMissingLayer):
                    continue
                if isinstance(layer.mlp, native.DeepseekV4MoE):
                    example_moe = layer.mlp
                    self.moe_mlp_layers.append(layer.mlp)
                    self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx, **kwargs)

    def forward_ubatches_layer_major(
        self,
        ubatch_metadata: list[Any],
    ) -> list[Any]:
        return self.model.forward_ubatches_layer_major(ubatch_metadata)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        if self.afd_role != "attention":
            return None
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(_iter_role_weights(weights, role=self.afd_role))


class AFDDeepSeekMultiTokenPredictorLayer(native_mtp.DeepSeekMultiTokenPredictorLayer):
    """Role-selective DSV4 MTP layer for the pinned v0.23 Ascend stack."""

    # Patch reason: upstream MTP allocates Attention, HC, head, and MoE together.
    # Patch functionality: allocate only the active AFD role and remote the MoE.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        # ### PATCH START: role-selective MTP construction.
        nn.Module.__init__(self)
        self.afd_role = parse_afd_config(vllm_config, validate=False).role
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        self.device = native_mtp.current_platform.device_type
        self.is_v32 = hasattr(config, "index_topk")

        if self.afd_role == "attention":
            self.e_proj = native_mtp.ReplicatedLinear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.e_proj",
                return_bias=False,
            )
            self.h_proj = native_mtp.ReplicatedLinear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.h_proj",
                return_bias=False,
            )
            self.enorm = native_mtp.RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.hnorm = native_mtp.RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            topk_indices_buffer = (
                torch.empty(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.index_topk,
                    dtype=torch.int32,
                    device=self.device,
                )
                if self.is_v32
                else None
            )
            self.shared_head = native_mtp.SharedHead(
                config=config,
                prefix=prefix,
                quant_config=quant_config,
            )
            self.hc_eps = config.hc_eps
            self.hc_mult = hc_mult = config.hc_mult
            hc_dim = hc_mult * config.hidden_size
            self.hc_head_fn = nn.Parameter(
                torch.empty(hc_mult, hc_dim, dtype=torch.float32)
            )
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
            self.norm_eps = config.rms_norm_eps
        else:
            topk_indices_buffer = None

        self.mtp_block = AFDDeepseekV4DecoderLayer(
            vllm_config,
            prefix,
            config=self.config,
            topk_indices_buffer=topk_indices_buffer,
            is_draft_layer=True,
        )
        # ### PATCH END

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        # ### PATCH START: Attention-only remote-MoE execution.
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 MTP FFN role is connector-driven")
        if inputs_embeds is None:
            raise RuntimeError("DSV4 MTP Attention requires inputs_embeds")
        # ### PATCH END
        inputs_embeds = torch.where(
            positions.unsqueeze(-1) == 0,
            0,
            inputs_embeds,
        )
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = previous_hidden_states.view(
            -1,
            self.hc_mult,
            self.config.hidden_size,
        )
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.e_proj(inputs_embeds).unsqueeze(-2) + self.h_proj(
            previous_hidden_states
        )
        # ### PATCH START: bind this proposal iteration to the MTP phase.
        proxy = self.mtp_block.mlp
        if isinstance(proxy, AFDDeepseekV4RemoteMoEProxy):
            proxy.speculative_step = spec_step_index
        # ### PATCH END
        hidden_states, _ = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        return hidden_states

    def compute_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mtp_block.compute_ffn_output(hidden_states)


class AFDDeepSeekMultiTokenPredictor(native_mtp.DeepSeekMultiTokenPredictor):
    """Role-aware container for the single DSV4 MTP layer."""

    # Patch reason: upstream constructs embeddings and full MTP layers on both roles.
    # Patch functionality: construct role-owned modules and expose layer bounds.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # ### PATCH START: role-selective MTP container.
        nn.Module.__init__(self)
        self.afd_role = parse_afd_config(vllm_config, validate=False).role
        config = vllm_config.model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)
        # vLLM-Ascend caches the target model's layer-index capability and
        # subsequently reads these attributes from the draft model as well.
        self.start_layer = 0
        self.end_layer = self.num_mtp_layers
        self.layers = nn.ModuleDict(
            {
                str(index): AFDDeepSeekMultiTokenPredictorLayer(
                    vllm_config,
                    f"{prefix}.{index}",
                )
                for index in range(self.num_mtp_layers)
            }
        )
        if self.afd_role == "attention":
            self.embed_tokens = native_mtp.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
            self.logits_processor = native_mtp.LogitsProcessor(config.vocab_size)
        else:
            self.embed_tokens = native.PPMissingLayer()
            self.logits_processor = None
        # ### PATCH END

    # Patch reason: upstream permits embedding lookup on every constructed role.
    # Patch functionality: enforce Attention ownership of draft embeddings.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # ### PATCH START: Attention-only embedding ownership.
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 MTP FFN role does not own embeddings")
        # ### PATCH END
        return self.embed_tokens(input_ids)

    # Patch reason: upstream permits full MTP execution on every constructed role.
    # Patch functionality: run draft orchestration only on Attention.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # ### PATCH START: Attention-only draft orchestration.
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 MTP FFN role is connector-driven")
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        # ### PATCH END
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            spec_step_idx,
        )

    # Patch reason: upstream exposes logits for every constructed role.
    # Patch functionality: enforce Attention ownership of draft logits.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # ### PATCH START: Attention-only logits ownership.
        if self.afd_role != "attention" or self.logits_processor is None:
            raise RuntimeError("DSV4 MTP logits belong to the Attention role")
        # ### PATCH END
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(current_step_idx)]
        hidden_states = hidden_states.view(
            -1,
            mtp_layer.hc_mult,
            mtp_layer.config.hidden_size,
        )
        hidden_states = mtp_layer.hc_head(
            hidden_states,
            mtp_layer.hc_head_fn,
            mtp_layer.hc_head_scale,
            mtp_layer.hc_head_base,
        )
        return self.logits_processor(
            mtp_layer.shared_head.head,
            mtp_layer.shared_head(hidden_states),
        )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        return self.layers[str(layer_idx)].compute_ffn_output(hidden_states)


# ### PATCH START: keep multi-step eager draft orchestration outside compile.
# The pinned proposer omits spec_step_idx. Guard-free compiled reuse freezes
# the Python iteration counter below at step zero, violating the eager AFD
# wire protocol on the second draft token. Graph draft uses static headers;
# it and single-token draft retain their existing compilation paths.
@native_mtp.support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "hidden_states": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    },
    enable_if=lambda config: (
        not (
            config.speculative_config is not None
            and config.speculative_config.enforce_eager
            and config.speculative_config.num_speculative_tokens > 1
        )
    ),
)
# ### PATCH END: keep multi-step eager draft orchestration outside compile.
class AFDDeepSeekV4MTP(native_mtp.DeepSeekV4MTP):
    """Strict AFD role wrapper for the native DSV4 MTP model."""

    # Patch reason: upstream constructs a full draft model for every worker.
    # Patch functionality: use the role-aware MTP container.
    # Signature: matches upstream; no added parameters.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # ### PATCH START: role-aware top-level MTP model.
        nn.Module.__init__(self)
        self.afd_role = parse_afd_config(vllm_config, validate=False).role
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = AFDDeepSeekMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=native_mtp.maybe_prefix(prefix, "mtp"),
        )
        speculative_config = vllm_config.speculative_config
        self._afd_num_speculative_tokens = int(
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 1
        )
        self._afd_next_speculative_step = 0
        self.set_moe_parameters()
        # ### PATCH END

    # Patch reason: the pinned merged proposer omits ``spec_step_idx`` on every
    # repeated invocation of a single MTP layer.
    # Patch functionality: number implicit invocations within each N-token draft
    # while preserving an explicitly supplied step.
    # Signature: compatible with upstream; no added parameters.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int | None = None,
    ) -> torch.Tensor:
        # ### PATCH START: recover the omitted merged-draft iteration index.
        del intermediate_tensors
        if spec_step_idx is None:
            spec_step_idx = self._afd_next_speculative_step
            self._afd_next_speculative_step = (
                spec_step_idx + 1
            ) % self._afd_num_speculative_tokens
        # ### PATCH END
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            int(spec_step_idx),
        )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        if self.afd_role != "ffn":
            raise RuntimeError("DSV4 MTP Attention does not own local MoE weights")
        return self.model.compute_ffn_output(hidden_states, layer_idx)

    def attach_afd_connector(self, connector: object) -> None:
        if self.afd_role != "attention":
            return
        for layer in self.model.layers.values():
            proxy = layer.mtp_block.mlp
            if isinstance(proxy, AFDDeepseekV4RemoteMoEProxy):
                proxy.attach_connector(connector)

    # Patch reason: upstream loader receives both Attention and FFN checkpoint keys.
    # Patch functionality: filter original keys once, then reuse the native loader.
    # Signature: matches upstream; no added parameters. Native delegation is retained
    # because its quantized expert mappings must remain pinned to the target stack.
    # Upstream: vllm_ascend/models/deepseek_v4_mtp.py
    # Commit: 3da28f9414583d2d0b672a8f06d1fae142404bda
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ### PATCH START: one-shot role filtering before native load.
        role_weights = _iter_mtp_role_weights(weights, role=self.afd_role)
        # ### PATCH END
        return super().load_weights(role_weights)


__all__ = ["AFDDeepseekV4ForCausalLM", "AFDDeepSeekV4MTP"]
