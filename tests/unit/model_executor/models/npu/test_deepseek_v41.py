# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU API seams only; native allocation/compute needs separate evidence."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm.distributed import parallel_state  # noqa: E402
from vllm.model_executor.models.interfaces import (  # noqa: E402
    get_mixture_of_experts_model,
)

from afd_plugin.connectors import AFDForwardContextMetadata  # noqa: E402
from afd_plugin.model_executor.models.npu import deepseek_v41 as adapter  # noqa: E402
from tests.unit.fakes.deepseek_v41 import (  # noqa: E402
    ROUTER_DELTA_SCALE,
    Attention,
    LayerNorm,
    MoE,
    RemoteMoEPeer,
    attention_config,
    ffn_config,
    make_attention_loader,
)
from tests.unit.fakes.vllm_config import v41_dspark_config  # noqa: E402


@pytest.fixture
def attention_env(monkeypatch):
    monkeypatch.setattr(
        adapter.AFDDeepseekV41AttentionDecoderLayer, "attention_cls", Attention
    )
    monkeypatch.setattr(adapter.native, "RMSNorm", LayerNorm)
    monkeypatch.setattr(adapter.native, "enable_dsa_cp", lambda: False)
    monkeypatch.setattr(adapter.native, "uses_a5_packed_cache", lambda: False)
    return attention_config()


@pytest.fixture
def proxy_env(monkeypatch):
    peer = RemoteMoEPeer()
    metadata = AFDForwardContextMetadata(
        tokens_start_loc=[0],
        requests_start_loc=[0],
        stage_idx=0,
        connector=peer,
        tokens_lens=[2],
        num_stages=1,
        transaction_id="native-target",
        tokens_unpadded_lens=[2],
    )
    monkeypatch.setattr(
        adapter, "get_afd_metadata_from_forward_context", lambda: metadata
    )
    return peer, metadata


def test_attention_decoder_allocates_attention_hc_and_norms_without_moe(attention_env):
    layer = adapter.AFDDeepseekV41AttentionDecoderLayer(
        attention_env,
        "model.layers.0",
        config=attention_env.model_config.hf_config,
    )

    names = set(dict(layer.named_parameters()))
    assert "self_attn.indexer.compressor.wgate.weight" in names
    assert "post_attention_layernorm.weight" in names
    assert {"hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base"} <= names
    assert not any(name.startswith("mlp.") for name in names)
    assert list(layer.mlp.parameters()) == []


def test_dspark_target_keeps_the_remote_moe_role(attention_env):
    reduced = attention_env.model_config.hf_config
    native = v41_dspark_config()
    attention_env.model_config = native.model_config
    attention_env.speculative_config = native.speculative_config

    layer = adapter.AFDDeepseekV41AttentionDecoderLayer(
        attention_env, "model.layers.0", config=reduced
    )

    assert isinstance(layer.mlp, adapter.AFDDeepseekV41RemoteMoE)
    assert list(layer.mlp.parameters()) == []
    assert attention_env.speculative_config is native.speculative_config


@pytest.mark.parametrize("invalid", ["role", "gate", "draft", "sp"])
def test_attention_decoder_rejects_unsupported_role_state(invalid):
    config = attention_config()
    if invalid == "role":
        config.additional_config["afd"]["role"] = "ffn"
    elif invalid == "gate":
        config.additional_config["afd"]["compute_gate_on_attention"] = True
    elif invalid == "sp":
        config.parallel_config.use_sequence_parallel_moe = True

    with pytest.raises(ValueError, match="V4.1 Attention construction requires"):
        adapter.AFDDeepseekV41AttentionDecoderLayer(
            config, "model.layers.0", is_draft_layer=invalid == "draft"
        )


def test_remote_moe_preserves_independent_router_values_and_protocol_context(proxy_env):
    peer, metadata = proxy_env
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)
    fp32 = hidden.float() + 0.00031
    ids = torch.tensor([7, 2], dtype=torch.int32)
    proxy = adapter.AFDDeepseekV41RemoteMoE(layer_idx=3)

    output = proxy(hidden.clone(), input_ids=ids, hidden_states_fp32=fp32)

    expected = ((fp32 - hidden.float()) * ROUTER_DELTA_SCALE + ids[:, None]).bfloat16()
    torch.testing.assert_close(output, expected)
    payload = peer.payloads[0]
    torch.testing.assert_close(payload.hidden_states_fp32, fp32)
    torch.testing.assert_close(payload.input_ids, ids)
    assert payload.context.metadata.layer_idx == 3
    assert payload.context.metadata.stage_idx == metadata.stage_idx
    assert payload.context.metadata.seq_lens == [2]


@pytest.mark.parametrize("valid_tokens", [0, 1])
def test_remote_moe_zeros_transfer_padding_without_mutating_native_token_history(
    proxy_env, valid_tokens
):
    peer, metadata = proxy_env
    metadata.tokens_unpadded_lens = [valid_tokens]
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)
    raw_ids = torch.tensor([7, 2, 19, 23], dtype=torch.int32)
    original_ids = raw_ids.clone()
    proxy = adapter.AFDDeepseekV41RemoteMoE(layer_idx=3)
    proxy(hidden, input_ids=raw_ids, hidden_states_fp32=hidden.float())
    expected_ids = original_ids[:2].clone()
    expected_ids[valid_tokens:] = 0
    torch.testing.assert_close(peer.payloads[0].input_ids, expected_ids)
    torch.testing.assert_close(raw_ids, original_ids)


@pytest.mark.parametrize("invalid", ["ids", "fp32", "sp", "stage", "ubatches"])
def test_remote_moe_rejects_incomplete_or_unsupported_inputs(proxy_env, invalid):
    _, metadata = proxy_env
    if invalid == "stage":
        metadata.stage_idx = 1
    elif invalid == "ubatches":
        metadata.num_stages = 2
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)
    proxy = adapter.AFDDeepseekV41RemoteMoE(layer_idx=0)

    with pytest.raises(ValueError, match="V4.1 HCCL proxy"):
        proxy(
            hidden,
            input_ids=None if invalid == "ids" else torch.zeros(2, dtype=torch.int32),
            hidden_states_fp32=None if invalid == "fp32" else hidden.float(),
            already_sequence_parallel=invalid == "sp",
        )


def test_remote_moe_requires_a_current_afd_context(monkeypatch):
    monkeypatch.setattr(adapter, "get_afd_metadata_from_forward_context", lambda: None)
    proxy = adapter.AFDDeepseekV41RemoteMoE(layer_idx=0)
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="requires AFD forward metadata"):
        proxy(
            hidden,
            input_ids=torch.zeros(2, dtype=torch.int32),
            hidden_states_fp32=hidden.float(),
        )


@pytest.fixture
def model_env(monkeypatch):
    monkeypatch.setattr(adapter.native, "DeepseekV41MoE", MoE)
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1),
    )
    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1),
    )
    monkeypatch.setattr(
        adapter.native,
        "get_ascend_config",
        lambda: SimpleNamespace(mix_placement=False),
    )
    return ffn_config()


def test_attention_keeps_global_config_without_claiming_local_experts(model_env):
    model = make_attention_loader(
        adapter.AFDDeepseekV41AttentionLLMForCausalLM, model_env
    )

    assert get_mixture_of_experts_model(model) is None
    assert model.config is model_env.model_config.hf_config
    assert model.config.num_hidden_layers == 2
    assert model.config.n_routed_experts == 2


@pytest.mark.parametrize("prefix", ["", "model."])
@pytest.mark.parametrize(
    "checkpoint_path,parameter_path",
    [
        (
            "layers.0.attn.indexer.compressor.wgate.weight",
            "model.layers.0.self_attn.indexer.compressor.wgate.weight",
        ),
        ("layers.0.ffn_norm.weight", "model.layers.0.post_attention_layernorm.weight"),
        ("layers.0.hc_ffn_fn", "model.layers.0.hc_ffn_fn"),
        ("layers.0.engram.proj.weight", "model.layers.0.engram.proj.weight"),
    ],
)
def test_attention_native_loader_keeps_owned_namespaces_and_skips_ffn(
    model_env, prefix, checkpoint_path, parameter_path
):
    model = make_attention_loader(
        adapter.AFDDeepseekV41AttentionLLMForCausalLM, model_env
    )
    parameter = dict(model.named_parameters())[parameter_path]
    expected = torch.arange(parameter.numel(), dtype=parameter.dtype).reshape_as(
        parameter
    )
    weights = [(prefix + checkpoint_path, expected)]
    weights.extend(
        (root + "layers.0." + stage + "." + leaf, torch.empty(1))
        for root in ("", "model.")
        for stage in ("ffn", "mlp")
        for leaf in (
            "gate.weight",
            "shared_experts.gate_proj.weight",
            "experts.0.w1.weight",
        )
    )

    loaded = model.load_weights(iter(weights))

    assert loaded == {parameter_path}
    torch.testing.assert_close(parameter, expected)


def test_construction_allocates_only_ffn_parameters(model_env):
    model = adapter.AFDDeepseekV41FFNForCausalLM(vllm_config=model_env)
    assert set(dict(model.named_parameters())) == {
        f"model.layers.{layer}.mlp.{stage}.weight"
        for layer in range(2)
        for stage in ("gate", "experts")
    }
    assert model.get_experts_layer_indices() == (0, 1)
    assert model.model.has_engram is False
    assert model.num_routed_experts == 2


@pytest.mark.parametrize("pp_rank,owned_layer", [(0, 0), (1, 1)])
def test_pipeline_rank_constructs_only_its_ffn_layer(
    model_env, monkeypatch, pp_rank, owned_layer
):
    monkeypatch.setattr(
        parallel_state,
        "get_pp_group",
        lambda: SimpleNamespace(rank_in_group=pp_rank, world_size=2),
    )
    model = adapter.AFDDeepseekV41FFNForCausalLM(vllm_config=model_env)
    assert model.get_experts_layer_indices() == (owned_layer,)
    assert set(dict(model.named_parameters())) == {
        f"model.layers.{owned_layer}.mlp.gate.weight",
        f"model.layers.{owned_layer}.mlp.experts.weight",
    }
    with pytest.raises(ValueError, match="not owned"):
        model.compute_ffn_output(
            torch.zeros(1, 4),
            1 - owned_layer,
            input_ids=torch.tensor([7]),
            hidden_states_fp32=torch.zeros(1, 4),
        )


@pytest.mark.parametrize("invalid", ["attention", "attention-gate", "engram", "draft"])
def test_construction_rejects_incompatible_role_state(model_env, invalid):
    if invalid == "attention":
        model_env.additional_config["afd"]["role"] = "attention"
    elif invalid == "attention-gate":
        model_env.additional_config["afd"]["compute_gate_on_attention"] = True
    elif invalid == "engram":
        model_env.engram_config = SimpleNamespace()
    else:
        model_env.speculative_config = SimpleNamespace()
    with pytest.raises(ValueError, match="V4.1 FFN construction requires"):
        adapter.AFDDeepseekV41FFNForCausalLM(vllm_config=model_env)


@pytest.mark.parametrize("already_sequence_parallel", [False, True])
def test_compute_retains_fp32_router_values_and_raw_token_ids(
    model_env, already_sequence_parallel
):
    model = adapter.AFDDeepseekV41FFNForCausalLM(vllm_config=model_env)
    router = torch.tensor([[1.0001, 2.0001, 3.0001, 4.0001]])
    token_ids = torch.tensor([7])
    result = model.compute_ffn_output(
        router.bfloat16(),
        0,
        input_ids=token_ids,
        hidden_states_fp32=router,
        already_sequence_parallel=already_sequence_parallel,
    )
    torch.testing.assert_close(result, router + 7 + int(already_sequence_parallel))


@pytest.mark.parametrize("invalid", ["precision", "shape", "layer"])
def test_compute_rejects_invalid_input(model_env, invalid):
    model = adapter.AFDDeepseekV41FFNForCausalLM(vllm_config=model_env)
    router = torch.zeros(1, 4, dtype=torch.float32)
    if invalid == "precision":
        router = router.bfloat16()
    elif invalid == "shape":
        router = router[:, :2]
    with pytest.raises(ValueError, match="FP32|shapes must match|not owned"):
        model.compute_ffn_output(
            torch.zeros(1, 4),
            -1 if invalid == "layer" else 0,
            input_ids=torch.tensor([7]),
            hidden_states_fp32=router,
        )


@pytest.mark.parametrize(
    "gate_name",
    [
        "layers.0.ffn.gate.weight",
        "model.layers.0.ffn.gate.weight",
        "model.layers.0.mlp.gate.weight",
    ],
)
def test_native_loader_loads_ffn_and_ignores_other_checkpoint_names(
    model_env, gate_name
):
    model = adapter.AFDDeepseekV41FFNForCausalLM(vllm_config=model_env)
    excluded_names = [
        "layers.0.attn.indexer.compressor.wgate.weight",
        "layers.0.engram.embed.weight",
        "layers.0.ffn_norm.weight",
        "layers.0.hc_ffn_fn",
        "mtp.layers.0.ffn.gate.weight",
        "vision.layers.0.ffn.gate.weight",
        "language_model.layers.0.ffn.gate.weight",
        "layers.invalid.ffn.gate.weight",
        "embed.weight",
        "head.weight",
    ]
    expected = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    weights = [(gate_name, expected)] + [
        (name, torch.empty(1)) for name in excluded_names
    ]
    loaded = model.load_weights(iter(weights))
    assert loaded == {"model.layers.0.mlp.gate.weight"}
    torch.testing.assert_close(model.model.layers[0].mlp.gate.weight, expected)
