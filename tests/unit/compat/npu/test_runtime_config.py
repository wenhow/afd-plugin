# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Role configuration outcomes, without a device or upstream imports."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.runtime_config import (
    DeepseekV41TargetShape,
    get_npu_ffn_config,
    resolve_deepseek_v41_target_shape,
    validate_deepseek_v41_graph_speculation,
    validate_deepseek_v41_pd,
    validate_deepseek_v41_speculation,
)
from tests.unit.fakes.vllm_config import (
    npu_vllm_config,
    pd_kv_transfer_config,
    v41_dspark_config,
)


@pytest.mark.parametrize("steps", [None, 1, 5, 15])
def test_target_shape_retains_only_immutable_target_width_across_ffn_isolation(steps):
    source = v41_dspark_config(role="ffn")
    if steps is None:
        source.speculative_config = None
    else:
        source.speculative_config.num_speculative_tokens = steps
    before = deepcopy(source)

    shape = resolve_deepseek_v41_target_shape(source)
    isolated = get_npu_ffn_config(source)

    assert shape == DeepseekV41TargetShape(1 if steps is None else steps + 1)
    assert isolated.speculative_config is isolated.engram_config is None
    assert source == before
    with pytest.raises(FrozenInstanceError):
        shape.target_query_len = 1


@pytest.mark.parametrize("query_len", [True, False, 0, -1, 1.5, "3"])
def test_target_shape_rejects_nonpositive_or_noninteger_width(query_len):
    with pytest.raises(ValueError, match="positive integer"):
        DeepseekV41TargetShape(query_len)


@pytest.mark.parametrize("steps", [True, 0, -1, 1.5])
def test_target_shape_rejects_invalid_native_speculative_steps(steps):
    source = v41_dspark_config(role="ffn")
    source.speculative_config.num_speculative_tokens = steps
    with pytest.raises(ValueError, match="positive integer speculative steps"):
        resolve_deepseek_v41_target_shape(source)


def test_target_shape_rejects_width_above_native_token_capacity_before_isolation():
    source = v41_dspark_config(role="ffn")
    source.speculative_config.num_speculative_tokens = 16
    before = deepcopy(source)
    with pytest.raises(ValueError, match="exceeds native token capacity"):
        resolve_deepseek_v41_target_shape(source)
    assert source == before


def test_target_shape_keeps_other_models_on_the_existing_path():
    assert resolve_deepseek_v41_target_shape(npu_vllm_config()) is None


@pytest.mark.parametrize("steps", [None, 1, 2, 5])
def test_graph_target_shape_keeps_fixed_dspark_and_eager_draft_config(steps):
    config = v41_dspark_config()
    if steps is None:
        config.speculative_config = None
    else:
        config.speculative_config.num_speculative_tokens = steps
    before = deepcopy(config)
    assert validate_deepseek_v41_graph_speculation(config) == DeepseekV41TargetShape(
        1 if steps is None else steps + 1
    )
    assert config == before


@pytest.mark.parametrize(
    "boundary,reason",
    [
        ("draft_graph", "eager DSpark draft"),
        ("adaptive", "adaptive verification off"),
        ("pd", "KV transfer/PD"),
        ("method", "method='dspark'"),
    ],
)
def test_graph_target_shape_rejects_unqualified_dspark_combinations(boundary, reason):
    config = v41_dspark_config()
    if boundary == "draft_graph":
        config.speculative_config.enforce_eager = False
    elif boundary == "adaptive":
        config.speculative_config.enable_adaptive_verification = True
    elif boundary == "pd":
        config.kv_transfer_config = pd_kv_transfer_config()
    else:
        config.speculative_config.method = "mtp"
    before = deepcopy(config)
    with pytest.raises(ValueError, match=reason):
        validate_deepseek_v41_graph_speculation(config)
    assert config == before


def test_graph_target_shape_keeps_foreign_models_on_the_existing_path():
    assert validate_deepseek_v41_graph_speculation(npu_vllm_config()) is None


@pytest.mark.parametrize("has_engram", [False, True])
@pytest.mark.parametrize(
    "model_type,architectures",
    [
        ("deepseek_v41", None),
        ("deepseek_v41", ["DeepseekV41ForCausalLM"]),
        ("custom", ["DeepseekV41ForCausalLM"]),
        ("custom", ["AFDDeepseekV41ForCausalLM"]),
    ],
)
def test_ffn_config_disables_engram_without_changing_model_identity(
    has_engram, model_type, architectures
):
    engram = SimpleNamespace(dp_shared_memory=False) if has_engram else None
    source = npu_vllm_config(
        role="ffn",
        model_type=model_type,
        architectures=architectures,
        engram_config=engram,
    )

    result = get_npu_ffn_config(source)

    assert result.engram_config is None
    assert result is not source
    assert result.model_config is source.model_config
    assert source.engram_config is engram
    if engram is not None:
        assert engram.dp_shared_memory is False


@pytest.mark.parametrize(
    "model_type,architecture",
    [
        ("deepseek_v2", "DeepseekV2ForCausalLM"),
        ("deepseek_v4", "DeepseekV4ForCausalLM"),
        ("custom", "AFDDeepseekV4ForCausalLM"),
        ("custom", "DeepseekV410ForCausalLM"),
    ],
)
def test_ffn_config_preserves_existing_model_configurations(model_type, architecture):
    source = npu_vllm_config(
        role="ffn",
        model_type=model_type,
        architectures=[architecture],
        engram_config=SimpleNamespace(dp_shared_memory=False),
    )

    assert get_npu_ffn_config(source) is source
    assert source.engram_config.dp_shared_memory is False


@pytest.mark.parametrize("field", ["enable_dbo", "use_ubatching"])
def test_ffn_config_isolates_dspark_u2_without_changing_target_scheduling(field):
    source = v41_dspark_config(role="ffn")
    speculative = source.speculative_config
    source.engram_config = SimpleNamespace(dp_shared_memory=False)
    setattr(source.parallel_config, field, True)
    source.parallel_config.num_ubatches = 2

    result = get_npu_ffn_config(source)

    assert source.speculative_config is speculative
    assert source.engram_config.dp_shared_memory is False
    assert result.speculative_config is None and result.engram_config is None
    assert result.model_config is source.model_config
    assert result.parallel_config is source.parallel_config
    assert result.parallel_config.num_ubatches == 2


@pytest.mark.parametrize("connector", ["CAMP2pAFDConnector", "CAMAsyncAFDConnector"])
def test_ffn_config_preserves_non_hccl_dspark_u2_rejection(connector):
    source = v41_dspark_config(role="ffn")
    source.additional_config["afd"]["connector"] = connector
    source.parallel_config.enable_dbo = source.parallel_config.use_ubatching = True
    source.parallel_config.num_ubatches = source.parallel_config.ubatch_size = 2
    before = deepcopy(source)

    with pytest.raises(ValueError, match="DSpark with U1.*non-HCCL connectors"):
        get_npu_ffn_config(source)

    assert source == before


def test_dspark_validation_retains_native_target_and_draft_metadata():
    config = v41_dspark_config(tensor_parallel_size=4)
    before = deepcopy(config)

    validate_deepseek_v41_speculation(config)

    assert config == before
    assert config.model_config.hf_text_config.n_routed_experts == 384
    assert (
        config.speculative_config.draft_model_config.hf_text_config.n_routed_experts
        == 128
    )


@pytest.mark.parametrize("field", ["enable_dbo", "use_ubatching"])
def test_dspark_u2_rejects_adaptive_layout_before_ffn_role_isolation(field):
    source = v41_dspark_config(role="ffn")
    source.speculative_config.enable_adaptive_verification = True
    source.parallel_config.num_ubatches = 2
    setattr(source.parallel_config, field, True)
    before = deepcopy(source)

    with pytest.raises(ValueError, match="U2 requires adaptive verification off"):
        get_npu_ffn_config(source)

    assert source == before


def test_dspark_u1_keeps_native_adaptive_verification_configuration():
    config = v41_dspark_config()
    config.speculative_config.enable_adaptive_verification = True
    before = deepcopy(config)

    validate_deepseek_v41_speculation(config)

    assert config == before


@pytest.mark.parametrize(
    "owner,field,value,reason",
    [
        ("spec", "method", "mtp", "method='dspark'"),
        ("spec", "method", "eagle3", "method='dspark'"),
        ("spec", "rejection_sample_method", "synthetic", "real acceptance"),
        ("draft", "architectures", ["DSparkDraftModel"], "normalized"),
        ("draft_hf", "architectures", ["DeepseekV41ForCausalLM"], "normalized"),
        ("draft_hf", "model_type", "deepseek_v4", "normalized"),
        ("draft", "model", "other-checkpoint", "same target checkpoint"),
        ("draft_parallel", "tensor_parallel_size", 2, "equal target and draft TP"),
        ("target", "dspark_target_layer_ids", [], "one-based aux IDs"),
        ("target", "dspark_target_layer_ids", [0, 38, 39], "one-based aux IDs"),
        ("target", "dspark_target_layer_ids", [37, 38, 41], "one-based aux IDs"),
        ("target", "dspark_target_layer_ids", [39, 38, 37], "one-based aux IDs"),
        ("target", "dspark_target_layer_ids", [37, 37, 39], "one-based aux IDs"),
        ("target", "dspark_target_layer_ids", [True, 38, 39], "one-based aux IDs"),
        ("draft_hf", "dspark_target_layer_ids", [36, 38, 39], "one-based aux IDs"),
        ("target", "dspark_block_size", 0, "checkpoint draft metadata"),
        ("draft_hf", "dspark_block_size", 3, "checkpoint draft metadata"),
        ("draft_hf", "n_mtp_layers", 5, "checkpoint draft metadata"),
        ("draft_hf", "n_routed_experts", 384, "checkpoint draft metadata"),
        ("draft_hf", "num_experts_per_tok", 6, "checkpoint draft metadata"),
    ],
)
def test_dspark_rejects_configuration_drift(owner, field, value, reason):
    config = v41_dspark_config(tensor_parallel_size=4)
    spec = config.speculative_config
    owners = {
        "spec": spec,
        "draft": spec.draft_model_config,
        "draft_hf": spec.draft_model_config.hf_config,
        "draft_parallel": spec.draft_parallel_config,
        "target": config.model_config.hf_text_config,
    }
    setattr(owners[owner], field, value)

    with pytest.raises(ValueError, match=reason):
        validate_deepseek_v41_speculation(config)


def test_ffn_projection_rejects_classic_mtp_without_mutating_source():
    source = v41_dspark_config(role="ffn")
    source.speculative_config.method = "mtp"
    before = deepcopy(source)

    with pytest.raises(ValueError, match="method='dspark'"):
        get_npu_ffn_config(source)

    assert source == before


def test_ffn_config_rejects_a_v41_attention_role():
    source = npu_vllm_config(model_type="deepseek_v41")

    with pytest.raises(ValueError, match="role"):
        get_npu_ffn_config(source)


@pytest.mark.parametrize("tensor_parallel_size", [1, 4])
def test_ffn_config_leaves_dspark_and_engram_on_attention(tensor_parallel_size):
    source = v41_dspark_config(role="ffn", tensor_parallel_size=tensor_parallel_size)
    source.engram_config = SimpleNamespace(dp_shared_memory=False)
    speculative = source.speculative_config

    result = get_npu_ffn_config(source)

    assert result.speculative_config is None
    assert result.engram_config is None
    assert result.model_config is source.model_config
    assert source.speculative_config is speculative
    assert source.speculative_config.draft_model_config.architectures == [
        "DeepseekV41DSparkModel"
    ]
    assert source.engram_config.dp_shared_memory is False


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("num_stages", [1, 2])
@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_consumer"])
def test_hccl_ffn_config_isolates_pd_before_construction_without_changing_target(
    tp_size, num_stages, kv_role
):
    transfer = pd_kv_transfer_config(tp_size=tp_size)
    transfer.kv_role = kv_role
    source = npu_vllm_config(
        role="ffn",
        connector="P2pHcclAFDConnector",
        model_type="deepseek_v41",
        cudagraph_mode="NONE",
        kv_transfer_config=transfer,
        engram_config=SimpleNamespace(dp_shared_memory=False),
        tensor_parallel_size=tp_size,
        num_attention_ranks=tp_size,
        num_ffn_ranks=tp_size,
        num_ubatches=num_stages,
        use_ubatching=num_stages == 2,
        enable_dbo=num_stages == 2,
    )
    before = deepcopy(source)

    result = get_npu_ffn_config(source)

    assert result.kv_transfer_config is None
    assert result.engram_config is result.speculative_config is None
    assert result.model_config is source.model_config
    assert result.parallel_config is source.parallel_config
    assert source == before and source.kv_transfer_config is transfer


@pytest.mark.parametrize("connector", ["CAMP2pAFDConnector", "CAMAsyncAFDConnector"])
def test_non_hccl_ffn_config_retains_existing_pd_configuration(connector):
    transfer = pd_kv_transfer_config(connector="AnotherNativeKVConnector")
    source = npu_vllm_config(
        role="ffn",
        connector=connector,
        model_type="deepseek_v41",
        kv_transfer_config=transfer,
    )

    result = get_npu_ffn_config(source)

    assert result.kv_transfer_config is source.kv_transfer_config is transfer


@pytest.mark.parametrize(
    "section,field,value,reason",
    [
        ("transfer", "kv_connector", "SfaRemoteD2HConnector", "native Mooncake"),
        ("transfer", "kv_connector", "MultiConnector", "native Mooncake"),
        ("transfer", "kv_connector_module_path", "custom.pd", "native Mooncake"),
        ("transfer", "kv_role", "kv_both", "separate producer or consumer"),
        ("transfer", "kv_role", None, "separate producer or consumer"),
        ("model_config", "enforce_eager", False, "eager execution"),
        ("cache_config", "enable_prefix_caching", True, "prefix caching"),
        ("parallel_config", "pipeline_parallel_size", 2, "pipeline_parallel_size=1"),
        ("parallel_config", "data_parallel_size", 2, "data_parallel_size=1"),
        (
            "parallel_config",
            "prefill_context_parallel_size",
            2,
            "context_parallel_size=1",
        ),
        (
            "parallel_config",
            "decode_context_parallel_size",
            2,
            "context_parallel_size=1",
        ),
    ],
)
def test_pd_rejection_happens_before_ffn_isolation_and_preserves_source(
    section, field, value, reason
):
    source = npu_vllm_config(
        role="ffn",
        connector="P2pHcclAFDConnector",
        model_type="deepseek_v41",
        cudagraph_mode="NONE",
        kv_transfer_config=pd_kv_transfer_config(),
    )
    target = (
        source.kv_transfer_config if section == "transfer" else vars(source)[section]
    )
    setattr(target, field, value)
    before = deepcopy(source)

    with pytest.raises(ValueError, match=reason):
        get_npu_ffn_config(source)

    assert source == before


@pytest.mark.parametrize("side", ["prefill", "decode"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("tp_size", 2),
        ("tp_size", True),
        ("tp_size", None),
        ("dp_size", 2),
        ("dp_size", None),
        ("pp_size", 2),
    ],
)
def test_pd_requires_explicit_equal_native_tp_and_unpartitioned_placements(
    side, field, value
):
    source = npu_vllm_config(
        connector="P2pHcclAFDConnector",
        model_type="deepseek_v41",
        cudagraph_mode="NONE",
        kv_transfer_config=pd_kv_transfer_config(),
    )
    source.kv_transfer_config.kv_connector_extra_config[side][field] = value
    before = deepcopy(source)

    with pytest.raises(ValueError, match="configured equal P/D"):
        validate_deepseek_v41_pd(source)

    assert source == before


@pytest.mark.parametrize("num_stages", [1, 2])
def test_validated_dspark_pd_remains_rejected_before_ffn_draft_is_removed(num_stages):
    source = v41_dspark_config(role="ffn")
    source.kv_transfer_config = pd_kv_transfer_config()
    source.compilation_config.cudagraph_mode.name = "NONE"
    source.parallel_config.num_ubatches = num_stages
    source.parallel_config.enable_dbo = source.parallel_config.use_ubatching = (
        num_stages == 2
    )
    before = deepcopy(source)

    with pytest.raises(ValueError, match="PD requires speculation off"):
        get_npu_ffn_config(source)

    assert source == before


@pytest.mark.parametrize("feature", ["graph", "sparse_offload"])
def test_pd_does_not_admit_unqualified_native_cache_modes(feature):
    source = npu_vllm_config(
        role="ffn",
        connector="P2pHcclAFDConnector",
        model_type="deepseek_v41",
        cudagraph_mode="NONE",
        kv_transfer_config=pd_kv_transfer_config(),
    )
    if feature == "graph":
        source.compilation_config.cudagraph_mode.name = "FULL"
    else:
        source.additional_config["sparse_kv_offload_config"] = {"enabled": True}
    before = deepcopy(source)

    with pytest.raises(ValueError, match="without Graph|sparse KV offload disabled"):
        get_npu_ffn_config(source)

    assert source == before
