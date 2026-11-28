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
    uses_deepseek_v41_dspark_pd,
    validate_deepseek_v41_graph_speculation,
    validate_deepseek_v41_pd,
    validate_deepseek_v41_speculation,
)
from tests.unit.fakes.vllm_config import (
    configure_hccl_ep,
    npu_vllm_config,
    pd_kv_transfer_config,
    v41_dspark_config,
    v41_dspark_pd_config,
    v41_dspark_pd_graph_config,
)


def test_ffn_ep_rejects_unqualified_pd_before_role_isolation():
    source = configure_hccl_ep(v41_dspark_pd_config(role="ffn"))
    source.speculative_config = None
    source.kv_transfer_config = pd_kv_transfer_config()
    before = deepcopy(source)
    with pytest.raises(ValueError, match="attested Mooncake"):
        get_npu_ffn_config(source)
    assert source == before


@pytest.fixture
def ep_pd_source():
    source = configure_hccl_ep(v41_dspark_pd_config(role="ffn"), dp_rank=3)
    source.parallel_config.nnodes = 1
    source.speculative_config.num_speculative_tokens = 1
    source.kv_transfer_config.kv_port = 14579
    for layout in source.kv_transfer_config.kv_connector_extra_config.values():
        layout["dp_size"] = source.parallel_config.data_parallel_size
    return source


@pytest.mark.parametrize("steps", [None, 1, 2])
@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_consumer"])
def test_ep_pd_preserves_original_target_across_ffn_isolation(
    ep_pd_source, steps, kv_role
):
    source = ep_pd_source
    source.kv_transfer_config.kv_role = kv_role
    if steps is None:
        source.speculative_config = None
    else:
        source.speculative_config.num_speculative_tokens = steps
    source.engram_config = SimpleNamespace(dp_shared_memory=True)
    before = deepcopy(source)

    isolated = get_npu_ffn_config(source)

    assert source == before
    assert isolated.kv_transfer_config is None
    assert isolated.engram_config is isolated.speculative_config is None
    assert isolated.parallel_config is source.parallel_config


def test_ep_pd_rejects_foreign_dp_declaration_before_ffn_isolation(ep_pd_source):
    ep_pd_source.kv_transfer_config.kv_connector_extra_config["decode"]["dp_size"] = 1
    before = deepcopy(ep_pd_source)
    with pytest.raises(ValueError, match="configured equal P/D"):
        get_npu_ffn_config(ep_pd_source)
    assert ep_pd_source == before


@pytest.mark.parametrize("steps", [1, 2, 5])
@pytest.mark.parametrize("ubatches", [1, 2])
def test_ffn_ep_preserves_dspark_target_width_without_owning_the_drafter(
    steps, ubatches
):
    source = configure_hccl_ep(v41_dspark_config(role="ffn"))
    source.speculative_config.num_speculative_tokens = steps
    source.parallel_config.num_ubatches = ubatches
    source.parallel_config.use_ubatching = ubatches == 2
    before = deepcopy(source)

    shape = resolve_deepseek_v41_target_shape(source)
    isolated = get_npu_ffn_config(source)

    assert shape == DeepseekV41TargetShape(1 + steps)
    assert isolated.speculative_config is None
    assert isolated.parallel_config is source.parallel_config
    assert source == before


@pytest.mark.parametrize(
    ("axis", "changes", "reason"),
    [
        ("speculative_config", {"num_speculative_tokens": 3}, "eager N=1/2/5 draft"),
        ("speculative_config", {"enforce_eager": False}, "eager N=1/2/5 draft"),
        ("speculative_config", {"enable_adaptive_verification": True}, "adaptive"),
        ("parallel_config", {"num_ubatches": 2}, "with U1 or U2"),
        (
            "parallel_config",
            {"use_ubatching": True, "num_ubatches": 3},
            "with U1 or U2",
        ),
    ],
)
def test_ep_dspark_rejects_unsupported_draft_mode_before_ffn_isolation(
    axis, changes, reason
):
    source = configure_hccl_ep(v41_dspark_config(role="ffn"))
    source.speculative_config.num_speculative_tokens = 1
    vars(getattr(source, axis)).update(changes)
    with pytest.raises(ValueError, match=reason):
        get_npu_ffn_config(source)


def test_ep_dspark_requires_experts_divisible_by_the_native_dp_world():
    config = configure_hccl_ep(v41_dspark_config(), dp_size=3)
    config.speculative_config.num_speculative_tokens = 1
    with pytest.raises(ValueError, match="evenly sharded expert counts"):
        validate_deepseek_v41_speculation(config)


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
        ("model_config", "enable_sleep_mode", True, "sleep disabled"),
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


@pytest.mark.parametrize("module", [None, "afd_plugin.compat.npu.kv_layout_connector"])
def test_pd_keeps_the_exact_native_or_attesting_connector_on_attention_only(module):
    transfer = pd_kv_transfer_config()
    transfer.kv_connector_module_path = module
    source = npu_vllm_config(
        role="ffn",
        connector="P2pHcclAFDConnector",
        model_type="deepseek_v41",
        cudagraph_mode="NONE",
        kv_transfer_config=transfer,
    )
    isolated = get_npu_ffn_config(source)
    assert source.kv_transfer_config is transfer
    assert isolated.kv_transfer_config is None


def test_pd_rejects_weight_updates_before_clearing_ffn_owned_fields():
    source = npu_vllm_config(
        role="ffn",
        connector="P2pHcclAFDConnector",
        model_type="deepseek_v41",
        cudagraph_mode="NONE",
        kv_transfer_config=pd_kv_transfer_config(),
    )
    source.weight_transfer_config = SimpleNamespace()
    with pytest.raises(ValueError, match="weight updates disabled"):
        get_npu_ffn_config(source)
    assert source.kv_transfer_config is not None


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


@pytest.mark.parametrize("tp_size", [1, 4])
def test_exact_dspark_pd_keeps_prefix_ownership_on_attention(tp_size):
    source = v41_dspark_pd_config(role="ffn", tensor_parallel_size=tp_size)
    before = deepcopy(source)

    assert uses_deepseek_v41_dspark_pd(source)
    validate_deepseek_v41_pd(source)
    isolated = get_npu_ffn_config(source)

    assert source == before
    assert isolated.kv_transfer_config is None
    assert isolated.speculative_config is isolated.engram_config is None
    assert isolated.parallel_config is source.parallel_config


@pytest.mark.parametrize("tp_size", [1, 4])
@pytest.mark.parametrize("dbo,ubatch_size", [(False, 2), (True, 1), (True, 3)])
@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_consumer"])
def test_u2_pd_preserves_target_config_across_ffn_isolation(
    tp_size, dbo, ubatch_size, kv_role
):
    source = v41_dspark_pd_config(role="ffn", tensor_parallel_size=tp_size)
    source.kv_transfer_config.kv_role = kv_role
    parallel = source.parallel_config
    parallel.use_ubatching, parallel.enable_dbo = True, dbo
    parallel.num_ubatches, parallel.ubatch_size = 2, ubatch_size
    before = deepcopy(source)

    validate_deepseek_v41_pd(source)
    isolated = get_npu_ffn_config(source)

    assert source == before
    assert isolated.kv_transfer_config is None
    assert isolated.speculative_config is isolated.engram_config is None
    assert isolated.parallel_config is source.parallel_config
    assert isolated.parallel_config.num_ubatches == 2


@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize("stages", [True, 0, 1, 3, 2.0])
def test_dspark_pd_rejects_invalid_u2_stages(kv_role, stages):
    source = v41_dspark_pd_config()
    source.kv_transfer_config.kv_role = kv_role
    source.parallel_config.use_ubatching = True
    source.parallel_config.num_ubatches = stages
    before = deepcopy(source)

    with pytest.raises(ValueError, match="requires U1"):
        validate_deepseek_v41_pd(source)

    assert source == before


@pytest.mark.parametrize(
    "connector,use_ubatching",
    [("CAMP2pAFDConnector", True), ("P2pHcclAFDConnector", False)],
)
@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_consumer"])
def test_dspark_pd_requires_hccl_u2_schedule(connector, use_ubatching, kv_role):
    source = v41_dspark_pd_config()
    source.kv_transfer_config.kv_role = kv_role
    source.additional_config["afd"]["connector"] = connector
    source.parallel_config.use_ubatching = use_ubatching
    source.parallel_config.enable_dbo = True
    source.parallel_config.num_ubatches = 2
    before = deepcopy(source)

    with pytest.raises(ValueError, match="requires U1"):
        validate_deepseek_v41_pd(source)

    assert source == before


@pytest.mark.parametrize(
    "boundary, reason",
    [
        ("native_module", "speculation off"),
        ("draft_graph", "eager draft"),
        ("adaptive", "fixed verification"),
        ("ubatching", "U1"),
        ("dbo", "U1"),
    ],
)
def test_dspark_pd_rejects_modes_without_prefix_protocol_support(boundary, reason):
    source = v41_dspark_pd_config(role="ffn")
    if boundary == "native_module":
        source.kv_transfer_config.kv_connector_module_path = None
    elif boundary == "draft_graph":
        source.speculative_config.enforce_eager = False
    elif boundary == "adaptive":
        source.speculative_config.enable_adaptive_verification = True
    elif boundary == "ubatching":
        source.parallel_config.use_ubatching = True
    else:
        source.parallel_config.enable_dbo = True
    before = deepcopy(source)

    with pytest.raises(ValueError, match=reason):
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


@pytest.fixture
def pd_target_graph_config():
    """Original D target declaration, before any role/storage binding."""
    return v41_dspark_pd_graph_config(
        dtype=None,
        compilation_mode=SimpleNamespace(name="NONE"),
        cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
    )


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize(
    "stages,use_ubatching,enable_dbo",
    [(0, False, False), (1, False, False), (2, True, False), (2, True, True)],
)
@pytest.mark.parametrize("explicit_order", [False, True])
@pytest.mark.parametrize("has_dspark", [False, True])
def test_pd_target_graph_preserves_the_original_d_declaration_before_isolation(
    pd_target_graph_config,
    role,
    stages,
    use_ubatching,
    enable_dbo,
    explicit_order,
    has_dspark,
):
    source = pd_target_graph_config
    if not has_dspark:
        source.speculative_config = None
    source.additional_config["afd"]["role"] = role
    if explicit_order:
        source.additional_config["afd"]["connector_extra_config"] = {
            "graph_exchange_order": "stage_serial"
        }
    parallel = source.parallel_config
    parallel.num_ubatches = stages
    parallel.use_ubatching, parallel.enable_dbo = use_ubatching, enable_dbo
    before = deepcopy(source)

    validate_deepseek_v41_pd(source)
    assert validate_deepseek_v41_graph_speculation(source) == DeepseekV41TargetShape(
        3 if has_dspark else 1
    )
    if role == "ffn":
        isolated = get_npu_ffn_config(source)
        assert isolated.kv_transfer_config is None
        assert isolated.speculative_config is isolated.engram_config is None
        assert isolated.parallel_config is source.parallel_config
        assert isolated.compilation_config is source.compilation_config
    assert source == before


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("has_dspark", [False, True])
@pytest.mark.parametrize("enable_dbo", [False, True])
def test_pd_cross_layer_graph_preserves_exact_u2_consumer_before_isolation(
    pd_target_graph_config, role, has_dspark, enable_dbo
):
    source = pd_target_graph_config
    if not has_dspark:
        source.speculative_config = None
    source.additional_config["afd"]["role"] = role
    source.additional_config["afd"]["connector_extra_config"][
        "graph_exchange_order"
    ] = "cross_layer_v1"
    parallel = source.parallel_config
    parallel.use_ubatching = True
    parallel.num_ubatches = 2
    parallel.enable_dbo = enable_dbo
    before = deepcopy(source)

    validate_deepseek_v41_pd(source)
    assert validate_deepseek_v41_graph_speculation(source) == DeepseekV41TargetShape(
        3 if has_dspark else 1
    )
    if role == "ffn":
        isolated = get_npu_ffn_config(source)
        assert isolated.kv_transfer_config is None
        assert isolated.speculative_config is isolated.engram_config is None
        assert isolated.parallel_config is source.parallel_config
        assert isolated.compilation_config is source.compilation_config
    assert source == before


@pytest.mark.parametrize("has_dspark", [False, True])
@pytest.mark.parametrize(
    "stages,use_ubatching,enable_dbo,exchange_order",
    [
        (0, False, False, "cross_layer_v1"),
        (1, False, False, "cross_layer_v1"),
        (2, False, False, "cross_layer_v1"),
        (True, True, True, "cross_layer_v1"),
        (2.0, True, False, "cross_layer_v1"),
        (3, True, False, "cross_layer_v1"),
        (2, True, False, "bulk_v1"),
    ],
)
def test_pd_cross_layer_graph_rejects_non_u2_and_bulk_modes(
    pd_target_graph_config,
    has_dspark,
    stages,
    use_ubatching,
    enable_dbo,
    exchange_order,
):
    source = pd_target_graph_config
    if not has_dspark:
        source.speculative_config = None
    source.additional_config["afd"]["connector_extra_config"][
        "graph_exchange_order"
    ] = exchange_order
    parallel = source.parallel_config
    parallel.num_ubatches = stages
    parallel.use_ubatching, parallel.enable_dbo = use_ubatching, enable_dbo
    before = deepcopy(source)

    with pytest.raises(ValueError):
        validate_deepseek_v41_pd(source)
    with pytest.raises(ValueError):
        validate_deepseek_v41_graph_speculation(source)
    assert source == before


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("transfer", "kv_role", "kv_producer"),
        ("transfer", "kv_role", "kv_both"),
        ("transfer", "kv_connector_module_path", None),
        ("transfer", "kv_connector", "OtherConnector"),
        ("afd", "connector", "CAMP2pAFDConnector"),
        ("model", "enforce_eager", True),
        ("graph_mode", "name", "FULL"),
        ("graph_mode", "name", "PIECEWISE"),
        ("compile_mode", "name", "VLLM_COMPILE"),
        ("compilation", "use_inductor_graph_partition", True),
        ("extra", "graph_exchange_order", "bulk_v1"),
        ("extra", "graph_exchange_order", "cross_layer_v1"),
        ("parallel", "num_ubatches", True),
        ("parallel", "num_ubatches", 2.0),
        ("parallel", "num_ubatches", 2),
        ("parallel", "num_ubatches", 3),
        ("parallel", "use_ubatching", True),
        ("parallel", "enable_dbo", True),
        ("draft", "enforce_eager", False),
        ("draft", "enable_adaptive_verification", True),
        ("cache", "enable_prefix_caching", True),
        ("model", "enable_sleep_mode", True),
        ("source", "weight_transfer_config", SimpleNamespace()),
        ("additional", "sparse_kv_offload_config", {"enabled": True}),
        ("parallel", "data_parallel_size", 2),
        ("parallel", "pipeline_parallel_size", 2),
        ("parallel", "prefill_context_parallel_size", 2),
        ("parallel", "decode_context_parallel_size", 2),
        ("decode_placement", "tp_size", 2),
        ("draft_model", "model", "foreign-checkpoint"),
    ],
)
def test_pd_target_graph_rejects_unqualified_original_target_modes(
    pd_target_graph_config, section, field, value
):
    source = pd_target_graph_config
    sections = {
        "source": source,
        "model": source.model_config,
        "compilation": source.compilation_config,
        "compile_mode": source.compilation_config.mode,
        "graph_mode": source.compilation_config.cudagraph_mode,
        "parallel": source.parallel_config,
        "transfer": source.kv_transfer_config,
        "draft": source.speculative_config,
        "draft_model": source.speculative_config.draft_model_config,
        "cache": source.cache_config,
        "additional": source.additional_config,
        "afd": source.additional_config["afd"],
        "extra": source.additional_config["afd"]["connector_extra_config"],
        "decode_placement": source.kv_transfer_config.kv_connector_extra_config[
            "decode"
        ],
    }
    target = sections[section]
    if isinstance(target, dict):
        target[field] = value
    else:
        setattr(target, field, value)
    before = deepcopy(source)

    with pytest.raises(ValueError):
        validate_deepseek_v41_pd(source)
    with pytest.raises(ValueError):
        validate_deepseek_v41_graph_speculation(source)

    assert source == before


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("transfer", "kv_connector_module_path", None),
        ("transfer", "kv_role", "kv_producer"),
        ("graph", "name", "FULL"),
        ("extra", "graph_exchange_order", "cross_layer_v1"),
        ("parallel", "num_ubatches", True),
        ("cache", "enable_prefix_caching", True),
        ("decode", "tp_size", 2),
    ],
)
def test_pd_target_only_graph_rejects_unsafe_consumer_modes(
    pd_target_graph_config, section, field, value
):
    source = pd_target_graph_config
    source.speculative_config = None
    sections = {
        "transfer": source.kv_transfer_config,
        "graph": source.compilation_config.cudagraph_mode,
        "extra": source.additional_config["afd"]["connector_extra_config"],
        "parallel": source.parallel_config,
        "cache": source.cache_config,
        "decode": source.kv_transfer_config.kv_connector_extra_config["decode"],
    }
    target = sections[section]
    if isinstance(target, dict):
        target[field] = value
    else:
        setattr(target, field, value)
    before = deepcopy(source)

    with pytest.raises(ValueError):
        validate_deepseek_v41_pd(source)
    with pytest.raises(ValueError):
        validate_deepseek_v41_graph_speculation(source)

    assert source == before
