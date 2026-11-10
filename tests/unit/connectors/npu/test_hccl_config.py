# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Configuration admission without constructing devices or process groups."""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu.runtime_config import (
    DeepseekV41TargetShape,
    get_npu_ffn_config,
    resolve_deepseek_v41_target_shape,
)
from afd_plugin.config import (
    HCCL_P2P_CONNECTOR,
    SUPPORTED_AFD_CONNECTORS,
    AFDConfig,
    afd_config_from_mapping,
    validate_afd_config,
)
from tests.unit.fakes.vllm_config import hccl_vllm_config, pd_kv_transfer_config


@pytest.fixture(scope="module")
def hccl_config_module():
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    return importlib.import_module("afd_plugin.connectors.npu.hccl_config")


@pytest.fixture
def runtime(hccl_config_module):
    return hccl_vllm_config(
        dtype=hccl_config_module.torch.bfloat16,
        compilation_mode=hccl_config_module.CompilationMode.NONE,
        cudagraph_mode=hccl_config_module.CUDAGraphMode.NONE,
    )


@pytest.fixture
def afd_config():
    return AFDConfig(connector=HCCL_P2P_CONNECTOR)


@pytest.fixture
def graph_runtime(runtime, hccl_config_module):
    runtime.model_config.enforce_eager = False
    runtime.compilation_config.cudagraph_mode = (
        hccl_config_module.CUDAGraphMode.FULL_DECODE_ONLY
    )
    runtime.parallel_config.rank = 0
    return runtime


def test_common_config_accepts_hccl_without_importing_device_runtime():
    config = afd_config_from_mapping({"connector": HCCL_P2P_CONNECTOR})

    assert config.connector in SUPPORTED_AFD_CONNECTORS
    assert config.num_attention_ranks == config.num_ffn_ranks == 1
    assert not config.compute_gate_on_attention
    assert not config.async_dp


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("tp_size", [1, 2, 8])
def test_common_config_accepts_equal_tp_rank_counts(role, tp_size):
    validate_afd_config(
        AFDConfig(
            connector=HCCL_P2P_CONNECTOR,
            role=role,
            num_attention_ranks=tp_size,
            num_ffn_ranks=tp_size,
        )
    )


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"num_attention_ranks": 2}, "equal positive A/F"),
        ({"num_ffn_ranks": 2}, "equal positive A/F"),
        ({"num_attention_ranks": True}, "equal positive A/F"),
        ({"num_ffn_ranks": 1.0}, "equal positive A/F"),
        ({"compute_gate_on_attention": True}, "gate on FFN"),
        ({"async_dp": True}, "async mode"),
    ],
)
def test_common_config_rejects_unsupported_hccl_topology(afd_config, changes, reason):
    with pytest.raises(ValueError, match=reason):
        validate_afd_config(replace(afd_config, **changes))


@pytest.mark.parametrize("port", [65535, 65536, 0, -1, True, 65534.0])
def test_common_config_checks_the_entire_tp_pair_port_range(afd_config, port):
    config = replace(afd_config, num_attention_ranks=2, num_ffn_ranks=2, port=port)
    with pytest.raises(ValueError, match="full port range"):
        validate_afd_config(config)


def test_common_config_allows_last_pair_to_use_the_largest_tcp_port(afd_config):
    validate_afd_config(
        replace(afd_config, num_attention_ranks=2, num_ffn_ranks=2, port=65534)
    )


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"max_tokens": None},
        {"max_tokens": 4},
        {"graph_exchange_order": "stage_serial"},
        {"max_tokens": 4, "graph_exchange_order": "bulk_v1"},
        {"graph_exchange_order": "cross_layer_v1"},
    ],
)
def test_extra_config_round_trip(hccl_config_module, raw):
    extra = hccl_config_module.HcclExtraInfo.from_mapping(raw)

    assert extra.max_tokens == (raw or {}).get("max_tokens")
    order = (raw or {}).get("graph_exchange_order", "stage_serial")
    assert extra.graph_exchange_order == order
    expected = {"max_tokens": extra.max_tokens}
    if order != "stage_serial":
        expected["graph_exchange_order"] = order
    assert extra.to_mapping() == expected
    assert hccl_config_module.HcclExtraInfo.from_mapping(extra.to_mapping()) == extra


@pytest.mark.parametrize(
    "value", [None, True, False, 0, 1, [], {}, "", "bulk", "BULK_V1"]
)
def test_extra_config_requires_an_exact_graph_exchange_order(hccl_config_module, value):
    with pytest.raises(ValueError, match="graph_exchange_order must be"):
        hccl_config_module.HcclExtraInfo.from_mapping({"graph_exchange_order": value})


@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, 1.5, "4", [], {}])
def test_extra_config_requires_positive_integer_capacity(hccl_config_module, value):
    with pytest.raises(ValueError, match="positive integer"):
        hccl_config_module.HcclExtraInfo.from_mapping({"max_tokens": value})


def test_extra_config_rejects_unknown_fields(hccl_config_module):
    with pytest.raises(ValueError, match="unknown.*hccl_buffer_size"):
        hccl_config_module.HcclExtraInfo.from_mapping({"hccl_buffer_size": 200})


@pytest.mark.parametrize("raw", [[], "", 1])
def test_extra_config_requires_mapping(hccl_config_module, raw):
    with pytest.raises(TypeError, match="mapping"):
        hccl_config_module.HcclExtraInfo.from_mapping(raw)


def test_runtime_admission_preserves_config_and_reports_native_shape(
    hccl_config_module, runtime, afd_config
):
    before = deepcopy(runtime)

    capabilities = hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)

    assert capabilities.hidden_size == 16
    assert capabilities.vocab_size == 1024
    assert capabilities.layer_indices == (0, 1)
    assert capabilities.max_tokens == 16
    assert runtime == before


@pytest.mark.parametrize("tp_size,role_rank", [(2, 0), (2, 1), (8, 7)])
def test_runtime_publishes_native_tp_size_and_exact_pair_coordinate(
    hccl_config_module, afd_config, tp_size, role_rank
):
    runtime = hccl_vllm_config(
        dtype=hccl_config_module.torch.bfloat16,
        compilation_mode=hccl_config_module.CompilationMode.NONE,
        cudagraph_mode=hccl_config_module.CUDAGraphMode.NONE,
        tensor_parallel_size=tp_size,
    )
    config = replace(afd_config, num_attention_ranks=tp_size, num_ffn_ranks=tp_size)
    capabilities = hccl_config_module.validate_hccl_runtime(runtime, config, role_rank)
    assert (capabilities.native_tp_size, capabilities.pair_rank) == (tp_size, role_rank)


def test_native_wrapper_identity_uses_nested_text_dimensions(
    hccl_config_module, runtime, afd_config
):
    text = runtime.model_config.hf_text_config
    runtime.model_config.hf_config = SimpleNamespace(
        model_type="deepseek_v41",
        architectures=["DeepseekV41ForCausalLM"],
        text_config=text,
    )
    capabilities = hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)
    assert (capabilities.hidden_size, capabilities.vocab_size) == (16, 1024)
    assert capabilities.layer_indices == (0, 1)


@pytest.mark.parametrize(("max_tokens", "expected"), [(None, 16), (4, 4), (32, 16)])
def test_runtime_capacity_is_bounded_by_scheduler(
    hccl_config_module, runtime, afd_config, max_tokens, expected
):
    config = replace(afd_config, connector_extra_config={"max_tokens": max_tokens})

    assert (
        hccl_config_module.validate_hccl_runtime(runtime, config, 0).max_tokens
        == expected
    )


@pytest.mark.parametrize("num_ubatches", [0, 1])
def test_runtime_accepts_native_logical_u1(
    hccl_config_module, runtime, afd_config, num_ubatches
):
    runtime.parallel_config.num_ubatches = num_ubatches

    hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)


@pytest.mark.parametrize("num_ubatches", [0, 1, 2])
def test_staged_runtime_admits_counts_protocol_without_relaxing_legacy_u1(
    hccl_config_module, runtime, afd_config, num_ubatches
):
    parallel = runtime.parallel_config
    parallel.num_ubatches = num_ubatches
    parallel.enable_dbo = parallel.use_ubatching = num_ubatches == 2
    before = deepcopy(runtime)

    caps = hccl_config_module.validate_staged_hccl_runtime(runtime, afd_config, 0)

    assert caps.local_counts_version == 1
    assert (caps.attention_size, caps.ffn_size, caps.native_tp_size) == (1, 1, 1)
    assert (caps.hidden_size, caps.vocab_size, caps.max_tokens) == (16, 1024, 16)
    assert caps.layer_indices == (0, 1)
    assert runtime == before
    if num_ubatches == 2:
        with pytest.raises(ValueError, match="U1"):
            hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)


@pytest.mark.parametrize("order", ["bulk_v1", "cross_layer_v1"])
@pytest.mark.parametrize("staged", [False, True])
def test_eager_admission_rejects_graph_exchange_orders(
    hccl_config_module, runtime, afd_config, order, staged
):
    config = replace(afd_config, connector_extra_config={"graph_exchange_order": order})
    validate = (
        hccl_config_module.validate_staged_hccl_runtime
        if staged
        else hccl_config_module.validate_hccl_runtime
    )
    with pytest.raises(ValueError, match="requires Graph execution"):
        validate(runtime, config, 0)


@pytest.mark.parametrize(
    ("section", "field", "value", "reason"),
    [
        ("parallel_config", "num_ubatches", 3, "U1 or native U2"),
        ("parallel_config", "enable_expert_parallel", True, "expert parallelism"),
        ("parallel_config", "use_sequence_parallel_moe", True, "sequence parallelism"),
        ("parallel_config", "enable_eplb", True, "EPLB"),
        ("compilation_config", "use_inductor_graph_partition", True, "modes NONE"),
        (
            None,
            "kv_transfer_config",
            pd_kv_transfer_config(connector="UnsupportedKVConnector"),
            "KV transfer/PD",
        ),
        (None, "lora_config", True, "LoRA"),
    ],
)
def test_staged_runtime_preserves_unimplemented_feature_boundaries(
    hccl_config_module, runtime, afd_config, section, field, value, reason
):
    target = runtime if section is None else vars(runtime)[section]
    setattr(target, field, value)

    with pytest.raises(ValueError, match=reason):
        hccl_config_module.validate_staged_hccl_runtime(runtime, afd_config, 0)


def test_explicit_graph_component_reports_its_protocol_without_mutating_config(
    hccl_config_module, graph_runtime, afd_config
):
    before = deepcopy(graph_runtime)
    caps = hccl_config_module.validate_staged_hccl_runtime(
        graph_runtime, afd_config, 0, graph=True
    )
    assert caps.graph_execution_version == caps.local_counts_version == 1
    assert (caps.attention_size, caps.ffn_size, caps.native_tp_size) == (1, 1, 1)
    assert graph_runtime == before
    with pytest.raises(ValueError, match="enforce_eager=True"):
        hccl_config_module.validate_staged_hccl_runtime(graph_runtime, afd_config, 0)


@pytest.mark.parametrize(
    "section,field,value,reason",
    [
        ("model_config", "enforce_eager", True, "enforce_eager=False"),
        ("compilation_config", "cudagraph_mode", "NONE", "FULL_DECODE_ONLY"),
        ("compilation_config", "cudagraph_mode", "FULL", "FULL_DECODE_ONLY"),
        ("compilation_config", "mode", "COMPILE", "compilation NONE"),
        (
            "compilation_config",
            "use_inductor_graph_partition",
            True,
            "compilation NONE",
        ),
        ("parallel_config", "data_parallel_size", 2, "data_parallel_size=1"),
        (None, "kv_transfer_config", pd_kv_transfer_config(), "KV transfer/PD"),
        (None, "lora_config", True, "LoRA"),
    ],
)
def test_explicit_graph_component_preserves_its_restricted_execution_contract(
    hccl_config_module, graph_runtime, afd_config, section, field, value, reason
):
    target = graph_runtime if section is None else vars(graph_runtime)[section]
    setattr(target, field, value)
    with pytest.raises(ValueError, match=reason):
        hccl_config_module.validate_staged_hccl_runtime(
            graph_runtime, afd_config, 0, graph=True
        )


@pytest.fixture
def graph_binding(hccl_config_module, graph_runtime):
    afd_config = hccl_config_module.parse_afd_config(graph_runtime)
    caps = hccl_config_module.validate_staged_hccl_runtime(
        graph_runtime, afd_config, 0, graph=True
    )
    return SimpleNamespace(
        role="attention",
        role_rank=0,
        local_device=0,
        negotiated_capabilities=caps,
        closed=False,
        terminal=False,
        plan=None,
        graph_reservation=None,
    )


def test_explicit_graph_wire_binds_to_actual_rank_and_native_device(
    hccl_config_module, graph_runtime, graph_binding
):
    hccl_config_module.validate_staged_hccl_binding(
        graph_runtime, graph_binding, SimpleNamespace(type="npu", index=0)
    )
    with pytest.raises(ValueError, match="idle transport"):
        hccl_config_module.validate_staged_hccl_binding(
            graph_runtime, graph_binding, SimpleNamespace(type="npu", index=1)
        )


@pytest.mark.parametrize("order", ["stage_serial", "bulk_v1", "cross_layer_v1"])
def test_graph_wire_binds_only_to_the_configured_exchange_order(
    hccl_config_module, graph_runtime, graph_binding, order
):
    graph_runtime.additional_config["afd"]["connector_extra_config"] = {
        "graph_exchange_order": order
    }
    graph_binding.negotiated_capabilities = replace(
        graph_binding.negotiated_capabilities, graph_exchange_order=order
    )
    hccl_config_module.validate_staged_hccl_binding(
        graph_runtime, graph_binding, SimpleNamespace(type="npu", index=0)
    )
    foreign_order = "bulk_v1" if order == "stage_serial" else "stage_serial"
    graph_binding.negotiated_capabilities = replace(
        graph_binding.negotiated_capabilities, graph_exchange_order=foreign_order
    )
    with pytest.raises(ValueError, match="idle transport"):
        hccl_config_module.validate_staged_hccl_binding(
            graph_runtime, graph_binding, SimpleNamespace(type="npu", index=0)
        )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("tensor_parallel_size", 2, "tensor_parallel_size"),
        ("pipeline_parallel_size", 2, "pipeline_parallel_size"),
        ("data_parallel_size", 2, "data_parallel_size"),
        ("prefill_context_parallel_size", 2, "prefill_context_parallel_size"),
        ("decode_context_parallel_size", 2, "decode_context_parallel_size"),
        ("data_parallel_rank", 1, "data_parallel_rank"),
        ("enable_expert_parallel", True, "expert parallelism"),
        ("enable_eplb", True, "EPLB or elastic EP"),
        ("enable_elastic_ep", True, "EPLB or elastic EP"),
        ("enable_fault_tolerance", True, "fault tolerance"),
        ("use_sequence_parallel_moe", True, "sequence parallelism"),
        ("enable_dbo", True, "U1"),
        ("use_ubatching", True, "U1"),
        ("num_ubatches", 2, "U1"),
    ],
)
def test_runtime_rejects_parallel_expansion(
    hccl_config_module, runtime, afd_config, field, value, reason
):
    setattr(runtime.parallel_config, field, value)

    with pytest.raises(ValueError, match=reason):
        hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)


@pytest.mark.parametrize("role_rank", [-1, 1, False, 0.0])
def test_runtime_requires_single_role_rank(
    hccl_config_module, runtime, afd_config, role_rank
):
    with pytest.raises(ValueError, match="role_rank in"):
        hccl_config_module.validate_hccl_runtime(runtime, afd_config, role_rank)


@pytest.mark.parametrize(
    ("model_type", "architectures"),
    [
        ("deepseek_v4", ["DeepseekV4ForCausalLM"]),
        ("deepseek_v4", ["DeepseekV41ForCausalLM"]),
        ("custom", ["DeepseekV41ForCausalLM"]),
        ("deepseek_v41", ["DeepseekV4ForCausalLM"]),
        ("deepseek_v41", ["DeepseekV41ForCausalLM", "OtherForCausalLM"]),
        ("deepseek_v41", None),
    ],
)
def test_runtime_requires_exact_v41_identity(
    hccl_config_module, runtime, afd_config, model_type, architectures
):
    runtime.model_config.hf_config.model_type = model_type
    runtime.model_config.hf_config.architectures = architectures

    with pytest.raises(ValueError, match="exact DeepseekV41"):
        hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)


def test_runtime_accepts_existing_afd_v41_architecture(
    hccl_config_module, runtime, afd_config
):
    runtime.model_config.hf_config.architectures = ["AFDDeepseekV41ForCausalLM"]

    hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("enforce_eager", False, "enforce_eager"),
        ("is_multimodal_model", True, "decoder-only text"),
        ("is_encoder_decoder", True, "decoder-only text"),
        ("dtype", "bfloat16", "BF16"),
    ],
)
def test_runtime_rejects_unsupported_model_execution(
    hccl_config_module, runtime, afd_config, field, value, reason
):
    setattr(runtime.model_config, field, value)

    with pytest.raises(ValueError, match=reason):
        hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)


@pytest.mark.parametrize("language_model_only", [False, True])
def test_runtime_admits_native_multimodal_wrapper_only_in_text_mode(
    hccl_config_module, runtime, afd_config, language_model_only
):
    runtime.model_config.is_multimodal_model = True
    runtime.model_config.multimodal_config = SimpleNamespace(
        language_model_only=language_model_only
    )
    expected = (
        nullcontext()
        if language_model_only
        else pytest.raises(ValueError, match="decoder-only text")
    )

    with expected:
        hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)


def test_runtime_rejects_sequence_parallel_compilation_pass(
    hccl_config_module, runtime, afd_config
):
    runtime.compilation_config.pass_config.enable_sp = True

    with pytest.raises(ValueError, match="sequence parallelism"):
        hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)


@pytest.mark.parametrize(
    "field", ["mode", "cudagraph_mode", "use_inductor_graph_partition"]
)
def test_runtime_rejects_compilation_or_graphs(
    hccl_config_module, runtime, afd_config, field
):
    choices = {
        "mode": hccl_config_module.CompilationMode.VLLM_COMPILE,
        "cudagraph_mode": hccl_config_module.CUDAGraphMode.FULL,
        "use_inductor_graph_partition": True,
    }
    setattr(runtime.compilation_config, field, choices[field])

    with pytest.raises(ValueError, match="modes NONE"):
        hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("use_v2_model_runner", False, "ModelRunner V2"),
        (
            "kv_transfer_config",
            pd_kv_transfer_config(connector="UnsupportedKVConnector"),
            "KV transfer/PD",
        ),
        ("lora_config", True, "LoRA"),
    ],
)
def test_runtime_rejects_unsupported_optional_features(
    hccl_config_module, runtime, afd_config, field, value, reason
):
    setattr(runtime, field, value)

    with pytest.raises(ValueError, match=reason):
        hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)


@pytest.fixture
def dspark_runtime(hccl_config_module):
    return hccl_vllm_config(
        dtype=hccl_config_module.torch.bfloat16,
        compilation_mode=hccl_config_module.CompilationMode.NONE,
        cudagraph_mode=hccl_config_module.CUDAGraphMode.NONE,
        dspark=True,
    )


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("num_stages", [1, 2])
@pytest.mark.parametrize("kv_role", ["kv_producer", "kv_consumer"])
def test_eager_pd_keeps_native_attention_transfer_and_equal_private_ffn_capabilities(
    hccl_config_module, tp_size, num_stages, kv_role
):
    runtime = hccl_vllm_config(
        dtype=hccl_config_module.torch.bfloat16,
        compilation_mode=hccl_config_module.CompilationMode.NONE,
        cudagraph_mode=hccl_config_module.CUDAGraphMode.NONE,
        tensor_parallel_size=tp_size,
        num_ubatches=num_stages,
    )
    native_config = pytest.importorskip("vllm.config")
    runtime.kv_transfer_config = native_config.KVTransferConfig(
        **vars(pd_kv_transfer_config(tp_size=tp_size)),
        kv_buffer_device="npu",
    )
    runtime.kv_transfer_config.kv_role = kv_role
    source = deepcopy(runtime)
    source.additional_config["afd"]["role"] = "ffn"
    attention_afd = hccl_config_module.parse_afd_config(runtime)
    ffn_afd = hccl_config_module.parse_afd_config(source)
    validate = (
        hccl_config_module.validate_staged_hccl_runtime
        if num_stages == 2
        else hccl_config_module.validate_hccl_runtime
    )

    attention_caps = validate(runtime, attention_afd, 0)
    with pytest.raises(ValueError, match="KV transfer/PD off on FFN"):
        validate(source, ffn_afd, 0)
    private = get_npu_ffn_config(source)
    ffn_caps = validate(private, ffn_afd, 0)

    assert ffn_caps == attention_caps
    assert private.kv_transfer_config is None
    assert (
        runtime.kv_transfer_config is not None and source.kv_transfer_config is not None
    )
    assert private.model_config is source.model_config
    assert private.parallel_config is source.parallel_config


@pytest.mark.parametrize("num_stages", [1, 2])
def test_eager_pd_rejects_native_dspark_without_prompt_draft_ingress_contract(
    hccl_config_module, dspark_runtime, num_stages
):
    runtime = dspark_runtime
    runtime.kv_transfer_config = pd_kv_transfer_config()
    runtime.parallel_config.num_ubatches = num_stages
    runtime.parallel_config.enable_dbo = runtime.parallel_config.use_ubatching = (
        num_stages == 2
    )
    afd = hccl_config_module.parse_afd_config(runtime)
    before = deepcopy(runtime)

    with pytest.raises(ValueError, match="PD requires speculation off"):
        hccl_config_module.validate_staged_hccl_runtime(runtime, afd, 0)

    assert runtime == before


def test_hccl_admits_native_attention_local_dspark_components(
    hccl_config_module, dspark_runtime, afd_config
):
    capabilities = hccl_config_module.validate_hccl_runtime(
        dspark_runtime, afd_config, 0
    )

    assert capabilities.layer_indices == tuple(range(40))
    assert dspark_runtime.speculative_config.draft_model_config.architectures == [
        "DeepseekV41DSparkModel"
    ]


@pytest.mark.parametrize("num_ubatches", [1, 2])
def test_staged_dspark_target_admits_eager_attention_without_consuming_the_drafter(
    hccl_config_module, dspark_runtime, afd_config, num_ubatches
):
    parallel = dspark_runtime.parallel_config
    parallel.num_ubatches = num_ubatches
    parallel.enable_dbo = parallel.use_ubatching = num_ubatches == 2
    before = deepcopy(dspark_runtime)

    caps = hccl_config_module.validate_staged_hccl_runtime(
        dspark_runtime, afd_config, 0
    )

    assert caps.local_counts_version == 1 and caps.graph_execution_version == 0
    assert caps.layer_indices == tuple(range(40))
    assert dspark_runtime == before


@pytest.mark.parametrize("boundary", ["ffn", "checkpoint", "method"])
def test_staged_dspark_target_keeps_role_graph_and_checkpoint_boundaries(
    hccl_config_module, dspark_runtime, afd_config, boundary
):
    if boundary == "checkpoint":
        dspark_runtime.speculative_config.draft_model_config.model = "foreign-model"
    elif boundary == "method":
        dspark_runtime.speculative_config.method = "mtp"
    role = replace(afd_config, role="ffn") if boundary == "ffn" else afd_config
    reason = {
        "ffn": "speculation off on FFN",
        "checkpoint": "same target checkpoint",
        "method": "method='dspark'",
    }[boundary]
    with pytest.raises(ValueError, match=reason):
        hccl_config_module.validate_staged_hccl_runtime(dspark_runtime, role, 0)


@pytest.fixture
def dspark_graph_runtime(dspark_runtime, hccl_config_module):
    config = dspark_runtime
    config.model_config.enforce_eager = False
    config.compilation_config.cudagraph_mode = (
        hccl_config_module.CUDAGraphMode.FULL_DECODE_ONLY
    )
    config.parallel_config.rank = 0
    return config


@pytest.mark.parametrize("steps", [1, 2, 5])
@pytest.mark.parametrize("order", ["stage_serial", "bulk_v1", "cross_layer_v1"])
def test_graph_peers_retain_target_width_after_private_ffn_isolation(
    hccl_config_module, dspark_graph_runtime, steps, order
):
    source = dspark_graph_runtime
    source.speculative_config.num_speculative_tokens = steps
    source.additional_config["afd"]["connector_extra_config"] = {
        "graph_exchange_order": order
    }
    ffn = deepcopy(source)
    ffn.additional_config["afd"]["role"] = "ffn"
    shape = resolve_deepseek_v41_target_shape(ffn)
    private = get_npu_ffn_config(ffn)
    a_caps = hccl_config_module.validate_staged_hccl_runtime(
        source, hccl_config_module.parse_afd_config(source), 0, graph=True
    )
    f_caps = hccl_config_module.validate_staged_hccl_runtime(
        private,
        hccl_config_module.parse_afd_config(private),
        0,
        graph=True,
        target_shape=shape,
    )
    assert a_caps == f_caps
    assert a_caps.graph_exchange_order == order
    assert a_caps.target_query_len == steps + 1
    assert private.speculative_config is private.engram_config is None
    assert source.speculative_config.num_speculative_tokens == steps


@pytest.mark.parametrize("provided_query_len", [1, 2, 7])
def test_graph_attention_rejects_a_foreign_retained_target_shape(
    hccl_config_module, dspark_graph_runtime, afd_config, provided_query_len
):
    with pytest.raises(ValueError, match="match Attention config"):
        hccl_config_module.validate_staged_hccl_runtime(
            dspark_graph_runtime,
            afd_config,
            0,
            graph=True,
            target_shape=DeepseekV41TargetShape(provided_query_len),
        )


def test_graph_target_width_cannot_exceed_the_transfer_ceiling(
    hccl_config_module, dspark_graph_runtime, afd_config
):
    narrow = replace(afd_config, connector_extra_config={"max_tokens": 4})
    with pytest.raises(ValueError, match="exceeds native token capacity"):
        hccl_config_module.validate_staged_hccl_runtime(
            dspark_graph_runtime, narrow, 0, graph=True
        )


def test_hccl_rejects_classic_mtp(hccl_config_module, dspark_runtime, afd_config):
    dspark_runtime.speculative_config.method = "mtp"

    with pytest.raises(ValueError, match="method='dspark'"):
        hccl_config_module.validate_hccl_runtime(dspark_runtime, afd_config, 0)


def test_hccl_ffn_requires_the_draft_free_role_view(
    hccl_config_module, dspark_runtime, afd_config
):
    with pytest.raises(ValueError, match="speculation off on FFN"):
        hccl_config_module.validate_hccl_runtime(
            dspark_runtime, replace(afd_config, role="ffn"), 0
        )


@pytest.mark.parametrize("capacity", [0, -1, True, 4.0])
def test_runtime_rejects_invalid_scheduler_capacity(
    hccl_config_module, runtime, afd_config, capacity
):
    runtime.scheduler_config.max_num_batched_tokens = capacity

    with pytest.raises(ValueError, match="scheduler"):
        hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)


@pytest.mark.parametrize("layer_count", [0, -1, True, 3.0, 4_097])
def test_runtime_rejects_invalid_layer_count(
    hccl_config_module, runtime, afd_config, layer_count
):
    runtime.model_config.hf_config.num_hidden_layers = layer_count

    with pytest.raises(ValueError, match="num_hidden_layers"):
        hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)
