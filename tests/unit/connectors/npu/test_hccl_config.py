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

from afd_plugin.config import (
    HCCL_P2P_CONNECTOR,
    SUPPORTED_AFD_CONNECTORS,
    AFDConfig,
    afd_config_from_mapping,
    validate_afd_config,
)
from tests.unit.fakes.vllm_config import hccl_vllm_config


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


@pytest.mark.parametrize("raw", [None, {}, {"max_tokens": None}, {"max_tokens": 4}])
def test_extra_config_round_trip(hccl_config_module, raw):
    extra = hccl_config_module.HcclExtraInfo.from_mapping(raw)

    assert extra.max_tokens == (raw or {}).get("max_tokens")
    assert hccl_config_module.HcclExtraInfo.from_mapping(extra.to_mapping()) == extra


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
        ("speculative_config", True, "speculative decoding"),
        ("kv_transfer_config", True, "KV transfer/PD"),
        ("lora_config", True, "LoRA"),
    ],
)
def test_runtime_rejects_unsupported_optional_features(
    hccl_config_module, runtime, afd_config, field, value, reason
):
    setattr(runtime, field, value)

    with pytest.raises(ValueError, match=reason):
        hccl_config_module.validate_hccl_runtime(runtime, afd_config, 0)


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
