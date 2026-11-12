# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU AFD feature validation rejects unsupported combinations before execution."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest

from afd_plugin.compat.npu import (
    fail_if_unsupported_npu_afd_features,
    npu_afd_num_ubatches,
)
from afd_plugin.compat.npu.runtime_config import get_npu_ffn_config
from afd_plugin.config import AFDConfig
from tests.unit.fakes.vllm_config import (
    async_moe_config as _async_moe_config,
)
from tests.unit.fakes.vllm_config import (
    npu_vllm_config as _vllm_config,
)
from tests.unit.fakes.vllm_config import (
    v41_dspark_config,
    v41_eager_qualification_config,
)

TWO_UBATCH_DBO = dict(
    enable_dbo=True, use_ubatching=True, num_ubatches=2, ubatch_size=4
)
ASYNC_CAM = dict(connector="CAMAsyncAFDConnector", async_dp=True)


@pytest.fixture(autouse=True)
def clear_v41_qualification_environment(monkeypatch):
    monkeypatch.delenv("AFD_DSV41_EAGER_QUALIFICATION", raising=False)


@pytest.mark.parametrize(
    "role,connector,model_type,message",
    [
        ("attention", "CAMAsyncAFDConnector", "deepseek_v4", "FFN role"),
        ("ffn", "CAMP2pAFDConnector", "deepseek_v4", "async CAM"),
        ("ffn", "CAMAsyncAFDConnector", "deepseek_v2", "only DeepSeek V4"),
    ],
)
def test_layered_switch_rejects_other_roles_connectors_and_models(
    monkeypatch, role, connector, model_type, message
):
    config = _vllm_config(role=role, connector=connector, model_type=model_type)
    afd_config = AFDConfig(role=role, connector=connector)
    monkeypatch.setenv("AFD_ASYNC_CAM_LAYERED_GMM", "1")

    with pytest.raises(RuntimeError, match=message):
        fail_if_unsupported_npu_afd_features(config, afd_config=afd_config)


@pytest.fixture
def vllm_connector_runtime():
    # Only connector parsing requires vLLM; early config rejection is CPU safe.
    pytest.importorskip("vllm")


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize(
    "connector", ["CAMP2pAFDConnector", "CAMAsyncAFDConnector", "P2pNcclAFDConnector"]
)
@pytest.mark.parametrize(
    "model_type,architecture",
    [
        ("deepseek_v41", None),
        ("deepseek_v41", "DeepseekV41ForCausalLM"),
        ("custom", "DeepseekV41ForCausalLM"),
        ("custom", "AFDDeepseekV41ForCausalLM"),
        ("deepseek_v4", "DeepseekV41ForCausalLM"),
    ],
)
def test_npu_validation_rejects_v41_before_connector_initialization(
    role, connector, model_type, architecture
):
    config = _vllm_config(role=role, connector=connector)
    config.model_config.hf_config.model_type = model_type
    config.model_config.hf_config.architectures = (
        [architecture] if architecture else None
    )
    before = deepcopy(config)
    with pytest.raises(RuntimeError, match="DeepSeek V4.1 NPU AFD is not available"):
        fail_if_unsupported_npu_afd_features(config)
    assert config == before


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_dspark_components_do_not_enable_full_v41_model_execution(role):
    config = v41_dspark_config(role=role)
    before = deepcopy(config)

    with pytest.raises(RuntimeError, match="DeepSeek V4.1 NPU AFD is not available"):
        fail_if_unsupported_npu_afd_features(config)

    assert config == before


@pytest.fixture
def v41_eager_candidate():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.config import CompilationMode
    from vllm.config.compilation import CUDAGraphMode

    def build(*, role="attention", rank=0, tensor_parallel_size=4):
        return v41_eager_qualification_config(
            dtype=torch.bfloat16,
            compilation_mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
            role=role,
            rank=rank,
            tensor_parallel_size=tensor_parallel_size,
        )

    return build


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("rank", [0, 3])
@pytest.mark.parametrize("load_format", ["auto", "safetensors"])
def test_v41_eager_qualification_accepts_only_explicit_role_candidate(
    monkeypatch, v41_eager_candidate, role, rank, load_format
):
    source = v41_eager_candidate(role=role, rank=rank)
    source.load_config.load_format = load_format
    original = deepcopy(source)
    config = get_npu_ffn_config(source) if role == "ffn" else source
    before = deepcopy(config)
    monkeypatch.setenv("AFD_DSV41_EAGER_QUALIFICATION", "1")

    fail_if_unsupported_npu_afd_features(config)

    assert config == before
    assert source == original
    assert source.engram_config.cpu_offload is True
    if role == "ffn":
        assert config.engram_config is None


@pytest.mark.parametrize("flag", [None, "0"])
def test_v41_eager_candidate_remains_rejected_by_default(
    monkeypatch, v41_eager_candidate, flag
):
    if flag is not None:
        monkeypatch.setenv("AFD_DSV41_EAGER_QUALIFICATION", flag)
    with pytest.raises(RuntimeError, match="DeepSeek V4.1 NPU AFD is not available"):
        fail_if_unsupported_npu_afd_features(v41_eager_candidate())


@pytest.mark.parametrize("flag", ["", "true", "01", "2"])
def test_v41_eager_qualification_requires_an_exact_flag(monkeypatch, flag):
    monkeypatch.setenv("AFD_DSV41_EAGER_QUALIFICATION", flag)
    with pytest.raises(
        ValueError, match="AFD_DSV41_EAGER_QUALIFICATION must be 0 or 1"
    ):
        fail_if_unsupported_npu_afd_features(v41_dspark_config())


@pytest.mark.parametrize(
    "container,field,value,message",
    [
        (None, "use_v2_model_runner", False, "ModelRunner V2"),
        ("parallel_config", "rank", -1, "role_rank"),
        ("parallel_config", "rank", 4, "role_rank"),
        ("parallel_config", "rank", True, "role_rank"),
        ("parallel_config", "tensor_parallel_size", 2, "tensor_parallel_size"),
        ("parallel_config", "data_parallel_size", 2, "data_parallel_size"),
        ("parallel_config", "enable_expert_parallel", True, "expert parallelism"),
        ("parallel_config", "use_sequence_parallel_moe", True, "sequence parallelism"),
        ("parallel_config", "enable_dbo", True, "eager U1"),
        ("model_config", "enforce_eager", False, "enforce_eager"),
        ("model_config", "is_multimodal_model", True, "decoder-only text"),
        ("model_config", "enable_sleep_mode", True, "sleep/weight updates"),
        ("model_config", "quantization", "fp8", "deepseek_v4_fp8"),
        (None, "quant_config", None, "deepseek_v4_fp8"),
        ("load_config", "load_format", "dummy", "real auto/safetensors"),
        ("scheduler_config", "max_num_seqs", 2, "one sequence"),
        ("scheduler_config", "async_scheduling", True, "sync scheduling"),
        ("scheduler_config", "async_scheduling", None, "sync scheduling"),
        ("scheduler_config", "max_num_batched_tokens", 1025, "limits <= 1024"),
        ("model_config", "max_model_len", 1025, "limits <= 1024"),
        ("model_config", "max_model_len", 0, "limits <= 1024"),
        ("cache_config", "enable_prefix_caching", True, "prefix caching off"),
        (None, "speculative_config", SimpleNamespace(), "speculation off"),
        (None, "kv_transfer_config", SimpleNamespace(), "KV transfer/PD off"),
        (None, "weight_transfer_config", SimpleNamespace(), "sleep/weight updates"),
        (None, "lora_config", SimpleNamespace(), "LoRA"),
        (None, "engram_config", None, "Attention CPU Engram"),
        ("engram_config", "cpu_offload", False, "Attention CPU Engram"),
        ("engram_config", "dp_shared_memory", True, "Attention CPU Engram"),
        ("engram_config", "embedding_across_dp", True, "Attention CPU Engram"),
    ],
)
def test_v41_eager_qualification_rejects_configuration_outside_candidate(
    monkeypatch, v41_eager_candidate, container, field, value, message
):
    config = v41_eager_candidate()
    owner = config if container is None else vars(config)[container]
    setattr(owner, field, value)
    before = deepcopy(config)
    monkeypatch.setenv("AFD_DSV41_EAGER_QUALIFICATION", "1")

    with pytest.raises(ValueError, match=message):
        fail_if_unsupported_npu_afd_features(config)

    assert config == before


def test_v41_eager_qualification_rejects_ffn_before_role_isolation(
    monkeypatch, v41_eager_candidate
):
    config = v41_eager_candidate(role="ffn")
    monkeypatch.setenv("AFD_DSV41_EAGER_QUALIFICATION", "1")
    with pytest.raises(ValueError, match="FFN Engram isolation"):
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize("tp_size", [1, 2, 8])
def test_v41_eager_qualification_rejects_other_valid_native_topologies(
    monkeypatch, v41_eager_candidate, tp_size
):
    config = v41_eager_candidate(tensor_parallel_size=tp_size)
    monkeypatch.setenv("AFD_DSV41_EAGER_QUALIFICATION", "1")
    with pytest.raises(ValueError, match="A4/F4 with native TP4"):
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    "additional_config,layered_gmm,message",
    [
        (
            {"sparse_kv_offload_config": {"enabled": True}},
            False,
            "sparse KV offload off",
        ),
        ({"enable_shared_expert_dp": True}, False, "uniform native TP"),
        (
            {"finegrained_tp_config": {"mlp_tensor_parallel_size": 2}},
            False,
            "uniform native TP",
        ),
        ({}, True, "Async CAM layered GMM off"),
    ],
)
def test_v41_eager_qualification_rejects_unqualified_runtime_flags(
    monkeypatch, v41_eager_candidate, additional_config, layered_gmm, message
):
    config = v41_eager_candidate()
    config.additional_config.update(additional_config)
    monkeypatch.setenv("AFD_DSV41_EAGER_QUALIFICATION", "1")
    monkeypatch.setenv("AFD_ASYNC_CAM_LAYERED_GMM", "1" if layered_gmm else "0")
    with pytest.raises(ValueError, match=message):
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    "model_type,architecture",
    [
        ("deepseek_v4", None),
        ("custom", "DeepseekV4ForCausalLM"),
        ("custom", "AFDDeepseekV4ForCausalLM"),
    ],
)
def test_npu_validation_rejects_v4_without_token_id_transport(model_type, architecture):
    config = _vllm_config(connector="P2pNcclAFDConnector")
    config.model_config.hf_config.model_type = model_type
    config.model_config.hf_config.architectures = (
        [architecture] if architecture else None
    )
    with pytest.raises(RuntimeError, match="DSV4 NPU AFD supports only"):
        fail_if_unsupported_npu_afd_features(config)


def _config_with(*, model=None, max_num_batched_tokens=None, **kwargs):
    config = _vllm_config(**kwargs)
    for name, value in (model or {}).items():
        setattr(config.model_config, name, value)
    if max_num_batched_tokens is not None:
        config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    return config


def _mla_dbo(cudagraph_mode, **model_fields):
    return _config_with(
        use_mla=True,
        cudagraph_mode=cudagraph_mode,
        model={
            "hf_text_config": SimpleNamespace(model_type="deepseek_v2", **model_fields)
        },
        **TWO_UBATCH_DBO,
    )


@pytest.mark.parametrize(
    ("config", "error"),
    [
        pytest.param(_config_with(model={"enforce_eager": False}), None, id="graph"),
        pytest.param(_config_with(**TWO_UBATCH_DBO), None, id="two-ubatches"),
        pytest.param(
            _config_with(**{**TWO_UBATCH_DBO, "num_ubatches": 4}),
            "exactly two ubatches",
            id="four-ubatches",
        ),
        pytest.param(
            _config_with(extra_config={"compute_gate_on_attention": True}),
            "compute_gate_on_attention",
            id="gate-on-attention",
        ),
        pytest.param(
            _config_with(extra_config={"quant_mode": 1}), "quant_mode=0", id="quant"
        ),
        pytest.param(
            _config_with(connector="P2pNcclAFDConnector", extra_config={"core_num": 8}),
            "does not support connector_extra_config",
            id="connector-extra-parser",
        ),
        pytest.param(_mla_dbo("FULL"), "FULL_DECODE_ONLY", id="mla-full"),
        pytest.param(
            _mla_dbo("FULL_AND_PIECEWISE"), "FULL_DECODE_ONLY", id="mla-mixed"
        ),
        pytest.param(_mla_dbo("FULL_DECODE_ONLY"), None, id="mla-decode"),
        pytest.param(_mla_dbo("FULL", index_topk=8), None, id="sparse"),
        # Compressor models select the DSA backend, whose graph-params update is
        # a no-op, so the MLA DBO graph rules do not apply.
        pytest.param(
            _mla_dbo("FULL", index_topk=512, compress_ratios=[0, 4, 128]),
            None,
            id="compressor",
        ),
        pytest.param(
            _config_with(
                use_mla=True,
                cudagraph_mode="FULL_DECODE_ONLY",
                speculative_config=object(),
                **TWO_UBATCH_DBO,
            ),
            "does not support speculative decoding",
            id="mla-draft",
        ),
        pytest.param(
            _config_with(**{**ASYNC_CAM, "async_dp": False}),
            "async=true",
            id="async-off",
        ),
        pytest.param(
            _config_with(model={"enforce_eager": False}, **ASYNC_CAM),
            "only eager",
            id="async-graph",
        ),
        pytest.param(
            _config_with(use_ubatching=True, **ASYNC_CAM),
            "DBO requires exactly two ubatches",
            id="async-native-ubatching",
        ),
        pytest.param(
            _config_with(enable_dbo=True, **ASYNC_CAM),
            "DBO requires exactly two ubatches",
            id="async-native-dbo",
        ),
        *(
            pytest.param(
                _config_with(
                    use_mla=True, extra_config={"dynamicQuant": value}, **ASYNC_CAM
                ),
                error,
                id=f"async-dynamic-quant-{value}",
            )
            for value, error in ((0, None), ("1", None), (2, "dynamicQuant"))
        ),
        pytest.param(
            _config_with(
                compute_gate_on_attention=True,
                use_v2_model_runner=True,
                use_mla=True,
                tensor_parallel_size=2,
                num_attention_ranks=2,
                extra_config={"attn_ranks_per_dp": 2},
                max_num_batched_tokens=5,
                **ASYNC_CAM,
            ),
            "divisible by attn_ranks_per_dp",
            id="async-mrv2-fractional-tp-capacity",
        ),
        pytest.param(_async_moe_config(), None, id="moe-request-tp1"),
        pytest.param(
            _async_moe_config(tensor_parallel_size=2, num_attention_ranks=2),
            None,
            id="moe-token-attention-tp2",
        ),
        pytest.param(
            _async_moe_config(role="ffn"),
            None,
            id="moe-token-ffn-tp1",
        ),
        pytest.param(
            _async_moe_config(compute_gate_on_attention=False),
            "compute_gate_on_attention",
            id="moe-missing-attention-gate",
        ),
        pytest.param(
            _async_moe_config(num_ubatches=3),
            "exactly two",
            id="moe-three-stages",
        ),
        pytest.param(
            _async_moe_config(decode_context_parallel_size=2),
            "CP",
            id="moe-decode-context-parallel",
        ),
        pytest.param(
            _async_moe_config(prefill_context_parallel_size=2),
            "CP",
            id="moe-prefill-context-parallel",
        ),
        pytest.param(
            _async_moe_config(tensor_parallel_size=2, attn_ranks_per_dp=1),
            "attn_ranks_per_dp",
            id="moe-attention-cam-tp-mismatch",
        ),
    ],
)
@pytest.mark.usefixtures("vllm_connector_runtime")
def test_npu_feature_validation(config, error):
    with (
        pytest.raises((RuntimeError, ValueError), match=error)
        if error
        else nullcontext()
    ):
        fail_if_unsupported_npu_afd_features(config)


def test_npu_afd_num_ubatches_reads_two_ubatch_dbo():
    assert npu_afd_num_ubatches(_vllm_config(**TWO_UBATCH_DBO)) == 2


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize(
    ("factor", "max_num_batched_tokens", "rejected"),
    [
        ("0.015625", 8192, True),
        ("0.015625", 4096, False),
        ("0.5", 8192, False),
        (None, 262144, False),
        ("0", 16, True),
        ("nan", 16, True),
        ("invalid", 16, True),
        ("0.01_5625", 4096, True),
    ],
)
@pytest.mark.usefixtures("vllm_connector_runtime")
def test_npu_async_feature_validation_checks_receive_capacity(
    monkeypatch,
    role,
    factor,
    max_num_batched_tokens,
    rejected,
):
    if factor is None:
        monkeypatch.delenv("BATCH_SIZE_FACTOR", raising=False)
    else:
        monkeypatch.setenv("BATCH_SIZE_FACTOR", factor)
    config = _vllm_config(
        role=role,
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        use_mla=True,
        use_v2_model_runner=True,
        num_attention_ranks=2,
        tensor_parallel_size=2 if role == "attention" else 1,
        extra_config={"attn_ranks_per_dp": 2},
    )
    config.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    if rejected:
        with pytest.raises(RuntimeError, match="BATCH_SIZE_FACTOR"):
            fail_if_unsupported_npu_afd_features(config)
    else:
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    "extra_config,additional_config,model_type,use_mla,error",
    [
        (
            {},
            {"enable_shared_expert_dp": True},
            "deepseek_v2",
            True,
            "shared-expert DP",
        ),
        *[
            (
                {},
                {"finegrained_tp_config": {key: 2}},
                "deepseek_v2",
                True,
                "finegrained TP",
            )
            for key in (
                "oproj_tensor_parallel_size",
                "mlp_tensor_parallel_size",
                "embedding_tensor_parallel_size",
                "lmhead_tensor_parallel_size",
            )
        ],
        ({}, {}, "qwen3_moe", True, "DeepSeek-V2/V3 MLA"),
        ({}, {}, "deepseek_v2", False, "DeepSeek-V2/V3 MLA"),
    ],
)
@pytest.mark.usefixtures("vllm_connector_runtime")
def test_npu_async_mrv2_feature_boundaries(
    extra_config, additional_config, model_type, use_mla, error
):
    config = _vllm_config(
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        use_v2_model_runner=True,
        use_mla=use_mla,
        extra_config=extra_config,
    )
    config.additional_config.update(additional_config)
    config.model_config.hf_text_config.model_type = model_type
    with pytest.raises(RuntimeError, match=error):
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize(
    "model_type,use_mla",
    [("deepseek_v2", True), ("deepseek_v3", True), ("deepseek_v4", False)],
)
@pytest.mark.parametrize("tp_size,use_sp", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("local_ubatching", [False, True])
@pytest.mark.usefixtures("vllm_connector_runtime")
def test_npu_async_mrv2_features_allow_model_layouts(
    model_type, use_mla, tp_size, use_sp, local_ubatching
):
    config = _vllm_config(
        connector="CAMAsyncAFDConnector",
        async_dp=True,
        compute_gate_on_attention=True,
        use_v2_model_runner=True,
        use_mla=use_mla,
        tensor_parallel_size=tp_size,
        use_sequence_parallel_moe=use_sp,
        use_ubatching=local_ubatching,
        num_ubatches=2,
        num_attention_ranks=2 * tp_size,
        data_parallel_size=2,
        extra_config={
            "attn_ranks_per_dp": tp_size,
            "dynamicQuant": 1,
        },
    )
    config.model_config.hf_text_config.model_type = model_type
    config.additional_config["enable_flashcomm1"] = use_sp
    fail_if_unsupported_npu_afd_features(config)
