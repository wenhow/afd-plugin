# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from afd_plugin.compat.npu.feature_validation import (
    _fail_if_unsupported_dspark_features,
)
from afd_plugin.config import parse_afd_config
from afd_plugin.validation import (
    ATTENTION_WORKER_FQCN,
    FFN_WORKER_FQCN,
    NPU_ATTENTION_WORKER_FQCN,
    NPU_FFN_WORKER_FQCN,
    assert_compatible_afd_stack,
    validate_gpu_model_runner_v2_config,
    validate_npu_model_runner_v2_config,
)
from tests.unit.fakes.vllm_config import window_dspark_config

MODEL_UTILS_MODULE = "afd_plugin.model_executor.models.model_utils"


@pytest.mark.parametrize("stages", [1, 2, 4])
@pytest.mark.parametrize(
    "target_eager,draft_eager",
    [
        (True, True),
        (False, True),
        (False, False),
        (True, False),
    ],
)
def test_dspark_mode_policy_is_independent_of_target_stages(
    stages,
    target_eager,
    draft_eager,
):
    config = window_dspark_config(stages, target_eager, draft_eager)
    if target_eager and not draft_eager:
        with pytest.raises(RuntimeError, match="draft graphs require a graph target"):
            _fail_if_unsupported_dspark_features(config, parse_afd_config(config))
    else:
        _fail_if_unsupported_dspark_features(config, parse_afd_config(config))


@pytest.mark.parametrize("stages", [1, 2])
def test_graph_verification_requires_fixed_width_independent_of_stages(stages):
    config = window_dspark_config(stages, False, False)
    config.speculative_config.enable_adaptive_verification = True
    with pytest.raises(RuntimeError, match="fixed draft width"):
        _fail_if_unsupported_dspark_features(config, parse_afd_config(config))


def _vllm_like_config(*, afd, worker_cls):
    return SimpleNamespace(
        additional_config={} if afd is None else {"afd": afd},
        parallel_config=SimpleNamespace(worker_cls=worker_cls),
    )


@pytest.mark.parametrize(
    "afd, worker_cls, worker_override",
    [
        pytest.param(
            {"role": "attention"},
            ATTENTION_WORKER_FQCN,
            None,
            id="attention-matching-worker",
        ),
        pytest.param({"role": "ffn"}, FFN_WORKER_FQCN, None, id="ffn-matching-worker"),
        pytest.param(
            {"role": "attention", "connector": "CAMP2pAFDConnector"},
            NPU_ATTENTION_WORKER_FQCN,
            NPU_ATTENTION_WORKER_FQCN,
            id="npu-worker-override",
        ),
        pytest.param(
            {"role": "attention", "connector": "CAMAsyncAFDConnector"},
            NPU_ATTENTION_WORKER_FQCN,
            None,
            id="async-connector-npu-attention-worker",
        ),
        pytest.param(
            {"role": "ffn", "connector": "CAMAsyncAFDConnector"},
            NPU_FFN_WORKER_FQCN,
            None,
            id="async-connector-npu-ffn-worker",
        ),
    ],
)
def test_stack_validation_accepts(afd, worker_cls, worker_override):
    config = assert_compatible_afd_stack(
        _vllm_like_config(afd=afd, worker_cls=worker_cls),
        caller="test",
        expected_role=afd["role"],
        expected_worker_qualname_override=worker_override,
    )

    assert config.role == afd["role"]
    if "connector" in afd:
        assert config.connector == afd["connector"]


@pytest.mark.parametrize(
    "afd, worker_cls, expected_fragments",
    [
        pytest.param(
            None,
            ATTENTION_WORKER_FQCN,
            ("requires additional_config",),
            id="missing-afd-config",
        ),
        pytest.param(
            {"role": "ffn"},
            ATTENTION_WORKER_FQCN,
            ("invalid worker class", "remove --worker-cls"),
            id="wrong-worker",
        ),
        pytest.param(
            {"role": "attention"},
            "auto",
            ("remained 'auto'", "ensure the AFD general plugin is loaded"),
            id="auto-worker",
        ),
        pytest.param(
            {"role": "attention", "connector": "CAMAsyncAFDConnector"},
            ATTENTION_WORKER_FQCN,
            ("requires Ascend NPU worker",),
            id="async-connector-gpu-worker",
        ),
    ],
)
def test_stack_validation_rejects(afd, worker_cls, expected_fragments):
    with pytest.raises(ValueError) as exc_info:
        assert_compatible_afd_stack(
            _vllm_like_config(afd=afd, worker_cls=worker_cls),
            caller="test",
            expected_role=None if afd is None else afd["role"],
        )

    for fragment in expected_fragments:
        assert fragment in str(exc_info.value)


@pytest.fixture
def mrv2_config(monkeypatch):
    # Model registration is tested with the real vLLM runtime elsewhere.
    # Isolate that import so deployment policy tests remain CPU/dependency-safe.
    model_utils = ModuleType(MODEL_UTILS_MODULE)
    monkeypatch.setattr(
        model_utils, "has_afd_model_registration", lambda config: True, raising=False
    )
    monkeypatch.setitem(sys.modules, model_utils.__name__, model_utils)

    return SimpleNamespace(
        use_v2_model_runner=True,
        additional_config={
            "afd": {
                "role": "attention",
                "num_attention_ranks": 2,
                "num_ffn_ranks": 2,
            }
        },
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=True,
            enable_elastic_ep=False,
            enable_eplb=False,
            use_sequence_parallel_moe=False,
            enable_dbo=True,
            use_ubatching=True,
            num_ubatches=2,
        ),
        model_config=SimpleNamespace(
            enforce_eager=True,
            hf_config=SimpleNamespace(architectures=["Qwen3MoeForCausalLM"]),
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode="FULL_DECODE_ONLY",
            pass_config=SimpleNamespace(enable_sp=False),
        ),
        speculative_config=None,
    )


# backend -> (validator, device_type, AFD config defaults for that backend)
MRV2_BACKENDS = {
    "gpu": (validate_gpu_model_runner_v2_config, "cuda", {}),
    "npu": (
        validate_npu_model_runner_v2_config,
        "npu",
        {"connector": "CAMP2pAFDConnector"},
    ),
}


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_routed_transport_preserves_runner_and_gate_policy(mrv2_config, role):
    mrv2_config.additional_config["afd"].update(
        role=role, connector="DeepEPM2NAFDConnector", compute_gate_on_attention=True
    )
    validate_gpu_model_runner_v2_config(
        mrv2_config, expected_role=role, device_type="cuda"
    )
    mrv2_config.additional_config["afd"]["compute_gate_on_attention"] = False
    with pytest.raises(RuntimeError, match="requires compute_gate_on_attention=true"):
        validate_gpu_model_runner_v2_config(
            mrv2_config, expected_role=role, device_type="cuda"
        )


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize(
    "architecture, accepted",
    [
        pytest.param("Qwen3MoeForCausalLM", True, id="qwen3-moe"),
        # Runners validate the wrapped model's AFD-prefixed architecture.
        pytest.param("AFDQwen3MoeForCausalLM", True, id="afd-qwen3-moe"),
        pytest.param("DeepseekV2ForCausalLM", False, id="deepseek-v2"),
        pytest.param("DeepseekV3ForCausalLM", False, id="deepseek-v3"),
        pytest.param("AFDDeepseekV3ForCausalLM", False, id="afd-deepseek-v3"),
        pytest.param("Qwen3_5MoeForConditionalGeneration", False, id="qwen3-5-moe"),
    ],
)
def test_routed_transport_requires_model_that_routes_on_attention(
    monkeypatch, mrv2_config, role, architecture, accepted
):
    mrv2_config.additional_config["afd"].update(
        role=role, connector="DeepEPM2NAFDConnector", compute_gate_on_attention=True
    )
    mrv2_config.model_config.hf_config.architectures = [architecture]

    if accepted:
        validate_gpu_model_runner_v2_config(
            mrv2_config, expected_role=role, device_type="cuda"
        )
        return
    # Workers validate before the connector joins the AFD world; rejecting a
    # model must not need the model runtime.
    monkeypatch.setitem(sys.modules, MODEL_UTILS_MODULE, None)
    with pytest.raises(RuntimeError, match="DeepEPM2NAFDConnector supports only"):
        validate_gpu_model_runner_v2_config(
            mrv2_config, expected_role=role, device_type="cuda"
        )


def _symmetric_ranks(dp_size, tp_size):
    return {
        "parallel_config.data_parallel_size": dp_size,
        "parallel_config.tensor_parallel_size": tp_size,
        "afd.num_attention_ranks": dp_size * tp_size,
        "afd.num_ffn_ranks": dp_size * tp_size,
    }


GRAPH = {"model_config.enforce_eager": False}
V1_RUNNER = {"use_v2_model_runner": False}
PIPELINE_PARALLEL = {"parallel_config.pipeline_parallel_size": 2}


def _run_mrv2_validator(config, backend, role, overrides):
    validator, device_type, afd_defaults = MRV2_BACKENDS[backend]
    afd = config.additional_config["afd"]
    afd.update(afd_defaults, role=role)
    for path, value in overrides.items():
        if path.startswith("afd."):
            afd[path.removeprefix("afd.")] = value
            continue
        *parents, attr = path.split(".")
        target = config
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, attr, value)
    validator(config, expected_role=role, device_type=device_type)


@pytest.mark.parametrize(
    "backend, role, overrides",
    [
        pytest.param("gpu", "attention", {}, id="gpu-attention-eager"),
        pytest.param("gpu", "ffn", {}, id="gpu-ffn-eager"),
        pytest.param("gpu", "attention", GRAPH, id="gpu-attention-full-decode-graph"),
        pytest.param("gpu", "ffn", GRAPH, id="gpu-ffn-full-decode-graph"),
        pytest.param(
            "gpu", "attention", _symmetric_ranks(1, 1), id="gpu-attention-dp1-tp1"
        ),
        pytest.param("gpu", "ffn", _symmetric_ranks(1, 2), id="gpu-ffn-dp1-tp2"),
        pytest.param(
            "gpu",
            "attention",
            {**_symmetric_ranks(2, 2), **GRAPH},
            id="gpu-attention-dp2-tp2-graph",
        ),
        pytest.param(
            "gpu",
            "attention",
            {"afd.compute_gate_on_attention": True},
            id="gpu-compute-gate-on-attention",
        ),
        pytest.param(
            "gpu",
            "attention",
            {"afd.num_ffn_ranks": 1},
            id="gpu-asymmetric-attention-dp2",
        ),
        pytest.param(
            "gpu",
            "ffn",
            {"afd.num_ffn_ranks": 1, "parallel_config.data_parallel_size": 1},
            id="gpu-asymmetric-ffn-dp1",
        ),
        pytest.param("npu", "attention", {}, id="npu-sync-dbo-attention-eager"),
        pytest.param("npu", "ffn", {}, id="npu-sync-dbo-ffn-eager"),
        pytest.param("npu", "attention", GRAPH, id="npu-sync-dbo-attention-full-graph"),
        pytest.param("npu", "ffn", GRAPH, id="npu-sync-dbo-ffn-full-graph"),
    ],
)
def test_mrv2_validator_accepts_supported_deployment(
    mrv2_config, backend, role, overrides
):
    _run_mrv2_validator(mrv2_config, backend, role, overrides)


@pytest.mark.parametrize(
    "backend, role, overrides, expected_error",
    [
        pytest.param(
            "gpu",
            "attention",
            V1_RUNNER,
            "AFD GPU requires ModelRunnerV2",
            id="gpu-v1-attention",
        ),
        pytest.param(
            "gpu", "ffn", V1_RUNNER, "AFD GPU requires ModelRunnerV2", id="gpu-v1-ffn"
        ),
        pytest.param(
            "gpu",
            "attention",
            {"parallel_config.num_ubatches": 1},
            "exactly two microbatches",
            id="gpu-one-ubatch",
        ),
        pytest.param(
            "gpu",
            "attention",
            {"parallel_config.num_ubatches": 3},
            "exactly two microbatches",
            id="gpu-three-ubatches",
        ),
        pytest.param(
            "npu",
            "attention",
            V1_RUNNER,
            "AFD NPU requires ModelRunnerV2",
            id="npu-v1-attention",
        ),
        pytest.param(
            "npu", "ffn", V1_RUNNER, "AFD NPU requires ModelRunnerV2", id="npu-v1-ffn"
        ),
        pytest.param(
            "npu",
            "attention",
            PIPELINE_PARALLEL,
            "does not support PP or CP",
            id="npu-pipeline-parallel-attention",
        ),
        pytest.param(
            "npu",
            "ffn",
            PIPELINE_PARALLEL,
            "does not support PP or CP",
            id="npu-pipeline-parallel-ffn",
        ),
    ],
)
def test_mrv2_validator_rejects_before_importing_model_runtime(
    monkeypatch, mrv2_config, backend, role, overrides, expected_error
):
    # Unsupported deployments must fail before importing or constructing a
    # runner. Block the runtime module so an import before rejection fails
    # visibly even when vLLM is installed in the test environment.
    monkeypatch.setitem(sys.modules, MODEL_UTILS_MODULE, None)

    with pytest.raises(RuntimeError, match=expected_error):
        _run_mrv2_validator(mrv2_config, backend, role, overrides)


@pytest.mark.parametrize(
    "connector, role, method, slots, allowed",
    [
        ("WindowAFDConnector", "attention", "dspark", 2, True),
        ("WindowAFDConnector", "attention", "mtp", 2, False),
        ("WindowAFDConnector", "attention", "dspark", 1, False),
        ("WindowAFDConnector", "ffn", "dspark", 2, False),
        ("CAMP2pAFDConnector", "attention", "dspark", 2, False),
    ],
)
def test_npu_dbo_speculative_deployment(
    mrv2_config, connector, role, method, slots, allowed
):
    overrides = {
        **GRAPH,
        **_symmetric_ranks(4, 1),
        "afd.connector": connector,
        "afd.async": connector == "WindowAFDConnector",
        "afd.compute_gate_on_attention": connector == "WindowAFDConnector",
        "afd.connector_extra_config": {"micro_batch_num": slots},
        "model_config.hf_config.architectures": ["DeepseekV41ForCausalLM"],
        "speculative_config": SimpleNamespace(method=method, enforce_eager=True),
    }
    if allowed:
        _run_mrv2_validator(mrv2_config, "npu", role, overrides)
    else:
        with pytest.raises(RuntimeError, match="only for Window Attention with DSpark"):
            _run_mrv2_validator(mrv2_config, "npu", role, overrides)


def test_window_dspark_local_target_does_not_require_multiple_attention_dp_ranks(
    mrv2_config,
):
    overrides = {
        **GRAPH,
        **_symmetric_ranks(1, 1),
        "afd.num_ffn_ranks": 4,
        "afd.connector": "WindowAFDConnector",
        "afd.async": True,
        "afd.compute_gate_on_attention": True,
        "afd.connector_extra_config": {"micro_batch_num": 2},
        "model_config.hf_config.architectures": ["DeepseekV41ForCausalLM"],
        "speculative_config": SimpleNamespace(method="dspark", enforce_eager=False),
    }
    _run_mrv2_validator(mrv2_config, "npu", "attention", overrides)
