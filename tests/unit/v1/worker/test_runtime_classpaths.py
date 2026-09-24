from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from afd_plugin.validation import (
    ATTENTION_MODEL_RUNNER_FQCN,
    ATTENTION_WORKER_FQCN,
    FFN_MODEL_RUNNER_FQCN,
    FFN_WORKER_FQCN,
    NPU_ATTENTION_MODEL_RUNNER_FQCN,
    NPU_ATTENTION_WORKER_FQCN,
    NPU_FFN_MODEL_RUNNER_FQCN,
    NPU_FFN_WORKER_FQCN,
    UBATCH_WRAPPER_FQCN,
    resolve_class_from_qualname,
)

GPU_RUNTIME_CLASS_PATHS = [
    ATTENTION_WORKER_FQCN,
    ATTENTION_MODEL_RUNNER_FQCN,
    FFN_WORKER_FQCN,
    FFN_MODEL_RUNNER_FQCN,
    UBATCH_WRAPPER_FQCN,
    "afd_plugin.v1.worker:AFDAttentionWorker",
]

NPU_RUNTIME_CLASS_PATHS = [
    NPU_ATTENTION_WORKER_FQCN,
    NPU_ATTENTION_MODEL_RUNNER_FQCN,
    NPU_FFN_WORKER_FQCN,
    NPU_FFN_MODEL_RUNNER_FQCN,
]

V026_OVERRIDE_CONTRACTS = [
    ("attention_worker", "AFDAttentionWorker", "Worker", "__init__"),
    ("attention_worker", "AFDAttentionWorker", "Worker", "init_device"),
    ("ffn_worker", "AFDFFNWorker", "Worker", "__init__"),
    ("ffn_worker", "AFDFFNWorker", "Worker", "init_device"),
    ("ffn_worker", "AFDFFNWorker", "Worker", "get_kv_cache_spec"),
    ("ffn_worker", "AFDFFNWorker", "Worker", "initialize_from_config"),
    ("ffn_worker", "AFDFFNWorker", "Worker", "compile_or_warm_up_model"),
    ("ffn_worker", "AFDFFNWorker", "Worker", "execute_model"),
    ("ffn_worker", "AFDFFNWorker", "Worker", "shutdown"),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "load_model",
    ),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "_build_attention_metadata",
    ),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "_determine_batch_execution_and_padding",
    ),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "_model_forward",
    ),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "execute_model",
    ),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "_dummy_run",
    ),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "_warmup_and_capture",
    ),
    (
        "attention_model_runner",
        "AFDAttentionModelRunner",
        "GPUModelRunner",
        "shutdown",
    ),
    ("ubatch_wrapper", "AFDUBatchWrapper", "UBatchWrapper", "__init__"),
    (
        "ubatch_wrapper",
        "AFDUBatchWrapper",
        "UBatchWrapper",
        "_create_sm_control_context",
    ),
    (
        "ubatch_wrapper",
        "AFDUBatchWrapper",
        "UBatchWrapper",
        "_make_ubatch_metadata",
    ),
]


def _call_contract(callable_obj):
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(callable_obj).parameters.values()
    ]


@pytest.mark.parametrize(
    "qualname",
    GPU_RUNTIME_CLASS_PATHS + NPU_RUNTIME_CLASS_PATHS,
)
def test_runtime_class_paths_are_plugin_paths(qualname):
    assert qualname.startswith("afd_plugin.v1.worker")


@pytest.mark.vllm_runtime
@pytest.mark.parametrize("qualname", GPU_RUNTIME_CLASS_PATHS)
def test_gpu_runtime_class_paths_resolve_when_vllm_is_available(qualname):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")

    cls = resolve_class_from_qualname(qualname)

    assert isinstance(cls, type)
    assert cls.__module__.startswith("afd_plugin.v1.worker")


@pytest.mark.vllm_runtime
@pytest.mark.parametrize(
    ("module_name", "afd_class_name", "native_class_name", "method_name"),
    V026_OVERRIDE_CONTRACTS,
)
def test_gpu_v1_overrides_match_native_call_contract(
    module_name,
    afd_class_name,
    native_class_name,
    method_name,
):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")

    if module_name == "attention_worker":
        from vllm.v1.worker import gpu_worker as native_module

        from afd_plugin.v1.worker import attention_worker as afd_module
    elif module_name == "ffn_worker":
        from vllm.v1.worker import gpu_worker as native_module

        from afd_plugin.v1.worker import ffn_worker as afd_module
    elif module_name == "attention_model_runner":
        from vllm.v1.worker import gpu_model_runner as native_module

        from afd_plugin.v1.worker import attention_model_runner as afd_module
    else:
        from vllm.v1.worker import gpu_ubatch_wrapper as native_module

        from afd_plugin.v1.worker import ubatch_wrapper as afd_module

    afd_method = getattr(getattr(afd_module, afd_class_name), method_name)
    native_method = getattr(getattr(native_module, native_class_name), method_name)

    afd_contract = _call_contract(afd_method)
    native_contract = _call_contract(native_method)
    if method_name == "_warmup_and_capture" and len(afd_contract) == len(
        native_contract
    ) + 1:
        # vLLM 0.26 added an optional profiler argument. Keep it in the plugin
        # signature so one implementation remains callable on 0.23 and 0.26.
        assert afd_contract[:-1] == native_contract
        assert afd_contract[-1][0] == "profiler"
        assert afd_contract[-1][2] is None
    else:
        assert afd_contract == native_contract


@pytest.mark.vllm_runtime
@pytest.mark.parametrize("qualname", NPU_RUNTIME_CLASS_PATHS)
def test_npu_runtime_class_paths_resolve_when_vllm_ascend_is_available(qualname):
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    pytest.importorskip("vllm_ascend")

    cls = resolve_class_from_qualname(qualname)

    assert isinstance(cls, type)
    assert cls.__module__.startswith("afd_plugin.v1.worker.npu")


def test_npu_attention_backend_override_matches_native_call_contract():
    source_path = (
        Path(__file__).parents[4]
        / "afd_plugin/v1/worker/npu/attention_model_runner.py"
    )
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    runner_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "AFDNPUAttentionModelRunner"
    )
    method = next(
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "initialize_attn_backend"
    )
    assert [argument.arg for argument in method.args.args] == [
        "self",
        "kv_cache_config",
    ]
    assert method.args.defaults == []
    parent_call = next(
        call
        for call in ast.walk(method)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "initialize_attn_backend"
    )
    assert len(parent_call.args) == 1
    assert isinstance(parent_call.args[0], ast.Name)
    assert parent_call.args[0].id == "kv_cache_config"
    assert parent_call.keywords == []
