# SPDX-License-Identifier: Apache-2.0
"""Native-control backend selection contracts; no NPU execution."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from afd_plugin.model_executor.models.npu.dsv41_packed_cache import (  # noqa: E402
    AFDMixedQuantPackedCacheOps,
    native,
)
from afd_plugin.v1.worker.npu.dsv41_native_worker import (  # noqa: E402
    DSV41NativeTP4ControlWorker,
    NPUWorker,
)
from tests.unit.fakes.dsv41_packed_cache import make_native_control_case  # noqa: E402


@pytest.fixture
def native_control_case(monkeypatch):
    case = make_native_control_case(
        DSV41NativeTP4ControlWorker, native.MixedQuantPackedCacheOps
    )

    def load(owner):
        owner.model_runner = SimpleNamespace(model=case.model)

    monkeypatch.setattr(NPUWorker, "load_model", load)
    return case


def test_native_control_changes_only_its_attention_backend(native_control_case):
    case = native_control_case
    native_qsmla = native.MixedQuantPackedCacheOps.qsmla
    case.worker.load_model()

    assert case.model["attention"].packed_cache_ops is AFDMixedQuantPackedCacheOps
    assert case.model["unrelated"].packed_cache_ops is native.MixedQuantPackedCacheOps
    assert native.MixedQuantPackedCacheOps.qsmla is native_qsmla


@pytest.mark.parametrize("backend", [None, type])
def test_native_control_does_not_replace_other_backends(native_control_case, backend):
    case = native_control_case
    case.model["attention"].packed_cache_ops = backend

    with pytest.raises(ValueError, match="no A5 packed Attention"):
        case.worker.load_model()
    assert case.model["attention"].packed_cache_ops is backend


def test_native_load_failure_keeps_original_error(native_control_case, monkeypatch):
    def failed_load(owner):
        raise RuntimeError("original weight load failure")

    monkeypatch.setattr(NPUWorker, "load_model", failed_load)
    with pytest.raises(RuntimeError, match="original weight load failure"):
        native_control_case.worker.load_model()
    assert (
        native_control_case.model["attention"].packed_cache_ops
        is native.MixedQuantPackedCacheOps
    )


def test_native_control_rejects_other_tp_sizes_before_loading(native_control_case):
    native_control_case.worker.vllm_config.parallel_config.tensor_parallel_size = 1
    with pytest.raises(ValueError, match="requires TP4"):
        native_control_case.worker.load_model()
