# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU head-independence and per-Attention selection, not A5 qualification."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from afd_plugin.model_executor.models.npu import deepseek_v41  # noqa: E402
from afd_plugin.model_executor.models.npu import (  # noqa: E402
    dsv41_packed_cache as adapter,
)
from tests.unit.fakes.dsv41_packed_cache import (  # noqa: E402
    headwise_attention_with_sink,
    make_packed_attention_inputs,
    packed_64_head_attention_reference,
)


@pytest.fixture
def headwise_backend(monkeypatch):
    monkeypatch.setattr(adapter.native, "qsmla", packed_64_head_attention_reference)


@pytest.mark.parametrize("tokens", [1, 3])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tp4_keeps_headwise_attention_with_sink(
    headwise_backend, tokens, compressed, dtype
):
    inputs = make_packed_attention_inputs(
        tokens=tokens, compressed=compressed, dtype=dtype
    )
    expected = headwise_attention_with_sink(*inputs.arguments(), **inputs.options())

    output = adapter.AFDMixedQuantPackedCacheOps.qsmla(
        *inputs.arguments(), **inputs.options()
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.shape == inputs.q.shape
    assert output.dtype == torch.bfloat16
    assert output.is_contiguous()


def test_tp4_keeps_query_sinks_and_shared_cache_values(headwise_backend):
    inputs = make_packed_attention_inputs()
    tensors = (
        inputs.q,
        inputs.sinks,
        inputs.ori_kv,
        inputs.cmp_kv,
        inputs.compressed_indices,
        inputs.compressed_lengths,
        inputs.metadata.swa.ori_sparse_indices,
        inputs.metadata.swa.ori_topk_length,
    )
    snapshots = [tensor.clone() for tensor in tensors]

    adapter.AFDMixedQuantPackedCacheOps.qsmla(*inputs.arguments(), **inputs.options())

    for tensor, snapshot in zip(tensors, snapshots, strict=True):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)


def test_native_64_heads_return_the_native_output(monkeypatch):
    inputs = make_packed_attention_inputs(heads=64)
    native_output = headwise_attention_with_sink(
        *inputs.arguments(), **inputs.options()
    )
    monkeypatch.setattr(adapter.native, "qsmla", lambda *a, **kw: native_output)

    output = adapter.AFDMixedQuantPackedCacheOps.qsmla(
        *inputs.arguments(), **inputs.options()
    )

    assert output is native_output


def test_other_head_counts_keep_the_native_rejection(headwise_backend):
    inputs = make_packed_attention_inputs(heads=32)

    with pytest.raises(ValueError, match="N1 must be 64 .* got 32"):
        adapter.AFDMixedQuantPackedCacheOps.qsmla(
            *inputs.arguments(), **inputs.options()
        )


@pytest.mark.parametrize("backend", ["packed", "none", "other"])
def test_attention_selects_only_the_exact_native_packed_backend(monkeypatch, backend):
    native_backend = adapter.native.MixedQuantPackedCacheOps
    selected = {"packed": native_backend, "none": None, "other": type}[backend]
    original_qsmla = native_backend.qsmla

    def initialize(self, *args, **kwargs):
        self.packed_cache_ops = selected

    monkeypatch.setattr(
        deepseek_v41.native.DeepseekV41Attention, "__init__", initialize
    )
    attention = deepseek_v41.AFDDeepseekV41Attention(None, None)

    expected = adapter.AFDMixedQuantPackedCacheOps if backend == "packed" else selected
    assert attention.packed_cache_ops is expected
    assert native_backend.qsmla is original_qsmla
