# SPDX-License-Identifier: Apache-2.0
"""DSpark expert tensors and gate rows must reach the native loader intact."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm_ascend")

from afd_plugin.model_executor.models.npu.deepseek_v41_dspark import (  # noqa: E402
    _checked_draft_weights,
)


def test_streams_all_draft_experts_and_original_tensors():
    count = 128
    tensor = torch.ones(1)
    weights = [("head.weight", tensor)]
    for stage in range(3):
        prefix = f"mtp.{stage}.ffn"
        for field in ("weight", "bias", "bias_vl"):
            weights.append((f"{prefix}.gate.{field}", torch.ones(count)))
        for expert in range(count):
            weights.append((f"{prefix}.experts.{expert}.w1.weight", tensor))
    loaded = list(_checked_draft_weights(iter(weights), count))
    assert [name for name, _ in loaded] == [name for name, _ in weights]
    assert all(actual is expected for (_, actual), (_, expected) in zip(loaded, weights))


@pytest.mark.parametrize("field", ["weight", "bias", "bias_vl"])
@pytest.mark.parametrize(("rows", "configured"), [(128, 72), (72, 128)])
def test_rejects_mismatched_draft_gate(field, rows, configured):
    weights = [(f"mtp.0.ffn.gate.{field}", torch.ones(rows))]
    with pytest.raises(ValueError, match=f"has {rows} rows, expected {configured}"):
        list(_checked_draft_weights(weights, configured))


def test_rejects_out_of_range_expert_instead_of_dropping_it():
    weights = [("mtp.2.ffn.experts.72.w1.scale", torch.ones(1))]
    with pytest.raises(ValueError, match="exceeds the configured 72 draft experts"):
        list(_checked_draft_weights(weights, 72))
