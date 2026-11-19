# SPDX-License-Identifier: Apache-2.0
"""DSpark checkpoint views preserve the expert IDs used by the native loader."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm_ascend")

from afd_plugin.model_executor.models.npu.deepseek_v41_dspark import (  # noqa: E402
    _draft_weights,
)


@pytest.mark.parametrize(("on_disk", "configured"), [(8, 8), (8, 3), (3, 3)])
def test_draft_expert_view(on_disk, configured):
    gate = torch.arange(on_disk * 2).reshape(on_disk, 2)
    bias = torch.arange(on_disk)
    share = torch.ones(1)
    weights = {
        "mtp.2.ffn.gate.weight": gate,
        "mtp.2.ffn.gate.bias": bias,
        "mtp.2.ffn.shared_experts.w1.weight": share,
        "mtp.2.markov_head.embed.weight": share,
        "head.weight": share,
    }
    for expert in range(on_disk):
        weights[f"mtp.2.ffn.experts.{expert}.w1.scale"] = share
    loaded = dict(_draft_weights(weights.items(), configured))
    assert torch.equal(loaded.pop("mtp.2.ffn.gate.weight"), gate[:configured])
    assert torch.equal(loaded.pop("mtp.2.ffn.gate.bias"), bias[:configured])
    assert loaded.keys() == {
        "mtp.2.ffn.shared_experts.w1.weight",
        "mtp.2.markov_head.embed.weight",
        "head.weight",
        *(f"mtp.2.ffn.experts.{expert}.w1.scale" for expert in range(configured)),
    }
    assert all(value is share for value in loaded.values())


def test_draft_router_rejects_incomplete_checkpoint():
    with pytest.raises(ValueError, match="has 2 rows, expected 3"):
        list(_draft_weights([("mtp.0.ffn.gate.bias", torch.zeros(2))], 3))
