# SPDX-License-Identifier: Apache-2.0
"""Load the complete DSpark draft with its checkpoint-defined expert topology."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from vllm_ascend.models.deepseek_v41.dspark import DSparkDeepseekV41ForCausalLM

if TYPE_CHECKING:
    import torch


def _checked_draft_weights(
    weights: Iterable[tuple[str, torch.Tensor]], num_experts: int
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        # Validate global IDs even when the loader streams only this EP rank's
        # shards. Pass valid tensors to Ascend without changing their contents.
        if name.startswith("mtp."):
            parts = name.split(".", 5)
            if len(parts) >= 5 and parts[1].isdigit() and parts[2] == "ffn":
                if parts[3] == "experts" and parts[4].isdigit():
                    if int(parts[4]) >= num_experts:
                        raise ValueError(
                            f"{name} exceeds the configured {num_experts} draft experts"
                        )
                elif len(parts) == 5 and parts[3] == "gate" and parts[4] in (
                    "weight",
                    "bias",
                    "bias_vl",
                ):
                    if tensor.shape[0] != num_experts:
                        raise ValueError(
                            f"{name} has {tensor.shape[0]} rows, expected {num_experts}"
                        )
        yield name, tensor


class AFDDeepseekV41DSparkModel(DSparkDeepseekV41ForCausalLM):
    """Preserve native draft EP placement, sampling and attention."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(
            _checked_draft_weights(weights, self.config.n_routed_experts)
        )
        # The native loader also accepts target-only embeddings and heads. A
        # pruned checkpoint can keep these while omitting the entire drafter.
        if not any(name.endswith(".main_proj.weight") for name in loaded):
            raise ValueError("DSpark checkpoint is missing the mtp main_proj weight")
        return loaded
