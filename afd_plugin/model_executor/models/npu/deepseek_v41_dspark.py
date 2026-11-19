# SPDX-License-Identifier: Apache-2.0
"""Load a DeepSeek V4.1 checkpoint view into the native Ascend DSpark draft."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from vllm_ascend.models.deepseek_v41.dspark import DSparkDeepseekV41ForCausalLM

if TYPE_CHECKING:
    import torch


def _draft_weights(
    weights: Iterable[tuple[str, torch.Tensor]], num_experts: int
) -> Iterator[tuple[str, torch.Tensor]]:
    for name, tensor in weights:
        # The native loader also consumes embeddings and head parameters.
        # Only mtp.<stage>.ffn weights depend on the routed expert view.
        if name.startswith("mtp."):
            parts = name.split(".", 5)
            if len(parts) >= 5 and parts[1].isdigit() and parts[2] == "ffn":
                if parts[3] == "experts" and parts[4].isdigit():
                    if int(parts[4]) >= num_experts:
                        continue
                elif len(parts) == 5 and parts[3] == "gate" and parts[4] in (
                    "weight",
                    "bias",
                    "bias_vl",
                ):
                    if tensor.shape[0] < num_experts:
                        raise ValueError(
                            f"{name} has {tensor.shape[0]} rows, expected {num_experts}"
                        )
                    if tensor.shape[0] > num_experts:
                        tensor = tensor[:num_experts]
        yield name, tensor


class AFDDeepseekV41DSparkModel(DSparkDeepseekV41ForCausalLM):
    """Preserve native draft EP placement, sampling and attention."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(
            _draft_weights(weights, self.config.n_routed_experts)
        )
        # The native loader also accepts target-only embeddings and heads. A
        # pruned checkpoint can keep these while omitting the entire drafter.
        if not any(name.endswith(".main_proj.weight") for name in loaded):
            raise ValueError("DSpark checkpoint is missing the mtp main_proj weight")
        return loaded
