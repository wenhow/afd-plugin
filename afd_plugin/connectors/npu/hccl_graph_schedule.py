# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""One paired static parent-wire schedule, constructed outside graph capture.

INPUT is A send-three/F receive-three; OUTPUT is F send-one/A receive-one.
Zero-row stages remain in the geometry but produce no action on either peer.
The schedule neither posts transport nor changes native layer-major compute.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class HcclGraphAction:
    kind: Literal["input", "output"]
    layer_idx: int
    stage_idx: int


@dataclass(frozen=True, slots=True)
class HcclGraphSchedule:
    """The exact cross_layer_v1 actions shared by both role consumers."""

    layer_indices: tuple[int, ...]
    stage_rows: tuple[int, ...]
    actions: tuple[HcclGraphAction, ...] = field(init=False)
    compute_steps: tuple[tuple[int, int], ...] = field(init=False)

    def __post_init__(self) -> None:
        if (
            type(self.layer_indices) is not tuple
            or not self.layer_indices
            or any(type(layer) is not int or layer < 0 for layer in self.layer_indices)
            or tuple(sorted(set(self.layer_indices))) != self.layer_indices
            or type(self.stage_rows) is not tuple
            or len(self.stage_rows) != 2
            or any(type(rows) is not int or rows < 0 for rows in self.stage_rows)
        ):
            raise ValueError("cross-layer schedule requires ordered layers and U2 rows")
        stages = tuple(stage for stage, rows in enumerate(self.stage_rows) if rows)
        steps = tuple(
            (layer, stage) for layer in self.layer_indices for stage in stages
        )
        actions = [
            HcclGraphAction("input", self.layer_indices[0], stage) for stage in stages
        ]
        for offset, layer in enumerate(self.layer_indices):
            for stage in stages:
                actions.append(HcclGraphAction("output", layer, stage))
                if offset + 1 < len(self.layer_indices):
                    actions.append(
                        HcclGraphAction("input", self.layer_indices[offset + 1], stage)
                    )
        object.__setattr__(self, "actions", tuple(actions))
        object.__setattr__(self, "compute_steps", steps)
