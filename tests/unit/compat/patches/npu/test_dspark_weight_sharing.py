# SPDX-License-Identifier: Apache-2.0
"""Exact vocabulary comparisons with a CPU-only tensor workload."""

import pytest

torch = pytest.importorskip("torch")

from afd_plugin.compat.patches.npu.dspark_weight_sharing import (  # noqa: E402
    WEIGHT_COMPARE_CHUNK_ROWS,
    equal_weight_rows_on_cpu,
)


@pytest.mark.parametrize("difference", [None, "first_row", "last_row", "shape"])
def test_blocked_weight_comparison_keeps_torch_equal_semantics(difference):
    draft = torch.zeros((WEIGHT_COMPARE_CHUNK_ROWS + 1, 4), dtype=torch.bfloat16)
    target = draft.clone()
    if difference == "first_row":
        target[0, 1] = 1
    elif difference == "last_row":
        target[-1, 1] = 1
    elif difference == "shape":
        target = target[:-1]
    assert equal_weight_rows_on_cpu(draft, target) == (difference is None)
