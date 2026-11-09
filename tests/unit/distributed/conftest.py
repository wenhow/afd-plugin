# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared real CPU protocol contracts; no simulated device operations."""

import pytest

from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_plan import HcclParentPlan, HcclPlanCapabilities


@pytest.fixture
def capabilities():
    return HcclPlanCapabilities(16, 128, 8, (0, 1, 3), 1, 2)


@pytest.fixture
def parent_plan():
    return HcclParentPlan(
        "parent-1",
        (0, 1, 3),
        (HcclStageLayout(0, (3,), (4,), 2), HcclStageLayout(1, (0,), (2,), 2)),
    )
