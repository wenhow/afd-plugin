# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Construction/lifecycle seams only; tensor behavior needs the real NPU probe."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")

from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.distributed.hccl_plan import HcclPlanCapabilities


@pytest.fixture
def capabilities():
    return HcclPlanCapabilities(16, 128, 8, (0, 2), 2, 2, native_tp_size=2)


@pytest.mark.parametrize(
    "role,rank,world_rank,coordinator",
    [
        ("ffn", 0, 0, False),
        ("ffn", 1, 1, False),
        ("attention", 0, 2, True),
        ("attention", 1, 3, False),
    ],
)
def test_rank_identity_preserves_ffn_first_world_order(
    capabilities, role, rank, world_rank, coordinator
):
    transport = HcclStagedTransport(
        role, rank, capabilities, init_method="tcp://localhost:1", local_device=0
    )
    assert transport.world_rank == world_rank
    assert transport.is_coordinator is coordinator
    transport.close()
    transport.close()
    with pytest.raises(RuntimeError, match="not active"):
        transport.finish_forward()


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "draft"),
        ("role_rank", True),
        ("role_rank", -1),
        ("role_rank", 2),
        ("local_device", False),
        ("local_device", -1),
        ("timeout_seconds", 0),
        ("timeout_seconds", True),
    ],
)
def test_construction_rejects_invalid_endpoint_configuration(
    capabilities, field, value
):
    options = dict(
        role="attention",
        role_rank=0,
        capabilities=capabilities,
        init_method="tcp://localhost:1",
        local_device=0,
        timeout_seconds=60,
    )
    with pytest.raises(ValueError):
        HcclStagedTransport(**{**options, field: value})


def test_closed_endpoint_cannot_be_initialized(capabilities):
    transport = HcclStagedTransport(
        "attention", 0, capabilities, init_method="tcp://localhost:1", local_device=0
    )
    transport.close()
    with pytest.raises(RuntimeError, match="cannot be initialized again"):
        transport.initialize()
