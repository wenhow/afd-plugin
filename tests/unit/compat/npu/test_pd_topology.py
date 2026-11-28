# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""V4.1 EP/PD layout admission without a native transfer engine."""

import pytest

from afd_plugin.compat.npu.pd_topology import validate_ep_pd_topology
from afd_plugin.compat.npu.runtime_config import HCCL_ATTESTED_PD_MODULE
from afd_plugin.config import MAX_TCP_PORT
from tests.unit.fakes.vllm_config import (
    configure_hccl_ep,
    npu_vllm_config,
    pd_kv_transfer_config,
)

DP_SIZE = 4
LAST_VALID_BASE = MAX_TCP_PORT - DP_SIZE + 1


@pytest.fixture
def ep_pd_config():
    transfer = pd_kv_transfer_config()
    transfer.kv_connector_module_path = HCCL_ATTESTED_PD_MODULE
    transfer.kv_port = LAST_VALID_BASE
    return configure_hccl_ep(
        npu_vllm_config(
            model_type="deepseek_v41", kv_transfer_config=transfer, nnodes=1
        ),
        dp_size=DP_SIZE,
        dp_rank=DP_SIZE - 1,
    )


def test_ep_pd_admits_last_rank_when_full_handshake_range_fits(ep_pd_config):
    assert (
        validate_ep_pd_topology(
            ep_pd_config,
            ep_pd_config.kv_transfer_config,
            attested_module=HCCL_ATTESTED_PD_MODULE,
        )
        == DP_SIZE
    )


@pytest.fixture(params=[False, True], ids=["microbatching", "dbo"])
def ep_pd_u2_config(ep_pd_config, request):
    parallel = ep_pd_config.parallel_config
    parallel.enable_dbo = request.param
    parallel.use_ubatching = True
    parallel.num_ubatches = 2
    return ep_pd_config


def test_ep_pd_admits_native_u2_on_decode(ep_pd_u2_config):
    transfer = ep_pd_u2_config.kv_transfer_config
    transfer.kv_role = "kv_consumer"
    assert (
        validate_ep_pd_topology(
            ep_pd_u2_config, transfer, attested_module=HCCL_ATTESTED_PD_MODULE
        )
        == DP_SIZE
    )


def test_ep_pd_requires_u1_on_prefill(ep_pd_u2_config):
    transfer = ep_pd_u2_config.kv_transfer_config
    transfer.kv_role = "kv_producer"
    with pytest.raises(ValueError, match="U1 without DBO except native U2 on D"):
        validate_ep_pd_topology(
            ep_pd_u2_config, transfer, attested_module=HCCL_ATTESTED_PD_MODULE
        )


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("kv_transfer_config", "kv_connector_module_path", None, "attested Mooncake"),
        ("kv_transfer_config", "kv_port", LAST_VALID_BASE + 1, "one valid KV"),
        ("parallel_config", "data_parallel_size_local", 2, "static local DP/EP"),
        ("parallel_config", "data_parallel_rank", DP_SIZE, "static local DP/EP"),
        ("parallel_config", "nnodes", 2, "static local DP/EP"),
        ("parallel_config", "enable_dbo", True, "U1 without DBO"),
        ("parallel_config", "num_ubatches", 2, "U1 without DBO"),
        ("parallel_config", "use_sequence_parallel_moe", True, "static local DP/EP"),
    ],
)
def test_ep_pd_rejects_unsupported_replica_coordinates(
    ep_pd_config, section, field, value, message
):
    setattr(vars(ep_pd_config)[section], field, value)
    with pytest.raises(ValueError, match=message):
        validate_ep_pd_topology(
            ep_pd_config,
            ep_pd_config.kv_transfer_config,
            attested_module=HCCL_ATTESTED_PD_MODULE,
        )
