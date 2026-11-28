# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native Mooncake replica coordinates for V4.1 TP1/EP PD."""

from __future__ import annotations

from typing import TYPE_CHECKING

from afd_plugin.config import MAX_TCP_PORT

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig, VllmConfig


def validate_ep_pd_topology(
    config: VllmConfig,
    transfer: KVTransferConfig,
    *,
    attested_module: str,
) -> int:
    """Return the native DP size for the matching P/D role declarations.

    Locked Ascend advertises each P replica's own handshake base to D.
    Single-node TP1/PP1/CP1 makes that base the configured KV port plus DP
    rank. Replica caches are independent even when they share an engine ID.
    P publishes request-owned prefixes from U1. D can split its native target
    parent into two stages without changing the producer's KV coordinates.
    """
    parallel = config.parallel_config
    dp_size = parallel.data_parallel_size
    if transfer.kv_connector_module_path != attested_module:
        raise ValueError("V4.1 EP PD requires the attested Mooncake connector")
    if (
        parallel.tensor_parallel_size != 1
        or type(dp_size) is not int
        or dp_size <= 1
        or parallel.data_parallel_size_local != dp_size
        or parallel.nnodes != 1
        or not 0 <= parallel.data_parallel_rank < dp_size
        or parallel.enable_eplb
        or parallel.enable_elastic_ep
        or parallel.use_sequence_parallel_moe
    ):
        raise ValueError("V4.1 EP PD requires single-node TP1 with static local DP/EP")
    stages = parallel.num_ubatches
    native_u1 = (
        not parallel.enable_dbo and not parallel.use_ubatching and stages in (0, 1)
    )
    native_decode_u2 = (
        transfer.kv_role == "kv_consumer" and parallel.use_ubatching and stages == 2
    )
    if type(stages) is not int or not (native_u1 or native_decode_u2):
        raise ValueError("V4.1 EP PD requires U1 without DBO except native U2 on D")
    port = transfer.kv_port
    if type(port) is not int or not 1 <= port <= MAX_TCP_PORT - dp_size + 1:
        raise ValueError("V4.1 EP PD requires one valid KV handshake port per DP rank")
    return dp_size
