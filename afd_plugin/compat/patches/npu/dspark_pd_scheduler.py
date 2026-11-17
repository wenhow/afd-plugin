# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Keep request-owned DSpark P pages live through the typed read lease."""

from __future__ import annotations

from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import SupportsHMA
from vllm.v1.request import Request
from vllm_ascend.patch.platform.patch_balance_schedule import Scheduler

from afd_plugin.compat.npu.kv_layout_connector import MooncakeHybridConnector
from afd_plugin.compat.npu.runtime_config import uses_deepseek_v41_dspark_pd

# Ascend 7e37e29a patch_balance_schedule.py:72 preserves the actual native base,
# then :897 replaces the public Scheduler alias with its BalanceScheduler.
# Patch the defining base so cached native aliases and derived schedulers agree.
# Immutable native baseline for signature review even after installation. The
# copied function never delegates through this reference or changes it.
NATIVE_CONNECTOR_FINISHED = Scheduler._connector_finished


# Upstream: vllm/v1/core/sched/scheduler.py, Scheduler._connector_finished,
# ced6857afa0ea7b2e3f0846a62e1394e90f15607:2842-2896. Exact signature/body
# except the marked final-eviction guard. Ready already names physical pages
# when native sampling returns; native final SWA eviction must not free those
# pages before a read/cancel lease drains. Keep normal delay-free/completion
# ownership, including cancellation. Cost: retain the final P chunk's otherwise
# expired SWA blocks until existing worker completion, potentially more than a
# fixed tail pair; no new state, synchronization or thread. Remove when native
# retirement supports connector-owned page leases before final SWA eviction.
def _connector_finished(self, request: Request) -> tuple[bool, dict[str, Any] | None]:
    """
    Invoke the KV connector request_finished() method if applicable.

    Returns optional kv transfer parameters to be included with the
    request outputs.
    """
    if self.connector is None:
        return False, None

    finished_partial_tails: list[tuple[int, int, int]] = []
    kv_transfer_config = self.vllm_config.kv_transfer_config
    if kv_transfer_config is not None and kv_transfer_config.is_kv_producer:
        finished_partial_tails = self.kv_cache_manager.finalize_partial_tail_offloads(
            request
        )

    # Free any out-of-window prefix blocks before we hand the block table to
    # the connector, on the processed-token basis (see `allocate_slots`).
    # ### PATCH START: retain leased DSpark P pages until worker completion
    params = request.kv_transfer_params
    retain_source = (
        type(self.connector) is MooncakeHybridConnector
        and uses_deepseek_v41_dspark_pd(self.vllm_config)
        and kv_transfer_config is not None
        and kv_transfer_config.kv_role == "kv_producer"
        and params is not None
        and params.get("do_remote_decode")
    )
    if not retain_source:
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            processed_computed_tokens=max(
                0, request.num_computed_tokens - request.num_in_flight_tokens
            ),
            num_prompt_tokens=request.num_prompt_tokens,
        )
    # ### PATCH END: retain leased DSpark P pages until worker completion

    block_ids = self.kv_cache_manager.get_block_ids_for_computed_tokens(
        request_id=request.request_id,
        num_computed_tokens=request.num_computed_tokens,
    )
    partial_tail_delay = False
    if finished_partial_tails:
        partial_tail_delay = self.connector.register_finished_partial_tail(
            request,
            block_ids,
            finished_partial_tails,
        )

    if not isinstance(self.connector, SupportsHMA):
        # NOTE(Kuntai): We should deprecate this code path after we enforce
        # all connectors to support HMA.
        # Hybrid memory allocator should be already turned off for this
        # code path, but let's double-check here.
        assert len(self.kv_cache_config.kv_cache_groups) == 1
        delay_free, kv_xfer_params = self.connector.request_finished(
            request, block_ids[0]
        )
    else:
        delay_free, kv_xfer_params = self.connector.request_finished_all_groups(
            request, block_ids
        )
    return delay_free or partial_tail_delay, kv_xfer_params


def install_dspark_pd_scheduler_retention() -> None:
    """Install idempotently only when the exact source connector is selected."""
    Scheduler._connector_finished = _connector_finished
