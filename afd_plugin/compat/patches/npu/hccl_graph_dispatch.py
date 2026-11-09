# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""One owning execute thread's pre-input-preparation graph decision hook.

The native selected gather and dispatch still run. The caller supplies the
controller that reserves the all-peer mode and returns its final descriptor.
No global registry, transport or graph/model ownership is introduced here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import get_ident
from types import MethodType
from typing import TYPE_CHECKING, TypeAlias

from vllm.config import CUDAGraphMode, ParallelConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.model_runner import BatchReqState

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner


@dataclass(frozen=True, slots=True)
class HcclGraphDispatchArguments:
    cudagraph_manager: CudaGraphManager | None
    num_reqs: int
    num_tokens: int
    uniform_token_count: int | None
    dp_size: int
    dp_rank: int
    max_query_len: int | None
    need_eager: bool
    num_active_loras: int
    parallel_config: ParallelConfig | None
    allow_ubatching: bool
    uniform_decode: bool
    dp_sync: DPSyncState | None


HcclGraphDispatchSelector: TypeAlias = Callable[
    [BatchReqState | None, HcclGraphDispatchArguments, BatchExecutionDescriptor],
    BatchExecutionDescriptor,
]


class _HcclGraphDispatchScope:
    def __init__(
        self, owner: NPUModelRunner, select: HcclGraphDispatchSelector
    ) -> None:
        self.owner, self.select = owner, select
        self.original_gather = owner.gather_batch_req_state
        self.original_dispatch = model_runner.dispatch_cg_and_sync_dp
        self.thread_id = get_ident()
        self.gathered = self.dispatched = False
        self.batch_state: BatchReqState | None = None
        self.uniform_token_count: int | None = None

    def _require_owner_thread(self) -> None:
        if get_ident() != self.thread_id:
            raise RuntimeError(
                "native graph dispatch scope belongs to one execute thread"
            )

    # Upstream: Ascend NPUModelRunner.gather_batch_req_state @ 7e37e29a:615.
    # Reason: native dispatch receives dimensions but not the actual ordered
    # CPU batch state. Retain the selected owner's real return before dispatch.
    # Signature matches that override exactly, including no return annotation.
    # Delegation exception: selected Ascend/GPU gather owns ordering and history
    # coordinates; copying it would bypass its selected instance behavior.
    # Remove when upstream provides batch state at a pre-preparation callback.
    def gather_batch_req_state(
        self, scheduler_output: SchedulerOutput, dummy_run: bool
    ):
        # ### PATCH START: retain the native ordered CPU batch once
        self._require_owner_thread()
        if self.gathered:
            raise RuntimeError("native graph scope requires one authoritative gather")
        result = self.original_gather(scheduler_output, dummy_run)
        self.batch_state, self.uniform_token_count = result
        self.gathered = True
        # ### PATCH END: retain the native ordered CPU batch once
        return result

    # Upstream: gpu/dp_utils.py:dispatch_cg_and_sync_dp @ ced6857a:214-228,
    # imported into gpu/model_runner.py and called before prepare_inputs.
    # Reason: all-peer graph/fallback agreement must precede native Engram and
    # metadata preparation. Functionality: retain native dispatch/DPSync and
    # let the controller return the final descriptor from the real CPU state.
    # Signature matches all 13 native parameters and the return annotation.
    # Delegation exception: retain the selected imported dispatcher, including
    # any existing backend scope. Copying its body would bypass that selection
    # and duplicate DP synchronization/reuse. The controller owns the additional
    # U-specific lookup and wire agreement.
    # Remove when native execute exposes a pre-input-preparation decision hook.
    def dispatch_cg_and_sync_dp(
        self,
        cudagraph_manager: CudaGraphManager | None,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        dp_size: int,
        dp_rank: int,
        max_query_len: int | None = None,
        need_eager: bool = False,
        num_active_loras: int = 0,
        parallel_config: ParallelConfig | None = None,
        allow_ubatching: bool = False,
        uniform_decode: bool = False,
        dp_sync: DPSyncState | None = None,
    ) -> tuple[BatchExecutionDescriptor, DPSyncState | None]:
        # ### PATCH START: decide before native input/Engram preparation
        self._require_owner_thread()
        if not self.gathered or self.dispatched:
            raise RuntimeError("native graph dispatch requires one owning batch gather")
        if cudagraph_manager is not self.owner.cudagraph_manager:
            raise ValueError("native graph dispatch does not belong to this runner")
        if (
            parallel_config is not None
            and parallel_config is not self.owner.parallel_config
        ):
            raise ValueError(
                "native graph dispatch uses another runner's parallel config"
            )
        if dp_size != 1 or dp_rank != 0:
            raise ValueError("native graph dispatch scope requires DP1 rank zero")
        if (
            uniform_token_count != self.uniform_token_count
            or self.batch_state is not None
            and (
                num_reqs != len(self.batch_state.req_ids)
                or num_tokens != self.batch_state.num_tokens
            )
        ):
            raise ValueError(
                "native dispatch dimensions do not match its gathered batch"
            )
        self.dispatched = True
        effective_eager = (
            need_eager
            or self.batch_state is None
            or dp_sync is not None
            and dp_sync.eager
        )
        arguments = HcclGraphDispatchArguments(
            cudagraph_manager,
            num_reqs,
            num_tokens,
            uniform_token_count,
            dp_size,
            dp_rank,
            max_query_len,
            effective_eager,
            num_active_loras,
            parallel_config,
            allow_ubatching,
            uniform_decode,
            dp_sync,
        )
        # ### PATCH END: decide before native input/Engram preparation
        native_descriptor, native_sync = self.original_dispatch(
            cudagraph_manager,
            num_reqs,
            num_tokens,
            uniform_token_count,
            dp_size,
            dp_rank,
            max_query_len=max_query_len,
            need_eager=effective_eager,
            num_active_loras=num_active_loras,
            parallel_config=parallel_config,
            allow_ubatching=allow_ubatching,
            uniform_decode=uniform_decode,
            dp_sync=dp_sync,
        )
        # ### PATCH START: a committed controller decision owns the final mode
        descriptor = self.select(self.batch_state, arguments, native_descriptor)
        if not isinstance(descriptor, BatchExecutionDescriptor):
            raise ValueError("graph selector must return a native execution descriptor")
        if effective_eager and descriptor.cg_mode != CUDAGraphMode.NONE:
            raise ValueError("forced-eager native dispatch cannot select a graph")
        # ### PATCH END: a committed controller decision owns the final mode
        return descriptor, native_sync


@contextmanager
def use_hccl_graph_dispatch(
    owner: NPUModelRunner, select: HcclGraphDispatchSelector
) -> Iterator[None]:
    """Bind one native execute invocation; concurrency is intentionally excluded.

    The temporarily imported bound method identifies an already active scope,
    so nested scopes for either the same or a different runner are rejected.
    The real gather tuple and original native DP synchronization are preserved.
    """
    dispatch = model_runner.dispatch_cg_and_sync_dp
    gather = owner.gather_batch_req_state
    if any(
        isinstance(binding, MethodType)
        and isinstance(binding.__self__, _HcclGraphDispatchScope)
        for binding in (dispatch, gather)
    ):
        raise RuntimeError("native graph dispatch scopes cannot be nested")
    scope = _HcclGraphDispatchScope(owner, select)
    had_override = "gather_batch_req_state" in vars(owner)
    try:
        owner.gather_batch_req_state = scope.gather_batch_req_state
        model_runner.dispatch_cg_and_sync_dp = scope.dispatch_cg_and_sync_dp
        yield
    finally:
        model_runner.dispatch_cg_and_sync_dp = scope.original_dispatch
        if had_override:
            owner.gather_batch_req_state = scope.original_gather
        else:
            del owner.gather_batch_req_state
