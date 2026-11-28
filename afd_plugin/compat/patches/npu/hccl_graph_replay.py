# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Retain native ACL replay while scoping one accepted metadata handoff."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from threading import get_ident
from types import MethodType

import torch
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager


class _HcclGraphReplayScope:
    def __init__(
        self,
        manager: ModelAclGraphManager,
        before_replay: Callable[
            [BatchExecutionDescriptor], AbstractContextManager[None]
        ],
        after_replay: Callable[[BatchExecutionDescriptor], None],
        fail: Callable[[], None],
    ) -> None:
        self.manager = manager
        self.before_replay, self.after_replay, self.fail = (
            before_replay,
            after_replay,
            fail,
        )
        self.original_replay = manager.run_fullgraph
        self.thread_id = get_ident()
        self.started = False


@contextmanager
def use_hccl_graph_replay(
    manager: ModelAclGraphManager,
    *,
    before_replay: Callable[[BatchExecutionDescriptor], AbstractContextManager[None]],
    after_replay: Callable[[BatchExecutionDescriptor], None],
    fail: Callable[[], None],
) -> Iterator[None]:
    """Bind one owner thread's replay; callbacks run outside the native graph.

    The caller supplies a context that temporarily hands native lifecycle code
    the accepted metadata. Native replay and its finally/finish_replay remain
    authoritative. Completion follows the successful original call, after the
    temporary metadata has been restored. Concurrent use is unsupported.
    """
    original = manager.run_fullgraph
    if isinstance(original, MethodType) and any(
        isinstance(cell.cell_contents, _HcclGraphReplayScope)
        for cell in original.__func__.__closure__ or ()
    ):
        raise RuntimeError("native graph replay scopes cannot be nested")
    scope = _HcclGraphReplayScope(manager, before_replay, after_replay, fail)
    had_override = "run_fullgraph" in vars(manager)

    # Upstream: ModelAclGraphManager.run_fullgraph @ Ascend 7e37e29a:176-209.
    # Reason: U2 prepares child metadata but native ACL replay reads the parent
    # ModelState field. Functionality: scope the accepted metadata handoff and
    # perform HCCL completion outside replay without copying native lifecycle.
    # Signature matches this Ascend override exactly, including its return type.
    # Delegation exception: native prepare_engram/graph update/finish_replay
    # remain selected-backend behavior; expansion would duplicate that owner.
    # The caller restricts V4.1 SFA/DSA to their pinned no-op update contract;
    # this wrapper does not flatten children or add generic backend support.
    # Remove when native ACL replay consumes explicit parent/child metadata.
    def run_fullgraph(
        self, desc: BatchExecutionDescriptor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        # ### PATCH START: one accepted native replay and outside completion
        try:
            if (
                self is not scope.manager
                or get_ident() != scope.thread_id
                or scope.started
            ):
                raise RuntimeError(
                    "native graph replay requires one owning-thread invocation"
                )
            scope.started = True
            with scope.before_replay(desc):
                result = scope.original_replay(desc)
            scope.after_replay(desc)
            return result
        except BaseException:
            scope.fail()
            raise
        # ### PATCH END: one accepted native replay and outside completion

    try:
        manager.run_fullgraph = MethodType(run_fullgraph, manager)
        yield
    finally:
        if had_override:
            manager.run_fullgraph = original
        else:
            del manager.run_fullgraph
