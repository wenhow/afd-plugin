# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Graph ownership outside capture, scoped to one fixed transport session.

Native Attention managers still own replay and model-state updates. This cache
only advertises fully published entries and keeps every captured dependency
alive. A caller supplies its device synchronization operation explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from afd_plugin.distributed.hccl_graph import HcclGraphKey

MAX_GRAPH_CACHE_ENTRIES = 1_024

ResourcesT = TypeVar("ResourcesT")


class HcclReplayGraph(Protocol):
    def replay(self) -> None: ...

    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class HcclGraphCacheEntry(Generic[ResourcesT]):
    key: HcclGraphKey
    generation: int
    graph: HcclReplayGraph
    resources: ResourcesT


class HcclGraphCache(Generic[ResourcesT]):
    """Hold graph lifetimes; a failed capture makes the whole cache terminal.

    Create a fresh instance for each transport session. Model/layer/TP/payload
    capabilities are immutable within that session. ``generation`` is the local
    native input/metadata storage generation, never a cross-peer pointer value.
    A generation mismatch is a miss and keeps the old storage alive for cleanup.
    Recapture of an existing key requires a new session in this first version.
    """

    def __init__(self, synchronize: Callable[[], None]) -> None:
        self._synchronize = synchronize
        self._pending: dict[str, HcclGraphCacheEntry[ResourcesT]] = {}
        self._ready: dict[str, HcclGraphCacheEntry[ResourcesT]] = {}
        self.terminal = self.closed = False

    @property
    def is_empty(self) -> bool:
        """Whether no graph dependencies remain, including pending captures."""
        return not self._pending and not self._ready

    def _require_active(self) -> None:
        if self.terminal or self.closed:
            raise RuntimeError("graph cache is not active")

    @staticmethod
    def _validate_generation(generation: int) -> None:
        if type(generation) is not int or generation < 0:
            raise ValueError("graph storage generation must be a nonnegative integer")

    def reserve_capture(
        self,
        key: HcclGraphKey,
        generation: int,
        graph: HcclReplayGraph,
        resources: ResourcesT,
    ) -> None:
        """Retain graph and dependencies before capture or an ownership handoff.

        Native managers may retain the graph during capture themselves. In
        that case the caller must keep its entire resource bundle alive until
        this handoff, then publish only after capture has returned successfully.
        """
        self._require_active()
        self._validate_generation(generation)
        digest = key.digest
        if digest in self._pending or digest in self._ready:
            raise ValueError("graph key already reserved; restart before recapture")
        if len(self._pending) + len(self._ready) >= MAX_GRAPH_CACHE_ENTRIES:
            raise ValueError("graph cache entry limit reached")
        self._pending[digest] = HcclGraphCacheEntry(key, generation, graph, resources)

    def publish_capture(self, key: HcclGraphKey, generation: int) -> None:
        """Advertise readiness only after capture returned and the device finished."""
        self._require_active()
        self._validate_generation(generation)
        entry = self._pending.get(key.digest)
        if entry is None or entry.generation != generation:
            raise ValueError(
                "capture publication requires its reserved key and generation"
            )
        try:
            self._synchronize()
        except BaseException:
            self.terminal = True
            raise
        self._ready[key.digest] = entry
        del self._pending[key.digest]

    def find(
        self, key: HcclGraphKey, generation: int
    ) -> HcclGraphCacheEntry[ResourcesT] | None:
        self._require_active()
        self._validate_generation(generation)
        entry = self._ready.get(key.digest)
        return entry if entry is not None and entry.generation == generation else None

    def contains(self, key: HcclGraphKey) -> bool:
        """Reject recapture even when old or pending storage is unavailable."""
        self._require_active()
        return key.digest in self._ready or key.digest in self._pending

    def fail(self) -> None:
        """Retain every graph/buffer for supervised cleanup after a device failure."""
        self.terminal = True

    def close(self) -> None:
        """Synchronize and reset before releasing even failed/pending resources.

        The owner must first stop all graph users and peer execution. A failure
        retains references and propagates to external supervision; it never
        claims that a blocked device operation has been cancelled.
        """
        if self.closed:
            return
        self.terminal = True
        self._synchronize()
        for entry in (*self._ready.values(), *self._pending.values()):
            entry.graph.reset()
        self._ready.clear()
        self._pending.clear()
        self.closed = True
