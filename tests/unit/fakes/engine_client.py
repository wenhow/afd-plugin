# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native engine-client process outcomes for service supervision tests."""

from dataclasses import dataclass
from types import SimpleNamespace


class FFNProcessExit:
    """One OS process-exit boundary, without constructing engine processes."""

    def __init__(self, manager, rank, returncode):
        self.manager = manager
        self.sentinel = rank
        self.name = f"FFN-{rank}"
        self.returncode = returncode
        self.exitcode = None

    def join(self):
        self.exitcode = self.returncode
        self.manager.outcomes[self.name] = self.exitcode


@dataclass
class FFNProcessManager:
    outcomes: dict[str, int]
    monitor_error: Exception | None = None
    stopped: bool = False

    def monitor_engine_liveness(self) -> None:
        if self.monitor_error is not None:
            raise self.monitor_error
        self.stopped = True

    def finished_procs(self) -> dict[str, int]:
        return dict(self.outcomes)


class FFNServiceClient:
    def __init__(
        self, manager: FFNProcessManager, shutdown_error: Exception | None = None
    ) -> None:
        self.resources = SimpleNamespace(engine_manager=manager)
        self.shutdown_error = shutdown_error
        self.closed = False

    def shutdown(self, timeout: float | None = None) -> None:
        del timeout
        self.closed = True
        if self.shutdown_error is not None:
            raise self.shutdown_error


def make_ffn_exit_service(returncodes, *, timeout_after_first=False):
    manager = FFNProcessManager({})
    manager.processes = [
        FFNProcessExit(manager, rank, code) for rank, code in enumerate(returncodes)
    ]
    pending = iter(manager.processes)

    def ready_processes(handles, timeout=None):
        if timeout_after_first and timeout is not None:
            return []
        return [next(pending).sentinel]

    return FFNServiceClient(manager), ready_processes
