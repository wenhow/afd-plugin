# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native engine-client process outcomes for service supervision tests."""

from dataclasses import dataclass
from types import SimpleNamespace


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
