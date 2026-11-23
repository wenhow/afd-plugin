# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Request admission logging for the A5 expert-parallel diagnostic run."""

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

logger = init_logger(__name__)


class DSV41EPDiagnosticScheduler(Scheduler):
    def add_request(self, request: Request) -> None:
        super().add_request(request)
        logger.info(
            "DSV41 EP request admitted: dp_rank=%d request_id=%s",
            self.parallel_config.data_parallel_rank,
            request.request_id,
        )
