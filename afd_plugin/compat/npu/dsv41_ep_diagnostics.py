# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Request admission logging for the A5 expert-parallel diagnostic run."""

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from afd_plugin.entrypoints.dsv41_ep_stack import register_stack_dump

logger = init_logger(f"vllm.{__name__}")


class DSV41EPDiagnosticScheduler(Scheduler):
    _stack_dump_registered = False

    def add_request(self, request: Request) -> None:
        if not self._stack_dump_registered:
            register_stack_dump()
            self._stack_dump_registered = True
        super().add_request(request)
        logger.info(
            "DSV41 EP request admitted: dp_rank=%d request_id=%s",
            self.parallel_config.data_parallel_rank,
            request.request_id,
        )

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        output = super().schedule(throttle_prefills=throttle_prefills)
        if output.total_num_scheduled_tokens:
            logger.info(
                "DSV41 EP scheduled tokens: dp_rank=%d requests=%s",
                self.parallel_config.data_parallel_rank,
                output.num_scheduled_tokens,
            )
        return output
