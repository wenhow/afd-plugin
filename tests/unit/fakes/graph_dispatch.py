# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Selected native-owner collaborator for pre-prepare CPU scope tests."""

from vllm.config import ParallelConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.model_runner import BatchReqState


class NativeGraphDispatchOwner:
    def __init__(
        self,
        result: tuple[BatchReqState | None, int | None],
        parallel_config: ParallelConfig,
        manager: CudaGraphManager | None = None,
    ) -> None:
        self.result = result
        self.parallel_config = parallel_config
        self.cudagraph_manager = manager

    def gather_batch_req_state(
        self, scheduler_output: SchedulerOutput, dummy_run: bool
    ):
        return self.result
