# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native output supervision with CPU construction, diagnostic and queue leaves.

EngineCore scheduling/output updates, UniProc dispatch/futures and WorkerProc's
response assembly stay real. The runner is the already registered PD CPU owner;
no executor/worker startup, NPU completion or process supervision runs here.
"""

from collections import deque
from queue import Queue
from types import SimpleNamespace

from vllm.v1.engine import core as engine_core
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.executor.uniproc_executor import UniProcExecutor


class RunnerOutputBoundary:
    """Map the native worker RPC names to the existing actual model runner.

    The completion fault is armed only after sample_tokens has returned its actual
    AsyncOutput. Its inherited get_output and all native executor consumers remain
    unchanged. Cache/protocol snapshots describe that already advanced frontier.
    """

    def __init__(self, runner, transfer, *, fail_completion):
        self.runner = runner
        self.transfer = transfer
        self.fail_completion = fail_completion

    def execute_model(self, scheduler_output):
        return self.runner.owner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output):
        output = self.runner.owner.sample_tokens(grammar_output)
        self.pages_after_sampling = tuple(
            page.clone() for page in self.transfer.destination_cache.pages
        )
        self.protocol_after_sampling = tuple(self.transfer.sent)
        output.copy_event.failed = self.fail_completion
        return output


def make_native_output_engine_case(monkeypatch, runner, transfer, *, fail_completion):
    """Construct native consumers around a scheduler ready for first D work.

    The only native module substitution here suppresses diagnostic file writes.
    The actual error context, scheduler, dispatch, output and Future methods run.
    A one-entry batch queue drains this one output; this does not claim that a
    deployed async scheduler cannot have later batches already in flight.
    """
    monkeypatch.setattr(engine_core, "dump_engine_exception", lambda *args: None)
    driver = RunnerOutputBoundary(runner, transfer, fail_completion=fail_completion)
    executor = UniProcExecutor.__new__(UniProcExecutor)
    executor.driver_worker = driver
    engine = engine_core.EngineCore.__new__(engine_core.EngineCore)
    engine.vllm_config = transfer.config
    engine.scheduler, engine.model_executor = transfer.scheduler, executor
    engine.log_stats = False
    engine.aborts_queue = Queue()
    engine.batch_queue, engine.batch_queue_size = deque(), 1
    engine.is_pooling_model, engine.is_ec_consumer = False, True
    return SimpleNamespace(
        engine=engine, driver=driver, runner=runner, transfer=transfer
    )


def make_native_worker_output_case():
    """Replace only the IPC queue/construction around real enqueue_output."""
    responses = Queue()
    worker = WorkerProc.__new__(WorkerProc)
    worker.worker_response_mq = SimpleNamespace(enqueue=responses.put_nowait)
    return worker, responses
