# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run one V4.1 HCCL FFN engine with a native local startup client.

Native ``vllm serve --headless`` expects another frontend to answer its
EngineCore startup handshake. This standalone F role instead retains the
native synchronous client's local sockets, startup handshake and cleanup,
without exposing an HTTP generation service or sending scheduled requests.
"""

from __future__ import annotations

import signal
from collections.abc import Sequence
from types import FrameType

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.plugins import load_general_plugins
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.core_client import SyncMPClient
from vllm.v1.engine.utils import CoreEngineProcManager
from vllm.v1.executor import Executor

from afd_plugin.compat.npu.feature_validation import (
    fail_if_unsupported_npu_afd_features,
)
from afd_plugin.compat.npu.runtime_config import get_npu_ffn_config
from afd_plugin.config import HCCL_P2P_CONNECTOR, parse_afd_config
from afd_plugin.validation import validate_npu_model_runner_v2_config

SUCCESS_MESSAGE = "AFD FFN launcher exited with all engine exit codes zero."


class AFDFFNClient(SyncMPClient):
    """Let the connector daemon finish through its paired Attention STOP."""

    # Upstream: MPClient.start_engine_core_monitor, vllm/v1/engine/core_client.py
    # at ced6857afa0ea7b2e3f0846a62e1394e90f15607. Signature matches exactly.
    # Reason: native request clients classify every engine exit as unexpected;
    # an FFN service normally exits after Attention's acknowledged HCCL STOP.
    # Functionality: this entry point monitors the same native process manager
    # synchronously and checks every exit code before releasing client resources.
    # Remove when native clients expose a connector-service exit policy.
    def start_engine_core_monitor(self):
        # ### PATCH START: explicit FFN service exit owner
        return None
        # ### PATCH END: explicit FFN service exit owner


def validate_ffn_configuration(vllm_config: VllmConfig) -> None:
    """Reject other roles and unsupported layouts before creating processes."""
    afd_config = parse_afd_config(vllm_config, expected_role="ffn")
    if afd_config.connector != HCCL_P2P_CONNECTOR:
        raise ValueError("standalone NPU FFN requires P2pHcclAFDConnector")
    parallel = vllm_config.parallel_config
    if (
        parallel.distributed_executor_backend != "mp"
        or parallel.data_parallel_size != 1
        or parallel.data_parallel_size_local != 1
        or parallel.data_parallel_rank != 0
        or parallel.nnodes != 1
        or parallel.node_rank != 0
    ):
        raise ValueError("standalone NPU FFN requires one local DP1 mp engine")
    if (
        vllm_config.speculative_config is not None
        or vllm_config.kv_transfer_config is not None
    ):
        raise ValueError("standalone NPU FFN qualification requires DSpark/PD off")
    validate_npu_model_runner_v2_config(
        vllm_config,
        expected_role="ffn",
        device_type=vllm_config.device_config.device_type,
    )
    # Use the worker's existing private role view for preflight. Preserve the
    # original configuration so FFNWorker still performs its native isolation.
    fail_if_unsupported_npu_afd_features(get_npu_ffn_config(vllm_config))


def run_ffn(vllm_config: VllmConfig, *, log_stats: bool) -> int:
    """Bootstrap natively, wait for the resident engine, then clean up."""
    validate_ffn_configuration(vllm_config)
    client = AFDFFNClient(
        vllm_config,
        executor_class=Executor.get_class(vllm_config),
        log_stats=log_stats,
    )
    try:
        manager = client.resources.engine_manager
        if not isinstance(manager, CoreEngineProcManager):
            raise TypeError("standalone NPU FFN requires the native process manager")
        manager.monitor_engine_liveness()
        finished = manager.finished_procs()
        if len(finished) != vllm_config.parallel_config.data_parallel_size_local:
            raise RuntimeError("standalone NPU FFN exited with unfinished engines")
        for returncode in finished.values():
            if returncode:
                # Preserve ordinary native error codes and map a signal to the
                # shell convention instead of reporting a killed engine as clean.
                return returncode if returncode > 0 else 128 - returncode
    finally:
        client.shutdown(timeout=vllm_config.shutdown_timeout)
    print(SUCCESS_MESSAGE, flush=True)
    return 0


def _exit_on_signal(signum: int, _frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def main(argv: Sequence[str] | None = None) -> int:
    load_general_plugins()
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser(description=__doc__))
    engine_args = EngineArgs.from_cli_args(parser.parse_args(argv))
    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER,
    )
    previous_handlers = {
        signum: signal.signal(signum, _exit_on_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        return run_ffn(vllm_config, log_stats=not engine_args.disable_log_stats)
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
