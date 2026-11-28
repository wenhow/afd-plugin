# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run the native vLLM CLI with the A5 EP shutdown relay."""

import os
import runpy
import signal
from types import FrameType

SHUTDOWN_RELAY_ENV = "AFD_DSV41_EP_SHUTDOWN_RELAY"


def relay_shutdown(_signum: int, frame: FrameType | None) -> None:
    # The native multi-API frame owns the shutdown_requested flag. Its local
    # callback remains available even if the signal registration has changed.
    caller = frame
    while caller is not None:
        if (
            caller.f_globals.get("__name__") == "vllm.entrypoints.cli.serve"
            and caller.f_code.co_name == "run_multi_api_server"
        ):
            handler = caller.f_locals.get("signal_handler")
            if callable(handler):
                os.write(
                    2,
                    b"DSV41 EP shutdown: invoking native vLLM signal handler "
                    b"from active serve frame\n",
                )
                handler(signal.SIGTERM, frame)
                return
            break
        caller = caller.f_back
    os.write(2, b"DSV41 EP shutdown: native vLLM signal handler unavailable\n")


if __name__ == "__main__":
    if os.environ.get(SHUTDOWN_RELAY_ENV) == "1":
        signal.signal(signal.SIGUSR2, relay_shutdown)
    runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__", alter_sys=True)
