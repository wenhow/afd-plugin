# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run the native vLLM CLI with A5 EP lifecycle diagnostics."""

import faulthandler
import os
import runpy
import signal
from pathlib import Path
from types import FrameType

STACK_DIR_ENV = "AFD_DSV41_EP_STACK_DIR"
SHUTDOWN_RELAY_ENV = "AFD_DSV41_EP_SHUTDOWN_RELAY"


def register_stack_dump() -> None:
    directory = os.environ.get(STACK_DIR_ENV)
    if directory is None:
        return
    path = Path(directory) / f"{os.getpid()}.log"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        faulthandler.register(signal.SIGUSR1, file=fd, all_threads=True)
    except BaseException:
        os.close(fd)
        raise
    # faulthandler keeps the descriptor for the rest of this process's life.


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
    register_stack_dump()
    if os.environ.get(SHUTDOWN_RELAY_ENV) == "1":
        signal.signal(signal.SIGUSR2, relay_shutdown)
    runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__", alter_sys=True)
