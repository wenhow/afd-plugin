# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run the native vLLM CLI with optional A5 EP stack dumps."""

import faulthandler
import os
import runpy
import signal
from pathlib import Path

STACK_DIR_ENV = "AFD_DSV41_EP_STACK_DIR"


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


if __name__ == "__main__":
    register_stack_dump()
    runpy.run_module("vllm.entrypoints.cli.main", run_name="__main__", alter_sys=True)
