# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Capture existing native serving handlers without promoting plugin loggers."""

import ast
import logging

import pytest

SESSION_CONFIGURATION_PREFIX = "AFD HCCL session configuration: "


@pytest.fixture
def serving_logs(caplog):
    # Native serving emits INFO under vllm while the root remains WARNING.
    serving_logger = logging.getLogger("vllm")
    serving_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING):
            caplog.handler.setLevel(logging.NOTSET)
            yield caplog
    finally:
        serving_logger.removeHandler(caplog.handler)


def structured_log_records(caplog, prefix):
    return [
        ast.literal_eval(message[len(prefix) :])
        for record in caplog.records
        if (message := record.getMessage()).startswith(prefix)
    ]
