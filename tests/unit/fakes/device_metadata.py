# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native metadata lifecycle with CPU task execution; no stream/device behavior."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    DeviceMetadataTask,
    TargetDeviceMetadata,
)


class CPUTaskExecutor:
    """The device boundary of real TargetDeviceMetadata, explicitly on CPU."""

    def __init__(self):
        self.submission_in_flight = False

    def submit(self, tasks):
        self.submission_in_flight = True
        for task in tasks:
            task.run()

    def wait(self, stage, group_id):
        if not self.submission_in_flight:
            raise AssertionError("metadata consumer has no submitted parent")

    def release(self):
        self.submission_in_flight = False


class CPUStageMetadataProvider:
    def __init__(self, value):
        self.value = value
        self.output = torch.zeros(1)

    @contextmanager
    def defer_device_metadata(self, *, in_graph=False):
        if not in_graph:
            raise AssertionError("FULL parent must retain graph-side metadata tasks")
        yield

    def take_device_metadata_tasks(self):
        return (
            DeviceMetadataTask(
                DeviceMetadataStage.ATTENTION,
                lambda: self.output.fill_(self.value),
                id(self.output),
            ),
        )

    def group(self):
        return SimpleNamespace(get_metadata_builder=lambda index: self)


def make_cpu_target_metadata():
    # Only the native CPU build/forward/retirement state machine runs. The real
    # TargetDeviceMetadata constructor allocates an NPU stream and is excluded.
    metadata = object.__new__(TargetDeviceMetadata)
    metadata.executor = CPUTaskExecutor()
    metadata._tasks = ()
    metadata._failed = False
    providers = (CPUStageMetadataProvider(10), CPUStageMetadataProvider(11))
    groups = [[[provider.group()]] for provider in providers]
    return metadata, providers, groups
