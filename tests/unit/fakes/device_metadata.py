# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native metadata lifecycle with CPU task execution; no stream/device behavior."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    DeviceMetadataTask,
    TargetDeviceMetadata,
)
from vllm_ascend.worker.v2.model_states import default as ascend_model_state


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


def _allocate_cpu_target_metadata() -> TargetDeviceMetadata:
    # Only the native CPU build/forward/retirement state machine runs. The real
    # TargetDeviceMetadata constructor allocates an NPU stream and is excluded.
    metadata = object.__new__(TargetDeviceMetadata)
    metadata.executor = CPUTaskExecutor()
    metadata._tasks = ()
    metadata._failed = False
    return metadata


def make_cpu_target_metadata():
    metadata = _allocate_cpu_target_metadata()
    providers = (CPUStageMetadataProvider(10), CPUStageMetadataProvider(11))
    groups = [[[provider.group()]] for provider in providers]
    return metadata, providers, groups


def install_native_target_metadata_cpu(
    monkeypatch: pytest.MonkeyPatch,
    model_state: EngramModelState,
    model_bundle: SimpleNamespace,
) -> TargetDeviceMetadata:
    """Keep native prepare/run_build/finish with a CPU backend metadata leaf.

    ModelState has already been constructed for the same prebound models and
    registered views. Native prepare_attn supplies its decomposed operands;
    this installer never observes or substitutes an InputBatch. Empty native
    attention groups produce no tasks: no producer, stream or device overlap
    is simulated. TargetDeviceMetadata.run_build itself attaches the executor
    to each returned resource, and native finish retires the same owner.
    """

    def build_cpu_metadata(
        *,
        num_reqs,
        query_start_loc_gpu,
        positions,
        seq_lens,
        is_prefilling,
        block_tables,
        **kwargs,
    ):
        return model_bundle.prepare_target_attention(
            num_reqs=num_reqs,
            query_start_loc_gpu=query_start_loc_gpu,
            positions=positions,
            seq_lens=seq_lens,
            is_prefilling=is_prefilling,
            block_tables=block_tables,
        )

    monkeypatch.setattr(ascend_model_state, "build_attn_metadata", build_cpu_metadata)
    metadata = _allocate_cpu_target_metadata()
    model_state.device_metadata = metadata
    return metadata
