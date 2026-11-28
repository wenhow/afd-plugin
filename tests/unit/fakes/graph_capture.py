# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host factory notifications with an explicit CPU capture-body boundary."""

from types import SimpleNamespace

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


class CaptureNotifications:
    def __init__(self, *, failure=None):
        self.failure = failure
        self.messages = []
        self.terminal = False

    def before_factory(self, descriptor, warmup):
        self.messages.append(("reserve", descriptor, warmup))
        if self.failure == "reserve":
            raise RuntimeError("reservation failed")

    def create_forward_fn(self, descriptor, warmup):
        self.messages.append(("prepare", descriptor, warmup))
        if self.failure == "prepare":
            raise RuntimeError("preparation failed")

        def forward(mode):
            self.messages.append(("forward", descriptor, warmup, mode))
            if self.failure == "forward":
                raise RuntimeError("forward failed")

        return forward

    def after_capture(self):
        self.messages.append(("flush",))
        if self.failure == "flush":
            raise RuntimeError("flush failed")

    def fail(self):
        self.terminal = True


def make_capture_boundary_case(manager_type, *, failure=None):
    manager = object.__new__(manager_type)
    descriptor = BatchExecutionDescriptor(
        CUDAGraphMode.FULL, 4, 4, uniform_token_count=1, num_ubatches=2
    )
    manager.graphs = {}
    manager.host_descriptors = (descriptor,)
    manager.capture_failure = failure
    return SimpleNamespace(manager=manager, descriptor=descriptor)


def cpu_capture_boundary(
    manager, create_forward_fn, progress_bar_desc="Capturing CUDA graphs"
):
    """Only the upstream factory call contract, without graph/device execution."""
    if manager.capture_failure == "native":
        raise RuntimeError("native capture failed")
    for descriptor in manager.host_descriptors:
        create_forward_fn(descriptor, True)(CUDAGraphMode.NONE)
        create_forward_fn(descriptor, False)(CUDAGraphMode.NONE)
        # This is host bookkeeping to verify after_capture ordering. It is
        # deliberately not an executable graph or simulated device object.
        manager.graphs[descriptor] = "host capture boundary returned"
