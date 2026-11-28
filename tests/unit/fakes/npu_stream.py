# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host stream selection and native public-query seams; no NPU execution."""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import local
from types import SimpleNamespace

import torch
from vllm import platforms
from vllm.utils import torch_utils


@dataclass(frozen=True)
class HostStream:
    label: str
    device: torch.device


class HostStreamRecorder:
    def __init__(self, device):
        self.trace = []
        self.parent = HostStream("parent", device)
        self.compute = HostStream("compute", device)
        self.current = self.parent
        self.fail_enter = False

    def current_stream(self, device=None):
        return self.current

    @contextmanager
    def stream(self, stream):
        if self.fail_enter:
            raise RuntimeError("native stream entry failed")
        previous, self.current = self.current, stream
        self.trace.append(("enter", stream.label))
        try:
            yield
        finally:
            self.current = previous
            self.trace.append(("leave", stream.label))


class HostNPUPlatform:
    """Only the actual vLLM stream-cache cold-query platform boundary."""

    def __init__(self, runtime):
        self.current_stream = runtime.current_stream

    def is_cuda(self):
        return False

    def is_rocm(self):
        return False

    def is_cpu(self):
        return False


def make_npu_stream_case(monkeypatch, module, *, cached="parent"):
    runtime = HostStreamRecorder(torch.device("cpu"))
    cache = local()
    previous = runtime.parent
    if cached == "parent":
        cache.value = previous
    elif cached == "foreign":
        previous = HostStream("cached", runtime.parent.device)
        cache.value = previous
    elif cached == "none":
        cache.value = None
    elif cached != "unset":
        raise ValueError("unknown stream-cache setup")
    monkeypatch.setattr(module.torch, "npu", runtime, raising=False)
    monkeypatch.setattr(torch_utils, "_current_stream_tls", cache)
    monkeypatch.setattr(platforms, "current_platform", HostNPUPlatform(runtime))
    return SimpleNamespace(
        runtime=runtime,
        previous=previous,
        query=torch_utils.current_stream,
    )
