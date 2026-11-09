# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Host metadata handoff around an explicit CPU native-replay boundary."""

from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import torch

from tests.unit.fakes.graph_capture import make_capture_boundary_case


class ReplayNotifications:
    def __init__(self, manager, *, failure=None):
        self.manager, self.failure = manager, failure
        self.previous_metadata = manager.metadata
        self.accepted_metadata = SimpleNamespace(rows=4)
        self.messages = []
        self.terminal = False

    @contextmanager
    def before_replay(self, descriptor):
        if self.failure == "before":
            raise RuntimeError("metadata handoff failed")
        self.manager.metadata = self.accepted_metadata
        self.messages.append("accepted metadata")
        try:
            yield
        finally:
            self.manager.metadata = self.previous_metadata
            self.messages.append("restored metadata")

    def after_replay(self, descriptor):
        if self.manager.metadata is not self.previous_metadata:
            raise AssertionError("COMPLETE preceded metadata restoration")
        self.messages.append("COMPLETE")
        if self.failure == "after":
            raise RuntimeError("COMPLETE failed")

    def fail(self):
        self.terminal = True


def cpu_native_replay(manager, descriptor):
    """Observable host boundary; no graph, NPU launch or kernel is emulated."""
    if manager.replay_failure == "native":
        raise RuntimeError("native replay failed")
    if manager.metadata is not manager.notifications.accepted_metadata:
        raise AssertionError("native lifecycle read stale metadata")
    manager.notifications.messages.append("native replay")
    return torch.full((manager.metadata.rows,), descriptor.num_tokens)


def make_replay_boundary_case(manager_type, *, failure=None):
    case = make_capture_boundary_case(manager_type)
    case.manager.metadata = SimpleNamespace(rows=1)
    case.manager.replay_failure = failure
    case.manager.run_fullgraph = MethodType(cpu_native_replay, case.manager)
    case.notifications = ReplayNotifications(case.manager, failure=failure)
    case.manager.notifications = case.notifications
    return case
