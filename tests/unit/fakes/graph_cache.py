# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Graph ownership seam; this fake executes no device work."""

from dataclasses import dataclass


@dataclass
class GraphResources:
    value: int = 0


class ReplayGraph:
    def __init__(self, *, reset_error=False):
        self.reset_error = reset_error
        self.closed = False

    def replay(self):
        if self.closed:
            raise RuntimeError("graph was reset")

    def reset(self):
        if self.reset_error:
            raise RuntimeError("graph reset failed")
        self.closed = True


class DeviceCompletion:
    def __init__(self):
        self.failed = False

    def synchronize(self):
        if self.failed:
            raise RuntimeError("device completion failed")
