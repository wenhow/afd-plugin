# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit diagnostic record and requested-sync protocol, without device IO."""

from types import SimpleNamespace


class StageDiagnosticRecorder:
    def __init__(self):
        self.records = []
        self.events = []
        self.fail_sync = False
        self.fail_event = None

    def emit(self, record):
        self.records.append(dict(record))
        self.events.append(record["event"])
        if record["event"] == self.fail_event:
            raise RuntimeError("requested diagnostic sink failed")

    def synchronize(self):
        self.events.append("device_sync")
        if self.fail_sync:
            raise RuntimeError("requested diagnostic device sync failed")


def make_diagnostics_case(api, selection, *, world_rank=0, role_rank=0):
    recorder = StageDiagnosticRecorder()
    diagnostics = api.HcclStageDiagnostics(
        selection,
        world_rank=world_rank,
        role_rank=role_rank,
        emit=recorder.emit,
        synchronize=recorder.synchronize,
    )
    return SimpleNamespace(
        diagnostics=diagnostics,
        recorder=recorder,
        records=recorder.records,
        events=recorder.events,
    )
