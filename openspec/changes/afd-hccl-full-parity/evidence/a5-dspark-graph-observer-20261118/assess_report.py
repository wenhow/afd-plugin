#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize the supplied failed scene without importing a device runtime."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def main() -> None:
    directory = Path(__file__).resolve().parent
    source = directory / "graph-u2-cancel-input.zip"
    with zipfile.ZipFile(source) as archive:
        summary = json.loads(archive.read("summary.json"))
        consumer = json.loads(archive.read("kv_consumer.json"))
        startup = consumer["native_live_owner"]["target_graph"]
        log = archive.read("kv_consumer.log").decode("utf-8", errors="replace")
        result = {
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "files": len([item for item in archive.infolist() if not item.is_dir()]),
            "summary": summary,
            "consumer_success": consumer["success"],
            "first_recorded_exception": consumer["traceback"],
            "target_graph_decode": consumer["target_graph_decode"],
            "startup": {
                name: startup[name]
                for name in (
                    "native_capture_completed",
                    "capture_completed_without_requests",
                    "capture_completed_without_pd_metadata",
                    "capture_kv_bytes_unchanged",
                    "capture_physical_pages_unchanged",
                    "replay_observed",
                )
            },
            "native_groups_destroyed": consumer["native_groups_destroyed"],
            "heap_corruption_also_observed": "corrupted double-linked list" in log,
            "full_model_accepted": consumer["full_model_accepted"],
            "reject_success_evidence": "user_report_only_not_in_this_zip",
            "frozen_source_identity": "outer_source_lock_and_snapshot_not_supplied",
            "verdict": "failed_observer_assertion_then_process_abort_not_accepted",
            "after_fix_A5_replay_cancel_and_clean_exit": "pending",
        }
    (directory / "assessment.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
