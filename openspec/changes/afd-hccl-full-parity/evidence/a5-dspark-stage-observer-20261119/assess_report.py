#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assess the two supplied scene archives against their original outer summary."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def main() -> None:
    directory = Path(__file__).resolve().parent
    summary = json.loads((directory / "outer-summary.json").read_text("utf-8"))
    scenes = {scene["name"]: scene for scene in summary["scenes"]}
    with zipfile.ZipFile(directory / "reject-input.zip") as archive:
        prefix = "dspark-pd-reject-continue-ready/"
        marker = json.loads(archive.read(prefix + "consumer-verified.json"))
        producer = json.loads(archive.read(prefix + "ffn_producer.json"))
        assert marker["run_id"] == producer["run_id"]
        reject = {
            "run_id": marker["run_id"],
            "outer_scene": scenes["dspark-pd-reject-continue-ready"],
            "supplied_ffn_producer_success": producer["success"],
            "detailed_missing_reports": [
                name
                for name in ("summary.json", "kv_consumer.json", "kv_producer.json")
                if prefix + name not in archive.namelist()
            ],
            "numerical_acceptance_counts_reaudit": "not_in_supplied_role_reports",
        }
    with zipfile.ZipFile(directory / "graph-input.zip") as archive:
        prefix = "dspark-pd-graph-u2-cancel-replace/"
        inner = json.loads(archive.read(prefix + "summary.json"))
        consumer = json.loads(archive.read(prefix + "kv_consumer.json"))
        detail = consumer["target_graph_decode"]
        startup = consumer["native_live_owner"]["target_graph"]
        parents = startup["host_records"]
        last = next(
            row
            for row in reversed(parents)
            if row["event"] == "graph_parent_scope_returned"
        )
        returned = [
            {key: row[key] for key in ("event", "layer", "stage", "phase")}
            for row in parents
            if row["event"] == "compute_returned"
            and row["transaction_id"] == last["transaction_id"]
        ]
        graph = {
            "inner_summary": inner,
            "first_recorded_exception": consumer["traceback"],
            "observed_python_entrypoint": detail["python_target_entrypoint"],
            "observed_python_calls": detail["python_target_calls"],
            "expected_target_batches": detail["expected_target_batches"],
            "last_parent_returned": last,
            "compute_returned_host_records": returned,
            "native_capture_completed": startup["native_capture_completed"],
            "capture_kv_bytes_unchanged": startup["capture_kv_bytes_unchanged"],
            "capture_physical_pages_unchanged": startup[
                "capture_physical_pages_unchanged"
            ],
            "replay_observed": startup["replay_observed"],
            "heap_corruption_also_observed": "corrupted double-linked list"
            in archive.read(prefix + "kv_consumer.log").decode(),
            "native_groups_destroyed": consumer["native_groups_destroyed"],
            "cancel_replace_completed": consumer["native_dspark_request_lifecycle"][
                "concurrent_survivor_replacement_executed"
            ],
            "full_model_accepted": consumer["full_model_accepted"],
        }
    result = {
        "input_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("graph-input.zip", "reject-input.zip", "outer-summary.json")
        },
        "reported_runtime_source_sha256": summary["runtime_source_sha256"],
        "matches_04da709_frozen_archive": summary["runtime_source_sha256"]
        == "1bdf52cdfc3eb5f7f6250111e4bcb462cda9e1565910ea202c2c92466ff34d49",
        "runtime_preflight_success": summary["runtime_preflight"]["success"],
        "outer_success": summary["success"],
        "outer_traceback": summary["traceback"],
        "reject": reject,
        "graph": graph,
        "verdict": "reject_passed_in_outer_summary_graph_observer_failed_then_abort",
        "NPU_occupancy": "excluded_from_diagnosis_per_user",
        "A5_replay_cancel_replace_and_clean_exit_after_fix": "pending",
    }
    (directory / "assessment.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
