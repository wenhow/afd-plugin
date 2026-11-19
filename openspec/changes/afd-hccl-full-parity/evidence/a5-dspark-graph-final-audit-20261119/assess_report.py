#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record the supplied partial scene evidence without inventing its inner error."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


def main() -> None:
    directory = Path(__file__).resolve().parent
    path = directory / "initial-input.zip"
    with zipfile.ZipFile(path) as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]
        summary_name = next(name for name in names if name.endswith("/summary.json"))
        summary = json.loads(archive.read(summary_name))
        prefix = next(
            name for name in names if name.endswith("/consumer-verified.json")
        )
        prefix = prefix.removesuffix("consumer-verified.json")
        marker = json.loads(archive.read(prefix + "consumer-verified.json"))
        ffn = json.loads(archive.read(prefix + "ffn_producer.json"))
        assert marker["run_id"] == ffn["run_id"]
        scenes = {scene["name"]: scene for scene in summary["scenes"]}
        errors = {}
        for name in names:
            if name.endswith(".log"):
                raw = archive.read(name).decode("utf-8", errors="replace")
                lines = raw.splitlines()
                errors[name] = [
                    {"line": number, "text": line}
                    for number, line in enumerate(lines, 1)
                    if any(
                        word in line
                        for word in (
                            "Traceback",
                            "AssertionError",
                            "corrupted double-linked list",
                            "Fatal Python error",
                            "ERROR ",
                        )
                    )
                ]
        result = {
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "run_id": marker["run_id"],
            "reported_runtime_source_sha256": summary["runtime_source_sha256"],
            "matches_51a0954_archive": summary["runtime_source_sha256"]
            == "c576f79af0732bfcedc3deab35d8e0edac34a0f56e4475a0e61a5193d2212379",
            "runtime_preflight_success": summary["runtime_preflight"]["success"],
            "reject_scene": scenes["dspark-pd-reject-continue-ready"],
            "graph_scene": scenes["dspark-pd-graph-u2-cancel-replace"],
            "consumer_verified_marker_present": True,
            "ffn_producer_success": ffn["success"],
            "ffn_producer_native_daemon_retired": ffn["native_ffn"][
                "native_daemon_retired"
            ],
            "supplied_log_errors": errors,
            "missing_inner_reports": [
                prefix + name
                for name in (
                    "summary.json", "kv_producer.json", "kv_consumer.json",
                    "ffn_consumer.json",
                )
                if prefix + name not in names
            ],
            "outer_traceback": summary["traceback"],
            "verdict": "graph_process_returned_1_inner_exception_not_supplied",
            "physical_overlap_performance_or_full_checkpoint_accepted": False,
            "NPU_occupancy": "excluded_from_diagnosis_per_user",
        }
    (directory / "assessment.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
