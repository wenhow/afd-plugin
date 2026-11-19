#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the real frozen report validators without starting workers or devices."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import tarfile
import tempfile
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
DIRECTORY = Path(__file__).resolve().parent
OUTER_INPUT = (
    ROOT / "openspec/changes/afd-hccl-full-parity/evidence/"
    "a5-dspark-graph-final-audit-20261119/initial-input.zip"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "reports/dsv41-a5-functional/runtime-source.tar.gz",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-device-audit", action="store_true")
    parser.add_argument(
        "--expected-devices", help="Attention,FFN override for negatives"
    )
    args = parser.parse_args()
    manifest = json.loads((DIRECTORY / "input-manifest.json").read_text("utf-8"))
    with tarfile.open(DIRECTORY / "inner-reports.tar.gz") as archive:
        for name, expected_identity in manifest.items():
            recorded = archive.extractfile(name).read()
            assert len(recorded) == expected_identity["bytes"]
            assert hashlib.sha256(recorded).hexdigest() == expected_identity["sha256"]
        raw = archive.extractfile("summary.json").read()
        summary = json.loads(raw)
        for role in ("kv_consumer", "kv_producer", "ffn_consumer"):
            assert summary["workers"][role] == json.load(
                archive.extractfile(f"{role}.json")
            )
    with zipfile.ZipFile(OUTER_INPUT) as archive:
        name = next(n for n in archive.namelist() if n.endswith("/summary.json"))
        outer = json.loads(archive.read(name))
    command = next(
        item["process"]["command"]
        for item in outer["scenes"]
        if item["name"] == "dspark-pd-graph-u2-cancel-replace"
    )
    devices = [
        int(value) for value in command[command.index("--devices") + 1].split(",")
    ]
    expected = {"attention": devices[1], "ffn": devices[3]}
    if args.expected_devices:
        attention_device, ffn_device = map(int, args.expected_devices.split(","))
        expected = {"attention": attention_device, "ffn": ffn_device}
    result = {
        "success": False,
        "scope": "read_only_post_run_audit_of_recorded_NPU_component",
        "new_NPU_execution": False,
        "original_run_id": summary["run_id"],
        "original_success": summary["success"],
        "original_traceback": summary["traceback"],
        "original_summary_sha256": hashlib.sha256(raw).hexdigest(),
        "role_exit_codes": summary["returncodes"],
        "expected_local_devices": expected,
        "expected_devices_source": (
            "explicit negative override"
            if args.expected_devices
            else "outer scene --devices role placement"
        ),
        "bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
    }
    try:
        assert all(code == 0 for code in summary["returncodes"].values())
        assert not summary["forced_killed_pids"] and not summary["timed_out"]
        assert all(worker["success"] for worker in summary["workers"].values())
        with tempfile.TemporaryDirectory(
            prefix="dspark-report-audit-", dir=ROOT / ".tmp"
        ) as name:
            snapshot = Path(name)
            with tarfile.open(args.bundle) as archive:
                for member in archive.getmembers():
                    if not member.isfile():
                        continue
                    path = (snapshot / member.name).resolve()
                    assert path.is_relative_to(snapshot.resolve())
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.extractfile(member).read())
            sys.path[:0] = [str(snapshot / "tests/npu"), str(snapshot)]
            decode = importlib.import_module(
                "hccl_native_attention_pd_dspark_decode_probe"
            )
            assert set(summary["returncodes"]) == set(decode.ROLES)
            cross = importlib.import_module(
                "hccl_native_attention_pd_dspark_cross_layer_probe"
            )
            workers = summary["workers"]
            producer, consumer = workers["kv_producer"], workers["kv_consumer"]
            result["recorded_graph_observer_accepted"] = consumer[
                "target_graph_decode"
            ]["accepted"]
            assert producer["typed_done_observed"]
            assert consumer["native_decode_and_verification_accepted"]
            assert len(producer["sources"]) == 3
            assert set(producer["typed_done_source_ids"]) == {
                item["request_id"] for item in producer["sources"]
            }
            for role in ("ffn_producer", "ffn_consumer"):
                assert workers[role]["native_ffn"]["native_daemon_retired"]
            result["producer_protocol"] = decode.verify_pair_messages(
                producer, workers["ffn_producer"], (len(decode.PROMPT) - 1,) * 3
            )
            arguments = {} if args.legacy_device_audit else {"local_devices": expected}
            result["consumer_protocol"] = cross.verify_cross_layer_lifecycle(
                consumer, workers["ffn_consumer"], **arguments
            )
        result["success"] = True
    except Exception:
        result["audit_traceback"] = traceback.format_exc()
    result["native_runtime_imports"] = sorted(
        name
        for name in sys.modules
        if name.split(".")[0] in ("torch", "torch_npu", "vllm", "vllm_ascend")
    )
    if result["native_runtime_imports"]:
        result["success"] = False
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "success",
                    "expected_local_devices",
                    "native_runtime_imports",
                )
            }
        )
    )
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
