"""Audit the two original A5 uploads without importing a device runtime."""

import hashlib
import json
import zipfile
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[4]
ARTIFACT_ARCHIVE = "a095f8a7ad73413a9fa98abd3eceaf49.zip"
REPORT_ARCHIVE = "e19a3975e09d411a838711d3b1e7c245.zip"
PREVIOUS_INPUT_SHA256 = (
    "6bf7ca17a89d2e6dbcbc1c9878b567a5e2a3dca2357d1be2a84eaad37e93d6e0"
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    with (
        zipfile.ZipFile(EVIDENCE / ARTIFACT_ARCHIVE) as artifacts,
        zipfile.ZipFile(EVIDENCE / REPORT_ARCHIVE) as reports,
    ):
        summary = json.loads(reports.read("summary.json"))
        result = json.loads(reports.read("native-eager-u1-result.json"))
        process = json.loads(reports.read("native-eager-u1.process.json"))
        runtime = json.loads(reports.read("native-runtime.json"))
        assert summary["success"] and summary["diagnostic_only"]
        assert summary["npu_acceptance"] == "diagnostic_only"
        assert len(summary["scenes"]) == 1
        assert summary["scenes"][0]["name"] == "native-eager-u1"
        assert summary["scenes"][0]["result"] == result
        preflight = summary["runtime_preflight"]
        assert preflight["success"] and preflight["returncode"] == 0
        assert not preflight["timed_out"]
        assert not preflight["forced_killed_pids"]
        assert not preflight["surviving_owned_pids"]
        assert process["returncode"] == 0 and not process["timed_out"]
        assert not process["forced_killed_pids"]
        assert not process["surviving_owned_pids"]
        for name in (
            "success",
            "model_loaded",
            "attention_kv_initialized",
            "compile_or_warm_up_returned",
            "inference_executed",
            "mode_execution_accepted",
            "request_lifecycle_accepted",
            "functional_worker_component_accepted",
            "loaded_parameters_unchanged",
            "worker_shutdown_completed",
            "native_groups_destroyed",
            "ascend_groups_completed",
            "model_groups_completed",
            "distributed_environment_completed",
        ):
            assert result[name], name
        assert not result["cleanup_failures"] and not result["afd_modules_imported"]
        assert result["configuration"]["load_format"] == "dummy"
        assert runtime["sources_match"] and runtime["source_count"] == 194
        assert all(item["matches"] for item in runtime["sources"].values())
        lock = json.loads(
            (ROOT / "reports/dsv41-a5-functional/bundle-lock.json").read_bytes()
        )
        assert summary["runtime_source_sha256"] == lock["runtime_source_sha256"]
        preparation = result["operator_environment_preparation"]
        assert preparation["available"]
        assert not preparation["extension_already_imported"]
        assert preparation["after"].split(":")[0] == preparation["vendor"]
        assert all(
            part in preparation["after"].split(":")
            for part in preparation["before"].split(":")
        )
        lifecycle = result["request_lifecycle"]
        assert json.loads(artifacts.read("lifecycle-progress.json")) == lifecycle
        assert json.loads(artifacts.read("loaded-model.json")) == result["loaded_model"]
        assert lifecycle["scheduler_empty"] and lifecycle["worker_requests_retired"]
        assert lifecycle["replacement_slot_matches_retired"]
        hidden = []
        for step in lifecycle["steps"]:
            record = step["hidden"]
            data = artifacts.read(record["raw_file"])
            assert len(data) == record["bytes"]
            assert digest(data) == record["sha256"]
            assert record["finite"]
            hidden.append({"ordinal": step["ordinal"], **record})
        calls = result["router_input_capture"]["calls"]
        for call in calls:
            name = Path(call["metadata_file"]).name
            assert json.loads(artifacts.read("router-inputs/" + name)) == call
            data = artifacts.read("router-inputs/" + Path(call["inputs_file"]).name)
            assert digest(data) == call["inputs_sha256"]
            assert call["capture_complete"] and call["operator_called"]
            assert call["operator_returned"]
            assert all(item["finite"] for item in call["tensors"].values())
        assert calls[0]["inputs_sha256"] == PREVIOUS_INPUT_SHA256
        previous_input = (
            EVIDENCE.parent
            / "a5-router-import-order-20261118/native-inputs/call-001.pt"
        )
        assert digest(previous_input.read_bytes()) == PREVIOUS_INPUT_SHA256
        assessment = {
            "uploads": {
                name: digest((EVIDENCE / name).read_bytes())
                for name in (ARTIFACT_ARCHIVE, REPORT_ARCHIVE)
            },
            "runtime_source_sha256": summary["runtime_source_sha256"],
            "source_files": summary["source_files"],
            "production_files": summary["production_files"],
            "native_sources_match": runtime["sources_match"],
            "native_source_count": runtime["source_count"],
            "package_versions": runtime["package_versions"],
            "geometry": result["first_declaration"],
            "device": result["actual_device"],
            "scope": result["scope"],
            "success": result["success"],
            "diagnostic_only": summary["diagnostic_only"],
            "npu_acceptance": summary["npu_acceptance"],
            "npu_occupancy_excluded_per_user": summary["npu_occupancy_excluded"],
            "process_returncode": process["returncode"],
            "timed_out": process["timed_out"],
            "forced_killed_pids": process["forced_killed_pids"],
            "surviving_owned_pids": process["surviving_owned_pids"],
            "operator_environment_preparation": preparation,
            "router_calls_verified": len(calls),
            "first_input_matches_previous_failure": True,
            "first_input_sha256": calls[0]["inputs_sha256"],
            "symbol_providers_before_call": calls[0]["symbol_providers_before_call"],
            "libraries_before_call": calls[0]["libraries_before_call"],
            "request_rounds": len(lifecycle["steps"]),
            "outputs": lifecycle["outputs"],
            "scheduler_empty": lifecycle["scheduler_empty"],
            "worker_requests_retired": lifecycle["worker_requests_retired"],
            "replacement_slot_matches_retired": (
                lifecycle["replacement_slot_matches_retired"]
            ),
            "hidden_artifacts_verified": hidden,
            "worker_shutdown_completed": result["worker_shutdown_completed"],
            "native_groups_destroyed": result["native_groups_destroyed"],
            "cleanup_failures": result["cleanup_failures"],
            "full_checkpoint_accepted": result["full_checkpoint_accepted"],
            "numerical_equivalence_accepted": result["numerical_equivalence_accepted"],
            "dspark_accepted": result["dspark_accepted"],
            "pd_accepted": result["pd_accepted"],
            "engram_accepted": result["engram_accepted"],
            "performance_accepted": result["performance_accepted"],
            "seven_scene_suite_accepted": False,
        }
    (EVIDENCE / "assessment.json").write_text(
        json.dumps(assessment, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                name: assessment[name]
                for name in (
                    "success",
                    "npu_acceptance",
                    "request_rounds",
                    "router_calls_verified",
                    "first_input_matches_previous_failure",
                    "worker_shutdown_completed",
                    "native_groups_destroyed",
                    "seven_scene_suite_accepted",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
