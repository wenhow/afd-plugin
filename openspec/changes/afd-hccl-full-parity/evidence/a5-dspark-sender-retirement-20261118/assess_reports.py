"""Independently audit the supplied reports; never rewrite original outcomes."""

import hashlib
import json
import tarfile
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
FIELDS = ("shape", "dtype", "numel", "bytes", "sha256", "finite")


def main():
    with tarfile.open(EVIDENCE / "uploaded-reports.tar.gz") as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    summary = json.loads(files["summary.json"])
    scenes = summary["scenes"]
    native = scenes[0]["result"]
    checked = {}
    for scene in scenes[:5]:
        process = scene["process"]
        assert process["success"] and process["returncode"] == 0
        assert not process["timed_out"]
        assert not process["forced_killed_pids"]
        assert not process["surviving_owned_pids"]
        result = scene["result"]
        roles = result.get("roles", {"native": result})
        for role in roles.values():
            assert role["success"] and role["functional_worker_component_accepted"]
            assert not role["cleanup_failures"]
            assert role["native_groups_destroyed"]
        if "attention" in roles:
            attention, ffn = roles["attention"], roles["ffn"]
            names = attention["canonical_parameter_groups"]["ffn_to_native_names"]
            params = {
                **attention["loaded_model"]["parameters"],
                **{
                    names[name]: value
                    for name, value in ffn["loaded_model"]["parameters"].items()
                },
            }
            assert set(params) == set(native["loaded_model"]["parameters"])
            for name, value in params.items():
                expected = native["loaded_model"]["parameters"][name]
                assert all(value[k] == expected[k] for k in FIELDS), name
            assert (
                attention["request_lifecycle"]["outputs"]
                == native["request_lifecycle"]["outputs"]
            )
            checked[scene["name"]] = {"parameters": len(params), "outputs_match": True}
    failed = "dspark-pd-reject-continue-ready/"
    dspark = json.loads(files[failed + "summary.json"])
    producer = json.loads(files[failed + "kv_producer.json"])
    consumer = json.loads(files[failed + "kv_consumer.json"])
    assert dspark["returncodes"] == {
        "kv_producer": -6,
        "kv_consumer": 0,
        "ffn_producer": 0,
        "ffn_consumer": 0,
    }
    assert b"corrupted double-linked list" in files[failed + "kv_producer.log"]
    assert producer["success"] and consumer["success"]
    assert producer["ready_frontier"]["status"] == "passed"
    assert consumer["native_post_rejection_continuation_accepted"]
    assert consumer["typed_read_and_whole_pool_oracle_passed"]
    assert not summary["success"] and not dspark["success"]
    result = {
        "uploaded_archive_sha256": hashlib.sha256(
            (EVIDENCE / "uploaded-reports.tar.gz").read_bytes()
        ).hexdigest(),
        "file_sha256": {n: hashlib.sha256(v).hexdigest() for n, v in files.items()},
        "runtime_source_sha256": summary["runtime_source_sha256"],
        "normal_scenes_independently_checked": checked,
        "dspark_returncodes": dspark["returncodes"],
        "dspark_component_observations_only": {
            "ready_frontier": producer["ready_frontier"]["status"],
            "rejected_per_step": [
                s["native_num_rejected"] for s in consumer["decode_steps"]
            ],
            "sampled_per_step": [
                s["sampled_token_ids"] for s in consumer["decode_steps"]
            ],
        },
        "overall_original_success": False,
        "npu_occupancy_excluded_per_user": True,
        "full_checkpoint_accepted": False,
        "native_heap_corruption_root_cause": "unconfirmed without native crash stack",
    }
    (EVIDENCE / "assessment.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
