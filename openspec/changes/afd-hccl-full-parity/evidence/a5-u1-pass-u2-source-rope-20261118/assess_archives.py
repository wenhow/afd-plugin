"""Verify U1 artifacts and preserve the first U2 failure from uploaded ZIPs."""

import hashlib
import json
import zipfile
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parent
UPLOADS = (
    "8c282423171142818bdacce74b45ed6a.zip",
    "6c4d6f688c8f4106a85897c3ff665742.zip",
    "89ceeac6f8f74cfcabe3aeeb86948d81.zip",
)
IDENTITY_FIELDS = ("shape", "dtype", "numel", "bytes", "sha256", "finite")


def main():
    with zipfile.ZipFile(EVIDENCE / UPLOADS[1]) as archive:
        pair = json.loads(archive.read("result.json"))
    attention, ffn = pair["roles"]["attention"], pair["roles"]["ffn"]
    assert pair["success"] and pair["returncodes"] == {"attention": 0, "ffn": 0}
    for role in (attention, ffn):
        assert role["success"] and role["functional_worker_component_accepted"]
        assert role["worker_shutdown_completed"] and role["native_groups_destroyed"]
        assert not role["cleanup_failures"]
    with zipfile.ZipFile(EVIDENCE / UPLOADS[0]) as archive:
        assert json.loads(archive.read("attention/result.json")) == attention
        for step in attention["request_lifecycle"]["steps"]:
            hidden = step["hidden"]
            data = archive.read("attention/" + hidden["raw_file"])
            assert len(data) == hidden["bytes"]
            assert hashlib.sha256(data).hexdigest() == hidden["sha256"]
    native_archive = EVIDENCE.parent / (
        "a5-native-vendor-first-pass-20261118/e19a3975e09d411a838711d3b1e7c245.zip"
    )
    with zipfile.ZipFile(native_archive) as archive:
        native = json.loads(archive.read("native-eager-u1-result.json"))
    names = attention["canonical_parameter_groups"]["ffn_to_native_names"]
    parameters = {
        **attention["loaded_model"]["parameters"],
        **{
            names[name]: item
            for name, item in ffn["loaded_model"]["parameters"].items()
        },
    }
    assert set(parameters) == set(native["loaded_model"]["parameters"])
    for name, actual in parameters.items():
        expected = native["loaded_model"]["parameters"][name]
        assert all(actual[key] == expected[key] for key in IDENTITY_FIELDS), name
    assert (
        attention["request_lifecycle"]["outputs"]
        == native["request_lifecycle"]["outputs"]
    )
    with zipfile.ZipFile(EVIDENCE / UPLOADS[2]) as archive:
        process = json.loads(archive.read("afd-eager-u2-streams.process.json"))
        failed = {}
        control_kinds = {}
        for role in ("attention", "ffn"):
            failed[role] = json.loads(
                archive.read(f"afd-eager-u2-streams/{role}/result.json")
            )
            control = json.loads(
                archive.read(f"afd-eager-u2-streams/{role}/staged-controls.json")
            )
            control_kinds[role] = [
                item["encoded_message"]["kind"] for item in control["messages"]
            ]
            assert control_kinds[role] == ["HELLO", "HELLO"]
        assert (
            "Enable device metadata before building compressor metadata"
            in (failed["attention"]["traceback"])
        )
        assert all(not item["inference_executed"] for item in failed.values())
        assert process["returncode"] == 1 and not process["timed_out"]
        assert not process["forced_killed_pids"]
        assert not process["surviving_owned_pids"]
    assessment = {
        "uploads": {
            name: hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
            for name in UPLOADS
        },
        "npu_occupancy_excluded_per_user": True,
        "u1": {
            "pair_success": pair["success"],
            "returncodes": pair["returncodes"],
            "scope": attention["scope"],
            "devices": [attention["actual_device"], ffn["actual_device"]],
            "geometry": attention["first_declaration"],
            "request_rounds": len(attention["request_lifecycle"]["steps"]),
            "outputs": attention["request_lifecycle"]["outputs"],
            "canonical_parameter_count": len(parameters),
            "matches_previous_native_parameters": True,
            "matches_previous_native_outputs": True,
            "native_comparison_scope": "Independent audit against saved diagnostic run",
            "source_identity_limit": "Current suite source report not uploaded",
            "supervisor_limit": "U1 supervisor report absent; role exits available",
            "role_cleanup_failures": {
                "attention": attention["cleanup_failures"],
                "ffn": ffn["cleanup_failures"],
            },
        },
        "u2": {
            "success": False,
            "process": process,
            "attention_first_error": failed["attention"]["traceback"],
            "ffn_followup_error": failed["ffn"]["traceback"],
            "control_kinds": control_kinds,
            "inference_executed": False,
            "cleanup_failures": {
                role: value["cleanup_failures"] for role, value in failed.items()
            },
        },
        "u2_fix_device_validated": False,
        "seven_scene_suite_accepted": False,
        "full_checkpoint_accepted": False,
        "engram_accepted": False,
        "performance_accepted": False,
    }
    (EVIDENCE / "assessment.json").write_text(
        json.dumps(assessment, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "U1 roles passed; 79 parameter identities and six-round outputs match "
        "the saved native baseline. U2 failed before compute; raw failure preserved."
    )


if __name__ == "__main__":
    main()
