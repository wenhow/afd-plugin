# SPDX-License-Identifier: Apache-2.0
"""Stdlib CLI and child-environment checks; no native runtime or device."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "tests/npu/dsv41_a5_functional_validation.py"
TIMEOUT_SECONDS = 60
INHERITED_PATHS = MappingProxyType(
    {
        "ASCEND_CUSTOM_OPP_PATH": "/existing/vendor",
        "LD_LIBRARY_PATH": "/existing/runtime/lib",
    }
)


def load_launcher():
    spec = importlib.util.spec_from_file_location("a5_launcher", LAUNCHER)
    launcher = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(LAUNCHER.parent), *sys.path]):
        spec.loader.exec_module(launcher)
    return launcher


class TestTransformerVendor(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="a5-vendor-cli-", dir=ROOT / ".tmp"
        )
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.vendor = self.directory / "transformer vendor"
        library = self.vendor / "op_api/lib/libcust_opapi.so"
        library.parent.mkdir(parents=True)
        # Metadata checks file presence only; this fixture is never loaded.
        library.write_bytes(b"CPU CLI fixture, not an operator library\n")

    def run_cli(
        self,
        *,
        vendor=None,
        diagnose_native=False,
        diagnose_router=False,
        dspark_only=False,
        ignore_npu_occupancy=False,
    ):
        output = self.directory / "result"
        command = [
            sys.executable,
            "-B",
            "-S",
            str(LAUNCHER),
            "--model",
            str(self.directory),
            "--output-dir",
            str(output),
        ]
        if vendor is not None:
            command.extend(("--transformer-vendor", str(vendor)))
        if diagnose_native:
            command.append("--diagnose-native")
        if diagnose_router:
            command.append("--diagnose-router")
        if dspark_only:
            command.append("--dspark-only")
        if ignore_npu_occupancy:
            command.append("--ignore-npu-occupancy")
        result = subprocess.run(
            command,
            env={**os.environ, **INHERITED_PATHS},
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
        )
        return result, output

    def test_vendor_environment_reaches_a_fresh_python_process(self):
        launcher = load_launcher()
        with patch.dict(os.environ, INHERITED_PATHS):
            before = dict(os.environ)
            environment = launcher.runtime_environment(self.directory, self.vendor)
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-S",
                    "-c",
                    "import json,os; print(json.dumps({name:os.environ[name] "
                    "for name in ('ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH')}))",
                ],
                env=environment,
                capture_output=True,
                check=True,
                timeout=TIMEOUT_SECONDS,
            )
            self.assertEqual(dict(os.environ), before)
        observed = json.loads(result.stdout)
        self.assertEqual(
            observed["ASCEND_CUSTOM_OPP_PATH"],
            str(self.vendor) + os.pathsep + INHERITED_PATHS["ASCEND_CUSTOM_OPP_PATH"],
        )
        self.assertEqual(
            observed["LD_LIBRARY_PATH"],
            str(self.vendor / "op_api/lib")
            + os.pathsep
            + INHERITED_PATHS["LD_LIBRARY_PATH"],
        )

    def test_cli_uses_vendor_environment_for_all_metadata_scenes(self):
        result, output = self.run_cli(vendor=self.vendor)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        report = json.loads((output / "summary.json").read_bytes())
        self.assertTrue(report["success"])
        self.assertEqual(report["transformer_vendor"], str(self.vendor))
        self.assertEqual(
            report["runtime_environment"]["ASCEND_CUSTOM_OPP_PATH"],
            str(self.vendor) + os.pathsep + INHERITED_PATHS["ASCEND_CUSTOM_OPP_PATH"],
        )
        self.assertEqual(len(report["scenes"]), 7)
        for scene in report["scenes"]:
            metadata = json.loads((output / (scene["name"] + ".log")).read_bytes())
            self.assertIs(metadata["runtime_imported"], False)
            if scene["name"] in {"native-eager-u1", "afd-eager-u1"}:
                self.assertIs(metadata["capture_router_inputs_requested"], False)

    def test_cli_preserves_inherited_paths_without_vendor_override(self):
        result, output = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        report = json.loads((output / "summary.json").read_bytes())
        self.assertTrue(report["success"])
        self.assertIsNone(report["transformer_vendor"])
        self.assertEqual(report["runtime_environment"], INHERITED_PATHS)

    def test_missing_vendor_library_stops_before_output_creation(self):
        result, output = self.run_cli(vendor=self.directory / "missing vendor")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"requires op_api/lib/libcust_opapi.so", result.stderr)
        self.assertFalse(output.exists())

    def test_native_diagnosis_runs_one_synchronous_child_without_accepting_suite(self):
        result, output = self.run_cli(vendor=self.vendor, diagnose_native=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        report = json.loads((output / "summary.json").read_bytes())
        self.assertTrue(report["success"])
        self.assertIs(report["diagnostic_only"], True)
        self.assertEqual(report["ascend_launch_blocking"], "1")
        self.assertEqual(report["npu_acceptance"], "pending_on_A5")
        self.assertNotIn("native_afd_comparison", report)
        self.assertEqual([s["name"] for s in report["scenes"]], ["native-eager-u1"])
        metadata = json.loads((output / "native-eager-u1.log").read_bytes())
        self.assertEqual(metadata["ascend_launch_blocking"], "1")
        self.assertIs(metadata["runtime_imported"], False)
        self.assertIs(metadata["functional_worker_component_accepted"], False)
        self.assertIs(metadata["capture_router_inputs_requested"], True)

    def test_occupancy_exclusion_requires_diagnostic_mode(self):
        result, output = self.run_cli(ignore_npu_occupancy=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"requires a diagnostic mode", result.stderr)
        self.assertFalse(output.exists())

    def test_diagnostic_metadata_records_explicit_occupancy_exclusion(self):
        result, output = self.run_cli(diagnose_native=True, ignore_npu_occupancy=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        report = json.loads((output / "summary.json").read_bytes())
        self.assertIs(report["npu_occupancy_excluded"], True)
        self.assertIs(report["diagnostic_only"], True)
        self.assertEqual(report["npu_acceptance"], "pending_on_A5")

    def test_router_diagnosis_isolates_six_cases_without_model_or_runtime_imports(self):
        result, output = self.run_cli(vendor=self.vendor, diagnose_router=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        report = json.loads((output / "summary.json").read_bytes())
        self.assertTrue(report["success"])
        self.assertIs(report["diagnostic_only"], True)
        self.assertIs(report["router_diagnostic"], True)
        self.assertEqual(report["ascend_launch_blocking"], "1")
        self.assertEqual(report["npu_acceptance"], "pending_on_A5")
        self.assertNotIn("native_afd_comparison", report)
        self.assertEqual(len(report["scenes"]), 6)
        cases = set()
        for scene in report["scenes"]:
            metadata = json.loads((output / (scene["name"] + ".log")).read_bytes())
            cases.add(metadata["case"])
            self.assertEqual(metadata["ascend_launch_blocking"], "1")
            self.assertIs(metadata["runtime_imported"], False)
            self.assertIs(metadata["model_executed"], False)
            self.assertIs(metadata["collective_executed"], False)
            self.assertIs(metadata["operator_returned"], False)
            self.assertIs(metadata["functional_worker_component_accepted"], False)
        self.assertEqual(
            cases,
            {
                "e64-k2-fp32",
                "e384-k2-fp32",
                "e384-k6-fp32",
                "e64-k2-bf16",
                "e384-k6-bf16",
                "e384-k6-fp32-vision",
            },
        )

    def test_native_and_router_diagnoses_are_mutually_exclusive(self):
        result, output = self.run_cli(diagnose_native=True, diagnose_router=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"not allowed with argument", result.stderr)
        self.assertFalse(output.exists())

    def test_dspark_only_selects_two_components_without_accepting_suite(self):
        result, output = self.run_cli(vendor=self.vendor, dspark_only=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        report = json.loads((output / "summary.json").read_bytes())
        self.assertEqual(
            [scene["name"] for scene in report["scenes"]],
            ["dspark-pd-reject-continue-ready", "dspark-pd-graph-u2-cancel-replace"],
        )
        self.assertTrue(report["dspark_only"])
        self.assertEqual(report["npu_acceptance"], "pending_on_A5")
        self.assertNotIn("native_afd_comparison", report)

    def test_dspark_only_rejects_native_diagnosis(self):
        result, output = self.run_cli(dspark_only=True, diagnose_native=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"not allowed with argument", result.stderr)
        self.assertFalse(output.exists())

    def test_synchronous_environment_does_not_change_parent_or_default_children(self):
        launcher = load_launcher()
        with patch.dict(os.environ, {"ASCEND_LAUNCH_BLOCKING": "0"}):
            before = dict(os.environ)
            diagnostic = launcher.runtime_environment(
                self.directory, self.vendor, diagnose_native=True
            )
            normal = launcher.runtime_environment(self.directory, self.vendor)
            self.assertEqual(dict(os.environ), before)
        self.assertEqual(diagnostic["ASCEND_LAUNCH_BLOCKING"], "1")
        self.assertEqual(normal["ASCEND_LAUNCH_BLOCKING"], "0")


class TestInstalledVendorEnvironment(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        package = self.directory / "vllm_ascend"
        package.mkdir()
        (package / "__init__.py").write_text("raise RuntimeError('runtime imported')")
        self.vendor = package / "_cann_ops_custom/vendors/custom_transformer"
        self.library = self.vendor / "op_api/lib/libcust_opapi.so"
        self.library.parent.mkdir(parents=True)
        self.library.write_bytes(b"presence fixture; never dlopen this file")

    def prepare_in_child(self, inherited):
        script = (
            "import json,os,sys;"
            "from dsv41_a5_router_probe import prepare_native_operator_environment;"
            "record=prepare_native_operator_environment();"
            "print(json.dumps({'record':record,'ld':os.environ['LD_LIBRARY_PATH'],"
            "'runtime':[n for n in sys.modules if n.split('.')[0] in "
            "('vllm_ascend','torch','torch_npu','vllm')]}))"
        )
        return subprocess.run(
            [sys.executable, "-B", "-S", "-c", script],
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    (str(self.directory), str(LAUNCHER.parent))
                ),
                "ASCEND_CUSTOM_OPP_PATH": inherited,
                "LD_LIBRARY_PATH": "external-transformer-library",
            },
            capture_output=True,
            check=True,
            timeout=TIMEOUT_SECONDS,
        )

    def test_installed_vendor_precedes_external_roots_without_runtime_imports(self):
        for entries in (("external", "fla"), ("external", str(self.vendor), "fla")):
            with self.subTest(entries=entries):
                result = self.prepare_in_child(os.pathsep.join(entries))
                report = json.loads(result.stdout)
                self.assertEqual(
                    report["record"]["after"].split(os.pathsep),
                    [str(self.vendor), "external", "fla"],
                )
                self.assertIs(report["record"]["extension_already_imported"], False)
                self.assertEqual(report["runtime"], [])
                self.assertEqual(report["ld"], "external-transformer-library")

    def test_missing_installed_vendor_retains_external_environment(self):
        self.library.unlink()
        result = self.prepare_in_child("external")
        report = json.loads(result.stdout)
        self.assertIs(report["record"]["available"], False)
        self.assertEqual(report["record"]["before"], "external")
        self.assertNotIn("after", report["record"])
        self.assertEqual(report["runtime"], [])


class TestRouterDiagnostics(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.launcher = load_launcher()
        self.args = SimpleNamespace(
            run=True,
            python=sys.executable,
            _devices=(0, 1, 2, 3),
            output_dir=self.directory,
            timeout=10,
            ignore_npu_occupancy=False,
        )

    def test_child_exit_zero_without_operator_result_fails_diagnosis(self):
        report = {"scenes": []}
        with (
            patch.object(
                self.launcher.shared,
                "run_bounded",
                return_value={"success": True, "returncode": 0},
            ),
            patch.object(
                self.launcher.shared,
                "npu_snapshot",
                return_value={"all_eight_idle": True},
            ),
            self.assertRaisesRegex(RuntimeError, "router cases failed"),
        ):
            self.launcher.diagnose_router(self.args, self.directory, {}, report)
        self.assertEqual(len(report["scenes"]), 6)

    def test_failed_child_keeps_other_results_after_idle_check(self):
        def child_result(command, environment, snapshot, log, timeout):
            case = command[command.index("--case") + 1]
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir()
            success = case != "e64-k2-fp32"
            (output / "result.json").write_text(
                json.dumps(
                    {
                        "success": success,
                        "operator_returned": success,
                        "reference_matched": success,
                        "diagnostic_only": True,
                        "model_executed": False,
                        "collective_executed": False,
                    }
                )
            )
            return {"success": success, "returncode": int(not success)}

        report = {"scenes": []}
        with (
            patch.object(self.launcher.shared, "run_bounded", side_effect=child_result),
            patch.object(
                self.launcher.shared,
                "npu_snapshot",
                return_value={"all_eight_idle": True},
            ),
            self.assertRaisesRegex(RuntimeError, "router cases failed"),
        ):
            self.launcher.diagnose_router(self.args, self.directory, {}, report)
        self.assertEqual(len(report["scenes"]), 6)
        self.assertEqual(sum(s["result"]["success"] for s in report["scenes"]), 5)
        self.assertTrue(all(s["npu_after"]["all_eight_idle"] for s in report["scenes"]))


class TestActualRouterReplay(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.launcher = load_launcher()
        self.args = SimpleNamespace(
            run=True,
            python=sys.executable,
            _devices=(0, 1, 2, 3),
            output_dir=self.directory,
            timeout=10,
            ignore_npu_occupancy=True,
        )
        self.item = {"name": "native-eager-u1", "process": {"success": False}}
        self.capture = self.directory / "native-eager-u1/router-inputs/call-001.json"
        self.capture.parent.mkdir(parents=True)

    def write_capture(self, *, finite=True, returned=False):
        self.capture.write_text(
            json.dumps(
                {
                    "capture_complete": True,
                    "operator_called": True,
                    "operator_returned": returned,
                    "tensors": {"x": {"finite": finite}},
                }
            )
        )
        self.capture.with_suffix(".pt").write_bytes(b"transport fixture only")

    def test_replay_retains_original_failure_and_records_independent_result(self):
        self.write_capture()

        def replay_result(command, environment, snapshot, log, timeout):
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir()
            (output / "result.json").write_text(json.dumps({"success": True}))
            return {"success": True, "returncode": 0}

        with (
            patch.object(
                self.launcher.shared, "run_bounded", side_effect=replay_result
            ),
            patch.object(
                self.launcher.shared,
                "npu_snapshot",
                return_value={"all_eight_idle": False},
            ),
        ):
            self.launcher.replay_failed_router(self.args, self.directory, {}, self.item)
        replay = self.item["router_actual_replay"]
        self.assertIs(replay["attempted"], True)
        self.assertIs(replay["result"]["success"], True)
        self.assertIs(self.item["process"]["success"], False)
        self.assertIs(replay["npu_before"]["all_eight_idle"], False)

    def test_replay_skips_nonfinite_or_successful_capture(self):
        for finite, returned in ((False, False), (True, True)):
            with self.subTest(finite=finite, returned=returned):
                self.write_capture(finite=finite, returned=returned)
                with patch.object(
                    self.launcher.shared,
                    "npu_snapshot",
                    return_value={"all_eight_idle": False},
                ):
                    self.launcher.replay_failed_router(
                        self.args, self.directory, {}, self.item
                    )
                self.assertIs(self.item["router_actual_replay"]["attempted"], False)
                self.assertIn("skip_reason", self.item["router_actual_replay"])

    def test_replay_requires_owned_child_cleanup(self):
        self.write_capture()
        self.item["process"]["surviving_owned_pids"] = [123]
        self.launcher.replay_failed_router(self.args, self.directory, {}, self.item)
        self.assertIs(self.item["router_actual_replay"]["attempted"], False)
        self.assertEqual(
            self.item["router_actual_replay"]["skip_reason"],
            "native child processes remain",
        )


if __name__ == "__main__":
    unittest.main()
