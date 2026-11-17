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
from types import MappingProxyType
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

    def run_cli(self, *, vendor=None, diagnose_native=False):
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
        result = subprocess.run(
            command,
            env={**os.environ, **INHERITED_PATHS},
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
        )
        return result, output

    def test_vendor_environment_reaches_a_fresh_python_process(self):
        spec = importlib.util.spec_from_file_location("a5_launcher", LAUNCHER)
        launcher = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(LAUNCHER.parent), *sys.path]):
            spec.loader.exec_module(launcher)
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

    def test_synchronous_environment_does_not_change_parent_or_default_children(self):
        spec = importlib.util.spec_from_file_location("a5_debug_launcher", LAUNCHER)
        launcher = importlib.util.module_from_spec(spec)
        with patch.object(sys, "path", [str(LAUNCHER.parent), *sys.path]):
            spec.loader.exec_module(launcher)
        with patch.dict(os.environ, {"ASCEND_LAUNCH_BLOCKING": "0"}):
            before = dict(os.environ)
            diagnostic = launcher.runtime_environment(
                self.directory, self.vendor, diagnose_native=True
            )
            normal = launcher.runtime_environment(self.directory, self.vendor)
            self.assertEqual(dict(os.environ), before)
        self.assertEqual(diagnostic["ASCEND_LAUNCH_BLOCKING"], "1")
        self.assertEqual(normal["ASCEND_LAUNCH_BLOCKING"], "0")


if __name__ == "__main__":
    unittest.main()
