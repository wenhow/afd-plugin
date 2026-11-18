"""Exercise diagnostic capture with real CPU PyTorch operators, without NPU work."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

import torch
from dsv41_a5_router_probe import capture_router_inputs, restore_tensor_layout
from torch.utils._python_dispatch import TorchDispatchMode


class CaptureChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Real CPU kernels exercise the Python observer across an opaque
        # custom-op boundary. This does not simulate a device kernel.
        cls.library = torch.library.Library("dsv41_capture_cpu", "DEF")
        cls.library.define("add(Tensor left, Tensor right, float alpha=1) -> Tensor")
        cls.library.define("mm(Tensor left, Tensor right) -> Tensor")
        cls.library.define("outer(Tensor left, Tensor right) -> Tensor")
        cls.library.impl(
            "add",
            lambda left, right, alpha=1: torch.add(left, right, alpha=alpha),
            "CPU",
        )
        cls.library.impl("mm", torch.mm, "CPU")
        cls.library.impl(
            "outer",
            lambda left, right: torch.ops.dsv41_capture_cpu.add(left, right),
            "CPU",
        )

    def setUp(self):
        self.output = Path(sys.argv[1]) / self.id().split(".")[-1]

    def test_inputs_and_result_are_preserved(self):
        left, right = torch.arange(6).reshape(2, 3), torch.ones(2, 3)
        with capture_router_inputs(self.output, "dsv41_capture_cpu::add") as capture:
            actual = torch.ops.dsv41_capture_cpu.add(left, right, alpha=2)
        torch.testing.assert_close(actual, left + right * 2)
        record = capture["calls"][0]
        saved = torch.load(record["inputs_file"], weights_only=True)
        torch.testing.assert_close(saved["left"], left)
        torch.testing.assert_close(saved["right"], right)
        self.assertEqual(saved["alpha"], 2)
        self.assertTrue(record["capture_complete"])
        self.assertTrue(record["operator_returned"])
        self.assertEqual(record["tensors"]["left"]["shape"], [2, 3])

    def test_original_error_and_inputs_are_retained(self):
        with (
            self.assertRaisesRegex(RuntimeError, "shapes cannot be multiplied"),
            capture_router_inputs(self.output, "dsv41_capture_cpu::mm"),
        ):
            torch.ops.dsv41_capture_cpu.mm(torch.ones(2, 3), torch.ones(4, 2))
        record = json.loads((self.output / "call-001.json").read_text())
        self.assertTrue(record["capture_complete"])
        self.assertTrue(record["operator_called"])
        self.assertFalse(record["operator_returned"])
        self.assertTrue(Path(record["inputs_file"]).is_file())

    def test_nonfinite_values_are_reported_without_changing_the_call(self):
        value = torch.tensor([float("nan"), float("inf"), 1.0])
        with capture_router_inputs(self.output, "dsv41_capture_cpu::add") as capture:
            actual = torch.ops.dsv41_capture_cpu.add(value, torch.zeros_like(value))
        torch.testing.assert_close(actual, value, equal_nan=True)
        tensor = capture["calls"][0]["tensors"]["left"]
        self.assertFalse(tensor["finite"])
        self.assertEqual(tensor["nonfinite_count"], 2)
        self.assertEqual(tensor["finite_min"], 1.0)

    def test_noncontiguous_layout_and_storage_offset_are_recreated(self):
        value = torch.arange(40).reshape(5, 8)[1:4, 1:7:2].transpose(0, 1)
        with capture_router_inputs(self.output, "dsv41_capture_cpu::add") as capture:
            torch.ops.dsv41_capture_cpu.add(value, torch.zeros_like(value))
        record = capture["calls"][0]
        saved = torch.load(record["inputs_file"], weights_only=True)
        restored = restore_tensor_layout(
            saved["left"], record["tensors"]["left"], "cpu"
        )
        torch.testing.assert_close(restored, value)
        self.assertEqual(restored.stride(), value.stride())
        self.assertEqual(restored.storage_offset(), value.storage_offset())

    def test_inner_call_is_captured_through_opaque_container(self):
        left, right = torch.ones(2, 3), torch.ones(2, 3)
        seen = []

        class DispatchControl(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                seen.append(func._schema.name)
                return func(*args, **(kwargs or {}))

        with DispatchControl():
            torch.ops.dsv41_capture_cpu.outer(left, right)
        self.assertNotIn("dsv41_capture_cpu::add", seen)
        with capture_router_inputs(self.output, "dsv41_capture_cpu::add") as capture:
            actual = torch.ops.dsv41_capture_cpu.outer(left, right)
        torch.testing.assert_close(actual, left + right)
        self.assertEqual(len(capture["calls"]), 1)
        self.assertTrue(capture["calls"][0]["operator_returned"])

    def test_prior_thread_profiler_is_restored_after_failure(self):
        def prior(frame, event, result):
            return None

        previous = sys.getprofile()
        try:
            sys.setprofile(prior)
            with (
                self.assertRaisesRegex(RuntimeError, "deliberate"),
                capture_router_inputs(self.output, "dsv41_capture_cpu::add"),
            ):
                raise RuntimeError("deliberate")
            self.assertIs(sys.getprofile(), prior)
        finally:
            sys.setprofile(previous)


if __name__ == "__main__":
    manifest = json.loads(Path("source-manifest.json").read_text())
    assert all(
        hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
        for name, digest in manifest.items()
    )
    destination = Path(sys.argv[1])
    destination.mkdir()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CaptureChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    npu_initialized = (
        sys.modules["torch_npu"].npu.is_initialized()
        if "torch_npu" in sys.modules
        else False
    )
    summary = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "torch_version": torch.__version__,
        "npu_initialized": npu_initialized,
        "source_unchanged": all(
            hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
            for name, digest in manifest.items()
        ),
        "scope": "CPU capture and layout restoration only; no A5 operator execution",
        "success": result.wasSuccessful() and not npu_initialized,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    raise SystemExit(int(not summary["success"]))
