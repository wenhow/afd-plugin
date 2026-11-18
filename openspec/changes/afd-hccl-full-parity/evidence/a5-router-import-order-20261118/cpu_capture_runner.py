"""Exercise diagnostic capture with real CPU PyTorch operators, without NPU work."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

import torch
from dsv41_a5_router_probe import (
    capture_router_inputs,
    operator_symbol_providers,
    restore_tensor_layout,
)
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

    def test_loaded_symbol_provider_is_inspected_without_loading_missing_library(self):
        paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libc.so" in line
        }
        self.assertEqual(len(paths), 1)
        path = paths.pop()
        missing = str(self.output / "not-loaded.so")
        providers = operator_symbol_providers([path, missing], ("malloc",))
        self.assertEqual(len(providers), 1)
        self.assertEqual(Path(providers[0]["provider"]).resolve(), Path(path).resolve())
        self.assertEqual(providers[0]["symbol"], "malloc")


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
    values = torch.load(
        "native-inputs/call-001.pt", map_location="cpu", weights_only=True
    )
    actual = json.loads(Path("native-inputs/replay-result.json").read_text())
    assert values["tid2eid"] is None and values["bias_vl"] is None
    assert (values["norm_type"], values["renorm"], values["group_count"]) == (2, 0, 1)
    scores = torch.nn.functional.softplus(values["x"].float()).sqrt()
    ids = torch.topk(scores + values["bias"], values["k"], dim=-1, sorted=True).indices
    weights = scores.gather(1, ids)
    weights = (
        weights / weights.sum(dim=-1, keepdim=True) * values["routed_scaling_factor"]
    )
    actual_weights = torch.tensor(actual["output_weights"])
    reference = {
        "ids_match": ids.tolist() == actual["output_ids"],
        "max_weight_absolute_error": float((weights - actual_weights).abs().max()),
        "weights_match": bool(
            torch.allclose(weights, actual_weights, atol=1e-5, rtol=0)
        ),
        "inputs_sha256": hashlib.sha256(
            Path("native-inputs/call-001.pt").read_bytes()
        ).hexdigest(),
        "scope": "CPU reference comparison with the uploaded actual A5 replay result",
    }
    (destination / "actual-input-reference.json").write_text(
        json.dumps(reference, indent=2) + "\n"
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
        "actual_input_reference": reference,
        "success": result.wasSuccessful()
        and not npu_initialized
        and reference["ids_match"]
        and reference["weights_match"],
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    raise SystemExit(int(not summary["success"]))
