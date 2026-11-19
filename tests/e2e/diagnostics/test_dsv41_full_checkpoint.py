# SPDX-License-Identifier: Apache-2.0
"""Executable host checks for the qualification CLI, without NPU execution."""

import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def load_driver():
    with patch.object(sys, "path", [str(ROOT / "tools/diagnostics"), *sys.path]):
        return importlib.import_module("run_dsv41_full_checkpoint")


def response(prompt):
    return {
        "choices": [
            {
                "token_ids": list(range(32)),
                "prompt_token_ids": [len(prompt)],
                "finish_reason": "length",
            }
        ],
        "usage": {"completion_tokens": 32},
    }


class TestFullCheckpointPlans(unittest.TestCase):
    def test_modes_keep_native_quantized_weights_and_cpu_engram(self):
        driver = load_driver()
        for mode in driver.MODES:
            for steps in (0, 1, 2):
                with self.subTest(mode=mode, steps=steps):
                    plan = driver.make_plan(
                        Path("/model"),
                        Path("/vendor"),
                        ROOT,
                        mode=mode,
                        spec_tokens=steps,
                    )
                    for settings in plan["roles"].values():
                        argv = settings["argv"]
                        self.assertEqual(
                            argv[argv.index("--load-format") + 1], "safetensors"
                        )
                        self.assertEqual(
                            argv[argv.index("--quantization") + 1], "deepseek_v4_fp8"
                        )
                        self.assertEqual(
                            argv[argv.index("--engram-config.cpu_offload") + 1], "true"
                        )
                        self.assertEqual(
                            "--enforce-eager" in argv, mode.startswith("eager")
                        )
                        self.assertEqual("--enable-dbo" in argv, mode.endswith("u2"))
                        self.assertEqual("--speculative-config" in argv, bool(steps))
                    self.assertFalse(plan["full_model_qualification"])

    def test_native_control_has_matching_tp_and_no_afd_role(self):
        plan = load_driver().make_plan(
            Path("/model"), Path("/vendor"), ROOT, native_control=True
        )
        self.assertEqual(set(plan["roles"]), {"attention"})
        argv = plan["roles"]["attention"]["argv"]
        self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "4")
        self.assertNotIn("afd", json.loads(argv[argv.index("--additional-config") + 1]))

    def test_pd_config_keeps_equal_tp_and_attested_connector_on_declared_roles(self):
        driver = load_driver()
        for role in ("producer", "consumer"):
            plan = driver.make_plan(Path("/model"), Path("/vendor"), ROOT, pd_role=role)
            for settings in plan["roles"].values():
                argv = settings["argv"]
                transfer = json.loads(argv[argv.index("--kv-transfer-config") + 1])
                self.assertEqual(transfer["kv_role"], "kv_" + role)
                self.assertEqual(transfer["kv_connector_module_path"], driver.PD_MODULE)
                self.assertEqual(transfer["kv_buffer_device"], "npu")
                self.assertEqual(
                    transfer["kv_connector_extra_config"]["prefill"]["tp_size"], 4
                )
                self.assertEqual(
                    transfer["kv_connector_extra_config"]["decode"]["tp_size"], 4
                )

    def test_devices_addresses_ports_and_disabled_overlap_reach_launch_plan(self):
        driver = load_driver()
        plan = driver.make_plan(
            Path("/model"),
            Path("/vendor"),
            ROOT,
            mode="graph-u2",
            pd_role="consumer",
            engine_id="decode-node",
            attention_devices=(8, 9, 10, 11),
            ffn_devices=(12, 13, 14, 15),
            bind_host="0.0.0.0",
            kv_ip="10.0.0.8",
            port_offset=100,
            stream_overlap=False,
        )
        self.assertEqual(plan["physical_devices"], list(range(8, 16)))
        self.assertEqual(plan["api_url"], "http://127.0.0.1:18131")
        for settings in plan["roles"].values():
            argv = settings["argv"]
            transfer = json.loads(argv[argv.index("--kv-transfer-config") + 1])
            self.assertEqual(transfer["kv_ip"], "10.0.0.8")
            self.assertEqual(transfer["kv_port"], driver.DEFAULT_KV_PORT + 100)
            additional = json.loads(argv[argv.index("--additional-config") + 1])
            self.assertNotIn(
                "graph_exchange_order", additional["afd"]["connector_extra_config"]
            )
            self.assertEqual(
                settings["environment"]["AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"], "0"
            )

    def test_overlapping_role_devices_are_rejected_before_launch(self):
        with self.assertRaisesRegex(ValueError, "distinct devices"):
            load_driver().make_plan(
                Path("/model"),
                Path("/vendor"),
                ROOT,
                ffn_devices=(0, 5, 6, 7),
            )


class TestFullCheckpointReference(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.reference = self.directory / "native"
        self.reference.mkdir()
        self.error = "RuntimeError: selected idle devices were not confirmed"
        for name, report in (
            ("launch-plan.json", {"native_control": True}),
            (
                "summary.json",
                {"request_passed": False, "clean_shutdown": False, "error": self.error},
            ),
        ):
            (self.reference / name).write_text(json.dumps(report))

    def test_failed_native_reference_reports_original_failure_before_afd_start(self):
        driver = load_driver()
        argv = [
            "run_dsv41_full_checkpoint.py",
            "--model",
            str(self.directory / "model"),
            "--transformer-vendor",
            str(self.directory / "vendor"),
            "--output-root",
            str(self.directory / "output"),
            "--reference-run",
            str(self.reference),
            "--run",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "platform", "linux"),
            patch.object(
                driver.launch, "collect_preflight", return_value={"passed": True}
            ),
            patch.object(
                driver.launch,
                "run_candidate",
                side_effect=RuntimeError("unexpected service start"),
            ),
            self.assertRaises(ValueError) as raised,
        ):
            driver.main()

        message = str(raised.exception)
        self.assertIn(str(self.reference), message)
        self.assertIn("request_passed=False", message)
        self.assertIn(self.error, message)
        self.assertIn("AFD services were not started", message)


class TestFullCheckpointRequests(unittest.TestCase):
    def test_dspark_requires_a_later_step_after_rejection(self):
        driver = load_driver()
        for accepted in ([1, 0], [0, 1]):
            with (
                self.subTest(accepted=accepted),
                tempfile.TemporaryDirectory() as temporary,
            ):
                directory = Path(temporary)

                def reply(request, per_step_accepted=accepted, **kwargs):
                    payload = json.loads(request.data)
                    body = response(payload["prompt"])
                    body["metrics"] = {
                        "speculative_decoding": {
                            "num_spec_tokens": 1,
                            "num_spec_steps": 2,
                            "num_draft_tokens": 2,
                            "num_accepted_draft_tokens": 1,
                            "acceptance_histogram": [1, 1],
                            "per_step_accepted": per_step_accepted,
                            "per_step_drafted": [1, 1],
                            "draft_acceptance_rate": 0.5,
                            "mean_acceptance_length": 1.5,
                        }
                    }
                    raw = (
                        b'data: {"choices":[{"text":"generated"}]}\n\n'
                        if payload["stream"]
                        else json.dumps(body).encode()
                    )
                    stream = io.BytesIO(raw)
                    stream.status = 200
                    return stream

                with patch("urllib.request.urlopen", side_effect=reply):
                    if accepted == [1, 0]:
                        with self.assertRaisesRegex(RuntimeError, "continuation"):
                            driver.run_requests(
                                directory, None, spec_tokens=1, prefill_url=None
                            )
                    else:
                        report = driver.run_requests(
                            directory, None, spec_tokens=1, prefill_url=None
                        )
                        self.assertTrue(report["rejection_continuation_observed"])
                saved = json.loads((directory / "request-matrix.json").read_bytes())
                self.assertEqual(saved["passed"], accepted == [0, 1])

    def test_parallel_waves_and_post_cancel_request_compare_to_native(self):
        driver = load_driver()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "afd"
            baseline = Path(temporary) / "native"
            directory.mkdir()
            for index, prompt in enumerate(
                (*driver.PROMPTS[:2], *driver.PROMPTS[:2], driver.PROMPTS[2])
            ):
                target = baseline / f"request-{index:02d}"
                target.mkdir(parents=True)
                (target / "response.json").write_text(json.dumps(response(prompt)))

            def reply(request, **kwargs):
                payload = json.loads(request.data)
                raw = (
                    b'data: {"choices":[{"text":"generated"}]}\n\n'
                    if payload["stream"]
                    else json.dumps(response(payload["prompt"])).encode()
                )
                stream = io.BytesIO(raw)
                stream.status = 200
                return stream

            with patch("urllib.request.urlopen", side_effect=reply):
                report = driver.run_requests(
                    directory, baseline, spec_tokens=0, prefill_url=None
                )

            self.assertTrue(report["passed"])
            self.assertEqual(len(report["requests"]), 5)
            self.assertTrue((directory / "cancel-first-chunk.json").is_file())
            for index in range(5):
                comparison = json.loads(
                    (
                        directory / f"request-{index:02d}" / "token-comparison.json"
                    ).read_text()
                )
                self.assertEqual(comparison["status"], "tokens_match")
            self.assertFalse(report["hardware_acceptance"])


if __name__ == "__main__":
    unittest.main()
