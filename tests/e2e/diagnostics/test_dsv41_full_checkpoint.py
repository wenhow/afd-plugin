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


class RepeatabilityReplies:
    """HTTP protocol fixture; never represents model or device computations."""

    def __init__(self, difference=None):
        self.difference = difference
        self.completed = 0
        self.cancelled = False

    def open(self, request, **kwargs):
        payload = json.loads(request.data)
        if payload["stream"]:
            self.cancelled = True
            raw = b'data: {"choices":[{"text":"generated"}]}\n\n'
        else:
            body = response(payload["prompt"])
            tokens = body["choices"][0]["token_ids"]
            if (
                self.difference == "all"
                or (self.difference == "warmup" and self.completed < 2)
                or (self.difference == "before" and self.completed == 4)
                or (self.difference == "after" and self.cancelled)
            ):
                tokens[0] += 100
            body["choices"][0]["logprobs"] = (
                {
                    "tokens": [f"token_id:{token}" for token in tokens],
                    "token_logprobs": [-0.1] * len(tokens),
                    "top_logprobs": [
                        {
                            f"token_id:{token + offset}": -0.1 - offset
                            for offset in range(5)
                        }
                        for token in tokens
                    ],
                }
                if payload.get("logprobs")
                else None
            )
            if self.difference == "missing_logprobs":
                body["choices"][0]["logprobs"] = None
            self.completed += 1
            raw = json.dumps(body).encode()
        stream = io.BytesIO(raw)
        stream.status = 200
        return stream


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
        self.assertEqual(
            argv[argv.index("--worker-cls") + 1],
            "afd_plugin.v1.worker.npu.dsv41_native_worker.DSV41NativeTP4ControlWorker",
        )
        self.assertEqual(plan["attention_head_adapter"], "tp4-query-16-to-64-v1")

    def test_repeatability_limits_active_requests_and_records_protocol(self):
        driver = load_driver()
        native_protocol = None
        for native, mode in (
            (True, "eager-u1"),
            (False, "eager-u1"),
            (False, "graph-u1"),
        ):
            with self.subTest(native=native, mode=mode):
                plan = driver.make_plan(
                    Path("/model"),
                    Path("/vendor"),
                    ROOT,
                    native_control=native,
                    mode=mode,
                    repeatability=True,
                )
                for settings in plan["roles"].values():
                    argv = settings["argv"]
                    self.assertEqual(argv[argv.index("--max-num-seqs") + 1], "1")
                    self.assertEqual(
                        argv[argv.index("--logprobs-mode") + 1], "raw_logprobs"
                    )
                self.assertEqual(plan["request_suite"], "repeatability")
                self.assertEqual(
                    plan["repeatability_protocol"]["client_concurrency"], 1
                )
                if native:
                    native_protocol = plan["repeatability_protocol"]
                if mode == "graph-u1":
                    self.assertEqual(plan["repeatability_protocol"], native_protocol)
                    compilation = json.loads(
                        argv[argv.index("--compilation-config") + 1]
                    )
                    self.assertEqual(compilation["cudagraph_mode"], "FULL_DECODE_ONLY")
                    self.assertNotIn("--enforce-eager", argv)

    def test_repeatability_rejects_combinations_and_insufficient_rounds(self):
        driver = load_driver()
        for options in (
            {"mode": "eager-u2"},
            {"mode": "graph-u2"},
            {"spec_tokens": 1},
            {"pd_role": "consumer"},
            {"warmup_rounds": 0},
            {"repeat_rounds": 1},
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "repeatability requires"),
            ):
                driver.make_plan(
                    Path("/model"),
                    Path("/vendor"),
                    ROOT,
                    repeatability=True,
                    **options,
                )

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

    def invoke_reference(self, *options):
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
            *options,
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

        return str(raised.exception)

    def test_failed_native_reference_reports_original_failure_before_afd_start(self):
        message = self.invoke_reference()
        self.assertIn(str(self.reference), message)
        self.assertIn("request_passed=False", message)
        self.assertIn(self.error, message)
        self.assertIn("AFD services were not started", message)

    def test_reference_without_matching_head_adapter_is_rejected(self):
        (self.reference / "summary.json").write_text(
            json.dumps({"request_passed": True, "clean_shutdown": True})
        )
        message = self.invoke_reference()
        self.assertIn("attention_head_adapter=None", message)
        self.assertIn("tp4-query-16-to-64-v1", message)
        self.assertIn("AFD services were not started", message)

    def test_matrix_reference_cannot_start_repeatability_services(self):
        driver = load_driver()
        driver.launch.save_json(
            self.reference / "summary.json",
            {
                "request_passed": True,
                "clean_shutdown": True,
            },
        )
        driver.launch.save_json(
            self.reference / "launch-plan.json",
            {
                "native_control": True,
                "attention_head_adapter": driver.TP4_HEAD_ADAPTER,
            },
        )
        message = self.invoke_reference("--repeatability")
        self.assertIn("request suite differs", message)
        self.assertIn("AFD services were not started", message)

    def test_repeatability_reference_rejects_incomplete_or_different_protocol(self):
        driver = load_driver()
        plan = driver.make_plan(
            Path("/model"),
            Path("/vendor"),
            ROOT,
            native_control=True,
            repeatability=True,
        )
        driver.launch.save_json(self.reference / "launch-plan.json", plan)
        driver.launch.save_json(
            self.reference / "summary.json",
            {
                "request_passed": True,
                "clean_shutdown": True,
            },
        )
        for completed, passed, protocol in (
            (False, True, plan["repeatability_protocol"]),
            (True, False, plan["repeatability_protocol"]),
            (True, True, {**plan["repeatability_protocol"], "repeat_rounds": 9}),
        ):
            with self.subTest(completed=completed, passed=passed, protocol=protocol):
                driver.launch.save_json(
                    self.reference / "repeatability.json",
                    {
                        "completed": completed,
                        "passed": passed,
                        "protocol": protocol,
                    },
                )
                message = self.invoke_reference("--repeatability")
                self.assertIn(
                    "incomplete, unstable or has different parameters", message
                )
                self.assertIn("AFD services were not started", message)


class TestRepeatabilityRequests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.driver = load_driver()
        self.protocol = self.driver.make_plan(
            Path("/model"),
            Path("/vendor"),
            ROOT,
            repeatability=True,
            repeat_rounds=2,
            repeat_logprobs=5,
        )["repeatability_protocol"]

    def run_suite(self, name, *, difference=None, reference=None):
        directory = self.directory / name
        directory.mkdir()
        replies = RepeatabilityReplies(difference)
        with patch("urllib.request.OpenerDirector.open", side_effect=replies.open):
            report = self.driver.repeatability_check.run_repeatability(
                directory,
                reference,
                api_url="http://127.0.0.1:18031",
                protocol=self.protocol,
                cancel=lambda target: self.driver.cancel_stream(
                    target, prefill_url=None
                ),
            )
        return report

    def test_stable_measured_rounds_pass_with_warmup_difference_recorded(self):
        report = self.run_suite("native", difference="warmup")
        self.assertTrue(report["passed"])
        self.assertFalse(report["warmup_matches_measured_baseline"])
        self.assertTrue(report["before_cancel_stable"])
        self.assertTrue(report["after_cancel_matches_before"])
        self.assertFalse(report["model_equivalence_accepted"])
        self.assertFalse(report["hardware_acceptance"])

    def test_identical_native_reference_passes(self):
        self.run_suite("native")
        report = self.run_suite("afd", reference=self.directory / "native")
        self.assertTrue(report["native_tokens_match"])
        self.assertTrue(report["passed"])

    def test_default_token_only_protocol_completes_without_logprobs(self):
        self.protocol.update(logprobs=0, return_tokens_as_token_ids=False)
        report = self.run_suite("native")
        self.assertTrue(report["passed"])
        response_path = (
            self.directory / "native/before-cancel-00/request-00/response.json"
        )
        body = json.loads(response_path.read_bytes())
        self.assertIsNone(body["choices"][0]["logprobs"])

    def test_token_only_difference_fails_without_requiring_scores(self):
        self.protocol.update(logprobs=0, return_tokens_as_token_ids=False)
        with self.assertRaisesRegex(RuntimeError, "token comparison failed"):
            self.run_suite("native", difference="after")
        report = json.loads((self.directory / "native/repeatability.json").read_bytes())
        self.assertTrue(report["completed"])
        self.assertFalse(report["passed"])
        divergent = next(
            item for item in report["comparisons"] if item["status"] == "tokens_differ"
        )
        self.assertEqual(divergent["first_output_mismatch"]["index"], 0)
        self.assertIsNone(divergent["first_divergence_logprobs"])

    def test_differences_fail_after_saving_all_measured_rounds(self):
        self.run_suite("native")
        for difference, before, after, native in (
            ("before", False, True, False),
            ("after", True, False, False),
            ("all", True, True, False),
        ):
            with self.subTest(difference=difference):
                with self.assertRaisesRegex(RuntimeError, "token comparison failed"):
                    self.run_suite(
                        difference,
                        difference=difference,
                        reference=self.directory / "native",
                    )
                report = json.loads(
                    (self.directory / difference / "repeatability.json").read_bytes()
                )
                self.assertTrue(report["completed"])
                self.assertFalse(report["passed"])
                self.assertEqual(len(report["requests"]), 10)
                self.assertEqual(report["before_cancel_stable"], before)
                self.assertEqual(report["after_cancel_matches_before"], after)
                self.assertEqual(report["native_tokens_match"], native)
                divergent = next(
                    item
                    for item in report["comparisons"]
                    if item["status"] == "tokens_differ"
                )
                self.assertEqual(divergent["first_output_mismatch"]["index"], 0)
                self.assertAlmostEqual(
                    divergent["first_divergence_logprobs"]["actual"]["top1_minus_top2"],
                    1.0,
                )
                self.assertTrue(
                    (self.directory / difference / "cancel-first-chunk.json").is_file()
                )

    def test_missing_logprobs_leave_incomplete_report_and_response(self):
        with self.assertRaisesRegex(ValueError, "token-ID logprobs"):
            self.run_suite("bad", difference="missing_logprobs")
        target = self.directory / "bad"
        report = json.loads((target / "repeatability.json").read_bytes())
        self.assertFalse(report["completed"])
        self.assertFalse(report["passed"])
        self.assertTrue((target / "warmup-00/request-00/response.json").is_file())
        self.assertEqual(report["cancellation"], "not_started")


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

                with patch("urllib.request.OpenerDirector.open", side_effect=reply):
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

            with patch("urllib.request.OpenerDirector.open", side_effect=reply):
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
