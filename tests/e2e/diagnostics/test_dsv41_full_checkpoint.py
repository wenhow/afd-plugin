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
    def test_ep_diagnostics_use_native_scheduler_with_admission_logging_on_a(self):
        driver = load_driver()
        for native in (False, True):
            with self.subTest(native=native):
                plan = driver.make_plan(
                    Path("/model"),
                    Path("/vendor"),
                    ROOT,
                    expert_parallel=True,
                    native_control=native,
                    ep_diagnostics=True,
                    attention_stop_signal="SIGTERM",
                )
                argv = plan["roles"]["attention"]["argv"]
                self.assertTrue(plan["ep_diagnostics"])
                self.assertEqual(plan["attention_stop_signal"], "SIGTERM")
                self.assertEqual(
                    argv[argv.index("--scheduler-cls") + 1],
                    "afd_plugin.compat.npu.dsv41_ep_diagnostics."
                    "DSV41EPDiagnosticScheduler",
                )
                self.assertEqual(
                    argv[argv.index("--logprobs-mode") + 1], "raw_logprobs"
                )
                if not native:
                    self.assertNotIn("--scheduler-cls", plan["roles"]["ffn"]["argv"])

    def test_attention_stop_signal_requires_ep_diagnostics(self):
        driver = load_driver()
        with self.assertRaisesRegex(ValueError, "attention stop signal requires"):
            driver.make_plan(
                Path("/model"), Path("/vendor"), ROOT, attention_stop_signal="SIGTERM"
            )

    def test_ep_plans_select_tp1_dp4_native_and_af_roles(self):
        driver = load_driver()
        for native in (False, True):
            with self.subTest(native=native):
                plan = driver.make_plan(
                    Path("/model"),
                    Path("/vendor"),
                    ROOT,
                    expert_parallel=True,
                    native_control=native,
                )
                self.assertEqual(plan["native_tp_size"], 1)
                self.assertEqual(plan["native_dp_size"], 4)
                self.assertIsNone(plan["attention_head_adapter"])
                for settings in plan["roles"].values():
                    argv = settings["argv"]
                    self.assertIn("--enable-expert-parallel", argv)
                    self.assertEqual(
                        argv[argv.index("--tensor-parallel-size") + 1], "1"
                    )
                    self.assertEqual(argv[argv.index("--data-parallel-size") + 1], "4")
                    self.assertEqual(
                        argv[argv.index("--data-parallel-size-local") + 1], "4"
                    )
                    self.assertEqual(
                        argv[argv.index("--engram-config.dp_shared_memory") + 1], "true"
                    )
                if native:
                    argv = plan["roles"]["attention"]["argv"]
                    self.assertEqual(
                        argv[argv.index("--worker-cls") + 1],
                        "vllm_ascend.worker.worker.NPUWorker",
                    )

    def test_ep_plan_rejects_unintegrated_combinations(self):
        driver = load_driver()
        for options, message in (
            ({"trace_native_prefill": True}, "requires native prefill trace off"),
            (
                {"mode": "eager-u2", "pd_role": "producer"},
                "producer qualification uses eager U1",
            ),
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, message),
            ):
                driver.make_plan(
                    Path("/model"),
                    Path("/vendor"),
                    ROOT,
                    expert_parallel=True,
                    **options,
                )

    def test_ep_graph_u2_keeps_consumer_draft_and_serial_exchange(self):
        plan = load_driver().make_plan(
            Path("/model"),
            Path("/vendor"),
            ROOT,
            mode="graph-u2",
            spec_tokens=2,
            expert_parallel=True,
            pd_role="consumer",
        )
        for role in plan["roles"].values():
            self.assertIn("--enable-dbo", role["argv"])
            self.assertEqual(
                role["environment"]["AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"], "0"
            )
            extra = json.loads(
                role["argv"][role["argv"].index("--additional-config") + 1]
            )
            self.assertEqual(
                extra["afd"]["connector_extra_config"].get(
                    "graph_exchange_order", "stage_serial"
                ),
                "stage_serial",
            )

    def test_ep_graph_u1_plan_keeps_tp1_dp4_and_fixed_decode_capture(self):
        plan = load_driver().make_plan(
            Path("/model"), Path("/vendor"), ROOT, mode="graph-u1", expert_parallel=True
        )
        self.assertTrue(plan["expert_parallel"])
        self.assertEqual((plan["native_tp_size"], plan["native_dp_size"]), (1, 4))
        for settings in plan["roles"].values():
            argv = settings["argv"]
            self.assertNotIn("--enforce-eager", argv)
            compilation = json.loads(argv[argv.index("--compilation-config") + 1])
            self.assertEqual(compilation["cudagraph_mode"], "FULL_DECODE_ONLY")
            self.assertEqual(compilation["cudagraph_capture_sizes"], [1, 2])

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

    def test_native_prefill_trace_keeps_repeatability_protocol(self):
        driver = load_driver()
        options = dict(native_control=True, repeatability=True, repeat_logprobs=5)
        paths = (Path("/model"), Path("/vendor"), ROOT)
        baseline = driver.make_plan(*paths, **options)
        trace = driver.make_plan(*paths, **options, trace_native_prefill=True)
        self.assertEqual(
            trace["repeatability_protocol"], baseline["repeatability_protocol"]
        )
        self.assertTrue(trace["trace_native_prefill"])
        argv = trace["roles"]["attention"]["argv"]
        self.assertTrue(
            argv[argv.index("--worker-cls") + 1].endswith(
                ".DSV41NativeTP4PrefillTraceWorker"
            )
        )

    def test_native_prefill_trace_requires_native_repeatability(self):
        driver = load_driver()
        for options in ({"native_control": True}, {"repeatability": True}):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "requires native repeatability"),
            ):
                driver.make_plan(
                    Path("/model"),
                    Path("/vendor"),
                    ROOT,
                    trace_native_prefill=True,
                    **options,
                )

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

    def test_ep_pd_declares_four_native_replicas_on_both_sides(self):
        driver = load_driver()
        kv_ip = "192.0.2.8"
        port_offset = 100
        for role, mode in (("producer", "eager-u1"), ("consumer", "graph-u1")):
            with self.subTest(role=role):
                plan = driver.make_plan(
                    Path("/model"),
                    Path("/vendor"),
                    ROOT,
                    expert_parallel=True,
                    mode=mode,
                    pd_role=role,
                    spec_tokens=1,
                    kv_ip=kv_ip,
                    port_offset=port_offset,
                )
                kv_base = driver.DEFAULT_KV_PORT + port_offset
                self.assertTrue(set(range(kv_base, kv_base + 4)) <= set(plan["ports"]))
                attention = plan["roles"]["attention"]
                self.assertEqual(attention["environment"]["VLLM_HOST_IP"], kv_ip)
                for settings in plan["roles"].values():
                    argv = settings["argv"]
                    transfer = json.loads(argv[argv.index("--kv-transfer-config") + 1])
                    self.assertEqual(transfer["kv_role"], "kv_" + role)
                    self.assertEqual(transfer["kv_port"], kv_base)
                    for side in ("prefill", "decode"):
                        self.assertEqual(
                            transfer["kv_connector_extra_config"][side],
                            {"tp_size": 1, "dp_size": 4, "pp_size": 1},
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
        self.candidate = None
        self.preflight = {
            "passed": True,
            "checkpoint": {"sha256": {"config.json": "a", "index": "b"}},
        }
        for name, report in (
            ("launch-plan.json", {"native_control": True}),
            (
                "summary.json",
                {"request_passed": False, "clean_shutdown": False, "error": self.error},
            ),
        ):
            (self.reference / name).write_text(json.dumps(report))

    def run_reference(self, *options):
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
                driver.launch, "collect_preflight", return_value=self.preflight
            ),
            patch.object(
                driver.launch,
                "run_candidate",
                side_effect=self.record_candidate,
            ),
        ):
            return driver.main()

    def record_candidate(self, plan, _output, *_args, **_kwargs):
        self.candidate = plan["native_reference"]
        return 0

    def invoke_reference(self, *options):
        with self.assertRaises(ValueError) as raised:
            self.run_reference(*options)
        self.assertIsNone(self.candidate)
        return str(raised.exception)

    def write_timed_out_ep_reference(self):
        driver = load_driver()
        plan = driver.make_plan(
            Path("/model"),
            Path("/vendor"),
            ROOT,
            native_control=True,
            expert_parallel=True,
        )
        records = {
            "launch-plan.json": plan,
            "summary.json": {
                "request_passed": True,
                "clean_shutdown": False,
                "forced_cleanup": True,
                "survivors": [],
                "exit_codes": {"attention": -9},
                "npu_after": {"exit_code": 0, "idle_devices": list(range(4))},
            },
            "lifecycle.json": [
                {"event": "request_completed"},
                {"event": "exit_timeout", "role": "attention"},
            ],
            "preflight.json": self.preflight,
            "request-matrix.json": {
                "passed": True,
                "requests": [{"passed": True, "http_status": 200}] * 13,
            },
        }
        for name, data in records.items():
            driver.launch.save_json(self.reference / name, data)
        for index in range(13):
            request = self.reference / f"request-{index:02d}"
            request.mkdir()
            driver.launch.save_json(request / "response.json", response("prompt"))
            driver.launch.save_json(
                request / "request-routing.json",
                {"dp_rank": 1 if index == 12 else index % 4},
            )

    def test_ep_timeout_reference_retains_failed_shutdown(self):
        self.write_timed_out_ep_reference()
        self.assertEqual(self.run_reference("--expert-parallel"), 0)
        self.assertTrue(self.candidate["diagnostic_only"])
        self.assertIs(self.candidate["clean_shutdown"], False)
        self.assertEqual(self.candidate["exit_codes"], {"attention": -9})

    def test_ep_diagnostics_recheck_archived_survivors_and_all_candidate_devices(self):
        self.write_timed_out_ep_reference()
        driver = load_driver()
        summary_path = self.reference / "summary.json"
        summary = json.loads(summary_path.read_bytes())
        summary["survivors"] = [{"pid": 42, "start_ticks": 100}]
        summary["npu_after"]["idle_devices"] = [0, 3]
        driver.launch.save_json(summary_path, summary)
        previous_worker = driver.launch.ProcessIdentity(42, 1, 10, 10, 100, "D")
        reused_pid = driver.launch.ProcessIdentity(42, 1, 20, 20, 101, "S")
        options = ("--expert-parallel", "--ep-diagnostics")

        for worker, devices, error in (
            (previous_worker, list(range(8)), "live native PIDs=[42]"),
            (reused_pid, list(range(7)), "devices not confirmed idle=[7]"),
            (reused_pid, list(range(8)), None),
        ):
            with (
                self.subTest(error=error),
                patch.object(driver.launch, "process_table", return_value={42: worker}),
                patch.object(
                    driver.launch,
                    "npu_snapshot",
                    return_value={"exit_code": 0, "idle_devices": devices},
                ),
            ):
                if error is not None:
                    self.assertIn(error, self.invoke_reference(*options))
                else:
                    self.assertEqual(self.run_reference(*options), 0)
                    self.assertTrue(self.candidate["diagnostic_only"])
                    self.assertFalse(self.candidate["clean_shutdown"])
                    self.assertTrue(self.candidate["cleanup_recheck"]["passed"])
                    self.assertEqual(
                        self.candidate["cleanup_recheck"]["archived_survivors"],
                        summary["survivors"],
                    )
        self.assertEqual(json.loads(summary_path.read_bytes()), summary)

    def test_diagnostic_reference_rejects_missing_response_or_failed_cleanup(self):
        self.write_timed_out_ep_reference()
        options = ("--expert-parallel",)
        raw = self.reference / "request-12/response.json"
        raw.unlink()
        self.assertIn("request-12", self.invoke_reference(*options))
        load_driver().launch.save_json(raw, response("prompt"))
        summary_path = self.reference / "summary.json"
        summary = json.loads(summary_path.read_bytes())
        summary["survivors"] = [{"pid": 42}]
        load_driver().launch.save_json(summary_path, summary)
        self.assertIn("no survivors", self.invoke_reference(*options))

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

    def test_native_prefill_trace_is_ineligible_even_if_requests_pass(self):
        (self.reference / "summary.json").write_text(
            json.dumps({"request_passed": True, "clean_shutdown": True})
        )
        plan = load_driver().make_plan(
            Path("/model"),
            Path("/vendor"),
            ROOT,
            native_control=True,
            repeatability=True,
            trace_native_prefill=True,
        )
        (self.reference / "launch-plan.json").write_text(json.dumps(plan))
        self.assertIn(
            "native prefill trace is diagnostic only", self.invoke_reference()
        )

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
    def test_ep_diagnostics_finish_the_matrix_despite_token_mismatches(self):
        driver = load_driver()
        with tempfile.TemporaryDirectory() as temporary:
            native = Path(temporary) / "native"
            afd = Path(temporary) / "afd"
            native.mkdir()
            afd.mkdir()
            options = dict(
                spec_tokens=0,
                prefill_url=None,
                data_parallel_size=4,
                ep_diagnostics=True,
            )
            replies = RepeatabilityReplies()
            with patch("urllib.request.OpenerDirector.open", side_effect=replies.open):
                self.assertTrue(driver.run_requests(native, None, **options)["passed"])

            replies = RepeatabilityReplies(difference="all")
            with (
                patch("urllib.request.OpenerDirector.open", side_effect=replies.open),
                self.assertRaisesRegex(RuntimeError, "completed with token mismatches"),
            ):
                driver.run_requests(afd, native, **options)
            report = json.loads((afd / "request-matrix.json").read_bytes())
            self.assertFalse(report["passed"])
            self.assertEqual(len(report["requests"]), 13)
            self.assertTrue((afd / "cancel-first-chunk.json").is_file())
            response = json.loads((afd / "request-12/response.json").read_bytes())
            scores = response["choices"][0]["logprobs"]["top_logprobs"]
            self.assertEqual(len(scores), 32)

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

    def test_ep_matrix_sends_one_request_to_each_dp_rank_per_wave(self):
        driver = load_driver()
        observed_ranks = []

        def reply(request, **kwargs):
            payload = json.loads(request.data)
            if payload["stream"]:
                raw = b'data: {"choices":[{"text":"generated"}]}\n\n'
            else:
                observed_ranks.append(request.get_header("X-data-parallel-rank"))
                raw = json.dumps(response(payload["prompt"])).encode()
            stream = io.BytesIO(raw)
            stream.status = 200
            return stream

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("urllib.request.OpenerDirector.open", side_effect=reply),
        ):
            report = driver.run_requests(
                Path(temporary),
                None,
                spec_tokens=0,
                prefill_url=None,
                data_parallel_size=4,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(sorted(observed_ranks[:4]), ["0", "1", "2", "3"])
        self.assertEqual(
            sorted(observed_ranks[4:12]), ["0", "0", "1", "1", "2", "2", "3", "3"]
        )
        self.assertEqual(observed_ranks[12:], ["1"])

    def test_ep_pd_matrix_pulls_from_the_rotated_prefill_dp_rank(self):
        driver = load_driver()
        prefill_url = "http://127.0.0.1:18131"
        dp_size = 4

        def reply(request, **kwargs):
            payload = json.loads(request.data)
            rank = request.get_header("X-data-parallel-rank")
            if request.full_url.startswith(prefill_url):
                self.assertTrue(payload["kv_transfer_params"]["do_remote_decode"])
                data = {
                    "kv_transfer_params": {
                        "afd_engram_history_v1": {},
                        "remote_port": driver.DEFAULT_KV_PORT + int(rank or 0),
                    }
                }
                raw = json.dumps(data).encode()
            elif payload["stream"]:
                raw = b'data: {"choices":[{"text":"generated"}]}\n\n'
            else:
                self.assertEqual(
                    payload["kv_transfer_params"]["remote_port"],
                    driver.DEFAULT_KV_PORT + (int(rank) + 1) % dp_size,
                )
                self.assertTrue(payload["kv_transfer_params"]["do_remote_prefill"])
                raw = json.dumps(response(payload["prompt"])).encode()
            stream = io.BytesIO(raw)
            stream.status = 200
            return stream

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("urllib.request.OpenerDirector.open", side_effect=reply),
        ):
            directory = Path(temporary)
            report = driver.run_requests(
                directory,
                None,
                spec_tokens=0,
                prefill_url=prefill_url,
                data_parallel_size=dp_size,
            )
            routing = directory / "request-12" / "prefill-routing.json"
            self.assertEqual(json.loads(routing.read_bytes()), {"dp_rank": 2})
        self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main()
