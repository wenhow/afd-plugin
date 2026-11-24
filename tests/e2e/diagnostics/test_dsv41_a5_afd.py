# SPDX-License-Identifier: Apache-2.0
"""Host contracts for full-checkpoint diagnostics; no model or NPU execution."""

from __future__ import annotations

import importlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import ExitStack, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from afd_plugin.entrypoints.dsv41_ep_stack import relay_shutdown

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / (
    "openspec/changes/afd-hccl-full-parity/evidence/a5-full-checkpoint-guide-20261119"
)


def load_launcher():
    with patch.object(sys, "path", [str(ROOT / "tools/diagnostics"), *sys.path]):
        return importlib.import_module("run_dsv41_a5_afd")


class TestAttentionStop(unittest.TestCase):
    def test_relay_uses_native_serve_callback_after_signal_registration_changes(self):
        namespace = {
            "__name__": "vllm.entrypoints.cli.serve",
            "relay_shutdown": relay_shutdown,
            "signal": signal,
            "sys": sys,
        }
        exec(
            """
def run_multi_api_server():
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = signum == signal.SIGTERM
        raise SystemExit

    def wait_for_child():
        relay_shutdown(0, sys._getframe())

    try:
        wait_for_child()
    except SystemExit:
        return shutdown_requested
    return False
""",
            namespace,
        )
        with patch.object(signal, "getsignal", return_value=signal.SIG_IGN):
            self.assertTrue(namespace["run_multi_api_server"]())

    def test_root_signal_is_recorded_with_the_matching_timeout_snapshot(self):
        launcher = load_launcher()
        for stop_signal in (signal.SIGINT, signal.SIGTERM):
            with (
                self.subTest(stop_signal=stop_signal),
                tempfile.TemporaryDirectory() as root,
            ):
                directory = Path(root)
                supervisor = launcher.Supervision(directory)
                process = Mock(pid=42)
                process.poll.return_value = None
                attention = SimpleNamespace(process=process)
                with (
                    patch.object(Path, "read_text", return_value="SigIgn:\t00000002\n"),
                    patch.object(supervisor, "wait_exit", return_value=False),
                ):
                    supervisor.stop_attention(
                        attention, stop_signal=stop_signal, trace_signal=True
                    )
                process.send_signal.assert_called_once_with(stop_signal)
                events = json.loads((directory / "lifecycle.json").read_bytes())
                self.assertEqual(events[1]["signal"], stop_signal.name)
                self.assertEqual(events[1]["target"], "root")
                self.assertEqual(
                    events[3]["phase"], f"{stop_signal.name.lower()}_timeout"
                )
                self.assertEqual(events[3]["status"], {"SigIgn": "00000002"})


def write_ep_graph_logs(directory, max_stages, *, missing_replay_rank=False):
    startup_prefix = "AFD HCCL session configuration: "
    stage_prefix = "AFD HCCL stage diagnostic: "
    for role in ("attention", "ffn"):
        lines = []
        for rank in range(2):
            peer = {
                "role": role,
                "role_rank": rank,
                "world_rank": rank + (2 if role == "attention" else 0),
                "local_device": rank,
                "configured.AFD_HCCL_EAGER_U2_STREAM_OVERLAP": False,
                "configured.AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP": False,
                "configured.AFD_HCCL_GRAPH_U2_HYBRID_DAG": False,
                "configured.AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM": False,
                "configured.AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM": False,
                "configured.AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER": False,
                "configured.AFD_HCCL_STAGE_DIAGNOSTICS": True,
                "configured.AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS": False,
            }
            startup = {
                "event": "session_configuration",
                **peer,
                "negotiated.expert_parallel": True,
                "negotiated.native_tp_size": 1,
                "negotiated.attention_size": 2,
                "negotiated.ffn_size": 2,
                "negotiated.graph_execution_version": 1,
                "negotiated.max_stages": max_stages,
            }
            lines.append(f"{startup_prefix}{startup!r}")
            if role == "ffn":
                profile = {
                    **peer,
                    "event": "graph_parent_reserved",
                    "phase": "eager",
                    "mode": "profile",
                }
                lines.append(f"{stage_prefix}{profile!r}")
            lines.append(f"{stage_prefix}{{'event': 'configuration'}}")
            for phase in ("capture", "replay", "eager"):
                if missing_replay_rank and (phase, role, rank) == ("replay", "ffn", 1):
                    continue
                for record in ep_graph_parent_records(peer, phase, max_stages):
                    lines.append(f"{stage_prefix}{record!r}")
        (directory / f"{role}-server.log").write_text(
            "\n".join(lines), encoding="utf-8"
        )


def ep_graph_parent_records(peer, phase, max_stages):
    eager = phase == "eager"
    parent = {
        **peer,
        "transaction_id": phase,
        "phase": phase,
        "mode": {"capture": "warmup", "replay": "target", "eager": "dummy"}[phase],
        "reason": "requested",
        "key_present": not eager,
        "key_digest": "" if eager else "a" * 64,
        "decision_digest": "b" * 64,
        "num_stages": 1 if eager else max_stages,
        "boundary": "host_control",
    }
    attention = peer["role"] == "attention"
    boundaries = [("reserved", "graph_reservation")]
    if phase == "replay":
        api = "run_fullgraph" if attention else "graph.replay"
        boundaries.extend((("replay_begin", api), ("replay_returned", api)))
    if eager:
        api = "native_eager_scope" if attention else "execute_hccl_v41_parent"
        boundaries.append(("scope_returned", api))
    else:
        boundaries.extend(
            (
                ("finish_begin", "complete_graph_forward"),
                ("finish_returned", "complete_graph_forward"),
            )
        )
    for event, api in boundaries:
        yield {
            **parent,
            "event": f"graph_parent_{event}",
            "boundary_api": api,
            "outcome": "begin" if event.endswith("_begin") else "returned",
        }


class TestEpGraphLogVerification(unittest.TestCase):
    @contextmanager
    def graph_u1_services(self, directory):
        launcher = load_launcher()
        roles = ("ffn", "attention")
        children = [
            SimpleNamespace(
                role=role,
                process=SimpleNamespace(returncode=-9, poll=lambda: -9),
                stream=io.BytesIO(),
                text=(directory / f"{role}-server.log").read_text,
            )
            for role in roles
        ]
        owner = Mock(processes=children, forced_cleanup=True)
        owner.start.side_effect = children
        owner.survivors.return_value = []
        plan = {
            "mode": "graph-u1",
            "request_suite": "ep-rank-matrix-v2",
            "expert_parallel": True,
            "native_dp_size": 2,
            "physical_devices": list(range(4)),
            "repo_root": str(ROOT),
            "roles": {
                role: {
                    "argv": [],
                    "environment": {launcher.EP_GRAPH_TRACE_ENV: "1"},
                }
                for role in roles
            },
        }
        with patch.multiple(
            launcher,
            Supervision=Mock(return_value=owner),
            check_resources=Mock(),
            probe_library=Mock(),
            wait_ready=Mock(),
            npu_snapshot=Mock(
                return_value={"exit_code": 0, "idle_devices": list(range(4))}
            ),
        ):
            yield launcher, owner, plan

    def test_graph_u1_pass_precedes_shutdown_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_ep_graph_logs(directory, 1)

            def stop_attention(*_args, **_kwargs):
                report = json.loads(
                    (directory / "graph-verification.json").read_bytes()
                )
                self.assertTrue(report["passed"])
                raise TimeoutError("attention teardown timed out")

            with self.graph_u1_services(directory) as (launcher, owner, plan):
                owner.stop_attention.side_effect = stop_attention
                code = launcher.run_candidate(
                    plan,
                    directory,
                    Path("/vendor"),
                    1,
                    None,
                    request_runner=lambda *_: {"passed": True},
                )
            report = json.loads((directory / "summary.json").read_bytes())
            self.assertTrue(report["request_passed"])
            self.assertTrue(report["graph_verification"]["passed"])
            self.assertFalse(report["clean_shutdown"])
            self.assertIn("attention teardown timed out", report["shutdown_error"])
            self.assertEqual(code, 1)

    def test_graph_u1_request_timeout_collects_stacks_before_shutdown(self):
        def request_timeout(*_args):
            raise TimeoutError("first wave timed out")

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_ep_graph_logs(directory, 1)
            with self.graph_u1_services(directory) as (launcher, owner, plan):
                stack_dir = directory.resolve() / launcher.EP_GRAPH_STACK_DIR

                def capture_stacks(_attention, name):
                    (directory / name / "42.log").write_text(
                        "worker waiting for graph query"
                    )

                def stop_attention(*_args, **_kwargs):
                    self.assertTrue((stack_dir / "42.log").is_file())

                owner.request_attention_stacks.side_effect = capture_stacks
                owner.stop_attention.side_effect = stop_attention
                code = launcher.run_candidate(
                    plan,
                    directory,
                    Path("/vendor"),
                    1,
                    None,
                    request_runner=request_timeout,
                )
            report = json.loads((directory / "summary.json").read_bytes())
            self.assertEqual(report["error"], "TimeoutError: first wave timed out")
            self.assertFalse(report["request_passed"])
            self.assertNotIn("shutdown_error", report)
            self.assertEqual(code, 1)

    def test_reports_capture_replay_and_runtime_eager_from_complete_logs(self):
        for max_stages in (1, 2):
            with (
                self.subTest(max_stages=max_stages),
                tempfile.TemporaryDirectory() as tmp,
            ):
                directory = Path(tmp)
                write_ep_graph_logs(directory, max_stages)
                result = load_launcher().verify_ep_graph_run(directory, 2, max_stages)
                self.assertEqual(
                    result["transactions"],
                    {phase: [phase] for phase in ("capture", "replay", "eager")},
                )
                self.assertTrue(result["passed"])
                self.assertFalse(result["device_completion_verified_by_logs_alone"])

    def test_rejects_replay_with_a_missing_ffn_peer(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_ep_graph_logs(directory, 2, missing_replay_rank=True)
            with self.assertRaisesRegex(ValueError, "replay: missing participant.*ffn"):
                load_launcher().verify_ep_graph_run(directory, 2, 2)


def write_ep_eager_u2_logs(directory, *, runtime="complete"):
    windows = {}
    for role in ("attention", "ffn"):
        path = directory / f"{role}-server.log"
        with path.open("wb") as log:
            for transaction, variant in (("startup", "complete"), ("live", runtime)):
                start = log.tell()
                for rank in range(4):
                    if variant == "missing-rank" and (role, rank) == ("ffn", 3):
                        continue
                    for record in ep_eager_u2_records(role, rank, transaction, variant):
                        log.write(f"AFD HCCL stage diagnostic: {record!r}\n".encode())
                windows[role] = (start, log.tell())
    return windows


def ep_eager_u2_records(role, rank, transaction, variant):
    attention = role == "attention"
    progress = {
        "role": role,
        "role_rank": rank,
        "world_rank": rank + (4 if attention else 0),
        "transaction_id": "other"
        if variant == "other-transaction" and not attention
        else transaction,
        "phase": "eager",
        "num_stages": 2,
        "boundary": "host_enqueue",
        "actual.mode": "eager-streamed" if variant == "streamed" else "serial",
        "actual.compute_stream": "parent",
        "actual.send_stream": "parent",
        "actual.recv_stream": "parent",
        "in_profile_run": variant == "profile",
        "is_warmup": variant == "warmup",
        "layer": 42,
    }
    operations = (
        ("compute", "send", "recv") if attention else ("recv", "compute", "send")
    )
    for stage in (0, 1):
        if variant == "missing-stage" and (role, rank, stage) == ("ffn", 3, 1):
            continue
        for operation in operations:
            yield {**progress, "event": f"{operation}_returned", "stage": stage}
    if attention:
        yield {**progress, "event": "collapse_returned", "stage": 1}


class TestEpEagerU2LogVerification(unittest.TestCase):
    def test_reports_both_stages_for_the_same_live_parent_on_eight_ranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            windows = write_ep_eager_u2_logs(directory)
            result = load_launcher().verify_ep_eager_u2_run(directory, 4, windows)
            self.assertTrue(result["passed"])
            self.assertEqual(result["world_ranks"], list(range(8)))
            self.assertEqual(result["transactions"], {"live": 42})

    def test_rejects_complete_startup_when_live_parent_has_insufficient_evidence(self):
        for variant in (
            "warmup",
            "profile",
            "missing-rank",
            "missing-stage",
            "other-transaction",
        ):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                windows = write_ep_eager_u2_logs(directory, runtime=variant)
                with self.assertRaisesRegex(
                    ValueError, "no live serial eager U2 parent"
                ):
                    load_launcher().verify_ep_eager_u2_run(directory, 4, windows)

    def test_rejects_streamed_u2_as_evidence_of_the_serial_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            windows = write_ep_eager_u2_logs(directory, runtime="streamed")
            with self.assertRaisesRegex(ValueError, "actual.mode.*serial"):
                load_launcher().verify_ep_eager_u2_run(directory, 4, windows)


def completion(*, mismatch=None, prompt=(1, 2, 3)):
    tokens = list(range(32))
    if mismatch is not None:
        tokens[mismatch] += 100
    return {
        "choices": [
            {
                "token_ids": tokens,
                "prompt_token_ids": list(prompt),
                "finish_reason": "length",
            }
        ],
        "usage": {"completion_tokens": len(tokens)},
    }


@contextmanager
def validation_service(health_status=200):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(health_status)
            self.end_headers()

        def do_POST(self):
            if health_status != 200:
                self.send_error(health_status)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            response = completion()
            if payload.get("kv_transfer_params", {}).get("do_remote_decode"):
                response["kv_transfer_params"] = {
                    "afd_engram_history_v1": "opaque-http-test-value"
                }
            body = (
                b'data: {"choices":[{"text":"generated"}]}\n\n'
                if payload["stream"]
                else json.dumps(response).encode()
            )
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


class TestValidationHttp(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        proxy = self.stack.enter_context(validation_service(403))
        self.stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "http_proxy": proxy,
                    "https_proxy": proxy,
                    "no_proxy": "",
                    "NO_PROXY": "",
                },
            )
        )
        # Recreate urllib's default transport under the inherited proxy settings.
        self.stack.enter_context(
            patch("urllib.request._opener", urllib.request.build_opener())
        )
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.launcher = load_launcher()

    def test_health_reaches_service_with_inherited_proxy(self):
        with validation_service() as url:
            self.assertTrue(self.launcher.health(url, directory=self.directory))
        report = json.loads((self.directory / "health.json").read_bytes())
        self.assertEqual(report["client"], "curl")
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["status_code"], 200)
        self.assertIsNone(report["error"])

    def test_unhealthy_service_keeps_http_failure_diagnostic(self):
        with validation_service(503) as url:
            self.assertFalse(self.launcher.health(url, directory=self.directory))
        report = json.loads((self.directory / "health.json").read_bytes())
        self.assertEqual(report["status_code"], 503)
        self.assertIn("503", report["error"])
        self.assertFalse(report["passed"])

    def test_missing_curl_preserves_error_and_fails_without_waiting_for_startup(self):
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("curl not found")),
            self.assertRaisesRegex(RuntimeError, "curl"),
        ):
            self.launcher.health(directory=self.directory)
        report = json.loads((self.directory / "health.json").read_bytes())
        self.assertFalse(report["passed"])
        self.assertIn("curl not found", report["error"])

    def test_curl_timeout_preserves_error_and_does_not_pass_readiness(self):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(["curl"], 3),
        ):
            self.assertFalse(self.launcher.health(directory=self.directory))
        report = json.loads((self.directory / "health.json").read_bytes())
        self.assertFalse(report["passed"])
        self.assertIn("TimeoutExpired", report["error"])

    def test_incomplete_curl_transfer_cannot_pass_on_http_status_alone(self):
        reply = subprocess.CompletedProcess(
            ["curl"], 28, stdout="200", stderr="transfer timed out"
        )
        with patch("subprocess.run", return_value=reply):
            self.assertFalse(self.launcher.health(directory=self.directory))
        report = json.loads((self.directory / "health.json").read_bytes())
        self.assertEqual(report["exit_code"], 28)
        self.assertEqual(report["status_code"], 200)
        self.assertIn("transfer timed out", report["error"])

    def test_generation_and_cancellation_reach_service_with_inherited_proxy(self):
        with validation_service() as url:
            with patch.object(
                sys, "path", [str(ROOT / "tools/diagnostics"), *sys.path]
            ):
                driver = importlib.import_module("run_dsv41_full_checkpoint")
            report = driver.run_requests(
                self.directory, None, spec_tokens=0, prefill_url=url, api_url=url
            )
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["requests"]), 5)
        self.assertTrue((self.directory / "cancel-first-chunk.json").is_file())
        prefill = json.loads(
            (self.directory / "request-00/prefill-response.json").read_bytes()
        )
        self.assertEqual(
            prefill["kv_transfer_params"]["afd_engram_history_v1"],
            "opaque-http-test-value",
        )


class TestFullCheckpointPreflight(unittest.TestCase):
    def test_preflight_matches_locked_git_source_identities(self):
        launcher = load_launcher()
        preflight = sys.modules[launcher.collect_preflight.__module__]
        evidence = json.loads((EVIDENCE / "native-pins-red.json").read_text("utf-8"))
        expected = {row["path"]: row["locked_sha256"] for row in evidence["pins"]}
        draft_pins = json.loads(
            (
                EVIDENCE.parent / "a5-dspark-heads-eager-u2-20261120/source-pins.json"
            ).read_text("utf-8")
        )
        expected.update(
            {row["path"]: row["locked_sha256"] for row in draft_pins["pins"]}
        )
        actual = {
            source.package + "/" + source.path: source.sha256
            for source in preflight.DEFAULT_PROFILE.sources
        }
        self.assertEqual(actual, expected)


class TestFullCheckpointRequest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="afd-checkpoint-response-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.launcher = load_launcher()

    def send(self, baseline=None):
        reference = None
        if baseline is not None:
            reference = self.directory / "native.json"
            reference.write_text(json.dumps(baseline), encoding="utf-8")
        reply = io.BytesIO(json.dumps(completion()).encode())
        reply.status = 200
        with patch("urllib.request.OpenerDirector.open", return_value=reply):
            return self.launcher.send_request(self.directory, reference)

    def comparison(self):
        return json.loads((self.directory / "token-comparison.json").read_text("utf-8"))

    def test_without_reference_only_execution_is_reported(self):
        result = self.send()
        self.assertTrue(result["passed"])
        self.assertIsNone(result["reference_token_ids_match"])
        comparison = self.comparison()
        self.assertEqual(comparison["status"], "not_checked")
        self.assertFalse(comparison["model_equivalence_accepted"])

    def test_equal_prompt_and_output_tokens_are_recorded(self):
        self.send(completion())
        comparison = self.comparison()
        self.assertEqual(comparison["status"], "tokens_match")
        self.assertTrue(comparison["prompt_token_ids_match"])
        self.assertIsNone(comparison["first_output_mismatch"])
        self.assertFalse(comparison["model_equivalence_accepted"])

    def test_differences_preserve_the_first_mismatch_before_failure(self):
        cases = ((0, (1, 2, 3)), (31, (1, 2, 3)), (None, (9, 2, 3)))
        for index, prompt in cases:
            with self.subTest(index=index, prompt=prompt):
                with self.assertRaises(RuntimeError):
                    self.send(completion(mismatch=index, prompt=prompt))
                comparison = self.comparison()
                self.assertEqual(comparison["status"], "tokens_differ")
                self.assertEqual(
                    comparison["prompt_token_ids_match"], prompt == (1, 2, 3)
                )
                if index is not None:
                    self.assertEqual(
                        comparison["first_output_mismatch"],
                        {
                            "index": index,
                            "reference_token_id": index + 100,
                            "actual_token_id": index,
                        },
                    )

    def test_missing_reference_prompt_is_rejected_with_a_report(self):
        baseline = completion()
        baseline["choices"][0].pop("prompt_token_ids")
        with self.assertRaisesRegex(RuntimeError, "reference"):
            self.send(baseline)
        self.assertEqual(self.comparison()["status"], "invalid_reference")

    def test_invalid_reference_files_cannot_be_treated_as_absent(self):
        for content in ("null", "[]", "truncated-json"):
            with self.subTest(content=content):
                reference = self.directory / "invalid-reference.json"
                reference.write_text(content, encoding="utf-8")
                reply = io.BytesIO(json.dumps(completion()).encode())
                reply.status = 200
                with (
                    patch("urllib.request.OpenerDirector.open", return_value=reply),
                    self.assertRaisesRegex(RuntimeError, "reference"),
                ):
                    self.launcher.send_request(self.directory, reference)
                self.assertEqual(self.comparison()["status"], "invalid_reference")


if __name__ == "__main__":
    unittest.main()
