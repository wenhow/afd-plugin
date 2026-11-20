# SPDX-License-Identifier: Apache-2.0
"""Host contracts for full-checkpoint diagnostics; no model or NPU execution."""

from __future__ import annotations

import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from contextlib import ExitStack, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / (
    "openspec/changes/afd-hccl-full-parity/evidence/a5-full-checkpoint-guide-20261119"
)


def load_launcher():
    with patch.object(sys, "path", [str(ROOT / "tools/diagnostics"), *sys.path]):
        return importlib.import_module("run_dsv41_a5_afd")


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
