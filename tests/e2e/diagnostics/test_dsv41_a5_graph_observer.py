# SPDX-License-Identifier: Apache-2.0
"""Frozen harness host-observation contracts; no torch, runtime or device."""

from __future__ import annotations

import json
import sys
import tarfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def load_observer(suffix="graph"):
    bundle = ROOT / "reports/dsv41-a5-functional/runtime-source.tar.gz"
    helper = f"tests/npu/hccl_native_attention_pd_dspark_{suffix}_probe.py"
    with tarfile.open(bundle) as archive:
        source = archive.extractfile(helper).read()
    module = ModuleType("frozen_graph_observer")
    exec(compile(source, helper, "exec"), module.__dict__)
    return module


def recorded_device_audit():
    directory = ROOT / (
        "openspec/changes/afd-hccl-full-parity/evidence/a5-dspark-device-audit-20261119"
    )
    with tarfile.open(directory / "inner-reports.tar.gz") as archive:
        workers = json.load(archive.extractfile("summary.json"))["workers"]
    audit = json.loads((directory / "offline-green.json").read_text("utf-8"))
    return (
        workers,
        audit["consumer_protocol"]["parents"],
        audit["expected_local_devices"],
    )


class Attention:
    def forward(self, positions, hidden_states, scaling=None):
        return hidden_states


class Target:
    def __init__(self):
        self.layers = [SimpleNamespace(self_attn=Attention())]

    def embed_input_ids(self, input_ids):
        return input_ids


class Wrapper:
    def __init__(self):
        self.language_model = SimpleNamespace(model=Target())

    def forward(self, *args, **kwargs):
        raise AssertionError("the staged path bypasses the wrapper")


class Connector:
    def publish_dspark_ready(self, ready_batch):
        return None


class Owner:
    def __init__(
        self, records, *, requests, eager_calls, replay_calls, prepare_embeds, rows
    ):
        self.model = Wrapper()
        self.records = records
        self.requests = requests
        self.eager_calls = eager_calls
        self.replay_calls = replay_calls
        self.prepare_embeds = prepare_embeds
        self.rows = rows
        self.kv_connector = SimpleNamespace(kv_connector=Connector())
        self.hccl_v41_graph_coordinator = SimpleNamespace(
            control=SimpleNamespace(cache=SimpleNamespace(closed=False)),
            transport=SimpleNamespace(closed=False),
        )
        self.execute_model_state = SimpleNamespace(
            hidden_states=SimpleNamespace(data_ptr=lambda: 123),
            aux_hidden_states=[],
        )

    def execute_model(self, step, intermediate_tensors=None):
        target = self.model.language_model.model
        if self.prepare_embeds:
            # Native MRV2 prepares the parent embeddings before eager or replay.
            target.embed_input_ids(SimpleNamespace(numel=lambda: 3 * self.requests))
        calls = self.eager_calls if step.phase == "eager" else self.replay_calls
        for _ in range(calls):
            positions = SimpleNamespace(numel=lambda: self.rows)
            hidden = (
                positions if self.prepare_embeds else target.embed_input_ids(positions)
            )
            target.layers[0].self_attn.forward(positions, hidden)
        self.records.append(
            {
                "event": "graph_parent_finish_returned",
                "transaction_id": str(len(self.records)),
                "phase": step.phase,
                "mode": "decode",
                "reason": "recorded host fixture",
                "key_digest": "retained" if step.phase == "replay" else "",
                "key_present": step.phase == "replay",
            }
        )
        return None


def scenario(
    requests, *, eager_calls=None, replay_calls=0, prepare_embeds=True, rows=3
):
    records = []
    owner = Owner(
        records,
        requests=requests,
        eager_calls=requests if eager_calls is None else eager_calls,
        replay_calls=replay_calls,
        prepare_embeds=prepare_embeds,
        rows=rows,
    )
    report = {
        "native_live_owner": {
            "target_graph": {
                "capture_kv_bytes_unchanged": True,
                "capture_physical_pages_unchanged": True,
                "host_records": records,
            }
        },
        "native_decode_and_verification_accepted": True,
        "two_request_native_decode_accepted": True,
        "receiver_retired": True,
        "native_owner_shutdown_returned": True,
    }
    storage = {
        "entries": [
            {
                "key_digest": "retained",
                "num_reqs": requests,
                "num_stages": requests,
                "hidden": 123,
                "auxiliary": [],
            }
        ]
    }
    return owner, report, storage


def execute_pair(observer, owner, report, storage, requests):
    with (
        patch.object(observer, "graph_storage", return_value=storage),
        observer.observe_graph_decode(
            SimpleNamespace(owner=owner),
            report,
            target_requests=requests,
            allow_u2=requests == 2,
        ),
    ):
        for phase in ("eager", "replay"):
            owner.execute_model(
                SimpleNamespace(
                    total_num_scheduled_tokens=3 * requests,
                    finished_req_ids=set(),
                    phase=phase,
                )
            )
        owner.hccl_v41_graph_coordinator.control.cache.closed = True
        owner.hccl_v41_graph_coordinator.transport.closed = True


class TestGraphObserver(unittest.TestCase):
    def test_staged_eager_and_replay_are_observed_without_wrapper_forward(self):
        observer = load_observer()
        for requests, prepare_embeds in ((1, True), (2, True), (1, False), (2, False)):
            with self.subTest(requests=requests, prepare_embeds=prepare_embeds):
                owner, report, storage = scenario(
                    requests, prepare_embeds=prepare_embeds
                )
                execute_pair(observer, owner, report, storage, requests)
                result = report["target_graph_decode"]
                self.assertTrue(result["accepted"])
                self.assertEqual(
                    [row["python_target_calls"] for row in result["executions"]],
                    [requests, 0],
                )
                self.assertEqual(
                    [row["python_target_rows"] for row in result["executions"]],
                    [[3] * requests, []],
                )
                self.assertNotIn(
                    "forward",
                    vars(owner.model.language_model.model.layers[0].self_attn),
                )

    def test_missing_eager_or_python_execution_during_replay_is_rejected(self):
        observer = load_observer()
        cases = ((0, 0, 3), (1, 0, 3), (2, 1, 3), (2, 0, 6))
        for eager_calls, replay_calls, rows in cases:
            with self.subTest(
                eager_calls=eager_calls, replay_calls=replay_calls, rows=rows
            ):
                owner, report, storage = scenario(
                    2, eager_calls=eager_calls, replay_calls=replay_calls, rows=rows
                )
                with self.assertRaises(AssertionError):
                    execute_pair(observer, owner, report, storage, 2)
                self.assertFalse(report["target_graph_decode"]["accepted"])
                self.assertEqual(
                    report["target_graph_decode"]["last_returned_execution"][
                        "python_target_calls"
                    ],
                    replay_calls if replay_calls else eager_calls,
                )
                self.assertNotIn(
                    "forward",
                    vars(owner.model.language_model.model.layers[0].self_attn),
                )


class TestGraphDeviceAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workers, cls.parents, cls.local_devices = recorded_device_audit()

    def test_host_records_require_the_requested_device_for_each_role(self):
        observer = load_observer()
        service = self.workers["ffn_consumer"]["native_ffn"]
        records = {
            "attention": self.workers["kv_consumer"]["native_live_owner"][
                "target_graph"
            ]["host_records"],
            "ffn": [*service["host_records"], *service["body_records"]],
        }
        for role, values in records.items():
            with self.subTest(role=role):
                options = {
                    "role": role,
                    "graph_compute_overlap": True,
                    "graph_all_axes": True,
                }
                observer.verify_graph_host_records(
                    values,
                    self.parents,
                    local_device=self.local_devices[role],
                    **options,
                )
                with self.assertRaises(AssertionError):
                    observer.verify_graph_host_records(
                        values,
                        self.parents,
                        local_device=2 if role == "attention" else 6,
                        **options,
                    )

    def test_pipeline_storage_requires_the_requested_device_for_each_role(self):
        observer = load_observer("overlap")
        cross = load_observer("cross_layer")
        attention = self.workers["kv_consumer"]["target_graph_decode"]["storage_before"]
        ffn = self.workers["ffn_consumer"]["native_ffn"]["captured_entries"]
        for role, entries in (("attention", attention["entries"]), ("ffn", ffn)):
            pipeline = next(
                item["pipeline_storage"]
                for item in entries
                if item["pipeline_storage"] is not None
            )
            options = {
                "bank": pipeline["bank"],
                "graph_all_axes": True,
                "graph_exchange_order": "cross_layer_v1",
            }
            with (
                self.subTest(role=role),
                patch.dict(
                    sys.modules,
                    {"hccl_native_attention_pd_dspark_cross_layer_probe": cross},
                ),
            ):
                observer.verify_pipeline_storage(
                    pipeline, device=self.local_devices[role], **options
                )
                with self.assertRaises(AssertionError):
                    observer.verify_pipeline_storage(
                        pipeline, device=2 if role == "attention" else 6, **options
                    )


if __name__ == "__main__":
    unittest.main()
