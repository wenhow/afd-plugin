# SPDX-License-Identifier: Apache-2.0
"""Frozen harness host-observation contracts; no torch, runtime or device."""

from __future__ import annotations

import tarfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
HELPER = "tests/npu/hccl_native_attention_pd_dspark_graph_probe.py"


def load_observer():
    bundle = ROOT / "reports/dsv41-a5-functional/runtime-source.tar.gz"
    with tarfile.open(bundle) as archive:
        source = archive.extractfile(HELPER).read()
    module = ModuleType("frozen_graph_observer")
    exec(compile(source, HELPER, "exec"), module.__dict__)
    return module


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


if __name__ == "__main__":
    unittest.main()
