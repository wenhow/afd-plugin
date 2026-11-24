# SPDX-License-Identifier: Apache-2.0
"""CPU probes of the replay control boundary, not NPU device execution."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest


@pytest.fixture
def replay_hook():
    path = Path(__file__).resolve().parents[5] / (
        "afd_plugin/v1/worker/npu/attention_model_runner_v2.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_use_afd_fullgraph_replay_hook"
    ]
    namespace = dict(
        contextmanager=contextmanager,
        MethodType=MethodType,
        _AFD_FULLGRAPH_HOOK_MARKER="_afd_fullgraph_replay_hook_active",
    )
    future = ast.parse("from __future__ import annotations").body
    exec(
        compile(
            ast.Module(body=future + functions, type_ignores=[]), str(path), "exec"
        ),
        namespace,
    )
    return namespace["_use_afd_fullgraph_replay_hook"]


@pytest.mark.parametrize("has_control_plane", [False, True])
def test_single_stage_replay_respects_connector_control_plane(
    replay_hook, has_control_plane
):
    messages = []
    output = ["complete target hidden states"]
    connector = SimpleNamespace(control_plane=messages if has_control_plane else None)

    def send(metadata, slices):
        assert connector.control_plane is not None, "connector has no control plane"
        connector.control_plane.append(metadata)

    manager = SimpleNamespace(run_fullgraph=lambda desc: output)
    runner = SimpleNamespace(
        cudagraph_manager=manager,
        connector=connector,
        build_afd_metadata=lambda *args: SimpleNamespace(),
        build_capture_dp_metadata=lambda tokens: {"tokens": tokens},
        send_dp_metadata=send,
        _afd_is_graph_replaying=False,
    )
    desc = SimpleNamespace(num_ubatches=1, num_tokens=5)
    with replay_hook(runner, 5, verifying_drafts=True):
        assert manager.run_fullgraph(desc) is output
    assert messages == ([{"tokens": 5}] if has_control_plane else [])
    assert manager.run_fullgraph(desc) is output
