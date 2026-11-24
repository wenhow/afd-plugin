# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch policy probes; these do not emulate NPU graph execution."""

import ast
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest


class Mode(Enum):
    NONE = 0
    FULL = 1


@dataclass(frozen=True)
class Descriptor:
    cg_mode: Mode
    num_tokens: int
    num_reqs: int
    num_ubatches: int = 1
    uniform_token_count: int | None = None
    num_active_loras: int = 0


@pytest.fixture
def dispatch():
    # Match the existing GPU dispatch probes: load the actual scoped policy
    # against a small native DP boundary, without importing device runtimes.
    path = Path(__file__).resolve().parents[5] / (
        "afd_plugin/compat/patches/npu/mrv2_ubatching.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "use_async_cam_dbo_dispatch"
    ]

    def native(manager, requests, tokens, width, dp_size, dp_rank, **kwargs):
        # Return the agreed counts at the public dispatch seam. A peer may be
        # idle or much larger; the target's local descriptor must stay local.
        counts = [tokens, 0, 100][:dp_size]
        return Descriptor(Mode.NONE, tokens, requests), counts

    module = SimpleNamespace(dispatch_cg_and_sync_dp=native)
    namespace = dict(
        contextmanager=contextmanager,
        replace=replace,
        model_runner=module,
        CUDAGraphMode=Mode,
        BatchExecutionDescriptor=Descriptor,
        get_num_ubatches=lambda config: config.num_ubatches,
    )
    future = ast.parse("from __future__ import annotations").body
    exec(
        compile(
            ast.Module(body=future + functions, type_ignores=[]), str(path), "exec"
        ),
        namespace,
    )
    return namespace["use_async_cam_dbo_dispatch"], module


@pytest.mark.parametrize("stages", [1, 2, 4])
@pytest.mark.parametrize("dp_size", [1, 3])
def test_local_verification_graph_keeps_draft_dp_counts(dispatch, stages, dp_size):
    scope, module = dispatch
    graph = Descriptor(Mode.FULL, stages * 5, stages, stages, 5)
    manager = SimpleNamespace(graphs={graph: "captured graph"})
    with scope(dspark_query_width=5):
        desc, counts = module.dispatch_cg_and_sync_dp(
            manager,
            1,
            5,
            5,
            dp_size,
            0,
            uniform_decode=True,
            allow_ubatching=stages > 1,
            parallel_config=SimpleNamespace(num_ubatches=stages),
        )
    assert desc == graph
    assert counts == [graph.num_tokens, 0, 100][:dp_size]


@pytest.mark.parametrize("stages", [1, 2, 4])
@pytest.mark.parametrize("tokens,width,force_eager", [(3, None, False), (5, 5, True)])
def test_prefill_and_profiling_keep_eager_target(
    dispatch, stages, tokens, width, force_eager
):
    scope, module = dispatch
    graph = Descriptor(Mode.FULL, stages * 5, stages, stages, 5)
    with scope(dspark_query_width=5):
        desc, counts = module.dispatch_cg_and_sync_dp(
            SimpleNamespace(graphs={graph: "graph"}),
            1,
            tokens,
            width,
            3,
            0,
            need_eager=force_eager,
            uniform_decode=width == 5,
            allow_ubatching=stages > 1,
            parallel_config=SimpleNamespace(num_ubatches=stages),
        )
    assert desc.cg_mode is Mode.NONE
    assert desc.num_ubatches == stages
    assert counts == [max(tokens, stages), 0, 100]


@pytest.mark.parametrize("stages", [1, 2, 4])
def test_idle_single_request_keeps_real_request_count_for_draft_sync(dispatch, stages):
    scope, module = dispatch
    padded_rows = max(2, stages)
    graph = Descriptor(Mode.FULL, padded_rows * 5, padded_rows, stages, 5)
    with scope(dspark_query_width=5, allow_target_graph=False):
        desc, counts = module.dispatch_cg_and_sync_dp(
            SimpleNamespace(graphs={graph: "graph"}),
            1,
            5,
            5,
            3,
            0,
            uniform_decode=True,
            allow_ubatching=stages > 1,
            parallel_config=SimpleNamespace(num_ubatches=stages),
        )
    # make_dummy uses descriptor.num_reqs as real rows; draft DP agreed on
    # one request. Graph padding must not inflate this to two or four rows.
    assert desc.cg_mode is Mode.NONE
    assert desc.num_reqs == 1
    assert desc.num_tokens == 5
    assert counts == [5, 0, 100]
