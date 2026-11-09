# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# ruff: noqa: E402
"""Actual vLLM public stream queries with a host NPU-selection boundary."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.compat.npu import stream as module
from tests.unit.fakes.npu_stream import HostStream, make_npu_stream_case


@pytest.mark.parametrize("cached", ["parent", "foreign", "none", "unset"])
@pytest.mark.parametrize("failed", [False, True])
def test_native_and_vllm_queries_share_the_scoped_stream_and_restore(
    monkeypatch, cached, failed
):
    case = make_npu_stream_case(monkeypatch, module, cached=cached)
    if failed:
        with (
            pytest.raises(RuntimeError, match="compute failed"),
            module.use_npu_current_stream(case.runtime.compute),
        ):
            assert case.runtime.current_stream() is case.runtime.compute
            assert case.query() is case.runtime.compute
            raise RuntimeError("compute failed")
    else:
        with module.use_npu_current_stream(case.runtime.compute):
            assert case.runtime.current_stream() is case.runtime.compute
            assert case.query() is case.runtime.compute
    assert case.runtime.current_stream() is case.runtime.parent
    assert case.query() is case.previous


def test_nested_stream_scopes_restore_the_enclosing_native_and_vllm_stream(monkeypatch):
    case = make_npu_stream_case(monkeypatch, module)
    nested = HostStream("nested", case.runtime.compute.device)
    with module.use_npu_current_stream(case.runtime.compute):
        with module.use_npu_current_stream(nested):
            assert case.runtime.current_stream() is nested
            assert case.query() is nested
        assert case.runtime.current_stream() is case.runtime.compute
        assert case.query() is case.runtime.compute
    assert case.runtime.current_stream() is case.runtime.parent
    assert case.query() is case.previous


def test_native_context_entry_failure_leaves_public_stream_queries_unchanged(
    monkeypatch,
):
    case = make_npu_stream_case(monkeypatch, module)
    case.runtime.fail_enter = True
    with (
        pytest.raises(RuntimeError, match="native stream entry failed"),
        module.use_npu_current_stream(case.runtime.compute),
    ):
        pytest.fail("native stream entry did not fail")
    assert case.runtime.current_stream() is case.runtime.parent
    assert case.query() is case.previous
