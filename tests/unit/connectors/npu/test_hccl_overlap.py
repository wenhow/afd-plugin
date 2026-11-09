# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pure settings contracts; no runtime or device capability is implied."""

import importlib.util
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType

import pytest

ENVIRONMENT_FIELDS = (
    ("AFD_HCCL_EAGER_U2_STREAM_OVERLAP", "eager_u2_stream_overlap", True),
    ("AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP", "graph_u2_compute_overlap", True),
    ("AFD_HCCL_GRAPH_U2_HYBRID_DAG", "graph_u2_hybrid_dag", True),
    (
        "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM",
        "graph_u2_attention_three_stream",
        True,
    ),
    ("AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM", "graph_u2_ffn_recv_stream", True),
    ("AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER", "graph_u2_ffn_cross_layer", True),
    ("AFD_HCCL_STAGE_DIAGNOSTICS", "stage_diagnostics", False),
    ("AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS", "ffn_compute_sync_diagnostics", False),
)


@pytest.fixture
def overlap(monkeypatch):
    # Package namespaces load device connectors. Execute the actual standalone
    # settings module so this seam also runs on a host without torch or vLLM.
    name = "_afd_unit_hccl_overlap"
    path = (
        Path(__file__).resolve().parents[4]
        / "afd_plugin/connectors/npu/hccl_overlap.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("num_stages", (1, 2))
def test_unset_environment_preserves_donor_defaults_and_u1_gate(overlap, num_stages):
    settings = overlap.HcclOverlapSettings.from_environment({}, num_stages=num_stages)
    expected = {name: default for name, _, default in ENVIRONMENT_FIELDS}
    expected["AFD_HCCL_EAGER_U2_STREAM_OVERLAP"] = num_stages > 1
    assert settings.effective_settings() == expected


@pytest.mark.parametrize("name,field,default", ENVIRONMENT_FIELDS)
@pytest.mark.parametrize("raw,expected", ((" \t0\n", False), (" \t1\n", True)))
def test_binary_values_are_stripped_and_explicit_zero_is_retained(
    overlap, name, field, default, raw, expected
):
    environment = {name: raw}
    if name == "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM":
        environment["AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER"] = "0"
    settings = overlap.HcclOverlapSettings.from_environment(environment, num_stages=2)
    assert settings.effective_settings()[name] is expected


@pytest.mark.parametrize("name,field,default", ENVIRONMENT_FIELDS)
@pytest.mark.parametrize("raw", ("", "true", "2", "01"))
def test_invalid_binary_values_name_the_rejected_setting(
    overlap, name, field, default, raw
):
    with pytest.raises(RuntimeError, match=f"{name} must be 0 or 1"):
        overlap.HcclOverlapSettings.from_environment({name: raw}, num_stages=2)


@pytest.mark.parametrize("raw", ("", "true", "0", "1"))
def test_u1_does_not_parse_or_enable_the_eager_environment(overlap, raw):
    settings = overlap.HcclOverlapSettings.from_environment(
        {"AFD_HCCL_EAGER_U2_STREAM_OVERLAP": raw}, num_stages=1
    )
    assert settings.eager_u2_stream_overlap is False


@pytest.mark.parametrize("name,field,default", ENVIRONMENT_FIELDS[1:])
def test_u1_still_validates_all_graph_and_diagnostic_settings(
    overlap, name, field, default
):
    with pytest.raises(RuntimeError, match=f"{name} must be 0 or 1"):
        overlap.HcclOverlapSettings.from_environment({name: "invalid"}, num_stages=1)


@pytest.mark.parametrize("num_stages", (1, 2))
def test_cross_layer_requires_receive_stream_even_when_u1(overlap, num_stages):
    with pytest.raises(
        RuntimeError,
        match=(
            "AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=1 requires "
            "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=1"
        ),
    ):
        overlap.HcclOverlapSettings.from_environment(
            {"AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM": "0"}, num_stages=num_stages
        )


@pytest.mark.parametrize("num_stages", (0, -1, True, 1.0))
def test_stage_count_requires_a_positive_integer(overlap, num_stages):
    with pytest.raises(ValueError, match="positive integer"):
        overlap.HcclOverlapSettings.from_environment({}, num_stages=num_stages)


@pytest.mark.parametrize("name,field,default", ENVIRONMENT_FIELDS)
def test_direct_settings_require_bool_fields(overlap, name, field, default):
    with pytest.raises(ValueError, match="settings must be bool"):
        replace(overlap.HcclOverlapSettings(), **{field: 1})


def test_settings_are_frozen(overlap):
    settings = overlap.HcclOverlapSettings()
    with pytest.raises(FrozenInstanceError):
        settings.graph_u2_hybrid_dag = False


def test_diagnostic_export_and_environment_are_independent_snapshots(overlap):
    environment = {"AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP": "0"}
    settings = overlap.HcclOverlapSettings.from_environment(
        MappingProxyType(environment), num_stages=2
    )
    environment["AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"] = "1"
    record = settings.effective_settings()
    record["AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"] = True
    assert settings.effective_settings()["AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"] is False
