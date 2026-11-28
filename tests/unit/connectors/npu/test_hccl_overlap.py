# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pure settings and selection; no runtime or device capability is implied."""

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


@pytest.mark.parametrize("phase", ("graph_warmup", "graph_capture", "graph_replay"))
@pytest.mark.parametrize(
    "role,num_stages,enabled,logical",
    (
        ("attention", 1, (), ("parent", "parent", "parent")),
        ("ffn", 1, (), ("parent", "parent", "parent")),
        (
            "attention",
            2,
            (
                "graph_u2_compute_overlap",
                "graph_u2_hybrid_dag",
                "graph_u2_attention_three_stream",
            ),
            ("compute", "send", "recv"),
        ),
        (
            "ffn",
            2,
            (
                "graph_u2_compute_overlap",
                "graph_u2_ffn_recv_stream",
                "graph_u2_ffn_cross_layer",
            ),
            ("compute", "send", "recv"),
        ),
    ),
)
def test_graph_selection_distinguishes_logical_plan_from_parent_transport(
    overlap, phase, role, num_stages, enabled, logical
):
    selection = overlap.select_hccl_overlap(
        overlap.HcclOverlapSettings(), role=role, phase=phase, num_stages=num_stages
    )
    expected = {name: field in enabled for name, field, _ in ENVIRONMENT_FIELDS}
    assert selection.effective_settings() == expected
    record = selection.diagnostic_record()
    assert (
        tuple(record[f"logical.{kind}_stream"] for kind in ("compute", "send", "recv"))
        == logical
    )
    assert record["physical.compute_stream"] == logical[0]
    assert record["physical.send_stream"] == record["physical.recv_stream"] == "parent"
    assert record["phase"] == phase
    assert record["parent_transport_override"] is True


@pytest.mark.parametrize("role", ("attention", "ffn"))
@pytest.mark.parametrize("num_stages", (1, 2))
def test_eager_selection_never_enables_graph_axes(overlap, role, num_stages):
    selection = overlap.select_hccl_overlap(
        overlap.HcclOverlapSettings(), role=role, phase="eager", num_stages=num_stages
    )
    assert selection.effective_settings() == {
        name: name == overlap.EAGER_U2_STREAM_OVERLAP_ENV and num_stages == 2
        for name, _, _ in ENVIRONMENT_FIELDS
    }
    record = selection.diagnostic_record()
    expected = (
        ("parent", "parent", "parent")
        if num_stages == 1
        else ("compute", "send", "recv")
    )
    for domain in ("logical", "physical"):
        assert (
            tuple(
                record[f"{domain}.{kind}_stream"]
                for kind in ("compute", "send", "recv")
            )
            == expected
        )
    assert record["parent_transport_override"] is False


@pytest.mark.parametrize("role", ("attention", "ffn"))
@pytest.mark.parametrize("phase", ("graph_warmup", "graph_capture", "graph_replay"))
@pytest.mark.parametrize("all_off", (False, True))
def test_compute_off_keeps_graph_phase_with_configured_and_effective_values(
    overlap, role, phase, all_off
):
    changes = {"graph_u2_compute_overlap": False}
    if all_off:
        changes.update(
            {
                field: False
                for _, field, _ in ENVIRONMENT_FIELDS
                if field.startswith("graph_")
            }
        )
    settings = replace(overlap.HcclOverlapSettings(), **changes)
    selection = overlap.select_hccl_overlap(
        settings, role=role, phase=phase, num_stages=2
    )
    record = selection.diagnostic_record()
    assert record["phase"] == phase
    assert all(not value for value in selection.effective_settings().values())
    assert record["configured.AFD_HCCL_EAGER_U2_STREAM_OVERLAP"] is True
    assert record["configured.AFD_HCCL_GRAPH_U2_HYBRID_DAG"] is not all_off
    assert record["effective.AFD_HCCL_EAGER_U2_STREAM_OVERLAP"] is False
    assert record["logical.compute_stream"] == "parent"
    assert record["physical.send_stream"] == record["physical.recv_stream"] == "parent"


@pytest.mark.parametrize(
    "role,changes,enabled,logical",
    (
        (
            "attention",
            {"graph_u2_hybrid_dag": False},
            ("graph_u2_compute_overlap", "graph_u2_attention_three_stream"),
            ("compute", "send", "recv"),
        ),
        (
            "attention",
            {"graph_u2_attention_three_stream": False},
            ("graph_u2_compute_overlap", "graph_u2_hybrid_dag"),
            ("compute", "parent", "parent"),
        ),
        (
            "ffn",
            {"graph_u2_ffn_cross_layer": False},
            ("graph_u2_compute_overlap", "graph_u2_ffn_recv_stream"),
            ("compute", "send", "recv"),
        ),
        (
            "ffn",
            {"graph_u2_ffn_recv_stream": False, "graph_u2_ffn_cross_layer": False},
            ("graph_u2_compute_overlap",),
            ("compute", "send", "parent"),
        ),
    ),
)
def test_graph_axes_preserve_independent_logical_choices(
    overlap, role, changes, enabled, logical
):
    selection = overlap.select_hccl_overlap(
        replace(overlap.HcclOverlapSettings(), **changes),
        role=role,
        phase="graph_capture",
        num_stages=2,
    )
    assert selection.effective_settings() == {
        name: field in enabled for name, field, _ in ENVIRONMENT_FIELDS
    }
    record = selection.diagnostic_record()
    assert (
        tuple(record[f"logical.{kind}_stream"] for kind in ("compute", "send", "recv"))
        == logical
    )
    assert record["physical.send_stream"] == record["physical.recv_stream"] == "parent"


@pytest.mark.parametrize(
    "role,phase,num_stages,compute_sync",
    (
        ("attention", "eager", 1, False),
        ("ffn", "eager", 1, True),
        ("ffn", "eager", 2, False),
        ("ffn", "graph_warmup", 1, False),
        ("ffn", "graph_capture", 1, False),
        ("ffn", "graph_replay", 1, False),
        ("ffn", "graph_capture", 2, False),
    ),
)
def test_diagnostics_keep_stage_logging_and_gate_live_serial_ffn_sync(
    overlap, role, phase, num_stages, compute_sync
):
    settings = overlap.HcclOverlapSettings(
        stage_diagnostics=True, ffn_compute_sync_diagnostics=True
    )
    selection = overlap.select_hccl_overlap(
        settings, role=role, phase=phase, num_stages=num_stages
    )
    record = selection.diagnostic_record()
    assert record["configured.AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS"] is True
    assert record["effective.AFD_HCCL_STAGE_DIAGNOSTICS"] is True
    assert record["effective.AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS"] is compute_sync


@pytest.mark.parametrize(
    "changes,message",
    (
        ({"role": "decoder"}, "role"),
        ({"role": None}, "role"),
        ({"role": True}, "role"),
        ({"phase": "NONE"}, "phase"),
        ({"phase": "capture"}, "phase"),
        ({"phase": None}, "phase"),
        ({"phase": False}, "phase"),
        ({"num_stages": 0}, "positive integer"),
        ({"num_stages": -1}, "positive integer"),
        ({"num_stages": True}, "positive integer"),
        ({"num_stages": 1.0}, "positive integer"),
        ({"settings": None}, "overlap settings"),
        ({"settings": {}}, "overlap settings"),
    ),
)
def test_selection_rejects_invalid_explicit_inputs(overlap, changes, message):
    arguments = {
        "settings": overlap.HcclOverlapSettings(),
        "role": "attention",
        "phase": "graph_capture",
        "num_stages": 2,
        **changes,
    }
    with pytest.raises(ValueError, match=message):
        overlap.select_hccl_overlap(**arguments)


def test_selection_is_frozen_and_exports_fresh_diagnostic_snapshots(overlap):
    selection = overlap.select_hccl_overlap(
        overlap.HcclOverlapSettings(),
        role="attention",
        phase="graph_capture",
        num_stages=2,
    )
    with pytest.raises(FrozenInstanceError):
        selection.num_stages = 1
    record = selection.diagnostic_record()
    record["configured.AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"] = False
    record["effective.AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"] = False
    record["physical.send_stream"] = "send"
    effective = selection.effective_settings()
    effective["AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"] = False
    assert (
        selection.diagnostic_record()["configured.AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"]
        is True
    )
    assert (
        selection.diagnostic_record()["effective.AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP"]
        is True
    )
    assert selection.physical_send_stream == "parent"


def test_one_fixed_settings_snapshot_projects_u1_and_u2_without_mutation(overlap):
    settings = overlap.HcclOverlapSettings()
    u1 = overlap.select_hccl_overlap(
        settings, role="attention", phase="graph_capture", num_stages=1
    )
    u2 = replace(u1, num_stages=2)
    assert u1.settings == u2.settings == settings
    assert u1.logical_compute_stream == "parent"
    assert u2.logical_compute_stream == "compute"
    assert u1.physical_send_stream == u2.physical_send_stream == "parent"
