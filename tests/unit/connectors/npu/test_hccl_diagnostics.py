# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pure structured diagnostics; requested-sync callbacks do not execute NPU IO."""

import logging
import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit.fakes.hccl_diagnostics import make_diagnostics_case


@pytest.fixture
def api():
    root = Path(__file__).resolve().parents[4] / "afd_plugin/connectors/npu"
    return SimpleNamespace(
        **runpy.run_path(str(root / "hccl_overlap.py")),
        **{
            name: value
            for name, value in runpy.run_path(str(root / "hccl_diagnostics.py")).items()
            if name.startswith("HcclStage")
        },
    )


@pytest.fixture
def case(api):
    selection = api.select_hccl_overlap(
        api.HcclOverlapSettings(
            stage_diagnostics=True, ffn_compute_sync_diagnostics=True
        ),
        role="ffn",
        phase="eager",
        num_stages=1,
    )
    return make_diagnostics_case(api, selection, world_rank=3, role_rank=1)


def test_default_diagnostics_emit_no_configuration_progress_or_sync(api):
    selection = api.select_hccl_overlap(
        api.HcclOverlapSettings(), role="ffn", phase="eager", num_stages=1
    )
    case = make_diagnostics_case(api, selection)
    run = case.diagnostics.bind_execution(
        layer_indices=(0, 2), actual_compute_stream="parent"
    )
    run.record("compute_begin", 0, 0)
    run.synchronize_compute(0, 0)
    assert case.records == case.events == []


def test_configuration_distinguishes_eligible_eager_u2_from_actual_serial_execution(
    api,
):
    selection = api.select_hccl_overlap(
        api.HcclOverlapSettings(stage_diagnostics=True),
        role="ffn",
        phase="eager",
        num_stages=2,
    )
    case = make_diagnostics_case(api, selection)
    case.diagnostics.bind_execution(
        layer_indices=(0, 1), actual_compute_stream="parent"
    )
    (record,) = case.records
    assert record["configured.AFD_HCCL_EAGER_U2_STREAM_OVERLAP"] is True
    assert record["effective.AFD_HCCL_EAGER_U2_STREAM_OVERLAP"] is True
    assert record["logical.recv_stream"] == record["physical.recv_stream"] == "recv"
    assert record["actual.mode"] == "serial"
    assert record["actual.recv_stream"] == record["actual.send_stream"] == "parent"
    assert record["actual.compute_pipeline_enabled"] is False


@pytest.mark.parametrize("phase", ["graph_warmup", "graph_capture", "graph_replay"])
@pytest.mark.parametrize("actual", ["parent", "compute"])
def test_graph_records_actual_streams_without_claiming_physical_receive_overlap(
    api, phase, actual
):
    selection = api.select_hccl_overlap(
        api.HcclOverlapSettings(stage_diagnostics=True),
        role="ffn",
        phase=phase,
        num_stages=2,
    )
    case = make_diagnostics_case(api, selection)
    case.diagnostics.bind_execution(
        layer_indices=(0, 1),
        actual_compute_stream=actual,
        is_warmup=phase == "graph_warmup",
    )
    (record,) = case.records
    assert record["logical.recv_stream"] == "recv"
    assert record["physical.recv_stream"] == record["actual.recv_stream"] == "parent"
    assert record["actual.compute_pipeline_enabled"] is (actual == "compute")
    assert record["actual.ffn_compute_sync"] is False


def test_progress_records_only_first_last_layers_and_host_enqueue_boundaries(case):
    run = case.diagnostics.bind_execution(
        layer_indices=(0, 2, 7), actual_compute_stream="parent"
    )
    for layer in (0, 2, 7):
        run.record("recv_begin", layer, 0)
        run.record("recv_returned", layer, 0)
    assert [(record["event"], record["layer"]) for record in case.records[1:]] == [
        (event, layer) for layer in (0, 7) for event in ("recv_begin", "recv_returned")
    ]
    assert all(record["boundary"] == "host_enqueue" for record in case.records)
    assert all(
        record["world_rank"] == 3 and record["role_rank"] == 1
        for record in case.records
    )


def test_explicit_live_u1_sync_records_real_requested_callback_before_complete(case):
    run = case.diagnostics.bind_execution(
        layer_indices=(0, 2), actual_compute_stream="parent"
    )
    run.synchronize_compute(0, 0)
    assert case.events == [
        "configuration",
        "compute_sync_begin",
        "device_sync",
        "compute_sync_complete",
    ]
    assert [record["boundary"] for record in case.records[1:]] == [
        "explicit_synchronize"
    ] * 2


def test_sync_failure_has_no_false_complete_record(case):
    case.recorder.fail_sync = True
    run = case.diagnostics.bind_execution(
        layer_indices=(0,), actual_compute_stream="parent"
    )
    with pytest.raises(RuntimeError, match="requested diagnostic device sync failed"):
        run.synchronize_compute(0, 0)
    assert case.events == ["configuration", "compute_sync_begin", "device_sync"]


@pytest.mark.parametrize("profile,warmup", [(True, False), (False, True), (True, True)])
def test_profile_and_warmup_suppress_requested_compute_sync(case, profile, warmup):
    run = case.diagnostics.bind_execution(
        layer_indices=(0,),
        actual_compute_stream="parent",
        in_profile_run=profile,
        is_warmup=warmup,
    )
    run.synchronize_compute(0, 0)
    assert case.events == ["configuration"]
    assert case.records[0]["actual.ffn_compute_sync"] is False


@pytest.mark.parametrize(
    "role,phase,stages",
    [
        ("attention", "eager", 1),
        ("ffn", "eager", 2),
        ("ffn", "graph_warmup", 1),
        ("ffn", "graph_capture", 1),
        ("ffn", "graph_replay", 2),
    ],
)
def test_ineligible_role_stage_or_graph_phase_never_requests_sync(
    api, role, phase, stages
):
    selection = api.select_hccl_overlap(
        api.HcclOverlapSettings(ffn_compute_sync_diagnostics=True),
        role=role,
        phase=phase,
        num_stages=stages,
    )
    case = make_diagnostics_case(api, selection)
    run = case.diagnostics.bind_execution(
        layer_indices=(0,), actual_compute_stream="parent"
    )
    run.synchronize_compute(0, 0)
    assert case.events == ["configuration"]


def test_active_sync_requires_an_explicit_callback_before_progress(case):
    diagnostics = replace(case.diagnostics, synchronize=None)
    with pytest.raises(ValueError, match="require a synchronizer"):
        diagnostics.bind_execution(layer_indices=(0,), actual_compute_stream="parent")
    assert case.records == []


@pytest.mark.parametrize(
    "changes", [{"world_rank": -1}, {"role_rank": True}, {"role_rank": 1.0}]
)
def test_invalid_rank_rejected_without_records(case, changes):
    with pytest.raises(ValueError, match="nonnegative integers"):
        replace(case.diagnostics, **changes)
    assert case.records == []


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"layer_indices": ()}, "layers"),
        ({"layer_indices": (1, 0)}, "layers"),
        ({"layer_indices": (True,)}, "layers"),
        ({"actual_compute_stream": "recv"}, "compute stream"),
        ({"actual_compute_stream": "compute"}, "serial parent"),
        ({"in_profile_run": 1}, "markers"),
        ({"is_warmup": None}, "markers"),
    ],
)
def test_invalid_actual_execution_rejected_before_configuration(case, changes, message):
    with pytest.raises(ValueError, match=message):
        case.diagnostics.bind_execution(
            **{"layer_indices": (0,), "actual_compute_stream": "parent", **changes}
        )
    assert case.records == []


@pytest.mark.parametrize(
    "event,layer,stage",
    [
        ("done", 0, 0),
        ("recv_begin", 1, 0),
        ("recv_begin", 0, 1),
        ("recv_begin", True, 0),
    ],
)
def test_invalid_stage_record_is_rejected_without_partial_progress(
    case, event, layer, stage
):
    run = case.diagnostics.bind_execution(
        layer_indices=(0,), actual_compute_stream="parent"
    )
    with pytest.raises(ValueError, match="event|coordinates"):
        run.record(event, layer, stage)
    assert case.events == ["configuration"]


def test_default_sink_uses_standard_logging_without_global_logger(case, caplog):
    diagnostics = replace(case.diagnostics, emit=None)
    with caplog.at_level(logging.WARNING):
        diagnostics.bind_execution(layer_indices=(0,), actual_compute_stream="parent")
    assert "AFD HCCL stage diagnostic" in caplog.text
    assert "configuration" in caplog.text
