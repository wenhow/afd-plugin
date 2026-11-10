# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Native CPU slicing/dispatch contracts; no device graph behavior is simulated."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
)
from vllm.v1.worker.gpu.ubatch_utils import create_ubatch_slices

from afd_plugin.distributed.hccl_plan import HcclPlanCapabilities
from afd_plugin.v1.worker.npu.hccl_v41_graph import (
    HcclV41StageGeometry,
    capture_hccl_v41_geometry,
    dispatch_hccl_v41_graph,
    preview_hccl_v41_geometry,
    validate_hccl_v41_capture_coverage,
)
from tests.unit.fakes.batch_req_state import make_batch_req_state
from tests.unit.fakes.cudagraph_manager import make_native_graph_manager
from tests.unit.fakes.hccl_graph_stage import make_graph_stage_bank
from tests.unit.fakes.vllm_config import npu_parallel_config


@pytest.fixture
def full_descriptor():
    return BatchExecutionDescriptor(
        CUDAGraphMode.FULL, 8, 8, uniform_token_count=1, num_ubatches=2
    )


@pytest.fixture
def native_manager(full_descriptor):
    # Only the real native dispatch method is exercised. Preloaded CPU
    # candidates stand in for graph registration; no graph/device is created.
    return make_native_graph_manager(
        (replace(full_descriptor, num_ubatches=1), full_descriptor)
    )


@pytest.fixture
def parallel_config():
    return npu_parallel_config(
        use_ubatching=True,
        num_ubatches=2,
        dbo_decode_token_threshold=4,
        dbo_prefill_token_threshold=6,
    )


@pytest.mark.parametrize("query_len", [2, 3])
@pytest.mark.parametrize("requests", [1, 3])
@pytest.mark.parametrize("num_ubatches", [1, 2])
def test_target_width_preview_uses_actual_native_request_slicing(
    query_len, requests, num_ubatches
):
    descriptor = BatchExecutionDescriptor(
        CUDAGraphMode.FULL,
        3 * query_len,
        3,
        uniform_token_count=query_len,
        num_ubatches=num_ubatches,
    )
    state = make_batch_req_state((query_len,) * requests)
    expected = create_ubatch_slices(
        SimpleNamespace(
            num_scheduled_tokens=state.num_scheduled_tokens,
            num_tokens_after_padding=descriptor.num_tokens,
            num_reqs_after_padding=descriptor.num_reqs,
            num_reqs=requests,
        ),
        num_ubatches,
    )

    geometry = preview_hccl_v41_geometry(state, descriptor, target_query_len=query_len)

    assert geometry.stages == tuple(
        HcclV41StageGeometry(
            stage.request_slice.start,
            stage.request_slice.stop,
            stage.token_slice.start,
            stage.token_slice.stop,
        )
        for stage in expected
    )
    if query_len == 3 and requests == 3 and num_ubatches == 2:
        assert geometry.stages == (
            HcclV41StageGeometry(0, 2, 0, 4),
            HcclV41StageGeometry(1, 3, 4, 9),
        )


@pytest.mark.parametrize("query_len", [2, 3])
@pytest.mark.parametrize("num_ubatches", [1, 2])
def test_target_capture_matches_native_uniform_dummy_and_dispatch(
    query_len, num_ubatches, parallel_config
):
    descriptor = BatchExecutionDescriptor(
        CUDAGraphMode.FULL,
        3 * query_len,
        3,
        uniform_token_count=query_len,
        num_ubatches=num_ubatches,
    )
    state = make_batch_req_state((query_len,) * 3)
    manager = make_native_graph_manager((descriptor,))
    selected = dispatch_hccl_v41_graph(
        manager,
        state,
        parallel_config=parallel_config,
        allow_ubatching=num_ubatches == 2,
        target_query_len=query_len,
    )
    captured = capture_hccl_v41_geometry(descriptor, target_query_len=query_len)

    assert selected.candidate == captured
    assert selected.eager.descriptor.num_reqs == 3
    assert selected.eager.descriptor.num_tokens == 3 * query_len


@pytest.mark.parametrize("scheduled", [(1, 3), (2, 3), (1, 1)])
def test_variable_target_width_falls_back_before_native_preparation(
    scheduled, parallel_config
):
    selected = dispatch_hccl_v41_graph(
        None,
        make_batch_req_state(scheduled),
        parallel_config=parallel_config,
        allow_ubatching=True,
        target_query_len=3,
    )
    assert selected.candidate is selected.eager
    assert selected.eager.descriptor.num_tokens == sum(scheduled)
    assert selected.eager.descriptor.num_reqs == len(scheduled)


@pytest.mark.parametrize("query_len", [2, 3])
def test_capture_coverage_requires_descriptors_for_the_agreed_target(query_len):
    descriptor = BatchExecutionDescriptor(
        CUDAGraphMode.FULL,
        3 * query_len,
        3,
        uniform_token_count=query_len,
        num_ubatches=2,
    )
    caps = HcclPlanCapabilities(
        4,
        128,
        16,
        (0, 1),
        1,
        1,
        1,
        local_counts_version=1,
        graph_execution_version=1,
        target_query_len=query_len,
    )
    coverage = validate_hccl_v41_capture_coverage((descriptor,), caps)

    assert coverage.geometries == (
        capture_hccl_v41_geometry(descriptor, target_query_len=query_len),
    )
    with pytest.raises(ValueError, match="uniform target"):
        validate_hccl_v41_capture_coverage(
            (descriptor,), replace(caps, target_query_len=1)
        )
    with pytest.raises(ValueError, match="repeat the same"):
        validate_hccl_v41_capture_coverage(
            (descriptor, replace(descriptor, max_query_len=query_len)), caps
        )


@pytest.mark.parametrize("num_ubatches", [1, 2])
@pytest.mark.parametrize("actual_tokens", [1, 4, 5, 7, 8])
def test_preview_matches_actual_native_padded_slicing(
    full_descriptor, actual_tokens, num_ubatches
):
    state = make_batch_req_state((1,) * actual_tokens)
    descriptor = replace(full_descriptor, num_ubatches=num_ubatches)
    expected = create_ubatch_slices(
        SimpleNamespace(
            num_scheduled_tokens=state.num_scheduled_tokens,
            num_tokens_after_padding=8,
            num_reqs_after_padding=8,
            num_reqs=actual_tokens,
        ),
        num_ubatches,
    )

    geometry = preview_hccl_v41_geometry(state, descriptor)

    assert geometry.stages == tuple(
        HcclV41StageGeometry(
            stage.request_slice.start,
            stage.request_slice.stop,
            stage.token_slice.start,
            stage.token_slice.stop,
        )
        for stage in expected
    )


@pytest.mark.parametrize("scheduled", [(2, 3), (1, 2, 4), (3, 1, 1, 2)])
def test_eager_preview_preserves_actual_native_ordered_request_splits(scheduled):
    state = make_batch_req_state(scheduled, has_prefill=True)
    descriptor = BatchExecutionDescriptor(
        CUDAGraphMode.NONE, state.num_tokens, len(scheduled), num_ubatches=2
    )
    expected = create_ubatch_slices(
        SimpleNamespace(
            num_scheduled_tokens=state.num_scheduled_tokens,
            num_tokens_after_padding=state.num_tokens,
            num_reqs_after_padding=len(scheduled),
            num_reqs=len(scheduled),
        ),
        2,
    )

    geometry = preview_hccl_v41_geometry(state, descriptor)

    assert geometry.stages == tuple(
        HcclV41StageGeometry(
            stage.request_slice.start,
            stage.request_slice.stop,
            stage.token_slice.start,
            stage.token_slice.stop,
        )
        for stage in expected
    )


def test_runtime_and_capture_request_shapes_differ_at_padding_boundary(full_descriptor):
    runtime = preview_hccl_v41_geometry(make_batch_req_state((1,) * 4), full_descriptor)
    captured = capture_hccl_v41_geometry(full_descriptor)

    assert runtime.stages[-1] == HcclV41StageGeometry(3, 8, 4, 8)
    assert captured.stages[-1] == HcclV41StageGeometry(4, 8, 4, 8)
    assert runtime != captured


@pytest.mark.parametrize("num_ubatches", [1, 2])
@pytest.mark.parametrize("tokens", [3, 8])
def test_capture_preview_matches_real_uniform_dummy_vector(
    full_descriptor, tokens, num_ubatches
):
    descriptor = replace(
        full_descriptor, num_tokens=tokens, num_reqs=tokens, num_ubatches=num_ubatches
    )

    captured = capture_hccl_v41_geometry(descriptor)

    assert captured == preview_hccl_v41_geometry(
        make_batch_req_state((1,) * tokens), descriptor
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"cg_mode": CUDAGraphMode.NONE},
        {"uniform_token_count": 2},
        {"uniform_token_count": None},
        {"num_reqs": 4},
        {"num_active_loras": 1},
        {"max_query_len": 2},
        {"num_ubatches": 3},
        {"num_tokens": 0, "num_reqs": 0},
    ],
)
def test_capture_rejects_unsupported_descriptor(full_descriptor, changes):
    with pytest.raises(ValueError, match="FULL uniform target"):
        capture_hccl_v41_geometry(replace(full_descriptor, **changes))


@pytest.mark.parametrize(
    ("actual_tokens", "allow_ubatching", "expected_ubatches"),
    [(3, True, 1), (4, True, 2), (5, False, 1)],
)
def test_dispatch_uses_native_threshold_and_matching_graph_family(
    native_manager, parallel_config, actual_tokens, allow_ubatching, expected_ubatches
):
    selection = dispatch_hccl_v41_graph(
        native_manager,
        make_batch_req_state((1,) * actual_tokens),
        parallel_config=parallel_config,
        allow_ubatching=allow_ubatching,
    )

    assert selection.candidate.descriptor.cg_mode == CUDAGraphMode.FULL
    assert selection.candidate.descriptor.num_ubatches == expected_ubatches
    assert selection.candidate.descriptor.num_tokens == 8
    assert selection.eager.descriptor.num_tokens == actual_tokens
    assert selection.eager.descriptor.num_reqs == actual_tokens
    assert selection.eager.descriptor.num_ubatches == expected_ubatches


def test_u2_cache_miss_does_not_rename_u1_hit(native_manager, parallel_config):
    native_manager._candidates[(4, 0)] = [
        BatchExecutionDescriptor(CUDAGraphMode.FULL, 8, 8, uniform_token_count=1)
    ]

    selection = dispatch_hccl_v41_graph(
        native_manager,
        make_batch_req_state((1,) * 4),
        parallel_config=parallel_config,
        allow_ubatching=True,
    )

    assert selection.candidate == selection.eager
    assert selection.candidate.descriptor == BatchExecutionDescriptor(
        CUDAGraphMode.NONE, 4, 4, num_ubatches=2
    )


@pytest.mark.parametrize(
    ("scheduled", "prefill", "kwargs", "expected_ubatches", "expected_loras"),
    [
        ((1,) * 4, False, {"need_eager": True}, 2, 0),
        ((1,) * 4, True, {}, 1, 0),
        ((2, 2), False, {}, 1, 0),
        ((2, 2, 2), True, {}, 2, 0),
        ((1,) * 4, False, {"num_active_loras": 1}, 2, 1),
    ],
)
def test_unsupported_or_forced_steps_use_fresh_eager_layout(
    parallel_config, scheduled, prefill, kwargs, expected_ubatches, expected_loras
):
    selection = dispatch_hccl_v41_graph(
        None,
        make_batch_req_state(scheduled, has_prefill=prefill),
        parallel_config=parallel_config,
        allow_ubatching=True,
        **kwargs,
    )

    assert selection.candidate == selection.eager
    assert selection.eager.descriptor == BatchExecutionDescriptor(
        CUDAGraphMode.NONE,
        sum(scheduled),
        len(scheduled),
        num_active_loras=expected_loras,
        num_ubatches=expected_ubatches,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"uniform_token_count": None},
        {"uniform_token_count": 2},
        {"num_active_loras": 1},
        {"num_reqs": 4},
    ],
)
def test_unsupported_native_cache_candidate_falls_back_before_prepare(
    native_manager, parallel_config, full_descriptor, changes
):
    native_manager._candidates[(4, 0)] = [replace(full_descriptor, **changes)]

    selection = dispatch_hccl_v41_graph(
        native_manager,
        make_batch_req_state((1,) * 4),
        parallel_config=parallel_config,
        allow_ubatching=True,
    )

    assert selection.candidate == selection.eager
    assert selection.eager.descriptor.cg_mode == CUDAGraphMode.NONE
    assert selection.eager.descriptor.num_tokens == 4


@pytest.mark.parametrize("scheduled", [(), (0,), (-1, 2)])
def test_preview_rejects_invalid_native_counts(full_descriptor, scheduled):
    with pytest.raises(ValueError, match="nonempty native scheduled batch"):
        preview_hccl_v41_geometry(make_batch_req_state(scheduled), full_descriptor)


def test_eager_preview_rejects_retained_graph_padding(full_descriptor):
    with pytest.raises(ValueError, match="fresh actual batch dimensions"):
        preview_hccl_v41_geometry(
            make_batch_req_state((1,) * 4),
            replace(full_descriptor, cg_mode=CUDAGraphMode.NONE),
        )


def test_graph_dispatch_requires_manager_for_eligible_decode(parallel_config):
    with pytest.raises(ValueError, match="initialized native manager"):
        dispatch_hccl_v41_graph(
            None,
            make_batch_req_state((1,) * 4),
            parallel_config=parallel_config,
            allow_ubatching=True,
        )


@pytest.mark.parametrize(
    "changes", [{"data_parallel_size": 2}, {"data_parallel_rank": 1}]
)
def test_local_dispatch_rejects_unsynchronized_dp(parallel_config, changes):
    with pytest.raises(ValueError, match="native DP1 rank zero"):
        dispatch_hccl_v41_graph(
            None,
            make_batch_req_state((1,) * 4),
            parallel_config=npu_parallel_config(**{**vars(parallel_config), **changes}),
            allow_ubatching=True,
            need_eager=True,
        )


@pytest.mark.parametrize("native_tp_size", [1, 2, 4])
def test_capture_coverage_keeps_native_u1_u2_order_and_per_rank_storage_budget(
    full_descriptor, native_tp_size
):
    bank = make_graph_stage_bank(rows=(4, 4), hidden=16, native_tp_size=native_tp_size)
    descriptors = (full_descriptor, replace(full_descriptor, num_ubatches=1))

    report = validate_hccl_v41_capture_coverage(descriptors, bank.capabilities)

    assert tuple(item.descriptor for item in report.geometries) == descriptors
    assert report.geometries[0].stages == (
        HcclV41StageGeometry(0, 4, 0, 4),
        HcclV41StageGeometry(4, 8, 4, 8),
    )
    assert report.geometries[1].stages == (HcclV41StageGeometry(0, 8, 0, 8),)
    payloads = tuple(
        tensor
        for exchange in bank.exchanges
        for tensor in (
            exchange.buffers.hidden_states,
            exchange.buffers.hidden_states_fp32,
            exchange.buffers.input_ids,
            exchange.buffers.output,
        )
    )
    owned = payloads + bank.valid_counts + bank.row_indices + bank.padding_masks
    assert report.bank_bytes_per_rank[0] == sum(
        tensor.numel() * tensor.element_size() for tensor in owned
    )
    assert report.wire_bytes_per_layer_per_rank[0] == sum(
        tensor.numel() * tensor.element_size() for tensor in payloads
    )
    assert report.bank_bytes_per_rank == (1104, 1100)


@pytest.mark.parametrize("changes", [{"max_tokens": 7}, {"max_stages": 1}])
def test_capture_coverage_rejects_native_layout_over_negotiated_capacity(
    full_descriptor, changes
):
    caps = make_graph_stage_bank(rows=(4, 4)).capabilities
    with pytest.raises(ValueError, match="exceeds HCCL session capacity"):
        validate_hccl_v41_capture_coverage((full_descriptor,), replace(caps, **changes))


def test_capture_coverage_rejects_equivalent_native_query_one_aliases(full_descriptor):
    caps = make_graph_stage_bank(rows=(4, 4)).capabilities
    with pytest.raises(ValueError, match="repeat the same graph layout"):
        validate_hccl_v41_capture_coverage(
            (full_descriptor, replace(full_descriptor, max_query_len=1)), caps
        )


def test_capture_coverage_requires_graph_capability_before_native_factories(
    full_descriptor,
):
    caps = make_graph_stage_bank(rows=(4, 4)).capabilities
    with pytest.raises(ValueError, match="graph/counts v1 paired TP"):
        validate_hccl_v41_capture_coverage(
            (full_descriptor,), replace(caps, graph_execution_version=0)
        )


def test_empty_native_capture_list_has_no_bank_allocations():
    caps = make_graph_stage_bank().capabilities
    report = validate_hccl_v41_capture_coverage((), caps)
    assert report.geometries == ()
    assert report.bank_bytes_per_rank == ()
    assert report.wire_bytes_per_layer_per_rank == ()
