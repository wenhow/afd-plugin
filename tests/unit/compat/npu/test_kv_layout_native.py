# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# ruff: noqa: E402
"""Real native KV types and materialized CPU storage; no device behavior."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from afd_plugin.compat.npu.kv_layout import (
    compare_kv_layouts,
    layout_from_json,
    validate_kv_transfer_blocks,
)
from afd_plugin.compat.npu.kv_layout_native import collect_deepseek_v41_kv_layout
from tests.unit.fakes.kv_cache import make_native_kv_cache_case


@pytest.mark.parametrize("include_draft", [False, True])
def test_native_grouping_and_real_shared_views_survive_diagnostic_round_trip(
    include_draft,
):
    case = make_native_kv_cache_case(include_draft=include_draft)
    snapshot = collect_deepseek_v41_kv_layout(case.config, case.caches)
    assert layout_from_json(snapshot.to_json()) == snapshot
    assert snapshot.device_type == "cpu" and snapshot.num_blocks == 3
    assert snapshot.pages[0].block_stride_bytes == 24
    assert {view.storage_alias for view in snapshot.views} == {0}
    index = tuple(v for v in snapshot.views if v.layer_name == case.names.index)
    assert [(v.dtype, v.page_offset_bytes) for v in index] == [
        ("torch.int8", 16),
        ("torch.float16", 20),
    ]
    aliased = tuple(v for v in snapshot.views if v.layer_name != case.names.index)
    assert all(v.page_offset_bytes == 0 for v in aliased)
    assert tuple(spec.layer_name for g in snapshot.groups for spec in g.specs) == tuple(
        name for group in case.config.kv_cache_groups for name in group.layer_names
    )


def test_native_collector_outputs_feed_capacity_aware_request_validation():
    source = make_native_kv_cache_case(num_blocks=3)
    destination = make_native_kv_cache_case(num_blocks=7, allocation_offset=8)
    src = collect_deepseek_v41_kv_layout(source.config, source.caches)
    dst = collect_deepseek_v41_kv_layout(destination.config, destination.caches)
    assert compare_kv_layouts(src, dst).compatible
    validate_kv_transfer_blocks(src, dst, ((2,), (1,), (0,)), ((6,), (5,), (4,)))
    with pytest.raises(ValueError, match="group 2 block 1"):
        validate_kv_transfer_blocks(
            src, dst, ((2,), (1,), (0, 3)), ((6,), (5,), (4, 5))
        )


@pytest.mark.parametrize(
    "invalid", ["count", "dtype", "shape", "alias", "meta", "missing", "extra"]
)
def test_native_collector_rejects_inconsistent_materialized_components(invalid):
    case = make_native_kv_cache_case()
    if invalid == "count":
        case.caches[case.names.index] = case.caches[case.names.index][0]
    elif invalid == "dtype":
        case.caches[case.names.index] = (
            case.caches[case.names.index][0].view(torch.uint8),
            case.caches[case.names.index][1],
        )
    elif invalid == "shape":
        case.caches[case.names.main] = case.caches[case.names.main][:, :, :, :3]
    elif invalid == "alias":
        case.caches[case.names.state] = case.caches[case.names.state].clone()
    elif invalid == "meta":
        case.caches[case.names.main] = torch.empty((3, 2, 1, 4), device="meta")
    elif invalid == "missing":
        del case.caches[case.names.main]
    else:
        case.caches["unregistered"] = case.caches[case.names.main]
    with pytest.raises((ValueError, KeyError)):
        collect_deepseek_v41_kv_layout(case.config, case.caches)


@pytest.mark.parametrize("invalid", ["blocks", "stride", "host", "model", "spec-size"])
def test_native_collector_rejects_unsupported_or_out_of_bounds_descriptors(invalid):
    case = make_native_kv_cache_case()
    if invalid == "blocks":
        case.config.num_blocks += 1
    elif invalid == "stride":
        case.config.kv_cache_tensors[0].layer_stride = 24
    elif invalid == "host":
        case.config.kv_cache_tensors[0].host_resident = True
    else:
        group = case.config.kv_cache_groups[0]
        spec = group.kv_cache_spec.kv_cache_specs[case.names.main]
        group.kv_cache_spec.kv_cache_specs[case.names.main] = replace(
            spec,
            **(
                {"model_version": "deepseek_v4"}
                if invalid == "model"
                else {"page_size_padded": 25}
            ),
        )
    with pytest.raises(ValueError):
        collect_deepseek_v41_kv_layout(case.config, case.caches)


def test_native_group_transfer_flags_define_the_complete_request_map():
    case = make_native_kv_cache_case(include_draft=True)
    case.config.kv_cache_groups[-1].is_eagle_group = True
    case.config.kv_cache_groups[-1].enable_kv_transfer = False
    snapshot = collect_deepseek_v41_kv_layout(case.config, case.caches)
    assert snapshot.groups[-1].is_draft and not snapshot.groups[-1].enable_transfer
    validate_kv_transfer_blocks(snapshot, snapshot, ((), (), ()), ((), (), ()))
    with pytest.raises(ValueError, match="every ordered transferable group"):
        validate_kv_transfer_blocks(snapshot, snapshot, ((),) * 4, ((),) * 4)
