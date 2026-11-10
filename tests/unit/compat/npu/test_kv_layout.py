# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Public comparison/CLI semantics without runtime dependencies."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from afd_plugin.compat.npu.kv_layout import (
    compare_kv_layouts,
    layout_from_json,
    load_kv_layout,
    main,
    validate_kv_transfer_blocks,
)
from tests.unit.fakes.kv_layout import make_kv_layout, make_slab_kv_layout


def test_registration_layout_allows_different_local_capacity_and_allocation_offset():
    source = make_kv_layout(num_blocks=4)
    destination = make_kv_layout(num_blocks=8, allocation_offset=16)
    assert compare_kv_layouts(source, destination).compatible
    validate_kv_transfer_blocks(source, destination, ((3, 1), (2,)), ((7, 4), (6,)))


def test_disjoint_pages_in_one_slab_allow_different_capacity_and_slab_distance():
    source = make_slab_kv_layout(num_blocks=4)
    destination = make_slab_kv_layout(num_blocks=8, page_offsets=(16, 288))
    independent = make_kv_layout(num_blocks=8, page_count=2)
    assert compare_kv_layouts(source, destination).compatible
    assert compare_kv_layouts(source, independent).compatible


def test_cross_descriptor_partial_overlap_is_outside_the_shared_page_contract():
    with pytest.raises(ValueError, match="partially overlap"):
        make_slab_kv_layout(page_offsets=(0, 32))


@pytest.mark.parametrize(
    "source_ids,destination_ids",
    [(((0, 1), ()), ((2, 2), ())), (((0,), (1,)), ((2,), (2,)))],
)
def test_whole_page_mapping_rejects_conflicts_within_and_across_logical_groups(
    source_ids, destination_ids
):
    snapshot = make_kv_layout()
    with pytest.raises(ValueError, match="whole-page destination mapping"):
        validate_kv_transfer_blocks(snapshot, snapshot, source_ids, destination_ids)


def test_shared_page_logical_groups_allow_consistent_repeated_mappings():
    snapshot = make_kv_layout()
    validate_kv_transfer_blocks(snapshot, snapshot, ((0, 0), (0,)), ((2, 2), (2,)))


@pytest.mark.parametrize(
    "difference",
    [
        "group-order",
        "spec",
        "draft",
        "transfer",
        "dtype",
        "offset",
        "stride",
        "alias",
        "layout",
    ],
)
def test_semantic_difference_rejects_the_whole_transfer(difference):
    source = make_kv_layout(page_count=2)
    destination = source
    if difference == "group-order":
        destination = replace(source, groups=source.groups[::-1])
    elif difference in ("spec", "draft", "transfer"):
        group = source.groups[0]
        if difference == "spec":
            spec = replace(group.specs[0], fields=(("block_size", "int:8"),))
            group = replace(group, specs=(spec,))
        elif difference == "draft":
            group = replace(group, is_draft=True)
        else:
            group = replace(group, enable_transfer=False)
        destination = replace(source, groups=(group, *source.groups[1:]))
    elif difference in ("dtype", "offset", "stride"):
        view = source.views[1]
        if difference == "dtype":
            view = replace(view, dtype="torch.uint8")
        elif difference == "offset":
            view = replace(view, page_offset_bytes=14, storage_offset_bytes=14)
        else:
            view = replace(view, strides_bytes=(32, 3, 1))
        destination = replace(source, views=(source.views[0], view, *source.views[2:]))
    elif difference == "alias":
        pages = (source.pages[0], replace(source.pages[1], storage_alias=0))
        views = tuple(replace(view, storage_alias=0) for view in source.views)
        destination = replace(source, pages=pages, views=views)
    else:
        destination = replace(source, cache_layout="different")
    assert not compare_kv_layouts(source, destination).compatible
    with pytest.raises(ValueError, match="incompatible KV layouts"):
        validate_kv_transfer_blocks(source, destination, ((),) * 4, ((),) * 4)


@pytest.mark.parametrize(
    "invalid",
    [
        "blocks",
        "bool-blocks",
        "stride",
        "storage",
        "page",
        "missing",
        "index",
        "element-bool",
    ],
)
def test_invalid_runtime_snapshot_is_rejected_as_a_whole(invalid):
    snapshot = make_kv_layout()
    with pytest.raises(ValueError):
        if invalid == "blocks":
            replace(snapshot, num_blocks=3)
        elif invalid == "bool-blocks":
            replace(snapshot, num_blocks=True)
        elif invalid == "storage":
            replace(snapshot.views[-1], storage_size_bytes=24)
        elif invalid == "element-bool":
            replace(snapshot.views[1], element_bytes=True)
        elif invalid == "page":
            view = replace(snapshot.views[-1], page_index=1)
            replace(snapshot, views=(*snapshot.views[:-1], view))
        elif invalid == "missing":
            replace(snapshot, views=snapshot.views[1:])
        elif invalid == "index":
            view = replace(snapshot.views[-1], tensor_index=3)
            replace(snapshot, views=(*snapshot.views[:-1], view))
        else:
            view = replace(snapshot.views[-1], strides_bytes=(30, 2, 2))
            replace(snapshot, views=(*snapshot.views[:-1], view))


@pytest.mark.parametrize(
    "source_ids,destination_ids",
    [
        (((4,), ()), ((0,), ())),
        (((0,), ()), ((8,), ())),
        (((0,), ()), ((False,), ())),
        (((0,), (1, 2)), ((0,), (1,))),
        (((0,),), ((0,),)),
        (((0,), (), ()), ((0,), (), ())),
    ],
)
def test_each_request_block_is_checked_without_shortening(source_ids, destination_ids):
    with pytest.raises(ValueError):
        validate_kv_transfer_blocks(
            make_kv_layout(), make_kv_layout(num_blocks=8), source_ids, destination_ids
        )


def test_snapshot_round_trip_rechecks_all_materialized_semantics(tmp_path):
    snapshot = make_kv_layout(num_blocks=3, allocation_offset=8)
    path = tmp_path / "snapshot.json"
    path.write_text(snapshot.to_json(), encoding="utf-8")
    assert load_kv_layout(path) == snapshot
    with pytest.raises(FrozenInstanceError):
        snapshot.num_blocks = 99


def test_relative_placement_of_pages_sharing_an_allocation_is_semantic():
    source = make_kv_layout(page_count=2)
    shared_views = tuple(
        replace(view, storage_alias=0, storage_size_bytes=256) for view in source.views
    )
    source = replace(
        source,
        pages=(source.pages[0], replace(source.pages[1], storage_alias=0)),
        views=shared_views,
    )
    destination = replace(
        source,
        pages=(source.pages[0], replace(source.pages[1], storage_offset_bytes=128)),
        views=tuple(
            replace(view, storage_offset_bytes=view.storage_offset_bytes + 128)
            if view.page_index == 1
            else view
            for view in source.views
        ),
    )
    assert not compare_kv_layouts(source, destination).compatible


@pytest.mark.parametrize(
    "invalid",
    ["unknown", "version", "boolean", "missing", "tuple", "syntax", "duplicate"],
)
def test_serialized_snapshot_rejects_malformed_or_inconsistent_fields(invalid):
    payload = json.loads(make_kv_layout().to_json())
    if invalid == "unknown":
        payload["address"] = 123
    elif invalid == "version":
        payload["schema_version"] = 2
    elif invalid == "boolean":
        payload["views"][-1]["shape"][0] = True
    elif invalid == "missing":
        payload["views"].pop()
    elif invalid == "tuple":
        payload["groups"][0]["layer_names"] = "layer"
    elif invalid == "duplicate":
        payload["groups"].append(payload["groups"][0])
    text = "{" if invalid == "syntax" else json.dumps(payload)
    with pytest.raises(ValueError):
        layout_from_json(text)


def test_serialized_snapshot_rejects_duplicate_json_fields():
    text = (
        make_kv_layout()
        .to_json()
        .replace('"num_blocks": 4', '"num_blocks": 4, "num_blocks": 4')
    )
    with pytest.raises(ValueError, match="duplicate JSON field"):
        layout_from_json(text)


@pytest.mark.parametrize("request_blocks", [False, True])
def test_public_cli_consumes_snapshots_and_optional_request_maps(
    tmp_path, capsys, request_blocks
):
    source, destination = tmp_path / "source.json", tmp_path / "destination.json"
    source.write_text(make_kv_layout().to_json())
    destination.write_text(make_kv_layout(num_blocks=8).to_json())
    argv = [str(source), str(destination)]
    if request_blocks:
        src_ids, dst_ids = tmp_path / "src-ids.json", tmp_path / "dst-ids.json"
        src_ids.write_text("[[3,0],[1]]")
        dst_ids.write_text("[[7,2],[5]]")
        argv += ["--source-blocks", str(src_ids), "--destination-blocks", str(dst_ids)]
    assert main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["compatible"] and result["request_blocks_validated"] == request_blocks
    assert (result["source_num_blocks"], result["destination_num_blocks"]) == (4, 8)


def test_public_cli_reports_registration_mismatch(tmp_path, capsys):
    source, destination = tmp_path / "source.json", tmp_path / "destination.json"
    source.write_text(make_kv_layout().to_json())
    destination.write_text(
        replace(make_kv_layout(), cache_layout="different").to_json()
    )
    assert main([str(source), str(destination)]) == 1
    assert json.loads(capsys.readouterr().out)["differences"] == ["cache_layout"]


def test_public_cli_rejects_a_late_invalid_block(tmp_path, capsys):
    path, ids = tmp_path / "layout.json", tmp_path / "blocks.json"
    path.write_text(make_kv_layout().to_json())
    ids.write_text("[[0,1],[0,4]]")
    with pytest.raises(SystemExit) as error:
        main(
            [
                str(path),
                str(path),
                "--source-blocks",
                str(ids),
                "--destination-blocks",
                str(ids),
            ]
        )
    assert error.value.code == 2
    assert "group 1 block 1" in capsys.readouterr().err
