# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Read-only materialized KV layout comparison, independent of runtime imports.

This is a diagnostic consumer, not a connector handshake or PD admission.
Absolute addresses are deliberately absent. Allocation-relative offsets and
capacities validate each snapshot; page-relative layout defines compatibility.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

LAYOUT_SCHEMA_VERSION = 1
MAX_LAYOUT_BYTES = 8 * 1024 * 1024
MAX_LAYOUT_RECORDS = 16_384
MAX_LAYOUT_DIMENSIONS = 8
MAX_LAYOUT_TEXT = 2_048
DTYPE_BYTES = (
    ("torch.bool", 1),
    ("torch.uint8", 1),
    ("torch.int8", 1),
    ("torch.float8_e4m3fn", 1),
    ("torch.float8_e5m2", 1),
    ("torch.float16", 2),
    ("torch.bfloat16", 2),
    ("torch.int16", 2),
    ("torch.float32", 4),
    ("torch.int32", 4),
    ("torch.float64", 8),
    ("torch.int64", 8),
)
JsonValue: TypeAlias = (
    str | int | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


def _integer(value: int, label: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value or len(value) > MAX_LAYOUT_TEXT:
        raise ValueError(f"{label} must be bounded nonempty text")


def _names(value: tuple[str, ...], label: str) -> None:
    if not isinstance(value, tuple) or not 1 <= len(value) <= MAX_LAYOUT_RECORDS:
        raise ValueError(f"{label} must be a nonempty tuple of unique names")
    for name in value:
        _text(name, label)
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must be a nonempty tuple of unique names")


@dataclass(frozen=True, slots=True)
class KVLayoutSpec:
    layer_name: str
    spec_type: str
    fields: tuple[tuple[str, str], ...]
    page_size_bytes: int
    tensor_count: int = 1

    def __post_init__(self) -> None:
        _text(self.layer_name, "spec layer")
        _text(self.spec_type, "spec type")
        _integer(self.page_size_bytes, "spec page size", minimum=1)
        _integer(self.tensor_count, "spec tensor count", minimum=1)
        if self.tensor_count > 2:
            raise ValueError("V4.1 layers expose one cache tensor or an index pair")
        if not isinstance(self.fields, tuple) or not self.fields:
            raise ValueError("spec fields must be a nonempty tuple")
        names = []
        for pair in self.fields:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("spec field must be a name/value pair")
            _text(pair[0], "spec field name")
            _text(pair[1], "spec field value")
            names.append(pair[0])
        if len(set(names)) != len(names):
            raise ValueError("spec field names must be unique")


@dataclass(frozen=True, slots=True)
class KVLayoutGroup:
    layer_names: tuple[str, ...]
    specs: tuple[KVLayoutSpec, ...]
    is_draft: bool
    host_resident: bool
    enable_transfer: bool
    role: str

    def __post_init__(self) -> None:
        _names(self.layer_names, "group layers")
        if not isinstance(self.specs, tuple) or any(
            not isinstance(spec, KVLayoutSpec) for spec in self.specs
        ):
            raise ValueError("group specs must be typed immutable records")
        if tuple(spec.layer_name for spec in self.specs) != self.layer_names:
            raise ValueError("group specs must follow the exact ordered layers")
        if any(
            type(flag) is not bool
            for flag in (self.is_draft, self.host_resident, self.enable_transfer)
        ):
            raise ValueError("group flags must be booleans")
        _text(self.role, "group role")


@dataclass(frozen=True, slots=True)
class KVLayoutPage:
    layers: tuple[str, ...]
    size_bytes: int
    offset_bytes: int
    layer_stride_bytes: int
    block_stride_bytes: int
    storage_alias: int
    storage_offset_bytes: int

    def __post_init__(self) -> None:
        _names(self.layers, "page layers")
        for name, value in (
            ("page size", self.size_bytes),
            ("block stride", self.block_stride_bytes),
        ):
            _integer(value, name, minimum=1)
        for name, value in (
            ("descriptor offset", self.offset_bytes),
            ("layer stride", self.layer_stride_bytes),
            ("storage alias", self.storage_alias),
            ("page storage offset", self.storage_offset_bytes),
        ):
            _integer(value, name)
        if self.layer_stride_bytes != 0:
            raise ValueError("V4.1 collector supports shared-page layer_stride=0 only")


@dataclass(frozen=True, slots=True)
class KVLayoutView:
    layer_name: str
    tensor_index: int
    page_index: int
    dtype: str
    element_bytes: int
    shape: tuple[int, ...]
    strides_bytes: tuple[int, ...]
    storage_alias: int
    storage_offset_bytes: int
    storage_size_bytes: int
    page_offset_bytes: int

    def __post_init__(self) -> None:
        _text(self.layer_name, "view layer")
        for name, value in (
            ("tensor index", self.tensor_index),
            ("page index", self.page_index),
            ("storage alias", self.storage_alias),
            ("view storage offset", self.storage_offset_bytes),
            ("view page offset", self.page_offset_bytes),
        ):
            _integer(value, name)
        _integer(self.storage_size_bytes, "storage size", minimum=1)
        _integer(self.element_bytes, "element size", minimum=1)
        if (self.dtype, self.element_bytes) not in DTYPE_BYTES:
            raise ValueError("view dtype/element size is unsupported or inconsistent")
        if (
            not isinstance(self.shape, tuple)
            or not isinstance(self.strides_bytes, tuple)
            or not 1 <= len(self.shape) <= MAX_LAYOUT_DIMENSIONS
            or len(self.shape) != len(self.strides_bytes)
        ):
            raise ValueError("view shape and byte strides must have equal bounded rank")
        for dimension, stride in zip(self.shape, self.strides_bytes, strict=True):
            _integer(dimension, "view dimension", minimum=1)
            _integer(stride, "view byte stride", minimum=int(dimension > 1))
            if stride % self.element_bytes:
                raise ValueError("byte strides must align to the view dtype")
        if self.storage_offset_bytes % self.element_bytes:
            raise ValueError("storage offset must align to the view dtype")
        span = self.element_bytes + sum(
            (dimension - 1) * stride
            for dimension, stride in zip(self.shape, self.strides_bytes, strict=True)
        )
        if self.storage_offset_bytes + span > self.storage_size_bytes:
            raise ValueError("materialized view exceeds its own storage capacity")


@dataclass(frozen=True, slots=True)
class KVLayoutSnapshot:
    num_blocks: int
    device_type: str
    cache_layout: str | None
    groups: tuple[KVLayoutGroup, ...]
    pages: tuple[KVLayoutPage, ...]
    views: tuple[KVLayoutView, ...]
    schema_version: int = LAYOUT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != LAYOUT_SCHEMA_VERSION
        ):
            raise ValueError("unsupported KV layout schema version")
        _integer(self.num_blocks, "num_blocks", minimum=1)
        if self.device_type not in ("cpu", "npu"):
            raise ValueError("materialized V4.1 views require CPU or NPU storage")
        if self.cache_layout is not None:
            _text(self.cache_layout, "cache layout")
        for records, record_type in (
            (self.groups, KVLayoutGroup),
            (self.pages, KVLayoutPage),
            (self.views, KVLayoutView),
        ):
            if (
                not isinstance(records, tuple)
                or not 1 <= len(records) <= MAX_LAYOUT_RECORDS
                or any(not isinstance(record, record_type) for record in records)
            ):
                raise ValueError("layout must contain bounded immutable typed records")
        grouped = tuple(name for group in self.groups for name in group.layer_names)
        specs = {spec.layer_name: spec for group in self.groups for spec in group.specs}
        placed = tuple(name for page in self.pages for name in page.layers)
        if len(set(grouped)) != len(grouped) or len(set(placed)) != len(placed):
            raise ValueError("each named layer must occur once in groups and pages")
        if set(grouped) != set(placed) or set(grouped) != {
            view.layer_name for view in self.views
        }:
            raise ValueError(
                "all logical layers require exactly their materialized views"
            )
        aliases: dict[int, int] = {}
        page_aliases: set[int] = set()
        for page_index, page in enumerate(self.pages):
            if page.storage_alias not in page_aliases:
                if page.storage_alias != len(page_aliases):
                    raise ValueError("storage aliases must use first-occurrence order")
                page_aliases.add(page.storage_alias)
            if page.size_bytes < (
                page.offset_bytes + self.num_blocks * page.block_stride_bytes
            ):
                raise ValueError("descriptor cannot hold its own num_blocks")
            page_views = tuple(v for v in self.views if v.page_index == page_index)
            if {view.layer_name for view in page_views} != set(page.layers):
                raise ValueError("every page must own exactly its named runtime views")
            for layer in page.layers:
                components = tuple(v for v in page_views if v.layer_name == layer)
                if tuple(v.tensor_index for v in components) != tuple(
                    range(specs[layer].tensor_count)
                ):
                    raise ValueError(
                        "layer tensor indices must be ordered and complete"
                    )
                first_offset = min(view.page_offset_bytes for view in components)
                if (
                    first_offset + specs[layer].page_size_bytes
                    > page.block_stride_bytes
                ):
                    raise ValueError("native spec exceeds its physical shared page")
            for view in page_views:
                if (
                    view.shape[0] != self.num_blocks
                    or view.strides_bytes[0] != page.block_stride_bytes
                    or view.storage_alias != page.storage_alias
                    or view.storage_offset_bytes
                    != page.storage_offset_bytes + view.page_offset_bytes
                ):
                    raise ValueError(
                        "runtime block count/stride/alias/page offset mismatch"
                    )
                previous = aliases.setdefault(
                    view.storage_alias, view.storage_size_bytes
                )
                if previous != view.storage_size_bytes:
                    raise ValueError("one storage alias must have one allocation size")
                if (
                    page.storage_offset_bytes + page.size_bytes - page.offset_bytes
                    > view.storage_size_bytes
                ):
                    raise ValueError("whole registered page exceeds runtime storage")
                row_span = view.element_bytes + sum(
                    (dimension - 1) * stride
                    for dimension, stride in zip(
                        view.shape[1:], view.strides_bytes[1:], strict=True
                    )
                )
                if view.page_offset_bytes + row_span > page.block_stride_bytes:
                    raise ValueError("runtime component crosses its physical page")
        if any(view.page_index >= len(self.pages) for view in self.views):
            raise ValueError("runtime view refers to an unknown physical page")
        # A descriptor is registered as one whole page pool. Two descriptors
        # can be disjoint regions of a shared slab or exact aliases, but a
        # partial overlap is outside this narrow shared-page contract.
        ranges = sorted(
            (
                page.storage_alias,
                page.storage_offset_bytes,
                page.storage_offset_bytes + self.num_blocks * page.block_stride_bytes,
            )
            for page in self.pages
        )
        for previous, current in zip(ranges, ranges[1:], strict=False):
            if (
                previous[0] == current[0]
                and current[1] < previous[2]
                and previous[1:] != current[1:]
            ):
                raise ValueError("distinct descriptor page pools partially overlap")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2) + "\n"


@dataclass(frozen=True, slots=True)
class KVLayoutComparison:
    differences: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.differences


def _page_aliases(snapshot: KVLayoutSnapshot) -> tuple[int, ...]:
    aliases: dict[tuple[int, int, int], int] = {}
    return tuple(
        aliases.setdefault(
            (
                page.storage_alias,
                page.storage_offset_bytes,
                snapshot.num_blocks * page.block_stride_bytes,
            ),
            index,
        )
        for index, page in enumerate(snapshot.pages)
    )


def compare_kv_layouts(
    source: KVLayoutSnapshot, destination: KVLayoutSnapshot
) -> KVLayoutComparison:
    """Compare all named registration semantics; capacities may differ."""
    differences = []
    for name, left, right in (
        ("device_type", source.device_type, destination.device_type),
        ("cache_layout", source.cache_layout, destination.cache_layout),
        ("groups", source.groups, destination.groups),
    ):
        if left != right:
            differences.append(name)
    # Disjoint descriptor pools register independently, even when their local
    # allocator puts them in one slab. Slab distances change with capacity.
    # Compare exact page aliases, not opaque allocation identity or distance.
    source_aliases = _page_aliases(source)
    destination_aliases = _page_aliases(destination)
    source_pages = tuple(
        (
            page.layers,
            page.offset_bytes,
            page.layer_stride_bytes,
            page.block_stride_bytes,
            source_aliases[index],
        )
        for index, page in enumerate(source.pages)
    )
    destination_pages = tuple(
        (
            page.layers,
            page.offset_bytes,
            page.layer_stride_bytes,
            page.block_stride_bytes,
            destination_aliases[index],
        )
        for index, page in enumerate(destination.pages)
    )
    if source_pages != destination_pages:
        differences.append("pages: ordered names/offsets/strides/aliases")
    source_views = tuple(
        (
            view.layer_name,
            view.tensor_index,
            view.page_index,
            view.dtype,
            view.element_bytes,
            view.shape[1:],
            view.strides_bytes,
            source_aliases[view.page_index],
            view.page_offset_bytes,
        )
        for view in source.views
    )
    destination_views = tuple(
        (
            view.layer_name,
            view.tensor_index,
            view.page_index,
            view.dtype,
            view.element_bytes,
            view.shape[1:],
            view.strides_bytes,
            destination_aliases[view.page_index],
            view.page_offset_bytes,
        )
        for view in destination.views
    )
    if source_views != destination_views:
        differences.append(
            "views: ordered names/dtypes/shapes/strides/aliases/page offsets"
        )
    return KVLayoutComparison(tuple(differences))


def validate_kv_transfer_blocks(
    source: KVLayoutSnapshot,
    destination: KVLayoutSnapshot,
    source_block_ids: tuple[tuple[int, ...], ...],
    destination_block_ids: tuple[tuple[int, ...], ...],
) -> None:
    """Validate one request's ordered transferable groups without truncation.

    IDs refer to each side's own global block-ID pool. This read-only consumer
    neither submits a transfer nor attests native remote metadata.
    """
    comparison = compare_kv_layouts(source, destination)
    if not comparison.compatible:
        raise ValueError(
            "incompatible KV layouts: " + "; ".join(comparison.differences)
        )
    group_count = sum(group.enable_transfer for group in source.groups)
    if (
        not isinstance(source_block_ids, tuple)
        or not isinstance(destination_block_ids, tuple)
        or len(source_block_ids) != group_count
        or len(destination_block_ids) != group_count
    ):
        raise ValueError("block IDs must cover every ordered transferable group")
    page_aliases = _page_aliases(source)
    layer_pools = {
        layer: page_aliases[index]
        for index, page in enumerate(source.pages)
        for layer in page.layers
    }
    transfer_groups = tuple(group for group in source.groups if group.enable_transfer)
    destinations: dict[tuple[int, int], int] = {}
    for group_index, (src, dst) in enumerate(
        zip(source_block_ids, destination_block_ids, strict=True)
    ):
        if (
            not isinstance(src, tuple)
            or not isinstance(dst, tuple)
            or len(src) != len(dst)
        ):
            raise ValueError(f"transfer group {group_index} block counts differ")
        for block_index, (src_id, dst_id) in enumerate(zip(src, dst, strict=True)):
            if (
                type(src_id) is not int
                or not 0 <= src_id < source.num_blocks
                or type(dst_id) is not int
                or not 0 <= dst_id < destination.num_blocks
            ):
                raise ValueError(
                    f"transfer group {group_index} block {block_index} is out of bounds"
                )
            # Native compressed transfer copies the entire physical page.
            # Different logical groups can name that same pool; one destination
            # block must never receive two different source pages. An identical
            # repeated pair is harmless and remains allowed.
            for pool in {
                layer_pools[layer] for layer in transfer_groups[group_index].layer_names
            }:
                previous = destinations.setdefault((pool, dst_id), src_id)
                if previous != src_id:
                    raise ValueError(
                        f"transfer group {group_index} block {block_index} "
                        "conflicts with a whole-page destination mapping"
                    )


def _record(value: JsonValue, names: tuple[str, ...]) -> dict[str, JsonValue]:
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("layout record has missing or unknown fields")
    return value


def _array(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list) or len(value) > MAX_LAYOUT_RECORDS:
        raise ValueError("layout arrays must be bounded lists")
    return value


def _unique_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def layout_from_json(text: str) -> KVLayoutSnapshot:
    """Read an untrusted bounded snapshot, rechecking every typed invariant."""
    if len(text.encode("utf-8")) > MAX_LAYOUT_BYTES:
        raise ValueError("layout snapshot exceeds the byte limit")
    try:
        raw = _record(
            json.loads(text, object_pairs_hook=_unique_keys),
            (
                "num_blocks",
                "device_type",
                "cache_layout",
                "groups",
                "pages",
                "views",
                "schema_version",
            ),
        )
        groups = []
        for value in _array(raw["groups"]):
            group = _record(
                value,
                (
                    "layer_names",
                    "specs",
                    "is_draft",
                    "host_resident",
                    "enable_transfer",
                    "role",
                ),
            )
            specs = []
            for spec_value in _array(group["specs"]):
                spec = _record(
                    spec_value,
                    (
                        "layer_name",
                        "spec_type",
                        "fields",
                        "page_size_bytes",
                        "tensor_count",
                    ),
                )
                specs.append(
                    KVLayoutSpec(
                        spec["layer_name"],
                        spec["spec_type"],
                        tuple(tuple(_array(pair)) for pair in _array(spec["fields"])),
                        spec["page_size_bytes"],
                        spec["tensor_count"],
                    )
                )
            groups.append(
                KVLayoutGroup(
                    tuple(_array(group["layer_names"])),
                    tuple(specs),
                    group["is_draft"],
                    group["host_resident"],
                    group["enable_transfer"],
                    group["role"],
                )
            )
        pages = []
        for value in _array(raw["pages"]):
            page = _record(
                value,
                (
                    "layers",
                    "size_bytes",
                    "offset_bytes",
                    "layer_stride_bytes",
                    "block_stride_bytes",
                    "storage_alias",
                    "storage_offset_bytes",
                ),
            )
            pages.append(
                KVLayoutPage(
                    tuple(_array(page["layers"])),
                    page["size_bytes"],
                    page["offset_bytes"],
                    page["layer_stride_bytes"],
                    page["block_stride_bytes"],
                    page["storage_alias"],
                    page["storage_offset_bytes"],
                )
            )
        views = []
        for value in _array(raw["views"]):
            view = _record(
                value,
                (
                    "layer_name",
                    "tensor_index",
                    "page_index",
                    "dtype",
                    "element_bytes",
                    "shape",
                    "strides_bytes",
                    "storage_alias",
                    "storage_offset_bytes",
                    "storage_size_bytes",
                    "page_offset_bytes",
                ),
            )
            views.append(
                KVLayoutView(
                    view["layer_name"],
                    view["tensor_index"],
                    view["page_index"],
                    view["dtype"],
                    view["element_bytes"],
                    tuple(_array(view["shape"])),
                    tuple(_array(view["strides_bytes"])),
                    view["storage_alias"],
                    view["storage_offset_bytes"],
                    view["storage_size_bytes"],
                    view["page_offset_bytes"],
                )
            )
        return KVLayoutSnapshot(
            raw["num_blocks"],
            raw["device_type"],
            raw["cache_layout"],
            tuple(groups),
            tuple(pages),
            tuple(views),
            raw["schema_version"],
        )
    except (TypeError, KeyError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("malformed KV layout snapshot") from exc


def load_kv_layout(path: Path) -> KVLayoutSnapshot:
    if path.stat().st_size > MAX_LAYOUT_BYTES:
        raise ValueError("layout snapshot exceeds the byte limit")
    return layout_from_json(path.read_text(encoding="utf-8"))


def _load_blocks(path: Path) -> tuple[tuple[int, ...], ...]:
    if path.stat().st_size > MAX_LAYOUT_BYTES:
        raise ValueError("block mapping exceeds the byte limit")
    try:
        return tuple(
            tuple(_array(group)) for group in _array(json.loads(path.read_text()))
        )
    except (TypeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("malformed block mapping") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-blocks", type=Path)
    parser.add_argument("--destination-blocks", type=Path)
    args = parser.parse_args(argv)
    if (args.source_blocks is None) != (args.destination_blocks is None):
        parser.error("both source and destination block mappings are required")
    try:
        source, destination = (
            load_kv_layout(args.source),
            load_kv_layout(args.destination),
        )
        comparison = compare_kv_layouts(source, destination)
        blocks_validated = False
        if args.source_blocks is not None:
            validate_kv_transfer_blocks(
                source,
                destination,
                _load_blocks(args.source_blocks),
                _load_blocks(args.destination_blocks),
            )
            blocks_validated = True
        print(
            json.dumps(
                {
                    "compatible": comparison.compatible,
                    "differences": comparison.differences,
                    "source_num_blocks": source.num_blocks,
                    "destination_num_blocks": destination.num_blocks,
                    "request_blocks_validated": blocks_validated,
                    "scope": (
                        "read-only layout comparison; native PD handshake not checked"
                    ),
                }
            )
        )
        return 0 if comparison.compatible else 1
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
