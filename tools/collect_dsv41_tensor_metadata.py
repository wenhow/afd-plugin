#!/usr/bin/env python3
"""Collect V4.1 checkpoint metadata without reading tensor payloads.

Uses only the standard library. Safetensors format reference:
https://github.com/safetensors/safetensors/blob/main/README.md#format
"""

import argparse
import gzip
import hashlib
import json
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

HEADER_PREFIX_BYTES = 8
MAX_HEADER_BYTES = 100_000_000
TARGET_ARCHITECTURE = "DeepseekV41ForCausalLM"


def unique_json_object(pairs: list[tuple]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON key: {name}")
        result[name] = value
    return result


def read_json(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=unique_json_object)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data, hashlib.sha256(raw).hexdigest()


def read_shard(path: Path) -> dict:
    file_bytes = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(HEADER_PREFIX_BYTES)
        if len(prefix) != HEADER_PREFIX_BYTES:
            raise ValueError("Truncated safetensors length prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes > MAX_HEADER_BYTES:
            raise ValueError(f"Header exceeds {MAX_HEADER_BYTES} bytes")
        if header_bytes > file_bytes - HEADER_PREFIX_BYTES:
            raise ValueError("Header extends beyond shard file")
        raw = stream.read(header_bytes)
        if len(raw) != header_bytes or not raw.startswith(b"{"):
            raise ValueError("Truncated or invalid safetensors JSON header")
    header = json.loads(raw, object_pairs_hook=unique_json_object)
    if not isinstance(header, dict):
        raise ValueError("Expected safetensors header object")
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    data_bytes = file_bytes - HEADER_PREFIX_BYTES - header_bytes
    spans = []
    for name, tensor in tensors.items():
        if not isinstance(tensor, dict):
            raise ValueError(f"Invalid tensor metadata: {name}")
        dtype = tensor.get("dtype")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"Invalid dtype: {name}")
        if not isinstance(shape, list) or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise ValueError(f"Invalid shape: {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= data_bytes
        ):
            raise ValueError(f"Invalid data offsets: {name}")
        spans.append((offsets[0], offsets[1], name))
    covered = 0
    for start, end, name in sorted(spans):
        if start != covered:
            raise ValueError(f"Overlapping or unindexed tensor bytes: {name}")
        covered = end
    if covered != data_bytes:
        raise ValueError("Unindexed trailing tensor bytes")
    return {
        "file": path.name,
        "file_bytes": file_bytes,
        "header_bytes": header_bytes,
        "header_sha256": hashlib.sha256(prefix + raw).hexdigest(),
        "tensor_bytes": data_bytes,
        "tensors": tensors,
    }


def collect_metadata(model_dir: Path) -> dict:
    config, config_sha256 = read_json(model_dir / "config.json")
    if config.get("architectures") != [TARGET_ARCHITECTURE]:
        raise ValueError(f"Expected architectures=[{TARGET_ARCHITECTURE!r}]")
    index, index_sha256 = read_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Missing or empty index weight_map")
    if any(
        not isinstance(name, str) or not isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise ValueError("Index weight_map must map tensor names to shard filenames")
    report = {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir.resolve()),
        "config": config,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "index_metadata": index.get("metadata", {}),
        "weight_map": weight_map,
        "shards": [],
        "errors": [],
        "tensor_count": 0,
        "total_tensor_bytes": 0,
        "complete": False,
        "payload_read": False,
    }
    for filename in sorted(set(weight_map.values())):
        shard_path = PurePosixPath(filename)
        if (
            shard_path.is_absolute()
            or ".." in shard_path.parts
            or "\\" in filename
            or ":" in filename
            or not filename.endswith(".safetensors")
        ):
            report["errors"].append(f"Unsafe or unsupported shard path: {filename}")
            continue
        try:
            shard = read_shard(model_dir / filename)
        except (OSError, ValueError, UnicodeError) as error:
            report["errors"].append(f"{filename}: {error}")
            continue
        shard["file"] = filename
        expected = {name for name, owner in weight_map.items() if owner == filename}
        actual = set(shard["tensors"])
        for name in sorted(expected - actual):
            report["errors"].append(f"{filename}: indexed tensor missing: {name}")
        for name in sorted(actual - expected):
            report["errors"].append(
                f"{filename}: unindexed or wrong-owner tensor: {name}"
            )
        report["shards"].append(shard)
        report["tensor_count"] += len(actual)
        report["total_tensor_bytes"] += shard["tensor_bytes"]
    metadata = report["index_metadata"]
    if (
        isinstance(metadata, dict)
        and "total_size" in metadata
        and metadata["total_size"] != report["total_tensor_bytes"]
    ):
        report["errors"].append("Index total_size differs from collected tensor bytes")
    report["complete"] = not report["errors"]
    return report


def write_report(report: dict, output: Path, model_dir: Path) -> None:
    if output.resolve().is_relative_to(model_dir.resolve()):
        raise ValueError("Output must be outside the checkpoint directory")
    opener = gzip.open if output.suffix == ".gz" else open
    with opener(output, "xt", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="V4.1 checkpoint directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON or JSON.gz outside model"
    )
    args = parser.parse_args()
    try:
        report = collect_metadata(args.model)
        write_report(report, args.output, args.model)
    except (OSError, ValueError, UnicodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(
        f"complete={report['complete']} tensors={report['tensor_count']} "
        f"bytes={report['total_tensor_bytes']} output={args.output}"
    )
    for error in report["errors"]:
        print(error, file=sys.stderr)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
