#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Check the bounded A5 AFD candidate without importing runtime packages.

Reads package metadata, source bytes and checkpoint headers only. No device,
operator library, tensor payload, installation or service is touched. A passing
report permits a candidate launch; it does not qualify full-model execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TypeAlias

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)
MAX_HEADER_BYTES = 64 * 1024**2
HEADER_LENGTH_BYTES = 8
GIT_TIMEOUT_SECONDS = 10
REQUIRED_PACKAGES = (
    "torch",
    "torch-npu",
    "vllm",
    "vllm-ascend",
    "fastapi",
    "vllm-afd-plugin",
)
REQUIRED_PLUGINS = (
    ("afd", "afd_plugin:register_afd"),
    ("ascend_model", "vllm_ascend:register_model"),
    ("ascend_model_loader", "vllm_ascend:register_model_loader"),
    ("ascend_kv_connector", "vllm_ascend:register_connector"),
    ("ascend_service_profiling", "vllm_ascend:register_service_profiling"),
)


@dataclass(frozen=True)
class SourceIdentity:
    package: str
    path: str
    sha256: str


@dataclass(frozen=True)
class PreflightProfile:
    config_sha256: str
    index_sha256: str
    shard_count: int
    sources: tuple[SourceIdentity, ...]


# vLLM is pinned to ced6857a; the bounded Ascend closure is the successful A5
# checkout 4ec6dc77. The model/wrapper/Engram/runner files also match 7e37e29a.
# These are byte identities, not a claim that the two whole revisions are equal.
DEFAULT_PROFILE = PreflightProfile(
    "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
    "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8",
    48,
    (
        SourceIdentity(
            "vllm",
            "config/vllm.py",
            "956b812e5a719bcbfa3a3958801361b38b9c928c96b8c073311bbb96376dfb7a",
        ),
        SourceIdentity(
            "vllm",
            "v1/worker/gpu/model_runner.py",
            "174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce",
        ),
        SourceIdentity(
            "vllm",
            "entrypoints/cli/main.py",
            "5aac2864e27d6ccc09ec3a8f2630ebbfb74292b84f27d2ae1d4ea2a6f5ecd80c",
        ),
        SourceIdentity(
            "vllm",
            "v1/engine/core.py",
            "268bf40534a853812867ae8e80040f190983cb364b41182b55969aace20c93f5",
        ),
        SourceIdentity(
            "vllm",
            "v1/engine/utils.py",
            "d09f86b107b3c63b62f40a6590036be5b711778ea997fc5a6af6de94fdd2ccd3",
        ),
        SourceIdentity(
            "vllm",
            "v1/engine/core_client.py",
            "fd6ca0cb7499b3086eef2fc57674b7ace56ef1c648f09e92e144b255b79018f8",
        ),
        SourceIdentity(
            "vllm",
            "v1/executor/multiproc_executor.py",
            "3ed0149a8a8e1550c4f062af40e645d1b734e70f99981bc5bf15f0360507648a",
        ),
        SourceIdentity(
            "vllm",
            "entrypoints/cli/serve.py",
            "c182c88347f1eb3440a816986f1c2025a2ab2a13236cd68b0ad0146250d031ff",
        ),
        SourceIdentity(
            "vllm_ascend",
            "quantization/methods/w4a8/w4a8_mxfp4.py",
            "71240fac4f2eaa87300127d8f0e520c55596746ccbb52852357655499cb842f4",
        ),
        SourceIdentity(
            "vllm_ascend",
            "worker/v2/spec_decode/dspark/speculator.py",
            "0cce54c5b5218bb428d737158221a5953d59c34370f1cfbd7054d794c3111390",
        ),
        SourceIdentity(
            "vllm_ascend",
            "ascend_forward_context.py",
            "714270fa4a2217247a46b1fb59fc86b9b271c055f2ced7113f566c1fb4c4c9bb",
        ),
        SourceIdentity(
            "vllm_ascend",
            "ops/fused_moe/dataclass/fused_experts.py",
            "f154ad3ed1ffcf5ec4ab4a46f3d7df74af07cd3b7f44b9e71ed3407994d0f590",
        ),
        SourceIdentity(
            "vllm_ascend",
            "models/deepseek_v41/model.py",
            "989aec6db1e344dca76159dcbef0d5876874facedaa36b47e993146d5b8a2b9f",
        ),
        SourceIdentity(
            "vllm_ascend",
            "models/deepseek_v41/vl_model.py",
            "bb98a2e2a0e24c18aafe17229eb593937aaed2e37c1fd7cc2003c45ce872d282",
        ),
        SourceIdentity(
            "vllm_ascend",
            "models/deepseek_v41/engram/model_state.py",
            "677cb13bb86ad80a154e0ff9a43620aa3344e7e93170a05e27aba0cc1f10aa8b",
        ),
        SourceIdentity(
            "vllm_ascend",
            "worker/v2/model_runner.py",
            "a3471a9f181b45742b8f985c86137cdaafc2871d2ef6f96aa6a45236a242969b",
        ),
    ),
)


def git_identity(directory: Path) -> dict[str, JSONValue]:
    """Read identity without refreshing the index or changing any Git files."""
    results: dict[str, JSONValue] = {}
    for name, arguments in (
        ("root", ("rev-parse", "--show-toplevel")),
        ("head", ("rev-parse", "HEAD")),
        (
            "source_status",
            (
                "status",
                "--porcelain=v1",
                "--untracked-files=no",
                "--",
                "afd_plugin",
                "pyproject.toml",
                "tools/diagnostics",
            ),
        ),
    ):
        try:
            process = subprocess.run(
                ["git", "--no-optional-locks", "-C", str(directory), *arguments],
                capture_output=True,
                text=True,
                timeout=GIT_TIMEOUT_SECONDS,
                check=False,
            )
            results[name] = {
                "returncode": process.returncode,
                "stdout": process.stdout.strip(),
                "stderr": process.stderr.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as error:
            results[name] = {"error": str(error)}
    return results


def inspect_checkpoint(model: Path, profile: PreflightProfile) -> dict[str, JSONValue]:
    """Check fixed config/index and header structure without reading payloads."""
    actual = {
        name: hashlib.sha256((model / name).read_bytes()).hexdigest()
        for name in ("config.json", "model.safetensors.index.json")
    }
    config = json.loads((model / "config.json").read_bytes())
    weight_map = json.loads((model / "model.safetensors.index.json").read_bytes())[
        "weight_map"
    ]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index requires a nonempty weight_map")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise ValueError("checkpoint weight_map requires string names and shard paths")
    shards = sorted(set(weight_map.values()))
    if len(shards) != profile.shard_count:
        raise ValueError(
            f"expected {profile.shard_count} indexed shards, found {len(shards)}"
        )
    seen: set[str] = set()
    records: list[JSONValue] = []
    for name in shards:
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or ":" in name
        ):
            raise ValueError(f"checkpoint shard must be a relative path: {name!r}")
        shard = model / name
        with shard.open("rb") as stream:
            length_bytes = stream.read(HEADER_LENGTH_BYTES)
            if len(length_bytes) != HEADER_LENGTH_BYTES:
                raise ValueError(f"truncated safetensors header length: {name}")
            length = struct.unpack("<Q", length_bytes)[0]
            if not 0 < length <= MAX_HEADER_BYTES:
                raise ValueError(f"invalid safetensors header length: {name}")
            header_bytes = stream.read(length)
            if len(header_bytes) != length:
                raise ValueError(f"truncated safetensors header: {name}")
        header = json.loads(header_bytes)
        if not isinstance(header, dict):
            raise ValueError(f"checkpoint header must be a JSON object: {name}")
        payload_limit = shard.stat().st_size - HEADER_LENGTH_BYTES - length
        count = 0
        for tensor_name, tensor in header.items():
            if tensor_name == "__metadata__":
                continue
            if tensor_name in seen or weight_map.get(tensor_name) != name:
                raise ValueError(
                    f"duplicate or misindexed checkpoint tensor: {tensor_name}"
                )
            start, end = tensor["data_offsets"]
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start <= end <= payload_limit
            ):
                raise ValueError(
                    f"checkpoint tensor extends outside shard: {tensor_name}"
                )
            if not isinstance(tensor["dtype"], str) or any(
                type(size) is not int or size < 0 for size in tensor["shape"]
            ):
                raise ValueError(f"invalid checkpoint tensor descriptor: {tensor_name}")
            seen.add(tensor_name)
            count += 1
        records.append(
            {
                "name": name,
                "bytes": shard.stat().st_size,
                "header_bytes": HEADER_LENGTH_BYTES + length,
                "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
                "tensors": count,
            }
        )
    if seen != set(weight_map):
        raise ValueError("checkpoint headers do not contain every indexed tensor")
    matches = (
        actual["config.json"] == profile.config_sha256
        and actual["model.safetensors.index.json"] == profile.index_sha256
        and config["model_type"] == "deepseek_v41"
        and config["architectures"] == ["DeepseekV41ForCausalLM"]
    )
    return {
        "path": str(model),
        "sha256": actual,
        "expected_config_sha256": profile.config_sha256,
        "expected_index_sha256": profile.index_sha256,
        "shard_count": len(shards),
        "tensor_count": len(seen),
        "shards": records,
        "passed": matches,
        "payload_hashes_verified": False,
        "reference_header_hashes_verified": False,
        "scope": (
            "Fixed config/index identity and structural headers only; "
            "payload integrity and unmodified tensor values are not proved"
        ),
    }


def collect_preflight(
    model: Path,
    repo_root: Path,
    *,
    profile: PreflightProfile = DEFAULT_PROFILE,
    expected_python: tuple[int, int] = (3, 11),
) -> dict[str, JSONValue]:
    """Collect one report; injected profiles/version are for CPU fixtures only.

    The production CLI exposes no identity or version relaxation switches.
    """
    model, repo_root = model.resolve(), repo_root.resolve()
    errors: list[JSONValue] = []
    report: dict[str, JSONValue] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "A5 eager/U1 A4/F4 candidate preflight; no runtime acceptance",
        "python": {"executable": sys.executable, "version": sys.version},
        "repo_root": str(repo_root),
        "full_model_qualification": False,
        "device_probe_started": False,
        "tensor_payload_bytes_read": 0,
        "sources": {},
        "source_reference_commits": {
            "vllm": "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
            "vllm_ascend": "4ec6dc7722ec4f7f8005a86570238ac4377c5d43",
            "scope": "Bounded file identities only, not whole-revision equivalence",
        },
    }
    if sys.version_info[:2] != expected_python:
        errors.append(
            f"candidate requires Python {expected_python[0]}.{expected_python[1]}"
        )
    try:
        report["checkpoint"] = checkpoint = inspect_checkpoint(model, profile)
        if not checkpoint["passed"]:
            errors.append("checkpoint config/index identity or model family differs")
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        report["checkpoint"] = {
            "path": str(model),
            "passed": False,
            "error": str(error),
        }
        errors.append(f"checkpoint: {error}")
    versions: dict[str, JSONValue] = {}
    for name in REQUIRED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            errors.append(f"missing package metadata: {name}")
    report["packages"] = versions
    plugins = {
        entry.name: entry.value
        for entry in importlib.metadata.entry_points(group="vllm.general_plugins")
    }
    report["general_plugins"] = plugins
    for name, value in REQUIRED_PLUGINS:
        if plugins.get(name) != value:
            errors.append(f"plugin entry point differs: {name}")
    roots: dict[str, Path] = {}
    for name in ("afd_plugin", "vllm", "vllm_ascend"):
        # Only top-level lookup: never execute package __init__ or import a child.
        spec = importlib.util.find_spec(name)
        locations = list(spec.submodule_search_locations or ()) if spec else []
        if len(locations) != 1:
            errors.append(f"expected one installed package root: {name}")
        else:
            roots[name] = Path(locations[0]).resolve()
    report["package_roots"] = {name: str(path) for name, path in roots.items()}
    if roots.get("afd_plugin") != repo_root / "afd_plugin":
        errors.append("installed afd_plugin does not resolve to this checkout")
    sources: dict[str, JSONValue] = {}
    for source in profile.sources:
        key = f"{source.package}/{source.path}"
        try:
            path = roots[source.package] / source.path
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            sources[key] = {
                "path": str(path),
                "sha256": digest,
                "expected_sha256": source.sha256,
                "matches": digest == source.sha256,
            }
            if digest != source.sha256:
                errors.append(f"installed source differs: {key}")
        except (KeyError, OSError) as error:
            sources[key] = {"matches": False, "error": str(error)}
            errors.append(f"installed source unavailable: {key}")
    report["sources"] = sources
    identity = git_identity(repo_root)
    report["repo_git"] = identity
    head = identity["head"]
    status = identity["source_status"]
    git_root = identity["root"]
    if (
        head.get("returncode") != 0
        or not re.fullmatch("[0-9a-f]{40}", head.get("stdout", ""))
        or git_root.get("returncode") != 0
        or Path(git_root.get("stdout", "")).resolve() != repo_root
        or status.get("returncode") != 0
        or status.get("stdout")
    ):
        errors.append(
            "checkout Git identity unavailable or tracked source modifications present"
        )
    report["errors"] = errors
    report["passed"] = not errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = collect_preflight(args.model, args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"preflight: {args.output.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
