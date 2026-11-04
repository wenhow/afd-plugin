"""Collect C0 environment evidence to stdout without importing inference packages."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

COMMAND_TIMEOUT_SECONDS = 30
MAX_TEXT_BYTES = 1024 * 1024
RUNTIME_PACKAGES = (
    "vllm",
    "vllm-ascend",
    "vllm-afd-plugin",
    "torch",
    "torch-npu",
    "fastapi",
    "starlette",
    "pydantic",
    "transformers",
    "safetensors",
    "triton-ascend",
    "mooncake-transfer-engine",
    "mooncake-transfer-engine-npu",
)
ASCEND_VERSION_PATTERNS = (
    "driver/version.info",
    "driver/version.cfg",
    "ascend-toolkit/latest/*version*",
    "ascend-toolkit/latest/*-linux/ascend_toolkit_install.info",
    "cann/latest/*version*",
    "nnal/atb/latest/*version*",
)


def run_command(command: list[str]) -> dict:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
        return {
            "command": command,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "exit_code": None, "error": str(error)}


def read_text_file(path: Path) -> dict:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return {"path": str(path), "status": "too_large"}
        content = path.read_bytes()
        return {
            "path": str(path),
            "sha256": hashlib.sha256(content).hexdigest(),
            "text": content.decode("utf-8", errors="replace"),
        }
    except OSError as error:
        return {"path": str(path), "status": "unavailable", "error": str(error)}


def collect_package(name: str) -> dict:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {"status": "not_installed"}
    metadata_text = distribution.read_text("METADATA")
    result = {
        "version": distribution.version,
        "location": str(distribution.locate_file("")),
        "requires_dist": distribution.requires or [],
        "metadata_sha256": (
            hashlib.sha256(metadata_text.encode("utf-8")).hexdigest()
            if metadata_text is not None
            else None
        ),
    }
    # Keep provenance useful without copying authenticated repository URLs.
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            provenance = json.loads(direct_url)
            result["vcs_info"] = provenance.get("vcs_info")
            result["editable"] = provenance.get("dir_info", {}).get("editable")
        except json.JSONDecodeError as error:
            result["provenance_error"] = str(error)
    return result


def collect_source(path: str, requirement: str) -> dict:
    # Prevent git status from refreshing the checkout's index or fsmonitor.
    git = ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", path]
    return {
        "path": path,
        "head": run_command([*git, "rev-parse", "HEAD"]),
        "status": run_command([*git, "status", "--porcelain"]),
        "head_requirements": run_command([*git, "show", f"HEAD:{requirement}"]),
        "working_requirements": read_text_file(Path(path) / requirement),
    }


def collect_model(config_path: str) -> dict:
    path = Path(config_path)
    result = {"config": read_text_file(path)}
    index_path = path.parent / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index_bytes = index_path.read_bytes()
            index = json.loads(index_bytes)
            weight_map = index.get("weight_map", {})
            result["weight_index"] = {
                "path": str(index_path),
                "sha256": hashlib.sha256(index_bytes).hexdigest(),
                "metadata": index.get("metadata", {}),
                "tensor_count": len(weight_map),
                "shard_count": len(set(weight_map.values())),
            }
        except (OSError, ValueError) as error:
            result["weight_index"] = {"error": str(error)}
    else:
        result["weight_index"] = {"status": "not_found"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-ref", default="unspecified")
    parser.add_argument(
        "--model-config", help="Read config and index only, never weights"
    )
    parser.add_argument("--vllm-source")
    parser.add_argument("--ascend-source")
    parser.add_argument("--afd-source")
    parser.add_argument("--ascend-root", default="/usr/local/Ascend")
    args = parser.parse_args()

    ascend_root = Path(args.ascend_root)
    version_files = sorted(
        {
            path
            for pattern in ASCEND_VERSION_PATTERNS
            for path in ascend_root.glob(pattern)
            if path.is_file()
        }
    )
    # Metadata reads and pip check do not import torch/vLLM or run NPU operators.
    report = {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_ref_user_reported": args.image_ref,
        "host": {
            "hostname": platform.node(),
            "system": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "python_executable": sys.executable,
            "os_release": read_text_file(Path("/etc/os-release")),
        },
        "packages": {name: collect_package(name) for name in RUNTIME_PACKAGES},
        "installed_versions": sorted(
            (
                {"name": dist.metadata["Name"], "version": dist.version}
                for dist in metadata.distributions()
            ),
            key=lambda item: item["name"] or "",
        ),
        "pip_check": run_command([sys.executable, "-B", "-m", "pip", "check"]),
        "npu_smi": run_command(["npu-smi", "info"]),
        "ascend_version_files": [read_text_file(path) for path in version_files],
        "ascend_root_entries": sorted(str(path) for path in ascend_root.glob("*")),
        "device_nodes": sorted(str(path) for path in Path("/dev").glob("davinci*")),
        "ascend_paths": {
            key: os.environ.get(key)
            for key in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH")
        },
        "memory": read_text_file(Path("/proc/meminfo")),
        "shared_memory": (
            dict(
                zip(
                    ("total", "used", "free"),
                    shutil.disk_usage("/dev/shm"),
                    strict=True,
                )
            )
            if Path("/dev/shm").is_dir()
            else None
        ),
        "sources": {},
        "model": {"status": "not_provided"},
        "scope": "inventory only; imports, operators and model runtime not tested",
    }
    for name, path, requirement in (
        ("vllm", args.vllm_source, "requirements/common.txt"),
        ("vllm_ascend", args.ascend_source, "requirements.txt"),
        ("afd", args.afd_source, "pyproject.toml"),
    ):
        if path:
            report["sources"][name] = collect_source(path, requirement)
    if args.model_config:
        report["model"] = collect_model(args.model_config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
