"""Read installed stream/event source without importing torch or starting NPU."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SSH_CONFIG = "C:/Users/zq/.atomgitdevenv/.ssh/config"
SSH_HOST = "devenvc_rissvg.4ff038c993bf40219c03182f786a9def.atomgit.0"
REMOTE_PYTHON = "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0/bin/python"
REMOTE_SITE = Path(
    "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0/lib/python3.12/site-packages"
)
SOURCE_PATHS = (
    "torch_npu/npu/streams.py",
    "torch_npu/npu/__init__.py",
    "torch_npu/npu/utils.py",
    "torch_npu/utils/_tensor_methods.py",
)
API_NAMES = frozenset(
    ("__init__", "__new__", "record", "wait", "current_stream", "record_stream")
)


def read_installed_source() -> dict:
    sources = {}
    for relative in SOURCE_PATHS:
        path = REMOTE_SITE / relative
        if not path.is_file():
            sources[relative] = {"exists": False}
            continue
        data = path.read_bytes()
        source = data.decode("utf-8")
        tree = ast.parse(source)
        apis = []
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in API_NAMES
            ):
                start = min(
                    (item.lineno for item in node.decorator_list),
                    default=node.lineno,
                )
                apis.append(
                    {
                        "name": node.name,
                        "line_start": start,
                        "line_end": node.end_lineno,
                        "arguments": ast.unparse(node.args),
                        "source": "\n".join(
                            source.splitlines()[start - 1 : node.end_lineno]
                        ),
                    }
                )
        sources[relative] = {
            "exists": True,
            "absolute_path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "line_count": len(source.splitlines()),
            "api_excerpts": apis,
            "source": source,
        }
    return {
        "scope": "installed source read only; no torch import or device execution",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "torch_npu_version": importlib.metadata.version("torch-npu"),
        "torch_imported": "torch" in sys.modules,
        "torch_npu_imported": "torch_npu" in sys.modules,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-read", action="store_true")
    args = parser.parse_args()
    if args.remote_read:
        print(json.dumps(read_installed_source(), indent=2))
        return
    collector = Path(__file__).resolve()
    destination = collector.parent / "torch-npu-stream-api-20261110.json"
    if destination.exists():
        parser.error("keep the existing raw source evidence immutable")
    command = (
        "LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:${LD_LIBRARY_PATH} "
        + shlex.quote(REMOTE_PYTHON)
        + " -B - --remote-read"
    )
    result = subprocess.run(
        [
            "ssh",
            "-F",
            SSH_CONFIG,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            SSH_HOST,
            command,
        ],
        input=collector.read_bytes(),
        capture_output=True,
        timeout=60,
        check=True,
    )
    report = json.loads(result.stdout)
    report["collection_command"] = command
    report["collector_sha256"] = hashlib.sha256(collector.read_bytes()).hexdigest()
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(destination)
    print(
        json.dumps(
            {
                "torch_npu_version": report["torch_npu_version"],
                "torch_imported": report["torch_imported"],
                "torch_npu_imported": report["torch_npu_imported"],
                "sources": {
                    path: {
                        key: item[key]
                        for key in ("exists", "sha256", "line_count")
                        if key in item
                    }
                    for path, item in report["sources"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
