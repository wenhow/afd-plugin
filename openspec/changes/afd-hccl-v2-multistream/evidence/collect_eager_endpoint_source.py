"""Read installed eager P2P APIs and Work declarations without importing torch."""

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
    "torch/distributed/distributed_c10d.py",
    "torch/_C/_distributed_c10d.pyi",
    "torch_npu/distributed/distributed_c10d.py",
    "torch_npu/distributed/__init__.py",
)
DECLARATIONS = frozenset(("isend", "irecv", "send", "recv", "Work"))


def read_sources() -> dict:
    sources = {}
    for relative in SOURCE_PATHS:
        path = REMOTE_SITE / relative
        if not path.is_file():
            sources[relative] = {"exists": False}
            continue
        data = path.read_bytes()
        source = data.decode("utf-8")
        lines = source.splitlines()
        declarations = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (
                node.name in DECLARATIONS
            ):
                start = min(
                    (item.lineno for item in node.decorator_list), default=node.lineno
                )
                declarations.append(
                    {
                        "name": node.name,
                        "line_start": start,
                        "line_end": node.end_lineno,
                        "source": "\n".join(lines[start - 1 : node.end_lineno]),
                    }
                )
        sources[relative] = {
            "exists": True,
            "absolute_path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "declarations": declarations,
            "p2p_reference_lines": [
                {"line": number, "text": line}
                for number, line in enumerate(lines, 1)
                if any(word in line for word in ("isend", "irecv", ".send", ".recv"))
            ],
        }
    return {
        "scope": "installed source declarations only; no torch import or NPU job",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "torch-npu")
        },
        "torch_imported": "torch" in sys.modules,
        "torch_npu_imported": "torch_npu" in sys.modules,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-read", action="store_true")
    args = parser.parse_args()
    if args.remote_read:
        print(json.dumps(read_sources(), indent=2))
        return
    collector = Path(__file__).resolve()
    destination = collector.parent / "eager-endpoint-source-20261110.json"
    if destination.exists():
        parser.error("retain the existing source evidence")
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


if __name__ == "__main__":
    main()
