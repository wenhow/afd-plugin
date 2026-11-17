"""Record locked source contracts without importing or changing upstreams."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from harness import git, read_json

ASCEND_PATHS = (
    "vllm_ascend/ascend_config.py",
    "vllm_ascend/platform.py",
    "vllm_ascend/ascend_forward_context.py",
    "vllm_ascend/attention/mla_v1.py",
    "vllm_ascend/compilation/acl_graph.py",
    "vllm_ascend/worker/worker.py",
    "vllm_ascend/worker/v2/model_runner.py",
    "vllm_ascend/distributed/parallel_state.py",
    "vllm_ascend/models/deepseek_v41/model.py",
    "vllm_ascend/models/deepseek_v41/vl_model.py",
    "vllm_ascend/models/deepseek_v41/engram/model_state.py",
    "vllm_ascend/device/device_op.py",
)
VLLM_PATHS = (
    "vllm/config/model.py",
    "vllm/config/vllm.py",
    "vllm/model_executor/model_loader/base_loader.py",
    "vllm/model_executor/model_loader/utils.py",
    "vllm/distributed/parallel_state.py",
    "vllm/forward_context.py",
    "vllm/v1/worker/ubatching.py",
    "vllm/v1/worker/gpu/model_runner.py",
    "vllm/v1/worker/gpu/ubatch_utils.py",
    "vllm/v1/worker/gpu/input_batch.py",
)


def symbols(source: str) -> dict:
    lines = source.splitlines()
    result = {}

    def visit(nodes: list[ast.stmt], prefix: str = "") -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Include the complete annotated signature and source body hash.
                first_body_line = node.body[0].lineno
                signature_lines = lines[node.lineno - 1 : first_body_line - 1]
                while signature_lines and (
                    not signature_lines[-1].strip()
                    or signature_lines[-1].lstrip().startswith("#")
                ):
                    signature_lines.pop()
                signature = "\n".join(signature_lines)
                body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
                result[prefix + node.name] = {
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                    "signature": signature,
                    "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                }
            elif isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + ".")

    visit(ast.parse(source).body)
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    lock = read_json(root / ".agents/dsv41/sources.lock.json")
    destination = root / "openspec/changes/afd-v030-rfc-runtime/evidence"
    destination.mkdir(parents=True, exist_ok=True)
    report = {"source_lock": lock, "source_objects": [], "contracts": {}}
    for item in lock["sources"]:
        repo = Path(lock["repositories"][item["repo"]]["path"])
        object_name = item["commit"] + ":" + item["path"]
        git(repo, "cat-file", "-e", object_name)
        report["source_objects"].append(
            {"id": item["id"], "object": object_name, "verified": True}
        )

    for name, paths in (("vllm_ascend", ASCEND_PATHS), ("vllm", VLLM_PATHS)):
        repo = Path(lock["repositories"][name]["path"])
        target = lock["repositories"][name]["commit"]
        for path in paths:
            source = git(repo, "show", target + ":" + path).decode("utf-8")
            current = symbols(source)
            entry = {"commit": target, "symbols": current}
            if name == "vllm_ascend":
                previous = lock["previous_ascend_reference"]
                exists = git(repo, "ls-tree", previous, "--", path)
                old = (
                    symbols(git(repo, "show", previous + ":" + path).decode("utf-8"))
                    if exists
                    else {}
                )
                entry["previous_commit"] = previous
                entry["changes"] = {
                    symbol: {
                        "previous": old.get(symbol),
                        "target": current.get(symbol),
                    }
                    for symbol in sorted(old.keys() | current.keys())
                    if old.get(symbol) != current.get(symbol)
                }
            report["contracts"][path] = entry

    for name, path in (
        ("vllm", "requirements/common.txt"),
        ("vllm_ascend", "requirements.txt"),
    ):
        repo = lock["repositories"][name]
        source = git(Path(repo["path"]), "show", repo["commit"] + ":" + path)
        report[name + "_fastapi"] = [
            line for line in source.decode().splitlines() if "fastapi" in line.lower()
        ]
    output = destination / "source-contracts.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"source_objects": len(report["source_objects"]), "output": str(output)}
        )
    )


if __name__ == "__main__":
    main()
