#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Prepare a source-only A5 DSpark bundle from the reviewed immutable baseline.

This command reads Git objects and local files only. It does not import the
runtime, install packages, contact a host, or initialize a device. Running the
prepared bundle does not require this original evidence archive to be present.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

BASELINE = (
    "openspec/changes/afd-hccl-full-parity/evidence/"
    "a3-text-u2-config-admission-20261116/engine-args-pd-green-001/source.tar.gz"
)
BASELINE_SHA256 = "af0c18647dc8f25bb42a901ae40a74bb0f796f75ceef23dc6259ca50964cebdd"
DEFAULT_OUTPUT = "reports/dsv41-a5-dspark-functional"
LAUNCHER = "tests/npu/dsv41_dspark_a5_validation.py"
ENTRY = "tests/npu/hccl_native_attention_pd_dspark_decode_probe.py"
BASELINE_SOURCE_COUNT = 447
PRODUCTION_COUNT = 146
BASELINE_HELPER_COUNT = 29
NATIVE_PYTHON_COUNT = 190
GIT_TIMEOUT_SECONDS = 60
A5_RUNTIME = "tests/npu/dsv41_dspark_a5_runtime.py"
A5_NATIVE_SOURCES = (
    "vllm_ascend/device/hardware_profile.py",
    "vllm_ascend/device/device_config.py",
    "vllm_ascend/device/hardware.py",
    "vllm_ascend/device/device_op.py",
    "vllm_ascend/ops/packaged_attention.py",
    "vllm_ascend/models/deepseek_v41/mixed_quant_attention.py",
    "vllm_ascend/models/deepseek_v41/cache_config.py",
    "vllm_ascend/platform.py",
    "vllm_ascend/utils.py",
)
KERNEL_REFERENCES = (
    "csrc/attention/kv_compress_epilog_v2/README.md",
    "csrc/attention/kv_compress_epilog_v2/op_kernel/kv_compress_epilog_v2_quant_fp8.h",
    "csrc/attention/kv_compress_epilog_v2/op_kernel/kv_compress_epilog_v2_common.h",
    "csrc/attention/kv_compress_epilog_v2/op_kernel/kv_compress_epilog_v2_kernel.h",
    "csrc/attention/kv_compress_epilog_v2/op_kernel/kv_compress_epilog_v2_layout_4d.h",
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def archive_files(path: Path) -> dict[str, bytes]:
    files = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or ":" in member.name
                or member.name in files
            ):
                raise ValueError(f"unsafe or duplicate archive member: {member.name}")
            files[member.name] = archive.extractfile(member).read()
    return files


def local_helper_closure(files: dict[str, bytes]) -> tuple[str, ...]:
    pending, selected = [ENTRY], set()
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected.add(name)
        tree = ast.parse(files[name], filename=name)
        for node in ast.walk(tree):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports = [node.module]
            for module in imports:
                module = module.removeprefix("tests.npu.").split(".", 1)[0]
                if module.startswith(("hccl_", "dsv41_", "afd_")) and (
                    module.endswith("_probe") or module == "dsv41_dspark_a5_runtime"
                ):
                    dependency = f"tests/npu/{module}.py"
                    if dependency not in files:
                        raise ValueError(f"missing local helper: {dependency}")
                    pending.append(dependency)
    return tuple(sorted(selected))


def a5_helper_successors(original: dict[str, bytes], root: Path) -> tuple[dict, dict]:
    """Apply reviewable fixture-only successors to exact immutable donor bytes.

    These are controlled component helpers, not upstream or production patches.
    Exact match counts prevent silently adapting a different donor revision.
    """
    files, changes = dict(original), {}

    def replace(name: str, before: str, after: str) -> None:
        source = files[name].decode("utf-8")
        if source.count(before) != 1:
            raise ValueError(f"A5 donor replacement is not unique: {name}: {before}")
        files[name] = source.replace(before, after).encode("utf-8")

    spec = "tests/npu/dsv41_dspark_speculator_probe.py"
    replace(
        spec,
        "from pathlib import Path\n",
        "from pathlib import Path\n\n"
        "from dsv41_dspark_a5_runtime import NATIVE_ATTENTION_HEADS\n",
    )
    replace(
        spec,
        '"num_attention_heads": 1,',
        '"num_attention_heads": NATIVE_ATTENTION_HEADS,',
    )
    replace(
        spec,
        "        assert hf.hc_mult == HC_MULT and hf.num_attention_heads == 1\n",
        "        assert hf.hc_mult == HC_MULT\n"
        "        assert hf.num_attention_heads == NATIVE_ATTENTION_HEADS\n",
    )

    decode = ENTRY
    replace(
        decode,
        '        report["custom_ops_enabled"] = enable_custom_op()\n'
        '        assert report["custom_ops_enabled"]\n',
        "        from dsv41_dspark_a5_runtime import native_a5_bootstrap\n\n"
        '        report["native_a5_bootstrap"] = native_a5_bootstrap()\n'
        '        report["custom_ops_enabled"] = False\n',
    )
    replace(decode, "            enable_custom_op,\n", "")
    replace(
        decode,
        '"""Map a real BF16 SWA row into the complete physical descriptor span."""',
        '"""Map a native typed SWA row into its complete physical descriptor span."""',
    )
    replace(
        decode,
        "    from hccl_native_attention_pd_dspark_probe import "
        "ROUTER_DELTA, ROUTER_SCALE\n",
        "    from dsv41_dspark_a5_runtime import (\n"
        "        encode_reference_row,\n"
        "        read_reference_row,\n"
        "    )\n"
        "    from hccl_native_attention_pd_dspark_probe import "
        "ROUTER_DELTA, ROUTER_SCALE\n",
    )
    replace(
        decode,
        "            previous.append(\n"
        "                torch.frombuffer(\n"
        "                    bytearray(expected[pool][begin : begin + size]),\n"
        "                    dtype=torch.bfloat16,\n"
        "                ).clone()\n"
        "            )",
        "            previous.append(\n"
        "                read_reference_row(\n"
        "                    cache, bytes(expected[pool][begin : begin + size])\n"
        "                )\n"
        "            )",
    )
    replace(
        decode,
        "            raw = row.contiguous().view(torch.uint8).numpy().tobytes()",
        "            raw = encode_reference_row(cache, row)",
    )

    prep = "tests/npu/hccl_native_attention_pd_dspark_probe.py"
    replace(
        prep,
        "            assert cache.ndim == 4 and cache.shape[-1] == HIDDEN_SIZE\n",
        "            if cache.dtype == torch.uint8:\n"
        "                from dsv41_dspark_a5_runtime import "
        "controlled_prefix_attention\n\n"
        "                return controlled_prefix_attention(\n"
        "                    cache, slots, positions, hidden_states\n"
        "                )\n"
        "            assert cache.ndim == 4 and cache.shape[-1] == HIDDEN_SIZE\n",
    )

    run = "tests/npu/hccl_native_attention_pd_dspark_run_probe.py"
    replace(
        run,
        "    assert owner.execute_model(step) is None\n",
        "    from hccl_native_attention_pd_dspark_decode_probe import "
        "target_reference\n\n"
        "    layout, pages = physical_pages(bindings)\n"
        "    before_target = [bytes(page.cpu().numpy()) for page in pages]\n"
        "    oracle_hidden, oracle_aux, oracle_pages, oracle_layers = "
        "target_reference(\n"
        "        bindings, pages, before_target, blocks, PROMPT[:-1],\n"
        "        tuple(range(len(PROMPT) - 1)), step.new_block_ids_to_zero\n"
        "    )\n"
        '    report["independent_layer_oracles"] = oracle_layers\n'
        '    report["before_target"] = snapshot_pages(\n'
        '        pages, args.output_dir, "producer-before-target"\n'
        "    )\n"
        "    assert owner.execute_model(step) is None\n"
        "    assert [bytes(page.cpu().numpy()) for page in pages] == oracle_pages\n"
        '    report["after_target"] = snapshot_pages(\n'
        '        pages, args.output_dir, "producer-after-target"\n'
        "    )\n"
        '    report["whole_pool_target_write_oracle_accepted"] = True\n',
    )
    replace(
        run,
        "    assert len(state.aux_hidden_states) == 2\n",
        "    assert len(state.aux_hidden_states) == 2\n"
        "    assert torch.equal(state.hidden_states.cpu(), oracle_hidden)\n"
        "    assert all(torch.equal(actual.cpu(), expected) for actual, expected in\n"
        "               zip(state.aux_hidden_states, oracle_aux, strict=True))\n",
    )
    files[A5_RUNTIME] = (root / A5_RUNTIME).read_bytes()
    for name in (spec, decode, prep, run):
        changes[name] = {
            "donor_sha256": sha256(original[name]),
            "successor_sha256": sha256(files[name]),
            "unified_diff": "".join(
                difflib.unified_diff(
                    original[name].decode().splitlines(keepends=True),
                    files[name].decode().splitlines(keepends=True),
                    fromfile="donor/" + name,
                    tofile="A5/" + name,
                )
            ),
        }
    return files, changes


def read_locked_sources(repo: dict, names: list[str]) -> dict[str, bytes]:
    result = subprocess.run(
        ["git", "-C", repo["path"], "cat-file", "--batch"],
        input=("\n".join(f"{repo['commit']}:{name}" for name in names) + "\n").encode(),
        capture_output=True,
        check=True,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    stream, sources = io.BytesIO(result.stdout), {}
    for name in names:
        header = stream.readline().decode("ascii").split()
        if len(header) != 3 or header[1] != "blob":
            raise ValueError(f"missing locked Git blob: {name}")
        sources[name] = stream.read(int(header[2]))
        if stream.read(1) != b"\n":
            raise ValueError("invalid Git blob framing")
    if stream.read():
        raise ValueError("unexpected data after Git blob batch")
    return sources


def native_identity(files: dict[str, bytes], repositories: dict) -> dict:
    expected = {}
    for name, raw in files.items():
        if name.endswith(".json"):
            payload = json.loads(raw)
            for source, digest in payload.get("source_expected", {}).items():
                if source in expected and expected[source] != digest:
                    raise ValueError(f"conflicting native source pin: {source}")
                expected[source] = digest
    native = {
        name: value
        for name, value in expected.items()
        if name.startswith(("vllm/", "vllm_ascend/"))
    }
    if len(native) != NATIVE_PYTHON_COUNT:
        raise ValueError(f"unexpected native Python pin count: {len(native)}")
    omitted = sorted(set(expected) - set(native))
    if omitted != ["torch_npu/profiler/profiler.py"]:
        raise ValueError(f"unexpected non-Git native pins: {omitted}")
    extra_names = sorted(set(A5_NATIVE_SOURCES) - set(native))
    extra = read_locked_sources(repositories["vllm_ascend"], extra_names)
    native.update({name: sha256(raw) for name, raw in extra.items()})
    for package in ("vllm", "vllm_ascend"):
        repo = repositories[package]
        names = sorted(name for name in native if name.startswith(package + "/"))
        for name, raw in read_locked_sources(repo, names).items():
            if sha256(raw) != native[name]:
                raise ValueError(f"locked Git source differs: {name}")
    references = read_locked_sources(
        repositories["vllm_ascend"], list(KERNEL_REFERENCES)
    )
    return {
        "scope": "native Python Git source identity; no A3 binary expectations",
        "commits": {
            name: repositories[name]["commit"] for name in ("vllm", "vllm_ascend")
        },
        "source_expected": dict(sorted(native.items())),
        "baseline_native_python_pin_count": NATIVE_PYTHON_COUNT,
        "additional_a5_source_pins": extra_names,
        "packed_oracle_kernel_source_sha256": {
            name: sha256(raw) for name, raw in references.items()
        },
        "excluded_a3_actual_source_pins": omitted,
        "binary_identity_policy": "record the existing A5 installation only",
    }


def write_archive(path: Path, files: dict[str, bytes]) -> None:
    with (
        path.open("wb") as output,
        gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        for name, raw in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            member.mode = 0o644
            member.mtime = 0
            tar.addfile(member, io.BytesIO(raw))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    baseline = args.baseline or root / BASELINE
    output = args.output_dir or root / DEFAULT_OUTPUT
    if output.exists():
        parser.error("--output-dir must not exist; prepared bundles are immutable")
    if sha256(baseline.read_bytes()) != BASELINE_SHA256:
        raise ValueError("baseline archive differs from the reviewed source identity")
    original = archive_files(baseline)
    manifest = json.loads(original.pop("manifest.json"))
    if len(original) != BASELINE_SOURCE_COUNT or set(original) != set(manifest):
        raise ValueError("baseline source count or manifest membership differs")
    for name, raw in original.items():
        if sha256(raw) != manifest[name]:
            raise ValueError(f"baseline source hash mismatch: {name}")
    repositories = json.loads(
        (root / ".agents/dsv41/sources.lock.json").read_text("utf-8")
    )["repositories"]
    identity = native_identity(original, repositories)
    original_helpers = local_helper_closure(original)
    if len(original_helpers) != BASELINE_HELPER_COUNT:
        raise ValueError(f"unexpected donor helper closure: {len(original_helpers)}")
    successors, changes = a5_helper_successors(original, root)
    helpers = local_helper_closure(successors)
    selected = {
        name: raw for name, raw in original.items() if name.startswith("afd_plugin/")
    }
    if len(selected) != PRODUCTION_COUNT:
        raise ValueError("baseline production source count differs")
    production = {name: sha256(raw) for name, raw in sorted(selected.items())}
    selected.update({name: successors[name] for name in helpers})
    selected["tests/__init__.py"] = original["tests/__init__.py"]
    selected[LAUNCHER] = (root / LAUNCHER).read_bytes()
    selected["native-source-identity.json"] = json_bytes(identity)
    selected["a5-helper-delta.json"] = json_bytes(
        {
            "production_sources_changed": [],
            "donor_helpers": len(original_helpers),
            "unchanged_donor_helpers": len(original_helpers) - len(changes),
            "changes": changes,
            "new_runtime_helper": {
                "path": A5_RUNTIME,
                "sha256": sha256(selected[A5_RUNTIME]),
            },
            "geometry": {
                "donor_query_heads": 1,
                "a5_query_heads": 64,
                "hidden_size": 512,
                "head_size": 512,
                "q_lora_rank": 64,
            },
            "swa_abi": {
                "data_bytes": 512,
                "bf16_scales": 16,
                "row_bytes": 544,
                "native_writer": "custom::kv_compress_epilog_v2",
            },
        }
    )
    for name, raw in selected.items():
        if not name.endswith((".py", ".json")):
            raise ValueError(f"unexpected non-source member: {name}")
        if name.endswith(".py"):
            compile(raw, name, "exec")
    source_manifest = {
        "baseline_archive_sha256": BASELINE_SHA256,
        "source_count": len(selected),
        "production_count": PRODUCTION_COUNT,
        "helper_count": len(helpers),
        "sources": {name: sha256(raw) for name, raw in sorted(selected.items())},
    }
    packaged = {**selected, "bundle-manifest.json": json_bytes(source_manifest)}
    output.mkdir(parents=True)
    archive = output / "runtime-source.tar.gz"
    write_archive(archive, packaged)
    reread = archive_files(archive)
    if reread != packaged:
        raise ValueError("prepared archive round trip differs")
    lock = {
        "baseline_archive_sha256": BASELINE_SHA256,
        "runtime_source_sha256": sha256(archive.read_bytes()),
        "launcher_sha256": sha256(selected[LAUNCHER]),
        "production_sources": production,
    }
    (output / "bundle-lock.json").write_bytes(json_bytes(lock))
    preparation = {
        "status": "source_prepared_not_device_validated",
        "baseline_archive": BASELINE,
        "baseline_archive_sha256": BASELINE_SHA256,
        "baseline_sources_verified": len(original),
        "runtime_source_sha256": lock["runtime_source_sha256"],
        "launcher_sha256": lock["launcher_sha256"],
        "builder_sha256": sha256(Path(__file__).read_bytes()),
        "source_count": len(selected),
        "python_sources_compiled": sum(name.endswith(".py") for name in selected),
        "production_count": PRODUCTION_COUNT,
        "production_exact_baseline": True,
        "native_python_git_pins_verified": len(identity["source_expected"]),
        "helpers": {name: sha256(successors[name]) for name in helpers},
        "unchanged_donor_helpers": len(original_helpers) - len(changes),
        "helper_deltas": {
            name: {key: value for key, value in detail.items() if key != "unified_diff"}
            for name, detail in changes.items()
        },
        "contains_binary_or_vendor_payload": False,
        "uses_live_production_sources": False,
        "npu_validation": "pending_on_A5",
    }
    (output / "preparation.json").write_bytes(json_bytes(preparation))
    print(json.dumps(preparation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
