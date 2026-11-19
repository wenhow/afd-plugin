#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Parse qualification CLIs and construct real EngineArgs, without loading weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from pathlib import Path
from unittest.mock import patch

import run_dsv41_full_checkpoint as launch


def check_matrix(model: Path, output: Path) -> int:
    # Explicit diagnostic entry only; no model construction or device execution.
    import torch
    import torch_npu  # noqa: F401
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins

    before = torch.npu.is_initialized()
    if before:
        raise RuntimeError("require an uninitialized NPU runtime")
    records = []
    source = model / "config.json"
    original = source.read_bytes()
    with patch.object(
        torch.npu,
        "_lazy_init",
        side_effect=RuntimeError("CPU configuration check forbids NPU initialization"),
    ):
        current_platform.pre_register_and_update()
        load_general_plugins()
        from vllm.engine.arg_utils import EngineArgs
        from vllm.entrypoints.launchers.cli_args import make_arg_parser
        from vllm.utils.argparse_utils import FlexibleArgumentParser

        import afd_plugin.compat.patches.config_validation  # noqa: F401
        from afd_plugin.compat.npu import apply_afd_ascend_config_patch_if_needed
        from afd_plugin.compat.npu.feature_validation import (
            fail_if_unsupported_npu_afd_features,
        )
        from afd_plugin.entrypoints.npu_ffn import validate_ffn_configuration

        apply_afd_ascend_config_patch_if_needed()
        parsers = {
            "attention": make_arg_parser(FlexibleArgumentParser()),
            "ffn": EngineArgs.add_cli_args(FlexibleArgumentParser()),
        }
        scenarios = (
            [("native", 0, None)]
            + [
                (mode, steps, pd_role)
                for mode in launch.MODES
                for steps in (0, 1, 2)
                for pd_role in (None, "consumer")
            ]
            + [("eager-u1", steps, "producer") for steps in (0, 1, 2)]
        )
        for mode, steps, pd_role in scenarios:
            plan = launch.make_plan(
                model,
                Path("/unused-vendor"),
                Path(__file__).resolve().parents[2],
                mode="eager-u1" if mode == "native" else mode,
                spec_tokens=steps,
                pd_role=pd_role,
                native_control=mode == "native",
            )
            for role, settings in plan["roles"].items():
                record = {
                    "mode": mode,
                    "steps": steps,
                    "pd_role": pd_role,
                    "role": role,
                }
                try:
                    with patch.dict(os.environ, settings["environment"]):
                        argv = settings["argv"]
                        tail = (
                            argv[argv.index(str(model)) :]
                            if role == "attention"
                            else argv[argv.index("--model") :]
                        )
                        parsed = parsers[role].parse_args(tail)
                        if role == "attention":
                            parsed.model = parsed.model_tag
                        config = EngineArgs.from_cli_args(parsed).create_engine_config()
                        # Native EngineCore handshakes repeat this validation
                        # after AFD restores the real connector backend.
                        config.__post_init__()
                        if role == "ffn":
                            validate_ffn_configuration(config)
                        elif mode != "native":
                            fail_if_unsupported_npu_afd_features(config)
                        assert config.parallel_config.use_ubatching is mode.endswith(
                            "u2"
                        )
                        assert config.compilation_config.cudagraph_mode.name == (
                            "FULL_DECODE_ONLY" if mode.startswith("graph") else "NONE"
                        )
                        assert config.model_config.enforce_eager is (
                            mode == "native" or mode.startswith("eager")
                        )
                        assert config.model_config.quantization == "deepseek_v4_fp8"
                        assert config.load_config.load_format == "safetensors"
                        assert config.engram_config.cpu_offload is True
                        assert (config.speculative_config is not None) is bool(steps)
                        assert (config.kv_transfer_config is not None) is bool(pd_role)
                        record.update(
                            passed=True,
                            quantization=config.model_config.quantization,
                            cpu_engram=config.engram_config.cpu_offload,
                            use_ubatching=config.parallel_config.use_ubatching,
                            num_ubatches=config.parallel_config.num_ubatches,
                            cudagraph_mode=config.compilation_config.cudagraph_mode.name,
                            native_revalidation_passed=True,
                        )
                except (Exception, SystemExit) as error:
                    record.update(
                        passed=False,
                        error=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc(),
                    )
                records.append(record)
    after = torch.npu.is_initialized()
    report = {
        "passed": all(item["passed"] for item in records) and not after,
        "npu_initialized_before": before,
        "npu_initialized_after": after,
        "checkpoint_config_sha256": hashlib.sha256(original).hexdigest(),
        "checkpoint_config_unchanged": original == source.read_bytes(),
        "weight_loading": False,
        "model_construction": False,
        "hardware_acceptance": False,
        "scenarios": records,
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] and report["checkpoint_config_unchanged"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new result file")
    return check_matrix(args.model.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
