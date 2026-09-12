#!/usr/bin/env python3
"""Resolve model-format-specific vLLM launch arguments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

NATIVE_A5_PROFILE = "deepseek-v4-native"
ASCEND_PROFILE = "ascend"
PASSTHROUGH_PROFILE = "model-config"


def _load_model_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"model config does not exist: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"model config is not valid JSON: {config_path}: {exc}"
        ) from exc
    if not isinstance(config, dict):
        raise ValueError(f"model config must contain a JSON object: {config_path}")
    return config


def _is_native_a5(config: dict[str, Any]) -> bool:
    quant_config = config.get("quantization_config")
    return (
        config.get("model_type") == "deepseek_v4"
        and isinstance(quant_config, dict)
        and quant_config.get("quant_method") == "fp8"
    )


def _validate_native_a5(config: dict[str, Any]) -> None:
    quant_config = config.get("quantization_config")
    expected = {
        "model_type": (config.get("model_type"), "deepseek_v4"),
        "architectures": (
            config.get("architectures"),
            ["DeepseekV4ForCausalLM"],
        ),
        "num_nextn_predict_layers": (
            config.get("num_nextn_predict_layers"),
            1,
        ),
        "quant_method": (
            quant_config.get("quant_method")
            if isinstance(quant_config, dict)
            else None,
            "fp8",
        ),
        "activation_scheme": (
            quant_config.get("activation_scheme")
            if isinstance(quant_config, dict)
            else None,
            "dynamic",
        ),
        "fmt": (
            quant_config.get("fmt") if isinstance(quant_config, dict) else None,
            "e4m3",
        ),
        "scale_fmt": (
            quant_config.get("scale_fmt") if isinstance(quant_config, dict) else None,
            "ue8m0",
        ),
        "weight_block_size": (
            quant_config.get("weight_block_size")
            if isinstance(quant_config, dict)
            else None,
            [128, 128],
        ),
    }
    mismatches = [
        f"{name}={actual!r} (expected {wanted!r})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(
            "unsupported DeepSeek-V4-Flash A5 checkpoint contract: "
            + ", ".join(mismatches)
        )
    expert_dtype = config.get("expert_dtype")
    if expert_dtype not in (None, "fp4"):
        raise ValueError(
            "unsupported DeepSeek-V4-Flash A5 checkpoint contract: "
            f"expert_dtype={expert_dtype!r} (expected missing or 'fp4')"
        )


def resolve_launch(
    model_path: Path,
    quantization: str,
    block_size: str,
    load_strategy: str,
    kv_cache_dtype: str,
) -> tuple[list[str], dict[str, Any]]:
    config = _load_model_config(model_path)
    checkpoint_quant = config.get("quantization_config")
    checkpoint_quant_method = (
        checkpoint_quant.get("quant_method")
        if isinstance(checkpoint_quant, dict)
        else None
    )
    quant_description = model_path / "quant_model_description.json"

    if checkpoint_quant_method == "mxfp8":
        raise ValueError(
            "the model uses quant_method='mxfp8', but vLLM-Ascend v0.23.0 "
            "supports the original DeepSeek-V4-Flash config with quant_method='fp8' "
            "and weight_block_size=[128, 128]; restore the official config.json first"
        )

    if quantization == "auto":
        if _is_native_a5(config):
            profile = NATIVE_A5_PROFILE
        elif quant_description.is_file() or checkpoint_quant_method == ASCEND_PROFILE:
            profile = ASCEND_PROFILE
        else:
            profile = PASSTHROUGH_PROFILE
    else:
        profile = quantization

    if profile == NATIVE_A5_PROFILE:
        _validate_native_a5(config)
    elif profile == ASCEND_PROFILE:
        if _is_native_a5(config):
            raise ValueError(
                "MODEL_QUANTIZATION=ascend is incompatible with the DeepSeek-V4 "
                "native checkpoint; use deepseek-v4-native or auto"
            )
        if (
            not quant_description.is_file()
            and checkpoint_quant_method != ASCEND_PROFILE
        ):
            raise ValueError(
                "MODEL_QUANTIZATION=ascend requires quant_model_description.json "
                "or quant_method=ascend"
            )
    elif profile != PASSTHROUGH_PROFILE:
        raise ValueError(f"unsupported MODEL_QUANTIZATION={profile!r}")

    if block_size == "auto":
        resolved_block_size = 32 if profile == NATIVE_A5_PROFILE else 128
    else:
        try:
            resolved_block_size = int(block_size)
        except ValueError as exc:
            raise ValueError(
                f"MODEL_BLOCK_SIZE must be auto or a positive integer: {block_size}"
            ) from exc
        if resolved_block_size <= 0:
            raise ValueError(f"MODEL_BLOCK_SIZE must be positive: {block_size}")

    if load_strategy == "auto":
        resolved_load_strategy = "prefetch" if profile == NATIVE_A5_PROFILE else "lazy"
    elif load_strategy in {"lazy", "prefetch"}:
        resolved_load_strategy = load_strategy
    else:
        raise ValueError(
            "MODEL_SAFETENSORS_LOAD_STRATEGY must be auto, lazy, or prefetch: "
            f"{load_strategy}"
        )
    if not kv_cache_dtype:
        raise ValueError("KV_CACHE_DTYPE must not be empty")

    launch_args = [
        "--safetensors-load-strategy",
        resolved_load_strategy,
        "--block-size",
        str(resolved_block_size),
        "--kv-cache-dtype",
        kv_cache_dtype,
    ]
    if profile == ASCEND_PROFILE:
        launch_args.extend(("--quantization", "ascend"))

    description = {
        "model_path": str(model_path.resolve()),
        "model_type": config.get("model_type"),
        "expert_dtype": config.get("expert_dtype"),
        "checkpoint_quant_method": checkpoint_quant_method,
        "profile": profile,
        "effective_hf_quant_method": checkpoint_quant_method,
        "block_size": resolved_block_size,
        "safetensors_load_strategy": resolved_load_strategy,
        "kv_cache_dtype": kv_cache_dtype,
        "speculative_method": (
            "deepseek_mtp" if profile == NATIVE_A5_PROFILE else "mtp"
        ),
        "uses_hf_quantization_override": False,
    }
    return launch_args, description


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--quantization",
        choices=("auto", ASCEND_PROFILE, NATIVE_A5_PROFILE, PASSTHROUGH_PROFILE),
        default="auto",
    )
    parser.add_argument("--block-size", default="auto")
    parser.add_argument("--safetensors-load-strategy", default="auto")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument(
        "--get",
        choices=("profile", "speculative_method"),
        help="Print one resolved description field instead of launch arguments.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        launch_args, description = resolve_launch(
            args.model_path,
            args.quantization,
            args.block_size,
            args.safetensors_load_strategy,
            args.kv_cache_dtype,
        )
    except ValueError as exc:
        print(f"[model-launch] ERROR: {exc}", file=sys.stderr)
        return 2

    if args.get:
        print(description[args.get])
    elif args.describe:
        print(json.dumps(description, sort_keys=True, indent=2))
    else:
        print("\n".join(launch_args))
        print(
            "[model-launch] "
            f"profile={description['profile']} "
            f"checkpoint_quant={description['checkpoint_quant_method']} "
            f"block_size={description['block_size']} "
            f"load={description['safetensors_load_strategy']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
