# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Build deterministic MooncakeHybridConnector configuration for DSV4 PD."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def build_mooncake_pd_config(
    *,
    role: str,
    engine_id: str,
    kv_port: int,
    prefill_dp_size: int,
    prefill_tp_size: int,
    decode_dp_size: int,
    decode_tp_size: int,
    ascend_local_comm_res_path: str | None = None,
) -> dict[str, Any]:
    if role not in {"kv_producer", "kv_consumer"}:
        raise ValueError("role must be kv_producer or kv_consumer")
    if not engine_id.strip():
        raise ValueError("engine_id must be non-empty")
    if not 1 <= kv_port <= 65535:
        raise ValueError("kv_port must be in 1..65535")
    sizes = {
        "prefill_dp_size": prefill_dp_size,
        "prefill_tp_size": prefill_tp_size,
        "decode_dp_size": decode_dp_size,
        "decode_tp_size": decode_tp_size,
    }
    for name, value in sizes.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    extra_config: dict[str, Any] = {
        "prefill": {
            "dp_size": prefill_dp_size,
            "tp_size": prefill_tp_size,
        },
        "decode": {
            "dp_size": decode_dp_size,
            "tp_size": decode_tp_size,
        },
    }
    local_comm_res_path = (ascend_local_comm_res_path or "").strip()
    if local_comm_res_path:
        if not os.path.isabs(local_comm_res_path):
            raise ValueError("ascend_local_comm_res_path must be absolute")
        extra_config["ascend_local_comm_res_path"] = local_comm_res_path

    return {
        "kv_connector": "MooncakeHybridConnector",
        "kv_role": role,
        "kv_port": kv_port,
        "engine_id": engine_id,
        "kv_parallel_size": 1,
        "kv_connector_extra_config": extra_config,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=("kv_producer", "kv_consumer"))
    parser.add_argument("--engine-id", required=True)
    parser.add_argument("--kv-port", required=True, type=int)
    parser.add_argument("--prefill-dp-size", required=True, type=int)
    parser.add_argument("--prefill-tp-size", required=True, type=int)
    parser.add_argument("--decode-dp-size", required=True, type=int)
    parser.add_argument("--decode-tp-size", required=True, type=int)
    parser.add_argument(
        "--ascend-local-comm-res-path",
        default=os.environ.get("ASCEND_LOCAL_COMM_RES_PATH"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = build_mooncake_pd_config(
        role=args.role,
        engine_id=args.engine_id,
        kv_port=args.kv_port,
        prefill_dp_size=args.prefill_dp_size,
        prefill_tp_size=args.prefill_tp_size,
        decode_dp_size=args.decode_dp_size,
        decode_tp_size=args.decode_tp_size,
        ascend_local_comm_res_path=args.ascend_local_comm_res_path,
    )
    print(json.dumps(config, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
