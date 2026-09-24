# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Check the fixed vLLM-Ascend Mooncake metadata contract without a model."""

from __future__ import annotations

from types import SimpleNamespace


def main() -> int:
    from mooncake.engine import TransferEngine
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
        MooncakeConnectorMetadata,
        MooncakeConnectorWorker,
    )

    assert TransferEngine is not None

    metadata = MooncakeConnectorMetadata()
    metadata.add_new_req(
        request_id="m9-contract-request",
        local_block_ids=[1, 2],
        num_external_tokens=128,
        kv_transfer_params={
            "remote_block_ids": [11, 12],
            "remote_engine_id": "m9-prefill",
            "remote_request_id": "m9-contract-request",
            "remote_host": "127.0.0.1",
            "remote_port": 30000,
            "remote_ptp_size": 1,
            "remote_multi_nodes_meta_mapping": {},
            "num_prompt_blocks": 2,
        },
    )
    request = metadata.requests["m9-contract-request"]
    assert request.remote_engine_id == "m9-prefill"
    assert request.remote_port == 30000
    assert request.num_external_tokens == 128

    topologies = {
        "prefill": {"dp_size": 8, "tp_size": 1},
        "decode": {"dp_size": 8, "tp_size": 1},
    }
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            get_from_extra_config=lambda key, default: topologies.get(key, default)
        )
    )
    worker = object.__new__(MooncakeConnectorWorker)
    worker._get_prefill_decode_size(config)
    assert worker._prefill_dp_size == 8
    assert worker._prefill_tp_size == 1
    assert worker._decode_dp_size == 8
    assert worker._decode_tp_size == 1
    assert worker._decode_pp_size == 1

    print("MooncakeHybridConnector metadata contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
