"""Probe native/A/F worker construction and optional single-rank device setup."""

import argparse
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path

import torch
import vllm_ascend.ops  # noqa: F401
from vllm.config import EngramConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.config.device import DeviceConfig
from vllm.config.load import LoadConfig
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_engram_dp_group,
    get_pp_group,
    get_tp_group,
    get_world_group,
)
from vllm.platforms import current_platform
from vllm.plugins import load_general_plugins

from afd_plugin.compat.npu.runtime import apply_afd_ascend_patches_if_needed

CHECKPOINT = "/mnt/workspace/models/DeepSeek-V4.1-Flash"
MAX_MODEL_LEN = 128
DATA_PARALLEL_SIZE = 2
DEVICE_PROBE_PARALLEL_SIZE = 1
DEVICE_PROBE_MEMORY_UTILIZATION = 0.01


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("native", "attention", "ffn"))
    parser.add_argument("--init-device", action="store_true")
    args = parser.parse_args()
    current_platform.pre_register_and_update()
    load_general_plugins()
    model_config = ModelConfig(
        model=CHECKPOINT,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        skip_tokenizer_init=True,
    )
    config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(load_format="dummy"),
    )
    # API construction cell only. Attach after native validation, as in the
    # existing role-config probe. The connector is an inert parser placeholder.
    config.engram_config = EngramConfig(cpu_offload=True, dp_shared_memory=False)
    parallel_size = (
        DEVICE_PROBE_PARALLEL_SIZE if args.init_device else DATA_PARALLEL_SIZE
    )
    config.parallel_config.data_parallel_size = parallel_size
    if args.init_device:
        config.device_config = DeviceConfig(device="npu")
        config.parallel_config.tensor_parallel_size = DEVICE_PROBE_PARALLEL_SIZE
        config.parallel_config.pipeline_parallel_size = DEVICE_PROBE_PARALLEL_SIZE
        config.parallel_config.prefill_context_parallel_size = (
            DEVICE_PROBE_PARALLEL_SIZE
        )
        config.parallel_config.decode_context_parallel_size = DEVICE_PROBE_PARALLEL_SIZE
        config.cache_config.gpu_memory_utilization = DEVICE_PROBE_MEMORY_UTILIZATION
    original_engram = config.engram_config
    if args.role != "native":
        config.additional_config = {
            "afd": {"role": args.role, "connector": "CAMP2pAFDConnector"}
        }
        apply_afd_ascend_patches_if_needed()
    # Native imports are lazy: plugin/platform patches must be installed first.
    from vllm_ascend.distributed.parallel_state import (
        destroy_ascend_model_parallel,
        get_mc2_group,
    )
    from vllm_ascend.distributed.parallel_state import (
        model_parallel_initialized as ascend_model_parallel_initialized,
    )
    from vllm_ascend.worker.worker import NPUWorker

    from afd_plugin.v1.worker.npu.attention_worker import AFDNPUAttentionWorker
    from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker

    worker_class = {
        "native": NPUWorker,
        "attention": AFDNPUAttentionWorker,
        "ffn": AFDNPUFFNWorker,
    }[args.role]
    report = {
        "role": args.role,
        "native_constructor_executed": False,
        "device_initialized": False,
        "model_loaded": False,
        "connector_initialized": False,
        "data_parallel_config_size": parallel_size,
        "device_groups_initialized": False,
        "nontrivial_engram_group_verified": False,
        "afd_shutdown_executed": False,
    }
    with tempfile.TemporaryDirectory(prefix="worker-device-") as temp:
        with ExitStack() as cleanup:
            if args.init_device:
                # ExitStack runs every callback even if another cleanup fails;
                # cleanup errors propagate rather than becoming passed probes.
                cleanup.callback(destroy_distributed_environment)
                cleanup.callback(destroy_model_parallel)
                cleanup.callback(destroy_ascend_model_parallel)
            init_method = Path(temp, "store").as_uri() if args.init_device else "unused"
            with set_current_vllm_config(config):
                worker = worker_class(config, 0, 0, init_method)
            report["native_constructor_executed"] = True
            if args.init_device:
                # F shutdown assumes an initialized runner/connector. This cell
                # creates neither, so use native shutdown's supported no-runner
                # path; do not invent a runner or claim AFD lifecycle coverage.
                cleanup.callback(NPUWorker.shutdown, worker)
            assert worker.model_config is model_config
            assert worker.use_v2_model_runner
            if args.role == "ffn":
                assert worker.vllm_config is not config
                assert worker.vllm_config.engram_config is None
                assert config.engram_config is original_engram
                assert not original_engram.dp_shared_memory
            else:
                assert worker.vllm_config is config
                assert worker.vllm_config.engram_config is original_engram
                assert original_engram.dp_shared_memory is (not args.init_device)
            if args.init_device:
                # Exercise only the inherited device/group implementation.
                # AFD init_device additionally creates a guarded role runner.
                with set_current_vllm_config(worker.vllm_config):
                    worker.device = worker._init_device()
                assert worker.device == torch.device("npu:0")
                assert torch.npu.current_device() == 0
                groups = {
                    "world": get_world_group(),
                    "tp": get_tp_group(),
                    "pp": get_pp_group(),
                    "dp": get_dp_group(),
                    "mc2": get_mc2_group(),
                }
                group_sizes = {name: group.world_size for name, group in groups.items()}
                assert all(
                    size == DEVICE_PROBE_PARALLEL_SIZE for size in group_sizes.values()
                )
                backend = torch.distributed.get_backend(groups["world"].device_group)
                assert backend == "hccl"
                # All three roles legitimately have no Engram DP group at
                # world=1. A's retained config does not prove a multi-rank group.
                assert get_engram_dp_group() is None
                report.update(
                    device_initialized=True,
                    device_groups_initialized=True,
                    device=str(worker.device),
                    backend=backend,
                    group_sizes=group_sizes,
                    engram_group_present=False,
                )
            report.update(
                engram_on_worker=worker.vllm_config.engram_config is not None,
                source_engram_retained=config.engram_config is original_engram,
                source_engram_dp_shared_memory=original_engram.dp_shared_memory,
                model_config_preserved=worker.model_config is model_config,
                use_v2_model_runner=worker.use_v2_model_runner,
            )
        if args.init_device:
            assert not torch.distributed.is_initialized()
            assert not ascend_model_parallel_initialized()
            assert get_engram_dp_group() is None
            report["native_shutdown_and_group_cleanup_completed"] = True
    print(
        "WORKER_CONSTRUCTION_RESULT " + json.dumps(report),
        flush=True,
    )


if __name__ == "__main__":
    main()
