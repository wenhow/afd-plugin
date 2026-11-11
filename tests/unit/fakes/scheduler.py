# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Actual native PD scheduler with CPU configuration and worker-output leaves.

Scheduler construction, the external AFD connector factory, Request, the hybrid
KV manager, schedule and update_from_output remain native. Configuration-only
adapters avoid checkpoint/tokenizer loading and reuse native derived methods.
No target model, worker read, sampling or device operation runs in this helper.
"""

from copy import deepcopy
from types import SimpleNamespace

from transformers import PretrainedConfig
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    EncoderCacheManagerConfig,
    KVTransferConfig,
    ObservabilityConfig,
    SchedulerConfig,
    SpeculativeConfig,
    StructuredOutputsConfig,
    VllmConfig,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm_ascend import utils as ascend_utils
from vllm_ascend.ascend_config import SchedulerConfig as AscendSchedulerConfig
from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_hybrid_connector

from tests.unit.fakes.kv_layout_connector import DSPARK_CPU_COMPRESS_RATIO

SCHEDULER_MAX_MODEL_LEN = 64
SCHEDULER_MAX_REQUESTS = 4
SCHEDULER_MAX_BATCHED_TOKENS = 16
SCHEDULER_BLOCK_SIZE = 4
SCHEDULER_HASH_BLOCK_SIZE = 2
SCHEDULER_MAX_OUTPUT_TOKENS = 8
SCHEDULER_HANDSHAKE_PORT = 9000
SCHEDULER_DESTINATION_ENGINE = "destination-scheduler-engine"


class SchedulerSpeculativeConfig(SimpleNamespace):
    """Checkpoint metadata with the pinned native DSpark scheduling methods.

    The original shared config has already selected DSpark. The extra fields
    are its native validator's fixed scheduling metadata, not replacement
    scheduling or drafting algorithms. In particular DSpark uses Eagle-style
    hidden states, one prefill lookahead, and native parallel-draft budgeting.
    """

    use_eagle = SpeculativeConfig.use_eagle
    use_eagle_block_drop = SpeculativeConfig.use_eagle_block_drop
    use_multi_module_mtp = SpeculativeConfig.use_multi_module_mtp
    use_dflash = SpeculativeConfig.use_dflash
    use_dspark = SpeculativeConfig.use_dspark
    uses_draft_model = SpeculativeConfig.uses_draft_model
    max_num_new_slots_for_drafting = SpeculativeConfig.max_num_new_slots_for_drafting


class SchedulerVllmConfig(SimpleNamespace):
    """Metadata-only model config, retaining native scheduling derivations."""

    max_concurrent_batches = VllmConfig.max_concurrent_batches
    max_in_flight_tokens = VllmConfig.max_in_flight_tokens
    num_speculative_tokens = VllmConfig.num_speculative_tokens
    num_lookahead_tokens = VllmConfig.num_lookahead_tokens


def make_native_pd_scheduler_case(
    monkeypatch,
    *,
    config: SimpleNamespace,
    kv_cache_config: KVCacheConfig,
    request_id: str,
    prompt_token_ids: list[int],
    kv_transfer_params: dict,
    sampling_params: SamplingParams | None = None,
):
    """Build and admit one actual D request, without inventing worker completion.

    Call the returned scheduler's real schedule/update_from_output methods.
    Worker KVConnectorOutput is supplied explicitly by the caller. The first
    schedule exposes the actual D allocation in connector metadata; this helper
    neither starts a receiver nor assumes that a registered cache was filled.

    The input DSpark count and source schema stay unchanged. Recompute policy
    is resolved from actual additional_config into the native Ascend subconfig;
    it is not inferred from logical prompt length or overwritten with a mode.
    """
    prepared = SchedulerVllmConfig(**deepcopy(vars(config)))
    model = prepared.model_config
    # Complete the same checkpoint metadata as the registered-prefix fixture.
    # Native Mooncake derives its N-1 compressed prefill boundary from this
    # field; typed cache specs and the source schema are left unchanged.
    text = PretrainedConfig(**vars(model.hf_text_config))
    if "compress_ratios" not in vars(text):
        text.compress_ratios = [DSPARK_CPU_COMPRESS_RATIO]
    model.hf_config = model.hf_text_config = text
    model.uses_mrope = False
    model.is_encoder_decoder = False
    model.is_multimodal_model = False
    model.is_diffusion = False
    model.max_model_len = SCHEDULER_MAX_MODEL_LEN
    model.enable_return_routed_experts = False
    model.return_sampling_mask = False
    model.skip_tokenizer_init = True
    prepared.is_mm_encoder_only = False
    prepared.lora_config = None
    prepared.kv_events_config = None
    prepared.ec_transfer_config = None
    prepared.ec_manager_config = EncoderCacheManagerConfig()
    prepared.observability_config = ObservabilityConfig()
    prepared.structured_outputs_config = StructuredOutputsConfig()
    prepared.diffusion_config = None
    prepared.parallel_config.data_parallel_index = 0
    prepared.parallel_config.distributed_executor_backend = "uni"
    prepared.scheduler_config = SchedulerConfig(
        max_model_len=SCHEDULER_MAX_MODEL_LEN,
        is_encoder_decoder=False,
        max_num_seqs=SCHEDULER_MAX_REQUESTS,
        max_num_batched_tokens=SCHEDULER_MAX_BATCHED_TOKENS,
        disable_hybrid_kv_cache_manager=False,
        async_scheduling=False,
    )
    prepared.cache_config = CacheConfig(
        block_size=SCHEDULER_BLOCK_SIZE, enable_prefix_caching=False
    )
    prepared.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    prepared.speculative_config = SchedulerSpeculativeConfig(
        **(
            {
                "num_speculative_tokens_per_batch_size": None,
                "disable_eagle_block_drop": False,
                "parallel_drafting": True,
            }
            | vars(prepared.speculative_config)
        )
    )
    original = prepared.kv_transfer_config
    prepared.kv_transfer_config = KVTransferConfig(
        kv_connector=original.kv_connector,
        kv_connector_module_path=original.kv_connector_module_path,
        kv_role="kv_consumer",
        engine_id=SCHEDULER_DESTINATION_ENGINE,
        kv_buffer_device="cpu",
        kv_port=SCHEDULER_HANDSHAKE_PORT,
        kv_connector_extra_config=deepcopy(original.kv_connector_extra_config),
    )
    ascend_config = SimpleNamespace(
        scheduler_config=AscendSchedulerConfig.from_additional_config(
            prepared.additional_config
        )
    )
    # CPU configuration boundary: do not initialize or publish a process-wide
    # Ascend engine. Both real Mooncake construction and the real recompute
    # predicate observe this same typed policy; monkeypatch restores bindings.
    monkeypatch.setattr(
        mooncake_hybrid_connector, "init_ascend_config", lambda config: ascend_config
    )
    monkeypatch.setattr(
        mooncake_hybrid_connector, "get_ascend_config", lambda: ascend_config
    )
    monkeypatch.setattr(ascend_utils, "get_ascend_config", lambda: ascend_config)
    structured_output = StructuredOutputManager(prepared)
    scheduler = Scheduler(
        prepared,
        kv_cache_config,
        structured_output,
        block_size=SCHEDULER_BLOCK_SIZE,
        hash_block_size=SCHEDULER_HASH_BLOCK_SIZE,
    )
    request = Request(
        request_id,
        list(prompt_token_ids),
        sampling_params
        if sampling_params is not None
        else SamplingParams(max_tokens=SCHEDULER_MAX_OUTPUT_TOKENS),
        None,
    )
    request.kv_transfer_params = deepcopy(kv_transfer_params)
    scheduler.add_request(request)
    return SimpleNamespace(
        config=prepared,
        ascend_config=ascend_config,
        scheduler=scheduler,
        connector=scheduler.get_kv_connector(),
        request=request,
        close=scheduler.shutdown,
    )
