# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native request admission with CPU memory and staged-write kernel leaves.

The real Ascend request state, block tables, sampler registration and inherited
GPUModelRunner.add_requests/update_requests remain intact. The returned batch
state is produced by the actual Ascend gather method, including its configured
PD recompute classification. No target forward, input preparation, sampler
execution, checkpoint load or device operation is supplied by this helper.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker import utils as worker_utils
from vllm.v1.worker.block_table import get_block_table_width
from vllm.v1.worker.gpu import buffer_utils
from vllm.v1.worker.gpu.lora_utils import LoraState
from vllm.v1.worker.gpu.model_runner import BatchReqState
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer
from vllm_ascend.attention.dsa_v41 import DeepseekV41CacheBackend
from vllm_ascend.worker.v2.block_table import AscendBlockTables
from vllm_ascend.worker.v2.states import AscendRequestState

from afd_plugin.compat.npu.kv_layout_native import KVCacheValue
from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
    AFDNPUAttentionModelRunnerV2,
)

ZERO_KERNEL_WORD_BYTES = 4


class CPUUvaBuffer:
    """Host storage for the native buffer pool's accelerator-view boundary."""

    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()

    def uva(self, n: int | None = None) -> torch.Tensor:
        return self.cpu if n is None else self.cpu[:n]


def cpu_tensor_copy(
    data: list | np.ndarray | torch.Tensor,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit host-copy leaf; never request pinned accelerator allocation."""
    if device is not None and torch.device(device).type != "cpu":
        raise ValueError("request-admission fixture only copies CPU tensors")
    tensor = torch.as_tensor(data, dtype=dtype, device="cpu")
    return tensor.clone() if out is None else out.copy_(tensor)


class CPUStagedWriteKernel:
    """Replace only the native Triton stores, preserving staged-write bodies.

    Fused writes resolve pointers through this instance's known CPU outputs.
    This does not dereference arbitrary addresses or create a global registry.
    """

    def __init__(self):
        self.outputs: dict[int, torch.Tensor] = {}

    def bind_outputs(self, outputs: Sequence[torch.Tensor]) -> None:
        self.outputs = {tensor.data_ptr(): tensor for tensor in outputs}

    def __getitem__(self, grid):
        return self.write

    def write(
        self,
        output: torch.Tensor,
        stride: int | torch.Tensor,
        indices: torch.Tensor,
        starts: torch.Tensor,
        contents: torch.Tensor,
        cumulative_lengths: torch.Tensor,
        group_ids: torch.Tensor | None,
        *,
        BLOCK_SIZE: int,  # noqa: N803 - exact native Triton launch keywords
        MULTI_GROUP: bool,  # noqa: N803 - exact native Triton launch keywords
    ) -> None:
        previous = 0
        for row, start, stop, ordinal in zip(
            indices.tolist(),
            starts.tolist(),
            cumulative_lengths.tolist(),
            range(len(indices)),
            strict=True,
        ):
            if MULTI_GROUP:
                group = int(group_ids[ordinal])
                target = self.outputs[int(output[group])]
            else:
                target = output
            values = contents[previous:stop]
            target.reshape(target.shape[0], -1)[row, start : start + len(values)].copy_(
                values
            )
            previous = stop


class CPUBlockZeroKernel:
    """Execute native zeroer segments only within known destination storage.

    Segment selection, overlapping-span deduplication and skipped tuple/cache
    types remain the actual KVBlockZeroer's responsibility. This leaf never
    dereferences an arbitrary address or broadens a segment to a whole page.
    """

    def __init__(self, caches: Mapping[str, KVCacheValue]):
        self.storages: dict[int, torch.Tensor] = {}
        for value in caches.values():
            for tensor in (value,) if isinstance(value, torch.Tensor) else value:
                if tensor.device.type != "cpu":
                    raise ValueError("block-zero fixture requires CPU cache storage")
                storage = tensor.untyped_storage()
                self.storages[storage.data_ptr()] = torch.empty(
                    0, dtype=torch.uint8
                ).set_(storage, 0, (storage.nbytes(),), (1,))

    def __getitem__(self, grid):
        return self.zero

    def zero(
        self,
        addresses: torch.Tensor,
        strides: torch.Tensor,
        page_sizes: torch.Tensor,
        block_ids: torch.Tensor,
        *,
        BLOCK_SIZE: int,  # noqa: N803 - exact native Triton launch keyword
    ) -> None:
        for address, stride, page_size in zip(
            addresses.tolist(), strides.tolist(), page_sizes.tolist(), strict=True
        ):
            for block_id in block_ids.tolist():
                if block_id < 0:
                    raise ValueError("block-zero fixture received a negative block")
                start = address + block_id * stride * ZERO_KERNEL_WORD_BYTES
                stop = start + page_size * ZERO_KERNEL_WORD_BYTES
                for base, storage in self.storages.items():
                    if base <= start <= stop <= base + storage.numel():
                        storage[start - base : stop - base].zero_()
                        break
                else:
                    raise ValueError("native block-zero segment escaped D storage")


@dataclass(frozen=True)
class NativeRequestAdmission:
    owner: AFDNPUAttentionModelRunnerV2

    def admit(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[BatchReqState, int | None]:
        """Consume actual NewRequestData without replacing the native methods."""
        self.owner.add_requests(scheduler_output)
        self.owner.update_requests(scheduler_output)
        self.owner.block_tables.apply_staged_writes()
        return self.owner.gather_batch_req_state(scheduler_output, dummy_run=False)


def make_native_request_admission_case(
    monkeypatch: pytest.MonkeyPatch,
    owner: AFDNPUAttentionModelRunnerV2,
    *,
    kv_cache_views: Mapping[str, KVCacheValue],
) -> NativeRequestAdmission:
    """Attach empty native request state to an existing eager/U1 CPU owner.

    Model, speculator, consumer, cache views and connector are never replaced.
    The text-only model-state fields complete the existing uninitialized CPU
    owner; they do not disable a loaded model's enabled feature. Equal logical
    and kernel block sizes are the explicit CPU backend boundary, without
    claiming an NPU attention backend selected them. Actual native block-zero
    metadata is built from the registered views; only its device copy/kernel
    leaves execute on CPU, retaining tuple skips and exact byte spans.
    """
    if owner.device.type != "cpu":
        raise ValueError("request-admission fixture requires a CPU owner")
    config = owner.vllm_config
    if config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("request-admission fixture requires the actual PP1 role")
    state_fields = vars(owner.model_state)
    if any(
        state_fields.get(name) is not None
        for name in ("rope_state", "prompt_embeds_state", "_offload_request_states")
    ):
        raise ValueError("request-admission fixture requires uninitialized text state")

    kernel = CPUStagedWriteKernel()
    # Four CPU memory/operator leaves. Native pool, staging, fusion, request
    # admission and block-table methods execute without replacement.
    monkeypatch.setattr(buffer_utils, "UvaBuffer", CPUUvaBuffer)
    monkeypatch.setattr(buffer_utils, "is_uva_available", lambda: True)
    monkeypatch.setattr(buffer_utils, "async_tensor_h2d", cpu_tensor_copy)
    monkeypatch.setattr(buffer_utils, "_apply_write_kernel", kernel)

    owner.max_num_reqs = config.scheduler_config.max_num_seqs
    owner.max_num_tokens = config.scheduler_config.max_num_batched_tokens
    owner.max_model_len = config.model_config.max_model_len
    owner.vocab_size = config.model_config.hf_text_config.vocab_size
    owner.num_speculative_steps = config.speculative_config.num_speculative_tokens
    owner.decode_query_len = (
        owner.num_speculative_steps + owner.model_state.num_new_sampled_tokens_per_step
    )
    owner.is_first_pp_rank = owner.is_last_pp_rank = True
    owner.adaptive_verification = None
    owner.req_states = AscendRequestState(
        max_num_reqs=owner.max_num_reqs,
        max_model_len=owner.max_model_len,
        max_num_batched_tokens=owner.max_num_tokens,
        num_speculative_steps=owner.num_speculative_steps,
        vocab_size=owner.vocab_size,
        device=owner.device,
    )
    owner.lora_state = LoraState(owner.max_num_reqs)
    owner.prompt_logprobs_worker = PromptLogprobsWorker(owner.max_num_reqs)
    owner.sampler = Sampler(
        owner.max_num_reqs,
        owner.vocab_size,
        owner.device,
        owner.req_states,
        num_speculative_tokens=owner.num_speculative_steps,
    )
    # These are the unchanged native text-only registration branches. The
    # existing zero-work owner has no initialized rope/prompt/offload state.
    owner.model_state.rope_state = None
    owner.model_state.prompt_embeds_state = None
    owner.model_state._offload_request_states = None

    groups = owner.kv_cache_config.kv_cache_groups
    block_sizes = [group.kv_cache_spec.block_size for group in groups]
    attn_groups = [
        AttentionGroup(
            backend=DeepseekV41CacheBackend,
            layer_names=[name],
            kv_cache_spec=(
                group.kv_cache_spec.kv_cache_specs[name]
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else group.kv_cache_spec
            ),
            kv_cache_group_id=index,
        )
        for index, group in enumerate(groups)
        for name in group.layer_names
    ]
    # The real worker performs this startup before requests arrive. Build its
    # metadata locally without changing the bound model's forward context.
    # Preserve actual tuples: native KVBlockZeroer deliberately skips them.
    zero_kernel = CPUBlockZeroKernel(kv_cache_views)
    monkeypatch.setattr(worker_utils, "async_tensor_h2d", cpu_tensor_copy)
    monkeypatch.setattr(worker_utils, "_zero_kv_blocks_kernel", zero_kernel)
    owner.kv_block_zeroer = KVBlockZeroer(
        owner.device,
        attn_groups,
        block_sizes,
        {
            name: SimpleNamespace(kv_cache=value)
            for name, value in kv_cache_views.items()
        },
        owner.kv_cache_config.num_blocks,
    )
    widths = [
        get_block_table_width(
            group.kv_cache_spec.max_num_blocks_per_req(config, owner.max_model_len),
            group.kv_cache_spec.block_size,
            token_alignment=group.kv_cache_spec.block_table_token_alignment,
        )
        for group in groups
    ]
    slot_mapping = [
        (
            group.kv_cache_spec.first_spec
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else group.kv_cache_spec
        ).uses_slot_mapping
        for group in groups
    ]
    owner.block_tables = AscendBlockTables(
        block_sizes=block_sizes,
        max_num_reqs=owner.max_num_reqs,
        max_num_batched_tokens=owner.max_num_tokens,
        max_num_blocks_per_group=widths,
        device=owner.device,
        kernel_block_sizes=block_sizes,
        slot_mapping_enabled=slot_mapping,
    )
    kernel.bind_outputs([table.gpu for table in owner.block_tables.block_tables])
    return NativeRequestAdmission(owner)
