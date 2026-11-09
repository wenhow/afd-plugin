# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Observe native U1 FULL preparation before its model callback executes."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import get_ident
from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import create_forward_context
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.ubatch_utils import UBatchState
from vllm.v1.worker.ubatch_utils import UBatchSlice
from vllm.v1.worker.utils import AttentionGroup
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

if TYPE_CHECKING:
    from afd_plugin.v1.worker.npu.hccl_v41_ubatch_runner import HcclV41GraphOwner


class _HcclV41U1PreparationScope:
    def __init__(
        self,
        model_state: EngramModelState,
        vllm_config: VllmConfig,
        resolve_owner: Callable[[], HcclV41GraphOwner | None],
    ) -> None:
        self.model_state, self.vllm_config = model_state, vllm_config
        self.resolve_owner = resolve_owner
        self.original_prepare = model_state.prepare_attn
        self.thread_id = get_ident()


@contextmanager
def use_hccl_v41_u1_preparation(
    model_state: EngramModelState,
    vllm_config: VllmConfig,
    resolve_owner: Callable[[], HcclV41GraphOwner | None],
) -> Iterator[None]:
    """Bind one selected ModelState, leaving U2 and ordinary calls native.

    The resolver returns an accepted U1 FULL owner or None. Its selection is a
    local read, not wire agreement. This scope prepares no history, producer,
    model callback or completion. NPU U2 preparation temporarily replaces the
    selected prepare_attn method itself and restores this binding afterwards.
    Concurrent use of the selected ModelState is unsupported.
    """
    original = model_state.prepare_attn
    if isinstance(original, MethodType) and any(
        isinstance(cell.cell_contents, _HcclV41U1PreparationScope)
        for cell in original.__func__.__closure__ or ()
    ):
        raise RuntimeError("native U1 preparation scopes cannot be nested")
    scope = _HcclV41U1PreparationScope(model_state, vllm_config, resolve_owner)
    had_override = "prepare_attn" in vars(model_state)

    # Upstream: EngramModelState.prepare_attn @ Ascend 7e37e29a:77-99.
    # Reason: native U1 capture hides its actual dummy batch inside the capture
    # helper, before the forward context. Functionality: retain selected native
    # metadata preparation, then accept its real single-stage FULL layout.
    # Signature matches upstream exactly; no added parameters. Delegation
    # exception: selected ModelState preparation owns backend metadata/task
    # builders and Engram mode; copying it would bypass its backend behavior.
    # Keep native NONE/for_capture=True arguments unchanged during capture,
    # then retain FULL for the parent's later prepare_inputs selection.
    # Remove when native preparation exposes its actual batch to the plugin.
    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        # ### PATCH START: accept actual U1 preparation before model work
        if self is not scope.model_state:
            raise ValueError("native U1 preparation belongs to the selected ModelState")
        owner = scope.resolve_owner()
        previous_mode = self._cg_mode
        try:
            if get_ident() != scope.thread_id:
                raise RuntimeError("native U1 preparation belongs to its owning thread")
            if owner is not None:
                if ubatch_idx != 0 or not (
                    cudagraph_mode == CUDAGraphMode.FULL
                    or cudagraph_mode == CUDAGraphMode.NONE
                    and for_capture
                ):
                    raise ValueError(
                        "native U1 graph preparation requires FULL child zero"
                    )
                owner.require_full(input_batch, for_capture=for_capture)
            # ### PATCH END: accept actual U1 preparation before model work
            metadata = scope.original_prepare(
                input_batch,
                cudagraph_mode,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
                for_capture=for_capture,
                ubatch_idx=ubatch_idx,
            )
            # ### PATCH START: retain FULL producer mode and one actual child
            if owner is not None:
                self._cg_mode = CUDAGraphMode.FULL
                context = create_forward_context(
                    metadata,
                    scope.vllm_config,
                    dp_metadata=None,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    slot_mapping=build_slot_mappings_by_layer(
                        slot_mappings, kv_cache_config
                    ),
                    is_padding=input_batch.is_padding,
                )
                context.additional_kwargs.update(
                    current_platform.set_additional_forward_context(
                        metadata,
                        scope.vllm_config,
                        None,
                        num_tokens=input_batch.num_tokens_after_padding,
                        cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    )
                )
                context.additional_kwargs.update(
                    ubatch_idx=0, num_ubatches=1, dbo_enabled=False
                )
                context.ubatch_idx = 0
                state = UBatchState(
                    [
                        UBatchSlice(
                            slice(0, input_batch.num_reqs_after_padding),
                            slice(0, input_batch.num_tokens_after_padding),
                        )
                    ],
                    [context],
                )
                owner.prepared(input_batch, state, for_capture=for_capture)
            return metadata
        except BaseException:
            if owner is not None:
                self._cg_mode = previous_mode
                owner.fail()
            raise
        # ### PATCH END: retain FULL producer mode and one actual child

    try:
        model_state.prepare_attn = MethodType(prepare_attn, model_state)
        yield
    finally:
        if had_override:
            model_state.prepare_attn = original
        else:
            del model_state.prepare_attn
