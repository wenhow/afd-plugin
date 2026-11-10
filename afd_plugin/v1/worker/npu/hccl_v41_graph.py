# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU geometry and native graph dispatch before V4.1 input preparation.

The caller retains the actual ordered BatchReqState returned by native gather.
These values describe layout, not graph/cache availability or Engram ownership.
No transport, model, device tensor allocation or production admission lives here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from vllm.config import CUDAGraphMode, ParallelConfig
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.model_runner import BatchReqState
from vllm.v1.worker.ubatch_utils import (
    check_ubatch_thresholds,
    get_num_ubatches,
    maybe_create_ubatch_slices,
)

from afd_plugin.distributed.hccl_plan import HcclPlanCapabilities

DEFAULT_TARGET_QUERY_LEN = 1
SUPPORTED_UBATCHES = (1, 2)
BF16_ELEMENT_BYTES = 2
FP32_ELEMENT_BYTES = 4
INT32_ELEMENT_BYTES = 4
BOOL_ELEMENT_BYTES = 1


@dataclass(frozen=True, slots=True)
class HcclV41StageGeometry:
    request_start: int
    request_stop: int
    token_start: int
    token_stop: int


@dataclass(frozen=True, slots=True)
class HcclV41GraphGeometry:
    descriptor: BatchExecutionDescriptor
    stages: tuple[HcclV41StageGeometry, ...]


@dataclass(frozen=True, slots=True)
class HcclV41GraphDispatch:
    """Local native candidate and the fresh layout used for an all-peer miss."""

    candidate: HcclV41GraphGeometry
    eager: HcclV41GraphGeometry


@dataclass(frozen=True, slots=True)
class HcclV41GraphCoverage:
    """Native capture order and exact bank budgets for each descriptor.

    Byte tuples follow ``geometries``. Each value is per paired-TP rank, not
    the sum across Attention ranks or layers. Native model buffers, graph
    pools and output storage are additional allocations.
    """

    geometries: tuple[HcclV41GraphGeometry, ...]
    bank_bytes_per_rank: tuple[int, ...]
    wire_bytes_per_layer_per_rank: tuple[int, ...]


def _batch_dimensions(batch_state: BatchReqState) -> tuple[int, int]:
    num_reqs = len(batch_state.req_ids)
    scheduled = batch_state.num_scheduled_tokens
    if (
        num_reqs <= 0
        or scheduled.ndim != 1
        or len(scheduled) != num_reqs
        or not np.issubdtype(scheduled.dtype, np.integer)
        or np.any(scheduled <= 0)
        or int(scheduled.sum()) != batch_state.num_tokens
    ):
        raise ValueError("graph geometry requires a nonempty native scheduled batch")
    return num_reqs, batch_state.num_tokens


def _uniform_decode(batch_state: BatchReqState, target_query_len: int) -> bool:
    return not batch_state.has_prefill and bool(
        np.all(batch_state.num_scheduled_tokens == target_query_len)
    )


def _validate_target_query_len(target_query_len: int) -> None:
    if type(target_query_len) is not int or target_query_len <= 0:
        raise ValueError("target query length must be a positive integer")


def _validate_full_descriptor(
    descriptor: BatchExecutionDescriptor, target_query_len: int
) -> None:
    _validate_target_query_len(target_query_len)
    if (
        descriptor.cg_mode != CUDAGraphMode.FULL
        or descriptor.uniform_token_count != target_query_len
        or descriptor.max_query_len not in (None, target_query_len)
        or descriptor.num_active_loras != 0
        or descriptor.num_tokens <= 0
        or descriptor.num_reqs is None
        or descriptor.num_reqs <= 0
        or descriptor.num_tokens != descriptor.num_reqs * target_query_len
        or descriptor.num_ubatches not in SUPPORTED_UBATCHES
        or descriptor.num_tokens < descriptor.num_ubatches
    ):
        raise ValueError("graph geometry requires a FULL uniform target descriptor")


# Native source: gpu/ubatch_utils.py:create_ubatch_slices @ ced6857a:49-75.
# Reason: before native InputBatch preparation, only ordered CPU counts and
# descriptor dimensions exist. This new API supplies those fields explicitly;
# native maybe_create_ubatch_slices retains the token/request split algorithm.
# Remove the projection when upstream exposes CPU stage geometry before prepare.
def _stage_geometry(
    scheduled: np.ndarray,
    actual_requests: int,
    padded_requests: int,
    padded_tokens: int,
    num_ubatches: int,
) -> tuple[HcclV41StageGeometry, ...]:
    if num_ubatches == 1:
        return (HcclV41StageGeometry(0, padded_requests, 0, padded_tokens),)
    _, slices = maybe_create_ubatch_slices(
        True, scheduled, padded_tokens, padded_requests, num_ubatches
    )
    assert slices is not None
    return tuple(
        HcclV41StageGeometry(
            # ### PATCH START: native request bounds from explicit CPU dimensions
            min(stage.request_slice.start, actual_requests - 1),
            min(stage.request_slice.stop, padded_requests),
            # ### PATCH END: native request bounds from explicit CPU dimensions
            stage.token_slice.start,
            stage.token_slice.stop,
        )
        for stage in slices
    )


def preview_hccl_v41_geometry(
    batch_state: BatchReqState,
    descriptor: BatchExecutionDescriptor,
    *,
    target_query_len: int = DEFAULT_TARGET_QUERY_LEN,
) -> HcclV41GraphGeometry:
    """Preview the exact native layout in the fixed target-query/CP1 scope.

    FULL requires the agreed uniform target width. Eager uses native actual
    dimensions; graph-padding retained after a mode-only replacement is rejected.
    """
    num_reqs, num_tokens = _batch_dimensions(batch_state)
    if descriptor.num_ubatches not in SUPPORTED_UBATCHES:
        raise ValueError("graph geometry supports native U1 or U2")
    if descriptor.cg_mode == CUDAGraphMode.FULL:
        _validate_full_descriptor(descriptor, target_query_len)
        if not _uniform_decode(batch_state, target_query_len):
            raise ValueError("FULL graph geometry requires uniform target inputs")
        padded_requests = descriptor.num_reqs
        assert padded_requests is not None
        if descriptor.num_tokens < num_tokens or padded_requests < num_reqs:
            raise ValueError("FULL descriptor cannot truncate the native batch")
        padded_tokens = descriptor.num_tokens
    elif descriptor.cg_mode == CUDAGraphMode.NONE:
        if descriptor.num_tokens != num_tokens or descriptor.num_reqs != num_reqs:
            raise ValueError("eager geometry requires fresh actual batch dimensions")
        padded_requests, padded_tokens = num_reqs, num_tokens
    else:
        raise ValueError("graph geometry supports FULL or eager execution")
    if padded_tokens < descriptor.num_ubatches:
        raise ValueError("native stages require at least one padded token each")
    return HcclV41GraphGeometry(
        descriptor,
        _stage_geometry(
            batch_state.num_scheduled_tokens,
            num_reqs,
            padded_requests,
            padded_tokens,
            descriptor.num_ubatches,
        ),
    )


def eager_hccl_v41_geometry(
    descriptor: BatchExecutionDescriptor,
) -> HcclV41GraphGeometry:
    """Project a native eager U1 dummy/empty descriptor without request state.

    Real gathered batches use ``preview_hccl_v41_geometry``. A native dummy
    dispatch already supplies its dimensions and needs no synthetic request
    vector; reservation validates the resulting complete parent geometry.
    """
    if (
        descriptor.cg_mode != CUDAGraphMode.NONE
        or descriptor.num_ubatches != 1
        or descriptor.num_reqs is None
    ):
        raise ValueError("native dummy reservation requires an eager U1 descriptor")
    return HcclV41GraphGeometry(
        descriptor,
        (HcclV41StageGeometry(0, descriptor.num_reqs, 0, descriptor.num_tokens),),
    )


def capture_hccl_v41_geometry(
    descriptor: BatchExecutionDescriptor,
    *,
    target_query_len: int = DEFAULT_TARGET_QUERY_LEN,
) -> HcclV41GraphGeometry:
    """Preview native FULL target dummy capture without device inputs.

    InputBatch.make_dummy @ ced6857a:138-148 divides tokens across requests;
    for a uniform target descriptor each scheduled count is exactly target q.
    """
    _validate_full_descriptor(descriptor, target_query_len)
    num_reqs = descriptor.num_reqs
    assert num_reqs is not None
    scheduled = np.full(num_reqs, target_query_len, dtype=np.int32)
    return HcclV41GraphGeometry(
        descriptor,
        _stage_geometry(
            scheduled,
            num_reqs,
            num_reqs,
            descriptor.num_tokens,
            descriptor.num_ubatches,
        ),
    )


def validate_hccl_v41_capture_coverage(
    descriptors: Sequence[BatchExecutionDescriptor],
    capabilities: HcclPlanCapabilities,
) -> HcclV41GraphCoverage:
    """Validate the real native capture list before any factory input writes.

    Native VllmConfig and CudaGraphManager resolve sizes and U2 twins. Preserve
    that order and request/token geometry instead of padding scalar totals
    again on FFN. In this paired-TP protocol each FFN rank receives one peer's
    stages, so ``max_tokens`` bounds a parent per rank. It is not multiplied by
    Attention size. A later peer reservation still agrees the exact full key.
    """
    capabilities.validate()
    if (
        capabilities.graph_execution_version != 1
        or capabilities.local_counts_version != 1
        or not capabilities.attention_size
        == capabilities.ffn_size
        == capabilities.native_tp_size
    ):
        raise ValueError("graph capture coverage requires graph/counts v1 paired TP")
    geometries = []
    bank_bytes = []
    wire_bytes = []
    seen = set()
    hidden_bytes = (
        BF16_ELEMENT_BYTES + FP32_ELEMENT_BYTES + BF16_ELEMENT_BYTES
    ) * capabilities.hidden_size
    for descriptor in descriptors:
        geometry = capture_hccl_v41_geometry(
            descriptor, target_query_len=capabilities.target_query_len
        )
        canonical = replace(descriptor, max_query_len=capabilities.target_query_len)
        if canonical in seen:
            raise ValueError("native capture descriptors repeat the same graph layout")
        if (
            descriptor.num_tokens > capabilities.max_tokens
            or descriptor.num_ubatches > capabilities.max_stages
        ):
            raise ValueError("native capture descriptor exceeds HCCL session capacity")
        seen.add(canonical)
        geometries.append(geometry)
        bank_bytes.append(
            (
                hidden_bytes
                + INT32_ELEMENT_BYTES
                + INT32_ELEMENT_BYTES
                + BOOL_ELEMENT_BYTES
            )
            * descriptor.num_tokens
            + INT32_ELEMENT_BYTES * descriptor.num_ubatches
        )
        wire_bytes.append((hidden_bytes + INT32_ELEMENT_BYTES) * descriptor.num_tokens)
    return HcclV41GraphCoverage(tuple(geometries), tuple(bank_bytes), tuple(wire_bytes))


def dispatch_hccl_v41_graph(
    manager: CudaGraphManager | None,
    batch_state: BatchReqState,
    *,
    parallel_config: ParallelConfig,
    allow_ubatching: bool,
    need_eager: bool = False,
    max_query_len: int | None = None,
    num_active_loras: int = 0,
    target_query_len: int = DEFAULT_TARGET_QUERY_LEN,
) -> HcclV41GraphDispatch:
    """Use native thresholds and an actual U-specific manager cache lookup.

    The caller performs all-peer availability agreement before preparing inputs.
    A local NONE result is a cache miss, not a peer descriptor mismatch. Global
    fallback uses ``eager`` and never renames a padded FULL descriptor to NONE.
    """
    _validate_target_query_len(target_query_len)
    num_reqs, num_tokens = _batch_dimensions(batch_state)
    if (
        parallel_config.data_parallel_size != 1
        or parallel_config.data_parallel_rank != 0
    ):
        raise ValueError("local graph dispatch requires native DP1 rank zero")
    configured_ubatches = get_num_ubatches(parallel_config)
    if configured_ubatches not in SUPPORTED_UBATCHES:
        raise ValueError("local graph dispatch supports native U1 or U2")
    uniform_decode = _uniform_decode(batch_state, target_query_len)
    num_ubatches = 1
    if (
        allow_ubatching
        and num_tokens >= configured_ubatches
        and check_ubatch_thresholds(parallel_config, num_tokens, uniform_decode)
    ):
        num_ubatches = configured_ubatches
    eager = preview_hccl_v41_geometry(
        batch_state,
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
            num_ubatches=num_ubatches,
        ),
    )
    if need_eager or not uniform_decode or num_active_loras != 0:
        return HcclV41GraphDispatch(eager, eager)
    if manager is None:
        raise ValueError("graph dispatch requires an initialized native manager")
    descriptor = manager.dispatch(
        num_reqs,
        num_tokens,
        target_query_len,
        num_active_loras=num_active_loras,
        max_query_len=max_query_len,
        num_ubatches=num_ubatches,
    )
    if descriptor.cg_mode != CUDAGraphMode.FULL:
        return HcclV41GraphDispatch(eager, eager)
    try:
        _validate_full_descriptor(descriptor, target_query_len)
    except ValueError:
        # A native varlen/unsupported candidate is an early eager fallback;
        # current input/metadata preparation has not started yet.
        return HcclV41GraphDispatch(eager, eager)
    candidate = preview_hccl_v41_geometry(
        batch_state, descriptor, target_query_len=target_query_len
    )
    return HcclV41GraphDispatch(candidate, eager)
