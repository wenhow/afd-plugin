# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Actual native CPU batch-state values in an explicit native batch order."""

import numpy as np
from vllm.v1.worker.gpu.model_runner import BatchReqState


def make_batch_req_state(
    scheduled: tuple[int, ...], *, has_prefill: bool = False
) -> BatchReqState:
    num_reqs = len(scheduled)
    return BatchReqState(
        req_ids=[f"req-{index}" for index in range(num_reqs)],
        num_scheduled_tokens=np.asarray(scheduled, dtype=np.int32),
        num_tokens=sum(scheduled),
        idx_mapping_np=np.arange(num_reqs, dtype=np.intp),
        prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        is_prefilling_np=np.full(num_reqs, has_prefill, dtype=np.bool_),
        has_prefill=has_prefill,
    )
