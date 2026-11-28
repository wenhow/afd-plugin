# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Keep native NPU stream selection and vLLM's existing stream cache aligned."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from vllm.utils import torch_utils


# Source: vllm/utils/torch_utils.py:792-848 @ ced6857afa0ea7b2e3f0846a62e1394e90f15607.
# Reason: native NPU stream contexts do not update vLLM's CUDA-set_stream TLS
# cache. Native helpers may otherwise observe the parent during side compute.
# Functionality: scope the actual NPU stream and that existing thread's cache,
# restoring its precise prior value or absent state even when compute raises.
# This composition API does not replace an upstream function. It follows the
# existing NPUUBatchContext.update_stream boundary without a new registry.
# Remove when the upstream platform stream context updates its own TLS cache.
@contextmanager
def use_npu_current_stream(stream: torch.npu.Stream) -> Iterator[None]:
    """Expose the selected stream through both native and vLLM public queries."""
    # ### PATCH START: scope native NPU selection with the existing vLLM TLS
    cache = torch_utils._current_stream_tls
    had_value = "value" in vars(cache)
    previous = vars(cache).get("value")
    # Enter first: a native context-entry failure must not change vLLM's cache.
    with torch.npu.stream(stream):
        try:
            cache.value = stream
            yield
        finally:
            if had_value:
                cache.value = previous
            else:
                del cache.value
    # ### PATCH END: scope native NPU selection with the existing vLLM TLS
