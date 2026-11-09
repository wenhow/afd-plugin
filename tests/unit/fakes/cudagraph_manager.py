# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real native CPU graph dispatch with registered candidates; no device graphs."""

from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager


def make_native_graph_manager(descriptors, graphs=None):
    descriptors = tuple(descriptors)
    manager = CudaGraphManager.__new__(CudaGraphManager)
    manager._graphs_captured = True
    manager._lora_dispatch_map = {}
    manager._max_lora_case = 0
    maximum = max((descriptor.num_tokens for descriptor in descriptors), default=0)
    manager._candidates = {
        (tokens, 0): list(descriptors) for tokens in range(1, maximum + 1)
    }
    manager.graphs = {} if graphs is None else dict(graphs)
    return manager
