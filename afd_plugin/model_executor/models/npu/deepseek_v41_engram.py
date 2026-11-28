# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A-owned Engram preparation for native TP graph execution.

TP-sharded lookups contain collectives that the pinned native in-graph
producer excludes. Produce once outside replay and join on its parent stream;
captured children then read the same persistent rows without consuming events.
Native request history, speculative correction and final retirement stay native.
"""

from __future__ import annotations

from typing import Any

from vllm.config.compilation import CUDAGraphMode
from vllm_ascend.models.deepseek_v41.engram.embedding import (
    AscendParallelEngramEmbedding,
)
from vllm_ascend.models.deepseek_v41.engram.model_state import EngramModelState


class AFDDeepseekV41EngramModelState(EngramModelState):
    # Upstream: EngramModelState.__init__ @ Ascend 7e37e29a, engram/model_state.py.
    # Reason: TP>1 cannot capture the native Engram collective producer.
    # Functionality: choose graph-external preparation for that native layout.
    # Signature matches upstream. Delegation retains all native history and
    # metadata storage; no replacement request/token state is introduced.
    # Cost: one parent producer and stream waits per replay, outside the graph.
    # Remove when native supports captured TP-sharded producers for AFD stages.
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        # ### PATCH START: TP graph producer selection
        self.uses_external_engram_graph_producer = (
            self.lookback_token_ids is not None
            and not vllm_config.model_config.enforce_eager
            and not model.supports_engram_graph_producer
        )
        # ### PATCH END: TP graph producer selection

    # Upstream: EngramModelState.prepare_engram_dummy_inputs @ Ascend 7e37e29a.
    # Reason: native priming installs ExternalEvents consumed/reset by one full
    # model; two AFD children cannot each reset that same producer record.
    # Functionality: capture only persistent lookup/mask addresses, without
    # priming an external event. Native slotless dummy history is preserved.
    # Signature matches upstream. Delegation exception: unchanged native route
    # owns captured producer inputs when the backend can capture that producer.
    def prepare_engram_dummy_inputs(
        self, num_reqs: int, num_tokens: int
    ) -> dict[str, Any]:
        # ### PATCH START: event-free TP capture inputs
        if not self.uses_external_engram_graph_producer:
            return super().prepare_engram_dummy_inputs(num_reqs, num_tokens)
        self._engram_inputs = None
        window = self.lookback_token_ids
        assert window is not None
        window.fill_(-1)
        buffers, mask = self.model.language_model.model._get_engram_input_buffers()
        if num_tokens > mask.shape[0]:
            raise ValueError("Engram token count exceeds the output buffer capacity")
        mask[:num_tokens].zero_()
        return {
            "lookback_token_ids": window,
            "engram_lookups": buffers,
            "engram_mask": mask,
        }
        # ### PATCH END: event-free TP capture inputs

    # Upstream: EngramModelState.prepare_engram @ Ascend 7e37e29a.
    # Reason: native FULL preparation publishes one-shot ExternalEvents, while
    # TP AFD graphs need reusable rows already ready on their parent stream.
    # Functionality: run one ordinary native producer, join its plain events
    # before graph replay/fork, and omit readiness objects from captured inputs.
    # Signature matches upstream. Delegation retains eager and native in-graph
    # behavior. Failure retirement remains execute_model's native finally.
    def prepare_engram(self) -> dict[str, Any]:
        # ### PATCH START: one TP producer before graph replay
        pending = self._engram_inputs
        if (
            not self.uses_external_engram_graph_producer
            or pending is None
            or pending["cg_mode"] != CUDAGraphMode.FULL
        ):
            return super().prepare_engram()
        # NONE selects plain reusable events; request coordinates and valid
        # token counts remain exactly those gathered once by the native parent.
        produced = self.model.prepare_engram_inputs(
            **{**pending, "cg_mode": CUDAGraphMode.NONE}
        )
        mask_ready = produced.get("engram_mask_ready_event")
        if mask_ready is not None:
            AscendParallelEngramEmbedding.wait_lookup(mask_ready, external=False)
        for ready in (produced.get("engram_pending") or {}).values():
            AscendParallelEngramEmbedding.wait_lookup(ready, external=False)
        self._engram_inputs = None
        return {
            name: value
            for name, value in produced.items()
            if name
            not in ("engram_pending", "engram_mask_ready_event", "engram_graph_events")
        }
        # ### PATCH END: one TP producer before graph replay
