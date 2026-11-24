# SPDX-License-Identifier: Apache-2.0
"""Ascend DSpark replay using draft-owned attention metadata."""

from types import MethodType

import torch
from vllm.forward_context import get_forward_context, set_forward_context
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import update_full_graph_params
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager


# Upstream: vllm-ascend 8d4409d6, DFlashAclGraphManager._graph_replay.
# Target metadata describes target tokens and may be split across stages.
# The caller has already built the complete draft batch's live metadata.
# Signature matches upstream. Remove when Ascend uses draft metadata here.
def _graph_replay(self, desc, attn_backend, num_tokens, draft_attn_metadatas):
    self.update_stream.wait_stream(torch.npu.current_stream())
    # Same parent replay as the native method. Calling self.run_fullgraph
    # would redispatch into this instance method.
    result = super(DFlashAclGraphManager, self).run_fullgraph(desc)
    num_tokens_across_dp = torch.full([self.speculator.dp_size], num_tokens)
    with set_forward_context(
        # ### PATCH START: use the current draft attention metadata.
        draft_attn_metadatas[0],
        # ### PATCH END: use the current draft attention metadata.
        self.vllm_config,
        num_tokens=num_tokens,
        cudagraph_runtime_mode=desc.cg_mode,
        num_tokens_across_dp=num_tokens_across_dp,
        batch_descriptor=None,
        slot_mapping=None,
    ):
        _EXTRA_CTX.is_draft_model = True
        _EXTRA_CTX.is_draft_model_prefill = False
        update_full_graph_params(
            attn_backend,
            self.update_stream,
            get_forward_context(),
            num_tokens,
            self.vllm_config,
            self.speculator.speculative_config,
            draft_attn_metadatas=draft_attn_metadatas,
        )
    return result


def install_window_dspark_replay(manager: DFlashAclGraphManager) -> None:
    """Bind the legacy replay branch only on the current worker's draft manager."""
    manager._graph_replay = MethodType(_graph_replay, manager)
