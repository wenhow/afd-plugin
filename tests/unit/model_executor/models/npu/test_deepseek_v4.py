# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Old V4 role payloads on CPU; no native model/device acceptance."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm.forward_context import override_forward_context  # noqa: E402

from afd_plugin.connectors import AFDForwardContextMetadata  # noqa: E402
from afd_plugin.model_executor.models import deepseek_v2  # noqa: E402
from afd_plugin.model_executor.models.npu import deepseek_v4 as adapter  # noqa: E402
from tests.unit.fakes.deepseek_v4 import (  # noqa: E402
    ROUTER_DELTA_SCALE,
    RoleBoundaryPeer,
    RoutingMoE,
    ffn_layer,
)


@pytest.fixture
def role_peer(monkeypatch):
    monkeypatch.setattr(adapter.native, "DeepseekV4MoE", RoutingMoE)
    peer = RoleBoundaryPeer(ffn_layer(adapter.AFDDeepseekV4DecoderLayer))
    metadata = AFDForwardContextMetadata(
        tokens_start_loc=[0],
        requests_start_loc=[0],
        stage_idx=0,
        connector=peer,
        tokens_lens=[2],
        num_stages=1,
    )
    monkeypatch.setattr(
        deepseek_v2, "get_afd_metadata_from_forward_context", lambda: metadata
    )
    monkeypatch.setattr(
        deepseek_v2, "maybe_apply_dbo_yield", lambda tensor, *, role: tensor
    )
    return peer


@pytest.mark.parametrize("explicit_ids", [False, True])
@pytest.mark.parametrize("independent_router_input", [False, True])
def test_role_roundtrip_preserves_router_precision_and_raw_token_ids(
    role_peer, explicit_ids, independent_router_input
):
    hidden = torch.ones(2, 4, dtype=torch.bfloat16)
    fp32 = hidden.float() + 0.00031 if independent_router_input else None
    ids = torch.tensor([7, -1], dtype=torch.int32)
    original_ids = ids.clone()
    context = SimpleNamespace(input_ids=ids, ubatch_idx=0)
    proxy = adapter.AFDDeepseekV4RemoteMoE(layer_idx=3)

    with override_forward_context(context):
        output = proxy(
            hidden,
            input_ids=ids if explicit_ids else None,
            hidden_states_fp32=fp32,
        )

    delta = 0 if fp32 is None else (fp32 - hidden.float()) * ROUTER_DELTA_SCALE
    torch.testing.assert_close(output, (delta + ids[:, None]).bfloat16().expand(2, 4))
    payload = role_peer.payloads[0]
    assert payload.context.metadata.layer_idx == 3
    assert payload.context.metadata.seq_lens == [2]
    assert payload.input_ids.dtype == torch.int64
    torch.testing.assert_close(payload.input_ids, ids.to(torch.int64))
    torch.testing.assert_close(ids, original_ids)
    if fp32 is None:
        assert payload.hidden_states_fp32 is None
    else:
        torch.testing.assert_close(payload.hidden_states_fp32, fp32)
        assert not torch.equal(payload.hidden_states_fp32, hidden.float())
