from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.connectors import (  # noqa: E402
    AFDConnectorFactory,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.npu import p2p_hccl as hccl_module  # noqa: E402
from afd_plugin.connectors.npu.p2p_hccl import (  # noqa: E402
    HCCLAttentionGraphStreamPlan,
    HCCLMTPHeader,
    HCCLP2PTransferState,
    P2pHcclAFDConnector,
)


def _vllm_config(
    *,
    num_ubatches: int = 1,
    dsv4: bool = True,
    max_num_batched_tokens: int = 16,
    mtp: bool = False,
    mtp_draft_enforce_eager: bool = True,
    num_speculative_tokens: int = 1,
    enforce_eager: bool = True,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
):
    return SimpleNamespace(
        additional_config={"afd": {"connector_extra_config": {}}},
        parallel_config=SimpleNamespace(
            data_parallel_size=data_parallel_size,
            data_parallel_rank=0,
            prefill_context_parallel_size=1,
            tensor_parallel_size=tensor_parallel_size,
            num_ubatches=num_ubatches,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=enforce_eager,
            hf_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"] if dsv4 else [],
                hidden_size=4,
                hc_mult=4,
                num_hidden_layers=3,
                vocab_size=32,
            ),
        ),
        speculative_config=(
            SimpleNamespace(
                method="mtp",
                enforce_eager=mtp_draft_enforce_eager,
                num_speculative_tokens=num_speculative_tokens,
            )
            if mtp
            else None
        ),
    )


def _afd_config(*, role: str, attention: int = 1, ffn: int = 1):
    return AFDConfig(
        connector="P2pHcclAFDConnector",
        role=role,
        num_attention_ranks=attention,
        num_ffn_ranks=ffn,
    )


def _connector(
    *,
    role: str,
    role_rank: int = 0,
    attention: int = 1,
    ffn: int = 1,
    num_ubatches: int = 1,
    max_num_batched_tokens: int = 16,
    mtp: bool = False,
    mtp_draft_enforce_eager: bool = True,
    num_speculative_tokens: int = 1,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
):
    connector = P2pHcclAFDConnector(
        0,
        0,
        _vllm_config(
            num_ubatches=num_ubatches,
            max_num_batched_tokens=max_num_batched_tokens,
            mtp=mtp,
            mtp_draft_enforce_eager=mtp_draft_enforce_eager,
            num_speculative_tokens=num_speculative_tokens,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
        ),
        _afd_config(role=role, attention=attention, ffn=ffn),
        role_rank,
    )
    connector._initialized = True
    connector.data_pg_list = [object() for _ in range(num_ubatches)]
    connector.ids_pg_list = [object() for _ in range(num_ubatches)]
    connector.input_ids_buffers = [
        torch.empty(16, dtype=torch.int32) for _ in range(num_ubatches)
    ]
    connector.mtp_header_buffers = [
        torch.empty(4 + ffn, dtype=torch.int32) for _ in range(num_ubatches)
    ]
    if ffn > attention:
        connector.mtp_header_buffers_by_peer = [
            {
                peer_rank: torch.empty(4 + ffn, dtype=torch.int32)
                for peer_rank in connector.mapping.ffn_peer_ranks
            }
            for _ in range(num_ubatches)
        ]
    return connector


def _attention_context(*, layer_idx: int, stage_idx: int, num_tokens: int):
    return AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_len=num_tokens,
        ),
    )


def _mtp_attention_context(*, num_tokens: int):
    return AFDTransferContext(
        metadata=AFDTransferMetadata.create_attention_metadata(
            layer_idx=0,
            stage_idx=0,
            seq_len=num_tokens,
            phase="mtp",
            speculative_step=0,
        ),
    )


def test_p2p_hccl_factory_registration():
    connector = AFDConnectorFactory.create_connector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
    )

    assert isinstance(connector, P2pHcclAFDConnector)
    assert connector.requires_input_ids is True
    assert connector.topology.role_rank == 0
    assert connector.is_initialized is False


@pytest.mark.parametrize(
    ("role", "role_rank", "expected_subgroup", "expected_peers"),
    [
        ("attention", 0, 0, (0, 2, 3)),
        ("attention", 3, 1, (1, 4, 5)),
        ("ffn", 0, 0, (0, 2, 3)),
        ("ffn", 1, 1, (1, 4, 5)),
    ],
)
def test_p2p_hccl_accepts_integer_multiple_topology(
    role,
    role_rank,
    expected_subgroup,
    expected_peers,
):
    connector = P2pHcclAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role=role, attention=4, ffn=2),
        role_rank,
    )

    assert connector.ratio == 2
    assert connector.mapping.subgroup_index == expected_subgroup
    assert connector.mapping.subgroup_ranks == expected_peers


@pytest.mark.parametrize(
    ("attention", "ffn"),
    [(3, 2), (0, 1), (1, 0), (-1, 1), (1, -1)],
)
def test_p2p_hccl_rejects_invalid_unequal_topology(attention, ffn):
    with pytest.raises(ValueError, match="P2P AFD connectors require"):
        P2pHcclAFDConnector(
            0,
            0,
            _vllm_config(),
            _afd_config(role="attention", attention=attention, ffn=ffn),
            0,
        )


def test_p2p_hccl_attention_sends_ids_before_hidden(monkeypatch):
    connector = _connector(role="attention")
    events = []
    forward_context = SimpleNamespace(afd_input_ids_pretransferred=False)
    monkeypatch.setattr(hccl_module, "get_forward_context", lambda: forward_context)
    monkeypatch.setattr(
        hccl_module,
        "maybe_apply_dbo_yield",
        lambda tensor, **_kwargs: events.append(("yield", tensor.clone())),
    )

    def send(tensor, *, dst, group):
        kind = "ids" if tensor.dtype == torch.int32 else "hidden"
        events.append((kind, dst, group, tensor.clone()))

    monkeypatch.setattr(hccl_module.dist, "send", send)
    hidden = torch.ones((3, 4), dtype=torch.bfloat16)
    connector.send_attn_output(
        hidden,
        _attention_context(layer_idx=0, stage_idx=0, num_tokens=3),
        input_ids=torch.tensor([-1, 2, 31], dtype=torch.int64),
    )

    assert [event[0] for event in events] == ["ids", "yield", "hidden"]
    assert events[0][1:3] == (0, connector.ids_pg_list[0])
    assert events[2][1:3] == (0, connector.data_pg_list[0])
    assert events[0][3].dtype == torch.int32


def test_p2p_hccl_mtp_sends_fixed_header_before_moe_input(
    monkeypatch,
):
    connector = _connector(role="attention", num_ubatches=2, mtp=True)
    events = []
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda tensor, *, dst, group: events.append(
            (tensor.clone(), dst, group),
        ),
    )
    hidden = torch.ones((3, 4), dtype=torch.bfloat16)

    connector.send_attn_output(
        hidden,
        _mtp_attention_context(num_tokens=3),
        num_tokens_across_dp=torch.tensor([3], dtype=torch.int32),
    )

    assert len(events) == 2
    header, dst, group = events[0]
    assert (dst, group) == (0, connector.ids_pg_list[0])
    assert header.dtype == torch.int32
    assert header[1:].tolist() == [0, 3, 1, 3]
    assert events[1][1:] == (0, connector.data_pg_list[0])
    assert events[1][0].shape == (3, 4)


def test_p2p_hccl_mtp_u2_uses_sync_stage_zero_receive(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2, mtp=True)
    connector.a2f_send_stream = object()
    connector.f2a_recv_stream = object()
    connector.attention_pipeline_events = {(0, 0): object()}
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(dbo_enabled=True, num_ubatches=2),
    )
    calls = []
    monkeypatch.setattr(
        connector,
        "_enqueue_attention_receive",
        lambda *_args, **_kwargs: pytest.fail("MTP must not enter decoder pipeline"),
    )
    monkeypatch.setattr(
        connector,
        "_recv_tensor",
        lambda tensor, *, src, group: calls.append((tensor, src, group)),
    )
    hidden = torch.empty((3, 4), dtype=torch.bfloat16)

    output = connector.recv_ffn_output(
        hidden,
        ubatch_idx=0,
        phase="mtp",
    )

    assert output is hidden
    assert calls == [(hidden, 0, connector.data_pg_list[0])]


def test_p2p_hccl_mtp_header_uses_graph_send_while_compiling(monkeypatch):
    connector = _connector(role="attention", mtp=True)
    events = []
    monkeypatch.setattr(hccl_module.torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda tensor, *, dst, group: events.append(
            (tensor.clone(), dst, group),
        ),
    )

    connector.send_mtp_header(
        num_tokens=3,
        speculative_step=0,
        num_tokens_across_dp=torch.tensor([3], dtype=torch.int32),
        stage_idx=0,
    )

    assert len(events) == 1
    header, dst, group = events[0]
    assert (dst, group) == (0, connector.ids_pg_list[0])
    assert header[1:].tolist() == [0, 3, 1, 3]

    with pytest.raises(ValueError, match="token-count vector"):
        connector.send_mtp_header(
            num_tokens=3,
            speculative_step=0,
            num_tokens_across_dp=torch.tensor([1, 2], dtype=torch.int32),
            stage_idx=0,
        )


def test_p2p_hccl_mtp_header_carries_each_configured_speculative_step(monkeypatch):
    connector = _connector(
        role="attention",
        mtp=True,
        num_speculative_tokens=3,
    )
    headers = []
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda tensor, *, dst, group: headers.append(tensor.clone()),
    )

    for speculative_step in range(3):
        connector.send_mtp_header(
            num_tokens=3,
            speculative_step=speculative_step,
            num_tokens_across_dp=torch.tensor([3], dtype=torch.int32),
            stage_idx=0,
        )

    assert [int(header[1]) for header in headers] == [0, 1, 2]
    with pytest.raises(RuntimeError, match="speculative step must be in"):
        connector.send_mtp_header(
            num_tokens=3,
            speculative_step=3,
            num_tokens_across_dp=torch.tensor([3], dtype=torch.int32),
            stage_idx=0,
        )


def test_p2p_hccl_full_draft_graph_prepares_header_before_send(monkeypatch):
    connector = _connector(
        role="attention",
        role_rank=3,
        attention=4,
        ffn=2,
        mtp=True,
        mtp_draft_enforce_eager=False,
    )
    payload = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(
                num_tokens_across_dp_cpu=torch.tensor([2, 3, 4, 5]),
            ),
            1: AFDDPMetadata(
                num_tokens_across_dp_cpu=torch.tensor([1, 2, 3, 4]),
            ),
        },
        is_graph_capturing=True,
        is_warmup=False,
    )
    connector.control_plane.update_state_from_dp_metadata(payload)
    prepared = connector.mtp_header_buffers[0].clone()
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda tensor, *, dst, group: sent.append((tensor.clone(), dst, group)),
    )

    connector.send_mtp_header(
        num_tokens=9,
        speculative_step=0,
        num_tokens_across_dp=torch.tensor([99]),
        stage_idx=0,
    )

    assert prepared.tolist() == [0x4D545031, 0, 9, 2, 8, 16]
    assert len(sent) == 1
    header, dst, group = sent[0]
    assert torch.equal(header, prepared)
    assert (dst, group) == (1, connector.ids_pg_list[0])


def test_p2p_hccl_full_draft_graph_rejects_unprepared_header():
    connector = _connector(
        role="attention",
        mtp=True,
        mtp_draft_enforce_eager=False,
    )

    with pytest.raises(RuntimeError, match="was not prepared"):
        connector.send_mtp_header(
            num_tokens=3,
            speculative_step=0,
            num_tokens_across_dp=torch.tensor([3]),
            stage_idx=0,
        )


def test_p2p_hccl_mtp_projects_attention_counts_to_unequal_ffn_world(
    monkeypatch,
):
    connector = _connector(
        role="attention",
        role_rank=3,
        attention=4,
        ffn=2,
        mtp=True,
    )
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda tensor, *, dst, group: sent.append((tensor.clone(), dst, group)),
    )

    connector.send_mtp_header(
        num_tokens=5,
        speculative_step=0,
        num_tokens_across_dp=torch.tensor([2, 3, 4, 5], dtype=torch.int32),
        stage_idx=0,
    )

    assert len(sent) == 1
    header, dst, group = sent[0]
    assert (dst, group) == (1, connector.ids_pg_list[0])
    assert header.tolist() == [0x4D545031, 0, 5, 2, 5, 9]


def test_p2p_hccl_mtp_fans_out_header_and_projected_counts(monkeypatch):
    connector = _connector(
        role="attention",
        attention=1,
        ffn=2,
        mtp=True,
        num_speculative_tokens=3,
    )
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda tensor, *, dst, group: sent.append((tensor.clone(), dst, group)),
    )

    connector.send_mtp_header(
        num_tokens=5,
        speculative_step=2,
        num_tokens_across_dp=torch.tensor([5], dtype=torch.int32),
        stage_idx=0,
    )

    assert [(dst, header.tolist()) for header, dst, _group in sent] == [
        (0, [0x4D545031, 2, 3, 2, 3, 2]),
        (1, [0x4D545031, 2, 2, 2, 3, 2]),
    ]
    assert all(group is connector.ids_pg_list[0] for _, _, group in sent)


def test_p2p_hccl_mtp_expands_dp_counts_for_equal_tp2(monkeypatch):
    connector = _connector(
        role="attention",
        role_rank=3,
        attention=4,
        ffn=4,
        mtp=True,
        tensor_parallel_size=2,
        data_parallel_size=2,
    )
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda tensor, *, dst, group: sent.append((tensor.clone(), dst, group)),
    )

    connector.send_mtp_header(
        num_tokens=5,
        speculative_step=0,
        num_tokens_across_dp=torch.tensor([3, 5], dtype=torch.int32),
        stage_idx=0,
    )

    assert len(sent) == 1
    header, dst, group = sent[0]
    assert (dst, group) == (3, connector.ids_pg_list[0])
    assert header.tolist() == [0x4D545031, 0, 5, 4, 3, 3, 5, 5]


def test_p2p_hccl_control_plane_rejects_mismatched_tp_size():
    connector = _connector(
        role="ffn",
        attention=2,
        ffn=2,
        tensor_parallel_size=2,
    )
    payload = AFDControlPayload(
        dp_metadata_list={0: AFDDPMetadata([3])},
        is_graph_capturing=False,
        is_warmup=False,
        tensor_parallel_size=1,
    )

    with pytest.raises(RuntimeError, match="matching Attention/FFN"):
        connector.control_plane.update_state_from_dp_metadata(payload)


def test_p2p_hccl_mtp_rejects_pre_hc_three_dimensional_hidden():
    connector = _connector(role="attention", mtp=True)

    with pytest.raises(ValueError, match="post-HC MoE input shape"):
        connector.send_attn_output(
            torch.ones((3, 4, 4), dtype=torch.bfloat16),
            _mtp_attention_context(num_tokens=3),
            num_tokens_across_dp=torch.tensor([3], dtype=torch.int32),
        )


def test_p2p_hccl_ffn_receives_mtp_header_and_separate_hidden_buffer(monkeypatch):
    connector = _connector(role="ffn", mtp=True)
    connector.mtp_hidden_recv_buffers[0] = torch.empty(
        (3, 4),
        dtype=torch.bfloat16,
    )
    events = []

    def recv(tensor, *, src, group):
        events.append((src, group, tuple(tensor.shape)))
        if tensor.dtype == torch.int32:
            tensor.copy_(torch.tensor([0x4D545031, 0, 3, 1, 3]))
        else:
            tensor.fill_(7)

    monkeypatch.setattr(hccl_module.dist, "recv", recv)

    header = connector.recv_mtp_header(stage_idx=0)
    payload = connector.recv_attn_output(
        ubatch_idx=0,
        layer_idx=0,
        phase="mtp",
        speculative_step=header.speculative_step,
        num_tokens=header.num_tokens,
    )

    assert header == HCCLMTPHeader(
        num_tokens=3,
        speculative_step=0,
        num_tokens_across_dp=torch.tensor([3], dtype=torch.int32),
    )
    assert events == [
        (1, connector.ids_pg_list[0], (5,)),
        (1, connector.data_pg_list[0], (3, 4)),
    ]
    assert payload.context.metadata.phase == "mtp"
    assert payload.input_ids is None
    assert payload.hidden_states.shape == (3, 4)
    assert torch.equal(payload.hidden_states, torch.full_like(payload.hidden_states, 7))


def test_p2p_hccl_ffn_aggregates_unequal_mtp_headers_and_splits_output(
    monkeypatch,
):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=4,
        ffn=2,
        max_num_batched_tokens=12,
        mtp=True,
    )
    connector.mtp_hidden_recv_buffers[0] = torch.empty((9, 4), dtype=torch.bfloat16)
    headers = {
        4: torch.tensor([0x4D545031, 0, 4, 2, 5, 9], dtype=torch.int32),
        5: torch.tensor([0x4D545031, 0, 5, 2, 5, 9], dtype=torch.int32),
    }
    events = []

    def recv(tensor, *, src, group):
        events.append((src, group, tuple(tensor.shape)))
        if tensor.dtype == torch.int32:
            tensor.copy_(headers[src])
        else:
            tensor.fill_(src)

    sent = []
    monkeypatch.setattr(hccl_module.dist, "recv", recv)
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda tensor, *, dst, group: sent.append((tensor.clone(), dst, group)),
    )

    header = connector.recv_mtp_header(stage_idx=0)
    payload = connector.recv_attn_output(
        ubatch_idx=0,
        layer_idx=0,
        phase="mtp",
        speculative_step=header.speculative_step,
        num_tokens=header.num_tokens,
    )
    connector.send_ffn_output(payload.hidden_states + 1, payload.context)

    assert header.num_tokens == 9
    assert header.speculative_step == 0
    assert header.num_tokens_across_dp.tolist() == [5, 9]
    assert events == [
        (4, connector.ids_pg_list[0], (6,)),
        (5, connector.ids_pg_list[0], (6,)),
        (4, connector.data_pg_list[0], (4, 4)),
        (5, connector.data_pg_list[0], (5, 4)),
    ]
    assert payload.context.metadata.seq_lens == [4, 5]
    assert payload.hidden_states[:, 0].tolist() == [4] * 4 + [5] * 5
    assert [(dst, tensor.shape[0]) for tensor, dst, _group in sent] == [
        (4, 4),
        (5, 5),
    ]
    assert connector.mtp_stage_layouts == {}


def test_p2p_hccl_ffn_captures_mtp_headers_from_static_peer_layout(monkeypatch):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=4,
        ffn=2,
        max_num_batched_tokens=12,
        mtp=True,
    )
    receives = []
    monkeypatch.setattr(
        connector,
        "_recv_tensor",
        lambda tensor, *, src, group: receives.append(
            (tensor, src, group),
        ),
    )

    header = connector.recv_mtp_header_for_graph(
        stage_idx=0,
        attention_peer_counts=(2, 3, 4, 5),
    )

    assert header.num_tokens == 9
    assert header.speculative_step == 0
    assert header.num_tokens_across_dp.tolist() == [5, 9]
    assert [(src, group) for _tensor, src, group in receives] == [
        (4, connector.ids_pg_list[0]),
        (5, connector.ids_pg_list[0]),
    ]
    assert connector.mtp_stage_layouts[0].seq_lens == (4, 5)


def test_p2p_hccl_ffn_captures_fanout_mtp_header_layout(monkeypatch):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=1,
        ffn=2,
        mtp=True,
    )
    receives = []
    monkeypatch.setattr(
        connector,
        "_recv_tensor",
        lambda tensor, *, src, group: receives.append((tensor, src, group)),
    )

    header = connector.recv_mtp_header_for_graph(
        stage_idx=0,
        attention_peer_counts=(5,),
    )

    assert header.num_tokens == 2
    assert header.num_tokens_across_dp.tolist() == [3, 2]
    assert [(src, group) for _tensor, src, group in receives] == [
        (2, connector.ids_pg_list[0]),
    ]
    assert connector.mtp_stage_layouts[0].peer_slices == ((2, 0, 2),)


def test_p2p_hccl_ffn_graph_header_rejects_wrong_attention_layout():
    connector = _connector(role="ffn", mtp=True)

    with pytest.raises(ValueError, match="Attention world size"):
        connector.recv_mtp_header_for_graph(
            stage_idx=0,
            attention_peer_counts=(1, 2),
        )


def test_p2p_hccl_ffn_rejects_inconsistent_unequal_mtp_headers(monkeypatch):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=4,
        ffn=2,
        mtp=True,
    )
    headers = iter(
        (
            torch.tensor([0x4D545031, 0, 4, 2, 5, 9], dtype=torch.int32),
            torch.tensor([0x4D545031, 0, 5, 2, 5, 8], dtype=torch.int32),
        )
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        lambda tensor, **_kwargs: tensor.copy_(next(headers)),
    )

    with pytest.raises(RuntimeError, match="peer headers disagree"):
        connector.recv_mtp_header(stage_idx=0)

    assert connector.mtp_stage_layouts == {}


def test_p2p_hccl_ffn_rejects_unequal_mtp_subgroup_total_mismatch(monkeypatch):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=4,
        ffn=2,
        mtp=True,
    )
    headers = iter(
        (
            torch.tensor([0x4D545031, 0, 4, 2, 5, 8], dtype=torch.int32),
            torch.tensor([0x4D545031, 0, 5, 2, 5, 8], dtype=torch.int32),
        )
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        lambda tensor, **_kwargs: tensor.copy_(next(headers)),
    )

    with pytest.raises(RuntimeError, match="does not match the FFN count vector"):
        connector.recv_mtp_header(stage_idx=0)

    assert connector.mtp_stage_layouts == {}


def test_p2p_hccl_mtp_header_layout_must_be_consumed_once(monkeypatch):
    connector = _connector(role="ffn", mtp=True)
    connector.mtp_hidden_recv_buffers[0] = torch.empty((3, 4), dtype=torch.bfloat16)

    def recv(tensor, **_kwargs):
        if tensor.dtype == torch.int32:
            tensor.copy_(torch.tensor([0x4D545031, 0, 3, 1, 3], dtype=torch.int32))
        else:
            tensor.fill_(1)

    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        recv,
    )

    connector.recv_mtp_header(stage_idx=0)
    with pytest.raises(RuntimeError, match="unconsumed header layout"):
        connector.recv_mtp_header(stage_idx=0)

    connector.recv_attn_output(
        ubatch_idx=0,
        layer_idx=0,
        phase="mtp",
        num_tokens=3,
    )
    with pytest.raises(RuntimeError, match="requires a preceding header"):
        connector.recv_attn_output(
            ubatch_idx=0,
            layer_idx=0,
            phase="mtp",
            num_tokens=3,
        )


def test_p2p_hccl_pretransferred_ids_send_only_hidden(monkeypatch):
    connector = _connector(role="attention")
    events = []
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(afd_input_ids_pretransferred=True),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda tensor, **_kwargs: events.append(tensor.clone()),
    )

    connector.send_attn_output(
        torch.ones((2, 4), dtype=torch.bfloat16),
        _attention_context(layer_idx=0, stage_idx=0, num_tokens=2),
        input_ids=torch.tensor([1, 2]),
    )

    assert len(events) == 1
    assert events[0].dtype == torch.bfloat16


def test_p2p_hccl_stage_groups_are_independent(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(afd_input_ids_pretransferred=True),
    )
    sent_groups = []
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda _tensor, *, dst, group: sent_groups.append((dst, group)),
    )

    for stage_idx in (0, 1):
        connector.send_attn_output(
            torch.ones((2, 4), dtype=torch.bfloat16),
            _attention_context(layer_idx=1, stage_idx=stage_idx, num_tokens=2),
        )

    assert sent_groups == [
        (0, connector.data_pg_list[0]),
        (0, connector.data_pg_list[1]),
    ]


def test_p2p_hccl_attention_yields_after_ffn_receive(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)
    events = []
    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        lambda tensor, *, src, group: events.append(("recv", src, group)),
    )
    monkeypatch.setattr(
        hccl_module,
        "maybe_apply_dbo_yield",
        lambda tensor, **kwargs: events.append(("yield", kwargs["role"], tensor)),
    )
    output = torch.empty((2, 4), dtype=torch.bfloat16)

    assert connector.recv_ffn_output(output, ubatch_idx=1) is output
    assert events == [
        ("recv", 0, connector.data_pg_list[1]),
        ("yield", "attention", output),
    ]


def test_p2p_hccl_attention_stream_pipeline_orders_sync_send_recv(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)
    calls = []
    active_stream = [None]
    compute_stream = object()
    send_stream = object()
    recv_stream = object()

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append((self.name, "record", stream))

        def wait(self, stream):
            calls.append((self.name, "wait", stream))

    @contextmanager
    def use_stream(stream):
        previous = active_stream[0]
        active_stream[0] = stream
        try:
            yield
        finally:
            active_stream[0] = previous

    connector.a2f_send_stream = send_stream
    connector.f2a_recv_stream = recv_stream
    connector.attention_pipeline_events = {
        (1, 0): hccl_module.HCCLAttentionPipelineEvents(
            compute_done=FakeEvent("compute"),
            send_done=FakeEvent("send"),
            recv_done=FakeEvent("recv"),
        )
    }
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(dbo_enabled=True, num_ubatches=2),
    )
    monkeypatch.setattr(hccl_module.torch.npu, "current_stream", lambda: compute_stream)
    monkeypatch.setattr(hccl_module.torch.npu, "stream", use_stream)
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda _tensor, *, dst, group: calls.append(
            ("dist.send", active_stream[0], dst, group)
        ),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        lambda tensor, *, src, group: calls.append(
            ("dist.recv", active_stream[0], src, group, tensor)
        ),
    )
    monkeypatch.setattr(
        hccl_module,
        "maybe_apply_dbo_yield",
        lambda tensor, **_kwargs: calls.append(("yield", tensor)),
    )
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)

    connector.send_attn_output(
        hidden,
        _attention_context(layer_idx=1, stage_idx=0, num_tokens=2),
    )
    output = connector.recv_ffn_output(hidden, ubatch_idx=0)

    assert output is not hidden
    assert connector.pending_attention_transfers == {}
    assert calls == [
        ("compute", "record", compute_stream),
        ("compute", "wait", send_stream),
        ("dist.send", send_stream, 0, connector.data_pg_list[0]),
        ("send", "record", send_stream),
        ("send", "wait", recv_stream),
        ("dist.recv", recv_stream, 0, connector.data_pg_list[0], output),
        ("recv", "record", recv_stream),
        ("yield", output),
        ("recv", "wait", compute_stream),
    ]


def test_p2p_hccl_layer_major_defers_receive_wait(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)
    calls = []
    active_stream = [None]
    compute_stream = object()
    send_stream = object()
    recv_stream = object()

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append((self.name, "record", stream))

        def wait(self, stream):
            calls.append((self.name, "wait", stream))

    @contextmanager
    def use_stream(stream):
        previous = active_stream[0]
        active_stream[0] = stream
        try:
            yield
        finally:
            active_stream[0] = previous

    connector.a2f_send_stream = send_stream
    connector.f2a_recv_stream = recv_stream
    connector.attention_pipeline_events = {
        (1, 0): hccl_module.HCCLAttentionPipelineEvents(
            compute_done=FakeEvent("compute"),
            send_done=FakeEvent("send"),
            recv_done=FakeEvent("recv"),
        )
    }
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            afd_layer_major_u2=True,
            dbo_enabled=True,
            num_ubatches=2,
        ),
    )
    monkeypatch.setattr(hccl_module.torch.npu, "current_stream", lambda: compute_stream)
    monkeypatch.setattr(hccl_module.torch.npu, "stream", use_stream)
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda _tensor, *, dst, group: calls.append(
            ("dist.send", active_stream[0], dst, group)
        ),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        lambda tensor, *, src, group: calls.append(
            ("dist.recv", active_stream[0], src, group, tensor)
        ),
    )
    monkeypatch.setattr(
        hccl_module,
        "maybe_apply_dbo_yield",
        lambda tensor, **_kwargs: calls.append(("yield", tensor)),
    )
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)

    connector.send_attn_output(
        hidden,
        _attention_context(layer_idx=1, stage_idx=0, num_tokens=2),
    )
    output = connector.recv_ffn_output(hidden, ubatch_idx=0)

    assert output is not hidden
    assert tuple(connector.attention_receive_dependencies) == (0,)
    with pytest.raises(RuntimeError, match="pipeline is not idle"):
        connector.require_attention_pipeline_idle()
    assert calls[-1][0] == "recv"

    connector.wait_for_attention_stage_receive(stage_idx=0, tensor=output)

    assert connector.attention_receive_dependencies == {}
    connector.require_attention_pipeline_idle()
    assert calls[-1] == ("recv", "wait", compute_stream)
    assert not any(call[0] == "yield" for call in calls)


def test_p2p_hccl_attention_stream_pipeline_requires_pretransferred_ids(
    monkeypatch,
):
    connector = _connector(role="attention", num_ubatches=2)
    connector.a2f_send_stream = object()
    connector.f2a_recv_stream = object()
    connector.attention_pipeline_events = {
        (0, 0): hccl_module.HCCLAttentionPipelineEvents(
            compute_done=object(),
            send_done=object(),
            recv_done=object(),
        )
    }
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            afd_input_ids_pretransferred=False,
            dbo_enabled=True,
            num_ubatches=2,
        ),
    )

    with pytest.raises(RuntimeError, match="requires input_ids to be pretransferred"):
        connector.send_attn_output(
            torch.ones((2, 4), dtype=torch.bfloat16),
            _attention_context(layer_idx=0, stage_idx=0, num_tokens=2),
            input_ids=torch.tensor([1, 2], dtype=torch.int32),
        )


def test_p2p_hccl_attention_stream_pipeline_is_inactive_without_forward_context(
    monkeypatch,
):
    connector = _connector(role="attention", num_ubatches=2)
    connector.a2f_send_stream = object()
    connector.f2a_recv_stream = object()
    connector.attention_pipeline_events = {(1, 0): object()}
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: (_ for _ in ()).throw(AssertionError("no forward context")),
    )
    sent = []
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda tensor, *, dst, group: sent.append((tensor, dst, group)),
    )

    hidden = torch.ones((2, 4), dtype=torch.bfloat16)
    connector.send_attn_output(
        hidden,
        _attention_context(layer_idx=1, stage_idx=0, num_tokens=2),
    )

    assert sent == [(hidden, 0, connector.data_pg_list[0])]
    assert connector.pending_attention_transfers == {}


def test_p2p_hccl_attention_stream_pipeline_is_inactive_for_graph_ubatching(
    monkeypatch,
):
    connector = _connector(role="attention", num_ubatches=2)
    connector.a2f_send_stream = object()
    connector.f2a_recv_stream = object()
    connector.attention_pipeline_events = {(1, 0): object()}
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            afd_graph_ubatching=True,
            dbo_enabled=True,
            num_ubatches=2,
        ),
    )
    sent = []
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda tensor, *, dst, group: sent.append((tensor, dst, group)),
    )

    hidden = torch.ones((2, 4), dtype=torch.bfloat16)
    connector.send_attn_output(
        hidden,
        _attention_context(layer_idx=1, stage_idx=0, num_tokens=2),
    )

    assert sent == [(hidden, 0, connector.data_pg_list[0])]
    assert connector.pending_attention_transfers == {}


def test_p2p_hccl_graph_uses_only_the_compute_side_stream(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)
    connector.a2f_send_stream = object()
    connector.f2a_recv_stream = object()
    connector.attention_graph_compute_stream = object()
    connector.attention_graph_stream_plan = HCCLAttentionGraphStreamPlan(
        compute_stream=connector.attention_graph_compute_stream,
    )
    connector.attention_pipeline_events = {(1, 0): object()}
    connector.attention_graph_events = {(1, 0): object()}
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            afd_graph_ubatching=True,
            afd_layer_major_u2=True,
            dbo_enabled=True,
            num_ubatches=2,
        ),
    )

    assert connector._attention_stream_pipeline_active() is False
    assert connector.attention_graph_compute_pipeline_active() is True


def test_p2p_hccl_graph_pipeline_skips_python_state_during_compile(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)

    class CompileUnsafeEvents(dict):
        def __bool__(self):
            raise AssertionError("Dynamo path must not inspect the event dictionary")

    connector.attention_graph_stream_plan = HCCLAttentionGraphStreamPlan(
        compute_stream=object(),
    )
    connector.attention_graph_events = CompileUnsafeEvents({(1, 0): object()})
    monkeypatch.setattr(hccl_module.torch.compiler, "is_compiling", lambda: True)

    assert connector.attention_graph_compute_pipeline_active() is False


def test_p2p_hccl_graph_keeps_send_and_receive_on_parent_stream(monkeypatch):
    monkeypatch.setenv("AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM", "0")
    connector = _connector(role="attention", num_ubatches=2)
    calls = []
    parent_stream = object()

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append((self.name, "record", stream))

        def wait(self, stream):
            calls.append((self.name, "wait", stream))

    connector.a2f_send_stream = object()
    connector.f2a_recv_stream = object()
    connector.attention_pipeline_events = {(1, 0): object()}
    connector.attention_graph_stream_plan = HCCLAttentionGraphStreamPlan(
        compute_stream=object(),
    )
    connector.attention_graph_events = {
        (1, 0): hccl_module.HCCLAttentionGraphEvents(
            ready=FakeEvent("ready"),
            compute_done=FakeEvent("compute_done"),
            send_done=FakeEvent("send_done"),
            recv_done=FakeEvent("recv_done"),
        )
    }
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            afd_graph_ubatching=True,
            afd_layer_major_u2=True,
            dbo_enabled=True,
            num_ubatches=2,
        ),
    )
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda _tensor, *, dst, group, stream=None: calls.append(
            ("send_tensor", dst, group, stream)
        ),
    )
    monkeypatch.setattr(
        connector,
        "_recv_tensor",
        lambda _tensor, *, src, group, stream=None: calls.append(
            ("recv_tensor", src, group, stream)
        ),
    )
    monkeypatch.setattr(
        hccl_module,
        "maybe_apply_dbo_yield",
        lambda *_args, **_kw: None,
    )
    monkeypatch.setattr(
        hccl_module.torch.npu,
        "current_stream",
        lambda: parent_stream,
    )
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)

    connector.send_attn_output(
        hidden,
        _attention_context(layer_idx=1, stage_idx=0, num_tokens=2),
    )
    output = connector.recv_ffn_output(hidden, ubatch_idx=0, layer_idx=1)

    assert calls == [
        ("send_tensor", 0, connector.data_pg_list[0], None),
        ("send_done", "record", parent_stream),
        ("send_done", "wait", parent_stream),
        ("recv_tensor", 0, connector.data_pg_list[0], None),
        ("recv_done", "record", parent_stream),
    ]
    assert output is hidden
    assert connector.pending_attention_transfers == {}
    assert connector.attention_receive_dependencies == {}


@pytest.mark.parametrize(
    ("three_stream_enabled", "expected_side_transport"),
    [("1", True), ("0", False)],
)
def test_p2p_hccl_initializes_configured_attention_graph_stream_plan(
    monkeypatch,
    three_stream_enabled,
    expected_side_transport,
):
    monkeypatch.setenv(
        "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM",
        three_stream_enabled,
    )
    connector = _connector(role="attention", num_ubatches=2)
    send_stream = object()
    recv_stream = object()
    compute_stream = object()
    streams = iter((send_stream, recv_stream, compute_stream))
    monkeypatch.setattr(
        hccl_module.torch.npu, "Stream", lambda **_kwargs: next(streams)
    )
    monkeypatch.setattr(hccl_module.torch.npu, "Event", object)

    connector._initialize_attention_stream_pipeline()

    plan = connector.attention_graph_stream_plan
    assert plan is not None
    assert plan.compute_stream is compute_stream
    assert plan.send_stream is (send_stream if expected_side_transport else None)
    assert plan.recv_stream is (recv_stream if expected_side_transport else None)


def test_attention_graph_stream_plan_decouples_logical_and_physical_streams():
    parent_stream = object()
    compute_stream = object()
    send_stream = object()
    recv_stream = object()

    parent_transport = HCCLAttentionGraphStreamPlan(
        compute_stream=compute_stream,
    )
    three_stream = HCCLAttentionGraphStreamPlan(
        compute_stream=compute_stream,
        send_stream=send_stream,
        recv_stream=recv_stream,
    )

    assert parent_transport.resolve("compute", parent_stream) is compute_stream
    assert parent_transport.resolve("send", parent_stream) is parent_stream
    assert parent_transport.resolve("recv", parent_stream) is parent_stream
    assert three_stream.resolve("send", parent_stream) is send_stream
    assert three_stream.resolve("recv", parent_stream) is recv_stream


def test_p2p_hccl_graph_routes_transport_through_stream_plan(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)
    calls = []
    parent_stream = object()
    send_stream = object()
    recv_stream = object()
    active_stream = [parent_stream]

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append((self.name, "record", stream))

        def wait(self, stream):
            calls.append((self.name, "wait", stream))

    @contextmanager
    def use_stream(stream):
        previous = active_stream[0]
        active_stream[0] = stream
        calls.append(("enter", stream))
        try:
            yield
        finally:
            calls.append(("exit", stream))
            active_stream[0] = previous

    connector.attention_graph_stream_plan = HCCLAttentionGraphStreamPlan(
        compute_stream=object(),
        send_stream=send_stream,
        recv_stream=recv_stream,
    )
    connector.attention_graph_events = {
        (1, 0): hccl_module.HCCLAttentionGraphEvents(
            ready=FakeEvent("ready"),
            compute_done=FakeEvent("compute_done"),
            send_done=FakeEvent("send_done"),
            recv_done=FakeEvent("recv_done"),
        )
    }
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            afd_graph_ubatching=True,
            afd_layer_major_u2=True,
            dbo_enabled=True,
            num_ubatches=2,
        ),
    )
    monkeypatch.setattr(
        hccl_module.torch.npu,
        "current_stream",
        lambda: active_stream[0],
    )
    monkeypatch.setattr(hccl_module.torch.npu, "stream", use_stream)
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda _tensor, *, dst, group, stream=None: calls.append(
            ("send_tensor", dst, group, stream)
        ),
    )
    monkeypatch.setattr(
        connector,
        "_recv_tensor",
        lambda _tensor, *, src, group, stream=None: calls.append(
            ("recv_tensor", src, group, stream)
        ),
    )
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)

    connector.send_attn_output(
        hidden,
        _attention_context(layer_idx=1, stage_idx=0, num_tokens=2),
    )
    connector.recv_ffn_output(hidden, ubatch_idx=0, layer_idx=1)

    assert calls == [
        ("enter", send_stream),
        ("send_tensor", 0, connector.data_pg_list[0], send_stream),
        ("send_done", "record", send_stream),
        ("exit", send_stream),
        ("enter", recv_stream),
        ("send_done", "wait", recv_stream),
        ("recv_tensor", 0, connector.data_pg_list[0], recv_stream),
        ("recv_done", "record", recv_stream),
        ("exit", recv_stream),
    ]


def test_p2p_hccl_graph_compute_pipeline_can_be_disabled_for_comparison(
    monkeypatch,
):
    monkeypatch.setenv("AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP", "0")
    connector = _connector(role="attention", num_ubatches=2)
    connector.a2f_send_stream = object()
    connector.f2a_recv_stream = object()
    connector.attention_pipeline_events = {(1, 0): object()}
    connector.attention_graph_compute_stream = object()
    connector.attention_graph_stream_plan = HCCLAttentionGraphStreamPlan(
        compute_stream=connector.attention_graph_compute_stream,
    )
    connector.attention_graph_events = {(1, 0): object()}
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            afd_graph_ubatching=True,
            afd_layer_major_u2=True,
            dbo_enabled=True,
            num_ubatches=2,
        ),
    )

    assert connector.graph_u2_compute_overlap_enabled is False
    assert connector.attention_graph_compute_pipeline_active() is False


def test_p2p_hccl_rejects_invalid_graph_compute_overlap_value(monkeypatch):
    monkeypatch.setenv("AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP", "invalid")

    with pytest.raises(
        RuntimeError,
        match="AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP must be 0 or 1",
    ):
        _connector(role="attention", num_ubatches=2)


def test_p2p_hccl_graph_hybrid_dag_can_be_disabled_for_comparison(monkeypatch):
    monkeypatch.setenv("AFD_HCCL_GRAPH_U2_HYBRID_DAG", "0")

    connector = _connector(role="attention", num_ubatches=2)

    assert connector.graph_u2_hybrid_dag_enabled is False


def test_p2p_hccl_rejects_invalid_graph_hybrid_dag_value(monkeypatch):
    monkeypatch.setenv("AFD_HCCL_GRAPH_U2_HYBRID_DAG", "invalid")

    with pytest.raises(
        RuntimeError,
        match="AFD_HCCL_GRAPH_U2_HYBRID_DAG must be 0 or 1",
    ):
        _connector(role="attention", num_ubatches=2)


@pytest.mark.parametrize(
    "name",
    [
        "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM",
        "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM",
        "AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER",
    ],
)
def test_p2p_hccl_rejects_invalid_graph_physical_pipeline_values(
    monkeypatch,
    name,
):
    monkeypatch.setenv(name, "invalid")

    with pytest.raises(RuntimeError, match=rf"{name} must be 0 or 1"):
        _connector(role="attention", num_ubatches=2)


def test_p2p_hccl_graph_physical_pipeline_defaults_are_enabled(monkeypatch):
    for name in (
        "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM",
        "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM",
        "AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER",
    ):
        monkeypatch.delenv(name, raising=False)
    connector = _connector(role="attention", num_ubatches=2)

    assert connector.graph_u2_attention_three_stream_enabled is True
    assert connector.graph_u2_ffn_recv_stream_enabled is True
    assert connector.graph_u2_ffn_cross_layer_enabled is True


def test_p2p_hccl_ffn_cross_layer_requires_receive_stream(monkeypatch):
    monkeypatch.setenv("AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM", "0")
    monkeypatch.setenv("AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER", "1")

    with pytest.raises(
        RuntimeError,
        match=(
            "AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=1 requires "
            "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=1"
        ),
    ):
        _connector(role="ffn", num_ubatches=2)


def test_p2p_hccl_attention_graph_compute_orders_fork_and_join(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)
    calls = []
    active_stream = [None]
    parent_stream = object()
    compute_stream = object()
    tensor = torch.ones((2, 4), dtype=torch.bfloat16)

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append((self.name, "record", stream))

        def wait(self, stream):
            calls.append((self.name, "wait", stream))

    @contextmanager
    def use_stream(stream):
        previous = active_stream[0]
        active_stream[0] = stream
        try:
            yield
        finally:
            active_stream[0] = previous

    connector.a2f_send_stream = object()
    connector.f2a_recv_stream = object()
    connector.attention_graph_compute_stream = compute_stream
    connector.attention_graph_stream_plan = HCCLAttentionGraphStreamPlan(
        compute_stream=compute_stream,
    )
    connector.attention_pipeline_events = {(1, 0): object()}
    connector.attention_graph_events = {
        (1, 0): hccl_module.HCCLAttentionGraphEvents(
            ready=FakeEvent("ready"),
            compute_done=FakeEvent("compute"),
            send_done=FakeEvent("send"),
            recv_done=FakeEvent("recv"),
        )
    }
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            afd_graph_ubatching=True,
            afd_layer_major_u2=True,
            dbo_enabled=True,
            num_ubatches=2,
        ),
    )
    monkeypatch.setattr(
        hccl_module.torch.npu,
        "current_stream",
        lambda: parent_stream,
    )
    monkeypatch.setattr(hccl_module.torch.npu, "stream", use_stream)
    monkeypatch.setattr(
        connector,
        "_record_stream",
        lambda value, stream: calls.append(("tensor", value, stream)),
    )

    with connector.attention_graph_compute(
        layer_idx=1,
        stage_idx=0,
        tensors=(tensor,),
    ):
        calls.append(("body", active_stream[0]))
    connector.wait_for_attention_graph_compute(
        layer_idx=1,
        stage_idx=0,
        tensors=(tensor,),
    )

    assert calls == [
        ("ready", "record", parent_stream),
        ("ready", "wait", compute_stream),
        ("tensor", tensor, compute_stream),
        ("body", compute_stream),
        ("compute", "record", compute_stream),
        ("compute", "wait", parent_stream),
        ("tensor", tensor, parent_stream),
    ]


def test_p2p_hccl_attention_graph_compute_waits_for_own_prior_receive(
    monkeypatch,
):
    connector = _connector(role="attention", num_ubatches=2)
    calls = []
    parent_stream = object()
    compute_stream = object()

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append((self.name, "record", stream))

        def wait(self, stream):
            calls.append((self.name, "wait", stream))

    @contextmanager
    def use_stream(_stream):
        yield

    connector.a2f_send_stream = object()
    connector.f2a_recv_stream = object()
    connector.attention_pipeline_events = {(0, 0): object()}
    connector.attention_graph_stream_plan = HCCLAttentionGraphStreamPlan(
        compute_stream=compute_stream,
    )
    connector.attention_graph_events = {
        (0, 0): hccl_module.HCCLAttentionGraphEvents(
            ready=FakeEvent("previous_ready"),
            compute_done=FakeEvent("previous_compute"),
            send_done=FakeEvent("previous_send"),
            recv_done=FakeEvent("previous_recv"),
        ),
        (1, 0): hccl_module.HCCLAttentionGraphEvents(
            ready=FakeEvent("current_ready"),
            compute_done=FakeEvent("current_compute"),
            send_done=FakeEvent("current_send"),
            recv_done=FakeEvent("current_recv"),
        ),
    }
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            afd_graph_ubatching=True,
            afd_layer_major_u2=True,
            dbo_enabled=True,
            num_ubatches=2,
        ),
    )
    monkeypatch.setattr(
        hccl_module.torch.npu,
        "current_stream",
        lambda: parent_stream,
    )
    monkeypatch.setattr(hccl_module.torch.npu, "stream", use_stream)

    with connector.attention_graph_compute(
        layer_idx=1,
        stage_idx=0,
        wait_for_receive_layer_idx=0,
    ):
        calls.append(("body", compute_stream))

    assert calls == [
        ("previous_recv", "wait", compute_stream),
        ("body", compute_stream),
        ("current_compute", "record", compute_stream),
    ]


def test_p2p_hccl_attention_stream_pipeline_is_inactive_while_compiling(
    monkeypatch,
):
    connector = _connector(role="attention", num_ubatches=2)
    connector.a2f_send_stream = object()
    connector.f2a_recv_stream = object()
    connector.attention_pipeline_events = {(1, 0): object()}
    monkeypatch.setattr(hccl_module.torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: (_ for _ in ()).throw(AssertionError("must not inspect context")),
    )

    assert connector._attention_stream_pipeline_active() is False


@pytest.mark.parametrize(
    ("is_compiling", "is_graph_capturing", "is_warmup", "uses_graph_ops"),
    [
        (False, False, False, False),
        (False, True, True, False),
        (False, True, False, True),
        (True, False, False, True),
    ],
)
def test_p2p_hccl_graph_transport_activation(
    monkeypatch,
    is_compiling,
    is_graph_capturing,
    is_warmup,
    uses_graph_ops,
):
    connector = _connector(role="attention")
    connector.is_graph_capturing = is_graph_capturing
    connector.is_warmup = is_warmup
    monkeypatch.setattr(
        hccl_module.torch.compiler,
        "is_compiling",
        lambda: is_compiling,
    )

    assert connector._graph_transport_active() is uses_graph_ops


def test_p2p_hccl_capture_uses_graph_send_recv(monkeypatch):
    connector = _connector(role="ffn")
    connector.is_graph_capturing = True
    graph_calls = []
    side_stream = object()
    monkeypatch.setattr(hccl_module.torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(
        connector,
        "_record_stream",
        lambda tensor, stream: graph_calls.append(("record", tensor, stream)),
    )
    monkeypatch.setattr(
        hccl_module,
        "_graph_hccl_send",
        lambda tensor, *, dst, group: graph_calls.append(("send", tensor, dst, group)),
    )
    monkeypatch.setattr(
        hccl_module,
        "_graph_hccl_recv",
        lambda tensor, *, src, group: graph_calls.append(("recv", tensor, src, group)),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda *_args, **_kwargs: pytest.fail("capture must not use dist.send"),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        lambda *_args, **_kwargs: pytest.fail("capture must not use dist.recv"),
    )
    tensor = torch.ones((2, 4), dtype=torch.bfloat16)

    connector._send_tensor(
        tensor,
        dst=1,
        group=connector.data_pg_list[0],
        stream=side_stream,
    )
    connector._recv_tensor(
        tensor,
        src=1,
        group=connector.data_pg_list[0],
        stream=side_stream,
    )

    assert graph_calls == [
        ("record", tensor, side_stream),
        ("send", tensor, 1, connector.data_pg_list[0]),
        ("record", tensor, side_stream),
        ("recv", tensor, 1, connector.data_pg_list[0]),
    ]


def test_p2p_hccl_dynamic_ubatch_capture_uses_graph_send_recv(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)
    graph_calls = []
    monkeypatch.setattr(hccl_module.torch.compiler, "is_compiling", lambda: False)
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(afd_graph_ubatching=True),
    )
    monkeypatch.setattr(
        hccl_module.torch.npu,
        "is_current_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(
        hccl_module,
        "_graph_hccl_send",
        lambda tensor, *, dst, group: graph_calls.append(("send", tensor, dst, group)),
    )
    monkeypatch.setattr(
        hccl_module,
        "_graph_hccl_recv",
        lambda tensor, *, src, group: graph_calls.append(("recv", tensor, src, group)),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda *_args, **_kwargs: pytest.fail("capture must not use dist.send"),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        lambda *_args, **_kwargs: pytest.fail("capture must not use dist.recv"),
    )
    tensor = torch.ones((2, 4), dtype=torch.bfloat16)

    connector._send_tensor(tensor, dst=1, group=connector.data_pg_list[0])
    connector._recv_tensor(tensor, src=1, group=connector.data_pg_list[0])

    assert graph_calls == [
        ("send", tensor, 1, connector.data_pg_list[0]),
        ("recv", tensor, 1, connector.data_pg_list[0]),
    ]


def test_p2p_hccl_attention_uses_mapped_ffn_rank(monkeypatch):
    connector = _connector(
        role="attention",
        role_rank=3,
        attention=4,
        ffn=2,
    )
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(afd_input_ids_pretransferred=True),
    )
    events = []
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda _tensor, *, dst, group: events.append(("send", dst, group)),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        lambda _tensor, *, src, group: events.append(("recv", src, group)),
    )
    monkeypatch.setattr(hccl_module, "maybe_apply_dbo_yield", lambda *_a, **_k: None)

    connector.send_attn_output(
        torch.ones((2, 4), dtype=torch.bfloat16),
        _attention_context(layer_idx=1, stage_idx=0, num_tokens=2),
    )
    connector.recv_ffn_output(torch.empty((2, 4), dtype=torch.bfloat16))

    assert events == [
        ("send", 1, connector.data_pg_list[0]),
        ("recv", 1, connector.data_pg_list[0]),
    ]


def test_p2p_hccl_attention_fans_out_ids_hidden_and_gathers_output(monkeypatch):
    connector = _connector(role="attention", attention=1, ffn=2)
    sent_ids = []
    sent_hidden = []
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda tensor, *, dst, group: sent_ids.append(
            (tensor.clone(), dst, group),
        ),
    )
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda tensor, *, dst, group: sent_hidden.append(
            (tensor.clone(), dst, group),
        ),
    )
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)

    connector.send_input_ids(torch.arange(5), ubatch_idx=0)
    connector._send_attention_tensor(hidden, group=connector.data_pg_list[0])

    assert [(dst, tensor.tolist()) for tensor, dst, _group in sent_ids] == [
        (0, [0, 1, 2]),
        (1, [3, 4]),
    ]
    assert [(dst, tuple(tensor.shape)) for tensor, dst, _group in sent_hidden] == [
        (0, (3, 4)),
        (1, (2, 4)),
    ]
    assert torch.equal(sent_hidden[0][0], hidden[:3])
    assert torch.equal(sent_hidden[1][0], hidden[3:])

    def recv(tensor, *, src, group):
        assert group is connector.data_pg_list[0]
        tensor.fill_(src + 1)

    monkeypatch.setattr(connector, "_recv_tensor", recv)
    output = torch.empty_like(hidden)
    connector._recv_attention_tensor(output, group=connector.data_pg_list[0])

    assert output[:, 0].tolist() == [1, 1, 1, 2, 2]


def test_p2p_hccl_attention_fanout_pads_and_discards_dummy_token(monkeypatch):
    connector = _connector(role="attention", attention=1, ffn=2)
    sent = []
    monkeypatch.setattr(
        connector,
        "_send_tensor",
        lambda tensor, *, dst, group: sent.append((tensor.clone(), dst, group)),
    )
    hidden = torch.full((1, 4), 7, dtype=torch.float32)

    connector._send_attention_tensor(hidden, group=connector.data_pg_list[0])

    assert [tuple(tensor.shape) for tensor, _dst, _group in sent] == [(1, 4), (1, 4)]
    assert torch.equal(sent[0][0], hidden)
    assert torch.equal(sent[1][0], torch.zeros_like(hidden))

    def recv(tensor, *, src, group):
        tensor.fill_(src + 11)

    monkeypatch.setattr(connector, "_recv_tensor", recv)
    output = torch.empty_like(hidden)
    connector._recv_attention_tensor(output, group=connector.data_pg_list[0])

    assert output.tolist() == [[11, 11, 11, 11]]


def test_p2p_hccl_ffn_receives_ids_then_hidden_and_returns_state(monkeypatch):
    connector = _connector(role="ffn")
    connector.dp_metadata_list = {
        0: AFDDPMetadata(torch.tensor([3], dtype=torch.int32)),
    }
    events = []
    hidden_buffer = torch.empty((3, 4), dtype=torch.bfloat16)
    connector.hidden_recv_buffers[0] = hidden_buffer

    def recv(tensor, *, src, group):
        if tensor.dtype == torch.int32:
            events.append(("ids", src, group))
            tensor.copy_(torch.tensor([-1, 0, 31], dtype=torch.int32))
        else:
            events.append(("hidden", src, group))
            tensor.fill_(2)
        return src

    monkeypatch.setattr(hccl_module.dist, "recv", recv)
    payload = connector.recv_attn_output(ubatch_idx=0, layer_idx=0)

    assert events == [
        ("ids", 1, connector.ids_pg_list[0]),
        ("hidden", 1, connector.data_pg_list[0]),
    ]
    assert payload.input_ids.tolist() == [-1, 0, 31]
    assert torch.equal(payload.hidden_states, torch.full_like(hidden_buffer, 2))
    assert isinstance(payload.context.states, HCCLP2PTransferState)
    assert payload.context.states.num_tokens == 3
    assert payload.context.states.peer_ranks == (1,)
    assert payload.context.states.seq_lens == (3,)
    assert payload.context.states.peer_slices == ((1, 0, 3),)


def test_p2p_hccl_ffn_aggregates_unequal_peer_payloads_and_splits_output(
    monkeypatch,
):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=4,
        ffn=2,
        max_num_batched_tokens=12,
    )
    connector.dp_metadata_list = {
        0: AFDDPMetadata(torch.tensor([2, 3, 4, 5], dtype=torch.int32)),
    }
    connector.hidden_recv_buffers[0] = torch.empty((9, 4), dtype=torch.bfloat16)
    events = []

    def recv(tensor, *, src, group):
        kind = "ids" if tensor.dtype == torch.int32 else "hidden"
        events.append((kind, src, tuple(tensor.shape), group))
        tensor.fill_(src)

    sent = []
    monkeypatch.setattr(hccl_module.dist, "recv", recv)
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda tensor, *, dst, group: sent.append(
            (tensor.clone(), dst, group),
        ),
    )

    payload = connector.recv_attn_output(ubatch_idx=0, layer_idx=0)

    assert events == [
        ("ids", 4, (4,), connector.ids_pg_list[0]),
        ("ids", 5, (5,), connector.ids_pg_list[0]),
        ("hidden", 4, (4, 4), connector.data_pg_list[0]),
        ("hidden", 5, (5, 4), connector.data_pg_list[0]),
    ]
    assert payload.context.metadata.seq_lens == [4, 5]
    assert payload.input_ids.tolist() == [4] * 4 + [5] * 5
    assert payload.hidden_states[:, 0].tolist() == [4] * 4 + [5] * 5
    assert payload.context.states == HCCLP2PTransferState(
        stage_idx=0,
        num_tokens=9,
        peer_ranks=(4, 5),
        seq_lens=(4, 5),
        peer_slices=((4, 0, 4), (5, 4, 9)),
    )

    ffn_output = torch.arange(36, dtype=torch.float32).reshape(9, 4)
    connector.send_ffn_output(ffn_output, payload.context)

    assert [(dst, tensor.shape[0]) for tensor, dst, _group in sent] == [
        (4, 4),
        (5, 5),
    ]
    assert torch.equal(sent[0][0], ffn_output[:4])
    assert torch.equal(sent[1][0], ffn_output[4:])


def test_p2p_hccl_ffn_graph_uses_each_unequal_peer_slice(monkeypatch):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=4,
        ffn=2,
        max_num_batched_tokens=12,
    )
    connector.is_graph_capturing = True
    connector.dp_metadata_list = {
        0: AFDDPMetadata(torch.tensor([2, 3, 4, 5], dtype=torch.int32)),
    }
    connector.hidden_recv_buffers[0] = torch.empty((9, 4), dtype=torch.bfloat16)
    graph_calls = []
    monkeypatch.setattr(hccl_module.torch.compiler, "is_compiling", lambda: False)

    def graph_recv(tensor, *, src, group):
        graph_calls.append(("recv", src, tuple(tensor.shape), group))

    def graph_send(tensor, *, dst, group):
        graph_calls.append(("send", dst, tuple(tensor.shape), group))

    monkeypatch.setattr(hccl_module, "_graph_hccl_recv", graph_recv)
    monkeypatch.setattr(hccl_module, "_graph_hccl_send", graph_send)
    monkeypatch.setattr(
        hccl_module.dist,
        "recv",
        lambda *_args, **_kwargs: pytest.fail("capture must not use dist.recv"),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda *_args, **_kwargs: pytest.fail("capture must not use dist.send"),
    )

    payload = connector.recv_attn_output(ubatch_idx=0, layer_idx=1)
    connector.send_ffn_output(payload.hidden_states, payload.context)

    assert graph_calls == [
        ("recv", 4, (4, 4), connector.data_pg_list[0]),
        ("recv", 5, (5, 4), connector.data_pg_list[0]),
        ("send", 4, (4, 4), connector.data_pg_list[0]),
        ("send", 5, (5, 4), connector.data_pg_list[0]),
    ]


def test_p2p_hccl_ffn_send_uses_matching_stage_and_attention_rank(monkeypatch):
    connector = _connector(role="ffn", num_ubatches=2)
    sent = []
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda tensor, *, dst, group: sent.append((tensor.clone(), dst, group)),
    )
    context = AFDTransferContext(
        metadata=AFDTransferMetadata.create_ffn_metadata(
            layer_idx=2,
            stage_idx=1,
            seq_lens=[3],
        ),
        states=HCCLP2PTransferState(
            stage_idx=1,
            num_tokens=3,
            peer_ranks=(1,),
            seq_lens=(3,),
        ),
    )

    connector.send_ffn_output(
        torch.ones((3, 4), dtype=torch.bfloat16),
        context,
        ubatch_idx=1,
    )

    assert sent[0][1:] == (1, connector.data_pg_list[1])


def test_p2p_hccl_ffn_stream_pipeline_keeps_sync_send_recv(monkeypatch):
    connector = _connector(role="ffn", num_ubatches=2)
    connector.dp_metadata_list = {
        1: AFDDPMetadata(torch.tensor([2], dtype=torch.int32)),
    }
    connector.hidden_recv_buffers[1] = torch.empty((2, 4), dtype=torch.bfloat16)
    calls = []
    active_stream = [None]
    recv_stream = object()
    send_stream = object()

    class FakeEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            calls.append((self.name, "record", stream))

        def wait(self, stream):
            calls.append((self.name, "wait", stream))

    @contextmanager
    def use_stream(stream):
        previous = active_stream[0]
        active_stream[0] = stream
        try:
            yield
        finally:
            active_stream[0] = previous

    monkeypatch.setattr(hccl_module.torch.npu, "stream", use_stream)

    def recv(tensor, *, src, group):
        calls.append(("dist.recv", active_stream[0], src, group))
        tensor.fill_(2)

    monkeypatch.setattr(hccl_module.dist, "recv", recv)
    monkeypatch.setattr(
        hccl_module.dist,
        "send",
        lambda _tensor, *, dst, group: calls.append(
            ("dist.send", active_stream[0], dst, group)
        ),
    )
    previous_send = FakeEvent("previous_send")
    recv_done = FakeEvent("recv_done")
    payload, _ = connector.recv_attn_output_streamed(
        ubatch_idx=1,
        layer_idx=1,
        max_num_tokens=2,
        recv_stream=recv_stream,
        wait_event=previous_send,
        done_event=recv_done,
    )
    compute_done = FakeEvent("compute_done")
    send_done = FakeEvent("send_done")
    connector.send_ffn_output_streamed(
        torch.ones((2, 4), dtype=torch.bfloat16),
        payload.context,
        ubatch_idx=1,
        send_stream=send_stream,
        wait_event=compute_done,
        done_event=send_done,
    )

    assert calls == [
        ("previous_send", "wait", recv_stream),
        ("dist.recv", recv_stream, 1, connector.data_pg_list[1]),
        ("recv_done", "record", recv_stream),
        ("compute_done", "wait", send_stream),
        ("dist.send", send_stream, 1, connector.data_pg_list[1]),
        ("send_done", "record", send_stream),
    ]


def test_p2p_hccl_control_plane_prepares_stage_buffers(monkeypatch):
    connector = _connector(role="ffn", num_ubatches=2)
    prepared = []
    monkeypatch.setattr(
        connector,
        "prepare_stage_buffer",
        lambda stage_idx, num_tokens: prepared.append((stage_idx, num_tokens)),
    )
    payload = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(torch.tensor([3], dtype=torch.int32)),
            1: AFDDPMetadata(torch.tensor([5], dtype=torch.int32)),
        },
        is_graph_capturing=False,
        is_warmup=False,
    )

    connector.control_plane.update_state_from_dp_metadata(payload)

    assert prepared == [(0, 3), (1, 5)]
    assert connector.dp_metadata_list == payload.dp_metadata_list


def test_p2p_hccl_control_plane_prepares_aggregate_unequal_buffers(monkeypatch):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=4,
        ffn=2,
        num_ubatches=2,
    )
    prepared = []
    monkeypatch.setattr(
        connector,
        "prepare_stage_buffer",
        lambda stage_idx, num_tokens: prepared.append((stage_idx, num_tokens)),
    )
    payload = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(torch.tensor([2, 3, 4, 5], dtype=torch.int32)),
            1: AFDDPMetadata(torch.tensor([7, 6, 1, 2], dtype=torch.int32)),
        },
        is_graph_capturing=False,
        is_warmup=False,
    )

    connector.control_plane.update_state_from_dp_metadata(payload)

    assert prepared == [(0, 9), (1, 3)]
    assert connector.stage_layouts[0].peer_slices == ((4, 0, 4), (5, 4, 9))
    assert connector.stage_layouts[1].peer_slices == ((4, 0, 1), (5, 1, 3))


@pytest.mark.parametrize(
    ("role_rank", "expected_peer", "expected_tokens"),
    [
        (0, 4, 3),
        (1, 4, 2),
        (2, 5, 2),
        (3, 5, 1),
    ],
)
def test_p2p_hccl_control_plane_prepares_ffn_fanout_shard(
    monkeypatch,
    role_rank,
    expected_peer,
    expected_tokens,
):
    connector = _connector(
        role="ffn",
        role_rank=role_rank,
        attention=2,
        ffn=4,
    )
    prepared = []
    monkeypatch.setattr(
        connector,
        "prepare_stage_buffer",
        lambda stage_idx, num_tokens: prepared.append((stage_idx, num_tokens)),
    )
    payload = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(torch.tensor([5, 3], dtype=torch.int32)),
        },
        is_graph_capturing=False,
        is_warmup=False,
    )

    connector.control_plane.update_state_from_dp_metadata(payload)

    layout = connector.stage_layouts[0]
    assert layout.peer_ranks == (expected_peer,)
    assert layout.seq_lens == (expected_tokens,)
    assert prepared == [(0, expected_tokens)]


def test_p2p_hccl_reuses_control_plane_stage_layout(monkeypatch):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=4,
        ffn=2,
    )
    prepared = []
    calls = []
    original = hccl_module._attention_token_counts
    monkeypatch.setattr(
        connector,
        "prepare_stage_buffer",
        lambda stage_idx, num_tokens: prepared.append((stage_idx, num_tokens)),
    )

    def record_counts(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(hccl_module, "_attention_token_counts", record_counts)
    payload = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(torch.tensor([2, 3, 4, 5], dtype=torch.int32)),
        },
        is_graph_capturing=False,
        is_warmup=False,
    )

    connector.control_plane.update_state_from_dp_metadata(payload)
    first = connector._stage_layout(0, fallback=1)
    second = connector._stage_layout(0, fallback=99)

    assert first is second
    assert first.seq_lens == (4, 5)
    assert prepared == [(0, 9)]
    assert len(calls) == 1


def test_p2p_hccl_control_plane_uses_one_sender_per_subgroup(monkeypatch):
    payload = AFDControlPayload(
        dp_metadata_list={},
        is_graph_capturing=False,
        is_warmup=False,
    )
    sent = []
    monkeypatch.setattr(
        hccl_module,
        "send_control_payload",
        lambda value, *, dst, group, device: sent.append(
            (value, dst, group, device),
        ),
    )

    for role_rank in range(4):
        connector = _connector(
            role="attention",
            role_rank=role_rank,
            attention=4,
            ffn=2,
        )
        connector.p2p_pg = object()
        connector.control_plane.send_dp_metadata_list(payload)

    assert [(dst, device.type) for _value, dst, _group, device in sent] == [
        (0, "cpu"),
        (1, "cpu"),
    ]


def test_p2p_hccl_control_plane_receives_from_first_subgroup_attention(monkeypatch):
    connector = _connector(
        role="ffn",
        role_rank=1,
        attention=4,
        ffn=2,
    )
    connector.p2p_pg = object()
    expected = AFDControlPayload(
        dp_metadata_list={},
        is_graph_capturing=False,
        is_warmup=False,
    )
    calls = []

    def recv(*, src, group, device):
        calls.append((src, group, device))
        return expected

    monkeypatch.setattr(hccl_module, "recv_control_payload", recv)

    assert connector.control_plane.recv_dp_metadata_list() is expected
    assert calls == [(4, connector.p2p_pg, torch.device("cpu"))]


def test_p2p_hccl_control_plane_fans_out_from_one_attention(monkeypatch):
    payload = AFDControlPayload(
        dp_metadata_list={},
        is_graph_capturing=False,
        is_warmup=False,
    )
    attention = _connector(role="attention", attention=1, ffn=2)
    attention.p2p_pg = object()
    sent = []
    monkeypatch.setattr(
        hccl_module,
        "send_control_payload",
        lambda value, *, dst, group, device: sent.append(
            (value, dst, group, device),
        ),
    )

    attention.control_plane.send_dp_metadata_list(payload)

    assert [(dst, device.type) for _value, dst, _group, device in sent] == [
        (0, "cpu"),
        (1, "cpu"),
    ]

    for role_rank in (0, 1):
        ffn = _connector(
            role="ffn",
            role_rank=role_rank,
            attention=1,
            ffn=2,
        )
        ffn.p2p_pg = object()
        received = []
        monkeypatch.setattr(
            hccl_module,
            "recv_control_payload",
            lambda *, src, group, device, received=received: (
                received.append(src) or payload
            ),
        )
        assert ffn.control_plane.recv_dp_metadata_list() is payload
        assert received == [2]


@pytest.mark.parametrize(
    ("attention_size", "ratio"),
    [(48, 3), (64, 4)],
)
def test_p2p_hccl_scaleout_control_plane_has_one_sender_per_ffn(
    monkeypatch,
    attention_size,
    ratio,
):
    payload = AFDControlPayload(
        dp_metadata_list={},
        is_graph_capturing=False,
        is_warmup=False,
    )
    active_sender = -1
    sent = []

    def send(value, *, dst, group, device):
        sent.append((active_sender, value, dst, group, device))

    monkeypatch.setattr(hccl_module, "send_control_payload", send)
    for role_rank in range(attention_size):
        active_sender = role_rank
        connector = _connector(
            role="attention",
            role_rank=role_rank,
            attention=attention_size,
            ffn=16,
        )
        connector.p2p_pg = object()
        connector.control_plane.send_dp_metadata_list(payload)

    assert [(source, destination) for source, _, destination, _, _ in sent] == [
        (ffn_rank * ratio, ffn_rank) for ffn_rank in range(16)
    ]
    assert all(value is payload for _, value, _, _, _ in sent)
    assert all(device == torch.device("cpu") for _, _, _, _, device in sent)


@pytest.mark.parametrize(
    ("attention_size", "ffn_rank", "expected_peer_world_ranks"),
    [
        (48, 5, (31, 32, 33)),
        (48, 10, (46, 47, 48)),
        (48, 15, (61, 62, 63)),
        (64, 3, (28, 29, 30, 31)),
        (64, 4, (32, 33, 34, 35)),
        (64, 15, (76, 77, 78, 79)),
    ],
)
def test_p2p_hccl_scaleout_token_fan_in_matches_static_peer_contract(
    monkeypatch,
    attention_size,
    ffn_rank,
    expected_peer_world_ranks,
):
    connector = _connector(
        role="ffn",
        role_rank=ffn_rank,
        attention=attention_size,
        ffn=16,
        max_num_batched_tokens=256,
    )
    prepared = []
    monkeypatch.setattr(
        connector,
        "prepare_stage_buffer",
        lambda stage_idx, num_tokens: prepared.append((stage_idx, num_tokens)),
    )
    counts = torch.arange(1, attention_size + 1, dtype=torch.int32)
    connector.control_plane.update_state_from_dp_metadata(
        AFDControlPayload(
            dp_metadata_list={0: AFDDPMetadata(counts)},
            is_graph_capturing=False,
            is_warmup=False,
        ),
    )

    first_attention_rank = ffn_rank * connector.ratio
    expected_seq_lens = tuple(
        range(
            first_attention_rank + 1,
            first_attention_rank + connector.ratio + 1,
        ),
    )
    layout = connector.stage_layouts[0]
    assert layout.peer_ranks == expected_peer_world_ranks
    assert layout.seq_lens == expected_seq_lens
    assert layout.num_tokens == sum(expected_seq_lens)
    assert layout.peer_slices[-1][2] == sum(expected_seq_lens)
    assert prepared == [(0, sum(expected_seq_lens))]


def test_p2p_hccl_mtp_phase_marker_uses_control_sender(monkeypatch):
    connector = _connector(
        role="attention",
        role_rank=2,
        attention=4,
        ffn=2,
    )
    connector.p2p_pg = object()
    sent = []
    monkeypatch.setattr(
        hccl_module,
        "send_control_payload",
        lambda value, *, dst, group, device: sent.append(
            (value, dst, group, device),
        ),
    )

    connector.control_plane.send_mtp_phase_ready(graph_replay=True)

    assert len(sent) == 1
    payload, dst, group, device = sent[0]
    assert payload.mtp_phase_ready is True
    assert payload.mtp_phase_graph_replay is True
    assert payload.dp_metadata_list == {}
    assert dst == 1
    assert group is connector.p2p_pg
    assert device == torch.device("cpu")


def test_p2p_hccl_mtp_phase_receive_stashes_next_target(monkeypatch):
    connector = _connector(role="ffn")
    connector.p2p_pg = object()
    next_target = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(torch.tensor([3], dtype=torch.int32)),
        },
        is_graph_capturing=False,
        is_warmup=False,
        mtp_phase_control_enabled=True,
    )
    monkeypatch.setattr(
        connector.control_plane,
        "_recv_payload",
        lambda: next_target,
    )

    assert connector.control_plane.recv_mtp_phase_ready() is False
    assert connector.control_plane.recv_dp_metadata_list() is next_target
    assert connector.mtp_phase_control_enabled is True


def test_p2p_hccl_mtp_phase_receive_consumes_marker(monkeypatch):
    connector = _connector(role="ffn")
    marker = AFDControlPayload(
        dp_metadata_list={},
        is_graph_capturing=False,
        is_warmup=False,
        mtp_phase_ready=True,
        mtp_phase_graph_replay=True,
    )
    monkeypatch.setattr(
        connector.control_plane,
        "_recv_payload",
        lambda: marker,
    )

    assert connector.control_plane.recv_mtp_phase_ready() is True
    assert connector.mtp_phase_graph_replay is True
    assert connector.control_plane._pending_payload is None


def test_p2p_hccl_rejects_aggregate_buffer_overflow(monkeypatch):
    connector = _connector(
        role="ffn",
        attention=2,
        ffn=1,
        max_num_batched_tokens=8,
    )
    monkeypatch.setattr(
        hccl_module.torch,
        "empty",
        lambda *_args, **_kwargs: pytest.fail("must fail before allocating"),
    )
    payload = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(torch.tensor([4, 5], dtype=torch.int32)),
        },
        is_graph_capturing=False,
        is_warmup=False,
    )

    with pytest.raises(ValueError, match="aggregate token count.*increase FFN"):
        connector.control_plane.update_state_from_dp_metadata(payload)


def test_p2p_hccl_close_destroys_all_groups(monkeypatch):
    connector = _connector(role="attention", num_ubatches=2)
    connector.p2p_pg = object()
    groups = [
        connector.p2p_pg,
        *connector.ids_pg_list,
        *connector.data_pg_list,
    ]
    destroyed = []
    monkeypatch.setattr(
        hccl_module.dist,
        "destroy_process_group",
        lambda group: destroyed.append(group),
    )

    connector.close()

    assert destroyed == groups
    assert connector.data_pg_list == []
    assert connector.ids_pg_list == []
    assert connector.input_ids_buffers == []
    assert connector.hidden_recv_buffers == {}
    assert connector.stage_layouts == {}
    assert connector.mtp_stage_layouts == {}
    assert connector.is_initialized is False


def test_p2p_hccl_partial_init_failure_destroys_created_groups(monkeypatch):
    connector = P2pHcclAFDConnector(
        0,
        0,
        _vllm_config(),
        _afd_config(role="attention"),
        0,
    )
    data_group = object()
    calls = iter((data_group, RuntimeError("IDs group failed")))

    def init_group(**_kwargs):
        result = next(calls)
        if isinstance(result, BaseException):
            raise result
        return result

    destroyed = []
    monkeypatch.setattr(hccl_module, "init_afd_process_group", init_group)
    monkeypatch.setattr(
        hccl_module.dist,
        "destroy_process_group",
        lambda group: destroyed.append(group),
    )

    with pytest.raises(RuntimeError, match="IDs group failed"):
        connector.init_afd_connector()

    assert destroyed == [data_group]
    assert connector.data_pg_list == []
    assert connector.ids_pg_list == []
    assert connector.is_initialized is False


def test_p2p_hccl_graph_init_registers_hccl_graph_ops_before_groups(monkeypatch):
    connector = P2pHcclAFDConnector(
        0,
        0,
        _vllm_config(enforce_eager=False),
        _afd_config(role="attention"),
        0,
    )
    calls = []
    monkeypatch.setattr(
        hccl_module,
        "_ensure_graph_hccl_ops_registered",
        lambda: calls.append("register"),
    )
    monkeypatch.setattr(
        hccl_module,
        "init_afd_process_group",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("stop after register")),
    )

    with pytest.raises(RuntimeError, match="stop after register"):
        connector.init_afd_connector()

    assert calls == ["register"]


@pytest.mark.parametrize("input_ids", [[-2], [32]])
def test_p2p_hccl_rejects_invalid_input_ids(monkeypatch, input_ids):
    connector = _connector(role="attention")
    monkeypatch.setattr(
        hccl_module,
        "get_forward_context",
        lambda: SimpleNamespace(afd_input_ids_pretransferred=False),
    )

    with pytest.raises(ValueError, match="-1 padding"):
        connector.send_attn_output(
            torch.ones((1, 4), dtype=torch.bfloat16),
            _attention_context(layer_idx=0, stage_idx=0, num_tokens=1),
            input_ids=torch.tensor(input_ids),
        )


def test_p2p_hccl_rejects_non_integer_input_ids_before_send():
    connector = _connector(role="attention")

    with pytest.raises(TypeError, match="int32 or int64"):
        connector._validate_input_ids(torch.ones(1, dtype=torch.float32), 1)


def test_p2p_hccl_device_input_ids_skip_host_value_readback():
    connector = _connector(role="attention")
    device_ids = torch.empty(1, dtype=torch.int64, device="meta")

    connector._validate_input_ids(device_ids, 1)


def test_p2p_hccl_module_has_no_camp2p_custom_op_reference():
    source = __import__("inspect").getsource(hccl_module)

    assert "torch.ops.vllm.afd_camp2p" not in source
    assert "torch.ops.afd_ascend" not in source


def test_p2p_hccl_graph_lowering_omits_dynamic_shape_guard(monkeypatch):
    calls = []

    class FakeOp:
        def default(self, *args):
            calls.append(args)
            return args[0]

    monkeypatch.setattr(
        hccl_module.torch,
        "ops",
        SimpleNamespace(
            npu_define=SimpleNamespace(_send=FakeOp(), _recv=FakeOp()),
        ),
    )
    monkeypatch.setattr(
        hccl_module.dist,
        "get_process_group_ranks",
        lambda group: [0, 1] if group == "data-group" else pytest.fail(),
    )
    monkeypatch.setattr(
        hccl_module.c10d,
        "_get_group_tag",
        lambda group: "afd-data" if group == "data-group" else pytest.fail(),
    )
    tensor = torch.zeros((2, 4), dtype=torch.bfloat16)

    hccl_module._graph_hccl_send(tensor, dst=1, group="data-group")
    hccl_module._graph_hccl_recv(tensor, src=0, group="data-group")

    assert calls[0][1:] == (1, [0, 1], "afd-data", 0, None, None)
    assert calls[1][1:] == (0, [0, 1], "afd-data", 0, None, None)
