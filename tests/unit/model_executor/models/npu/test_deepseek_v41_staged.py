# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU execution/state seams, using native forward as the sequence reference."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.forward_context import (  # noqa: E402
    get_forward_context,
    override_forward_context,
)

from afd_plugin.model_executor.models.npu import (  # noqa: E402
    deepseek_v41_staged as module,
)
from afd_plugin.model_executor.models.npu.deepseek_v41_staged import (  # noqa: E402
    DeepseekV41EngramInputs,
    run_deepseek_v41_stages,
)
from tests.unit.fakes.deepseek_v41 import (  # noqa: E402
    contract_ffn,
    make_staged_case,
    make_staged_progress,
)
from tests.unit.fakes.hccl import StagedModelPeer  # noqa: E402
from tests.unit.fakes.hccl_eager_model import make_eager_model_case  # noqa: E402
from tests.unit.fakes.hccl_eager_pipeline import make_eager_progress  # noqa: E402


@pytest.fixture
def case():
    return make_staged_case()


@pytest.mark.parametrize("split", [(2, 3), (0, 3), (3, 0), (0, 0)])
@pytest.mark.parametrize("engram", [False, True])
def test_eager_loans_preserve_native_hc_aux_and_parent_token_order(
    monkeypatch, split, engram
):
    setup = make_eager_model_case(monkeypatch, module, split=split, engram=engram)
    case = setup.case
    if sum(split):
        expected = case.model(**case.parent_inputs)
        expected_mtp = case.mtp_rows.clone()
    else:
        empty = torch.empty((0, 4), dtype=torch.bfloat16)
        expected = (empty, [empty, empty])
        expected_mtp = torch.empty((0, 8), dtype=torch.bfloat16)

    output = run_deepseek_v41_stages(
        case.model,
        case.stages,
        case.contexts,
        case.slices,
        setup.peer,
        pipeline=setup.pipeline,
        compute_stream=setup.runtime.compute,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(case.mtp_rows, expected_mtp, rtol=0, atol=0)
    assert not setup.pipeline.outputs
    assert setup.runtime.current == setup.runtime.parent
    assert not setup.pipeline.finished  # The execution owner completes the parent.


def test_eager_posts_both_stages_before_consuming_the_first_output(monkeypatch):
    setup = make_eager_model_case(monkeypatch, module, valid=(1, 1))
    case = setup.case
    original_ids = case.parent_inputs["input_ids"].clone()

    run_deepseek_v41_stages(
        case.model,
        case.stages,
        case.contexts,
        case.slices,
        setup.peer,
        pipeline=setup.pipeline,
        compute_stream=setup.runtime.compute,
    )

    assert [message[:3] for message in setup.pipeline.messages] == [
        ("post", 0, 0),
        ("post", 0, 1),
        ("consume", 0, 0),
        ("post", 1, 0),
        ("consume", 0, 1),
        ("post", 1, 1),
        ("consume", 1, 0),
        ("consume", 1, 1),
    ]
    torch.testing.assert_close(case.parent_inputs["input_ids"], original_ids)
    for kind, _, _stage, *payload in setup.pipeline.messages:
        if kind == "post":
            hidden, fp32, ids = payload
            assert hidden.dtype == torch.bfloat16 and fp32.dtype == torch.float32
            assert torch.any(fp32 != hidden.float())
            assert ids[1:].count_nonzero() == 0


def test_eager_output_failure_restores_native_bindings_and_marks_owner_terminal(
    monkeypatch,
):
    setup = make_eager_model_case(monkeypatch, module, fail_step=(0, 1))
    case = setup.case
    parent = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE, ubatch_idx=-1)
    original = case.model.shared_attention_state

    with override_forward_context(parent):
        with pytest.raises(RuntimeError, match="injected eager output failure"):
            run_deepseek_v41_stages(
                case.model,
                case.stages,
                case.contexts,
                case.slices,
                setup.peer,
                pipeline=setup.pipeline,
                compute_stream=setup.runtime.compute,
            )
        assert get_forward_context() is parent

    assert setup.peer.terminal and setup.pipeline.failed
    assert setup.runtime.current == setup.runtime.parent
    assert case.model.shared_attention_state is original
    assert all(layer.self_attn.shared_state is original for layer in case.model.layers)


def test_interleaved_continuations_preserve_native_hc_and_delayed_pre_mix(case):
    layer = case.model.layers[0]
    hidden = [
        case.model.embed_input_ids(stage.input_ids).unsqueeze(1).repeat(1, 2, 1)
        for stage in case.stages
    ]
    pre = [torch.tensor([[1.0, 0.0]]).repeat(rows.shape[0], 1) for rows in hidden]
    expected = [
        layer(stage.positions, rows, mix, input_ids=stage.input_ids)
        for stage, rows, mix in zip(case.stages, hidden, pre, strict=True)
    ]

    pending = [
        layer.prepare_remote_ffn(stage.positions, rows, mix)
        for stage, rows, mix in zip(case.stages, hidden, pre, strict=True)
    ]
    for index in (1, 0):
        x, fp32, continuation = pending[index]
        output = layer.complete_remote_ffn(
            contract_ffn(x, fp32, case.stages[index].input_ids), continuation
        )
        torch.testing.assert_close(output, expected[index], rtol=0, atol=0)


@pytest.mark.parametrize("split", [(5,), (2, 3), (0, 3), (3, 0), (0, 0)])
@pytest.mark.parametrize("engram", [False, True])
def test_stages_match_native_forward_and_aux_layer_entry(split, engram):
    case = make_staged_case(split, engram=engram)
    if sum(split):
        expected = case.model(**case.parent_inputs)
        expected_mtp = case.mtp_rows.clone()
    else:
        empty = torch.empty((0, 4), dtype=torch.bfloat16)
        expected = (empty, [empty, empty])
        expected_mtp = torch.empty((0, 8), dtype=torch.bfloat16)
    peer = StagedModelPeer(case.plan)

    output = run_deepseek_v41_stages(
        case.model, case.stages, case.contexts, case.slices, peer
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(case.mtp_rows, expected_mtp, rtol=0, atol=0)
    assert [(layer, stage) for layer, stage, *_ in peer.messages] == [
        (layer, stage) for layer in (0, 1) for stage, rows in enumerate(split) if rows
    ]


@pytest.mark.parametrize("split", [(0, 3), (3, 0), (0, 0)])
def test_locally_empty_stage_still_participates_in_global_payload_order(split):
    case = make_staged_case(split, remote_active=True)
    peer = StagedModelPeer(case.plan)

    run_deepseek_v41_stages(case.model, case.stages, case.contexts, case.slices, peer)

    assert [(layer, stage) for layer, stage, *_ in peer.messages] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    for _, stage, hidden, fp32, ids in peer.messages:
        assert hidden.shape == fp32.shape == (split[stage], 4)
        assert hidden.dtype == torch.bfloat16 and fp32.dtype == torch.float32
        assert ids.shape == (split[stage],)


def test_transfer_padding_sanitizes_a_private_copy_of_parent_raw_ids():
    case = make_staged_case(valid=(1, 1))
    case.parent_inputs["input_ids"][1] = 29
    case.parent_inputs["input_ids"][3:] = 31
    original = case.parent_inputs["input_ids"].clone()
    peer = StagedModelPeer(case.plan)

    run_deepseek_v41_stages(case.model, case.stages, case.contexts, case.slices, peer)

    torch.testing.assert_close(case.parent_inputs["input_ids"], original)
    for _, stage, hidden, fp32, ids in peer.messages:
        expected = original[case.slices[stage]].clone()
        expected[1:] = 0
        torch.testing.assert_close(ids, expected)
        assert torch.any(fp32 != hidden.float())


def test_remote_failure_restores_parent_context_and_all_attention_bindings(case):
    parent = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE, ubatch_idx=-1)
    original = case.model.shared_attention_state
    peer = StagedModelPeer(case.plan, fail_step=(0, 1))

    with override_forward_context(parent):
        with pytest.raises(RuntimeError, match="injected remote failure"):
            run_deepseek_v41_stages(
                case.model, case.stages, case.contexts, case.slices, peer
            )
        assert get_forward_context() is parent

    assert case.model.shared_attention_state is original
    assert all(layer.self_attn.shared_state is original for layer in case.model.layers)


@pytest.mark.parametrize(
    "invalid", ["graph", "child_index", "positions", "raw_ids", "sp"]
)
def test_incompatible_child_state_is_rejected_before_payload(case, invalid):
    if invalid == "graph":
        case.contexts[0].cudagraph_runtime_mode = CUDAGraphMode.FULL
    elif invalid == "child_index":
        case.contexts[0].ubatch_idx = 1
    elif invalid == "sp":
        case.model.use_sequence_parallel = True
    else:
        changes = (
            {"positions": torch.ones((2, 2))}
            if invalid == "positions"
            else {"input_ids": torch.ones(1, dtype=torch.int32)}
        )
        case.stages = (replace(case.stages[0], **changes), case.stages[1])
    peer = StagedModelPeer(case.plan)

    with pytest.raises(ValueError, match="V4.1"):
        run_deepseek_v41_stages(
            case.model, case.stages, case.contexts, case.slices, peer
        )

    assert peer.messages == []


def test_engram_slices_share_parent_storage_and_readiness_handles():
    lookup = torch.arange(20).reshape(5, 4)
    mask = torch.ones(5, dtype=torch.bool)
    ready, pending = SimpleNamespace(), {1: SimpleNamespace()}
    parent = DeepseekV41EngramInputs({1: lookup}, mask, pending, ready)

    first, second = parent.slice(slice(0, 2)), parent.slice(slice(2, 5))
    second.lookups[1].fill_(7)
    second.mask.fill_(False)

    torch.testing.assert_close(first.lookups[1], torch.arange(8).reshape(2, 4))
    torch.testing.assert_close(lookup[2:], torch.full((3, 4), 7))
    assert mask.tolist() == [True, True, False, False, False]
    assert first.pending is second.pending is pending
    assert first.mask_ready_event is second.mask_ready_event is ready


@pytest.mark.parametrize(
    "tokens",
    [
        slice(None, 1),
        slice(-1, 1),
        slice(3, 2),
        slice(0, 6),
        slice(0, 2, True),
        slice(0, 2, 1.0),
    ],
)
def test_engram_slice_rejects_invalid_parent_coordinates(tokens):
    parent = DeepseekV41EngramInputs({1: torch.zeros(5, 4)}, torch.ones(5))

    with pytest.raises(ValueError, match="Engram stage"):
        parent.slice(tokens)


@pytest.mark.parametrize("split", [(5,), (2, 3)])
def test_serial_attention_diagnostics_keep_native_outputs_and_parent_streams(split):
    case = make_staged_case(split)
    peer = StagedModelPeer(case.plan)
    diagnostics = make_staged_progress(peer)
    expected = case.model(**case.parent_inputs)
    result = run_deepseek_v41_stages(
        case.model,
        case.stages,
        case.contexts,
        case.slices,
        peer,
        progress=diagnostics.progress,
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    stage_zero = [
        r["event"]
        for r in diagnostics.recorder.records
        if r.get("layer") == 0 and r.get("stage") == 0
    ]
    assert stage_zero == [
        "prepare_begin",
        "prepare_returned",
        "compute_begin",
        "compute_returned",
        "send_begin",
        "send_returned",
        "recv_begin",
        "recv_returned",
        "continuation_begin",
        "continuation_returned",
    ]
    assert diagnostics.recorder.events[-2:] == ["collapse_begin", "collapse_returned"]
    assert all(r["actual.mode"] == "serial" for r in diagnostics.recorder.records)
    assert all(
        r["actual.ffn_compute_sync"] is False for r in diagnostics.recorder.records
    )
    assert "device_sync" not in diagnostics.recorder.events


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("boundary", ["transaction", "rank", "actual_streams"])
def test_foreign_attention_progress_rejects_before_embedding_or_payload(
    monkeypatch, streamed, boundary
):
    setup = make_eager_model_case(monkeypatch, module)
    case, peer = setup.case, setup.peer
    embeddings = []
    case.model.embed_tokens.register_forward_hook(
        lambda *args: embeddings.append("embedded")
    )
    diagnostics = (
        make_eager_progress(setup.pipeline, setup.runtime.compute)
        if streamed
        else make_staged_progress(peer)
    )
    progress = diagnostics.progress
    if boundary == "transaction":
        progress = replace(progress, transaction_id="foreign")
    elif boundary == "rank":
        progress = replace(
            progress, diagnostics=replace(progress.diagnostics, world_rank=7)
        )
    else:
        progress = replace(
            progress,
            actual_execution=replace(
                progress.actual_execution,
                send_stream="send" if streamed else "parent",
                mode="eager-streamed",
                stream_aliases=None,
            ),
        )
    with pytest.raises(ValueError, match="parent|executor streams"):
        run_deepseek_v41_stages(
            case.model,
            case.stages,
            case.contexts,
            case.slices,
            peer,
            pipeline=setup.pipeline if streamed else None,
            compute_stream=setup.runtime.compute if streamed else None,
            progress=progress,
        )
    assert peer.messages == setup.pipeline.messages == []
    assert embeddings == []
    assert not setup.pipeline.failed


def test_streamed_attention_continuation_uses_issuing_layer_coordinates(monkeypatch):
    setup = make_eager_model_case(monkeypatch, module)
    case = setup.case
    diagnostics = make_eager_progress(setup.pipeline, setup.runtime.compute)
    result = run_deepseek_v41_stages(
        case.model,
        case.stages,
        case.contexts,
        case.slices,
        setup.peer,
        pipeline=setup.pipeline,
        compute_stream=setup.runtime.compute,
        progress=diagnostics.progress,
    )
    expected = case.model(**case.parent_inputs)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    continuations = [
        (r["layer"], r["stage"])
        for r in diagnostics.recorder.records
        if r["event"] == "continuation_begin"
    ]
    assert continuations == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all(
        r["actual.mode"] == "eager-streamed" for r in diagnostics.recorder.records
    )
    assert "device_sync" not in diagnostics.recorder.events


@pytest.mark.parametrize(
    "event",
    ["prepare_begin", "compute_returned", "continuation_begin", "collapse_returned"],
)
def test_attention_sink_failure_retains_live_owner_and_restores_bindings(
    monkeypatch, event
):
    setup = make_eager_model_case(monkeypatch, module)
    case = setup.case
    original = case.model.shared_attention_state
    diagnostics = make_eager_progress(setup.pipeline, setup.runtime.compute)
    diagnostics.recorder.fail_event = event
    with pytest.raises(RuntimeError, match="diagnostic sink failed"):
        run_deepseek_v41_stages(
            case.model,
            case.stages,
            case.contexts,
            case.slices,
            setup.peer,
            pipeline=setup.pipeline,
            compute_stream=setup.runtime.compute,
            progress=diagnostics.progress,
        )
    assert setup.pipeline.failed and setup.peer.terminal
    assert not setup.pipeline.finished
    assert setup.runtime.current == setup.runtime.parent
    assert case.model.shared_attention_state is original
