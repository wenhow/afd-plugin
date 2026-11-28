# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU view/binding seams; no Ascend operator or model computation."""

from contextlib import nullcontext

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from afd_plugin.model_executor.models.npu.deepseek_v41_stage_state import (  # noqa: E402
    DeepseekV41StageAttentionStates,
)
from tests.unit.fakes.deepseek_v41 import make_shared_attention_model  # noqa: E402


@pytest.fixture
def model():
    return make_shared_attention_model()


def buffers(state):
    return (
        state.topk_indices,
        state.candidates,
        state.candidate_lengths,
        state.topk_lengths,
    )


def test_stage_writes_reach_only_their_disjoint_parent_rows(model):
    parent = model.shared_attention_state
    for buffer in buffers(parent):
        buffer.fill_(37)
    states = DeepseekV41StageAttentionStates(model, (slice(0, 2), slice(2, 5)))
    for stage_idx, value in enumerate((10, 20)):
        with states.bind(stage_idx) as state:
            for buffer in buffers(state):
                buffer.fill_(value)
    for buffer in buffers(parent):
        expected = torch.full_like(buffer, 37)
        expected[:2] = 10
        expected[2:5] = 20
        torch.testing.assert_close(buffer, expected)


def test_stage_source_and_later_consumer_share_the_same_index_rows(model):
    states = DeepseekV41StageAttentionStates(model, (slice(0, 3), slice(3, 6)))
    with states.bind(0):
        source = model.layers[0].self_attn.shared_state
        source.topk_indices.fill_(17)
        source.candidates.fill_(19)
    with states.bind(1):
        source = model.layers[0].self_attn.shared_state
        source.topk_indices.fill_(23)
        source.candidates.fill_(29)
    with states.bind(0) as state:
        consumer = model.layers[2].self_attn.shared_state
        assert consumer is state is model.shared_attention_state
        torch.testing.assert_close(
            consumer.topk_indices, torch.full_like(consumer.topk_indices, 17)
        )
        torch.testing.assert_close(
            consumer.candidates, torch.full_like(consumer.candidates, 19)
        )


@pytest.mark.parametrize("fail", [False, True])
def test_parent_bindings_and_storage_are_restored_after_stage_exit(model, fail):
    parent = model.shared_attention_state
    parent_buffers = buffers(parent)
    states = DeepseekV41StageAttentionStates(model, (slice(0, 3), slice(3, 6)))
    with (
        pytest.raises(RuntimeError, match="consumer failed") if fail else nullcontext(),
        states.bind(1) as state,
    ):
        assert model.shared_attention_state is state
        assert all(layer.self_attn.shared_state is state for layer in model.layers)
        if fail:
            raise RuntimeError("consumer failed")
    assert model.shared_attention_state is parent
    assert all(layer.self_attn.shared_state is parent for layer in model.layers)
    assert all(
        actual is original
        for actual, original in zip(buffers(parent), parent_buffers, strict=True)
    )
    assert model.topk_indices_buffer is parent.topk_indices
    assert model.candidate_indices_buffer is parent.candidates
    with states.bind(0):
        pass


@pytest.mark.parametrize(
    "slices",
    [
        (slice(0, 0),),
        (slice(0, 0), slice(0, 0)),
        (slice(0, 0), slice(0, 4)),
        (slice(0, 4), slice(4, 4)),
    ],
)
def test_empty_children_and_active_extent_below_capacity_are_valid(model, slices):
    parent = model.shared_attention_state
    before = tuple(buffer.clone() for buffer in buffers(parent))
    states = DeepseekV41StageAttentionStates(model, slices)
    for stage_idx, token_slice in enumerate(slices):
        with states.bind(stage_idx) as state:
            for buffer in buffers(state):
                assert buffer.shape[0] == token_slice.stop - token_slice.start
                if not buffer.shape[0]:
                    buffer.fill_(41)
    for buffer, expected in zip(buffers(parent), before, strict=True):
        torch.testing.assert_close(buffer, expected)


def test_optional_native_length_buffers_stay_absent():
    model = make_shared_attention_model(with_lengths=False)
    states = DeepseekV41StageAttentionStates(model, (slice(0, 2), slice(2, 6)))
    with states.bind(1) as state:
        assert state.candidate_lengths is None
        assert state.topk_lengths is None


@pytest.mark.parametrize(
    "slices",
    [
        (),
        [slice(0, 6)],
        (slice(0, 2), slice(2, 4), slice(4, 6)),
        (slice(None, 6),),
        (slice(0, None),),
        (slice(1, 6),),
        (slice(0, -1),),
        (slice(0, 6, -1),),
        (slice(0, 6, 2),),
        (slice(0, 6, True),),
        (slice(False, 6),),
        (slice(0, True),),
        (slice(0.0, 6),),
        (slice(0, 6.0),),
        (slice(0, 4), slice(3, 6)),
        (slice(0, 2), slice(3, 6)),
        ("0:6",),
    ],
)
def test_invalid_slice_layout_never_changes_parent_bindings(model, slices):
    parent = model.shared_attention_state
    with pytest.raises(ValueError, match="slices"):
        DeepseekV41StageAttentionStates(model, slices)
    assert model.shared_attention_state is parent
    assert all(layer.self_attn.shared_state is parent for layer in model.layers)


@pytest.mark.parametrize(
    "field", ["topk_indices", "candidates", "candidate_lengths", "topk_lengths"]
)
def test_stage_extent_must_fit_every_present_parent_buffer(model, field):
    parent = model.shared_attention_state
    parent_buffers = dict(
        zip(
            ("topk_indices", "candidates", "candidate_lengths", "topk_lengths"),
            buffers(parent),
            strict=True,
        )
    )
    setattr(parent, field, parent_buffers[field][:4])
    with pytest.raises(ValueError, match="capacity"):
        DeepseekV41StageAttentionStates(model, (slice(0, 3), slice(3, 5)))
    assert model.shared_attention_state is parent


def test_noncontiguous_parent_storage_is_rejected_before_binding(model):
    parent = model.shared_attention_state
    parent.topk_indices = torch.empty((6, 6), dtype=torch.int32)[:, ::2]
    with pytest.raises(ValueError, match="contiguous"):
        DeepseekV41StageAttentionStates(model, (slice(0, 3), slice(3, 6)))
    assert model.shared_attention_state is parent


@pytest.mark.parametrize("stage_idx", [-1, 2, True, False, 0.0, "0", None])
def test_stage_coordinate_is_an_explicit_prepared_integer(model, stage_idx):
    parent = model.shared_attention_state
    states = DeepseekV41StageAttentionStates(model, (slice(0, 3), slice(3, 6)))
    with pytest.raises(ValueError, match="stage_idx"), states.bind(stage_idx):
        pass
    assert model.shared_attention_state is parent


def test_nested_binding_is_rejected_without_disturbing_the_active_stage(model):
    parent = model.shared_attention_state
    states = DeepseekV41StageAttentionStates(model, (slice(0, 3), slice(3, 6)))
    with states.bind(0) as state:
        with pytest.raises(RuntimeError, match="not reentrant"), states.bind(1):
            pass
        assert model.shared_attention_state is state
        assert all(layer.self_attn.shared_state is state for layer in model.layers)
    assert model.shared_attention_state is parent


def test_second_manager_cannot_override_another_active_binding(model):
    first = DeepseekV41StageAttentionStates(model, (slice(0, 3), slice(3, 6)))
    second = DeepseekV41StageAttentionStates(model, (slice(0, 3), slice(3, 6)))
    with first.bind(0) as state:
        with pytest.raises(RuntimeError, match="unbound parent"), second.bind(1):
            pass
        assert model.shared_attention_state is state
    with second.bind(1):
        pass


def test_broken_layer_field_is_detected_before_any_binding_changes(model):
    parent = model.shared_attention_state
    states = DeepseekV41StageAttentionStates(model, (slice(0, 3), slice(3, 6)))
    del model.layers[2].self_attn.shared_state
    with pytest.raises(AttributeError, match="shared_state"), states.bind(0):
        pass
    assert model.shared_attention_state is parent
    assert model.layers[0].self_attn.shared_state is parent
    assert model.layers[1].self_attn.shared_state is parent
    model.layers[2].self_attn.shared_state = parent
    with states.bind(0):
        pass
