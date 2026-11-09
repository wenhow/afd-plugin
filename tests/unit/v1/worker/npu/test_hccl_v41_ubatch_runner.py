# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU V2 slicing/parent ownership seams; no native device model execution."""

from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode  # noqa: E402
from vllm.forward_context import (  # noqa: E402
    create_forward_context,
    get_forward_context,
    override_forward_context,
)

from afd_plugin.compat.patches.npu import mrv2_ubatching as factory  # noqa: E402
from afd_plugin.distributed.hccl import HcclStageLayout  # noqa: E402
from afd_plugin.distributed.hccl_plan import HcclParentPlan  # noqa: E402
from afd_plugin.v1.worker.npu import hccl_v41_ubatch_runner as module  # noqa: E402
from tests.unit.fakes.deepseek_v41 import (  # noqa: E402
    ParentEngramState,
    make_staged_model,
)
from tests.unit.fakes.hccl import StagedModelPeer  # noqa: E402
from tests.unit.fakes.input_batch import (  # noqa: E402
    make_ascend_batch,
    make_hccl_ubatch_seam,
)


@pytest.fixture
def runner_factory(monkeypatch):
    # RoPE/device metadata are separate seams; native slicing and forward
    # context construction below execute their real locked implementations.
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda _: None
    )
    monkeypatch.setattr(
        module.current_platform, "set_additional_forward_context", lambda *a, **k: {}
    )

    def build(*, dp_size=1, model_state=None):
        return make_hccl_ubatch_seam(
            module.HcclV41UBatchRunner,
            dp_size=dp_size,
            model_state=model_state or ParentEngramState({}),
        )

    return build


@pytest.mark.parametrize("dp_size", [1, 3])
@pytest.mark.parametrize("real", [2, 5, 7])
def test_prepare_retains_native_child_coordinates_without_local_moe_dp(
    runner_factory, dp_size, real
):
    runner = runner_factory(dp_size=dp_size)
    batch = make_ascend_batch(real)
    slots = torch.arange(8, dtype=torch.int32).reshape(1, 8)
    prepared = runner.prepare(batch, (torch.tensor([[31, 32]]),), slots)
    assert [(s.token_slice.start, s.token_slice.stop) for s in prepared.slices] == [
        (0, 4),
        (4, 8),
    ]
    for stage_idx, context in enumerate(prepared.forward_contexts):
        child = context.attn_metadata["child"]
        token_slice = prepared.slices[stage_idx].token_slice
        assert child.padded_tokens == 4
        assert child.valid_tokens == (
            min(real, 4) if stage_idx == 0 else max(0, real - 4)
        )
        torch.testing.assert_close(child.input_ids, batch.input_ids[token_slice])
        torch.testing.assert_close(child.positions, batch.positions[token_slice])
        torch.testing.assert_close(context.slot_mapping["layer"], slots[0, token_slice])
        torch.testing.assert_close(context.is_padding, batch.is_padding[token_slice])
        assert child.block_tables[0].tolist() == [[31, 32]]
        assert context.dp_metadata is None
        assert context.ubatch_idx == stage_idx
        assert context.additional_kwargs["dbo_enabled"] is False
    children = [context.attn_metadata["child"] for context in prepared.forward_contexts]
    assert children[0].seq_lens_np.tolist() == [12 + min(real, 4)]
    assert children[1].seq_lens_np.tolist() == [12 + real]
    assert children[1].query_start_loc.tolist() == [0, max(0, real - 4)]
    assert batch.seq_lens_np.tolist() == [12 + real]


def test_parent_input_preparation_observes_eager_mode_after_native_child_preparation(
    runner_factory,
):
    runner = runner_factory()
    runner.model_state._cg_mode = CUDAGraphMode.FULL
    batch = make_ascend_batch()
    runner.prepare(batch, (), torch.arange(8, dtype=torch.int32).reshape(1, 8))
    assert (
        runner.model_state.prepare_inputs(batch, None)["producer_mode"]
        == CUDAGraphMode.NONE
    )


def test_failed_child_preparation_preserves_the_previous_parent_mode(runner_factory):
    runner = runner_factory()
    runner.model_state._cg_mode = CUDAGraphMode.FULL
    batch = make_ascend_batch()

    def unavailable_metadata(*args, **kwargs):
        raise RuntimeError("child metadata unavailable")

    runner._prepare_attn = unavailable_metadata
    with pytest.raises(RuntimeError, match="child metadata unavailable"):
        runner.prepare(batch, (), torch.arange(8, dtype=torch.int32).reshape(1, 8))
    assert (
        runner.model_state.prepare_inputs(batch, None)["producer_mode"]
        == CUDAGraphMode.FULL
    )


@pytest.mark.parametrize(
    "mode,for_capture", [(CUDAGraphMode.FULL, False), (CUDAGraphMode.NONE, True)]
)
def test_prepare_rejects_graph_execution_before_native_slicing(
    runner_factory, mode, for_capture
):
    runner = runner_factory()
    with pytest.raises(ValueError, match="eager execution"):
        runner.prepare(None, (), None, cg_mode=mode, for_capture=for_capture)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipeline_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("use_sequence_parallel_moe", True),
    ],
)
def test_prepare_rejects_unimplemented_parallel_layouts(runner_factory, field, value):
    runner = runner_factory()
    setattr(runner.parallel_config, field, value)
    with pytest.raises(ValueError, match="PP1, CP1 and non-SP"):
        runner.prepare(None, (), None)


@pytest.fixture
def staged_case(runner_factory):
    def build(*, engram=True, fail_step=None):
        inner = make_staged_model(capacity=8, engram=engram)
        producer = (
            {
                "engram_lookups": {
                    i: torch.full((8, 4), i + 1, dtype=torch.bfloat16) for i in range(2)
                },
                "engram_mask": torch.ones(8, dtype=torch.bool),
            }
            if engram
            else {}
        )
        state = ParentEngramState(producer)
        runner = runner_factory(model_state=state)
        batch = make_ascend_batch()
        prepared = runner.prepare(
            batch, (), torch.arange(8, dtype=torch.int32).reshape(1, 8)
        )
        parent = create_forward_context({"parent": "metadata"}, runner.vllm_config)
        state.parent_context = parent
        plan = HcclParentPlan(
            "runner-parent",
            (0, 1),
            (
                HcclStageLayout(0, (4,), (4,), 1),
                HcclStageLayout(1, (1,), (4,), 1),
            ),
        )
        return SimpleNamespace(
            runner=runner,
            inner=inner,
            model=SimpleNamespace(language_model=SimpleNamespace(model=inner)),
            inputs={
                "input_ids": batch.input_ids,
                "positions": batch.positions,
                "intermediate_tensors": None,
                "inputs_embeds": None,
            },
            producer=producer,
            prepared=prepared,
            parent=parent,
            peer=StagedModelPeer(plan, fail_step=fail_step),
        )

    return build


@pytest.mark.parametrize("engram", [False, True])
@pytest.mark.parametrize("entry", ["explicit", "native"])
def test_staged_run_consumes_parent_producer_and_returns_native_parent_outputs(
    staged_case, engram, entry
):
    case = staged_case(engram=engram)
    reference = make_staged_model(capacity=8, engram=engram)
    kwargs = case.producer or {
        "engram_lookups": {},
        "engram_mask": torch.empty(0, dtype=torch.bool),
    }
    with override_forward_context(case.parent):
        expected = reference(**case.inputs, **kwargs)
        if entry == "native":
            case.peer.initialized, case.peer.closed = True, False
            case.runner.transport = case.peer
            output, auxiliary = case.runner.run(case.model, case.inputs, case.prepared)
        else:
            output, auxiliary = case.runner.run_staged(
                case.model, case.inputs, case.prepared, case.peer
            )
        assert get_forward_context() is case.parent
    torch.testing.assert_close(output[:5], expected[0][:5], rtol=0, atol=0)
    for actual, wanted in zip(auxiliary, expected[1], strict=True):
        torch.testing.assert_close(actual[:5], wanted[:5], rtol=0, atol=0)
    assert output.shape == (8, 4)
    assert case.inputs["input_ids"].tolist() == list(range(8))
    assert case.peer.completed and not case.peer.terminal
    assert [(layer, stage) for layer, stage, *_ in case.peer.messages] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


@pytest.mark.parametrize("entry", ["explicit", "native"])
def test_staged_failure_is_terminal_and_restores_parent_context_and_attention_owner(
    staged_case, entry
):
    case = staged_case(fail_step=(0, 1))
    parent_state = case.inner.shared_attention_state
    with override_forward_context(case.parent):
        with pytest.raises(RuntimeError, match="injected remote failure"):
            if entry == "native":
                case.peer.initialized, case.peer.closed = True, False
                case.runner.transport = case.peer
                case.runner.run(case.model, case.inputs, case.prepared)
            else:
                case.runner.run_staged(
                    case.model, case.inputs, case.prepared, case.peer
                )
        assert get_forward_context() is case.parent
    assert case.peer.terminal and not case.peer.completed
    assert case.inner.shared_attention_state is parent_state
    assert all(
        layer.self_attn.shared_state is parent_state for layer in case.inner.layers
    )


@pytest.mark.parametrize(
    "boundary", ["parent-graph", "engram-graph", "intermediate", "raw-ids"]
)
def test_staged_run_rejects_unsupported_parent_contracts(staged_case, boundary):
    case = staged_case()
    if boundary == "parent-graph":
        case.parent.cudagraph_runtime_mode = CUDAGraphMode.FULL
    elif boundary == "engram-graph":
        case.inputs["engram_graph_events"] = True
    elif boundary == "intermediate":
        case.inputs["intermediate_tensors"] = object()
    else:
        case.inputs["input_ids"] = None
    with override_forward_context(case.parent):
        with pytest.raises(ValueError, match="eager PP1|raw token IDs"):
            case.runner.run_staged(case.model, case.inputs, case.prepared, case.peer)
        assert get_forward_context() is case.parent
    assert case.peer.terminal and not case.peer.completed
    assert case.peer.messages == []


@pytest.mark.parametrize("state", ["uninitialized", "closed", "terminal", "no-plan"])
def test_native_run_rejects_an_inactive_or_unaccepted_transport(staged_case, state):
    case = staged_case()
    case.runner.transport = case.peer
    case.peer.initialized = state != "uninitialized"
    case.peer.closed = state == "closed"
    case.peer.terminal = state == "terminal"
    if state == "no-plan":
        case.peer.plan = None
    with override_forward_context(case.parent):
        with pytest.raises((ValueError, RuntimeError), match="active HCCL|accepted"):
            case.runner.run(case.model, case.inputs, case.prepared)
        assert get_forward_context() is case.parent
    assert case.peer.messages == []
    assert not case.peer.completed


@pytest.fixture
def scoped_factory_case(monkeypatch, staged_case):
    case = staged_case()
    case.runner.vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    case.runner.parallel_config.rank = 0
    case.peer.negotiated_capabilities = SimpleNamespace(
        native_tp_size=1, attention_size=1, max_stages=2
    )

    # This test owns factory selection, not device allocation. The separately
    # validated native prepare/context state is the CPU constructor boundary.
    def cpu_constructor(runner, *args, **kwargs):
        runner.__dict__.update(case.runner.__dict__)

    monkeypatch.setattr(module.NPUUBatchRunner, "__init__", cpu_constructor)
    case.owner = SimpleNamespace(kernel_block_sizes=[1])

    def construct():
        return factory.model_runner.maybe_build_ubatch_runner(
            case.runner.vllm_config,
            torch.device("cpu"),
            case.runner.model_state,
            [],
            case.runner.kv_cache_config,
            2,
        )

    case.construct = construct
    return case


def test_explicit_factory_binds_native_dp1_run_and_restores_both_native_aliases(
    scoped_factory_case,
):
    case = scoped_factory_case
    original_factory = factory.model_runner.maybe_build_ubatch_runner
    original_sm = factory.ubatch_utils.create_sm_control_context
    with factory.use_npu_ubatch_factory(case.owner, hccl_transport=case.peer):
        runner = case.construct()
        assert isinstance(runner, module.HcclV41UBatchRunner)
        case.peer.initialized, case.peer.closed = True, False
        with override_forward_context(case.parent):
            output, _ = runner.run(case.model, case.inputs, case.prepared)
    assert output.shape == (8, 4)
    assert case.peer.completed and not case.peer.terminal
    assert factory.model_runner.maybe_build_ubatch_runner is original_factory
    assert factory.ubatch_utils.create_sm_control_context is original_sm


@pytest.mark.parametrize(
    "boundary,error",
    [
        ("dp-size", "DP1 rank zero"),
        ("dp-rank", "DP1 rank zero"),
        ("tp-size", "transport native TP size"),
        ("role-size", "role size must match"),
        ("native-rank", "native rank does not match"),
        ("stage-capacity", "requires negotiated U2"),
    ],
)
def test_explicit_factory_rejects_unproven_native_group_mappings(
    scoped_factory_case, boundary, error
):
    case = scoped_factory_case
    parallel = case.runner.parallel_config
    if boundary == "dp-size":
        parallel.data_parallel_size = 2
    elif boundary == "dp-rank":
        parallel.data_parallel_rank = 1
    elif boundary == "tp-size":
        parallel.tensor_parallel_size = 2
    elif boundary == "role-size":
        case.peer.negotiated_capabilities.attention_size = 2
    elif boundary == "stage-capacity":
        case.peer.negotiated_capabilities.max_stages = 1
    else:
        parallel.tensor_parallel_size = 2
        case.peer.negotiated_capabilities.native_tp_size = 2
        case.peer.negotiated_capabilities.attention_size = 2
        parallel.rank = 1
    with (
        pytest.raises(ValueError, match=error),
        factory.use_npu_ubatch_factory(case.owner, hccl_transport=case.peer),
    ):
        case.construct()
    assert case.peer.messages == []
    assert not case.peer.completed


def test_omitted_transport_preserves_existing_hccl_dp1_factory_admission(
    scoped_factory_case,
):
    case = scoped_factory_case
    with factory.use_npu_ubatch_factory(case.owner):
        assert case.construct() is None


@pytest.mark.parametrize(
    "boundary",
    ["disabled", "u1", "u3", "graph", "non-eager", "connector", "role", "peer-role"],
)
def test_explicit_factory_rejects_unsupported_construction_without_silent_fallback(
    scoped_factory_case, boundary
):
    case = scoped_factory_case
    config = case.runner.vllm_config
    if boundary == "disabled":
        config.parallel_config.use_ubatching = False
    elif boundary in {"u1", "u3"}:
        config.parallel_config.num_ubatches = int(boundary[1])
    elif boundary == "graph":
        config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    elif boundary == "non-eager":
        config.model_config.enforce_eager = False
    elif boundary == "connector":
        config.additional_config["afd"]["connector"] = "CAMP2pAFDConnector"
    elif boundary == "role":
        config.additional_config["afd"]["role"] = "ffn"
    else:
        case.peer.role = "ffn"
    original_factory = factory.model_runner.maybe_build_ubatch_runner
    original_sm = factory.ubatch_utils.create_sm_control_context
    with (
        pytest.raises(ValueError, match="eager Attention U2"),
        factory.use_npu_ubatch_factory(case.owner, hccl_transport=case.peer),
    ):
        case.construct()
    assert factory.model_runner.maybe_build_ubatch_runner is original_factory
    assert factory.ubatch_utils.create_sm_control_context is original_sm
