# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU V2 slicing/parent ownership seams; no native device model execution."""

from concurrent.futures import ThreadPoolExecutor
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
from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext  # noqa: E402

from afd_plugin.compat.patches.npu import mrv2_ubatching as factory  # noqa: E402
from afd_plugin.distributed.hccl import HcclStageLayout  # noqa: E402
from afd_plugin.distributed.hccl_plan import HcclParentPlan  # noqa: E402
from afd_plugin.v1.worker.npu import hccl_v41_ubatch_runner as module  # noqa: E402
from afd_plugin.v1.worker.npu.hccl_v41_graph_session import (  # noqa: E402
    HcclV41GraphSession,
)
from tests.unit.fakes.deepseek_v41 import (  # noqa: E402
    ParentEngramState,
    make_staged_model,
)
from tests.unit.fakes.device_metadata import make_cpu_target_metadata  # noqa: E402
from tests.unit.fakes.hccl import StagedModelPeer  # noqa: E402
from tests.unit.fakes.hccl_v41_ubatch import (  # noqa: E402
    GraphParentModel,
    GraphUBatchOwner,
)
from tests.unit.fakes.input_batch import (  # noqa: E402
    make_ascend_batch,
    make_ascend_decode_batch,
    make_hccl_ubatch_seam,
)
from tests.unit.fakes.vllm_config import v41_dspark_config  # noqa: E402


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

    def build(*, dp_size=1, model_state=None, max_num_reqs=2, graph_owner=None):
        return make_hccl_ubatch_seam(
            module.HcclV41UBatchRunner,
            dp_size=dp_size,
            model_state=model_state or ParentEngramState({}),
            max_num_reqs=max_num_reqs,
            graph_owner=graph_owner,
        )

    return build


def test_eager_staged_run_rejects_foreign_wire_before_native_producer(runner_factory):
    runner = runner_factory()
    peer = StagedModelPeer(None)
    runner.transport = peer
    runner.eager_session = SimpleNamespace(transport=peer)
    with pytest.raises(ValueError, match="bound transport"):
        runner.run_staged(None, {}, None, StagedModelPeer(None))
    assert not runner.model_state.consumed
    assert peer.messages == []


@pytest.mark.parametrize("invalid", ["uninitialized", "closed", "terminal"])
def test_eager_binding_rejects_a_dead_previously_warmed_session(
    runner_factory, invalid
):
    runner = runner_factory()
    peer = SimpleNamespace(initialized=True, closed=False, terminal=False)
    runner.transport = peer
    session = SimpleNamespace(transport=peer, initialized=True)
    setattr(
        peer,
        "initialized" if invalid == "uninitialized" else invalid,
        invalid != "uninitialized",
    )
    with pytest.raises(RuntimeError, match="idle eager session"):
        runner.bind_eager_session(session)


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


@pytest.fixture
def graph_case(runner_factory, monkeypatch):
    # ModelWithContext itself runs. Stream-capture detection is its explicit
    # device boundary; this fixture neither captures nor simulates an NPU graph.
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)

    def build(*, reject=None, metadata=False, fail=False):
        state = ParentEngramState({})
        runner = runner_factory(model_state=state, max_num_reqs=8)
        parent = create_forward_context(
            {}, runner.vllm_config, cudagraph_runtime_mode=CUDAGraphMode.FULL
        )
        peer = SimpleNamespace(terminal=False)
        owner = GraphUBatchOwner(peer, parent, reject=reject)
        runner.graph_owner = owner
        providers = ()
        if metadata:
            state.device_metadata, providers, runner.stage_attn_groups = (
                make_cpu_target_metadata()
            )
        return SimpleNamespace(
            runner=runner,
            owner=owner,
            peer=peer,
            parent=parent,
            batch=make_ascend_decode_batch(),
            slots=torch.arange(8, dtype=torch.int32).reshape(1, 8),
            model=ModelWithContext(
                GraphParentModel(providers=providers, fail=fail),
                device_metadata=state.device_metadata,
            ),
        )

    return build


@pytest.mark.parametrize("for_capture", [False, True])
def test_full_prepare_preserves_native_child_modes_and_parent_producer_choice(
    graph_case, for_capture
):
    case = graph_case()
    prepared = case.runner.prepare(
        case.batch, (), case.slots, CUDAGraphMode.FULL, for_capture
    )
    assert [(stage.request_slice, stage.token_slice) for stage in prepared.slices] == [
        (slice(0, 4), slice(0, 4)),
        (slice(4, 8), slice(4, 8)),
    ]
    assert all(
        context.cudagraph_runtime_mode == CUDAGraphMode.FULL
        for context in prepared.forward_contexts
    )
    assert case.runner.model_state.prepare_inputs(case.batch, None) == {
        "producer_mode": CUDAGraphMode.FULL
    }
    eager = case.runner.prepare(case.batch, (), case.slots)
    assert all(
        context.cudagraph_runtime_mode == CUDAGraphMode.NONE
        for context in eager.forward_contexts
    )
    assert case.runner.model_state.prepare_inputs(case.batch, None) == {
        "producer_mode": CUDAGraphMode.NONE
    }


@pytest.mark.parametrize(
    "reject,error", [("before", "reservation is unavailable"), ("after", "geometry")]
)
def test_failed_full_prepare_is_terminal_and_restores_parent_producer_mode(
    graph_case, reject, error
):
    case = graph_case(reject=reject)
    with pytest.raises(RuntimeError, match=error):
        case.runner.prepare(case.batch, (), case.slots, CUDAGraphMode.FULL, True)
    assert case.runner.model_state.prepare_inputs(case.batch, None) == {
        "producer_mode": CUDAGraphMode.NONE
    }
    assert case.peer.terminal


def test_failed_full_child_builder_is_terminal_and_restores_parent_producer_mode(
    graph_case,
):
    case = graph_case()

    def unavailable_metadata(*args, **kwargs):
        raise RuntimeError("child metadata unavailable")

    case.runner._prepare_attn = unavailable_metadata
    with pytest.raises(RuntimeError, match="child metadata unavailable"):
        case.runner.prepare(case.batch, (), case.slots, CUDAGraphMode.FULL, True)
    assert case.runner.model_state.prepare_inputs(case.batch, None) == {
        "producer_mode": CUDAGraphMode.NONE
    }
    assert case.peer.terminal


@pytest.mark.parametrize("metadata", [False, True])
def test_capture_callback_keeps_one_native_parent_wrapper_and_both_metadata_frontiers(
    graph_case, metadata
):
    case = graph_case(metadata=metadata)
    prepared = case.runner.prepare(case.batch, (), case.slots, CUDAGraphMode.FULL, True)
    if metadata:
        assert all(
            resource.device_metadata_executor
            is case.runner.model_state.device_metadata.executor
            for context in prepared.forward_contexts
            for resource in context.attn_metadata.values()
        )
    outer = create_forward_context({}, case.runner.vllm_config)
    finish = case.runner.begin_capturable_run(
        case.model, {"input_ids": case.batch.input_ids}, prepared, True
    )
    with override_forward_context(outer):
        output = finish()
        assert get_forward_context() is outer
    expected = torch.tensor([10.0, 11.0]) if metadata else case.batch.input_ids + 1
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    if metadata:
        # Public native retirement must be legal after ModelWithContext's
        # producer/consumer join, without host re-submission on replay.
        case.runner.model_state.device_metadata.finish_replay()
    assert not case.peer.terminal


def test_capture_callback_failure_is_terminal_and_restores_the_callers_context(
    graph_case,
):
    case = graph_case(fail=True)
    prepared = case.runner.prepare(case.batch, (), case.slots, CUDAGraphMode.FULL, True)
    finish = case.runner.begin_capturable_run(
        case.model, {"input_ids": case.batch.input_ids}, prepared, True
    )
    outer = create_forward_context({}, case.runner.vllm_config)
    with override_forward_context(outer):
        with pytest.raises(RuntimeError, match="parent model failed"):
            finish()
        assert get_forward_context() is outer
    assert case.peer.terminal


@pytest.mark.parametrize("boundary", ["repeat", "foreign-thread"])
def test_capture_callback_rejects_repeated_or_foreign_thread_execution(
    graph_case, boundary
):
    case = graph_case()
    prepared = case.runner.prepare(case.batch, (), case.slots, CUDAGraphMode.FULL, True)
    finish = case.runner.begin_capturable_run(
        case.model, {"input_ids": case.batch.input_ids}, prepared, True
    )
    if boundary == "repeat":
        finish()
        with pytest.raises(RuntimeError, match="once on its owning thread"):
            finish()
    else:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(finish)
            with pytest.raises(RuntimeError, match="once on its owning thread"):
                result.result()
    assert case.peer.terminal


@pytest.mark.parametrize(
    "boundary", ["eager", "not-capture", "different-state", "model"]
)
def test_graph_callback_rejects_an_unprepared_or_incompatible_parent(
    graph_case, boundary
):
    case = graph_case()
    mode = CUDAGraphMode.NONE if boundary == "eager" else CUDAGraphMode.FULL
    prepared = case.runner.prepare(
        case.batch, (), case.slots, mode, mode == CUDAGraphMode.FULL
    )
    if boundary == "different-state":
        prepared = prepared._replace(forward_contexts=list(prepared.forward_contexts))
    model = case.model if boundary != "model" else SimpleNamespace()
    with pytest.raises(ValueError, match="prepared FULL capture"):
        case.runner.begin_capturable_run(
            model,
            {"input_ids": case.batch.input_ids},
            prepared,
            boundary != "not-capture",
        )


@pytest.fixture
def graph_constructor(scoped_factory_case):
    case = scoped_factory_case
    config = case.runner.vllm_config
    config.model_config.enforce_eager = False
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    config.speculative_config = None
    config.model_config.hf_config.model_type = "deepseek_v41"
    config.model_config.hf_config.architectures = ["DeepseekV41ForCausalLM"]
    case.peer.negotiated_capabilities.graph_execution_version = 1
    case.peer.negotiated_capabilities.local_counts_version = 1
    case.peer.negotiated_capabilities.ffn_size = 1
    case.peer.negotiated_capabilities.target_query_len = 1
    case.graph_owner = GraphUBatchOwner(case.peer, case.parent)

    def construct():
        return module.HcclV41UBatchRunner(
            config,
            torch.device("cpu"),
            case.runner.model_state,
            [],
            case.runner.kv_cache_config,
            2,
            kernel_block_sizes=[1],
            timeout_s=1,
            transport=case.peer,
            graph_owner=case.graph_owner,
        )

    case.construct = construct
    return case


def test_explicit_graph_owner_admits_only_controlled_native_graph_construction(
    graph_constructor,
):
    case = graph_constructor
    runner = case.construct()
    with pytest.raises(ValueError, match="prepared FULL capture"):
        runner.begin_capturable_run(case.model, case.inputs, case.prepared)
    assert not case.peer.completed and case.peer.messages == []


@pytest.fixture
def dspark_graph_constructor(graph_constructor):
    case = graph_constructor
    config = case.runner.vllm_config
    source = v41_dspark_config()
    config.model_config = source.model_config
    config.model_config.enforce_eager = False
    config.speculative_config = source.speculative_config
    config.speculative_config.num_speculative_tokens = 2
    case.peer.negotiated_capabilities.target_query_len = 3
    return case


def test_graph_owner_accepts_a_fixed_dspark_target_and_eager_local_draft(
    dspark_graph_constructor,
):
    case = dspark_graph_constructor
    runner = case.construct()
    with pytest.raises(ValueError, match="prepared FULL capture"):
        runner.begin_capturable_run(case.model, case.inputs, case.prepared)
    assert not case.peer.completed and case.peer.messages == []


@pytest.mark.parametrize(
    "boundary,reason",
    [
        ("shape", "negotiated target shape"),
        ("draft_graph", "eager DSpark draft"),
        ("adaptive", "adaptive verification off"),
    ],
)
def test_graph_owner_rejects_mismatched_or_unqualified_dspark_before_forward(
    dspark_graph_constructor, boundary, reason
):
    case = dspark_graph_constructor
    if boundary == "shape":
        case.peer.negotiated_capabilities.target_query_len = 1
    elif boundary == "draft_graph":
        case.runner.vllm_config.speculative_config.enforce_eager = False
    else:
        case.runner.vllm_config.speculative_config.enable_adaptive_verification = True
    with pytest.raises(ValueError, match=reason):
        case.construct()
    assert not case.peer.completed and case.peer.messages == []


@pytest.mark.parametrize(
    "boundary,error",
    [
        ("eager", "FULL_DECODE_ONLY"),
        ("none", "FULL_DECODE_ONLY"),
        ("full", "FULL_DECODE_ONLY"),
        ("speculation", "method='dspark'"),
        ("graph-version", "graph protocol v1"),
        ("counts-version", "graph protocol v1"),
        ("ffn-size", "paired native TP"),
        ("owner-transport", "own the bound transport"),
    ],
)
def test_graph_construction_rejects_unaccepted_modes_and_protocols(
    graph_constructor, boundary, error
):
    case = graph_constructor
    config = case.runner.vllm_config
    if boundary == "eager":
        config.model_config.enforce_eager = True
    elif boundary in {"none", "full"}:
        config.compilation_config.cudagraph_mode = (
            CUDAGraphMode.NONE if boundary == "none" else CUDAGraphMode.FULL
        )
    elif boundary == "speculation":
        config.speculative_config = v41_dspark_config().speculative_config
        config.speculative_config.method = "mtp"
    elif boundary == "graph-version":
        case.peer.negotiated_capabilities.graph_execution_version = 0
    elif boundary == "counts-version":
        case.peer.negotiated_capabilities.local_counts_version = 0
    elif boundary == "ffn-size":
        case.peer.negotiated_capabilities.ffn_size = 2
    else:
        case.graph_owner.transport = SimpleNamespace(terminal=False)
    with pytest.raises(ValueError, match=error):
        case.construct()
    assert not case.peer.completed and case.peer.messages == []


def test_native_factory_injects_persistent_graph_session_before_the_first_phase(
    graph_constructor,
):
    case = graph_constructor
    session = HcclV41GraphSession(case.peer)
    with factory.use_npu_ubatch_factory(
        case.owner, hccl_transport=case.peer, hccl_graph_owner=session
    ):
        runner = factory.model_runner.maybe_build_ubatch_runner(
            case.runner.vllm_config,
            torch.device("cpu"),
            case.runner.model_state,
            [],
            case.runner.kv_cache_config,
            2,
        )
        with pytest.raises(ValueError, match="armed phase owner"):
            runner.prepare(
                make_ascend_decode_batch(),
                (),
                torch.zeros(1, 8, dtype=torch.int32),
                CUDAGraphMode.FULL,
                True,
            )
    assert case.peer.terminal
    assert not case.peer.completed and case.peer.messages == []


@pytest.mark.parametrize("missing_transport", [False, True])
def test_native_factory_rejects_a_graph_owner_from_another_transport(
    graph_constructor, missing_transport
):
    case = graph_constructor
    session = HcclV41GraphSession(SimpleNamespace(role="attention"))
    original = factory.model_runner.maybe_build_ubatch_runner
    with (
        pytest.raises(ValueError, match="injected HCCL transport"),
        factory.use_npu_ubatch_factory(
            case.owner,
            hccl_transport=None if missing_transport else case.peer,
            hccl_graph_owner=session,
        ),
    ):
        case.construct()
    assert factory.model_runner.maybe_build_ubatch_runner is original
    assert not case.peer.completed and case.peer.messages == []
