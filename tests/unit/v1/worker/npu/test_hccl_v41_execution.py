# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Native-call Python seams only; arithmetic models do not verify device ops."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.v1.worker.gpu import model_runner as native_model_runner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from afd_plugin.connectors.npu.hccl_staged import HcclStagedAFDConnector
from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector
from afd_plugin.distributed.hccl_graph import HcclGraphDecision
from afd_plugin.model_executor.models import forward_context as context_module
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    HCCL_V41_PARENT_FORWARD_KEY,
    AFDDeepseekV41AttentionForCausalLM,
)
from afd_plugin.v1.worker.npu import attention_model_runner_v2 as attention
from afd_plugin.v1.worker.npu import hccl_v41_execution as execution
from afd_plugin.v1.worker.npu import hccl_v41_ubatch_runner as ubatch
from tests.unit.fakes.deepseek_v41 import ParentEngramState, make_staged_model
from tests.unit.fakes.hccl import StagedAttentionPeer
from tests.unit.fakes.hccl_eager_model import make_eager_model_case
from tests.unit.fakes.input_batch import (
    make_ascend_batch,
    make_ascend_speculative_batch,
    make_hccl_ubatch_seam,
)
from tests.unit.fakes.vllm_config import hccl_vllm_config


def test_explicit_eager_parent_consumes_one_producer_and_completes_ticket_owner(
    monkeypatch,
):
    setup = make_eager_model_case(monkeypatch, execution)
    case = setup.case
    parent = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE, ubatch_idx=-1)
    state = ParentEngramState(
        {
            key: value
            for key, value in case.parent_inputs.items()
            if key.startswith("engram_")
        },
        parent_context=parent,
    )
    with override_forward_context(parent):
        output = execution.execute_hccl_v41_stages(
            case.wrapper,
            state,
            case.parent_inputs,
            case.contexts,
            case.slices,
            setup.peer,
            eager_pipeline=setup.pipeline,
            compute_stream=setup.runtime.compute,
        )
        assert get_forward_context() is parent

    expected = case.model(**case.parent_inputs)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert state.consumed and setup.pipeline.finished and setup.peer.completed


@pytest.mark.parametrize("invalid", ["foreign_transport", "foreign_plan", "finished"])
def test_eager_parent_rejects_stale_owner_before_consuming_native_producer(
    monkeypatch, invalid
):
    setup = make_eager_model_case(monkeypatch, execution)
    state = ParentEngramState({})
    if invalid == "foreign_transport":
        setup.pipeline.transport = StagedAttentionPeer()
    elif invalid == "foreign_plan":
        setup.pipeline.plan = replace(setup.case.plan, transaction_id="another-parent")
    else:
        setup.pipeline.finished = True

    with pytest.raises(ValueError, match="exact active transport/plan owner"):
        execution.execute_hccl_v41_stages(
            setup.case.wrapper,
            state,
            {},
            (),
            (),
            setup.peer,
            eager_pipeline=setup.pipeline,
            compute_stream=setup.runtime.compute,
        )

    assert not state.consumed and setup.peer.messages == []


@pytest.fixture
def native_case(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda _: None
    )
    monkeypatch.setattr(
        ubatch.current_platform, "set_additional_forward_context", lambda *a, **k: {}
    )
    monkeypatch.setattr(attention, "step_afd_npu_profiler", lambda _: None)

    # One upstream execution collaborator seam. Real Ascend input slicing,
    # forward-context construction, nn.Module pre-hook and AFD entry run below.
    # Native request/KV execution and retirement are not simulated as evidence.
    def native_execute(self, scheduler_output, *args, **kwargs):
        batch = (
            make_ascend_speculative_batch(scheduler_output.total_num_scheduled_tokens)
            if self.vllm_config.speculative_config is not None
            else make_ascend_batch(scheduler_output.total_num_scheduled_tokens)
        )
        state = self.model_state
        state.consumed = False
        prepared = None
        if self.select_u2:
            prepared = self.ubatch_runner.prepare(
                batch, (), torch.arange(8, dtype=torch.int32).reshape(1, 8)
            )
        self.native_steps.append(SimpleNamespace(batch=batch, prepared=prepared))
        parent = context_module.forward_context_module.create_forward_context(
            None,
            self.vllm_config,
            batch_descriptor=SimpleNamespace(num_tokens=8),
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        state.parent_context = parent
        model_inputs = dict(
            input_ids=batch.input_ids,
            positions=batch.positions,
            intermediate_tensors=None,
            inputs_embeds=None,
        )
        with override_forward_context(parent):
            output = (
                self.ubatch_runner.run(self.model, model_inputs, prepared)
                if prepared is not None
                else self.model(**model_inputs)
            )
            assert get_forward_context() is parent
        return output

    monkeypatch.setattr(NPUModelRunner, "execute_model", native_execute)

    def build(*, with_ubatches=True, bind=True, dspark=False):
        config = hccl_vllm_config(
            dtype=torch.bfloat16,
            compilation_mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
            dspark=dspark,
        )
        config.model_config.hf_text_config.hidden_size = 4
        config.model_config.hf_text_config.vocab_size = 32
        if dspark:
            # The arithmetic model has two target layers. Keep target and
            # native-draft aux metadata aligned with those actual layer IDs.
            config.model_config.hf_text_config.num_hidden_layers = 2
            config.model_config.hf_text_config.dspark_target_layer_ids = [1, 2]
            draft_text = config.speculative_config.draft_model_config.hf_text_config
            draft_text.dspark_target_layer_ids = [1, 2]
        config.compilation_config.static_forward_context = {}
        config.parallel_config.use_ubatching = True
        config.parallel_config.num_ubatches = 2
        config.parallel_config.rank = 0
        inner = make_staged_model(capacity=8)
        model = object.__new__(AFDDeepseekV41AttentionForCausalLM)
        torch.nn.Module.__init__(model)
        model.language_model = SimpleNamespace(model=inner)
        producer = dict(
            engram_lookups={i: torch.ones(8, 4).bfloat16() for i in range(2)},
            engram_mask=torch.ones(8, dtype=torch.bool),
        )
        state = ParentEngramState(producer)
        model.register_forward_pre_hook(
            state._prepare_engram_before_forward, with_kwargs=True
        )
        owner = object.__new__(attention.AFDNPUAttentionModelRunnerV2)
        owner.vllm_config, owner.model, owner.model_state = config, model, state
        owner.device = SimpleNamespace(type="npu", index=0)
        owner.connector = object.__new__(P2pHcclAFDConnector)
        owner.connector._initialized = False
        owner.ubatch_runner, owner.prof, owner.cudagraph_manager = None, None, None
        owner._afd_transaction_counter = 0
        owner.select_u2 = False
        owner.native_steps = []
        peer = StagedAttentionPeer()
        if bind:
            owner.bind_hccl_v41_transport(peer)
        if with_ubatches:
            owner.ubatch_runner = make_hccl_ubatch_seam(
                ubatch.HcclV41UBatchRunner, model_state=state
            )
            owner.ubatch_runner.vllm_config = config
            owner.ubatch_runner.parallel_config = config.parallel_config
            owner.ubatch_runner.transport = peer
        return SimpleNamespace(
            owner=owner,
            peer=peer,
            state=state,
            producer=producer,
            native_steps=owner.native_steps,
        )

    return build


@pytest.mark.parametrize(
    "u2,dummy,profile,prepared_runner,valid,padded",
    [
        (False, False, False, True, (5,), (8,)),
        (True, False, False, True, (4, 1), (4, 4)),
        (False, True, True, False, (0,), (8,)),
        (False, True, False, True, (0,), (8,)),
        (True, True, False, True, (0, 0), (4, 4)),
    ],
)
def test_native_entry_submits_actual_stage_counts_and_completes_parent(
    native_case, u2, dummy, profile, prepared_runner, valid, padded
):
    case = native_case(with_ubatches=prepared_runner)
    case.owner.select_u2 = u2
    output, auxiliary = case.owner.execute_model(
        SimpleNamespace(total_num_scheduled_tokens=5),
        dummy_run=dummy,
        is_profile=profile,
    )
    assert output.shape == (8, 4) and len(auxiliary) == 2
    proposal = case.peer.proposals[0]
    assert proposal.valid_tokens == valid and proposal.padded_tokens == padded
    assert proposal.mode == ("profile" if profile else "dummy" if dummy else "target")
    assert case.peer.completed and case.peer.plan is None and not case.peer.terminal


def test_same_native_entry_alternates_u1_and_u2_without_reusing_parent_ids(native_case):
    case = native_case()
    for u2 in (False, True, False, True):
        case.owner.select_u2 = u2
        case.owner.execute_model(SimpleNamespace(total_num_scheduled_tokens=5))
    assert [len(plan.stages) for plan in case.peer.forwards] == [1, 2, 1, 2]
    assert [plan.transaction_id for plan in case.peer.forwards] == [
        "native-1",
        "native-2",
        "native-3",
        "native-4",
    ]
    assert case.peer.plan is None and not case.peer.terminal


def test_dspark_target_keeps_query_coordinates_and_ordered_aux_through_completion(
    native_case,
):
    reference = native_case(dspark=True)
    expected, expected_aux = reference.owner.execute_model(
        SimpleNamespace(total_num_scheduled_tokens=5)
    )
    case = native_case(dspark=True)
    case.owner.select_u2 = True
    output, auxiliary = case.owner.execute_model(
        SimpleNamespace(total_num_scheduled_tokens=5)
    )
    assert output.shape == (8, 4) and len(auxiliary) == 2
    torch.testing.assert_close(output, expected)
    for actual, native_u1 in zip(auxiliary, expected_aux, strict=True):
        torch.testing.assert_close(actual, native_u1)
    # The first native aux is layer entry, before its Engram injection.
    inner = case.owner.model.language_model.model
    torch.testing.assert_close(auxiliary[0], inner.embed_tokens(torch.arange(8)))
    assert not torch.equal(auxiliary[1], auxiliary[0])
    # Native prepare exposes the actual per-child query metadata. The second
    # request crosses the token boundary: 3 queries become 2+1. Sampling fields
    # stay on the parent; no child sampler or acceptance result is invented.
    batch, prepared = case.native_steps[0].batch, case.native_steps[0].prepared
    assert batch.num_draft_tokens_per_req.tolist() == [1, 2]
    assert batch.logits_indices.tolist() == list(range(5))
    assert batch.cu_num_logits.tolist() == [0, 2, 5]
    children = [context.attn_metadata["child"] for context in prepared.forward_contexts]
    assert [child.query_start_loc.tolist() for child in children] == [[0, 2, 4], [0, 1]]
    assert [child.positions.tolist() for child in children] == [
        list(range(4)),
        list(range(4, 8)),
    ]
    assert case.peer.proposals[-1].valid_tokens == (4, 1)
    assert case.peer.completed and case.peer.plan is None


def test_completed_dspark_u2_target_leaves_native_draft_forward_attention_local(
    native_case,
):
    case = native_case(dspark=True)
    case.owner.select_u2 = True
    _, auxiliary = case.owner.execute_model(
        SimpleNamespace(total_num_scheduled_tokens=5)
    )
    assert case.peer.completed and case.peer.plan is None

    def local_draft(input_ids, **kwargs):
        # A real native drafter context must not inherit the completed target's
        # typed parent sidecar or AFD metadata provider. Arithmetic is a model
        # collaborator only; this test makes no native sampling/device claim.
        context = get_forward_context()
        assert HCCL_V41_PARENT_FORWARD_KEY not in (context.additional_kwargs or {})
        assert context_module.get_afd_metadata_from_forward_context() is None
        return auxiliary[0][: input_ids.numel()] + input_ids[:, None]

    draft = SimpleNamespace(
        vllm_config=case.owner.vllm_config,
        input_buffers=SimpleNamespace(
            input_ids=torch.arange(5), positions=torch.arange(5)
        ),
        model=local_draft,
    )
    result = DFlashSpeculator._run_model(draft, 5, None, None, None)
    torch.testing.assert_close(result, auxiliary[0][:5] + torch.arange(5)[:, None])
    assert len(case.peer.forwards) == 1
    assert [(layer, stage) for layer, stage, *_ in case.peer.messages] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert case.peer.completed and case.peer.plan is None


@pytest.mark.parametrize("u2,padded", [(False, (8,)), (True, (4, 4))])
def test_reserved_eager_execution_reuses_the_agreed_transaction(
    native_case, u2, padded
):
    case = native_case()
    case.owner.select_u2 = u2
    case.peer.negotiated_capabilities = replace(
        case.peer.negotiated_capabilities, graph_execution_version=1
    )
    decision = HcclGraphDecision(
        "agreed-eager", "eager", "target", None, padded, "requested"
    )
    case.peer.graph_reservation = decision

    # The outer graph lifecycle already owns this reservation. Exercise its
    # eager execution seam directly; the main runner now owns that outer
    # reservation lifecycle and cannot accept a manually injected decision.
    with (
        torch.inference_mode(),
        execution.HcclV41Execution(
            case.owner, case.peer, mode="target", valid_tokens=5, dummy_run=False
        ).scope(),
    ):
        output, _ = NPUModelRunner.execute_model(
            case.owner, SimpleNamespace(total_num_scheduled_tokens=5)
        )

    assert output.shape == (8, 4)
    assert case.peer.forwards[-1].transaction_id == "agreed-eager"
    assert case.peer.forwards[-1].graph_execution == decision
    assert case.peer.plan is None and not case.peer.terminal


def test_native_u1_pre_hook_failure_is_terminal_after_acceptance(native_case):
    case = native_case(with_ubatches=False)

    def failed_producer():
        raise RuntimeError("parent producer failed")

    case.state.prepare_engram = failed_producer
    with pytest.raises(RuntimeError, match="parent producer failed"):
        case.owner.execute_model(SimpleNamespace(total_num_scheduled_tokens=5))
    assert case.peer.terminal and case.peer.plan is not None
    assert case.peer.messages == []


@pytest.mark.parametrize(
    "boundary", ["duplicate", "legacy-started", "kv-started", "adapter-transport"]
)
def test_binding_rejects_a_second_or_late_session(native_case, boundary):
    case = native_case(with_ubatches=False, bind=boundary == "duplicate")
    if boundary == "legacy-started":
        case.owner.connector._initialized = True
    elif boundary == "kv-started":
        case.owner.ubatch_runner = SimpleNamespace()
    elif boundary == "adapter-transport":
        connector = object.__new__(HcclStagedAFDConnector)
        connector.transport = StagedAttentionPeer()
        connector.transport.initialized = False
        case.owner.connector = connector
    with pytest.raises(RuntimeError, match="before connector/KV startup"):
        case.owner.bind_hccl_v41_transport(case.peer)
    assert case.peer.forwards == []


def test_local_dp1_dispatch_preserves_threshold_fallback(native_case):
    case = native_case()
    config = case.owner.vllm_config.parallel_config
    config.dbo_prefill_token_threshold = 4
    with execution.HcclV41Execution(
        case.owner, case.peer, mode="target", valid_tokens=5, dummy_run=False
    ).scope():
        descriptors = [
            native_model_runner.dispatch_cg_and_sync_dp(
                None,
                1,
                tokens,
                None,
                1,
                0,
                need_eager=True,
                parallel_config=config,
                allow_ubatching=True,
            )[0]
            for tokens in (3, 5)
        ]
    assert [descriptor.num_ubatches for descriptor in descriptors] == [1, 2]


@pytest.mark.parametrize("stage_capacity", [1, 2])
def test_load_model_negotiates_before_native_u2_admission(
    monkeypatch, native_case, stage_capacity
):
    case = native_case(with_ubatches=False)
    case.peer.initialized = False
    monkeypatch.setattr(NPUModelRunner, "load_model", lambda *a, **k: None)

    def initialize():
        case.peer.initialized = True
        case.peer.negotiated_capabilities = replace(
            case.peer.capabilities, max_stages=stage_capacity
        )

    case.peer.initialize = initialize
    if stage_capacity == 1:
        with pytest.raises(ValueError, match="negotiated U2 capacity"):
            case.owner.load_model()
    else:
        case.owner.load_model()
    assert case.peer.initialized and case.peer.forwards == []


@pytest.mark.parametrize("native_fails", [False, True])
def test_shutdown_releases_staged_adapter_only_after_native_cleanup(
    monkeypatch, native_case, native_fails
):
    case = native_case(with_ubatches=False)
    connector = object.__new__(HcclStagedAFDConnector)
    connector.transport = case.peer
    case.owner.connector = connector
    monkeypatch.setattr(attention, "stop_afd_npu_profiler", lambda _: None)

    def native_shutdown(self):
        assert not case.peer.closed
        assert case.peer.messages == [("STOP",)]
        if native_fails:
            raise RuntimeError("native cleanup failed")

    monkeypatch.setattr(NPUModelRunner, "shutdown", native_shutdown)
    if native_fails:
        with pytest.raises(RuntimeError, match="native cleanup failed"):
            case.owner.shutdown()
        assert not case.peer.closed
    else:
        case.owner.shutdown()
        assert case.peer.closed


@pytest.mark.parametrize("native_u2", [False, True])
def test_binding_accepts_normal_staged_adapter_for_native_u1_or_u2(
    native_case, native_u2
):
    case = native_case(with_ubatches=False, bind=False)
    connector = object.__new__(HcclStagedAFDConnector)
    connector.transport = case.peer
    case.peer.initialized = False
    case.owner.connector = connector
    parallel = case.owner.vllm_config.parallel_config
    parallel.use_ubatching, parallel.num_ubatches = native_u2, 2 if native_u2 else 0
    case.owner.bind_hccl_v41_transport(case.peer)
    case.peer.initialized = True
    output, _ = case.owner.execute_model(SimpleNamespace(total_num_scheduled_tokens=5))
    assert output.shape == (8, 4) and case.peer.completed


@pytest.mark.parametrize("native_u2", [False, True])
def test_normal_runner_constructor_binds_selected_staged_adapter(
    monkeypatch, native_u2
):
    config = hccl_vllm_config(
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
    )
    config.model_config.hf_text_config.hidden_size = 4
    config.model_config.hf_text_config.vocab_size = 32
    config.parallel_config.rank = 0
    config.parallel_config.use_ubatching = native_u2
    config.parallel_config.num_ubatches = 2 if native_u2 else 0
    peer = StagedAttentionPeer()
    peer.initialized = False
    connector = object.__new__(HcclStagedAFDConnector)
    connector.transport = peer

    # The closed model guard is deliberately outside this constructor Python
    # seam. Native allocation and normal CLI startup still need device evidence.
    monkeypatch.setattr(
        attention, "fail_if_unsupported_npu_afd_features", lambda _: None
    )
    monkeypatch.setattr(attention, "create_npu_runtime_connector", lambda *a: connector)
    monkeypatch.setattr(attention, "_resolve_world_ranks", lambda: (0, 1))
    monkeypatch.setattr(attention, "create_afd_npu_profiler", lambda _: None)

    def native_init(self, vllm_config, device):
        self.vllm_config, self.device, self.ubatch_runner = vllm_config, device, None

    monkeypatch.setattr(NPUModelRunner, "__init__", native_init)
    runner = attention.AFDNPUAttentionModelRunnerV2(
        config, SimpleNamespace(type="npu", index=0)
    )
    assert runner.connector is connector and runner.hccl_v41_transport is peer
    assert not connector.is_initialized
