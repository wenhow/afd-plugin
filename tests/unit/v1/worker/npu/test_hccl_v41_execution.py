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
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from afd_plugin.connectors.npu.hccl_staged import HcclStagedAFDConnector
from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector
from afd_plugin.distributed.hccl_graph import HcclGraphDecision
from afd_plugin.model_executor.models import forward_context as context_module
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    AFDDeepseekV41AttentionForCausalLM,
)
from afd_plugin.v1.worker.npu import attention_model_runner_v2 as attention
from afd_plugin.v1.worker.npu import hccl_v41_execution as execution
from afd_plugin.v1.worker.npu import hccl_v41_ubatch_runner as ubatch
from tests.unit.fakes.deepseek_v41 import ParentEngramState, make_staged_model
from tests.unit.fakes.hccl import StagedAttentionPeer
from tests.unit.fakes.input_batch import make_ascend_batch, make_hccl_ubatch_seam
from tests.unit.fakes.vllm_config import hccl_vllm_config


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
        batch = make_ascend_batch(scheduler_output.total_num_scheduled_tokens)
        state = self.model_state
        state.consumed = False
        prepared = None
        if self.select_u2:
            prepared = self.ubatch_runner.prepare(
                batch, (), torch.arange(8, dtype=torch.int32).reshape(1, 8)
            )
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

    def build(*, with_ubatches=True, bind=True):
        config = hccl_vllm_config(
            dtype=torch.bfloat16,
            compilation_mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
        )
        config.model_config.hf_text_config.hidden_size = 4
        config.model_config.hf_text_config.vocab_size = 32
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
        owner.ubatch_runner, owner.prof = None, None
        owner._afd_transaction_counter = 0
        owner.select_u2 = False
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
        return SimpleNamespace(owner=owner, peer=peer, state=state, producer=producer)

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


@pytest.mark.parametrize("u2,padded", [(False, (8,)), (True, (4, 4))])
def test_graph_eager_fallback_uses_the_previously_agreed_transaction(
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

    output, _ = case.owner.execute_model(SimpleNamespace(total_num_scheduled_tokens=5))

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
