# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Host lifecycle entry seams; no NPU graph or transfer-engine I/O."""

from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")
msgspec = pytest.importorskip("msgspec")

import torch
from vllm import SamplingParams
from vllm.config import CUDAGraphMode
from vllm.utils import torch_utils
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.kv_cache_interface import (
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.sfa_v1 import AscendSFABackend
from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from afd_plugin.compat.npu import kv_layout_connector as pd_adapter
from afd_plugin.compat.npu.dspark_pd import (
    NULL_BLOCK_ID,
    DsparkPdReadQuery,
    read_lease_from_payload,
)
from afd_plugin.compat.npu.runtime_config import validate_deepseek_v41_pd
from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.model_executor.models import forward_context as context_module
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    HCCL_V41_PARENT_FORWARD_KEY,
)
from afd_plugin.model_executor.models.npu.dsv41_packed_cache import (
    native as packed_native,
)
from afd_plugin.v1.worker.npu import attention_model_runner_v2 as runner_module
from afd_plugin.v1.worker.npu import hccl_v41_graph_coordinator as coordinator_module
from tests.unit.fakes.attention_model_runner_v2 import (
    install_graph_runner_coordinator,
    make_dspark_pd_runner_case,
    make_graph_runner_case,
    make_legacy_attention_startup_case,
    queue_graph_runner_stop,
)
from tests.unit.fakes.block_table import install_native_block_table_kernels_cpu
from tests.unit.fakes.deepseek_v41 import PD_CPU_TARGET_TOKEN
from tests.unit.fakes.dspark_speculator import make_dspark_proposal_case
from tests.unit.fakes.dsv41_packed_cache import (
    headwise_attention_with_sink,
    make_packed_attention_inputs,
    packed_64_head_attention_reference,
)
from tests.unit.fakes.engine_core import (
    make_native_output_engine_case,
    make_native_worker_output_case,
)
from tests.unit.fakes.hccl_v41_graph_bootstrap import (
    make_attention_eager_bootstrap_case,
    make_graph_bootstrap_case,
    set_graph_bootstrap_environment,
)
from tests.unit.fakes.hccl_v41_graph_eager import run_graph_eager_parent
from tests.unit.fakes.input_batch import install_native_input_preparation_cpu
from tests.unit.fakes.kv_connector import (
    FULL_REPLAY_HISTORY_TOKENS,
    SpeculativeLogitsBoundary,
    advance_native_dspark_sampling_case,
    complete_native_pd_handoff,
    initialize_native_pd_sampling,
    initialize_native_pd_target_execution,
    make_native_dspark_factory_case,
    make_native_dspark_sampling_case,
    make_native_full_kv_replay_case,
    make_native_full_kv_sampling_case,
    make_native_kv_runner_case,
    make_native_pd_graph_case,
    make_native_pd_prefill_owner_case,
    make_native_u2_prefill_prefix_case,
    make_scheduled_pd_runner_case,
    native_dspark_eager_reference,
    native_pd_target_step_reference,
)
from tests.unit.fakes.kv_layout_connector import (
    DESTINATION_SENTINEL,
    make_scheduled_dspark_read_case,
)
from tests.unit.fakes.logging import (
    SESSION_CONFIGURATION_PREFIX,
    structured_log_records,
)
from tests.unit.fakes.logging import serving_logs as serving_logs
from tests.unit.fakes.request_state import (
    make_native_request_admission_case,
    native_zeroed_cache_reference,
)

PD_RETIRED_OUTPUT_TOKENS = 2
PD_REPLACEMENT_PROMPT = (40, 41, 42)
PD_REPLACEMENT_MAX_TOKENS = 3


@pytest.fixture
def scheduled_pd_zero_case(monkeypatch, request):
    transfer, runner = make_scheduled_pd_runner_case(
        monkeypatch, prompt_length=request.param
    )
    request.addfinalizer(transfer.close)
    return transfer, runner


@pytest.fixture
def native_pd_target_case(scheduled_pd_zero_case, monkeypatch):
    transfer, runner = scheduled_pd_zero_case
    resumed = complete_native_pd_handoff(runner)
    initialize_native_pd_target_execution(monkeypatch, runner, scheduler_output=resumed)
    return transfer, runner, resumed


@pytest.fixture
def native_pd_u2_case(monkeypatch, request):
    transfer, runner = make_scheduled_pd_runner_case(
        monkeypatch, prompt_length=request.param, stages=2
    )
    request.addfinalizer(transfer.close)
    resumed = complete_native_pd_handoff(runner)
    initialize_native_pd_target_execution(monkeypatch, runner, scheduler_output=resumed)
    request.addfinalizer(runner.target_hook.remove)
    initialize_native_pd_sampling(monkeypatch, runner)
    return transfer, runner, resumed


@pytest.fixture
def native_pd_graph_case(monkeypatch, request):
    stages, fail_replay = (
        request.param if isinstance(request.param, tuple) else (request.param, False)
    )
    transfer, runner = make_native_pd_graph_case(
        monkeypatch, stages=stages, fail_replay=fail_replay
    )
    request.addfinalizer(transfer.close)
    if stages == 2:
        request.addfinalizer(runner.target_hook.remove)
    return transfer, runner


@pytest.fixture
def native_u2_prefill_case(monkeypatch):
    case = make_native_u2_prefill_prefix_case(monkeypatch)
    with monkeypatch.context() as execution_patch:
        execution_patch.setattr(
            "vllm.distributed.kv_transfer.kv_transfer_state._KV_CONNECTOR_AGENT", None
        )
        assert case.owner.execute_model(case.scheduler) is None
    return case


@pytest.mark.parametrize("boundary", ["execute", "proposal"])
def test_pd_ready_waits_for_native_p_sampling(  # long-test: public Ready frontier
    monkeypatch, boundary
):
    case = make_native_pd_prefill_owner_case(monkeypatch)
    owner, prefix, producer = case.owner, case.prefix, case.producer
    assert owner.vllm_config is prefix.config
    assert owner.dspark_pd_consumer.vllm_config is owner.vllm_config
    assert owner.dspark_pd_consumer.binding.target_model is owner.model
    assert all(request.sampling_params.max_tokens == 1 for request in case.requests)
    assert case.scheduler.has_sync_kv_loads is False
    assert owner.execute_model(case.scheduler) is None
    state = owner.execute_model_state
    assert len(state.aux_hidden_states) == 3
    positions = state.input_batch.positions.clone()
    offsets = state.input_batch.query_start_loc_np.copy()
    expected = sum(
        (index + 1) * auxiliary.float()
        for index, auxiliary in enumerate(state.aux_hidden_states)
    ).to(torch.bfloat16)[:, :2]
    if boundary == "execute":
        assert producer.request_ready(case.query)["ok"] is False
    case.observations.enabled = True
    output = owner.sample_tokens(None).get_output()
    assert output.sampled_token_ids == [[PD_CPU_TARGET_TOKEN]] * len(case.requests)
    # Observe the public peer result while native proposal/store is executing.
    # Equal final bytes alone would not prove leased storage was quiescent.
    assert case.observations.replies
    assert all(reply["ok"] is False for reply in case.observations.replies)
    lease = read_lease_from_payload(producer.request_ready(case.query)["lease"])
    assert lease.ready == prefix.expected_ready
    middle = state.input_batch.req_ids.index(prefix.source.request_id)
    for group, name in zip(prefix.draft_group_ids, prefix.draft_names, strict=True):
        cache = prefix.caches[name]
        for position in range(prefix.source.computed_prefix_tokens):
            block = prefix.source.allocation_block_ids[group][
                position // cache.shape[1]
            ]
            torch.testing.assert_close(
                cache[block, position % cache.shape[1], 0],
                expected[offsets[middle] + position],
                rtol=0,
                atol=0,
            )
    torch.testing.assert_close(state.input_batch.positions, positions, rtol=0, atol=0)
    assert producer.complete_read(lease) == b"ACK"
    assert producer.request_ready(case.query)["ok"] is False


@pytest.mark.parametrize("boundary", ["execute", "shutdown"])
def test_pending_p_sampling_preserves_native_state_before_reentry(
    monkeypatch, boundary
):
    case = make_native_pd_prefill_owner_case(monkeypatch)
    owner = case.owner
    assert owner.execute_model(case.scheduler) is None
    state = owner.execute_model_state
    views = tuple(case.prefix.caches[name] for name in case.prefix.draft_names)
    before = tuple(view.clone() for view in views)
    with pytest.raises(RuntimeError, match="sampling|pending"):
        if boundary == "execute":
            owner.execute_model(case.scheduler)
        else:
            owner.prepare_shutdown()
    assert owner.execute_model_state is state
    assert case.producer.request_ready(case.query)["ok"] is False
    for actual, unchanged in zip(views, before, strict=True):
        torch.testing.assert_close(actual, unchanged, rtol=0, atol=0)


def test_p_proposal_failure_retains_cause_and_rejects_ready_retry(monkeypatch, request):
    case = make_native_pd_prefill_owner_case(monkeypatch)
    cause = RuntimeError("controlled native P proposal failure")

    def fail_proposal(module, args):
        raise cause

    draft_layer = next(iter(case.owner.speculator.model.model.layers.values()))
    hook = draft_layer.self_attn.register_forward_pre_hook(fail_proposal)
    request.addfinalizer(hook.remove)
    assert case.owner.execute_model(case.scheduler) is None
    with pytest.raises(
        RuntimeError, match="controlled native P proposal failure"
    ) as caught:
        case.owner.sample_tokens(None)
    assert caught.value is cause
    assert case.owner.dspark_pd_consumer.terminal
    assert case.producer.request_ready(case.query)["ok"] is False
    with pytest.raises(RuntimeError, match="terminal"):
        case.owner.sample_tokens(None)
    assert case.producer.request_ready(case.query)["ok"] is False


@pytest.mark.parametrize("stages", [1, 2])
def test_same_p_owner_keeps_prefill_coordinates_until_ready(  # long-test: one parent
    monkeypatch, stages
):
    case = make_native_pd_prefill_owner_case(monkeypatch, stages=stages)
    owner, prefix = case.owner, case.prefix
    config, consumer, model = owner.vllm_config, owner.dspark_pd_consumer, owner.model
    assert config is prefix.config and consumer.vllm_config is config
    assert consumer.binding.target_model is model
    assert case.scheduler.has_sync_kv_loads is False
    assert owner.execute_model(case.scheduler) is None
    state = owner.execute_model_state
    batch = state.input_batch
    assert batch.query_start_loc_np.tolist() == [0, 2, 7, 13]
    assert batch.num_scheduled_tokens.tolist() == [2, 5, 6]
    assert batch.prefill_len_np.tolist() == [2, 5, 6]
    assert batch.num_computed_tokens_np.tolist() == [0, 0, 0]
    assert len(state.aux_hidden_states) == 3
    positions, offsets = batch.positions.clone(), batch.query_start_loc_np.copy()
    projected = sum(
        (index + 1) * auxiliary.float()
        for index, auxiliary in enumerate(state.aux_hidden_states)
    ).to(torch.bfloat16)[:, :2]
    for group in range(len(prefix.layout.groups)):
        allocated = [
            set(source.allocation_block_ids[group]) - {NULL_BLOCK_ID}
            for source in case.sources
        ]
        assert all(
            left.isdisjoint(right)
            for index, left in enumerate(allocated)
            for right in allocated[index + 1 :]
        )
    if stages == 2:
        (plan,) = [
            message.plan
            for direction, _, _, message in case.transport.transcript
            if direction == "send" and message.kind == "FORWARD"
        ]
        assert plan.mode == "target" and plan.layer_indices == (0, 1, 2)
        assert [stage.attention_valid_tokens for stage in plan.stages] == [(6,), (7,)]
        assert [stage.attention_padded_tokens for stage in plan.stages] == [(6,), (7,)]
        assert [
            message.transaction_id
            for direction, _, _, message in case.transport.transcript
            if direction == "recv" and message.kind == "COMPLETE"
        ] == [plan.transaction_id]
    assert case.producer.request_ready(case.query)["ok"] is False
    case.observations.enabled = True
    output = owner.sample_tokens(None).get_output()
    assert output.sampled_token_ids == [[PD_CPU_TARGET_TOKEN]] * len(case.requests)
    assert case.observations.replies
    assert all(reply["ok"] is False for reply in case.observations.replies)
    lease = read_lease_from_payload(case.producer.request_ready(case.query)["lease"])
    assert lease.ready == prefix.expected_ready
    sources = {source.request_id: source for source in case.sources}
    for row, request_id in enumerate(batch.req_ids):
        source = sources[request_id]
        for group, name in zip(prefix.draft_group_ids, prefix.draft_names, strict=True):
            cache = prefix.caches[name]
            for position in range(source.computed_prefix_tokens):
                block = source.allocation_block_ids[group][position // cache.shape[1]]
                torch.testing.assert_close(
                    cache[block, position % cache.shape[1], 0],
                    projected[offsets[row] + position],
                    rtol=0,
                    atol=0,
                )
    torch.testing.assert_close(batch.positions, positions, rtol=0, atol=0)
    assert owner.vllm_config is config and owner.dspark_pd_consumer is consumer
    assert consumer.binding.target_model is owner.model is model
    assert owner.execute_model_state is None and not consumer.terminal
    assert case.producer.complete_read(lease) == b"ACK"
    assert case.producer.request_ready(case.query)["ok"] is False


def test_u2_prefill_keeps_parent_aux(  # long-test: three-request projection
    native_u2_prefill_case,
):
    case = native_u2_prefill_case
    owner, prefix = case.owner, case.prefix
    state = owner.execute_model_state
    batch = state.input_batch
    assert batch.req_ids == case.expected_target.req_ids
    assert batch.query_start_loc_np.tolist() == [0, 2, 7, 13]
    assert batch.num_scheduled_tokens.tolist() == [2, 5, 6]
    assert batch.prefill_len_np.tolist() == [2, 5, 6]
    assert batch.num_computed_tokens_np.tolist() == [0, 0, 0]
    assert batch.positions.tolist() == [0, 1, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]
    assert batch.input_ids.tolist() == [0, 1, 0, 1, 2, 3, 4, 20, 21, 22, 23, 24, 25]
    (plan,) = [
        message.plan
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == "FORWARD"
    ]
    assert plan.mode == "target" and plan.layer_indices == (0, 1, 2)
    assert [stage.attention_valid_tokens for stage in plan.stages] == [(6,), (7,)]
    assert [stage.attention_padded_tokens for stage in plan.stages] == [(6,), (7,)]
    assert (
        batch.query_start_loc_np[1]
        < plan.stages[0].attention_valid_tokens[0]
        < batch.query_start_loc_np[2]
    )
    assert [
        message.transaction_id
        for direction, _, _, message in case.transport.transcript
        if direction == "recv" and message.kind == "COMPLETE"
    ] == [plan.transaction_id]
    torch.testing.assert_close(
        state.hidden_states, case.expected_target.hidden_states, rtol=0, atol=0
    )
    for actual, expected in zip(
        state.aux_hidden_states, case.expected_target.aux_hidden_states, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(case.pages, case.expected_target.pages, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert owner.vllm_config.kv_transfer_config is None
    assert owner.parallel_config.use_ubatching
    validate_deepseek_v41_pd(prefix.config)
    assert not prefix.config.parallel_config.use_ubatching
    assert prefix.consumer.binding.target_model is owner.model
    for group in range(len(prefix.kv_config.kv_cache_groups)):
        allocated = [
            set(source.allocation_block_ids[group]) - {0} for source in case.sources
        ]
        assert all(allocated)
        assert all(
            left.isdisjoint(right)
            for index, left in enumerate(allocated)
            for right in allocated[index + 1 :]
        )
    ready = prefix.consumer.consume_prefill(case.scheduler, state, case.sources)
    assert {item.source_request_id for item in ready} == set(batch.req_ids)
    assert {item.source_generation for item in ready} == {
        source.source_generation for source in case.sources
    }
    # The leaf reads prepared prefixes, not intra-chunk causal Attention.
    # The independent oracle models child segments, without claiming U1 parity.
    projected = sum(
        (index + 1) * layer.float()
        for index, layer in enumerate(case.expected_target.aux_hidden_states)
    )
    projected = projected.to(torch.bfloat16)[:, :2]
    sources = {source.request_id: source for source in case.sources}
    for row, request_id in enumerate(batch.req_ids):
        source = sources[request_id]
        for group, name in zip(prefix.draft_group_ids, prefix.draft_names, strict=True):
            cache = prefix.caches[name]
            for position in range(source.computed_prefix_tokens):
                block = source.allocation_block_ids[group][position // cache.shape[1]]
                torch.testing.assert_close(
                    cache[block, position % cache.shape[1], 0],
                    projected[batch.query_start_loc_np[row] + position],
                    rtol=0,
                    atol=0,
                )
    assert owner.execute_model_state is state and not prefix.consumer.terminal


def test_u2_prefill_reaches_pd_sample(  # long-test: Ready/read/DONE and sample
    native_u2_prefill_case,
    monkeypatch,
    request,
):
    source = native_u2_prefill_case
    state = source.owner.execute_model_state
    transfer = make_scheduled_dspark_read_case(
        monkeypatch,
        source.prefix,
        producer=source.producer,
        producer_execution=(source.scheduler, state, source.sources),
        sampling_params=SamplingParams(temperature=0),
    )
    request.addfinalizer(transfer.close)
    runner = make_native_kv_runner_case(
        monkeypatch, post_kv=False, sync_load=False, pd_read_case=transfer
    )
    complete_native_pd_handoff(runner, schedule_target=False)
    prefix_tokens = source.prefix.source.computed_prefix_tokens
    assert transfer.request.num_computed_tokens == prefix_tokens
    resumed = transfer.scheduler.schedule()
    assert resumed.scheduled_new_reqs[0].num_computed_tokens == prefix_tokens
    scheduled_width = resumed.num_scheduled_tokens[transfer.request.request_id]
    assert transfer.request.num_computed_tokens == prefix_tokens + scheduled_width
    initialize_native_pd_target_execution(monkeypatch, runner, scheduler_output=resumed)
    initialize_native_pd_sampling(monkeypatch, runner)
    assert runner.owner.execute_model(resumed) is None
    actual = runner.owner.execute_model_state
    torch.testing.assert_close(
        actual.hidden_states, runner.expected_target.hidden_states, rtol=0, atol=0
    )
    output = runner.owner.sample_tokens(None).get_output()
    assert output.sampled_token_ids == [[PD_CPU_TARGET_TOKEN]]
    drafts = runner.owner.take_draft_token_ids()
    assert drafts.req_ids == [transfer.request.request_id]
    assert drafts.draft_token_ids == [[-1] * runner.owner.num_speculative_steps]
    index = runner.owner.req_states.req_id_to_index[transfer.request.request_id]
    assert runner.owner.req_states.draft_tokens[index].tolist() == list(
        range(
            PD_CPU_TARGET_TOKEN + 1,
            PD_CPU_TARGET_TOKEN + 1 + runner.owner.num_speculative_steps,
        )
    )
    assert transfer.request.num_computed_tokens == prefix_tokens + scheduled_width
    assert runner.owner.model is transfer.loaded_destination.target_model
    assert runner.owner.model is not source.owner.model
    assert runner.owner.speculator is transfer.loaded_destination.speculator
    assert source.owner.execute_model_state is state
    assert not source.prefix.consumer.terminal
    kinds = [msgspec.msgpack.decode(message)[0] for message in transfer.sent]
    assert kinds == [
        pd_adapter.GET_META_MSG,
        pd_adapter.GET_DSPARK_READY_MSG,
        pd_adapter.DONE_DSPARK_RECV_MSG,
    ]


@pytest.fixture
def native_pd_output_case(scheduled_pd_zero_case, monkeypatch, fail_completion):
    transfer, runner = scheduled_pd_zero_case
    complete_native_pd_handoff(runner, schedule_target=False)
    initialize_native_pd_target_execution(monkeypatch, runner)
    initialize_native_pd_sampling(monkeypatch, runner)
    return make_native_output_engine_case(
        monkeypatch, runner, transfer, fail_completion=fail_completion
    )


@pytest.mark.parametrize("native_pd_graph_case", [1, 2], indirect=True)
def test_native_pd_graph_handoff_enters_cached_replay(  # long-test: PD Graph parent
    native_pd_graph_case, monkeypatch
):
    transfer, case = native_pd_graph_case
    owner, destination = case.owner, transfer.destination_cache
    # Supplied static bundle startup has not polled or consumed a PD message.
    # Its CPU allocator label does not prove real NPU device admission/capture.
    assert case.startup_pd_messages == ()
    assert (
        owner.vllm_config.compilation_config.cudagraph_mode
        == CUDAGraphMode.FULL_DECODE_ONLY
    )
    assert transfer.prefix_case.config.model_config.enforce_eager
    scheduled = complete_native_pd_handoff(case)
    request = scheduled.scheduled_new_reqs[0]
    width = scheduled.num_scheduled_tokens[request.req_id]
    assert width == 3 and request.num_computed_tokens == 5
    protocol = tuple(transfer.sent)
    expected = native_pd_target_step_reference(
        case,
        scheduled,
        input_ids=torch.tensor([5, 0, 0]),
        computed=5,
        block_ids=request.block_ids,
        stages=owner.parallel_config.num_ubatches,
    )
    assert owner.execute_model(scheduled) is None
    first = owner.execute_model_state
    assert first.input_batch.input_ids.tolist() == [5, 0, 0]
    assert first.input_batch.positions.tolist() == [5, 6, 7]
    torch.testing.assert_close(
        first.hidden_states, expected.hidden_states, rtol=0, atol=0
    )
    for actual, target in zip(destination.pages, expected.pages, strict=True):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    sampled = owner.sample_tokens(None).get_output()
    assert sampled.sampled_token_ids == [[PD_CPU_TARGET_TOKEN]]
    transfer.scheduler.update_from_output(scheduled, sampled)
    transfer.scheduler.update_draft_token_ids(owner.take_draft_token_ids())
    scheduled = transfer.scheduler.schedule()
    assert scheduled.scheduled_cached_reqs.req_ids == [request.req_id]
    assert scheduled.num_scheduled_tokens == {request.req_id: 3}

    def forbid_target_python(module, inputs):
        raise AssertionError("inert FULL replay must not rerun target Python")

    hook = owner.model.register_forward_pre_hook(forbid_target_python)
    try:
        assert owner.execute_model(scheduled) is None
    finally:
        hook.remove()
    step = owner.execute_model_state
    assert step.input_batch.input_ids.tolist() == [7, 8, 9]
    assert step.input_batch.positions.tolist() == [6, 7, 8]
    assert step.input_batch.seq_lens[:1].tolist() == [9]
    hidden, auxiliary = case.recorded_outputs
    torch.testing.assert_close(step.hidden_states, hidden, rtol=0, atol=0)
    for actual, target in zip(step.aux_hidden_states, auxiliary, strict=True):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    plans = [
        message.plan
        for direction, _, _, message in case.transport.transcript
        if direction == "send"
        and message.kind == "FORWARD"
        and message.plan.mode == "target"
        and sum(stage.attention_valid_tokens[0] for stage in message.plan.stages)
    ]
    assert [plan.graph_execution.phase for plan in plans] == ["eager", "replay"]
    assert [stage.attention_valid_tokens for stage in plans[-1].stages] == (
        [(3,)] if owner.parallel_config.num_ubatches == 1 else [(1,), (2,)]
    )
    monkeypatch.setattr(
        owner.model.language_model,
        "compute_logits",
        SpeculativeLogitsBoundary(4, 128, ordinal_rows=True, predictions=(8, 7, 7)),
    )
    output = owner.sample_tokens(None).get_output()
    assert output.sampled_token_ids == [[8, 7]]
    index = owner.req_states.req_id_to_index[request.req_id]
    assert owner.req_states.draft_tokens[index].tolist() == [8, 9]
    assert owner.req_states.all_token_ids.gpu[index, :9].tolist() == [
        0,
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        7,
    ]
    assert owner.req_states.num_computed_tokens.gpu[index].item() == 8
    transfer.scheduler.update_from_output(scheduled, output)
    assert tuple(transfer.sent) == protocol
    assert not owner.dspark_pd_consumer.terminal
    for actual, target in zip(
        destination.source_pages, case.source_pages_before, strict=True
    ):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)


@pytest.mark.parametrize("native_pd_graph_case", [(2, True)], indirect=True, ids=["u2"])
def test_native_pd_graph_replay_failure_retains_owner(  # long-test: failed PD replay
    native_pd_graph_case,
):
    transfer, case = native_pd_graph_case
    owner, destination = case.owner, transfer.destination_cache
    scheduled = complete_native_pd_handoff(case)
    assert owner.execute_model(scheduled) is None
    output = owner.sample_tokens(None).get_output()
    transfer.scheduler.update_from_output(scheduled, output)
    transfer.scheduler.update_draft_token_ids(owner.take_draft_token_ids())
    scheduled = transfer.scheduler.schedule()
    protocol = tuple(transfer.sent)
    model, speculator, consumer = (
        owner.model,
        owner.speculator,
        owner.dspark_pd_consumer,
    )
    pages = tuple(destination.pages)
    addresses = tuple(page.data_ptr() for page in pages)
    before = len(case.transport.transcript)

    with pytest.raises(
        RuntimeError, match="controlled D PD FULL replay failure"
    ) as error:
        owner.execute_model(scheduled)

    assert error.value is case.replay_error
    assert consumer.terminal and case.cache.terminal and case.transport.terminal
    assert not case.cache.is_empty and not case.cache.closed
    assert not case.transport.closed and not case.recorded_graph.closed
    assert owner.model is model and owner.speculator is speculator
    assert owner.dspark_pd_consumer is consumer
    assert tuple(transfer.sent) == protocol
    # Native preparation may clear cache padding before the failed device
    # boundary. Failure retains those bytes; neither retry nor shutdown may
    # mutate them again. This does not promise transactional KV rollback.
    values = tuple(page.clone() for page in destination.pages)
    for actual, expected in zip(
        destination.source_pages, case.source_pages_before, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    messages = [
        (direction, message)
        for direction, _, _, message in case.transport.transcript[before:]
    ]
    assert [
        message.kind
        for direction, message in messages
        if direction == "send" and message.kind == "FORWARD"
    ] == ["FORWARD"]
    assert any(
        direction == "recv" and message.kind == "ACCEPT"
        for direction, message in messages
    )
    assert not any(message.kind in ("COMPLETE", "STOP") for _, message in messages)
    assert (
        next(
            message
            for direction, message in messages
            if direction == "send" and message.kind == "FORWARD"
        ).plan.graph_execution.phase
        == "replay"
    )
    failed_transcript = tuple(case.transport.transcript)
    with pytest.raises(RuntimeError, match="DSpark prefix consumer is terminal"):
        owner.execute_model(scheduled)
    with pytest.raises(RuntimeError, match="DSpark prefix consumer is terminal"):
        owner.prepare_shutdown()
    assert tuple(transfer.sent) == protocol
    assert tuple(case.transport.transcript) == failed_transcript
    assert tuple(page.data_ptr() for page in destination.pages) == addresses
    for actual, expected in zip(destination.pages, values, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("scheduled_pd_zero_case", [1, 6], indirect=True)
@pytest.mark.parametrize("engine_step", ["step", "step_with_batch_queue"])
@pytest.mark.parametrize("fail_completion", [False, True])
def test_pd_output_completion_precedes_scheduler_update(  # long-test: output seam
    native_pd_output_case, engine_step, fail_completion
):
    case = native_pd_output_case
    engine, transfer, owner = case.engine, case.transfer, case.runner.owner
    step = engine.step if engine_step == "step" else engine.step_with_batch_queue
    if fail_completion:
        with pytest.raises(RuntimeError, match="device completion failed"):
            step()
        assert list(transfer.request.output_token_ids) == []
    else:
        outputs, executed = step()
        assert executed
        assert [
            (output.request_id, output.new_token_ids)
            for batch in outputs.values()
            for output in batch.outputs
        ] == [(transfer.request.request_id, [PD_CPU_TARGET_TOKEN])]
        assert list(transfer.request.output_token_ids) == [PD_CPU_TARGET_TOKEN]

    # Both native sample/proposal and KV post_forward have already completed.
    # A delayed copy error does not rewind the worker to the scheduler frontier.
    prompt = list(transfer.request.prompt_token_ids)
    state = owner.req_states
    index = state.req_id_to_index[transfer.request.request_id]
    assert state.total_len.gpu[index].item() == len(prompt) + 1
    assert state.num_computed_tokens.gpu[index].item() == len(prompt)
    assert state.all_token_ids.gpu[index, : len(prompt) + 1].tolist() == (
        prompt + [PD_CPU_TARGET_TOKEN]
    )
    assert state.draft_tokens[index].tolist() == list(
        range(
            PD_CPU_TARGET_TOKEN + 1,
            PD_CPU_TARGET_TOKEN + 1 + owner.num_speculative_steps,
        )
    )
    assert owner.execute_model_state is None
    assert tuple(transfer.sent) == case.driver.protocol_after_sampling
    assert [msgspec.msgpack.decode(message)[0] for message in transfer.sent] == [
        pd_adapter.GET_META_MSG,
        pd_adapter.GET_DSPARK_READY_MSG,
        pd_adapter.DONE_DSPARK_RECV_MSG,
    ]
    for actual, expected in zip(
        transfer.destination_cache.pages, case.driver.pages_after_sampling, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(
        transfer.destination_cache.source_pages,
        case.runner.source_pages_before,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("scheduled_pd_zero_case", [1, 6], indirect=True)
@pytest.mark.parametrize("fail_completion", [False, True])
def test_native_pd_worker_output_reports_completion_failure(
    native_pd_output_case, fail_completion
):
    case = native_pd_output_case
    scheduled = case.transfer.scheduler.schedule()
    assert case.engine.model_executor.execute_model(scheduled) is None
    output = case.driver.sample_tokens(None)
    worker, responses = make_native_worker_output_case()

    worker.enqueue_output(output)

    status, payload = responses.get_nowait()
    if fail_completion:
        assert status is WorkerProc.ResponseStatus.FAILURE
        assert payload == "device completion failed"
    else:
        assert status is WorkerProc.ResponseStatus.SUCCESS
        assert payload.req_ids == [case.transfer.request.request_id]
        assert payload.sampled_token_ids == [[PD_CPU_TARGET_TOKEN]]
    assert responses.empty()
    assert list(case.transfer.request.output_token_ids) == []
    state = case.runner.owner.req_states
    index = state.req_id_to_index[case.transfer.request.request_id]
    assert (
        state.total_len.gpu[index].item()
        == len(case.transfer.request.prompt_token_ids) + 1
    )
    assert tuple(case.transfer.sent) == case.driver.protocol_after_sampling
    for actual, expected in zip(
        case.transfer.destination_cache.pages,
        case.driver.pages_after_sampling,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture
def retired_native_pd_case(native_pd_target_case, monkeypatch, request):
    """Retire an actual two-round D request, then reuse its ID and native owner.

    The replacement is a local prefill on the same registered consumer, not a
    second remote transfer. Native admission, sampling, scheduler retirement
    and block allocation remain real. Engram is disabled in this CPU model.
    """
    transfer, runner, scheduled = native_pd_target_case
    owner, previous = runner.owner, transfer.request
    if request.param == RequestStatus.FINISHED_LENGTH_CAPPED:
        previous.max_tokens = previous.sampling_params.max_tokens = (
            PD_RETIRED_OUTPUT_TOKENS
        )
    initialize_native_pd_sampling(monkeypatch, runner)
    for round_index in range(PD_RETIRED_OUTPUT_TOKENS):
        assert owner.execute_model(scheduled) is None
        output = owner.sample_tokens(None).get_output()
        transfer.scheduler.update_from_output(scheduled, output)
        if round_index == 0:
            transfer.scheduler.update_draft_token_ids(owner.take_draft_token_ids())
            scheduled = transfer.scheduler.schedule()
    if request.param == RequestStatus.FINISHED_ABORTED:
        transfer.scheduler.finish_requests(previous.request_id, request.param)
    previous_index = owner.req_states.req_id_to_index[previous.request_id]
    replacement = Request(
        previous.request_id,
        list(PD_REPLACEMENT_PROMPT),
        SamplingParams(temperature=0, max_tokens=PD_REPLACEMENT_MAX_TOKENS),
        None,
    )
    transfer.scheduler.add_request(replacement)
    scheduled = transfer.scheduler.schedule()
    expected = native_pd_target_step_reference(
        runner,
        scheduled,
        input_ids=torch.tensor(replacement.prompt_token_ids),
        computed=0,
        block_ids=scheduled.scheduled_new_reqs[0].block_ids,
    )
    return SimpleNamespace(
        transfer=transfer,
        runner=runner,
        previous=previous,
        previous_index=previous_index,
        replacement=replacement,
        scheduled=scheduled,
        expected=expected,
        expected_status=request.param,
    )


@pytest.mark.parametrize("scheduled_pd_zero_case", [1, 6], indirect=True)
@pytest.mark.parametrize(
    "retired_native_pd_case",
    [RequestStatus.FINISHED_LENGTH_CAPPED, RequestStatus.FINISHED_ABORTED],
    indirect=True,
)
def test_native_pd_finished_id_reuse_starts_with_fresh_history(  # long-test: ID reuse
    retired_native_pd_case,
):
    case = retired_native_pd_case
    transfer, runner = case.transfer, case.runner
    owner, destination = runner.owner, transfer.destination_cache
    scheduled, expected = case.scheduled, case.expected
    request_id, prompt = case.replacement.request_id, case.replacement.prompt_token_ids
    assert case.previous.status == case.expected_status
    assert (
        list(case.previous.output_token_ids)
        == [PD_CPU_TARGET_TOKEN] * PD_RETIRED_OUTPUT_TOKENS
    )
    assert scheduled.finished_req_ids == {request_id}
    assert scheduled.num_scheduled_tokens == {request_id: len(prompt)}
    assert scheduled.scheduled_spec_decode_tokens == {}
    assert scheduled.scheduled_new_reqs[0].num_computed_tokens == 0
    assert scheduled.kv_connector_metadata.requests == {}
    assert scheduled.kv_connector_metadata.dspark_destinations == {}

    assert owner.execute_model(scheduled) is None
    state = owner.req_states
    index = state.req_id_to_index[request_id]
    # Reusing the actual native slot makes stale counters/history observable.
    assert index == case.previous_index
    assert state.num_reqs == 1
    assert state.total_len.gpu[index].item() == len(prompt)
    assert state.num_computed_tokens.gpu[index].item() == 0
    assert state.all_token_ids.gpu[index, : len(prompt)].tolist() == prompt
    assert state.draft_tokens[index].tolist() == [0] * owner.num_speculative_steps
    # Native PenaltiesState initializes histogram rows only when penalties are
    # enabled. This greedy/no-penalty request must not require unused bins to
    # be cleared; its live history, target input and draft context are checked.
    step = owner.execute_model_state
    assert step.input_batch.input_ids.tolist() == prompt
    assert step.input_batch.positions.tolist() == list(range(len(prompt)))
    torch.testing.assert_close(
        step.hidden_states, expected.hidden_states, rtol=0, atol=0
    )
    for actual, target in zip(
        step.aux_hidden_states, expected.aux_hidden_states, strict=True
    ):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    for actual, target in zip(destination.pages, expected.pages, strict=True):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)

    output = owner.sample_tokens(None).get_output()
    assert output.req_ids == [request_id]
    assert output.sampled_token_ids == [[PD_CPU_TARGET_TOKEN]]
    assert state.total_len.gpu[index].item() == len(prompt) + 1
    assert state.num_computed_tokens.gpu[index].item() == len(prompt)
    assert state.all_token_ids.gpu[index, : len(prompt) + 1].tolist() == (
        prompt + [PD_CPU_TARGET_TOKEN]
    )
    assert state.draft_tokens[index].tolist() == list(
        range(
            PD_CPU_TARGET_TOKEN + 1,
            PD_CPU_TARGET_TOKEN + 1 + owner.num_speculative_steps,
        )
    )
    combined = sum(
        (ordinal + 1) * hidden.float()
        for ordinal, hidden in enumerate(expected.aux_hidden_states)
    ).to(torch.bfloat16)
    block_ids = scheduled.scheduled_new_reqs[0].block_ids
    for group_id in owner.speculator.draft_kv_cache_group_ids:
        group = destination.config.kv_cache_groups[group_id]
        for position in range(len(prompt)):
            block, offset = divmod(position, group.kv_cache_spec.block_size)
            for name in group.layer_names:
                torch.testing.assert_close(
                    destination.caches[name][block_ids[group_id][block], offset, 0],
                    combined[position, :2],
                    rtol=0,
                    atol=0,
                )
    for actual, target in zip(
        destination.source_pages, runner.source_pages_before, strict=True
    ):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    assert [msgspec.msgpack.decode(frame)[0] for frame in transfer.sent] == [
        pd_adapter.GET_META_MSG,
        pd_adapter.GET_DSPARK_READY_MSG,
        pd_adapter.DONE_DSPARK_RECV_MSG,
    ]
    transfer.scheduler.update_from_output(scheduled, output)
    assert list(case.replacement.output_token_ids) == [PD_CPU_TARGET_TOKEN]
    assert case.replacement.num_computed_tokens == len(prompt)
    assert owner.execute_model_state is None
    assert owner.model is transfer.loaded_destination.target_model
    assert owner.speculator is transfer.loaded_destination.speculator


@pytest.mark.parametrize("scheduled_pd_zero_case", [1, 6], indirect=True)
def test_native_pd_target_consumes_received_cache(  # long-test: full first D execute
    native_pd_target_case,
):
    transfer, runner, resumed = native_pd_target_case
    destination, owner = transfer.destination_cache, runner.owner
    new_request = resumed.scheduled_new_reqs[0]
    computed = new_request.num_computed_tokens
    width = resumed.num_scheduled_tokens[transfer.request.request_id]
    expected = runner.expected_target
    assert owner.execute_model(resumed) is None
    step = owner.execute_model_state
    assert step.input_batch.input_ids.tolist() == runner.expected_input_ids.tolist()
    assert step.input_batch.positions.tolist() == runner.expected_positions.tolist()
    torch.testing.assert_close(
        step.hidden_states, expected.hidden_states, rtol=0, atol=0
    )
    for actual, target in zip(
        step.aux_hidden_states, expected.aux_hidden_states, strict=True
    ):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    torch.testing.assert_close(
        owner.model.get_mtp_target_hidden_states()[:width],
        expected.mtp_hidden_states,
        rtol=0,
        atol=0,
    )
    for actual, target in zip(destination.pages, expected.pages, strict=True):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    for actual, target in zip(
        destination.source_pages, runner.source_pages_before, strict=True
    ):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    index = owner.req_states.req_id_to_index[transfer.request.request_id]
    assert owner.req_states.total_len.gpu[index].item() == len(
        new_request.prompt_token_ids
    )
    assert owner.req_states.num_computed_tokens.gpu[index].item() == computed
    assert owner.model is transfer.loaded_destination.target_model
    assert owner.speculator is transfer.loaded_destination.speculator


@pytest.mark.parametrize("scheduled_pd_zero_case", [1, 6], indirect=True)
def test_native_pd_target_failure_retains_storage_and_rejects_retry(
    native_pd_target_case, monkeypatch
):
    transfer, runner, resumed = native_pd_target_case
    initialize_native_pd_sampling(monkeypatch, runner)
    owner, destination = runner.owner, transfer.destination_cache
    expected = native_zeroed_cache_reference(
        layout=destination.layout,
        config=destination.config,
        caches=destination.caches,
        pages=destination.pages,
        block_ids=resumed.new_block_ids_to_zero,
    )
    protocol = tuple(transfer.sent)
    cache_addresses = [page.data_ptr() for page in destination.pages]
    target_error = RuntimeError("controlled native PD target entry failure")

    def fail_target_entry(model, args):
        # The native eager execute has admitted the request, prepared metadata
        # and bound the connector. Fail at its target call, before KV writes.
        raise target_error

    hook = owner.model.register_forward_pre_hook(fail_target_entry)
    try:
        with pytest.raises(RuntimeError, match="PD target entry failure") as raised:
            owner.execute_model(resumed)
        assert raised.value is target_error
    finally:
        hook.remove()
    assert owner.dspark_pd_consumer.terminal
    assert owner.execute_model_state is None
    assert owner.sample_tokens(None) is None
    # Removing the injected fault must not reopen a failed registered owner.
    with pytest.raises(RuntimeError, match="DSpark prefix consumer is terminal"):
        owner.execute_model(resumed)
    assert tuple(transfer.sent) == protocol
    assert owner.model is transfer.loaded_destination.target_model
    assert owner.speculator is transfer.loaded_destination.speculator
    assert owner.dspark_pd_consumer.binding.target_model is owner.model
    assert owner.dspark_pd_consumer.binding.draft_model is owner.speculator.model
    assert [page.data_ptr() for page in destination.pages] == cache_addresses
    for actual, target in zip(destination.pages, expected, strict=True):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    for actual, target in zip(
        destination.source_pages, runner.source_pages_before, strict=True
    ):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)


@pytest.mark.parametrize("scheduled_pd_zero_case", [1, 6], indirect=True)
@pytest.mark.parametrize("failure_stage", ["target-head", "draft-backbone"])
def test_pd_sampling_failure_retains_storage_and_rejects_retry(  # long-test: PD owner
    native_pd_target_case, monkeypatch, failure_stage
):
    transfer, runner, resumed = native_pd_target_case
    initialize_native_pd_sampling(monkeypatch, runner)
    owner, destination = runner.owner, transfer.destination_cache
    assert owner.execute_model(resumed) is None
    protocol = tuple(transfer.sent)
    cache_addresses = [page.data_ptr() for page in destination.pages]
    sampling_error = RuntimeError(f"controlled PD {failure_stage} failure")

    def fail_sampling(*args, **kwargs):
        raise sampling_error

    if failure_stage == "target-head":
        monkeypatch.setattr(owner.model, "compute_logits", fail_sampling)
        hook = None
    else:
        # The native compile decorator bypasses the wrapper's Module hooks;
        # fail at its actual draft embedding call after context KV projection.
        hook = owner.speculator.model.model.embed_tokens.register_forward_pre_hook(
            fail_sampling
        )
    try:
        with pytest.raises(RuntimeError, match=failure_stage) as raised:
            owner.sample_tokens(None)
        assert raised.value is sampling_error
    finally:
        if hook is not None:
            hook.remove()

    assert owner.dspark_pd_consumer.terminal
    assert owner.execute_model_state is None
    assert owner.sample_tokens(None) is None
    state = owner.req_states
    index = state.req_id_to_index[transfer.request.request_id]
    prompt = resumed.scheduled_new_reqs[0].prompt_token_ids
    # The draft failure follows native postprocess; preserve that advanced
    # frontier rather than pretending that sampling/history were rolled back.
    advanced = int(failure_stage == "draft-backbone")
    assert state.total_len.gpu[index].item() == len(prompt) + advanced
    assert state.all_token_ids.gpu[index, : len(prompt) + advanced].tolist() == (
        prompt + [PD_CPU_TARGET_TOKEN] * advanced
    )
    pages_after_failure = [page.clone() for page in destination.pages]
    with pytest.raises(RuntimeError, match="DSpark prefix consumer is terminal"):
        owner.execute_model(resumed)
    assert tuple(transfer.sent) == protocol
    assert [page.data_ptr() for page in destination.pages] == cache_addresses
    for actual, expected in zip(destination.pages, pages_after_failure, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(
        destination.source_pages, runner.source_pages_before, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert owner.model is transfer.loaded_destination.target_model
    assert owner.speculator is transfer.loaded_destination.speculator


@pytest.mark.parametrize("scheduled_pd_zero_case", [1, 6], indirect=True)
def test_native_pd_terminal_owner_rejects_pending_sample(
    native_pd_target_case, monkeypatch
):
    transfer, runner, resumed = native_pd_target_case
    initialize_native_pd_sampling(monkeypatch, runner)
    owner = runner.owner
    assert owner.execute_model(resumed) is None
    pending = owner.execute_model_state
    protocol = tuple(transfer.sent)
    owner.dspark_pd_consumer.fail()

    with pytest.raises(RuntimeError, match="DSpark prefix consumer is terminal"):
        owner.sample_tokens(None)

    assert owner.execute_model_state is pending
    assert tuple(transfer.sent) == protocol


@pytest.mark.parametrize("scheduled_pd_zero_case", [1, 6], indirect=True)
def test_native_pd_first_sample_keeps_prompt_and_drafts(  # long-test: first D sample
    native_pd_target_case, monkeypatch
):
    transfer, runner, resumed = native_pd_target_case
    initialize_native_pd_sampling(monkeypatch, runner)
    owner, destination = runner.owner, transfer.destination_cache
    prompt = resumed.scheduled_new_reqs[0].prompt_token_ids
    assert owner.execute_model(resumed) is None
    batch = owner.execute_model_state.input_batch
    output = owner.sample_tokens(None).get_output()
    index = owner.req_states.req_id_to_index[transfer.request.request_id]
    assert output.req_ids == [transfer.request.request_id]
    assert output.sampled_token_ids == [[PD_CPU_TARGET_TOKEN]]
    assert owner.req_states.total_len.gpu[index].item() == len(prompt) + 1
    assert owner.req_states.num_computed_tokens.gpu[index].item() == len(prompt)
    assert owner.req_states.all_token_ids.gpu[index, : len(prompt) + 1].tolist() == (
        prompt + [PD_CPU_TARGET_TOKEN]
    )
    assert owner.req_states.draft_tokens[index].tolist() == list(
        range(
            PD_CPU_TARGET_TOKEN + 1,
            PD_CPU_TARGET_TOKEN + 1 + owner.num_speculative_steps,
        )
    )
    counts = owner.sampler.penalties_state.output_bin_counts[index]
    assert counts[PD_CPU_TARGET_TOKEN].item() == counts.sum().item() == 1
    combined = sum(
        (index + 1) * states.float()
        for index, states in enumerate(runner.expected_target.aux_hidden_states)
    ).to(torch.bfloat16)
    # Only one physical target row survives the actual greedy rejection.
    # The new draft context keeps that accepted prompt-tail projection.
    position = len(prompt) - 1
    for group_id in owner.speculator.draft_kv_cache_group_ids:
        group = destination.config.kv_cache_groups[group_id]
        block_ids = resumed.scheduled_new_reqs[0].block_ids[group_id]
        block, offset = divmod(position, group.kv_cache_spec.block_size)
        for name in group.layer_names:
            torch.testing.assert_close(
                destination.caches[name][block_ids[block], offset, 0],
                combined[0, :2],
                rtol=0,
                atol=0,
            )
    draft_names = owner.speculator.draft_attn_layer_names
    for page, actual, expected in zip(
        destination.layout.pages,
        destination.pages,
        runner.expected_target.pages,
        strict=True,
    ):
        if not set(page.layers) & draft_names:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(
        destination.source_pages, runner.source_pages_before, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert owner.execute_model_state is None
    assert batch.num_draft_tokens == resumed.total_num_scheduled_tokens - 1


@pytest.mark.parametrize("scheduled_pd_zero_case", [1, 6], indirect=True)
def test_native_pd_second_sample_keeps_rejection_frontier(  # long-test: two PD rounds
    native_pd_target_case, monkeypatch
):
    transfer, runner, resumed = native_pd_target_case
    initialize_native_pd_sampling(monkeypatch, runner)
    owner, destination = runner.owner, transfer.destination_cache
    request_id = transfer.request.request_id
    prompt = resumed.scheduled_new_reqs[0].prompt_token_ids
    assert owner.execute_model(resumed) is None
    first_output = owner.sample_tokens(None).get_output()
    assert first_output.sampled_token_ids == [[PD_CPU_TARGET_TOKEN]]

    # Match synchronous EngineCore: consume output, take native no-grammar
    # draft placeholders, then schedule. Real draft values stay on the owner.
    transfer.scheduler.update_from_output(resumed, first_output)
    drafts = owner.take_draft_token_ids()
    assert drafts.req_ids == [request_id]
    assert drafts.draft_token_ids == [[-1] * owner.num_speculative_steps]
    transfer.scheduler.update_draft_token_ids(drafts)
    scheduled = transfer.scheduler.schedule()
    assert scheduled.scheduled_new_reqs == []
    cached = scheduled.scheduled_cached_reqs
    assert cached.req_ids == [request_id]
    assert cached.resumed_req_ids == set()
    assert cached.num_computed_tokens == [len(prompt)]
    assert cached.num_output_tokens == [1]
    assert scheduled.scheduled_spec_decode_tokens == {
        request_id: [-1] * owner.num_speculative_steps
    }
    width = owner.num_speculative_steps + 1
    assert scheduled.num_scheduled_tokens == {request_id: width}
    # CachedRequestData contains only appended blocks. The independent oracle
    # needs the scheduler's entire absolute map, including retired null slots.
    block_ids = transfer.scheduler.kv_cache_manager.get_block_ids(request_id)
    expected_ids = torch.arange(PD_CPU_TARGET_TOKEN, PD_CPU_TARGET_TOKEN + width)
    expected = native_pd_target_step_reference(
        runner,
        scheduled,
        input_ids=expected_ids,
        computed=len(prompt),
        block_ids=block_ids,
    )
    assert owner.execute_model(scheduled) is None
    step = owner.execute_model_state
    assert step.input_batch.input_ids.tolist() == expected_ids.tolist()
    assert step.input_batch.positions.tolist() == list(
        range(len(prompt), len(prompt) + width)
    )
    torch.testing.assert_close(
        step.hidden_states, expected.hidden_states, rtol=0, atol=0
    )
    for actual, target in zip(
        step.aux_hidden_states, expected.aux_hidden_states, strict=True
    ):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    torch.testing.assert_close(
        owner.model.get_mtp_target_hidden_states()[:width],
        expected.mtp_hidden_states,
        rtol=0,
        atol=0,
    )
    for actual, target in zip(destination.pages, expected.pages, strict=True):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    index = owner.req_states.req_id_to_index[request_id]
    assert owner.req_states.total_len.gpu[index].item() == len(prompt) + 1
    assert owner.req_states.num_computed_tokens.gpu[index].item() == len(prompt)

    # The target head predicts 7 while the first real draft is 8: actual greedy
    # rejection retains only the anchor row and advances one accepted frontier.
    output = owner.sample_tokens(None).get_output()
    assert output.req_ids == [request_id]
    assert output.sampled_token_ids == [[PD_CPU_TARGET_TOKEN]]
    assert owner.req_states.total_len.gpu[index].item() == len(prompt) + 2
    assert owner.req_states.num_computed_tokens.gpu[index].item() == len(prompt) + 1
    assert owner.req_states.all_token_ids.gpu[index, : len(prompt) + 2].tolist() == (
        prompt + [PD_CPU_TARGET_TOKEN, PD_CPU_TARGET_TOKEN]
    )
    assert owner.req_states.draft_tokens[index].tolist() == expected_ids[1:].tolist()
    counts = owner.sampler.penalties_state.output_bin_counts[index]
    assert counts[PD_CPU_TARGET_TOKEN].item() == counts.sum().item() == 2
    combined = sum(
        (ordinal + 1) * states.float()
        for ordinal, states in enumerate(expected.aux_hidden_states)
    ).to(torch.bfloat16)
    for group_id in owner.speculator.draft_kv_cache_group_ids:
        group = destination.config.kv_cache_groups[group_id]
        block, offset = divmod(len(prompt), group.kv_cache_spec.block_size)
        for name in group.layer_names:
            torch.testing.assert_close(
                destination.caches[name][block_ids[group_id][block], offset, 0],
                combined[0, :2],
                rtol=0,
                atol=0,
            )
    for page, actual, target in zip(
        destination.layout.pages, destination.pages, expected.pages, strict=True
    ):
        if not set(page.layers) & owner.speculator.draft_attn_layer_names:
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
    for actual, target in zip(
        destination.source_pages, runner.source_pages_before, strict=True
    ):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    transfer.scheduler.update_from_output(scheduled, output)
    assert list(transfer.request.output_token_ids) == [PD_CPU_TARGET_TOKEN] * 2
    assert transfer.request.num_computed_tokens == len(prompt) + 1
    assert owner.execute_model_state is None
    assert owner.model is transfer.loaded_destination.target_model
    assert owner.speculator is transfer.loaded_destination.speculator


@pytest.mark.parametrize("native_pd_u2_case", [1, 6], indirect=True)
def test_native_pd_u2_keeps_handoff_and_rejection_frontier(  # long-test: U2 D rounds
    native_pd_u2_case, monkeypatch
):
    transfer, runner, scheduled = native_pd_u2_case
    owner, destination = runner.owner, transfer.destination_cache
    request_id = transfer.request.request_id
    prompt = list(transfer.request.prompt_token_ids)
    protocol = tuple(transfer.sent)
    assert owner.parallel_config.use_ubatching
    assert owner.parallel_config.num_ubatches == 2
    assert owner.vllm_config.kv_transfer_config.kv_role == "kv_consumer"
    assert owner.num_speculative_steps == 2
    history = list(prompt)
    for round_index, sampled in enumerate(([7], [8, 7])):
        computed = len(prompt) - 1 + round_index
        # Native initial prefill has no speculative rows for original prompt1.
        # Keep that U1 fallback; its next cached decode is the real q3 parent.
        width = 1 if len(prompt) == 1 and not round_index else 3
        stages = 1 if width == 1 else 2
        expected_ids = torch.tensor(
            [prompt[-1]] + [0] * (width - 1) if not round_index else [7, 8, 9]
        )
        block_ids = transfer.scheduler.kv_cache_manager.get_block_ids(request_id)
        expected = native_pd_target_step_reference(
            runner,
            scheduled,
            input_ids=expected_ids,
            computed=computed,
            block_ids=block_ids,
            stages=stages,
        )
        if round_index:
            # A declared CPU head supplies actual logits. Native rejection must
            # accept draft 8, reject draft 9, then preserve only two target rows.
            monkeypatch.setattr(
                owner.model.language_model,
                "compute_logits",
                SpeculativeLogitsBoundary(
                    4, 128, ordinal_rows=True, predictions=(8, 7, 7)
                ),
            )
        assert scheduled.num_scheduled_tokens == {request_id: width}
        assert owner.execute_model(scheduled) is None
        step = owner.execute_model_state
        assert step.input_batch.input_ids.tolist() == expected_ids.tolist()
        assert step.input_batch.positions.tolist() == list(
            range(computed, computed + width)
        )
        torch.testing.assert_close(
            step.hidden_states, expected.hidden_states, rtol=0, atol=0
        )
        for actual, target in zip(
            step.aux_hidden_states, expected.aux_hidden_states, strict=True
        ):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
        torch.testing.assert_close(
            owner.model.get_mtp_target_hidden_states()[:width],
            expected.mtp_hidden_states,
            rtol=0,
            atol=0,
        )
        for actual, target in zip(destination.pages, expected.pages, strict=True):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
        group_id = next(
            index
            for index, group in enumerate(destination.config.kv_cache_groups)
            if "model.layers.0.self_attn.swa_cache" in group.layer_names
        )
        block_size = destination.config.kv_cache_groups[
            group_id
        ].kv_cache_spec.block_size
        segments = ((0, width),) if stages == 1 else ((0, 1), (1, width))
        for child, (begin, end) in zip(
            runner.target_children[-stages:], segments, strict=True
        ):
            # Expected coordinates derive from the original scheduler query,
            # not from the native prepared child used by the CPU operator.
            positions = list(range(computed + begin, computed + end))
            if stages == 1:
                assert child.descriptor_tokens == width
            else:
                assert child.stage == int(begin != 0)
            assert child.positions.tolist() == positions
            assert child.query_start_loc.tolist() == [0, end - begin]
            assert child.seq_lens.tolist() == [computed + end]
            assert child.slot_mapping.tolist() == [
                block_ids[group_id][position // block_size] * block_size
                + position % block_size
                for position in positions
            ]
        plans = [
            message.plan
            for direction, _, _, message in runner.transport.transcript
            if direction == "send" and message.kind == "FORWARD"
        ]
        assert len(plans) == round_index + 1
        assert plans[-1].mode == "target" and plans[-1].layer_indices == (0, 1, 2)
        expected_counts = [(end - begin,) for begin, end in segments]
        assert [
            stage.attention_valid_tokens for stage in plans[-1].stages
        ] == expected_counts
        assert [
            stage.attention_padded_tokens for stage in plans[-1].stages
        ] == expected_counts
        assert [
            message.transaction_id
            for direction, _, _, message in runner.transport.transcript
            if direction == "recv" and message.kind == "COMPLETE"
        ] == [plan.transaction_id for plan in plans]

        output = owner.sample_tokens(None).get_output()
        assert output.req_ids == [request_id] and output.sampled_token_ids == [sampled]
        history.extend(sampled)
        index = owner.req_states.req_id_to_index[request_id]
        assert owner.req_states.total_len.gpu[index].item() == len(history)
        assert owner.req_states.num_computed_tokens.gpu[index].item() == computed + len(
            sampled
        )
        assert (
            owner.req_states.all_token_ids.gpu[index, : len(history)].tolist()
            == history
        )
        assert owner.req_states.draft_tokens[index].tolist() == [8, 9]
        combined = sum(
            (ordinal + 1) * states.float()
            for ordinal, states in enumerate(expected.aux_hidden_states)
        ).to(torch.bfloat16)
        for draft_group_id in owner.speculator.draft_kv_cache_group_ids:
            group = destination.config.kv_cache_groups[draft_group_id]
            for row in range(len(sampled)):
                block, offset = divmod(computed + row, group.kv_cache_spec.block_size)
                for name in group.layer_names:
                    torch.testing.assert_close(
                        destination.caches[name][
                            block_ids[draft_group_id][block], offset, 0
                        ],
                        combined[row, :2],
                        rtol=0,
                        atol=0,
                    )
        for page, actual, target in zip(
            destination.layout.pages, destination.pages, expected.pages, strict=True
        ):
            if not set(page.layers) & owner.speculator.draft_attn_layer_names:
                torch.testing.assert_close(actual, target, rtol=0, atol=0)
        for actual, target in zip(
            destination.source_pages, runner.source_pages_before, strict=True
        ):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
        assert tuple(transfer.sent) == protocol
        assert (
            owner.execute_model_state is None and not owner.dspark_pd_consumer.terminal
        )
        transfer.scheduler.update_from_output(scheduled, output)
        assert list(transfer.request.output_token_ids) == history[len(prompt) :]
        if not round_index:
            drafts = owner.take_draft_token_ids()
            assert drafts.draft_token_ids == [[-1, -1]]
            transfer.scheduler.update_draft_token_ids(drafts)
            scheduled = transfer.scheduler.schedule()
    counts = owner.sampler.penalties_state.output_bin_counts[index]
    assert counts[7].item() == 2 and counts[8].item() == 1
    assert counts.sum().item() == 3
    assert transfer.request.num_computed_tokens == len(prompt) + 2
    assert owner.model is transfer.loaded_destination.target_model
    assert owner.speculator is transfer.loaded_destination.speculator


@pytest.mark.parametrize("native_pd_u2_case", [6], indirect=True)
def test_native_pd_u2_proposal_failure_retains_owner(native_pd_u2_case):
    transfer, runner, scheduled = native_pd_u2_case
    owner, destination = runner.owner, transfer.destination_cache
    assert owner.execute_model(scheduled) is None
    protocol = tuple(transfer.sent)
    hccl_protocol = tuple(runner.transport.transcript)
    cache_addresses = [page.data_ptr() for page in destination.pages]
    failure = RuntimeError("controlled D U2 native draft Attention failure")

    def fail_draft(layer, inputs):
        raise failure

    draft_layer = next(iter(owner.speculator.model.model.layers.values()))
    hook = draft_layer.self_attn.register_forward_pre_hook(fail_draft)
    try:
        with pytest.raises(RuntimeError, match="D U2 native draft Attention") as raised:
            owner.sample_tokens(None)
        assert raised.value is failure
    finally:
        hook.remove()
    assert owner.dspark_pd_consumer.terminal
    assert owner.execute_model_state is None and owner.sample_tokens(None) is None
    retained = tuple(page.clone() for page in destination.pages)
    with pytest.raises(RuntimeError, match="DSpark prefix consumer is terminal"):
        owner.execute_model(scheduled)
    assert tuple(transfer.sent) == protocol
    assert tuple(runner.transport.transcript) == hccl_protocol
    assert [page.data_ptr() for page in destination.pages] == cache_addresses
    for actual, expected in zip(destination.pages, retained, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(
        destination.source_pages, runner.source_pages_before, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert owner.model is transfer.loaded_destination.target_model
    assert owner.speculator is transfer.loaded_destination.speculator


@pytest.mark.parametrize("scheduled_pd_zero_case", [1, 6], indirect=True)
def test_native_pd_completion_prepares_prompt_tail(  # long-test: P/D handoff
    scheduled_pd_zero_case, monkeypatch
):
    transfer, runner = scheduled_pd_zero_case
    prefix, destination = transfer.prefix_case, transfer.destination_cache
    source_before = tuple(page.clone() for page in destination.source_pages)
    first = transfer.scheduler.schedule()
    assert first.total_num_scheduled_tokens == 0
    assert transfer.request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    allocated = first.kv_connector_metadata.dspark_destinations[
        transfer.request.request_id
    ]
    # Derive the required absolute prefix coordinates from the cache contract,
    # independently of receiver addresses or the destination's block count.
    # P may publish a conservative SWA superset; D needs only the retained
    # context preceding its next token. A circular group retains one ring.
    retained_source_blocks = []
    computed = prefix.source.computed_prefix_tokens
    for group_index, group in enumerate(prefix.kv_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[group.layer_names[0]]
        start = 0
        end = (computed + spec.block_size - 1) // spec.block_size
        if is_circular_kv_cache_spec(spec):
            end = 1
        elif isinstance(spec, SlidingWindowSpec):
            skipped = max(
                0, computed - spec.sliding_window + 1 - spec.extra_retained_tokens
            )
            start = skipped // spec.block_size
        required = prefix.source.allocation_block_ids[group_index][start:end]
        assert len(required) == len(allocated.block_ids[group_index]) == end - start
        published = prefix.source.transfer_block_ids[group_index]
        assert tuple(block for block in published if block in required) == required
        retained_source_blocks.append(required)
    expected = [
        torch.full_like(page, DESTINATION_SENTINEL) for page in destination.pages
    ]
    for page_index, page in enumerate(prefix.layout.pages):
        stride = page.block_stride_bytes
        for group_index, group in enumerate(prefix.layout.groups):
            if not set(page.layers) & set(group.layer_names):
                continue
            for source_id, destination_id in zip(
                retained_source_blocks[group_index],
                allocated.block_ids[group_index],
                strict=True,
            ):
                expected[page_index][
                    destination_id * stride : (destination_id + 1) * stride
                ] = source_before[page_index][
                    source_id * stride : (source_id + 1) * stride
                ]
    first_output = runner.owner.execute_model(first)
    assert first_output.req_ids == []
    assert first_output.kv_connector_output is None
    transfer.scheduler.update_from_output(first, first_output)
    for page in destination.pages:
        assert bool(torch.all(page == DESTINATION_SENTINEL))

    transfer.process_next_queued_request()
    for actual, target in zip(destination.pages, expected, strict=True):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    for actual, unchanged in zip(destination.source_pages, source_before, strict=True):
        torch.testing.assert_close(actual, unchanged, rtol=0, atol=0)
    pending = transfer.scheduler.schedule()
    assert pending.total_num_scheduled_tokens == 0
    assert transfer.request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    completed = runner.owner.execute_model(pending)
    assert completed.req_ids == []
    assert completed.kv_connector_output.finished_recving == {
        transfer.request.request_id
    }
    assert not completed.kv_connector_output.failed_recving
    transfer.scheduler.update_from_output(pending, completed)
    resumed = transfer.scheduler.schedule()
    new_request = resumed.scheduled_new_reqs[0]
    assert new_request.prompt_token_ids == list(
        range(prefix.source.original_prompt_tokens)
    )
    assert new_request.num_computed_tokens == prefix.source.original_prompt_tokens - 1
    query_width = (
        1
        if prefix.source.original_prompt_tokens == 1
        else 1 + transfer.config.speculative_config.num_speculative_tokens
    )
    assert resumed.num_scheduled_tokens == {transfer.request.request_id: query_width}
    zero_ids = resumed.new_block_ids_to_zero
    transferred_ids = {block for group in allocated.block_ids for block in group}
    assert zero_ids and NULL_BLOCK_ID not in zero_ids
    assert not set(zero_ids) & transferred_ids
    expected_after_admission = native_zeroed_cache_reference(
        layout=destination.layout,
        config=destination.config,
        caches=destination.caches,
        pages=expected,
        block_ids=zero_ids,
    )
    admission = make_native_request_admission_case(
        monkeypatch, runner.owner, kv_cache_views=runner.kv_views
    )
    batch, uniform = admission.admit(resumed)
    state = runner.owner.req_states
    index = state.req_id_to_index[transfer.request.request_id]
    prompt = new_request.prompt_token_ids
    assert state.all_token_ids.gpu[index, : len(prompt)].tolist() == prompt
    assert state.prompt_len.np[index] == state.prefill_len.np[index] == len(prompt)
    assert state.total_len.gpu[index].item() == len(prompt)
    assert (
        state.num_computed_tokens.gpu[index].item() == new_request.num_computed_tokens
    )
    assert state.num_computed_tokens_np[index] == new_request.num_computed_tokens
    assert (
        state.draft_tokens[index].tolist()
        == [0] * transfer.config.speculative_config.num_speculative_tokens
    )
    for table, block_ids in zip(
        runner.owner.block_tables.block_tables, new_request.block_ids, strict=True
    ):
        assert table.gpu[index, : len(block_ids)].tolist() == block_ids
    assert batch.req_ids == [transfer.request.request_id]
    assert batch.num_tokens == query_width
    assert batch.num_scheduled_tokens.tolist() == [query_width]
    assert batch.num_computed_prefill_tokens_np.tolist() == [
        new_request.num_computed_tokens
    ]
    assert batch.is_prefilling_np.tolist() == [True]
    assert batch.has_prefill and uniform is None
    for actual, target in zip(destination.pages, expected_after_admission, strict=True):
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
    install_native_input_preparation_cpu(monkeypatch, runner.owner)
    install_native_block_table_kernels_cpu(monkeypatch, runner.owner.block_tables)
    prepared = runner.owner.prepare_inputs(
        resumed,
        batch,
        BatchExecutionDescriptor(CUDAGraphMode.NONE, query_width, 1),
    )
    computed = new_request.num_computed_tokens
    assert prepared.input_ids.tolist() == [prompt[-1]] + [0] * (query_width - 1)
    assert prepared.positions.tolist() == list(range(computed, computed + query_width))
    assert prepared.query_start_loc.tolist() == [0, query_width]
    assert prepared.seq_lens.tolist() == [computed + query_width]
    assert prepared.seq_lens_cpu_upper_bound.tolist() == [computed + query_width]
    assert prepared.cu_num_logits.tolist() == [0, query_width]
    assert prepared.logits_indices.tolist() == list(range(query_width))
    assert prepared.num_draft_tokens == query_width - 1
    assert prepared.expanded_idx_mapping.tolist() == [index] * query_width
    assert prepared.expanded_local_pos.tolist() == list(range(query_width))
    assert prepared.attn_state == (
        AscendAttentionState.PrefillNoCache
        if len(prompt) == 1
        else AscendAttentionState.ChunkedPrefill
    )
    gathered = runner.owner.block_tables.gather_block_tables(prepared.idx_mapping, 1)
    slots = runner.owner.block_tables.compute_slot_mappings(
        prepared.idx_mapping,
        prepared.query_start_loc,
        prepared.positions,
        prepared.num_tokens,
    )
    for group_index, (group, block_ids) in enumerate(
        zip(prefix.kv_config.kv_cache_groups, new_request.block_ids, strict=True)
    ):
        assert gathered[group_index][0, : len(block_ids)].tolist() == block_ids
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[group.layer_names[0]]
        expected_slots = (
            [PAD_SLOT_ID] * query_width
            if is_circular_kv_cache_spec(spec)
            else [
                block_ids[position // spec.block_size] * spec.block_size
                + position % spec.block_size
                for position in range(computed, computed + query_width)
            ]
        )
        assert slots[group_index].tolist() == expected_slots
    assert [msgspec.msgpack.decode(frame)[0] for frame in transfer.sent] == [
        pd_adapter.GET_META_MSG,
        pd_adapter.GET_DSPARK_READY_MSG,
        pd_adapter.DONE_DSPARK_RECV_MSG,
    ]
    assert (
        transfer.producer.worker.kv_send_thread.get_and_clear_finished_requests()
        == {prefix.source.request_id}
    )


def test_dspark_pd_runner_publishes_only_the_complete_registered_prefix(monkeypatch):
    case = make_dspark_pd_runner_case(monkeypatch)
    prefix, registration = case.prefix, case.registration
    query = DsparkPdReadQuery(
        prefix.expected_ready,
        "destination-engine",
        "destination-generation",
        prefix.source.transfer_block_ids,
        "runner-query",
    )
    assert case.owner.dspark_pd_consumer.binding.target_model is prefix.target_model
    assert prefix.kv_config.dspark_draft_layer_names == frozenset(prefix.draft_names)
    assert case.step(0, 2) is None
    assert registration.request_ready(query)["ok"] is False
    assert case.step(2, 3) is None
    reply = registration.request_ready(query)
    assert reply["ok"] is True
    lease = read_lease_from_payload(reply["lease"])
    assert lease.ready == prefix.expected_ready
    for group_id, name in zip(prefix.draft_group_ids, prefix.draft_names, strict=True):
        size = prefix.kv_config.kv_cache_groups[group_id].kv_cache_spec.block_size
        actual = torch.stack(
            [
                prefix.caches[name][
                    prefix.source.allocation_block_ids[group_id][position // size],
                    position % size,
                    0,
                ]
                for position in range(prefix.source.computed_prefix_tokens)
            ]
        )
        expected = torch.tensor(
            [[240 + 60 * pos, 246 + 60 * pos] for pos in range(5)],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert registration.complete_read(lease) == b"ACK"
    assert case.transport.transcript == []


@pytest.mark.parametrize("failure", ["target", "projection", "metadata"])
def test_dspark_pd_runner_failure_retains_storage_without_ready(monkeypatch, failure):
    case = make_dspark_pd_runner_case(monkeypatch, failure=failure)
    prefix = case.prefix
    query = DsparkPdReadQuery(
        prefix.expected_ready,
        "destination-engine",
        "destination-generation",
        prefix.source.transfer_block_ids,
        "failed-runner-query",
    )
    with pytest.raises(RuntimeError):
        case.step(0, 5)
    assert case.owner.dspark_pd_consumer.terminal
    assert case.owner.dspark_pd_consumer.binding.draft_model is prefix.model
    assert case.registration.request_ready(query)["ok"] is False
    with pytest.raises(RuntimeError, match="terminal"):
        case.step(0, 5)
    assert case.transport.transcript == []


@pytest.mark.parametrize(
    "reinitialization,message", [("model", "single-use"), ("kv", "pre-registration")]
)
def test_dspark_pd_runner_cannot_replace_bound_models_or_kv(
    monkeypatch, reinitialization, message
):
    case = make_dspark_pd_runner_case(monkeypatch)
    with pytest.raises(RuntimeError, match=message):
        if reinitialization == "model":
            case.owner.load_model()
        else:
            case.owner.initialize_kv_cache(case.prefix.kv_config)
    assert case.owner.dspark_pd_consumer.binding.draft_model is case.prefix.model


@pytest.fixture
def runner_case(monkeypatch):
    monkeypatch.setattr(runner_module, "stop_afd_npu_profiler", lambda _: None)
    monkeypatch.setattr(
        coordinator_module, "_get_graph_update_backend", lambda _: AscendSFABackend
    )
    return make_graph_runner_case


@pytest.mark.parametrize("post_kv", [False, True])
def test_native_zero_return_uses_the_current_graph_lifecycle(
    runner_case, monkeypatch, post_kv
):
    case = runner_case()
    case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    if post_kv:
        case.owner.bind_hccl_v41_graph_coordinator(
            install_graph_runner_coordinator(case)
        )
    result = object()
    monkeypatch.setattr(NPUModelRunner, "execute_model", lambda *a, **k: result)

    assert (
        case.owner.execute_model(
            SimpleNamespace(total_num_scheduled_tokens=0),
            dummy_run=True,
            is_profile=True,
        )
        is result
    )

    plans = [
        message.plan
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == "FORWARD"
    ]
    assert len(plans) == 1
    assert plans[0].mode == "profile"
    assert plans[0].graph_execution.phase == "eager"
    assert plans[0].stages[0].attention_padded_tokens == (0,)
    assert case.transport.plan is None and case.transport.graph_reservation is None


@pytest.mark.parametrize("entry", ["execute", "capture", "kv"])
def test_unbound_graph_lifecycle_fails_before_native_work(runner_case, entry):
    case = runner_case()
    with pytest.raises(RuntimeError, match="bind.*graph"):
        if entry == "execute":
            case.owner.execute_model(SimpleNamespace(total_num_scheduled_tokens=4))
        elif entry == "capture":
            case.owner.capture_model()
        else:
            case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    assert case.transport.transcript == []


def test_pre_kv_binding_rejects_another_runner_before_protocol_work(runner_case):
    case, foreign = runner_case(), runner_case()
    with pytest.raises(RuntimeError, match="idle pre-KV owners"):
        case.owner.bind_hccl_v41_graph_eager(foreign.eager, case.session)
    assert case.transport.transcript == []


def test_graph_binding_cannot_replace_an_already_bound_lifecycle(runner_case):
    case = runner_case()
    case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    with pytest.raises(RuntimeError, match="idle pre-KV owners"):
        case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    assert case.transport.transcript == []


def test_real_kv_coordinator_rejects_a_different_pre_kv_session(runner_case):
    case, foreign = runner_case(), runner_case()
    case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    coordinator = install_graph_runner_coordinator(case)
    coordinator.session = foreign.session
    with pytest.raises(RuntimeError, match="pre-KV session owners"):
        case.owner.bind_hccl_v41_graph_coordinator(coordinator)
    assert case.transport.transcript == []


@pytest.mark.parametrize("failure", [None, "cache", "native"])
def test_shutdown_retains_native_owners_until_graph_reset_succeeds(
    runner_case, monkeypatch, failure
):
    case = runner_case(fail_close=failure == "cache")
    case.owner.bind_hccl_v41_graph_eager(case.eager, case.session)
    queue_graph_runner_stop(case)

    def native_shutdown(owner):
        case.events.append("native-close")
        if failure == "native":
            raise RuntimeError("native shutdown failed")

    monkeypatch.setattr(NPUModelRunner, "shutdown", native_shutdown)
    if failure:
        with pytest.raises(RuntimeError, match="synchronization|shutdown"):
            case.owner.shutdown()
        assert case.owner.hccl_v41_graph_eager is case.eager
        assert not case.transport.closed
    else:
        case.owner.shutdown()
        assert case.owner.hccl_v41_graph_eager is None
        assert case.cache.closed and case.transport.closed

    expected = ["graph-sync"]
    if failure != "cache":
        expected.append("native-close")
    if failure is None:
        expected.append("groups-close")
    assert case.events == expected
    assert any(
        direction == "recv" and message.kind == "STOP_ACK"
        for direction, _, _, message in case.transport.transcript
    )


@pytest.fixture
def automatic_graph_factory(runner_case, monkeypatch):
    set_graph_bootstrap_environment(monkeypatch)
    monkeypatch.setattr(runner_module.torch.npu, "synchronize", lambda *a: None)

    def create(**kwargs):
        case = make_graph_bootstrap_case(**kwargs)
        monkeypatch.setattr(
            NPUModelRunner,
            "load_model",
            lambda owner, *a, **k: case.native.load_model(owner, *a, **k),
        )
        monkeypatch.setattr(
            NPUModelRunner,
            "initialize_kv_cache",
            lambda owner, *a, **k: case.native.initialize_kv_cache(owner, *a, **k),
        )
        return case

    return create


@pytest.fixture
def automatic_graph_case(automatic_graph_factory):
    return automatic_graph_factory()


@pytest.fixture
def automatic_eager_factory(monkeypatch):
    def create(**kwargs):
        case = make_attention_eager_bootstrap_case(monkeypatch, runner_module, **kwargs)
        monkeypatch.setattr(
            NPUModelRunner,
            "load_model",
            lambda owner, *a, **k: case.native.load_model(owner, *a, **k),
        )
        monkeypatch.setattr(
            NPUModelRunner,
            "initialize_kv_cache",
            lambda owner, *a, **k: case.native.initialize_kv_cache(owner, *a, **k),
        )
        return case

    return create


@pytest.fixture
def packed_draft_load_case(monkeypatch):
    monkeypatch.setattr(packed_native, "qsmla", packed_64_head_attention_reference)
    monkeypatch.setattr(
        packed_native.MixedQuantPackedCacheOps,
        "qsmla",
        staticmethod(packed_64_head_attention_reference),
    )
    return make_legacy_attention_startup_case(
        monkeypatch, dspark=True, dspark_backend=packed_native.MixedQuantPackedCacheOps
    )


@pytest.mark.parametrize("tokens", [1, 3])
def test_loaded_dspark_draft_accepts_tp4_queries(packed_draft_load_case, tokens):
    case = packed_draft_load_case
    inputs = make_packed_attention_inputs(tokens=tokens, compressed=False)
    expected = headwise_attention_with_sink(*inputs.arguments(), **inputs.options())

    case.owner.load_model()
    output = case.draft.attention.packed_cache_ops.qsmla(
        *inputs.arguments(), **inputs.options()
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="N1 must be 64 .* got 16"):
        case.draft.unrelated.packed_cache_ops.qsmla(
            *inputs.arguments(), **inputs.options()
        )


@pytest.mark.parametrize("backend", [None, type])
def test_dspark_load_keeps_alternative_attention_backends(
    packed_draft_load_case, backend
):
    case = packed_draft_load_case
    case.draft.attention.packed_cache_ops = backend

    case.owner.load_model()

    assert case.draft.attention.packed_cache_ops is backend


@pytest.mark.parametrize("excluded", ["other-model", "other-method"])
def test_other_drafts_keep_native_head_rejection(packed_draft_load_case, excluded):
    case = packed_draft_load_case
    if excluded == "other-model":
        case.config.model_config.hf_config.model_type = "deepseek_v2"
        case.config.model_config.hf_config.architectures = ["DeepseekV2ForCausalLM"]
    else:
        case.config.speculative_config.method = "mtp"
    inputs = make_packed_attention_inputs(compressed=False)

    case.owner.load_model()

    with pytest.raises(ValueError, match="N1 must be 64 .* got 16"):
        case.draft.attention.packed_cache_ops.qsmla(
            *inputs.arguments(), **inputs.options()
        )


@pytest.mark.parametrize("graph", [False, True])
def test_attention_startup_logs_frozen_zero_settings_and_independent_graph_family(
    automatic_eager_factory, automatic_graph_factory, monkeypatch, serving_logs, graph
):
    case = (
        automatic_graph_factory() if graph else automatic_eager_factory(enabled=False)
    )
    values = {name: "0" for name in HcclOverlapSettings().effective_settings()}
    case.owner.connector.overlap_settings = HcclOverlapSettings.from_environment(
        values, num_stages=2
    )
    for name in values:
        monkeypatch.setenv(name, "1")
    case.owner.load_model()
    records = structured_log_records(serving_logs, SESSION_CONFIGURATION_PREFIX)
    assert len(records) == 1
    record = records[0]
    assert record["event"] == "session_configuration" and record["phase"] == "startup"
    assert record["role"] == "attention" and record["snapshot_present"] is True
    assert record["negotiated.graph_execution_version"] == int(graph)
    assert record["negotiated.eager_pipeline_version"] == 0
    assert {name: record[f"configured.{name}"] for name in values} == {
        name: False for name in values
    }


@pytest.mark.parametrize("enabled", [False, True])
def test_legacy_attention_freezes_settings_before_load_and_logs_one_session(
    monkeypatch, serving_logs, enabled
):
    names = HcclOverlapSettings().effective_settings()
    for name in names:
        monkeypatch.setenv(name, str(int(enabled)))
    case = make_legacy_attention_startup_case(monkeypatch)
    for name in names:
        monkeypatch.setenv(name, "invalid-after-construction")

    case.owner.load_model()
    case.owner.load_model()

    records = structured_log_records(serving_logs, SESSION_CONFIGURATION_PREFIX)
    assert len(records) == 1
    record = records[0]
    assert record["protocol"] == "legacy_pair"
    assert record["snapshot_present"] is True and record["role"] == "attention"
    assert record["world_rank"] == 1 and record["pair_index"] == 0
    expected = {name: enabled for name in names}
    expected["AFD_HCCL_EAGER_U2_STREAM_OVERLAP"] = False
    assert {name: record[f"configured.{name}"] for name in names} == expected
    assert not any(
        name.startswith(("actual.", "effective.", "logical.", "physical."))
        for name in record
    )


@pytest.mark.parametrize("failure", ["load", "hello"])
def test_legacy_attention_startup_failure_emits_no_configuration(
    monkeypatch, serving_logs, failure
):
    case = make_legacy_attention_startup_case(monkeypatch, fail_load=failure == "load")
    if failure == "hello":

        def failed_hello():
            raise RuntimeError("controlled legacy HELLO failure")

        monkeypatch.setattr(case.owner.connector, "init_afd_connector", failed_hello)

    with pytest.raises(RuntimeError, match="native load|legacy HELLO"):
        case.owner.load_model()

    assert structured_log_records(serving_logs, SESSION_CONFIGURATION_PREFIX) == []


@pytest.mark.parametrize("other_model", [False, True])
def test_attention_other_connector_ignores_hccl_environment(
    monkeypatch, serving_logs, other_model
):
    for name in HcclOverlapSettings().effective_settings():
        monkeypatch.setenv(name, "invalid-unused-setting")

    case = make_legacy_attention_startup_case(
        monkeypatch, other_connector=True, other_model=other_model
    )
    case.owner.load_model()

    assert case.owner.connector.is_initialized
    assert structured_log_records(serving_logs, SESSION_CONFIGURATION_PREFIX) == []


def test_legacy_attention_rejects_invalid_diagnostic_settings_before_load(monkeypatch):
    monkeypatch.setenv("AFD_HCCL_STAGE_DIAGNOSTICS", "invalid")

    with pytest.raises(RuntimeError, match="AFD_HCCL_STAGE_DIAGNOSTICS must be 0 or 1"):
        make_legacy_attention_startup_case(monkeypatch)


def test_attention_startup_logs_missing_snapshot_without_environment_guesses(
    automatic_eager_factory, monkeypatch, serving_logs
):
    case = automatic_eager_factory(enabled=False)
    case.owner.connector.overlap_settings = None
    monkeypatch.setenv("AFD_HCCL_EAGER_U2_STREAM_OVERLAP", "1")
    case.owner.load_model()
    records = structured_log_records(serving_logs, SESSION_CONFIGURATION_PREFIX)
    assert len(records) == 1 and records[0]["snapshot_present"] is False
    assert records[0]["negotiated.eager_pipeline_version"] == 0
    assert not any(name.startswith("configured.") for name in records[0])


@pytest.mark.parametrize("failure", ["load", "warmup", "hello"])
def test_attention_failed_startup_emits_no_session_configuration(
    automatic_eager_factory, monkeypatch, serving_logs, failure
):
    case = automatic_eager_factory()
    if failure == "load":
        case.native.fail_load = True
    elif failure == "warmup":
        case.host.backend.fail_warmup = True
    else:
        case.transport.initialized = False

        def failed_hello():
            raise RuntimeError("controlled HELLO failure")

        monkeypatch.setattr(case.transport, "initialize", failed_hello)
    with pytest.raises(RuntimeError, match="native load|warmup|HELLO"):
        case.owner.load_model()
    assert structured_log_records(serving_logs, SESSION_CONFIGURATION_PREFIX) == []


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("diagnostics", [False, True])
def test_eager_loaded_family_is_bound_only_after_native_kv_and_ignores_env_changes(
    automatic_eager_factory, monkeypatch, enabled, diagnostics
):
    case = automatic_eager_factory(enabled=enabled, diagnostics=diagnostics)
    case.owner.load_model()
    prepared = case.owner.hccl_v41_eager_bootstrap.session
    binding = case.owner.hccl_v41_diagnostics
    assert (prepared is not None) == enabled
    assert (binding is not None) == diagnostics
    assert case.owner.ubatch_runner is None
    monkeypatch.setenv("AFD_HCCL_EAGER_U2_STREAM_OVERLAP", "invalid-after-construction")
    case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    assert case.owner.ubatch_runner.eager_session is prepared
    assert case.owner.ubatch_runner.diagnostic_binding is binding
    assert case.events == ["native-load", "native-kv"]
    assert case.transport.plan is None and not case.transport.terminal


@pytest.mark.parametrize("failure", ["load", "warmup", "kv"])
def test_eager_setup_failure_retains_resources_and_prevents_native_reload(
    automatic_eager_factory, failure
):
    case = automatic_eager_factory()
    if failure == "load":
        case.native.fail_load = True
    elif failure == "warmup":
        case.host.backend.fail_warmup = True
    else:
        case.owner.load_model()
        case.native.fail_kv = True
    with pytest.raises(RuntimeError, match="native load|warmup|native KV"):
        if failure == "kv":
            case.owner.initialize_kv_cache(case.owner.kv_cache_config)
        else:
            case.owner.load_model()
    if failure != "load":
        assert case.owner.hccl_v41_eager_bootstrap.session is not None
    assert case.transport.terminal and not case.transport.closed
    before = list(case.events)
    with pytest.raises(RuntimeError, match="single-use"):
        case.owner.load_model()
    assert case.events == before
    assert all(message.kind != "STOP" for _, _, _, message in case.transport.transcript)


def test_eager_second_kv_lifetime_fails_before_native_storage_changes(
    automatic_eager_factory,
):
    case = automatic_eager_factory()
    case.owner.load_model()
    case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    first = case.owner.ubatch_runner.eager_session
    with pytest.raises(RuntimeError, match="first prepared load"):
        case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    assert case.events == ["native-load", "native-kv"]
    assert case.owner.ubatch_runner.eager_session is first
    assert case.transport.terminal and not case.transport.closed


def test_eager_manual_snapshot_none_preserves_serial_family(
    automatic_eager_factory, monkeypatch
):
    case = automatic_eager_factory(enabled=False)
    case.owner.connector.overlap_settings = None
    monkeypatch.setenv("AFD_HCCL_EAGER_U2_STREAM_OVERLAP", "1")
    case.owner.load_model()
    case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    assert case.owner.ubatch_runner.eager_session is None
    assert case.host.runtime.allocated == []
    assert case.transport.negotiated_capabilities.eager_pipeline_version == 0


@pytest.mark.parametrize("fail_native", [False, True])
def test_eager_shutdown_releases_session_only_after_native_cleanup(
    automatic_eager_factory, monkeypatch, fail_native
):
    case = automatic_eager_factory()
    case.owner.load_model()
    case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    bootstrap = case.owner.hccl_v41_eager_bootstrap
    queue_graph_runner_stop(case)

    def native_shutdown(owner):
        if fail_native:
            raise RuntimeError("native cleanup failed")

    monkeypatch.setattr(NPUModelRunner, "shutdown", native_shutdown)
    if fail_native:
        with pytest.raises(RuntimeError, match="native cleanup failed"):
            case.owner.shutdown()
        assert case.owner.hccl_v41_eager_bootstrap is bootstrap
        assert case.owner.ubatch_runner.eager_session is bootstrap.session
        assert not case.transport.closed
    else:
        case.owner.shutdown()
        assert case.owner.hccl_v41_eager_bootstrap is None
        assert case.owner.ubatch_runner is None
        assert case.transport.closed


def test_normal_dspark_u2_load_and_kv_retain_target_query_and_graph_session(
    automatic_graph_factory,
):
    case = automatic_graph_factory(stages=2, target_query_len=3)
    case.owner.load_model()
    bridge = case.owner.hccl_v41_graph_eager
    case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    coordinator = case.owner.hccl_v41_graph_coordinator
    assert coordinator.control.cache is bridge.control.cache
    assert coordinator.session is case.owner.hccl_v41_graph_session
    assert coordinator.transport.negotiated_capabilities.target_query_len == 3
    assert coordinator.owner.decode_query_len == 3
    assert case.owner.ubatch_runner.graph_owner is coordinator.session
    assert coordinator.control.generation() == 0 and not case.transport.terminal


def test_normal_load_profiles_before_kv_then_installs_the_same_graph_owners(
    automatic_graph_case, monkeypatch
):
    case = automatic_graph_case
    case.owner.load_model()
    bridge = case.owner.hccl_v41_graph_eager
    monkeypatch.setattr(
        NPUModelRunner,
        "execute_model",
        lambda owner, *a, **k: run_graph_eager_parent(case, dummy=True, tokens=4)[1],
    )
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.input_batch.update_cos_sin", lambda _: None
    )
    output, auxiliary = case.owner.execute_model(
        SimpleNamespace(total_num_scheduled_tokens=4), dummy_run=True, is_profile=True
    )
    case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    coordinator = case.owner.hccl_v41_graph_coordinator
    assert output.shape == (4, 4) and len(auxiliary) == 2
    assert coordinator.control.cache is bridge.control.cache
    assert coordinator.control.generation is bridge.control.generation
    assert coordinator.control.generation() == 0
    assert coordinator.session is case.owner.hccl_v41_graph_session
    assert not coordinator.control.cache.terminal
    plans = [
        message.plan
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == "FORWARD"
    ]
    assert len(plans) == 1 and plans[0].mode == "profile"
    assert plans[0].graph_execution.phase == "eager"
    assert case.transport.plan is None and case.transport.graph_reservation is None


def test_automatic_graph_settings_stay_fixed_through_real_kv_handoff(
    automatic_graph_case, monkeypatch
):
    case = automatic_graph_case
    case.owner.load_model()
    settings = case.owner.hccl_v41_graph_eager.overlap_settings
    monkeypatch.setenv("AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP", "0")
    monkeypatch.setenv("AFD_HCCL_STAGE_DIAGNOSTICS", "1")
    case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    assert case.owner.hccl_v41_graph_coordinator.overlap_settings is settings
    assert settings.graph_u2_compute_overlap and not settings.stage_diagnostics


def test_automatic_kv_cannot_write_after_acknowledged_shutdown(
    automatic_graph_case,
):
    case = automatic_graph_case
    case.owner.load_model()
    queue_graph_runner_stop(case)
    case.owner.prepare_shutdown()
    with pytest.raises(RuntimeError, match="first idle load"):
        case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    assert case.events == ["native-load"]
    assert case.owner.hccl_v41_graph_eager.control.cache.closed
    assert not case.transport.closed


@pytest.mark.parametrize("entry", ["load", "kv"])
def test_automatic_graph_reinitialization_preserves_the_first_valid_storage(
    automatic_graph_case, entry
):
    case = automatic_graph_case
    case.owner.load_model()
    if entry == "kv":
        case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    first = case.owner.hccl_v41_graph_eager.control.cache
    with pytest.raises(RuntimeError, match="single-use|first idle load"):
        if entry == "load":
            case.owner.load_model()
        else:
            case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    assert case.events == (
        ["native-load", "native-kv"] if entry == "kv" else ["native-load"]
    )
    assert case.owner.hccl_v41_graph_eager.control.cache is first
    assert not first.closed and not first.terminal and not case.transport.terminal


@pytest.mark.parametrize("entry", ["load", "kv"])
def test_automatic_graph_native_initialization_failure_retains_terminal_owners(
    automatic_graph_case, entry
):
    case = automatic_graph_case
    if entry == "load":
        case.native.fail_load = True
    else:
        case.owner.load_model()
        case.native.fail_kv = True
    with pytest.raises(RuntimeError, match="native load|native KV"):
        if entry == "load":
            case.owner.load_model()
        else:
            case.owner.initialize_kv_cache(case.owner.kv_cache_config)
    assert case.transport.terminal and not case.transport.closed
    if entry == "kv":
        assert case.owner.hccl_v41_graph_eager.control.cache.terminal
        assert not case.owner.hccl_v41_graph_eager.control.cache.closed
    with pytest.raises(RuntimeError, match="single-use|first idle load"):
        if entry == "load":
            case.owner.load_model()
        else:
            case.owner.initialize_kv_cache(case.owner.kv_cache_config)


@pytest.mark.parametrize("post_kv", [False, True])
@pytest.mark.parametrize("sync_load", [False, True])
def test_real_native_zero_work_preserves_kv_results_and_single_retirement(
    monkeypatch, post_kv, sync_load
):
    case = make_native_kv_runner_case(monkeypatch, post_kv=post_kv, sync_load=sync_load)

    output = case.owner.execute_model(case.scheduler)

    result = output.kv_connector_output
    assert result.finished_sending == {"sent"}
    assert result.finished_recving == {"received", "failed"}
    assert result.failed_recving == {"failed"}
    assert result.invalid_block_ids == {3, 7}
    assert case.kv_transfer.bound_metadata is None
    assert case.kv_transfer.registered_views is case.kv_views
    assert not case.auxiliary_device.lookup_live
    assert not case.auxiliary_device.metadata_live
    assert case.auxiliary_device.reset_events is False
    # This is the native no_forward lifecycle, including no wait_for_save;
    # it is not a rewritten execute body or a hand-invoked pre/post sequence.
    expected = ["kv-bind", "kv-load", "kv-finish"]
    if not sync_load:
        expected = ["kv-bind", "kv-finish", "kv-load"]
    assert case.native_events == expected + [
        "kv-results",
        "kv-clear",
        "engram-finish",
        "metadata-finish",
    ]
    plans = [
        message.plan
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == "FORWARD"
    ]
    assert len(plans) == 1
    assert plans[0].mode == "target"
    assert plans[0].graph_execution.phase == "eager"
    assert plans[0].stages[0].attention_valid_tokens == (0,)
    assert plans[0].stages[0].attention_padded_tokens == (0,)
    assert case.transport.tensor_peer.messages == []
    assert case.transport.plan is None and case.transport.graph_reservation is None
    assert not case.transport.terminal


@pytest.mark.parametrize("post_kv", [False, True])
@pytest.mark.parametrize("sync_load", [False, True])
def test_real_native_zero_work_load_error_retires_engram_without_a_false_parent(
    monkeypatch, post_kv, sync_load
):
    case = make_native_kv_runner_case(
        monkeypatch, post_kv=post_kv, sync_load=sync_load, fail_load=True
    )

    with pytest.raises(RuntimeError, match="controlled KV load failure") as raised:
        case.owner.execute_model(case.scheduler)

    assert raised.value is case.kv_transfer.load_error
    assert case.kv_transfer.bound_metadata is case.scheduler.kv_connector_metadata
    assert case.kv_transfer.registered_views is case.kv_views
    assert not case.auxiliary_device.lookup_live
    assert not case.auxiliary_device.metadata_live
    assert case.auxiliary_device.reset_events is True
    expected = ["kv-bind", "kv-load"]
    if not sync_load:
        expected = ["kv-bind", "kv-finish", "kv-load"]
    assert case.native_events == expected + ["engram-failed", "metadata-finish"]
    assert not any(
        direction == "send" and message.kind in ("FORWARD", "COMPLETE")
        for direction, _, _, message in case.transport.transcript
    )
    assert case.transport.tensor_peer.messages == []
    assert case.transport.terminal and not case.transport.closed
    assert case.cache.terminal and not case.cache.closed


def test_real_native_full_replay_fixture_returns_none_before_sampling(monkeypatch):
    case = make_native_full_kv_replay_case(monkeypatch)
    resources = case.recorded_resources

    assert case.owner.execute_model(case.scheduler) is None

    assert case.kv_transfer.bound_metadata is case.scheduler.kv_connector_metadata
    assert case.kv_transfer.registered_views is case.kv_views
    assert case.native_events == [
        "inputs-ready",
        "attn-ready",
        "kv-bind",
        "kv-load",
        "replay-device-boundary",
        "metadata-replay-finish",
        "kv-finish",
        "engram-finish",
        "metadata-finish",
    ]
    assert not case.auxiliary_device.lookup_live
    assert not case.auxiliary_device.metadata_live
    assert not case.auxiliary_device.replay_live
    assert case.auxiliary_device.reset_events is False
    entry = case.cache.find(resources.bank.key, case.generation())
    assert entry.resources is resources and entry.graph is case.recorded_graph
    assert case.manager.hidden_states is resources.hidden_states
    assert not case.recorded_graph.closed
    messages = [
        message
        for direction, _, _, message in case.transport.transcript[
            len(case.replay_baseline) :
        ]
        if direction == "send" and message.kind in ("FORWARD", "COMPLETE")
    ]
    assert [message.kind for message in messages] == ["FORWARD", "COMPLETE"]
    plan = messages[0].plan
    assert plan.graph_execution.phase == "replay"
    assert plan.stages[0].attention_valid_tokens == (case.batch.num_tokens,)
    assert plan.stages[0].attention_padded_tokens == resources.bank.stage_rows
    assert messages[1].transaction_id == plan.transaction_id
    assert messages[1].plan_digest == plan.digest
    assert case.transport.tensor_peer.messages == []
    assert all(endpoint.messages == [] for endpoint in case.model_case.endpoints)
    assert case.transport.plan is None and case.transport.graph_reservation is None
    assert not case.transport.terminal


@pytest.mark.parametrize("stages", [1, 2])
def test_dspark_native_rejection_history_and_next_proposal(  # long-test: native cycle
    monkeypatch, stages
):
    case = make_native_dspark_sampling_case(
        monkeypatch, make_dspark_proposal_case, stages=stages
    )
    owner = case.owner
    assert owner.execute_model(case.scheduler) is None
    assert case.kv_transfer.bound_metadata is case.scheduler.kv_connector_metadata
    protocol = tuple(case.transport.transcript)
    assert [
        message.kind
        for direction, _, _, message in protocol[len(case.replay_baseline) :]
        if direction == "send" and message.kind in ("FORWARD", "COMPLETE")
    ] == ["FORWARD", "COMPLETE"]
    plan = next(
        message.plan
        for direction, _, _, message in protocol[len(case.replay_baseline) :]
        if direction == "send" and message.kind == "FORWARD"
    )
    expected_rows = (9,) if stages == 1 else (4, 5)
    assert tuple(stage.attention_valid_tokens for stage in plan.stages) == tuple(
        (rows,) for rows in expected_rows
    )
    assert plan.graph_execution.key.padded_tokens == expected_rows
    assert [
        (stage.request_start, stage.request_stop, stage.token_start, stage.token_stop)
        for stage in plan.graph_execution.key.stage_layouts
    ] == ([(0, 3, 0, 9)] if stages == 1 else [(0, 2, 0, 4), (1, 3, 4, 9)])
    assert not case.auxiliary_device.lookup_live
    assert not case.auxiliary_device.metadata_live
    assert not case.auxiliary_device.replay_live

    async_output = owner.sample_tokens(None)
    output = async_output.get_output()

    assert output.req_ids == case.batch.req_ids
    assert output.sampled_token_ids == [[11, 12, 13], [29], [31, 39]]
    assert async_output.sampler_output.num_sampled.tolist() == [3, 1, 2]
    assert async_output.sampler_output.num_rejected.tolist() == [0, 2, 1]
    state = owner.req_states
    expected = case.initial_history.clone()
    expected[0, 13:16] = torch.tensor([11, 12, 13])
    expected[1, 13] = 29
    expected[2, 13:15] = torch.tensor([31, 39])
    assert torch.equal(state.all_token_ids.gpu, expected)
    assert state.last_sampled_tokens.tolist() == [13, 29, 39]
    assert state.total_len.gpu.tolist() == [16, 14, 15]
    assert state.num_computed_tokens.gpu.tolist() == [15, 13, 14]
    assert state.draft_tokens.tolist() == [[14, 15], [13, 14], [6, 7]]
    assert output.kv_connector_output.finished_sending == {"sent"}
    assert case.kv_transfer.bound_metadata is None
    assert tuple(case.transport.transcript) == protocol
    assert case.transport.tensor_peer.messages == []
    next_inputs = owner.model_state.prepare_engram_inputs(case.batch, state)
    assert next_inputs["lookback_token_ids"].tolist() == [
        [12, 11, 10, 11],
        [20, 11, 10, 9],
        [31, 30, 11, 10],
        [-1, -1, -1, -1],
    ]
    assert next_inputs["engram_query_start_loc"].tolist() == [0, 3, 6, 9]
    assert next_inputs["engram_valid_token_count"].tolist() == [9]
    assert tuple(case.transport.transcript) == protocol
    assert not case.cache.terminal and not case.transport.terminal


@pytest.mark.parametrize("stages", [1, 2])
def test_native_dspark_proposal_failure_retains_post_rejection_history_and_kv(
    monkeypatch, stages
):
    case = make_native_dspark_sampling_case(
        monkeypatch, make_dspark_proposal_case, fail_backbone=True, stages=stages
    )
    owner = case.owner
    assert owner.execute_model(case.scheduler) is None
    protocol = tuple(case.transport.transcript)

    with pytest.raises(
        RuntimeError, match="controlled DSpark backbone failure"
    ) as raised:
        owner.sample_tokens(None)

    assert raised.value is case.proposal.model.error
    assert owner.req_states.last_sampled_tokens.tolist() == [13, 29, 39]
    assert owner.req_states.total_len.gpu.tolist() == [16, 14, 15]
    assert owner.req_states.num_computed_tokens.gpu.tolist() == [15, 13, 14]
    assert owner.req_states.draft_tokens.tolist() == [[-1, -1]] * 3
    assert case.kv_transfer.bound_metadata is case.scheduler.kv_connector_metadata
    assert tuple(case.transport.transcript) == protocol
    assert case.transport.tensor_peer.messages == []
    assert not case.cache.terminal and not case.transport.terminal
    assert not case.recorded_graph.closed
    assert owner.sample_tokens(None) is None


@pytest.mark.parametrize("stages", [1, 2])
def test_dspark_cycles_use_updated_queries_and_history(  # long-test: two native steps
    monkeypatch, stages
):
    case = make_native_dspark_sampling_case(
        monkeypatch, make_dspark_proposal_case, stages=stages
    )
    owner = case.owner
    assert owner.execute_model(case.scheduler) is None
    assert owner.sample_tokens(None).get_output().sampled_token_ids == [
        [11, 12, 13],
        [29],
        [31, 39],
    ]
    next_batch = advance_native_dspark_sampling_case(case)

    assert owner.execute_model(case.scheduler) is None
    assert owner.model_state.lookback_token_ids.tolist() == [
        [12, 11, 10, 11],
        [20, 11, 10, 9],
        [31, 30, 11, 10],
        [-1, -1, -1, -1],
    ]
    assert owner.req_states.num_computed_tokens_np.tolist() == [15, 13, 14]
    protocol = tuple(case.transport.transcript)
    async_output = owner.sample_tokens(None)
    output = async_output.get_output()
    assert output.sampled_token_ids == [[41], [13, 42], [6, 7, 44]]
    assert async_output.sampler_output.num_sampled.tolist() == [1, 2, 3]
    assert async_output.sampler_output.num_rejected.tolist() == [2, 1, 0]
    state = owner.req_states
    assert torch.equal(state.all_token_ids.gpu, case.expected_second_history)
    assert state.total_len.gpu.tolist() == [17, 16, 18]
    assert state.num_computed_tokens.gpu.tolist() == [16, 15, 17]
    assert state.last_sampled_tokens.tolist() == [41, 42, 44]
    assert state.draft_tokens.tolist() == [[8, 9], [9, 10], [11, 12]]
    assert output.kv_connector_output.finished_sending == {"sent"}
    assert case.kv_transfer.bound_metadata is None
    assert not case.auxiliary_device.lookup_live
    assert not case.auxiliary_device.metadata_live
    assert not case.auxiliary_device.replay_live
    next_inputs = owner.model_state.prepare_engram_inputs(next_batch, state)
    assert next_inputs["lookback_token_ids"].tolist() == [
        [13, 12, 11, 10],
        [13, 29, 20, 11],
        [7, 6, 39, 31],
        [-1, -1, -1, -1],
    ]
    forwards = [
        message
        for direction, _, _, message in protocol[len(case.replay_baseline) :]
        if direction == "send" and message.kind in ("FORWARD", "COMPLETE")
    ]
    assert [message.kind for message in forwards] == [
        "FORWARD",
        "COMPLETE",
        "FORWARD",
        "COMPLETE",
    ]
    first, second = forwards[0].plan, forwards[2].plan
    assert first.transaction_id != second.transaction_id
    assert first.graph_execution.key == second.graph_execution.key
    assert first.graph_execution.key.padded_tokens == ((9,) if stages == 1 else (4, 5))
    assert tuple(case.transport.transcript) == protocol
    assert case.transport.tensor_peer.messages == []
    entry = case.cache.find(case.recorded_resources.bank.key, case.generation())
    assert entry.graph is case.recorded_graph
    assert entry.resources is case.recorded_resources
    assert not case.cache.terminal and not case.transport.terminal


@pytest.mark.parametrize("stages", [1, 2])
def test_second_dspark_proposal_failure_retains_advanced_history_and_kv(
    monkeypatch, stages
):
    case = make_native_dspark_sampling_case(
        monkeypatch, make_dspark_proposal_case, stages=stages
    )
    owner = case.owner
    assert owner.execute_model(case.scheduler) is None
    assert owner.sample_tokens(None).get_output().sampled_token_ids == [
        [11, 12, 13],
        [29],
        [31, 39],
    ]
    advance_native_dspark_sampling_case(case, fail_backbone=True)
    assert owner.execute_model(case.scheduler) is None
    protocol = tuple(case.transport.transcript)

    with pytest.raises(
        RuntimeError, match="controlled second DSpark backbone failure"
    ) as raised:
        owner.sample_tokens(None)

    assert raised.value is case.proposal.model.error
    assert owner.req_states.last_sampled_tokens.tolist() == [41, 42, 44]
    assert owner.req_states.total_len.gpu.tolist() == [17, 16, 18]
    assert owner.req_states.num_computed_tokens.gpu.tolist() == [16, 15, 17]
    assert owner.req_states.draft_tokens.tolist() == [[14, 15], [13, 14], [6, 7]]
    assert torch.equal(owner.req_states.all_token_ids.gpu, case.expected_second_history)
    assert case.kv_transfer.bound_metadata is case.scheduler.kv_connector_metadata
    assert tuple(case.transport.transcript) == protocol
    assert case.transport.tensor_peer.messages == []
    assert not case.cache.terminal and not case.transport.terminal
    assert not case.recorded_graph.closed
    assert owner.sample_tokens(None) is None


@pytest.mark.parametrize("stages", [1, 2])
def test_native_dspark_hit_miss_hit_keeps_history(  # long-test: three native cycles
    monkeypatch, stages
):
    case = make_native_dspark_sampling_case(
        monkeypatch, make_dspark_proposal_case, stages=stages, eager_fallback=True
    )
    owner, state = case.owner, case.owner.req_states
    key = case.recorded_resources.bank.key
    entry = case.cache.find(key, case.generation())
    retained = entry.resources.hidden_states
    retained_value, retained_storage = retained.clone(), retained.data_ptr()
    expected = (
        dict(
            readiness="ready",
            predictions=None,
            tokens=[[11, 12, 13], [29], [31, 39]],
            sampled=[3, 1, 2],
            rejected=[0, 2, 1],
            computed=[15, 13, 14],
            total=[16, 14, 15],
            last=[13, 29, 39],
            drafts=[[14, 15], [13, 14], [6, 7]],
        ),
        dict(
            readiness="missing",
            predictions=(41, 15, 16, 13, 42, 43, 6, 7, 44),
            tokens=[[41], [13, 42], [6, 7, 44]],
            sampled=[1, 2, 3],
            rejected=[2, 1, 0],
            computed=[16, 15, 17],
            total=[17, 16, 18],
            last=[41, 42, 44],
            drafts=[[8, 9], [9, 10], [11, 12]],
        ),
        dict(
            readiness="ready",
            predictions=(8, 47, 48, 9, 10, 49, 50, 12, 51),
            tokens=[[8, 47], [9, 10, 49], [50]],
            sampled=[2, 3, 1],
            rejected=[1, 0, 2],
            computed=[18, 18, 18],
            total=[19, 19, 19],
            last=[47, 49, 50],
            drafts=[[14, 15], [16, 0], [0, 1]],
        ),
    )
    for index, result in enumerate(expected):
        if index:
            advance_native_dspark_sampling_case(case, predictions=result["predictions"])
        # Only the external F report changes. The same real coordinator owns
        # native dispatch/reservation, preparation, replay or eager execution.
        case.transport.states[0] = result["readiness"]
        previous_history = state.all_token_ids.gpu.clone()
        assert owner.execute_model(case.scheduler) is None
        assert torch.equal(state.all_token_ids.gpu, previous_history)
        assert case.kv_transfer.bound_metadata is case.scheduler.kv_connector_metadata
        assert not case.auxiliary_device.lookup_live
        assert not case.auxiliary_device.metadata_live
        assert not case.auxiliary_device.replay_live
        if result["readiness"] == "missing":
            # ExecuteModelState is native's published execute->sample consumer
            # seam. Read its actual outputs; never replace them with the oracle.
            published = owner.execute_model_state
            hidden, auxiliary = native_dspark_eager_reference(case)
            torch.testing.assert_close(published.hidden_states, hidden)
            for actual, reference in zip(
                published.aux_hidden_states, auxiliary, strict=True
            ):
                torch.testing.assert_close(actual, reference)
            assert not torch.equal(published.hidden_states, retained_value)
            assert state.num_computed_tokens_np.tolist() == [15, 13, 14]
        protocol = tuple(case.transport.transcript)
        sampled = owner.sample_tokens(None)
        output = sampled.get_output()
        assert output.sampled_token_ids == result["tokens"]
        assert sampled.sampler_output.num_sampled.tolist() == result["sampled"]
        assert sampled.sampler_output.num_rejected.tolist() == result["rejected"]
        assert state.num_computed_tokens.gpu.tolist() == result["computed"]
        assert state.total_len.gpu.tolist() == result["total"]
        assert state.last_sampled_tokens.tolist() == result["last"]
        assert state.draft_tokens.tolist() == result["drafts"]
        assert output.kv_connector_output.finished_sending == {"sent"}
        assert case.kv_transfer.bound_metadata is None
        assert tuple(case.transport.transcript) == protocol
        draft_context = case.proposal.model.contexts[-1]
        assert HCCL_V41_PARENT_FORWARD_KEY not in (
            draft_context.additional_kwargs or {}
        )
        assert (
            context_module.get_afd_metadata_from_forward_context(draft_context) is None
        )
        assert case.cache.find(key, case.generation()) is entry
        assert retained.data_ptr() == retained_storage
        torch.testing.assert_close(retained, retained_value)

    expected_history = case.initial_history.clone()
    expected_history[0, 13:19] = torch.tensor([11, 12, 13, 41, 8, 47])
    expected_history[1, 13:19] = torch.tensor([29, 13, 42, 9, 10, 49])
    expected_history[2, 13:19] = torch.tensor([31, 39, 6, 7, 44, 50])
    assert torch.equal(state.all_token_ids.gpu, expected_history)
    next_inputs = owner.model_state.prepare_engram_inputs(case.batch, state)
    assert next_inputs["lookback_token_ids"].tolist() == [
        [8, 41, 13, 12],
        [10, 9, 42, 13],
        [44, 7, 6, 39],
        [-1, -1, -1, -1],
    ]
    messages = [
        message
        for direction, _, _, message in protocol[len(case.replay_baseline) :]
        if direction == "send" and message.kind in ("FORWARD", "COMPLETE")
    ]
    assert [message.kind for message in messages] == ["FORWARD", "COMPLETE"] * 3
    plans = [message.plan for message in messages if message.kind == "FORWARD"]
    assert [plan.graph_execution.phase for plan in plans] == [
        "replay",
        "eager",
        "replay",
    ]
    assert len({plan.transaction_id for plan in plans}) == 3
    assert plans[0].graph_execution.key == plans[2].graph_execution.key == key
    payloads = case.transport.tensor_peer.messages
    for layer in (0, 1):
        layer_payloads = [payload for payload in payloads if payload[0] == layer]
        torch.testing.assert_close(
            torch.cat([payload[4] for payload in layer_payloads]),
            torch.tensor([13, 14, 15, 29, 13, 14, 39, 6, 7], dtype=torch.int32),
        )
        assert all(payload[3].dtype == torch.float32 for payload in layer_payloads)
        assert any(
            not torch.equal(payload[3], payload[2].float())
            for payload in layer_payloads
        )
    assert not case.transport.terminal and not case.cache.terminal


def test_real_native_full_replay_fixture_failure_retains_kv_and_graph_owners(
    monkeypatch,
):
    case = make_native_full_kv_replay_case(monkeypatch, fail_replay=True)
    resources = case.recorded_resources

    with pytest.raises(RuntimeError, match="controlled FULL replay failure") as raised:
        case.owner.execute_model(case.scheduler)

    assert raised.value is case.replay_error
    assert case.kv_transfer.bound_metadata is case.scheduler.kv_connector_metadata
    assert case.kv_transfer.registered_views is case.kv_views
    assert case.native_events == [
        "inputs-ready",
        "attn-ready",
        "kv-bind",
        "kv-load",
        "replay-device-boundary",
        "metadata-replay-finish",
        "engram-failed",
        "metadata-finish",
    ]
    assert not case.auxiliary_device.lookup_live
    assert not case.auxiliary_device.metadata_live
    assert not case.auxiliary_device.replay_live
    assert case.auxiliary_device.reset_events is True
    assert case.manager.hidden_states is resources.hidden_states
    assert not case.recorded_graph.closed
    assert not case.cache.is_empty
    assert case.cache.terminal and not case.cache.closed
    assert case.transport.terminal and not case.transport.closed
    messages = [
        message
        for direction, _, _, message in case.transport.transcript[
            len(case.replay_baseline) :
        ]
        if direction == "send" and message.kind in ("FORWARD", "COMPLETE")
    ]
    assert [message.kind for message in messages] == ["FORWARD"]
    assert messages[0].plan.graph_execution.phase == "replay"
    assert case.transport.tensor_peer.messages == []
    assert all(endpoint.messages == [] for endpoint in case.model_case.endpoints)


@pytest.mark.parametrize("fail_backbone", [False, True])
def test_native_factory_outputs_survive_dspark_proposal(  # long-test: native factory
    monkeypatch, fail_backbone
):
    case = make_native_dspark_factory_case(
        monkeypatch, make_dspark_proposal_case, fail_backbone=fail_backbone
    )
    owner, runtime = case.owner, case.capture_runtime
    assert owner.capture_model() == 0
    coordinator = owner.hccl_v41_graph_coordinator
    key = coordinator.control.key_for(coordinator.coverage.geometries[0])
    entry = case.cache.find(key, case.generation())
    resources = entry.resources
    retained, auxiliary = resources.hidden_states, resources.auxiliary
    expected_hidden, expected_auxiliary = case.capture_reference
    torch.testing.assert_close(retained, expected_hidden)
    for actual, expected in zip(auxiliary, expected_auxiliary, strict=True):
        torch.testing.assert_close(actual, expected)
    retained_values = (retained.clone(),) + tuple(t.clone() for t in auxiliary)
    assert resources.input_buffers is owner.input_buffers
    assert resources.manager is owner.cudagraph_manager
    assert resources.pipeline.selection.phase == "graph_capture"
    assert not resources.pipeline.policy.hybrid
    capture_plans = [
        message.plan
        for direction, _, _, message in case.transport.transcript
        if direction == "send" and message.kind == "FORWARD"
    ]
    assert [plan.graph_execution.phase for plan in capture_plans] == [
        "eager",
        "warmup",
        "capture",
    ]
    assert all(
        stage.attention_valid_tokens == (0,)
        for plan in capture_plans
        for stage in plan.stages
    )
    assert key.padded_tokens == (4, 5)
    for exchange in resources.bank.exchanges:
        packets = exchange.endpoint.messages
        for hidden, router, ids in zip(
            packets[::3], packets[1::3], packets[2::3], strict=True
        ):
            assert hidden.dtype == torch.bfloat16 and router.dtype == torch.float32
            torch.testing.assert_close(router, hidden.float() + 0.00031)
            assert ids.dtype == torch.int32 and torch.equal(ids, torch.zeros_like(ids))
    assert all(
        record[-2:] == (True, True)
        for record in runtime.trace
        if record[0] == "attention"
    )
    assert ("parent_stream", "output", True, True) in runtime.trace
    assert all(
        record[-1] == "parent" for record in runtime.trace if record[0] == "wire"
    )
    assert torch.npu.current_stream() is runtime.parent
    assert torch_utils.current_stream() is runtime.parent
    before_replay = tuple(runtime.trace)
    protocol_start = len(case.transport.transcript)

    # Native execute(None) publishes the persistent factory outputs to its
    # sample consumer. The inert device replay does not rerun Python compute.
    assert owner.execute_model(case.scheduler) is None
    published = owner.execute_model_state
    torch.testing.assert_close(published.hidden_states, expected_hidden)
    for actual, expected in zip(
        published.aux_hidden_states, expected_auxiliary, strict=True
    ):
        torch.testing.assert_close(actual, expected)
    assert tuple(runtime.trace) == before_replay
    assert not case.auxiliary_device.lookup_live
    assert not case.auxiliary_device.metadata_live
    assert not case.auxiliary_device.replay_live
    assert torch.equal(owner.req_states.all_token_ids.gpu, case.initial_history)
    assert case.kv_transfer.bound_metadata is case.scheduler.kv_connector_metadata
    completed_protocol = tuple(case.transport.transcript)
    assert [
        message.kind
        for direction, _, _, message in completed_protocol[protocol_start:]
        if direction == "send" and message.kind in ("FORWARD", "COMPLETE")
    ] == ["FORWARD", "COMPLETE"]
    if fail_backbone:
        with pytest.raises(RuntimeError, match="controlled DSpark backbone failure"):
            owner.sample_tokens(None)
        assert case.kv_transfer.bound_metadata is case.scheduler.kv_connector_metadata
        assert owner.req_states.draft_tokens.tolist() == [[-1, -1]] * 3
    else:
        sampled = owner.sample_tokens(None)
        output = sampled.get_output()
        assert output.sampled_token_ids == [[11, 12, 13], [29], [31, 39]]
        assert sampled.sampler_output.num_sampled.tolist() == [3, 1, 2]
        assert sampled.sampler_output.num_rejected.tolist() == [0, 2, 1]
        assert owner.req_states.draft_tokens.tolist() == [[14, 15], [13, 14], [6, 7]]
        assert output.kv_connector_output.finished_sending == {"sent"}
        assert case.kv_transfer.bound_metadata is None
    state = owner.req_states
    expected_history = case.initial_history.clone()
    expected_history[0, 13:16] = torch.tensor([11, 12, 13])
    expected_history[1, 13] = 29
    expected_history[2, 13:15] = torch.tensor([31, 39])
    assert torch.equal(state.all_token_ids.gpu, expected_history)
    assert state.num_computed_tokens.gpu.tolist() == [15, 13, 14]
    assert state.total_len.gpu.tolist() == [16, 14, 15]
    assert state.last_sampled_tokens.tolist() == [13, 29, 39]
    assert ("draft_parent", True, True) in runtime.trace
    draft_context = case.proposal.model.contexts[-1]
    assert HCCL_V41_PARENT_FORWARD_KEY not in (draft_context.additional_kwargs or {})
    assert context_module.get_afd_metadata_from_forward_context(draft_context) is None
    assert tuple(case.transport.transcript) == completed_protocol
    assert case.cache.find(key, case.generation()) is entry
    for actual, expected in zip((retained,) + auxiliary, retained_values, strict=True):
        torch.testing.assert_close(actual, expected)
    assert (
        not entry.graph.closed
        and not case.cache.terminal
        and not case.transport.terminal
    )
    assert torch.npu.current_stream() is runtime.parent
    assert torch_utils.current_stream() is runtime.parent


def test_real_native_full_replay_fixture_sampling_returns_tokens_and_kv_results(
    monkeypatch,
):
    case = make_native_full_kv_sampling_case(monkeypatch)
    assert case.owner.execute_model(case.scheduler) is None
    assert case.kv_transfer.bound_metadata is case.scheduler.kv_connector_metadata
    completed_protocol = tuple(case.transport.transcript)

    async_output = case.owner.sample_tokens(None)
    output = async_output.get_output()

    assert output.req_ids == case.batch.req_ids
    assert output.sampled_token_ids == [[0], [1], [2], [3]]
    assert output.prompt_logprobs_dict == {}
    result = output.kv_connector_output
    assert result.finished_sending == {"sent"}
    assert result.finished_recving == {"received", "failed"}
    assert result.failed_recving == {"failed"}
    assert result.invalid_block_ids == {3, 7}
    expected = case.initial_history.clone()
    expected[:, FULL_REPLAY_HISTORY_TOKENS] = torch.arange(case.batch.num_reqs)
    state = case.owner.req_states
    assert torch.equal(state.all_token_ids.gpu, expected)
    assert state.last_sampled_tokens.tolist() == [0, 1, 2, 3]
    assert state.total_len.gpu.tolist() == [FULL_REPLAY_HISTORY_TOKENS + 1] * 4
    assert (
        state.num_computed_tokens.gpu.tolist() == [FULL_REPLAY_HISTORY_TOKENS + 1] * 4
    )
    assert case.kv_transfer.bound_metadata is None
    assert case.kv_transfer.registered_views is case.kv_views
    assert case.native_events[-4:] == [
        "sample-boundary",
        "kv-wait-save",
        "kv-results",
        "kv-clear",
    ]
    assert tuple(case.transport.transcript) == completed_protocol
    assert case.transport.tensor_peer.messages == []
    assert all(endpoint.messages == [] for endpoint in case.model_case.endpoints)
    resources = case.recorded_resources
    entry = case.cache.find(resources.bank.key, case.generation())
    assert entry.graph is case.recorded_graph and entry.resources is resources
    assert not case.cache.terminal and not case.transport.terminal
    assert case.owner.sample_tokens(None) is None
    assert async_output.get_output() is output


def test_real_native_full_replay_fixture_sampling_error_retains_unposted_kv(
    monkeypatch,
):
    case = make_native_full_kv_sampling_case(monkeypatch, fail_sample=True)
    assert case.owner.execute_model(case.scheduler) is None
    completed_protocol = tuple(case.transport.transcript)
    retired_events = tuple(case.native_events)

    with pytest.raises(RuntimeError, match="controlled sampler failure") as raised:
        case.owner.sample_tokens(None)

    assert raised.value is case.sampling_error
    assert case.owner.sample_tokens(None) is None
    assert tuple(case.native_events) == retired_events + ("sample-boundary",)
    assert case.kv_transfer.bound_metadata is case.scheduler.kv_connector_metadata
    assert case.kv_transfer.registered_views is case.kv_views
    assert torch.equal(case.owner.req_states.all_token_ids.gpu, case.initial_history)
    assert case.owner.req_states.last_sampled_tokens.tolist() == [-1] * 4
    assert (
        case.owner.req_states.total_len.gpu.tolist() == [FULL_REPLAY_HISTORY_TOKENS] * 4
    )
    assert tuple(case.transport.transcript) == completed_protocol
    assert case.transport.tensor_peer.messages == []
    assert all(endpoint.messages == [] for endpoint in case.model_case.endpoints)
    resources = case.recorded_resources
    entry = case.cache.find(resources.bank.key, case.generation())
    assert entry.graph is case.recorded_graph and entry.resources is resources
    assert not case.cache.terminal and not case.transport.terminal
    assert not case.recorded_graph.closed
