# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Native CPU preparation/control assembly with explicit device boundaries.

No graph or NPU kernel is simulated. The host capture boundary executes CPU
arithmetic once per factory phase and records an inert retained graph handle.
"""

from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import replace
from itertools import count
from types import MethodType, SimpleNamespace

import torch
from torch import nn
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.v1.worker.gpu import model_runner as native_runner
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager, ModelWithContext

from afd_plugin.connectors.npu.hccl_graph_cache import HcclGraphCache
from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_graph import (
    HcclGraphCandidate,
    HcclGraphReport,
    decide_graph_execution,
    graph_capture_sequence_digest,
)
from afd_plugin.distributed.hccl_plan import (
    HcclParentPlan,
    HcclPlanCapabilities,
    HcclPlanMessage,
)
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    AFDDeepseekV41AttentionForCausalLM,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph import capture_hccl_v41_geometry
from afd_plugin.v1.worker.npu.hccl_v41_graph_session import HcclV41GraphSession
from afd_plugin.v1.worker.npu.hccl_v41_ubatch_runner import HcclV41UBatchRunner
from tests.unit.fakes.batch_req_state import make_batch_req_state
from tests.unit.fakes.deepseek_v41 import ParentEngramState, make_staged_model
from tests.unit.fakes.deepseek_v41_graph import GraphFFNEndpoint, make_graph_model_case
from tests.unit.fakes.graph_cache import DeviceCompletion, ReplayGraph
from tests.unit.fakes.graph_capture import cpu_capture_boundary
from tests.unit.fakes.hccl_graph_stage import make_graph_stage_bank
from tests.unit.fakes.hccl_v41_graph_control import make_attention_graph_control_case
from tests.unit.fakes.input_batch import make_ascend_decode_batch, make_hccl_ubatch_seam
from tests.unit.fakes.vllm_config import v41_dspark_config


def make_coordinator_case(
    *,
    stages=2,
    descriptors=None,
    prefix=None,
    role_rank=0,
    exchange_order="stage_serial",
    target_query_len=1,
    engram=False,
):
    tokens = 4 if target_query_len == 1 else 3 * target_query_len
    descriptor = BatchExecutionDescriptor(
        CUDAGraphMode.FULL,
        tokens,
        tokens // target_query_len,
        uniform_token_count=target_query_len,
        num_ubatches=stages,
    )
    descriptors = (descriptor,) if descriptors is None else tuple(descriptors)
    geometry = capture_hccl_v41_geometry(descriptor, target_query_len=target_query_len)
    rows = tuple(stage.token_stop - stage.token_start for stage in geometry.stages)
    scheduled = (target_query_len,) * descriptor.num_reqs
    capacity = max((8, *(item.num_tokens for item in descriptors)))
    capabilities = HcclPlanCapabilities(
        4,
        128,
        capacity,
        (0, 1),
        1,
        1,
        1,
        local_counts_version=1,
        graph_execution_version=1,
        graph_exchange_order=exchange_order,
        target_query_len=target_query_len,
    )
    case = make_attention_graph_control_case(
        descriptor=descriptor,
        local_descriptors=descriptors,
        cached=False,
        peer_states={0: "missing"},
        capabilities=capabilities,
        rows=rows,
        scheduled=scheduled,
    )
    if role_rank:
        capabilities = HcclPlanCapabilities(
            4,
            128,
            capacity,
            (0, 1),
            2,
            2,
            2,
            local_counts_version=1,
            graph_execution_version=1,
            graph_exchange_order=exchange_order,
            target_query_len=target_query_len,
        )
        case = make_attention_graph_control_case(
            descriptor=descriptor,
            local_descriptors=descriptors,
            cached=False,
            role_rank=role_rank,
            capabilities=capabilities,
            rows=rows,
            scheduled=scheduled,
        )
    model_case = make_graph_model_case(rows, engram=engram)
    wrapper = AFDDeepseekV41AttentionForCausalLM.__new__(
        AFDDeepseekV41AttentionForCausalLM
    )
    nn.Module.__init__(wrapper)
    wrapper.language_model = nn.Module()
    wrapper.language_model.model = model_case.model
    state = ParentEngramState({})
    state.model = wrapper
    state.attn_metadata = {"previous-generation": SimpleNamespace()}
    runner = make_hccl_ubatch_seam(
        HcclV41UBatchRunner, model_state=state, max_num_reqs=8
    )
    config = runner.vllm_config
    if target_query_len > 1:
        original = v41_dspark_config(tensor_parallel_size=capabilities.native_tp_size)
        config.speculative_config = original.speculative_config
        config.speculative_config.num_speculative_tokens = target_query_len - 1
        config.model_config.model = original.model_config.model
        vars(config.model_config.hf_text_config).update(
            vars(original.model_config.hf_text_config)
        )
        config.model_config.hf_text_config.dspark_target_layer_ids = [1, 2]
        draft = config.speculative_config.draft_model_config.hf_text_config
        draft.dspark_target_layer_ids = [1, 2]
    config.model_config.hf_config.model_type = "deepseek_v41"
    config.model_config.hf_config.architectures = ["DeepseekV41ForCausalLM"]
    config.additional_config["afd"]["connector_extra_config"] = {
        "graph_exchange_order": exchange_order
    }
    config.model_config.enforce_eager = False
    config.model_config.dtype = torch.bfloat16
    config.model_config.hf_text_config.hidden_size = capabilities.hidden_size
    config.model_config.hf_text_config.vocab_size = capabilities.vocab_size
    config.model_config.hf_text_config.num_hidden_layers = 2
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    config.parallel_config.tensor_parallel_size = capabilities.native_tp_size
    config.compilation_config.static_forward_context = {}
    session = HcclV41GraphSession(case.transport)
    runner.graph_owner, runner.transport = session, case.transport
    manager = object.__new__(ModelAclGraphManager)
    manager.__dict__.update(vars(case.manager))
    manager._capture_descs = {CUDAGraphMode.FULL: list(descriptors)}
    manager._max_full_descs_to_capture = prefix
    manager.hidden_states, manager.aux_hidden_states = None, []
    manager.intermediate_tensors = None
    manager.host_descriptors = descriptors[:prefix]
    manager.capture_failure = None
    manager.ubatch_runner = runner if stages == 2 else None
    owner = SimpleNamespace(
        vllm_config=config,
        decode_query_len=target_query_len,
        parallel_config=config.parallel_config,
        model=wrapper,
        model_state=state,
        input_buffers=SimpleNamespace(),
        intermediate_tensors=None,
        ubatch_runner=manager.ubatch_runner,
        cudagraph_manager=manager,
        hccl_v41_transport=case.transport,
        hccl_v41_graph_eager=None,
        attn_groups=[],
        block_tables=SimpleNamespace(),
        kv_cache_config=runner.kv_cache_config,
        kv_caches=[torch.zeros(1)],
        pcp_manager=None,
        device=SimpleNamespace(type="npu", index=0),
    )
    manager.model_runner = owner
    case.control.manager, case.manager = manager, manager
    ids = count()
    case.owner, case.session, case.runner = owner, session, runner
    case.transport.data_groups = [endpoint.group for endpoint in model_case.endpoints]
    case.model_case = model_case
    case.next_transaction_id = lambda: f"native-{next(ids)}"
    case.capture_generations = []
    case.boundary_failure = None
    case.native_state = case.batch
    case.runtime_metadata = []
    owner.gather_batch_req_state = lambda output, dummy: (
        None if dummy else case.native_state,
        None if dummy or case.native_state is None else target_query_len,
    )
    return case


def queue_preflight(case, keys, *, prefix=None, empty=False):
    """A real responder wire round, including the empty accepted parent."""
    candidate = HcclGraphCandidate(
        "eager" if empty else "replay",
        "warmup",
        None if empty else keys[0],
        (0,),
        capture_sequence_digest=graph_capture_sequence_digest(
            tuple(keys), max_full_descs_to_capture=prefix
        ),
    )
    decision = decide_graph_execution(
        "preflight", candidate, (HcclGraphReport("unsupported"),)
    )
    capabilities = case.transport.negotiated_capabilities
    plan = HcclParentPlan(
        "preflight",
        capabilities.layer_indices,
        (HcclStageLayout(0, (0, 0), (0, 0), 2, 2),),
        "warmup",
        decision,
    )
    messages = [
        HcclPlanMessage(
            "GRAPH_QUERY", transaction_id="preflight", graph_candidate=candidate
        ),
        HcclPlanMessage("GRAPH_DECISION", graph_decision=decision),
        HcclPlanMessage(
            "GRAPH_ACK",
            transaction_id="preflight",
            world_rank=2,
            graph_digest=decision.digest,
        ),
        HcclPlanMessage("COUNTS_REQUEST", transaction_id="preflight"),
        HcclPlanMessage("FORWARD", plan=plan),
        HcclPlanMessage(
            "ACCEPT", transaction_id="preflight", world_rank=2, plan_digest=plan.digest
        ),
        HcclPlanMessage(
            "COMPLETE", transaction_id="preflight", plan_digest=plan.digest
        ),
    ]
    case.transport.incoming[2].clear()
    case.transport.incoming[2].extend(messages)
    return candidate


def cpu_allocate_bank(
    case, transport, key, *, device, mask_padding=False, runtime=None
):
    """CPU storage/operator boundary; no NPU allocation or device claim."""
    endpoints = tuple(GraphFFNEndpoint() for _ in key.stage_layouts)
    for index, endpoint in enumerate(endpoints):
        endpoint.group = transport.data_groups[index]
        endpoint.stage_idx, endpoint.runtime = index, runtime
    original = make_graph_stage_bank(
        rows=key.padded_tokens,
        endpoint_factory=lambda index: endpoints[index],
        target_query_len=key.uniform_token_count,
    )
    bank = HcclGraphStageBank(
        role="attention",
        role_rank=transport.role_rank,
        capabilities=transport.negotiated_capabilities,
        key=key,
        exchanges=original.exchanges,
    )
    if mask_padding:
        for index, endpoint in enumerate(endpoints):
            receive = endpoint.receive

            def receive_valid(tensor, shape, *, receive=receive, index=index):
                receive(tensor, shape)
                # The explicit CPU peer obeys the same accepted valid-count
                # mask as FFN Graph. Capture dummy rows must return zero.
                tensor[bank.valid_counts[index] :].zero_()

            endpoint.receive = receive_valid
    return bank


def cpu_runner_capture(case, owner, *, profile_only=False):
    """Only native factory/wrapper host semantics, without device capture."""
    if owner is not case.owner:
        raise AssertionError("capture must receive the selected native runner")
    manager = owner.cudagraph_manager
    model = ModelWithContext(
        owner.model, device_metadata=owner.model_state.device_metadata
    )

    def factory(descriptor, warmup):
        if case.boundary_failure == "prepare":
            raise RuntimeError("native preparation failed")
        batch = make_ascend_decode_batch(
            descriptor.num_tokens, target_query_len=descriptor.uniform_token_count
        )
        slots = torch.zeros(1, descriptor.num_tokens, dtype=torch.int32)
        inputs = {"input_ids": batch.input_ids, "positions": batch.positions}
        if case.model_case.model.has_engram:
            inputs.update(
                (name, case.model_case.graph_inputs[name])
                for name in (
                    "lookback_token_ids",
                    "engram_query_start_loc",
                    "engram_valid_token_count",
                )
            )
        if descriptor.num_ubatches == 2:
            prepared = owner.ubatch_runner.prepare(
                batch, (), slots, CUDAGraphMode.FULL, True
            )
            finish = owner.ubatch_runner.begin_capturable_run(
                model, inputs, prepared, for_capture=True
            )
            context = nullcontext
        else:
            metadata = owner.model_state.prepare_attn(
                batch,
                CUDAGraphMode.NONE,
                (),
                slots,
                owner.attn_groups,
                owner.kv_cache_config,
                for_capture=True,
            )

            def context():
                return set_forward_context(
                    metadata,
                    owner.vllm_config,
                    cudagraph_runtime_mode=CUDAGraphMode.NONE,
                )

            def finish():
                return model(**inputs)

        # The native factory prepares the parent after its actual metadata.
        if (
            owner.model_state.prepare_inputs(batch, None)["producer_mode"]
            != CUDAGraphMode.FULL
        ):
            raise AssertionError("native capture lost FULL parent input preparation")
        case.capture_generations.append((descriptor, warmup, batch))

        def forward(mode):
            before = tuple(case.transport.transcript)
            with context():
                result = finish()
            if tuple(case.transport.transcript) != before:
                raise AssertionError("tensor body performed control communication")
            manager.hidden_states, auxiliary = result
            manager.aux_hidden_states = auxiliary

        return forward

    CudaGraphManager.capture(manager, factory)
    return 17 if profile_only else 23


def cpu_target_manager_capture(case, manager, *args, **kwargs):
    """Replace only the model-manager factory/device boundary, not native order."""
    if manager is not case.manager:
        raise AssertionError("target capture must use its retained manager")
    cpu_runner_capture(case, case.owner)


def retain_native_target_capture(case):
    """Snapshot the public ready target before invoking a local draft boundary."""
    entry = case.cache.find(native_keys(case)[0], 0)
    if entry is None:
        raise AssertionError("draft started before target capture was published")
    case.target_entry = entry
    case.target_output = entry.resources.hidden_states
    case.target_value = case.target_output.clone()
    case.target_auxiliary = tuple(entry.resources.auxiliary)
    case.target_auxiliary_values = tuple(t.clone() for t in case.target_auxiliary)
    case.target_storage = case.target_output.data_ptr()
    case.target_transcript = tuple(case.transport.transcript)
    case.target_produced = tuple(case.model_case.producer_parents)
    case.capture_order.append("target-published")


def prepare_native_draft_capture(case, *, fail_draft=False):
    """Exercise native outer capture and a local NONE-manager draft boundary.

    DFlash's real NONE-manager capture has no factory forward. The explicit
    real _run_model call below additionally checks context isolation while the
    target capture hooks are still scoped. It is not a simulated device graph
    or a claim that native NONE capture itself executes that forward.
    """
    owner = case.owner
    owner.model_state.supports_mm_inputs = False
    owner.lora_config = None
    owner.maybe_setup_dummy_loras = lambda _: nullcontext()
    owner.use_aux_hidden_state_outputs = True
    owner.adaptive_verification = None
    owner._draft_workspace_lane = "cpu-draft-workspace"
    case.capture_order = []
    owner.kv_connector = SimpleNamespace(
        reset_capture_state=lambda: case.capture_order.append("native-kv-reset")
    )
    case.target_metadata = []
    producer = case.model_case.model.captured_engram_inputs

    @contextmanager
    def observed_producer(*args):
        case.target_metadata.append(get_forward_context().attn_metadata)
        with producer(*args) as produced:
            yield produced

    case.model_case.model.captured_engram_inputs = observed_producer
    draft_config = copy(owner.vllm_config)
    draft_config.model_config = copy(
        owner.vllm_config.speculative_config.draft_model_config
    )
    draft_config.model_config.enforce_eager = True
    draft_config.speculative_config = None
    draft_manager = DFlashCudaGraphManager.__new__(DFlashCudaGraphManager)
    draft_manager.cudagraph_mode = CUDAGraphMode.NONE
    draft_manager._capture_descs = {}
    draft_manager.host_descriptors = ()
    draft_manager.capture_failure = None
    draft_manager.graphs = {}
    draft = DFlashSpeculator.__new__(DFlashSpeculator)
    draft.vllm_config = draft_config
    draft.query_cudagraph_manager = draft_manager
    draft.sample_indices = torch.ones(3, dtype=torch.int64)
    draft.sample_pos = torch.ones(3, dtype=torch.int64)
    draft.sample_idx_mapping = torch.zeros(3, dtype=torch.int64)
    draft.input_buffers = SimpleNamespace(
        input_ids=torch.arange(3), positions=torch.arange(3)
    )
    draft.block_tables, draft.attn_groups = owner.block_tables, []
    draft.kv_cache_config, draft.max_model_len = owner.kv_cache_config, 32
    draft._group_causal = True
    case.draft_metadata = {"draft-only": SimpleNamespace()}
    case.draft_contexts = []
    case.draft_output = None

    def local_draft(input_ids, positions, inputs_embeds):
        case.draft_contexts.append(get_forward_context())
        case.capture_order.append("local-draft-forward")
        if fail_draft:
            raise RuntimeError("local draft forward failed")
        return input_ids[:, None] + positions[:, None] + 100

    draft.model = local_draft

    def capture():
        # Public ready-cache lookup must already succeed before native calls
        # speculator.capture. Preserve its resources for post-failure checks.
        retain_native_target_capture(case)
        DFlashSpeculator.capture(draft)
        case.capture_order.append("native-draft-capture-returned")
        case.draft_output = DFlashSpeculator._run_model(
            draft, 3, case.draft_metadata, None, None
        )

    draft.capture = capture
    owner.speculator = case.draft = draft
    return case


def cpu_retained_capture(
    manager, create_forward_fn, progress_bar_desc="Capturing CUDA graphs"
):
    """Host factory boundary with inert graph lifetime handles."""
    cpu_capture_boundary(manager, create_forward_fn, progress_bar_desc)
    manager.graphs = {descriptor: ReplayGraph() for descriptor in manager.graphs}


def native_keys(case):
    return tuple(
        case.control.key_for(
            capture_hccl_v41_geometry(
                descriptor,
                target_query_len=case.transport.negotiated_capabilities.target_query_len,
            )
        )
        for descriptor in case.manager.host_descriptors
    )


def dispatch_runtime(case, *, dummy=False, need_eager=False, tokens=None):
    """Call the selected real gather/dispatch seam; no input writes yet."""
    state, uniform = case.owner.gather_batch_req_state(None, dummy)
    rows = (0 if state is None else state.num_tokens) if tokens is None else tokens
    requests = rows if state is None else len(state.req_ids)
    descriptor, sync = native_runner.dispatch_cg_and_sync_dp(
        case.manager,
        requests,
        rows,
        uniform,
        dp_size=case.owner.parallel_config.data_parallel_size,
        dp_rank=case.owner.parallel_config.data_parallel_rank,
        need_eager=need_eager,
        parallel_config=case.owner.parallel_config,
        allow_ubatching=case.owner.ubatch_runner is not None,
        uniform_decode=uniform == case.owner.decode_query_len,
    )
    if sync is not None:
        raise AssertionError("DP1 must preserve native None synchronization")
    return descriptor


def prepare_runtime_full(case, descriptor):
    """Run actual selected native CPU U1/U2 preparation for a replay."""
    batch = make_ascend_decode_batch(
        descriptor.num_tokens, target_query_len=descriptor.uniform_token_count
    )
    batch.is_padding.fill_(False)
    slots = torch.zeros(1, descriptor.num_tokens, dtype=torch.int32)
    if descriptor.num_ubatches == 2:
        case.runner.prepare(batch, (), slots, CUDAGraphMode.FULL, False)
    else:
        case.owner.model_state.prepare_attn(
            batch,
            CUDAGraphMode.FULL,
            (),
            slots,
            case.owner.attn_groups,
            case.owner.kv_cache_config,
            for_capture=False,
        )
    return batch


def install_runtime_replay_boundary(case, *, fail=False):
    """Native ACL/device replay is explicitly excluded, with live metadata read."""

    def replay(manager, descriptor):
        if fail:
            raise RuntimeError("native replay failed")
        metadata = manager.model_runner.model_state.attn_metadata
        child = metadata["child"]
        if child.padded_tokens != descriptor.num_tokens // descriptor.num_ubatches:
            raise AssertionError("native lifecycle received stale parent metadata")
        case.runtime_metadata.append(metadata)
        return manager.hidden_states, manager.aux_hidden_states

    case.manager.run_fullgraph = MethodType(replay, case.manager)


def install_runtime_eager_tensor_boundary(case):
    """Reuse the shared public CPU tensor peer, keeping real control/cursors.

    The import is local because the pre-KV concept reuses this module's case
    builder. Only NPU tensor posting/receiving is replaced; control, native
    preparation, continuation and model wrapper dispatch remain real.
    """
    from tests.unit.fakes.hccl_v41_graph_eager import PreKVAttentionPeer

    template = PreKVAttentionPeer(case.transport.negotiated_capabilities)
    case.transport.tensor_peer = template.tensor_peer
    case.transport.send_layer_inputs = MethodType(
        PreKVAttentionPeer.send_layer_inputs, case.transport
    )
    case.transport.receive_layer_output = MethodType(
        PreKVAttentionPeer.receive_layer_output, case.transport
    )
    case.owner.model.register_forward_pre_hook(
        case.owner.model_state._prepare_engram_before_forward, with_kwargs=True
    )


def run_runtime_eager_parent(case, descriptor, *, token_offset, position_offset):
    """Actual U1/U2 preparation and eager wrapper with fresh CPU parent inputs.

    This is a component call, not GPUModelRunner.execute_model or sampling.
    Native metadata/device operators use the capture fixture's CPU leaves.
    """
    batch = make_ascend_decode_batch(
        descriptor.num_tokens, target_query_len=case.owner.decode_query_len
    )
    batch.is_dummy = False
    batch.is_padding.fill_(False)
    batch.is_prefilling_np.fill(False)
    batch.input_ids.add_(token_offset)
    batch.positions.add_(position_offset)
    slots = torch.zeros(1, descriptor.num_tokens, dtype=torch.int32)
    prepared = None
    metadata = None
    if descriptor.num_ubatches == 2:
        prepared = case.runner.prepare(batch, (), slots)
    else:
        metadata = case.owner.model_state.prepare_attn(
            batch,
            CUDAGraphMode.NONE,
            (),
            slots,
            case.owner.attn_groups,
            case.owner.kv_cache_config,
        )
    context = set_forward_context(
        metadata,
        case.owner.vllm_config,
        batch_descriptor=descriptor,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    inputs = dict(input_ids=batch.input_ids, positions=batch.positions)
    with context:
        case.owner.model_state.parent_context = get_forward_context()
        result = (
            case.owner.model(**inputs)
            if prepared is None
            else case.runner.run(case.owner.model, inputs, prepared)
        )
    return batch, result


class PaddingOnlyFFN(nn.Module):
    """Independent all-padding FFN oracle, without a model/device claim."""

    def forward(self, hidden_states, **kwargs):
        return torch.zeros_like(hidden_states)


def reference_runtime_eager_output(batch, *, embed_tokens=None, padding_only=False):
    """Unsplit CPU schedule; optional frozen CPU weights are borrowed read-only."""
    model = make_staged_model(capacity=batch.num_tokens_after_padding, engram=False)
    if embed_tokens is not None:
        model.embed_tokens = embed_tokens
    if padding_only:
        for layer in model.layers:
            layer.mlp = PaddingOnlyFFN()
    return model(
        input_ids=batch.input_ids,
        positions=batch.positions,
        intermediate_tensors=None,
        engram_lookups={},
        engram_mask=torch.empty(0, dtype=torch.bool),
    )


def set_runtime_rows(case, rows):
    case.native_state = None if not rows else make_batch_req_state((1,) * rows)


def replace_cached_bundle(case, entry, resources):
    """Use only the public cache API to publish a mismatched retained bundle."""
    cache = HcclGraphCache(DeviceCompletion().synchronize)
    cache.reserve_capture(entry.key, entry.generation, entry.graph, resources)
    cache.publish_capture(entry.key, entry.generation)
    case.control.cache = case.cache = cache


def replace_cached_pipeline(case, entry):
    """Replace the public bundle with fresh same-bank/policy mutable owners."""
    execution = copy(entry.resources.execution)
    execution.pipeline = copy(execution.pipeline)
    execution.parent_body = copy(execution.parent_body)
    execution.parent_body.stages = copy(execution.parent_body.stages)
    execution.parent_body.stages.pipeline = execution.pipeline
    replace_cached_bundle(case, entry, replace(entry.resources, execution=execution))
