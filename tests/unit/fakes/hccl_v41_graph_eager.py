# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pre-KV host protocol and CPU model seams; no NPU/HCCL emulation."""

import torch
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.forward_context import override_forward_context

from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.model_executor.models import forward_context as context_module
from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
    AFDNPUAttentionModelRunnerV2,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph_eager import HcclV41GraphEager
from afd_plugin.v1.worker.npu.hccl_v41_graph_session import HcclV41GraphSession
from tests.unit.fakes.batch_req_state import make_batch_req_state
from tests.unit.fakes.hccl import StagedModelPeer
from tests.unit.fakes.hccl_diagnostics import StageDiagnosticRecorder
from tests.unit.fakes.hccl_v41_graph_control import AttentionGraphControlPeer
from tests.unit.fakes.hccl_v41_graph_coordinator import (
    dispatch_runtime,
    make_coordinator_case,
)
from tests.unit.fakes.input_batch import make_ascend_decode_batch
from tests.unit.fakes.vllm_config import hccl_vllm_config


class PreKVAttentionPeer(AttentionGraphControlPeer):
    """Real control codec/cursor with a public CPU tensor-operator boundary."""

    def __init__(self, capabilities):
        super().__init__(capabilities=capabilities, states={0: "missing"})
        self.tensor_peer = StagedModelPeer(None)

    def send_layer_inputs(self, layer_idx, stage_idx, hidden, fp32, ids):
        self._require_active("attention")
        self._require_step(layer_idx, stage_idx)
        self.tensor_peer.send_layer_inputs(layer_idx, stage_idx, hidden, fp32, ids)

    def receive_layer_output(self, layer_idx, stage_idx):
        self._require_active("attention")
        self._require_step(layer_idx, stage_idx)
        output = self.tensor_peer.receive_layer_output(layer_idx, stage_idx)
        self._advance()
        return output


def make_graph_eager_case(*, stages=1, scheduled=(1, 1, 1, 1), stage_diagnostics=False):
    """Reuse native/model/control builders, then select pre-KV owner fields."""
    case = make_coordinator_case(stages=stages)
    peer = PreKVAttentionPeer(case.transport.negotiated_capabilities)
    config = hccl_vllm_config(
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )
    config.model_config.enforce_eager = False
    config.model_config.hf_text_config.hidden_size = 4
    config.model_config.hf_text_config.vocab_size = 128
    config.parallel_config.use_ubatching = stages == 2
    config.parallel_config.num_ubatches = stages
    config.parallel_config.rank = 0
    config.compilation_config.static_forward_context = {}
    owner = object.__new__(AFDNPUAttentionModelRunnerV2)
    owner.__dict__.update(vars(case.owner))
    owner.vllm_config, owner.parallel_config = config, config.parallel_config
    owner.cudagraph_manager, owner.hccl_v41_transport = None, peer
    owner.hccl_v41_graph_eager = owner.hccl_v41_graph_session = None
    owner.hccl_v41_graph_coordinator = None
    owner.prof = None
    case.diagnostics = StageDiagnosticRecorder()
    case.overlap_settings = (
        HcclOverlapSettings(stage_diagnostics=True, ffn_compute_sync_diagnostics=True)
        if stage_diagnostics
        else None
    )
    case.owner, case.manager, case.transport = owner, None, peer
    case.control.manager, case.control.transport = None, peer
    case.session = HcclV41GraphSession(peer)
    case.runner.vllm_config = config
    case.runner.parallel_config = config.parallel_config
    case.runner.transport, case.runner.graph_owner = peer, case.session
    case.native_state = make_batch_req_state(scheduled)
    owner.gather_batch_req_state = lambda output, dummy: (
        None if dummy else case.native_state,
        1 if not dummy and all(count == 1 for count in scheduled) else None,
    )
    owner.model.register_forward_pre_hook(
        owner.model_state._prepare_engram_before_forward, with_kwargs=True
    )
    case.eager = HcclV41GraphEager(
        owner,
        case.control,
        next_transaction_id=case.next_transaction_id,
        overlap_settings=case.overlap_settings,
        diagnostic_sink=case.diagnostics.emit if stage_diagnostics else None,
    )
    owner.hccl_v41_graph_eager = case.eager
    return case


def run_graph_eager_parent(case, *, dummy=False, tokens=None):
    """Selected native CPU slicing/context/model seam, excluding native execute."""
    descriptor = dispatch_runtime(case, dummy=dummy, tokens=tokens)
    batch = make_ascend_decode_batch(descriptor.num_tokens)
    prepared = None
    if descriptor.num_ubatches == 2:
        prepared = case.owner.ubatch_runner.prepare(
            batch, (), torch.zeros(1, descriptor.num_tokens, dtype=torch.int32)
        )
    state = case.owner.model_state
    state.consumed = False
    context = context_module.forward_context_module.create_forward_context(
        None,
        case.owner.vllm_config,
        batch_descriptor=descriptor,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    state.parent_context = context
    inputs = dict(
        input_ids=batch.input_ids,
        positions=batch.positions,
        intermediate_tensors=None,
        inputs_embeds=None,
    )
    with override_forward_context(context):
        output = (
            case.owner.model(**inputs)
            if prepared is None
            else case.owner.ubatch_runner.run(case.owner.model, inputs, prepared)
        )
    return descriptor, output
