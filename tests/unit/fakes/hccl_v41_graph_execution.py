# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real graph control and native CPU slicing at the parent-owner boundary."""

from types import SimpleNamespace

from torch import nn
from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext

from afd_plugin.distributed.hccl_plan import HcclPlanCapabilities
from afd_plugin.model_executor.models.npu.deepseek_v41 import (
    AFDDeepseekV41AttentionForCausalLM,
)
from afd_plugin.v1.worker.npu.hccl_v41_graph_execution import HcclV41GraphExecution
from afd_plugin.v1.worker.npu.hccl_v41_ubatch_runner import HcclV41UBatchRunner
from tests.unit.fakes.deepseek_v41 import ParentEngramState
from tests.unit.fakes.deepseek_v41_graph import make_graph_model_case
from tests.unit.fakes.hccl_v41_graph_control import make_attention_graph_control_case
from tests.unit.fakes.input_batch import make_ascend_decode_batch, make_hccl_ubatch_seam


def make_graph_execution_case(*, warmup=True, engram=False, pipeline_factory=None):
    model_case = make_graph_model_case((2, 2), engram=engram)
    capabilities = HcclPlanCapabilities(
        4,
        128,
        4,
        (0, 1),
        1,
        1,
        1,
        local_counts_version=1,
        graph_execution_version=1,
    )
    case = make_attention_graph_control_case(
        cached=False,
        peer_states={0: "missing"},
        capabilities=capabilities,
    )
    reservation = case.control.reserve_capture(
        case.descriptor, warmup=warmup, transaction_id="owner-capture"
    )
    wrapper = AFDDeepseekV41AttentionForCausalLM.__new__(
        AFDDeepseekV41AttentionForCausalLM
    )
    nn.Module.__init__(wrapper)
    wrapper.language_model = nn.Module()
    wrapper.language_model.model = model_case.model
    state = ParentEngramState({})
    runner = make_hccl_ubatch_seam(
        HcclV41UBatchRunner, model_state=state, max_num_reqs=4
    )
    case.transport.data_groups = [peer.group for peer in model_case.endpoints]
    execution = HcclV41GraphExecution(
        runner.vllm_config,
        wrapper,
        state,
        case.control,
        reservation,
        model_case.bank,
        pipeline=(
            pipeline_factory(model_case.bank) if pipeline_factory is not None else None
        ),
    )
    runner.graph_owner, runner.transport = execution, case.transport
    batch = make_ascend_decode_batch(4)
    case.batch = batch
    # Keep actual native CPU batch tensors as the parent's input storage.
    model_case.graph_inputs.update(input_ids=batch.input_ids, positions=batch.positions)
    return SimpleNamespace(
        **vars(case),
        model_case=model_case,
        runner=runner,
        execution=execution,
        reservation=reservation,
        inputs=model_case.graph_inputs,
        model=ModelWithContext(wrapper, device_metadata=state.device_metadata),
    )
