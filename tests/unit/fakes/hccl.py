# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A CPU queue wire for connector seam tests; it simulates no device behavior."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.forward_context import get_forward_context

from afd_plugin.config import HCCL_P2P_CONNECTOR, AFDConfig
from afd_plugin.connectors.npu import p2p_hccl as connector_module
from afd_plugin.connectors.npu.hccl_protocol import (
    HcclCapabilities,
    HcclForwardPlan,
    HcclRole,
)
from afd_plugin.connectors.npu.hccl_transport import HcclLayerInputs
from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector
from afd_plugin.distributed.hccl_plan import HcclParentPlan
from tests.unit.fakes.vllm_config import hccl_vllm_config

ROUTER_DELTA_SCALE = 4_096
WORKER_JOIN_TIMEOUT_SECONDS = 3
STAGED_FFN_HIDDEN_SIZE = 16


class StagedModelPeer:
    """CPU protocol seam with controlled arithmetic, no simulated device ops."""

    def __init__(self, plan, *, role_rank=0, fail_step=None):
        self.plan = plan
        self.role = "attention"
        self.role_rank = role_rank
        self.terminal = False
        self.completed = False
        self.messages = []
        self.pending = None
        self.fail_step = fail_step

    def send_layer_inputs(self, layer_idx, stage_idx, hidden, fp32, ids):
        if self.pending is not None:
            raise AssertionError("previous reply must be consumed")
        if get_forward_context().ubatch_idx != stage_idx:
            raise AssertionError("payload must use the matching native child context")
        self.messages.append(
            (layer_idx, stage_idx, hidden.clone(), fp32.clone(), ids.clone())
        )
        self.pending = self.messages[-1]

    def receive_layer_output(self, layer_idx, stage_idx):
        if self.fail_step == (layer_idx, stage_idx):
            raise RuntimeError("injected remote failure")
        layer, stage, hidden, fp32, ids = self.pending
        assert (layer, stage) == (layer_idx, stage_idx)
        self.pending = None
        return ((fp32 - hidden.float()) * ROUTER_DELTA_SCALE + ids[:, None]).to(
            hidden.dtype
        )

    def finish_forward(self):
        assert self.pending is None
        self.completed = True


def make_hccl_connector_pair(
    monkeypatch, *, layer_count=2, tensor_parallel_size=1, role_rank=0, wire=None
):
    """Construct actual adapters with a CPU protocol wire and no process groups."""
    wire = wire or HcclLoopbackSession()
    monkeypatch.setattr(connector_module, "HcclEagerTransport", wire.endpoint)
    connectors = []
    for role in ("attention", "ffn"):
        config = hccl_vllm_config(
            dtype=torch.bfloat16,
            compilation_mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
            role=role,
            tensor_parallel_size=tensor_parallel_size,
        )
        config.model_config.hf_config.num_hidden_layers = layer_count
        connector = P2pHcclAFDConnector(
            role_rank,
            role_rank,
            config,
            AFDConfig(
                connector=HCCL_P2P_CONNECTOR,
                role=role,
                num_attention_ranks=tensor_parallel_size,
                num_ffn_ranks=tensor_parallel_size,
            ),
            role_rank,
        )
        connector.init_afd_connector()
        connectors.append(connector)
    return tuple(connectors)


def hccl_layer_tensors():
    hidden = torch.arange(48, dtype=torch.float32).reshape(3, 16).to(torch.bfloat16)
    fp32 = hidden.float() + 0.00031
    ids = torch.tensor([17, 9, 0], dtype=torch.int32)
    return hidden, fp32, ids


def make_hccl_runner(runner_type, connector, model):
    runner = object.__new__(runner_type)
    runner.connector = connector
    runner.vllm_config = connector.vllm_config
    runner.model = model
    return runner


def make_hccl_worker(worker_type, runner):
    """Bypass native worker/device construction to exercise its serving lifecycle."""
    worker = object.__new__(worker_type)
    worker.model_runner = runner
    worker.vllm_config = runner.vllm_config
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker._ffn_thread = None
    worker._ffn_shutdown_event = None
    worker._ffn_loop_error = None
    worker._hccl_stop_received = False
    worker._cpu_binding_attempted = True
    return worker


def join_hccl_worker(worker):
    """Bound test deadlocks without sleeping or asserting execution duration."""
    worker._ffn_thread.join(timeout=WORKER_JOIN_TIMEOUT_SECONDS)
    if worker._ffn_thread.is_alive():
        raise TimeoutError("CPU HCCL worker test did not finish")


class HcclCpuForwardContext:
    def __init__(self):
        self.current = None

    @contextmanager
    def activate(self, **kwargs):
        previous = self.current
        layer_count = kwargs["vllm_config"].model_config.hf_config.num_hidden_layers
        self.current = SimpleNamespace(
            afd_metadata=kwargs["afd_metadata"],
            input_ids=kwargs["input_ids"],
            num_tokens=kwargs["num_tokens"],
            num_tokens_across_dp=kwargs["num_tokens_across_dp"],
            model_instance=kwargs["model_instance"],
            in_profile_run=kwargs["in_profile_run"],
            all_moe_layers=tuple(f"model.layers.{i}.mlp" for i in range(layer_count)),
            moe_layer_index=None,
        )
        try:
            yield self.current
        finally:
            self.current = previous


class HcclContractModel(torch.nn.Module):
    """An observable model boundary; this arithmetic simulates no native MoE."""

    def __init__(self, context, plan):
        super().__init__()
        self.context = context
        self.plan = plan

    def compute_ffn_output(
        self, hidden_states, layer_idx, *, input_ids, hidden_states_fp32
    ):
        context = self.context.current
        if context is None:
            raise RuntimeError("FFN compute requires a current execution context")
        metadata = context.afd_metadata
        if (
            metadata.transaction_id != self.plan.transaction_id
            or metadata.tokens_lens != [self.plan.padded_tokens]
            or metadata.tokens_unpadded_lens != [self.plan.valid_tokens]
            or metadata.stage_idx != 0
            or metadata.num_stages != 1
            or context.num_tokens != self.plan.padded_tokens
            or context.in_profile_run != (self.plan.mode == "profile")
            or context.moe_layer_index != layer_idx
            or not torch.equal(context.input_ids, input_ids)
        ):
            raise ValueError("FFN execution context does not match the accepted plan")
        return (
            (hidden_states_fp32 - hidden_states.float()) * ROUTER_DELTA_SCALE
            + input_ids[:, None].float()
            + layer_idx
        ).to(torch.bfloat16)


class HcclFailingModel(torch.nn.Module):
    def compute_ffn_output(self, hidden_states, layer_idx, **kwargs):
        raise ValueError("controlled FFN compute failure")


class StagedFFNPeer:
    """Accepted FFN protocol seam; no groups or device operations are created."""

    def __init__(self, plan, *, role_rank=0, fail_at=None):
        self.plan = plan
        self.role = "ffn"
        self.role_rank = role_rank
        self.initialized = True
        self.closed = self.terminal = self.completed = False
        self.fail_at = fail_at
        self.messages = []
        self.inputs = {}
        self.pending = None
        self.steps = deque(
            (layer, stage.stage_idx)
            for layer in plan.layer_indices
            for stage in plan.stages
            if any(stage.attention_padded_tokens)
        )
        for layer, stage_idx in self.steps:
            peers = plan.stages[stage_idx].for_ffn(role_rank)
            rows = sum(peer.padded_tokens for peer in peers)
            hidden = torch.arange(
                rows * STAGED_FFN_HIDDEN_SIZE, dtype=torch.float32
            ).reshape(rows, STAGED_FFN_HIDDEN_SIZE)
            hidden = (hidden + layer + stage_idx).to(torch.bfloat16)
            fp32 = hidden.float() + (stage_idx + 1) * 0.00031
            ids = torch.zeros(rows, dtype=torch.int32)
            for peer in peers:
                start, stop = peer.ffn_offset, peer.ffn_offset + peer.valid_tokens
                ids[start:stop] = torch.arange(stop - start) + 1
            self.inputs[layer, stage_idx] = HcclLayerInputs(hidden, fp32, ids)

    def receive_layer(self, layer_idx, stage_idx):
        if self.fail_at == "receive":
            raise RuntimeError("injected staged receive failure")
        if self.pending is not None or self.steps[0] != (layer_idx, stage_idx):
            raise AssertionError("FFN must receive in layer-major stage order")
        self.messages.append(("RECEIVE", layer_idx, stage_idx))
        self.pending = (layer_idx, stage_idx)
        return self.inputs[self.pending]

    def send_layer_output(self, layer_idx, stage_idx, output):
        if self.fail_at == "send":
            raise RuntimeError("injected staged send failure")
        if self.pending != (layer_idx, stage_idx):
            raise AssertionError("FFN output must match its received aggregate")
        expected_shape = self.inputs[self.pending].hidden_states.shape
        if output.shape != expected_shape or output.dtype != torch.bfloat16:
            raise AssertionError("FFN must return its complete native output")
        self.messages.append(("OUTPUT", layer_idx, stage_idx, output.clone()))
        self.steps.popleft()
        self.pending = None

    def finish_forward(self):
        if self.fail_at == "complete":
            raise RuntimeError("injected staged COMPLETE failure")
        if self.steps or self.pending is not None:
            raise AssertionError("parent COMPLETE must follow every output")
        self.messages.append(("COMPLETE", self.plan.transaction_id))
        self.completed = True
        self.plan = None


class StagedFFNContractModel(torch.nn.Module):
    """Check native compute context while exposing raw FP32/ID propagation."""

    def __init__(self, context, plan, role_rank, *, fail=False):
        super().__init__()
        self.context, self.plan, self.role_rank = context, plan, role_rank
        self.fail = fail

    def compute_ffn_output(
        self,
        hidden_states,
        layer_idx,
        *,
        input_ids,
        hidden_states_fp32,
        already_sequence_parallel=False,
    ):
        if self.fail:
            raise ValueError("controlled staged FFN compute failure")
        context = self.context.current
        if context is None:
            raise RuntimeError("staged FFN requires a current execution context")
        metadata = context.afd_metadata
        peers = self.plan.stages[metadata.stage_idx].for_ffn(self.role_rank)
        rows = sum(peer.padded_tokens for peer in peers)
        if (
            metadata.transaction_id != self.plan.transaction_id
            or metadata.num_stages != len(self.plan.stages)
            or metadata.tokens_lens != [rows]
            or metadata.tokens_unpadded_lens
            != [sum(peer.valid_tokens for peer in peers)]
            or context.num_tokens != rows
            or context.num_tokens_across_dp.tolist() != [rows]
            or context.in_profile_run != (self.plan.mode == "profile")
            or context.moe_layer_index != layer_idx
            or context.model_instance is not self
            or not torch.equal(context.input_ids, input_ids)
            or already_sequence_parallel
        ):
            raise ValueError("staged FFN context does not match the accepted plan")
        return staged_ffn_expected_output(
            HcclLayerInputs(hidden_states, hidden_states_fp32, input_ids), layer_idx
        )


def staged_ffn_expected_output(payload, layer_idx):
    return (
        (payload.hidden_states_fp32 - payload.hidden_states.float())
        * ROUTER_DELTA_SCALE
        + payload.input_ids[:, None].float()
        + layer_idx
    ).to(torch.bfloat16)


def make_staged_ffn_case(runner_type, plan: HcclParentPlan, *, role_rank=0):
    """Build an already-accepted parent without worker/device initialization."""
    peer = StagedFFNPeer(plan, role_rank=role_rank)
    config = hccl_vllm_config(
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        role="ffn",
        tensor_parallel_size=plan.stages[0].native_tp_size,
    )
    config.parallel_config.rank = role_rank
    config.model_config.hf_config.num_hidden_layers = max(plan.layer_indices) + 1
    connector = SimpleNamespace(vllm_config=config)
    context = HcclCpuForwardContext()
    model = StagedFFNContractModel(context, plan, role_rank)
    runner = make_hccl_runner(runner_type, connector, model)
    return SimpleNamespace(runner=runner, peer=peer, context=context, plan=plan)


class HcclCpuDeviceRuntime:
    """Worker device hooks only; no device execution is represented."""

    @staticmethod
    def set_device(device):
        pass

    @staticmethod
    def synchronize():
        pass


class PendingHcclWorkerThread:
    def join(self, timeout=None):
        pass

    def is_alive(self):
        return True


class ExitedHcclWorkerThread(PendingHcclWorkerThread):
    def is_alive(self):
        return False


def unexpected_hccl_forward_context(**kwargs):
    raise AssertionError("empty HCCL plans must not create a forward context")


class HcclLoopbackSession:
    def __init__(self) -> None:
        self.plans: deque[HcclForwardPlan] = deque()
        self.inputs: deque[HcclLayerInputs] = deque()
        self.outputs: deque[torch.Tensor] = deque()
        self.stopped = False

    def endpoint(
        self,
        role: HcclRole,
        capabilities: HcclCapabilities,
        *,
        init_method: str,
        local_device: int,
        timeout_seconds: int,
    ) -> HcclLoopbackTransport:
        return HcclLoopbackTransport(self, capabilities)


class HcclEndpointRegistry:
    """Route CPU wire payloads by the peer's actual rendezvous endpoint."""

    def __init__(self):
        self.sessions = {}

    def endpoint(
        self, role, capabilities, *, init_method, local_device, timeout_seconds
    ):
        session = self.sessions.setdefault(init_method, HcclLoopbackSession())
        return session.endpoint(
            role,
            capabilities,
            init_method=init_method,
            local_device=local_device,
            timeout_seconds=timeout_seconds,
        )


class HcclLoopbackTransport:
    def __init__(
        self, session: HcclLoopbackSession, capabilities: HcclCapabilities
    ) -> None:
        self.session = session
        self.negotiated_capabilities = capabilities

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        pass

    def begin_forward(self, plan: HcclForwardPlan) -> None:
        plan.validate(self.negotiated_capabilities)
        self.session.plans.append(plan)

    def receive_forward(self) -> HcclForwardPlan | None:
        if self.session.stopped:
            return None
        return self.session.plans.popleft()

    def send_layer_inputs(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        hidden_states_fp32: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> None:
        self.session.inputs.append(
            HcclLayerInputs(
                hidden_states.clone(), hidden_states_fp32.clone(), input_ids.clone()
            )
        )

    def receive_layer(self, layer_idx: int) -> HcclLayerInputs:
        return self.session.inputs.popleft()

    def send_layer_output(self, layer_idx: int, output: torch.Tensor) -> None:
        self.session.outputs.append(output.clone())

    def receive_layer_output(self, layer_idx: int) -> torch.Tensor:
        return self.session.outputs.popleft()

    def stop(self) -> None:
        self.session.stopped = True
