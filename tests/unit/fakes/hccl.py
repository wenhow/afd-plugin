# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A CPU queue wire for connector seam tests; it simulates no device behavior."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from vllm.config.compilation import CompilationMode, CUDAGraphMode

from afd_plugin.config import HCCL_P2P_CONNECTOR, AFDConfig
from afd_plugin.connectors.npu import p2p_hccl as connector_module
from afd_plugin.connectors.npu.hccl_protocol import (
    HcclCapabilities,
    HcclForwardPlan,
    HcclRole,
)
from afd_plugin.connectors.npu.hccl_transport import HcclLayerInputs
from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector
from tests.unit.fakes.vllm_config import hccl_vllm_config

ROUTER_DELTA_SCALE = 4_096
WORKER_JOIN_TIMEOUT_SECONDS = 3


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
