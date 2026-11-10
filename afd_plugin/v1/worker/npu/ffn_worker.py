# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU FFN-side worker for AFD execution."""

from __future__ import annotations

import logging
import os
import threading
from functools import partial
from typing import TYPE_CHECKING, cast

import torch
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.platforms import current_platform
from vllm.v1.worker.worker_base import CompilationTimes
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.cpu_binding import bind_cpus
from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.compat.npu import (
    apply_afd_ascend_patches_if_needed,
    fail_if_unsupported_npu_afd_features,
    fix_all2all_backend_for_afd,
    npu_afd_num_ubatches,
)
from afd_plugin.compat.npu.profiler import (
    AFDNPUProfilerSession,
    afd_npu_profiler_owner,
    step_afd_npu_profiler,
)
from afd_plugin.compat.npu.runtime_config import (
    DeepseekV41TargetShape,
    get_npu_ffn_config,
    resolve_deepseek_v41_target_shape,
)
from afd_plugin.config import AFD_ASYNC_CONNECTOR, parse_afd_config
from afd_plugin.connectors import AFDConnectorFactory
from afd_plugin.connectors.npu.async_cam import CAMAsyncAFDConnector
from afd_plugin.connectors.npu.hccl_config import validate_staged_hccl_binding
from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.connectors.npu.hccl_staged import HcclStagedAFDConnector
from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector
from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.npu.async_ffn_model_runner import AsyncNPUFFNModelRunner
from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner
from afd_plugin.v1.worker.npu.hccl_v41_graph_bootstrap import (
    create_hccl_v41_ffn_graph_service,
)
from afd_plugin.validation import (
    NPU_FFN_WORKER_FQCN,
    assert_compatible_afd_stack,
    validate_npu_model_runner_v2_config,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

    from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
from afd_plugin.v1.worker.npu.hccl_v41_diagnostics import HcclV41DiagnosticBinding
from afd_plugin.v1.worker.npu.hccl_v41_eager import HcclV41EagerSession

logger = logging.getLogger(__name__)

FFN_SHUTDOWN_TIMEOUT_SECONDS = 5


def create_npu_ffn_model_runner(
    vllm_config: VllmConfig,
    device: torch.device,
    *,
    target_shape: DeepseekV41TargetShape | None = None,
) -> AFDNPUFFNModelRunner | AsyncNPUFFNModelRunner:
    """Select the execution owner from the connector, with no MRV1 fallback.

    Plugin-owned target_shape comes from the worker's original V4.1 target,
    before role isolation. None preserves the existing non-V4.1 factory path.
    """
    validate_npu_model_runner_v2_config(
        vllm_config, expected_role="ffn", device_type=device.type
    )
    afd_config = parse_afd_config(vllm_config, expected_role="ffn")
    if afd_config.connector == AFD_ASYNC_CONNECTOR:
        connector = AFDConnectorFactory.create_connector(
            int(get_world_group().rank), int(device.index), vllm_config, afd_config
        )
        return AsyncNPUFFNModelRunner(
            vllm_config, device, cast(CAMAsyncAFDConnector, connector)
        )
    if target_shape is None:
        return AFDNPUFFNModelRunner(vllm_config, device)
    return AFDNPUFFNModelRunner(vllm_config, device, target_shape=target_shape)


class AFDNPUFFNWorker(NPUWorker):
    """FFN worker that owns a connector-driven NPU daemon loop."""

    afd_expected_role = "ffn"

    # Upstream: vllm_ascend/worker/worker.py, NPUWorker.__init__,
    # 7e37e29a7803d6897a7113088e2b771f83205bc0.
    # Reason: native Ascend derives Engram settings before binding worker config.
    # Functionality: retain immutable target scheduling shape, then detach F's
    # Engram/draft bindings before any parent code.
    # Signature matches upstream, including no return annotation; no added args.
    # Delegation exception: the large native constructor owns ops registration,
    # config derivation, logging and cache/sleep setup. Keep that lifecycle in
    # super(); this subclass adds only the role view and existing daemon state.
    # Remove the role view when upstream exposes an FFN-only Engram owner.
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        **kwargs,
    ):
        apply_afd_ascend_patches_if_needed()
        # ### PATCH START: A-owned Engram and draft
        self.target_shape = resolve_deepseek_v41_target_shape(vllm_config)
        vllm_config = get_npu_ffn_config(vllm_config)
        # ### PATCH END: A-owned Engram and draft
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
            **kwargs,
        )
        self._ffn_thread: threading.Thread | None = None
        self._ffn_shutdown_event: threading.Event | None = None
        self._ffn_loop_error: BaseException | None = None
        self._hccl_stop_received = False
        self._hccl_v41_transport: HcclStagedTransport | None = None
        self._cpu_binding_attempted = False

    def init_device(self) -> None:
        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDNPUFFNWorker.init_device",
            expected_role="ffn",
            expected_worker_qualname_override=NPU_FFN_WORKER_FQCN,
        )
        fail_if_unsupported_npu_afd_features(self.vllm_config)
        fix_all2all_backend_for_afd(self.vllm_config)
        validate_npu_model_runner_v2_config(
            self.vllm_config,
            expected_role="ffn",
            device_type="npu",
        )

        self.device = self._init_device()
        init_workspace_manager(
            self.device,
            npu_afd_num_ubatches(self.vllm_config),
        )
        self.vllm_config.model_config = get_afd_model_config(
            self.vllm_config.model_config,
            device_type="npu",
        )
        self.model_runner = create_npu_ffn_model_runner(
            self.vllm_config, self.device, target_shape=self.target_shape
        )
        if isinstance(self.model_runner.connector, HcclStagedAFDConnector):
            self.bind_hccl_v41_transport(self.model_runner.connector.transport)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {}

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.model_runner.initialize_kv_cache(kv_cache_config)
        if self._hccl_v41_transport is None:
            self.model_runner.initialize_afd_connector()
        elif not self._hccl_v41_transport.initialized:
            self._hccl_v41_transport.initialize()
        self.start_ffn_server_loop()

    def bind_hccl_v41_transport(self, transport: HcclStagedTransport) -> None:
        """Inject the v3 service before connector initialization or thread start.

        This explicit component API leaves the model admission guard intact.
        The worker owns the supplied transport until acknowledged peer STOP.
        """
        if (
            self._hccl_v41_transport is not None
            or self._ffn_thread is not None
            or self.model_runner.connector.is_initialized
        ):
            raise RuntimeError("bind staged HCCL before FFN connector/service startup")
        if not isinstance(self.model_runner, AFDNPUFFNModelRunner):
            raise TypeError("staged HCCL requires the synchronous NPU FFN runner")
        connector = self.model_runner.connector
        if (
            isinstance(connector, HcclStagedAFDConnector)
            and transport is not connector.transport
        ):
            raise ValueError("staged HCCL FFN binding must use its connector transport")
        if transport.role != "ffn":
            raise ValueError("staged HCCL FFN binding requires an FFN transport")
        validate_staged_hccl_binding(
            self.model_runner.vllm_config,
            transport,
            self.device,
            target_shape=self.target_shape,
        )
        self._hccl_v41_transport = transport

    def _uses_hccl_service(self) -> bool:
        return self._hccl_v41_transport is not None or isinstance(
            self.model_runner.connector, P2pHcclAFDConnector
        )

    def bind_hccl_v41_eager_session(self, session: HcclV41EagerSession) -> None:
        """Bind caller-warmed streams before starting the resident receiver."""
        if (
            self._ffn_thread is not None
            or session.transport is not self._hccl_v41_transport
            or not isinstance(self.model_runner, AFDNPUFFNModelRunner)
        ):
            raise RuntimeError("bind eager FFN streams before resident service startup")
        self.model_runner.bind_hccl_v41_eager_session(session)

    def bind_hccl_v41_diagnostics(self, binding: HcclV41DiagnosticBinding) -> None:
        """Install rank diagnostics before the service thread can receive."""
        if (
            self._ffn_thread is not None
            or binding.transport is not self._hccl_v41_transport
            or not isinstance(self.model_runner, AFDNPUFFNModelRunner)
        ):
            raise RuntimeError("bind FFN diagnostics before resident service startup")
        self.model_runner.bind_hccl_v41_diagnostics(binding)

    def compile_or_warm_up_model(self) -> CompilationTimes:
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        raise RuntimeError(
            "AFD NPU FFN workers are connector-driven; scheduler-driven "
            "execute_model() is not supported.",
        )

    def start_ffn_server_loop(self) -> None:
        self._validate_profiler_owner()
        profiler = self.model_runner.prof
        if isinstance(profiler, AFDNPUProfilerSession):
            status = profiler.status()
            if status["failed"] or status["closed"]:
                # A deferred profiler belongs to one serving-thread lifetime.
                # In particular, do not recreate a legacy connector after its
                # old loop error has been observed/cleared by management.
                raise RuntimeError(
                    "cannot restart a failed or closed AFD profiler owner"
                )
        if self._ffn_thread is not None and self._ffn_thread.is_alive():
            self.raise_ffn_loop_error_if_any()
            return

        self.raise_ffn_loop_error_if_any()
        connector = self.model_runner.connector
        staged = self._hccl_v41_transport
        if staged is not None:
            if self._hccl_stop_received:
                raise RuntimeError("cannot restart an acknowledged staged HCCL STOP")
            if staged.closed or staged.terminal:
                raise RuntimeError("cannot restart a terminal staged HCCL service")
            if not staged.initialized:
                staged.initialize()
            if (
                staged.negotiated_capabilities.graph_execution_version == 1
                and self.model_runner.hccl_v41_graph_service is None
            ):
                try:
                    service = create_hccl_v41_ffn_graph_service(
                        self.model_runner,
                        staged,
                        synchronize=partial(torch.npu.synchronize, self.device),
                        overlap_settings=HcclOverlapSettings.from_environment(
                            dict(os.environ),
                            num_stages=2
                            if self.vllm_config.parallel_config.use_ubatching
                            else 1,
                        ),
                    )
                    self.model_runner.bind_hccl_v41_graph_service(service)
                except BaseException:
                    # No STOP or PG destroy can be inferred from local setup
                    # failure; loaded model/transport remain supervisor-owned.
                    staged.terminal = True
                    raise
        elif not connector.is_initialized:
            self.model_runner.initialize_afd_connector()

        self._bind_cpus_once()
        self._ffn_shutdown_event = threading.Event()
        self._ffn_loop_error = None
        self._hccl_stop_received = False

        def ffn_worker_loop() -> None:
            try:
                with afd_npu_profiler_owner(self.model_runner.prof):
                    self._run_ffn_server_loop()
            except Exception as exc:
                shutdown_event = self._ffn_shutdown_event
                profiler = self.model_runner.prof
                profiler_failed = (
                    isinstance(profiler, AFDNPUProfilerSession)
                    and profiler.status()["failed"]
                )
                if (
                    not self._uses_hccl_service()
                    and not self._requires_async_drain()
                    and not profiler_failed
                    and shutdown_event is not None
                    and shutdown_event.is_set()
                ):
                    logger.debug(
                        "AFD NPU FFN receive loop stopped during shutdown",
                        exc_info=True,
                    )
                    return
                self._ffn_loop_error = exc
                logger.exception("AFD NPU FFN worker loop failed")
            else:
                if self._requires_async_drain():
                    assert isinstance(self.model_runner, AsyncNPUFFNModelRunner)
                    self.model_runner._graph_receiver_exited = True

        self._ffn_thread = threading.Thread(
            target=ffn_worker_loop,
            name="afd-npu-ffn-worker-loop",
            daemon=True,
        )
        self._ffn_thread.start()

    def _validate_profiler_owner(self) -> None:
        if (
            isinstance(self.model_runner.prof, AFDNPUProfilerSession)
            and self.profiler_config is not None
            and self.profiler_config.profiler is not None
        ):
            raise ValueError(
                "AFD_NPU_FFN_PROFILER_ENABLE and native --profiler-config "
                "cannot be enabled together"
            )

    # Upstream: NPUWorker.profile, vllm_ascend/worker/worker.py:1223,
    # 7e37e29a7803d6897a7113088e2b771f83205bc0. Signature matches exactly.
    # Reason: F executes in a daemon, while native profile RPC runs on the
    # management thread and controls a separate native profiler.
    # Functionality: explicit role-env sessions accept nonblocking requests;
    # the serving owner applies them at boundaries and reports pending state.
    # Delegation exception: without the role-env session, preserve the entire
    # native config/trace-name/lifecycle behavior through super(). Remove this
    # override when native Worker supports connector-driven profiler owners.
    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # ### PATCH START: F profiler execution owner
        self._validate_profiler_owner()
        session = self.model_runner.prof
        if isinstance(session, AFDNPUProfilerSession):
            self.raise_ffn_loop_error_if_any()
            session.request(is_start, profile_prefix)
            return
        # ### PATCH END: F profiler execution owner
        return super().profile(is_start, profile_prefix)

    def afd_profiler_status(self) -> dict[str, bool | int | str | None]:
        """Management RPC: pending requests do not assert trace collection."""
        session = self.model_runner.prof
        if isinstance(session, AFDNPUProfilerSession):
            return {"mode": "afd-role", **session.status()}
        return {
            "mode": "native",
            "configured": self.profiler_config is not None
            and self.profiler_config.profiler is not None,
            # Native control state alone does not prove that F's daemon was
            # recorded or that native iteration limits were advanced there.
            "control_running": self.profiler is not None and self.profiler.is_running,
        }

    def _bind_cpus_once(self) -> None:
        if self._cpu_binding_attempted:
            return
        self._cpu_binding_attempted = True

        if not get_ascend_config().enable_cpu_binding:
            return

        try:
            physical_npu_id = current_platform.device_id_to_physical_device_id(
                self.local_rank
            )
            bind_cpus(self.local_rank, npu_id=physical_npu_id)
        except Exception as exc:
            logger.warning(
                "Bind cpus failed in rank%s: %s Skip binding cpu.",
                self.local_rank,
                exc,
            )

    def _run_ffn_server_loop(self) -> None:
        event = self._ffn_shutdown_event
        if event is None:
            return

        torch.npu.set_device(self.device)
        while not event.is_set():
            session = self.model_runner.prof
            if isinstance(session, AFDNPUProfilerSession):
                session.before_step()
            if self._hccl_v41_transport is not None:
                assert isinstance(self.model_runner, AFDNPUFFNModelRunner)
                if not self.model_runner.execute_hccl_v41_step(
                    self._hccl_v41_transport
                ):
                    self._hccl_stop_received = True
                    return
                # COMPLETE already synchronizes this device before release.
                step_afd_npu_profiler(session)
                continue
            if isinstance(self.model_runner.connector, P2pHcclAFDConnector):
                plan = self.model_runner.connector.receive_forward_plan()
                if plan is None:
                    self._hccl_stop_received = True
                    return
                self.model_runner.execute_hccl_step(plan)
                # Observe deferred device failures in the serving thread; no
                # failed transaction may proceed to another FORWARD receive.
                torch.npu.synchronize()
                step_afd_npu_profiler(session)
                continue
            if isinstance(self.model_runner, AsyncNPUFFNModelRunner):
                if not self.model_runner.execute_connector_driven_step():
                    return
                if (
                    self.model_runner.effective_mode == "eager"
                    and self.model_runner._two_stream is None
                ):
                    torch.npu.synchronize()
                continue
            control_plane = self.model_runner.connector.control_plane
            assert control_plane is not None
            payload = control_plane.recv_dp_metadata_list()
            dp_metadata_list = payload.dp_metadata_list
            is_attn_graph_capturing = payload.is_graph_capturing
            is_warmup = payload.is_warmup
            is_profile = payload.is_profile
            is_graph_replaying = payload.is_graph_replaying

            self.model_runner.execute_ffn_step(
                dp_metadata_list=dp_metadata_list,
                is_graph_capturing=is_attn_graph_capturing,
                is_warmup=is_warmup,
                is_profile=is_profile,
                is_graph_replaying=is_graph_replaying,
            )
            torch.npu.synchronize()

    def _requires_async_drain(self) -> bool:
        return isinstance(self.model_runner, AsyncNPUFFNModelRunner) and (
            self.model_runner.effective_mode == "full"
            or self.model_runner._two_stream is not None
        )

    def raise_ffn_loop_error_if_any(self) -> None:
        error = self._ffn_loop_error
        if error is not None:
            if not self._requires_async_drain() and not self._uses_hccl_service():
                self._ffn_loop_error = None
            raise RuntimeError("AFD NPU FFN worker loop failed") from error

    def poll_hccl_ffn_service(self) -> bool:
        """RPC status: False only after peer STOP; failures remain terminal."""
        if not self._uses_hccl_service():
            raise TypeError("HCCL service polling requires P2pHcclAFDConnector")
        self.raise_ffn_loop_error_if_any()
        if self._hccl_stop_received:
            return False
        thread = self._ffn_thread
        if thread is None:
            raise RuntimeError("HCCL FFN service has not been started")
        if not thread.is_alive():
            raise RuntimeError("HCCL FFN service exited without peer STOP")
        return True

    def stop_ffn_server_loop(self) -> None:
        event = self._ffn_shutdown_event
        if self._uses_hccl_service():
            # A owns STOP. Join the receiver after its STOP_ACK; destroying a
            # group under a blocking receive is not a normal shutdown protocol.
            thread = self._ffn_thread
            if thread is not None:
                thread.join(timeout=FFN_SHUTDOWN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    raise RuntimeError(
                        "HCCL FFN is awaiting peer STOP or device completion; "
                        "retain resources and terminate the failed worker process"
                    )
            self.raise_ffn_loop_error_if_any()
            staged = self._hccl_v41_transport
            if staged is not None:
                if staged.initialized and not self._hccl_stop_received:
                    raise RuntimeError(
                        "staged HCCL service requires peer STOP; retain resources"
                    )
                if staged.negotiated_capabilities.graph_execution_version == 1:
                    self.model_runner.close_hccl_v41_graph_service(staged)
                staged.close()
            else:
                self.model_runner.connector.close()
            if event is not None:
                event.set()
            self._ffn_thread = None
            self._ffn_shutdown_event = None
            return
        if self._requires_async_drain():
            if event is not None:
                event.set()
            thread = self._ffn_thread
            if thread is not None:
                thread.join(timeout=FFN_SHUTDOWN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    raise RuntimeError(
                        "Async graph receive is pending; retain resources"
                    )
            self.raise_ffn_loop_error_if_any()
            self.model_runner.shutdown()
            self._ffn_thread = None
            self._ffn_shutdown_event = None
            return
        if event is not None:
            event.set()

        # CAM recv blocks in the connector operator. Release the communicator
        # first so the daemon can observe the shutdown event, then wait for it
        # before the parent runner releases model tensors.
        self.model_runner.connector.close()
        thread = self._ffn_thread
        if thread is not None:
            thread.join(timeout=FFN_SHUTDOWN_TIMEOUT_SECONDS)
            if thread.is_alive():
                raise RuntimeError(
                    "AFD NPU FFN worker loop did not stop after connector close",
                )
        self._ffn_thread = None
        self._ffn_shutdown_event = None
        self.raise_ffn_loop_error_if_any()

    def shutdown(self) -> None:
        # Stop the connector-driven daemon before NPUWorker releases the model
        # runner and its tensors.
        self.stop_ffn_server_loop()
        super().shutdown()


__all__ = ["AFDNPUFFNWorker"]
