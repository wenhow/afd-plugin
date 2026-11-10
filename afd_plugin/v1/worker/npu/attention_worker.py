# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU Attention-side worker for AFD execution."""

from __future__ import annotations

from vllm.config import VllmConfig
from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.compat.npu import (
    apply_afd_ascend_patches_if_needed,
    fail_if_unsupported_npu_afd_features,
    fix_all2all_backend_for_afd,
    npu_afd_num_ubatches,
)
from afd_plugin.compat.npu.profiler import AFDNPUProfilerSession
from afd_plugin.compat.npu.runtime_config import configure_async_cam_dbo
from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
    AFDNPUAttentionModelRunnerV2,
)
from afd_plugin.validation import (
    NPU_ATTENTION_WORKER_FQCN,
    assert_compatible_afd_stack,
    validate_npu_model_runner_v2_config,
)


class AFDNPUAttentionWorker(NPUWorker):
    """Attention worker that creates an AFD-aware vLLM-Ascend runner."""

    afd_expected_role = "attention"

    # Upstream: vllm_ascend/worker/worker.py, NPUWorker.__init__, 7e37e29a.
    # Reason: install AFD config/loader hooks before native worker derivation.
    # Functionality: preserve the native config, including Attention's Engram.
    # Signature: matches upstream; no added parameters. Delegate the large
    # hardware/ops/profiler constructor to retain its native state ownership.
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
        **kwargs,
    ):
        # ### PATCH START: AFD runtime hooks
        apply_afd_ascend_patches_if_needed()
        # ### PATCH END: AFD runtime hooks
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
            **kwargs,
        )

    def init_device(self) -> None:
        assert_compatible_afd_stack(
            self.vllm_config,
            caller="AFDNPUAttentionWorker.init_device",
            expected_role="attention",
            expected_worker_qualname_override=NPU_ATTENTION_WORKER_FQCN,
        )
        configure_async_cam_dbo(self.vllm_config)
        fail_if_unsupported_npu_afd_features(self.vllm_config)
        fix_all2all_backend_for_afd(self.vllm_config)
        validate_npu_model_runner_v2_config(
            self.vllm_config,
            expected_role="attention",
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
        self.model_runner = AFDNPUAttentionModelRunnerV2(
            self.vllm_config,
            self.device,
        )

    # Upstream: NPUWorker.profile, vllm_ascend/worker/worker.py:1223,
    # 7e37e29a7803d6897a7113088e2b771f83205bc0. Exact native signature.
    # Reason: native RPC controls a different profiler from the AFD role-env
    # runner. Functionality: queue requests for its execute thread; native RPC
    # remains delegated in full when the role-env session is disabled.
    # Delegation exception: preserve native trace naming/configuration rather
    # than copying that unrelated implementation. Remove when native Worker
    # supports a runner-owned profiler session and status callback.
    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        # ### PATCH START: Attention profiler execution owner
        session = self.model_runner.prof
        if isinstance(session, AFDNPUProfilerSession):
            if (
                self.profiler_config is not None
                and self.profiler_config.profiler is not None
            ):
                raise ValueError(
                    "AFD_NPU_ATTENTION_PROFILER_ENABLE and native --profiler-config "
                    "cannot be enabled together"
                )
            session.request(is_start, profile_prefix)
            return
        # ### PATCH END: Attention profiler execution owner
        return super().profile(is_start, profile_prefix)

    def afd_profiler_status(self) -> dict[str, bool | int | str | None]:
        """RPC status; execute-step counts include legal None and KV-only calls."""
        session = self.model_runner.prof
        if isinstance(session, AFDNPUProfilerSession):
            return {"mode": "afd-role", "step_unit": "execute", **session.status()}
        return {
            "mode": "native",
            "configured": self.profiler_config is not None
            and self.profiler_config.profiler is not None,
            "control_running": self.profiler is not None and self.profiler.is_running,
        }

    # Upstream: NPUWorker.shutdown, vllm_ascend/worker/worker.py:365,
    # 7e37e29a7803d6897a7113088e2b771f83205bc0. Exact native signature.
    # Reason: native Worker releases its KV/weight-transfer owners before
    # runner.shutdown, which is too early if role-env export fails.
    # Functionality: deferred A sessions first complete the runner's idempotent
    # STOP/cache/export preparation, then preserve all native cleanup through
    # delegation. Disabled role-env keeps the original native order unchanged.
    # Delegation exception/removal: replace with a native pre-release callback
    # when Worker exposes one; no native KV or weight cleanup is copied here.
    def shutdown(self) -> None:
        # ### PATCH START: export before native resource release
        # Native worker_main also shuts down partially constructed workers.
        if (
            "model_runner" in self.__dict__
            and self.model_runner is not None
            and isinstance(self.model_runner.prof, AFDNPUProfilerSession)
        ):
            self.model_runner.prepare_shutdown()
        # ### PATCH END: export before native resource release
        super().shutdown()


__all__ = ["AFDNPUAttentionWorker"]
