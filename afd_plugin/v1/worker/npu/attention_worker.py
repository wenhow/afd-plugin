# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""NPU Attention-side worker for AFD execution."""

from __future__ import annotations

from typing import Any

from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.worker.worker import NPUWorker

from afd_plugin.compat.npu import (
    apply_afd_ascend_patches_if_needed,
    fail_if_unsupported_npu_afd_features,
    fix_all2all_backend_for_afd,
    npu_afd_num_ubatches,
)
from afd_plugin.compat.npu.runtime_config import configure_async_cam_dbo
from afd_plugin.config import uses_window_dspark
from afd_plugin.model_executor.models.model_utils import get_afd_model_config
from afd_plugin.v1.worker.npu.attention_model_runner_v2 import (
    AFDNPUAttentionModelRunnerV2,
)
from afd_plugin.v1.worker.npu.window_dspark import configure_window_dspark_draft
from afd_plugin.validation import (
    NPU_ATTENTION_WORKER_FQCN,
    assert_compatible_afd_stack,
    validate_npu_model_runner_v2_config,
)


class AFDNPUAttentionWorker(NPUWorker):
    """Attention worker that creates an AFD-aware vLLM-Ascend runner."""

    afd_expected_role = "attention"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        apply_afd_ascend_patches_if_needed()
        super().__init__(*args, **kwargs)

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
        window_dspark = uses_window_dspark(self.vllm_config)
        # MRV2 selects lane 1 during draft loading, capture and proposal.
        # Each lane has its own per-ubatch workspace retained by its graph.
        init_workspace_manager(
            self.device,
            npu_afd_num_ubatches(self.vllm_config),
            num_lanes=2 if window_dspark else 1,
        )
        self.vllm_config.model_config = get_afd_model_config(
            self.vllm_config.model_config,
            device_type="npu",
        )
        # MRV2 selects and loads the draft separately from the target model.
        spec = self.vllm_config.speculative_config
        if spec is not None and window_dspark:
            draft = spec.draft_model_config
            configure_window_dspark_draft(self.vllm_config.model_config, draft)
            spec.update_arch_()
        self.model_runner = AFDNPUAttentionModelRunnerV2(
            self.vllm_config,
            self.device,
        )


__all__ = ["AFDNPUAttentionWorker"]
