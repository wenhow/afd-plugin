# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Scoped native MRV2 initialization for plugin-owned V4.1 cache zeroing."""

from vllm.v1.worker.utils import KVBlockZeroer
from vllm_ascend.worker.v2.model_runner import NPUModelRunner

from afd_plugin.compat.npu.runtime_config import (
    HCCL_ATTESTED_PD_MODULE,
    is_deepseek_v41_model,
)
from afd_plugin.config import parse_optional_afd_config


# Upstream: vllm/v1/worker/gpu/model_runner.py:763-771 @ ced6857a;
# Ascend NPUModelRunner inherits this method at 7e37e29a.
# Patch reason: native V4.1 cache carriers wrap Tensor/tuple components in a
# list, so the generic zeroer silently skips their scheduled fresh blocks.
# Patch functionality: expose physical tile metadata only for V4.1 AFD or this
# plugin's exact attested PD connector; all other configs use the copied body.
# Signature: matches upstream exactly; no added parameters.
# Cost: metadata views/groups are built once at KV initialization. No storage
# copy, per-step wrapper, synchronization or extra device kernel is added.
# Removal: remove when native MRV2 zeroing accepts V4.1 bound cache tiles.
def init_kv_zero_meta(self) -> None:
    # ### PATCH START: V4.1 plugin cache tiles
    transfer = self.vllm_config.kv_transfer_config
    if is_deepseek_v41_model(self.vllm_config) and (
        parse_optional_afd_config(self.vllm_config, validate=False) is not None
        or (
            transfer is not None
            and transfer.kv_connector == "MooncakeHybridConnector"
            and transfer.kv_connector_module_path == HCCL_ATTESTED_PD_MODULE
        )
    ):
        # Worker-local loading avoids importing V4.1 device code for unrelated
        # native models during their otherwise unchanged initialization.
        from afd_plugin.compat.npu.kv_zeroing import DeepseekV41KVBlockZeroer

        self.kv_block_zeroer = DeepseekV41KVBlockZeroer(self)
        return
    # ### PATCH END: V4.1 plugin cache tiles
    self.kv_block_zeroer = KVBlockZeroer(
        self.device,
        attn_groups_iter=(g for groups in self.attn_groups for g in groups),
        kernel_block_sizes=self.kernel_block_sizes,
        static_forward_context=self.compilation_config.static_forward_context,
        num_blocks=self.kv_cache_config.num_blocks,
    )


def apply_dsv41_kv_zeroing_patch() -> None:
    """Install on the NPU subclass; the shared GPU base remains unchanged."""
    NPUModelRunner._init_kv_zero_meta = init_kv_zero_meta
