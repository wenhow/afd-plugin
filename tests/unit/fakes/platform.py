# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""A CUDA-like vLLM platform for device-selection tests on hosts without GPUs."""

from __future__ import annotations

import pytest


def stub_platform_forward_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep GPU CPU seams on native context APIs without device/TP setup."""
    # Deferred: shared fakes must remain importable without the vLLM runtime.
    from vllm import forward_context
    from vllm.platforms.interface import Platform

    monkeypatch.setattr(
        forward_context.current_platform,
        "set_additional_forward_context",
        Platform.set_additional_forward_context,
    )


def cuda_like_platform(monkeypatch: pytest.MonkeyPatch) -> type:
    """A platform that maps device ids like vLLM's CUDA platform, starting with
    no assigned physical ids and no ``CUDA_VISIBLE_DEVICES``."""
    # Deferred: this fakes module must import on CPU-only hosts without vLLM.
    from vllm.platforms import interface

    class CudaLikePlatform(interface.Platform):
        device_control_env_var = "CUDA_VISIBLE_DEVICES"

    # vLLM keeps the assigned ids process-wide; start every case unset.
    monkeypatch.setattr(interface, "_assigned_physical_gpu_ids", None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return CudaLikePlatform
