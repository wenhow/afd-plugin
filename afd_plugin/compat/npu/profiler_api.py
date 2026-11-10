# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Opt-in native endpoint plugin for execution-owned AFD NPU profilers."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictBool, StrictStr
from starlette.datastructures import State

from afd_plugin.compat.npu.profiler import afd_npu_profiler_config
from afd_plugin.config import AFDRole, parse_afd_config

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient
    from vllm.tasks import SupportedTask

PROFILER_RPC_TIMEOUT_SECONDS = 30.0


class _ProfilerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_start: StrictBool
    profile_prefix: StrictStr | None = None


@dataclass(frozen=True)
class _ProfilerApiBinding:
    engine_client: EngineClient
    role: AFDRole


async def _collective_rpc(
    binding: _ProfilerApiBinding,
    method: Literal["profile", "afd_profiler_status"],
    *,
    kwargs: dict[str, bool | str | None] | None = None,
) -> list[dict[str, bool | int | str | None] | None]:
    try:
        return await binding.engine_client.collective_rpc(
            method, timeout=PROFILER_RPC_TIMEOUT_SECONDS, kwargs=kwargs
        )
    except TimeoutError as error:
        # A timed-out request may still be applied by the execution owner.
        # A later status read, rather than an HTTP success, resolves its state.
        raise HTTPException(
            status_code=504, detail="AFD profiler RPC timed out"
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=502, detail="AFD profiler RPC failed"
        ) from error


class AFDNPUProfilerEndpointPlugin:
    """Implement the pinned native EndpointPlugin protocol without a patch.

    The endpoint entry is independent of the existing ``afd`` general plugin.
    All backend start/step/stop/export remains on the worker's execution owner.
    This object stores no engine binding: each FastAPI app owns its own binding.
    """

    name = "afd_npu_profiler"
    required_tasks: tuple[SupportedTask, ...] = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        router = APIRouter(prefix="/afd/profiler", tags=["AFD profiler"])

        @router.get("")
        async def profiler_status(request: Request) -> JSONResponse:
            binding = cast(
                _ProfilerApiBinding, request.app.state.afd_npu_profiler_binding
            )
            workers = await _collective_rpc(binding, "afd_profiler_status")
            return JSONResponse({"role": binding.role, "workers": workers})

        @router.post("")
        async def request_profiler(
            body: _ProfilerRequest, request: Request
        ) -> JSONResponse:
            binding = cast(
                _ProfilerApiBinding, request.app.state.afd_npu_profiler_binding
            )
            await _collective_rpc(
                binding,
                "profile",
                kwargs={
                    "is_start": body.is_start,
                    "profile_prefix": body.profile_prefix,
                },
            )
            # An RPC reply acknowledges desired state. It does not establish
            # that an idle/blocked owner applied it or exported an NPU trace.
            return JSONResponse({"accepted": True, "role": binding.role})

        app.include_router(router)

    async def init_state(
        self, engine_client: EngineClient | None, state: State, args: Namespace
    ) -> None:
        # Native init_app_state publishes these exact objects before this hook.
        # Validate here: the native loader logs/skips factory exceptions, while
        # initializer errors fail startup before the application serves HTTP.
        if engine_client is None:
            raise ValueError("AFD profiler endpoint requires an existing engine client")
        config = state.vllm_config
        if (
            state.engine_client is not engine_client
            or config is not engine_client.vllm_config
        ):
            raise ValueError("AFD profiler endpoint requires this app's engine client")
        if config.device_config.device_type != "npu":
            raise ValueError("AFD profiler endpoint requires an NPU AFD configuration")
        afd_config = parse_afd_config(config)
        profiler_config = afd_npu_profiler_config(afd_config.role)
        if not profiler_config.enabled:
            raise ValueError(
                "AFD profiler endpoint requires the role's profiler "
                "environment to be enabled"
            )
        if (
            config.profiler_config is not None
            and config.profiler_config.profiler is not None
        ):
            raise ValueError(
                "AFD role profiler and native --profiler-config "
                "cannot be enabled together"
            )
        state.afd_npu_profiler_binding = _ProfilerApiBinding(
            engine_client, afd_config.role
        )


__all__ = ["AFDNPUProfilerEndpointPlugin"]
