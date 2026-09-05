# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Loopback-only API middleware for explicit AFD profiler control."""

from __future__ import annotations

import ipaddress
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

AFD_PROFILE_START_PATH = "/afd/profile/start"
AFD_PROFILE_STOP_PATH = "/afd/profile/stop"
AFD_PROFILE_FINALIZE_PATH = "/afd/profile/finalize"


async def afd_profile_control_middleware(
    request: Any,
    call_next: Any,
) -> Any:
    """Start or stop plugin profilers without stopping serving processes."""

    from starlette.responses import Response

    path = request.url.path.rstrip("/")
    profile_paths = {
        AFD_PROFILE_START_PATH,
        AFD_PROFILE_STOP_PATH,
        AFD_PROFILE_FINALIZE_PATH,
    }
    if request.method != "POST" or path not in profile_paths:
        return await call_next(request)

    client = request.client
    try:
        is_loopback = (
            client is not None and ipaddress.ip_address(client.host).is_loopback
        )
    except ValueError:
        is_loopback = False
    if not is_loopback:
        return Response(status_code=403)

    engine_client: Any = getattr(request.app.state, "engine_client", None)
    if engine_client is None:
        return Response(status_code=503)
    action = "start" if path == AFD_PROFILE_START_PATH else "stop"
    started_at = time.monotonic()
    logger.warning(
        "AFD NPU profiler API %s request started: client=%s, path=%s",
        action,
        client.host,
        path,
    )
    try:
        if action == "start":
            await engine_client.start_profile()
        else:
            await engine_client.stop_profile()
    except Exception:
        logger.exception(
            "AFD NPU profiler API %s request failed after %.3fs",
            action,
            time.monotonic() - started_at,
        )
        raise
    logger.warning(
        "AFD NPU profiler API %s request completed in %.3fs",
        action,
        time.monotonic() - started_at,
    )
    return Response(status_code=200)


__all__ = [
    "AFD_PROFILE_FINALIZE_PATH",
    "AFD_PROFILE_START_PATH",
    "AFD_PROFILE_STOP_PATH",
    "afd_profile_control_middleware",
]
