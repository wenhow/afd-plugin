# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Explicit local profiler management for an existing headless HCCL F core.

The EngineCore management thread polls this bounded Unix socket. Requests only
reach its existing executor; profiler work stays on each worker's execution
owner. No listener, device import or thread is created unless configured.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from afd_plugin.compat.npu.profiler import afd_npu_profiler_config
from afd_plugin.config import HCCL_P2P_CONNECTOR, parse_afd_config

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor import Executor

CONTROL_PATH_ENV = "AFD_NPU_FFN_PROFILER_CONTROL_PATH"
MAX_CONTROL_CLIENTS = 4
MAX_REQUESTS_PER_POLL = 1
MAX_REQUEST_BYTES = 4_096
MAX_RESPONSE_BYTES = 65_536
CLIENT_IDLE_SECONDS = 5.0
RPC_TIMEOUT_SECONDS = 30.0
CLIENT_TIMEOUT_SECONDS = RPC_TIMEOUT_SECONDS + CLIENT_IDLE_SECONDS
SOCKET_PERMISSIONS = 0o600
ControlOperation = Literal["status", "start", "stop"]
JsonValue: TypeAlias = (
    bool | int | float | str | None | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass
class _Connection:
    socket: socket.socket
    deadline: float
    incoming: bytearray = field(default_factory=bytearray)
    outgoing: bytes | None = None
    sent: int = 0


class FfnProfilerControl:
    """One owner-thread listener; one request and reply per connection."""

    def __init__(
        self,
        path: Path,
        executor: Executor,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not path.is_absolute():
            raise ValueError("FFN profiler control socket path must be absolute")
        self.path, self.executor, self.clock = path, executor, clock
        self._clients: list[_Connection] = []
        self._identity: tuple[int, int] | None = None
        self._closed = False
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            # bind refuses existing files, sockets and dangling symlinks. Never
            # remove another process's endpoint in order to start this one.
            self._listener.bind(str(path))
            stat = path.lstat()
            self._identity = stat.st_dev, stat.st_ino
            path.chmod(SOCKET_PERMISSIONS)
            self._listener.setblocking(False)
            self._listener.listen(MAX_CONTROL_CLIENTS)
        except BaseException:
            with suppress(OSError):
                self.close()
            raise

    def _reply(self, client: _Connection, value: dict[str, JsonValue]) -> None:
        try:
            encoded = (json.dumps(value, allow_nan=False) + "\n").encode()
        except (TypeError, ValueError, RecursionError):
            encoded = b'{"ok":false,"error":"invalid_rpc_response"}\n'
        if len(encoded) > MAX_RESPONSE_BYTES:
            encoded = b'{"ok":false,"error":"response_too_large"}\n'
        client.outgoing = encoded
        client.deadline = self.clock() + CLIENT_IDLE_SECONDS

    def _handle(self, client: _Connection, raw: bytes) -> None:
        try:
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("request must be a JSON object")
            operation = value.get("operation")
            if operation == "status" and set(value) == {"operation"}:
                method, kwargs = "afd_profiler_status", None
            elif operation in ("start", "stop") and set(value) <= {
                "operation",
                "profile_prefix",
            }:
                prefix = value.get("profile_prefix")
                if prefix is not None and not isinstance(prefix, str):
                    raise ValueError("profile_prefix must be a string or null")
                method = "profile"
                kwargs = {"is_start": operation == "start", "profile_prefix": prefix}
            else:
                raise ValueError("unsupported profiler command or fields")
        except (ValueError, UnicodeError, RecursionError):
            self._reply(client, {"ok": False, "error": "invalid_request"})
            return
        try:
            workers = self.executor.collective_rpc(
                method, timeout=RPC_TIMEOUT_SECONDS, kwargs=kwargs
            )
        except TimeoutError:
            self._reply(client, {"ok": False, "error": "rpc_timeout"})
        except Exception:
            self._reply(client, {"ok": False, "error": "rpc_failed"})
        else:
            self._reply(
                client,
                {"ok": True, "role": "ffn", "workers": workers}
                if operation == "status"
                else {"ok": True, "role": "ffn", "accepted": True},
            )

    def poll(self) -> None:
        """Poll bounded client IO; a fixed executor RPC has its own timeout.

        The idle deadline bounds incomplete client IO, not the synchronous
        executor call. Worker inference remains on its existing daemon thread.
        """
        if self._closed:
            return
        for _ in range(MAX_CONTROL_CLIENTS - len(self._clients)):
            try:
                connection, _ = self._listener.accept()
            except BlockingIOError:
                break
            connection.setblocking(False)
            self._clients.append(
                _Connection(connection, self.clock() + CLIENT_IDLE_SECONDS)
            )
        request_budget = MAX_REQUESTS_PER_POLL
        for client in tuple(self._clients):
            try:
                if self.clock() >= client.deadline:
                    self._drop(client)
                    continue
                if client.outgoing is None:
                    if b"\n" not in client.incoming:
                        chunk = client.socket.recv(MAX_REQUEST_BYTES + 1)
                        if not chunk:
                            self._drop(client)
                            continue
                        client.incoming.extend(chunk)
                    if len(client.incoming) > MAX_REQUEST_BYTES:
                        self._reply(client, {"ok": False, "error": "request_too_large"})
                    elif b"\n" in client.incoming and request_budget:
                        request_budget -= 1
                        line, trailing = bytes(client.incoming).split(b"\n", 1)
                        if trailing.strip():
                            self._reply(
                                client, {"ok": False, "error": "invalid_request"}
                            )
                        else:
                            self._handle(client, line)
                if client.outgoing is not None:
                    client.sent += client.socket.send(client.outgoing[client.sent :])
                    if client.sent == len(client.outgoing):
                        self._drop(client)
            except BlockingIOError:
                continue
            except (ConnectionError, OSError):
                self._drop(client)

    def _drop(self, client: _Connection) -> None:
        self._clients.remove(client)
        with suppress(OSError):
            client.socket.close()

    def close(self) -> None:
        """Close local IO; unlink only the pathname this instance bound."""
        self._closed = True
        for client in tuple(self._clients):
            self._drop(client)
        cleanup_error = None
        try:
            self._listener.close()
        except OSError as error:
            cleanup_error = error
        if self._identity is not None:
            try:
                identity = self.path.lstat()
                if (
                    stat.S_ISSOCK(identity.st_mode)
                    and (identity.st_dev, identity.st_ino) == self._identity
                ):
                    self.path.unlink()
                self._identity = None
            except FileNotFoundError:
                self._identity = None
            except OSError as error:
                if cleanup_error is None:
                    cleanup_error = error
        if cleanup_error is not None:
            raise cleanup_error


def create_ffn_profiler_control(
    config: VllmConfig,
    executor: Executor,
    *,
    environment: Mapping[str, str] | None = None,
) -> FfnProfilerControl | None:
    """Reject an explicit incompatible endpoint before starting F workers."""
    env = os.environ if environment is None else environment
    raw = env.get(CONTROL_PATH_ENV)
    if raw is None:
        return None
    afd = parse_afd_config(config, expected_role="ffn")
    if config.device_config.device_type != "npu" or afd.connector != HCCL_P2P_CONNECTOR:
        raise ValueError("FFN profiler control requires NPU HCCL")
    if not afd_npu_profiler_config("ffn").enabled:
        raise ValueError("FFN profiler control requires the role profiler enabled")
    if (
        config.profiler_config is not None
        and config.profiler_config.profiler is not None
    ):
        raise ValueError("FFN profiler control conflicts with native profiler config")
    return FfnProfilerControl(Path(raw), executor)


def request_ffn_profiler(
    path: Path, operation: ControlOperation, *, profile_prefix: str | None = None
) -> dict[str, JsonValue]:
    """Send one fixed command to a running core's explicitly configured socket."""
    request: dict[str, JsonValue] = {"operation": operation}
    if profile_prefix is not None:
        request["profile_prefix"] = profile_prefix
    payload = (json.dumps(request) + "\n").encode()
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("profiler control request exceeds its byte limit")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(CLIENT_TIMEOUT_SECONDS)
        connection.connect(str(path))
        connection.sendall(payload)
        reply = bytearray()
        while b"\n" not in reply:
            chunk = connection.recv(MAX_RESPONSE_BYTES + 1)
            if not chunk:
                raise ConnectionError("profiler control closed before a reply")
            reply.extend(chunk)
            if len(reply) > MAX_RESPONSE_BYTES:
                raise ValueError("profiler control response exceeds its byte limit")
    value = json.loads(reply)
    if not isinstance(value, dict) or type(value.get("ok")) is not bool:
        raise ValueError("invalid profiler control response")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("operation", choices=("status", "start", "stop"))
    parser.add_argument("--profile-prefix")
    args = parser.parse_args(argv)
    response = request_ffn_profiler(
        args.socket, args.operation, profile_prefix=args.profile_prefix
    )
    print(json.dumps(response, indent=2))
    return 0 if response["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
