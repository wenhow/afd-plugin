# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bind slotless Engram PD history to the original text request.

The native V2 request state reconstructs lookback from the complete original
prompt and the accepted computed-token position. No lookup buffers or F-owned
history cross the KV transport. This envelope verifies those original tokens
before D admits a remote prefix; it is not authentication of a remote peer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

ENGRAM_HISTORY_PARAM = "afd_engram_history_v1"
ENGRAM_HISTORY_VERSION = 1


def engram_history_payload(
    token_ids: Sequence[int], source_request_id: str, source_engine_id: str
) -> dict[str, int | str]:
    if not token_ids or any(type(token) is not int or token < 0 for token in token_ids):
        raise ValueError("Engram PD requires complete original text prompt IDs")
    if not source_request_id or not source_engine_id:
        raise ValueError("Engram PD requires source request and engine identity")
    return {
        "version": ENGRAM_HISTORY_VERSION,
        "prompt_tokens": len(token_ids),
        "prompt_sha256": hashlib.sha256(
            json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "source_request_id": source_request_id,
        "source_engine_id": source_engine_id,
    }


def validate_engram_history(
    payload: Mapping[str, int | str],
    token_ids: Sequence[int],
    source_request_id: str,
    source_engine_id: str,
) -> None:
    expected = engram_history_payload(token_ids, source_request_id, source_engine_id)
    if (
        not isinstance(payload, Mapping)
        or payload.keys() != expected.keys()
        or any(
            type(payload[name]) is not type(value) or payload[name] != value
            for name, value in expected.items()
        )
    ):
        raise ValueError("Engram PD source history does not match the original prompt")
