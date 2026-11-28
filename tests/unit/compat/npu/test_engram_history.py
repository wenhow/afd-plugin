# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""PD history envelopes are CPU-only protocol records, not device history."""

import json

import pytest

from afd_plugin.compat.npu.engram_history import (
    engram_history_payload,
    validate_engram_history,
)


def test_history_round_trip_keeps_original_prompt_and_source_identity():
    tokens = [10, 20, 30, 40]
    payload = json.loads(json.dumps(engram_history_payload(tokens, "p-1", "engine-1")))

    validate_engram_history(payload, tokens, "p-1", "engine-1")

    assert tokens == [10, 20, 30, 40]
    assert payload["prompt_tokens"] == len(tokens)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("prompt_tokens", 3),
        ("prompt_sha256", "0" * 64),
        ("source_request_id", "other"),
        ("source_engine_id", "other"),
        ("unknown", "unexpected"),
    ],
)
def test_history_rejects_changed_or_foreign_prefix(field, value):
    tokens = [10, 20, 30, 40]
    payload = {**engram_history_payload(tokens, "p-1", "engine-1"), field: value}

    with pytest.raises(ValueError, match="does not match"):
        validate_engram_history(payload, tokens, "p-1", "engine-1")


@pytest.mark.parametrize("tokens", [[10, 20, 30], [10, 20, 30, 41], [20, 10, 30, 40]])
def test_history_rejects_truncated_or_reordered_original_prompt(tokens):
    payload = engram_history_payload([10, 20, 30, 40], "p-1", "engine-1")
    with pytest.raises(ValueError, match="does not match"):
        validate_engram_history(payload, tokens, "p-1", "engine-1")


@pytest.mark.parametrize("tokens", [[], [True], [-1], [1.5]])
def test_history_requires_original_text_token_ids(tokens):
    with pytest.raises(ValueError, match="original text"):
        engram_history_payload(tokens, "p", "engine")
