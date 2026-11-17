# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real native config-method outcomes; no EngineArgs or device execution."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from vllm import envs
from vllm.config import VllmConfig

from afd_plugin.compat.npu import runtime
from afd_plugin.compat.patches.npu import ubatching_config
from tests.unit.fakes.vllm_config import (
    native_dbo_config_snapshot,
    native_v41_dbo_config,
)

NATIVE_SPECULATION = "dual batch overlap with speculative decoding"
NATIVE_MULTIMODAL = "dual batch overlap with multimodal models"
NATIVE_BASE_RESTRICTIONS = [NATIVE_SPECULATION, NATIVE_MULTIMODAL]


@pytest.fixture
def native_config_patch(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", True)
    monkeypatch.setattr(
        ubatching_config, "current_platform", SimpleNamespace(device_type="npu")
    )
    # Register restoration before the public facade installs the real hook.
    monkeypatch.setattr(
        VllmConfig,
        "_get_dbo_unsupported_features",
        VllmConfig._get_dbo_unsupported_features,
    )
    runtime.apply_afd_ascend_config_patch_if_needed()


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("target_graph", [False, True])
def test_exact_v41_dspark_u2_retains_native_config_fields(
    native_config_patch, role, target_graph
):
    config = native_v41_dbo_config(role=role, target_graph=target_graph)
    before = native_dbo_config_snapshot(config)

    assert config._get_dbo_unsupported_features() == []
    assert native_dbo_config_snapshot(config) == before


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("target_graph", [False, True])
def test_text_only_v41_u2_retains_native_config_fields(
    native_config_patch, role, target_graph
):
    config = native_v41_dbo_config(role=role, target_graph=target_graph, dspark=False)
    before = native_dbo_config_snapshot(config)

    assert config.speculative_config is None
    assert config._get_dbo_unsupported_features() == []
    assert native_dbo_config_snapshot(config) == before


@pytest.mark.parametrize("raw_ubatch_size", [0, 1, 2, 3])
def test_text_only_native_dbo_uses_effective_two_stages(
    native_config_patch, raw_ubatch_size
):
    config = native_v41_dbo_config(dspark=False)
    config.parallel_config.enable_dbo = True
    config.parallel_config.ubatch_size = raw_ubatch_size
    before = native_dbo_config_snapshot(config)

    assert config.parallel_config.num_ubatches == 2
    assert config._get_dbo_unsupported_features() == []
    assert native_dbo_config_snapshot(config) == before


@pytest.mark.parametrize(
    "boundary",
    [
        "no_afd",
        "cuda",
        "cam_p2p",
        "cam_async",
        "old_v4",
        "u1",
        "u3",
        "float_stages",
        "engram_config",
        "engram_layers",
        "vision",
        "no_mm_config",
        "compile",
        "graph_partition",
        "inconsistent_target_graph",
    ],
)
def test_text_only_other_scopes_keep_native_multimodal_restriction(
    native_config_patch, monkeypatch, boundary
):
    config = native_v41_dbo_config(dspark=False, boundary=boundary)
    if boundary == "cuda":
        monkeypatch.setattr(
            ubatching_config, "current_platform", SimpleNamespace(device_type="cuda")
        )
    before = native_dbo_config_snapshot(config)

    assert config._get_dbo_unsupported_features() == [NATIVE_MULTIMODAL]
    assert native_dbo_config_snapshot(config) == before


@pytest.mark.parametrize(
    "boundary,expected",
    [
        ("lora", ["dual batch overlap with LoRA", NATIVE_MULTIMODAL]),
        (
            "pp",
            ["dual batch overlap with pipeline parallelism", NATIVE_MULTIMODAL],
        ),
        (
            "dcp",
            ["dual batch overlap with context parallelism", NATIVE_MULTIMODAL],
        ),
        (
            "pcp",
            ["dual batch overlap with context parallelism", NATIVE_MULTIMODAL],
        ),
        ("encoder_decoder", [NATIVE_MULTIMODAL]),
        ("hybrid", [NATIVE_MULTIMODAL, "dual batch overlap with hybrid models"]),
        (
            "piecewise",
            [NATIVE_MULTIMODAL, "dual batch overlap with PIECEWISE CUDA graphs"],
        ),
        (
            "encoder_only",
            [NATIVE_MULTIMODAL, "dual batch overlap with encoder only models"],
        ),
    ],
)
def test_text_only_native_unsupported_features_retain_order_and_messages(
    native_config_patch, boundary, expected
):
    config = native_v41_dbo_config(dspark=False, boundary=boundary)
    before = native_dbo_config_snapshot(config)

    assert config._get_dbo_unsupported_features() == expected
    assert native_dbo_config_snapshot(config) == before


@pytest.mark.parametrize(
    "explicit_v2,expected",
    [(None, ["dual batch overlap"]), (False, [NATIVE_MULTIMODAL])],
)
def test_text_only_implicit_or_disabled_v2_retains_native_restrictions(
    native_config_patch, monkeypatch, explicit_v2, expected
):
    config = native_v41_dbo_config(dspark=False)
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", explicit_v2)
    before = native_dbo_config_snapshot(config)

    assert config._get_dbo_unsupported_features() == expected
    assert native_dbo_config_snapshot(config) == before


@pytest.mark.parametrize("raw_ubatch_size", [0, 1, 2, 3])
def test_native_dbo_effective_two_stages_admits_raw_stage_values(
    native_config_patch, raw_ubatch_size
):
    config = native_v41_dbo_config()
    config.parallel_config.enable_dbo = True
    config.parallel_config.ubatch_size = raw_ubatch_size
    before = native_dbo_config_snapshot(config)

    assert config.parallel_config.num_ubatches == 2
    assert config._get_dbo_unsupported_features() == []
    assert native_dbo_config_snapshot(config) == before


@pytest.mark.parametrize(
    "boundary",
    [
        "no_afd",
        "cuda",
        "cam_p2p",
        "cam_async",
        "old_v4",
        "u1",
        "u3",
        "float_stages",
        "other_method",
        "adaptive",
        "draft_graph",
        "engram_config",
        "engram_layers",
        "vision",
        "no_mm_config",
        "compile",
        "graph_partition",
        "inconsistent_target_graph",
    ],
)
def test_other_scopes_keep_native_restrictions(
    native_config_patch, monkeypatch, boundary
):
    config = native_v41_dbo_config(boundary=boundary)
    if boundary == "cuda":
        monkeypatch.setattr(
            ubatching_config, "current_platform", SimpleNamespace(device_type="cuda")
        )
    before = native_dbo_config_snapshot(config)

    assert config._get_dbo_unsupported_features() == NATIVE_BASE_RESTRICTIONS
    assert native_dbo_config_snapshot(config) == before


@pytest.mark.parametrize(
    "boundary,expected",
    [
        ("lora", ["dual batch overlap with LoRA", *NATIVE_BASE_RESTRICTIONS]),
        (
            "pp",
            [
                NATIVE_SPECULATION,
                "dual batch overlap with pipeline parallelism",
                NATIVE_MULTIMODAL,
            ],
        ),
        (
            "dcp",
            [
                NATIVE_SPECULATION,
                "dual batch overlap with context parallelism",
                NATIVE_MULTIMODAL,
            ],
        ),
        (
            "pcp",
            [
                NATIVE_SPECULATION,
                "dual batch overlap with context parallelism",
                NATIVE_MULTIMODAL,
            ],
        ),
        ("encoder_decoder", NATIVE_BASE_RESTRICTIONS),
        (
            "hybrid",
            [*NATIVE_BASE_RESTRICTIONS, "dual batch overlap with hybrid models"],
        ),
        (
            "piecewise",
            [
                *NATIVE_BASE_RESTRICTIONS,
                "dual batch overlap with PIECEWISE CUDA graphs",
            ],
        ),
        (
            "encoder_only",
            [*NATIVE_BASE_RESTRICTIONS, "dual batch overlap with encoder only models"],
        ),
    ],
)
def test_native_unsupported_features_retain_order_and_messages(
    native_config_patch, boundary, expected
):
    config = native_v41_dbo_config(boundary=boundary)
    before = native_dbo_config_snapshot(config)

    assert config._get_dbo_unsupported_features() == expected
    assert native_dbo_config_snapshot(config) == before


@pytest.mark.parametrize(
    "explicit_v2,expected",
    [(None, ["dual batch overlap"]), (False, NATIVE_BASE_RESTRICTIONS)],
)
def test_implicit_or_disabled_v2_retains_native_restrictions(
    native_config_patch, monkeypatch, explicit_v2, expected
):
    config = native_v41_dbo_config()
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", explicit_v2)
    before = native_dbo_config_snapshot(config)

    assert config._get_dbo_unsupported_features() == expected
    assert native_dbo_config_snapshot(config) == before
