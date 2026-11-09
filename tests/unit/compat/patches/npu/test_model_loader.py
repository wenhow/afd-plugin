# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Observe construction/load outcomes through the real native loader API."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")
pytest.importorskip("torch_npu")

from vllm.model_executor.model_loader import get_model_loader  # noqa: E402
from vllm.model_executor.model_loader import utils as native_utils  # noqa: E402
from vllm.model_executor.model_loader.base_loader import BaseModelLoader  # noqa: E402
from vllm.model_executor.model_loader.default_loader import (  # noqa: E402
    DefaultModelLoader,
)

from afd_plugin.compat.patches.npu import model_loader as adapter  # noqa: E402
from afd_plugin.model_executor.models.npu import deepseek_v41  # noqa: E402
from tests.unit.fakes.model_loader import (  # noqa: E402
    AttentionModel,
    FFNModel,
    NativeModel,
    dspark_draft_model_config,
    loader_config,
)


@pytest.fixture
def loader_env(monkeypatch):
    monkeypatch.setattr(
        native_utils, "get_model_architecture", lambda config: (NativeModel, "native")
    )
    monkeypatch.setattr(deepseek_v41, "AFDDeepseekV41FFNForCausalLM", FFNModel)
    monkeypatch.setattr(
        deepseek_v41, "AFDDeepseekV41AttentionForCausalLM", AttentionModel
    )
    with patch.object(BaseModelLoader, "create_model", BaseModelLoader.create_model):
        adapter.apply_afd_model_loader_patch()
        yield loader_config()


@pytest.mark.parametrize("load_format", ["auto", "safetensors", "dummy"])
@pytest.mark.parametrize(
    "role,model_class", [("attention", AttentionModel), ("ffn", FFNModel)]
)
def test_supported_loader_constructs_role_without_rewriting_hf_identity(
    loader_env, load_format, role, model_class
):
    loader_env.load_config.load_format = load_format
    loader_env.additional_config["afd"]["role"] = role
    source = loader_env.model_config
    loader = get_model_loader(loader_env.load_config)
    model = loader.create_model(loader_env, source, prefix="role")
    assert type(model) is model_class
    assert model.prefix == "role"
    assert model.vllm_config is loader_env
    assert model.vllm_config.model_config is source
    assert loader_env.model_config is source
    assert source.hf_config.architectures == ["DeepseekV41ForCausalLM"]


@pytest.mark.parametrize(
    "role,model_class", [("attention", AttentionModel), ("ffn", FFNModel)]
)
def test_dummy_loader_retains_native_loading_and_eval(loader_env, role, model_class):
    loader_env.additional_config["afd"]["role"] = role
    loader = get_model_loader(loader_env.load_config)
    model = loader.load_model(loader_env, loader_env.model_config)
    assert type(model) is model_class
    assert model.vllm_config.model_config is loader_env.model_config
    assert loader_env.model_config.hf_config.architectures == ["DeepseekV41ForCausalLM"]
    assert not model.training
    assert model.weight.device.type == "cpu"
    assert torch.isfinite(model.weight).all()


@pytest.mark.parametrize(
    "model_type,device,role",
    [
        ("deepseek_v41", "npu", None),
        ("deepseek_v41", "cuda", "ffn"),
        ("deepseek_v4", "npu", "ffn"),
        ("deepseek_v2", "npu", "attention"),
    ],
)
def test_unaffected_paths_retain_native_model(loader_env, model_type, device, role):
    config = loader_config(model_type=model_type, device=device, role=role)
    # Other model families keep their own native architecture declaration.
    if model_type != "deepseek_v41":
        config.model_config.hf_config.architectures = ["ExistingForCausalLM"]
    loader = get_model_loader(config.load_config)
    model = loader.load_model(config, config.model_config)
    assert type(model) is NativeModel
    assert not model.training
    assert torch.isfinite(model.weight).all()


def test_invalid_role_fails_before_model_construction(loader_env):
    loader_env.additional_config["afd"]["role"] = "invalid"
    loader = get_model_loader(loader_env.load_config)
    with pytest.raises(RuntimeError, match="requires attention or ffn role"):
        loader.create_model(loader_env, loader_env.model_config)


def test_other_connector_is_rejected(loader_env):
    loader_env.additional_config["afd"]["connector"] = "CAMP2pAFDConnector"
    loader = get_model_loader(loader_env.load_config)
    with pytest.raises(ValueError, match="requires P2pHcclAFDConnector"):
        loader.create_model(loader_env, loader_env.model_config)


@pytest.mark.parametrize("load_format", ["auto", "safetensors", "dummy"])
def test_dspark_uses_native_draft_despite_the_ambient_afd_target(
    loader_env, load_format
):
    loader_env.additional_config["afd"]["role"] = "attention"
    loader_env.load_config.load_format = load_format
    target = loader_env.model_config
    draft = dspark_draft_model_config(target)
    loader = get_model_loader(loader_env.load_config)

    model = loader.create_model(loader_env, draft, prefix="dspark_head")

    assert type(model) is NativeModel
    assert model.prefix == "dspark_head"
    assert model.vllm_config.model_config is target
    assert target.hf_config.architectures == ["DeepseekV41ForCausalLM"]
    assert draft.hf_config.architectures == ["DeepseekV41DSparkModel"]


@pytest.mark.parametrize("role", ["attention", "ffn"])
def test_generic_fp8_cannot_silently_select_fp4_expert_loading(loader_env, role):
    loader_env.additional_config["afd"]["role"] = role
    loader_env.quant_config = SimpleNamespace(get_name=lambda: "fp8")
    loader = get_model_loader(loader_env.load_config)
    with pytest.raises(ValueError, match="--quantization deepseek_v4_fp8"):
        loader.create_model(loader_env, loader_env.model_config)


@pytest.mark.parametrize("load_format", ["pt", "npcache"])
def test_unqualified_load_format_is_rejected(loader_env, load_format):
    loader_env.load_config.load_format = load_format
    loader = get_model_loader(loader_env.load_config)
    with pytest.raises(ValueError, match="requires native default"):
        loader.create_model(loader_env, loader_env.model_config)


def test_unqualified_loader_subclass_is_rejected(loader_env):
    class UnqualifiedLoader(DefaultModelLoader):
        pass

    loader = UnqualifiedLoader(loader_env.load_config)
    with pytest.raises(ValueError, match="requires native default"):
        loader.create_model(loader_env, loader_env.model_config)
