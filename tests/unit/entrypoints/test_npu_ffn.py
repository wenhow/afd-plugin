# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# ruff: noqa: E402
# Native imports follow the CPU-only collection guards.

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")

from vllm.config import CompilationMode, CUDAGraphMode

from afd_plugin.entrypoints import npu_ffn
from tests.unit.fakes.engine_client import FFNProcessManager, FFNServiceClient
from tests.unit.fakes.vllm_config import v41_eager_qualification_config


@pytest.fixture
def ffn_config(monkeypatch):
    monkeypatch.setenv("AFD_DSV41_EAGER_QUALIFICATION", "1")
    config = v41_eager_qualification_config(
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        role="ffn",
    )
    parallel = config.parallel_config
    parallel.distributed_executor_backend = "mp"
    parallel.data_parallel_size_local = 1
    parallel.nnodes = 1
    parallel.node_rank = 0
    config.device_config = SimpleNamespace(device_type="npu")
    config.shutdown_timeout = 180
    return config


def test_ffn_entrypoint_preserves_original_engram_configuration(ffn_config):
    npu_ffn.validate_ffn_configuration(ffn_config)

    assert ffn_config.engram_config.cpu_offload is True
    assert ffn_config.engram_config.dp_shared_memory is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("distributed_executor_backend", "ray"),
        ("data_parallel_size", 2),
        ("data_parallel_size_local", 2),
        ("data_parallel_rank", 1),
        ("nnodes", 2),
        ("node_rank", 1),
    ],
)
def test_ffn_entrypoint_rejects_other_process_layouts(ffn_config, field, value):
    setattr(ffn_config.parallel_config, field, value)

    with pytest.raises(ValueError, match="one local DP1 mp engine"):
        npu_ffn.validate_ffn_configuration(ffn_config)


@pytest.mark.parametrize("field", ["speculative_config", "kv_transfer_config"])
def test_ffn_entrypoint_rejects_dspark_pd_before_isolation(ffn_config, field):
    setattr(ffn_config, field, SimpleNamespace())

    with pytest.raises(ValueError, match="DSpark/PD off"):
        npu_ffn.validate_ffn_configuration(ffn_config)


@pytest.mark.parametrize(
    "role,connector",
    [("attention", "P2pHcclAFDConnector"), ("ffn", "CAMP2pAFDConnector")],
)
def test_ffn_entrypoint_rejects_another_role_or_connector(ffn_config, role, connector):
    ffn_config.additional_config["afd"].update(role=role, connector=connector)

    with pytest.raises(ValueError):
        npu_ffn.validate_ffn_configuration(ffn_config)


@pytest.fixture
def service(monkeypatch, request):
    client = FFNServiceClient(FFNProcessManager(request.param))
    monkeypatch.setattr(npu_ffn, "CoreEngineProcManager", FFNProcessManager)
    monkeypatch.setattr(npu_ffn, "AFDFFNClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(npu_ffn.Executor, "get_class", lambda config: None)
    return client


@pytest.mark.parametrize(
    "service,expected",
    [({"EngineCore": 0}, 0), ({"EngineCore": 7}, 7), ({"EngineCore": -15}, 143)],
    indirect=["service"],
)
def test_ffn_entrypoint_preserves_engine_outcomes_and_closes_resources(
    ffn_config, service, expected, capsys
):
    result = npu_ffn.run_ffn(ffn_config, log_stats=False)

    assert result == expected
    assert service.closed
    assert (npu_ffn.SUCCESS_MESSAGE in capsys.readouterr().out) is (expected == 0)


@pytest.mark.parametrize(
    "service", [{}, {"EngineCore": 0, "unexpected": 0}], indirect=True
)
def test_ffn_entrypoint_rejects_incomplete_or_extra_engine_outcomes(
    ffn_config, service, capsys
):
    with pytest.raises(RuntimeError, match="unfinished engines"):
        npu_ffn.run_ffn(ffn_config, log_stats=False)

    assert service.closed
    assert npu_ffn.SUCCESS_MESSAGE not in capsys.readouterr().out


@pytest.mark.parametrize("failure_stage", ["monitor", "shutdown"])
@pytest.mark.parametrize("service", [{"EngineCore": 0}], indirect=True)
def test_ffn_entrypoint_propagates_supervision_errors_without_success(
    ffn_config, service, failure_stage, capsys
):
    error = RuntimeError(f"{failure_stage} failed")
    if failure_stage == "monitor":
        service.resources.engine_manager.monitor_error = error
    else:
        service.shutdown_error = error

    with pytest.raises(RuntimeError) as caught:
        npu_ffn.run_ffn(ffn_config, log_stats=False)

    assert caught.value is error
    assert service.closed
    assert npu_ffn.SUCCESS_MESSAGE not in capsys.readouterr().out
