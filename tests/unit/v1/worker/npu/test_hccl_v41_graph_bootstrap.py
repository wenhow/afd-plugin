# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Graph assembly host contracts; no device/KV allocation is simulated."""

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
pytest.importorskip("vllm_ascend")

from vllm_ascend.attention.sfa_v1 import AscendSFABackend

from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.v1.worker.npu import hccl_v41_graph_coordinator as coordinator_module
from afd_plugin.v1.worker.npu.hccl_v41_graph_bootstrap import (
    HcclV41AttentionGraphBootstrap,
)
from tests.unit.fakes.hccl_v41_graph_bootstrap import make_graph_bootstrap_case


@pytest.fixture
def assemble_case(monkeypatch):
    monkeypatch.setattr(
        coordinator_module, "_get_graph_update_backend", lambda _: AscendSFABackend
    )

    def create(**kwargs):
        case = make_graph_bootstrap_case(**kwargs)
        case.bootstrap = HcclV41AttentionGraphBootstrap(
            case.owner,
            synchronize=lambda: case.events.append("graph-sync"),
            overlap_settings=HcclOverlapSettings(),
        )
        case.bootstrap.bind_loaded_model()
        return case

    return create


@pytest.fixture
def assembled(assemble_case):
    return assemble_case()


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("query_len", [1, 3])
def test_pre_kv_and_real_kv_share_one_storage_generation_and_session(
    assemble_case, stages, query_len
):
    case = assemble_case(stages=stages, target_query_len=query_len)
    bootstrap = case.bootstrap
    bootstrap.begin_kv_init()
    case.native.initialize_kv_cache(case.owner, case.owner.kv_cache_config)
    bootstrap.complete_kv_init()

    pre = case.owner.hccl_v41_graph_eager.control
    post = case.owner.hccl_v41_graph_coordinator.control
    assert pre.generation() == post.generation() == 0
    assert pre.generation is post.generation
    assert pre.cache is post.cache is bootstrap.cache
    assert pre.metadata_shapes is post.metadata_shapes is None
    assert case.owner.hccl_v41_graph_session is bootstrap.session
    assert (
        case.owner.hccl_v41_graph_eager.overlap_settings is bootstrap.overlap_settings
    )
    assert (
        case.owner.hccl_v41_graph_coordinator.overlap_settings
        is bootstrap.overlap_settings
    )
    assert case.transport.transcript == []


@pytest.mark.parametrize("completed", [False, True])
def test_second_kv_attempt_is_rejected_without_a_new_phase(assembled, completed):
    case = assembled
    case.bootstrap.begin_kv_init()
    if completed:
        case.native.initialize_kv_cache(case.owner, case.owner.kv_cache_config)
        case.bootstrap.complete_kv_init()
    with pytest.raises(RuntimeError, match="first idle load"):
        case.bootstrap.begin_kv_init()
    assert not case.bootstrap.cache.terminal and not case.transport.terminal
    assert case.transport.transcript == []


@pytest.mark.parametrize("retirement", ["closed", "failed"])
def test_kv_cannot_start_after_public_cache_retirement(assembled, retirement):
    case = assembled
    if retirement == "closed":
        case.bootstrap.cache.close()
    else:
        case.bootstrap.cache.fail()
    with pytest.raises(RuntimeError, match="first idle load"):
        case.bootstrap.begin_kv_init()
    assert case.transport.transcript == []
    assert not case.transport.closed


@pytest.mark.parametrize("entry", ["bind", "kv", "complete"])
def test_failed_assembly_retains_resources_and_refuses_reuse(assembled, entry):
    case = assembled
    case.bootstrap.fail()
    with pytest.raises(RuntimeError):
        if entry == "bind":
            case.bootstrap.bind_loaded_model()
        elif entry == "kv":
            case.bootstrap.begin_kv_init()
        else:
            case.bootstrap.complete_kv_init()
    assert case.bootstrap.cache.terminal and not case.bootstrap.cache.closed
    assert case.transport.terminal and not case.transport.closed
    assert case.owner.hccl_v41_graph_eager.control.cache is case.bootstrap.cache
    assert case.transport.transcript == []


def test_duplicate_bootstrap_keeps_loaded_model_on_binding_failure(assembled):
    case = assembled
    duplicate = HcclV41AttentionGraphBootstrap(
        case.owner,
        synchronize=lambda: None,
        overlap_settings=HcclOverlapSettings(),
    )
    with pytest.raises(RuntimeError, match="idle pre-KV owners"):
        duplicate.bind_loaded_model()
    assert duplicate.cache.terminal and not duplicate.cache.closed
    assert case.owner.model is case.owner.model_state.model
    assert case.owner.hccl_v41_graph_eager is case.bootstrap.eager
    assert case.transport.terminal and not case.transport.closed
