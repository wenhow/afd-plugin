# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Adapter construction and payload contracts on CPU, with no device claims."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import partial

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.distributed import parallel_state

from afd_plugin.compat.npu.feature_validation import (
    fail_if_unsupported_npu_afd_features,
)
from afd_plugin.compat.npu.runtime_config import (
    DeepseekV41TargetShape,
    get_npu_ffn_config,
    resolve_deepseek_v41_target_shape,
)
from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu import hccl_staged as adapter_module
from afd_plugin.connectors.npu.hccl_config import (
    uses_staged_hccl_runtime,
    validate_staged_hccl_binding,
)
from afd_plugin.connectors.npu.hccl_overlap import HcclOverlapSettings
from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_plan import (
    HcclParentPlan,
    HcclPlanMessage,
    decode_hccl_plan_message,
    encode_hccl_plan_message,
)
from afd_plugin.validation import validate_npu_model_runner_v2_config
from tests.unit.fakes.hccl import (
    HcclCopyFailingTensor,
    ScriptedStagedControlPeer,
    hccl_layer_tensors,
    make_registered_hccl_runtime_connector,
    make_staged_connector_pair,
)
from tests.unit.fakes.vllm_config import (
    configure_hccl_ep,
    hccl_vllm_config,
    v41_dspark_pd_graph_config,
)


@pytest.fixture
def runtime():
    config = hccl_vllm_config(
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
    )
    config.parallel_config.rank = 0
    return config


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("dp_rank", [0, 3])
def test_ep_factory_binds_global_world_rank_to_dp_role_and_local_device(
    runtime, role, dp_rank
):
    configure_hccl_ep(runtime, dp_rank=dp_rank)
    runtime.additional_config["afd"]["role"] = role
    afd = parse_afd_config(runtime, expected_role=role)
    connector = adapter_module.create_npu_runtime_connector(dp_rank, 3, runtime, afd)
    transport = connector.transport
    validate_staged_hccl_binding(runtime, transport, torch.device("npu", 3))
    assert isinstance(connector, adapter_module.HcclStagedAFDConnector)
    assert transport.role_rank == dp_rank
    assert transport.world_rank == (4 + dp_rank if role == "attention" else dp_rank)
    assert transport.negotiated_capabilities.expert_parallel
    assert not connector.is_initialized and not transport.initialized


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("num_ubatches", [0, 1, 2])
def test_runtime_selector_preserves_u1_and_builds_uninitialized_u2_session(
    runtime, role, num_ubatches
):
    runtime.additional_config["afd"]["role"] = role
    parallel = runtime.parallel_config
    parallel.num_ubatches = num_ubatches
    parallel.use_ubatching = parallel.enable_dbo = num_ubatches == 2
    afd = parse_afd_config(runtime, expected_role=role)

    validate_npu_model_runner_v2_config(runtime, expected_role=role, device_type="npu")
    connector = adapter_module.create_npu_runtime_connector(0, 3, runtime, afd)

    assert not connector.is_initialized
    if num_ubatches == 2:
        assert isinstance(connector, adapter_module.HcclStagedAFDConnector)
        assert not connector.transport.initialized
        assert connector.transport.role == role
        assert connector.transport.local_device == 3
        assert connector.transport.init_method == f"tcp://{afd.host}:{afd.port}"
        assert connector.transport.negotiated_capabilities.local_counts_version == 1
    else:
        assert isinstance(connector, P2pHcclAFDConnector)
    with pytest.raises(RuntimeError, match="acceptance are pending"):
        fail_if_unsupported_npu_afd_features(runtime)


@pytest.mark.parametrize("role_rank", [0, 1])
def test_runtime_selector_preserves_native_tp_coordinate(
    monkeypatch, runtime, role_rank
):
    parallel = runtime.parallel_config
    parallel.tensor_parallel_size = 2
    parallel.rank = role_rank
    parallel.num_ubatches = 2
    parallel.enable_dbo = parallel.use_ubatching = True
    runtime.additional_config["afd"].update(num_attention_ranks=2, num_ffn_ranks=2)
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_rank", lambda: role_rank
    )
    afd = parse_afd_config(runtime)

    connector = adapter_module.create_npu_runtime_connector(role_rank, 3, runtime, afd)

    assert connector.role_rank == role_rank
    assert connector.transport.role_rank == role_rank
    assert connector.transport.negotiated_capabilities.native_tp_size == 2
    assert connector.transport.init_method == f"tcp://{afd.host}:{afd.port}"


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_runtime_hello_declares_the_frozen_eager_wire_family(
    runtime, role, enabled, graph
):
    runtime.additional_config["afd"]["role"] = role
    runtime.parallel_config.num_ubatches = 2
    runtime.parallel_config.enable_dbo = runtime.parallel_config.use_ubatching = True
    runtime.model_config.enforce_eager = not graph
    runtime.compilation_config.cudagraph_mode = (
        CUDAGraphMode.FULL_DECODE_ONLY if graph else CUDAGraphMode.NONE
    )
    settings = (
        None
        if enabled is None
        else HcclOverlapSettings(eager_u2_stream_overlap=enabled)
    )
    connector = adapter_module.create_npu_runtime_connector(
        0,
        0,
        runtime,
        parse_afd_config(runtime, expected_role=role),
        overlap_settings=settings,
    )
    caps = connector.transport.capabilities
    hello = encode_hccl_plan_message(
        HcclPlanMessage(
            "HELLO",
            role=role,
            world_rank=0 if role == "ffn" else caps.ffn_size,
            capabilities=caps,
        )
    )
    declared = decode_hccl_plan_message(hello).capabilities

    assert declared.eager_pipeline_version == int(enabled is True and not graph)
    assert declared.graph_execution_version == int(graph)
    assert connector.overlap_settings is settings
    assert not connector.is_initialized


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_factory_declares_native_tp4_stream_selection(
    monkeypatch, runtime, enabled
):
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0)
    runtime.additional_config["afd"].update(num_attention_ranks=4, num_ffn_ranks=4)
    runtime.parallel_config.tensor_parallel_size = 4
    runtime.parallel_config.num_ubatches = 2
    runtime.parallel_config.enable_dbo = runtime.parallel_config.use_ubatching = True
    settings = HcclOverlapSettings(eager_u2_stream_overlap=enabled)
    connector = adapter_module.create_npu_runtime_connector(
        0,
        0,
        runtime,
        parse_afd_config(runtime),
        overlap_settings=settings,
    )
    declared = decode_hccl_plan_message(
        encode_hccl_plan_message(
            HcclPlanMessage(
                "HELLO",
                role="attention",
                world_rank=4,
                capabilities=connector.transport.capabilities,
            )
        )
    ).capabilities
    assert declared.eager_pipeline_version == int(enabled)
    assert (declared.attention_size, declared.ffn_size) == (4, 4)
    assert not connector.is_initialized


@pytest.fixture
def configuration_connector(monkeypatch, runtime):
    """Real adapter construction with host transport state, without device init."""

    def create(*, role="attention", role_rank=0, graph=False, settings=None):
        runtime.additional_config["afd"].update(
            role=role, num_attention_ranks=2, num_ffn_ranks=2
        )
        parallel = runtime.parallel_config
        parallel.tensor_parallel_size = 2
        parallel.rank = role_rank
        parallel.num_ubatches = 2
        parallel.enable_dbo = parallel.use_ubatching = True
        monkeypatch.setattr(
            parallel_state, "get_tensor_model_parallel_rank", lambda: role_rank
        )
        runtime.model_config.enforce_eager = not graph
        runtime.compilation_config.cudagraph_mode = (
            CUDAGraphMode.FULL_DECODE_ONLY if graph else CUDAGraphMode.NONE
        )
        connector = adapter_module.create_npu_runtime_connector(
            role_rank,
            0,
            runtime,
            parse_afd_config(runtime, expected_role=role),
            overlap_settings=settings,
        )
        connector.transport = ScriptedStagedControlPeer(
            role, role_rank, connector.transport.capabilities, {}, initialized=False
        )
        return connector

    return create


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("role_rank", [0, 1])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_session_configuration_exports_negotiated_identity_and_all_settings(
    configuration_connector, role, role_rank, graph, enabled
):
    settings = HcclOverlapSettings.from_environment(
        {
            name: str(int(enabled))
            for name in HcclOverlapSettings().effective_settings()
        },
        num_stages=2,
    )
    connector = configuration_connector(
        role=role, role_rank=role_rank, graph=graph, settings=settings
    )
    # Seed an idle host session, not a claim that HELLO or HCCL ran here.
    connector.transport.initialized = True
    connector.transport.negotiated_capabilities = replace(
        connector.transport.capabilities, max_tokens=32
    )

    assert connector.configuration_record() == {
        "event": "session_configuration",
        "phase": "startup",
        "role": role,
        "world_rank": role_rank + (2 if role == "attention" else 0),
        "role_rank": role_rank,
        "local_device": 0,
        "snapshot_present": True,
        "negotiated.graph_execution_version": int(graph),
        "negotiated.eager_pipeline_version": int(enabled and not graph),
        "negotiated.native_tp_size": 2,
        "negotiated.attention_size": 2,
        "negotiated.ffn_size": 2,
        "negotiated.max_stages": 2,
        "negotiated.max_tokens": 32,
        "configured.AFD_HCCL_EAGER_U2_STREAM_OVERLAP": enabled,
        "configured.AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP": enabled,
        "configured.AFD_HCCL_GRAPH_U2_HYBRID_DAG": enabled,
        "configured.AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM": enabled,
        "configured.AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM": enabled,
        "configured.AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER": enabled,
        "configured.AFD_HCCL_STAGE_DIAGNOSTICS": enabled,
        "configured.AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS": enabled,
    }


@pytest.mark.parametrize("graph", [False, True])
def test_session_configuration_leaves_missing_settings_unknown(
    configuration_connector, graph
):
    connector = configuration_connector(graph=graph)
    connector.transport.initialized = True

    record = connector.configuration_record()

    assert record["snapshot_present"] is False
    assert record["negotiated.graph_execution_version"] == int(graph)
    assert record["negotiated.eager_pipeline_version"] == 0
    assert not any(name.startswith("configured.") for name in record)


def test_session_configuration_retains_the_snapshot_after_environment_changes(
    monkeypatch, configuration_connector
):
    name = "AFD_HCCL_EAGER_U2_STREAM_OVERLAP"
    connector = configuration_connector(settings=HcclOverlapSettings())
    connector.transport.initialized = True
    original = connector.configuration_record()
    original[f"configured.{name}"] = False
    monkeypatch.setenv(name, "0")

    record = connector.configuration_record()

    assert record[f"configured.{name}"] is True
    assert record["negotiated.eager_pipeline_version"] == 1


@pytest.mark.parametrize("state", ["uninitialized", "closed", "terminal", "parent"])
def test_session_configuration_requires_an_idle_live_transport(
    configuration_connector, state
):
    connector = configuration_connector()
    transport = connector.transport
    transport.initialized = state != "uninitialized"
    transport.closed = state == "closed"
    transport.terminal = state == "terminal"
    if state == "parent":
        transport.plan = HcclParentPlan(
            "configuration-parent", (0, 1), (HcclStageLayout(0, (1, 1), (1, 1), 2),)
        )

    with pytest.raises(RuntimeError, match="idle live transport"):
        connector.configuration_record()


def test_session_configuration_rejects_a_live_graph_reservation(
    monkeypatch, configuration_connector
):
    connector = configuration_connector(graph=True)
    connector.transport.initialized = True
    # The external transport's accepted reservation is the idle boundary here.
    monkeypatch.setattr(
        ScriptedStagedControlPeer,
        "graph_reservation",
        property(lambda transport: "accepted-graph-reservation"),
    )

    with pytest.raises(RuntimeError, match="idle live transport"):
        connector.configuration_record()


def test_dspark_component_keeps_the_existing_u1_connector():
    runtime = hccl_vllm_config(
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        dspark=True,
    )
    afd = parse_afd_config(runtime)

    connector = adapter_module.create_npu_runtime_connector(0, 0, runtime, afd)

    assert not uses_staged_hccl_runtime(runtime, afd)
    assert isinstance(connector, P2pHcclAFDConnector)
    assert not connector.is_initialized


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize(
    "class_name", ["RegisteredHcclConnector", "RegisteredNonHcclConnector"]
)
@pytest.mark.parametrize("snapshot_present", [False, True])
def test_legacy_runtime_factory_preserves_registry_replacement_and_optional_snapshot(
    monkeypatch, role, class_name, snapshot_present
):
    settings = (
        HcclOverlapSettings(eager_u2_stream_overlap=False) if snapshot_present else None
    )
    connector = make_registered_hccl_runtime_connector(
        monkeypatch,
        adapter_module,
        class_name=class_name,
        overlap_settings=settings,
        role=role,
    )

    assert type(connector).__name__ == class_name
    assert not connector.is_initialized
    if isinstance(connector, P2pHcclAFDConnector):
        assert connector.overlap_settings is settings
        connector.init_afd_connector()
        record = connector.configuration_record()
        assert record["protocol"] == "legacy_pair"
        assert record["snapshot_present"] is snapshot_present
        if snapshot_present:
            assert record["configured.AFD_HCCL_EAGER_U2_STREAM_OVERLAP"] is False


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("enable_dbo", [False, True])
def test_dspark_u2_selects_the_same_target_protocol_after_ffn_role_isolation(
    monkeypatch, role, tp_size, enable_dbo
):
    config = hccl_vllm_config(
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        role=role,
        tensor_parallel_size=tp_size,
        dspark=True,
        num_ubatches=2,
    )
    config.parallel_config.enable_dbo = enable_dbo
    config.parallel_config.rank = tp_size - 1
    source_spec = config.speculative_config
    if role == "ffn":
        config = get_npu_ffn_config(config)
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_rank", lambda: tp_size - 1
    )
    afd = parse_afd_config(config, expected_role=role)

    validate_npu_model_runner_v2_config(config, expected_role=role, device_type="npu")
    connector = adapter_module.create_npu_runtime_connector(tp_size - 1, 3, config, afd)

    assert isinstance(connector, adapter_module.HcclStagedAFDConnector)
    assert not connector.is_initialized
    assert connector.transport.role == role
    assert connector.transport.role_rank == tp_size - 1
    assert connector.transport.negotiated_capabilities.native_tp_size == tp_size
    assert (config.speculative_config is source_spec) == (role == "attention")
    with pytest.raises(RuntimeError, match="acceptance are pending"):
        fail_if_unsupported_npu_afd_features(config)


@pytest.mark.parametrize("boundary", ["method", "checkpoint", "adaptive"])
def test_dspark_u2_selector_rejects_invalid_draft_instead_of_using_legacy_u1(boundary):
    config = hccl_vllm_config(
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        dspark=True,
        num_ubatches=2,
    )
    if boundary == "method":
        config.speculative_config.method = "mtp"
    elif boundary == "checkpoint":
        config.speculative_config.draft_model_config.model = "foreign-model"
    else:
        config.speculative_config.enable_adaptive_verification = True
    reason = {
        "method": "method='dspark'",
        "checkpoint": "same target checkpoint",
        "adaptive": "U2 requires adaptive verification off",
    }[boundary]
    with pytest.raises(ValueError, match=reason):
        adapter_module.create_npu_runtime_connector(
            0, 0, config, parse_afd_config(config)
        )


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("query_len", [1, 3])
@pytest.mark.parametrize("order", ["stage_serial", "bulk_v1", "cross_layer_v1"])
def test_normal_graph_factory_and_validation_select_the_same_role_protocol(
    role, stages, query_len, order
):
    runtime = hccl_vllm_config(
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        role=role,
        dspark=query_len > 1,
        num_ubatches=stages,
    )
    runtime.model_config.enforce_eager = False
    runtime.parallel_config.rank = 0
    runtime.additional_config["afd"]["connector_extra_config"] = {
        "graph_exchange_order": order
    }
    if query_len > 1:
        runtime.speculative_config.num_speculative_tokens = query_len - 1
    shape = DeepseekV41TargetShape(query_len)
    if role == "ffn":
        runtime = get_npu_ffn_config(runtime)
    afd = parse_afd_config(runtime, expected_role=role)
    validate_npu_model_runner_v2_config(runtime, expected_role=role, device_type="npu")

    connector = adapter_module.create_npu_runtime_connector(
        0, 0, runtime, afd, target_shape=shape
    )

    assert isinstance(connector, adapter_module.HcclStagedAFDConnector)
    caps = connector.transport.negotiated_capabilities
    assert caps.graph_execution_version == caps.local_counts_version == 1
    assert caps.target_query_len == query_len
    assert caps.graph_exchange_order == order
    assert not connector.is_initialized
    assert runtime.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    assert not runtime.model_config.enforce_eager
    if role == "ffn":
        assert runtime.speculative_config is None and runtime.engram_config is None


@pytest.fixture
def pd_target_graph_runtime():
    return partial(
        v41_dspark_pd_graph_config,
        dtype=torch.bfloat16,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
    )


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("stages", [0, 1, 2])
@pytest.mark.parametrize("tp_size", [1, 2])
def test_pd_target_graph_factory_keeps_d_ownership_and_the_original_query_width(
    monkeypatch, pd_target_graph_runtime, role, stages, tp_size
):
    source = pd_target_graph_runtime(
        role=role, tensor_parallel_size=tp_size, num_ubatches=stages
    )
    source.parallel_config.rank = tp_size - 1
    before = deepcopy(source)
    target_shape = resolve_deepseek_v41_target_shape(source)
    runtime = get_npu_ffn_config(source) if role == "ffn" else source
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_rank", lambda: tp_size - 1
    )
    afd = parse_afd_config(runtime, expected_role=role)

    validate_npu_model_runner_v2_config(runtime, expected_role=role, device_type="npu")
    connector = adapter_module.create_npu_runtime_connector(
        tp_size - 1, 3, runtime, afd, target_shape=target_shape
    )

    assert isinstance(connector, adapter_module.HcclStagedAFDConnector)
    assert not connector.is_initialized
    caps = connector.transport.capabilities
    assert caps.graph_execution_version == caps.local_counts_version == 1
    assert caps.eager_pipeline_version == 0
    assert caps.target_query_len == 3 and caps.native_tp_size == tp_size
    assert caps.graph_exchange_order == "stage_serial"
    assert connector.transport.role == role and connector.role_rank == tp_size - 1
    assert (runtime.kv_transfer_config is source.kv_transfer_config) == (
        role == "attention"
    )
    if role == "ffn":
        assert runtime.speculative_config is runtime.engram_config is None
    assert source == before
    with pytest.raises(RuntimeError, match="acceptance are pending"):
        fail_if_unsupported_npu_afd_features(runtime)


@pytest.mark.parametrize(
    "boundary",
    ["producer", "native_pd", "full", "bulk"],
)
def test_pd_target_graph_factory_rejects_other_pd_and_exchange_modes(
    pd_target_graph_runtime, boundary
):
    runtime = pd_target_graph_runtime(num_ubatches=2)
    if boundary == "producer":
        runtime.kv_transfer_config.kv_role = "kv_producer"
    elif boundary == "native_pd":
        runtime.kv_transfer_config.kv_connector_module_path = None
    elif boundary == "full":
        runtime.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    else:
        runtime.additional_config["afd"]["connector_extra_config"] = {
            "graph_exchange_order": "bulk_v1"
            if boundary == "bulk"
            else "cross_layer_v1"
        }
    before = deepcopy(runtime)

    with pytest.raises(ValueError):
        validate_npu_model_runner_v2_config(
            runtime, expected_role="attention", device_type="npu"
        )
    with pytest.raises(ValueError):
        adapter_module.create_npu_runtime_connector(
            0, 0, runtime, parse_afd_config(runtime)
        )

    assert runtime == before


@pytest.mark.parametrize("with_spec", [False, True])
def test_pd_graph_factory_admits_existing_target_and_cross_layer_contracts(
    pd_target_graph_runtime, with_spec
):
    runtime = pd_target_graph_runtime(num_ubatches=2)
    if with_spec:
        runtime.additional_config["afd"]["connector_extra_config"] = {
            "graph_exchange_order": "cross_layer_v1"
        }
    else:
        runtime.speculative_config = None
    before = deepcopy(runtime)
    validate_npu_model_runner_v2_config(
        runtime, expected_role="attention", device_type="npu"
    )
    connector = adapter_module.create_npu_runtime_connector(
        0, 0, runtime, parse_afd_config(runtime)
    )
    cap = connector.transport.capabilities
    assert cap.graph_execution_version == 1
    assert cap.target_query_len == (3 if with_spec else 1)
    assert cap.graph_exchange_order == (
        "cross_layer_v1" if with_spec else "stage_serial"
    )
    assert not connector.is_initialized and runtime == before


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("held_state", ["pd", "draft"])
def test_pd_target_graph_ffn_runtime_rejects_attention_owned_state(
    pd_target_graph_runtime, stages, held_state
):
    source = pd_target_graph_runtime(role="ffn", num_ubatches=stages)
    runtime = deepcopy(source)
    if held_state == "pd":
        runtime.speculative_config = None
    else:
        runtime.kv_transfer_config = None

    with pytest.raises(ValueError):
        adapter_module.create_npu_runtime_connector(
            0,
            0,
            runtime,
            parse_afd_config(runtime, expected_role="ffn"),
            target_shape=DeepseekV41TargetShape(3),
        )


@pytest.mark.parametrize("role", ["attention", "ffn"])
@pytest.mark.parametrize("stages", [0, 2])
@pytest.mark.parametrize("order", ["bulk_v1", "cross_layer_v1"])
def test_eager_factory_and_common_validation_reject_graph_order_before_rendezvous(
    runtime, role, stages, order
):
    runtime.additional_config["afd"].update(
        role=role, connector_extra_config={"graph_exchange_order": order}
    )
    parallel = runtime.parallel_config
    parallel.num_ubatches = stages
    parallel.enable_dbo = parallel.use_ubatching = stages == 2
    afd = parse_afd_config(runtime, expected_role=role)

    with pytest.raises(ValueError, match="requires Graph execution"):
        validate_npu_model_runner_v2_config(
            runtime, expected_role=role, device_type="npu"
        )
    with pytest.raises(ValueError, match="requires Graph execution"):
        adapter_module.create_npu_runtime_connector(0, 0, runtime, afd)


@pytest.mark.parametrize("boundary", ["sleep", "weight_transfer", "mode"])
def test_normal_graph_factory_rejects_storage_or_mode_changes_before_rendezvous(
    runtime, boundary
):
    runtime.model_config.enforce_eager = False
    runtime.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    if boundary == "sleep":
        runtime.model_config.enable_sleep_mode = True
    elif boundary == "weight_transfer":
        runtime.weight_transfer_config = object()
    else:
        runtime.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    reason = "fixed storage" if boundary != "mode" else "FULL_DECODE_ONLY"
    with pytest.raises(ValueError, match=reason):
        adapter_module.create_npu_runtime_connector(
            0, 0, runtime, parse_afd_config(runtime)
        )


@pytest.fixture
def pair(monkeypatch):
    case = make_staged_connector_pair(monkeypatch, adapter_module)
    case.session.accept(
        HcclParentPlan(
            "adapter-parent",
            (0, 1),
            (HcclStageLayout(0, (2,), (3,), 1), HcclStageLayout(1, (1,), (2,), 1)),
        )
    )
    return case


def test_adapter_preserves_each_layer_stage_fp32_ids_and_output_buffer(pair):
    hidden, fp32, ids = hccl_layer_tensors()
    expected_messages = []
    for layer in (0, 1):
        for stage, rows in ((0, 3), (1, 2)):
            stage_ids = ids[:rows].clone()
            stage_ids[2 if stage == 0 else 1 :] = 0
            context = AFDTransferContext(AFDTransferMetadata(layer, stage, [rows]))
            pair.attention.send_attn_output(
                hidden[:rows],
                context,
                hidden_states_fp32=fp32[:rows],
                input_ids=stage_ids,
            )
            payload = pair.ffn.recv_attn_output(stage, layer_idx=layer)
            assert torch.equal(payload.hidden_states, hidden[:rows])
            assert torch.equal(payload.hidden_states_fp32, fp32[:rows])
            assert torch.equal(payload.input_ids, stage_ids)
            result = (payload.hidden_states_fp32 + layer + stage).to(torch.bfloat16)
            pair.ffn.send_ffn_output(result, payload.context)
            output = torch.empty_like(hidden[:rows])
            assert (
                pair.attention.recv_ffn_output(output, stage, context=context) is output
            )
            assert torch.equal(output, result)
            expected_messages.extend((("A2F", layer, stage), ("F2A", layer, stage)))
    assert pair.session.messages == expected_messages
    pair.attention.close()
    pair.ffn.close()
    assert not pair.attention.is_initialized and not pair.ffn.is_initialized


@pytest.mark.parametrize("boundary", ["stage", "capacity", "transaction"])
def test_adapter_rejects_context_mismatch_before_sending_payload(pair, boundary):
    hidden, fp32, ids = hccl_layer_tensors()
    context = AFDTransferContext(AFDTransferMetadata(0, 0, [3]))
    if boundary == "stage":
        context.metadata.stage_idx = 2
    elif boundary == "capacity":
        context.metadata.seq_lens = [2]
    else:
        context.states = adapter_module.HcclStagedTransferState("stale", 0, 0)

    with pytest.raises(ValueError, match="accepted parent"):
        pair.attention.send_attn_output(
            hidden, context, hidden_states_fp32=fp32, input_ids=ids
        )

    assert pair.session.messages == []


def test_adapter_rejects_wrong_receive_buffer_without_consuming_reply(pair):
    hidden, fp32, ids = hccl_layer_tensors()
    context = AFDTransferContext(AFDTransferMetadata(0, 0, [3]))
    pair.attention.send_attn_output(
        hidden, context, hidden_states_fp32=fp32, input_ids=ids
    )
    payload = pair.ffn.recv_attn_output(0, layer_idx=0)
    pair.ffn.send_ffn_output(payload.hidden_states, payload.context)

    with pytest.raises(ValueError, match="buffer must match"):
        pair.attention.recv_ffn_output(torch.empty(1, 16), context=context)

    output = torch.empty_like(hidden)
    assert torch.equal(pair.attention.recv_ffn_output(output, context=context), hidden)


def test_adapter_marks_consumed_reply_terminal_when_output_copy_fails(pair):
    hidden, fp32, ids = hccl_layer_tensors()
    context = AFDTransferContext(AFDTransferMetadata(0, 0, [3]))
    pair.attention.send_attn_output(
        hidden, context, hidden_states_fp32=fp32, input_ids=ids
    )
    payload = pair.ffn.recv_attn_output(0, layer_idx=0)
    pair.ffn.send_ffn_output(payload.hidden_states, payload.context)
    output = torch.empty_like(hidden).as_subclass(HcclCopyFailingTensor)

    with pytest.raises(RuntimeError, match="controlled output copy failure"):
        pair.attention.recv_ffn_output(output, context=context)

    assert pair.attention.transport.terminal
