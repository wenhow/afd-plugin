# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Runtime imports follow CPU dependency gates.
# ruff: noqa: E402
"""Adapter construction and payload contracts on CPU, with no device claims."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from vllm.config import CompilationMode, CUDAGraphMode
from vllm.distributed import parallel_state

from afd_plugin.compat.npu.feature_validation import (
    fail_if_unsupported_npu_afd_features,
)
from afd_plugin.compat.npu.runtime_config import get_npu_ffn_config
from afd_plugin.config import parse_afd_config
from afd_plugin.connectors.metadata import AFDTransferContext, AFDTransferMetadata
from afd_plugin.connectors.npu import hccl_staged as adapter_module
from afd_plugin.connectors.npu.hccl_config import uses_staged_hccl_runtime
from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector
from afd_plugin.distributed.hccl import HcclStageLayout
from afd_plugin.distributed.hccl_plan import HcclParentPlan
from afd_plugin.validation import validate_npu_model_runner_v2_config
from tests.unit.fakes.hccl import (
    HcclCopyFailingTensor,
    hccl_layer_tensors,
    make_staged_connector_pair,
)
from tests.unit.fakes.vllm_config import hccl_vllm_config


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
