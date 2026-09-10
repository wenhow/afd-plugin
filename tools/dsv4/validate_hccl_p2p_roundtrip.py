#!/usr/bin/env python3
"""Validate blocking HCCL AFD transfers on physical NPUs."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def _port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _token_counts(
    attention_size: int,
    *,
    step_idx: int,
    stage_idx: int,
    tensor_parallel_size: int = 1,
) -> list[int]:
    dp_size = attention_size // tensor_parallel_size
    dp_counts = [
        2 + ((step_idx + stage_idx + dp_rank) % 4) for dp_rank in range(dp_size)
    ]
    return [
        dp_counts[attention_rank // tensor_parallel_size]
        for attention_rank in range(attention_size)
    ]


def _input_id_values(
    attention_rank: int,
    num_tokens: int,
    *,
    step_idx: int,
    stage_idx: int,
) -> list[int]:
    start = 1 + step_idx * 40 + stage_idx * 16 + attention_rank * 6
    values = list(range(start, start + num_tokens))
    if step_idx == 0 and stage_idx == 0 and attention_rank == 0:
        values[0] = -1
    return values


def _hidden_value(attention_rank: int, *, step_idx: int, stage_idx: int) -> int:
    return 100 * step_idx + 10 * stage_idx + attention_rank


def _mtp_token_counts(
    attention_size: int,
    *,
    step_idx: int,
    tensor_parallel_size: int = 1,
) -> list[int]:
    dp_size = attention_size // tensor_parallel_size
    dp_counts = [1 + ((2 * step_idx + dp_rank) % 3) for dp_rank in range(dp_size)]
    return [
        dp_counts[attention_rank // tensor_parallel_size]
        for attention_rank in range(attention_size)
    ]


def _dp_token_counts(counts: list[int], *, tensor_parallel_size: int) -> list[int]:
    return counts[::tensor_parallel_size]


def _mtp_hidden_value(attention_rank: int, *, step_idx: int) -> int:
    return 40 + 20 * step_idx + attention_rank


def _balanced_split_sizes(num_tokens: int, num_peers: int) -> list[int]:
    transport_tokens = max(num_tokens, num_peers)
    base, remainder = divmod(transport_tokens, num_peers)
    return [base + (offset < remainder) for offset in range(num_peers)]


def _ffn_token_counts(counts: list[int], *, ffn_size: int) -> list[int]:
    attention_size = len(counts)
    if attention_size >= ffn_size:
        ratio = attention_size // ffn_size
        return [
            sum(counts[rank * ratio : (rank + 1) * ratio]) for rank in range(ffn_size)
        ]
    ratio = ffn_size // attention_size
    return [
        shard_tokens
        for attention_tokens in counts
        for shard_tokens in _balanced_split_sizes(attention_tokens, ratio)
    ]


def _ffn_peer_shards(
    counts: list[int],
    *,
    ffn_size: int,
    ffn_rank: int,
) -> list[tuple[int, int, int]]:
    """Return ``(attention_rank, start, end)`` shards received by one FFN rank."""

    attention_size = len(counts)
    if attention_size >= ffn_size:
        ratio = attention_size // ffn_size
        return [
            (attention_rank, 0, counts[attention_rank])
            for attention_rank in range(ffn_rank * ratio, (ffn_rank + 1) * ratio)
        ]
    ratio = ffn_size // attention_size
    attention_rank, fanout_index = divmod(ffn_rank, ratio)
    split_sizes = _balanced_split_sizes(counts[attention_rank], ratio)
    start = sum(split_sizes[:fanout_index])
    return [(attention_rank, start, start + split_sizes[fanout_index])]


def _expected_shard_values(
    values: list[int],
    *,
    start: int,
    end: int,
    padding_value: int,
) -> list[int]:
    if end > len(values):
        values = [*values, *([padding_value] * (end - len(values)))]
    return values[start:end]


def _validate_graph_transport(
    *,
    connector,
    role: str,
    role_rank: int,
    attention_size: int,
    stages: int,
    step_idx: int,
    tensor_parallel_size: int,
    multistream: bool,
) -> list[dict[str, object]]:
    import torch
    import torch.distributed as dist

    from afd_plugin.connectors.metadata import (
        AFDControlPayload,
        AFDDPMetadata,
        AFDTransferContext,
        AFDTransferMetadata,
    )

    counts_by_stage = {
        stage_idx: _token_counts(
            attention_size,
            step_idx=step_idx,
            stage_idx=stage_idx,
            tensor_parallel_size=tensor_parallel_size,
        )
        for stage_idx in range(stages)
    }
    control_payload = AFDControlPayload(
        dp_metadata_list={
            stage_idx: AFDDPMetadata(
                torch.tensor(
                    _dp_token_counts(
                        counts,
                        tensor_parallel_size=tensor_parallel_size,
                    ),
                    dtype=torch.int32,
                )
            )
            for stage_idx, counts in counts_by_stage.items()
        },
        is_graph_capturing=True,
        is_warmup=False,
        tensor_parallel_size=tensor_parallel_size,
    )
    if role == "attention":
        connector.control_plane.update_state_from_dp_metadata(control_payload)
        connector.control_plane.send_dp_metadata_list(control_payload)
    else:
        received_control = connector.control_plane.recv_dp_metadata_list()
        connector.control_plane.update_state_from_dp_metadata(received_control)

    static_hidden: dict[int, torch.Tensor] = {}
    returned_buffers: dict[int, torch.Tensor] = {}
    received_ids: dict[int, list[int]] = {}
    for stage_idx, counts in counts_by_stage.items():
        if role == "attention":
            num_tokens = counts[role_rank]
            input_ids = torch.tensor(
                _input_id_values(
                    role_rank,
                    num_tokens,
                    step_idx=step_idx,
                    stage_idx=stage_idx,
                ),
                dtype=torch.int32,
                device="npu:0",
            )
            connector.send_input_ids(input_ids, ubatch_idx=stage_idx)
            static_hidden[stage_idx] = torch.full(
                (num_tokens, 16),
                _hidden_value(role_rank, step_idx=step_idx, stage_idx=stage_idx),
                dtype=torch.bfloat16,
                device="npu:0",
            )
            returned_buffers[stage_idx] = torch.empty_like(static_hidden[stage_idx])
            continue

        peer_shards = _ffn_peer_shards(
            counts,
            ffn_size=connector.ffn_size,
            ffn_rank=role_rank,
        )
        ids = connector.recv_input_ids(
            sum(end - start for _rank, start, end in peer_shards),
            ubatch_idx=stage_idx,
        )
        received_ids[stage_idx] = ids.cpu().tolist()

    dist.barrier(group=connector.p2p_pg)
    graph = torch.npu.NPUGraph()
    connector.is_graph_capturing = True
    graph_compute_stream = torch.npu.Stream() if multistream else None
    graph_send_stream = torch.npu.Stream() if multistream else None
    graph_recv_stream = torch.npu.Stream() if multistream else None
    graph_ready_events = (
        {stage_idx: torch.npu.Event() for stage_idx in counts_by_stage}
        if multistream
        else None
    )
    graph_compute_events = (
        {stage_idx: torch.npu.Event() for stage_idx in counts_by_stage}
        if multistream
        else None
    )
    graph_send_events = (
        {stage_idx: torch.npu.Event() for stage_idx in counts_by_stage}
        if multistream
        else None
    )
    graph_recv_events = (
        {stage_idx: torch.npu.Event() for stage_idx in counts_by_stage}
        if multistream
        else None
    )
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        if role == "attention" and multistream:
            assert graph_compute_stream is not None
            assert graph_send_stream is not None
            assert graph_recv_stream is not None
            assert graph_ready_events is not None
            assert graph_compute_events is not None
            assert graph_send_events is not None
            assert graph_recv_events is not None
            prepared_hidden = {}
            capture_stream = torch.npu.current_stream()
            for stage_idx, hidden in static_hidden.items():
                ready_event = graph_ready_events[stage_idx]
                ready_event.record(capture_stream)
                with torch.npu.stream(graph_compute_stream):
                    ready_event.wait(graph_compute_stream)
                    hidden.record_stream(graph_compute_stream)
                    prepared_hidden[stage_idx] = hidden + 0
                    graph_compute_events[stage_idx].record(graph_compute_stream)
            for stage_idx, hidden in prepared_hidden.items():
                with torch.npu.stream(graph_send_stream):
                    graph_compute_events[stage_idx].wait(graph_send_stream)
                    hidden.record_stream(graph_send_stream)
                    context = AFDTransferContext(
                        metadata=AFDTransferMetadata.create_attention_metadata(
                            layer_idx=1,
                            stage_idx=stage_idx,
                            seq_len=int(hidden.shape[0]),
                        ),
                    )
                    connector.send_attn_output(hidden, context)
                    graph_send_events[stage_idx].record(graph_send_stream)
            for stage_idx in prepared_hidden:
                with torch.npu.stream(graph_recv_stream):
                    graph_send_events[stage_idx].wait(graph_recv_stream)
                    returned_buffers[stage_idx].record_stream(graph_recv_stream)
                    returned_buffers[stage_idx] = connector.recv_ffn_output(
                        returned_buffers[stage_idx],
                        ubatch_idx=stage_idx,
                    )
                    graph_recv_events[stage_idx].record(graph_recv_stream)
            for recv_event in graph_recv_events.values():
                recv_event.wait(capture_stream)
        elif role == "attention":
            for stage_idx, hidden in static_hidden.items():
                context = AFDTransferContext(
                    metadata=AFDTransferMetadata.create_attention_metadata(
                        layer_idx=1,
                        stage_idx=stage_idx,
                        seq_len=int(hidden.shape[0]),
                    ),
                )
                connector.send_attn_output(hidden, context)
                returned_buffers[stage_idx] = connector.recv_ffn_output(
                    returned_buffers[stage_idx],
                    ubatch_idx=stage_idx,
                )
        elif multistream:
            assert graph_compute_stream is not None
            assert graph_send_stream is not None
            assert graph_recv_stream is not None
            assert graph_ready_events is not None
            assert graph_compute_events is not None
            assert graph_send_events is not None
            assert graph_recv_events is not None
            ffn_outputs = {}
            ffn_contexts = {}
            capture_stream = torch.npu.current_stream()
            for stage_idx in counts_by_stage:
                graph_ready_events[stage_idx].record(capture_stream)
                with torch.npu.stream(graph_recv_stream):
                    graph_ready_events[stage_idx].wait(graph_recv_stream)
                    payload = connector.recv_attn_output(
                        ubatch_idx=stage_idx,
                        layer_idx=1,
                    )
                    payload.hidden_states.record_stream(graph_recv_stream)
                    graph_recv_events[stage_idx].record(graph_recv_stream)
                with torch.npu.stream(graph_compute_stream):
                    graph_recv_events[stage_idx].wait(graph_compute_stream)
                    payload.hidden_states.record_stream(graph_compute_stream)
                    ffn_outputs[stage_idx] = payload.hidden_states + 1
                    ffn_contexts[stage_idx] = payload.context
                    graph_compute_events[stage_idx].record(graph_compute_stream)
            for stage_idx, ffn_output in ffn_outputs.items():
                with torch.npu.stream(graph_send_stream):
                    graph_compute_events[stage_idx].wait(graph_send_stream)
                    ffn_output.record_stream(graph_send_stream)
                    connector.send_ffn_output(
                        ffn_output,
                        ffn_contexts[stage_idx],
                        ubatch_idx=stage_idx,
                    )
                    graph_send_events[stage_idx].record(graph_send_stream)
            for send_event in graph_send_events.values():
                send_event.wait(capture_stream)
        else:
            for stage_idx in counts_by_stage:
                payload = connector.recv_attn_output(
                    ubatch_idx=stage_idx,
                    layer_idx=1,
                )
                connector.send_ffn_output(
                    payload.hidden_states + 1,
                    payload.context,
                    ubatch_idx=stage_idx,
                )
    connector.is_graph_capturing = False
    torch.npu.synchronize()
    dist.barrier(group=connector.p2p_pg)

    checks: list[dict[str, object]] = []
    if role == "attention":
        for stage_idx, hidden in static_hidden.items():
            checks.append(
                {
                    "phase": "graph_capture",
                    "stage": stage_idx,
                    "tokens": int(hidden.shape[0]),
                    "captured": True,
                    "multistream": multistream,
                    "physical_streams": (
                        ["recv", "compute", "send"] if multistream else ["parent"]
                    ),
                }
            )
    else:
        for stage_idx, counts in counts_by_stage.items():
            expected_ids: list[int] = []
            peer_shards = _ffn_peer_shards(
                counts,
                ffn_size=connector.ffn_size,
                ffn_rank=role_rank,
            )
            for attention_rank, start, end in peer_shards:
                source_values = _input_id_values(
                    attention_rank,
                    counts[attention_rank],
                    step_idx=step_idx,
                    stage_idx=stage_idx,
                )
                expected_ids.extend(
                    _expected_shard_values(
                        source_values,
                        start=start,
                        end=end,
                        padding_value=0,
                    )
                )
            if received_ids[stage_idx] != expected_ids:
                raise AssertionError(
                    f"graph-external input IDs mismatch for ffn={role_rank} "
                    f"stage={stage_idx}"
                )
            checks.append(
                {
                    "phase": "graph_capture",
                    "stage": stage_idx,
                    "peer_tokens": [end - start for _, start, end in peer_shards],
                    "input_ids_external": True,
                    "fan_in_out": True,
                    "multistream": multistream,
                    "physical_streams": (
                        ["recv", "compute", "send"] if multistream else ["parent"]
                    ),
                }
            )

    replay_step_idx = step_idx + 7
    if role == "attention":
        for stage_idx, hidden in static_hidden.items():
            hidden.fill_(
                _hidden_value(
                    role_rank,
                    step_idx=replay_step_idx,
                    stage_idx=stage_idx,
                )
            )
    dist.barrier(group=connector.p2p_pg)
    replay_count = 100 if multistream else 1
    for _ in range(replay_count):
        graph.replay()
        torch.npu.synchronize()
        dist.barrier(group=connector.p2p_pg)

    if role == "attention":
        for stage_idx, hidden in static_hidden.items():
            if not torch.equal(returned_buffers[stage_idx].cpu(), (hidden + 1).cpu()):
                raise AssertionError(
                    "graph replay round-trip mismatch for "
                    f"attention={role_rank} stage={stage_idx}"
                )
            checks.append(
                {
                    "phase": "graph_replay",
                    "stage": stage_idx,
                    "replays": replay_count,
                    "tokens": int(hidden.shape[0]),
                    "updated_input": True,
                    "roundtrip": True,
                }
            )
    else:
        checks.append(
            {
                "phase": "graph_replay",
                "stages": stages,
                "replays": replay_count,
                "fan_in_out": True,
            }
        )
    return checks


def _validate_mtp_graph_transport(
    *,
    connector,
    role: str,
    role_rank: int,
    attention_size: int,
    ffn_size: int,
    stages: int,
    step_idx: int,
    tensor_parallel_size: int,
    num_speculative_tokens: int,
) -> list[dict[str, object]]:
    """Match a merged Attention graph to the production FFN capture/replay."""

    import torch
    import torch.distributed as dist

    from afd_plugin.connectors.metadata import (
        AFDControlPayload,
        AFDDPMetadata,
        AFDTransferContext,
        AFDTransferMetadata,
    )

    attention_peer_counts = [0] * attention_size
    for stage_idx in range(stages):
        stage_counts = _token_counts(
            attention_size,
            step_idx=step_idx,
            stage_idx=stage_idx,
            tensor_parallel_size=tensor_parallel_size,
        )
        for attention_rank, count in enumerate(stage_counts):
            attention_peer_counts[attention_rank] += count

    control_payload = AFDControlPayload(
        dp_metadata_list={
            0: AFDDPMetadata(
                torch.tensor(
                    _dp_token_counts(
                        attention_peer_counts,
                        tensor_parallel_size=tensor_parallel_size,
                    ),
                    dtype=torch.int32,
                )
            )
        },
        is_graph_capturing=True,
        is_warmup=False,
        tensor_parallel_size=tensor_parallel_size,
    )
    if role == "attention":
        connector.control_plane.update_state_from_dp_metadata(control_payload)
        connector.control_plane.send_dp_metadata_list(control_payload)
    else:
        received_control = connector.control_plane.recv_dp_metadata_list()
        connector.control_plane.update_state_from_dp_metadata(received_control)

    static_hidden = None
    returned_buffers = []
    runner = None
    if role == "attention":
        num_tokens = attention_peer_counts[role_rank]
        static_hidden = torch.full(
            (num_tokens, 16),
            _mtp_hidden_value(role_rank, step_idx=step_idx),
            dtype=torch.bfloat16,
            device="npu:0",
        )
        returned_buffers = [
            torch.empty_like(static_hidden) for _ in range(num_speculative_tokens)
        ]
    else:
        # Load worker modules only in the FFN subprocess, after torch_npu and
        # Ascend ops are initialized. Keep the real runner's graph lifecycle;
        # substitute a small +2 kernel for model weights in this transport test.
        import vllm_ascend.ops  # noqa: F401

        from afd_plugin.v1.worker.npu.ffn_model_runner import AFDNPUFFNModelRunner

        expected_ffn_counts = _ffn_token_counts(
            attention_peer_counts,
            ffn_size=ffn_size,
        )

        class RoundtripFFNRunner(AFDNPUFFNModelRunner):
            def _mtp_ffn_forward(
                self,
                stage_ids,
                *,
                header=None,
                expected_speculative_step=None,
            ):
                assert stage_ids == [0]
                assert header.speculative_step == expected_speculative_step == 0
                if header.num_tokens_across_dp.tolist() != expected_ffn_counts:
                    raise AssertionError("MTP graph FFN count projection mismatch")
                payload = self.connector.recv_attn_output(
                    ubatch_idx=0,
                    layer_idx=0,
                    phase="mtp",
                    speculative_step=header.speculative_step,
                    num_tokens=header.num_tokens,
                )
                output = payload.hidden_states + 2
                self.connector.send_ffn_output(output, payload.context, ubatch_idx=0)
                return output

        runner = object.__new__(RoundtripFFNRunner)
        runner.vllm_config = connector.vllm_config
        runner.speculative_config = connector.vllm_config.speculative_config
        runner.connector = connector
        runner.max_num_tokens = connector.max_num_batched_tokens
        runner.cudagraph_batch_sizes = ()
        runner.use_aclgraph = True
        runner.graph_pool = torch.npu.graph_pool_handle()
        runner._mtp_acl_graphs = {}

    dist.barrier(group=connector.p2p_pg)
    graph = torch.npu.NPUGraph()
    connector.is_graph_capturing = True
    if role == "attention":
        assert static_hidden is not None
        hidden = static_hidden
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
            # Like _run_merged_draft, record all N sends/receives before leaving
            # capture. Feeding each result into the next step detects missing
            # or duplicated iterations, rather than checking only the last send.
            for returned_buffer in returned_buffers:
                context = AFDTransferContext(
                    metadata=AFDTransferMetadata.create_attention_metadata(
                        layer_idx=0,
                        stage_idx=0,
                        seq_len=int(hidden.shape[0]),
                        phase="mtp",
                        speculative_step=0,
                    ),
                )
                connector.send_attn_output(
                    hidden,
                    context,
                    num_tokens_across_dp=torch.tensor(
                        _dp_token_counts(
                            attention_peer_counts,
                            tensor_parallel_size=tensor_parallel_size,
                        ),
                        dtype=torch.int32,
                    ),
                )
                connector.recv_ffn_output(
                    returned_buffer,
                    ubatch_idx=0,
                    phase="mtp",
                )
                hidden = returned_buffer
    else:
        assert runner is not None
        runner._capture_mtp_graphs(control_payload.dp_metadata_list)
    connector.is_graph_capturing = False
    torch.npu.synchronize()
    dist.barrier(group=connector.p2p_pg)

    checks: list[dict[str, object]] = []
    if role == "attention":
        checks.append(
            {
                "phase": "mtp_graph_capture",
                "tokens": attention_peer_counts[role_rank],
                "peer_tokens": attention_peer_counts,
                "captured": True,
                "captured_steps": num_speculative_tokens,
            }
        )
        assert static_hidden is not None
    else:
        checks.append(
            {
                "phase": "mtp_graph_capture",
                "peer_tokens": [
                    end - start
                    for _, start, end in _ffn_peer_shards(
                        attention_peer_counts,
                        ffn_size=ffn_size,
                        ffn_rank=role_rank,
                    )
                ],
                "header_fan_in": True,
                "captured": True,
                "captured_steps": num_speculative_tokens,
                "production_ffn_runner": True,
            }
        )

    dist.barrier(group=connector.p2p_pg)
    # Exercise both duplicate-capture and live replay with fresh inputs. Each
    # replay must consume one complete N-step proposal on each side.
    for replay_idx in range(2):
        if role == "attention":
            assert static_hidden is not None
            static_hidden.fill_(
                _mtp_hidden_value(role_rank, step_idx=step_idx + 7 + replay_idx)
            )
            graph.replay()
        else:
            assert runner is not None
            if replay_idx == 0:
                runner._capture_mtp_graphs(control_payload.dp_metadata_list)
            else:
                runner._execute_mtp_after_target(control_payload.dp_metadata_list)
        torch.npu.synchronize()
        if role == "attention":
            assert static_hidden is not None
            for draft_step, returned_buffer in enumerate(returned_buffers, 1):
                expected = static_hidden + 2 * draft_step
                if not torch.equal(returned_buffer.cpu(), expected.cpu()):
                    raise AssertionError(
                        f"Merged MTP replay mismatch: attention={role_rank}, "
                        f"replay={replay_idx}, draft_step={draft_step}"
                    )
        dist.barrier(group=connector.p2p_pg)

    if role == "attention":
        assert static_hidden is not None
        checks.append(
            {
                "phase": "mtp_graph_replay",
                "tokens": int(static_hidden.shape[0]),
                "updated_input": True,
                "roundtrip": True,
                "replays": 2,
                "steps_per_replay": num_speculative_tokens,
            }
        )
    else:
        checks.append(
            {
                "phase": "mtp_graph_replay",
                "fan_in_out": True,
                "replays": 2,
                "steps_per_replay": num_speculative_tokens,
            }
        )
    return checks


def _worker(
    role: str,
    role_rank: int,
    physical_device: int,
    attention_size: int,
    ffn_size: int,
    port: int,
    stages: int,
    steps: int,
    enable_mtp: bool,
    num_speculative_tokens: int,
    graph_transport: bool,
    graph_multistream: bool,
    mtp_graph_transport: bool,
    tensor_parallel_size: int,
    result_path: Path,
) -> None:
    connector = None
    result: dict[str, object] = {
        "role": role,
        "role_rank": role_rank,
        "physical_device": physical_device,
        "passed": False,
    }
    try:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(physical_device)
        os.environ["HCCL_EXEC_TIMEOUT"] = "0"

        import torch
        import torch.distributed as dist
        import torch_npu  # noqa: F401

        from afd_plugin.config import AFDConfig
        from afd_plugin.connectors.metadata import (
            AFDControlPayload,
            AFDDPMetadata,
            AFDTransferContext,
            AFDTransferMetadata,
        )
        from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector

        torch.npu.set_device(0)
        if graph_transport:
            world_rank = role_rank if role == "ffn" else ffn_size + role_rank
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{port}",
                world_size=attention_size + ffn_size,
                rank=world_rank,
            )
        vllm_config = SimpleNamespace(
            additional_config={"afd": {"connector_extra_config": {}}},
            parallel_config=SimpleNamespace(
                data_parallel_size=(
                    attention_size // tensor_parallel_size
                    if role == "attention"
                    else ffn_size // tensor_parallel_size
                ),
                data_parallel_rank=role_rank // tensor_parallel_size,
                prefill_context_parallel_size=1,
                tensor_parallel_size=tensor_parallel_size,
                num_ubatches=stages,
            ),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
            model_config=SimpleNamespace(
                dtype=torch.bfloat16,
                enforce_eager=not graph_transport,
                hf_config=SimpleNamespace(
                    architectures=["DeepseekV4ForCausalLM"],
                    hidden_size=16,
                    num_hidden_layers=2,
                    vocab_size=128,
                ),
            ),
            speculative_config=(
                SimpleNamespace(
                    method="mtp",
                    enforce_eager=not mtp_graph_transport,
                    num_speculative_tokens=num_speculative_tokens,
                )
                if enable_mtp
                else None
            ),
        )
        afd_config = AFDConfig(
            connector="P2pHcclAFDConnector",
            role=role,
            host="127.0.0.1",
            port=port,
            num_attention_ranks=attention_size,
            num_ffn_ranks=ffn_size,
        )
        connector = P2pHcclAFDConnector(
            rank=role_rank,
            local_rank=0,
            vllm_config=vllm_config,
            afd_config=afd_config,
            role_rank=role_rank,
        )
        connector.init_afd_connector()

        checks: list[dict[str, object]] = []
        for step_idx in range(steps):
            counts_by_stage = {
                stage_idx: _token_counts(
                    attention_size,
                    step_idx=step_idx,
                    stage_idx=stage_idx,
                    tensor_parallel_size=tensor_parallel_size,
                )
                for stage_idx in range(stages)
            }
            control_payload = AFDControlPayload(
                dp_metadata_list={
                    stage_idx: AFDDPMetadata(
                        torch.tensor(
                            _dp_token_counts(
                                counts,
                                tensor_parallel_size=tensor_parallel_size,
                            ),
                            dtype=torch.int32,
                        ),
                    )
                    for stage_idx, counts in counts_by_stage.items()
                },
                is_graph_capturing=False,
                is_warmup=False,
                tensor_parallel_size=tensor_parallel_size,
            )
            if role == "attention":
                connector.control_plane.update_state_from_dp_metadata(
                    control_payload,
                )
                connector.control_plane.send_dp_metadata_list(control_payload)
            else:
                received_control = connector.control_plane.recv_dp_metadata_list()
                connector.control_plane.update_state_from_dp_metadata(
                    received_control,
                )

            for stage_idx, counts in counts_by_stage.items():
                if role == "attention":
                    num_tokens = counts[role_rank]
                    id_values = _input_id_values(
                        role_rank,
                        num_tokens,
                        step_idx=step_idx,
                        stage_idx=stage_idx,
                    )
                    input_ids = torch.tensor(
                        id_values,
                        dtype=torch.int32,
                        device="npu:0",
                    )
                    hidden = torch.full(
                        (num_tokens, 16),
                        _hidden_value(
                            role_rank,
                            step_idx=step_idx,
                            stage_idx=stage_idx,
                        ),
                        dtype=torch.bfloat16,
                        device="npu:0",
                    )
                    connector.send_input_ids(input_ids, ubatch_idx=stage_idx)
                    context = AFDTransferContext(
                        metadata=AFDTransferMetadata.create_attention_metadata(
                            layer_idx=1,
                            stage_idx=stage_idx,
                            seq_len=num_tokens,
                        ),
                    )
                    connector.send_attn_output(hidden, context)
                    returned = connector.recv_ffn_output(
                        ref_tensor=torch.empty_like(hidden),
                        ubatch_idx=stage_idx,
                    )
                    if not torch.equal(returned.cpu(), (hidden + 1).cpu()):
                        raise AssertionError(
                            "round-trip mismatch for "
                            f"attention={role_rank} step={step_idx} "
                            f"stage={stage_idx}",
                        )
                    checks.append(
                        {
                            "step": step_idx,
                            "stage": stage_idx,
                            "tokens": num_tokens,
                            "roundtrip": True,
                        },
                    )
                    continue

                peer_shards = _ffn_peer_shards(
                    counts,
                    ffn_size=ffn_size,
                    ffn_rank=role_rank,
                )
                peer_counts = [end - start for _, start, end in peer_shards]
                aggregate_tokens = sum(peer_counts)
                received_ids = connector.recv_input_ids(
                    aggregate_tokens,
                    ubatch_idx=stage_idx,
                )
                received = connector.recv_attn_output(
                    ubatch_idx=stage_idx,
                    layer_idx=0,
                    input_ids=received_ids,
                )
                expected_ids: list[int] = []
                expected_hidden_values: list[int] = []
                for attention_rank, start, end in peer_shards:
                    id_values = _input_id_values(
                        attention_rank,
                        counts[attention_rank],
                        step_idx=step_idx,
                        stage_idx=stage_idx,
                    )
                    expected_ids.extend(
                        _expected_shard_values(
                            id_values,
                            start=start,
                            end=end,
                            padding_value=0,
                        ),
                    )
                    hidden_values = [
                        _hidden_value(
                            attention_rank,
                            step_idx=step_idx,
                            stage_idx=stage_idx,
                        )
                    ] * counts[attention_rank]
                    expected_hidden_values.extend(
                        _expected_shard_values(
                            hidden_values,
                            start=start,
                            end=end,
                            padding_value=0,
                        )
                    )
                if received.input_ids.cpu().tolist() != expected_ids:
                    raise AssertionError(
                        f"input IDs mismatch for ffn={role_rank} "
                        f"step={step_idx} stage={stage_idx}",
                    )
                actual_hidden_values = received.hidden_states[:, 0].cpu().tolist()
                if actual_hidden_values != expected_hidden_values:
                    raise AssertionError(
                        f"hidden aggregation mismatch for ffn={role_rank} "
                        f"step={step_idx} stage={stage_idx}",
                    )
                if received.context.metadata.seq_lens != peer_counts:
                    raise AssertionError(
                        f"peer lengths mismatch: "
                        f"{received.context.metadata.seq_lens} != {peer_counts}",
                    )
                connector.send_ffn_output(
                    received.hidden_states + 1,
                    received.context,
                    ubatch_idx=stage_idx,
                )
                checks.append(
                    {
                        "step": step_idx,
                        "stage": stage_idx,
                        "peer_tokens": peer_counts,
                        "aggregate_tokens": aggregate_tokens,
                        "ids": True,
                        "hidden": True,
                        "output_split": True,
                    },
                )

            if not enable_mtp or mtp_graph_transport:
                continue

            mtp_counts = _mtp_token_counts(
                attention_size,
                step_idx=step_idx,
                tensor_parallel_size=tensor_parallel_size,
            )
            expected_ffn_counts = _ffn_token_counts(
                mtp_counts,
                ffn_size=ffn_size,
            )
            for speculative_step in range(num_speculative_tokens):
                value_step = step_idx * num_speculative_tokens + speculative_step
                if role == "attention":
                    num_tokens = mtp_counts[role_rank]
                    hidden = torch.full(
                        (num_tokens, 16),
                        _mtp_hidden_value(role_rank, step_idx=value_step),
                        dtype=torch.bfloat16,
                        device="npu:0",
                    )
                    context = AFDTransferContext(
                        metadata=AFDTransferMetadata.create_attention_metadata(
                            layer_idx=0,
                            stage_idx=0,
                            seq_len=num_tokens,
                            phase="mtp",
                            speculative_step=speculative_step,
                        ),
                    )
                    connector.send_attn_output(
                        hidden,
                        context,
                        num_tokens_across_dp=torch.tensor(
                            _dp_token_counts(
                                mtp_counts,
                                tensor_parallel_size=tensor_parallel_size,
                            ),
                            dtype=torch.int32,
                        ),
                    )
                    returned = connector.recv_ffn_output(
                        ref_tensor=torch.empty_like(hidden),
                        ubatch_idx=0,
                        phase="mtp",
                    )
                    if not torch.equal(returned.cpu(), (hidden + 2).cpu()):
                        raise AssertionError(
                            "MTP round-trip mismatch for "
                            f"attention={role_rank} step={step_idx} "
                            f"proposal={speculative_step}"
                        )
                    checks.append(
                        {
                            "phase": "mtp",
                            "step": step_idx,
                            "speculative_step": speculative_step,
                            "tokens": num_tokens,
                            "ffn_tokens": expected_ffn_counts,
                            "roundtrip": True,
                        }
                    )
                    continue

                peer_shards = _ffn_peer_shards(
                    mtp_counts,
                    ffn_size=ffn_size,
                    ffn_rank=role_rank,
                )
                peer_counts = [end - start for _, start, end in peer_shards]
                header = connector.recv_mtp_header(stage_idx=0)
                if header.speculative_step != speculative_step:
                    raise AssertionError(
                        "MTP speculative step mismatch for "
                        f"ffn={role_rank}: {header.speculative_step} != "
                        f"{speculative_step}"
                    )
                if header.num_tokens != sum(peer_counts):
                    raise AssertionError(
                        f"MTP aggregate token mismatch for ffn={role_rank}: "
                        f"{header.num_tokens} != {sum(peer_counts)}"
                    )
                if header.num_tokens_across_dp.tolist() != expected_ffn_counts:
                    raise AssertionError(
                        f"MTP FFN counts mismatch: "
                        f"{header.num_tokens_across_dp.tolist()} != "
                        f"{expected_ffn_counts}"
                    )
                received = connector.recv_attn_output(
                    ubatch_idx=0,
                    layer_idx=0,
                    phase="mtp",
                    speculative_step=header.speculative_step,
                    num_tokens=header.num_tokens,
                )
                expected_hidden_values: list[int] = []
                for attention_rank, start, end in peer_shards:
                    hidden_values = [
                        _mtp_hidden_value(attention_rank, step_idx=value_step)
                    ] * mtp_counts[attention_rank]
                    expected_hidden_values.extend(
                        _expected_shard_values(
                            hidden_values,
                            start=start,
                            end=end,
                            padding_value=0,
                        )
                    )
                actual_hidden_values = received.hidden_states[:, 0].cpu().tolist()
                if actual_hidden_values != expected_hidden_values:
                    raise AssertionError(
                        f"MTP hidden aggregation mismatch for ffn={role_rank} "
                        f"step={step_idx} proposal={speculative_step}"
                    )
                if received.context.metadata.seq_lens != peer_counts:
                    raise AssertionError(
                        f"MTP peer lengths mismatch: "
                        f"{received.context.metadata.seq_lens} != {peer_counts}"
                    )
                connector.send_ffn_output(
                    received.hidden_states + 2,
                    received.context,
                    ubatch_idx=0,
                )
                checks.append(
                    {
                        "phase": "mtp",
                        "step": step_idx,
                        "speculative_step": speculative_step,
                        "peer_tokens": peer_counts,
                        "aggregate_tokens": header.num_tokens,
                        "ffn_tokens": expected_ffn_counts,
                        "header_fan_in": True,
                        "hidden_fan_in": True,
                        "output_split": True,
                    }
                )

        if graph_transport:
            checks.extend(
                _validate_graph_transport(
                    connector=connector,
                    role=role,
                    role_rank=role_rank,
                    attention_size=attention_size,
                    stages=stages,
                    step_idx=2,
                    tensor_parallel_size=tensor_parallel_size,
                    multistream=graph_multistream,
                )
            )
        if mtp_graph_transport:
            checks.extend(
                _validate_mtp_graph_transport(
                    connector=connector,
                    role=role,
                    role_rank=role_rank,
                    attention_size=attention_size,
                    ffn_size=ffn_size,
                    stages=stages,
                    step_idx=2,
                    tensor_parallel_size=tensor_parallel_size,
                    num_speculative_tokens=num_speculative_tokens,
                )
            )

        torch.npu.synchronize()
        result.update(passed=True, checks=checks)
    except BaseException:
        result["error"] = traceback.format_exc()
    finally:
        if connector is not None and connector.is_initialized:
            try:
                connector.close()
            except BaseException:
                result["passed"] = False
                result["close_error"] = traceback.format_exc()
        default_group_closed = True
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
            default_group_closed = not dist.is_initialized()
        except BaseException:
            default_group_closed = False
            result["passed"] = False
            result["default_group_close_error"] = traceback.format_exc()
        result["closed"] = bool(
            connector is not None
            and not connector.is_initialized
            and default_group_closed,
        )
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
        )
    if not result["passed"]:
        raise SystemExit(1)


def _parse_devices(raw: str) -> list[int]:
    try:
        devices = [int(value) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "devices must be a comma-separated integer list",
        ) from exc
    if not devices or any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError("devices must be non-negative")
    return devices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=29831)
    parser.add_argument("--attention-devices", type=_parse_devices, default=[0])
    parser.add_argument("--ffn-devices", type=_parse_devices, default=[8])
    parser.add_argument("--stages", type=int, choices=(1, 2), default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--enable-mtp", action="store_true")
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        choices=(1, 2, 3),
        default=1,
    )
    parser.add_argument("--graph-transport", action="store_true")
    parser.add_argument("--graph-multistream", action="store_true")
    parser.add_argument("--mtp-graph-transport", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    attention_size = len(args.attention_devices)
    ffn_size = len(args.ffn_devices)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.mtp_graph_transport and not args.enable_mtp:
        parser.error("--mtp-graph-transport requires --enable-mtp")
    if args.num_speculative_tokens != 1 and not args.enable_mtp:
        parser.error("--num-speculative-tokens requires --enable-mtp")
    if args.graph_multistream and not args.graph_transport:
        parser.error("--graph-multistream requires --graph-transport")
    if args.graph_multistream and args.stages != 2:
        parser.error("--graph-multistream requires --stages 2")
    if args.mtp_graph_transport and not args.graph_transport:
        parser.error("--mtp-graph-transport requires --graph-transport")
    larger_size = max(attention_size, ffn_size)
    smaller_size = min(attention_size, ffn_size)
    if larger_size % smaller_size != 0:
        parser.error("Attention and FFN counts must have an integer ratio")
    if (
        attention_size % args.tensor_parallel_size != 0
        or ffn_size % args.tensor_parallel_size != 0
    ):
        parser.error("Attention and FFN counts must be divisible by TP size")
    if args.tensor_parallel_size == 2 and attention_size != ffn_size:
        parser.error("TP2 component validation requires equal Attention and FFN counts")
    all_devices = [*args.attention_devices, *args.ffn_devices]
    if len(set(all_devices)) != len(all_devices):
        parser.error("Attention and FFN device lists must not overlap")
    if not _port_is_free(args.port):
        raise RuntimeError(f"port {args.port} is already in use")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    result_paths: dict[tuple[str, int], Path] = {}
    process_specs: list[tuple[str, int, int]] = []
    for role, devices in (
        ("ffn", args.ffn_devices),
        ("attention", args.attention_devices),
    ):
        for role_rank, physical_device in enumerate(devices):
            result_paths[(role, role_rank)] = (
                args.output.parent / f"{role}_{role_rank}_result.json"
            )
            process_specs.append((role, role_rank, physical_device))

    context = mp.get_context("spawn")
    workers = [
        context.Process(
            target=_worker,
            args=(
                role,
                role_rank,
                physical_device,
                attention_size,
                ffn_size,
                args.port,
                args.stages,
                args.steps,
                args.enable_mtp,
                args.num_speculative_tokens,
                args.graph_transport,
                args.graph_multistream,
                args.mtp_graph_transport,
                args.tensor_parallel_size,
                result_paths[(role, role_rank)],
            ),
            name=f"hccl-p2p-{role}-{role_rank}",
        )
        for role, role_rank, physical_device in process_specs
    ]
    for worker in workers:
        worker.start()

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if all(worker.exitcode is not None for worker in workers):
            break
        if any(
            worker.exitcode is not None and worker.exitcode != 0 for worker in workers
        ):
            break
        time.sleep(0.2)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        if worker.is_alive():
            worker.join(timeout=30)

    exit_codes = {worker.name: worker.exitcode for worker in workers}
    results = []
    for role, role_rank, physical_device in process_specs:
        result_path = result_paths[(role, role_rank)]
        if result_path.is_file():
            results.append(json.loads(result_path.read_text()))
        else:
            results.append(
                {
                    "role": role,
                    "role_rank": role_rank,
                    "physical_device": physical_device,
                    "passed": False,
                    "error": "role result file was not written",
                },
            )
    passed = (
        len(results) == len(workers)
        and all(bool(result.get("passed")) for result in results)
        and all(exit_code == 0 for exit_code in exit_codes.values())
    )
    summary = {
        "passed": passed,
        "topology": {
            "attention_size": attention_size,
            "ffn_size": ffn_size,
            "tensor_parallel_size": args.tensor_parallel_size,
            "attention_data_parallel_size": (
                attention_size // args.tensor_parallel_size
            ),
            "ffn_data_parallel_size": ffn_size // args.tensor_parallel_size,
            "ratio": larger_size // smaller_size,
            "direction": "fan_out" if ffn_size > attention_size else "fan_in",
            "attention_devices": args.attention_devices,
            "ffn_devices": args.ffn_devices,
        },
        "port": args.port,
        "stages": args.stages,
        "steps": args.steps,
        "enable_mtp": args.enable_mtp,
        "num_speculative_tokens": args.num_speculative_tokens,
        "graph_transport": args.graph_transport,
        "graph_multistream": args.graph_multistream,
        "mtp_graph_transport": args.mtp_graph_transport,
        "results": results,
        "exit_codes": exit_codes,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
