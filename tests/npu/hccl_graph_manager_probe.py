# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded native manager/phase/cache component with controlled arithmetic.

Uses the real base CudaGraphManager factory, dispatch and replay plus NPUGraph
HCCL operations. No checkpoint, native model/attention, Engram, production Graph
selector or full-model microbatch threads participate.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from afd_plugin.connectors.npu.hccl_graph_pipeline import HcclGraphComputePipeline
    from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank

HIDDEN_SIZE = 16
CAPACITY = 32
LAYERS = (0, 1)
CAPTURE_SIZES = (4, 8)
GROUP_TIMEOUT_SECONDS = 45
TOTAL_TIMEOUT_SECONDS = 300
REAP_TIMEOUT_SECONDS = 10
ROUTER_DELTA = 0.00031
ROUTER_SCALE = 4096
REPLAYS = 3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,2", help="FFN then Attention NPU")
    parser.add_argument("--model", required=True, help="configuration only; no weights")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=TOTAL_TIMEOUT_SECONDS)
    parser.add_argument(
        "--stage-bank",
        action="store_true",
        help="reuse the production tensor bank across layers",
    )
    parser.add_argument(
        "--compute-pipeline",
        action="store_true",
        help="fork controlled arithmetic on the production C6 compute pipeline",
    )
    parser.add_argument(
        "--pipeline-hybrid",
        type=int,
        choices=(0, 1),
        help="C6 stage-ready hybrid policy (default 1 with --compute-pipeline)",
    )
    parser.add_argument("--rank", type=int, choices=(0, 1), help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        args.devices = tuple(int(value) for value in args.devices.split(","))
    except ValueError:
        parser.error("--devices requires two distinct NPU indices")
    if len(args.devices) != 2 or len(set(args.devices)) != 2 or min(args.devices) < 0:
        parser.error("--devices requires two distinct nonnegative NPU indices")
    if not 0 < args.timeout <= TOTAL_TIMEOUT_SECONDS:
        parser.error("invalid bounded timeout")
    if args.rank is not None and args.port is None:
        parser.error("worker requires port")
    if args.compute_pipeline and not args.stage_bank:
        parser.error("--compute-pipeline requires --stage-bank")
    if args.pipeline_hybrid is not None and not args.compute_pipeline:
        parser.error("--pipeline-hybrid requires --compute-pipeline")
    if args.compute_pipeline and args.pipeline_hybrid is None:
        args.pipeline_hybrid = 1
    return args


def payload(torch, rows, valid, coordinate, iteration, previous=None):
    hidden = (
        torch.arange(rows * HIDDEN_SIZE, dtype=torch.float32).reshape(rows, HIDDEN_SIZE)
        / 8
        + coordinate
        + iteration
    ).bfloat16()
    fp32 = hidden.float() + ROUTER_DELTA * (iteration + 1)
    if previous is not None:
        hidden = (hidden.float() + previous.float()).bfloat16()
        fp32 = fp32 + previous.float()
    ids = torch.arange(rows, dtype=torch.int32) + coordinate * 11 + iteration
    ids[valid:] = 0
    expected = (
        hidden.float() + (fp32 - hidden.float()) * ROUTER_SCALE + ids[:, None].float()
    ).bfloat16()
    expected[valid:] = 0
    return hidden, fp32, ids, expected


@dataclass
class GraphBundle:
    exchanges: tuple
    valid_counts: tuple
    row_indices: tuple
    bank: HcclGraphStageBank | None = None
    observations: tuple = ()
    bank_addresses: tuple = ()
    pipeline: HcclGraphComputePipeline | None = None
    produced: tuple = ()
    continuations: tuple = ()


def run_worker(args):
    # Runtime imports are confined to isolated workers.
    import numpy as np
    import torch
    import torch.distributed as dist
    import vllm_ascend.ops  # noqa: F401
    from vllm.config import CUDAGraphMode, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.model_runner import BatchReqState
    from vllm.v1.worker.gpu.ubatch_utils import create_ubatch_slices
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.utils import adapt_patch
    from vllm_ascend.worker.v2.utils import communicator_switch, torch_cuda_wrapper

    from afd_plugin.compat.patches.npu.mrv2_ubatching import use_npu_ubatch_sm_control
    from afd_plugin.connectors.npu.hccl_graph import (
        HcclGraphBuffers,
        HcclGraphEndpoint,
        HcclGraphExchange,
    )
    from afd_plugin.connectors.npu.hccl_graph_cache import HcclGraphCache
    from afd_plugin.connectors.npu.hccl_staged_transport import HcclStagedTransport
    from afd_plugin.distributed.hccl_graph import HcclGraphReport
    from afd_plugin.distributed.hccl_plan import HcclLocalForward, HcclPlanCapabilities
    from afd_plugin.v1.worker.npu.hccl_v41_graph import (
        HcclV41GraphGeometry,
        HcclV41StageGeometry,
    )
    from afd_plugin.v1.worker.npu.hccl_v41_graph_control import HcclV41GraphControl
    from afd_plugin.v1.worker.npu.ubatch_runner import NPUUBatchRunner
    from tests.npu.hccl_staged_probe import build_config

    device = torch.device("npu", args.devices[args.rank])
    role = "ffn" if args.rank == 0 else "attention"
    report = {
        "rank": args.rank,
        "role": role,
        "device": str(device),
        "scope": "native base manager and HCCL phase/cache component",
        "checkpoint_weights_loaded": False,
        "engram_or_native_attention_executed": False,
        "transactions": [],
        "stage_bank": args.stage_bank,
        "compute_pipeline": args.compute_pipeline,
        "pipeline_hybrid": args.pipeline_hybrid,
        "bank_storage": [],
        "pipeline_storage": [],
    }
    transport = cache = None
    manager = None
    input_buffers = None
    active = None
    counter = 0
    generation = 0

    def native_check():
        value = torch.ones(1, device=device)
        dist.all_reduce(value, group=get_tp_group().device_group)
        assert value.item() == 1
        return value.cpu().tolist()

    def prepare_geometry(descriptor, real_tokens):
        # Actually construct the native input batch and call its native slice
        # helper after reservation. These controlled q1 dummy tensors exercise
        # geometry only; no runner/ModelState/history/attention is prepared.
        batch = InputBatch.make_dummy(
            real_tokens, real_tokens, input_buffers, max_query_len=1
        )
        batch = replace(
            batch,
            num_reqs_after_padding=descriptor.num_reqs,
            num_tokens_after_padding=descriptor.num_tokens,
            input_ids=input_buffers.input_ids[: descriptor.num_tokens],
            positions=input_buffers.positions[: descriptor.num_tokens],
            is_padding=input_buffers.is_padding[: descriptor.num_tokens],
        )
        if descriptor.num_ubatches == 1:
            stages = (
                HcclV41StageGeometry(0, descriptor.num_reqs, 0, descriptor.num_tokens),
            )
        else:
            stages = tuple(
                HcclV41StageGeometry(
                    stage.request_slice.start,
                    stage.request_slice.stop,
                    stage.token_slice.start,
                    stage.token_slice.stop,
                )
                for stage in create_ubatch_slices(batch, descriptor.num_ubatches)
            )
        return HcclV41GraphGeometry(descriptor, stages)

    def resolve(candidate):
        if candidate.key is None:
            return HcclGraphReport("unsupported")
        entry = cache.find(candidate.key, generation)
        return HcclGraphReport("ready" if entry is not None else "missing")

    def allocate(key, phase):
        if args.stage_bank:
            from afd_plugin.connectors.npu.hccl_graph_stage import HcclGraphStageBank

            bank = HcclGraphStageBank.allocate(transport, key, device=device)
            exchanges = tuple(exchange for _ in LAYERS for exchange in bank.exchanges)
            # Only the bank is production exchange storage. These separate
            # probe buffers retain each layer's oracle inputs/outputs, so a
            # later layer cannot hide an earlier bad exchange by overwriting it.
            observations = tuple(
                HcclGraphBuffers.allocate(rows, HIDDEN_SIZE, device) if rows else None
                for _ in LAYERS
                for rows in bank.stage_rows
            )
            pipeline, produced, continuations = None, (), ()
            if args.compute_pipeline:
                from afd_plugin.connectors.npu.hccl_graph_pipeline import (
                    HcclGraphComputePipeline,
                )

                pipeline = HcclGraphComputePipeline.allocate(
                    bank, hybrid=bool(args.pipeline_hybrid)
                )
                if role == "attention":
                    produced = tuple(
                        HcclGraphBuffers.allocate(rows, HIDDEN_SIZE, device)
                        if rows
                        else None
                        for _ in LAYERS
                        for rows in bank.stage_rows
                    )
                    continuations = tuple(
                        torch.empty_like(bank.buffers(stage).output) if rows else None
                        for stage, rows in enumerate(bank.stage_rows)
                    )
                report["pipeline_storage"].append(
                    {
                        "key": key.digest,
                        "phase": phase,
                        "hybrid": pipeline.hybrid,
                        "compute_stream": int(pipeline.compute_stream.npu_stream),
                        "positive_steps": len(pipeline.step_events),
                        "tail_event_pairs": len(pipeline.tail_events),
                    }
                )
            tensors = bank_tensors(bank)
            storage_bytes = sum(
                tensor.numel() * tensor.element_size() for tensor in tensors
            )
            assert (
                storage_bytes
                == (8 * HIDDEN_SIZE + 9) * key.num_tokens + 4 * key.num_ubatches
            )
            report["bank_storage"].append({"key": key.digest, "bytes": storage_bytes})
            return GraphBundle(
                exchanges=exchanges,
                valid_counts=(),
                row_indices=(),
                bank=bank,
                observations=observations,
                bank_addresses=tuple(tensor.data_ptr() for tensor in tensors),
                pipeline=pipeline,
                produced=produced,
                continuations=continuations,
            )
        exchanges, valid_counts, row_indices = [], [], []
        for _layer in LAYERS:
            for stage, rows in enumerate(key.padded_tokens):
                endpoint = HcclGraphEndpoint.from_process_group(
                    transport.data_groups[stage], 1 - args.rank
                )
                exchanges.append(
                    HcclGraphExchange(
                        endpoint,
                        HcclGraphBuffers.allocate(rows, HIDDEN_SIZE, device),
                        role=role,
                    )
                    if rows
                    else None
                )
                valid_counts.append(torch.zeros(1, dtype=torch.int32, device=device))
                row_indices.append(torch.arange(rows, device=device))
        return GraphBundle(tuple(exchanges), tuple(valid_counts), tuple(row_indices))

    def bank_tensors(bank):
        payloads = tuple(
            tensor
            for index, rows in enumerate(bank.stage_rows)
            if rows
            for tensor in (
                bank.buffers(index).hidden_states,
                bank.buffers(index).hidden_states_fp32,
                bank.buffers(index).input_ids,
                bank.buffers(index).output,
            )
        )
        return (*payloads, *bank.valid_counts, *bank.row_indices, *bank.padding_masks)

    def refill(bundle, plan, iteration):
        if bundle.bank is not None:
            bundle.bank.update_counts(plan)
        for index, exchange in enumerate(bundle.exchanges):
            if exchange is None:
                continue
            stage = plan.stages[index % len(plan.stages)]
            rows, valid = (
                stage.attention_padded_tokens[0],
                stage.attention_valid_tokens[0],
            )
            if bundle.bank is None:
                bundle.valid_counts[index].fill_(valid)
            if role == "attention":
                values = payload(torch, rows, valid, index, iteration)
                if bundle.bank is None:
                    exchange.update_inputs(
                        *(tensor.to(device) for tensor in values[:3])
                    )
                else:
                    observed = bundle.observations[index]
                    for target, value in zip(
                        (
                            observed.hidden_states,
                            observed.hidden_states_fp32,
                            observed.input_ids,
                        ),
                        values[:3],
                        strict=True,
                    ):
                        target.copy_(value.to(device))
                    # Deliberately dirty padded IDs; the captured bank body,
                    # rather than the external producer, must sanitize them.
                    observed.input_ids[valid:] = 127

    def body(bundle):
        # Tensor-only callback; no JSON, host cursor, COMPLETE or device readback.
        if bundle.pipeline is not None:
            pipeline_body(bundle)
            return
        for index, exchange in enumerate(bundle.exchanges):
            if exchange is None:
                continue
            if bundle.bank is not None:
                bank, observed = bundle.bank, bundle.observations[index]
                stage_index = index % len(bank.stage_rows)
                if role == "attention":
                    output = bank.attention_exchange(
                        stage_index,
                        observed.hidden_states,
                        observed.hidden_states_fp32,
                        observed.input_ids,
                    )
                    observed.output.copy_(output)
                else:
                    inputs = bank.receive_inputs(stage_index)
                    output = (
                        inputs.hidden_states.float()
                        + (inputs.hidden_states_fp32 - inputs.hidden_states.float())
                        * ROUTER_SCALE
                        + inputs.input_ids[:, None].float()
                    ).bfloat16()
                    bank.send_output(stage_index, output)
                for target, source in zip(
                    (
                        observed.hidden_states,
                        observed.hidden_states_fp32,
                        observed.input_ids,
                    ),
                    (
                        exchange.buffers.hidden_states,
                        exchange.buffers.hidden_states_fp32,
                        exchange.buffers.input_ids,
                    ),
                    strict=True,
                ):
                    target.copy_(source)
                if role == "ffn":
                    observed.output.copy_(exchange.buffers.output)
                continue
            if role == "attention":
                exchange.send_inputs()
                exchange.receive_output()
            else:
                exchange.receive_inputs()
                buffers = exchange.buffers
                output = (
                    buffers.hidden_states.float()
                    + (buffers.hidden_states_fp32 - buffers.hidden_states.float())
                    * ROUTER_SCALE
                    + buffers.input_ids[:, None].float()
                ).bfloat16()
                buffers.output.copy_(
                    output.masked_fill(
                        bundle.row_indices[index][:, None]
                        >= bundle.valid_counts[index],
                        0,
                    )
                )
                exchange.send_output()

    def pipeline_body(bundle):
        bank, pipeline = bundle.bank, bundle.pipeline
        positive = tuple(stage for stage, rows in enumerate(bank.stage_rows) if rows)
        with pipeline.run():
            for layer_index, layer in enumerate(LAYERS):
                if role == "attention":
                    for stage in positive:
                        index = layer_index * len(bank.stage_rows) + stage
                        observed = bundle.observations[index]
                        produced = bundle.produced[index]
                        previous = bank.buffers(stage).output if layer_index else None
                        dependencies = (
                            observed.hidden_states,
                            observed.hidden_states_fp32,
                            observed.input_ids,
                        ) + ((previous,) if previous is not None else ())
                        with pipeline.compute(layer, stage, inputs=dependencies):
                            if previous is None:
                                produced.hidden_states.copy_(observed.hidden_states)
                                produced.hidden_states_fp32.copy_(
                                    observed.hidden_states_fp32
                                )
                            else:
                                produced.hidden_states.copy_(
                                    observed.hidden_states.float() + previous.float()
                                )
                                produced.hidden_states_fp32.copy_(
                                    observed.hidden_states_fp32 + previous.float()
                                )
                            produced.input_ids.copy_(observed.input_ids)
                    for stage in positive:
                        index = layer_index * len(bank.stage_rows) + stage
                        produced = bundle.produced[index]
                        with pipeline.attention_exchange(
                            layer,
                            stage,
                            tensors=(
                                produced.hidden_states,
                                produced.hidden_states_fp32,
                                produced.input_ids,
                            ),
                        ):
                            bank.attention_exchange(
                                stage,
                                produced.hidden_states,
                                produced.hidden_states_fp32,
                                produced.input_ids,
                            )
                            observe_bank(bundle, index, stage)
                else:
                    for stage in positive:
                        index = layer_index * len(bank.stage_rows) + stage
                        with pipeline.ffn_receive(layer, stage):
                            inputs = bank.receive_inputs(stage)
                        with pipeline.compute(
                            layer,
                            stage,
                            inputs=(
                                inputs.hidden_states,
                                inputs.hidden_states_fp32,
                                inputs.input_ids,
                            ),
                        ):
                            output = (
                                inputs.hidden_states.float()
                                + (
                                    inputs.hidden_states_fp32
                                    - inputs.hidden_states.float()
                                )
                                * ROUTER_SCALE
                                + inputs.input_ids[:, None].float()
                            ).bfloat16()
                        with pipeline.ffn_send(layer, stage, tensors=(output,)):
                            bank.send_output(stage, output)
                            observe_bank(bundle, index, stage)
            if role == "attention":
                for stage in positive:
                    index = (len(LAYERS) - 1) * len(bank.stage_rows) + stage
                    previous = bank.buffers(stage).output
                    produced = bundle.produced[index].hidden_states
                    continuation = bundle.continuations[stage]
                    with pipeline.final_compute(stage, inputs=(previous, produced)):
                        torch.add(previous, produced, out=continuation)
                        continuation.masked_fill_(
                            bank.padding_masks[stage].unsqueeze(1), 0
                        )
                pipeline.join(
                    outputs=tuple(bundle.continuations[stage] for stage in positive)
                )
            else:
                pipeline.join(
                    outputs=tuple(
                        observed.output
                        for observed in bundle.observations
                        if observed is not None
                    )
                )

    def observe_bank(bundle, index, stage):
        observed, buffers = bundle.observations[index], bundle.bank.buffers(stage)
        for target, source in zip(
            (
                observed.hidden_states,
                observed.hidden_states_fp32,
                observed.input_ids,
                observed.output,
            ),
            (
                buffers.hidden_states,
                buffers.hidden_states_fp32,
                buffers.input_ids,
                buffers.output,
            ),
            strict=True,
        ):
            target.copy_(source)

    def verify(bundle, plan, iteration):
        if bundle.bank is not None:
            assert (
                tuple(tensor.data_ptr() for tensor in bank_tensors(bundle.bank))
                == bundle.bank_addresses
            )
        previous = {}
        last_inputs = {}
        for index, exchange in enumerate(bundle.exchanges):
            if exchange is None:
                continue
            stage = plan.stages[index % len(plan.stages)]
            values = payload(
                torch,
                stage.attention_padded_tokens[0],
                stage.attention_valid_tokens[0],
                index,
                iteration,
                previous.get(stage.stage_idx) if bundle.pipeline is not None else None,
            )
            previous[stage.stage_idx] = values[3]
            last_inputs[stage.stage_idx] = values[0]
            buffers = (
                exchange.buffers if bundle.bank is None else bundle.observations[index]
            )
            for actual, expected in zip(
                (
                    buffers.hidden_states,
                    buffers.hidden_states_fp32,
                    buffers.input_ids,
                    buffers.output,
                ),
                values,
                strict=True,
            ):
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        if bundle.pipeline is not None and role == "attention":
            for stage, output in enumerate(bundle.continuations):
                if output is None:
                    continue
                expected = previous[stage] + last_inputs[stage]
                expected[plan.stages[stage].attention_valid_tokens[0] :] = 0
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    def complete_active():
        nonlocal active
        if active is None:
            return
        desc, decision, bundle, plan = active
        torch.npu.synchronize(device)
        if decision.phase == "warmup":
            verify(bundle, plan, 0)
        if decision.phase == "capture":
            # Native manager has inserted the completed graph by this boundary.
            # Its factory/active tuple retained every dependency until handoff.
            cache.reserve_capture(
                decision.key, generation, manager.graphs[desc], bundle
            )
            cache.publish_capture(decision.key, generation)
        transport.complete_graph_forward(decision.key)
        report["transactions"].append(
            {
                "id": plan.transaction_id,
                "phase": decision.phase,
                "key": decision.key.digest,
            }
        )
        active = None

    def eager(plan, iteration):
        for layer_index, layer in enumerate(LAYERS):
            for stage in plan.stages:
                rows = stage.attention_padded_tokens[0]
                if not rows:
                    continue
                coordinate = layer_index * len(plan.stages) + stage.stage_idx
                values = payload(
                    torch, rows, stage.attention_valid_tokens[0], coordinate, iteration
                )
                if role == "attention":
                    transport.send_layer_inputs(
                        layer,
                        stage.stage_idx,
                        *(tensor.to(device) for tensor in values[:3]),
                    )
                    output = transport.receive_layer_output(layer, stage.stage_idx)
                else:
                    inputs = transport.receive_layer(layer, stage.stage_idx)
                    for actual, expected in zip(
                        (
                            inputs.hidden_states,
                            inputs.hidden_states_fp32,
                            inputs.input_ids,
                        ),
                        values[:3],
                        strict=True,
                    ):
                        torch.testing.assert_close(
                            actual.cpu(), expected, rtol=0, atol=0
                        )
                    output = values[3].to(device)
                    transport.send_layer_output(layer, stage.stage_idx, output)
                torch.testing.assert_close(output.cpu(), values[3], rtol=0, atol=0)
        transport.finish_forward()

    try:
        current_platform.pre_register_and_update()
        load_general_plugins()
        torch.npu.set_device(device)
        adapt_patch()
        config = build_config(1, args.model)
        config.parallel_config.ubatch_size = 2
        config.parallel_config.dbo_decode_token_threshold = 1
        config.compilation_config.cudagraph_capture_sizes = list(CAPTURE_SIZES)
        config.compilation_config.max_cudagraph_capture_size = max(CAPTURE_SIZES)
        init_ascend_config(config)
        with (
            set_current_vllm_config(config),
            torch_cuda_wrapper(),
            torch.inference_mode(),
        ):
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=device.index,
                distributed_init_method=f"tcp://127.0.0.1:{args.port + 1 + args.rank}",
                backend="hccl",
                timeout=timedelta(seconds=GROUP_TIMEOUT_SECONDS),
            )
            initialize_model_parallel(tensor_model_parallel_size=1, backend="hccl")
            report["native_tp_before"] = native_check()
            transport = HcclStagedTransport(
                role,
                0,
                HcclPlanCapabilities(
                    HIDDEN_SIZE,
                    1024,
                    CAPACITY,
                    LAYERS,
                    1,
                    1,
                    local_counts_version=1,
                    graph_execution_version=1,
                ),
                init_method=f"tcp://127.0.0.1:{args.port}",
                local_device=device.index,
                timeout_seconds=GROUP_TIMEOUT_SECONDS,
            )
            transport.initialize()
            cache = HcclGraphCache(lambda: torch.npu.synchronize(device))
            if role == "attention":
                # Real runner construction supplies native capture streams and
                # candidate U2 twins. No model-state methods or handoff threads
                # are used by this controlled callback (model_state is unused).
                with use_npu_ubatch_sm_control():
                    ubatches = NPUUBatchRunner(
                        config,
                        device,
                        None,
                        [],
                        KVCacheConfig(1, [], []),
                        16,
                        kernel_block_sizes=[],
                        timeout_s=GROUP_TIMEOUT_SECONDS,
                    )
                manager = CudaGraphManager(
                    config,
                    device,
                    CUDAGraphMode.FULL_DECODE_ONLY,
                    1,
                    ubatch_runner=ubatches,
                )
                input_buffers = InputBuffers(CAPACITY, CAPACITY, device)
                control = HcclV41GraphControl(
                    transport, manager, cache, lambda: generation
                )

                def factory(desc, warmup):
                    nonlocal counter, active
                    complete_active()
                    counter += 1
                    reservation = control.reserve_capture(
                        desc,
                        warmup=warmup,
                        transaction_id=f"capture-{counter}",
                    )
                    assert reservation is not None
                    decision = reservation.decision
                    key = decision.key
                    prepared = prepare_geometry(desc, desc.num_tokens)
                    control.verify_prepared(reservation, prepared)
                    plan = transport.submit_forward(
                        HcclLocalForward(
                            "warmup", (0,) * len(key.stage_layouts), key.padded_tokens
                        ),
                        transaction_id=decision.transaction_id,
                    )
                    bundle = allocate(key, decision.phase)
                    refill(bundle, plan, 0)
                    active = desc, decision, bundle, plan
                    return lambda cg_mode: body(bundle)

                with communicator_switch():
                    manager.capture(factory, progress_bar_desc="AFD paired component")
                complete_active()
                captured = tuple(manager.graphs)
                assert len(captured) == len(CAPTURE_SIZES) * 2
                assert {desc.num_ubatches for desc in captured} == {1, 2}
                # Full exact geometry replays, then F generation miss and an A
                # same-bucket/different-request-layout miss both choose eager
                # before native controlled InputBatch construction occurs.
                cases = [(desc, desc.num_tokens, False) for desc in captured]
                cases.extend([(captured[-1], captured[-1].num_tokens, True)])
                cases.append(
                    (
                        next(
                            desc
                            for desc in captured
                            if desc.num_tokens == 4 and desc.num_ubatches == 2
                        ),
                        2,
                        False,
                    )
                )
                for case_index, (wanted, real_tokens, force_ffn_miss) in enumerate(
                    cases
                ):
                    config.parallel_config.dbo_decode_token_threshold = (
                        1 if wanted.num_ubatches == 2 else CAPACITY
                    )
                    state = BatchReqState(
                        req_ids=[str(index) for index in range(real_tokens)],
                        num_tokens=real_tokens,
                        num_scheduled_tokens=np.ones(real_tokens, dtype=np.int32),
                        idx_mapping_np=np.arange(real_tokens, dtype=np.int32),
                        prefill_len_np=np.zeros(real_tokens, dtype=np.int32),
                        num_computed_prefill_tokens_np=np.zeros(
                            real_tokens, dtype=np.int32
                        ),
                        is_prefilling_np=np.zeros(real_tokens, dtype=np.bool_),
                        has_prefill=False,
                    )
                    fixed_replay_key = None
                    for iteration in range(REPLAYS):
                        transaction = f"replay-{case_index}-{iteration}"
                        # This flag is an explicit probe schedule communicated
                        # as iteration identity; production has no bypass knob.
                        if force_ffn_miss:
                            transaction += "-ffn-generation-miss"
                        reservation = control.reserve_batch(
                            state,
                            parallel_config=config.parallel_config,
                            allow_ubatching=wanted.num_ubatches == 2,
                            max_query_len=1,
                            transaction_id=transaction,
                        )
                        assert reservation is not None
                        decision, geometry = reservation.decision, reservation.geometry
                        key = decision.key
                        prepared = prepare_geometry(geometry.descriptor, real_tokens)
                        control.verify_prepared(reservation, prepared)
                        controlled_valid = real_tokens
                        if (
                            args.compute_pipeline
                            and not force_ffn_miss
                            and real_tokens == wanted.num_tokens
                        ):
                            # Keep the native dummy's physical request/token
                            # geometry fixed. Only this controlled component's
                            # device-valid scalars and oracle payloads vary.
                            controlled_valid = (
                                real_tokens,
                                1,
                                real_tokens - 1,
                            )[iteration]
                            assert decision.phase == "replay"
                            assert geometry.descriptor == wanted
                            if fixed_replay_key is None:
                                fixed_replay_key = key
                            assert key == fixed_replay_key
                        valid = tuple(
                            max(
                                0,
                                min(stage.token_stop, controlled_valid)
                                - stage.token_start,
                            )
                            for stage in geometry.stages
                        )
                        plan = transport.submit_forward(
                            HcclLocalForward("target", valid, decision.padded_tokens),
                            transaction_id=transaction,
                        )
                        assert (
                            tuple(
                                stage.attention_valid_tokens[0] for stage in plan.stages
                            )
                            == valid
                        )
                        assert sum(valid) == controlled_valid
                        if force_ffn_miss or real_tokens != wanted.num_tokens:
                            assert decision.phase == "eager"
                        if decision.phase == "eager":
                            eager(plan, iteration)
                        else:
                            entry = cache.find(key, generation)
                            assert entry is not None
                            refill(entry.resources, plan, iteration)
                            manager.run_fullgraph(geometry.descriptor)
                            torch.npu.synchronize(device)
                            verify(entry.resources, plan, iteration)
                            transport.complete_graph_forward(key)
                        report["transactions"].append(
                            {
                                "id": transaction,
                                "phase": decision.phase,
                                "reason": decision.reason,
                                "native_input_batch_geometry_verified": True,
                                "key": key.digest if key is not None else None,
                                "controlled_valid_tokens": controlled_valid,
                                "valid_tokens_by_stage": valid,
                                "padded_tokens_by_stage": decision.padded_tokens,
                            }
                        )
                transport.stop_attention()
            else:
                while True:
                    # Transaction IDs are intentionally not visible to the
                    # resolver. Alternate the local generation only through a
                    # known component schedule after the captured replays.
                    target_count = sum(
                        r["id"].startswith("replay-") for r in report["transactions"]
                    )
                    miss_start = len(CAPTURE_SIZES) * 2 * REPLAYS
                    generation = int(miss_start <= target_count < miss_start + REPLAYS)
                    decision = transport.receive_graph_reservation(resolve)
                    if decision is None:
                        break
                    plan = transport.receive_forward()
                    assert plan is not None
                    iteration = (
                        int(decision.transaction_id.split("-")[2])
                        if decision.transaction_id.startswith("replay-")
                        else 0
                    )
                    if decision.phase == "eager":
                        eager(plan, iteration)
                    elif decision.phase == "replay":
                        entry = cache.find(decision.key, generation)
                        assert entry is not None
                        refill(entry.resources, plan, iteration)
                        entry.graph.replay()
                        torch.npu.synchronize(device)
                        verify(entry.resources, plan, iteration)
                        transport.complete_graph_forward(decision.key)
                    else:
                        bundle = allocate(decision.key, decision.phase)
                        refill(bundle, plan, 0)
                        if decision.phase == "warmup":
                            body(bundle)
                        else:
                            graph = torch.npu.NPUGraph()
                            cache.reserve_capture(
                                decision.key, generation, graph, bundle
                            )
                            with torch.npu.graph(graph):
                                body(bundle)
                            cache.publish_capture(decision.key, generation)
                        torch.npu.synchronize(device)
                        if decision.phase == "warmup":
                            verify(bundle, plan, 0)
                        transport.complete_graph_forward(decision.key)
                    report["transactions"].append(
                        {
                            "id": decision.transaction_id,
                            "phase": decision.phase,
                            "reason": decision.reason,
                            "key": decision.key.digest
                            if decision.key is not None
                            else None,
                            "controlled_valid_tokens": sum(
                                stage.attention_valid_tokens[0] for stage in plan.stages
                            ),
                            "valid_tokens_by_stage": tuple(
                                stage.attention_valid_tokens[0] for stage in plan.stages
                            ),
                            "padded_tokens_by_stage": decision.padded_tokens,
                        }
                    )
            cache.close()
            if manager is not None:
                manager.graphs.clear()
            transport.close()
            report["native_tp_after_close"] = native_check()
            report["status"] = "passed"
    except BaseException as exc:
        if cache is not None:
            cache.fail()
        if transport is not None:
            transport.terminal = True
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
        raise
    finally:
        (args.output_dir / f"rank-{args.rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        # Failure cleanup is left to this probe's bounded supervisor; a device
        # exception does not make a graph/group destruct sequence safe to retry.
        if report.get("status") == "passed":
            destroy_model_parallel()
            destroy_distributed_environment()


def run_supervisor(args):
    from tests.npu.hccl_staged_probe import select_port_block

    args.output_dir.mkdir(parents=True, exist_ok=True)
    port = select_port_block(3)
    processes, handles = [], []
    started = time.monotonic()
    summary = {
        "scope": __doc__,
        "exit_codes": [],
        "supervisor_cleanup": [],
        "stage_bank": args.stage_bank,
        "compute_pipeline": args.compute_pipeline,
        "pipeline_hybrid": args.pipeline_hybrid,
    }
    try:
        for rank in range(2):
            handle = (args.output_dir / f"rank-{rank}.log").open("wb")
            handles.append(handle)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--devices",
                ",".join(map(str, args.devices)),
                "--model",
                args.model,
                "--output-dir",
                str(args.output_dir),
                "--rank",
                str(rank),
                "--port",
                str(port),
            ]
            if args.stage_bank:
                command.append("--stage-bank")
            if args.compute_pipeline:
                command.extend(
                    [
                        "--compute-pipeline",
                        "--pipeline-hybrid",
                        str(args.pipeline_hybrid),
                    ]
                )
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        while any(process.poll() is None for process in processes):
            failed = any(process.poll() not in (None, 0) for process in processes)
            if failed or time.monotonic() - started > args.timeout:
                summary["failure"] = "worker failure" if failed else "bounded timeout"
                break
            time.sleep(0.1)
    finally:
        for process in processes:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                    summary["supervisor_cleanup"].append(
                        {"pid": process.pid, "signal": "SIGTERM"}
                    )
        for process in processes:
            try:
                process.wait(timeout=REAP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                    summary["supervisor_cleanup"].append(
                        {"pid": process.pid, "signal": "SIGKILL"}
                    )
                process.wait(timeout=REAP_TIMEOUT_SECONDS)
        for handle in handles:
            handle.close()
        summary["exit_codes"] = [process.returncode for process in processes]
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["status"] = (
            "passed"
            if summary["exit_codes"] == [0, 0] and "failure" not in summary
            else "failed"
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    options = parse_args()
    if options.rank is None:
        raise SystemExit(run_supervisor(options))
    run_worker(options)
