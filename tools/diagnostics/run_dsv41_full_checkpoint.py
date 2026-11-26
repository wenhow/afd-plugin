#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Qualify real FP8/FP4 weights and CPU Engram in standalone or PD modes.

Default execution writes a reviewable launch plan only. --run performs the
locked preflight and owns the services through completion and cleanup. A PD
producer stays alive until its explicit stop file appears on that node.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import signal
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import dsv41_repeatability as repeatability_check
import run_dsv41_a5_afd as launch
from collect_dsv41_dspark_requests import validate_response
from preflight_dsv41_a5_afd import DEFAULT_PROFILE, SourceIdentity

MODES = ("eager-u1", "eager-u2", "graph-u1", "graph-u2")
SPEC_TOKEN_CHOICES = (0, 1, 2, 5)
PD_MODULE = "afd_plugin.compat.npu.kv_layout_connector"
CANCEL_OUTPUT_TOKENS = 256
EP_FIRST_WAVE_TOKEN_INDEX = 2
EP_SCHEDULE_LINE = re.compile(
    r"DSV41 EP scheduled tokens: dp_rank=(\d+) requests=(\{[^}]*\})"
)
DEFAULT_KV_PORT = 14579
TP4_HEAD_ADAPTER = "tp4-query-16-to-64-v1"
FULL_CHECKPOINT_PROFILE = replace(
    DEFAULT_PROFILE,
    sources=DEFAULT_PROFILE.sources
    + (
        SourceIdentity(
            "vllm",
            "config/engram.py",
            "34043a22385a041d4e9be8857f64174c990c2ca8308b97b6a952b5ab96c20ee5",
        ),
        SourceIdentity(
            "vllm",
            "engine/arg_utils.py",
            "ead004ca266419cd25f54a7aefc0e20e1eb046f363b7f5cc977e14f52ee44bb1",
        ),
        SourceIdentity(
            "vllm_ascend",
            "worker/v2/aclgraph_utils.py",
            "b2271eb4aab5876e85370897236a99ece7e9a9541b05f2fc0dd923b32bb8193d",
        ),
        SourceIdentity(
            "vllm_ascend",
            "worker/worker.py",
            "51cfb3da18046af8c72ccefe940de5f24d9502bfdae75b941e1798c0a165cda4",
        ),
        SourceIdentity(
            "vllm_ascend",
            "distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py",
            "a6a9c3a870b7d577fb2a39435067fe378646b6ab39bc669140641beba92bdf84",
        ),
    ),
)
PROMPTS = (
    "Explain how to keep a simple daily work log.",
    "Continue this number sequence and explain the rule: 2, 4, 8, 16.",
    "Read these words and describe the pattern: " + "red blue green " * 70,
)


def set_option(argv: list[str], name: str, value: str) -> None:
    if name in argv:
        argv[argv.index(name) + 1] = value
    else:
        argv.extend((name, value))


def make_plan(
    model: Path,
    vendor: Path,
    repo: Path,
    *,
    mode: str = "eager-u1",
    spec_tokens: int = 0,
    native_control: bool = False,
    expert_parallel: bool = False,
    pd_role: str | None = None,
    engine_id: str | None = None,
    attention_devices: tuple[int, ...] = (0, 1, 2, 3),
    ffn_devices: tuple[int, ...] = (4, 5, 6, 7),
    bind_host: str = "127.0.0.1",
    kv_ip: str = "127.0.0.1",
    port_offset: int = 0,
    stream_overlap: bool = True,
    repeatability: bool = False,
    warmup_rounds: int = repeatability_check.DEFAULT_WARMUP_ROUNDS,
    repeat_rounds: int = repeatability_check.DEFAULT_REPEAT_ROUNDS,
    repeat_logprobs: int = 0,
    trace_native_prefill: bool = False,
    ep_diagnostics: bool = False,
    ep_functional_only: bool = False,
    attention_stop_signal: str | None = None,
    ep_serial_first_wave: bool = False,
    ep_shutdown_smoke: bool = False,
) -> dict:
    if mode not in MODES or spec_tokens not in SPEC_TOKEN_CHOICES:
        raise ValueError("unsupported full-checkpoint qualification mode")
    if native_control and (
        mode != "eager-u1" or (spec_tokens and not expert_parallel) or pd_role
    ):
        raise ValueError("native control requires eager U1 without PD")
    if expert_parallel and trace_native_prefill:
        raise ValueError("EP qualification requires native prefill trace off")
    if trace_native_prefill and not (native_control and repeatability):
        raise ValueError("native prefill trace requires native repeatability")
    ep_u1_diagnostics = expert_parallel and not (
        spec_tokens or pd_role or repeatability
    )
    if ep_functional_only and (
        not expert_parallel
        or native_control
        or pd_role == "producer"
        or repeatability
        or ep_shutdown_smoke
    ):
        raise ValueError("EP functional validation requires an AFD EP request matrix")
    graph, u2 = mode.startswith("graph"), mode.endswith("u2")
    ep_graph_trace = graph and (ep_functional_only or (ep_u1_diagnostics and not u2))
    trace_shutdown = ep_diagnostics or ep_graph_trace or ep_functional_only
    if ep_diagnostics and (
        not ep_u1_diagnostics or mode not in ("eager-u1", "graph-u1")
    ):
        raise ValueError(
            "EP diagnostics require standalone EP eager/U1 or Graph/U1 with N=0"
        )
    if attention_stop_signal is not None and (
        not trace_shutdown
        or attention_stop_signal not in ("SIGINT", "SIGTERM", "SIGUSR2")
    ):
        raise ValueError(
            "attention stop signal requires EP functional validation, EP diagnostics "
            "or standalone EP Graph/U1 with N=0, and SIGINT, SIGTERM or SIGUSR2"
        )
    if ep_serial_first_wave and (not ep_diagnostics or mode != "eager-u1"):
        raise ValueError("serial first wave requires eager/U1 EP diagnostics")
    if ep_shutdown_smoke and (
        not ep_diagnostics or mode != "eager-u1" or ep_serial_first_wave
    ):
        raise ValueError(
            "EP shutdown smoke requires eager/U1 EP diagnostics "
            "without serial first wave"
        )
    if pd_role == "producer" and mode != "eager-u1":
        raise ValueError("PD producer qualification uses eager U1")
    if pd_role not in (None, "producer", "consumer"):
        raise ValueError("PD role must be producer or consumer")
    if repeatability and (
        mode not in ("eager-u1", "graph-u1")
        or spec_tokens
        or pd_role
        or warmup_rounds < repeatability_check.MIN_WARMUP_ROUNDS
        or repeat_rounds < repeatability_check.MIN_REPEAT_ROUNDS
        or repeat_logprobs not in (0, repeatability_check.LOGPROB_CANDIDATES)
    ):
        raise ValueError(
            "repeatability requires standalone eager/U1 or Graph/U1 without DSpark, "
            "at least one warmup round and two measured rounds, "
            "and --repeat-logprobs 0 or 5"
        )
    plan = launch.make_plan(model, vendor, repo)
    selected = attention_devices if native_control else attention_devices + ffn_devices
    if (
        len(attention_devices) != 4
        or len(ffn_devices) != 4
        or any(type(device) is not int or device < 0 for device in selected)
        or len(set(selected)) != len(selected)
        or any(
            not 0 < port + port_offset < 65536
            for port in (*plan["ports"], DEFAULT_KV_PORT + 3)
        )
    ):
        raise ValueError("require four distinct devices per role and valid ports")
    port_options = ("--port", "--master-port", "--data-parallel-rpc-port")
    stream_overlap = stream_overlap and not (expert_parallel and u2)
    width = spec_tokens + 1
    for role, settings in plan["roles"].items():
        argv, env = settings["argv"], settings["environment"]
        devices = attention_devices if role == "attention" else ffn_devices
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices))
        env["VLLM_PORT"] = str(int(env["VLLM_PORT"]) + port_offset)
        for option in port_options:
            if option in argv:
                set_option(
                    argv, option, str(int(argv[argv.index(option) + 1]) + port_offset)
                )
        if role == "attention":
            set_option(argv, "--host", bind_host)
            if trace_shutdown:
                argv[argv.index("vllm.entrypoints.cli.main")] = (
                    "afd_plugin.entrypoints.dsv41_ep_stack"
                )
                if attention_stop_signal == "SIGUSR2":
                    env["AFD_DSV41_EP_SHUTDOWN_RELAY"] = "1"
            if ep_diagnostics:
                set_option(argv, "--logprobs-mode", "raw_logprobs")
                set_option(
                    argv,
                    "--scheduler-cls",
                    "afd_plugin.compat.npu.dsv41_ep_diagnostics."
                    "DSV41EPDiagnosticScheduler",
                )
        if ep_diagnostics or ep_functional_only:
            env["AFD_DSV41_EP_DIAGNOSTICS"] = "1"
        if ep_graph_trace:
            env[launch.EP_GRAPH_TRACE_ENV] = "1"
        env.update(AFD_DSV41_EAGER_QUALIFICATION="0", AFD_DSV41_FULL_QUALIFICATION="1")
        if expert_parallel:
            set_option(argv, "--tensor-parallel-size", "1")
            set_option(argv, "--data-parallel-size", "4")
            set_option(argv, "--data-parallel-size-local", "4")
            argv[argv.index("--no-enable-expert-parallel")] = "--enable-expert-parallel"
            set_option(argv, "--all2all-backend", "flashinfer_all2allv")
            set_option(argv, "--engram-config.dp_shared_memory", "true")
        env["AFD_HCCL_STAGE_DIAGNOSTICS"] = "1"
        for feature in (
            "EAGER_U2_STREAM_OVERLAP",
            "GRAPH_U2_COMPUTE_OVERLAP",
            "GRAPH_U2_HYBRID_DAG",
            "GRAPH_U2_ATTENTION_THREE_STREAM",
            "GRAPH_U2_FFN_RECV_STREAM",
            "GRAPH_U2_FFN_CROSS_LAYER",
        ):
            env["AFD_HCCL_" + feature] = str(int(stream_overlap))
        env["AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS"] = "0"
        set_option(
            argv,
            "--max-num-seqs",
            str(repeatability_check.MAX_ACTIVE_REQUESTS) if repeatability else "2",
        )
        if repeatability:
            set_option(argv, "--logprobs-mode", "raw_logprobs")
        set_option(argv, "--max-num-batched-tokens", "128")
        argv[argv.index("--no-enable-chunked-prefill")] = "--enable-chunked-prefill"
        if u2:
            argv[argv.index("--no-enable-dbo")] = "--enable-dbo"
            set_option(argv, "--ubatch-size", "2")
            argv.extend(
                (
                    "--dbo-decode-token-threshold",
                    "2",
                    "--dbo-prefill-token-threshold",
                    "2",
                )
            )
        if graph:
            argv.remove("--enforce-eager")
            set_option(
                argv,
                "--compilation-config",
                json.dumps(
                    {
                        "mode": 0,
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [width, 2 * width],
                        "pass_config": {"enable_sp": False},
                    }
                ),
            )
        if spec_tokens:
            set_option(
                argv,
                "--speculative-config",
                json.dumps(
                    {
                        "method": "dspark",
                        "model": str(model),
                        "num_speculative_tokens": spec_tokens,
                        "enforce_eager": True,
                        "enable_adaptive_verification": False,
                    }
                ),
            )
            set_option(argv, "--per-request-spec-decode-metrics", "detailed")
        additional = json.loads(argv[argv.index("--additional-config") + 1])
        additional["afd"]["port"] += port_offset
        additional["multistream_engram_overlap"] = graph or u2
        if graph and u2 and stream_overlap:
            additional["afd"]["connector_extra_config"]["graph_exchange_order"] = (
                "cross_layer_v1"
            )
        if pd_role:
            if role == "attention":
                # Native Mooncake binds and advertises get_ip(), which reads
                # VLLM_HOST_IP independently of the KVTransferConfig field.
                env["VLLM_HOST_IP"] = kv_ip
            set_option(
                argv,
                "--kv-transfer-config",
                json.dumps(
                    {
                        "kv_connector": "MooncakeHybridConnector",
                        "kv_connector_module_path": PD_MODULE,
                        "kv_role": "kv_" + pd_role,
                        "engine_id": engine_id or "dsv41-" + pd_role,
                        "kv_buffer_device": "npu",
                        "kv_ip": kv_ip,
                        "kv_port": DEFAULT_KV_PORT + port_offset,
                        "kv_connector_extra_config": {
                            side: {
                                "tp_size": 1 if expert_parallel else 4,
                                "dp_size": 4 if expert_parallel else 1,
                                "pp_size": 1,
                            }
                            for side in ("prefill", "decode")
                        },
                    }
                ),
            )
        if native_control and role == "attention":
            additional.pop("afd")
            env["AFD_DSV41_FULL_QUALIFICATION"] = "0"
            if expert_parallel and not spec_tokens:
                worker_class = "vllm_ascend.worker.worker.NPUWorker"
            else:
                if expert_parallel:
                    worker = "DSV41NativeEPDSparkControlWorker"
                elif trace_native_prefill:
                    worker = "DSV41NativeTP4PrefillTraceWorker"
                else:
                    worker = "DSV41NativeTP4ControlWorker"
                worker_class = f"afd_plugin.v1.worker.npu.dsv41_native_worker.{worker}"
            set_option(argv, "--worker-cls", worker_class)
        set_option(argv, "--additional-config", json.dumps(additional))
    if native_control:
        plan["roles"].pop("ffn")
    plan.update(
        scope="full-checkpoint qualification; hardware acceptance pending",
        mode=mode,
        spec_tokens=spec_tokens,
        native_control=native_control,
        pd_role=pd_role,
        native_tp_size=1 if expert_parallel else 4,
        native_dp_size=4 if expert_parallel else 1,
        expert_parallel=expert_parallel,
        attention_head_adapter=None if expert_parallel else TP4_HEAD_ADAPTER,
        cpu_engram=True,
        required_devices=4 if native_control else 8,
        physical_devices=list(selected),
        api_url=f"http://127.0.0.1:{launch.API_PORT + port_offset}",
        ports=[port + port_offset for port in plan["ports"]],
        full_model_qualification=False,
        stream_overlap=stream_overlap,
        request_suite=(
            "ep-shutdown-smoke"
            if ep_shutdown_smoke
            else "repeatability"
            if repeatability
            else "ep-rank-matrix-v2"
            if expert_parallel
            else "matrix"
        ),
    )
    if trace_native_prefill:
        plan["trace_native_prefill"] = True
    if ep_diagnostics:
        plan["ep_diagnostics"] = True
        plan["ep_serial_first_wave"] = ep_serial_first_wave
    if ep_functional_only:
        plan["ep_functional_only"] = True
    if trace_shutdown:
        plan["attention_stop_signal"] = attention_stop_signal or "SIGINT"
    if repeatability:
        plan["repeatability_protocol"] = {
            "version": repeatability_check.PROTOCOL_VERSION,
            "warmup_rounds": warmup_rounds,
            "repeat_rounds": repeat_rounds,
            "prompts": list(PROMPTS[:2]),
            "max_num_seqs": repeatability_check.MAX_ACTIVE_REQUESTS,
            "client_concurrency": repeatability_check.MAX_ACTIVE_REQUESTS,
            "temperature": 0,
            "seed": 0,
            "output_tokens": launch.REQUEST_TOKENS,
            "logprobs": repeat_logprobs,
            "logprobs_mode": "raw_logprobs",
            "return_tokens_as_token_ids": bool(repeat_logprobs),
            "cancel_output_tokens": CANCEL_OUTPUT_TOKENS,
        }
    if pd_role:
        plan["ports"].extend(
            range(DEFAULT_KV_PORT + port_offset, DEFAULT_KV_PORT + port_offset + 4)
        )
    return plan


def prefill_request(
    directory: Path,
    url: str,
    payload: dict,
    *,
    data_parallel_rank: int | None = None,
) -> dict:
    request = {
        **payload,
        "max_tokens": 1,
        "stream": False,
        "kv_transfer_params": {"do_remote_decode": True},
    }
    launch.save_json(directory / "prefill-request.json", request)
    headers = {"Content-Type": "application/json"}
    if data_parallel_rank is not None:
        headers["X-data-parallel-rank"] = str(data_parallel_rank)
        launch.save_json(
            directory / "prefill-routing.json", {"dp_rank": data_parallel_rank}
        )
    with launch.open_service_request(
        urllib.request.Request(
            url.rstrip("/") + "/v1/completions",
            data=json.dumps(request).encode(),
            headers=headers,
        ),
        timeout=launch.REQUEST_TIMEOUT_SECONDS,
    ) as reply:
        response = json.load(reply)
    launch.save_json(directory / "prefill-response.json", response)
    params = response["kv_transfer_params"]
    if "afd_engram_history_v1" not in params:
        raise ValueError("P did not export original Engram history")
    return {**params, "do_remote_prefill": True, "do_remote_decode": False}


def record_cold_ep_first_wave(directory: Path, wave_size: int) -> None:
    """Record the requested rank, observed scheduler batches and third-token scores."""
    index = EP_FIRST_WAVE_TOKEN_INDEX
    samples = []
    for request_index in range(wave_size):
        target = directory / f"request-{request_index:02d}"
        sample = json.loads((target / "request-routing.json").read_bytes())
        comparison = json.loads((target / "token-comparison.json").read_bytes())
        sample.update(
            request=target.name,
            prompt_token_ids_match=comparison["prompt_token_ids_match"],
            first_output_mismatch=comparison["first_output_mismatch"],
        )
        for name, filename in (
            ("actual", "response.json"),
            ("reference", "reference-response.json"),
        ):
            path = target / filename
            if path.is_file():
                response = json.loads(path.read_bytes())
                tokens = response["choices"][0]["token_ids"]
                sample[name] = {
                    "response_id": response.get("id"),
                    "prefix_token_ids": tokens[:index],
                    "selected_token_id": tokens[index],
                    "logprobs": repeatability_check.token_logprobs_at(response, index),
                }
        samples.append(sample)

    scheduler_batches: dict[str, list[dict[str, int]]] = {}
    unreadable_scheduler_lines = []
    log = directory / "attention-server.log"
    if log.is_file():
        with log.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                matches = EP_SCHEDULE_LINE.findall(line)
                if not matches and "DSV41 EP scheduled tokens:" in line:
                    unreadable_scheduler_lines.append(line_number)
                for rank, scheduled in matches:
                    try:
                        batch = ast.literal_eval(scheduled)
                    except (SyntaxError, ValueError):
                        unreadable_scheduler_lines.append(line_number)
                    else:
                        if isinstance(batch, dict):
                            scheduler_batches.setdefault(rank, []).append(batch)
                        else:
                            unreadable_scheduler_lines.append(line_number)
    launch.save_json(
        directory / "ep-first-wave.json",
        {
            "token_index": index,
            "requests": samples,
            "scheduler_batches": scheduler_batches,
            "unreadable_scheduler_lines": unreadable_scheduler_lines,
        },
    )


def run_requests(
    directory: Path,
    reference: Path | None,
    *,
    spec_tokens: int,
    prefill_url: str | None,
    api_url: str = f"http://127.0.0.1:{launch.API_PORT}",
    data_parallel_size: int = 1,
    ep_diagnostics: bool = False,
    ep_serial_first_wave: bool = False,
    return_mismatch_report: bool = False,
) -> dict:
    collect_mismatches = ep_diagnostics or return_mismatch_report

    def one(
        index: int, prompt: str, dp_rank: int | None = None, *, warmup: bool = False
    ) -> dict:
        parent = directory / "first-wave-warmup" if warmup else directory
        target = parent / f"request-{index:02d}"
        target.mkdir()
        payload = {
            "model": "dsv41-afd",
            "prompt": prompt,
            "temperature": 0,
            "seed": 0,
            "max_tokens": launch.REQUEST_TOKENS,
            "ignore_eos": True,
            "n": 1,
            "return_token_ids": True,
            "stream": False,
        }
        if ep_diagnostics:
            payload.update(
                logprobs=repeatability_check.LOGPROB_CANDIDATES,
                return_tokens_as_token_ids=True,
            )
        if prefill_url:
            # Rotate P relative to D to exercise per-replica KV ownership.
            prefill_rank = (
                (dp_rank + 1) % data_parallel_size if dp_rank is not None else None
            )
            payload["kv_transfer_params"] = prefill_request(
                target, prefill_url, payload, data_parallel_rank=prefill_rank
            )
        baseline = (
            reference / target.name / "response.json"
            if reference and not warmup
            else None
        )
        result = launch.send_request(
            target,
            baseline,
            payload=payload,
            api_url=api_url,
            data_parallel_rank=dp_rank,
            collect_token_mismatch=collect_mismatches,
        )
        if ep_diagnostics:
            repeatability_check.validate_logprobs(
                json.loads((target / "response.json").read_bytes())
            )
        if spec_tokens:
            result["dspark"] = validate_response(
                json.loads((target / "response.json").read_bytes()),
                spec_tokens=spec_tokens,
                max_tokens=launch.REQUEST_TOKENS,
            )["speculative_decoding"]
        return result

    results = []
    directed = data_parallel_size > 1
    request_index = 0
    if ep_serial_first_wave:
        (directory / "first-wave-warmup").mkdir()
        for rank in range(data_parallel_size):
            one(rank, PROMPTS[0], rank, warmup=True)
    # EP waves use one then two requests per replica to exercise U1 and U2.
    # Repeating the same prompt across ranks keeps decode lengths comparable.
    for wave, prompt in enumerate(PROMPTS[:2]):
        wave_size = data_parallel_size * (wave + 1) if directed else 2
        prompts = (prompt,) * wave_size if directed else PROMPTS[:2]
        if wave == 0 and ep_serial_first_wave:
            results.extend(
                one(rank, prompt, rank) for rank in range(data_parallel_size)
            )
        else:
            with ThreadPoolExecutor(max_workers=wave_size) as pool:
                futures = [
                    pool.submit(
                        one,
                        request_index + i,
                        text,
                        i % data_parallel_size if directed else None,
                    )
                    for i, text in enumerate(prompts)
                ]
                results.extend(future.result() for future in futures)
        request_index += wave_size
        if wave == 0:
            if ep_diagnostics and directed and not ep_serial_first_wave:
                record_cold_ep_first_wave(directory, wave_size)
            cancel_stream(directory, prefill_url=prefill_url, api_url=api_url)
    results.append(one(request_index, PROMPTS[2], 1 if directed else None))
    report = {"passed": True, "requests": results, "hardware_acceptance": False}
    if collect_mismatches:
        report["passed"] = all(result["passed"] for result in results)
    if spec_tokens:
        metrics = [result["dspark"] for result in results]
        accepted = sum(item["num_accepted_draft_tokens"] for item in metrics)
        drafted = sum(item["num_draft_tokens"] for item in metrics)
        report.update(
            accepted_tokens=accepted,
            rejected_tokens=drafted - accepted,
            max_proposed_tokens_per_step=max(
                proposed for item in metrics for proposed in item["per_step_drafted"]
            ),
            acceptance_and_rejection_observed=0 < accepted < drafted,
            rejection_continuation_observed=any(
                any(
                    accepted_step < proposed_step
                    for accepted_step, proposed_step in zip(
                        item["per_step_accepted"][:-1],
                        item["per_step_drafted"][:-1],
                        strict=True,
                    )
                )
                for item in metrics
            ),
        )
        report["passed"] = (
            report["acceptance_and_rejection_observed"]
            and report["rejection_continuation_observed"]
        )
    launch.save_json(directory / "request-matrix.json", report)
    if not report["passed"]:
        if collect_mismatches and not spec_tokens:
            if return_mismatch_report:
                return report
            raise RuntimeError(
                "EP diagnostic matrix completed with token mismatches; "
                "see request-matrix.json"
            )
        raise RuntimeError(
            "DSpark requires acceptance, rejection and continuation; "
            "see request-matrix.json"
        )
    return report


def cancel_stream(
    directory: Path,
    *,
    prefill_url: str | None,
    api_url: str = f"http://127.0.0.1:{launch.API_PORT}",
) -> None:
    payload = {
        "model": "dsv41-afd",
        "prompt": PROMPTS[1],
        "temperature": 0,
        "seed": 0,
        "max_tokens": CANCEL_OUTPUT_TOKENS,
        "ignore_eos": True,
        "stream": True,
    }
    if prefill_url:
        payload["kv_transfer_params"] = prefill_request(directory, prefill_url, payload)
    launch.save_json(directory / "cancel-request.json", payload)
    with launch.open_service_request(
        urllib.request.Request(
            api_url.rstrip("/") + "/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=launch.REQUEST_TIMEOUT_SECONDS,
    ) as reply:
        for line in reply:
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                chunk = json.loads(line[len(b"data: ") :])
                if any(choice.get("text") for choice in chunk.get("choices", [])):
                    launch.save_json(directory / "cancel-first-chunk.json", chunk)
                    return
    raise RuntimeError("cancel request returned no generated chunk")


def hold_producer(directory: Path, reference: Path | None, *, stop_file: Path) -> dict:
    del reference
    launch.save_json(directory / "producer-ready.json", {"stop_file": str(stop_file)})
    print(f"P ready; after D finishes, create {stop_file} to stop P", flush=True)
    while not stop_file.exists():
        time.sleep(launch.POLL_SECONDS)
    return {"producer_hold_finished": True, "hardware_acceptance": False}


def validate_reference_shutdown_timeout(
    reference: Path,
    plan: dict,
    summary: dict,
    preflight: dict,
    *,
    recheck_cleanup: bool = False,
) -> None:
    """Verify a completed EP token matrix whose native launcher timed out at exit."""

    def load(name: str):
        return json.loads((reference / name).read_bytes())

    if (
        summary.get("error")
        or summary.get("shutdown_error")
        or summary.get("forced_cleanup") is not True
    ):
        raise ValueError("expected a native exit timeout followed by forced cleanup")
    if not recheck_cleanup and (
        summary.get("survivors") != []
        or not launch.selected_devices_idle(summary.get("npu_after", {}), plan)
    ):
        raise ValueError("expected no survivors and idle native devices")
    events = load("lifecycle.json")
    completed = next(
        (
            index
            for index, event in enumerate(events)
            if event["event"] == "request_completed"
        ),
        len(events),
    )
    if not any(
        event["event"] == "exit_timeout" and event.get("role") == "attention"
        for event in events[completed + 1 :]
    ):
        raise ValueError("missing native exit timeout after successful requests")

    reference_preflight = load("preflight.json")
    if (
        reference_preflight.get("passed") is not True
        or reference_preflight["checkpoint"]["sha256"]
        != preflight["checkpoint"]["sha256"]
    ):
        raise ValueError("native reference checkpoint identity differs")
    matrix = load("request-matrix.json")
    dp_size = plan["native_dp_size"]
    count = 3 * dp_size + 1
    if matrix.get("passed") is not True or len(matrix["requests"]) != count:
        raise ValueError("native reference request matrix is incomplete")
    for index, request in enumerate(matrix["requests"]):
        name = f"request-{index:02d}"
        response = load(f"{name}/response.json")
        ranking = load(f"{name}/request-routing.json")
        expected_rank = 1 if index == count - 1 else index % dp_size
        if (
            request.get("passed") is not True
            or request.get("http_status") != 200
            or ranking.get("dp_rank") != expected_rank
            or response["usage"]["completion_tokens"] != launch.REQUEST_TOKENS
            or launch.compare_response_tokens(response, response)["status"]
            != "tokens_match"
        ):
            raise ValueError(f"native reference response {index:02d} is incomplete")


def recheck_reference_cleanup(
    directory: Path, reference_summary: dict, candidate_plan: dict
) -> dict:
    table = launch.process_table()
    remaining = []
    for archived in reference_summary["survivors"]:
        current = table.get(archived["pid"])
        if (
            current is not None
            and current.started == archived["start_ticks"]
            and current.state != "Z"
        ):
            remaining.append({**archived, "state": current.state})
    snapshot = launch.npu_snapshot(directory, "native-reference-recheck-npu.log")
    idle = launch.selected_devices_idle(snapshot, candidate_plan)
    report = {
        "checked_utc": launch.utc_now(),
        "archived_survivors": reference_summary["survivors"],
        "archived_npu_after": reference_summary["npu_after"],
        "remaining_survivors": remaining,
        "npu": snapshot,
        "passed": not remaining and idle,
    }
    launch.save_json(directory / "native-reference-recheck.json", report)
    if not report["passed"]:
        pids = [item["pid"] for item in remaining]
        unconfirmed_devices = sorted(
            set(candidate_plan["physical_devices"])
            - (set(snapshot["idle_devices"]) if snapshot["exit_code"] == 0 else set())
        )
        raise ValueError(
            f"current cleanup recheck failed: live native PIDs={pids}, devices "
            f"not confirmed idle={unconfirmed_devices}; "
            "see native-reference-recheck.json"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("/home/models/DeepSeek-V4.1-Flash")
    )
    parser.add_argument("--transformer-vendor", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="eager-u1")
    parser.add_argument(
        "--spec-tokens", type=int, choices=SPEC_TOKEN_CHOICES, default=0
    )
    parser.add_argument("--native-control", action="store_true")
    parser.add_argument(
        "--expert-parallel",
        action="store_true",
        help="qualify TP1/DP4/EP4 on both roles with serial U2 exchange",
    )
    parser.add_argument("--pd-role", choices=("producer", "consumer"))
    parser.add_argument("--engine-id")
    parser.add_argument("--attention-devices", default="0,1,2,3")
    parser.add_argument("--ffn-devices", default="4,5,6,7")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--kv-ip", default="127.0.0.1")
    parser.add_argument("--port-offset", type=int, default=0)
    parser.add_argument("--stream-overlap", type=int, choices=(0, 1), default=1)
    parser.add_argument("--prefill-url")
    parser.add_argument("--producer-stop-file", type=Path)
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument("--reference-run", type=Path)
    validation.add_argument(
        "--ep-functional-only",
        action="store_true",
        help="validate AFD EP requests and execution without a native reference",
    )
    parser.add_argument(
        "--repeatability",
        action="store_true",
        help="serial eager/U1 or Graph/U1 requests before/after cancellation",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=repeatability_check.DEFAULT_WARMUP_ROUNDS,
    )
    parser.add_argument(
        "--repeat-rounds",
        type=int,
        default=repeatability_check.DEFAULT_REPEAT_ROUNDS,
    )
    parser.add_argument(
        "--repeat-logprobs",
        type=int,
        choices=(0, repeatability_check.LOGPROB_CANDIDATES),
        default=0,
        help="optional top-5 score collection; default reuses the token-only path",
    )
    parser.add_argument(
        "--trace-native-prefill",
        action="store_true",
        help="record the first two native sampling steps in attention-server.log",
    )
    parser.add_argument(
        "--ep-diagnostics",
        action="store_true",
        help="collect DP scheduling, top-5 scores and STOP diagnostics for EP U1",
    )
    parser.add_argument(
        "--attention-stop-signal",
        choices=("SIGINT", "SIGTERM", "SIGUSR2"),
        help="initial signal to the Attention root; SIGUSR2 invokes the native handler",
    )
    parser.add_argument(
        "--ep-serial-first-wave",
        action="store_true",
        help="warm up each DP rank and measure the first EP wave one request at a time",
    )
    parser.add_argument(
        "--ep-shutdown-smoke",
        action="store_true",
        help="send one DP1 request before verifying EP shutdown",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run and sys.platform != "linux":
        parser.error("--run requires the A5 Linux container")
    if args.trace_native_prefill and (
        not (args.native_control and args.repeatability) or args.reference_run
    ):
        parser.error("--trace-native-prefill requires --native-control --repeatability")
    if args.pd_role == "consumer" and not args.prefill_url:
        parser.error("PD consumer requires --prefill-url")
    if args.pd_role == "producer" and (
        args.producer_stop_file is None or args.producer_stop_file.exists()
    ):
        parser.error("PD producer requires a new --producer-stop-file")
    if args.ep_shutdown_smoke and args.reference_run is not None:
        parser.error("EP shutdown smoke takes no native reference")
    if (
        args.run
        and not args.native_control
        and args.pd_role != "producer"
        and not args.ep_shutdown_smoke
        and not args.ep_functional_only
        and args.reference_run is None
    ):
        parser.error(
            "execution requires --reference-run from the matching native control"
        )
    repo = Path(__file__).resolve().parents[2]
    plan = make_plan(
        args.model.resolve(),
        args.transformer_vendor.resolve(),
        repo,
        mode=args.mode,
        spec_tokens=args.spec_tokens,
        native_control=args.native_control,
        expert_parallel=args.expert_parallel,
        pd_role=args.pd_role,
        engine_id=args.engine_id,
        attention_devices=tuple(map(int, args.attention_devices.split(","))),
        ffn_devices=tuple(map(int, args.ffn_devices.split(","))),
        bind_host=args.bind_host,
        kv_ip=args.kv_ip,
        port_offset=args.port_offset,
        stream_overlap=bool(args.stream_overlap),
        repeatability=args.repeatability,
        warmup_rounds=args.warmup_rounds,
        repeat_rounds=args.repeat_rounds,
        repeat_logprobs=args.repeat_logprobs,
        trace_native_prefill=args.trace_native_prefill,
        ep_diagnostics=args.ep_diagnostics,
        ep_functional_only=args.ep_functional_only,
        attention_stop_signal=args.attention_stop_signal,
        ep_serial_first_wave=args.ep_serial_first_wave,
        ep_shutdown_smoke=args.ep_shutdown_smoke,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(prefix="full-" + args.mode + "-", dir=args.output_root)
    )
    launch.save_json(directory / "launch-plan.json", plan)
    print(f"results: {directory}", flush=True)
    if not args.run:
        return 0
    preflight = launch.collect_preflight(
        args.model, repo, profile=FULL_CHECKPOINT_PROFILE
    )
    launch.save_json(directory / "preflight.json", preflight)
    if not preflight["passed"]:
        launch.pack_run(directory)
        return 1
    if args.reference_run:
        reference_plan = json.loads(
            (args.reference_run / "launch-plan.json").read_bytes()
        )
        if reference_plan.get("trace_native_prefill"):
            raise ValueError(
                "native prefill trace is diagnostic only; AFD services were not started"
            )
        reference_summary = json.loads(
            (args.reference_run / "summary.json").read_bytes()
        )
        reference_adapter = reference_plan.get("attention_head_adapter")
        if (
            not reference_plan["native_control"]
            or not reference_summary["request_passed"]
            or (
                not reference_summary["clean_shutdown"]
                and plan["request_suite"] != "ep-rank-matrix-v2"
            )
            or reference_adapter != plan["attention_head_adapter"]
            or reference_plan.get("native_tp_size", 4) != plan["native_tp_size"]
            or reference_plan.get("native_dp_size", 1) != plan["native_dp_size"]
            or reference_plan.get("expert_parallel", False) != plan["expert_parallel"]
            or (
                plan["expert_parallel"]
                and reference_plan.get("spec_tokens", 0) != plan["spec_tokens"]
            )
        ):
            raise ValueError(
                f"native reference {args.reference_run} is unusable: "
                f"native_control={reference_plan['native_control']}, "
                f"request_passed={reference_summary['request_passed']}, "
                f"clean_shutdown={reference_summary['clean_shutdown']}; "
                f"attention_head_adapter={reference_adapter!r} "
                f"(expected {plan['attention_head_adapter']!r}); "
                f"reference error={reference_summary.get('error', 'none')}. "
                "See its summary.json; AFD services were not started."
            )
        if reference_plan.get("request_suite", "matrix") != plan["request_suite"]:
            raise ValueError(
                "native reference request suite differs; AFD services were not started"
            )
        if (
            reference_plan.get("ep_serial_first_wave", False)
            != args.ep_serial_first_wave
        ):
            raise ValueError(
                "native reference first-wave scheduling differs; "
                "AFD services were not started"
            )
        cleanup_recheck = None
        if not reference_summary["clean_shutdown"]:
            try:
                can_recheck = args.ep_diagnostics or (
                    args.expert_parallel
                    and args.mode == "graph-u1"
                    and args.pd_role is None
                )
                recheck = can_recheck and (
                    reference_summary["survivors"] != []
                    or not launch.selected_devices_idle(
                        reference_summary["npu_after"], reference_plan
                    )
                )
                validate_reference_shutdown_timeout(
                    args.reference_run,
                    reference_plan,
                    reference_summary,
                    preflight,
                    recheck_cleanup=recheck,
                )
                if recheck:
                    cleanup_recheck = recheck_reference_cleanup(
                        directory, reference_summary, plan
                    )
            except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
                raise ValueError(
                    f"native reference shutdown diagnostic rejected: {error}. "
                    "AFD services were not started."
                ) from error
        plan["native_reference"] = {
            "directory": str(args.reference_run),
            "request_passed": reference_summary["request_passed"],
            "clean_shutdown": reference_summary["clean_shutdown"],
            "exit_codes": reference_summary.get("exit_codes"),
            "forced_cleanup": reference_summary.get("forced_cleanup"),
            "diagnostic_only": not reference_summary["clean_shutdown"],
        }
        if cleanup_recheck is not None:
            plan["native_reference"]["cleanup_recheck"] = cleanup_recheck
        launch.save_json(directory / "launch-plan.json", plan)
        if args.repeatability:
            reference_report = json.loads(
                (args.reference_run / "repeatability.json").read_bytes()
            )
            if (
                not reference_report["completed"]
                or not reference_report["passed"]
                or reference_report["protocol"] != plan["repeatability_protocol"]
                or reference_plan["repeatability_protocol"]
                != plan["repeatability_protocol"]
            ):
                raise ValueError(
                    "native repeatability reference is incomplete, unstable or has "
                    "different parameters; AFD services were not started"
                )
    if args.pd_role == "producer":
        request_runner = partial(
            hold_producer, stop_file=args.producer_stop_file.resolve()
        )
    elif args.ep_shutdown_smoke:
        request_runner = partial(
            launch.send_request, api_url=plan["api_url"], data_parallel_rank=1
        )
    elif args.repeatability:
        request_runner = partial(
            repeatability_check.run_repeatability,
            api_url=plan["api_url"],
            protocol=plan["repeatability_protocol"],
            cancel=partial(cancel_stream, prefill_url=None, api_url=plan["api_url"]),
        )
    else:
        request_runner = partial(
            run_requests,
            spec_tokens=args.spec_tokens,
            prefill_url=args.prefill_url,
            api_url=plan["api_url"],
            data_parallel_size=plan["native_dp_size"],
            ep_diagnostics=args.ep_diagnostics,
            ep_serial_first_wave=args.ep_serial_first_wave,
            return_mismatch_report=(
                plan["roles"]["attention"]["environment"].get(launch.EP_GRAPH_TRACE_ENV)
                == "1"
            ),
        )
    previous_handler = signal.signal(signal.SIGTERM, launch.handle_termination)
    try:
        return launch.run_candidate(
            plan,
            directory,
            args.transformer_vendor,
            args.startup_timeout,
            args.reference_run,
            request_runner=request_runner,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        launch.pack_run(directory)


if __name__ == "__main__":
    raise SystemExit(main())
