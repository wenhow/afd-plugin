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
import json
import signal
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import run_dsv41_a5_afd as launch
from collect_dsv41_dspark_requests import validate_response
from preflight_dsv41_a5_afd import DEFAULT_PROFILE, SourceIdentity

MODES = ("eager-u1", "eager-u2", "graph-u1", "graph-u2")
PD_MODULE = "afd_plugin.compat.npu.kv_layout_connector"
CANCEL_OUTPUT_TOKENS = 256
DEFAULT_KV_PORT = 14579
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
    pd_role: str | None = None,
    engine_id: str | None = None,
    attention_devices: tuple[int, ...] = (0, 1, 2, 3),
    ffn_devices: tuple[int, ...] = (4, 5, 6, 7),
    bind_host: str = "127.0.0.1",
    kv_ip: str = "127.0.0.1",
    port_offset: int = 0,
    stream_overlap: bool = True,
) -> dict:
    if mode not in MODES or spec_tokens not in (0, 1, 2):
        raise ValueError("unsupported full-checkpoint qualification mode")
    if native_control and (mode != "eager-u1" or spec_tokens or pd_role):
        raise ValueError("native control requires eager U1 without DSpark/PD")
    if pd_role == "producer" and mode != "eager-u1":
        raise ValueError("PD producer qualification uses eager U1")
    if pd_role not in (None, "producer", "consumer"):
        raise ValueError("PD role must be producer or consumer")
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
    graph, u2 = mode.startswith("graph"), mode.endswith("u2")
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
        env.update(AFD_DSV41_EAGER_QUALIFICATION="0", AFD_DSV41_FULL_QUALIFICATION="1")
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
        set_option(argv, "--max-num-seqs", "2")
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
                            "prefill": {"tp_size": 4, "dp_size": 1, "pp_size": 1},
                            "decode": {"tp_size": 4, "dp_size": 1, "pp_size": 1},
                        },
                    }
                ),
            )
        if native_control and role == "attention":
            additional.pop("afd")
            env["AFD_DSV41_FULL_QUALIFICATION"] = "0"
        set_option(argv, "--additional-config", json.dumps(additional))
    if native_control:
        plan["roles"].pop("ffn")
    plan.update(
        scope="full-checkpoint qualification; hardware acceptance pending",
        mode=mode,
        spec_tokens=spec_tokens,
        native_control=native_control,
        pd_role=pd_role,
        native_tp_size=4,
        cpu_engram=True,
        required_devices=4 if native_control else 8,
        physical_devices=list(selected),
        api_url=f"http://127.0.0.1:{launch.API_PORT + port_offset}",
        ports=[port + port_offset for port in plan["ports"]],
        full_model_qualification=False,
        stream_overlap=stream_overlap,
    )
    if pd_role:
        plan["ports"].extend(
            range(DEFAULT_KV_PORT + port_offset, DEFAULT_KV_PORT + port_offset + 4)
        )
    return plan


def prefill_request(directory: Path, url: str, payload: dict) -> dict:
    request = {
        **payload,
        "max_tokens": 1,
        "stream": False,
        "kv_transfer_params": {"do_remote_decode": True},
    }
    launch.save_json(directory / "prefill-request.json", request)
    with urllib.request.urlopen(
        urllib.request.Request(
            url.rstrip("/") + "/v1/completions",
            data=json.dumps(request).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=launch.REQUEST_TIMEOUT_SECONDS,
    ) as reply:
        response = json.load(reply)
    launch.save_json(directory / "prefill-response.json", response)
    params = response["kv_transfer_params"]
    if "afd_engram_history_v1" not in params:
        raise ValueError("P did not export original Engram history")
    return {**params, "do_remote_prefill": True, "do_remote_decode": False}


def run_requests(
    directory: Path,
    reference: Path | None,
    *,
    spec_tokens: int,
    prefill_url: str | None,
    api_url: str = f"http://127.0.0.1:{launch.API_PORT}",
) -> dict:
    def one(index: int, prompt: str) -> dict:
        target = directory / f"request-{index:02d}"
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
        if prefill_url:
            payload["kv_transfer_params"] = prefill_request(
                target, prefill_url, payload
            )
        baseline = reference / target.name / "response.json" if reference else None
        result = launch.send_request(target, baseline, payload=payload, api_url=api_url)
        if spec_tokens:
            result["dspark"] = validate_response(
                json.loads((target / "response.json").read_bytes()),
                spec_tokens=spec_tokens,
                max_tokens=launch.REQUEST_TOKENS,
            )["speculative_decoding"]
        return result

    results = []
    # Concurrent short requests exercise U2; subsequent waves reuse slots.
    # The final long prompt exercises chunked prefill and single-request U1
    # fallback. Each reference uses the same wave/prompt in the TP4 control.
    for wave in range(2):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(one, wave * 2 + i, prompt)
                for i, prompt in enumerate(PROMPTS[:2])
            ]
            results.extend(future.result() for future in futures)
        if wave == 0:
            cancel_stream(directory, prefill_url=prefill_url, api_url=api_url)
    results.append(one(4, PROMPTS[2]))
    report = {"passed": True, "requests": results, "hardware_acceptance": False}
    if spec_tokens:
        metrics = [result["dspark"] for result in results]
        accepted = sum(item["num_accepted_draft_tokens"] for item in metrics)
        drafted = sum(item["num_draft_tokens"] for item in metrics)
        report.update(
            accepted_tokens=accepted,
            rejected_tokens=drafted - accepted,
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
    with urllib.request.urlopen(
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("/home/models/DeepSeek-V4.1-Flash")
    )
    parser.add_argument("--transformer-vendor", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="eager-u1")
    parser.add_argument("--spec-tokens", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--native-control", action="store_true")
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
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=3600)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run and sys.platform != "linux":
        parser.error("--run requires the A5 Linux container")
    if args.pd_role == "consumer" and not args.prefill_url:
        parser.error("PD consumer requires --prefill-url")
    if args.pd_role == "producer" and (
        args.producer_stop_file is None or args.producer_stop_file.exists()
    ):
        parser.error("PD producer requires a new --producer-stop-file")
    if (
        args.run
        and not args.native_control
        and args.pd_role != "producer"
        and args.reference_run is None
    ):
        parser.error("execution requires --reference-run from the TP4 native control")
    repo = Path(__file__).resolve().parents[2]
    plan = make_plan(
        args.model.resolve(),
        args.transformer_vendor.resolve(),
        repo,
        mode=args.mode,
        spec_tokens=args.spec_tokens,
        native_control=args.native_control,
        pd_role=args.pd_role,
        engine_id=args.engine_id,
        attention_devices=tuple(map(int, args.attention_devices.split(","))),
        ffn_devices=tuple(map(int, args.ffn_devices.split(","))),
        bind_host=args.bind_host,
        kv_ip=args.kv_ip,
        port_offset=args.port_offset,
        stream_overlap=bool(args.stream_overlap),
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
        reference_summary = json.loads(
            (args.reference_run / "summary.json").read_bytes()
        )
        if (
            not reference_plan["native_control"]
            or not reference_summary["request_passed"]
            or not reference_summary["clean_shutdown"]
        ):
            raise ValueError("reference must be a successful native TP4 control run")
    request_runner = (
        partial(hold_producer, stop_file=args.producer_stop_file.resolve())
        if args.pd_role == "producer"
        else partial(
            run_requests,
            spec_tokens=args.spec_tokens,
            prefill_url=args.prefill_url,
            api_url=plan["api_url"],
        )
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
