# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Bounded real Mooncake/NPU PD component; controlled pages, no model allocator.

Only isolated workers import device packages. The supervisor requires a reviewed
prepare-only identity, eight idle boards and owned-PID cleanup. Native workers
have no stop/unregister API: DONE, executor drain and normal process exit are
separate observations. Optional plugin receiver retirement never unregisters
memory or closes the native engine.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.metadata
import json
import queue
import subprocess
import sys
import sysconfig
import threading
import time
import traceback
from datetime import timedelta
from pathlib import Path

import hccl_eager_backend_probe as supervision
import hccl_staged_probe as helpers

EVENT_PREFIX = "PD_LAYOUT_EVENT "
TOTAL_TIMEOUT_SECONDS = 240
REQUEST_TIMEOUT_SECONDS = 45
REAP_TIMEOUT_SECONDS = 10
POLL_SECONDS = 0.05
EXPECTED_IDLE_BOARDS = 8
SOURCE_PIN_COUNT = 33
PRODUCER_BLOCKS = 6
CONSUMER_BLOCKS = 4
BLOCK_SIZE = 4
PAGE_PADDING_BYTES = 8
SLAB_GUARD_BYTES = 32
PATTERN_PERIOD = 251
TRANSFER_CASES = ("empty", "mapped", "tail", "alias-conflict")
BINARY_PATHS = ("mooncake/engine.so", "mooncake/libtransfer_engine.so")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("transfer", "layout-mismatch"), required=True
    )
    parser.add_argument(
        "--devices", default="0,2", help="producer,consumer physical NPU"
    )
    parser.add_argument("--model", required=True, help="native config metadata only")
    parser.add_argument("--source-identity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=TOTAL_TIMEOUT_SECONDS)
    parser.add_argument(
        "--retire-receiver",
        action="store_true",
        help="opt in to plugin consumer admission/drain/thread retirement",
    )
    parser.add_argument(
        "--role", choices=("producer", "consumer"), help=argparse.SUPPRESS
    )
    parser.add_argument("--port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        args.devices = tuple(int(part) for part in args.devices.split(","))
    except ValueError:
        parser.error("devices must be two distinct nonnegative indices")
    if len(args.devices) != 2 or len(set(args.devices)) != 2 or min(args.devices) < 0:
        parser.error("devices must be two distinct nonnegative indices")
    if not REAP_TIMEOUT_SECONDS < args.timeout <= TOTAL_TIMEOUT_SECONDS:
        parser.error("timeout exceeds the bounded probe limit")
    if args.role and args.port is None:
        parser.error("worker requires a port block")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_preflight(identity_path):
    if "torch" in sys.modules or "torch_npu" in sys.modules:
        raise RuntimeError("supervisor must not import device packages")
    identity = json.loads(identity_path.read_text())
    site = Path(sysconfig.get_paths()["purelib"]).resolve()
    root = Path(__file__).resolve().parents[2]
    if Path(identity["snapshot_root"]).resolve() != root:
        raise RuntimeError("identity does not belong to this immutable snapshot")
    expected = identity["source_expected"]
    if len(expected) != SOURCE_PIN_COUNT or set(identity["binary_expected"]) != set(
        BINARY_PATHS
    ):
        raise RuntimeError("the reviewed 33-source/two-binary gate is required")
    checks = {name: sha256(site / name) == digest for name, digest in expected.items()}
    binaries = {
        name: sha256(site / name) == digest
        for name, digest in identity["binary_expected"].items()
    }
    sources = {}
    for name, digest in identity["manifest"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("manifest path escapes the snapshot")
        sources[name] = sha256(path) == digest
    model_root = Path(identity["options"]["model"]).resolve()
    model = {}
    expected_model = identity["model_config_expected"]
    if "config.json" not in expected_model:
        raise ValueError("prepared model config identity is required")
    actual_model_names = {"config.json"} | {
        path.name for path in model_root.glob("*.py")
    }
    if (model_root / "generation_config.json").exists():
        actual_model_names.add("generation_config.json")
    if actual_model_names != set(expected_model):
        raise ValueError("model config file set changed since preparation")
    for name, digest in expected_model.items():
        path = (model_root / name).resolve()
        if not path.is_relative_to(model_root):
            raise ValueError("model config path escapes the model directory")
        model[name] = sha256(path) == digest
    if (
        not all(checks.values())
        or not all(binaries.values())
        or not all(sources.values())
        or not all(model.values())
    ):
        raise RuntimeError("source or binary changed since preparation")
    return {
        "installed_sources": checks,
        "binaries": binaries,
        "snapshot_sources": sources,
        "model_config_sources": model,
        "mooncake_version": importlib.metadata.version("mooncake-transfer-engine-npu"),
        "torch_imported": False,
        "torch_npu_imported": False,
    }


def sample_idle(output_dir, label):
    result = subprocess.run(["npu-smi", "info"], capture_output=True, timeout=30)
    (output_dir / f"npu-{label}.log").write_bytes(result.stdout + result.stderr)
    idle = result.stdout.count(b"No running processes found in NPU")
    if result.returncode != 0 or idle != EXPECTED_IDLE_BOARDS:
        raise RuntimeError(f"eight idle boards required at {label}; observed {idle}")
    return idle


def emit(event, **fields):
    print(EVENT_PREFIX + json.dumps({"event": event, **fields}), flush=True)


def receive_command(expected):
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("supervisor control input closed")
    command = json.loads(line)
    if command["command"] != expected:
        raise RuntimeError(f"expected {expected}, got {command['command']}")
    return command


def materialize_pages(torch, device, *, producer, mismatch):
    # Device-only imports. These are native typed descriptors and real NPU views;
    # manual allocation deliberately does not claim native model KV allocation.
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
        KVCacheConfig,
        KVCacheTensor,
    )
    from vllm_ascend.core.kv_cache_interface import (
        AscendMLAAttentionSpec,
        AscendSlidingWindowMLASpec,
    )
    from vllm_ascend.models.deepseek_v41.cache_config import (
        get_layer_tuples,
        group_cache_specs,
        make_cache_groups,
    )

    prefix = "model.layers.0.self_attn"
    names = tuple(
        prefix + suffix
        for suffix in (
            ".long_kv_cache",
            ".indexer.k_cache",
            ".compressor.state",
            ".swa_cache",
        )
    )
    scale_dtype = torch.bfloat16 if mismatch and producer else torch.float16
    specs = {
        names[0]: AscendMLAAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.bfloat16,
            tokens_per_state=2,
            model_version="deepseek_v41",
            storage_block_size=2,
        ),
        names[1]: AscendMLAAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=2,
            dtype=torch.int8,
            tokens_per_state=2,
            model_version="deepseek_v41",
            storage_block_size=2,
            scale_dim=1,
            scale_dtype=scale_dtype,
        ),
        names[2]: CircularBufferSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=3,
            dtype=torch.bfloat16,
        ),
        names[3]: AscendSlidingWindowMLASpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=2,
            dtype=torch.bfloat16,
            sliding_window=4,
            model_version="deepseek_v41",
        ),
    }
    page_sizes, layer_tuples = get_layer_tuples(specs)
    if len(page_sizes) != 1 or len(layer_tuples) != 1 or page_sizes[0] != 24:
        raise RuntimeError("controlled native shared-page geometry changed")
    blocks = PRODUCER_BLOCKS if producer else CONSUMER_BLOCKS
    stride = page_sizes[0] + PAGE_PADDING_BYTES
    config = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=blocks * stride,
                layers=list(layer_tuples[0]),
                layer_stride=0,
                block_stride=stride,
            )
        ],
        kv_cache_groups=make_cache_groups(group_cache_specs(specs)),
    )
    raw = torch.empty(
        blocks * stride + SLAB_GUARD_BYTES, dtype=torch.uint8, device=device
    )
    views = {}
    for name, dtype, width, rows, offset in (
        (names[0], torch.bfloat16, 4, 2, 0),
        (names[1], torch.int8, 2, 2, 16),
        (names[2], torch.bfloat16, 3, 2, 0),
        (names[3], torch.bfloat16, 2, 4, 0),
    ):
        size = dtype.itemsize
        views[name] = torch.as_strided(
            raw.view(dtype),
            (blocks, rows, 1, width),
            (stride // size, width, width, 1),
            offset // size,
        )
    scale = torch.as_strided(
        raw.view(scale_dtype),
        (blocks, 2, 1, 1),
        (stride // 2, 1, 1, 1),
        20 // 2,
    )
    views[names[1]] = (views[names[1]], scale)
    return raw, config, views, stride


def cpu_pattern(torch, length, *, producer, iteration=0):
    return (
        (
            torch.arange(length, dtype=torch.int64) * (7 if producer else 5)
            + (11 if producer else 173 + iteration)
        )
        % PATTERN_PERIOD
    ).to(torch.uint8)


def request_mapping(name, groups):
    if name == "empty":
        return tuple(() for _ in range(groups)), tuple(() for _ in range(groups))
    source, destination = (
        ((3, 1), (0, 2))
        if name == "mapped"
        else (((0, 2, 3), (1, 2)) if name == "tail" else ((2,), (1,)))
    )
    remote = tuple(source for _ in range(groups))
    if name == "alias-conflict":
        remote = (*remote[:-1], (3,))
    return remote, tuple(destination for _ in range(groups))


def wait_finished(worker, expected, *, producer):
    deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        sent, received = worker.get_finished()
        finished = sent if producer else received
        if finished:
            if finished != {expected}:
                raise RuntimeError(f"unexpected completed request IDs: {finished}")
            return
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"native DONE did not retire {expected}")


def run_worker(args):
    import torch
    import vllm_ascend.ops  # noqa: F401
    from vllm.config import set_current_vllm_config
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
        collect_configured_register_regions,
    )
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
        MooncakeConnectorMetadata,
    )
    from vllm_ascend.utils import adapt_patch

    from afd_plugin.compat.npu.kv_layout import compare_kv_layouts, layout_from_json
    from afd_plugin.compat.npu.kv_layout_native import collect_deepseek_v41_kv_layout

    producer = args.role == "producer"
    coordinate = 0 if producer else 1
    device = torch.device("npu", args.devices[coordinate])
    report = {
        "role": args.role,
        "device": str(device),
        "success": False,
        "scope": "real native worker/TE/threads, controlled typed NPU pages",
        "native_allocator": False,
        "requests": [],
        "native_stop_or_unregister_api": False,
        "receiver_retirement_requested": args.retire_receiver and not producer,
    }
    output = args.output_dir / f"{args.role}.json"
    connector = raw = views = None
    try:
        current_platform.pre_register_and_update()
        load_general_plugins()
        torch.npu.set_device(device)
        adapt_patch()
        config = helpers.build_config(1, args.model)
        config.cache_config.block_size = BLOCK_SIZE
        config.cache_config.enable_prefix_caching = False
        config.scheduler_config.disable_hybrid_kv_cache_manager = False
        config.kv_transfer_config = KVTransferConfig(
            kv_connector="MooncakeHybridConnector",
            kv_connector_module_path="afd_plugin.compat.npu.kv_layout_connector",
            kv_role="kv_producer" if producer else "kv_consumer",
            engine_id="pd-layout-" + args.role,
            kv_port=args.port + coordinate,
            kv_connector_extra_config={
                "prefill": {"tp_size": 1, "dp_size": 1, "pp_size": 1},
                "decode": {"tp_size": 1, "dp_size": 1, "pp_size": 1},
            },
        )
        init_ascend_config(config)
        with set_current_vllm_config(config), torch.inference_mode():
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=device.index,
                distributed_init_method=f"tcp://127.0.0.1:{args.port + 2 + coordinate}",
                backend="hccl",
                timeout=timedelta(seconds=REQUEST_TIMEOUT_SECONDS),
            )
            initialize_model_parallel(tensor_model_parallel_size=1, backend="hccl")

            def native_check():
                value = torch.tensor([17], dtype=torch.int32, device=device)
                actual = get_tp_group().all_reduce(value).cpu().tolist()
                if actual != [17]:
                    raise RuntimeError("native TP domain changed")
                return actual

            report["native_tp_before"] = native_check()
            raw, kv_config, views, stride = materialize_pages(
                torch,
                device,
                producer=producer,
                mismatch=args.case == "layout-mismatch",
            )
            # Native has no unregister/stop. Keep the allocation and typed
            # descriptors alive through process finalization, including errors.
            atexit.register(lambda retained=(raw, kv_config, views, config): None)
            reference = cpu_pattern(torch, raw.numel(), producer=producer)
            raw.copy_(reference)
            torch.npu.synchronize()
            layout = collect_deepseek_v41_kv_layout(kv_config, views)
            if layout.device_type != "npu":
                raise RuntimeError("actual NPU collector required")
            (args.output_dir / f"{args.role}-layout.json").write_text(layout.to_json())
            regions = collect_configured_register_regions(kv_config, views)
            report.update(
                {
                    "page_stride_bytes": stride,
                    "slab_bytes": raw.numel(),
                    "configured_page_bytes": kv_config.num_blocks * stride,
                    "native_registration_regions": {
                        "pointers": regions.ptrs,
                        "lengths": regions.lengths,
                    },
                    "guard_bytes": SLAB_GUARD_BYTES,
                    "guard_scope": (
                        "outside configured pages; native region may include it"
                    ),
                    "layout_sha256": hashlib.sha256(
                        layout.to_json().encode()
                    ).hexdigest(),
                }
            )
            cls = KVConnectorFactory.get_connector_class(config.kv_transfer_config)
            connector = cls(config, KVConnectorRole.WORKER, kv_config)
            atexit.register(lambda retained=connector: None)
            connector.register_kv_caches(views)
            report["registration_and_thread_start_returned"] = True
            worker = connector.connector_worker
            cases = TRANSFER_CASES if args.case == "transfer" else ("layout-mismatch",)
            tracking = MooncakeConnectorMetadata()
            tracking.reqs_in_batch.update(
                ("p-" if producer else "d-") + name for name in cases
            )
            worker.start_load_kv(tracking)
            emit(
                "ready",
                role=args.role,
                host=worker.side_channel_host,
                port=worker.handshake_port,
                engine_id=worker.engine_id,
                layout=layout.to_json(),
            )
            command = receive_command("peer")
            peer = command["peer"]
            peer_layout = layout_from_json(peer["layout"])
            comparison = compare_kv_layouts(peer_layout, layout)
            if comparison.compatible != (args.case == "transfer"):
                raise RuntimeError(
                    "actual peer layout does not match the selected case"
                )
            receiver = worker.kv_recv_thread
            if not producer and args.case == "layout-mismatch":
                try:
                    receiver._get_remote_metadata(
                        peer["engine_id"], peer["host"], peer["port"]
                    )
                except ValueError as error:
                    if not str(error).startswith("incompatible KV layouts:"):
                        raise
                    report["negative_handshake_error"] = str(error)
                else:
                    raise RuntimeError("incompatible actual peer layout was accepted")
            emit("tracked", role=args.role)
            for iteration, name in enumerate(cases):
                receive_command("request:" + name)
                if producer:
                    wait_finished(worker, "p-" + name, producer=True)
                    torch.testing.assert_close(raw.cpu(), reference, rtol=0, atol=0)
                    row = {
                        "name": name,
                        "native_done_observed": True,
                        "source_unchanged": True,
                    }
                else:
                    before = cpu_pattern(
                        torch, raw.numel(), producer=False, iteration=iteration
                    )
                    raw.copy_(before)
                    torch.npu.synchronize()
                    remote, local = request_mapping(name, len(layout.groups))
                    metadata = MooncakeConnectorMetadata()
                    metadata.add_new_req(
                        "d-" + name,
                        local,
                        0 if name == "empty" else BLOCK_SIZE,
                        {
                            "remote_block_ids": remote,
                            "remote_engine_id": peer["engine_id"],
                            "remote_request_id": "p-" + name,
                            "remote_host": peer["host"],
                            "remote_port": peer["port"],
                            "remote_ptp_size": 1,
                            "remote_pcp_size": 1,
                        },
                    )
                    worker.start_load_kv(metadata)
                    wait_finished(worker, "d-" + name, producer=False)
                    receiver.request_queue.join()
                    invalid = worker.get_block_ids_with_load_errors()
                    rejected = name in ("alias-conflict", "layout-mismatch")
                    if invalid != ({1} if rejected else set()):
                        raise RuntimeError(f"unexpected native load errors: {invalid}")
                    expected = before.clone()
                    if not rejected:
                        source = cpu_pattern(
                            torch,
                            PRODUCER_BLOCKS * stride + SLAB_GUARD_BYTES,
                            producer=True,
                        )
                        for src_ids, dst_ids in zip(remote, local, strict=True):
                            for src, dst in zip(
                                src_ids[-len(dst_ids) :] if dst_ids else (),
                                dst_ids,
                                strict=True,
                            ):
                                expected[dst * stride : (dst + 1) * stride] = source[
                                    src * stride : (src + 1) * stride
                                ]
                    actual = raw.cpu()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    published = (
                        peer["engine_id"] in receiver.kv_caches_base_addr
                        and peer["port"]
                        in receiver.kv_caches_base_addr[peer["engine_id"]]
                    )
                    if (name in ("empty", "layout-mismatch")) and published:
                        raise RuntimeError("no peer address may publish in this case")
                    row = {
                        "name": name,
                        "native_done_observed": True,
                        "native_invalid_ids": sorted(invalid),
                        "rejected": rejected,
                        "whole_slab_bytes_equal": True,
                        "peer_addresses_published": published,
                        "raw_source_blocks": remote,
                        "local_blocks": local,
                        "destination_sha256": hashlib.sha256(
                            actual.numpy().tobytes()
                        ).hexdigest(),
                    }
                report["requests"].append(row)
                emit("request-finished", role=args.role, name=name)
            receive_command("drain")
            if not producer:
                if args.retire_receiver:
                    worker.retire_receiver(timeout=REQUEST_TIMEOUT_SECONDS)
                    report["receiver_retired"] = receiver.retired
                    report["receiver_input_thread_joined"] = not receiver.is_alive()
                    if not receiver.retired or receiver.is_alive():
                        raise RuntimeError("consumer receiver did not retire")
                else:
                    receiver.request_queue.join()
                    receiver.executor.shutdown(wait=True)
                report["request_queue_drained"] = True
                report["executor_shutdown_returned"] = True
            emit("drained", role=args.role)
            receive_command("exit")
            report["native_tp_after"] = native_check()
            destroy_model_parallel()
            destroy_distributed_environment()
            report["native_groups_destroyed"] = True
            report["registered_dependencies_retained_until_process_exit"] = True
            report["success"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(output, report)


def supervise(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "scope": "real NPU PD component; controlled pages, no model allocation/history",
        "case": args.case,
        "receiver_retirement_requested": args.retire_receiver,
        "devices": args.devices,
        "success": False,
        "owned_pid_start_ticks": {},
        "killed_owned_pids": [],
        "events": [],
    }
    children, threads, handles, ready = {}, [], [], {}
    mailbox = queue.Queue()
    started = time.monotonic()
    deadline = started + args.timeout - REAP_TIMEOUT_SECONDS

    def collect(role, process, log):
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if line.startswith(EVENT_PREFIX):
                    mailbox.put((role, json.loads(line[len(EVENT_PREFIX) :])))
        except BaseException:
            mailbox.put(
                (role, {"event": "reader-error", "error": traceback.format_exc()})
            )

    def send(role, command, **fields):
        children[role].stdin.write(json.dumps({"command": command, **fields}) + "\n")
        children[role].stdin.flush()

    def await_events(event, name=None):
        pending = set(children)
        found = {}
        while pending:
            for process in children.values():
                if process.poll() is None:
                    supervision.observe_owned(
                        process.pid, summary["owned_pid_start_ticks"]
                    )
                elif process.returncode != 0:
                    raise RuntimeError("a role failed; retain its first raw log")
            if time.monotonic() >= deadline:
                summary["timed_out"] = True
                raise TimeoutError(f"timed out at {event}:{name}")
            try:
                role, record = mailbox.get(timeout=POLL_SECONDS)
            except queue.Empty:
                continue
            summary["events"].append({"role": role, **record})
            if record["event"] != event or (
                name is not None and record.get("name") != name
            ):
                raise RuntimeError("unexpected control event ordering")
            if role not in pending:
                raise RuntimeError("duplicate control event")
            pending.remove(role)
            found[role] = record
        return found

    try:
        write_json(
            args.output_dir / "source-preflight.json",
            source_preflight(args.source_identity),
        )
        summary["idle_boards_before"] = sample_idle(args.output_dir, "before")
        port = helpers.select_port_block(4)
        for role in ("producer", "consumer"):
            log = (args.output_dir / f"{role}.log").open("w", encoding="utf-8")
            handles.append(log)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--case",
                    args.case,
                    "--devices",
                    ",".join(map(str, args.devices)),
                    "--model",
                    args.model,
                    "--source-identity",
                    str(args.source_identity),
                    "--output-dir",
                    str(args.output_dir),
                    "--role",
                    role,
                    "--port",
                    str(port),
                    *(["--retire-receiver"] if args.retire_receiver else []),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            children[role] = process
            supervision.observe_owned(process.pid, summary["owned_pid_start_ticks"])
            thread = threading.Thread(
                target=collect, args=(role, process, log), daemon=True
            )
            thread.start()
            threads.append(thread)
        ready = await_events("ready")
        for role in children:
            send(
                role,
                "peer",
                peer=ready["consumer" if role == "producer" else "producer"],
            )
        await_events("tracked")
        for name in TRANSFER_CASES if args.case == "transfer" else ("layout-mismatch",):
            for role in children:
                send(role, "request:" + name)
            await_events("request-finished", name)
        for role in children:
            send(role, "drain")
        await_events("drained")
        for role in children:
            send(role, "exit")
        while any(process.poll() is None for process in children.values()):
            for process in children.values():
                if process.poll() is None:
                    supervision.observe_owned(
                        process.pid, summary["owned_pid_start_ticks"]
                    )
            if time.monotonic() >= deadline:
                summary["timed_out"] = True
                raise TimeoutError("native process exit did not complete")
            time.sleep(POLL_SECONDS)
        summary["normal_process_exit"] = all(
            process.returncode == 0 for process in children.values()
        )
        summary["success"] = summary["normal_process_exit"]
    except BaseException:
        summary["error"] = traceback.format_exc()
    finally:
        cleanup_errors = []
        reap_deadline = time.monotonic() + REAP_TIMEOUT_SECONDS
        try:
            if any(process.poll() is None for process in children.values()):
                summary["success"] = False
                summary["killed_owned_pids"] = supervision.kill_owned(
                    summary["owned_pid_start_ticks"]
                )
        except BaseException:
            cleanup_errors.append(traceback.format_exc())
        for process in children.values():
            try:
                process.wait(timeout=max(0, reap_deadline - time.monotonic()))
            except BaseException:
                cleanup_errors.append(traceback.format_exc())
        for thread in threads:
            try:
                thread.join(timeout=max(0, reap_deadline - time.monotonic()))
                if thread.is_alive():
                    raise TimeoutError("owned output reader did not retire")
            except BaseException:
                cleanup_errors.append(traceback.format_exc())
        for handle in handles:
            try:
                handle.close()
            except BaseException:
                cleanup_errors.append(traceback.format_exc())
        summary["exit_codes"] = {
            role: process.returncode for role, process in children.items()
        }
        summary["workers"] = {}
        for role in children:
            path = args.output_dir / f"{role}.json"
            try:
                summary["workers"][role] = (
                    json.loads(path.read_text()) if path.exists() else None
                )
            except BaseException:
                summary["workers"][role] = None
                cleanup_errors.append(traceback.format_exc())
        summary["cleanup_errors"] = cleanup_errors
        summary["success"] = (
            summary["success"]
            and len(children) == 2
            and all(
                isinstance(report, dict) and report.get("success") is True
                for report in summary["workers"].values()
            )
            and not cleanup_errors
        )
        try:
            summary["idle_boards_after"] = sample_idle(args.output_dir, "after")
        except BaseException:
            summary["postcheck_error"] = traceback.format_exc()
            summary["success"] = False
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output_dir / "result.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.role:
        run_worker(parsed)
    else:
        raise SystemExit(supervise(parsed))
