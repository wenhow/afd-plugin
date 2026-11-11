# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Freeze/review, then explicitly run an isolated A3 PD component; no installs.

``prepare`` uses only stdlib/source reads and npu-smi, never imports torch or
Mooncake. ``run`` reuses that exact immutable source/binary identity. Results,
binary identities and effective environment stay in the requested local folder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import re
import shlex
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path, PurePosixPath
from types import MappingProxyType

PROBE = "tests/npu/pd_kv_layout_probe.py"
SUPERVISION = "tests/npu/hccl_eager_backend_probe.py"
CONFIG_HELPER = "tests/npu/hccl_staged_probe.py"
HARNESS = "tools/diagnostics/run_a3_pd_layout.py"
SOURCE_PIN_COUNT = 33
BINARY_PATHS = ("mooncake/engine.so", "mooncake/libtransfer_engine.so")
SOURCE_EXPECTED = MappingProxyType(
    {
        "vllm/model_executor/model_loader/base_loader.py": (
            "5291c330c4ed5634a9ea1d535364c26162b60251aa2af03fd409b88c34d836e2"
        ),
        "vllm/model_executor/model_loader/utils.py": (
            "aa398e74ee835537403e60186f2e9a81e8741a4b35fec68b47df6537716a45b7"
        ),
        "vllm/distributed/parallel_state.py": (
            "d9b10ad56833ebc08a753e1a050ef0989c30d4134021b7b7b00150b7bd01adc6"
        ),
        "vllm/config/parallel.py": (
            "f012ace89e3fb730a467f729f602679bc145c49f0d10721811c9d818570e7b2b"
        ),
        "vllm/forward_context.py": (
            "cc1473477dcc7b762c3901b25cc7bf35c06a2c42f193136fed553abf6fe33f0e"
        ),
        "vllm_ascend/distributed/parallel_state.py": (
            "b7ea7d6fb74530c50aec718d34266bae952ed96cc5a85766d7e208b952b10445"
        ),
        "vllm_ascend/worker/v2/model_runner.py": (
            "a3471a9f181b45742b8f985c86137cdaafc2871d2ef6f96aa6a45236a242969b"
        ),
        "vllm_ascend/models/deepseek_v41/model.py": (
            "989aec6db1e344dca76159dcbef0d5876874facedaa36b47e993146d5b8a2b9f"
        ),
        "vllm_ascend/models/deepseek_v41/dspark.py": (
            "fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601"
        ),
        "vllm_ascend/models/deepseek_v41/engram/model_state.py": (
            "677cb13bb86ad80a154e0ff9a43620aa3344e7e93170a05e27aba0cc1f10aa8b"
        ),
        "vllm_ascend/ascend_forward_context.py": (
            "848112214c9f56371b4bf43a59d39fadbed0e0e864581e300a3a89778d4083cb"
        ),
        "vllm/v1/worker/gpu/model_runner.py": (
            "174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce"
        ),
        "vllm/v1/worker/gpu/cudagraph_utils.py": (
            "6e9c042890603535e300a40df8ee159dbed1058a64a83ae50ff0329e332e05ff"
        ),
        "vllm/v1/worker/ubatch_utils.py": (
            "b34dce551b5195c414e6a3a997c75961cc7a17f24c7c980487fe3d3e2ba7a798"
        ),
        "vllm/v1/worker/gpu/ubatch_utils.py": (
            "5092e2e58af0561c8dc12e62847cfc6dda9ca20dfbd33e7f0cd77a83fe87f7c5"
        ),
        "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py": (
            "dcf4abdedbd2bddd0d76be198ccf5b4e467c46f645038c35fcab821491a35b03"
        ),
        "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py": (
            "f9c100f42f9326e48eb84e89281897e859ba761943f54d593523555ccd98dc63"
        ),
        "vllm/v1/worker/gpu/spec_decode/dflash/cudagraph.py": (
            "beab567ae2df6a2ba2cd92fe7ee452c6b959b8a7037a4b5f493993a1824619bb"
        ),
        "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py": (
            "51faf4e6b4d633863d1f7c74db84986fb7c15453e7ce0305d3d965abbd5d14c1"
        ),
        "vllm_ascend/worker/v2/aclgraph_utils.py": (
            "b2271eb4aab5876e85370897236a99ece7e9a9541b05f2fc0dd923b32bb8193d"
        ),
        "vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py": (
            "dc4ee25fee88ef1f82f24a64743904a0eda2d191c756e234136cfbb3b1ad1ecb"
        ),
        "vllm/v1/kv_cache_interface.py": (
            "1cf202f1a44d5bc5c3832b70b41507687ecceea3784e67a4f003bc0210d6eecb"
        ),
        "vllm_ascend/core/kv_cache_interface.py": (
            "b7356c5e499446a1bbf92be9490753336ed7f7d31e521735659491b613f370b3"
        ),
        "vllm_ascend/models/deepseek_v41/cache_config.py": (
            "467dddca82e2528ebbec14c78682486a39bffb4f096675d8b1d2d140f1a7014b"
        ),
        "vllm_ascend/worker/v2/attn_utils.py": (
            "08f36619b368f2c50a13a26d357001a688a94463e9c960db02382077e24a6b2b"
        ),
        "vllm_ascend/attention/dsa_v41.py": (
            "d9a9f3c02eda7ba50fbfe3de1e42022921539cb2c7c16947f63ef60fbfb8adb4"
        ),
        "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py": (
            "a6a9c3a870b7d577fb2a39435067fe378646b6ab39bc669140641beba92bdf84"
        ),
        "vllm/distributed/kv_transfer/kv_connector/factory.py": (
            "add99c93c587b0098b90b1f8fd6940bdd1f15c236c5c6f1a258411863cb902d3"
        ),
        "vllm/config/kv_transfer.py": (
            "506fae7a9d7f32cb3646156cf9249a856f6317dff6c4c85db70efd34418c429b"
        ),
        "vllm/distributed/kv_transfer/kv_connector/utils.py": (
            "317ed40dc837671ca6e666165222e641a0f9b3992aea63bbf54de7e13d40b3a4"
        ),
        "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/utils.py": (
            "9d2e1cbacde3def1892ed52fd9e65559a1b36f0f85c752f5897bd10ce7aa2ef2"
        ),
        "vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py": (
            "6f0dfdad686d44b3414809ab127f7b463354b4869a41d5b56385c203565f166a"
        ),
        "vllm_ascend/distributed/kv_transfer/__init__.py": (
            "0246650661223ecbaaad13ab4d2fa908c5fd47a59bdfd0b405f4efda6a14db5f"
        ),
    }
)
DEFAULT_HOST = "devenvc_rissvg.4ff038c993bf40219c03182f786a9def.atomgit.0"
DEFAULT_SSH_CONFIG = "C:/Users/zq/.atomgitdevenv/.ssh/config"
DEFAULT_REMOTE_PARENT = "/mnt/workspace/tmp/afd-c0-20261104.hVgsIf"
DEFAULT_VENV = "/mnt/workspace/code/.venvs/afd-v030-dsv41-c0"
DEFAULT_CANN_ENV = "/mnt/workspace/code/.ascend/cann-9.0.1/ascend-toolkit/set_env.sh"
DEFAULT_ATB_ENV = (
    "/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1/nnal/atb/set_env.sh"
)
DEFAULT_LIBRARY_PATH = (
    "/opt/buildtools/python-3.12.9/lib:/usr/local/Ascend/driver/lib64/driver:"
    "/usr/local/Ascend/driver/lib64/common"
)
EXPECTED_IDLE_BOARDS = 8
INNER_TIMEOUT_SECONDS = 240
OUTER_TIMEOUT_SECONDS = 280
SSH_TIMEOUT_SECONDS = 360
REAP_TIMEOUT_SECONDS = 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare", help="freeze only; no device children")
    prepare.add_argument("--label", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--model", required=True, help="native model config directory")
    prepare.add_argument("--host", default=DEFAULT_HOST)
    prepare.add_argument("--ssh-config", default=DEFAULT_SSH_CONFIG)
    prepare.add_argument("--remote-parent", default=DEFAULT_REMOTE_PARENT)
    prepare.add_argument("--venv", default=DEFAULT_VENV)
    prepare.add_argument("--cann-env", default=DEFAULT_CANN_ENV)
    prepare.add_argument("--atb-env", default=DEFAULT_ATB_ENV)
    prepare.add_argument("--extra-library-path", default=DEFAULT_LIBRARY_PATH)
    run = commands.add_parser("run", help="explicitly run a previously reviewed freeze")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument(
        "--identity-sha256", required=True, help="reviewed prepared.json SHA"
    )
    run.add_argument("--case", choices=("transfer", "layout-mismatch"), required=True)
    run.add_argument("--devices", default="0,2")
    run.add_argument("--label", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--retire-receiver",
        action="store_true",
        help="opt in to plugin receiver retirement; requires a new source freeze",
    )
    remote = commands.add_parser("remote", help=argparse.SUPPRESS)
    remote.add_argument("--remote-root", type=Path, required=True)
    remote.add_argument("--operation", choices=("prepare", "run"), required=True)
    remote.add_argument("--label")
    remote.add_argument("--case", choices=("transfer", "layout-mismatch"))
    remote.add_argument("--devices", default="0,2")
    remote.add_argument("--retire-receiver", action="store_true")
    args = parser.parse_args()
    if args.action != "remote" and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", args.label
    ):
        parser.error(
            "label must contain only ASCII letters, digits, underscore or dash"
        )
    if args.action == "run":
        if not re.fullmatch(r"[0-9a-f]{64}", args.identity_sha256):
            parser.error("identity-sha256 must be the reviewed lowercase SHA256")
        if not re.fullmatch(r"\d+,\d+", args.devices):
            parser.error("devices must be two distinct nonnegative indices")
        first, second = map(int, args.devices.split(","))
        if first == second:
            parser.error("devices must be distinct")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_env(snapshot, options):
    # Existing CANN/ATB scripts extend their own paths. Retain their output and
    # the inherited LD_LIBRARY_PATH/PYTHONPATH, then prepend this snapshot only.
    script = (
        "set -e\nsource " + shlex.quote(options["cann_env"]) + " >/dev/null\n"
        "source " + shlex.quote(options["atb_env"]) + " >/dev/null\nenv -0\n"
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, timeout=30, check=True
    )
    env = {
        key.decode(): value.decode()
        for field in result.stdout.split(b"\0")
        if field
        for key, value in [field.split(b"=", 1)]
    }
    env["LD_LIBRARY_PATH"] = (
        options["extra_library_path"] + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["PATH"] = options["venv"] + "/bin:" + env.get("PATH", "")
    env["PYTHONPATH"] = str(snapshot) + ":" + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    env["VLLM_PLUGINS"] = (
        "ascend,ascend_model,ascend_model_loader,ascend_kv_connector,"
        "ascend_service_profiling"
    )
    return env


def check_idle(folder, label, env):
    result = subprocess.run(
        ["npu-smi", "info"], env=env, capture_output=True, timeout=30
    )
    (folder / f"npu-{label}.log").write_bytes(result.stdout + result.stderr)
    count = result.stdout.count(b"No running processes found in NPU")
    if result.returncode != 0 or count != EXPECTED_IDLE_BOARDS:
        raise RuntimeError(f"eight idle boards required at {label}; observed {count}")
    return count


def check_manifest(snapshot, manifest):
    checks = {}
    for name, digest in manifest.items():
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot.resolve()):
            raise ValueError("manifest path escapes snapshot")
        checks[name] = sha256(path) == digest
    if not checks or not all(checks.values()):
        raise RuntimeError("immutable snapshot source mismatch")
    return checks


def remote_prepare(remote):
    folder = remote / "prepare"
    folder.mkdir()
    summary = {"success": False, "device_probe_started": False, "action": "prepare"}
    try:
        snapshot = remote / "snapshot"
        snapshot.mkdir()
        with tarfile.open(remote / "source.tar.gz") as archive:
            names = archive.getnames()
            if len(names) != len(set(names)):
                raise ValueError("archive has duplicate members")
            archive.extractall(snapshot, filter="data")
        manifest = json.loads((snapshot / "manifest.json").read_text())
        summary["snapshot_sources"] = check_manifest(snapshot, manifest)
        if sha256(Path(__file__)) != manifest[HARNESS]:
            raise RuntimeError("uploaded runner differs from snapshot runner")
        if len(SOURCE_EXPECTED) != SOURCE_PIN_COUNT:
            raise RuntimeError("reviewed 33-source identity is required")
        options = json.loads((remote / "options.json").read_text())
        env = prepare_env(snapshot, options)
        metadata = subprocess.run(
            [
                options["venv"] + "/bin/python",
                "-B",
                "-c",
                "import sys,sysconfig,json,importlib.metadata as m;"
                "print(json.dumps({'site':sysconfig.get_paths()['purelib'],"
                "'python':sys.version,'torch_imported':'torch' in sys.modules,"
                "'torch_npu_imported':'torch_npu' in sys.modules,"
                "'mooncake_imported':'mooncake' in sys.modules,"
                "'packages':{k:m.version(k) for k in "
                "['torch','torch-npu','vllm','vllm-ascend','fastapi',"
                "'mooncake-transfer-engine-npu']}},indent=2))",
            ],
            env=env,
            capture_output=True,
            timeout=30,
            check=True,
        )
        versions = json.loads(metadata.stdout)
        write_json(folder / "standalone-no-torch-preflight.json", versions)
        if any(
            versions[key]
            for key in ("torch_imported", "torch_npu_imported", "mooncake_imported")
        ):
            raise RuntimeError("prepare-only imported a device/transfer module")
        site = Path(versions["site"])
        actual = {name: sha256(site / name) for name in SOURCE_EXPECTED}
        checks = {
            name: actual[name] == digest for name, digest in SOURCE_EXPECTED.items()
        }
        write_json(folder / "source-check.json", checks)
        write_json(folder / "installed-source-identity.json", actual)
        if not all(checks.values()):
            raise RuntimeError("installed source pins mismatch; do not run")
        binaries = {name: sha256(site / name) for name in BINARY_PATHS}
        write_json(folder / "binary-identity.json", binaries)
        write_json(
            folder / "effective-env.json",
            {
                key: env.get(key)
                for key in (
                    "PYTHONPATH",
                    "LD_LIBRARY_PATH",
                    "ASCEND_HOME_PATH",
                    "ATB_HOME_PATH",
                    "VLLM_PLUGINS",
                )
            },
        )
        if not Path(options["model"]).is_dir():
            raise ValueError("model config directory does not exist")
        model_root = Path(options["model"])
        model_files = {model_root / "config.json"}
        generation_config = model_root / "generation_config.json"
        if generation_config.exists():
            model_files.add(generation_config)
        model_files.update(model_root.glob("*.py"))
        model_expected = {path.name: sha256(path) for path in sorted(model_files)}
        write_json(folder / "model-config-identity.json", model_expected)
        summary["idle_boards"] = check_idle(folder, "prepare", env)
        identity = {
            "scope": (
                "PD controlled-slab NPU component; preparation only, no device child"
            ),
            "snapshot_root": str(snapshot),
            "source_expected": dict(SOURCE_EXPECTED),
            "binary_expected": binaries,
            "manifest": manifest,
            "options": options,
            "packages": versions["packages"],
            "model_config_expected": model_expected,
        }
        write_json(remote / "prepared.json", identity)
        summary["prepared_identity_sha256"] = sha256(remote / "prepared.json")
        summary["source_pins"] = checks
        summary["success"] = True
    except BaseException:
        summary["error"] = traceback.format_exc()
    write_json(folder / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["success"] else 1


def remote_run(args):
    remote = args.remote_root
    folder = remote / "runs" / args.label
    folder.mkdir(parents=True, exist_ok=False)
    identity = json.loads((remote / "prepared.json").read_text())
    snapshot, options = Path(identity["snapshot_root"]), identity["options"]
    summary = {
        "success": False,
        "device_probe_started": False,
        "outer_timed_out": False,
        "killed_owned_pids": [],
    }
    process = env = supervision = None
    owned = {}
    started = time.monotonic()
    try:
        summary["snapshot_sources"] = check_manifest(snapshot, identity["manifest"])
        if sha256(Path(__file__)) != identity["manifest"][HARNESS]:
            raise RuntimeError("runner differs from prepared identity")
        env = prepare_env(snapshot, options)
        sys.path.insert(0, str(snapshot / "tests/npu"))
        supervision = load_module(snapshot / SUPERVISION, "pd_owned_processes")
        preflight = subprocess.run(
            [
                options["venv"] + "/bin/python",
                "-B",
                "-c",
                "import sys,json;from pathlib import Path;"
                "sys.path.insert(0,sys.argv[1]);import pd_kv_layout_probe as p;"
                "print(json.dumps(p.source_preflight(Path(sys.argv[2])),indent=2))",
                str(snapshot / "tests/npu"),
                str(remote / "prepared.json"),
            ],
            cwd=snapshot,
            env=env,
            capture_output=True,
            timeout=30,
        )
        (folder / "source-preflight.log").write_bytes(
            preflight.stdout + preflight.stderr
        )
        if preflight.returncode:
            raise RuntimeError("venv source/binary preflight failed; no device launch")
        write_json(folder / "source-preflight.json", json.loads(preflight.stdout))
        summary["idle_boards_before"] = check_idle(folder, "before", env)
        command = [
            options["venv"] + "/bin/python",
            "-B",
            str(snapshot / PROBE),
            "--case",
            args.case,
            "--devices",
            args.devices,
            "--model",
            options["model"],
            "--source-identity",
            str(remote / "prepared.json"),
            "--output-dir",
            str(folder / "probe"),
            "--timeout",
            str(INNER_TIMEOUT_SECONDS),
        ]
        if args.retire_receiver:
            command.append("--retire-receiver")
        summary["receiver_retirement_requested"] = args.retire_receiver
        summary["command"] = command
        with (folder / "probe.log").open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=snapshot,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            summary["device_probe_started"] = True
            while process.poll() is None:
                supervision.observe_owned(process.pid, owned)
                if time.monotonic() - started >= OUTER_TIMEOUT_SECONDS:
                    summary["outer_timed_out"] = True
                    break
                time.sleep(0.1)
            if process.poll() is not None:
                summary["returncode"] = process.returncode
                result_path = folder / "probe/result.json"
                summary["component"] = json.loads(result_path.read_text())
                summary["success"] = (
                    process.returncode == 0 and summary["component"]["success"]
                )
    except BaseException:
        summary["error"] = traceback.format_exc()
    finally:
        cleanup_errors = []
        if process is not None and process.poll() is None:
            summary["success"] = False
            try:
                summary["killed_owned_pids"] = supervision.kill_owned(owned)
                process.wait(timeout=REAP_TIMEOUT_SECONDS)
            except BaseException:
                cleanup_errors.append(traceback.format_exc())
        summary["owned_pid_start_ticks"] = owned
        summary["cleanup_errors"] = cleanup_errors
        if env is not None:
            try:
                summary["idle_boards_after"] = check_idle(folder, "after", env)
            except BaseException:
                summary["postcheck_error"] = traceback.format_exc()
                summary["success"] = False
        else:
            summary["postcheck_environment_not_ready"] = True
        summary["success"] = (
            summary["success"] and not cleanup_errors and not summary["outer_timed_out"]
        )
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(folder / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["success"] else 1


def ssh(options, command, *, data=None, timeout=SSH_TIMEOUT_SECONDS):
    command_line = ["ssh"]
    if options["ssh_config"]:
        command_line.extend(["-F", options["ssh_config"]])
    command_line.extend(
        ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", options["host"], command]
    )
    return subprocess.run(
        command_line, input=data, capture_output=True, timeout=timeout
    )


def ssh_checked(options, command, data=None):
    result = ssh(options, command, data=data)
    if result.returncode:
        raise RuntimeError(
            f"SSH failed ({result.returncode}): "
            + result.stderr.decode(errors="replace")
        )
    return result.stdout


def retrieve(options, remote, relative, output):
    raw = ssh_checked(
        options, "tar -czf - -C " + shlex.quote(remote) + " " + shlex.quote(relative)
    )
    archive_path = output / "results.tar.gz"
    archive_path.write_bytes(raw)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(output, filter="data")
    return sha256(archive_path)


def local_prepare(args):
    root = Path(__file__).resolve().parents[2]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = sorted(root.glob("afd_plugin/**/*.py"))
    files.extend(root / name for name in (PROBE, SUPERVISION, CONFIG_HELPER, HARNESS))
    contents = {path.relative_to(root).as_posix(): path.read_bytes() for path in files}
    manifest = {
        name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, data in {
            **contents,
            "manifest.json": json.dumps(manifest).encode(),
        }.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
    options = {
        key: value
        for key, value in vars(args).items()
        if key not in ("output_dir", "action")
    }
    remote = (
        ssh_checked(
            options,
            "mktemp -d "
            + shlex.quote(args.remote_parent + "/pd-layout-" + args.label + "-XXXXXX"),
        )
        .decode()
        .strip()
    )
    if not PurePosixPath(remote).is_relative_to(PurePosixPath(args.remote_parent)):
        raise ValueError("remote temporary path escapes named workspace")
    options["remote_root"] = remote
    write_json(output / "options.json", options)
    write_json(output / "manifest.json", manifest)
    archive_data = stream.getvalue()
    (output / "source.tar.gz").write_bytes(archive_data)
    ssh_checked(
        options, "cat > " + shlex.quote(remote + "/source.tar.gz"), archive_data
    )
    ssh_checked(
        options, "cat > " + shlex.quote(remote + "/runner.py"), contents[HARNESS]
    )
    ssh_checked(
        options,
        "cat > " + shlex.quote(remote + "/options.json"),
        json.dumps(options).encode(),
    )
    result = ssh(
        options,
        "python3 -B "
        + shlex.quote(remote + "/runner.py")
        + " remote --remote-root "
        + shlex.quote(remote)
        + " --operation prepare",
    )
    (output / "remote-prepare.log").write_bytes(result.stdout + result.stderr)
    identity_result = ssh(options, "cat " + shlex.quote(remote + "/prepared.json"))
    if identity_result.returncode == 0:
        (output / "prepared.json").write_bytes(identity_result.stdout)
    archive_hash = retrieve(options, remote, "prepare", output)
    write_json(
        output / "delivery-identity.json",
        {
            "returncode": result.returncode,
            "remote_root": remote,
            "harness_sha256": manifest[HARNESS],
            "probe_sha256": manifest[PROBE],
            "source_archive_sha256": sha256(output / "source.tar.gz"),
            "results_archive_sha256": archive_hash,
            "prepared_sha256": sha256(output / "prepared.json")
            if identity_result.returncode == 0
            else None,
            "device_probe_started": False,
        },
    )
    print(
        json.dumps(
            {
                "returncode": result.returncode,
                "remote_root": remote,
                "output": str(output),
                "device_probe_started": False,
            },
            indent=2,
        )
    )
    return result.returncode


def local_run(args):
    if sha256(args.prepared) != args.identity_sha256:
        raise ValueError("prepared identity differs from the reviewed SHA")
    identity = json.loads(args.prepared.read_text())
    options = identity["options"]
    remote = options["remote_root"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    current_identity = ssh_checked(
        options, "cat " + shlex.quote(remote + "/prepared.json")
    )
    if hashlib.sha256(current_identity).hexdigest() != args.identity_sha256:
        raise ValueError("remote identity differs from the reviewed SHA")
    command = (
        "python3 -B "
        + shlex.quote(remote + "/runner.py")
        + " remote --remote-root "
        + shlex.quote(remote)
        + " --operation run --label "
        + shlex.quote(args.label)
        + " --case "
        + args.case
        + " --devices "
        + shlex.quote(args.devices)
    )
    if args.retire_receiver:
        command += " --retire-receiver"
    result = ssh(options, command)
    (output / "remote-run.log").write_bytes(result.stdout + result.stderr)
    archive_hash = retrieve(options, remote, "runs/" + args.label, output)
    write_json(
        output / "delivery-identity.json",
        {
            "returncode": result.returncode,
            "prepared_sha256": args.identity_sha256,
            "remote_root": remote,
            "results_archive_sha256": archive_hash,
            "harness_sha256": identity["manifest"][HARNESS],
            "probe_sha256": identity["manifest"][PROBE],
        },
    )
    print(
        json.dumps({"returncode": result.returncode, "output": str(output)}, indent=2)
    )
    return result.returncode


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.action == "prepare":
        status = local_prepare(parsed)
    elif parsed.action == "run":
        status = local_run(parsed)
    elif parsed.operation == "prepare":
        status = remote_prepare(parsed.remote_root)
    else:
        status = remote_run(parsed)
    raise SystemExit(status)
