#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Native DP8/TP1/EP8 eager baseline with CPU-offloaded, shared Engram tables.
# The optional TP8 compatibility probe is independent of this baseline.

set -uo pipefail

if [[ "${1:-}" == "--help" ]]; then
    cat <<'HELP'
Usage: bash tools/diagnostics/run_dsv41_a5_native_eager.sh [--probe-only | --dspark N]
Optional environment: AFD_PYTHON, AFD_MODEL, AFD_TRANSFORMER_VENDOR.
Defaults: /usr/local/python3.11.10/bin/python, /home/models/DeepSeek-V4.1-Flash.
Transformer vendor defaults to $ASCEND_OPP_PATH/vendors/custom_transformer.
The console prints one fresh results directory. All other output goes to files.
Run in the existing validated A5 environment; no packages are installed.
Default: DP8/TP1/EP8, shared CPU Engram, eager, DSpark off, small text requests.
The default does not load AFD or its TP8 quantization compatibility patches.
--probe-only runs the separate TP8 activation diagnostic without serving.
--dspark N enables native same-checkpoint DSpark, N=1, 2 or 3, draft TP1/eager.
The complete draft increases HBM use. Start with one N=1 run on idle devices.
DSpark mode keeps DP8/TP1/EP8 and never loads the AFD runtime plugin.
HELP
    exit 0
fi
afd_probe_only=0
afd_dspark_tokens=0
if [[ "${1:-}" == "--probe-only" ]]; then
    afd_probe_only=1
    shift
elif [[ "${1:-}" == "--dspark" ]]; then
    case "${2:-}" in
        1|2|3) afd_dspark_tokens=$2 ;;
        *) printf '%s\n' '--dspark requires N=1, 2 or 3; use --help.' >&2; exit 2 ;;
    esac
    shift 2
fi
if (( $# != 0 )); then
    printf 'Unexpected argument; use --help.\n' >&2
    exit 2
fi

afd_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd) || exit 1
cd -- "$afd_repo_root" || exit 1
afd_python=${AFD_PYTHON:-/usr/local/python3.11.10/bin/python}
afd_model=${AFD_MODEL:-/home/models/DeepSeek-V4.1-Flash}
mkdir -p "$afd_repo_root/results" || exit 1
afd_run_prefix=dsv41-native-dp8-ep8
if (( afd_probe_only )); then
    afd_run_prefix=dsv41-tp8-activation-probe
elif (( afd_dspark_tokens )); then
    afd_run_prefix=dsv41-native-dspark-n$afd_dspark_tokens-dp8-ep8
fi
afd_native_run=$(mktemp -d "$afd_repo_root/results/$afd_run_prefix.XXXXXX") || exit 1
printf 'results: %s\n' "$afd_native_run"
exec > "$afd_native_run/launcher.log" 2>&1

finish() {
    local afd_exit_code=$?
    trap - EXIT
    printf '%s\n' "$afd_exit_code" > "$afd_native_run/launcher-exit-code.txt"
    npu-smi info > "$afd_native_run/npu-after.log" 2>&1 || true
    date -u
    exit "$afd_exit_code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# The A5 library probe found CompressorV2 in this vendor, not the built-in
# opapi libraries. Export before importing native extensions or spawning workers.
afd_opp_root=${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.2.0}/opp}
afd_transformer_vendor=${AFD_TRANSFORMER_VENDOR:-$afd_opp_root/vendors/custom_transformer}
if (( ! afd_probe_only )); then
    export ASCEND_CUSTOM_OPP_PATH="$afd_transformer_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
    export LD_LIBRARY_PATH="$afd_transformer_vendor/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling
export AFD_DSV41_MXFP8_SCALE_PATCH=0
if (( afd_probe_only )); then
    export VLLM_PLUGINS="$VLLM_PLUGINS,afd_dsv41_mxfp8"
    export AFD_DSV41_MXFP8_SCALE_PATCH=1
fi
# Match the official 950DT communication and CPU settings.
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=1600
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=2040
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
afd_speculative_args=()
if (( afd_dspark_tokens )); then
    afd_speculative_config=$("$afd_python" -c \
        'import json,sys; print(json.dumps({"method":"dspark","num_speculative_tokens":int(sys.argv[1]),"draft_tensor_parallel_size":1,"enforce_eager":True}))' \
        "$afd_dspark_tokens") || exit 1
    afd_speculative_args=(--speculative-config "$afd_speculative_config"
        --per-request-spec-decode-metrics detailed)
    printf 'Native DSpark N=%s: full draft adds weights/KV/workspace HBM; no fit guarantee.\n' \
        "$afd_dspark_tokens"
fi

{
    date -u
    "$afd_python" -V
    "$afd_python" -m pip show torch torch-npu vllm vllm-ascend fastapi
    git rev-parse HEAD
    git status --short
    git -C /opt/vllm-ascend rev-parse HEAD
    git -C /opt/vllm-ascend status --short
    printf 'python=%s\nmodel=%s\n' "$afd_python" "$afd_model"
    printf 'transformer_vendor=%s\n' "$afd_transformer_vendor"
    printf 'ASCEND_OPP_PATH=%s\nASCEND_CUSTOM_OPP_PATH=%s\n' "${ASCEND_OPP_PATH:-}" "${ASCEND_CUSTOM_OPP_PATH:-}"
    printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH:-}"
    printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf 'VLLM_PLUGINS=%s\n' "$VLLM_PLUGINS"
    printf 'AFD_DSV41_MXFP8_SCALE_PATCH=%s\n' "$AFD_DSV41_MXFP8_SCALE_PATCH"
    printf 'tp8_compat_probe_only=%s\n' "$afd_probe_only"
    printf 'native_dspark_tokens=%s\n' "$afd_dspark_tokens"
    printf 'speculative_config=%s\n' "${afd_speculative_config:-null}"
    printf 'HCCL_OP_EXPANSION_MODE=%s\nHCCL_BUFFSIZE=%s\n' "$HCCL_OP_EXPANSION_MODE" "$HCCL_BUFFSIZE"
    printf 'HCCL_CONNECT_TIMEOUT=%s\nHCCL_EXEC_TIMEOUT=%s\n' "$HCCL_CONNECT_TIMEOUT" "$HCCL_EXEC_TIMEOUT"
    printf 'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=%s\n' "$VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"
    printf 'OMP_PROC_BIND=%s\nOMP_NUM_THREADS=%s\n' "$OMP_PROC_BIND" "$OMP_NUM_THREADS"
    printf 'PYTORCH_NPU_ALLOC_CONF=%s\n' "$PYTORCH_NPU_ALLOC_CONF"
    free -h
    df -h /dev/shm
} > "$afd_native_run/environment.txt" 2>&1
npu-smi info > "$afd_native_run/npu-before.log" 2>&1 || exit 1

if (( ! afd_probe_only )); then
    "$afd_python" - "$afd_transformer_vendor" <<'PY' > "$afd_native_run/opapi-preflight.json" 2> "$afd_native_run/opapi-preflight.log"
import ctypes
import json
import os
import sys
from pathlib import Path

library_path = Path(sys.argv[1]) / "op_api/lib/libcust_opapi.so"
symbols = ("aclnnCompressorV2", "aclnnCompressorV2GetWorkspaceSize")
report = {"library": str(library_path), "loaded": False, "symbols": {}}
try:
    library = ctypes.CDLL(str(library_path), mode=os.RTLD_LAZY | os.RTLD_LOCAL)
    report["loaded"] = True
    for symbol in symbols:
        try:
            library[symbol]
            report["symbols"][symbol] = True
        except AttributeError:
            report["symbols"][symbol] = False
except OSError as error:
    report["error"] = str(error)
report["passed"] = report["loaded"] and all(report["symbols"].values())
report["scope"] = "Library load and symbol lookup only; no operator execution or inference"
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 1)
PY
    afd_opapi_status=$?
    if (( afd_opapi_status != 0 )); then
        printf 'CompressorV2 library check failed; see opapi-preflight.json and .log.\n'
        exit "$afd_opapi_status"
    fi
fi

"$afd_python" - "$afd_model" "$afd_probe_only" "$afd_dspark_tokens" <<'PY' > "$afd_native_run/preflight.json" 2> "$afd_native_run/preflight.log"
import hashlib
import importlib.util
import json
import sys
from importlib.metadata import entry_points
from pathlib import Path

root = Path(sys.argv[1])
probe_only = sys.argv[2] == "1"
dspark_tokens = int(sys.argv[3])
expected = {
    "config.json": "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
    "model.safetensors.index.json": "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8",
}
actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in expected}
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (root / name).is_file()]
plugins = {entry.name: entry.value for entry in entry_points(group="vllm.general_plugins")}
module_paths = {
    name: importlib.util.find_spec(name).origin
    for name in ("afd_plugin", "vllm", "vllm_ascend")
}
ascend_root = Path(module_paths["vllm_ascend"]).parent
vllm_root = Path(module_paths["vllm"]).parent
ascend_source_sha256 = {
    name: hashlib.sha256((ascend_root / name).read_bytes()).hexdigest()
    for name in (
        "quantization/methods/w4a8/w4a8_mxfp4.py",
        "ops/fused_moe/moe_mlp.py",
        "ops/fused_moe/dataclass/moe_mlp.py",
    )
}
matches = actual == expected and not missing and len(shards) == 48
plugin_matches = plugins.get("afd_dsv41_mxfp8") == "afd_plugin:register_dsv41_mxfp8_scale_patch"
dspark_source_sha256 = {}
missing_dspark_sources = []
if dspark_tokens:
    for package, package_root, paths in (
        ("vllm", vllm_root, (
            "config/speculative.py", "config/observability.py",
            "v1/worker/gpu/spec_decode/dspark/speculator.py",
            "v1/worker/gpu/spec_decode/dspark/utils.py",
            "v1/worker/gpu/spec_decode/dflash/speculator.py",
            "v1/worker/gpu/spec_decode/rejection_sampler.py",
            "v1/spec_decode/metrics.py",
        )),
        ("vllm_ascend", ascend_root, (
            "worker/v2/spec_decode/dspark/speculator.py",
            "models/deepseek_v41/dspark.py",
            "models/deepseek_v41/engram/model_state.py",
        )),
    ):
        for name in paths:
            source_path = package_root / name
            source_key = package + "/" + name
            if source_path.is_file():
                dspark_source_sha256[source_key] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            else:
                missing_dspark_sources.append(source_key)
report = {
    "mode": "tp8_compat_activation_probe" if probe_only else (
        "native_dspark_dp8_tp1_ep8_eager" if dspark_tokens else "native_dp8_tp1_ep8_eager"
    ),
    "checkpoint": str(root), "sha256": actual, "expected_sha256": expected,
    "shard_count": len(shards), "missing_shards": missing,
    "checkpoint_matches": matches, "plugin_entrypoint_matches": plugin_matches,
    "compat_plugin_required": probe_only,
    "module_paths": module_paths, "general_plugins": plugins,
    "ascend_source_sha256": ascend_source_sha256,
    "speculative_config": {
        "method": "dspark", "num_speculative_tokens": dspark_tokens,
        "draft_tensor_parallel_size": 1, "enforce_eager": True,
    } if dspark_tokens else None,
    "dspark_source_sha256": dspark_source_sha256,
    "missing_dspark_sources": missing_dspark_sources,
    "dspark_source_scope": "File identity only; native configuration validates at startup",
}
print(json.dumps(report, indent=2))
raise SystemExit(0 if matches and (not probe_only or plugin_matches) and not missing_dspark_sources else 1)
PY
afd_preflight_status=$?
if (( afd_preflight_status != 0 )); then
    printf 'Preflight failed; see preflight.json and preflight.log.\n'
    exit "$afd_preflight_status"
fi

if (( afd_probe_only )); then
    printf 'Running separate TP8 W13/SwiGLU/W2 diagnostic; see activation-probe.json and .log.\n'
    "$afd_python" -B tests/npu/dsv41_mxfp4_activation_probe.py --device 0 \
        --output "$afd_native_run/activation-probe.json" \
        > "$afd_native_run/activation-probe.log" 2>&1
    afd_probe_status=$?
    printf '%s\n' "$afd_probe_status" > "$afd_native_run/activation-probe-exit-code.txt"
    exit "$afd_probe_status"
fi

printf 'Starting native DP8/TP1/EP8 with shared CPU Engram; see native-server.log.\n'
"$afd_python" -B -m vllm.entrypoints.cli.main serve "$afd_model" \
    --host 127.0.0.1 --port 18030 --served-model-name dsv41-native \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 8 \
    --prefill-context-parallel-size 1 --decode-context-parallel-size 1 \
    --enable-expert-parallel --no-enable-dbo --ubatch-size 0 \
    --language-model-only --quantization deepseek_v4_fp8 --load-format safetensors \
    --tokenizer-mode deepseek_v41 --trust-remote-code --block-size 128 \
    --engram-config.cpu_offload true \
    --engram-config.embedding_across_dp false \
    --engram-config.dp_shared_memory true \
    --enforce-eager --max-model-len 1024 --max-num-seqs 1 --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.90 --no-async-scheduling --no-enable-prefix-caching \
    "${afd_speculative_args[@]}" \
    --generation-config vllm \
    --additional-config '{
        "enable_cpu_binding":true,
        "multistream_engram_overlap":true,
        "multistream_overlap_shared_expert":true,
        "multistream_dsv4_dsa_overlap":true,
        "combine_quant_mode":4,
        "enable_flashcomm1":false,
        "enable_dsa_cp":false
    }' \
    > "$afd_native_run/native-server.log" 2>&1
afd_native_status=$?
printf '%s\n' "$afd_native_status" > "$afd_native_run/server-exit-code.txt"
exit "$afd_native_status"
