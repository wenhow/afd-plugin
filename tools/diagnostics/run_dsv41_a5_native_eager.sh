#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Native TP8 control with the explicit MXFP compatibility plugin. Run on A5
# after the Linear and MoE component probes pass; this does not start AFD.

set -uo pipefail

if [[ "${1:-}" == "--help" ]]; then
    cat <<'HELP'
Usage: bash tools/diagnostics/run_dsv41_a5_native_eager.sh [--probe-only]
Optional environment: AFD_PYTHON, AFD_MODEL.
Defaults: /usr/local/python3.11.10/bin/python, /home/models/DeepSeek-V4.1-Flash.
The console prints one fresh results directory. All other output goes to files.
Run in the existing validated A5 environment; no packages are installed.
The activation probe runs before model loading. --probe-only stops after it.
HELP
    exit 0
fi
afd_probe_only=0
if [[ "${1:-}" == "--probe-only" ]]; then
    afd_probe_only=1
    shift
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
afd_native_run=$(mktemp -d "$afd_repo_root/results/dsv41-native-tp8.XXXXXX") || exit 1
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

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling,afd_dsv41_mxfp8
export AFD_DSV41_MXFP8_SCALE_PATCH=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export PYTHONUNBUFFERED=1

{
    date -u
    "$afd_python" -V
    "$afd_python" -m pip show torch torch-npu vllm vllm-ascend fastapi
    git rev-parse HEAD
    git status --short
    git -C /opt/vllm-ascend rev-parse HEAD
    git -C /opt/vllm-ascend status --short
    printf 'python=%s\nmodel=%s\n' "$afd_python" "$afd_model"
    printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
    printf 'VLLM_PLUGINS=%s\n' "$VLLM_PLUGINS"
    printf 'AFD_DSV41_MXFP8_SCALE_PATCH=%s\n' "$AFD_DSV41_MXFP8_SCALE_PATCH"
} > "$afd_native_run/environment.txt" 2>&1
npu-smi info > "$afd_native_run/npu-before.log" 2>&1 || exit 1

"$afd_python" - "$afd_model" <<'PY' > "$afd_native_run/preflight.json" 2> "$afd_native_run/preflight.log"
import hashlib
import importlib.util
import json
import sys
from importlib.metadata import entry_points
from pathlib import Path

root = Path(sys.argv[1])
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
report = {
    "checkpoint": str(root), "sha256": actual, "expected_sha256": expected,
    "shard_count": len(shards), "missing_shards": missing,
    "checkpoint_matches": matches, "plugin_entrypoint_matches": plugin_matches,
    "module_paths": module_paths, "general_plugins": plugins,
    "ascend_source_sha256": ascend_source_sha256,
}
print(json.dumps(report, indent=2))
raise SystemExit(0 if matches and plugin_matches else 1)
PY
afd_preflight_status=$?
if (( afd_preflight_status != 0 )); then
    printf 'Preflight failed; see preflight.json and preflight.log.\n'
    exit "$afd_preflight_status"
fi

printf 'Running W13/SwiGLU/W2 probe; see activation-probe.json and .log.\n'
"$afd_python" -B tests/npu/dsv41_mxfp4_activation_probe.py --device 0 \
    --output "$afd_native_run/activation-probe.json" \
    > "$afd_native_run/activation-probe.log" 2>&1
afd_probe_status=$?
printf '%s\n' "$afd_probe_status" > "$afd_native_run/activation-probe-exit-code.txt"
if (( afd_probe_status != 0 )); then
    printf 'Activation probe failed; full model not started.\n'
    exit "$afd_probe_status"
fi
if (( afd_probe_only )); then
    exit 0
fi

printf 'Starting native TP8; see native-server.log.\n'
"$afd_python" -B -m vllm.entrypoints.cli.main serve "$afd_model" \
    --host 127.0.0.1 --port 18030 --served-model-name dsv41-native \
    --tensor-parallel-size 8 --pipeline-parallel-size 1 --data-parallel-size 1 \
    --prefill-context-parallel-size 1 --decode-context-parallel-size 1 \
    --no-enable-expert-parallel --no-enable-dbo --ubatch-size 0 \
    --language-model-only --quantization deepseek_v4_fp8 --load-format safetensors \
    --engram-config.cpu_offload false \
    --engram-config.embedding_across_dp false \
    --engram-config.dp_shared_memory false \
    --enforce-eager --max-model-len 1024 --max-num-seqs 1 --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.90 --no-async-scheduling --no-enable-prefix-caching \
    --generation-config vllm \
    --additional-config '{"enable_flashcomm1":false,"enable_dsa_cp":false}' \
    > "$afd_native_run/native-server.log" 2>&1
afd_native_status=$?
printf '%s\n' "$afd_native_status" > "$afd_native_run/server-exit-code.txt"
exit "$afd_native_status"
