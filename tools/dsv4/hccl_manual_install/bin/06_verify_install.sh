#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

LOAD_VLLM_ASCEND_OPS=1
export LOAD_VLLM_ASCEND_OPS
# shellcheck source=activate_runtime.sh
source "${SCRIPT_DIR}/activate_runtime.sh"

export EXPECTED_VLLM_COMMIT="${VLLM_COMMIT}"
export EXPECTED_ASCEND_COMMIT="${VLLM_ASCEND_COMMIT}"
export EXPECTED_NPU_COUNT="$((ATTENTION_RANKS + FFN_RANKS))"
export EXPECTED_VLLM_ROOT="${VLLM_ROOT}"
export EXPECTED_ASCEND_ROOT="${VLLM_ASCEND_ROOT}"
export EXPECTED_AFD_ROOT="${AFD_PLUGIN_ROOT}"

ensure_dir "${STATE_ROOT}"
model_launch_description="${STATE_ROOT}/model-launch.json"
python "${SCRIPT_DIR}/model_launch_args.py" \
  --model-path "${MODEL_PATH}" \
  --quantization "${MODEL_QUANTIZATION}" \
  --block-size "${MODEL_BLOCK_SIZE}" \
  --safetensors-load-strategy "${MODEL_SAFETENSORS_LOAD_STRATEGY}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --describe >"${model_launch_description}"
export MODEL_LAUNCH_DESCRIPTION="${model_launch_description}"

python - <<'PY'
import json
from importlib.metadata import version
import os
from pathlib import Path

import torch
import torch_npu
import vllm
import vllm_ascend
import afd_plugin

from afd_plugin.connectors.npu.p2p_hccl import P2pHcclAFDConnector

assert torch.__version__.startswith("2.10.0"), torch.__version__
assert version("torch-npu") == "2.10.0.post2"
assert vllm.__version__.startswith("0.23.0"), vllm.__version__
expected_ascend_suffix = f"g{os.environ['EXPECTED_ASCEND_COMMIT'][:9]}"
assert version("vllm-ascend").endswith(expected_ascend_suffix)
assert version("transformers") == "5.5.4"
assert version("numpy") == "2.2.6"
assert P2pHcclAFDConnector.__name__ == "P2pHcclAFDConnector"
assert Path(vllm.__file__).resolve().is_relative_to(
    Path(os.environ["EXPECTED_VLLM_ROOT"]).resolve()
), vllm.__file__
assert Path(vllm_ascend.__file__).resolve().is_relative_to(
    Path(os.environ["EXPECTED_ASCEND_ROOT"]).resolve()
), vllm_ascend.__file__
assert Path(afd_plugin.__file__).resolve().is_relative_to(
    Path(os.environ["EXPECTED_AFD_ROOT"]).resolve()
), afd_plugin.__file__
assert torch.npu.is_available(), "torch-npu cannot see an Ascend device"
expected_npus = int(os.environ["EXPECTED_NPU_COUNT"])
assert torch.npu.device_count() >= expected_npus, (
    torch.npu.device_count(),
    expected_npus,
)
model_launch = json.loads(Path(os.environ["MODEL_LAUNCH_DESCRIPTION"]).read_text())
if model_launch["profile"] == "deepseek-v4-native":
    for owner, symbol in (
        (torch, "float8_e8m0fnu"),
        (torch_npu, "float4_e2m1fn_x2"),
        (torch_npu, "npu_dynamic_mx_quant"),
        (torch_npu, "npu_quant_matmul"),
        (torch_npu, "npu_grouped_matmul_swiglu_quant_v2"),
    ):
        assert hasattr(owner, symbol), f"native MXFP runtime is missing {symbol}"

print("DSV4_AFD_HCCL_RUNTIME_OK")
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("vllm", vllm.__version__)
print("vllm_ascend", version("vllm-ascend"))
print("transformers", version("transformers"))
print("numpy", version("numpy"))
print("vllm_root", Path(vllm.__file__).resolve())
print("vllm_ascend_root", Path(vllm_ascend.__file__).resolve())
print("afd_plugin_root", Path(afd_plugin.__file__).resolve())
PY

help_file="${STATE_ROOT}/vllm-help-all.txt"
vllm serve --help=all >"${help_file}"
for required_flag in \
  --additional-config \
  --compilation-config \
  --enable-dbo \
  --speculative-config; do
  grep -q -- "${required_flag}" "${help_file}" \
    || die "vLLM CLI is missing ${required_flag}"
done

env | sort \
  | grep -E '^(ASCEND|ATB|CANN|LD_LIBRARY_PATH|PYTHONPATH|VIRTUAL_ENV|VLLM_PLUGINS)=' \
  >"${STATE_ROOT}/verified-runtime-environment.txt"

log "Install verification passed"
