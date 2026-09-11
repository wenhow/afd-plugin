#!/usr/bin/env bash
# Source this file for the vLLM 0.23 + vllm-ascend rfc/vllm_cann stack.

export DSV4_RUNTIME_VENV="${DSV4_RUNTIME_VENV:-/mnt/workspace/code/.venvs/afd-v023-vllm-cann}"
export DSV4_VLLM_ROOT="${DSV4_VLLM_ROOT:-/mnt/workspace/code/vllm-release-v0.23.0}"
export DSV4_VLLM_ASCEND_ROOT="${DSV4_VLLM_ASCEND_ROOT:-/mnt/workspace/code/vllm-ascend-rfc-vllm-cann}"
if [[ -z "${DSV4_CANN_ROOT+x}" ]]; then
  export DSV4_CANN_ROOT="/mnt/workspace/code/.ascend/cann-9.0.0/cann-9.0.0"
  export DSV4_CANN_VERSION="${DSV4_CANN_VERSION:-9.0.0}"
  export DSV4_ATB_ROOT="${DSV4_ATB_ROOT:-/mnt/workspace/code/.ascend/cann-9.0.0/nnal/atb}"
else
  export DSV4_CANN_ROOT
  export DSV4_CANN_VERSION="${DSV4_CANN_VERSION:-}"
  export DSV4_ATB_ROOT="${DSV4_ATB_ROOT:-}"
fi
DSV4_VLLM_VENV="${DSV4_VLLM_VENV:-${DSV4_RUNTIME_VENV}}"
export DSV4_VLLM_VENV

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/activate_runtime.sh"
