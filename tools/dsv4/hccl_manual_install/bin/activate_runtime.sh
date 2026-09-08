#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

require_file "${CANN_ROOT}/set_env.sh"
require_file "${VENV_ROOT}/bin/python"

unset ASCEND_AICPU_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_TOOLKIT_HOME
unset ASCEND_CUSTOM_OPP_PATH ATB_HOME_PATH TOOLCHAIN_HOME VIRTUAL_ENV
export CMAKE_PREFIX_PATH=
export PYTHONPATH=
export PATH="${SYSTEM_PATH}"
export LD_LIBRARY_PATH="${PYTHON_LIBRARY_PATH}"

# CANN first, then the selected venv. This prevents another venv or toolkit
# from leaking into the deployment process.
# shellcheck disable=SC1090
source "${CANN_ROOT}/set_env.sh"

export VIRTUAL_ENV="${VENV_ROOT}"
export PATH="${VENV_ROOT}/bin:${PATH}"

# NNAL probes torch's C++ ABI with python3, so source it only after the venv is
# on PATH.
if [[ -f "${CANN_ROOT}/nnal/atb/set_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CANN_ROOT}/nnal/atb/set_env.sh"
fi

export DSV4_CANN_ROOT="${CANN_ROOT}"
export DSV4_RUNTIME_VENV="${VENV_ROOT}"
export DSV4_VLLM_ROOT="${VLLM_ROOT}"
export DSV4_VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT}"
export DSV4_VLLM_VENV="${VENV_ROOT}"
export SOC_VERSION
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd

case "${PATH}:${LD_LIBRARY_PATH:-}:${PYTHONPATH:-}:${ASCEND_HOME_PATH:-}" in
  *cann-9.1*)
    die "CANN 9.1 leaked into the fixed 9.0.0 runtime"
    ;;
esac

if is_true "${LOAD_VLLM_ASCEND_OPS:-0}"; then
  ops_env="${VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
  require_file "${ops_env}"
  # shellcheck disable=SC1090
  source "${ops_env}"
fi
