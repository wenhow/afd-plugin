#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

require_file "${VLLM_ROOT}/setup.py"
require_file "${VLLM_ASCEND_ROOT}/setup.py"
require_file "${AFD_PLUGIN_ROOT}/setup.py"

LOAD_VLLM_ASCEND_OPS=0
export LOAD_VLLM_ASCEND_OPS
# shellcheck source=activate_runtime.sh
source "${SCRIPT_DIR}/activate_runtime.sh"

ensure_dir "${STATE_ROOT}"
ensure_dir "${INSTALL_ROOT}/build-tmp"
export TMPDIR="${INSTALL_ROOT}/build-tmp"
export MAX_JOBS="${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN)}"

assert_zero_or_one INSTALL_UPSTREAM_STACK "${INSTALL_UPSTREAM_STACK}"
if is_true "${INSTALL_UPSTREAM_STACK}"; then
  log "Installing vLLM ${VLLM_COMMIT}"
  cd "${VLLM_ROOT}"
  VLLM_TARGET_DEVICE=empty \
  SETUPTOOLS_SCM_PRETEND_VERSION=0.23.0 \
    python -m pip install -v --no-build-isolation --no-deps --editable .

  log "Building and installing vLLM-Ascend ${VLLM_ASCEND_COMMIT}"
  cd "${VLLM_ASCEND_ROOT}"
  SETUPTOOLS_SCM_PRETEND_VERSION=0.1.dev1+g3da28f941 \
    python -m pip install -v --no-build-isolation --no-deps --editable .
else
  log "Reusing the validated vLLM and vLLM-Ascend editable installations"
fi

ops_env="${VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
require_file "${ops_env}"
require_file "${VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so"

log "Installing afd-plugin snapshot ${AFD_SNAPSHOT_ID}"
cd "${AFD_PLUGIN_ROOT}"
AFD_BUILD_ASCEND_OPS="${AFD_BUILD_ASCEND_OPS}" \
SETUPTOOLS_SCM_PRETEND_VERSION=1 \
  python -m pip install -v --no-build-isolation --no-deps --editable .

python -m pip list --format=freeze >"${STATE_ROOT}/python-packages-installed.txt"
log "Stack installation completed"
