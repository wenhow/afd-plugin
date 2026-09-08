#!/usr/bin/env bash
# Source this file before building or running the pinned DSV4 AFD stack.

DSV4_CANN_ROOT="${DSV4_CANN_ROOT:-/mnt/workspace/code/.ascend/cann-9.0.0/cann-9.0.0}"
DSV4_VLLM_VENV="${DSV4_VLLM_VENV:-/mnt/workspace/code/.venvs/afd-v026}"
DSV4_ATB_ROOT="${DSV4_ATB_ROOT:-}"
DSV4_EXTRA_OPP_ENV="${DSV4_EXTRA_OPP_ENV:-}"

source_vendor_env() {
  local vendor_env="$1"
  local had_nounset=0
  local source_rc=0
  shift
  case $- in
    *u*) had_nounset=1 ;;
  esac
  set +u
  source "${vendor_env}" "$@" || source_rc=$?
  if (( had_nounset )); then
    set -u
  else
    set +u
  fi
  return "${source_rc}"
}

if [[ ! -f "${DSV4_CANN_ROOT}/set_env.sh" ]]; then
  echo "Missing CANN set_env.sh: ${DSV4_CANN_ROOT}/set_env.sh" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ ! -x "${DSV4_VLLM_VENV}/bin/python" ]]; then
  echo "Missing DSV4 virtual environment: ${DSV4_VLLM_VENV}" >&2
  return 2 2>/dev/null || exit 2
fi

unset ASCEND_AICPU_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_TOOLKIT_HOME
unset ASCEND_CUSTOM_OPP_PATH ATB_HOME_PATH TOOLCHAIN_HOME VIRTUAL_ENV
export CMAKE_PREFIX_PATH=
export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib
export PYTHONPATH=
export PATH=/opt/buildtools/python-3.12.9/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

if ! source_vendor_env "${DSV4_CANN_ROOT}/set_env.sh"; then
  echo "Failed to source CANN environment: ${DSV4_CANN_ROOT}/set_env.sh" >&2
  return 2 2>/dev/null || exit 2
fi
export VIRTUAL_ENV="${DSV4_VLLM_VENV}"
export PATH="${DSV4_VLLM_VENV}/bin:${PATH}"
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd

DSV4_TORCH_PACKAGE="$("${DSV4_VLLM_VENV}/bin/python" -c \
  'import importlib.util; s = importlib.util.find_spec("torch"); print(next(iter(s.submodule_search_locations or []), "") if s else "")')"
DSV4_TORCH_LIB="${DSV4_TORCH_PACKAGE}/lib"
if [[ ! -d "${DSV4_TORCH_LIB}" ]]; then
  echo "Missing torch library directory: ${DSV4_TORCH_LIB}" >&2
  return 2 2>/dev/null || exit 2
fi
export LD_LIBRARY_PATH="${DSV4_TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ -z "${DSV4_ATB_ROOT}" && -f "${DSV4_CANN_ROOT}/nnal/atb/set_env.sh" ]]; then
  DSV4_ATB_ROOT="${DSV4_CANN_ROOT}/nnal/atb"
fi
if [[ -z "${DSV4_ATB_ROOT}" && -f "$(dirname "${DSV4_CANN_ROOT}")/nnal/atb/set_env.sh" ]]; then
  DSV4_ATB_ROOT="$(dirname "${DSV4_CANN_ROOT}")/nnal/atb"
fi
if [[ -z "${DSV4_ATB_ROOT}" && -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
  DSV4_ATB_ROOT=/usr/local/Ascend/nnal/atb
fi
if [[ -z "${DSV4_ATB_ROOT}" || ! -f "${DSV4_ATB_ROOT}/set_env.sh" ]]; then
  echo "Missing NNAL/ATB set_env.sh; set DSV4_ATB_ROOT" >&2
  return 2 2>/dev/null || exit 2
fi
if ! source_vendor_env "${DSV4_ATB_ROOT}/set_env.sh"; then
  echo "Failed to source NNAL/ATB environment: ${DSV4_ATB_ROOT}/set_env.sh" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ -n "${DSV4_EXTRA_OPP_ENV}" ]]; then
  if [[ ! -f "${DSV4_EXTRA_OPP_ENV}" ]]; then
    echo "Missing extra custom OPP environment: ${DSV4_EXTRA_OPP_ENV}" >&2
    return 2 2>/dev/null || exit 2
  fi
  if ! source_vendor_env "${DSV4_EXTRA_OPP_ENV}"; then
    echo "Failed to source extra custom OPP: ${DSV4_EXTRA_OPP_ENV}" >&2
    return 2 2>/dev/null || exit 2
  fi
fi

case "${PATH}:${LD_LIBRARY_PATH:-}:${PYTHONPATH:-}:${ASCEND_HOME_PATH:-}" in
  *cann-9.1.0*)
    echo "CANN 9.1.0 leaked into the pinned DSV4 runtime" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

# Keep caller-selected roots available when a role performs a second runtime
# gate before launching vLLM.
unset DSV4_TORCH_PACKAGE DSV4_TORCH_LIB
unset -f source_vendor_env
