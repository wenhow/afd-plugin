#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

preflight_scope="${1:-install}"
case "${preflight_scope}" in
  install|runtime) ;;
  *) die "Usage: $0 [install|runtime]" ;;
esac

required_commands=(bash awk grep curl npu-smi setsid nohup ps)
if [[ "${preflight_scope}" == "install" ]]; then
  required_commands+=(git tar sed find)
fi
for command_name in "${required_commands[@]}"; do
  require_command "${command_name}"
done

architecture="$(uname -m)"
if [[ "${preflight_scope}" == "install" \
  && "${architecture}" != "aarch64" ]] \
  && ! is_true "${ALLOW_NON_AARCH64}"; then
  die "Expected aarch64, got ${architecture}. Set ALLOW_NON_AARCH64=1 only for non-runtime checks."
fi

require_file "${CANN_ROOT}/set_env.sh"
resolved_cann="$(readlink -f "${CANN_ROOT}")"
cann_version_text="${resolved_cann}"
if [[ -x "${CANN_ROOT}/query_pkg_version.sh" ]]; then
  cann_version_text+=$'\n'"$("${CANN_ROOT}/query_pkg_version.sh" 2>&1 || true)"
fi
if [[ "${cann_version_text}" != *"9.0.0"* ]] \
  && ! is_true "${ALLOW_CANN_VERSION_MISMATCH}"; then
  die "CANN 9.0.0 not detected at ${resolved_cann}"
fi

if [[ "${CANN_ROOT}" == *"/home/develp/"* ]]; then
  die "CANN_ROOT contains the known /home/develp typo"
fi

if [[ "${PATH}:${LD_LIBRARY_PATH:-}:${PYTHONPATH:-}:${CMAKE_PREFIX_PATH:-}" == *"cann-9.1"* ]]; then
  die "Current shell contains CANN 9.1 paths. Open a clean shell before continuing."
fi

if [[ "${preflight_scope}" == "install" ]]; then
  if [[ "${PYTHON_BIN}" == */* ]]; then
    [[ -x "${PYTHON_BIN}" ]] || die "Python is not executable: ${PYTHON_BIN}"
  else
    require_command "${PYTHON_BIN}"
  fi
fi

if command -v ip >/dev/null 2>&1; then
  ip link show dev "${NIC_NAME}" >/dev/null 2>&1 \
    || die "Network interface does not exist: ${NIC_NAME}"
else
  [[ -d "/sys/class/net/${NIC_NAME}" ]] \
    || die "Network interface does not exist: ${NIC_NAME}"
  [[ -n "${HCCL_IF_IP}" ]] \
    || die "iproute2 is unavailable; set HCCL_IF_IP explicitly"
fi
resolved_ip="$(resolve_hccl_ip)"
[[ -n "${resolved_ip}" ]] || die "Could not resolve an IPv4 address on ${NIC_NAME}"

assert_positive_integer ATTENTION_RANKS "${ATTENTION_RANKS}"
assert_positive_integer FFN_RANKS "${FFN_RANKS}"
assert_zero_or_one ENABLE_MTP "${ENABLE_MTP}"
[[ "$(device_list_count "${ATTENTION_DEVICES}")" == "${ATTENTION_RANKS}" ]] \
  || die "ATTENTION_DEVICES count does not match ATTENTION_RANKS"
[[ "$(device_list_count "${FFN_DEVICES}")" == "${FFN_RANKS}" ]] \
  || die "FFN_DEVICES count does not match FFN_RANKS"
device_lists_are_disjoint "${ATTENTION_DEVICES}" "${FFN_DEVICES}" \
  || die "ATTENTION_DEVICES and FFN_DEVICES must be disjoint"
larger_ranks=$((ATTENTION_RANKS > FFN_RANKS ? ATTENTION_RANKS : FFN_RANKS))
smaller_ranks=$((ATTENTION_RANKS < FFN_RANKS ? ATTENTION_RANKS : FFN_RANKS))
(( larger_ranks % smaller_ranks == 0 )) \
  || die "Attention and FFN ranks must have an integer ratio"
ratio=$((larger_ranks / smaller_ranks))
if (( FFN_RANKS > ATTENTION_RANKS )); then
  required_ffn_tokens=$(((ATTENTION_MAX_NUM_BATCHED_TOKENS + ratio - 1) / ratio))
else
  required_ffn_tokens=$((ATTENTION_MAX_NUM_BATCHED_TOKENS * ratio))
fi
(( FFN_MAX_NUM_BATCHED_TOKENS >= required_ffn_tokens )) \
  || die "FFN_MAX_NUM_BATCHED_TOKENS must be at least ${required_ffn_tokens}"

npu_list="$(npu-smi info -l)"
npu_chip_count="$(awk -F: '/Chip Count/ {gsub(/[[:space:]]/, "", $2); sum += $2} END {print sum + 0}' <<<"${npu_list}")"
required_npu_count=$((ATTENTION_RANKS + FFN_RANKS))
if (( npu_chip_count < required_npu_count )); then
  die "A${ATTENTION_RANKS}F${FFN_RANKS} requires ${required_npu_count} NPU chips, detected ${npu_chip_count}"
fi

case "${EXECUTION_MODE}" in
  eager|full-decode-only) ;;
  *) die "Unsupported EXECUTION_MODE=${EXECUTION_MODE}" ;;
esac
case "${U_BATCHES}" in
  1|2) ;;
  *) die "U_BATCHES must be 1 or 2" ;;
esac
if [[ "${ENABLE_MTP}" == "1" ]]; then
  [[ "${MTP_NUM_SPECULATIVE_TOKENS}" =~ ^[1-3]$ ]] \
    || die "MTP supports num_speculative_tokens in [1, 3]"
  case "${EXECUTION_MODE}:${MTP_DRAFT_EXECUTION}" in
    eager:eager|full-decode-only:eager|full-decode-only:graph) ;;
    *) die "Unsupported target/draft execution: ${EXECUTION_MODE}/${MTP_DRAFT_EXECUTION}" ;;
  esac
fi

if [[ "${preflight_scope}" == "install" ]] && is_true "${OFFLINE}"; then
  require_dir "${WHEELHOUSE}"
fi

log "${preflight_scope^} preflight passed: ${architecture}, CANN=${resolved_cann}, NPU chips=${npu_chip_count}, HCCL IP=${resolved_ip}"
