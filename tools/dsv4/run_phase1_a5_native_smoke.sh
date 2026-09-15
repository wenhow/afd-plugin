#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MATRIX_RUNNER="${SCRIPT_DIR}/run_phase1_a5_matrix.sh"
NATIVE_LAUNCHER="${SCRIPT_DIR}/run_v023_native_baseline.sh"
FUNCTIONAL_SMOKE_TOOL="${SCRIPT_DIR}/run_pd_functional_smoke.py"
ACTION="${1:-help}"

usage() {
  cat <<'EOF'
Usage:
  bash tools/dsv4/run_phase1_a5_native_smoke.sh preflight
  bash tools/dsv4/run_phase1_a5_native_smoke.sh run

Required for run:
  PHASE1_NATIVE_OUTPUT_ROOT  Fresh evidence directory.

Optional:
  PHASE1_NATIVE_DEVICES=0,1,2,3
  PHASE1_NATIVE_API_PORT=8900
  PHASE1_NATIVE_READY_TIMEOUT_SECONDS=3600

This starts one official-style no-AFD DP4 Graph/MTP-off service, checks health,
the model list and one completion, then stops it. It never creates or compares
golden tokens.
EOF
}

die() {
  printf '[phase1-a5-native] ERROR: %s\n' "$*" >&2
  exit 2
}

npu_process_count() {
  npu-smi info | awk '
    /^\|[[:space:]]*NPU[[:space:]]+(ID|Chip)[[:space:]]*\|[[:space:]]*Process[[:space:]]+id[[:space:]]*\|/ {
      in_process_table=1
      next
    }
    in_process_table && /^\|[[:space:]]*[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+[[:space:]]*\|/ {count++}
    END {print count + 0}
  '
}

wait_for_empty_npus() {
  local attempts=0
  while (( attempts < 30 )); do
    (( "$(npu_process_count)" == 0 )) && return 0
    attempts=$((attempts + 1))
    sleep 2
  done
  return 1
}

run_smoke() {
  local output_root="${PHASE1_NATIVE_OUTPUT_ROOT:-}"
  [[ -n "${output_root}" ]] || die "PHASE1_NATIVE_OUTPUT_ROOT is required"
  [[ ! -e "${output_root}" ]] || die "Output root already exists: ${output_root}"

  bash "${MATRIX_RUNNER}" preflight-native
  source "${SCRIPT_DIR}/activate_v023_vllm_cann_runtime.sh"
  export PYTHONPATH="${REPO_ROOT}:${DSV4_VLLM_ROOT}:${DSV4_VLLM_ASCEND_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

  local devices="${PHASE1_NATIVE_DEVICES:-${ATTENTION_DEVICES:-0,1,2,3}}"
  local api_port="${PHASE1_NATIVE_API_PORT:-8900}"
  local ready_timeout="${PHASE1_NATIVE_READY_TIMEOUT_SECONDS:-3600}"
  local stop_timeout="${PHASE1_NATIVE_STOP_TIMEOUT_SECONDS:-${STOP_TIMEOUT_SECONDS:-180}}"
  local -a device_array
  IFS=',' read -r -a device_array <<<"${devices}"
  (( ${#device_array[@]} == 4 )) || die "Native baseline requires exactly four NPU devices"
  [[ "${devices}" == "0,1,2,3" ]] \
    || die "Phase-one A5 native baseline must use devices 0,1,2,3"
  [[ "${api_port}" =~ ^[0-9]+$ ]] || die "PHASE1_NATIVE_API_PORT must be an integer"
  [[ "${ready_timeout}" =~ ^[1-9][0-9]*$ ]] \
    || die "PHASE1_NATIVE_READY_TIMEOUT_SECONDS must be positive"
  [[ "${stop_timeout}" =~ ^[1-9][0-9]*$ ]] \
    || die "PHASE1_NATIVE_STOP_TIMEOUT_SECONDS must be positive"
  (( "$(npu_process_count)" == 0 )) || die "NPU processes exist before native smoke"

  "${DSV4_RUNTIME_VENV}/bin/python" - "${api_port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
PY

  mkdir -p "${output_root}"
  npu-smi info >"${output_root}/npu-before.txt"
  {
    printf 'validation_mode=functional_smoke\n'
    printf 'golden_checked=0\n'
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'devices=%s\n' "${devices}"
    printf 'data_parallel_size=4\n'
    printf 'tensor_parallel_size=1\n'
    printf 'execution_mode=full-decode-only\n'
    printf 'enable_mtp=0\n'
    printf 'mtp_num_speculative_tokens=0\n'
    printf 'mtp_draft_execution=off\n'
    printf 'model_path=%s\n' "${MODEL_PATH}"
    printf 'model_config_sha256=%s\n' "$(sha256sum "${MODEL_PATH}/config.json" | awk '{print $1}')"
    printf 'cann_root=%s\n' "$(readlink -f "${DSV4_CANN_ROOT}")"
    printf 'vllm_commit=%s\n' "$(git -C "${DSV4_VLLM_ROOT}" rev-parse HEAD)"
    printf 'vllm_ascend_commit=%s\n' "$(git -C "${DSV4_VLLM_ASCEND_ROOT}" rev-parse HEAD)"
    printf 'afd_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  } >"${output_root}/runtime.env"

  local native_pid= ready=0 deadline forced_stop=0 completed=0 cleanup_passed=0
  cleanup_native() {
    local attempts=0
    [[ -n "${native_pid}" ]] || return
    kill -TERM -- "-${native_pid}" 2>/dev/null || true
    while (( attempts < stop_timeout )); do
      if ! kill -0 "${native_pid}" 2>/dev/null; then
        break
      fi
      attempts=$((attempts + 1))
      sleep 1
    done
    if kill -0 "${native_pid}" 2>/dev/null; then
      forced_stop=1
      kill -KILL -- "-${native_pid}" 2>/dev/null || true
    fi
    wait "${native_pid}" 2>/dev/null || true
    native_pid=
  }
  finalize() {
    local exitcode="${1:-$?}"
    cleanup_native
    if wait_for_empty_npus; then
      cleanup_passed=1
    fi
    npu-smi info >"${output_root}/npu-after-stop.txt" 2>&1 || true
    {
      printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
      printf 'passed=%s\n' "${completed}"
      printf 'exitcode=%s\n' "${exitcode}"
      printf 'forced_stop=%s\n' "${forced_stop}"
      printf 'npu_cleanup_passed=%s\n' "${cleanup_passed}"
    } >"${output_root}/summary.env"
  }
  trap 'finalize $?' EXIT
  trap 'exit 130' TERM INT

  setsid env \
    API_HOST=127.0.0.1 API_PORT="${api_port}" \
    DATA_PARALLEL_RPC_PORT=$((api_port + 20000)) \
    MASTER_PORT=$((api_port + 20001)) \
    HCCL_IF_BASE_PORT=$((api_port + 44000)) \
    ASCEND_RT_VISIBLE_DEVICES="${devices}" \
    EXECUTION_MODE=full-decode-only \
    ENABLE_MTP=0 \
    MTP_NUM_SPECULATIVE_TOKENS=1 \
    MTP_DRAFT_EXECUTION=eager \
    TENSOR_PARALLEL_SIZE=1 \
    bash "${NATIVE_LAUNCHER}" \
    >"${output_root}/server.log" 2>&1 &
  native_pid=$!
  printf '%s\n' "${native_pid}" >"${output_root}/server.pid"

  deadline=$((SECONDS + ready_timeout))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://127.0.0.1:${api_port}/health" \
      >"${output_root}/health.json" 2>"${output_root}/health.stderr"; then
      ready=1
      break
    fi
    if ! kill -0 "${native_pid}" 2>/dev/null; then
      tail -160 "${output_root}/server.log" >&2
      die "Native service exited before ready"
    fi
    sleep 2
  done
  (( ready == 1 )) || die "Native service readiness timed out"

  curl -fsS "http://127.0.0.1:${api_port}/v1/models" \
    >"${output_root}/models.json"
  npu-smi info >"${output_root}/npu-ready.txt"
  "${DSV4_RUNTIME_VENV}/bin/python" "${FUNCTIONAL_SMOKE_TOOL}" \
    --endpoint "http://127.0.0.1:${api_port}/v1/completions" \
    --model dsv4-v023-native \
    --batch-sizes "1" \
    --output "${output_root}/functional_smoke.json"

  cleanup_native
  (( forced_stop == 0 )) || die "Native service required SIGKILL"
  wait_for_empty_npus || die "NPU cleanup failed after native smoke"
  cleanup_passed=1
  npu-smi info >"${output_root}/npu-after-stop.txt"
  if grep -Eiq \
    'EngineCore encountered a fatal error|RuntimeError: Worker failed with error|Communication_Error_Bind_IP_Port|error code is 507015' \
    "${output_root}/server.log"; then
    die "Fatal marker found in native service log"
  fi
  completed=1
  trap - EXIT TERM INT
  finalize 0
  printf '[phase1-a5-native] completed: %s\n' "${output_root}"
}

case "${ACTION}" in
  help|-h|--help) usage ;;
  preflight) bash "${MATRIX_RUNNER}" preflight-native ;;
  run) run_smoke ;;
  *) usage; die "Unknown action: ${ACTION}" ;;
esac
