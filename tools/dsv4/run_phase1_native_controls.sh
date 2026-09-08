#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NATIVE_LAUNCHER="${SCRIPT_DIR}/run_v023_native_baseline.sh"
GOLDEN_GENERATOR="${SCRIPT_DIR}/generate_golden.py"
MATRIX_RUNNER="${SCRIPT_DIR}/run_phase1_a5_matrix.sh"
ACTION="${1:-help}"
if (( $# > 0 )); then
  shift
fi

CONTROL_KEYS=(
  eager_mtp_off
  eager_mtp_n1
  eager_mtp_n2
  graph_target_draft_eager_mtp_n2
  graph_target_draft_graph_mtp_n3
)

usage() {
  cat <<'EOF'
Usage:
  bash tools/dsv4/run_phase1_native_controls.sh list
  bash tools/dsv4/run_phase1_native_controls.sh preflight
  bash tools/dsv4/run_phase1_native_controls.sh run [control-key ...]

Required for run:
  PHASE1_GOLDEN_ROOT  Output root for five path-matched controls.

Optional:
  PHASE1_NATIVE_DEVICES=0,1,2,3,4,5,6,7
  PHASE1_NATIVE_API_PORT=8900
  PHASE1_NATIVE_READY_TIMEOUT_SECONDS=1800
EOF
}

die() {
  printf '[phase1-native] ERROR: %s\n' "$*" >&2
  exit 2
}

control_arguments() {
  local control_key="$1"
  case "${control_key}" in
    eager_mtp_off)
      CONTROL_EXECUTION_MODE=eager
      CONTROL_ENABLE_MTP=0
      CONTROL_MTP_TOKENS=0
      CONTROL_DRAFT_EXECUTION=off
      ;;
    eager_mtp_n1)
      CONTROL_EXECUTION_MODE=eager
      CONTROL_ENABLE_MTP=1
      CONTROL_MTP_TOKENS=1
      CONTROL_DRAFT_EXECUTION=eager
      ;;
    eager_mtp_n2)
      CONTROL_EXECUTION_MODE=eager
      CONTROL_ENABLE_MTP=1
      CONTROL_MTP_TOKENS=2
      CONTROL_DRAFT_EXECUTION=eager
      ;;
    graph_target_draft_eager_mtp_n2)
      CONTROL_EXECUTION_MODE=full-decode-only
      CONTROL_ENABLE_MTP=1
      CONTROL_MTP_TOKENS=2
      CONTROL_DRAFT_EXECUTION=eager
      ;;
    graph_target_draft_graph_mtp_n3)
      CONTROL_EXECUTION_MODE=full-decode-only
      CONTROL_ENABLE_MTP=1
      CONTROL_MTP_TOKENS=3
      CONTROL_DRAFT_EXECUTION=graph
      ;;
    *) die "Unknown control key: ${control_key}" ;;
  esac
}

npu_process_count() {
  npu-smi info | awk '
    /\| NPU +Chip +\| Process id/ {in_process_table=1; next}
    in_process_table && /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+/ {count++}
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

run_controls() {
  local output_root="${PHASE1_GOLDEN_ROOT:-}"
  [[ -n "${output_root}" ]] || die "PHASE1_GOLDEN_ROOT is required"
  [[ ! -e "${output_root}" || -d "${output_root}" ]] \
    || die "PHASE1_GOLDEN_ROOT is not a directory: ${output_root}"
  mkdir -p "${output_root}"

  source "${SCRIPT_DIR}/activate_v023_vllm_cann_runtime.sh"
  export PYTHONPATH="${REPO_ROOT}:${DSV4_VLLM_ROOT}:${DSV4_VLLM_ASCEND_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

  local api_port="${PHASE1_NATIVE_API_PORT:-8900}"
  local devices="${PHASE1_NATIVE_DEVICES:-0,1,2,3,4,5,6,7}"
  local ready_timeout="${PHASE1_NATIVE_READY_TIMEOUT_SECONDS:-1800}"
  [[ "${api_port}" =~ ^[0-9]+$ ]] || die "PHASE1_NATIVE_API_PORT must be an integer"
  [[ "${ready_timeout}" =~ ^[1-9][0-9]*$ ]] \
    || die "PHASE1_NATIVE_READY_TIMEOUT_SECONDS must be positive"

  local selected_controls=("$@") control_key candidate found
  if (( ${#selected_controls[@]} == 0 )); then
    selected_controls=("${CONTROL_KEYS[@]}")
  fi
  for control_key in "${selected_controls[@]}"; do
    found=0
    for candidate in "${CONTROL_KEYS[@]}"; do
      [[ "${candidate}" == "${control_key}" ]] && found=1
    done
    (( found == 1 )) || die "Unknown control key: ${control_key}"
    [[ ! -e "${output_root}/${control_key}" ]] \
      || die "Control output already exists: ${output_root}/${control_key}"
  done

  local control_root native_pid ready deadline forced_stop
  native_pid=
  cleanup_native() {
    if [[ -z "${native_pid}" ]]; then
      return
    fi
    kill -TERM -- "-${native_pid}" 2>/dev/null || true
    forced_stop=1
    for _ in $(seq 1 60); do
      if ! kill -0 "${native_pid}" 2>/dev/null; then
        forced_stop=0
        break
      fi
      sleep 1
    done
    if (( forced_stop )); then
      kill -KILL -- "-${native_pid}" 2>/dev/null || true
    fi
    wait "${native_pid}" 2>/dev/null || true
    native_pid=
  }
  trap cleanup_native EXIT
  trap 'cleanup_native; exit 130' TERM INT

  for control_key in "${selected_controls[@]}"; do
    (( "$(npu_process_count)" == 0 )) \
      || die "NPU processes exist before ${control_key}"
    control_arguments "${control_key}"
    control_root="${output_root}/${control_key}"
    mkdir -p "${control_root}"
    npu-smi info >"${control_root}/npu-before.txt"
    {
      printf 'control_key=%s\n' "${control_key}"
      printf 'target_execution_mode=%s\n' "${CONTROL_EXECUTION_MODE}"
      printf 'mtp_draft_execution=%s\n' "${CONTROL_DRAFT_EXECUTION}"
      printf 'enable_mtp=%s\n' "${CONTROL_ENABLE_MTP}"
      printf 'mtp_num_speculative_tokens=%s\n' "${CONTROL_MTP_TOKENS}"
    } >"${control_root}/control.env"

    setsid env \
      API_HOST=127.0.0.1 API_PORT="${api_port}" \
      DATA_PARALLEL_RPC_PORT=$((api_port + 20000)) \
      MASTER_PORT=$((api_port + 20001)) \
      HCCL_IF_BASE_PORT=$((api_port + 44000)) \
      ASCEND_RT_VISIBLE_DEVICES="${devices}" \
      EXECUTION_MODE="${CONTROL_EXECUTION_MODE}" \
      ENABLE_MTP="${CONTROL_ENABLE_MTP}" \
      MTP_NUM_SPECULATIVE_TOKENS="${CONTROL_MTP_TOKENS}" \
      MTP_DRAFT_EXECUTION="${CONTROL_DRAFT_EXECUTION}" \
      TENSOR_PARALLEL_SIZE=1 \
      bash "${NATIVE_LAUNCHER}" \
      >"${control_root}/server.log" 2>&1 &
    native_pid=$!
    printf '%s\n' "${native_pid}" >"${control_root}/server.pid"

    ready=0
    deadline=$((SECONDS + ready_timeout))
    while (( SECONDS < deadline )); do
      if curl -fsS "http://127.0.0.1:${api_port}/health" >/dev/null 2>&1; then
        ready=1
        break
      fi
      if ! kill -0 "${native_pid}" 2>/dev/null; then
        tail -160 "${control_root}/server.log" >&2
        die "Native control exited before ready: ${control_key}"
      fi
      sleep 2
    done
    (( ready == 1 )) || die "Native control readiness timed out: ${control_key}"

    "${DSV4_RUNTIME_VENV}/bin/python" "${GOLDEN_GENERATOR}" \
      --endpoint "http://127.0.0.1:${api_port}/v1/completions" \
      --model dsv4-v023-native \
      --prompt-source "${SCRIPT_DIR}/phase1_prompts.json" \
      --output "${control_root}/golden_results.json" \
      --rounds 3 \
      --metadata baseline_kind=native_path_control \
      --metadata "control_key=${control_key}" \
      --metadata cann_version=9.0.0 \
      --metadata vllm_commit=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665 \
      --metadata vllm_ascend_commit=3da28f9414583d2d0b672a8f06d1fae142404bda \
      --metadata "target_execution_mode=${CONTROL_EXECUTION_MODE}" \
      --metadata "mtp_draft_execution=${CONTROL_DRAFT_EXECUTION}" \
      --metadata "enable_mtp=${CONTROL_ENABLE_MTP}" \
      --metadata "mtp_num_speculative_tokens=${CONTROL_MTP_TOKENS}"

    forced_stop=0
    cleanup_native
    (( forced_stop == 0 )) || die "Native control required SIGKILL: ${control_key}"
    wait_for_empty_npus || die "NPU cleanup failed after ${control_key}"
    npu-smi info >"${control_root}/npu-after-stop.txt"
    if grep -Eiq \
      'Traceback|Communication_Error|507015|EngineCore.*fatal|(^|[[:space:]])ERROR([[:space:]]|$)' \
      "${control_root}/server.log"; then
      die "Fatal marker found in native control log: ${control_key}"
    fi
    printf '[phase1-native] passed: %s\n' "${control_key}"
  done
  trap - EXIT TERM INT
  printf '[phase1-native] completed: %s\n' "${output_root}"
}

case "${ACTION}" in
  help|-h|--help) usage ;;
  list) printf '%s\n' "${CONTROL_KEYS[@]}" ;;
  preflight) bash "${MATRIX_RUNNER}" preflight-native ;;
  run)
    bash "${MATRIX_RUNNER}" preflight-native
    run_controls "$@"
    ;;
  *) usage; die "Unknown action: ${ACTION}" ;;
esac
