#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

bash "${SCRIPT_DIR}/01_preflight.sh" runtime
require_file "${VENV_ROOT}/bin/python"
require_file "${MODEL_PATH}/config.json"
require_file "${AFD_PLUGIN_ROOT}/afd_plugin/connectors/npu/p2p_hccl.py"

ensure_dir "${STATE_ROOT}"
ensure_dir "${LOG_ROOT}"

for role_name in attention ffn; do
  pid_path="${STATE_ROOT}/${role_name}.pid"
  if pid="$(read_pid_file "${pid_path}" 2>/dev/null)" && pid_is_alive "${pid}"; then
    die "${role_name} already appears to be running with PID ${pid}"
  fi
done

for port in \
  "${ATTENTION_API_PORT}" \
  "${FFN_PROCESS_PORT}" \
  "${AFD_PORT}" \
  "${ATTENTION_HCCL_IF_BASE_PORT}" \
  "${FFN_HCCL_IF_BASE_PORT}"; do
  port_is_listening "${port}" && die "Port is already listening: ${port}"
done

npu_output="$(npu-smi info)"
npu_process_count="$(awk '
  /\| NPU +Chip +\| Process id/ {in_process_table=1; next}
  in_process_table && /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+/ {count++}
  END {print count + 0}
' <<<"${npu_output}")"
if (( npu_process_count > 0 )) && ! is_true "${ALLOW_NPU_PROCESSES}"; then
  die "Detected ${npu_process_count} existing NPU processes"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
attention_log="${LOG_ROOT}/attention-${timestamp}.log"
ffn_log="${LOG_ROOT}/ffn-${timestamp}.log"
ln -sfn "$(basename "${attention_log}")" "${LOG_ROOT}/attention.log"
ln -sfn "$(basename "${ffn_log}")" "${LOG_ROOT}/ffn.log"

{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'vllm_commit=%s\n' "${VLLM_COMMIT}"
  printf 'vllm_ascend_commit=%s\n' "${VLLM_ASCEND_COMMIT}"
  printf 'afd_snapshot=%s\n' "${AFD_SNAPSHOT_ID}"
  printf 'cann_root=%s\n' "$(readlink -f "${CANN_ROOT}")"
  printf 'model_path=%s\n' "${MODEL_PATH}"
  printf 'mode=%s/U%s\n' "${EXECUTION_MODE}" "${U_BATCHES}"
  printf 'enable_mtp=%s\n' "${ENABLE_MTP}"
  printf 'hccl_ip=%s\n' "$(resolve_hccl_ip)"
} >"${STATE_ROOT}/last-run.env"

log "Starting FFN; log=${ffn_log}"
nohup setsid bash "${SCRIPT_DIR}/run_role.sh" ffn >"${ffn_log}" 2>&1 &
ffn_pid=$!
printf '%s\n' "${ffn_pid}" >"${STATE_ROOT}/ffn.pid"

sleep 2

log "Starting Attention; log=${attention_log}"
nohup setsid bash "${SCRIPT_DIR}/run_role.sh" attention \
  >"${attention_log}" 2>&1 &
attention_pid=$!
printf '%s\n' "${attention_pid}" >"${STATE_ROOT}/attention.pid"

sleep 3
if ! pid_is_alive "${ffn_pid}" || ! pid_is_alive "${attention_pid}"; then
  warn "A role exited immediately; stopping owned processes"
  bash "${SCRIPT_DIR}/09_stop.sh" || true
  tail -n 80 "${ffn_log}" "${attention_log}" >&2 || true
  die "Service startup failed"
fi

if ! is_true "${WAIT_READY}"; then
  log "Roles started: attention=${attention_pid}, ffn=${ffn_pid}. WAIT_READY=0, not waiting for health."
  exit 0
fi

deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
next_progress=$((SECONDS + 30))
while (( SECONDS < deadline )); do
  if ! pid_is_alive "${ffn_pid}" || ! pid_is_alive "${attention_pid}"; then
    bash "${SCRIPT_DIR}/09_stop.sh" || true
    die "A role exited before readiness; inspect ${LOG_ROOT}"
  fi
  if curl -fsS --max-time 5 \
    "http://${API_HOST}:${ATTENTION_API_PORT}/health" >/dev/null; then
    ffn_ready_count="$(
      { grep -o \
          'AFD FFN EngineCore started; workers run connector loop' \
          "${ffn_log}" 2>/dev/null || true; } | wc -l
    )"
    if (( ffn_ready_count >= FFN_RANKS )); then
      log "Service ready: Attention health OK, FFN loops=${ffn_ready_count}/${FFN_RANKS}"
      exit 0
    fi
  fi
  if (( SECONDS >= next_progress )); then
    log "Still waiting for model load/compile; elapsed=$((STARTUP_TIMEOUT_SECONDS - (deadline - SECONDS)))s"
    next_progress=$((SECONDS + 30))
  fi
  sleep 5
done

die "Readiness timed out after ${STARTUP_TIMEOUT_SECONDS}s; roles remain owned by this bundle"
