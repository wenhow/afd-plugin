#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD_SCRIPT="${SCRIPT_DIR}/pd.sh"
PERFORMANCE_TOOL="${SCRIPT_DIR}/../run_pd_performance.py"
NPU_MONITOR_TOOL="${SCRIPT_DIR}/../run_pd_npu_monitor.py"
PROFILE_SUMMARY_TOOL="${SCRIPT_DIR}/../summarize_pd_profiles.py"
PROFILE_ANALYSE_TOOL="${SCRIPT_DIR}/../analyse_pd_profile.sh"
FATAL_PATTERN='EngineCore encountered a fatal error|AFD NPU FFN worker loop failed|Mooncake transfer failed|Communication_Error|507015|Traceback'
ACTION="${1:-help}"
CONFIG_DIR="${2:-${PD_GRAPH_MATRIX_CONFIG_DIR:-/data/z00569729/config/pd-graph-matrix}}"
POINT="${3:-}"
ROLE="${4:-}"

CORE_POINTS=(
  control_graph_u1
  afd_graph_u1
  afd_graph_u2
  afd_graph_u2_mtp1
)
RATIO_POINTS=(
  afd_graph_u2
  afd_graph_u2_split_a8f8
  afd_graph_u2_split_a16f8
)
POINTS=("${CORE_POINTS[@]}" "${RATIO_POINTS[@]:1}")
COLOCATED_ROLES=(prefill decode proxy)
SPLIT_ROLES=(prefill_ffn attention proxy)

usage() {
  cat <<'EOF'
Usage:
  bash pd_graph_matrix.sh init <config-dir>
  bash pd_graph_matrix.sh list <config-dir>
  bash pd_graph_matrix.sh commands <config-dir> <point>
  bash pd_graph_matrix.sh print-config <config-dir> <point> <role>
  bash pd_graph_matrix.sh check|start|status|smoke|collect|profile-start|profile-check|profile-stop|profile-finalize|stop <config-dir> <point> <role>
  bash pd_graph_matrix.sh collect-final <config-dir> <point> <role>
  bash pd_graph_matrix.sh monitor-start|monitor-stop <config-dir> <point> <NPU-role>
  bash pd_graph_matrix.sh evidence|profile-analyse|profile-summary <config-dir> <AFD-point> <AFD-role>
  bash pd_graph_matrix.sh benchmark <config-dir> <point> <p1|p2|profile>
  bash pd_graph_matrix.sh compare <config-dir> <p1|p2>
  bash pd_graph_matrix.sh compare-ratio <config-dir> <p1|p2>

Points:
  control_graph_u1    PD control, Graph, U1, MTP off, P8+D8 (16 NPUs)
  afd_graph_u1        PD + AFD, Graph, U1, MTP off, P8+A8F8 (24 NPUs)
  afd_graph_u2        PD + AFD, Graph, U2, MTP off, P8+A8F8 (24 NPUs)
  afd_graph_u2_mtp1   PD + AFD, Graph, U2, eager draft MTP x1, P8+A8F8 (24 NPUs)
  afd_graph_u2_split_a8f8   PD + AFD, Graph U2, P8F8+A8 placement control (24 NPUs)
  afd_graph_u2_split_a16f8  PD + AFD, Graph U2, P8F8+A16 target (32 NPUs)

F0 uses the smoke action and never reads golden. benchmark fixes the main
workload to C32/1024/128/128 by default. P1 runs once; P2 runs three times.
EOF
}

die() {
  printf '[pd-graph-matrix] ERROR: %s\n' "$*" >&2
  exit 2
}

log() {
  printf '[pd-graph-matrix] %s\n' "$*"
}

warn() {
  printf '[pd-graph-matrix] WARNING: %s\n' "$*" >&2
}

load_point_spec() {
  local point="$1"
  MATRIX_EXPECT_EXECUTION_MODE=full-decode-only
  MATRIX_EXPECT_MTP_DRAFT_EXECUTION=eager
  MATRIX_EXPECT_MTP_NUM_SPECULATIVE_TOKENS=1
  MATRIX_PREFILL_NPUS=8
  MATRIX_CONTROL_TOTAL_NPUS=16
  MATRIX_RESERVED_NPUS=32
  MATRIX_CONTROL_RESERVED_NPUS=32
  MATRIX_SERVER_COUNT=2
  MATRIX_CONTROL_SERVER_COUNT=2
  MATRIX_PREFILL_SERVER_COUNT=1
  MATRIX_DECODE_SERVER_COUNT=1
  MATRIX_PLACEMENT=colocated
  MATRIX_ATTENTION_RANKS=8
  MATRIX_FFN_RANKS=8
  MATRIX_MAX_NUM_SEQS=16
  MATRIX_ATTENTION_MAX_NUM_BATCHED_TOKENS=4096
  MATRIX_FFN_MAX_NUM_BATCHED_TOKENS=4096
  MATRIX_MAX_CUDAGRAPH_CAPTURE_SIZE=8
  MATRIX_CUDAGRAPH_CAPTURE_SIZES="1 2 4 8"
  case "${point}" in
    control_graph_u1)
      MATRIX_EXPECT_VARIANT=pd_control
      MATRIX_EXPECT_U_BATCHES=1
      MATRIX_EXPECT_MTP=0
      MATRIX_DECODE_NPUS=8
      MATRIX_ATTENTION_NPUS=0
      MATRIX_FFN_NPUS=0
      MATRIX_RESOURCE_LABEL=P8+D8
      ;;
    afd_graph_u1)
      MATRIX_EXPECT_VARIANT=pd_afd
      MATRIX_EXPECT_U_BATCHES=1
      MATRIX_EXPECT_MTP=0
      MATRIX_DECODE_NPUS=16
      MATRIX_ATTENTION_NPUS=8
      MATRIX_FFN_NPUS=8
      MATRIX_RESOURCE_LABEL=P8+A8F8
      ;;
    afd_graph_u2)
      MATRIX_EXPECT_VARIANT=pd_afd
      MATRIX_EXPECT_U_BATCHES=2
      MATRIX_EXPECT_MTP=0
      MATRIX_DECODE_NPUS=16
      MATRIX_ATTENTION_NPUS=8
      MATRIX_FFN_NPUS=8
      MATRIX_RESOURCE_LABEL=P8+A8F8
      ;;
    afd_graph_u2_mtp1)
      MATRIX_EXPECT_VARIANT=pd_afd
      MATRIX_EXPECT_U_BATCHES=2
      MATRIX_EXPECT_MTP=1
      MATRIX_DECODE_NPUS=16
      MATRIX_ATTENTION_NPUS=8
      MATRIX_FFN_NPUS=8
      MATRIX_RESOURCE_LABEL=P8+A8F8
      ;;
    afd_graph_u2_split_a8f8)
      MATRIX_EXPECT_VARIANT=pd_afd
      MATRIX_EXPECT_U_BATCHES=2
      MATRIX_EXPECT_MTP=0
      MATRIX_PLACEMENT=split
      MATRIX_DECODE_NPUS=16
      MATRIX_ATTENTION_NPUS=8
      MATRIX_FFN_NPUS=8
      MATRIX_RESOURCE_LABEL=P8F8+A8
      ;;
    afd_graph_u2_split_a16f8)
      MATRIX_EXPECT_VARIANT=pd_afd
      MATRIX_EXPECT_U_BATCHES=2
      MATRIX_EXPECT_MTP=0
      MATRIX_PLACEMENT=split
      MATRIX_ATTENTION_RANKS=16
      MATRIX_DECODE_NPUS=24
      MATRIX_ATTENTION_NPUS=16
      MATRIX_FFN_NPUS=8
      MATRIX_MAX_NUM_SEQS=32
      MATRIX_FFN_MAX_NUM_BATCHED_TOKENS=8192
      MATRIX_MAX_CUDAGRAPH_CAPTURE_SIZE=16
      MATRIX_CUDAGRAPH_CAPTURE_SIZES="1 2 4 8 16"
      MATRIX_RESOURCE_LABEL=P8F8+A16
      ;;
    *) die "Unknown matrix point: ${point}" ;;
  esac
  MATRIX_TOTAL_NPUS=$((MATRIX_PREFILL_NPUS + MATRIX_DECODE_NPUS))
}

roles_for_point() {
  if [[ "${MATRIX_PLACEMENT}" == "split" ]]; then
    printf '%s\n' "${SPLIT_ROLES[@]}"
  else
    printf '%s\n' "${COLOCATED_ROLES[@]}"
  fi
}

validate_role() {
  case "$1" in
    prefill|decode|prefill_ffn|attention|proxy) ;;
    *) die "Unsupported role: $1" ;;
  esac
}

validate_device_list() {
  local name="$1"
  local value="$2"
  local expected_count="$3"
  local raw device
  local devices=()
  local -A seen=()
  IFS=',' read -r -a devices <<<"${value}"
  (( ${#devices[@]} == expected_count )) \
    || die "${name} must contain exactly ${expected_count} device IDs"
  for raw in "${devices[@]}"; do
    device="${raw//[[:space:]]/}"
    [[ "${device}" =~ ^[0-9]+$ ]] \
      || die "${name} contains an invalid device ID: ${raw}"
    [[ -z "${seen[${device}]+x}" ]] \
      || die "${name} contains duplicate device ID: ${device}"
    seen["${device}"]=1
  done
}

config_path() {
  printf '%s/%s-%s.env\n' "${CONFIG_DIR}" "$1" "$2"
}

write_role_config() {
  local point="$1"
  local role="$2"
  local output_path
  load_point_spec "${point}"
  output_path="$(config_path "${point}" "${role}")"
  cat >"${output_path}" <<EOF
# Generated by pd_graph_matrix.sh. Edit common.env, not this file.
MATRIX_CONFIG_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "\${MATRIX_CONFIG_DIR}/common.env"
unset MATRIX_CONFIG_DIR

MATRIX_POINT="${point}"
NODE_ROLE="${role}"
DEPLOYMENT_VARIANT="${MATRIX_EXPECT_VARIANT}"
DECODE_TP_SIZE="1"
AFD_PLACEMENT="${MATRIX_PLACEMENT}"
ATTENTION_RANKS="${MATRIX_ATTENTION_RANKS}"
FFN_RANKS="${MATRIX_FFN_RANKS}"
DECODE_DP_SIZE="${MATRIX_ATTENTION_RANKS}"
ATTENTION_DEVICES="$([[ "${MATRIX_ATTENTION_RANKS}" == "16" ]] && printf '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15' || printf '0,1,2,3,4,5,6,7')"
FFN_DEVICES="8,9,10,11,12,13,14,15"
MAX_NUM_SEQS="${MATRIX_MAX_NUM_SEQS}"
ATTENTION_MAX_NUM_BATCHED_TOKENS="${MATRIX_ATTENTION_MAX_NUM_BATCHED_TOKENS}"
FFN_MAX_NUM_BATCHED_TOKENS="${MATRIX_FFN_MAX_NUM_BATCHED_TOKENS}"
DECODE_MAX_CUDAGRAPH_CAPTURE_SIZE="${MATRIX_MAX_CUDAGRAPH_CAPTURE_SIZE}"
DECODE_CUDAGRAPH_CAPTURE_SIZES="${MATRIX_CUDAGRAPH_CAPTURE_SIZES}"
DECODE_EXECUTION_MODE="${MATRIX_EXPECT_EXECUTION_MODE}"
DECODE_U_BATCHES="${MATRIX_EXPECT_U_BATCHES}"
DECODE_ENABLE_MTP="${MATRIX_EXPECT_MTP}"
DECODE_MTP_DRAFT_EXECUTION="${MATRIX_EXPECT_MTP_DRAFT_EXECUTION}"
DECODE_MTP_NUM_SPECULATIVE_TOKENS="${MATRIX_EXPECT_MTP_NUM_SPECULATIVE_TOKENS}"
ENABLE_BATCH_INVARIANT="0"
VLLM_ASCEND_WORKTREE_MODE="clean"
AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP="1"
AFD_HCCL_GRAPH_U2_HYBRID_DAG="1"
AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM="1"
AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM="1"
AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER="1"

MATRIX_RUN_BASE="\${MATRIX_RUN_BASE:-/data/z00569729/run/dsv4-mooncake-pd-graph-matrix}"
RUN_ROOT="\${MATRIX_RUN_BASE}/${point}"
# CANN device writers are more restrictive than the Python FRAMEWORK writer.
# Keep raw captures on node-local storage by default; the run/evidence tree may
# remain on shared storage. The MATRIX_RUN_BASE basename isolates retries.
MATRIX_PROFILE_BASE="\${MATRIX_PROFILE_BASE:-/tmp/dsv4-pd-profile/\$(basename "\${MATRIX_RUN_BASE}")}"
STATE_ROOT="\${RUN_ROOT}/state/${role}"
LOG_ROOT="\${RUN_ROOT}/logs/${role}"
VALIDATION_ROOT="\${RUN_ROOT}/validation"
OUTPUT_ROOT="\${RUN_ROOT}/output"
PD_CONTROL_GOLDEN_PATH="\${RUN_ROOT}/deferred-f1/golden_results.json"
COLOCATED_PREFILL_PID_FILE="\${RUN_ROOT}/state/prefill/prefill.pid"
AFD_PROFILE_ATTENTION_DIR="\${MATRIX_PROFILE_BASE}/${point}/attention"
AFD_PROFILE_FFN_DIR="\${MATRIX_PROFILE_BASE}/${point}/ffn"
EOF
}

init_action() {
  [[ ! -e "${CONFIG_DIR}/common.env" ]] \
    || die "Matrix config already exists: ${CONFIG_DIR}/common.env"
  mkdir -p "${CONFIG_DIR}"
  cp "${SCRIPT_DIR}/config.env.example" "${CONFIG_DIR}/common.env"
  local repo_root afd_commit point role
  repo_root="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
  afd_commit="$(git -c safe.directory="${repo_root}" \
    -C "${repo_root}" rev-parse HEAD)" \
    || die "Cannot resolve the current afd-plugin commit: ${repo_root}"
  [[ "${afd_commit}" =~ ^[0-9a-f]{40}$ ]] \
    || die "Current afd-plugin HEAD is not a 40-character commit: ${afd_commit}"
  sed -i \
    "s/^AFD_PD_COMMIT=.*/AFD_PD_COMMIT=\"${afd_commit}\"/" \
    "${CONFIG_DIR}/common.env"
  local generated=0
  for point in "${POINTS[@]}"; do
    load_point_spec "${point}"
    while IFS= read -r role; do
      write_role_config "${point}" "${role}"
      generated=$((generated + 1))
    done < <(roles_for_point)
  done
  log "Created ${CONFIG_DIR}/common.env and ${generated} generated role configs"
  log "Pinned afd-plugin ${afd_commit} in ${CONFIG_DIR}/common.env"
  log "Review common.env: fixed IPs, NIC, model path, and commits must match the site"
}

compare_ratio_action() {
  local phase="${POINT}"
  case "${phase}" in p1|p2) ;; *) die "Compare phase must be p1 or p2" ;; esac
  local point summary output base_config
  local inputs=()
  for point in "${RATIO_POINTS[@]}"; do
    summary="$(last_summary_for_point "${point}" "${phase}")"
    inputs+=(--input "${summary}")
  done
  POINT=afd_graph_u2
  ROLE=proxy
  load_point_spec "${POINT}"
  base_config="$(require_generated_config)"
  source_and_validate_config "${base_config}"
  output="${MATRIX_RUN_BASE:-/data/z00569729/run/dsv4-mooncake-pd-graph-matrix}/comparison-af-ratio-${phase}-$(date +%Y%m%d_%H%M%S).json"
  "${VENV_ROOT}/bin/python" "${PERFORMANCE_TOOL}" compare-ratio \
    "${inputs[@]}" --output "${output}"
  log "AF ratio comparison: ${output}"
}

list_action() {
  printf '%-24s %-10s %-5s %-5s %-5s %-12s\n' \
    POINT VARIANT GRAPH U MTP RESOURCES
  local point
  for point in "${POINTS[@]}"; do
    load_point_spec "${point}"
    printf '%-24s %-10s %-5s %-5s %-5s %-12s\n' \
      "${point}" "${MATRIX_EXPECT_VARIANT}" yes \
      "${MATRIX_EXPECT_U_BATCHES}" "${MATRIX_EXPECT_MTP}" \
      "${MATRIX_RESOURCE_LABEL}"
  done
}

require_generated_config() {
  local path
  load_point_spec "${POINT}"
  validate_role "${ROLE}"
  path="$(config_path "${POINT}" "${ROLE}")"
  [[ -f "${path}" ]] || die "Missing generated config: ${path}; run init first"
  printf '%s\n' "${path}"
}

source_and_validate_config() {
  local config="$1"
  load_point_spec "${POINT}"
  set -a
  # shellcheck disable=SC1090
  source "${config}"
  set +a
  [[ "${MATRIX_POINT:-}" == "${POINT}" ]] || die "MATRIX_POINT mismatch"
  [[ "${DEPLOYMENT_VARIANT}" == "${MATRIX_EXPECT_VARIANT}" ]] \
    || die "DEPLOYMENT_VARIANT mismatch for ${POINT}"
  [[ "${DECODE_DP_SIZE}:${DECODE_TP_SIZE}" == "${MATRIX_ATTENTION_RANKS}:1" ]] \
    || die "Decode DP/TP mismatch for ${POINT}"
  [[ "${PREFILL_DP_SIZE}:${PREFILL_TP_SIZE}" == "2:4" ]] \
    || die "The primary PD Graph matrix is fixed to Prefill DP2/TP4"
  [[ "${PREFILL_IP}" != "${DECODE_IP}" ]] \
    || die "The primary PD Graph matrix requires separate Prefill and Decode hosts"
  [[ "${ALLOW_COLOCATED_PD_CONTROL}" == "0" ]] \
    || die "The primary PD Graph matrix forbids colocated PD control"
  [[ "${AFD_PLACEMENT}" == "${MATRIX_PLACEMENT}" ]] \
    || die "AFD_PLACEMENT mismatch for ${POINT}"
  [[ "${ATTENTION_RANKS}:${FFN_RANKS}" == "${MATRIX_ATTENTION_RANKS}:${MATRIX_FFN_RANKS}" ]] \
    || die "Attention/FFN rank mismatch for ${POINT}"
  [[ "${DECODE_EXECUTION_MODE}" == "${MATRIX_EXPECT_EXECUTION_MODE}" ]] \
    || die "DECODE_EXECUTION_MODE mismatch for ${POINT}"
  [[ "${DECODE_U_BATCHES}" == "${MATRIX_EXPECT_U_BATCHES}" ]] \
    || die "DECODE_U_BATCHES mismatch for ${POINT}"
  [[ "${DECODE_ENABLE_MTP}" == "${MATRIX_EXPECT_MTP}" ]] \
    || die "DECODE_ENABLE_MTP mismatch for ${POINT}"
  [[ "${DECODE_MTP_DRAFT_EXECUTION}" == "eager" ]] \
    || die "The primary matrix fixes the MTP draft to eager"
  [[ "${DECODE_MTP_NUM_SPECULATIVE_TOKENS}" == "1" ]] \
    || die "The primary matrix supports one speculative token"
  [[ "${ATTENTION_RANKS}" =~ ^[0-9]+$ && "${ATTENTION_RANKS}" -gt 0 ]] \
    || die "ATTENTION_RANKS must be a positive integer"
  [[ "${ENABLE_BATCH_INVARIANT}" == "0" ]] \
    || die "Batch invariance is deferred until after functional/performance work"
  validate_device_list PREFILL_DEVICES "${PREFILL_DEVICES}" 8
  validate_device_list ATTENTION_DEVICES "${ATTENTION_DEVICES}" "${MATRIX_ATTENTION_RANKS}"
  validate_device_list FFN_DEVICES "${FFN_DEVICES}" "${MATRIX_FFN_RANKS}"
  if [[ "${DEPLOYMENT_VARIANT}" == "pd_afd" && "${MATRIX_PLACEMENT}" == "colocated" ]]; then
    validate_device_list DECODE_AFD_DEVICES \
      "${ATTENTION_DEVICES},${FFN_DEVICES}" 16
  fi
}

monitor_devices() {
  if [[ "${ROLE}" == "prefill" ]]; then
    printf '%s\n' "${PREFILL_DEVICES}"
  elif [[ "${ROLE}" == "prefill_ffn" ]]; then
    printf '%s,%s\n' "${PREFILL_DEVICES}" "${FFN_DEVICES}"
  elif [[ "${ROLE}" == "attention" ]]; then
    printf '%s\n' "${ATTENTION_DEVICES}"
  elif [[ "${DEPLOYMENT_VARIANT}" == "pd_control" ]]; then
    printf '%s\n' "${ATTENTION_DEVICES}"
  else
    printf '%s,%s\n' "${ATTENTION_DEVICES}" "${FFN_DEVICES}"
  fi
}

monitor_pid_is_owned() {
  local pid="$1"
  local run_dir="$2"
  local cmdline
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${cmdline}" == *"${NPU_MONITOR_TOOL}"* && "${cmdline}" == *"${run_dir}"* ]]
}

pid_is_zombie() {
  [[ "$(awk '$1 == "State:" {print $2}' "/proc/$1/status" 2>/dev/null || true)" == "Z" ]]
}

monitor_start_action() {
  [[ "${ROLE}" == "prefill" || "${ROLE}" == "decode" \
    || "${ROLE}" == "prefill_ffn" || "${ROLE}" == "attention" ]] \
    || die "monitor-start is only valid on an NPU role"
  local config pid_path last_dir_path run_dir log_path pid devices
  config="$(require_generated_config)"
  source_and_validate_config "${config}"
  [[ -f "${NPU_MONITOR_TOOL}" ]] || die "Missing NPU monitor: ${NPU_MONITOR_TOOL}"
  pid_path="${STATE_ROOT}/npu-monitor.pid"
  last_dir_path="${STATE_ROOT}/last-monitor-dir"
  if [[ -f "${pid_path}" ]]; then
    read -r pid <"${pid_path}"
    [[ "${pid}" =~ ^[0-9]+$ ]] || die "Invalid NPU monitor PID: ${pid}"
    if kill -0 "${pid}" 2>/dev/null && ! pid_is_zombie "${pid}"; then
      [[ -f "${last_dir_path}" ]] || die "Live monitor PID has no output pointer: ${pid}"
      read -r run_dir <"${last_dir_path}"
      monitor_pid_is_owned "${pid}" "${run_dir}" \
        || die "Refusing stale/reused NPU monitor PID: ${pid}"
      die "NPU monitor is already running with PID ${pid}"
    fi
  fi
  run_dir="${RUN_ROOT}/monitor/${ROLE}-$(date +%Y%m%d_%H%M%S)"
  log_path="${run_dir}.log"
  devices="$(monitor_devices)"
  mkdir -p "${STATE_ROOT}" "${run_dir}"
  "${VENV_ROOT}/bin/python" "${NPU_MONITOR_TOOL}" preflight \
    --devices "${devices}" >"${run_dir}/preflight.json" \
    || die "NPU monitor preflight failed; inspect ${run_dir}/preflight.json"
  nohup setsid "${VENV_ROOT}/bin/python" "${NPU_MONITOR_TOOL}" run \
    --output-dir "${run_dir}" \
    --devices "${devices}" \
    --interval "${NPU_MONITOR_INTERVAL_SECONDS:-5}" \
    --max-samples "${NPU_MONITOR_MAX_SAMPLES:-3600}" \
    >"${log_path}" 2>&1 &
  pid=$!
  printf '%s\n' "${pid}" >"${pid_path}"
  printf '%s\n' "${run_dir}" >"${last_dir_path}"
  sleep 1
  kill -0 "${pid}" 2>/dev/null \
    || die "NPU monitor exited immediately; inspect ${log_path}"
  log "NPU monitor started: role=${ROLE} pid=${pid} devices=${devices} output=${run_dir}"
}

monitor_stop_action() {
  [[ "${ROLE}" == "prefill" || "${ROLE}" == "decode" \
    || "${ROLE}" == "prefill_ffn" || "${ROLE}" == "attention" ]] \
    || die "monitor-stop is only valid on an NPU role"
  local config pid_path last_dir_path pid run_dir deadline
  config="$(require_generated_config)"
  source_and_validate_config "${config}"
  pid_path="${STATE_ROOT}/npu-monitor.pid"
  last_dir_path="${STATE_ROOT}/last-monitor-dir"
  [[ -f "${pid_path}" ]] || die "No NPU monitor PID recorded for ${ROLE}"
  [[ -f "${last_dir_path}" ]] || die "Missing NPU monitor output pointer"
  read -r pid <"${pid_path}"
  read -r run_dir <"${last_dir_path}"
  [[ "${pid}" =~ ^[0-9]+$ ]] || die "Invalid NPU monitor PID: ${pid}"
  case "${run_dir}" in
    "${RUN_ROOT}"/monitor/*) ;;
    *) die "NPU monitor output is outside RUN_ROOT: ${run_dir}" ;;
  esac
  if kill -0 "${pid}" 2>/dev/null && ! pid_is_zombie "${pid}"; then
    monitor_pid_is_owned "${pid}" "${run_dir}" \
      || die "Refusing to signal stale/reused NPU monitor PID: ${pid}"
    kill -TERM "${pid}"
    deadline=$((SECONDS + 30))
    while kill -0 "${pid}" 2>/dev/null && (( SECONDS < deadline )); do
      sleep 1
    done
    ! kill -0 "${pid}" 2>/dev/null \
      || die "NPU monitor did not stop after SIGTERM: ${pid}"
  fi
  rm -f "${pid_path}"
  [[ -s "${run_dir}/summary.json" ]] \
    || die "Missing NPU monitor summary: ${run_dir}/summary.json"
  "${VENV_ROOT}/bin/python" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("passed") else 1)' \
    "${run_dir}/summary.json" || die "NPU monitor did not collect a valid sample"
  log "NPU monitor stopped: ${run_dir}/summary.json"
}

role_pid_names() {
  case "${ROLE}" in
    prefill) printf '%s\n' prefill ;;
    prefill_ffn) printf '%s\n' prefill ffn ;;
    attention) printf '%s\n' attention ;;
    proxy) printf '%s\n' proxy ;;
    decode)
      if [[ "${DEPLOYMENT_VARIANT}" == "pd_control" ]]; then
        printf '%s\n' decode-control
      else
        printf '%s\n' attention ffn
      fi
      ;;
  esac
}

current_role_has_fatal() {
  local name log_path
  while read -r name; do
    log_path="${LOG_ROOT}/${name}.log"
    [[ -f "${log_path}" ]] || continue
    grep -Eq "${FATAL_PATTERN}" "${log_path}" && return 0
  done < <(role_pid_names)
  return 1
}

evidence_action() {
  [[ "${ROLE}" == "decode" || "${ROLE}" == "attention" ]] \
    || die "evidence must run on decode or split Attention"
  local config attention_log expected count live_lines ranks rank_count
  config="$(require_generated_config)"
  source_and_validate_config "${config}"
  [[ "${DEPLOYMENT_VARIANT}" == "pd_afd" ]] \
    || die "stage evidence is only defined for AFD Decode"
  attention_log="${LOG_ROOT}/attention.log"
  [[ -f "${attention_log}" ]] || die "Missing Attention log: ${attention_log}"
  expected="${DECODE_U_BATCHES}"
  live_lines="$(grep -E \
    "AFD NPU Attention send_dp_metadata decision;.*stage_count=${expected} .*is_graph_capturing=False is_warmup=False" \
    "${attention_log}" || true)"
  count="$(printf '%s\n' "${live_lines}" | grep -c . || true)"
  ranks="$(printf '%s\n' "${live_lines}" \
    | sed -nE 's/.*world_rank=([0-9]+).*/\1/p' \
    | sort -n -u | paste -sd, -)"
  rank_count="$(printf '%s\n' "${ranks}" | tr ',' '\n' | grep -c . || true)"
  (( count > 0 && rank_count == ATTENTION_RANKS )) \
    || die "Online stage_count=${expected} reached ${rank_count}/${ATTENTION_RANKS} Attention ranks; run evidence after the concurrent benchmark/profile workload, not immediately after F0 smoke"
  ! current_role_has_fatal \
    || die "Fatal marker found while collecting stage evidence"
  mkdir -p "${STATE_ROOT}"
  {
    printf 'passed=1\n'
    printf 'matrix_point=%s\n' "${POINT}"
    printf 'expected_stage_count=%s\n' "${expected}"
    printf 'matching_log_records=%s\n' "${count}"
    printf 'expected_attention_ranks=%s\n' "${ATTENTION_RANKS}"
    printf 'observed_attention_ranks=%s\n' "${rank_count}"
    printf 'observed_world_ranks=%s\n' "${ranks}"
    printf 'attention_log=%s\n' "${attention_log}"
    printf 'collected_at=%s\n' "$(date --iso-8601=seconds)"
  } >"${STATE_ROOT}/stage-evidence.env"
  log "Stage evidence passed: stage_count=${expected}, records=${count}"
}

profile_summary_action() {
  [[ "${ROLE}" == "decode" || "${ROLE}" == "attention" \
    || "${ROLE}" == "prefill_ffn" ]] \
    || die "profile-summary must run on an AFD role node"
  local config session started_epoch session_cann
  local attention_log ffn_log attention_markers ffn_markers output
  config="$(require_generated_config)"
  source_and_validate_config "${config}"
  require_role_stopped
  [[ "${DEPLOYMENT_VARIANT}" == "pd_afd" && "${AFD_PROFILE_ENABLE:-0}" == "1" ]] \
    || die "profile-summary requires a pd_afd point with AFD_PROFILE_ENABLE=1"
  [[ -f "${PROFILE_SUMMARY_TOOL}" ]] \
    || die "Missing profile summarizer: ${PROFILE_SUMMARY_TOOL}"
  session="${STATE_ROOT}/profile-session.env"
  [[ -s "${session}" ]] || die "Missing profile session: ${session}"
  started_epoch="$(awk -F= '$1 == "started_epoch" {print $2}' "${session}")"
  session_cann="$(awk -F= '$1 == "cann_version" {print $2}' "${session}")"
  [[ "${started_epoch}" =~ ^[0-9]+$ ]] \
    || die "Invalid profile started_epoch in ${session}"
  [[ "${session_cann}" == "${CANN_VERSION}" ]] \
    || die "Profile session CANN version does not match current config"
  attention_log="${LOG_ROOT}/attention.log"
  ffn_log="${LOG_ROOT}/ffn.log"
  attention_markers=0
  ffn_markers=0
  if [[ -f "${attention_log}" ]]; then
    attention_markers="$(grep -c \
      'AFD NPU attention profiler started manually.*with_stack=False.*online_analysis=False' \
      "${attention_log}" || true)"
  fi
  if [[ -f "${ffn_log}" ]]; then
    ffn_markers="$(grep -c 'AFD NPU ffn profiler started manually.*with_stack=False.*online_analysis=False' \
      "${ffn_log}" || true)"
  fi
  output="${STATE_ROOT}/profile-summary.json"
  local role_args=()
  case "${ROLE}" in
    decode)
      if (( attention_markers == 0 || ffn_markers == 0 )); then
        warn "Profiler enable log markers are incomplete: attention=${attention_markers} (${attention_log}), ffn=${ffn_markers} (${ffn_log}); continuing with authoritative current-session artifact validation"
      fi
      role_args=(--attention-dir "${AFD_PROFILE_ATTENTION_DIR}" \
        --ffn-dir "${AFD_PROFILE_FFN_DIR}")
      ;;
    attention)
      if (( attention_markers == 0 )); then
        warn "Attention profiler enable log marker is missing: ${attention_log}; continuing with authoritative current-session artifact validation"
      fi
      role_args=(--role attention --role-dir "${AFD_PROFILE_ATTENTION_DIR}")
      ;;
    prefill_ffn)
      if (( ffn_markers == 0 )); then
        warn "FFN profiler enable log marker is missing: ${ffn_log}; continuing with authoritative current-session artifact validation"
      fi
      role_args=(--role ffn --role-dir "${AFD_PROFILE_FFN_DIR}")
      ;;
  esac
  "${VENV_ROOT}/bin/python" "${PROFILE_SUMMARY_TOOL}" "${role_args[@]}" \
      --started-epoch "${started_epoch}" \
      --expected-cann-version "${CANN_VERSION}" \
      --expect-manual-window \
      --artifact-timeout-seconds "${AFD_PROFILE_ARTIFACT_TIMEOUT_SECONDS:-300}" \
      --output "${output}"
  log "Profiler parse outputs validated: ${output}"
}

profile_analyse_action() {
  [[ "${ROLE}" == "decode" || "${ROLE}" == "attention" \
    || "${ROLE}" == "prefill_ffn" ]] \
    || die "profile-analyse must run on an AFD role node"
  local config session started_epoch root role_name role_dir
  local -a roots=()
  local -a role_specs=()
  config="$(require_generated_config)"
  source_and_validate_config "${config}"
  require_role_stopped
  [[ "${DEPLOYMENT_VARIANT}" == "pd_afd" && "${AFD_PROFILE_ENABLE:-0}" == "1" ]] \
    || die "profile-analyse requires a pd_afd point with AFD_PROFILE_ENABLE=1"
  [[ -f "${PROFILE_ANALYSE_TOOL}" ]] \
    || die "Missing profile analyzer: ${PROFILE_ANALYSE_TOOL}"
  session="${STATE_ROOT}/profile-session.env"
  [[ -s "${session}" ]] || die "Missing profile session: ${session}"
  started_epoch="$(awk -F= '$1 == "started_epoch" {print $2}' "${session}")"
  [[ "${started_epoch}" =~ ^[0-9]+$ ]] \
    || die "Invalid profile started_epoch in ${session}"
  case "${ROLE}" in
    decode)
      role_specs=("attention:${AFD_PROFILE_ATTENTION_DIR}" \
        "ffn:${AFD_PROFILE_FFN_DIR}")
      ;;
    attention) role_specs=("attention:${AFD_PROFILE_ATTENTION_DIR}") ;;
    prefill_ffn) role_specs=("ffn:${AFD_PROFILE_FFN_DIR}") ;;
  esac
  local spec
  for spec in "${role_specs[@]}"; do
    role_name="${spec%%:*}"
    role_dir="${spec#*:}"
    roots=()
    while IFS= read -r root; do
      roots+=("$(dirname "${root}")")
    done < <(find "${role_dir}" -type f -name 'profiler_info_*.json' \
      -newermt "@${started_epoch}" -print 2>/dev/null | sort)
    (( ${#roots[@]} == 1 )) \
      || die "${role_name} must contain exactly one current profiler root; found ${#roots[@]} under ${role_dir}"
    log "Analysing ${role_name} profile: ${roots[0]}"
    bash "${PROFILE_ANALYSE_TOOL}" "${config}" "${roots[0]}"
  done
  profile_summary_action
}

require_role_stopped() {
  local name pid_path pid
  while read -r name; do
    pid_path="${STATE_ROOT}/${name}.pid"
    [[ -f "${pid_path}" ]] || continue
    read -r pid <"${pid_path}"
    if [[ "${pid}" =~ ^[0-9]+$ ]] \
      && kill -0 "${pid}" 2>/dev/null \
      && ! pid_is_zombie "${pid}"; then
      die "${name} is still running with PID ${pid}; stop all roles before collect-final"
    fi
  done < <(role_pid_names)
  if [[ -f "${STATE_ROOT}/npu-monitor.pid" ]]; then
    read -r pid <"${STATE_ROOT}/npu-monitor.pid"
    if [[ "${pid}" =~ ^[0-9]+$ ]] \
      && kill -0 "${pid}" 2>/dev/null \
      && ! pid_is_zombie "${pid}"; then
      die "NPU monitor is still running with PID ${pid}"
    fi
  fi
}

json_field_is_true() {
  local path="$1"
  local field="$2"
  [[ -s "${path}" ]] || return 1
  "${VENV_ROOT}/bin/python" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get(sys.argv[2]) is True else 1)' \
    "${path}" "${field}" 2>/dev/null
}

collect_final_action() {
  local config monitor_dir performance_dir gate_status failure
  local -a gate_failures=()
  config="$(require_generated_config)"
  source_and_validate_config "${config}"
  require_role_stopped
  if current_role_has_fatal; then
    warn "Fatal markers found in ${ROLE} logs; preserving them in the collection artifact"
    gate_failures+=("fatal markers found in ${ROLE} logs")
  fi
  if [[ "${AFD_PROFILE_ENABLE}" == "0" ]]; then
    case "${ROLE}" in
      prefill|decode|prefill_ffn|attention)
        if [[ ! -f "${STATE_ROOT}/last-monitor-dir" ]]; then
          gate_failures+=("missing NPU monitor pointer for ${ROLE}")
        else
          read -r monitor_dir <"${STATE_ROOT}/last-monitor-dir"
          case "${monitor_dir}" in
            "${RUN_ROOT}"/monitor/${ROLE}-*)
              json_field_is_true "${monitor_dir}/summary.json" passed \
                || gate_failures+=("NPU monitor summary is missing or not passed: ${monitor_dir}/summary.json")
              ;;
            *) gate_failures+=("invalid NPU monitor output for ${ROLE}: ${monitor_dir}") ;;
          esac
        fi
        ;;
    esac
  else
    case "${ROLE}" in
      decode|prefill_ffn|attention) ;;
      *) die "Profile collect-final is only valid on an AFD role node" ;;
    esac
  fi
  if [[ ("${ROLE}" == "decode" || "${ROLE}" == "attention") \
    && "${DEPLOYMENT_VARIANT}" == "pd_afd" ]]; then
    if [[ ! -s "${STATE_ROOT}/stage-evidence.env" ]]; then
      gate_failures+=("missing AFD stage evidence")
    else
      grep -qx 'passed=1' "${STATE_ROOT}/stage-evidence.env" \
        || gate_failures+=("AFD stage evidence is not passed")
      grep -qx "expected_stage_count=${DECODE_U_BATCHES}" \
        "${STATE_ROOT}/stage-evidence.env" \
        || gate_failures+=("AFD stage evidence does not match U${DECODE_U_BATCHES}")
      grep -qx "observed_attention_ranks=${ATTENTION_RANKS}" \
        "${STATE_ROOT}/stage-evidence.env" \
        || gate_failures+=("AFD stage evidence does not cover all Attention ranks")
    fi
  fi
  if [[ "${AFD_PROFILE_ENABLE}" == "1" \
    && "${DEPLOYMENT_VARIANT}" == "pd_afd" \
    && ("${ROLE}" == "decode" || "${ROLE}" == "attention" \
      || "${ROLE}" == "prefill_ffn") ]]; then
    json_field_is_true "${STATE_ROOT}/profile-summary.json" passed \
      || gate_failures+=("AFD profile summary is missing or not passed: ${STATE_ROOT}/profile-summary.json")
  fi
  if [[ "${AFD_PROFILE_ENABLE}" == "0" && "${ROLE}" == "proxy" ]]; then
    if [[ ! -f "${STATE_ROOT}/last-performance-attempt-dir" ]]; then
      gate_failures+=("missing P2 performance attempt pointer")
    else
      read -r performance_dir <"${STATE_ROOT}/last-performance-attempt-dir"
      case "${performance_dir}" in
        "${VALIDATION_ROOT}"/performance-p2-*)
          if ! "${VENV_ROOT}/bin/python" -c \
            'import json,sys; d=json.load(open(sys.argv[1])); ok=d.get("phase")=="p2" and d.get("measurement_passed") is True; raise SystemExit(0 if ok else 1)' \
            "${performance_dir}/performance_summary.json" 2>/dev/null; then
            gate_failures+=("latest P2 measurement is missing or not passed: ${performance_dir}/performance_summary.json")
          fi
          ;;
        *) gate_failures+=("latest performance attempt is not P2: ${performance_dir}") ;;
      esac
    fi
  fi
  gate_status=passed
  if (( ${#gate_failures[@]} > 0 )); then
    gate_status=failed
    for failure in "${gate_failures[@]}"; do
      warn "Acceptance gate failed: ${failure}; collecting artifact anyway"
    done
  fi
  mkdir -p "${STATE_ROOT}"
  {
    printf 'status=%s\n' "${gate_status}"
    printf 'collected_at=%s\n' "$(date --iso-8601=seconds)"
    for failure in "${gate_failures[@]}"; do
      printf 'failure=%s\n' "${failure}"
    done
  } >"${STATE_ROOT}/collect-final-gates.txt"
  exec bash "${PD_SCRIPT}" collect "${config}"
}

commands_action() {
  load_point_spec "${POINT}"
  local role config
  printf '# Point: %s; run in this order on the labelled node.\n' "${POINT}"
  while IFS= read -r role; do
    config="$(config_path "${POINT}" "${role}")"
    printf '\n# %s node\n' "${role}"
    printf 'bash %q check %q\n' "${PD_SCRIPT}" "${config}"
    printf 'bash %q start %q\n' "${PD_SCRIPT}" "${config}"
  done < <(roles_for_point)
  config="$(config_path "${POINT}" proxy)"
  printf '\n# Proxy node: F0 only; no golden and no batch invariance.\n'
  printf 'bash %q smoke %q\n' "${PD_SCRIPT}" "${config}"
  printf '# Later P1/P2: bash %q benchmark %q %q p1\n' "$0" "${CONFIG_DIR}" "${POINT}"
  if [[ "${MATRIX_PLACEMENT}" == "split" ]]; then
    printf '# Stop in reverse order: proxy, attention, prefill_ffn.\n'
  else
    printf '# Stop in reverse order: proxy, decode, prefill.\n'
  fi
}

delegate_action() {
  local config
  config="$(require_generated_config)"
  source_and_validate_config "${config}"
  case "${ACTION}:${ROLE}" in
    smoke:proxy|check:*|start:*|status:*|collect:*|profile-start:*|profile-check:*|profile-stop:*|profile-finalize:*|stop:*|print-config:*) ;;
    smoke:*) die "smoke must run with role=proxy" ;;
    *) die "Unsupported delegated action: ${ACTION}" ;;
  esac
  exec bash "${PD_SCRIPT}" "${ACTION}" "${config}"
}

benchmark_action() {
  local phase="${ROLE}"
  ROLE=proxy
  local config run_dir repeats activation tool_phase record_performance
  config="$(require_generated_config)"
  source_and_validate_config "${config}"
  case "${phase}" in
    p1)
      repeats=1
      tool_phase=p1
      record_performance=1
      [[ "${AFD_PROFILE_ENABLE:-0}" == "0" ]] \
        || die "P1 requires AFD_PROFILE_ENABLE=0"
      ;;
    p2)
      repeats=3
      tool_phase=p2
      record_performance=1
      [[ "${AFD_PROFILE_ENABLE:-0}" == "0" ]] \
        || die "P2 requires AFD_PROFILE_ENABLE=0"
      ;;
    profile)
      repeats=1
      tool_phase=p1
      record_performance=0
      [[ "${DEPLOYMENT_VARIANT}" == "pd_afd" && "${AFD_PROFILE_ENABLE:-0}" == "1" ]] \
        || die "profile workload requires pd_afd and AFD_PROFILE_ENABLE=1"
      ;;
    *) die "Benchmark phase must be p1, p2, or profile: ${phase}" ;;
  esac
  bash "${PD_SCRIPT}" status "${config}"
  run_dir="${VALIDATION_ROOT}/performance-${phase}-$(date +%Y%m%d_%H%M%S)"
  activation="${AFD_PLUGIN_ROOT}/tools/dsv4/activate_runtime.sh"
  [[ -f "${activation}" ]] || die "Missing runtime activation: ${activation}"
  export DSV4_CANN_ROOT="${CANN_ROOT}" DSV4_CANN_VERSION="${CANN_VERSION}"
  export DSV4_ATB_ROOT="${ATB_ROOT:-}" DSV4_RUNTIME_VENV="${VENV_ROOT}"
  export DSV4_VLLM_VENV="${VENV_ROOT}"
  export DSV4_VLLM_ROOT="${VLLM_ROOT}" DSV4_VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT}"
  # shellcheck disable=SC1090
  source "${activation}"
  mkdir -p "${STATE_ROOT}"
  if (( record_performance )); then
    printf '%s\n' "${run_dir}" >"${STATE_ROOT}/last-performance-attempt-dir"
    printf '%s\n' "${run_dir}" >"${STATE_ROOT}/last-validation-dir"
  else
    printf '%s\n' "${run_dir}" >"${STATE_ROOT}/last-profile-dir"
  fi
  "${VENV_ROOT}/bin/python" "${PERFORMANCE_TOOL}" run \
    --output-dir "${run_dir}" --point "${POINT}" --phase "${tool_phase}" \
    --python-bin "${VENV_ROOT}/bin/python" --host 127.0.0.1 --port "${PROXY_PORT}" \
    --metrics-url "http://${DECODE_IP}:${DECODE_API_PORT}/metrics" \
    --served-model "${MODEL_NAME}" --tokenizer "${MODEL_PATH}" \
    --execution-mode "${DECODE_EXECUTION_MODE}" --u-batches "${DECODE_U_BATCHES}" \
    --mtp "${DECODE_ENABLE_MTP}" \
    --mtp-num-speculative-tokens "${DECODE_MTP_NUM_SPECULATIVE_TOKENS}" \
    --repeats "${repeats}" \
    --input-len "${PERFORMANCE_INPUT_LEN:-1024}" \
    --output-len "${PERFORMANCE_OUTPUT_LEN:-128}" \
    --num-prompts "${PERFORMANCE_NUM_PROMPTS:-128}" \
    --concurrency "${PERFORMANCE_CONCURRENCY:-32}" \
    --warmup-input-len "${PERFORMANCE_WARMUP_INPUT_LEN:-256}" \
    --warmup-output-len "${PERFORMANCE_WARMUP_OUTPUT_LEN:-16}" \
    --warmup-prompts "${PERFORMANCE_WARMUP_PROMPTS:-16}" \
    --warmup-concurrency "${PERFORMANCE_WARMUP_CONCURRENCY:-8}" \
    --benchmark-timeout "${PERFORMANCE_TIMEOUT_SECONDS:-3600}" \
    --prefill-npus "${MATRIX_PREFILL_NPUS}" --decode-npus "${MATRIX_DECODE_NPUS}" \
    --attention-npus "${MATRIX_ATTENTION_NPUS}" --ffn-npus "${MATRIX_FFN_NPUS}" \
    --total-npus "${MATRIX_TOTAL_NPUS}" \
    --control-total-npus "${MATRIX_CONTROL_TOTAL_NPUS}" \
    --reserved-npus "${MATRIX_RESERVED_NPUS}" \
    --control-reserved-npus "${MATRIX_CONTROL_RESERVED_NPUS}" \
    --server-count "${MATRIX_SERVER_COUNT}" \
    --control-server-count "${MATRIX_CONTROL_SERVER_COUNT}" \
    --prefill-server-count "${MATRIX_PREFILL_SERVER_COUNT}" \
    --decode-server-count "${MATRIX_DECODE_SERVER_COUNT}" \
    --cann-version "${CANN_VERSION}" --vllm-commit "${VLLM_COMMIT}" \
    --vllm-ascend-commit "${VLLM_ASCEND_COMMIT}" --afd-commit "${AFD_PD_COMMIT}"
  if (( record_performance )); then
    printf '%s\n' "${run_dir}" >"${STATE_ROOT}/last-performance-dir"
  fi
  log "${POINT} ${phase} complete: ${run_dir}/performance_summary.json"
}

last_summary_for_point() (
  local point="$1"
  local phase="$2"
  POINT="${point}"
  ROLE=proxy
  load_point_spec "${point}"
  local config last_dir summary
  config="$(require_generated_config)"
  source_and_validate_config "${config}"
  [[ -f "${STATE_ROOT}/last-performance-attempt-dir" ]] \
    || die "No performance attempt recorded for ${point}"
  read -r last_dir <"${STATE_ROOT}/last-performance-attempt-dir"
  summary="${last_dir}/performance_summary.json"
  [[ -f "${summary}" ]] || die "Missing performance summary for ${point}: ${summary}"
  "${VENV_ROOT}/bin/python" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); ok=d.get("phase")==sys.argv[2] and d.get("measurement_passed", d.get("passed")) is True; raise SystemExit(0 if ok else 1)' \
    "${summary}" "${phase}" \
    || die "Latest ${point} attempt is failed or is not phase ${phase}: ${summary}"
  printf '%s\n' "${summary}"
)

compare_action() {
  local phase="${POINT}"
  case "${phase}" in p1|p2) ;; *) die "Compare phase must be p1 or p2" ;; esac
  local point summary output base_config
  local inputs=()
  for point in "${CORE_POINTS[@]}"; do
    summary="$(last_summary_for_point "${point}" "${phase}")"
    inputs+=(--input "${summary}")
  done
  POINT=control_graph_u1
  ROLE=proxy
  load_point_spec "${POINT}"
  base_config="$(require_generated_config)"
  source_and_validate_config "${base_config}"
  output="${MATRIX_RUN_BASE:-/data/z00569729/run/dsv4-mooncake-pd-graph-matrix}/comparison-${phase}-$(date +%Y%m%d_%H%M%S).json"
  "${VENV_ROOT}/bin/python" "${PERFORMANCE_TOOL}" compare \
    "${inputs[@]}" --output "${output}"
  log "Matrix comparison: ${output}"
}

case "${ACTION}" in
  help|-h|--help) usage ;;
  init) init_action ;;
  list) list_action ;;
  commands)
    [[ -n "${POINT}" ]] || die "commands requires a point"
    commands_action
    ;;
  print-config|check|start|status|smoke|collect|profile-start|profile-check|profile-stop|profile-finalize|stop)
    [[ -n "${POINT}" && -n "${ROLE}" ]] \
      || die "${ACTION} requires a point and role"
    delegate_action
    ;;
  monitor-start|monitor-stop|evidence|profile-analyse|profile-summary)
    [[ -n "${POINT}" && -n "${ROLE}" ]] \
      || die "${ACTION} requires a point and role"
    case "${ACTION}" in
      monitor-start) monitor_start_action ;;
      monitor-stop) monitor_stop_action ;;
      evidence) evidence_action ;;
      profile-analyse) profile_analyse_action ;;
      profile-summary) profile_summary_action ;;
    esac
    ;;
  collect-final)
    [[ -n "${POINT}" && -n "${ROLE}" ]] \
      || die "collect-final requires a point and role"
    collect_final_action
    ;;
  benchmark)
    [[ -n "${POINT}" && -n "${ROLE}" ]] \
      || die "benchmark requires a point and p1|p2|profile"
    benchmark_action
    ;;
  compare)
    [[ -n "${POINT}" ]] || die "compare requires p1|p2"
    compare_action
    ;;
  compare-ratio)
    [[ -n "${POINT}" ]] || die "compare-ratio requires p1|p2"
    compare_ratio_action
    ;;
  *) usage; die "Unknown action: ${ACTION}" ;;
esac
