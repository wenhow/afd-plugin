#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-help}"
CONFIG_FILE="${2:-${PD_CONFIG_FILE:-${SCRIPT_DIR}/config.env}}"

usage() {
  cat <<'EOF'
Usage: bash pd.sh <action> [config.env]

Actions:
  init          Copy config.env.example to the requested config path.
  print-config  Print the non-secret effective role/topology configuration.
  install       Install M9 afd-plugin and optionally the delivered Mooncake wheel.
  check         Validate versions, runtime, network, NPUs, and local round-trip.
  start         Start the configured prefill, decode, split A/F, or proxy role.
  status        Check owned processes, readiness, and fatal log markers.
  smoke         Run F0 batch/cancellation/recovery checks without golden.
  record-control  Record a stable PD no-AFD control golden from the proxy.
  validate      Compare PD + AFD against the path-matched control golden.
  profile-start  Explicitly start Attention/FFN profilers while serving.
  profile-check  Verify that this node has entered raw CANN recording.
  profile-stop   Stop profilers and verify raw CANN capture while serving.
  profile-finalize  Backward-compatible alias for profile-stop.
  stop          Stop only process groups owned by this config.
  collect       Produce a redacted, size-capped support artifact.
EOF
}

log() {
  printf '[mooncake-pd:%s:%s] %s\n' \
    "${DEPLOYMENT_VARIANT:-unconfigured}" "${NODE_ROLE:-unconfigured}" "$*"
}

warn() {
  printf '[mooncake-pd:%s:%s] WARNING: %s\n' \
    "${DEPLOYMENT_VARIANT:-unconfigured}" "${NODE_ROLE:-unconfigured}" "$*" >&2
}

die() {
  printf '[mooncake-pd:%s:%s] ERROR: %s\n' \
    "${DEPLOYMENT_VARIANT:-unconfigured}" "${NODE_ROLE:-unconfigured}" "$*" >&2
  exit 2
}

if [[ "${ACTION}" == "help" || "${ACTION}" == "-h" || "${ACTION}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${ACTION}" == "init" ]]; then
  [[ ! -e "${CONFIG_FILE}" ]] || die "Config already exists: ${CONFIG_FILE}"
  mkdir -p "$(dirname "${CONFIG_FILE}")"
  cp "${SCRIPT_DIR}/config.env.example" "${CONFIG_FILE}"
  printf 'Created %s\n' "${CONFIG_FILE}"
  exit 0
fi

[[ -f "${CONFIG_FILE}" ]] || die "Missing config: ${CONFIG_FILE}; run: bash $0 init ${CONFIG_FILE}"
set -a
# shellcheck disable=SC1090
source "${CONFIG_FILE}"
set +a

: "${CODE_ROOT:=/data/z00569729/code}"
: "${VENV_ROOT:=${CODE_ROOT}/.venvs/afd-v023-vllm-cann}"
: "${VLLM_ROOT:=${CODE_ROOT}/vllm-release-v0.23.0}"
: "${VLLM_ASCEND_ROOT:=${CODE_ROOT}/vllm-ascend-rfc-vllm-cann}"
: "${AFD_PLUGIN_ROOT:=${CODE_ROOT}/afd-plugin}"
: "${CANN_ROOT:=/usr/local/Ascend/cann-9.0.0}"
: "${CANN_VERSION:=9.0.0}"
: "${ATB_ROOT:=}"
: "${MODEL_PATH:=/data/z00569729/models/DeepSeek-V4-Flash-w8a8-mtp}"
: "${DEPLOYMENT_VARIANT:=pd_afd}"
DEPLOYMENT_SLUG="${DEPLOYMENT_VARIANT//_/-}"
: "${RUN_ROOT:=/data/z00569729/run/dsv4-mooncake-${DEPLOYMENT_SLUG}}"
: "${STATE_ROOT:=${RUN_ROOT}/state/${NODE_ROLE}}"
: "${LOG_ROOT:=${RUN_ROOT}/logs/${NODE_ROLE}}"
: "${VALIDATION_ROOT:=${RUN_ROOT}/validation}"
: "${OUTPUT_ROOT:=${RUN_ROOT}/output}"
: "${NATIVE_GOLDEN_PATH:=${GOLDEN_PATH:-/data/z00569729/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json}}"
: "${PD_CONTROL_GOLDEN_PATH:=/data/z00569729/validation/dsv4_m9_pd_control/golden_results.json}"
: "${VLLM_COMMIT:=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665}"
: "${VLLM_ASCEND_COMMIT:=3da28f9414583d2d0b672a8f06d1fae142404bda}"
: "${VLLM_ASCEND_WORKTREE_MODE:=clean}"
: "${ENABLE_BATCH_INVARIANT:=0}"
: "${BATCH_INVARIANT_OPP_ROOT:=${CODE_ROOT}/.ascend/custom-opp/batch-invariant-a3-1.0.0}"
: "${MOONCAKE_INSTALL_MODE:=wheel}"
: "${MOONCAKE_VERSION:=0.3.9}"
: "${MOONCAKE_LIBRARY_DIR:=}"
: "${MOONCAKE_WHEEL_SHA256:=0f9964801b24fd683d6016e1196cc0606fc87b0285b45d89c433650b9477ca12}"
: "${PREFILL_API_PORT:=8100}"
: "${DECODE_API_PORT:=8910}"
: "${FFN_PROCESS_PORT:=8911}"
: "${PROXY_PORT:=9000}"
: "${AFD_PORT:=29761}"
: "${PREFILL_KV_PORT:=30000}"
: "${DECODE_KV_PORT:=30100}"
: "${CONTROL_DATA_PARALLEL_RPC_PORT:=29360}"
: "${CONTROL_MASTER_PORT:=29361}"
: "${PREFILL_HCCL_IF_BASE_PORT:=50000}"
: "${ATTENTION_HCCL_IF_BASE_PORT:=51000}"
: "${FFN_HCCL_IF_BASE_PORT:=52000}"
: "${MC_MIN_PRC_PORT:=15000}"
: "${MC_MAX_PRC_PORT:=17000}"
: "${PREFILL_DEVICES:=0,1,2,3,4,5,6,7}"
: "${ATTENTION_DEVICES:=0,1,2,3,4,5,6,7}"
: "${FFN_DEVICES:=8,9,10,11,12,13,14,15}"
: "${PREFILL_DP_SIZE:=2}"
: "${PREFILL_TP_SIZE:=4}"
: "${ATTENTION_RANKS:=8}"
: "${FFN_RANKS:=8}"
: "${DECODE_DP_SIZE:=8}"
: "${DECODE_TP_SIZE:=1}"
: "${AFD_PLACEMENT:=colocated}"
: "${DECODE_EXECUTION_MODE:=eager}"
: "${DECODE_U_BATCHES:=1}"
: "${DECODE_ENABLE_MTP:=0}"
: "${DECODE_MTP_DRAFT_EXECUTION:=eager}"
: "${DECODE_MTP_NUM_SPECULATIVE_TOKENS:=1}"
: "${DECODE_DBO_DECODE_TOKEN_THRESHOLD:=2}"
: "${DECODE_DBO_PREFILL_TOKEN_THRESHOLD:=12}"
: "${DECODE_MAX_CUDAGRAPH_CAPTURE_SIZE:=8}"
: "${DECODE_CUDAGRAPH_CAPTURE_SIZES:=1 2 4 8}"
: "${MAX_MODEL_LEN:=4096}"
: "${MAX_NUM_BATCHED_TOKENS:=4096}"
: "${ATTENTION_MAX_NUM_BATCHED_TOKENS:=${MAX_NUM_BATCHED_TOKENS}}"
: "${FFN_MAX_NUM_BATCHED_TOKENS:=${MAX_NUM_BATCHED_TOKENS}}"
: "${MAX_NUM_SEQS:=16}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${HCCL_BUFFSIZE:=2048}"
: "${OMP_NUM_THREADS:=10}"
: "${MODEL_NAME:=dsv4-afd}"
: "${INSTALL_SYSTEM_PACKAGES:=0}"
: "${RUN_LOCAL_ROUNDTRIP:=1}"
: "${ROUNDTRIP_PRODUCER_DEVICE:=0}"
: "${ROUNDTRIP_CONSUMER_DEVICE:=1}"
: "${WAIT_READY:=1}"
: "${STARTUP_TIMEOUT_SECONDS:=3600}"
: "${STOP_TIMEOUT_SECONDS:=300}"
: "${AFD_PROFILE_START_TIMEOUT_SECONDS:=60}"
: "${AFD_PROFILE_FINALIZE_TIMEOUT_SECONDS:=300}"
: "${AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS:=240}"
: "${FORCE_KILL:=0}"
: "${ALLOW_NPU_PROCESSES:=0}"
: "${ALLOW_COLOCATED_PD_CONTROL:=0}"
: "${COLOCATED_PREFILL_PID_FILE:=${RUN_ROOT}/state/prefill/prefill.pid}"
: "${VALIDATION_ROUNDS:=3}"
: "${VALIDATION_BATCH_SIZES:=1 8 32}"
: "${RUN_CANCELLATION_TEST:=1}"
: "${AFD_PROFILE_ENABLE:=0}"
: "${AFD_PROFILE_ATTENTION_DIR:=${RUN_ROOT}/profile/attention}"
: "${AFD_PROFILE_FFN_DIR:=${RUN_ROOT}/profile/ffn}"
: "${AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP:=1}"
: "${AFD_HCCL_GRAPH_U2_HYBRID_DAG:=1}"
: "${AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM:=1}"
: "${AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM:=1}"
: "${AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER:=1}"
: "${ARTIFACT_LOG_TAIL_BYTES:=262144}"
: "${ARTIFACT_MAX_BYTES:=2097152}"
: "${PORT_SNAPSHOT_TIMEOUT_SECONDS:=10}"

PYTHON_BIN="${VENV_ROOT}/bin/python"
PREFILL_SCRIPT="${AFD_PLUGIN_ROOT}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/prefill.sh"
CONTROL_DECODE_SCRIPT="${AFD_PLUGIN_ROOT}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/decode_control.sh"
ATTENTION_SCRIPT="${AFD_PLUGIN_ROOT}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_attention.sh"
FFN_SCRIPT="${AFD_PLUGIN_ROOT}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_ffn.sh"
PROXY_SCRIPT="${AFD_PLUGIN_ROOT}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_pd/proxy.sh"
RUNTIME_CHECK="${AFD_PLUGIN_ROOT}/tools/dsv4/check_mooncake_runtime.sh"
ROUNDTRIP_TOOL="${AFD_PLUGIN_ROOT}/tools/dsv4/check_mooncake_npu_roundtrip.py"
FUNCTIONAL_SMOKE_TOOL="${AFD_PLUGIN_ROOT}/tools/dsv4/run_pd_functional_smoke.py"
GOLDEN_VALIDATOR="${AFD_PLUGIN_ROOT}/recipe/npu/deepseek_v4/common/validate_golden.py"
GOLDEN_GENERATOR="${AFD_PLUGIN_ROOT}/tools/dsv4/generate_golden.py"
FATAL_PATTERN='EngineCore encountered a fatal error|AFD NPU FFN worker loop failed|Mooncake transfer failed|Communication_Error|507015|Traceback'

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

resolve_hostname() {
  local node_name=""
  if [[ -r /proc/sys/kernel/hostname ]]; then
    IFS= read -r node_name </proc/sys/kernel/hostname || true
  fi
  if [[ -n "${node_name}" ]]; then
    printf '%s\n' "${node_name}"
  elif [[ -n "${HOSTNAME:-}" ]]; then
    printf '%s\n' "${HOSTNAME}"
  elif command -v hostname >/dev/null 2>&1; then
    hostname
  elif command -v uname >/dev/null 2>&1; then
    uname -n
  else
    printf 'unknown\n'
  fi
}

run_as_root() {
  if (( EUID == 0 )); then
    "$@"
  else
    require_command sudo
    sudo "$@"
  fi
}

install_system_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    run_as_root apt-get update
    run_as_root apt-get install -y \
      iproute2 curl libgoogle-glog0v6t64 libjsoncpp25 libjemalloc2 netcat-openbsd
  elif command -v dnf >/dev/null 2>&1; then
    run_as_root dnf install -y iproute curl glog jsoncpp jemalloc nmap-ncat
  elif command -v yum >/dev/null 2>&1; then
    run_as_root yum install -y iproute curl glog jsoncpp jemalloc nmap-ncat
  else
    die "Unsupported package manager; install ip/ss, curl, glog, jsoncpp, jemalloc, and netcat manually"
  fi
}

assert_integer() {
  [[ "$2" =~ ^[0-9]+$ ]] || die "$1 must be a non-negative integer: $2"
}

pid_is_alive() {
  [[ "$1" =~ ^[0-9]+$ ]] && kill -0 "$1" 2>/dev/null
}

read_pid() {
  local path="$1"
  local pid
  [[ -f "${path}" ]] || return 1
  read -r pid <"${path}"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${pid}"
}

device_lists_are_disjoint() {
  local left_value="$1"
  local right_value="$2"
  local item
  local -a left_devices=()
  local -a right_devices=()
  local -A seen=()
  IFS=',' read -r -a left_devices <<<"${left_value}"
  IFS=',' read -r -a right_devices <<<"${right_value}"
  (( ${#left_devices[@]} > 0 && ${#right_devices[@]} > 0 )) || return 1
  for item in "${left_devices[@]}"; do
    [[ "${item}" =~ ^[0-9]+$ && -z "${seen[${item}]+present}" ]] || return 1
    seen["${item}"]=1
  done
  for item in "${right_devices[@]}"; do
    [[ "${item}" =~ ^[0-9]+$ && -z "${seen[${item}]+present}" ]] || return 1
    seen["${item}"]=1
  done
}

port_is_listening() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn | awk -v expected=":${port}" '$4 ~ expected "$" {found=1} END {exit !found}'
    return
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -lnt | awk -v expected=":${port}" \
      '$4 ~ expected "$" && $6 == "LISTEN" {found=1} END {exit !found}'
    return
  fi
  local port_hex
  printf -v port_hex '%04X' "${port}"
  awk -v expected="${port_hex}" '
    FNR > 1 {
      split($2, address, ":")
      if (toupper(address[2]) == expected && $4 == "0A") found=1
    }
    END {exit !found}
  ' /proc/net/tcp /proc/net/tcp6
}

git_head() {
  git -c safe.directory="$1" -C "$1" rev-parse HEAD
}

validate_vllm_ascend_worktree() {
  local status diff_sha
  status="$(git -c safe.directory="${VLLM_ASCEND_ROOT}" \
    -C "${VLLM_ASCEND_ROOT}" status --short --untracked-files=all)"
  case "${VLLM_ASCEND_WORKTREE_MODE}" in
    clean)
      [[ -z "${status}" ]] || die "vLLM-Ascend worktree is dirty"
      ;;
    batch_invariant_patch)
      [[ "${status}" == $' M tests/ut/test_batch_invariant.py\n M vllm_ascend/batch_invariant.py' ]] \
        || die "vLLM-Ascend changes do not match the delivered batch-invariant patch"
      diff_sha="$(git -c safe.directory="${VLLM_ASCEND_ROOT}" \
        -C "${VLLM_ASCEND_ROOT}" diff -- \
        tests/ut/test_batch_invariant.py vllm_ascend/batch_invariant.py \
        | sha256sum | awk '{print $1}')"
      [[ "${diff_sha}" == "cf97a0b6e509fbb128e847babbf8f01cc953f06cb3126936cc4111bbab60b897" ]] \
        || die "vLLM-Ascend batch-invariant patch fingerprint mismatch"
      ;;
    *)
      die "VLLM_ASCEND_WORKTREE_MODE must be clean or batch_invariant_patch"
      ;;
  esac
}

validate_afd_worktree() {
  local status
  status="$(git -c safe.directory="${AFD_PLUGIN_ROOT}" \
    -C "${AFD_PLUGIN_ROOT}" status --short --untracked-files=all)"
  if [[ -n "${status}" ]]; then
    printf '%s\n' "${status}" >&2
    die "afd-plugin worktree must be clean; commit the exact validation code first"
  fi
  return 0
}

validate_role() {
  case "${NODE_ROLE:-}" in
    prefill|decode|prefill_ffn|attention|proxy) ;;
    *) die "NODE_ROLE must be prefill, decode, prefill_ffn, attention, or proxy: ${NODE_ROLE:-unset}" ;;
  esac
}

validate_variant() {
  case "${DEPLOYMENT_VARIANT}" in
    pd_control|pd_afd) ;;
    *) die "DEPLOYMENT_VARIANT must be pd_control or pd_afd: ${DEPLOYMENT_VARIANT}" ;;
  esac
}

reject_placeholder() {
  local name="$1"
  local value="$2"
  [[ -n "${value}" && "${value}" != *CHANGE_ME* && "${value}" != *REPLACE_WITH* ]] \
    || die "${name} is not configured: ${value}"
}

validate_common_config() {
  validate_role
  validate_variant
  reject_placeholder PREFILL_IP "${PREFILL_IP:-}"
  reject_placeholder DECODE_IP "${DECODE_IP:-}"
  [[ "${AFD_PD_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] \
    || die "AFD_PD_COMMIT must be the delivered 40-character phase-one commit"
  assert_integer STARTUP_TIMEOUT_SECONDS "${STARTUP_TIMEOUT_SECONDS}"
  assert_integer STOP_TIMEOUT_SECONDS "${STOP_TIMEOUT_SECONDS}"
  assert_integer AFD_PROFILE_START_TIMEOUT_SECONDS \
    "${AFD_PROFILE_START_TIMEOUT_SECONDS}"
  assert_integer AFD_PROFILE_FINALIZE_TIMEOUT_SECONDS \
    "${AFD_PROFILE_FINALIZE_TIMEOUT_SECONDS}"
  assert_integer AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS \
    "${AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS}"
  assert_integer ARTIFACT_LOG_TAIL_BYTES "${ARTIFACT_LOG_TAIL_BYTES}"
  assert_integer ARTIFACT_MAX_BYTES "${ARTIFACT_MAX_BYTES}"
  assert_integer DECODE_DBO_DECODE_TOKEN_THRESHOLD "${DECODE_DBO_DECODE_TOKEN_THRESHOLD}"
  assert_integer DECODE_DBO_PREFILL_TOKEN_THRESHOLD "${DECODE_DBO_PREFILL_TOKEN_THRESHOLD}"
  assert_integer DECODE_MAX_CUDAGRAPH_CAPTURE_SIZE "${DECODE_MAX_CUDAGRAPH_CAPTURE_SIZE}"
  local capture_size
  [[ -n "${DECODE_CUDAGRAPH_CAPTURE_SIZES//[[:space:]]/}" ]] \
    || die "DECODE_CUDAGRAPH_CAPTURE_SIZES must not be empty"
  for capture_size in ${DECODE_CUDAGRAPH_CAPTURE_SIZES}; do
    assert_integer DECODE_CUDAGRAPH_CAPTURE_SIZES "${capture_size}"
    (( capture_size > 0 )) || die "CUDAGRAPH capture sizes must be positive"
  done
  [[ "${NATIVE_GOLDEN_PATH}" != "${PD_CONTROL_GOLDEN_PATH}" ]] \
    || die "Native and PD control golden paths must be different"
  [[ "${PREFILL_DP_SIZE}" == "2" && "${PREFILL_TP_SIZE}" == "4" ]] \
    || die "M9 baseline requires Prefill DP2/TP4"
  if [[ "${DEPLOYMENT_VARIANT}" == "pd_control" ]]; then
    case "${DECODE_DP_SIZE}:${DECODE_TP_SIZE}" in
      8:1|4:1|4:2) ;;
      *) die "Phase-one control supports Decode DP8/TP1, DP4/TP1, or DP4/TP2" ;;
    esac
  else
    [[ "${DECODE_TP_SIZE}" == "1" ]] \
      || die "Unequal A/F PD validation requires TP1"
    assert_integer ATTENTION_RANKS "${ATTENTION_RANKS}"
    assert_integer FFN_RANKS "${FFN_RANKS}"
    (( ATTENTION_RANKS > 0 && FFN_RANKS > 0 )) \
      || die "ATTENTION_RANKS and FFN_RANKS must be positive"
    local larger_ranks smaller_ranks ratio
    larger_ranks=$((ATTENTION_RANKS > FFN_RANKS ? ATTENTION_RANKS : FFN_RANKS))
    smaller_ranks=$((ATTENTION_RANKS < FFN_RANKS ? ATTENTION_RANKS : FFN_RANKS))
    (( larger_ranks % smaller_ranks == 0 )) \
      || die "Attention and FFN ranks must have an integer ratio"
    ratio=$((larger_ranks / smaller_ranks))
    (( DECODE_DP_SIZE == ATTENTION_RANKS )) \
      || die "DECODE_DP_SIZE must equal ATTENTION_RANKS under TP1"
    local required_ffn_tokens
    if (( FFN_RANKS > ATTENTION_RANKS )); then
      required_ffn_tokens=$(((ATTENTION_MAX_NUM_BATCHED_TOKENS + ratio - 1) / ratio))
    else
      required_ffn_tokens=$((ATTENTION_MAX_NUM_BATCHED_TOKENS * ratio))
    fi
    (( FFN_MAX_NUM_BATCHED_TOKENS >= required_ffn_tokens )) \
      || die "FFN_MAX_NUM_BATCHED_TOKENS must be at least ${required_ffn_tokens}"
  fi
  case "${AFD_PLACEMENT}" in
    colocated)
      [[ "${NODE_ROLE}" != "prefill_ffn" && "${NODE_ROLE}" != "attention" ]] \
        || die "Split roles require AFD_PLACEMENT=split"
      ;;
    split)
      [[ "${DEPLOYMENT_VARIANT}" == "pd_afd" ]] \
        || die "AFD_PLACEMENT=split requires DEPLOYMENT_VARIANT=pd_afd"
      [[ "${NODE_ROLE}" != "prefill" && "${NODE_ROLE}" != "decode" ]] \
        || die "Split placement uses prefill_ffn, attention, and proxy roles"
      device_lists_are_disjoint "${PREFILL_DEVICES}" "${FFN_DEVICES}" \
        || die "Split placement requires disjoint Prefill and FFN devices"
      ;;
    *) die "AFD_PLACEMENT must be colocated or split" ;;
  esac
  case "${DECODE_EXECUTION_MODE}" in
    eager|full-decode-only) ;;
    *) die "DECODE_EXECUTION_MODE must be eager or full-decode-only" ;;
  esac
  case "${DECODE_U_BATCHES}" in
    1|2) ;;
    *) die "DECODE_U_BATCHES must be 1 or 2" ;;
  esac
  case "${DECODE_ENABLE_MTP}" in
    0|1) ;;
    *) die "DECODE_ENABLE_MTP must be 0 or 1" ;;
  esac
  case "${AFD_PROFILE_ENABLE}" in
    0) ;;
    1)
      [[ "${DEPLOYMENT_VARIANT}" == "pd_afd" ]] \
        || die "AFD_PROFILE_ENABLE=1 is only valid for pd_afd"
      (( AFD_PROFILE_START_TIMEOUT_SECONDS > 0 )) \
        || die "AFD_PROFILE_START_TIMEOUT_SECONDS must be positive"
      (( AFD_PROFILE_FINALIZE_TIMEOUT_SECONDS > 0 )) \
        || die "AFD_PROFILE_FINALIZE_TIMEOUT_SECONDS must be positive"
      (( AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS > 0 )) \
        || die "AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS must be positive"
      (( STOP_TIMEOUT_SECONDS > AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS )) \
        || die "STOP_TIMEOUT_SECONDS must exceed AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS"
      ;;
    *) die "AFD_PROFILE_ENABLE must be 0 or 1" ;;
  esac
  local graph_u2_flag
  for graph_u2_flag in \
    AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP \
    AFD_HCCL_GRAPH_U2_HYBRID_DAG \
    AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM \
    AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM \
    AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER; do
    [[ "${!graph_u2_flag}" == "0" || "${!graph_u2_flag}" == "1" ]] \
      || die "${graph_u2_flag} must be 0 or 1"
  done
  if [[ "${DECODE_EXECUTION_MODE}:${DECODE_U_BATCHES}" == "full-decode-only:2" ]]; then
    [[ "${AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP}" == "1" \
      && "${AFD_HCCL_GRAPH_U2_HYBRID_DAG}" == "1" \
      && "${AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM}" == "1" \
      && "${AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM}" == "1" \
      && "${AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER}" == "1" ]] \
      || die "Graph U2 comparison requires hybrid DAG and all physical streams enabled"
  fi
  case "${ENABLE_BATCH_INVARIANT}" in
    0) ;;
    1)
      [[ "${VLLM_ASCEND_WORKTREE_MODE}" == "batch_invariant_patch" ]] \
        || die "ENABLE_BATCH_INVARIANT=1 requires VLLM_ASCEND_WORKTREE_MODE=batch_invariant_patch"
      require_file "${BATCH_INVARIANT_OPP_ROOT}/vendors/batch_invariant/bin/set_env.bash"
      ;;
    *) die "ENABLE_BATCH_INVARIANT must be 0 or 1" ;;
  esac
  case "${ALLOW_COLOCATED_PD_CONTROL}" in
    0) ;;
    1)
      [[ "${DEPLOYMENT_VARIANT}" == "pd_control" ]] \
        || die "ALLOW_COLOCATED_PD_CONTROL requires DEPLOYMENT_VARIANT=pd_control"
      [[ "${PREFILL_IP}" == "${DECODE_IP}" ]] \
        || die "Colocated PD control requires identical Prefill and Decode IPs"
      device_lists_are_disjoint "${PREFILL_DEVICES}" "${ATTENTION_DEVICES}" \
        || die "Colocated PD control requires valid, disjoint Prefill and Decode devices"
      ;;
    *) die "ALLOW_COLOCATED_PD_CONTROL must be 0 or 1" ;;
  esac
  case "${DECODE_MTP_DRAFT_EXECUTION}" in
    eager|graph) ;;
    *) die "DECODE_MTP_DRAFT_EXECUTION must be eager or graph" ;;
  esac
  [[ "${DECODE_MTP_NUM_SPECULATIVE_TOKENS}" =~ ^[1-3]$ ]] \
    || die "Phase-one MTP requires num_speculative_tokens in [1, 3]"
  if [[ "${DECODE_ENABLE_MTP}" == "1" ]]; then
    if [[ "${DECODE_EXECUTION_MODE}" == "eager" \
      && "${DECODE_MTP_DRAFT_EXECUTION}" != "eager" ]]; then
      die "Eager Decode requires DECODE_MTP_DRAFT_EXECUTION=eager"
    fi
  fi
  if [[ "${DECODE_TP_SIZE}" == "2" \
    && "${DECODE_EXECUTION_MODE}" == "full-decode-only" \
    && "${DECODE_U_BATCHES}" == "2" \
    && "${DECODE_ENABLE_MTP}" == "1" \
    && "${DECODE_MTP_DRAFT_EXECUTION}" == "graph" ]]; then
    die "TP2 full-draft Graph U2 + MTP is not validated"
  fi
  require_command git
  require_dir "${AFD_PLUGIN_ROOT}"
  require_dir "${VLLM_ROOT}"
  require_dir "${VLLM_ASCEND_ROOT}"
  require_file "${PYTHON_BIN}"
  [[ "$(git_head "${VLLM_ROOT}")" == "${VLLM_COMMIT}" ]] \
    || die "vLLM commit mismatch"
  [[ "$(git_head "${VLLM_ASCEND_ROOT}")" == "${VLLM_ASCEND_COMMIT}" ]] \
    || die "vLLM-Ascend commit mismatch"
  [[ "$(git_head "${AFD_PLUGIN_ROOT}")" == "${AFD_PD_COMMIT}" ]] \
    || die "afd-plugin commit mismatch"
  [[ -z "$(git -c safe.directory="${VLLM_ROOT}" -C "${VLLM_ROOT}" status --short)" ]] \
    || die "vLLM worktree is dirty"
  validate_vllm_ascend_worktree
  validate_afd_worktree
}

export_batch_invariant_env() {
  if [[ "${ENABLE_BATCH_INVARIANT}" == "1" ]]; then
    export DSV4_EXTRA_OPP_ENV="${BATCH_INVARIANT_OPP_ROOT}/vendors/batch_invariant/bin/set_env.bash"
    export VLLM_BATCH_INVARIANT=1
    export HCCL_DETERMINISTIC=true
    export LCCL_DETERMINISTIC=1
    export ATB_MATMUL_SHUFFLE_K_ENABLE=0
    export ATB_LLM_LCOC_ENABLE=0
  else
    unset DSV4_EXTRA_OPP_ENV VLLM_BATCH_INVARIANT
  fi
}

local_role_ip() {
  case "${NODE_ROLE}" in
    prefill|prefill_ffn) printf '%s\n' "${PREFILL_IP}" ;;
    decode|attention) printf '%s\n' "${DECODE_IP}" ;;
    proxy) printf '%s\n' "" ;;
  esac
}

validate_local_network() {
  [[ "${NODE_ROLE}" == "proxy" ]] && return 0
  local expected_ip
  expected_ip="$(local_role_ip)"
  [[ -d "/sys/class/net/${NIC_NAME}" ]] \
    || die "Network interface not found: ${NIC_NAME}"
  if command -v ip >/dev/null 2>&1; then
    ip -o -4 addr show dev "${NIC_NAME}" \
      | awk '{split($4, a, "/"); print a[1]}' \
      | grep -Fxq "${expected_ip}" \
      || die "${expected_ip} is not assigned to ${NIC_NAME}"
    return
  fi
  if command -v ifconfig >/dev/null 2>&1; then
    ifconfig "${NIC_NAME}" 2>/dev/null \
      | awk '
          /^[[:space:]]*inet[[:space:]]/ {
            for (i = 1; i <= NF; i++) {
              if ($i == "inet" && i < NF) {
                value = $(i + 1)
                sub(/^addr:/, "", value)
                print value
              }
            }
          }
        ' \
      | grep -Fxq "${expected_ip}" \
      || die "${expected_ip} is not assigned to ${NIC_NAME}"
    return
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -ie 2>/dev/null \
      | awk -v interface="${NIC_NAME}" '
          /^[^[:space:]]/ {
            current = $1
            sub(/:.*/, "", current)
          }
          current == interface && /^[[:space:]]*inet[[:space:]]/ {
            for (i = 1; i <= NF; i++) {
              if ($i == "inet" && i < NF) {
                value = $(i + 1)
                sub(/^addr:/, "", value)
                print value
              }
            }
          }
        ' \
      | grep -Fxq "${expected_ip}" \
      || die "${expected_ip} is not assigned to ${NIC_NAME}"
    return
  fi
  "${PYTHON_BIN}" - "${NIC_NAME}" "${expected_ip}" <<'PY' \
    || die "${expected_ip} is not the primary IPv4 assigned to ${NIC_NAME}"
import fcntl
import socket
import struct
import sys

interface, expected = sys.argv[1:]
request = struct.pack("256s", interface.encode()[:15])
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
    response = fcntl.ioctl(sock.fileno(), 0x8915, request)
actual = socket.inet_ntoa(response[20:24])
if actual != expected:
    print(f"{interface}: expected {expected}, found {actual}", file=sys.stderr)
    raise SystemExit(1)
PY
}

wheel_sha256() {
  sha256sum "${MOONCAKE_WHEEL}" | awk '{print $1}'
}

validate_wheel() {
  reject_placeholder MOONCAKE_WHEEL "${MOONCAKE_WHEEL:-}"
  require_file "${MOONCAKE_WHEEL}"
  [[ "$(wheel_sha256)" == "${MOONCAKE_WHEEL_SHA256}" ]] \
    || die "Mooncake wheel SHA256 mismatch"
}

validate_mooncake_install_mode() {
  case "${MOONCAKE_INSTALL_MODE}" in
    wheel|existing) ;;
    *) die "MOONCAKE_INSTALL_MODE must be wheel or existing: ${MOONCAKE_INSTALL_MODE}" ;;
  esac
}

validate_cann_version() {
  local resolved version_text version_file expected_regex
  [[ "${CANN_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
    || die "CANN_VERSION must use major.minor.patch format: ${CANN_VERSION}"
  resolved="$(readlink -f "${CANN_ROOT}")"
  version_text="${resolved}"
  if [[ -x "${CANN_ROOT}/query_pkg_version.sh" ]]; then
    version_text+=$'\n'"$("${CANN_ROOT}/query_pkg_version.sh" 2>&1 || true)"
  fi
  version_file="$(find "${CANN_ROOT}" -name version.info -type f -print -quit 2>/dev/null || true)"
  if [[ -n "${version_file}" ]]; then
    version_text+=$'\n'"$(head -n 20 "${version_file}" 2>/dev/null || true)"
  fi
  expected_regex="${CANN_VERSION//./\\.}"
  grep -Eq "(^|[^0-9])${expected_regex}([^0-9]|$)" <<<"${version_text}" \
    || die "CANN ${CANN_VERSION} not detected at ${resolved}"
}

resolve_atb_root() {
  if [[ -n "${ATB_ROOT}" ]]; then
    require_file "${ATB_ROOT}/set_env.sh"
    return 0
  fi
  if [[ -f "${CANN_ROOT}/nnal/atb/set_env.sh" ]]; then
    ATB_ROOT="${CANN_ROOT}/nnal/atb"
    return 0
  fi
  if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    ATB_ROOT=/usr/local/Ascend/nnal/atb
    return 0
  fi
  die "NNAL/ATB was not found; set ATB_ROOT to a directory containing set_env.sh"
}

installed_mooncake_version() {
  "${PYTHON_BIN}" -c \
    'from importlib.metadata import version; print(version("mooncake-transfer-engine"))'
}

validate_installed_mooncake() {
  local installed_version
  installed_version="$(installed_mooncake_version 2>/dev/null)" \
    || die "mooncake-transfer-engine is not installed in ${PYTHON_BIN}"
  [[ "${installed_version}" == "${MOONCAKE_VERSION}" ]] \
    || die "Mooncake version mismatch: expected ${MOONCAKE_VERSION}, got ${installed_version}"
}

write_mooncake_fingerprint() {
  local output_path="$1"
  local package_dir site_packages library library_dir candidate
  site_packages="$("${PYTHON_BIN}" -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  package_dir="$(PYTHONPATH="${site_packages}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -c \
    'import importlib.util; s=importlib.util.find_spec("mooncake"); print(next(iter(s.submodule_search_locations or []), "") if s else "")')"
  require_dir "${package_dir}"
  library_dir="${MOONCAKE_LIBRARY_DIR}"
  if [[ -z "${library_dir}" ]]; then
    for candidate in "${package_dir}" /usr/local/lib /usr/local/lib64; do
      if [[ -f "${candidate}/libtransfer_engine.so" \
        && -f "${candidate}/ascend_transport.so" ]]; then
        library_dir="${candidate}"
        break
      fi
    done
  fi
  require_dir "${library_dir}"
  : >"${output_path}"
  while IFS= read -r -d '' library; do
    printf '%s  %s\n' \
      "$(sha256sum "${library}" | awk '{print $1}')" \
      "${library#"${package_dir}"/}" >>"${output_path}"
  done < <(find "${package_dir}" -type f -name '*.so' -print0 | sort -z)
  if [[ "${library_dir}" != "${package_dir}" ]]; then
    for library in \
      "${library_dir}/libtransfer_engine.so" \
      "${library_dir}/ascend_transport.so"; do
      require_file "${library}"
      printf '%s  runtime/%s\n' \
        "$(sha256sum "${library}" | awk '{print $1}')" \
        "$(basename "${library}")" >>"${output_path}"
    done
  fi
  [[ -s "${output_path}" ]] || die "No Mooncake shared libraries found in ${package_dir}"
}

npu_process_pids() {
  npu-smi info | awk '
    /\| NPU +Chip +\| Process id/ {in_process_table=1; next}
    in_process_table && /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+/ {
      split($0, fields, "|")
      gsub(/[[:space:]]/, "", fields[3])
      if (fields[3] ~ /^[0-9]+$/) print fields[3]
    }
  ' | sort -u
}

npu_process_count() {
  local pids=()
  mapfile -t pids < <(npu_process_pids)
  printf '%s\n' "${#pids[@]}"
}

process_is_descendant_of() {
  local current_pid="$1"
  local ancestor_pid="$2"
  local parent_pid
  while [[ "${current_pid}" =~ ^[0-9]+$ ]] && (( current_pid > 1 )); do
    [[ "${current_pid}" == "${ancestor_pid}" ]] && return 0
    [[ -r "/proc/${current_pid}/status" ]] || return 1
    parent_pid="$(awk '/^PPid:/ {print $2}' "/proc/${current_pid}/status")"
    [[ "${parent_pid}" =~ ^[0-9]+$ ]] || return 1
    current_pid="${parent_pid}"
  done
  return 1
}

validate_colocated_control_processes() {
  [[ "${NODE_ROLE}" == "decode" && "${DEPLOYMENT_VARIANT}" == "pd_control" ]] \
    || die "Existing NPU processes are allowed only for a colocated control Decode"
  local prefill_pid actual_devices process_pid
  prefill_pid="$(read_pid "${COLOCATED_PREFILL_PID_FILE}")" \
    || die "Missing colocated Prefill PID: ${COLOCATED_PREFILL_PID_FILE}"
  pid_is_alive "${prefill_pid}" \
    || die "Colocated Prefill PID is not running: ${prefill_pid}"
  owned_process "${prefill_pid}" \
    || die "Colocated Prefill PID is not owned by this deployment: ${prefill_pid}"
  actual_devices="$(tr '\0' '\n' <"/proc/${prefill_pid}/environ" \
    | awk -F= '$1 == "PREFILL_DEVICES" {print $2; exit}')"
  [[ "${actual_devices}" == "${PREFILL_DEVICES}" ]] \
    || die "Colocated Prefill devices do not match this config"
  while IFS= read -r process_pid; do
    process_is_descendant_of "${process_pid}" "${prefill_pid}" \
      || die "NPU process ${process_pid} is not owned by colocated Prefill ${prefill_pid}"
  done < <(npu_process_pids)
  log "Allowing only colocated Prefill NPU processes; Decode devices are disjoint"
}

check_npus() {
  require_command npu-smi
  local expected=8
  if [[ "${NODE_ROLE}" == "decode" && "${DEPLOYMENT_VARIANT}" == "pd_afd" ]] \
    || [[ "${NODE_ROLE}" == "prefill_ffn" ]] \
    || [[ "${NODE_ROLE}" == "attention" && "${ATTENTION_RANKS}" == "16" ]]; then
    expected=16
  fi
  local detected
  detected="$(npu-smi info -l | awk -F: '/Chip Count/ {gsub(/[[:space:]]/, "", $2); sum += $2} END {print sum + 0}')"
  (( detected >= expected )) || die "${NODE_ROLE} requires ${expected} NPUs, detected ${detected}"
  local process_count
  process_count="$(npu_process_count)"
  if (( process_count > 0 )); then
    if is_true "${ALLOW_NPU_PROCESSES}"; then
      warn "ALLOW_NPU_PROCESSES permits ${process_count} existing NPU processes"
    elif is_true "${ALLOW_COLOCATED_PD_CONTROL}"; then
      validate_colocated_control_processes
    else
      die "Detected ${process_count} existing NPU processes"
    fi
  fi
}

owned_pid_names() {
  case "${NODE_ROLE}" in
    prefill) printf '%s\n' prefill ;;
    prefill_ffn) printf '%s\n' prefill ffn ;;
    attention) printf '%s\n' attention ;;
    decode)
      if [[ "${DEPLOYMENT_VARIANT}" == "pd_control" ]]; then
        printf '%s\n' decode-control
      else
        printf '%s\n' attention ffn
      fi
      ;;
    proxy) printf '%s\n' proxy ;;
  esac
}

owned_process() {
  local pid="$1"
  local cmdline
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${cmdline}" == *"${AFD_PLUGIN_ROOT}"* || "${cmdline}" == *"vllm"* ]]
}

ensure_not_running() {
  local name pid
  while read -r name; do
    if pid="$(read_pid "${STATE_ROOT}/${name}.pid" 2>/dev/null)" && pid_is_alive "${pid}"; then
      die "${name} is already running with PID ${pid}"
    fi
  done < <(owned_pid_names)
}

check_start_ports() {
  local ports=()
  case "${NODE_ROLE}" in
    prefill) ports=("${PREFILL_API_PORT}" "${PREFILL_KV_PORT}" "${PREFILL_HCCL_IF_BASE_PORT}") ;;
    prefill_ffn) ports=("${PREFILL_API_PORT}" "${FFN_PROCESS_PORT}" "${AFD_PORT}" \
      "${PREFILL_KV_PORT}" "${PREFILL_HCCL_IF_BASE_PORT}" "${FFN_HCCL_IF_BASE_PORT}") ;;
    attention) ports=("${DECODE_API_PORT}" "${DECODE_KV_PORT}" \
      "${ATTENTION_HCCL_IF_BASE_PORT}") ;;
    decode)
      if [[ "${DEPLOYMENT_VARIANT}" == "pd_control" ]]; then
        ports=("${DECODE_API_PORT}" "${DECODE_KV_PORT}" \
          "${ATTENTION_HCCL_IF_BASE_PORT}" "${CONTROL_DATA_PARALLEL_RPC_PORT}" \
          "${CONTROL_MASTER_PORT}")
      else
        ports=("${DECODE_API_PORT}" "${FFN_PROCESS_PORT}" "${AFD_PORT}" \
          "${DECODE_KV_PORT}" "${ATTENTION_HCCL_IF_BASE_PORT}" \
          "${FFN_HCCL_IF_BASE_PORT}")
      fi
      ;;
    proxy) ports=("${PROXY_PORT}") ;;
  esac
  local port
  for port in "${ports[@]}"; do
    if port_is_listening "${port}"; then
      die "Port is already listening: ${port}"
    fi
  done
  return 0
}

export_runtime_env() {
  local local_ip
  local_ip="$(local_role_ip)"
  export DSV4_CANN_ROOT="${CANN_ROOT}"
  export DSV4_CANN_VERSION="${CANN_VERSION}"
  export DSV4_ATB_ROOT="${ATB_ROOT}"
  export DSV4_RUNTIME_VENV="${VENV_ROOT}"
  export DSV4_VLLM_VENV="${VENV_ROOT}"
  export DSV4_VLLM_ROOT="${VLLM_ROOT}"
  export DSV4_VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT}"
  export_batch_invariant_env
  export MODEL_PATH VLLM_HOST_IP="${local_ip}" HCCL_IF_IP="${local_ip}"
  export GLOO_SOCKET_IFNAME="${NIC_NAME}" HCCL_SOCKET_IFNAME="${NIC_NAME}"
  export MC_MIN_PRC_PORT MC_MAX_PRC_PORT MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS
  export ATTENTION_MAX_NUM_BATCHED_TOKENS FFN_MAX_NUM_BATCHED_TOKENS
  export MAX_NUM_SEQS GPU_MEMORY_UTILIZATION HCCL_BUFFSIZE OMP_NUM_THREADS
  export PREFILL_DEVICES PREFILL_DP_SIZE PREFILL_TP_SIZE DECODE_DP_SIZE DECODE_TP_SIZE
  export ATTENTION_DEVICES FFN_DEVICES ATTENTION_RANKS FFN_RANKS
  export PREFILL_HCCL_IF_BASE_PORT ATTENTION_HCCL_IF_BASE_PORT FFN_HCCL_IF_BASE_PORT
  export CONTROL_DATA_PARALLEL_RPC_PORT CONTROL_MASTER_PORT MODEL_NAME
  if [[ "${AFD_PLACEMENT}" == "split" ]]; then
    export AFD_HOST="${PREFILL_IP}"
  else
    export AFD_HOST=127.0.0.1
  fi
  export AFD_PORT
  export TENSOR_PARALLEL_SIZE="${DECODE_TP_SIZE}"
  export EXECUTION_MODE="${DECODE_EXECUTION_MODE}"
  export U_BATCHES="${DECODE_U_BATCHES}"
  export ENABLE_MTP="${DECODE_ENABLE_MTP}"
  export MTP_DRAFT_EXECUTION="${DECODE_MTP_DRAFT_EXECUTION}"
  export MTP_NUM_SPECULATIVE_TOKENS="${DECODE_MTP_NUM_SPECULATIVE_TOKENS}"
  export DBO_DECODE_TOKEN_THRESHOLD="${DECODE_DBO_DECODE_TOKEN_THRESHOLD}"
  export DBO_PREFILL_TOKEN_THRESHOLD="${DECODE_DBO_PREFILL_TOKEN_THRESHOLD}"
  export MAX_CUDAGRAPH_CAPTURE_SIZE="${DECODE_MAX_CUDAGRAPH_CAPTURE_SIZE}"
  export CUDAGRAPH_CAPTURE_SIZES="${DECODE_CUDAGRAPH_CAPTURE_SIZES}"
  export MOONCAKE_ENGINE_ID MOONCAKE_KV_PORT MOONCAKE_LIBRARY_DIR
  if is_true "${AFD_PROFILE_ENABLE}"; then
    export VLLM_SHUTDOWN_TIMEOUT_SECONDS="${AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS}"
  else
    export VLLM_SHUTDOWN_TIMEOUT_SECONDS=0
  fi
  export TORCH_PROFILER_WITH_STACK=0
  export AFD_NPU_ATTENTION_PROFILER_ENABLE="${AFD_PROFILE_ENABLE}"
  export AFD_NPU_ATTENTION_PROFILER_WITH_STACK=0
  export AFD_NPU_ATTENTION_PROFILER_DIR="${AFD_PROFILE_ATTENTION_DIR}"
  export AFD_NPU_FFN_PROFILER_ENABLE="${AFD_PROFILE_ENABLE}"
  export AFD_NPU_FFN_PROFILER_WITH_STACK=0
  export AFD_NPU_FFN_PROFILER_DIR="${AFD_PROFILE_FFN_DIR}"
  export AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP
  export AFD_HCCL_GRAPH_U2_HYBRID_DAG
  export AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM
  export AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM
  export AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER
}

wait_http() {
  local url="$1"
  shift
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  local pid
  while (( SECONDS < deadline )); do
    for pid in "$@"; do
      pid_is_alive "${pid}" || return 1
    done
    curl --noproxy '*' -fsS --max-time 5 "${url}" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

new_log() {
  local name="$1"
  local path="${LOG_ROOT}/${name}-$(date +%Y%m%d_%H%M%S).log"
  ln -sfn "$(basename "${path}")" "${LOG_ROOT}/${name}.log"
  printf '%s\n' "${path}"
}

start_prefill() {
  export_runtime_env
  export API_HOST=0.0.0.0 API_PORT="${PREFILL_API_PORT}"
  export MOONCAKE_ENGINE_ID="dsv4-${DEPLOYMENT_SLUG}-prefill"
  export MOONCAKE_KV_PORT="${PREFILL_KV_PORT}"
  # Prefill is identical across all matrix points. AFD is a Decode-only
  # variable, so loading the plugin here would invalidate the control.
  export ENABLE_AFD_PLUGIN=0
  local log_path pid
  log_path="$(new_log prefill)"
  nohup setsid bash "${PREFILL_SCRIPT}" >"${log_path}" 2>&1 &
  pid=$!
  printf '%s\n' "${pid}" >"${STATE_ROOT}/prefill.pid"
  sleep 3
  pid_is_alive "${pid}" || die "Prefill exited immediately; inspect ${log_path}"
  if is_true "${WAIT_READY}"; then
    wait_http "http://127.0.0.1:${PREFILL_API_PORT}/health" "${pid}" \
      || die "Prefill readiness failed; inspect ${log_path}"
  fi
  log "Prefill started: pid=${pid}, log=${log_path}"
}

start_control_decode() {
  export_runtime_env
  export MOONCAKE_ENGINE_ID="dsv4-pd-control-decode"
  export MOONCAKE_KV_PORT="${DECODE_KV_PORT}"
  local log_path pid
  log_path="$(new_log decode-control)"
  API_HOST=0.0.0.0 API_PORT="${DECODE_API_PORT}" \
    nohup setsid bash "${CONTROL_DECODE_SCRIPT}" >"${log_path}" 2>&1 &
  pid=$!
  printf '%s\n' "${pid}" >"${STATE_ROOT}/decode-control.pid"
  sleep 3
  pid_is_alive "${pid}" || die "Control Decode exited immediately; inspect ${log_path}"
  if is_true "${WAIT_READY}"; then
    wait_http "http://127.0.0.1:${DECODE_API_PORT}/health" "${pid}" \
      || die "Control Decode readiness failed; inspect ${log_path}"
  fi
  if grep -Eq 'AFDDeepseekV4ForCausalLM|P2pHcclAFDConnector|AFD FFN EngineCore' \
    "${log_path}"; then
    die "PD control loaded AFD code; inspect ${log_path}"
  fi
  log "Control Decode started without AFD: pid=${pid}, log=${log_path}"
}

start_afd_decode() {
  export_runtime_env
  export MOONCAKE_ENGINE_ID=dsv4-afd-decode MOONCAKE_KV_PORT="${DECODE_KV_PORT}"
  local ffn_log attention_log ffn_pid attention_pid deadline ready_count
  ffn_log="$(new_log ffn)"
  attention_log="$(new_log attention)"
  API_HOST=0.0.0.0 API_PORT="${FFN_PROCESS_PORT}" \
    nohup setsid bash "${FFN_SCRIPT}" >"${ffn_log}" 2>&1 &
  ffn_pid=$!
  printf '%s\n' "${ffn_pid}" >"${STATE_ROOT}/ffn.pid"
  sleep 2
  ENABLE_PD=1 API_HOST=0.0.0.0 API_PORT="${DECODE_API_PORT}" \
    nohup setsid bash "${ATTENTION_SCRIPT}" >"${attention_log}" 2>&1 &
  attention_pid=$!
  printf '%s\n' "${attention_pid}" >"${STATE_ROOT}/attention.pid"
  sleep 3
  if ! pid_is_alive "${ffn_pid}" || ! pid_is_alive "${attention_pid}"; then
    warn "A Decode role exited immediately"
    stop_action || true
    die "Decode startup failed; inspect ${LOG_ROOT}"
  fi
  if is_true "${WAIT_READY}"; then
    deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
    while (( SECONDS < deadline )); do
      pid_is_alive "${ffn_pid}" && pid_is_alive "${attention_pid}" \
        || die "Decode role exited before readiness"
      ready_count="$( { grep -o 'AFD FFN EngineCore started; workers run connector loop' "${ffn_log}" 2>/dev/null || true; } | wc -l)"
      if (( ready_count >= FFN_RANKS )) \
        && curl -fsS --max-time 5 "http://127.0.0.1:${DECODE_API_PORT}/health" >/dev/null 2>&1; then
        log "Decode ready: Attention health OK, FFN loops=${ready_count}/${FFN_RANKS}"
        return 0
      fi
      sleep 5
    done
    die "Decode readiness timed out; inspect ${LOG_ROOT}"
  fi
  log "Decode started: attention=${attention_pid}, ffn=${ffn_pid}"
}

start_split_ffn() {
  export_runtime_env
  local ffn_log ffn_pid
  ffn_log="$(new_log ffn)"
  API_HOST=0.0.0.0 API_PORT="${FFN_PROCESS_PORT}" \
    nohup setsid bash "${FFN_SCRIPT}" >"${ffn_log}" 2>&1 &
  ffn_pid=$!
  printf '%s\n' "${ffn_pid}" >"${STATE_ROOT}/ffn.pid"
  sleep 3
  pid_is_alive "${ffn_pid}" \
    || die "Split FFN exited immediately; inspect ${ffn_log}"
  log "Split FFN started and is waiting for Attention: pid=${ffn_pid}, log=${ffn_log}"
}

start_split_attention() {
  export_runtime_env
  export MOONCAKE_ENGINE_ID=dsv4-afd-decode MOONCAKE_KV_PORT="${DECODE_KV_PORT}"
  local attention_log attention_pid
  attention_log="$(new_log attention)"
  ENABLE_PD=1 API_HOST=0.0.0.0 API_PORT="${DECODE_API_PORT}" \
    nohup setsid bash "${ATTENTION_SCRIPT}" >"${attention_log}" 2>&1 &
  attention_pid=$!
  printf '%s\n' "${attention_pid}" >"${STATE_ROOT}/attention.pid"
  sleep 3
  pid_is_alive "${attention_pid}" \
    || die "Split Attention exited immediately; inspect ${attention_log}"
  if is_true "${WAIT_READY}"; then
    wait_http "http://127.0.0.1:${DECODE_API_PORT}/health" "${attention_pid}" \
      || die "Split Attention readiness failed; inspect ${attention_log}"
  fi
  log "Split Attention ready: pid=${attention_pid}, ranks=${ATTENTION_RANKS}, log=${attention_log}"
}

start_prefill_ffn() {
  start_prefill
  start_split_ffn
}

start_proxy() {
  export PREFILL_HOSTS="${PREFILL_IP}" PREFILL_PORTS="${PREFILL_API_PORT}"
  export DECODE_HOSTS="${DECODE_IP}" DECODE_PORTS="${DECODE_API_PORT}"
  export PROXY_HOST=0.0.0.0 PROXY_PORT DSV4_RUNTIME_VENV="${VENV_ROOT}"
  export DSV4_VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT}"
  curl --noproxy '*' -fsS --max-time 10 \
    "http://${PREFILL_IP}:${PREFILL_API_PORT}/health" >/dev/null \
    || die "Prefill backend is not healthy"
  curl --noproxy '*' -fsS --max-time 10 \
    "http://${DECODE_IP}:${DECODE_API_PORT}/health" >/dev/null \
    || die "Decode backend is not healthy"
  local log_path pid
  log_path="$(new_log proxy)"
  nohup setsid bash "${PROXY_SCRIPT}" >"${log_path}" 2>&1 &
  pid=$!
  printf '%s\n' "${pid}" >"${STATE_ROOT}/proxy.pid"
  sleep 2
  pid_is_alive "${pid}" || die "Proxy exited immediately; inspect ${log_path}"
  if is_true "${WAIT_READY}"; then
    wait_http "http://127.0.0.1:${PROXY_PORT}/healthcheck" "${pid}" \
      || die "Proxy readiness failed; inspect ${log_path}"
  fi
  log "Proxy started: pid=${pid}, log=${log_path}"
}

status_action() {
  validate_role
  validate_variant
  local overall=0 name pid cmdline log_path ready_count transfer_count
  while read -r name; do
    if pid="$(read_pid "${STATE_ROOT}/${name}.pid" 2>/dev/null)" && pid_is_alive "${pid}"; then
      cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
      log "${name}: RUNNING pid=${pid} command=${cmdline}"
    else
      warn "${name}: NOT RUNNING"
      overall=1
    fi
  done < <(owned_pid_names)
  case "${NODE_ROLE}" in
    prefill)
      curl --noproxy '*' -fsS --max-time 5 \
        "http://127.0.0.1:${PREFILL_API_PORT}/health" >/dev/null 2>&1 \
        && log "Prefill health: OK" || { warn "Prefill health: NOT READY"; overall=1; }
      ;;
    prefill_ffn)
      curl --noproxy '*' -fsS --max-time 5 \
        "http://127.0.0.1:${PREFILL_API_PORT}/health" >/dev/null 2>&1 \
        && log "Prefill health: OK" || { warn "Prefill health: NOT READY"; overall=1; }
      log_path="${LOG_ROOT}/ffn.log"
      ready_count="$( { grep -o 'AFD FFN EngineCore started; workers run connector loop' "${log_path}" 2>/dev/null || true; } | wc -l)"
      log "FFN connector loops: ${ready_count}/${FFN_RANKS}"
      (( ready_count >= FFN_RANKS )) || overall=1
      ;;
    attention)
      curl --noproxy '*' -fsS --max-time 5 \
        "http://127.0.0.1:${DECODE_API_PORT}/health" >/dev/null 2>&1 \
        && log "Decode Attention health: OK" || { warn "Decode Attention health: NOT READY"; overall=1; }
      log_path="${LOG_ROOT}/attention.log"
      grep -Eq 'AFDDeepseekV4ForCausalLM|P2pHcclAFDConnector' "${log_path}" 2>/dev/null \
        || { warn "PD + AFD Attention log has no AFD runtime marker"; overall=1; }
      transfer_count="$( { grep -c 'KV cache transfer for request .* took .* remote_session_id' "${log_path}" 2>/dev/null || true; } )"
      log "Successful Mooncake KV transfer records: ${transfer_count}"
      ;;
    decode)
      curl --noproxy '*' -fsS --max-time 5 \
        "http://127.0.0.1:${DECODE_API_PORT}/health" >/dev/null 2>&1 \
        && log "Decode health: OK" || { warn "Decode health: NOT READY"; overall=1; }
      if [[ "${DEPLOYMENT_VARIANT}" == "pd_control" ]]; then
        log_path="${LOG_ROOT}/decode-control.log"
        if grep -Eq 'AFDDeepseekV4ForCausalLM|P2pHcclAFDConnector|AFD FFN EngineCore' \
          "${log_path}" 2>/dev/null; then
          warn "PD control contains AFD runtime markers"
          overall=1
        fi
      else
        log_path="${LOG_ROOT}/ffn.log"
        ready_count="$( { grep -o 'AFD FFN EngineCore started; workers run connector loop' "${log_path}" 2>/dev/null || true; } | wc -l)"
        log "FFN connector loops: ${ready_count}/${FFN_RANKS}"
        (( ready_count >= FFN_RANKS )) || overall=1
        log_path="${LOG_ROOT}/attention.log"
        grep -Eq 'AFDDeepseekV4ForCausalLM|P2pHcclAFDConnector' "${log_path}" 2>/dev/null \
          || { warn "PD + AFD Attention log has no AFD runtime marker"; overall=1; }
      fi
      transfer_count="$( { grep -c 'KV cache transfer for request .* took .* remote_session_id' "${log_path}" 2>/dev/null || true; } )"
      log "Successful Mooncake KV transfer records: ${transfer_count}"
      ;;
    proxy)
      curl --noproxy '*' -fsS --max-time 5 \
        "http://127.0.0.1:${PROXY_PORT}/healthcheck" >/dev/null 2>&1 \
        && log "Proxy health: OK" || { warn "Proxy health: NOT READY"; overall=1; }
      ;;
  esac
  local fatal=0
  while read -r name; do
    log_path="${LOG_ROOT}/${name}.log"
    [[ -f "${log_path}" ]] || continue
    if grep -En "${FATAL_PATTERN}" "${log_path}"; then
      fatal=1
    fi
  done < <(owned_pid_names)
  (( fatal == 0 )) || { warn "Fatal markers found"; overall=1; }
  return "${overall}"
}

wait_for_group_exit() {
  local pgid="$1"
  local deadline=$((SECONDS + STOP_TIMEOUT_SECONDS))
  while kill -0 -- "-${pgid}" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 2
  done
  ! kill -0 -- "-${pgid}" 2>/dev/null
}

wait_for_profile_raw_started() {
  local name="$1"
  local profile_dir profile_root raw_root
  case "${name}" in
    attention) profile_dir="${AFD_PROFILE_ATTENTION_DIR}" ;;
    ffn) profile_dir="${AFD_PROFILE_FFN_DIR}" ;;
    *) return 0 ;;
  esac

  local deadline=$((SECONDS + AFD_PROFILE_START_TIMEOUT_SECONDS))
  local -a profile_roots=()
  local -a raw_roots=()
  while (( SECONDS < deadline )); do
    profile_roots=()
    while IFS= read -r profile_root; do
      profile_roots+=("${profile_root}")
    done < <(find "${profile_dir}" -mindepth 1 -maxdepth 1 -type d \
      -name '*_ascend_pt' -print 2>/dev/null | sort)
    if (( ${#profile_roots[@]} == 1 )); then
      raw_roots=()
      while IFS= read -r raw_root; do
        raw_roots+=("${raw_root}")
      done < <(find "${profile_roots[0]}" -mindepth 1 -maxdepth 1 -type d \
        -name 'PROF_*' -print 2>/dev/null | sort)
      if (( ${#raw_roots[@]} == 1 )); then
        log "${name} CANN profiler is recording: root=${profile_roots[0]}, raw=${raw_roots[0]}"
        return 0
      fi
    fi
    sleep 1
  done
  die "${name} profile-start did not create exactly one *_ascend_pt/PROF_* raw capture under ${profile_dir}; benchmark was not started"
}

log_profile_raw_diagnostics() {
  local name="$1"
  local profile_dir="$2"
  local -a info_files=()
  local -a raw_roots=()
  local info_file profile_root raw_root
  local data_files data_bytes device_markers host_markers newest_file

  while IFS= read -r info_file; do
    info_files+=("${info_file}")
  done < <(find "${profile_dir}" -type f -name 'profiler_info_*.json' \
    -print 2>/dev/null | sort)
  if (( ${#info_files[@]} == 1 )); then
    profile_root="$(dirname "${info_files[0]}")"
    while IFS= read -r raw_root; do
      raw_roots+=("${raw_root}")
    done < <(find "${profile_root}" -mindepth 1 -maxdepth 1 -type d \
      -name 'PROF_*' -print 2>/dev/null | sort)
  fi

  data_files="$(find "${profile_dir}" -type f \
    -path '*/PROF_*/device_*/data/*' -size +0c \
    -printf '1\n' 2>/dev/null | wc -l)"
  data_bytes="$(find "${profile_dir}" -type f \
    -path '*/PROF_*/device_*/data/*' -size +0c \
    -printf '%s\n' 2>/dev/null | awk '{total += $1} END {print total + 0}')"
  device_markers="$(find "${profile_dir}" -type f \
    -path '*/PROF_*/device_*/*' -name 'end_info*.done' \
    -printf '1\n' 2>/dev/null | wc -l)"
  host_markers="$(find "${profile_dir}" -type f \
    -path '*/PROF_*/host/end_info.done' \
    -printf '1\n' 2>/dev/null | wc -l)"
  newest_file="$(find "${profile_dir}" -type f \
    -printf '%T@ %s %p\n' 2>/dev/null | sort -nr | head -n 1 || true)"
  log "${name} raw CANN state: info_files=${#info_files[@]}, "\
"raw_roots=${#raw_roots[@]}, device_data_files=${data_files}, "\
"device_data_bytes=${data_bytes}, device_end_markers=${device_markers}, "\
"host_end_markers=${host_markers}, newest=${newest_file:-none}"
}

wait_for_profile_raw_completion() {
  local name="$1"
  local required="${2:-0}"
  is_true "${AFD_PROFILE_ENABLE}" || return 0
  local profile_dir info_file profile_root device_marker host_marker device_data_file
  case "${name}" in
    attention) profile_dir="${AFD_PROFILE_ATTENTION_DIR}" ;;
    ffn) profile_dir="${AFD_PROFILE_FFN_DIR}" ;;
    *) return 0 ;;
  esac
  if [[ ! -d "${profile_dir}" ]]; then
    [[ "${required}" == "0" ]] && return 0
    die "${name} profiler directory does not exist: ${profile_dir}"
  fi
  info_file="$(find "${profile_dir}" -type f -name 'profiler_info_*.json' \
    -print -quit 2>/dev/null || true)"
  if [[ -z "${info_file}" ]]; then
    [[ "${required}" == "0" ]] && return 0
    die "${name} profiler did not produce profiler_info_*.json under ${profile_dir}"
  fi

  local deadline=$((SECONDS + AFD_PROFILE_FINALIZE_TIMEOUT_SECONDS))
  local next_diagnostic_at=${SECONDS}
  local -a info_files=()
  while (( SECONDS < deadline )); do
    info_files=()
    while IFS= read -r info_file; do
      info_files+=("${info_file}")
    done < <(find "${profile_dir}" -type f -name 'profiler_info_*.json' \
      -print 2>/dev/null | sort)
    if (( ${#info_files[@]} == 1 )); then
      profile_root="$(dirname "${info_files[0]}")"
      device_marker="$(find "${profile_root}" -type f \
        -path '*/device_*/*' -name 'end_info*.done' \
        -print -quit 2>/dev/null || true)"
      host_marker="$(find "${profile_root}" -type f \
        -path '*/PROF_*/host/end_info.done' \
        -print -quit 2>/dev/null || true)"
      device_data_file="$(find "${profile_root}" -type f \
        -path '*/PROF_*/device_*/data/*' -size +0c \
        -print -quit 2>/dev/null || true)"
      if [[ -n "${device_marker}" && -n "${host_marker}" \
        && -n "${device_data_file}" ]]; then
        log "${name} raw CANN profile finalized: device=${device_marker}, host=${host_marker}"
        return 0
      fi
    fi
    if (( SECONDS >= next_diagnostic_at )); then
      log_profile_raw_diagnostics "${name}" "${profile_dir}"
      next_diagnostic_at=$((SECONDS + 30))
    fi
    sleep 2
  done
  log_profile_raw_diagnostics "${name}" "${profile_dir}"
  die "${name} raw CANN profile did not finalize under ${profile_dir}; expected exactly one profiler root with non-empty device_*/data, device_*/end_info*.done, and host/end_info.done"
}

profile_start_action() {
  validate_common_config
  [[ "${DEPLOYMENT_VARIANT}" == "pd_afd" && "${AFD_PROFILE_ENABLE}" == "1" ]] \
    || die "profile-start requires pd_afd with AFD_PROFILE_ENABLE=1"
  case "${NODE_ROLE}" in
    decode|attention|prefill_ffn) ;;
    *) die "profile-start is valid only for decode, attention, or prefill_ffn" ;;
  esac
  [[ -s "${STATE_ROOT}/profile-session.env" ]] \
    || die "The running service was not started with AFD_PROFILE_ENABLE=1; stop it, enable Profile, and cold-start it before profile-start"
  [[ ! -f "${STATE_ROOT}/profile-started.env" ]] \
    || die "Profile is already started for this service instance"
  [[ ! -f "${STATE_ROOT}/profile-finalized.env" ]] \
    || die "Profile is already finalized for this service instance"

  if [[ "${NODE_ROLE}" == "decode" || "${NODE_ROLE}" == "attention" ]]; then
    require_command curl
    local attention_pid
    attention_pid="$(read_pid "${STATE_ROOT}/attention.pid")" \
      || die "Attention PID file is missing; start the service first"
    pid_is_alive "${attention_pid}" \
      || die "Attention is not running; Profile cannot be started"
    log "Explicitly starting Attention/FFN profilers through plugin API"
    curl --noproxy '*' -fsS --connect-timeout 10 \
      --max-time "${AFD_PROFILE_START_TIMEOUT_SECONDS}" \
      -X POST \
      "http://127.0.0.1:${DECODE_API_PORT}/afd/profile/start" >/dev/null \
      || die "Explicit plugin profile start failed; inspect ${LOG_ROOT}/attention.log"
    wait_for_profile_raw_started attention
    if [[ "${NODE_ROLE}" == "decode" ]]; then
      wait_for_profile_raw_started ffn
    fi
  else
    local ffn_pid
    ffn_pid="$(read_pid "${STATE_ROOT}/ffn.pid")" \
      || die "FFN PID file is missing; start the service first"
    pid_is_alive "${ffn_pid}" \
      || die "FFN is not running; Profile cannot be armed"
    log "Arming split FFN profile state; Attention profile-start sends the start payload"
  fi
  local started_tmp="${STATE_ROOT}/profile-started.env.tmp.$$"
  {
    printf 'started_epoch=%s\n' "$(date +%s)"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'control=%s\n' \
      "$([[ "${NODE_ROLE}" == "prefill_ffn" ]] && printf relayed || printf local-api)"
  } >"${started_tmp}"
  mv "${started_tmp}" "${STATE_ROOT}/profile-started.env"
  log "Plugin Profile window is open; run the benchmark before profile-stop"
}

profile_check_action() {
  validate_common_config
  [[ "${DEPLOYMENT_VARIANT}" == "pd_afd" && "${AFD_PROFILE_ENABLE}" == "1" ]] \
    || die "profile-check requires pd_afd with AFD_PROFILE_ENABLE=1"
  [[ -f "${STATE_ROOT}/profile-started.env" ]] \
    || die "Profile was not armed; run profile-start first"
  local -a profile_roles=()
  case "${NODE_ROLE}" in
    decode) profile_roles=(attention ffn) ;;
    attention) profile_roles=(attention) ;;
    prefill_ffn) profile_roles=(ffn) ;;
    *) die "profile-check is valid only for decode, attention, or prefill_ffn" ;;
  esac
  local name
  for name in "${profile_roles[@]}"; do
    wait_for_profile_raw_started "${name}"
  done
  log "Role-local CANN profiler recording state is ready"
}

profile_stop_action() {
  validate_common_config
  [[ "${DEPLOYMENT_VARIANT}" == "pd_afd" && "${AFD_PROFILE_ENABLE}" == "1" ]] \
    || die "profile-stop requires pd_afd with AFD_PROFILE_ENABLE=1"
  [[ -f "${STATE_ROOT}/profile-started.env" ]] \
    || die "Profile was not started; run profile-start before profile-stop"
  local -a profile_roles=()
  case "${NODE_ROLE}" in
    decode)
      profile_roles=(attention ffn)
      ;;
    attention)
      profile_roles=(attention)
      ;;
    prefill_ffn)
      profile_roles=(ffn)
      ;;
    *) die "profile-stop is valid only for decode, attention, or prefill_ffn" ;;
  esac

  if [[ "${NODE_ROLE}" == "decode" || "${NODE_ROLE}" == "attention" ]]; then
    require_command curl
    local attention_pid
    attention_pid="$(read_pid "${STATE_ROOT}/attention.pid")" \
      || die "Attention PID file is missing; stop Profile before stopping services"
    pid_is_alive "${attention_pid}" \
      || die "Attention is not running; Profile cannot be finalized explicitly"
    log "Explicitly stopping Attention/FFN profilers through plugin API"
    curl --noproxy '*' -fsS --connect-timeout 10 \
      --max-time "${AFD_PROFILE_FINALIZE_TIMEOUT_SECONDS}" \
      -X POST \
      "http://127.0.0.1:${DECODE_API_PORT}/afd/profile/stop" >/dev/null \
      || die "Explicit plugin profile stop failed; inspect ${LOG_ROOT}/attention.log"
  else
    log "Waiting for the Attention-relayed FFN profiler stop"
  fi

  local name
  for name in "${profile_roles[@]}"; do
    wait_for_profile_raw_completion "${name}" 1
  done
  printf 'finalized_at=%s\n' "$(date --iso-8601=seconds)" \
    >"${STATE_ROOT}/profile-finalized.env"
  log "Plugin Profile raw capture finalized; services remain running"
}

stop_name() {
  local name="$1"
  local pid_path="${STATE_ROOT}/${name}.pid"
  local pid pgid
  if ! pid="$(read_pid "${pid_path}" 2>/dev/null)"; then
    log "${name}: no PID file"
    wait_for_profile_raw_completion "${name}"
    return 0
  fi
  if ! pid_is_alive "${pid}"; then
    log "${name}: already stopped"
    rm -f "${pid_path}"
    wait_for_profile_raw_completion "${name}"
    return 0
  fi
  if is_true "${AFD_PROFILE_ENABLE}" \
    && [[ "${name}" == "attention" || "${name}" == "ffn" ]] \
    && [[ ! -f "${STATE_ROOT}/profile-finalized.env" ]]; then
    die "${name} Profile is not finalized; run profile-stop before stop"
  fi
  owned_process "${pid}" || die "Refusing to signal unrecognized PID ${pid}"
  pgid="$(ps -o pgid= -p "${pid}" | tr -d '[:space:]')"
  [[ "${pgid}" == "${pid}" ]] || die "PID ${pid} is not its process-group leader"
  if is_true "${AFD_PROFILE_ENABLE}" \
    && [[ "${name}" == "attention" || "${name}" == "ffn" ]]; then
    log "Gracefully stopping profiled ${name} supervisor ${pid}; process group ${pgid} remains available for supervised worker shutdown"
    kill -TERM "${pid}"
  else
    log "Stopping ${name} process group ${pgid}"
    kill -TERM -- "-${pgid}"
  fi
  if wait_for_group_exit "${pgid}"; then
    rm -f "${pid_path}"
    wait_for_profile_raw_completion "${name}"
    return 0
  fi
  if is_true "${FORCE_KILL}"; then
    warn "TERM timed out for ${name}; FORCE_KILL=1"
    kill -KILL -- "-${pgid}"
    wait_for_group_exit "${pgid}" || die "Process group ${pgid} did not exit"
    rm -f "${pid_path}"
    return 0
  fi
  die "${name} did not exit; inspect it before setting FORCE_KILL=1"
}

stop_action() {
  validate_role
  validate_variant
  mkdir -p "${STATE_ROOT}"
  case "${NODE_ROLE}" in
    proxy) stop_name proxy ;;
    attention) stop_name attention ;;
    prefill_ffn)
      stop_name ffn
      stop_name prefill
      ;;
    decode)
      if [[ "${DEPLOYMENT_VARIANT}" == "pd_control" ]]; then
        stop_name decode-control
      else
        stop_name attention
        stop_name ffn
      fi
      ;;
    prefill) stop_name prefill ;;
  esac
  log "Stop complete; run npu-smi info on NPU nodes"
}

install_action() {
  validate_common_config
  [[ "${NODE_ROLE}" == "proxy" ]] && { log "Proxy needs no Mooncake install"; return 0; }
  validate_mooncake_install_mode
  if is_true "${INSTALL_SYSTEM_PACKAGES}"; then
    install_system_packages
  fi
  AFD_BUILD_ASCEND_OPS=0 "${PYTHON_BIN}" -m pip install \
    --no-build-isolation --no-deps --editable "${AFD_PLUGIN_ROOT}"
  if [[ "${MOONCAKE_INSTALL_MODE}" == "wheel" ]]; then
    validate_wheel
    "${PYTHON_BIN}" -m pip install --no-deps --force-reinstall "${MOONCAKE_WHEEL}"
  else
    log "Keeping Mooncake already installed in the configured Python environment"
  fi
  validate_installed_mooncake
  "${PYTHON_BIN}" -m pip show vllm vllm-ascend vllm-afd-plugin mooncake-transfer-engine
  log "Install complete; next run: bash $0 check ${CONFIG_FILE}"
}

check_action() {
  validate_common_config
  validate_local_network
  require_command curl
  mkdir -p "${STATE_ROOT}" "${LOG_ROOT}" "${VALIDATION_ROOT}" "${OUTPUT_ROOT}"
  if [[ "${NODE_ROLE}" == "proxy" ]]; then
    require_file "${PROXY_SCRIPT}"
    require_file "${GOLDEN_VALIDATOR}"
    require_file "${GOLDEN_GENERATOR}"
    ensure_not_running
    check_start_ports
    log "Proxy preflight passed"
    return 0
  fi
  require_file "${CANN_ROOT}/set_env.sh"
  validate_cann_version
  resolve_atb_root
  require_file "${MODEL_PATH}/config.json"
  require_file "${RUNTIME_CHECK}"
  require_file "${ROUNDTRIP_TOOL}"
  validate_mooncake_install_mode
  if [[ "${MOONCAKE_INSTALL_MODE}" == "wheel" ]]; then
    validate_wheel
  fi
  validate_installed_mooncake
  check_npus
  ensure_not_running
  check_start_ports
  export DSV4_CANN_ROOT="${CANN_ROOT}" DSV4_CANN_VERSION="${CANN_VERSION}"
  export DSV4_ATB_ROOT="${ATB_ROOT}"
  export DSV4_RUNTIME_VENV="${VENV_ROOT}"
  export DSV4_VLLM_ROOT="${VLLM_ROOT}" DSV4_VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT}"
  export_batch_invariant_env
  bash "${RUNTIME_CHECK}" >"${STATE_ROOT}/runtime-check.log" 2>&1 \
    || { tail -n 100 "${STATE_ROOT}/runtime-check.log" >&2; die "Mooncake runtime check failed"; }
  if [[ "${ENABLE_BATCH_INVARIANT}" == "1" ]]; then
    bash -c 'source "$1"; exec python -c '\''import batch_invariant_ops; import vllm_ascend.batch_invariant as bi; assert bi.HAS_ASCENDC_BATCH_INVARIANT; print("batch-invariant backend: OK")'\''' \
      batch-invariant-check "${AFD_PLUGIN_ROOT}/tools/dsv4/activate_v023_vllm_cann_runtime.sh" \
      >"${STATE_ROOT}/batch-invariant-check.log" 2>&1 \
      || { tail -n 100 "${STATE_ROOT}/batch-invariant-check.log" >&2; die "Batch-invariant runtime check failed"; }
  fi
  write_mooncake_fingerprint "${STATE_ROOT}/mooncake-libraries.sha256"
  if is_true "${RUN_LOCAL_ROUNDTRIP}"; then
    if ! bash -c 'source "$1"; exec python "$2" --producer-device "$3" --consumer-device "$4" --host "$5" --interface "$6"' \
      pd-roundtrip "${RUNTIME_CHECK}" "${ROUNDTRIP_TOOL}" \
      "${ROUNDTRIP_PRODUCER_DEVICE}" "${ROUNDTRIP_CONSUMER_DEVICE}" \
      "$(local_role_ip)" "${NIC_NAME}" \
      >"${STATE_ROOT}/roundtrip.json" 2>"${STATE_ROOT}/roundtrip.stderr"; then
      tail -n 100 "${STATE_ROOT}/roundtrip.stderr" >&2 || true
      die "Mooncake local NPU round-trip failed; inspect ${STATE_ROOT}/roundtrip.stderr"
    fi
  fi
  log "Preflight passed; runtime=${STATE_ROOT}/runtime-check.log"
}

start_action() {
  local configured_roundtrip="${RUN_LOCAL_ROUNDTRIP}"
  RUN_LOCAL_ROUNDTRIP=0
  check_action
  RUN_LOCAL_ROUNDTRIP="${configured_roundtrip}"
  mkdir -p "${STATE_ROOT}" "${LOG_ROOT}"
  ensure_not_running
  if [[ -f "${STATE_ROOT}/npu-monitor.pid" ]]; then
    local monitor_pid
    read -r monitor_pid <"${STATE_ROOT}/npu-monitor.pid"
    if [[ "${monitor_pid}" =~ ^[0-9]+$ ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
      die "NPU monitor PID ${monitor_pid} is still live; stop it before restarting the service"
    fi
    rm -f "${STATE_ROOT}/npu-monitor.pid"
  fi
  # Pointers are scoped to this cold-started service instance.  This prevents
  # a skipped P2 monitor or benchmark from silently reusing P1 evidence.
  rm -f "${STATE_ROOT}/last-monitor-dir" \
    "${STATE_ROOT}/last-validation-dir" \
    "${STATE_ROOT}/collect-final-gates.txt"
  if [[ "${NODE_ROLE}" == "proxy" ]]; then
    rm -f "${STATE_ROOT}/last-performance-attempt-dir" \
      "${STATE_ROOT}/last-performance-dir"
  fi
  if [[ ("${NODE_ROLE}" == "decode" || "${NODE_ROLE}" == "attention" \
      || "${NODE_ROLE}" == "prefill_ffn") \
    && "${DEPLOYMENT_VARIANT}" == "pd_afd" ]]; then
    rm -f "${STATE_ROOT}/stage-evidence.env" \
      "${STATE_ROOT}/profile-summary.json" \
      "${STATE_ROOT}/profile-session.env" \
      "${STATE_ROOT}/profile-started.env" \
      "${STATE_ROOT}/profile-finalized.env"
    if is_true "${AFD_PROFILE_ENABLE}"; then
      local profile_file
      local profile_dirs=()
      case "${NODE_ROLE}" in
        decode) profile_dirs=("${AFD_PROFILE_ATTENTION_DIR}" "${AFD_PROFILE_FFN_DIR}") ;;
        attention) profile_dirs=("${AFD_PROFILE_ATTENTION_DIR}") ;;
        prefill_ffn) profile_dirs=("${AFD_PROFILE_FFN_DIR}") ;;
      esac
      for profile_file in "${profile_dirs[@]}"; do
        if [[ -d "${profile_file}" ]] \
          && find "${profile_file}" -type f -print -quit | grep -q .; then
          die "Profiler output directory is not empty; use a fresh RUN_ROOT: ${profile_file}"
        fi
      done
      {
        printf 'matrix_point=%s\n' "${MATRIX_POINT:-not-matrix}"
        printf 'started_epoch=%s\n' "$(date +%s)"
        printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
        printf 'cann_version=%s\n' "${CANN_VERSION}"
        printf 'profile_mode=manual-window\n'
        printf 'online_analysis=0\n'
        printf 'finalize_timeout_seconds=%s\n' \
          "${AFD_PROFILE_FINALIZE_TIMEOUT_SECONDS}"
        printf 'vllm_shutdown_timeout_seconds=%s\n' \
          "${AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS}"
        printf 'attention_dir=%s\n' "${AFD_PROFILE_ATTENTION_DIR}"
        printf 'ffn_dir=%s\n' "${AFD_PROFILE_FFN_DIR}"
      } >"${STATE_ROOT}/profile-session.env"
    fi
  fi
  case "${NODE_ROLE}" in
    prefill) require_file "${PREFILL_SCRIPT}"; start_prefill ;;
    prefill_ffn)
      require_file "${PREFILL_SCRIPT}"
      require_file "${FFN_SCRIPT}"
      start_prefill_ffn
      ;;
    attention)
      require_file "${ATTENTION_SCRIPT}"
      start_split_attention
      ;;
    decode)
      if [[ "${DEPLOYMENT_VARIANT}" == "pd_control" ]]; then
        require_file "${CONTROL_DECODE_SCRIPT}"
        start_control_decode
      else
        require_file "${ATTENTION_SCRIPT}"
        require_file "${FFN_SCRIPT}"
        start_afd_decode
      fi
      ;;
    proxy) require_file "${PROXY_SCRIPT}"; start_proxy ;;
  esac
}

control_golden_metadata_args() {
  printf '%s\n' \
    "baseline_kind=mooncake_pd_no_afd" \
    "deployment_variant=pd_control" \
    "cann_root=$(readlink -f "${CANN_ROOT}" 2>/dev/null || printf '%s' "${CANN_ROOT}")" \
    "cann_version=${CANN_VERSION}" \
    "vllm_commit=${VLLM_COMMIT}" \
    "vllm_ascend_commit=${VLLM_ASCEND_COMMIT}" \
    "afd_commit=${AFD_PD_COMMIT}" \
    "model_path=${MODEL_PATH}" \
    "prefill_dp_size=${PREFILL_DP_SIZE}" \
    "prefill_tp_size=${PREFILL_TP_SIZE}" \
    "decode_dp_size=${DECODE_DP_SIZE}" \
    "decode_tp_size=${DECODE_TP_SIZE}" \
    "execution_mode=${DECODE_EXECUTION_MODE}" \
    "u_batches=${DECODE_U_BATCHES}" \
    "mtp=${DECODE_ENABLE_MTP}" \
    "mtp_draft_execution=${DECODE_MTP_DRAFT_EXECUTION}" \
    "mtp_num_speculative_tokens=${DECODE_MTP_NUM_SPECULATIVE_TOKENS}" \
    "batch_invariant=${ENABLE_BATCH_INVARIANT}" \
    "mooncake_version=${MOONCAKE_VERSION}"
}

record_control_action() {
  [[ "${NODE_ROLE}" == "proxy" ]] \
    || die "record-control must run with NODE_ROLE=proxy"
  [[ "${DEPLOYMENT_VARIANT}" == "pd_control" ]] \
    || die "record-control requires DEPLOYMENT_VARIANT=pd_control"
  validate_common_config
  status_action
  require_file "${NATIVE_GOLDEN_PATH}"
  require_file "${GOLDEN_GENERATOR}"
  local run_dir endpoint output_path reference_comparison
  local metadata_args=()
  run_dir="${VALIDATION_ROOT}/pd-control-$(date +%Y%m%d_%H%M%S)"
  output_path="${run_dir}/golden_results.json"
  endpoint="http://127.0.0.1:${PROXY_PORT}/v1/completions"
  mkdir -p "${run_dir}" "$(dirname "${PD_CONTROL_GOLDEN_PATH}")" "${STATE_ROOT}"
  while IFS= read -r metadata_entry; do
    metadata_args+=(--metadata "${metadata_entry}")
  done < <(control_golden_metadata_args)
  if ! "${PYTHON_BIN}" "${GOLDEN_GENERATOR}" \
    --endpoint "${endpoint}" --model "${MODEL_NAME}" \
    --prompt-source "${NATIVE_GOLDEN_PATH}" --output "${output_path}" \
    --rounds "${VALIDATION_ROUNDS}" "${metadata_args[@]}"; then
    die "PD control was not stable; inspect ${output_path}"
  fi
  cp "${output_path}" "${PD_CONTROL_GOLDEN_PATH}"
  reference_comparison="$("${PYTHON_BIN}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); c=d["reference_comparison"]; print("{}/{}".format(c["exact_match_count"], c["request_count"]))' \
    "${output_path}")"
  {
    printf 'status=pd_control_golden_recorded\n'
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'pd_control_golden=%s\n' "${PD_CONTROL_GOLDEN_PATH}"
    printf 'native_vs_pd_control=%s\n' "${reference_comparison}"
  } >"${run_dir}/summary.env"
  printf '%s\n' "${run_dir}" >"${STATE_ROOT}/last-validation-dir"
  log "PD control golden recorded: ${PD_CONTROL_GOLDEN_PATH}"
  log "Native semantic reference vs PD control: ${reference_comparison} (informational)"
}

functional_smoke_action() {
  [[ "${NODE_ROLE}" == "proxy" ]] || die "smoke must run with NODE_ROLE=proxy"
  validate_common_config
  status_action
  require_file "${FUNCTIONAL_SMOKE_TOOL}"
  local run_dir endpoint cancel_rc
  run_dir="${VALIDATION_ROOT}/f0-functional-$(date +%Y%m%d_%H%M%S)"
  endpoint="http://127.0.0.1:${PROXY_PORT}/v1/completions"
  mkdir -p "${run_dir}" "${STATE_ROOT}"
  "${PYTHON_BIN}" "${FUNCTIONAL_SMOKE_TOOL}" \
    --endpoint "${endpoint}" --model "${MODEL_NAME}" \
    --batch-sizes "${VALIDATION_BATCH_SIZES}" \
    --output "${run_dir}/batches.json"
  if is_true "${RUN_CANCELLATION_TEST}"; then
    set +e
    curl -fsS --max-time 1 "${endpoint}" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${MODEL_NAME}\",\"prompt\":\"Write a detailed deterministic systems validation checklist.\",\"temperature\":0,\"seed\":1024,\"max_tokens\":512,\"stream\":false}" \
      >"${run_dir}/cancellation-response.json" 2>"${run_dir}/cancellation.stderr"
    cancel_rc=$?
    set -e
    printf '%s\n' "${cancel_rc}" >"${run_dir}/cancellation.exitcode"
    [[ "${cancel_rc}" == "28" ]] \
      || die "Cancellation gate expected curl exit 28, got ${cancel_rc}"
    "${PYTHON_BIN}" "${FUNCTIONAL_SMOKE_TOOL}" \
      --endpoint "${endpoint}" --model "${MODEL_NAME}" --batch-sizes "1" \
      --output "${run_dir}/recovery.json"
  fi
  status_action
  {
    printf 'status=f0_functional_smoke_passed_no_golden\n'
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'deployment_variant=%s\n' "${DEPLOYMENT_VARIANT}"
    printf 'decode_topology=DP%s/TP%s\n' "${DECODE_DP_SIZE}" "${DECODE_TP_SIZE}"
    printf 'execution_mode=%s\n' "${DECODE_EXECUTION_MODE}"
    printf 'u_batches=%s\n' "${DECODE_U_BATCHES}"
    printf 'mtp=%s\n' "${DECODE_ENABLE_MTP}"
    printf 'afd_profile=%s\n' "${AFD_PROFILE_ENABLE}"
    printf 'afd_profile_start_timeout_seconds=%s\n' \
      "${AFD_PROFILE_START_TIMEOUT_SECONDS}"
    printf 'afd_profile_attention_dir=%s\n' "${AFD_PROFILE_ATTENTION_DIR}"
    printf 'afd_profile_ffn_dir=%s\n' "${AFD_PROFILE_FFN_DIR}"
    printf 'mtp_draft_execution=%s\n' "${DECODE_MTP_DRAFT_EXECUTION}"
    printf 'batch_invariant=%s\n' "${ENABLE_BATCH_INVARIANT}"
    printf 'golden_checked=0\n'
  } >"${run_dir}/summary.env"
  printf '%s\n' "${run_dir}" >"${STATE_ROOT}/last-validation-dir"
  log "F0 functional smoke passed without golden: ${run_dir}"
}

validate_control_golden() {
  require_file "${PD_CONTROL_GOLDEN_PATH}"
  local expected_metadata=()
  while IFS= read -r metadata_entry; do
    expected_metadata+=("${metadata_entry}")
  done < <(control_golden_metadata_args)
  "${PYTHON_BIN}" -c '
import json
import sys

path = sys.argv[1]
rounds = int(sys.argv[2])
payload = json.load(open(path, encoding="utf-8"))
if payload.get("passed") is not True:
    raise SystemExit("PD control golden is not internally stable")
if payload.get("rounds") != rounds or payload.get("prompt_count") != 10:
    raise SystemExit("PD control golden must contain 10 prompts for the configured rounds")
expected = dict(item.split("=", 1) for item in sys.argv[3:])
actual = payload.get("metadata", {})
mismatched = {
    key: {"expected": value, "actual": actual.get(key)}
    for key, value in expected.items()
    if actual.get(key) != value
}
if mismatched:
    raise SystemExit(f"PD control golden metadata mismatch: {mismatched}")
sampling = payload.get("sampling", {})
if sampling != {"temperature": 0.0, "top_p": 1.0, "max_tokens": 16, "seed": 1024}:
    raise SystemExit(f"PD control golden sampling mismatch: {sampling}")
' "${PD_CONTROL_GOLDEN_PATH}" "${VALIDATION_ROUNDS}" "${expected_metadata[@]}"
}

validate_action() {
  [[ "${NODE_ROLE}" == "proxy" ]] || die "validate must run with NODE_ROLE=proxy"
  [[ "${DEPLOYMENT_VARIANT}" == "pd_afd" ]] \
    || die "validate requires DEPLOYMENT_VARIANT=pd_afd; use record-control for pd_control"
  validate_common_config
  status_action
  validate_control_golden
  require_file "${GOLDEN_VALIDATOR}"
  local run_dir endpoint cancel_rc
  run_dir="${VALIDATION_ROOT}/pd-afd-f0-$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${run_dir}" "${STATE_ROOT}"
  endpoint="http://127.0.0.1:${PROXY_PORT}/v1/completions"
  curl -fsS --max-time 600 "${endpoint}" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL_NAME}\",\"prompt\":\"PD smoke request.\",\"temperature\":0,\"seed\":1024,\"max_tokens\":16,\"stream\":false,\"return_token_ids\":true}" \
    >"${run_dir}/smoke.json"
  read -r -a batch_sizes <<<"${VALIDATION_BATCH_SIZES}"
  "${PYTHON_BIN}" "${GOLDEN_VALIDATOR}" \
    --endpoint "${endpoint}" --model "${MODEL_NAME}" \
    --golden "${PD_CONTROL_GOLDEN_PATH}" --rounds "${VALIDATION_ROUNDS}" \
    --batch-sizes "${batch_sizes[@]}" --output "${run_dir}/golden.json"
  if is_true "${RUN_CANCELLATION_TEST}"; then
    set +e
    curl -fsS --max-time 1 "${endpoint}" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"${MODEL_NAME}\",\"prompt\":\"Write a detailed deterministic systems validation checklist.\",\"temperature\":0,\"seed\":1024,\"max_tokens\":512,\"stream\":false}" \
      >"${run_dir}/cancellation-response.json" 2>"${run_dir}/cancellation.stderr"
    cancel_rc=$?
    set -e
    printf '%s\n' "${cancel_rc}" >"${run_dir}/cancellation.exitcode"
    [[ "${cancel_rc}" == "28" ]] || die "Cancellation gate expected curl exit 28, got ${cancel_rc}"
    curl -fsS --max-time 10 "http://127.0.0.1:${PROXY_PORT}/healthcheck" \
      >"${run_dir}/health-after-cancellation.json"
  fi
  {
    printf 'status=pd_control_vs_pd_afd_passed_decode_transfer_evidence_required\n'
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'comparison=pd_control_vs_pd_afd\n'
    printf 'matched_requests=%s\n' "$((VALIDATION_ROUNDS * 10))"
    printf 'golden=%s\n' "${run_dir}/golden.json"
  } >"${run_dir}/summary.env"
  printf '%s\n' "${run_dir}" >"${STATE_ROOT}/last-validation-dir"
  log "Path-matched PD control vs PD + AFD validation passed: ${run_dir}"
  log "Run collect on proxy, decode, and prefill; Decode artifact must contain KV transfer evidence"
}

copy_log_tail() {
  local source_path="$1"
  local output_path="$2"
  [[ -f "${source_path}" ]] || return 0
  tail -c "${ARTIFACT_LOG_TAIL_BYTES}" "${source_path}" >"${output_path}"
}

write_raw_tcp_tables() {
  local output_path="$1"
  {
    printf '# socket utility unavailable or timed out; raw Linux TCP tables\n'
    printf '\n# /proc/net/tcp\n'
    cat /proc/net/tcp
    printf '\n# /proc/net/tcp6\n'
    cat /proc/net/tcp6
  } >"${output_path}" 2>&1 || true
}

capture_port_snapshot() {
  local output_path="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp >"${output_path}" 2>&1 || true
  elif command -v netstat >/dev/null 2>&1 \
    && command -v timeout >/dev/null 2>&1; then
    if timeout "${PORT_SNAPSHOT_TIMEOUT_SECONDS}" netstat -lntp \
      >"${output_path}" 2>&1; then
      return 0
    fi
    warn "netstat port snapshot timed out; using /proc/net/tcp*"
    write_raw_tcp_tables "${output_path}"
  else
    write_raw_tcp_tables "${output_path}"
  fi
}

collect_action() {
  validate_role
  validate_variant
  require_command tar
  require_command sha256sum
  mkdir -p "${STATE_ROOT}" "${OUTPUT_ROOT}"
  local temp_dir archive artifact_slug timestamp size_bytes status_rc name log_path validation_dir
  temp_dir="$(mktemp -d "${STATE_ROOT}/collect.XXXXXX")"
  COLLECT_TEMP_DIR="${temp_dir}"
  trap '[[ -z "${COLLECT_TEMP_DIR:-}" ]] || rm -rf -- "${COLLECT_TEMP_DIR}"' EXIT
  timestamp="$(date +%Y%m%d_%H%M%S)"
  artifact_slug="${MATRIX_POINT:-${DEPLOYMENT_SLUG}}"
  [[ "${artifact_slug}" =~ ^[a-z0-9_-]+$ ]] \
    || die "Invalid artifact slug: ${artifact_slug}"
  archive="${OUTPUT_ROOT}/dsv4-m9-${artifact_slug}-${NODE_ROLE}-${timestamp}.tar.gz"
  {
    printf 'deployment_variant=%s\n' "${DEPLOYMENT_VARIANT}"
    printf 'matrix_point=%s\n' "${MATRIX_POINT:-not-matrix}"
    printf 'node_role=%s\n' "${NODE_ROLE}"
    printf 'collected_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'hostname=%s\n' "$(resolve_hostname)"
    printf 'prefill_ip=%s\n' "${PREFILL_IP}"
    printf 'decode_ip=%s\n' "${DECODE_IP}"
    printf 'nic_name=%s\n' "${NIC_NAME}"
    printf 'cann_root=%s\n' "$(readlink -f "${CANN_ROOT}" 2>/dev/null || printf '%s' "${CANN_ROOT}")"
    printf 'vllm_commit=%s\n' "$(git_head "${VLLM_ROOT}" 2>/dev/null || true)"
    printf 'vllm_ascend_commit=%s\n' "$(git_head "${VLLM_ASCEND_ROOT}" 2>/dev/null || true)"
    printf 'afd_commit=%s\n' "$(git_head "${AFD_PLUGIN_ROOT}" 2>/dev/null || true)"
    printf 'mooncake_install_mode=%s\n' "${MOONCAKE_INSTALL_MODE}"
    printf 'mooncake_version=%s\n' "${MOONCAKE_VERSION}"
    printf 'batch_invariant=%s\n' "${ENABLE_BATCH_INVARIANT}"
    printf 'decode_topology=DP%s/TP%s\n' "${DECODE_DP_SIZE}" "${DECODE_TP_SIZE}"
    printf 'afd_placement=%s\n' "${AFD_PLACEMENT}"
    printf 'afd_ranks=A%s/F%s\n' "${ATTENTION_RANKS}" "${FFN_RANKS}"
    printf 'attention_max_num_batched_tokens=%s\n' \
      "${ATTENTION_MAX_NUM_BATCHED_TOKENS}"
    printf 'ffn_max_num_batched_tokens=%s\n' "${FFN_MAX_NUM_BATCHED_TOKENS}"
    printf 'execution_mode=%s\n' "${DECODE_EXECUTION_MODE}"
    printf 'u_batches=%s\n' "${DECODE_U_BATCHES}"
    printf 'mtp=%s\n' "${DECODE_ENABLE_MTP}"
    printf 'afd_profile=%s\n' "${AFD_PROFILE_ENABLE}"
    printf 'afd_profile_attention_dir=%s\n' "${AFD_PROFILE_ATTENTION_DIR}"
    printf 'afd_profile_ffn_dir=%s\n' "${AFD_PROFILE_FFN_DIR}"
    printf 'graph_u2_compute_overlap=%s\n' \
      "${AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP}"
    printf 'graph_u2_hybrid_dag=%s\n' "${AFD_HCCL_GRAPH_U2_HYBRID_DAG}"
    printf 'graph_u2_attention_three_stream=%s\n' \
      "${AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM}"
    printf 'graph_u2_ffn_recv_stream=%s\n' \
      "${AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM}"
    printf 'graph_u2_ffn_cross_layer=%s\n' \
      "${AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER}"
    printf 'vllm_ascend_worktree_mode=%s\n' "${VLLM_ASCEND_WORKTREE_MODE}"
    if [[ "${MOONCAKE_INSTALL_MODE}" == "wheel" ]]; then
      printf 'mooncake_wheel_sha256=%s\n' "${MOONCAKE_WHEEL_SHA256}"
    else
      printf 'mooncake_wheel_sha256=not-used\n'
    fi
    if [[ -f "${STATE_ROOT}/mooncake-libraries.sha256" ]]; then
      printf 'mooncake_library_set_sha256=%s\n' \
        "$(sha256sum "${STATE_ROOT}/mooncake-libraries.sha256" | awk '{print $1}')"
    fi
  } >"${temp_dir}/summary.env"
  {
    git -c safe.directory="${VLLM_ROOT}" -C "${VLLM_ROOT}" status --short 2>/dev/null || true
    git -c safe.directory="${VLLM_ASCEND_ROOT}" -C "${VLLM_ASCEND_ROOT}" status --short 2>/dev/null || true
    git -c safe.directory="${AFD_PLUGIN_ROOT}" -C "${AFD_PLUGIN_ROOT}" status --short 2>/dev/null || true
  } >"${temp_dir}/git-status.txt"
  "${PYTHON_BIN}" -m pip show torch torch-npu vllm vllm-ascend vllm-afd-plugin mooncake-transfer-engine \
    >"${temp_dir}/python-packages.txt" 2>&1 || true
  if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info >"${temp_dir}/npu-smi.txt" 2>&1 || true
  fi
  capture_port_snapshot "${temp_dir}/ports.txt"
  set +e
  status_action >"${temp_dir}/status.txt" 2>&1
  status_rc=$?
  set -e
  printf '%s\n' "${status_rc}" >"${temp_dir}/status.exitcode"
  for name in prefill decode-control attention ffn proxy; do
    log_path="${LOG_ROOT}/${name}.log"
    copy_log_tail "${log_path}" "${temp_dir}/${name}.tail.log"
  done
  cp "${STATE_ROOT}/runtime-check.log" "${temp_dir}/" 2>/dev/null || true
  cp "${STATE_ROOT}/batch-invariant-check.log" "${temp_dir}/" 2>/dev/null || true
  cp "${STATE_ROOT}/mooncake-libraries.sha256" "${temp_dir}/" 2>/dev/null || true
  cp "${STATE_ROOT}/roundtrip.json" "${temp_dir}/" 2>/dev/null || true
  cp "${STATE_ROOT}/roundtrip.stderr" "${temp_dir}/" 2>/dev/null || true
  cp "${STATE_ROOT}/stage-evidence.env" "${temp_dir}/" 2>/dev/null || true
  if [[ -f "${STATE_ROOT}/last-monitor-dir" ]]; then
    local monitor_dir
    read -r monitor_dir <"${STATE_ROOT}/last-monitor-dir"
    case "${monitor_dir}" in
      "${RUN_ROOT}"/*)
        cp "${monitor_dir}/summary.json" "${temp_dir}/npu-monitor-summary.json" \
          2>/dev/null || true
        cp "${monitor_dir}/preflight.json" "${temp_dir}/npu-monitor-preflight.json" \
          2>/dev/null || true
        ;;
      *) warn "Ignoring monitor path outside RUN_ROOT: ${monitor_dir}" ;;
    esac
  fi
  cp "${STATE_ROOT}/profile-summary.json" "${temp_dir}/" 2>/dev/null || true
  cp "${STATE_ROOT}/profile-session.env" "${temp_dir}/" 2>/dev/null || true
  cp "${STATE_ROOT}/profile-started.env" "${temp_dir}/" 2>/dev/null || true
  cp "${STATE_ROOT}/profile-finalized.env" "${temp_dir}/" 2>/dev/null || true
  cp "${STATE_ROOT}/collect-final-gates.txt" "${temp_dir}/" 2>/dev/null || true
  if [[ -f "${STATE_ROOT}/last-validation-dir" ]]; then
    read -r validation_dir <"${STATE_ROOT}/last-validation-dir"
    case "${validation_dir}" in
      "${VALIDATION_ROOT}"/*)
        for name in batches.json recovery.json smoke.json golden.json golden_results.json performance_summary.json cancellation.exitcode cancellation.stderr health-after-cancellation.json summary.env; do
          [[ -f "${validation_dir}/${name}" ]] && cp "${validation_dir}/${name}" "${temp_dir}/validation-${name}"
        done
        for result_path in "${validation_dir}"/c*.json; do
          [[ -f "${result_path}" ]] \
            && cp "${result_path}" "${temp_dir}/validation-$(basename "${result_path}")"
        done
        ;;
      *) warn "Ignoring validation path outside VALIDATION_ROOT: ${validation_dir}" ;;
    esac
  fi
  if [[ -f "${STATE_ROOT}/last-profile-dir" ]]; then
    local profile_run_dir
    read -r profile_run_dir <"${STATE_ROOT}/last-profile-dir"
    case "${profile_run_dir}" in
      "${VALIDATION_ROOT}"/*)
        cp "${profile_run_dir}/performance_summary.json" \
          "${temp_dir}/profile-workload-summary.json" 2>/dev/null || true
        ;;
      *) warn "Ignoring profile workload path outside VALIDATION_ROOT: ${profile_run_dir}" ;;
    esac
  fi
  local transfer_log="${LOG_ROOT}/attention.log"
  if [[ "${NODE_ROLE}" == "decode" && "${DEPLOYMENT_VARIANT}" == "pd_control" ]]; then
    transfer_log="${LOG_ROOT}/decode-control.log"
  elif [[ "${NODE_ROLE}" == "prefill" ]]; then
    transfer_log="${LOG_ROOT}/prefill.log"
  fi
  { grep -Eh 'KV cache transfer for request .* took .* remote_session_id' \
      "${transfer_log}" 2>/dev/null || true; } \
    | tail -n 50 >"${temp_dir}/kv-transfer-evidence.txt"
  {
    while read -r name; do
      log_path="${LOG_ROOT}/${name}.log"
      [[ -f "${log_path}" ]] || continue
      grep -EHn "${FATAL_PATTERN}" "${log_path}" 2>/dev/null || true
    done < <(owned_pid_names)
  } | tail -n 200 >"${temp_dir}/fatal-markers.txt"
  find "${temp_dir}" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort >"${temp_dir}/manifest.txt"
  tar -czf "${archive}" -C "${temp_dir}" .
  size_bytes="$(stat -c '%s' "${archive}")"
  if (( size_bytes > ARTIFACT_MAX_BYTES )); then
    rm -f "${archive}"
    die "Artifact exceeded ${ARTIFACT_MAX_BYTES} bytes; lower ARTIFACT_LOG_TAIL_BYTES"
  fi
  (
    cd "$(dirname "${archive}")"
    sha256sum "$(basename "${archive}")"
  ) >"${archive}.sha256"
  rm -rf -- "${temp_dir}"
  COLLECT_TEMP_DIR=""
  trap - EXIT
  log "ARTIFACT=${archive}"
  log "ARTIFACT_SIZE_BYTES=${size_bytes}"
  log "SHA256_FILE=${archive}.sha256"
}

print_config_action() {
  validate_role
  validate_variant
  printf 'DEPLOYMENT_VARIANT=%s\n' "${DEPLOYMENT_VARIANT}"
  printf 'MATRIX_POINT=%s\n' "${MATRIX_POINT:-not-matrix}"
  printf 'NODE_ROLE=%s\n' "${NODE_ROLE}"
  printf 'PREFILL_IP=%s\n' "${PREFILL_IP:-}"
  printf 'DECODE_IP=%s\n' "${DECODE_IP:-}"
  printf 'NIC_NAME=%s\n' "${NIC_NAME:-}"
  printf 'AFD_PD_COMMIT=%s\n' "${AFD_PD_COMMIT:-}"
  printf 'MOONCAKE_INSTALL_MODE=%s\n' "${MOONCAKE_INSTALL_MODE}"
  printf 'MOONCAKE_VERSION=%s\n' "${MOONCAKE_VERSION}"
  printf 'MOONCAKE_LIBRARY_DIR=%s\n' "${MOONCAKE_LIBRARY_DIR:-auto}"
  printf 'CANN_ROOT=%s\n' "${CANN_ROOT}"
  printf 'CANN_VERSION=%s\n' "${CANN_VERSION}"
  printf 'ATB_ROOT=%s\n' "${ATB_ROOT:-auto}"
  printf 'VENV_ROOT=%s\n' "${VENV_ROOT}"
  printf 'MODEL_PATH=%s\n' "${MODEL_PATH}"
  printf 'NATIVE_GOLDEN_PATH=%s\n' "${NATIVE_GOLDEN_PATH}"
  printf 'PD_CONTROL_GOLDEN_PATH=%s\n' "${PD_CONTROL_GOLDEN_PATH}"
  printf 'DECODE_TOPOLOGY=DP%s/TP%s\n' "${DECODE_DP_SIZE}" "${DECODE_TP_SIZE}"
  printf 'AFD_PLACEMENT=%s\n' "${AFD_PLACEMENT}"
  printf 'AFD_RANKS=A%s/F%s\n' "${ATTENTION_RANKS}" "${FFN_RANKS}"
  printf 'ATTENTION_MAX_NUM_BATCHED_TOKENS=%s\n' "${ATTENTION_MAX_NUM_BATCHED_TOKENS}"
  printf 'FFN_MAX_NUM_BATCHED_TOKENS=%s\n' "${FFN_MAX_NUM_BATCHED_TOKENS}"
  printf 'DECODE_EXECUTION_MODE=%s\n' "${DECODE_EXECUTION_MODE}"
  printf 'DECODE_U_BATCHES=%s\n' "${DECODE_U_BATCHES}"
  printf 'DECODE_ENABLE_MTP=%s\n' "${DECODE_ENABLE_MTP}"
  printf 'DECODE_MTP_DRAFT_EXECUTION=%s\n' "${DECODE_MTP_DRAFT_EXECUTION}"
  printf 'ENABLE_BATCH_INVARIANT=%s\n' "${ENABLE_BATCH_INVARIANT}"
  printf 'AFD_PROFILE_ENABLE=%s\n' "${AFD_PROFILE_ENABLE}"
  printf 'AFD_PROFILE_START_TIMEOUT_SECONDS=%s\n' \
    "${AFD_PROFILE_START_TIMEOUT_SECONDS}"
  printf 'AFD_PROFILE_FINALIZE_TIMEOUT_SECONDS=%s\n' \
    "${AFD_PROFILE_FINALIZE_TIMEOUT_SECONDS}"
  printf 'AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS=%s\n' \
    "${AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS}"
  printf 'MATRIX_PROFILE_BASE=%s\n' "${MATRIX_PROFILE_BASE:-not-set}"
  printf 'AFD_PROFILE_ATTENTION_DIR=%s\n' "${AFD_PROFILE_ATTENTION_DIR}"
  printf 'AFD_PROFILE_FFN_DIR=%s\n' "${AFD_PROFILE_FFN_DIR}"
  printf 'AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=%s\n' "${AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP}"
  printf 'AFD_HCCL_GRAPH_U2_HYBRID_DAG=%s\n' "${AFD_HCCL_GRAPH_U2_HYBRID_DAG}"
  printf 'AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=%s\n' "${AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM}"
  printf 'AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=%s\n' "${AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM}"
  printf 'AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=%s\n' "${AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER}"
  printf 'ALLOW_COLOCATED_PD_CONTROL=%s\n' "${ALLOW_COLOCATED_PD_CONTROL}"
  printf 'VLLM_ASCEND_WORKTREE_MODE=%s\n' "${VLLM_ASCEND_WORKTREE_MODE}"
  printf 'STATE_ROOT=%s\n' "${STATE_ROOT}"
  printf 'LOG_ROOT=%s\n' "${LOG_ROOT}"
  printf 'OUTPUT_ROOT=%s\n' "${OUTPUT_ROOT}"
}

case "${ACTION}" in
  print-config) print_config_action ;;
  install) install_action ;;
  check) check_action ;;
  start) start_action ;;
  status) status_action ;;
  smoke) functional_smoke_action ;;
  record-control) record_control_action ;;
  validate) validate_action ;;
  profile-start) profile_start_action ;;
  profile-check) profile_check_action ;;
  profile-stop|profile-finalize) profile_stop_action ;;
  stop) stop_action ;;
  collect) collect_action ;;
  *) usage; die "Unknown action: ${ACTION}" ;;
esac
