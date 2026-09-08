#!/usr/bin/env bash

if [[ -n "${HCCL_MANUAL_INSTALL_COMMON_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
HCCL_MANUAL_INSTALL_COMMON_LOADED=1

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSIONS_FILE="${BUNDLE_ROOT}/manifest/versions.env"
CONFIG_FILE="${CONFIG_FILE:-${BUNDLE_ROOT}/config.env}"

if [[ -f "${VERSIONS_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${VERSIONS_FILE}"
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing config: ${CONFIG_FILE}" >&2
  echo "Copy ${BUNDLE_ROOT}/config.env.example to config.env and edit it." >&2
  return 2 2>/dev/null || exit 2
fi

set -a
# shellcheck disable=SC1090
source "${CONFIG_FILE}"
set +a

# Make system utilities from the configured base path available to preflight
# and packaging helpers without discarding the caller's path yet. The runtime
# activation script later resets PATH completely before sourcing CANN.
export PATH="${SYSTEM_PATH}:${PATH:-}"

: "${VLLM_COMMIT:=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665}"
: "${VLLM_ASCEND_COMMIT:=3da28f9414583d2d0b672a8f06d1fae142404bda}"
: "${AFD_SOURCE_COMMIT:=9db1fb981f5262986e7ea05938e9c6ed57b5a885}"
: "${AFD_SOURCE_TREE:=unset}"
: "${AFD_TARGET_COMMIT:=unset}"
: "${AFD_TARGET_TREE:=unset}"
: "${AFD_PATCH_SHA256:=unset}"
: "${AFD_SNAPSHOT_ID:=dsv4-afd-v023-cann900-phase1-external-v1}"
: "${BUNDLE_CONFIG_PROFILE:=generic}"
: "${BUNDLE_INCLUDES_AFD_SEED:=0}"
: "${AFD_SEED_BUNDLE_SHA256:=}"
: "${USE_AFD_SEED_BUNDLE:=0}"
: "${AFD_SEED_ROOT:=}"
: "${AFD_SEED_COMMIT:=}"
: "${INSTALL_PYTHON_DEPS:=1}"
: "${INSTALL_UPSTREAM_STACK:=1}"

log() {
  printf '[hccl-install] %s\n' "$*"
}

warn() {
  printf '[hccl-install] WARNING: %s\n' "$*" >&2
}

die() {
  printf '[hccl-install] ERROR: %s\n' "$*" >&2
  exit 2
}

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

ensure_dir() {
  mkdir -p "$1"
}

dir_is_empty() {
  [[ -d "$1" ]] || return 0
  [[ -z "$(find "$1" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}

git_unexpected_changes() {
  local target="$1"
  git -C "${target}" status --short \
    | grep -vE '^\?\? \.bundle-source-version$' \
    || true
}

require_clean_git_tree() {
  local target="$1" label="$2" changes
  changes="$(git_unexpected_changes "${target}")"
  [[ -z "${changes}" ]] \
    || die "${label} worktree is not clean: ${changes}"
}

resolve_hccl_ip() {
  if [[ -n "${HCCL_IF_IP:-}" ]]; then
    printf '%s\n' "${HCCL_IF_IP}"
    return 0
  fi
  command -v ip >/dev/null 2>&1 || return 1
  ip -o -4 addr show dev "${NIC_NAME}" \
    | awk 'NR == 1 {split($4, parts, "/"); print parts[1]}'
}

device_list_count() {
  awk -F, '{print NF}' <<<"$1"
}

device_lists_are_disjoint() {
  local first="$1" second="$2" device
  local -A seen=()
  IFS=',' read -r -a first_devices <<<"${first}"
  IFS=',' read -r -a second_devices <<<"${second}"
  for device in "${first_devices[@]}"; do
    seen["${device//[[:space:]]/}"]=1
  done
  for device in "${second_devices[@]}"; do
    [[ -z "${seen[${device//[[:space:]]/}]+x}" ]] || return 1
  done
}

pid_is_alive() {
  [[ "$1" =~ ^[0-9]+$ ]] && kill -0 "$1" 2>/dev/null
}

read_pid_file() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  local pid
  read -r pid <"${path}"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${pid}"
}

port_is_listening() {
  local port="$1"
  ss -ltn | awk -v expected=":${port}" '$4 ~ expected "$" {found=1} END {exit !found}'
}

assert_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

assert_zero_or_one() {
  local name="$1"
  local value="$2"
  [[ "${value}" == "0" || "${value}" == "1" ]] \
    || die "${name} must be 0 or 1: ${value}"
}

print_version_contract() {
  log "vLLM commit: ${VLLM_COMMIT}"
  log "vLLM-Ascend commit: ${VLLM_ASCEND_COMMIT}"
  log "afd-plugin download base: ${AFD_SOURCE_COMMIT}"
  log "afd-plugin target: ${AFD_TARGET_COMMIT} (${AFD_SNAPSHOT_ID})"
  log "bundle config profile: ${BUNDLE_CONFIG_PROFILE}"
}
