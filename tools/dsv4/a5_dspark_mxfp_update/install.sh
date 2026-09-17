#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-install}"
CONFIG_FILE="${2:-${SCRIPT_DIR}/config.env}"

die() {
  printf '[a5-dspark-mxfp] ERROR: %s\n' "$*" >&2
  exit 2
}

git_in() {
  local root="$1"
  shift
  git -c safe.directory="${root}" -C "${root}" "$@"
}

require_clean_head() {
  local root="$1" expected="$2" label="$3"
  [[ -d "${root}/.git" ]] || die "${label} is not a git checkout: ${root}"
  [[ "$(git_in "${root}" rev-parse HEAD)" == "${expected}" ]] \
    || die "${label} commit mismatch: expected ${expected}"
  [[ -z "$(git_in "${root}" status --short --untracked-files=all)" ]] \
    || die "${label} worktree is dirty: ${root}"
}

[[ -f "${SCRIPT_DIR}/VERSION.env" ]] || die "Missing VERSION.env"
[[ -f "${CONFIG_FILE}" ]] || die "Missing config: ${CONFIG_FILE}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/VERSION.env"
# shellcheck disable=SC1090
source "${CONFIG_FILE}"

case "${ACTION}" in
  install|check) ;;
  *) die "Usage: bash install.sh [install|check] [config.env]" ;;
esac

(
  cd "${SCRIPT_DIR}"
  sha256sum -c SHA256SUMS
)

[[ -f "${MODEL_PATH}/config.json" ]] || die "Missing model config: ${MODEL_PATH}/config.json"
[[ -f "${MODEL_PATH}/model.safetensors.index.json" ]] \
  || die "Missing model index: ${MODEL_PATH}/model.safetensors.index.json"
[[ "$(sha256sum "${MODEL_PATH}/config.json" | awk '{print $1}')" == "${MODEL_CONFIG_SHA256}" ]] \
  || die "DSpark config.json SHA256 mismatch"
[[ "$(sha256sum "${MODEL_PATH}/model.safetensors.index.json" | awk '{print $1}')" == "${MODEL_INDEX_SHA256}" ]] \
  || die "DSpark model.safetensors.index.json SHA256 mismatch"

if [[ "${ACTION}" == "install" ]]; then
  [[ -d "${VLLM_ASCEND_ROOT}/.git" ]] \
    || die "vLLM-Ascend is not a git checkout: ${VLLM_ASCEND_ROOT}"
  [[ -z "$(git_in "${VLLM_ASCEND_ROOT}" status --short --untracked-files=all)" ]] \
    || die "vLLM-Ascend worktree is dirty: ${VLLM_ASCEND_ROOT}"
  current_ascend_commit="$(git_in "${VLLM_ASCEND_ROOT}" rev-parse HEAD)"
  case "${current_ascend_commit}" in
    "${VLLM_ASCEND_BASE_COMMIT}")
      git_in "${VLLM_ASCEND_ROOT}" fetch \
        "${SCRIPT_DIR}/payload/vllm-ascend.bundle" \
        "refs/heads/fix/dsv4-compressed-mxfp-detection"
      git_in "${VLLM_ASCEND_ROOT}" merge --ff-only "${VLLM_ASCEND_PATCH_COMMIT}"
      ;;
    "${VLLM_ASCEND_PATCH_COMMIT}") ;;
    *)
      die "vLLM-Ascend must be ${VLLM_ASCEND_BASE_COMMIT} or ${VLLM_ASCEND_PATCH_COMMIT}; found ${current_ascend_commit}"
      ;;
  esac

  if [[ -e "${AFD_TARGET_ROOT}" ]]; then
    require_clean_head "${AFD_TARGET_ROOT}" "${AFD_PLUGIN_COMMIT}" "afd-plugin target"
  else
    mkdir -p "$(dirname "${AFD_TARGET_ROOT}")"
    git clone --no-checkout "${SCRIPT_DIR}/payload/afd-plugin.bundle" "${AFD_TARGET_ROOT}"
    git_in "${AFD_TARGET_ROOT}" checkout --detach "${AFD_PLUGIN_COMMIT}"
  fi
fi

require_clean_head "${VLLM_ASCEND_ROOT}" "${VLLM_ASCEND_PATCH_COMMIT}" "vLLM-Ascend"
require_clean_head "${AFD_TARGET_ROOT}" "${AFD_PLUGIN_COMMIT}" "afd-plugin target"
[[ -f "${VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash" ]] \
  || die "Existing vLLM-Ascend custom ops are missing; this source-only package does not rebuild them"

printf '[a5-dspark-mxfp] ready\n'
printf 'export AFD_PLUGIN_ROOT=%q\n' "${AFD_TARGET_ROOT}"
printf 'export DSV4_VLLM_ASCEND_ROOT=%q\n' "${VLLM_ASCEND_ROOT}"
printf 'export MODEL_PATH=%q\n' "${MODEL_PATH}"
printf 'vLLM-Ascend commit: %s\n' "${VLLM_ASCEND_PATCH_COMMIT}"
printf 'afd-plugin commit: %s\n' "${AFD_PLUGIN_COMMIT}"
