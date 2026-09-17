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

require_clean_tree() {
  local root="$1" expected_tree="$2" label="$3"
  git_in "${root}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "${label} is not a git checkout: ${root}"
  [[ -z "$(git_in "${root}" status --short --untracked-files=all)" ]] \
    || die "${label} worktree is dirty: ${root}"
  [[ "$(git_in "${root}" show -s --format=%T HEAD)" == "${expected_tree}" ]] \
    || die "${label} tree mismatch: expected ${expected_tree}"
}

install_vllm_ascend_patch() {
  local patch_file="${SCRIPT_DIR}/patches/vllm-ascend-3da28f9-to-18a0709.patch"
  local message_file="${SCRIPT_DIR}/patches/vllm-ascend-18a0709.message"
  local current_commit new_commit

  git_in "${VLLM_ASCEND_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "vLLM-Ascend is not a git checkout: ${VLLM_ASCEND_ROOT}"
  [[ -z "$(git_in "${VLLM_ASCEND_ROOT}" status --short --untracked-files=all)" ]] \
    || die "vLLM-Ascend worktree is dirty: ${VLLM_ASCEND_ROOT}"
  current_commit="$(git_in "${VLLM_ASCEND_ROOT}" rev-parse HEAD)"
  case "${current_commit}" in
    "${VLLM_ASCEND_PATCH_COMMIT}") return ;;
    "${VLLM_ASCEND_BASE_COMMIT}") ;;
    *)
      die "vLLM-Ascend must be ${VLLM_ASCEND_BASE_COMMIT} or ${VLLM_ASCEND_PATCH_COMMIT}; found ${current_commit}"
      ;;
  esac

  git_in "${VLLM_ASCEND_ROOT}" apply --check "${patch_file}"
  git_in "${VLLM_ASCEND_ROOT}" apply --index "${patch_file}"
  [[ "$(git_in "${VLLM_ASCEND_ROOT}" write-tree)" == "${VLLM_ASCEND_PATCH_TREE}" ]] \
    || die "vLLM-Ascend patched tree mismatch"

  new_commit="$(
    GIT_AUTHOR_NAME="${VLLM_ASCEND_AUTHOR_NAME}" \
    GIT_AUTHOR_EMAIL="${VLLM_ASCEND_AUTHOR_EMAIL}" \
    GIT_AUTHOR_DATE="@${VLLM_ASCEND_COMMIT_TIME} ${VLLM_ASCEND_COMMIT_TZ}" \
    GIT_COMMITTER_NAME="${VLLM_ASCEND_COMMITTER_NAME}" \
    GIT_COMMITTER_EMAIL="${VLLM_ASCEND_COMMITTER_EMAIL}" \
    GIT_COMMITTER_DATE="@${VLLM_ASCEND_COMMIT_TIME} ${VLLM_ASCEND_COMMIT_TZ}" \
      git -c safe.directory="${VLLM_ASCEND_ROOT}" \
        -C "${VLLM_ASCEND_ROOT}" commit-tree \
        "${VLLM_ASCEND_PATCH_TREE}" \
        -p "${VLLM_ASCEND_BASE_COMMIT}" \
        <"${message_file}"
  )" || die "Could not create vLLM-Ascend patch commit"
  [[ "${new_commit}" == "${VLLM_ASCEND_PATCH_COMMIT}" ]] \
    || die "Reconstructed vLLM-Ascend commit mismatch: ${new_commit}"
  git_in "${VLLM_ASCEND_ROOT}" update-ref \
    HEAD "${new_commit}" "${VLLM_ASCEND_BASE_COMMIT}"
}

install_afd_patch() {
  local patch_file="${SCRIPT_DIR}/patches/afd-plugin-66ec72f-to-mxfp.patch"

  if [[ -e "${AFD_TARGET_ROOT}" ]]; then
    require_clean_tree "${AFD_TARGET_ROOT}" "${AFD_PLUGIN_TARGET_TREE}" "afd-plugin target"
    return
  fi

  [[ "$(git_in "${AFD_SOURCE_ROOT}" rev-parse HEAD)" == "${AFD_PLUGIN_BASE_COMMIT}" ]] \
    || die "afd-plugin source must be ${AFD_PLUGIN_BASE_COMMIT}"
  require_clean_tree "${AFD_SOURCE_ROOT}" "${AFD_PLUGIN_BASE_TREE}" "afd-plugin source"
  mkdir -p "$(dirname "${AFD_TARGET_ROOT}")"
  git_in "${AFD_SOURCE_ROOT}" worktree add --detach \
    "${AFD_TARGET_ROOT}" "${AFD_PLUGIN_BASE_COMMIT}"
  git_in "${AFD_TARGET_ROOT}" apply --check "${patch_file}"
  git_in "${AFD_TARGET_ROOT}" apply --index "${patch_file}"
  [[ "$(git_in "${AFD_TARGET_ROOT}" write-tree)" == "${AFD_PLUGIN_TARGET_TREE}" ]] \
    || die "afd-plugin patched tree mismatch"
  GIT_AUTHOR_NAME=afd-plugin-delivery \
  GIT_AUTHOR_EMAIL=afd-plugin-delivery@localhost \
  GIT_COMMITTER_NAME=afd-plugin-delivery \
  GIT_COMMITTER_EMAIL=afd-plugin-delivery@localhost \
    git -c safe.directory="${AFD_TARGET_ROOT}" \
      -C "${AFD_TARGET_ROOT}" commit -q \
      -m "Apply A5 DSpark compressed MX compatibility patch"
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
  install_vllm_ascend_patch
  install_afd_patch
  if [[ "${SKIP_EDITABLE_INSTALL:-0}" != "1" ]]; then
    [[ -x "${VENV_ROOT}/bin/python" ]] \
      || die "Runtime venv Python is missing: ${VENV_ROOT}/bin/python"
    "${VENV_ROOT}/bin/python" -m pip install \
      --no-build-isolation --no-deps --editable "${AFD_TARGET_ROOT}"
  fi
fi

[[ "$(git_in "${VLLM_ASCEND_ROOT}" rev-parse HEAD)" == "${VLLM_ASCEND_PATCH_COMMIT}" ]] \
  || die "vLLM-Ascend commit mismatch after patch"
require_clean_tree "${VLLM_ASCEND_ROOT}" "${VLLM_ASCEND_PATCH_TREE}" "vLLM-Ascend"
require_clean_tree "${AFD_TARGET_ROOT}" "${AFD_PLUGIN_TARGET_TREE}" "afd-plugin target"
[[ -f "${VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash" ]] \
  || die "Existing vLLM-Ascend custom ops are missing; this source-only package does not rebuild them"

printf '[a5-dspark-mxfp] ready\n'
printf 'export AFD_PLUGIN_ROOT=%q\n' "${AFD_TARGET_ROOT}"
printf 'export DSV4_VLLM_ASCEND_ROOT=%q\n' "${VLLM_ASCEND_ROOT}"
printf 'export MODEL_PATH=%q\n' "${MODEL_PATH}"
printf 'vLLM-Ascend commit: %s\n' "${VLLM_ASCEND_PATCH_COMMIT}"
printf 'afd-plugin commit: %s\n' "$(git_in "${AFD_TARGET_ROOT}" rev-parse HEAD)"
printf 'afd-plugin tree: %s\n' "${AFD_PLUGIN_TARGET_TREE}"
