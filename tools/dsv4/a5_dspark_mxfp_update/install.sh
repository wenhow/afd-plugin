#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:-install}"
CONFIG_FILE="${2:-${SCRIPT_DIR}/config.env}"

die() {
  printf '[a5-dspark-official] ERROR: %s\n' "$*" >&2
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
  [[ "$(git_in "${root}" rev-parse 'HEAD^{tree}')" == "${expected_tree}" ]] \
    || die "${label} tree mismatch: expected ${expected_tree}"
}

restore_upstream_baselines() {
  local ascend_commit

  require_clean_tree "${VLLM_ROOT}" "${VLLM_TREE}" "vLLM"
  [[ "$(git_in "${VLLM_ROOT}" rev-parse HEAD)" == "${VLLM_COMMIT}" ]] \
    || die "vLLM commit mismatch: expected ${VLLM_COMMIT}"

  git_in "${VLLM_ASCEND_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "vLLM-Ascend is not a git checkout: ${VLLM_ASCEND_ROOT}"
  [[ -z "$(git_in "${VLLM_ASCEND_ROOT}" status --short --untracked-files=all)" ]] \
    || die "vLLM-Ascend worktree is dirty: ${VLLM_ASCEND_ROOT}"
  ascend_commit="$(git_in "${VLLM_ASCEND_ROOT}" rev-parse HEAD)"
  case "${ascend_commit}" in
    "${VLLM_ASCEND_BASE_COMMIT}") ;;
    "${VLLM_ASCEND_REVERTABLE_COMMIT}")
      git_in "${VLLM_ASCEND_ROOT}" checkout --detach \
        "${VLLM_ASCEND_BASE_COMMIT}"
      ;;
    *)
      die "vLLM-Ascend must be ${VLLM_ASCEND_BASE_COMMIT} or the known rollback source ${VLLM_ASCEND_REVERTABLE_COMMIT}; found ${ascend_commit}"
      ;;
  esac
  require_clean_tree "${VLLM_ASCEND_ROOT}" \
    "${VLLM_ASCEND_BASE_TREE}" "vLLM-Ascend"
  [[ "$(git_in "${VLLM_ASCEND_ROOT}" rev-parse HEAD)" == "${VLLM_ASCEND_BASE_COMMIT}" ]] \
    || die "vLLM-Ascend rollback failed"
}

install_afd_patch() {
  local patch_file source_commit source_tree

  if [[ -e "${AFD_TARGET_ROOT}" ]]; then
    require_clean_tree "${AFD_TARGET_ROOT}" "${AFD_PLUGIN_TARGET_TREE}" "afd-plugin target"
    return
  fi

  git_in "${AFD_SOURCE_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "afd-plugin source is not a git checkout: ${AFD_SOURCE_ROOT}"
  [[ -z "$(git_in "${AFD_SOURCE_ROOT}" status --short --untracked-files=all)" ]] \
    || die "afd-plugin source worktree is dirty: ${AFD_SOURCE_ROOT}"
  source_commit="$(git_in "${AFD_SOURCE_ROOT}" rev-parse HEAD)"
  source_tree="$(git_in "${AFD_SOURCE_ROOT}" rev-parse 'HEAD^{tree}')"
  case "${source_tree}" in
    "${AFD_PLUGIN_BASE_TREE}")
      patch_file="${SCRIPT_DIR}/patches/afd-plugin-66ec72f-to-official.patch"
      ;;
    "${AFD_PLUGIN_GUIDE_BASE_TREE}")
      patch_file="${SCRIPT_DIR}/patches/afd-plugin-1720b71-to-official.patch"
      ;;
    "${AFD_PLUGIN_R2_TREE}")
      patch_file="${SCRIPT_DIR}/patches/afd-plugin-7e58cc5-to-official.patch"
      ;;
    *)
      die "afd-plugin source tree is unsupported: commit=${source_commit} tree=${source_tree}"
      ;;
  esac
  mkdir -p "$(dirname "${AFD_TARGET_ROOT}")"
  git_in "${AFD_SOURCE_ROOT}" worktree add --detach \
    "${AFD_TARGET_ROOT}" "${source_commit}"
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
      -m "Apply A5 DSpark official-upstream validation patch"
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

if [[ "${ACTION}" == "install" ]]; then
  restore_upstream_baselines
  install_afd_patch
  if [[ "${SKIP_EDITABLE_INSTALL:-0}" != "1" ]]; then
    [[ -x "${VENV_ROOT}/bin/python" ]] \
      || die "Runtime venv Python is missing: ${VENV_ROOT}/bin/python"
    "${VENV_ROOT}/bin/python" -m pip install \
      --no-build-isolation --no-deps --editable "${AFD_TARGET_ROOT}"
  fi
fi

require_clean_tree "${VLLM_ROOT}" "${VLLM_TREE}" "vLLM"
[[ "$(git_in "${VLLM_ROOT}" rev-parse HEAD)" == "${VLLM_COMMIT}" ]] \
  || die "vLLM commit mismatch"
require_clean_tree "${VLLM_ASCEND_ROOT}" "${VLLM_ASCEND_BASE_TREE}" "vLLM-Ascend"
[[ "$(git_in "${VLLM_ASCEND_ROOT}" rev-parse HEAD)" == "${VLLM_ASCEND_BASE_COMMIT}" ]] \
  || die "vLLM-Ascend commit mismatch"
require_clean_tree "${AFD_TARGET_ROOT}" "${AFD_PLUGIN_TARGET_TREE}" "afd-plugin target"
[[ -f "${VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash" ]] \
  || die "Existing vLLM-Ascend custom ops are missing; this source-only package does not rebuild them"

printf '[a5-dspark-official] ready\n'
printf 'export AFD_PLUGIN_ROOT=%q\n' "${AFD_TARGET_ROOT}"
printf 'export DSV4_VLLM_ROOT=%q\n' "${VLLM_ROOT}"
printf 'export DSV4_VLLM_ASCEND_ROOT=%q\n' "${VLLM_ASCEND_ROOT}"
printf 'vLLM commit: %s\n' "${VLLM_COMMIT}"
printf 'vLLM-Ascend commit: %s\n' "${VLLM_ASCEND_BASE_COMMIT}"
printf 'afd-plugin commit: %s\n' "$(git_in "${AFD_TARGET_ROOT}" rev-parse HEAD)"
printf 'afd-plugin tree: %s\n' "${AFD_PLUGIN_TARGET_TREE}"
