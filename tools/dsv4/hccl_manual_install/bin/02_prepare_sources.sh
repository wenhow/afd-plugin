#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

prepare_target() {
  local target="$1"
  local label="$2"
  if ! dir_is_empty "${target}"; then
    if is_true "${REUSE_SOURCES}"; then
      log "Reusing ${label} source: ${target}"
      return 1
    fi
    die "${label} source directory is not empty: ${target}"
  fi
  ensure_dir "${target}"
  return 0
}

extract_source() {
  local archive="$1"
  local target="$2"
  local label="$3"
  require_file "${archive}"
  if prepare_target "${target}" "${label}"; then
    log "Extracting ${label}: ${archive}"
    tar -xzf "${archive}" -C "${target}"
  fi
}

clone_source() {
  local url="$1"
  local ref="$2"
  local target="$3"
  local label="$4"
  if prepare_target "${target}" "${label}"; then
    log "Cloning ${label}: ${url}"
    git clone --no-checkout "${url}" "${target}"
    git -C "${target}" checkout --detach "${ref}"
  fi
}

verify_git_head() {
  local target="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(git -C "${target}" rev-parse HEAD)"
  [[ "${actual}" == "${expected}" ]] \
    || die "${label} commit mismatch: expected ${expected}, got ${actual}"
}

prepare_afd_from_seed_bundle() {
  local seed_bundle="${BUNDLE_ROOT}/manifest/afd-plugin-from-seed.bundle"
  local actual_bundle_sha current_tree seed_changes
  [[ "${BUNDLE_INCLUDES_AFD_SEED}" == "1" ]] \
    || die "This package does not include an afd-plugin seed bundle"
  require_file "${seed_bundle}"
  [[ -n "${AFD_SEED_ROOT}" ]] || die "AFD_SEED_ROOT is required"
  [[ -n "${AFD_SEED_COMMIT}" ]] || die "AFD_SEED_COMMIT is required"
  git -C "${AFD_SEED_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "afd-plugin seed is not a Git worktree: ${AFD_SEED_ROOT}"
  verify_git_head "${AFD_SEED_ROOT}" "${AFD_SEED_COMMIT}" "afd-plugin seed"
  seed_changes="$(git_unexpected_changes "${AFD_SEED_ROOT}")"
  if [[ -n "${seed_changes}" ]]; then
    ensure_dir "${STATE_ROOT}"
    git -C "${AFD_SEED_ROOT}" status --short \
      >"${STATE_ROOT}/afd-seed-worktree-status.txt"
    git -C "${AFD_SEED_ROOT}" diff --binary HEAD \
      >"${STATE_ROOT}/afd-seed-local-changes.patch"
    warn "afd-plugin seed has local changes; preserving the seed and cloning only committed ${AFD_SEED_COMMIT} objects"
    warn "seed status and tracked diff saved under ${STATE_ROOT}"
  fi
  actual_bundle_sha="$(sha256sum "${seed_bundle}" | awk '{print $1}')"
  [[ "${actual_bundle_sha}" == "${AFD_SEED_BUNDLE_SHA256}" ]] \
    || die "afd-plugin seed bundle checksum mismatch"

  if ! dir_is_empty "${AFD_PLUGIN_ROOT}"; then
    is_true "${REUSE_SOURCES}" \
      || die "afd-plugin target directory is not empty: ${AFD_PLUGIN_ROOT}"
    verify_git_head "${AFD_PLUGIN_ROOT}" "${AFD_TARGET_COMMIT}" \
      "afd-plugin target"
    require_clean_git_tree "${AFD_PLUGIN_ROOT}" "afd-plugin target"
    current_tree="$(git -C "${AFD_PLUGIN_ROOT}" show -s --format=%T HEAD)"
    [[ "${current_tree}" == "${AFD_TARGET_TREE}" ]] \
      || die "afd-plugin target tree mismatch"
    log "Reusing afd-plugin target: ${AFD_PLUGIN_ROOT}"
    return
  fi

  ensure_dir "$(dirname "${AFD_PLUGIN_ROOT}")"
  log "Creating afd-plugin target from validated seed: ${AFD_SEED_ROOT}"
  git clone --no-checkout --no-hardlinks "${AFD_SEED_ROOT}" "${AFD_PLUGIN_ROOT}"
  git -C "${AFD_PLUGIN_ROOT}" fetch "${seed_bundle}" HEAD
  git -C "${AFD_PLUGIN_ROOT}" checkout --detach "${AFD_TARGET_COMMIT}"
  verify_git_head "${AFD_PLUGIN_ROOT}" "${AFD_TARGET_COMMIT}" \
    "afd-plugin target"
  require_clean_git_tree "${AFD_PLUGIN_ROOT}" "afd-plugin target"
  current_tree="$(git -C "${AFD_PLUGIN_ROOT}" show -s --format=%T HEAD)"
  [[ "${current_tree}" == "${AFD_TARGET_TREE}" ]] \
    || die "afd-plugin target tree mismatch"
}

apply_afd_patch() {
  local patch_file="${BUNDLE_ROOT}/manifest/afd-plugin-phase1.patch"
  local actual_patch_sha current_tree unexpected_untracked

  require_file "${patch_file}"
  actual_patch_sha="$(sha256sum "${patch_file}" | awk '{print $1}')"
  [[ "${actual_patch_sha}" == "${AFD_PATCH_SHA256}" ]] \
    || die "afd-plugin patch checksum mismatch"

  current_tree="$(git -C "${AFD_PLUGIN_ROOT}" write-tree)"
  git -C "${AFD_PLUGIN_ROOT}" diff --quiet \
    && git -C "${AFD_PLUGIN_ROOT}" diff --cached --quiet \
    || die "afd-plugin has tracked worktree changes"
  unexpected_untracked="$(
    git -C "${AFD_PLUGIN_ROOT}" ls-files --others --exclude-standard \
      | grep -vFx '.bundle-source-version' \
      || true
  )"
  [[ -z "${unexpected_untracked}" ]] \
    || die "afd-plugin has unexpected untracked files: ${unexpected_untracked}"

  if [[ "${current_tree}" == "${AFD_TARGET_TREE}" ]]; then
    log "Reusing patched afd-plugin release tree: ${AFD_SNAPSHOT_ID}"
    return
  fi
  verify_git_head "${AFD_PLUGIN_ROOT}" "${AFD_SOURCE_COMMIT}" "afd-plugin base"
  [[ "${current_tree}" == "${AFD_SOURCE_TREE}" ]] \
    || die "afd-plugin source has changes outside the expected release patch"

  log "Applying afd-plugin release patch: ${AFD_SNAPSHOT_ID}"
  git -C "${AFD_PLUGIN_ROOT}" apply --index "${patch_file}"
  current_tree="$(git -C "${AFD_PLUGIN_ROOT}" write-tree)"
  [[ "${current_tree}" == "${AFD_TARGET_TREE}" ]] \
    || die "afd-plugin patched tree does not match ${AFD_TARGET_COMMIT}"
  GIT_AUTHOR_NAME=afd-plugin-delivery \
  GIT_AUTHOR_EMAIL=afd-plugin-delivery@localhost \
  GIT_AUTHOR_DATE=2000-01-01T00:00:00Z \
  GIT_COMMITTER_NAME=afd-plugin-delivery \
  GIT_COMMITTER_EMAIL=afd-plugin-delivery@localhost \
  GIT_COMMITTER_DATE=2000-01-01T00:00:00Z \
    git -C "${AFD_PLUGIN_ROOT}" commit -q -m "Apply ${AFD_SNAPSHOT_ID}"
  [[ "$(git -C "${AFD_PLUGIN_ROOT}" show -s --format=%T HEAD)" == "${AFD_TARGET_TREE}" ]] \
    || die "afd-plugin delivery commit tree mismatch"
}

ensure_dir "${CODE_ROOT}"

if is_true "${USE_BUNDLED_SOURCES}"; then
  [[ "${BUNDLE_INCLUDES_SOURCES:-0}" == "1" ]] \
    || die "This is a slim package. Set USE_BUNDLED_SOURCES=0 to download sources."
  extract_source \
    "${BUNDLE_ROOT}/sources/vllm-release-v0.23.0.tar.gz" \
    "${VLLM_ROOT}" \
    "vLLM"
  extract_source \
    "${BUNDLE_ROOT}/sources/vllm-ascend-rfc-vllm-cann.tar.gz" \
    "${VLLM_ASCEND_ROOT}" \
    "vLLM-Ascend"
  extract_source \
    "${BUNDLE_ROOT}/sources/afd-plugin-phase1-snapshot.tar.gz" \
    "${AFD_PLUGIN_ROOT}" \
    "afd-plugin"
else
  clone_source "${VLLM_GIT_URL}" "${VLLM_COMMIT}" "${VLLM_ROOT}" "vLLM"
  verify_git_head "${VLLM_ROOT}" "${VLLM_COMMIT}" "vLLM"
  require_clean_git_tree "${VLLM_ROOT}" "vLLM"

  clone_source \
    "${VLLM_ASCEND_GIT_URL}" \
    "${VLLM_ASCEND_COMMIT}" \
    "${VLLM_ASCEND_ROOT}" \
    "vLLM-Ascend"
  verify_git_head "${VLLM_ASCEND_ROOT}" "${VLLM_ASCEND_COMMIT}" "vLLM-Ascend"
  require_clean_git_tree "${VLLM_ASCEND_ROOT}" "vLLM-Ascend"
  git -C "${VLLM_ASCEND_ROOT}" submodule update --init --recursive

  if is_true "${USE_AFD_SEED_BUNDLE}"; then
    prepare_afd_from_seed_bundle
  else
    [[ -n "${AFD_PLUGIN_REF}" ]] \
      || die "AFD_PLUGIN_REF is required when bundled sources are disabled"
    clone_source \
      "${AFD_PLUGIN_GIT_URL}" \
      "${AFD_PLUGIN_REF}" \
      "${AFD_PLUGIN_ROOT}" \
      "afd-plugin"
    apply_afd_patch
  fi
fi

require_file "${VLLM_ROOT}/setup.py"
require_file "${VLLM_ASCEND_ROOT}/setup.py"
require_file "${VLLM_ASCEND_ROOT}/csrc/third_party/catlass/CMakeLists.txt"
require_file "${AFD_PLUGIN_ROOT}/pyproject.toml"
require_file "${AFD_PLUGIN_ROOT}/afd_plugin/connectors/npu/p2p_hccl.py"
require_file \
  "${AFD_PLUGIN_ROOT}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_attention.sh"

printf '%s\n' "${VLLM_COMMIT}" >"${VLLM_ROOT}/.bundle-source-version"
printf '%s\n' "${VLLM_ASCEND_COMMIT}" >"${VLLM_ASCEND_ROOT}/.bundle-source-version"
printf '%s\n' "${AFD_TARGET_COMMIT} ${AFD_SNAPSHOT_ID}" \
  >"${AFD_PLUGIN_ROOT}/.bundle-source-version"
for source_root in "${VLLM_ROOT}" "${VLLM_ASCEND_ROOT}" "${AFD_PLUGIN_ROOT}"; do
  if [[ -d "${source_root}/.git" ]]; then
    grep -qxF '.bundle-source-version' "${source_root}/.git/info/exclude" 2>/dev/null \
      || printf '%s\n' '.bundle-source-version' >>"${source_root}/.git/info/exclude"
  fi
done

log "Sources are ready under ${CODE_ROOT}"
