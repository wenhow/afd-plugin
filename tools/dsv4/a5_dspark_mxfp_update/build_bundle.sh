#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace}"
VLLM_ASCEND_SOURCE_ROOT="${VLLM_ASCEND_SOURCE_ROOT:-${WORKSPACE_ROOT}/code/vllm-ascend-rfc-vllm-cann-3da28f941}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/delivery}"
PACKAGE_NAME="dsv4-a5-dspark-compressed-mxfp-patch-20260917"
VLLM_ASCEND_BASE_COMMIT=3da28f9414583d2d0b672a8f06d1fae142404bda
VLLM_ASCEND_PATCH_COMMIT=18a0709c88a6c1abed792f0f071bb0f9e8a5fc07
AFD_PLUGIN_BASE_COMMIT=66ec72fdea87cd3b6dfa483cc252305245c220c7

[[ "$(git -C "${VLLM_ASCEND_SOURCE_ROOT}" rev-parse HEAD)" == "${VLLM_ASCEND_PATCH_COMMIT}" ]]
[[ -z "$(git -C "${VLLM_ASCEND_SOURCE_ROOT}" status --short --untracked-files=all)" ]]
[[ -z "$(git -C "${REPO_ROOT}" status --short --untracked-files=all)" ]]
AFD_PLUGIN_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
AFD_PLUGIN_BASE_TREE="$(git -C "${REPO_ROOT}" show -s --format=%T "${AFD_PLUGIN_BASE_COMMIT}")"
AFD_PLUGIN_TARGET_TREE="$(git -C "${REPO_ROOT}" show -s --format=%T "${AFD_PLUGIN_COMMIT}")"
VLLM_ASCEND_PATCH_TREE="$(git -C "${VLLM_ASCEND_SOURCE_ROOT}" show -s --format=%T "${VLLM_ASCEND_PATCH_COMMIT}")"

staging_root="$(mktemp -d)"
trap 'rm -rf "${staging_root}"' EXIT
package_root="${staging_root}/${PACKAGE_NAME}"
mkdir -p "${package_root}/patches" "${OUTPUT_DIR}"

cp "${SCRIPT_DIR}/install.sh" "${package_root}/install.sh"
cp "${SCRIPT_DIR}/config.env.example" "${package_root}/config.env"
cp "${SCRIPT_DIR}/README_ZH.md" "${package_root}/README_ZH.md"
cp "${REPO_ROOT}/docs/npu/DEEPSEEK_V4_AFD_PHASE1_A5_VALIDATION_GUIDE_ZH.md" "${package_root}/"

git -C "${VLLM_ASCEND_SOURCE_ROOT}" diff --binary \
  "${VLLM_ASCEND_BASE_COMMIT}" "${VLLM_ASCEND_PATCH_COMMIT}" -- \
  >"${package_root}/patches/vllm-ascend-3da28f9-to-18a0709.patch"
git -C "${VLLM_ASCEND_SOURCE_ROOT}" cat-file commit \
  "${VLLM_ASCEND_PATCH_COMMIT}" \
  | sed '1,/^$/d' \
  >"${package_root}/patches/vllm-ascend-18a0709.message"
git -C "${REPO_ROOT}" diff --binary \
  "${AFD_PLUGIN_BASE_COMMIT}" "${AFD_PLUGIN_COMMIT}" -- \
  >"${package_root}/patches/afd-plugin-66ec72f-to-mxfp.patch"

{
  printf 'VLLM_ASCEND_BASE_COMMIT=%q\n' "${VLLM_ASCEND_BASE_COMMIT}"
  printf 'VLLM_ASCEND_PATCH_COMMIT=%q\n' "${VLLM_ASCEND_PATCH_COMMIT}"
  printf 'VLLM_ASCEND_PATCH_TREE=%q\n' "${VLLM_ASCEND_PATCH_TREE}"
  printf 'VLLM_ASCEND_AUTHOR_NAME=%q\n' wenhow
  printf 'VLLM_ASCEND_AUTHOR_EMAIL=%q\n' 47055533+wenhow@users.noreply.github.com
  printf 'VLLM_ASCEND_COMMITTER_NAME=%q\n' wenhow
  printf 'VLLM_ASCEND_COMMITTER_EMAIL=%q\n' 47055533+wenhow@users.noreply.github.com
  printf 'VLLM_ASCEND_COMMIT_TIME=%q\n' 1789613655
  printf 'VLLM_ASCEND_COMMIT_TZ=%q\n' +0800
  printf 'AFD_PLUGIN_BASE_COMMIT=%q\n' "${AFD_PLUGIN_BASE_COMMIT}"
  printf 'AFD_PLUGIN_BASE_TREE=%q\n' "${AFD_PLUGIN_BASE_TREE}"
  printf 'AFD_PLUGIN_TARGET_SOURCE_COMMIT=%q\n' "${AFD_PLUGIN_COMMIT}"
  printf 'AFD_PLUGIN_TARGET_TREE=%q\n' "${AFD_PLUGIN_TARGET_TREE}"
  printf 'MODEL_CONFIG_SHA256=%q\n' db3e4addebd459d7cc67b09790abf147edb963e1dd49dcf90c7332bbdb8bee26
  printf 'MODEL_INDEX_SHA256=%q\n' 98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b
} >"${package_root}/VERSION.env"

(
  cd "${package_root}"
  sha256sum \
    README_ZH.md \
    DEEPSEEK_V4_AFD_PHASE1_A5_VALIDATION_GUIDE_ZH.md \
    VERSION.env \
    install.sh \
    patches/afd-plugin-66ec72f-to-mxfp.patch \
    patches/vllm-ascend-18a0709.message \
    patches/vllm-ascend-3da28f9-to-18a0709.patch \
    >SHA256SUMS
)

tar -C "${staging_root}" -czf "${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}"
sha256sum "${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz"
