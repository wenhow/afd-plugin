#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace}"
VLLM_SOURCE_ROOT="${VLLM_SOURCE_ROOT:-${WORKSPACE_ROOT}/code/vllm-release-v0.23.0}"
VLLM_ASCEND_SOURCE_ROOT="${VLLM_ASCEND_SOURCE_ROOT:-${WORKSPACE_ROOT}/code/vllm-ascend-rfc-vllm-cann-3da28f941}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/delivery}"
PACKAGE_NAME="dsv4-a5-dspark-official-upstream-patch-20260917-r3"
VLLM_COMMIT=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
VLLM_ASCEND_BASE_COMMIT=3da28f9414583d2d0b672a8f06d1fae142404bda
VLLM_ASCEND_REVERTABLE_COMMIT=18a0709c88a6c1abed792f0f071bb0f9e8a5fc07
AFD_PLUGIN_BASE_COMMIT=66ec72fdea87cd3b6dfa483cc252305245c220c7
AFD_PLUGIN_GUIDE_BASE_COMMIT=1720b714b90f190cb6097bdf35f69f960c6ad798
AFD_PLUGIN_R2_COMMIT=7e58cc58e5d90c4558e3ea0fcfbe9a889668fea3

[[ "$(git -C "${VLLM_SOURCE_ROOT}" rev-parse HEAD)" == "${VLLM_COMMIT}" ]]
[[ -z "$(git -C "${VLLM_SOURCE_ROOT}" status --short --untracked-files=all)" ]]
[[ "$(git -C "${VLLM_ASCEND_SOURCE_ROOT}" rev-parse HEAD)" == "${VLLM_ASCEND_BASE_COMMIT}" ]]
[[ -z "$(git -C "${VLLM_ASCEND_SOURCE_ROOT}" status --short --untracked-files=all)" ]]
[[ -z "$(git -C "${REPO_ROOT}" status --short --untracked-files=all)" ]]
AFD_PLUGIN_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
VLLM_TREE="$(git -C "${VLLM_SOURCE_ROOT}" rev-parse "${VLLM_COMMIT}^{tree}")"
VLLM_ASCEND_BASE_TREE="$(git -C "${VLLM_ASCEND_SOURCE_ROOT}" rev-parse "${VLLM_ASCEND_BASE_COMMIT}^{tree}")"
AFD_PLUGIN_BASE_TREE="$(git -C "${REPO_ROOT}" rev-parse "${AFD_PLUGIN_BASE_COMMIT}^{tree}")"
AFD_PLUGIN_GUIDE_BASE_TREE="$(git -C "${REPO_ROOT}" rev-parse "${AFD_PLUGIN_GUIDE_BASE_COMMIT}^{tree}")"
AFD_PLUGIN_R2_TREE="$(git -C "${REPO_ROOT}" rev-parse "${AFD_PLUGIN_R2_COMMIT}^{tree}")"
AFD_PLUGIN_TARGET_TREE="$(git -C "${REPO_ROOT}" rev-parse "${AFD_PLUGIN_COMMIT}^{tree}")"

staging_root="$(mktemp -d)"
trap 'rm -rf "${staging_root}"' EXIT
package_root="${staging_root}/${PACKAGE_NAME}"
mkdir -p "${package_root}/patches" "${OUTPUT_DIR}"

cp "${SCRIPT_DIR}/install.sh" "${package_root}/install.sh"
cp "${SCRIPT_DIR}/config.env.example" "${package_root}/config.env"
cp "${SCRIPT_DIR}/README_ZH.md" "${package_root}/README_ZH.md"
cp "${REPO_ROOT}/docs/npu/DEEPSEEK_V4_AFD_PHASE1_A5_VALIDATION_GUIDE_ZH.md" "${package_root}/"

git -C "${REPO_ROOT}" diff --binary \
  "${AFD_PLUGIN_BASE_COMMIT}" "${AFD_PLUGIN_COMMIT}" -- \
  >"${package_root}/patches/afd-plugin-66ec72f-to-official.patch"
git -C "${REPO_ROOT}" diff --binary \
  "${AFD_PLUGIN_GUIDE_BASE_COMMIT}" "${AFD_PLUGIN_COMMIT}" -- \
  >"${package_root}/patches/afd-plugin-1720b71-to-official.patch"
git -C "${REPO_ROOT}" diff --binary \
  "${AFD_PLUGIN_R2_COMMIT}" "${AFD_PLUGIN_COMMIT}" -- \
  >"${package_root}/patches/afd-plugin-7e58cc5-to-official.patch"

{
  printf 'VLLM_COMMIT=%q\n' "${VLLM_COMMIT}"
  printf 'VLLM_TREE=%q\n' "${VLLM_TREE}"
  printf 'VLLM_ASCEND_BASE_COMMIT=%q\n' "${VLLM_ASCEND_BASE_COMMIT}"
  printf 'VLLM_ASCEND_BASE_TREE=%q\n' "${VLLM_ASCEND_BASE_TREE}"
  printf 'VLLM_ASCEND_REVERTABLE_COMMIT=%q\n' "${VLLM_ASCEND_REVERTABLE_COMMIT}"
  printf 'AFD_PLUGIN_BASE_COMMIT=%q\n' "${AFD_PLUGIN_BASE_COMMIT}"
  printf 'AFD_PLUGIN_BASE_TREE=%q\n' "${AFD_PLUGIN_BASE_TREE}"
  printf 'AFD_PLUGIN_GUIDE_BASE_COMMIT=%q\n' "${AFD_PLUGIN_GUIDE_BASE_COMMIT}"
  printf 'AFD_PLUGIN_GUIDE_BASE_TREE=%q\n' "${AFD_PLUGIN_GUIDE_BASE_TREE}"
  printf 'AFD_PLUGIN_R2_COMMIT=%q\n' "${AFD_PLUGIN_R2_COMMIT}"
  printf 'AFD_PLUGIN_R2_TREE=%q\n' "${AFD_PLUGIN_R2_TREE}"
  printf 'AFD_PLUGIN_TARGET_SOURCE_COMMIT=%q\n' "${AFD_PLUGIN_COMMIT}"
  printf 'AFD_PLUGIN_TARGET_TREE=%q\n' "${AFD_PLUGIN_TARGET_TREE}"
} >"${package_root}/VERSION.env"

(
  cd "${package_root}"
  sha256sum \
    README_ZH.md \
    DEEPSEEK_V4_AFD_PHASE1_A5_VALIDATION_GUIDE_ZH.md \
    VERSION.env \
    install.sh \
    patches/afd-plugin-1720b71-to-official.patch \
    patches/afd-plugin-66ec72f-to-official.patch \
    patches/afd-plugin-7e58cc5-to-official.patch \
    >SHA256SUMS
)

tar -C "${staging_root}" -czf "${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}"
sha256sum "${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz"
