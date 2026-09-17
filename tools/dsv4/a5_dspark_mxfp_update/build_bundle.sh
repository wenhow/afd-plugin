#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace}"
VLLM_ASCEND_SOURCE_ROOT="${VLLM_ASCEND_SOURCE_ROOT:-${WORKSPACE_ROOT}/code/vllm-ascend-rfc-vllm-cann-3da28f941}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/delivery}"
PACKAGE_NAME="dsv4-a5-dspark-compressed-mxfp-fix-20260917"
VLLM_ASCEND_BASE_COMMIT=3da28f9414583d2d0b672a8f06d1fae142404bda
VLLM_ASCEND_PATCH_COMMIT=18a0709c88a6c1abed792f0f071bb0f9e8a5fc07

[[ "$(git -C "${VLLM_ASCEND_SOURCE_ROOT}" rev-parse HEAD)" == "${VLLM_ASCEND_PATCH_COMMIT}" ]]
[[ -z "$(git -C "${VLLM_ASCEND_SOURCE_ROOT}" status --short --untracked-files=all)" ]]
[[ -z "$(git -C "${REPO_ROOT}" status --short --untracked-files=all)" ]]
AFD_PLUGIN_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"

staging_root="$(mktemp -d)"
trap 'rm -rf "${staging_root}"' EXIT
package_root="${staging_root}/${PACKAGE_NAME}"
mkdir -p "${package_root}/payload" "${OUTPUT_DIR}"

cp "${SCRIPT_DIR}/install.sh" "${package_root}/install.sh"
cp "${SCRIPT_DIR}/config.env.example" "${package_root}/config.env"
cp "${SCRIPT_DIR}/README_ZH.md" "${package_root}/README_ZH.md"
cp "${REPO_ROOT}/docs/npu/DEEPSEEK_V4_AFD_PHASE1_A5_VALIDATION_GUIDE_ZH.md" "${package_root}/"

git -C "${VLLM_ASCEND_SOURCE_ROOT}" bundle create \
  "${package_root}/payload/vllm-ascend.bundle" \
  fix/dsv4-compressed-mxfp-detection
git -C "${REPO_ROOT}" bundle create \
  "${package_root}/payload/afd-plugin.bundle" \
  feat/dsv4-afd-graph-u2-multistream-all-on-v1

{
  printf 'VLLM_ASCEND_BASE_COMMIT=%q\n' "${VLLM_ASCEND_BASE_COMMIT}"
  printf 'VLLM_ASCEND_PATCH_COMMIT=%q\n' "${VLLM_ASCEND_PATCH_COMMIT}"
  printf 'AFD_PLUGIN_COMMIT=%q\n' "${AFD_PLUGIN_COMMIT}"
  printf 'MODEL_CONFIG_SHA256=%q\n' db3e4addebd459d7cc67b09790abf147edb963e1dd49dcf90c7332bbdb8bee26
  printf 'MODEL_INDEX_SHA256=%q\n' 98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b
} >"${package_root}/VERSION.env"

(
  cd "${package_root}"
  sha256sum \
    README_ZH.md \
    DEEPSEEK_V4_AFD_PHASE1_A5_VALIDATION_GUIDE_ZH.md \
    VERSION.env \
    config.env \
    install.sh \
    payload/afd-plugin.bundle \
    payload/vllm-ascend.bundle \
    >SHA256SUMS
)

tar -C "${staging_root}" -czf "${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}"
sha256sum "${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz"
