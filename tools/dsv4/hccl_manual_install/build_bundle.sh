#!/usr/bin/env bash
set -euo pipefail

BUNDLE_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AFD_REPO_ROOT="$(cd "${BUNDLE_SOURCE_DIR}/../../.." && pwd)"
WORKSPACE_CODE_ROOT="$(dirname "${AFD_REPO_ROOT}")"
VLLM_SOURCE_ROOT="${VLLM_SOURCE_ROOT:-${WORKSPACE_CODE_ROOT}/vllm-release-v0.23.0}"
VLLM_ASCEND_SOURCE_ROOT="${VLLM_ASCEND_SOURCE_ROOT:-${WORKSPACE_CODE_ROOT}/vllm-ascend-rfc-vllm-cann}"
OUTPUT_DIR="${1:-/mnt/workspace/artifacts}"

VLLM_COMMIT=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
VLLM_ASCEND_COMMIT=3da28f9414583d2d0b672a8f06d1fae142404bda
AFD_SOURCE_COMMIT="${AFD_SOURCE_COMMIT:-9db1fb981f5262986e7ea05938e9c6ed57b5a885}"
AFD_RELEASE_REF="${AFD_RELEASE_REF:-HEAD}"
AFD_SNAPSHOT_ID="${AFD_SNAPSHOT_ID:-dsv4-afd-v023-cann900-phase1-external-v1}"
CONFIG_PROFILE="${CONFIG_PROFILE:-generic}"
INCLUDE_AFD_SEED_BUNDLE="${INCLUDE_AFD_SEED_BUNDLE:-}"
AFD_SEED_COMMIT="${AFD_SEED_COMMIT:-}"
AFD_TARGET_COMMIT="$(git -C "${AFD_REPO_ROOT}" rev-parse "${AFD_RELEASE_REF}^{commit}" 2>/dev/null)" \
  || { echo "afd-plugin release ref does not exist: ${AFD_RELEASE_REF}" >&2; exit 2; }
[[ -z "$(git -C "${AFD_REPO_ROOT}" status --short)" ]] \
  || { echo "afd-plugin worktree must be clean before packaging" >&2; exit 2; }
git -C "${AFD_REPO_ROOT}" merge-base --is-ancestor \
  "${AFD_SOURCE_COMMIT}" "${AFD_TARGET_COMMIT}" \
  || { echo "afd-plugin source commit is not an ancestor of the target" >&2; exit 2; }
AFD_SOURCE_TREE="$(git -C "${AFD_REPO_ROOT}" show -s --format=%T "${AFD_SOURCE_COMMIT}")"
AFD_TARGET_TREE="$(git -C "${AFD_REPO_ROOT}" show -s --format=%T "${AFD_TARGET_COMMIT}")"
INCLUDE_SOURCES="${INCLUDE_SOURCES:-0}"

case "${INCLUDE_SOURCES}" in
  1|true|TRUE|yes|YES|on|ON)
    include_sources=1
    package_flavor="with-sources"
    ;;
  0|false|FALSE|no|NO|off|OFF)
    include_sources=0
    package_flavor="slim"
    ;;
  *)
    echo "INCLUDE_SOURCES must be 0 or 1: ${INCLUDE_SOURCES}" >&2
    exit 2
    ;;
esac

case "${CONFIG_PROFILE}" in
  generic)
    INCLUDE_AFD_SEED_BUNDLE="${INCLUDE_AFD_SEED_BUNDLE:-0}"
    profile_suffix=""
    ;;
  a5-new-install)
    INCLUDE_AFD_SEED_BUNDLE="${INCLUDE_AFD_SEED_BUNDLE:-0}"
    profile_suffix="-${CONFIG_PROFILE}"
    if [[ "${AFD_SNAPSHOT_ID}" == "dsv4-afd-v023-cann900-phase1-external-v1" ]]; then
      AFD_SNAPSHOT_ID="dsv4-afd-v023-phase1-a5-external-v1"
    fi
    ;;
  dual-a3-reuse)
    INCLUDE_AFD_SEED_BUNDLE="${INCLUDE_AFD_SEED_BUNDLE:-1}"
    AFD_SEED_COMMIT="${AFD_SEED_COMMIT:-2164240b31efc8605bf84cc45afc628996669554}"
    profile_suffix="-${CONFIG_PROFILE}"
    ;;
  *)
    echo "Unsupported CONFIG_PROFILE: ${CONFIG_PROFILE}" >&2
    exit 2
    ;;
esac

timestamp="$(date +%Y%m%d_%H%M%S)"
package_name="dsv4-afd-hccl-manual-install-${package_flavor}${profile_suffix}-${timestamp}"

temp_root="$(mktemp -d)"
cleanup() {
  rm -rf "${temp_root}"
}
trap cleanup EXIT

payload_root="${temp_root}/${package_name}"
mkdir -p "${payload_root}/manifest"
cp -a "${BUNDLE_SOURCE_DIR}/." "${payload_root}/"
case "${CONFIG_PROFILE}" in
  generic)
    ;;
  a5-new-install)
    sed -i \
      -e 's|^CANN_ROOT=.*|CANN_ROOT="/CHANGE_ME/CANN_ROOT"|' \
      -e 's|^EXPECTED_CANN_VERSION=.*|EXPECTED_CANN_VERSION=""|' \
      -e 's|^SOC_VERSION=.*|SOC_VERSION="CHANGE_ME"|' \
      -e 's|^NIC_NAME=.*|NIC_NAME="CHANGE_ME"|' \
      "${payload_root}/config.env.example"
    ;;
  dual-a3-reuse)
    sed -i \
      -e 's|^INSTALL_ROOT=.*|INSTALL_ROOT="/data/z00569729/run/dsv4-afd-phase1-install"|' \
      -e 's|^CODE_ROOT=.*|CODE_ROOT="/data/z00569729/code"|' \
      -e 's|^VENV_ROOT=.*|VENV_ROOT="${CODE_ROOT}/.venvs/afd-v023-vllm-cann"|' \
      -e 's|^STATE_ROOT=.*|STATE_ROOT="${INSTALL_ROOT}/state"|' \
      -e 's|^LOG_ROOT=.*|LOG_ROOT="${INSTALL_ROOT}/logs"|' \
      -e 's|^AFD_PLUGIN_ROOT=.*|AFD_PLUGIN_ROOT="${CODE_ROOT}/afd-plugin-phase1-a5"|' \
      -e 's|^CANN_ROOT=.*|CANN_ROOT="/usr/local/Ascend/cann-9.0.0"|' \
      -e 's|^EXPECTED_CANN_VERSION=.*|EXPECTED_CANN_VERSION="9.0.0"|' \
      -e 's|^MODEL_PATH=.*|MODEL_PATH="/data/models/DeepSeek-V4-Flash-w8a8-mtp"|' \
      -e 's|^PYTHON_BIN=.*|PYTHON_BIN="${VENV_ROOT}/bin/python"|' \
      -e 's|^NIC_NAME=.*|NIC_NAME="enp23s0f3"|' \
      -e 's|^MAX_NUM_BATCHED_TOKENS=.*|MAX_NUM_BATCHED_TOKENS="4096"|' \
      -e 's|^MAX_NUM_SEQS=.*|MAX_NUM_SEQS="16"|' \
      -e 's|^REUSE_SOURCES=.*|REUSE_SOURCES="1"|' \
      -e 's|^USE_AFD_SEED_BUNDLE=.*|USE_AFD_SEED_BUNDLE="1"|' \
      -e 's|^AFD_SEED_ROOT=.*|AFD_SEED_ROOT="${CODE_ROOT}/afd-plugin"|' \
      -e "s|^AFD_SEED_COMMIT=.*|AFD_SEED_COMMIT=\"${AFD_SEED_COMMIT}\"|" \
      -e 's|^REUSE_VENV=.*|REUSE_VENV="1"|' \
      -e 's|^INSTALL_PYTHON_DEPS=.*|INSTALL_PYTHON_DEPS="0"|' \
      -e 's|^INSTALL_UPSTREAM_STACK=.*|INSTALL_UPSTREAM_STACK="0"|' \
      "${payload_root}/config.env.example"
    ;;
esac
cp "${payload_root}/config.env.example" "${payload_root}/config.env"
cp "${AFD_REPO_ROOT}/docs/npu/DEEPSEEK_V4_AFD_PHASE1_A5_MULTI_NODE_VALIDATION_GUIDE_ZH.md" \
  "${payload_root}/PHASE1_A5_MULTI_NODE_VALIDATION_GUIDE_ZH.md"

if [[ "${CONFIG_PROFILE}" == "dual-a3-reuse" ]]; then
  cp "${AFD_REPO_ROOT}/tools/dsv4/mooncake_pd_manual/config.env.example" \
    "${payload_root}/DUAL_A3_PD_COMMON.env.example"
  sed -i \
    -e 's|^AFD_PLUGIN_ROOT=.*|AFD_PLUGIN_ROOT="${CODE_ROOT}/afd-plugin-phase1-a5"|' \
    -e "s|^AFD_PD_COMMIT=.*|AFD_PD_COMMIT=\"${AFD_TARGET_COMMIT}\"|" \
    "${payload_root}/DUAL_A3_PD_COMMON.env.example"
fi

case "${INCLUDE_AFD_SEED_BUNDLE}" in
  1|true|TRUE|yes|YES|on|ON)
    [[ -n "${AFD_SEED_COMMIT}" ]] \
      || { echo "AFD_SEED_COMMIT is required for a seed bundle" >&2; exit 2; }
    git -C "${AFD_REPO_ROOT}" merge-base --is-ancestor \
      "${AFD_SEED_COMMIT}" "${AFD_TARGET_COMMIT}" \
      || { echo "AFD seed commit is not an ancestor of the target" >&2; exit 2; }
    git -C "${AFD_REPO_ROOT}" bundle create \
      "${payload_root}/manifest/afd-plugin-from-seed.bundle" \
      "${AFD_RELEASE_REF}" "^${AFD_SEED_COMMIT}"
    BUNDLE_INCLUDES_AFD_SEED=1
    AFD_SEED_BUNDLE_SHA256="$(sha256sum \
      "${payload_root}/manifest/afd-plugin-from-seed.bundle" | awk '{print $1}')"
    ;;
  0|false|FALSE|no|NO|off|OFF)
    BUNDLE_INCLUDES_AFD_SEED=0
    AFD_SEED_BUNDLE_SHA256=
    ;;
  *)
    echo "INCLUDE_AFD_SEED_BUNDLE must be 0 or 1" >&2
    exit 2
    ;;
esac

if (( include_sources == 1 )); then
  sed -i 's/^USE_BUNDLED_SOURCES=.*/USE_BUNDLED_SOURCES="1"/' \
    "${payload_root}/config.env"
fi

patch_file="${payload_root}/manifest/afd-plugin-phase1.patch"
git -C "${AFD_REPO_ROOT}" diff --binary \
  "${AFD_SOURCE_COMMIT}..${AFD_TARGET_COMMIT}" >"${patch_file}"
AFD_PATCH_SHA256="$(sha256sum "${patch_file}" | awk '{print $1}')"

# Prove that the portable patch reconstructs the exact release tree.
verify_stage="${temp_root}/verify-afd-patch"
mkdir -p "${verify_stage}"
git -C "${AFD_REPO_ROOT}" archive "${AFD_SOURCE_COMMIT}" \
  | tar -xf - -C "${verify_stage}"
git -C "${verify_stage}" init -q
git -C "${verify_stage}" add -A
git -C "${verify_stage}" apply --index "${patch_file}"
[[ "$(git -C "${verify_stage}" write-tree)" == "${AFD_TARGET_TREE}" ]] \
  || { echo "afd-plugin patch does not reconstruct ${AFD_RELEASE_REF}" >&2; exit 2; }

cat >"${payload_root}/manifest/versions.env" <<EOF
VLLM_COMMIT="${VLLM_COMMIT}"
VLLM_ASCEND_COMMIT="${VLLM_ASCEND_COMMIT}"
AFD_SOURCE_COMMIT="${AFD_SOURCE_COMMIT}"
AFD_SOURCE_TREE="${AFD_SOURCE_TREE}"
AFD_TARGET_COMMIT="${AFD_TARGET_COMMIT}"
AFD_TARGET_TREE="${AFD_TARGET_TREE}"
AFD_PATCH_SHA256="${AFD_PATCH_SHA256}"
AFD_SNAPSHOT_ID="${AFD_SNAPSHOT_ID}"
BUNDLE_INCLUDES_SOURCES="${include_sources}"
BUNDLE_CONFIG_PROFILE="${CONFIG_PROFILE}"
BUNDLE_INCLUDES_AFD_SEED="${BUNDLE_INCLUDES_AFD_SEED}"
AFD_SEED_COMMIT="${AFD_SEED_COMMIT}"
AFD_SEED_BUNDLE_SHA256="${AFD_SEED_BUNDLE_SHA256}"
EOF

if (( include_sources == 1 )); then
  [[ "$(git -C "${VLLM_SOURCE_ROOT}" rev-parse HEAD)" == "${VLLM_COMMIT}" ]] \
    || { echo "vLLM source commit mismatch" >&2; exit 2; }
  [[ "$(git -C "${VLLM_ASCEND_SOURCE_ROOT}" rev-parse HEAD)" == "${VLLM_ASCEND_COMMIT}" ]] \
    || { echo "vLLM-Ascend source commit mismatch" >&2; exit 2; }
  [[ -z "$(git -C "${VLLM_SOURCE_ROOT}" status --short)" ]] \
    || { echo "vLLM source worktree is dirty" >&2; exit 2; }
  [[ -z "$(git -C "${VLLM_ASCEND_SOURCE_ROOT}" status --short)" ]] \
    || { echo "vLLM-Ascend source worktree is dirty" >&2; exit 2; }

  mkdir -p "${payload_root}/sources"
  git -C "${VLLM_SOURCE_ROOT}" archive \
    --format=tar.gz \
    --output="${payload_root}/sources/vllm-release-v0.23.0.tar.gz" \
    "${VLLM_COMMIT}"

  ascend_stage="${temp_root}/ascend-source"
  mkdir -p "${ascend_stage}"
  git -C "${VLLM_ASCEND_SOURCE_ROOT}" archive "${VLLM_ASCEND_COMMIT}" \
    | tar -xf - -C "${ascend_stage}"
  mkdir -p "${ascend_stage}/csrc/third_party/catlass"
  git -C "${VLLM_ASCEND_SOURCE_ROOT}/csrc/third_party/catlass" archive HEAD \
    | tar -xf - -C "${ascend_stage}/csrc/third_party/catlass"
  mkdir -p "${ascend_stage}/csrc/third_party/catlass/3rdparty/googletest"
  git -C "${VLLM_ASCEND_SOURCE_ROOT}/csrc/third_party/catlass/3rdparty/googletest" \
    archive HEAD \
    | tar -xf - -C \
      "${ascend_stage}/csrc/third_party/catlass/3rdparty/googletest"
  tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
    -czf "${payload_root}/sources/vllm-ascend-rfc-vllm-cann.tar.gz" \
    -C "${ascend_stage}" .

  afd_stage="${temp_root}/afd-source"
  mkdir -p "${afd_stage}"
  git -C "${AFD_REPO_ROOT}" archive "${AFD_TARGET_COMMIT}" \
    | tar -xf - -C "${afd_stage}"
  tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
    -czf "${payload_root}/sources/afd-plugin-phase1-snapshot.tar.gz" \
    -C "${afd_stage}" .
fi

if [[ -n "${INCLUDE_WHEELHOUSE:-}" ]]; then
  [[ -d "${INCLUDE_WHEELHOUSE}" ]] \
    || { echo "INCLUDE_WHEELHOUSE is not a directory" >&2; exit 2; }
  cp -a "${INCLUDE_WHEELHOUSE}" "${payload_root}/wheelhouse"
fi

(
  cd "${payload_root}"
  find . -type f \
    ! -path './manifest/SHA256SUMS' \
    ! -path './config.env' \
    -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum \
    >manifest/SHA256SUMS
)

mkdir -p "${OUTPUT_DIR}"
archive_path="${OUTPUT_DIR}/${package_name}.tar.gz"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -czf "${archive_path}" \
  -C "${temp_root}" "${package_name}"
(
  cd "${OUTPUT_DIR}"
  sha256sum "$(basename "${archive_path}")" \
    >"$(basename "${archive_path}").sha256"
)

printf '%s\n' "${archive_path}"
printf '%s\n' "${archive_path}.sha256"
