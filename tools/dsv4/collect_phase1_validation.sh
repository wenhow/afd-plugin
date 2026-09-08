#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 <output.tar.gz> <validation-root> [validation-root ...]" >&2
  exit 2
fi

OUTPUT_PATH="$1"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAX_FILE_BYTES="${PHASE1_EVIDENCE_MAX_FILE_BYTES:-4194304}"
LOG_TAIL_BYTES="${PHASE1_EVIDENCE_LOG_TAIL_BYTES:-262144}"

[[ "${MAX_FILE_BYTES}" =~ ^[1-9][0-9]*$ ]] \
  || { echo "PHASE1_EVIDENCE_MAX_FILE_BYTES must be positive" >&2; exit 2; }
[[ "${LOG_TAIL_BYTES}" =~ ^[1-9][0-9]*$ ]] \
  || { echo "PHASE1_EVIDENCE_LOG_TAIL_BYTES must be positive" >&2; exit 2; }

temp_root="$(mktemp -d)"
cleanup() {
  rm -rf "${temp_root}"
}
trap cleanup EXIT
payload_root="${temp_root}/dsv4-phase1-evidence"
mkdir -p "${payload_root}/provenance" "${payload_root}/runs"

{
  printf 'collected_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'repo_root=%s\n' "${REPO_ROOT}"
  printf 'afd_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || printf unavailable)"
  printf 'afd_status_begin\n'
  git -C "${REPO_ROOT}" status --short 2>/dev/null || true
  printf 'afd_status_end\n'
  printf 'vllm_commit=%s\n' "$(git -C "${DSV4_VLLM_ROOT:-/nonexistent}" rev-parse HEAD 2>/dev/null || printf unavailable)"
  printf 'vllm_ascend_commit=%s\n' "$(git -C "${DSV4_VLLM_ASCEND_ROOT:-/nonexistent}" rev-parse HEAD 2>/dev/null || printf unavailable)"
  printf 'cann_root=%s\n' "$(readlink -f "${DSV4_CANN_ROOT:-/nonexistent}" 2>/dev/null || printf unavailable)"
} >"${payload_root}/provenance/runtime.env"

env | LC_ALL=C sort \
  | grep -E '^(ASCEND|ATB|CANN|DSV4|HCCL|LD_LIBRARY_PATH|PYTHONPATH|SOC_VERSION|VIRTUAL_ENV|VLLM_PLUGINS)=' \
  >"${payload_root}/provenance/environment.txt" || true
npu-smi info >"${payload_root}/provenance/npu-smi.txt" 2>&1 || true
if [[ -x "${DSV4_RUNTIME_VENV:-/nonexistent}/bin/python" ]]; then
  "${DSV4_RUNTIME_VENV}/bin/python" -m pip freeze \
    >"${payload_root}/provenance/pip-freeze.txt" 2>&1 || true
fi

run_index=0
for input_root in "$@"; do
  [[ -d "${input_root}" ]] \
    || { echo "Validation root is not a directory: ${input_root}" >&2; exit 2; }
  input_root="$(readlink -f "${input_root}")"
  run_index=$((run_index + 1))
  run_name="$(basename "${input_root}")"
  destination_root="${payload_root}/runs/$(printf '%02d' "${run_index}")-${run_name}"
  mkdir -p "${destination_root}"
  printf '%s\n' "${input_root}" >"${destination_root}/SOURCE_PATH"

  while IFS= read -r -d '' source_file; do
    relative_path="${source_file#"${input_root}"/}"
    destination_file="${destination_root}/${relative_path}"
    mkdir -p "$(dirname "${destination_file}")"
    file_size="$(stat -c %s "${source_file}")"
    case "${source_file}" in
      *.log|*.out|*.stderr)
        tail -c "${LOG_TAIL_BYTES}" "${source_file}" >"${destination_file}.tail"
        ;;
      *)
        if (( file_size <= MAX_FILE_BYTES )); then
          cp -a "${source_file}" "${destination_file}"
        else
          tail -c "${LOG_TAIL_BYTES}" "${source_file}" >"${destination_file}.tail"
        fi
        ;;
    esac
  done < <(
    find "${input_root}" -type f \
      ! -path '*/profiles/*' \
      ! -path '*/ASCEND_PROFILER_OUTPUT/*' \
      ! -path '*/PROF_*/*' \
      -print0
  )
done

(
  cd "${payload_root}"
  find . -type f ! -name SHA256SUMS -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum >SHA256SUMS
)
mkdir -p "$(dirname "${OUTPUT_PATH}")"
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -czf "${OUTPUT_PATH}" -C "${temp_root}" "$(basename "${payload_root}")"
sha256sum "${OUTPUT_PATH}" >"${OUTPUT_PATH}.sha256"
printf '%s\n%s\n' "${OUTPUT_PATH}" "${OUTPUT_PATH}.sha256"
