#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${REPO_ROOT}/recipe/npu/deepseek_v4/common/run_validation.py"
ACTION="${1:-help}"
if (( $# > 0 )); then
  shift
fi

CASES=(
  a8f8_eager_u1_mtp_off
  a8f8_eager_u1_n1
  a8f8_eager_u2_n2
  a8f8_graph_u1_n2
  a8f8_graph_u2_n3
  a4f8_eager_u1_n2
  a4f8_graph_u2_n3
  a8f4_eager_u1_n2
  a8f4_graph_u2_n3
)

usage() {
  cat <<'EOF'
Usage:
  bash tools/dsv4/run_phase1_a5_matrix.sh list
  bash tools/dsv4/run_phase1_a5_matrix.sh preflight
  bash tools/dsv4/run_phase1_a5_matrix.sh f0 [case ...]
  bash tools/dsv4/run_phase1_a5_matrix.sh f1 [case ...]

Required environment:
  PHASE1_GOLDEN       Same-stack native golden_results.json
  PHASE1_OUTPUT_BASE  Fresh output root (default includes a timestamp)

F0 runs one cold cycle, one validation round, and no idle-resume wait.
F1 runs two cold cycles, three rounds, batch 1/8/32, and a 30-minute
idle-resume gate by default. Override PHASE1_F1_IDLE_SECONDS only for diagnosis.
EOF
}

die() {
  printf '[phase1-a5] ERROR: %s\n' "$*" >&2
  exit 2
}

contains_case() {
  local requested="$1" candidate
  for candidate in "${CASES[@]}"; do
    [[ "${candidate}" == "${requested}" ]] && return 0
  done
  return 1
}

activate_and_audit() {
  source "${SCRIPT_DIR}/activate_v023_vllm_cann_runtime.sh"
  export PYTHONPATH="${REPO_ROOT}:${DSV4_VLLM_ROOT}:${DSV4_VLLM_ASCEND_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  local expected_vllm=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
  local expected_ascend=3da28f9414583d2d0b672a8f06d1fae142404bda
  [[ "$(git -C "${DSV4_VLLM_ROOT}" rev-parse HEAD)" == "${expected_vllm}" ]] \
    || die "vLLM commit mismatch"
  [[ "$(git -C "${DSV4_VLLM_ASCEND_ROOT}" rev-parse HEAD)" == "${expected_ascend}" ]] \
    || die "vLLM-Ascend commit mismatch"
  [[ -z "$(git -C "${DSV4_VLLM_ROOT}" status --short)" ]] \
    || die "vLLM worktree is dirty"
  [[ -z "$(git -C "${DSV4_VLLM_ASCEND_ROOT}" status --short)" ]] \
    || die "vLLM-Ascend worktree is dirty"
  [[ -z "$(git -C "${REPO_ROOT}" status --short)" ]] \
    || die "afd-plugin worktree is dirty"
  [[ "$(readlink -f "${DSV4_CANN_ROOT}")" == *cann-9.0.0* ]] \
    || die "DSV4_CANN_ROOT is not the fixed CANN 9.0.0 tree"
  [[ -x "${DSV4_RUNTIME_VENV}/bin/python" ]] || die "Runtime venv is missing"
  [[ -f "${MODEL_PATH:-/nonexistent}/config.json" ]] || die "MODEL_PATH is invalid"
  [[ -f "${PHASE1_GOLDEN:-/nonexistent}" ]] || die "PHASE1_GOLDEN is invalid"
  [[ -f "${RUNNER}" ]] || die "Validation runner is missing"

  local imported_roots=()
  mapfile -t imported_roots < <(
    cd "${REPO_ROOT}"
    "${DSV4_RUNTIME_VENV}/bin/python" - <<'PY'
from pathlib import Path

import afd_plugin
import vllm
import vllm_ascend

print(Path(afd_plugin.__file__).resolve().parent.parent)
print(Path(vllm.__file__).resolve().parent.parent)
print(Path(vllm_ascend.__file__).resolve().parent.parent)
PY
  )
  [[ "${imported_roots[0]:-}" == "$(readlink -f "${REPO_ROOT}")" ]] \
    || die "afd-plugin imports from ${imported_roots[0]:-unknown}, not ${REPO_ROOT}"
  [[ "${imported_roots[1]:-}" == "$(readlink -f "${DSV4_VLLM_ROOT}")" ]] \
    || die "vLLM imports from ${imported_roots[1]:-unknown}, not ${DSV4_VLLM_ROOT}"
  [[ "${imported_roots[2]:-}" == "$(readlink -f "${DSV4_VLLM_ASCEND_ROOT}")" ]] \
    || die "vLLM-Ascend imports from ${imported_roots[2]:-unknown}, not ${DSV4_VLLM_ASCEND_ROOT}"

  local npu_process_count
  npu_process_count="$(npu-smi info | awk '
    /\| NPU +Chip +\| Process id/ {in_process_table=1; next}
    in_process_table && /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*\|[[:space:]]*[0-9]+/ {count++}
    END {print count + 0}
  ')"
  (( npu_process_count == 0 )) \
    || die "Detected ${npu_process_count} existing NPU processes"
}

case_arguments() {
  local case_name="$1"
  CASE_ARGS=(
    --connector P2pHcclAFDConnector
    --attention-max-num-batched-tokens 4096
  )
  case "${case_name}" in
    a8f8_eager_u1_mtp_off)
      CASE_ARGS+=(--attention-devices 0,1,2,3,4,5,6,7 --ffn-devices 8,9,10,11,12,13,14,15 --ffn-max-num-batched-tokens 4096 --execution-mode eager --u-batches 1)
      ;;
    a8f8_eager_u1_n1)
      CASE_ARGS+=(--attention-devices 0,1,2,3,4,5,6,7 --ffn-devices 8,9,10,11,12,13,14,15 --ffn-max-num-batched-tokens 4096 --execution-mode eager --u-batches 1 --enable-mtp --mtp-num-speculative-tokens 1 --mtp-draft-execution eager)
      ;;
    a8f8_eager_u2_n2)
      CASE_ARGS+=(--attention-devices 0,1,2,3,4,5,6,7 --ffn-devices 8,9,10,11,12,13,14,15 --ffn-max-num-batched-tokens 4096 --execution-mode eager --u-batches 2 --enable-mtp --mtp-num-speculative-tokens 2 --mtp-draft-execution eager)
      ;;
    a8f8_graph_u1_n2)
      CASE_ARGS+=(--attention-devices 0,1,2,3,4,5,6,7 --ffn-devices 8,9,10,11,12,13,14,15 --ffn-max-num-batched-tokens 4096 --execution-mode full-decode-only --u-batches 1 --enable-mtp --mtp-num-speculative-tokens 2 --mtp-draft-execution eager)
      ;;
    a8f8_graph_u2_n3)
      CASE_ARGS+=(--attention-devices 0,1,2,3,4,5,6,7 --ffn-devices 8,9,10,11,12,13,14,15 --ffn-max-num-batched-tokens 4096 --execution-mode full-decode-only --u-batches 2 --enable-mtp --mtp-num-speculative-tokens 3 --mtp-draft-execution graph)
      ;;
    a4f8_eager_u1_n2)
      CASE_ARGS+=(--attention-devices 0,1,2,3 --ffn-devices 4,5,6,7,8,9,10,11 --ffn-max-num-batched-tokens 2048 --execution-mode eager --u-batches 1 --enable-mtp --mtp-num-speculative-tokens 2 --mtp-draft-execution eager)
      ;;
    a4f8_graph_u2_n3)
      CASE_ARGS+=(--attention-devices 0,1,2,3 --ffn-devices 4,5,6,7,8,9,10,11 --ffn-max-num-batched-tokens 2048 --execution-mode full-decode-only --u-batches 2 --enable-mtp --mtp-num-speculative-tokens 3 --mtp-draft-execution graph)
      ;;
    a8f4_eager_u1_n2)
      CASE_ARGS+=(--attention-devices 0,1,2,3,4,5,6,7 --ffn-devices 8,9,10,11 --ffn-max-num-batched-tokens 8192 --execution-mode eager --u-batches 1 --enable-mtp --mtp-num-speculative-tokens 2 --mtp-draft-execution eager)
      ;;
    a8f4_graph_u2_n3)
      CASE_ARGS+=(--attention-devices 0,1,2,3,4,5,6,7 --ffn-devices 8,9,10,11 --ffn-max-num-batched-tokens 8192 --execution-mode full-decode-only --u-batches 2 --enable-mtp --mtp-num-speculative-tokens 3 --mtp-draft-execution graph)
      ;;
    *) die "Unknown case: ${case_name}" ;;
  esac
}

run_matrix() {
  local phase="$1"
  shift
  local cycles rounds idle_seconds output_root case_name
  if [[ "${phase}" == "f0" ]]; then
    cycles=1
    rounds=1
    idle_seconds=0
  else
    cycles=2
    rounds=3
    idle_seconds="${PHASE1_F1_IDLE_SECONDS:-1800}"
  fi
  output_root="${PHASE1_OUTPUT_BASE:-/data/validation/dsv4-phase1-a5-$(date +%Y%m%d_%H%M%S)}/${phase}"
  [[ ! -e "${output_root}" ]] || die "Output root already exists: ${output_root}"
  mkdir -p "${output_root}"
  {
    printf 'phase=%s\n' "${phase}"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'afd_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    printf 'cann_root=%s\n' "$(readlink -f "${DSV4_CANN_ROOT}")"
  } >"${output_root}/matrix.env"
  npu-smi info >"${output_root}/npu-before.txt"

  if (( $# == 0 )); then
    set -- "${CASES[@]}"
  fi
  for case_name in "$@"; do
    contains_case "${case_name}" || die "Unknown case: ${case_name}"
    case_arguments "${case_name}"
    "${DSV4_RUNTIME_VENV}/bin/python" "${RUNNER}" \
      --output-dir "${output_root}/${case_name}" \
      --golden "${PHASE1_GOLDEN}" \
      --cycles "${cycles}" \
      --idle-seconds "${idle_seconds}" \
      --rounds "${rounds}" \
      --batch-sizes 1 8 32 \
      "${CASE_ARGS[@]}"
  done
  npu-smi info >"${output_root}/npu-after.txt"
  printf '[phase1-a5] completed: %s\n' "${output_root}"
}

case "${ACTION}" in
  help|-h|--help) usage ;;
  list) printf '%s\n' "${CASES[@]}" ;;
  preflight) activate_and_audit; printf '[phase1-a5] preflight passed\n' ;;
  f0|f1) activate_and_audit; run_matrix "${ACTION}" "$@" ;;
  *) usage; die "Unknown action: ${ACTION}" ;;
esac
