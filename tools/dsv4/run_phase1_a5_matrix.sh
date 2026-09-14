#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${REPO_ROOT}/recipe/npu/deepseek_v4/common/run_validation.py"
ACTION="${1:-help}"
if (( $# > 0 )); then
  shift
fi

EXACT_CASES=(
  a4f4_eager_u1_mtp_off
  a4f4_eager_u1_n1
  a4f4_eager_u2_n2
  a4f4_graph_u1_n2
  a4f4_graph_u2_n3
  a2f4_eager_u1_n2
  a2f4_graph_u2_n3
  a4f2_eager_u1_n2
  a4f2_graph_u2_n3
)
SMOKE_CASES=(
  a4f4_eager_u1_mtp_off
  a4f4_graph_u2_mtp_off
  a2f4_graph_u2_mtp_off
  a4f2_graph_u2_mtp_off
)

usage() {
  cat <<'EOF'
Usage:
  bash tools/dsv4/run_phase1_a5_matrix.sh list
  bash tools/dsv4/run_phase1_a5_matrix.sh list-smoke
  bash tools/dsv4/run_phase1_a5_matrix.sh list-deferred-exact
  bash tools/dsv4/run_phase1_a5_matrix.sh preflight-native
  bash tools/dsv4/run_phase1_a5_matrix.sh preflight [case ...]
  bash tools/dsv4/run_phase1_a5_matrix.sh preflight-smoke [case ...]
  bash tools/dsv4/run_phase1_a5_matrix.sh preflight-deferred-exact [case ...]
  bash tools/dsv4/run_phase1_a5_matrix.sh smoke [case ...]
  bash tools/dsv4/run_phase1_a5_matrix.sh f0 [case ...]
  bash tools/dsv4/run_phase1_a5_matrix.sh f1 [case ...]

Required environment for f0/f1 only:
  PHASE1_GOLDEN_ROOT  Five same-stack, path-matched native controls
  PHASE1_ENABLE_DEFERRED_EXACT=1  Explicit acknowledgement of deferred scope

Required environment for smoke/f0/f1:
  PHASE1_OUTPUT_BASE  Fresh output root (default includes a timestamp)

Smoke is the MTP-off A5 phase-one functional gate. It runs two cold cycles,
batch 1/8/32, cancellation recovery, U2/log/cleanup gates, and no golden
comparison. F0/F1 retain the deferred standalone exact matrix, but are not
part of the current A5 gate; MTP validation resumes only with dSpark.
EOF
}

die() {
  printf '[phase1-a5] ERROR: %s\n' "$*" >&2
  exit 2
}

contains_exact_case() {
  local requested="$1" candidate
  for candidate in "${EXACT_CASES[@]}"; do
    [[ "${candidate}" == "${requested}" ]] && return 0
  done
  return 1
}

contains_smoke_case() {
  local requested="$1" candidate
  for candidate in "${SMOKE_CASES[@]}"; do
    [[ "${candidate}" == "${requested}" ]] && return 0
  done
  return 1
}

audit_stack() {
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
  [[ -f "${DSV4_CANN_ROOT}/set_env.sh" ]] \
    || die "DSV4_CANN_ROOT does not contain set_env.sh"
  [[ -x "${DSV4_RUNTIME_VENV}/bin/python" ]] || die "Runtime venv is missing"
  [[ -f "${MODEL_PATH:-/nonexistent}/config.json" ]] || die "MODEL_PATH is invalid"
  [[ -f "${RUNNER}" ]] || die "Validation runner is missing"
  local custom_ops_root="${DSV4_VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
  [[ -f "${custom_ops_root}/bin/set_env.bash" ]] \
    || die "vLLM-Ascend custom_transformer ops are not built in the fixed source tree"
  [[ -f "${custom_ops_root}/op_api/lib/libcust_opapi.so" ]] \
    || die "vLLM-Ascend custom_transformer op_api library is missing"

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

control_key_for_case() {
  case "$1" in
    a4f4_eager_u1_mtp_off) printf 'eager_mtp_off\n' ;;
    a4f4_eager_u1_n1) printf 'eager_mtp_n1\n' ;;
    a4f4_eager_u2_n2|a2f4_eager_u1_n2|a4f2_eager_u1_n2)
      printf 'eager_mtp_n2\n'
      ;;
    a4f4_graph_u1_n2) printf 'graph_target_draft_eager_mtp_n2\n' ;;
    a4f4_graph_u2_n3|a2f4_graph_u2_n3|a4f2_graph_u2_n3)
      printf 'graph_target_draft_graph_mtp_n3\n'
      ;;
    *) die "Unknown case: $1" ;;
  esac
}

golden_for_case() {
  local control_key
  control_key="$(control_key_for_case "$1")"
  printf '%s/%s/golden_results.json\n' "${PHASE1_GOLDEN_ROOT}" "${control_key}"
}

control_metadata_for_key() {
  case "$1" in
    eager_mtp_off)
      EXPECTED_TARGET_EXECUTION=eager
      EXPECTED_DRAFT_EXECUTION=off
      EXPECTED_ENABLE_MTP=0
      EXPECTED_MTP_TOKENS=0
      ;;
    eager_mtp_n1)
      EXPECTED_TARGET_EXECUTION=eager
      EXPECTED_DRAFT_EXECUTION=eager
      EXPECTED_ENABLE_MTP=1
      EXPECTED_MTP_TOKENS=1
      ;;
    eager_mtp_n2)
      EXPECTED_TARGET_EXECUTION=eager
      EXPECTED_DRAFT_EXECUTION=eager
      EXPECTED_ENABLE_MTP=1
      EXPECTED_MTP_TOKENS=2
      ;;
    graph_target_draft_eager_mtp_n2)
      EXPECTED_TARGET_EXECUTION=full-decode-only
      EXPECTED_DRAFT_EXECUTION=eager
      EXPECTED_ENABLE_MTP=1
      EXPECTED_MTP_TOKENS=2
      ;;
    graph_target_draft_graph_mtp_n3)
      EXPECTED_TARGET_EXECUTION=full-decode-only
      EXPECTED_DRAFT_EXECUTION=graph
      EXPECTED_ENABLE_MTP=1
      EXPECTED_MTP_TOKENS=3
      ;;
    *) die "Unknown control key: $1" ;;
  esac
}

validate_golden_for_case() {
  local case_name="$1" control_key golden_path cann_root
  control_key="$(control_key_for_case "${case_name}")"
  control_metadata_for_key "${control_key}"
  golden_path="$(golden_for_case "${case_name}")"
  cann_root="$(readlink -f "${DSV4_CANN_ROOT}")"
  [[ -f "${golden_path}" ]] || die "Missing native control: ${golden_path}"
  jq -e \
    --arg control_key "${control_key}" \
    --arg target_execution "${EXPECTED_TARGET_EXECUTION}" \
    --arg draft_execution "${EXPECTED_DRAFT_EXECUTION}" \
    --arg enable_mtp "${EXPECTED_ENABLE_MTP}" \
    --arg mtp_tokens "${EXPECTED_MTP_TOKENS}" \
    --arg cann_root "${cann_root}" '
    .passed == true
    and .rounds >= 3
    and .prompt_count == 10
    and (.mismatched_prompt_indices | length == 0)
    and .metadata.baseline_kind == "native_path_control"
    and .metadata.control_key == $control_key
    and .metadata.cann_root == $cann_root
    and .metadata.vllm_commit == "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"
    and .metadata.vllm_ascend_commit == "3da28f9414583d2d0b672a8f06d1fae142404bda"
    and .metadata.target_execution_mode == $target_execution
    and .metadata.mtp_draft_execution == $draft_execution
    and .metadata.enable_mtp == $enable_mtp
    and .metadata.mtp_num_speculative_tokens == $mtp_tokens
  ' "${golden_path}" >/dev/null \
    || die "Invalid or mismatched native control: ${golden_path}"
}

activate_and_audit() {
  audit_stack
  [[ -d "${PHASE1_GOLDEN_ROOT:-/nonexistent}" ]] \
    || die "PHASE1_GOLDEN_ROOT is invalid"
  local requested_cases=("$@") case_name
  if (( ${#requested_cases[@]} == 0 )); then
    requested_cases=("${EXACT_CASES[@]}")
  fi
  for case_name in "${requested_cases[@]}"; do
    contains_exact_case "${case_name}" \
      || die "Case is not in the exact matrix: ${case_name}"
    validate_golden_for_case "${case_name}"
  done
}

activate_and_audit_smoke() {
  audit_stack
  local requested_cases=("$@") case_name
  if (( ${#requested_cases[@]} == 0 )); then
    requested_cases=("${SMOKE_CASES[@]}")
  fi
  for case_name in "${requested_cases[@]}"; do
    contains_smoke_case "${case_name}" \
      || die "Case is not in the A5 phase-one smoke matrix: ${case_name}"
  done
}

case_arguments() {
  local case_name="$1"
  CASE_ARGS=(
    --connector P2pHcclAFDConnector
    --attention-max-num-batched-tokens 4096
  )
  case "${case_name}" in
    a4f4_eager_u1_mtp_off)
      CASE_ARGS+=(--attention-devices 0,1,2,3 --ffn-devices 4,5,6,7 --ffn-max-num-batched-tokens 4096 --execution-mode eager --u-batches 1)
      ;;
    a4f4_graph_u2_mtp_off)
      CASE_ARGS+=(--attention-devices 0,1,2,3 --ffn-devices 4,5,6,7 --ffn-max-num-batched-tokens 4096 --execution-mode full-decode-only --u-batches 2)
      ;;
    a2f4_graph_u2_mtp_off)
      CASE_ARGS+=(--attention-devices 0,1 --ffn-devices 2,3,4,5 --ffn-max-num-batched-tokens 2048 --execution-mode full-decode-only --u-batches 2)
      ;;
    a4f2_graph_u2_mtp_off)
      CASE_ARGS+=(--attention-devices 0,1,2,3 --ffn-devices 4,5 --ffn-max-num-batched-tokens 8192 --execution-mode full-decode-only --u-batches 2)
      ;;
    a4f4_eager_u1_n1)
      CASE_ARGS+=(--attention-devices 0,1,2,3 --ffn-devices 4,5,6,7 --ffn-max-num-batched-tokens 4096 --execution-mode eager --u-batches 1 --enable-mtp --mtp-num-speculative-tokens 1 --mtp-draft-execution eager)
      ;;
    a4f4_eager_u2_n2)
      CASE_ARGS+=(--attention-devices 0,1,2,3 --ffn-devices 4,5,6,7 --ffn-max-num-batched-tokens 4096 --execution-mode eager --u-batches 2 --enable-mtp --mtp-num-speculative-tokens 2 --mtp-draft-execution eager)
      ;;
    a4f4_graph_u1_n2)
      CASE_ARGS+=(--attention-devices 0,1,2,3 --ffn-devices 4,5,6,7 --ffn-max-num-batched-tokens 4096 --execution-mode full-decode-only --u-batches 1 --enable-mtp --mtp-num-speculative-tokens 2 --mtp-draft-execution eager)
      ;;
    a4f4_graph_u2_n2)
      CASE_ARGS+=(--attention-devices 0,1,2,3 --ffn-devices 4,5,6,7 --ffn-max-num-batched-tokens 4096 --execution-mode full-decode-only --u-batches 2 --enable-mtp --mtp-num-speculative-tokens 2 --mtp-draft-execution graph)
      ;;
    a4f4_graph_u2_n3)
      CASE_ARGS+=(--attention-devices 0,1,2,3 --ffn-devices 4,5,6,7 --ffn-max-num-batched-tokens 4096 --execution-mode full-decode-only --u-batches 2 --enable-mtp --mtp-num-speculative-tokens 3 --mtp-draft-execution graph)
      ;;
    a2f4_eager_u1_n2)
      CASE_ARGS+=(--attention-devices 0,1 --ffn-devices 2,3,4,5 --ffn-max-num-batched-tokens 2048 --execution-mode eager --u-batches 1 --enable-mtp --mtp-num-speculative-tokens 2 --mtp-draft-execution eager)
      ;;
    a2f4_graph_u2_n3)
      CASE_ARGS+=(--attention-devices 0,1 --ffn-devices 2,3,4,5 --ffn-max-num-batched-tokens 2048 --execution-mode full-decode-only --u-batches 2 --enable-mtp --mtp-num-speculative-tokens 3 --mtp-draft-execution graph)
      ;;
    a4f2_eager_u1_n2)
      CASE_ARGS+=(--attention-devices 0,1,2,3 --ffn-devices 4,5 --ffn-max-num-batched-tokens 8192 --execution-mode eager --u-batches 1 --enable-mtp --mtp-num-speculative-tokens 2 --mtp-draft-execution eager)
      ;;
    a4f2_graph_u2_n3)
      CASE_ARGS+=(--attention-devices 0,1,2,3 --ffn-devices 4,5 --ffn-max-num-batched-tokens 8192 --execution-mode full-decode-only --u-batches 2 --enable-mtp --mtp-num-speculative-tokens 3 --mtp-draft-execution graph)
      ;;
    *) die "Unknown case: ${case_name}" ;;
  esac
}

run_matrix() {
  local phase="$1"
  shift
  local cycles rounds idle_seconds output_root case_name golden_path case_rc matrix_rc
  case "${phase}" in
    smoke)
      cycles=2
      rounds=1
      idle_seconds=0
      ;;
    f0)
      cycles=1
      rounds=1
      idle_seconds=0
      ;;
    f1)
      cycles=2
      rounds=3
      idle_seconds="${PHASE1_F1_IDLE_SECONDS:-1800}"
      ;;
    *) die "Unknown matrix phase: ${phase}" ;;
  esac
  output_root="${PHASE1_OUTPUT_BASE:-/data/validation/dsv4-phase1-a5-$(date +%Y%m%d_%H%M%S)}/${phase}"
  [[ ! -e "${output_root}" ]] || die "Output root already exists: ${output_root}"
  mkdir -p "${output_root}"
  {
    printf 'phase=%s\n' "${phase}"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'afd_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    printf 'cann_root=%s\n' "$(readlink -f "${DSV4_CANN_ROOT}")"
    printf 'golden_checked=%s\n' "$([[ "${phase}" == "smoke" ]] && printf 0 || printf 1)"
    if [[ "${phase}" != "smoke" ]]; then
      printf 'golden_root=%s\n' "$(readlink -f "${PHASE1_GOLDEN_ROOT}")"
    fi
  } >"${output_root}/matrix.env"
  npu-smi info >"${output_root}/npu-before.txt"

  if (( $# == 0 )); then
    if [[ "${phase}" == "smoke" ]]; then
      set -- "${SMOKE_CASES[@]}"
    else
      set -- "${EXACT_CASES[@]}"
    fi
  fi
  matrix_rc=0
  for case_name in "$@"; do
    if [[ "${phase}" == "smoke" ]]; then
      contains_smoke_case "${case_name}" \
        || die "Case is not in the A5 phase-one smoke matrix: ${case_name}"
    else
      contains_exact_case "${case_name}" \
        || die "Case is not in the exact matrix: ${case_name}"
    fi
    case_arguments "${case_name}"
    runner_args=(
      --output-dir "${output_root}/${case_name}" \
      --cycles "${cycles}" \
      --idle-seconds "${idle_seconds}" \
      --rounds "${rounds}" \
      --batch-sizes 1 8 32 \
      "${CASE_ARGS[@]}"
    )
    if [[ "${phase}" == "smoke" ]]; then
      runner_args+=(--functional-smoke)
    else
      golden_path="$(golden_for_case "${case_name}")"
      runner_args+=(--golden "${golden_path}")
    fi
    set +e
    # Case arguments are authoritative; do not inherit MTP defaults from config.env.
    env \
      ENABLE_MTP=0 \
      MTP_NUM_SPECULATIVE_TOKENS=1 \
      MTP_DRAFT_EXECUTION=eager \
      "${DSV4_RUNTIME_VENV}/bin/python" "${RUNNER}" "${runner_args[@]}"
    case_rc=$?
    set -e
    printf '%s\n' "${case_rc}" >"${output_root}/${case_name}.exitcode"
    if (( case_rc != 0 )); then
      matrix_rc="${case_rc}"
      printf '[phase1-a5] case failed: %s (exit %s)\n' \
        "${case_name}" "${case_rc}" >&2
      break
    fi
  done
  npu-smi info >"${output_root}/npu-after.txt"
  {
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'exitcode=%s\n' "${matrix_rc}"
  } >>"${output_root}/matrix.env"
  if (( matrix_rc == 0 )); then
    printf '[phase1-a5] completed: %s\n' "${output_root}"
  else
    printf '[phase1-a5] evidence retained: %s\n' "${output_root}" >&2
  fi
  return "${matrix_rc}"
}

case "${ACTION}" in
  help|-h|--help) usage ;;
  list|list-smoke) printf '%s\n' "${SMOKE_CASES[@]}" ;;
  list-deferred-exact) printf '%s\n' "${EXACT_CASES[@]}" ;;
  preflight-native) audit_stack; printf '[phase1-a5] native preflight passed\n' ;;
  preflight|preflight-smoke)
    activate_and_audit_smoke "$@"
    printf '[phase1-a5] MTP-off smoke preflight passed\n'
    ;;
  preflight-deferred-exact)
    [[ "${PHASE1_ENABLE_DEFERRED_EXACT:-0}" == "1" ]] \
      || die "exact preflight is deferred; MTP validation resumes with dSpark"
    activate_and_audit "$@"
    printf '[phase1-a5] deferred exact preflight passed\n'
    ;;
  smoke) activate_and_audit_smoke "$@"; run_matrix "${ACTION}" "$@" ;;
  f0|f1)
    [[ "${PHASE1_ENABLE_DEFERRED_EXACT:-0}" == "1" ]] \
      || die "f0/f1 are deferred; MTP validation resumes with dSpark"
    activate_and_audit "$@"
    run_matrix "${ACTION}" "$@"
    ;;
  *) usage; die "Unknown action: ${ACTION}" ;;
esac
