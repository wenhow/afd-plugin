#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/repro_a5_hccl_reducescatter.py"
OUTPUT_ROOT="${A5_HCCL_REPRO_OUTPUT_ROOT:-/data/validation/a5-hccl-rs-$(date +%Y%m%d_%H%M%S)}"
VISIBLE_DEVICES="${A5_HCCL_REPRO_DEVICES:-4,5}"

die() {
  printf '[a5-hccl-rs] ERROR: %s\n' "$*" >&2
  exit 2
}

[[ ! -e "${OUTPUT_ROOT}" ]] || die "Output root already exists: ${OUTPUT_ROOT}"
[[ "${VISIBLE_DEVICES}" =~ ^[0-9]+,[0-9]+$ ]] \
  || die "A5_HCCL_REPRO_DEVICES must contain exactly two device IDs"

source "${SCRIPT_DIR}/activate_v023_vllm_cann_runtime.sh"
[[ -x "${DSV4_RUNTIME_VENV}/bin/python" ]] || die "Runtime venv is missing"
[[ -f "${DSV4_CANN_ROOT}/set_env.sh" ]] || die "CANN root is invalid"
[[ -f "${WORKER}" ]] || die "ReduceScatter worker is missing"

mkdir -p "${OUTPUT_ROOT}/ascend-process-log"
export ASCEND_PROCESS_LOG_PATH="${OUTPUT_ROOT}/ascend-process-log"
export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"

{
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'devices=%s\n' "${ASCEND_RT_VISIBLE_DEVICES}"
  printf 'cann_root=%s\n' "$(readlink -f "${DSV4_CANN_ROOT}")"
  printf 'python=%s\n' "$(readlink -f "${DSV4_RUNTIME_VENV}/bin/python")"
  printf 'hccl_op_expansion_mode=%s\n' "${HCCL_OP_EXPANSION_MODE}"
  printf 'output_count=%s\n' '16777216'
  printf 'input_bytes=%s\n' '67108864'
  printf 'output_bytes=%s\n' '33554432'
} >"${OUTPUT_ROOT}/repro.env"
npu-smi info >"${OUTPUT_ROOT}/npu-before.txt"

run_case() {
  local case_name="$1" group_count="$2" prime_operations="$3" case_rc
  mkdir -p "${OUTPUT_ROOT}/${case_name}"
  set +e
  "${DSV4_RUNTIME_VENV}/bin/python" -m torch.distributed.run \
    --standalone \
    --nproc-per-node=2 \
    "${WORKER}" \
    --output-dir "${OUTPUT_ROOT}/${case_name}" \
    --group-count "${group_count}" \
    --prime-operations "${prime_operations}" \
    2>&1 | tee "${OUTPUT_ROOT}/${case_name}.console.log"
  case_rc="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "${case_rc}" >"${OUTPUT_ROOT}/${case_name}.exitcode"
  return "${case_rc}"
}

classification=standalone_repro_passed
final_rc=0
if run_case world_only 1 0; then
  # Materializing 20 communicators makes the final one group_name_19. Its
  # initial AllReduce plus 135 AllGathers places ReduceScatter at opIndex 136.
  if run_case group19_op136 20 135; then
    :
  else
    final_rc=$?
    classification=multi_communicator_or_sequence_failure
  fi
else
  final_rc=$?
  classification=standalone_hccl_reducescatter_failure
fi

npu-smi info >"${OUTPUT_ROOT}/npu-after.txt" || true
{
  printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'classification=%s\n' "${classification}"
  printf 'exitcode=%s\n' "${final_rc}"
} >>"${OUTPUT_ROOT}/repro.env"
printf '[a5-hccl-rs] classification=%s output=%s\n' \
  "${classification}" "${OUTPUT_ROOT}"
exit "${final_rc}"
