#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 attention|ffn" >&2
  exit 2
fi
role="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

LOAD_VLLM_ASCEND_OPS=1
export LOAD_VLLM_ASCEND_OPS
# shellcheck source=activate_runtime.sh
source "${SCRIPT_DIR}/activate_runtime.sh"

require_file "${MODEL_PATH}/config.json"

case "${role}" in
  attention)
    served_model_name="dsv4-afd"
    api_port="${ATTENTION_API_PORT}"
    role_ranks="${ATTENTION_RANKS}"
    visible_devices="${ATTENTION_DEVICES}"
    max_num_batched_tokens="${ATTENTION_MAX_NUM_BATCHED_TOKENS}"
    hccl_base_port="${ATTENTION_HCCL_IF_BASE_PORT}"
    hccl_buffsize="${HCCL_BUFFSIZE_ATTENTION}"
    ;;
  ffn)
    served_model_name="dsv4-afd-ffn"
    api_port="${FFN_PROCESS_PORT}"
    role_ranks="${FFN_RANKS}"
    visible_devices="${FFN_DEVICES}"
    max_num_batched_tokens="${FFN_MAX_NUM_BATCHED_TOKENS}"
    hccl_base_port="${FFN_HCCL_IF_BASE_PORT}"
    hccl_buffsize="${HCCL_BUFFSIZE_FFN}"
    ;;
  *)
    die "Unknown role: ${role}"
    ;;
esac

larger_ranks=$((ATTENTION_RANKS > FFN_RANKS ? ATTENTION_RANKS : FFN_RANKS))
smaller_ranks=$((ATTENTION_RANKS < FFN_RANKS ? ATTENTION_RANKS : FFN_RANKS))
(( larger_ranks % smaller_ranks == 0 )) \
  || die "Attention and FFN ranks must have an integer ratio"
ratio=$((larger_ranks / smaller_ranks))
if (( FFN_RANKS > ATTENTION_RANKS )); then
  required_ffn_tokens=$(((ATTENTION_MAX_NUM_BATCHED_TOKENS + ratio - 1) / ratio))
else
  required_ffn_tokens=$((ATTENTION_MAX_NUM_BATCHED_TOKENS * ratio))
fi
(( FFN_MAX_NUM_BATCHED_TOKENS >= required_ffn_tokens )) \
  || die "FFN_MAX_NUM_BATCHED_TOKENS must be at least ${required_ffn_tokens}"

export ASCEND_RT_VISIBLE_DEVICES="${visible_devices}"
export HCCL_IF_IP="$(resolve_hccl_ip)"
export HCCL_IF_BASE_PORT="${hccl_base_port}"
export GLOO_SOCKET_IFNAME
export HCCL_SOCKET_IFNAME
export OMP_PROC_BIND=false
export OMP_NUM_THREADS
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_BUFFSIZE="${hccl_buffsize}"
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_EXEC_TIMEOUT
export TASK_QUEUE_ENABLE=1
export VLLM_ENGINE_READY_TIMEOUT_S
unset VLLM_ASCEND_ENABLE_FLASHCOMM1

additional_config="$(printf \
  '{"afd":{"role":"%s","connector":"P2pHcclAFDConnector","host":"%s","port":%s,"num_attention_ranks":%s,"num_ffn_ranks":%s}}' \
  "${role}" "${AFD_HOST}" "${AFD_PORT}" "${ATTENTION_RANKS}" "${FFN_RANKS}")"

execution_args=()
case "${EXECUTION_MODE}" in
  eager)
    execution_args=(--enforce-eager)
    ;;
  full-decode-only)
    [[ "${U_BATCHES}" == "1" ]] || die "Graph requires U1"
    read -r -a capture_sizes <<<"${CUDAGRAPH_CAPTURE_SIZES}"
    execution_args=(
      --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}"
      --cudagraph-capture-sizes "${capture_sizes[@]}"
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    )
    ;;
  *)
    die "Unsupported EXECUTION_MODE=${EXECUTION_MODE}"
    ;;
esac

ubatch_args=()
case "${U_BATCHES}" in
  1) ;;
  2)
    [[ "${EXECUTION_MODE}" == "eager" ]] || die "U2 requires eager mode"
    ubatch_args=(
      --enable-dbo
      --dbo-decode-token-threshold "${DBO_DECODE_TOKEN_THRESHOLD}"
      --dbo-prefill-token-threshold "${DBO_PREFILL_TOKEN_THRESHOLD}"
    )
    ;;
  *)
    die "U_BATCHES must be 1 or 2"
    ;;
esac

mtp_args=()
case "${ENABLE_MTP}" in
  0) ;;
  1)
    [[ "${EXECUTION_MODE}" == "eager" && "${U_BATCHES}" == "1" ]] \
      || die "MTP M1 requires eager/U1"
    [[ "${MTP_NUM_SPECULATIVE_TOKENS}" =~ ^[1-3]$ ]] \
      || die "MTP supports num_speculative_tokens in [1, 3]"
    mtp_args=(
      --speculative-config
      "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_NUM_SPECULATIVE_TOKENS},\"enforce_eager\":true}"
    )
    ;;
  *)
    die "ENABLE_MTP must be 0 or 1"
    ;;
esac

command=(
  vllm serve "${MODEL_PATH}"
  --host "${API_HOST}"
  --port "${api_port}"
  --api-server-count 1
  --served-model-name "${served_model_name}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-batched-tokens "${max_num_batched_tokens}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --data-parallel-size "${role_ranks}"
  --tensor-parallel-size 1
  --enable-expert-parallel
  --seed 1024
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tokenizer-mode deepseek_v4
  --no-enable-prefix-caching
  --safetensors-load-strategy lazy
  --quantization ascend
  --block-size 128
  --additional-config "${additional_config}"
  "${mtp_args[@]}"
  "${ubatch_args[@]}"
  "${execution_args[@]}"
)

if [[ "${role}" == "attention" ]]; then
  exec "${command[@]}"
fi

shutdown_requested=0
vllm_pid=""

forward_shutdown() {
  shutdown_requested=1
  if [[ -n "${vllm_pid}" ]]; then
    kill -TERM "${vllm_pid}" 2>/dev/null || true
  fi
}

trap forward_shutdown TERM INT
set +e
"${command[@]}" &
vllm_pid=$!
wait "${vllm_pid}"
vllm_status=$?
set -e

if (( shutdown_requested )); then
  exit 0
fi
exit "${vllm_status}"
