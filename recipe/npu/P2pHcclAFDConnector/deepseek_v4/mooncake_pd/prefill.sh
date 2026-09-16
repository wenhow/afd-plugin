#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
export DSV4_VLLM_VENV="${DSV4_RUNTIME_VENV:-/mnt/workspace/code/.venvs/afd-v023-vllm-cann}"
source "${ROOT_DIR}/tools/dsv4/activate_runtime.sh"
DSV4_VLLM_ASCEND_ROOT="${DSV4_VLLM_ASCEND_ROOT:-/mnt/workspace/code/vllm-ascend-rfc-vllm-cann}"
source "${DSV4_VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
set -u

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8100}"
PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-2}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-4}"
DECODE_DP_SIZE="${DECODE_DP_SIZE:-8}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-1}"
MOONCAKE_ENGINE_ID="${MOONCAKE_ENGINE_ID:-dsv4-afd-prefill}"
MOONCAKE_KV_PORT="${MOONCAKE_KV_PORT:-30000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MODEL_QUANTIZATION="${MODEL_QUANTIZATION:-auto}"
MODEL_BLOCK_SIZE="${MODEL_BLOCK_SIZE:-auto}"
MODEL_SAFETENSORS_LOAD_STRATEGY="${MODEL_SAFETENSORS_LOAD_STRATEGY:-auto}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ENABLE_DSPARK="${ENABLE_DSPARK:-0}"
DSPARK_NUM_SPECULATIVE_TOKENS="${DSPARK_NUM_SPECULATIVE_TOKENS:-auto}"

export ASCEND_RT_VISIBLE_DEVICES="${PREFILL_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export HCCL_IF_IP="${HCCL_IF_IP:-192.169.91.106}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-${HCCL_IF_IP}}"
export HCCL_IF_BASE_PORT="${PREFILL_HCCL_IF_BASE_PORT:-50000}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1
export SOC_VERSION="${SOC_VERSION:-ascend910_9362}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-18000}"
case "${ENABLE_AFD_PLUGIN:-1}" in
  0)
    export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector
    ;;
  1)
    export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd
    ;;
  *)
    echo "ENABLE_AFD_PLUGIN must be 0 or 1" >&2
    exit 2
    ;;
esac
unset VLLM_ASCEND_ENABLE_FLASHCOMM1

source "${ROOT_DIR}/tools/dsv4/check_mooncake_runtime.sh"
MODEL_SPECULATIVE_METHOD="$(
  "${DSV4_VLLM_VENV}/bin/python" \
    "${ROOT_DIR}/tools/dsv4/hccl_manual_install/bin/model_launch_args.py" \
    --model-path "${MODEL_PATH}" \
    --quantization "${MODEL_QUANTIZATION}" \
    --block-size "${MODEL_BLOCK_SIZE}" \
    --safetensors-load-strategy "${MODEL_SAFETENSORS_LOAD_STRATEGY}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --get speculative_method
)"
MODEL_ARGS_OUTPUT="$(
  "${DSV4_VLLM_VENV}/bin/python" \
    "${ROOT_DIR}/tools/dsv4/hccl_manual_install/bin/model_launch_args.py" \
    --model-path "${MODEL_PATH}" \
    --quantization "${MODEL_QUANTIZATION}" \
    --block-size "${MODEL_BLOCK_SIZE}" \
    --safetensors-load-strategy "${MODEL_SAFETENSORS_LOAD_STRATEGY}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
)"
mapfile -t MODEL_ARGS <<<"${MODEL_ARGS_OUTPUT}"

case "${ENABLE_DSPARK}" in
  0)
    DSPARK_ARGS=()
    ;;
  1)
    DSPARK_CHECKPOINT_TOKENS="$(
      "${DSV4_VLLM_VENV}/bin/python" \
        "${ROOT_DIR}/tools/dsv4/hccl_manual_install/bin/model_launch_args.py" \
        --model-path "${MODEL_PATH}" \
        --quantization "${MODEL_QUANTIZATION}" \
        --block-size "${MODEL_BLOCK_SIZE}" \
        --safetensors-load-strategy "${MODEL_SAFETENSORS_LOAD_STRATEGY}" \
        --kv-cache-dtype "${KV_CACHE_DTYPE}" \
        --get dspark_block_size
    )"
    if [[ ! "${DSPARK_CHECKPOINT_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
      echo "ENABLE_DSPARK=1 requires a DSpark checkpoint with dspark_block_size" >&2
      exit 2
    fi
    if [[ "${DSPARK_NUM_SPECULATIVE_TOKENS}" == auto ]]; then
      DSPARK_NUM_SPECULATIVE_TOKENS="${DSPARK_CHECKPOINT_TOKENS}"
    elif [[ "${DSPARK_NUM_SPECULATIVE_TOKENS}" != "${DSPARK_CHECKPOINT_TOKENS}" ]]; then
      echo "DSPARK_NUM_SPECULATIVE_TOKENS must match checkpoint dspark_block_size=${DSPARK_CHECKPOINT_TOKENS}" >&2
      exit 2
    fi
    DSPARK_CONFIG="$(printf '{"method":"%s","num_speculative_tokens":%s,"enforce_eager":true,"draft_sample_method":"greedy"}' "${MODEL_SPECULATIVE_METHOD}" "${DSPARK_NUM_SPECULATIVE_TOKENS}")"
    DSPARK_ARGS=(--speculative-config "${DSPARK_CONFIG}")
    ;;
  *)
    echo "ENABLE_DSPARK must be 0 or 1" >&2
    exit 2
    ;;
esac

KV_TRANSFER_CONFIG="$(python "${ROOT_DIR}/tools/dsv4/mooncake_pd_config.py" \
  --role kv_producer \
  --engine-id "$MOONCAKE_ENGINE_ID" \
  --kv-port "$MOONCAKE_KV_PORT" \
  --prefill-dp-size "$PREFILL_DP_SIZE" \
  --prefill-tp-size "$PREFILL_TP_SIZE" \
  --decode-dp-size "$DECODE_DP_SIZE" \
  --decode-tp-size "$DECODE_TP_SIZE")"

exec vllm serve "$MODEL_PATH" \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --api-server-count 1 \
  --served-model-name dsv4-afd \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --data-parallel-size "$PREFILL_DP_SIZE" \
  --tensor-parallel-size "$PREFILL_TP_SIZE" \
  --all2all-backend flashinfer_all2allv \
  --enable-expert-parallel \
  --seed 1024 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tokenizer-mode deepseek_v4 \
  --no-enable-prefix-caching \
  "${MODEL_ARGS[@]}" \
  "${DSPARK_ARGS[@]}" \
  --enforce-eager \
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
