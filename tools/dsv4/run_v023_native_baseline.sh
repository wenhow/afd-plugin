#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/tools/dsv4/activate_v023_vllm_cann_runtime.sh"
set +u
source "${DSV4_VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
set -u

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8900}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv4-v023-native}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DATA_PARALLEL_RPC_PORT="${DATA_PARALLEL_RPC_PORT:-29350}"
MASTER_PORT="${MASTER_PORT:-29351}"
ENABLE_MTP="${ENABLE_MTP:-0}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-1}"
MTP_DRAFT_EXECUTION="${MTP_DRAFT_EXECUTION:-eager}"
EXECUTION_MODE="${EXECUTION_MODE:-eager}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-8}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-1 2 4 8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MODEL_QUANTIZATION="${MODEL_QUANTIZATION:-auto}"
MODEL_BLOCK_SIZE="${MODEL_BLOCK_SIZE:-auto}"
MODEL_SAFETENSORS_LOAD_STRATEGY="${MODEL_SAFETENSORS_LOAD_STRATEGY:-auto}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
MODEL_SPECULATIVE_METHOD="${MODEL_SPECULATIVE_METHOD:-auto}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

if [[ ! "${TENSOR_PARALLEL_SIZE}" =~ ^[12]$ ]]; then
  echo "TENSOR_PARALLEL_SIZE must be 1 or 2" >&2
  exit 2
fi
IFS=',' read -r -a visible_devices <<<"${ASCEND_RT_VISIBLE_DEVICES}"
device_count="${#visible_devices[@]}"
if (( device_count == 0 || device_count % TENSOR_PARALLEL_SIZE != 0 )); then
  echo "Visible NPU count must be divisible by TENSOR_PARALLEL_SIZE" >&2
  exit 2
fi
DATA_PARALLEL_SIZE=$((device_count / TENSOR_PARALLEL_SIZE))

if [[ "${MODEL_SPECULATIVE_METHOD}" == auto ]]; then
  MODEL_SPECULATIVE_METHOD="$(
    "${DSV4_RUNTIME_VENV}/bin/python" \
      "${ROOT_DIR}/tools/dsv4/hccl_manual_install/bin/model_launch_args.py" \
      --model-path "${MODEL_PATH}" \
      --quantization "${MODEL_QUANTIZATION}" \
      --block-size "${MODEL_BLOCK_SIZE}" \
      --safetensors-load-strategy "${MODEL_SAFETENSORS_LOAD_STRATEGY}" \
      --kv-cache-dtype "${KV_CACHE_DTYPE}" \
      --get speculative_method
  )"
fi

MTP_ARGS=()
case "${ENABLE_MTP}" in
  0)
    ;;
  1)
    if [[ ! "${MTP_NUM_SPECULATIVE_TOKENS}" =~ ^[1-3]$ ]]; then
      echo "MTP_NUM_SPECULATIVE_TOKENS must be in [1, 3]" >&2
      exit 2
    fi
    case "${EXECUTION_MODE}:${MTP_DRAFT_EXECUTION}" in
      eager:eager|full-decode-only:eager)
        mtp_draft_enforce_eager=true
        ;;
      full-decode-only:graph)
        mtp_draft_enforce_eager=false
        ;;
      *)
        echo "Unsupported target/draft execution pair: ${EXECUTION_MODE}/${MTP_DRAFT_EXECUTION}" >&2
        exit 2
        ;;
    esac
    MTP_ARGS=(
      --speculative-config
      "{\"method\":\"${MODEL_SPECULATIVE_METHOD}\",\"num_speculative_tokens\":${MTP_NUM_SPECULATIVE_TOKENS},\"enforce_eager\":${mtp_draft_enforce_eager}}"
    )
    ;;
  *)
    echo "ENABLE_MTP must be 0 or 1" >&2
    exit 2
    ;;
esac

case "${EXECUTION_MODE}" in
  eager)
    EXECUTION_ARGS=(--enforce-eager)
    ;;
  full-decode-only)
    read -r -a capture_size_args <<<"${CUDAGRAPH_CAPTURE_SIZES}"
    EXECUTION_ARGS=(
      --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}"
      --cudagraph-capture-sizes "${capture_size_args[@]}"
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    )
    ;;
  *)
    echo "EXECUTION_MODE must be eager or full-decode-only" >&2
    exit 2
    ;;
esac

export HCCL_IF_IP="${HCCL_IF_IP:-192.169.91.106}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-53000}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1
export SOC_VERSION="${SOC_VERSION:-ascend910_9362}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-18000}"
# This is the native control: the AFD plugin must not register or patch vLLM.
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector

MODEL_ARGS_OUTPUT="$(
  "${DSV4_RUNTIME_VENV}/bin/python" \
    "${ROOT_DIR}/tools/dsv4/hccl_manual_install/bin/model_launch_args.py" \
    --model-path "${MODEL_PATH}" \
    --quantization "${MODEL_QUANTIZATION}" \
    --block-size "${MODEL_BLOCK_SIZE}" \
    --safetensors-load-strategy "${MODEL_SAFETENSORS_LOAD_STRATEGY}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}"
)"
mapfile -t MODEL_ARGS <<<"${MODEL_ARGS_OUTPUT}"

exec vllm serve "${MODEL_PATH}" \
  --host "${API_HOST}" \
  --port "${API_PORT}" \
  --api-server-count 1 \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT}" \
  --master-port "${MASTER_PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --enable-expert-parallel \
  --seed 1024 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tokenizer-mode deepseek_v4 \
  --no-enable-prefix-caching \
  "${MODEL_ARGS[@]}" \
  "${MTP_ARGS[@]}" \
  "${EXECUTION_ARGS[@]}"
