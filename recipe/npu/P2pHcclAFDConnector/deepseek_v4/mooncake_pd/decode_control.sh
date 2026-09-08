#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
source "${ROOT_DIR}/recipe/npu/deepseek_v4/common/activate_role_runtime.sh"
set -u

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8910}"
MODEL_NAME="${MODEL_NAME:-dsv4-afd}"
DECODE_DP_SIZE="${DECODE_DP_SIZE:-8}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-1}"
PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-2}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-4}"
MOONCAKE_ENGINE_ID="${MOONCAKE_ENGINE_ID:-dsv4-pd-control-decode}"
MOONCAKE_KV_PORT="${MOONCAKE_KV_PORT:-30100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DATA_PARALLEL_RPC_PORT="${CONTROL_DATA_PARALLEL_RPC_PORT:-29360}"
MASTER_PORT="${CONTROL_MASTER_PORT:-29361}"
EXECUTION_MODE="${EXECUTION_MODE:-eager}"
U_BATCHES="${U_BATCHES:-1}"
ENABLE_MTP="${ENABLE_MTP:-0}"
MTP_DRAFT_EXECUTION="${MTP_DRAFT_EXECUTION:-eager}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-1}"
DBO_DECODE_TOKEN_THRESHOLD="${DBO_DECODE_TOKEN_THRESHOLD:-2}"
DBO_PREFILL_TOKEN_THRESHOLD="${DBO_PREFILL_TOKEN_THRESHOLD:-12}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-8}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-1 2 4 8}"

case "${DECODE_DP_SIZE}:${DECODE_TP_SIZE}" in
  8:1|4:1|4:2) ;;
  *)
    echo "Phase-one PD control supports Decode DP8/TP1, DP4/TP1, or DP4/TP2" >&2
    exit 2
    ;;
esac

case "${EXECUTION_MODE}" in
  eager)
    EXECUTION_ARGS=(--enforce-eager)
    ;;
  full-decode-only)
    read -r -a CAPTURE_SIZE_ARGS <<<"${CUDAGRAPH_CAPTURE_SIZES}"
    EXECUTION_ARGS=(
      --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}"
      --cudagraph-capture-sizes "${CAPTURE_SIZE_ARGS[@]}"
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    )
    ;;
  *)
    echo "Unsupported EXECUTION_MODE=${EXECUTION_MODE}" >&2
    exit 2
    ;;
esac

case "${U_BATCHES}" in
  1)
    UBATCH_ARGS=()
    ;;
  2)
    UBATCH_ARGS=(
      --enable-dbo
      --dbo-decode-token-threshold "${DBO_DECODE_TOKEN_THRESHOLD}"
      --dbo-prefill-token-threshold "${DBO_PREFILL_TOKEN_THRESHOLD}"
    )
    ;;
  *)
    echo "U_BATCHES must be 1 or 2" >&2
    exit 2
    ;;
esac

case "${ENABLE_MTP}" in
  0)
    MTP_ARGS=()
    ;;
  1)
    case "${EXECUTION_MODE}:${MTP_DRAFT_EXECUTION}" in
      eager:eager|full-decode-only:eager) MTP_DRAFT_ENFORCE_EAGER=true ;;
      full-decode-only:graph) MTP_DRAFT_ENFORCE_EAGER=false ;;
      *)
        echo "Unsupported target/draft execution: ${EXECUTION_MODE}/${MTP_DRAFT_EXECUTION}" >&2
        exit 2
        ;;
    esac
    if [[ ! "${MTP_NUM_SPECULATIVE_TOKENS}" =~ ^[1-3]$ ]]; then
      echo "DeepSeek-V4 MTP supports num_speculative_tokens in [1, 3]" >&2
      exit 2
    fi
    MTP_CONFIG="$(printf '{"method":"mtp","num_speculative_tokens":%s,"enforce_eager":%s}' "${MTP_NUM_SPECULATIVE_TOKENS}" "${MTP_DRAFT_ENFORCE_EAGER}")"
    MTP_ARGS=(--speculative-config "${MTP_CONFIG}")
    ;;
  *)
    echo "ENABLE_MTP must be 0 or 1" >&2
    exit 2
    ;;
esac

if [[ "${DECODE_TP_SIZE}" == "2" \
  && "${EXECUTION_MODE}" == "full-decode-only" \
  && "${U_BATCHES}" == "2" \
  && "${ENABLE_MTP}" == "1" \
  && "${MTP_DRAFT_EXECUTION}" == "graph" ]]; then
  echo "TP2 full-draft Graph U2 + MTP is not validated" >&2
  exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES="${ATTENTION_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_IF_IP="${HCCL_IF_IP:-192.169.91.106}"
export VLLM_HOST_IP="${VLLM_HOST_IP:-${HCCL_IF_IP}}"
export HCCL_IF_BASE_PORT="${ATTENTION_HCCL_IF_BASE_PORT:-51000}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"

# The path-matched control must load Ascend and Mooncake plugins only. Loading
# afd here would invalidate the PD no-AFD reference.
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector

source "${ROOT_DIR}/tools/dsv4/check_mooncake_runtime.sh"
KV_TRANSFER_CONFIG="$(python "${ROOT_DIR}/tools/dsv4/mooncake_pd_config.py" \
  --role kv_consumer \
  --engine-id "${MOONCAKE_ENGINE_ID}" \
  --kv-port "${MOONCAKE_KV_PORT}" \
  --prefill-dp-size "${PREFILL_DP_SIZE}" \
  --prefill-tp-size "${PREFILL_TP_SIZE}" \
  --decode-dp-size "${DECODE_DP_SIZE}" \
  --decode-tp-size "${DECODE_TP_SIZE}")"

exec vllm serve "${MODEL_PATH}" \
  --host "${API_HOST}" \
  --port "${API_PORT}" \
  --api-server-count 1 \
  --served-model-name "${MODEL_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --data-parallel-size "${DECODE_DP_SIZE}" \
  --data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT}" \
  --master-port "${MASTER_PORT}" \
  --tensor-parallel-size "${DECODE_TP_SIZE}" \
  --all2all-backend flashinfer_all2allv \
  --enable-expert-parallel \
  --seed 1024 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tokenizer-mode deepseek_v4 \
  --no-enable-prefix-caching \
  --safetensors-load-strategy lazy \
  --quantization ascend \
  --block-size 128 \
  --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
  "${MTP_ARGS[@]}" \
  "${UBATCH_ARGS[@]}" \
  "${EXECUTION_ARGS[@]}"
