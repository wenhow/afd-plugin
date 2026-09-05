#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "${ROOT_DIR}/recipe/npu/deepseek_v4/common/activate_role_runtime.sh"
set -u

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8910}"
AFD_HOST="${AFD_HOST:-127.0.0.1}"
AFD_PORT="${AFD_PORT:-29761}"
readonly AFD_CONNECTOR=P2pHcclAFDConnector
ATTENTION_RANKS="${ATTENTION_RANKS:-8}"
FFN_RANKS="${FFN_RANKS:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_BATCHED_TOKENS="${ATTENTION_MAX_NUM_BATCHED_TOKENS:-${MAX_NUM_BATCHED_TOKENS:-1024}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
EXECUTION_MODE="${EXECUTION_MODE:-eager}"
U_BATCHES="${U_BATCHES:-1}"
DBO_DECODE_TOKEN_THRESHOLD="${DBO_DECODE_TOKEN_THRESHOLD:-2}"
DBO_PREFILL_TOKEN_THRESHOLD="${DBO_PREFILL_TOKEN_THRESHOLD:-12}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-8}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-1 2 4 8}"
ENABLE_MTP="${ENABLE_MTP:-0}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-1}"
MTP_DRAFT_EXECUTION="${MTP_DRAFT_EXECUTION:-eager}"
AFD_ASYNC_SCHEDULING="${AFD_ASYNC_SCHEDULING:-auto}"
VLLM_SHUTDOWN_TIMEOUT_SECONDS="${VLLM_SHUTDOWN_TIMEOUT_SECONDS:-0}"
AFD_NPU_ATTENTION_PROFILER_ENABLE="${AFD_NPU_ATTENTION_PROFILER_ENABLE:-0}"
ENABLE_PD="${ENABLE_PD:-0}"
MOONCAKE_ENGINE_ID="${MOONCAKE_ENGINE_ID:-dsv4-afd-decode}"
MOONCAKE_KV_PORT="${MOONCAKE_KV_PORT:-30100}"
PREFILL_DP_SIZE="${PREFILL_DP_SIZE:-2}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-4}"

export ASCEND_RT_VISIBLE_DEVICES="${ATTENTION_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export HCCL_IF_IP="${HCCL_IF_IP:-192.169.91.106}"
export HCCL_IF_BASE_PORT="${ATTENTION_HCCL_IF_BASE_PORT:-51000}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"

if [[ ! "$VLLM_SHUTDOWN_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "VLLM_SHUTDOWN_TIMEOUT_SECONDS must be a non-negative integer" >&2
  exit 2
fi

case "$AFD_NPU_ATTENTION_PROFILER_ENABLE" in
  0)
    PROFILE_API_ARGS=()
    ;;
  1)
    PROFILE_API_ARGS=(
      --middleware
      afd_plugin.compat.npu.profile_api.afd_profile_control_middleware
    )
    ;;
  *)
    echo "AFD_NPU_ATTENTION_PROFILER_ENABLE must be 0 or 1" >&2
    exit 2
    ;;
esac

if [[ ! "$TENSOR_PARALLEL_SIZE" =~ ^[12]$ ]]; then
  echo "DeepSeek-V4 AFD supports TENSOR_PARALLEL_SIZE=1 or 2" >&2
  exit 2
fi
if ((ATTENTION_RANKS % TENSOR_PARALLEL_SIZE != 0)); then
  echo "ATTENTION_RANKS must be divisible by TENSOR_PARALLEL_SIZE" >&2
  exit 2
fi
if ((TENSOR_PARALLEL_SIZE > 1)); then
  if [[ "$ATTENTION_RANKS" != "$FFN_RANKS" ]]; then
    echo "DeepSeek-V4 HCCL P2P TP2 requires equal Attention and FFN ranks" >&2
    exit 2
  fi
fi
ATTENTION_DP_SIZE=$((ATTENTION_RANKS / TENSOR_PARALLEL_SIZE))

case "$ENABLE_MTP" in
  0)
    MTP_ARGS=()
    ;;
  1)
    if [[ "$MTP_NUM_SPECULATIVE_TOKENS" != "1" ]]; then
      echo "DeepSeek-V4 MTP supports exactly one speculative token" >&2
      exit 2
    fi
    case "$EXECUTION_MODE" in
      eager)
        if [[ "$MTP_DRAFT_EXECUTION" != "eager" ]]; then
          echo "DeepSeek-V4 eager target requires MTP_DRAFT_EXECUTION=eager" >&2
          exit 2
        fi
        MTP_DRAFT_ENFORCE_EAGER=true
        ;;
      full-decode-only)
        case "$MTP_DRAFT_EXECUTION" in
          eager) MTP_DRAFT_ENFORCE_EAGER=true ;;
          graph) MTP_DRAFT_ENFORCE_EAGER=false ;;
          *)
            echo "MTP_DRAFT_EXECUTION must be eager or graph" >&2
            exit 2
            ;;
        esac
        ;;
      *)
        echo "DeepSeek-V4 MTP supports eager or full-decode-only" >&2
        exit 2
        ;;
    esac
    MTP_CONFIG="$(printf '{"method":"mtp","num_speculative_tokens":1,"enforce_eager":%s}' "$MTP_DRAFT_ENFORCE_EAGER")"
    MTP_ARGS=(
      --speculative-config
      "$MTP_CONFIG"
    )
    ;;
  *)
    echo "ENABLE_MTP must be 0 or 1" >&2
    exit 2
    ;;
esac

ADDITIONAL_CONFIG="$(printf '{"afd":{"role":"attention","connector":"%s","host":"%s","port":%s,"num_attention_ranks":%s,"num_ffn_ranks":%s}}' "$AFD_CONNECTOR" "$AFD_HOST" "$AFD_PORT" "$ATTENTION_RANKS" "$FFN_RANKS")"

case "$EXECUTION_MODE" in
  eager)
    EXECUTION_ARGS=(--enforce-eager)
    ;;
  full-decode-only)
    read -r -a CAPTURE_SIZE_ARGS <<<"$CUDAGRAPH_CAPTURE_SIZES"
    EXECUTION_ARGS=(
      --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
      --cudagraph-capture-sizes "${CAPTURE_SIZE_ARGS[@]}"
      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    )
    ;;
  *)
    echo "Unsupported EXECUTION_MODE=$EXECUTION_MODE" >&2
    exit 2
    ;;
esac

case "$U_BATCHES" in
  1)
    UBATCH_ARGS=()
    ;;
  2)
    UBATCH_ARGS=(
      --enable-dbo
      --dbo-decode-token-threshold "$DBO_DECODE_TOKEN_THRESHOLD"
      --dbo-prefill-token-threshold "$DBO_PREFILL_TOKEN_THRESHOLD"
    )
    ;;
  *)
    echo "DeepSeek-V4 AFD supports U_BATCHES=1 or 2, got $U_BATCHES" >&2
    exit 2
    ;;
esac

case "$AFD_ASYNC_SCHEDULING" in
  auto)
    SCHEDULING_ARGS=()
    ;;
  on)
    SCHEDULING_ARGS=(--async-scheduling)
    ;;
  off)
    SCHEDULING_ARGS=(--no-async-scheduling)
    ;;
  *)
    echo "AFD_ASYNC_SCHEDULING must be auto, on, or off" >&2
    exit 2
    ;;
esac

case "$ENABLE_PD" in
  0)
    KV_TRANSFER_ARGS=()
    ;;
  1)
    export VLLM_HOST_IP="${VLLM_HOST_IP:-${HCCL_IF_IP}}"
    source "${ROOT_DIR}/tools/dsv4/check_mooncake_runtime.sh"
    KV_TRANSFER_CONFIG="$(python "${ROOT_DIR}/tools/dsv4/mooncake_pd_config.py" \
      --role kv_consumer \
      --engine-id "$MOONCAKE_ENGINE_ID" \
      --kv-port "$MOONCAKE_KV_PORT" \
      --prefill-dp-size "$PREFILL_DP_SIZE" \
      --prefill-tp-size "$PREFILL_TP_SIZE" \
      --decode-dp-size "$ATTENTION_DP_SIZE" \
      --decode-tp-size "$TENSOR_PARALLEL_SIZE")"
    KV_TRANSFER_ARGS=(--kv-transfer-config "$KV_TRANSFER_CONFIG")
    ;;
  *)
    echo "ENABLE_PD must be 0 or 1" >&2
    exit 2
    ;;
esac

exec vllm serve "$MODEL_PATH" \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --api-server-count 1 \
  --served-model-name dsv4-afd \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --data-parallel-size "$ATTENTION_DP_SIZE" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --all2all-backend flashinfer_all2allv \
  --enable-expert-parallel \
  --seed 1024 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --shutdown-timeout "$VLLM_SHUTDOWN_TIMEOUT_SECONDS" \
  --tokenizer-mode deepseek_v4 \
  --no-enable-prefix-caching \
  --safetensors-load-strategy lazy \
  --quantization ascend \
  --block-size 128 \
  --additional-config "$ADDITIONAL_CONFIG" \
  "${KV_TRANSFER_ARGS[@]}" \
  "${SCHEDULING_ARGS[@]}" \
  "${MTP_ARGS[@]}" \
  "${UBATCH_ARGS[@]}" \
  "${PROFILE_API_ARGS[@]}" \
  "${EXECUTION_ARGS[@]}"
