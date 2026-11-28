#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

fail() {
    printf 'Window recipe: %s\n' "$*" >&2
    exit 1
}

validate_integer() {
    local label="$1" value="$2" minimum="$3" maximum="$4"
    [[ "$value" =~ ^(0|[1-9][0-9]{0,5})$ ]] || fail "$label must be an integer"
    (( value >= minimum && value <= maximum )) ||
        fail "$label must be in [$minimum, $maximum]"
}

launch_window_role() {
    local role="$1" action="$2"
    local model_path="${MODEL_PATH:-}"
    local nic_name="${NIC_NAME:-}" hccl_ip="${HCCL_IF_IP:-}"
    local afd_host="${AFD_HOST:-}" afd_port="${AFD_PORT:-29761}"
    local attention_ranks="${ATTENTION_RANKS:-}" ffn_ranks="${FFN_RANKS:-}"
    local execution_mode="${EXECUTION_MODE:-eager}" batches="${U_BATCHES:-1}"
    local disable_engram="${DISABLE_ENGRAM:-0}" dspark_tokens="${DSPARK_TOKENS:-0}"
    local max_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-128}" max_seqs="${MAX_NUM_SEQS:-2}"
    local python_bin="${PYTHON_BIN:-python}" vllm_bin="${VLLM_BIN:-vllm}"
    local dp_start_rank="${DP_START_RANK:-0}"
    local role_ranks devices api_host api_port dp_address rpc_port master_port

    if [[ "$role" == attention ]]; then
        role_ranks="$attention_ranks"
        devices="${ATTENTION_DEVICES:-}"
        api_host="${ATTENTION_API_HOST:-127.0.0.1}"
        api_port="${ATTENTION_API_PORT:-8910}"
        dp_address="${ATTENTION_DP_ADDRESS:-127.0.0.1}"
        rpc_port="${ATTENTION_DP_RPC_PORT:-29550}"
        master_port="${ATTENTION_MASTER_PORT:-29501}"
    else
        role_ranks="$ffn_ranks"
        devices="${FFN_DEVICES:-}"
        api_host="127.0.0.1"
        api_port="${FFN_API_PORT:-8911}"
        dp_address="${FFN_DP_ADDRESS:-$afd_host}"
        rpc_port="${FFN_DP_RPC_PORT:-29560}"
        master_port="${FFN_MASTER_PORT:-29511}"
    fi

    [[ "$action" == run || "$action" == --dry-run ]] || fail 'usage: script [--dry-run]'
    [[ -n "$model_path" && -n "$nic_name" && -n "$hccl_ip" && -n "$afd_host" ]] ||
        fail 'set MODEL_PATH, NIC_NAME, HCCL_IF_IP and AFD_HOST'
    [[ "$devices" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail "set the $role device list"
    validate_integer ATTENTION_RANKS "$attention_ranks" 1 65535
    validate_integer FFN_RANKS "$ffn_ranks" 2 65535
    validate_integer DP_START_RANK "$dp_start_rank" 0 "$((role_ranks - 1))"
    validate_integer AFD_PORT "$afd_port" 1 65535
    validate_integer API_PORT "$api_port" 1 65535
    validate_integer DP_RPC_PORT "$rpc_port" 1 65535
    validate_integer MASTER_PORT "$master_port" 1 65535
    validate_integer U_BATCHES "$batches" 1 2
    validate_integer DISABLE_ENGRAM "$disable_engram" 0 1
    validate_integer MAX_NUM_BATCHED_TOKENS "$max_batched_tokens" 1 512
    validate_integer MAX_NUM_SEQS "$max_seqs" 1 "$max_batched_tokens"
    validate_integer DSPARK_TOKENS "$dspark_tokens" 0 "$((max_batched_tokens - 1))"
    if (( batches == 2 )); then
        [[ "$disable_engram" == 1 ]] || fail 'U2 requires DISABLE_ENGRAM=1 on both roles'
        (( max_seqs >= 2 && max_batched_tokens >= 2 )) ||
            fail 'U2 requires MAX_NUM_SEQS >= 2 and MAX_NUM_BATCHED_TOKENS >= 2'
        if [[ "$execution_mode" == graph ]] && (( dspark_tokens > 0 )); then
            local min_capture_tokens=$((2 * (dspark_tokens + 1)))
            (( max_batched_tokens >= min_capture_tokens )) ||
                fail "Graph/U2 DSpark requires MAX_NUM_BATCHED_TOKENS >= $min_capture_tokens"
        fi
    fi
    if (( batches == 2 && attention_ranks < 2 && dspark_tokens == 0 )); then
        fail 'U2 requires at least two Attention ranks when DSpark is disabled'
    fi

    local -a local_devices
    local -A seen=()
    local device
    IFS=, read -ra local_devices <<< "$devices"
    for device in "${local_devices[@]}"; do
        validate_integer device "$device" 0 65535
        [[ -z "${seen[$device]:-}" ]] || fail "device $device is listed twice"
        seen[$device]=1
    done
    (( dp_start_rank + ${#local_devices[@]} <= role_ranks )) ||
        fail "local $role DP range exceeds configured ranks"

    local additional_config
    additional_config="$("$python_bin" -c '
import json
import sys

role, host, port, attention, ffn, batches = sys.argv[1:]
print(json.dumps({
    "afd": {
        "role": role,
        "connector": "WindowAFDConnector",
        "async": True,
        "host": host,
        "port": int(port),
        "num_attention_ranks": int(attention),
        "num_ffn_ranks": int(ffn),
        "compute_gate_on_attention": True,
        "connector_extra_config": {"micro_batch_num": int(batches), "quant_mode": 4},
    },
    "enable_flashcomm1": False,
    "mix_placement": False,
}))' "$role" "$afd_host" "$afd_port" "$attention_ranks" "$ffn_ranks" "$batches")"

    local -a args=(
        --tensor-parallel-size 1 --enable-expert-parallel
        --all2all-backend flashinfer_all2allv --distributed-executor-backend mp
        --data-parallel-size "$role_ranks" --data-parallel-size-local "${#local_devices[@]}"
        --data-parallel-address "$dp_address" --data-parallel-rpc-port "$rpc_port"
        --master-port "$master_port" --language-model-only
        --dtype bfloat16 --quantization deepseek_v4_fp8 --load-format safetensors
        --tokenizer-mode deepseek_v41 --trust-remote-code --block-size 128
        --max-model-len "${MAX_MODEL_LEN:-4096}" --max-num-seqs "$max_seqs"
        --max-num-batched-tokens "$max_batched_tokens" --enable-chunked-prefill
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
        --no-enable-prefix-caching --no-async-scheduling --shutdown-timeout 30
        --additional-config "$additional_config"
    )
    if (( dp_start_rank == 0 )); then
        args+=(--host "$api_host" --port "$api_port" --api-server-count 1)
    else
        args+=(--headless --data-parallel-start-rank "$dp_start_rank")
    fi
    if [[ "$role" == attention && "$batches" == 2 ]]; then
        args+=(--enable-dbo --ubatch-size 2 --dbo-decode-token-threshold 2
              --dbo-prefill-token-threshold 2)
    else
        args+=(--no-enable-dbo --ubatch-size 0)
    fi
    if [[ "$disable_engram" == 1 ]]; then
        args+=(--hf-overrides '{"engram_layer_ids":[]}')
    fi

    local width=$((dspark_tokens + 1))
    if [[ "$role" == attention && "$dspark_tokens" != 0 ]]; then
        local speculative_config
        speculative_config="$("$python_bin" -c '
import json
import sys

print(json.dumps({
    "method": "dspark",
    "model": sys.argv[1],
    "num_speculative_tokens": int(sys.argv[2]),
    "enforce_eager": True,
    "enable_adaptive_verification": False,
}))' "$model_path" "$dspark_tokens")"
        args+=(--speculative-config "$speculative_config")
    fi
    case "$execution_mode" in
        eager)
            args+=(--enforce-eager --compilation-config
                  '{"mode":0,"cudagraph_mode":"NONE","pass_config":{"enable_sp":false}}')
            ;;
        graph)
            local capture_sizes="[$width]"
            if (( 2 * width <= max_batched_tokens )); then
                capture_sizes="[$width,$((2 * width))]"
            fi
            args+=(--compilation-config
                  "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":$capture_sizes,\"pass_config\":{\"enable_sp\":false}}")
            ;;
        *)
            fail 'EXECUTION_MODE must be eager or graph'
            ;;
    esac

    local -a launch_env=(
        "ASCEND_RT_VISIBLE_DEVICES=$devices"
        "HCCL_IF_IP=$hccl_ip" "HCCL_SOCKET_IFNAME=$nic_name" "GLOO_SOCKET_IFNAME=$nic_name"
        "VLLM_HOST_IP=$hccl_ip" VLLM_USE_V2_MODEL_RUNNER=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
        "VLLM_PLUGINS=${VLLM_PLUGINS:-ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling,afd}"
    )
    if [[ -n "${TRANSFORMER_VENDOR:-}" ]]; then
        launch_env+=(
            "ASCEND_CUSTOM_OPP_PATH=$TRANSFORMER_VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
            "LD_LIBRARY_PATH=$TRANSFORMER_VENDOR/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        )
    fi

    if [[ "$action" == --dry-run ]]; then
        printf '%q' env
        printf ' %q' "${launch_env[@]}" "$vllm_bin" serve "$model_path" "${args[@]}"
        printf '\n'
        return
    fi
    [[ -d "$model_path" ]] || fail 'MODEL_PATH must point to a local checkpoint directory'
    exec env "${launch_env[@]}" "$vllm_bin" serve "$model_path" "${args[@]}"
}
