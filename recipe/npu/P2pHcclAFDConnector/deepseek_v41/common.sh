#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=0
case "$#:${1:-}" in
    0:) ;;
    1:--dry-run) DRY_RUN=1 ;;
    *) printf 'Usage: %s [--dry-run]\n' "$0" >&2; exit 2 ;;
esac

fail() {
    printf '%s\n' "$*" >&2
    exit 2
}

run_command() {
    if (( DRY_RUN )); then
        printf '%q ' "$@"
        printf '\n'
        return
    fi
    exec "$@"
}

launch_role() {
    local role="$1"
    local mode="${DSV41_MODE:-eager-u1}"
    local spec_tokens="${DSV41_SPEC_TOKENS:-0}"
    local pd_role="${DSV41_PD_ROLE:-none}"
    local port_offset="${DSV41_PORT_OFFSET:-0}"
    local afd_host="${AFD_HOST:-127.0.0.1}"
    local kv_ip="${KV_IP:-}"
    local graph=0 u2=0
    local model="${MODEL_PATH:?Set MODEL_PATH to the DeepSeek-V4.1-Flash checkpoint}"
    local vendor="${TRANSFORMER_VENDOR:?Set TRANSFORMER_VENDOR to the Ascend transformer vendor directory}"
    local -a environment command scheduling speculation pd

    case "$mode" in
        eager-u1) ;;
        eager-u2) u2=1 ;;
        graph-u1) graph=1 ;;
        graph-u2) graph=1; u2=1 ;;
        *) fail "DSV41_MODE must be eager-u1, eager-u2, graph-u1 or graph-u2" ;;
    esac
    case "$spec_tokens" in
        0|1|2|5) ;;
        *) fail "DSV41_SPEC_TOKENS must be 0, 1, 2 or 5" ;;
    esac
    case "$pd_role" in
        none) ;;
        producer|consumer)
            [[ -n "$kv_ip" ]] || fail "Set KV_IP to the local address reachable by the other PD instance"
            ;;
        *) fail "DSV41_PD_ROLE must be none, producer or consumer" ;;
    esac
    if [[ "$pd_role" == producer && "$mode" != eager-u1 ]]; then
        fail "The PD producer requires DSV41_MODE=eager-u1"
    fi
    [[ "$port_offset" =~ ^(0|[1-9][0-9]{0,4})$ ]] && (( port_offset <= 35732 )) \
        || fail "DSV41_PORT_OFFSET must be an integer between 0 and 35732"

    local attention_devices="${ATTENTION_DEVICES:-0,1,2,3}"
    local ffn_devices="${FFN_DEVICES:-4,5,6,7}"
    local additional_config
    additional_config="$("$PYTHON_BIN" -c '
import json
import sys
role, host, port, attention, ffn, async_engram = sys.argv[1:]
def parse_devices(value):
    items = value.split(",")
    if len(items) != 4 or any(not item.isascii() or not item.isdecimal() for item in items):
        raise SystemExit("Each role requires a comma-separated list of four NPU IDs")
    return {int(item) for item in items}
if len(parse_devices(attention) | parse_devices(ffn)) != 8:
    raise SystemExit("Attention and FFN require eight distinct device IDs")
print(json.dumps({
    "afd": {"role": role, "connector": "P2pHcclAFDConnector",
            "host": host, "port": int(port), "num_attention_ranks": 4,
            "num_ffn_ranks": 4, "compute_gate_on_attention": False,
            "afd_process_group_timeout_s": 1200,
            "connector_extra_config": {"max_tokens": 1024}},
    "enable_cpu_binding": False,
    "multistream_engram_overlap": async_engram == "1",
    "multistream_overlap_shared_expert": False,
    "multistream_dsv4_dsa_overlap": False,
    "combine_quant_mode": 4,
    "enable_flashcomm1": False,
    "enable_dsa_cp": False,
}))
' "$role" "$afd_host" "$((29600 + port_offset))" "$attention_devices" "$ffn_devices" "$((graph || u2))")"

    local devices master_port rpc_port native_port
    if [[ "$role" == attention ]]; then
        devices="$attention_devices"
        master_port=29501
        rpc_port=29550
        native_port=29700
        command=("$PYTHON_BIN" -B -m afd_plugin.entrypoints.dsv41_ep_stack serve "$model"
            --host "${ATTENTION_HOST:-127.0.0.1}" --port "$((18031 + port_offset))"
            --served-model-name "${SERVED_MODEL_NAME:-dsv41-afd}")
        environment=("AFD_DSV41_EP_SHUTDOWN_RELAY=1")
    else
        devices="$ffn_devices"
        master_port=29511
        rpc_port=29560
        native_port=29800
        command=("$PYTHON_BIN" -B -m afd_plugin.entrypoints.npu_ffn --model "$model")
        environment=()
    fi

    local batch_tokens=128
    if (( u2 )); then
        scheduling=(--enable-dbo --ubatch-size 2
            --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 2)
    else
        scheduling=(--no-enable-dbo --ubatch-size 0)
    fi
    if (( graph )); then
        local width=$((spec_tokens + 1))
        scheduling+=(--compilation-config
            "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$width,$((2 * width))],\"pass_config\":{\"enable_sp\":false}}")
    else
        scheduling+=(--enforce-eager --compilation-config
            '{"mode":0,"cudagraph_mode":"NONE","pass_config":{"enable_sp":false}}')
    fi

    speculation=()
    if (( spec_tokens )); then
        speculation=(--speculative-config "$("$PYTHON_BIN" -c '
import json
import sys
print(json.dumps({"method": "dspark", "model": sys.argv[1],
                  "num_speculative_tokens": int(sys.argv[2]),
                  "enforce_eager": True, "enable_adaptive_verification": False}))
' "$model" "$spec_tokens")" --per-request-spec-decode-metrics detailed)
    fi

    pd=()
    if [[ "$pd_role" != none ]]; then
        pd=(--kv-transfer-config "$("$PYTHON_BIN" -c '
import json
import sys
role, ip, port, engine_id = sys.argv[1:]
parallel = {"tp_size": 1, "dp_size": 4, "pp_size": 1}
print(json.dumps({"kv_connector": "MooncakeHybridConnector",
                  "kv_connector_module_path": "afd_plugin.compat.npu.kv_layout_connector",
                  "kv_role": "kv_" + role, "engine_id": engine_id,
                  "kv_buffer_device": "npu", "kv_ip": ip, "kv_port": int(port),
                  "kv_connector_extra_config": {"prefill": parallel, "decode": parallel}}))
' "$pd_role" "$kv_ip" "$((14579 + port_offset))" "${DSV41_ENGINE_ID:-dsv41-$pd_role}")")
        if [[ "$role" == attention ]]; then
            environment+=("VLLM_HOST_IP=$kv_ip")
        fi
    fi

    if (( ! DRY_RUN )); then
        [[ -f "$model/config.json" ]] || fail "Checkpoint config missing: $model/config.json"
        [[ -f "$vendor/op_api/lib/libcust_opapi.so" ]] \
            || fail "Ascend custom operator library missing: $vendor/op_api/lib/libcust_opapi.so"
        printf 'Starting %s with launcher PID %s\n' "$role" "$$" >&2
    fi

    local plugins="ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling,afd"
    run_command env \
        "${environment[@]}" \
        "ASCEND_RT_VISIBLE_DEVICES=$devices" \
        "ASCEND_CUSTOM_OPP_PATH=$vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}" \
        "LD_LIBRARY_PATH=$vendor/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "VLLM_PORT=$((native_port + port_offset))" \
        "VLLM_PLUGINS=$plugins" \
        "VLLM_USE_V2_MODEL_RUNNER=1" \
        "VLLM_WORKER_MULTIPROC_METHOD=spawn" \
        "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=180" \
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000" \
        "AFD_DSV41_FULL_QUALIFICATION=1" "AFD_DSV41_EAGER_QUALIFICATION=0" \
        "AFD_DSV41_MXFP8_SCALE_PATCH=0" "AFD_ASYNC_CAM_LAYERED_GMM=0" \
        "AFD_HCCL_STAGE_DIAGNOSTICS=${AFD_HCCL_STAGE_DIAGNOSTICS:-0}" \
        "AFD_HCCL_EAGER_U2_STREAM_OVERLAP=0" "AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP=0" \
        "AFD_HCCL_GRAPH_U2_HYBRID_DAG=0" "AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM=0" \
        "AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM=0" "AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER=0" \
        "HCCL_OP_EXPANSION_MODE=CCU_SCHED" "HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-1600}" \
        "HCCL_CONNECT_TIMEOUT=1200" "HCCL_EXEC_TIMEOUT=2040" \
        "OMP_PROC_BIND=false" "OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}" \
        "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True" "PYTHONUNBUFFERED=1" \
        "${command[@]}" \
        --tensor-parallel-size 1 --data-parallel-size 4 --data-parallel-size-local 4 \
        --pipeline-parallel-size 1 --prefill-context-parallel-size 1 \
        --decode-context-parallel-size 1 --distributed-executor-backend mp \
        --enable-expert-parallel --all2all-backend flashinfer_all2allv \
        --language-model-only --dtype bfloat16 --quantization deepseek_v4_fp8 \
        --load-format safetensors --tokenizer-mode deepseek_v41 --trust-remote-code \
        --block-size 128 --max-model-len 1024 --max-num-seqs 2 \
        --max-num-batched-tokens "$batch_tokens" --gpu-memory-utilization 0.90 \
        --engram-config.cpu_offload true --engram-config.embedding_across_dp false \
        --engram-config.dp_shared_memory true --no-async-scheduling \
        --no-enable-prefix-caching --enable-chunked-prefill --generation-config vllm \
        --shutdown-timeout 180 --master-port "$((master_port + port_offset))" \
        --data-parallel-rpc-port "$((rpc_port + port_offset))" \
        --additional-config "$additional_config" \
        "${scheduling[@]}" "${speculation[@]}" "${pd[@]}"
}
