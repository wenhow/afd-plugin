#!/usr/bin/env bash
set -eo pipefail

VLLM_ASCEND_ROOT="${DSV4_VLLM_ASCEND_ROOT:-/mnt/workspace/code/vllm-ascend-rfc-vllm-cann}"
PROXY_SCRIPT="${VLLM_ASCEND_ROOT}/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
PYTHON_BIN="${DSV4_RUNTIME_VENV:-/mnt/workspace/code/.venvs/afd-v023-vllm-cann}/bin/python"

PREFILL_HOSTS="${PREFILL_HOSTS:-127.0.0.1}"
PREFILL_PORTS="${PREFILL_PORTS:-8100}"
DECODE_HOSTS="${DECODE_HOSTS:-127.0.0.1}"
DECODE_PORTS="${DECODE_PORTS:-8910}"
read -r -a PREFILL_HOST_ARGS <<<"$PREFILL_HOSTS"
read -r -a PREFILL_PORT_ARGS <<<"$PREFILL_PORTS"
read -r -a DECODE_HOST_ARGS <<<"$DECODE_HOSTS"
read -r -a DECODE_PORT_ARGS <<<"$DECODE_PORTS"

# httpx trusts proxy environment variables by default.  AFD backends are
# private-network services and must never be routed through an HTTP gateway.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

exec "$PYTHON_BIN" "$PROXY_SCRIPT" \
  --host "${PROXY_HOST:-0.0.0.0}" \
  --port "${PROXY_PORT:-9000}" \
  --workers "${PROXY_WORKERS:-1}" \
  --prefiller-hosts "${PREFILL_HOST_ARGS[@]}" \
  --prefiller-ports "${PREFILL_PORT_ARGS[@]}" \
  --decoder-hosts "${DECODE_HOST_ARGS[@]}" \
  --decoder-ports "${DECODE_PORT_ARGS[@]}"
