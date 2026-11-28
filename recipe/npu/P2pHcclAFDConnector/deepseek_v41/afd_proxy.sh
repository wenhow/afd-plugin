#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

run_command "$PYTHON_BIN" -B "$SCRIPT_DIR/../../../../tools/proxy_server.py" \
    --host "${PROXY_HOST:-127.0.0.1}" --port "${PROXY_PORT:-18030}" \
    --prefiller-hosts "${PREFILL_HOST:?Set PREFILL_HOST to the Attention API host on P}" \
    --prefiller-ports "${PREFILL_PORT:-18031}" \
    --decoder-hosts "${DECODE_HOST:?Set DECODE_HOST to the Attention API host on D}" \
    --decoder-ports "${DECODE_PORT:-19031}"
