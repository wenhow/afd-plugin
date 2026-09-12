#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

for command_name in sha256sum cp mv chmod date; do
  require_command "${command_name}"
done

official_config="${BUNDLE_ROOT}/model/DeepSeek-V4-Flash-config.json"
target_config="${MODEL_PATH}/config.json"
require_file "${official_config}"
require_file "${target_config}"
require_file "${MODEL_PATH}/model.safetensors.index.json"

if "${PYTHON_BIN}" - "${official_config}" "${target_config}" <<'PY'
import json
import sys
from pathlib import Path

expected = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
actual = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
raise SystemExit(expected != actual)
PY
then
  log "A5 model config already matches the bundled official config"
  exit 0
fi

"${PYTHON_BIN}" - "${target_config}" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert config.get("model_type") == "deepseek_v4", config.get("model_type")
assert config.get("architectures") == ["DeepseekV4ForCausalLM"], config.get("architectures")
assert config.get("num_nextn_predict_layers") == 1, config.get("num_nextn_predict_layers")
PY

ensure_dir "${STATE_ROOT}/model-config-backup"
timestamp="$(date +%Y%m%d_%H%M%S)"
current_sha="$(sha256sum "${target_config}" | awk '{print $1}')"
backup="${STATE_ROOT}/model-config-backup/config.json.${timestamp}.${current_sha}"
cp -p "${target_config}" "${backup}"

temporary_config="${MODEL_PATH}/.config.json.afd-${timestamp}"
cp "${official_config}" "${temporary_config}"
chmod --reference="${target_config}" "${temporary_config}" 2>/dev/null || chmod 0644 "${temporary_config}"
mv "${temporary_config}" "${target_config}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/model_launch_args.py" \
  --model-path "${MODEL_PATH}" \
  --quantization deepseek-v4-native \
  --block-size 32 \
  --safetensors-load-strategy prefetch \
  --kv-cache-dtype auto \
  --describe >"${STATE_ROOT}/model-config-after-restore.json"

log "Installed the bundled official DeepSeek-V4-Flash config: ${target_config}"
log "Previous config is recoverable from: ${backup}"
