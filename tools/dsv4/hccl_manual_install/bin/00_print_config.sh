#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

resolved_ip="$(resolve_hccl_ip || true)"

print_version_contract
cat <<EOF
[hccl-install] install root: ${INSTALL_ROOT}
[hccl-install] CANN root: ${CANN_ROOT}
[hccl-install] Python: ${PYTHON_BIN}
[hccl-install] venv: ${VENV_ROOT}
[hccl-install] model: ${MODEL_PATH}
[hccl-install] SoC: ${SOC_VERSION}
[hccl-install] NIC: ${NIC_NAME}
[hccl-install] HCCL IP: ${resolved_ip:-<unresolved>}
[hccl-install] topology: A${ATTENTION_RANKS}F${FFN_RANKS}
[hccl-install] Attention devices: ${ATTENTION_DEVICES}
[hccl-install] FFN devices: ${FFN_DEVICES}
[hccl-install] mode: ${EXECUTION_MODE}/U${U_BATCHES}, MTP=${ENABLE_MTP}
[hccl-install] bundled sources: ${USE_BUNDLED_SOURCES}
[hccl-install] AFD seed bundle: ${USE_AFD_SEED_BUNDLE}
[hccl-install] reuse venv: ${REUSE_VENV}
[hccl-install] install Python deps: ${INSTALL_PYTHON_DEPS}
[hccl-install] install upstream stack: ${INSTALL_UPSTREAM_STACK}
[hccl-install] offline Python deps: ${OFFLINE}
EOF
