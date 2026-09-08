#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/common.sh
source "${SCRIPT_DIR}/../lib/common.sh"

require_file "${VENV_ROOT}/bin/python"
require_file "${VLLM_ROOT}/requirements/common.txt"
require_file "${VLLM_ASCEND_ROOT}/requirements.txt"

python_bin="${VENV_ROOT}/bin/python"
pip_args=()

assert_zero_or_one INSTALL_PYTHON_DEPS "${INSTALL_PYTHON_DEPS}"
if ! is_true "${INSTALL_PYTHON_DEPS}"; then
  log "Auditing the validated Python environment without reinstalling dependencies"
  "${python_bin}" - <<'PY'
from importlib.metadata import version

expected = {
    "torch": "2.10.0",
    "torch-npu": "2.10.0.post2",
    "vllm": "0.23.0+empty",
    "vllm-ascend": "0.1.dev1+g3da28f941",
    "transformers": "5.5.4",
    "numpy": "2.2.6",
    "triton-ascend": "3.2.1",
}
actual = {name: version(name) for name in expected}
assert actual == expected, (actual, expected)
for name, value in actual.items():
    print(name, value)
PY
  ensure_dir "${STATE_ROOT}"
  "${python_bin}" -m pip list --format=freeze \
    >"${STATE_ROOT}/python-packages-reused.txt"
  log "Validated Python dependencies will be reused"
  exit 0
fi

if is_true "${OFFLINE}"; then
  require_dir "${WHEELHOUSE}"
  pip_args+=(--no-index --find-links "${WHEELHOUSE}")
else
  if [[ -n "${PIP_INDEX_URL}" ]]; then
    pip_args+=(--index-url "${PIP_INDEX_URL}")
  fi
  if [[ -n "${PIP_EXTRA_INDEX_URL}" ]]; then
    pip_args+=(--extra-index-url "${PIP_EXTRA_INDEX_URL}")
  fi
  if [[ -n "${PIP_TRUSTED_HOST}" ]]; then
    pip_args+=(--trusted-host "${PIP_TRUSTED_HOST}")
  fi
fi

log "Installing Python build dependencies"
"${python_bin}" -m pip install "${pip_args[@]}" --upgrade \
  pip "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8" \
  "setuptools-rust>=1.9.0" "packaging>=24.2" wheel jinja2 \
  "cmake>=3.26.1" ninja pybind11

log "Installing torch and Ascend Python runtime"
"${python_bin}" -m pip install "${pip_args[@]}" \
  torch==2.10.0 \
  torchvision==0.25.0 \
  torchaudio==2.10.0 \
  torch-npu==2.10.0.post2 \
  triton-ascend==3.2.1

log "Installing vLLM common requirements"
"${python_bin}" -m pip install "${pip_args[@]}" \
  -r "${VLLM_ROOT}/requirements/common.txt"

log "Installing vLLM-Ascend requirements"
"${python_bin}" -m pip install "${pip_args[@]}" \
  -r "${VLLM_ASCEND_ROOT}/requirements.txt"

# The target branch metadata still says torch-npu 2.10.0 and triton-ascend
# metadata pins numpy 1.26.4. The validated runtime intentionally restores
# these exact final versions after dependency resolution.
log "Restoring validated runtime pins"
"${python_bin}" -m pip install "${pip_args[@]}" \
  --upgrade --force-reinstall --no-deps \
  torch-npu==2.10.0.post2 \
  transformers==5.5.4 \
  numpy==2.2.6

ensure_dir "${STATE_ROOT}"
"${python_bin}" -m pip list --format=freeze \
  >"${STATE_ROOT}/python-packages-after-deps.txt"
log "Python dependencies installed"
