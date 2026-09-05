#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: bash analyse_pd_profile.sh <generated-role-config.env> <*_ascend_pt-root>

Analyse one stopped Attention/FFN profile with the exact CANN and Python
runtime pinned by the generated PD Graph matrix config.
EOF
}

die() {
  printf '[analyse-pd-profile] ERROR: %s\n' "$*" >&2
  exit 2
}

log() {
  printf '[analyse-pd-profile] %s\n' "$*"
}

[[ $# == 2 ]] || {
  usage
  exit 2
}

CONFIG_PATH="$1"
PROFILE_ROOT="$2"
[[ -f "${CONFIG_PATH}" ]] || die "Config does not exist: ${CONFIG_PATH}"
[[ -d "${PROFILE_ROOT}" ]] || die "Profile root does not exist: ${PROFILE_ROOT}"
CONFIG_PATH="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"
PROFILE_ROOT="$(cd "${PROFILE_ROOT}" && pwd)"
[[ "$(basename "${PROFILE_ROOT}")" != "ASCEND_PROFILER_OUTPUT" ]] \
  || die "Pass the parent *_ascend_pt root, not ASCEND_PROFILER_OUTPUT"
[[ -f "${PROFILE_ROOT}/profiler_info_0.json" ]] \
  || die "Missing profiler_info_0.json: ${PROFILE_ROOT}"

set -a
# shellcheck disable=SC1090
source "${CONFIG_PATH}"
set +a
: "${AFD_PLUGIN_ROOT:?AFD_PLUGIN_ROOT is required by the matrix config}"
: "${CANN_ROOT:?CANN_ROOT is required by the matrix config}"
: "${CANN_VERSION:?CANN_VERSION is required by the matrix config}"
: "${VENV_ROOT:?VENV_ROOT is required by the matrix config}"
[[ -f "${AFD_PLUGIN_ROOT}/tools/dsv4/activate_runtime.sh" ]] \
  || die "Missing runtime activator under AFD_PLUGIN_ROOT=${AFD_PLUGIN_ROOT}"
[[ -x "${VENV_ROOT}/bin/python" ]] \
  || die "Missing configured Python: ${VENV_ROOT}/bin/python"
[[ -f "${CANN_ROOT}/set_env.sh" ]] \
  || die "Missing configured CANN environment: ${CANN_ROOT}/set_env.sh"

resolved_cann="$(readlink -f "${CANN_ROOT}")"
version_text="${resolved_cann}"
if [[ -x "${CANN_ROOT}/query_pkg_version.sh" ]]; then
  version_text+=$'\n'"$("${CANN_ROOT}/query_pkg_version.sh" 2>&1 || true)"
fi
version_file="$(find "${CANN_ROOT}" -name version.info -type f \
  -print -quit 2>/dev/null || true)"
if [[ -n "${version_file}" ]]; then
  version_text+=$'\n'"$(head -n 20 "${version_file}" 2>/dev/null || true)"
fi
expected_cann_regex="${CANN_VERSION//./\\.}"
grep -Eq "(^|[^0-9])${expected_cann_regex}([^0-9]|$)" <<<"${version_text}" \
  || die "Configured CANN ${CANN_VERSION} was not detected at ${resolved_cann}"

captured_cann="$("${VENV_ROOT}/bin/python" - "${PROFILE_ROOT}/profiler_info_0.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream).get("cann_version") or "")
PY
)"
[[ -n "${captured_cann}" ]] \
  || die "profiler_info_0.json does not record cann_version"
[[ "${captured_cann}" == "${CANN_VERSION}" ]] \
  || die "Captured CANN ${captured_cann} does not match config CANN ${CANN_VERSION}"

mapfile -t raw_roots < <(find "${PROFILE_ROOT}" -maxdepth 1 -type d \
  -name 'PROF_*' -print | sort)
(( ${#raw_roots[@]} == 1 )) \
  || die "Expected exactly one raw PROF_* directory, found ${#raw_roots[@]}"
mapfile -t device_end_markers < <(find "${raw_roots[0]}" -type f \
  -path '*/device_*/*' -name 'end_info*.done' -print | sort)
mapfile -t host_end_markers < <(find "${raw_roots[0]}/host" -maxdepth 1 \
  -type f -name 'end_info.done' -print 2>/dev/null | sort)
(( ${#device_end_markers[@]} > 0 && ${#host_end_markers[@]} > 0 )) || die \
  "Raw CANN capture is incomplete: expected device_*/end_info*.done and host/end_info.done under ${raw_roots[0]}; offline analyse cannot repair an unfinalized capture, so rerun the Profile round after checking graceful profiler shutdown and free disk space"

required_outputs=(kernel_details.csv trace_view.json communication.json)
output_dir="${PROFILE_ROOT}/ASCEND_PROFILER_OUTPUT"
already_complete=1
for name in "${required_outputs[@]}"; do
  [[ -s "${output_dir}/${name}" ]] || already_complete=0
done
if (( already_complete )) && [[ -f "${output_dir}/analyse.done" ]]; then
  log "Derived outputs are already complete: ${output_dir}"
  exit 0
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
if [[ -d "${output_dir}" ]]; then
  backup_output="${PROFILE_ROOT}/ASCEND_PROFILER_OUTPUT.failed-${timestamp}"
  [[ ! -e "${backup_output}" ]] || die "Backup path already exists: ${backup_output}"
  mv "${output_dir}" "${backup_output}"
  log "Preserved incomplete derived output: ${backup_output}"
fi
if [[ -d "${PROFILE_ROOT}/logs" ]]; then
  backup_logs="${PROFILE_ROOT}/logs.failed-${timestamp}"
  [[ ! -e "${backup_logs}" ]] || die "Backup path already exists: ${backup_logs}"
  mv "${PROFILE_ROOT}/logs" "${backup_logs}"
  log "Preserved previous parser logs: ${backup_logs}"
fi

export DSV4_CANN_ROOT="${CANN_ROOT}"
export DSV4_CANN_VERSION="${CANN_VERSION}"
export DSV4_ATB_ROOT="${ATB_ROOT:-}"
export DSV4_RUNTIME_VENV="${VENV_ROOT}"
export DSV4_VLLM_VENV="${VENV_ROOT}"
export DSV4_VLLM_ROOT="${VLLM_ROOT:-}"
export DSV4_VLLM_ASCEND_ROOT="${VLLM_ASCEND_ROOT:-}"
# shellcheck disable=SC1091
source "${AFD_PLUGIN_ROOT}/tools/dsv4/activate_runtime.sh"

runtime_versions="$(python - <<'PY'
import json
import sys

import torch_npu

print(json.dumps({
    "python": sys.executable,
    "torch_npu": getattr(torch_npu, "__version__", "unknown"),
}))
PY
)"
log "Captured CANN: ${captured_cann}"
log "Analyzer runtime: ${runtime_versions}"

cd "${PROFILE_ROOT}"
python -c 'from torch_npu.profiler.profiler import analyse; analyse("./")'

missing=()
for name in "${required_outputs[@]}"; do
  [[ -s "${output_dir}/${name}" ]] || missing+=("${name}")
done
[[ -f "${output_dir}/analyse.done" ]] || missing+=(analyse.done)
if (( ${#missing[@]} > 0 )); then
  if [[ -d "${PROFILE_ROOT}/logs" ]]; then
    grep -RniE 'ERROR|Failed|Traceback|ERR[0-9]+' \
      "${PROFILE_ROOT}/logs" 2>/dev/null | tail -n 80 >&2 || true
  fi
  die "Analysis did not produce required artifacts: ${missing[*]}"
fi
log "Analysis outputs validated: ${output_dir}"
