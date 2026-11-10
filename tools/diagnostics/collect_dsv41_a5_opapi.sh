#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
# Inspect CompressorV2 library availability without importing torch or using NPU.

set -uo pipefail
if [[ "${1:-}" == "--help" ]]; then
    printf 'Usage: AFD_PYTHON=/path/to/python bash tools/diagnostics/collect_dsv41_a5_opapi.sh\n'
    printf 'Read-only library inspection. Prints one results directory; no installation or inference.\n'
    exit 0
fi
if (( $# != 0 )); then
    printf 'Unexpected argument; use --help.\n' >&2
    exit 2
fi

afd_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd) || exit 1
cd -- "$afd_repo_root" || exit 1
afd_python=${AFD_PYTHON:-/usr/local/python3.11.10/bin/python}
mkdir -p results || exit 1
afd_opapi_run=$(mktemp -d "$afd_repo_root/results/dsv41-opapi.XXXXXX") || exit 1
printf 'results: %s\n' "$afd_opapi_run"
exec > "$afd_opapi_run/collector.log" 2>&1
{
    date -u
    uname -a
    "$afd_python" -V
    "$afd_python" -m pip show cann_ops_transformer cann_ops_nn torch torch-npu vllm-ascend
    git rev-parse HEAD
    for afd_var in ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH ASCEND_CUSTOM_OPP_PATH LD_LIBRARY_PATH; do
        printf '%s=%s\n' "$afd_var" "${!afd_var-}"
    done
} > "$afd_opapi_run/environment.txt" 2>&1

# Bound discovery to the CANN installation and the installed Python package.
find /usr/local/Ascend -type f \( -name 'libopapi.so*' -o -name 'libopapi_transformer.so*' \
    -o -name 'libcust_opapi.so*' -o -name '*version.info' -o -name '*install.info' \) \
    > "$afd_opapi_run/cann-files.txt" 2> "$afd_opapi_run/discovery.log"
while IFS= read -r afd_info; do
    case "$afd_info" in
        *version.info|*install.info)
            printf '%s\n' "$afd_info"
            head -n 80 "$afd_info"
            ;;
    esac
done < "$afd_opapi_run/cann-files.txt" > "$afd_opapi_run/cann-versions.txt" 2>&1

"$afd_python" - "$afd_opapi_run" <<'PY' > "$afd_opapi_run/libraries.json" 2> "$afd_opapi_run/libraries.log"
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

output = Path(sys.argv[1])
names = ("libopapi_transformer.so", "libopapi.so", "libcust_opapi.so")
candidates = list(names)
for line in (output / "cann-files.txt").read_text().splitlines():
    if Path(line).name.startswith(names):
        candidates.append(line)
for directory in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
    if directory:
        candidates.extend(str(Path(directory) / name) for name in names if (Path(directory) / name).is_file())
for variable in ("ASCEND_CUSTOM_OPP_PATH", "ASCEND_OPP_PATH", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
    for directory in os.environ.get(variable, "").split(":"):
        if not directory:
            continue
        for relative in ("op_api/lib", "lib64", "lib"):
            candidates.extend(str(Path(directory) / relative / name) for name in names
                              if (Path(directory) / relative / name).is_file())
packages = {}
for name in ("cann_ops_transformer", "cann_ops_nn"):
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = {"installed": False}
        continue
    packages[name] = {"installed": True, "version": dist.version,
                      "root": str(dist.locate_file(""))}
    for entry in dist.files or ():
        if Path(entry).name.startswith(names):
            candidates.append(str(dist.locate_file(entry)))

# A fresh process per candidate avoids one probe changing later dlopen results.
probe = r'''
import ctypes
import json
import os
import sys
from pathlib import Path
record = {"requested": sys.argv[1]}
try:
    library = ctypes.CDLL(sys.argv[1], mode=os.RTLD_LAZY | os.RTLD_LOCAL)
    record["loaded"] = True
    record["symbols"] = {}
    for symbol in ("aclnnCompressorV2", "aclnnCompressorV2GetWorkspaceSize"):
        try:
            library[symbol]
            record["symbols"][symbol] = True
        except AttributeError:
            record["symbols"][symbol] = False
    record["mapped_opapi_paths"] = sorted({line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines() if "opapi" in line})
except OSError as exc:
    record.update(loaded=False, error=str(exc))
print(json.dumps(record))
'''
libraries = []
for candidate in dict.fromkeys(candidates):
    try:
        result = subprocess.run([sys.executable, "-c", probe, candidate],
                                capture_output=True, text=True, timeout=30)
        if result.returncode:
            record = {"requested": candidate, "exit_code": result.returncode,
                      "stdout": result.stdout, "stderr": result.stderr}
        else:
            record = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        record = {"requested": candidate, "error": "library load timed out after 30 seconds"}
    libraries.append(record)
report = {"scope": "Library discovery and symbol lookup only; no torch import, NPU execution or inference",
          "packages": packages, "libraries": libraries}
print(json.dumps(report, indent=2))
PY
afd_collect_status=$?
printf '%s\n' "$afd_collect_status" > "$afd_opapi_run/collector-exit-code.txt"
tar -czf "$afd_opapi_run.tar.gz" -C "$(dirname -- "$afd_opapi_run")" "$(basename -- "$afd_opapi_run")" || exit 1
exit "$afd_collect_status"
