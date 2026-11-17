#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source /mnt/workspace/code/.ascend/cann-9.2.0/ascend-toolkit/set_env.sh >/dev/null
source /mnt/workspace/code/.ascend/cann-9.2.0/nnal/atb/set_env.sh >/dev/null
export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH}
export PATH=/mnt/workspace/code/.venvs/afd-v030-dsv41-c0/bin:${PATH}
package=/mnt/workspace/tmp/afd-c0-20261104.hVgsIf/dspark-7e-qli-20261116-feigmgju/package
vendor=${package}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export PYTHONPATH=${PWD}/snapshot:${PWD}/snapshot/tests/npu:${package}:${PYTHONPATH}
export LD_LIBRARY_PATH=${vendor}/op_api/lib:${package}/vllm_ascend:${LD_LIBRARY_PATH}
export ASCEND_CUSTOM_OPP_PATH=${vendor}
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export TORCH_EXTENSIONS_DIR=/mnt/workspace/tmp/afd-c0-20261104.hVgsIf/functional-script-check-20261117-aed7k8py/cache/torch-extensions
export TRITON_CACHE_DIR=${PWD}/cache/triton
export XDG_CACHE_HOME=${PWD}/cache/xdg
export ASCEND_CACHE_PATH=${PWD}/cache/ascend-cache
export ASCEND_WORK_PATH=${PWD}/cache/ascend-work
export ASCEND_PROCESS_LOG_PATH=${PWD}/cache/ascend-logs
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$ASCEND_CACHE_PATH" "$ASCEND_WORK_PATH" "$ASCEND_PROCESS_LOG_PATH"
python -B verify_inputs.py --phase before-serialized
set +e
python -B check_scripts.py --snapshot "${PWD}/snapshot" --model /mnt/workspace/models/DeepSeek-V4.1-Flash --output-dir "${PWD}/results-serialized" --device 1
status=$?
python -B verify_inputs.py --phase after-serialized
preservation_status=$?
if [ "$preservation_status" -ne 0 ]; then exit "$preservation_status"; fi
exit "$status"
