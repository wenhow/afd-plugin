# A5 950DT 8×96 GiB：原生完整模型控制组

更新：2026-11-09。用户确认 950DT、8 卡、每卡显存显示 98304 MB。
本页按 NPU 工具通常的 MiB 口径解释为每卡 96 GiB，总计 768 GiB；
实际可用显存仍以运行前 `npu-smi info` 为准。
此前的 MXFP base、原生 wheel norm、缩小构造和 TP2 组件已经通过。
**以下完整 checkpoint 控制组尚未执行；模型支持和 AFD 总 guard 均未因此放开。**

## 容量候选

按同一 checkpoint 的 header、native A5 Engram FP8/E8M0 存储和已测得的两层 RoPE buffer 计算：

| 路径 | 设备数 | native TP | 每 rank 容量参考 GiB |
| --- | --- | --- | ---: |
| 原生完整模型 | 8 | 8 | 78.790 |
| AFD Attention | 4 | 4 | 69.135 |
| AFD FFN | 4 | 4 | 68.006 |
| 原生完整模型，Engram 仍在 NPU | 4 | 4 | 137.141，超出单卡容量 |

A 和原生估算均包含每 rank 20 GiB 的 40 层 RoPE 候选；Engram 表按 24 个 prime head
精确分片，其他 A 参数仍仅按 checkpoint 编码量理想均分。F 另计复制的 gate 和 FP32 cache。
这些数值不含完整复制参数、格式转换、KV、workspace、HCCL、框架和加载临时峰值。
原生 TP8 的 `gpu_memory_utilization=0.90` 对应每卡 86.4 GiB 预算，
容量参考距该预算只余约 7.61 GiB，并非已测可用空间。
细项与全部假设见 [预算 JSON](a5-topology-budget-20261109.json)。

因此先执行原生 TP8 控制组；其加载、路由、专家计算和首次请求通过后，再验证 A4/F4。
A4/F4 暂定 A 使用 0–3、F 使用 4–7，实际映射必须服从目标机器拓扑。
原生 TP8 与 AFD 两侧 TP4 改变了 native TP 大小：可作为明确记录此差异的输出对照，
不构成固定 TP 下的 AFD 等价或性能提升证据。若出现数值差异，要同时排查 TP 归约；
后续严格固定 TP 对照需另建容量可行、Engram 放置一致的验证单元。

## 先确认 checkpoint 与空闲设备

使用当前已经通过组件检查的 Python/CANN/ATB 环境，无需创建 venv 或更换包。
本次将 Engram 明确留在 NPU；上游默认可能启用 CPU offload，不能省略对应配置。
不更改 checkpoint，不缩小层数/专家/Engram，不启用 DSpark、graph、EP、SP 或 DBO。

在更新后的 afd-plugin 源码根目录运行，保留输出的新目录路径：

```bash
afd_python=/usr/local/python3.11.10/bin/python
afd_model=/home/models/DeepSeek-V4.1-Flash
mkdir -p results
afd_native_run=$(mktemp -d "$PWD/results/dsv41-native-tp8.XXXXXX")
printf '%s\n' "$afd_native_run"
npu-smi info > "$afd_native_run/npu-before.log" 2>&1
"$afd_python" - "$afd_model" <<'PY' > "$afd_native_run/checkpoint-identity.json"
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {
    "config.json": "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879",
    "model.safetensors.index.json": "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8",
}
actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in expected}
index = json.loads((root / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (root / name).is_file()]
report = {"path": str(root), "sha256": actual, "expected_sha256": expected,
          "shard_count": len(shards), "missing_shards": missing,
          "matches": actual == expected and not missing}
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["matches"] else 1)
PY
```

先查看 `checkpoint-identity.json`，只有 `matches=true` 且 48 个 shard 均存在才继续。
这核对配置/index 和文件存在性，不是完整权重 payload 校验。
同时查看 `npu-before.log`：8 个目标逻辑设备均须可用且显存充足。
下面假定该容器的目标设备为 0–7；若实际可见映射不同，先调整映射并记录。

## 启动原生 TP8

以下使用标准 vLLM CLI，限定加载 Ascend 插件，不加载 afd-plugin。
命令不绕过 AFD guard，不进行 AFD 推理。前台运行，日志中应确认 TP8、V2、
真实 safetensors、`deepseek_v4_fp8`、Engram `cpu_offload=False`、eager、无 speculation。
同时保留实际 router/MoE 选择、权重加载和内存日志；组件检查没有覆盖这些路径。

在同一个终端继续：

```bash
(
  set -u
  set -o pipefail
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export VLLM_USE_V2_MODEL_RUNNER=1
  export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling
  export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
  {
    date -u
    "$afd_python" -V
    "$afd_python" -m pip show torch torch-npu vllm vllm-ascend fastapi
    git -C /opt/vllm-ascend rev-parse HEAD
    git -C /opt/vllm-ascend status --short
  } > "$afd_native_run/environment.txt" 2>&1
  "$afd_python" -B -m vllm.entrypoints.cli.main serve "$afd_model" \
    --host 127.0.0.1 --port 18030 --served-model-name dsv41-native \
    --tensor-parallel-size 8 --pipeline-parallel-size 1 --data-parallel-size 1 \
    --prefill-context-parallel-size 1 --decode-context-parallel-size 1 \
    --no-enable-expert-parallel --no-enable-dbo --ubatch-size 0 \
    --language-model-only --quantization deepseek_v4_fp8 --load-format safetensors \
    --engram-config '{"cpu_offload":false,"embedding_across_dp":false,"dp_shared_memory":false}' \
    --enforce-eager --max-model-len 1024 --max-num-seqs 1 --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.90 --no-async-scheduling --no-enable-prefix-caching \
    --generation-config vllm \
    --additional-config '{"enable_flashcomm1":false,"enable_dsa_cp":false}' \
    2>&1 | tee "$afd_native_run/native-server.log"
  afd_native_status=$?
  printf '%s\n' "$afd_native_status" > "$afd_native_run/server-exit-code.txt"
  npu-smi info > "$afd_native_run/npu-after.log" 2>&1
  exit "$afd_native_status"
)
```

如果加载、profile、路由算子或首请求失败，保留本轮目录，先诊断原生根异常；
不要转去启动 AFD，也不要混入更换 checkpoint、关闭 Engram 或改 TP 的重试。
如需改变配置，另开新目录并明确记录该验证单元的差异。
该启动命令只完成静态参数核对；A5 容器实际解析和运行结果仍以日志为准。

## 发出首次正确性请求并结束

在第二个终端，先把 `afd_native_run` 改成刚才打印的绝对路径。
等待第一个终端报告服务就绪后执行；若 `/health` 失败，先查看服务日志。
本轮只发一个请求，保留输出 token IDs 和 logprobs 供后续同参数对照。

```bash
afd_native_run=/absolute/path/printed/above
(
  set -eu
  curl --fail --max-time 10 http://127.0.0.1:18030/health \
    > "$afd_native_run/health.txt"
  cat > "$afd_native_run/request.json" <<'JSON'
{"model":"dsv41-native","prompt":"The capital of France is","temperature":0,"seed":0,"max_tokens":32,"logprobs":5,"return_token_ids":true,"stream":false}
JSON
  npu-smi info > "$afd_native_run/npu-ready.log" 2>&1
  curl --fail-with-body --max-time 300 \
    -H 'Content-Type: application/json' \
    --data-binary "@$afd_native_run/request.json" \
    http://127.0.0.1:18030/v1/completions \
    > "$afd_native_run/response.json"
  npu-smi info > "$afd_native_run/npu-after-request.log" 2>&1
)
```

保存响应并确认无报错后，在第一个终端 Ctrl+C 正常结束服务，等待子进程退出及
`npu-after.log` 生成。不要用全局进程名清理其他作业。
一次 HTTP 成功只构成首次推理 smoke，不能代替连续/并发、请求回收、数值精度或性能验收。

## 回传最小结果

优先回传 `checkpoint-identity.json`、`environment.txt`、`request.json`、`response.json`、
`server-exit-code.txt` 和 `npu-before/ready/after.log`；没有生成的文件按实际失败阶段说明。
启动失败时最关键的是 `native-server.log` 中最先出现的 worker 根 traceback，
可先回传日志尾部 300 行及相关完整异常。完整日志继续保留本地。
原生通过并核对实际路由/FP4 和资源后，才进入 AFD A4/F4 开发验证。
