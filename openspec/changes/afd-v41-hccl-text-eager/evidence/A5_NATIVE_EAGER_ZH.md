# A5 950DT 8×96 GiB：原生完整模型控制组

更新：2026-11-09。用户确认 950DT、8 卡、每卡显存显示 98304 MB。
本页按 NPU 工具通常的 MiB 口径解释为每卡 96 GiB，总计 768 GiB；
实际可用显存仍以运行前 `npu-smi info` 为准。
此前的 MXFP base、原生 wheel norm、缩小构造和 TP2 组件已经通过。
**完整 checkpoint 控制组已执行两次启动：首次参数解析失败，第二次加载后在
MXFP8 scale 打包失败；服务和首请求尚未通过。模型支持和 AFD 总 guard 均未放开。**

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

## 第二次启动错误与修复前检查

回传包 `238c6abfd7a845eb9ed400ea68e7d275.gz` 中，Engram 三项参数已经正确解析，
48 个权重分片均存在；第二次启动退出码为 1。首个 worker 根异常为：

```text
w8a8_mxfp8.py:661 process_weights_after_loading
RuntimeError: shape '[5120, 4, 2]' is invalid for input of size 46080
```

当前 DS 专用 MXFP8 Linear 将 scale 直接两两打包，遗漏了奇数列的尾部补齐。
由 checkpoint 和 TP8 的形状推导，对应共享专家 `down_proj` 的 K=2304/8=288、
N=5120，展开后有 9 列 scale；需保留 9 列有效数据，并在最后一个 pair 的空位补零，
形成 `(5, 5120, 2)`。层名是形状与源码推导，原 traceback 没有直接打印层名。
权重和输入的逻辑 K 仍为 288，不改 checkpoint、TP 或 Engram 放置。
这与上游通用 MXFP8 Linear 已有的 odd-column 处理及
[官方 MX scale 布局](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_quant_matmul.md)
一致；目标 wheel 的实际算子仍需下面的小探针确认。

回传环境的 Ascend HEAD 为 `2db8549c548731611baeb169fa1ef13aa55fc6aa`，
不是项目锁定的 `7e37e29a7803d6897a7113088e2b771f83205bc0`；
这两个提交的相关 DS post-load 函数相同，单独切换版本不能解决此次 reshape。
保留该差异供验收追踪，不修改 `/opt/vllm-ascend`。

先在 **afd-plugin 源码目录** 更新分支，再刷新本插件的 editable 安装元数据。
以下假定 `origin` 已指向此前配置的 `wenhow/afd-plugin`，先核对输出；
有本地改动或分叉导致 Git 拒绝时，保留现场，不使用 reset/clean 强行更新。

```bash
git remote -v
git fetch origin feat/dsv41-flash-hccl-v030
git switch feat/dsv41-flash-hccl-v030
git pull --ff-only origin feat/dsv41-flash-hccl-v030
git rev-parse HEAD
"$afd_python" -m pip install --no-deps --no-build-isolation -e .
"$afd_python" - <<'PY'
from importlib.metadata import entry_points
plugins = {entry.name: entry.value for entry in entry_points(group="vllm.general_plugins")}
assert plugins["afd_dsv41_mxfp8"] == "afd_plugin:register_dsv41_mxfp8_scale_patch"
print(plugins["afd_dsv41_mxfp8"])
PY
```

只刷新 afd-plugin；沿用容器中的 FastAPI 0.123.10 及其他依赖。
新补丁默认关闭，由 `AFD_DSV41_MXFP8_SCALE_PATCH=1` 显式启用；
本页用 `VLLM_PLUGINS` 列表限定独立入口，保留该列表，避免默认发现其他插件。
它只修正量化 scale 打包，不注册 AFD 模型或 worker，不绕过 AFD guard。此次控制组由纯原生变为
**原生模型 + 单项量化兼容补丁**，后续输出对照须记录这一差异。

在上述 checkpoint 检查通过后的同一终端，先执行单卡探针：

```bash
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling
export ASCEND_RT_VISIBLE_DEVICES=0
"$afd_python" -B tests/npu/dsv41_mxfp8_linear_probe.py --device 0 \
  --output "$afd_native_run/mxfp8-linear-probe.json" \
  > "$afd_native_run/mxfp8-linear-probe.log" 2>&1
afd_probe_status=$?
cat "$afd_native_run/mxfp8-linear-probe.json"
printf 'probe exit=%s\n' "$afd_probe_status"
```

该探针真实执行 MX 动态量化与矩阵乘，覆盖 K=256/288/320、N=5120、
两种 scale 算法，并单独激活最后一个有效 group 对照 CPU 数值。
它不加载模型、checkpoint 或 Engram，不创建通信组。
只有退出码 0、`passed=true` 且 6 个 case 全部通过，才继续启动；
否则先回传 `mxfp8-linear-probe.json` 和 `.log`，不重复加载整模型。
这一步尚待用户在 A5 执行，CPU 打包测试不能代替该算子验证。

## 启动原生 TP8（显式量化修正）

以下使用标准 vLLM CLI，加载 Ascend 插件及独立的量化修正入口；不加载 `afd` 入口。
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
  export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling,afd_dsv41_mxfp8
  export AFD_DSV41_MXFP8_SCALE_PATCH=1
  export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
  {
    date -u
    "$afd_python" -V
    "$afd_python" -m pip show torch torch-npu vllm vllm-ascend fastapi
    git rev-parse HEAD
    git status --short
    env | sort | sed -n '/^AFD_DSV41_MXFP8_SCALE_PATCH=/p; /^VLLM_PLUGINS=/p'
    git -C /opt/vllm-ascend rev-parse HEAD
    git -C /opt/vllm-ascend status --short
  } > "$afd_native_run/environment.txt" 2>&1
  "$afd_python" -B -m vllm.entrypoints.cli.main serve "$afd_model" \
    --host 127.0.0.1 --port 18030 --served-model-name dsv41-native \
    --tensor-parallel-size 8 --pipeline-parallel-size 1 --data-parallel-size 1 \
    --prefill-context-parallel-size 1 --decode-context-parallel-size 1 \
    --no-enable-expert-parallel --no-enable-dbo --ubatch-size 0 \
    --language-model-only --quantization deepseek_v4_fp8 --load-format safetensors \
    --engram-config.cpu_offload false \
    --engram-config.embedding_across_dp false \
    --engram-config.dp_shared_memory false \
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

2026-11-09 17:20 用户回传的启动日志停在 `--engram-config` 的
`Invalid JSON: expected ident at line 2 column 0`，尚未进入模型加载。
原内联 JSON 本身合法；实际传入的文本可能在复制时被插入了换行，截断布尔字面量。
上面已改为锁定 vLLM 支持的逐字段写法，三项会合成为布尔值 `false`；
重试时用这三行替换原来的整行 `--engram-config '...'`。
第二次 A5 回传已确认这三项均解析为 `False` 并进入模型加载。
量化修正后的完整启动和首请求仍待验证；日志应出现
`Enabled DS MXFP8 odd scale-pair packing correction`，确认独立补丁入口已加载。

如果加载、profile、路由算子或首请求失败，保留本轮目录，先诊断原生根异常；
不要转去启动 AFD，也不要混入更换 checkpoint、关闭 Engram 或改 TP 的重试。
如需改变配置，另开新目录并明确记录该验证单元的差异。
更新后的启动命令和补丁仍以 A5 实际日志及首请求结果为准。

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

优先回传 `mxfp8-linear-probe.json`、`mxfp8-linear-probe.log`、
`checkpoint-identity.json`、`environment.txt`、`request.json`、`response.json`、
`server-exit-code.txt` 和 `npu-before/ready/after.log`；没有生成的文件按实际失败阶段说明。
启动失败时最关键的是 `native-server.log` 中最先出现的 worker 根 traceback，
可先回传日志尾部 300 行及相关完整异常。完整日志继续保留本地。
原生通过并核对实际路由/FP4 和资源后，才进入 AFD A4/F4 开发验证。
