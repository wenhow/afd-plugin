# A5 950DT 8×96 GiB：原生 DP8/TP1/EP8 eager 基线

更新：2026-11-10。用户确认 950DT、8 卡、每卡显示 98304 MB；
按 NPU 工具的 MiB 口径为每卡 96 GiB，实际可用显存以运行前采集为准。
**本轮改用 DP8/TP1/EP8 + Engram CPU offload/DP shared memory，
保留 eager、DSpark off 和小请求限制。最新回传已完成 8 rank 加载和服务启动，
`/health` 返回 200；首请求在原生 Attention 的 CompressorV2 动态库解析失败，返回 500。
不解除 AFD 模型 guard，完整原生推理与 AFD 对照仍未通过。**

## 当前下一步：检查 CompressorV2 算子库

回传 `b0a1f0ab6acf4ebc993ca3cc9342f698.gz` 已确认 `/dev/shm=512G`、
DP8/TP1/EP8、Engram CPU offload/shared memory 和默认关闭的 TP8 兼容补丁。
8 个 rank 均报告加载占用 44.29 GiB，KV 预算约 34.11–34.12 GiB/rank；
这是各阶段报告量，不是实测峰值。首请求时最早的 DP0 worker 在 `execute_dummy_batch`
进入 V4.1 Attention 压缩器后，`cann_ops_transformer` 无法解析
`aclnnCompressorV2` / `aclnnCompressorV2GetWorkspaceSize`。
因此 HTTP 500 是 worker 失败的后果，不是请求 JSON 校验错误，也不是此前的 2 GiB shm 限制。
详见 [本轮诊断](a5-20261110-native-ep8-request-001/diagnosis.json)。

Python wrapper 已导入不代表 ACLNN 动态库包含接口。当前仍需区分库缺失、版本缺少符号、
依赖库加载失败和搜索路径未生效，不据此盲目重装 Python 包或修改模型实现。
先在同一容器、同一 CANN shell 和 afd-plugin 目录执行只读检查：

```bash
git pull --ff-only origin feat/dsv41-flash-hccl-v030
bash tools/diagnostics/collect_dsv41_a5_opapi.sh
```

控制台仅打印 `results/dsv41-opapi.XXXXXX`，同时生成同名 `.tar.gz`，回传此压缩包。
工具保存指定 CANN 环境变量、包版本、CANN 库文件、绝对路径/裸库名的加载错误、
两个 CompressorV2 符号及实际已映射的 opapi 路径。
它不导入 torch、不使用 NPU、不加载 checkpoint、不执行算子、不安装包或改上游。
若已有库包含符号但裸库名加载失败，后续仅对确认的匹配目录修正环境；
若库不含接口，再按实际版本补齐匹配的 ops-transformer 二进制和 kernel 包。
检查成功仅表示采集完成；在环境问题定位前不重复加载整模型。

## 本轮配置与官方参考

参考 [vLLM Ascend v0.30.0 单机部署的 950DT 配置](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/DeepSeek-V4.1-Flash.html#51-single-node-colocated-deployment)。
这是保留官方并行布局和 Engram 放置的简化 smoke 配置，不是完整官方性能配方复现。

| 项目 | 本轮实际设置 |
| --- | --- |
| 并行与设备 | DP8 / TP1 / EP8，0–7 卡；PP/PCP/DCP=1 |
| 模型与量化 | 完整原 checkpoint，text-only，V2，`deepseek_v4_fp8`，safetensors |
| Engram | `cpu_offload=true`，`dp_shared_memory=true`，`embedding_across_dp=false`（官方默认） |
| 执行与负载 | eager；DSpark/DBO 关闭；每个 DP engine 的 `max_num_seqs=1`，长度/批 token 上限均 1024 |
| 其他 | block size 128，`deepseek_v41` tokenizer；async scheduling/prefix caching 关闭 |
| 原生优化 | 沿用官方 CPU binding、Engram/shared-expert/DSA overlap 和 `combine_quant_mode=4`；FlashComm1/DSA CP 关闭 |
| 插件 | 仅 Ascend 入口；默认不加载 `afd` 或 `afd_dsv41_mxfp8`，显式令量化兼容开关为 0 |

EP 分配完整专家，384 个专家由 8 个 rank 分担，每个 rank 48 个；
专家中间维度保持 2304，有 72 个 32-wide scale，SwiGLU 输入宽度为 4608。
此前 non-EP TP8 将单专家切成 288，出现 9 个 scale 和 576-wide SwiGLU 输入。
新布局避开这些特定的奇数 scale/128 对齐形状；这不保证其他算子或整模型已通过。
旧 DSV4 已使用相关 MXFP/GMM/SwiGLU 算子，不能把本轮异常归为首次引入这些算子。

官方文档验证的是 vLLM/vLLM-Ascend v0.30.0。项目仍锁定 vLLM `ced6857a…`、
Ascend RFC `7e37e29a…`，不自动改锁或切换上游；第四次回传实际 Ascend 为 `4ec6dc77…`，
包元数据仍显示旧 `2db8549c…`。保留此差异和本轮实际源文件 hash，不能直接套用官方版本的验收结论。

## 运行前准备

使用现有 `/usr/local/python3.11.10/bin/python` 与 `/home/models/DeepSeek-V4.1-Flash`，
不创建 venv，不修改 checkpoint、层数、专家数或 Engram 表，不重装上游依赖。
先确认 0–7 卡均空闲且属于本次作业；脚本记录 `npu-smi`，不会替用户清理其他作业。

Engram 从 HBM 移到主机 pinned/shared memory，需要足够的主机 RAM、`/dev/shm` 可用空间，
以及本地 DP worker 可共享的 IPC namespace。官方容器示例使用 `--shm-size=512g`；
已有容器若仍是很小的默认 `/dev/shm`，应先准备足够的共享内存再运行。
脚本将 `free -h`、`df -h /dev/shm` 保存到 `environment.txt`，不自行改容器配置。
此前 TP8 的 HBM 估算不适用于本轮；DP/EP 复制、host offload、KV、workspace 和加载峰值需重新实测。

## 更新并启动（全部日志写文件）

在 A5 的 afd-plugin 源码根目录更新已配置的分支；若有本地修改或分叉，保留现场，
不要用 reset/clean 覆盖。默认基线不要求安装或启用 TP8 兼容插件。

```bash
git pull --ff-only origin feat/dsv41-flash-hccl-v030
bash tools/diagnostics/run_dsv41_a5_native_eager.sh
```

需要覆盖路径时使用：

```bash
AFD_PYTHON=/usr/local/python3.11.10/bin/python AFD_MODEL=/home/models/DeepSeek-V4.1-Flash bash tools/diagnostics/run_dsv41_a5_native_eager.sh
```

控制台只打印 `results: .../results/dsv41-native-dp8-ep8.XXXXXX`。
服务前台运行，stdout/stderr 直接写 `native-server.log`，启动器写 `launcher.log`，不使用 `tee`。
脚本检查既定 config/index SHA256、48 个 shard 存在性并保存 `preflight.json`；
这不是全部权重 payload 校验。它同时记录模块路径、实际 Ascend 源文件 hash、包版本和运行前后设备状态。
默认不执行 K=288 的 TP8 探针，也不要求其插件入口存在。

日志中应确认 TP1、DP8、EP8、V2、真实 safetensors、`deepseek_v4_fp8`、
Engram `cpu_offload=True`/`dp_shared_memory=True`、eager、无 speculation。
`AFD_DSV41_MXFP8_SCALE_PATCH=0`，不应出现启用 DS MXFP 兼容修正的消息。
三项 Engram 配置继续采用逐字段 CLI，避免此前复制内联 JSON 时的布尔值截断。

将打印的绝对路径记为 `afd_native_run`。服务自然退出时记录 `server-exit-code.txt`，
启动器退出或 Ctrl+C 时记录 `launcher-exit-code.txt`；未生成文件按实际失败阶段说明。
无需 `cat`/`tail -f` 持续打印日志。失败时保留完整目录，先分析最早的 worker 根 traceback。

## 首次文本请求与退出

在第二个终端设置刚才的目录。以下 `/health` 失败会停止本次请求并将原因写入文件；
若模型尚在加载，等待后重试。全部请求输出也重定向，不在终端刷日志。

```bash
afd_native_run=/absolute/path/printed/above
(
  set -eu
  curl --silent --show-error --fail --max-time 10 http://127.0.0.1:18030/health \
    > "$afd_native_run/health.txt"
  cat > "$afd_native_run/request.json" <<'JSON'
{"model":"dsv41-native","prompt":"The capital of France is","temperature":0,"seed":0,"max_tokens":32,"logprobs":5,"return_token_ids":true,"stream":false}
JSON
  npu-smi info > "$afd_native_run/npu-ready.log" 2>&1
  curl --silent --show-error --fail-with-body --max-time 300 \
    -H 'Content-Type: application/json' \
    --data-binary "@$afd_native_run/request.json" \
    http://127.0.0.1:18030/v1/completions \
    > "$afd_native_run/response.json"
  npu-smi info > "$afd_native_run/npu-after-request.log" 2>&1
) > "$afd_native_run/request.log" 2>&1
afd_request_status=$?
printf '%s\n' "$afd_request_status" > "$afd_native_run/request-exit-code.txt"
```

保存响应后在第一个终端 Ctrl+C，等待 worker 退出和 `npu-after.log` 生成。
一次 HTTP 成功仅是首次推理 smoke，不代表连续/并发、数值精度、请求回收或性能验收。

回传该新目录，尤其是 `native-server.log`、`launcher.log`、`environment.txt`、
`preflight.json`、退出码文件、`request.json`/`response.json`/`request.log` 和设备状态日志。
可在服务停止后打包，不必复制终端 traceback：

```bash
tar -czf "$afd_native_run.tar.gz" -C "$(dirname "$afd_native_run")" "$(basename "$afd_native_run")"
```

本轮 DP8/EP8 + CPU Engram 与 AFD A4/F4 TP4 + NPU Engram 的并行和放置不同。
它先建立原生可用性基线，不构成固定 TP/Engram 放置的 AFD 等价或性能对照；
后续 AFD 验收须另建条件匹配的验证单元，不直接沿用本轮结果。
本地 [启动器检查记录](a5-20261110-native-ep8-baseline/launcher-validation.json)
仅覆盖 CPU stub 的传参、日志和退出码，以及语法/上游保护检查，不包含 NPU 执行。

## 可选：独立 TP8 兼容探针

以下仅用于继续诊断此前 TP8 路径，不是 DP8/EP8 基线的启动前置条件：

```bash
bash tools/diagnostics/run_dsv41_a5_native_eager.sh --probe-only
```

它单独建立 `results/dsv41-tp8-activation-probe.XXXXXX`，显式启用已有量化兼容入口，
执行 synthetic W13→SwiGLU/MX quant→W2，保存 `activation-probe.json`/`.log` 和退出码；
不会启动整模型。该模式仍要求此前安装的 `afd_dsv41_mxfp8` entry point，
必要时仅刷新 afd-plugin editable 安装，不安装或修改上游。

## 此前 TP8 容量候选与诊断历史

以下保留此前 Engram 留 NPU、EP 关闭的 TP8 证据和修复步骤，供回溯；
不是本轮 DP8/TP1/EP8 的操作顺序，也不要求重新执行旧探针。
完整服务及首请求一直未通过；12 项 MoE GMM 组件已回传通过，新增 SwiGLU 链路仍待设备执行。

### 历史容量参考

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

这是此前 Engram 留 NPU 的 TP8/A4F4 容量候选，不用于估算本轮 DP8/EP8 + CPU offload。
A4/F4 暂定 A 使用 0–3、F 使用 4–7，实际映射必须服从目标机器拓扑。
原生 TP8 与 AFD 两侧 TP4 改变了 native TP 大小：可作为明确记录此差异的输出对照，
不构成固定 TP 下的 AFD 等价或性能提升证据。若出现数值差异，要同时排查 TP 归约；
后续严格固定 TP 对照需另建容量可行、Engram 放置一致的验证单元。

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
当时的 TP8 验证要求退出码 0、`passed=true` 且 6 个 case 全部通过，才继续启动；
否则先回传 `mxfp8-linear-probe.json` 和 `.log`，不重复加载整模型。
第三次回传中该探针的 6 项均已通过；这不覆盖下面新发现的 MoE FP4 路径。

## 第三次启动：MXFP4 MoE scale 与下一步

回传包 `30a05b12216e4d89beed7b256fdd65b3.gz` 中，48/48 分片已读取，
首个 worker 于 20:08:09 在 routed MoE 的 post-load 退出：

```text
w4a8_mxfp4.py:608 process_weights_after_loading
RuntimeError: shape '[384, 5120, 4, 2]' is invalid for input of size 17694720
```

`17694720 = 384 × 5120 × 9`。W2 的逻辑 K=288，有 9 个 32-wide scale；
代码按 `9 // 2` 取整而尝试装入 8 列，导致 reshape 失败。
修正后的 scale 为 `(384, 5, 5120, 2)`，最后一对只补一个未使用的 byte。
这与 [官方 GMM MX scale 的 ceil(K/64) 配对布局](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md)
一致，目标 wheel 是否接受该 FP4/NZ 形状仍须单独执行探针。
完整根 traceback、环境及 hash 见 [第三次诊断](a5-20261109-native-startup-003/diagnosis.json)。

修复位于 afd-plugin 的 `mxfp4.py`，现有 `afd_dsv41_mxfp8` 入口和
`AFD_DSV41_MXFP8_SCALE_PATCH=1` 同时启用 Linear/MoE 两项修正。
它保留原生 FP4 有效权重、NZ 转换、逻辑维度与 TP/Engram；不修改 vLLM-Ascend。
W13/W2 和 EPLB 单 expert scale 使用相同打包，EPLB 的设备行为仍未验收。

### MoE 探针 20:55 回传：补齐物理 K

`ce3253b74327413faf3d2062c094305b.txt` 中前 6 项数值误差为 0，
覆盖 K=256 的 W13/W2 及 intermediate=288 的 W13。
第 7 项，即 W2 K=288、scale_alg=0，在算子 shape 校验阶段失败：

```text
161002 / EZ0009: The second dim of x must be 64 times
the second dim of perTokenScale.
```

9 个 scale 组成 5 对，但当前 kernel 要求实际 K=5×64=320。
此前仅补 scale 元数据，不能满足这个约束。新修复在原生 post-load 的 NZ 转换前
给 W2 的 packed FP4 K 轴补 16 个零 byte（32 个 FP4 零值），
并在生产使用的 `apply_gmm2` 中给 FP8 激活补 32 个零值。
有效量化字节保持不变，只把未使用的激活 scale byte 设为 E8M0 单位值，
避免未初始化的 NaN 元数据参与新增的零值 group。
checkpoint、W13 的有效输出宽度和逻辑 intermediate=288 均不变。

物理 W2/激活 K 变为 320，输入 scale 仍为 5 对。
每层 E=384、N=5120 的 W2 增加约 30 MiB，先前的 scale 补齐另增加约 1.875 MiB；
GMM2 还增加激活 padding 和 scale copy，不将其作为已验收的性能优化。
详见 [探针诊断](a5-20261109-mxfp4-probe-001/diagnosis.json) 和
[补丁审查](mxfp4-scale-patch-review.md)。

在 A5 回传中使用的 afd-plugin 源码目录，从当前开发分支更新：
已有相关本地改动或分叉时保留现场，不使用 reset/clean 强行覆盖。

```bash
git branch --show-current
git pull --ff-only origin feat/dsv41-flash-hccl-v030
git rev-parse HEAD
```

入口元数据没有变化，无需重新安装依赖。按前文新建结果目录并完成 checkpoint/设备检查后，
在同一终端先跑单卡 MoE 探针；已通过的 Linear 探针结果继续保留：

```bash
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling
export ASCEND_RT_VISIBLE_DEVICES=0
"$afd_python" -B tests/npu/dsv41_mxfp4_moe_probe.py --device 0 \
  --output "$afd_native_run/mxfp4-moe-probe.json" \
  > "$afd_native_run/mxfp4-moe-probe.log" 2>&1
afd_probe_status=$?
cat "$afd_native_run/mxfp4-moe-probe.json"
printf 'MoE probe exit=%s\n' "$afd_probe_status"
```

新版探针使用 2 个 synthetic experts，真实执行原生 DS post-load/NZ 转换及 FP4 GMM，
覆盖 W13/W2、intermediate=256/288/320 和两种 MX 动态量化算法，共 12 项。
W2 通过与生产相同的 `scheme.apply_gmm2` 调用，报告 logical/physical K、
实际 hook 源文件 hash；发生错误时也保留本次失败 case 的形状和调用入口。
每个 expert 分别检查全 K 与最后一个有效 group 的输出，对照 CPU 数值。
当时的 TP8 验证要求退出码 0、`passed=true` 且 12 项均通过才继续 TP8 启动；
否则回传 `mxfp4-moe-probe.json` 和 `.log`，先诊断组件异常。
探针不覆盖 routing、SwiGLU、TP 通信、Engram 或完整模型推理。

### 2026-11-10 回传：MoE 12 项通过，启动根异常待补

`61e977854ce049b8ad8e6454eb60bfd6.txt` 的 12 项均为 `passed=true`，
全 K 和最后一个有效 group 的最大绝对误差均为 0；W2 的逻辑 K=288、物理 K=320。
回传 `gmm2_sha256=b016ff5403061e2d045819c61a7d7bb0c518408e763469b86f256423015a982d`
与本地已交付补丁一致，相关 Ascend FP4 源文件也与锁定源文件同 hash。
这确认该单卡组件修正生效，仍不覆盖完整模型路由、SwiGLU、Engram 或 TP8。
结果见 [探针 JSON](a5-20261110-mxfp4-probe-002/probe.json) 和
[诊断记录](a5-20261110-mxfp4-probe-002/diagnosis.json)。

同次粘贴的启动片段在 09:13:40 报 TP6 worker 异常，调用已到达
`determine_available_memory → profile_run → _dummy_run → execute_model → V4.1 forward`。
片段结尾为未完成的 `/site-` 文件路径，缺少异常类型和消息，不能据此归因显存、算子或源码版本。
`determine_available_memory` 内执行了 profile 前向，出现这个函数名本身不能证明 OOM。

本地重新比对此前回传的 Ascend `2db8549c…` 与锁定 `7e37e29a…`：
当前片段涉及的 V4.1 `model.py`、`vl_model.py`、worker、V2 runner，以及 FP4 量化文件均逐字节相同。
这不证明所有文件或编译扩展一致，也不能确定本轮实际运行 HEAD；先补完整 traceback，
不把切换版本或重装包作为已确认的修复。

在 **afd-plugin 源码根目录** 执行
`"$afd_python" -m pip install --no-deps --no-build-isolation -e .`
只刷新该插件的 editable 安装及入口元数据，不会改变 Ascend 的 Git HEAD。
如果插件已 editable 安装且仅更新 Python 文件，通常无需重复安装；
editable 安装也不自动重编译上游 C++ 扩展，见 [pip 官方说明](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)。
探针直接安装补丁，不验证 CLI 的 general-plugin 入口，启动日志仍须确认下述启用消息。

后续补充的 09:15:11 片段来自 Ctrl+C 后的退出，包含父进程退出、SIGTERM 和
`Engine core initialization failed. See root cause above` 汇总，未补齐 09:13:40 的
worker 根异常。退出后的 shared-memory 清理警告也不能用于确定最初失败原因。
用户说明原 `server.log` 仅保存了 `bash: ger: command not found`；这条 shell 错误
没有包含模型异常。现改用下面的独立脚本直接重定向 stdout/stderr，避免复制多行命令
或 `tee` 导致控制台刷屏。保留本次旧日志，新运行使用独立目录。

### 第四次完整日志：融合 SwiGLU 输入宽度 576

`a9e556a7b78b4f9085a6ba01ca093ecb.gz` 已完整保存 worker 根异常。
8 个 rank 均报告完成权重加载，每 rank `Model loading took 61.84 GiB`；
这仅是加载阶段的报告量，不是 profile/HBM 峰值或可用 KV 空间。
10:25:10 首个 TP7 worker 报：

```text
apply_gmm1_act_quant -> torch.ops._C_ascend.npu_swiglu_group_quant
RuntimeError: In mx quant, the last dim of x should be divisible by 128, actual 576.
```

W13 输出为两半 `gate[288] + up[288]`，总宽 576。此前探针没有执行 SwiGLU，
其 12 项矩阵乘通过不能覆盖此约束。锁定 Ascend 的 `csrc/torch_binding.cpp:1843`
同样包含该校验，不能把这个具体错误仅归因于 HEAD 不同。
本轮 afd-plugin 为 `b7f86e1`，Ascend 源码 HEAD 实为 `4ec6dc77…`，
而 editable 包版本元数据仍显示 `…g2db8549c5`；本地没有该新 HEAD 的 Git 对象，
不声称其全部源码与锁定版本一致。下轮 preflight 增加相关实际源文件 hash。
本轮各 worker 已确认启用量化修正入口；checkpoint 配置/index 和 48 个分片存在性检查通过。
详见 [第四次诊断](a5-20261110-native-startup-004/diagnosis.json)。

新增修复复制锁定的原生 `apply_gmm1_act_quant`，仅在调用 SwiGLU 前
分别给 gate/up 补 32 个零值，形成 `gate[320] + up[320]`，总宽 640。
不能直接在整个 576-wide tensor 尾部追加零，否则会改变 gate/up 配对。
融合量化的输出宽 320 与之前已补齐的 W2 一致，保留有效元素、原生 clamp 和分组参数。
每个 token 多 64 个 BF16 零值；新增一次 padding/copy，无额外权重补齐。

`c279c4f` 版本的启动脚本先执行新增的 W13→融合 SwiGLU/MX 量化→W2 单卡探针，
覆盖 intermediate=256/288/320、clamp=0/6、cumulative/count 两种 group list，共 12 项。
与之前的矩阵乘探针分别记录，不将新探针的未执行状态视为通过。
探针使用两个 synthetic experts，检查不同 gate/up、最后一个有效 group、补零输出及 W2 数值；
不覆盖路由、EPLB/MegaMoE、TP/Engram 或完整 checkpoint。
当时只有它通过才继续加载 TP8 整模型。当前 DP8/TP1/EP8 默认不执行该探针；
仅显式 `--probe-only` 保留此诊断，全部日志继续直接写文件。
