# A5 950DT 8×96 GiB：原生完整模型控制组

更新：2026-11-10。用户确认 950DT、8 卡、每卡显存显示 98304 MB。
本页按 NPU 工具通常的 MiB 口径解释为每卡 96 GiB，总计 768 GiB；
实际可用显存仍以运行前 `npu-smi info` 为准。
此前的 MXFP base、原生 wheel norm、缩小构造和 TP2 组件已经通过。
**完整 checkpoint 控制组已回传三次启动：首次参数解析失败，第二次加载后在
MXFP8 Linear scale 打包失败；第三次 Linear 探针 6 项通过，但 routed MoE 的
MXFP4 scale 打包失败。后续 MoE 探针前 6 项通过，在 K=288 的 W2 遇到
WeightNz 的 K 对齐要求；补物理 K 对齐后的 12 项 MoE 探针已于 11-10 回传通过。
随后完整启动在 profile 的原生模型前向失败；第四次回传已确认融合 SwiGLU 的
576-wide 输入未满足 128 对齐。已补 gate/up 分半对齐，新增链路探针待 A5 执行。
服务和首请求尚未通过，模型支持和 AFD 总 guard 均未放开。**

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
只有退出码 0、`passed=true` 且 12 项均通过才继续下面 TP8 启动；
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

启动脚本现在先执行新增的 W13→融合 SwiGLU/MX 量化→W2 单卡探针，
覆盖 intermediate=256/288/320、clamp=0/6、cumulative/count 两种 group list，共 12 项。
与之前的矩阵乘探针分别记录，不将新探针的未执行状态视为通过。
探针使用两个 synthetic experts，检查不同 gate/up、最后一个有效 group、补零输出及 W2 数值；
不覆盖路由、EPLB/MegaMoE、TP/Engram 或完整 checkpoint。
默认只有它通过才继续加载整模型，全部日志继续直接写文件。

## 启动原生 TP8（显式量化修正）

以下使用标准 vLLM CLI，加载 Ascend 插件及独立的量化修正入口；不加载 `afd` 入口。
命令不绕过 AFD guard，不进行 AFD 推理。前台运行，日志中应确认 TP8、V2、
真实 safetensors、`deepseek_v4_fp8`、Engram `cpu_offload=False`、eager、无 speculation。
同时保留实际 router/MoE 选择、权重加载和内存日志；组件检查没有覆盖这些路径。

用户要求日志全部写入文件。使用
[`run_dsv41_a5_native_eager.sh`](../../../../tools/diagnostics/run_dsv41_a5_native_eager.sh)
保留上述 TP8 控制组配置：stdout/stderr 直接写入 `native-server.log`，不使用 `tee`。
控制台只打印新结果目录；`launcher.log` 记录启动器输出，`launcher-exit-code.txt`
记录正常/异常或 Ctrl+C 退出码，服务自然退出时另写 `server-exit-code.txt`。
新增探针的输出分别为 `activation-probe.json`、`activation-probe.log` 和
`activation-probe-exit-code.txt`；探针失败时停止，保留该轮结果，不加载整模型。
脚本重新记录 checkpoint header/index、48 个 shard 存在性、实际模块路径及插件入口，
并保存环境和运行前后 `npu-smi`。这不校验全部权重 payload 或设备空闲状态；
仍须先完成前文 checkpoint/空闲卡和探针检查，运行期间服务占用 0–7 卡。

在 afd-plugin 源码根目录执行；默认 Python 和模型路径与前文相同，可显式传入：

```bash
AFD_PYTHON="$afd_python" AFD_MODEL="$afd_model" bash tools/diagnostics/run_dsv41_a5_native_eager.sh
```

如只执行新增的单卡探针，在上述命令末尾追加 `--probe-only`。

将控制台打印的新路径赋给 `afd_native_run`，用于下面的请求和日志回传。
如果使用默认路径且没有前文变量，可直接运行
`bash tools/diagnostics/run_dsv41_a5_native_eager.sh`。
不要用 `cat`/`tail -f` 持续输出日志；发生失败后直接回传该目录的
`activation-probe.json`、`activation-probe.log`、`native-server.log`、`launcher.log`、
`environment.txt`、`preflight.json` 与退出码文件；未生成的文件按实际阶段说明。

2026-11-09 17:20 用户回传的启动日志停在 `--engram-config` 的
`Invalid JSON: expected ident at line 2 column 0`，尚未进入模型加载。
原内联 JSON 本身合法；实际传入的文本可能在复制时被插入了换行，截断布尔字面量。
上面已改为锁定 vLLM 支持的逐字段写法，三项会合成为布尔值 `false`；
重试时用这三行替换原来的整行 `--engram-config '...'`。
第二次 A5 回传已确认这三项均解析为 `False` 并进入模型加载。
量化修正后的完整启动和首请求仍待验证；日志应出现
`Enabled DS MXFP8 linear and MXFP4 MoE scale-pair corrections with MXFP4 GMM K alignment and fused SwiGLU alignment`，
确认独立补丁入口已加载。

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
`mxfp4-moe-probe.json`、`mxfp4-moe-probe.log`、
`checkpoint-identity.json`、`environment.txt`、`request.json`、`response.json`、
`server-exit-code.txt` 和 `npu-before/ready/after.log`；没有生成的文件按实际失败阶段说明。
启动失败时最关键的是 `native-server.log` 中最先出现的 worker 根 traceback，
优先回传完整日志文件，避免终端复制截断；完整日志继续保留本地。
原生通过并核对实际路由/FP4 和资源后，才进入 AFD A4/F4 开发验证。
