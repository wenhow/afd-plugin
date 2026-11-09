# A5 / 目标 Linux 环境手工验证入口

该入口采集当前开发阶段的算子与组件证据，方便在没有远程登录通路的目标机器上运行。
**目前完整 V4.1 AFD 模型 guard 仍关闭。工具不会绕过 guard，也不会宣称完整模型已验收。**
它不安装依赖、不升级运行时、不改 checkpoint、不启动 API 服务，也不终止其他作业。

## 在 A5 上切换和更新 afd-plugin

### 当前分支与交付状态

本次适配使用 Fork `git@github.com:wenhow/afd-plugin.git`，开发分支为`feat/dsv41-flash-hccl-v030`，起点为 `release/v0.30.0rc2` 的`39017a137d7809bd0ab7cda4a86697e2a6b7a522`。
本轮代码与操作说明通过上述开发分支交付；更新时核对交付消息中的提交 SHA，
并确认本地 HEAD 与刚 fetch 的远端分支一致。
该起点 SHA 只标识基线，不能代表本轮适配代码；当前直接更新 release 分支无法获得这些改动。
### 切换已有 Git 仓库

先把示例路径改成 A5 上实际的 **afd-plugin Git 仓库**。
确认 `git remote get-url origin` 指向上面的 `wenhow/afd-plugin` Fork；如果原仓库的 origin 指向其他项目或需要保留其状态，使用下一节的新目录 clone。
以下命令在子 shell 内执行，任何检查失败就停止，不会继续执行后续步骤：

```bash
(
  set -eu
  cd /mnt/workspace/code/afd-plugin
  afd_branch=feat/dsv41-flash-hccl-v030
  git remote get-url origin
  git branch --show-current
  git rev-parse HEAD
  git status --short
  if [ -n "$(git status --porcelain)" ]; then
    echo 'Local changes exist; keep this directory and clone into a new one.' >&2
    exit 1
  fi

  # A previous single-branch clone may only track the release branch.
  if ! git config --get-all remote.origin.fetch |
      grep -Fxq "+refs/heads/${afd_branch}:refs/remotes/origin/${afd_branch}"; then
    git remote set-branches --add origin "$afd_branch"
  fi
  git fetch origin "refs/heads/${afd_branch}:refs/remotes/origin/${afd_branch}"
  if git show-ref --verify --quiet "refs/heads/${afd_branch}"; then
    git switch "$afd_branch"
  else
    git switch --track -c "$afd_branch" "origin/$afd_branch"
  fi
  git merge --ff-only "origin/$afd_branch"
  if [ "$(git rev-parse HEAD)" != "$(git rev-parse "refs/remotes/origin/$afd_branch")" ]; then
    echo 'Local HEAD differs from the fetched branch; preserve it and use a new clone.' >&2
    exit 1
  fi
  git log -1 --format='%H %s'
  git status --short
)
```

以后更新同一分支时，重新执行此段即可：先检查本地改动，再 fetch、switch 和`merge --ff-only`，最后核对本地 HEAD 与刚 fetch 的远端引用完全一致。
若存在本地修改、未跟踪文件、未推送提交或分叉，保留原目录和历史，改用新目录；不要把旧分支的改动直接恢复到本次适配分支。

### 使用新目录

没有现有仓库，或现有仓库需要保留本地工作时，选择一个不存在的目录：

```bash
(
  set -eu
  git clone --single-branch --branch feat/dsv41-flash-hccl-v030 \
    git@github.com:wenhow/afd-plugin.git \
    /mnt/workspace/code/afd-plugin-dsv41-a5
  cd /mnt/workspace/code/afd-plugin-dsv41-a5
  git branch --show-current
  git rev-parse HEAD
  git rev-parse origin/feat/dsv41-flash-hccl-v030
  git status --short
)
```

两个 SHA 应一致，工作区应为空；同时与该次交付提供的提交 SHA 核对。
完成后，执行 `cd /mnt/workspace/code/afd-plugin-dsv41-a5` 进入新仓库再运行验证。
A5 若未配置 GitHub SSH，可将仓库 URL 换为 `https://github.com/wenhow/afd-plugin.git`。

此前的 `dsv41-manual-validation-20261107.tar.gz` 是不含 `.git` 的历史源码快照。
不能在该解压目录执行 `git switch` 或 `git pull`；保留其验证结果，另建 Git 仓库获取本轮代码。
当前操作总入口为 [交付与 A5 操作说明](../../../../docs/npu/DSV41_V030_HCCL_WORKFLOW_ZH.md)。

### 确认实际使用更新后的插件源码

进入所选的源码根目录，使用后续验证命令指定的同一个目标 Python。
组件采集器会把该根目录置于每个子进程的 `PYTHONPATH` 前端；
手工检查导入位置时也显式指定：

```bash
afd_python=/usr/local/python3.11.10/bin/python
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$afd_python" -c \
  'import importlib.util, pathlib; p=pathlib.Path(importlib.util.find_spec("afd_plugin").origin).resolve(); print(p); assert p == pathlib.Path("afd_plugin/__init__.py").resolve()'
```

输出应指向当前选定目录的 `afd_plugin/__init__.py`。
Git 分支/源码切换不会更新正在运行的 Python 进程；完成切换后重新启动本次采集命令。
本组件入口直接使用源码，无需重装 vLLM、vLLM-Ascend 或 FastAPI。

## 准备

使用上节准备的 afd-plugin 源码根目录，至少保留 `afd_plugin/`、`.agents/dsv41/sources.lock.json` 和本 evidence 目录下的五个 Python 文件：
`portable_qualification.py`、`probe_norm_precision.py`、`probe_mxfp_quant.py`、`probe_attention_construction.py`、`probe_hccl_tp.py`。
本次 A5 使用新建容器，Python 路径为 `/usr/local/python3.11.10/bin/python`，
没有 venv，也没有 ATB 环境脚本。下方所有命令均用这个解释器启动，
并通过 `--python` 把同一路径传给采集子进程。
锁定 vLLM、Ascend 和本插件均声明支持 Python 3.11；五个采集/探针脚本已通过
Python 3.11 语法检查。该入口采集实际容器中的算子与组件结果，完整模型另行验收。

先查看该解释器内已安装的包，新建容器可能还未具备运行依赖：

```bash
afd_python=/usr/local/python3.11.10/bin/python
"$afd_python" - <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

print("python:", sys.executable, sys.version)
missing = []
for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "fastapi"):
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, "MISSING")
        missing.append(name)
raise SystemExit(bool(missing))
PY
```

出现 `MISSING` 时，先补齐该 Python 3.11 环境中的对应依赖，再运行组件采集；
不要直接复制之前 A3 的 Python 3.12 venv 或 cp312 扩展。
包版本可读取只代表安装元数据存在，实际导入、动态库和算子仍由后续检查验证。
本工具使用标准库组织验证。外层监督要求 Linux 提供 pidfd 接口，用于按进程身份清理本次探针的子孙进程；实际容器内核和权限也需支持此接口。

锁定源码为 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`。
目标环境若版本不同，仍会保存实际版本与 SHA，但 preflight 返回失败；
不能把版本漂移下的结果合并为本次锁定版本验收。
FastAPI 以容器实际部署版本为准，已有验证基准为 `0.123.10`，工具不会重新安装。

先用 `npu-smi info` 确认你选择的逻辑设备空闲，并在不会同时使用这些设备的时段运行。
工具只记录运行前后状态，不擅自抢占设备。若已设置 `ASCEND_RT_VISIBLE_DEVICES`，下方 `--device` / `--tp-devices` 都是该进程可见空间内的逻辑编号。

## A5 一次收集组件证据

在源码根目录运行；将 checkpoint 和输出路径改成目标容器实际值。
本容器使用 `--python "$afd_python"`，省略 `--venv` 和 `--atb-env`。
下方命令沿用当前 shell 的 CANN 和动态库环境。
若 CANN 尚未初始化，先 source 容器中实际存在的 `set_env.sh`，
也可以给命令添加 `--cann-env /实际路径/set_env.sh`。
没有 ATB 环境脚本并不能证明 ATB 动态库缺失；无需为此填写一个不存在的脚本。
若实际导入报告 `libatb.so` 等库无法加载，保留错误并核对容器内对应库及其搜索路径。

```bash
afd_python=/usr/local/python3.11.10/bin/python
"$afd_python" -B openspec/changes/afd-v41-hccl-text-eager/evidence/portable_qualification.py \
  --python "$afd_python" \
  --checkpoint /path/to/DeepSeek-V4.1-Flash \
  --checks preflight,norm-native,mxfp \
  --device 0 \
  --output-dir /path/to/results/dsv41-target-001
```

上面的 A5 命令显式选择 `preflight,norm-native,mxfp`。
省略 `--checks` 时仍沿用 `preflight,norm,mxfp` 的严格诊断默认值。
每个检查独立子进程、独立日志，失败后仍继续收集后续检查。单个检查外层上限 240 秒；可用 `--timeout 300` 调整到 240–600 秒。内部 norm / TP 探针还有各自较短监督。
超时只清理本次命令及可验证的子孙进程，不搜索或清理其他模型进程。
输出目录和同名 `.tar.gz` 必须都是新路径，防止覆盖上轮失败证据。

检查的含义：

| 检查 | 收集内容 | 不代表的范围 |
| --- | --- | --- |
| preflight | Python/包版本；16 个安装源码 SHA 与锁定 Git 对象核对；完整 checkpoint config 与 SHA；自动及显式 `deepseek_v4_fp8` 的实际 quant config 类 | 不加载权重，不构造模型，不测量 HBM 峰值 |
| norm | 严格要求目标 profile 支持 `RMS_NORM_CAST` 且编译扩展可用；记录 norm 双返回、BF16→FP32 重建差和受控 gate/top-k | 无完整 decoder forward、Engram、真实 checkpoint 路由精度 |
| norm-native | 按实际 `_DS41_WHEEL_OPS and uses_a5_packed_cache()` 选择原生 decoder norm 方法的分支；记录两个选择值、源码 SHA、norm 结果和独立 top-k 状态 | 绕过 decoder 构造，仅运行其未修改的 norm 方法；无完整 forward 或 checkpoint 权重 |
| mxfp | 与锁定 `methods/base.py` 一致的 `npu_dynamic_mx_quant(..., dst_type=float8_e4m3fn)`；执行后同步设备 | 无 FP4 专家 GEMM、完整 MoE 或模型精度 |

MXFP 默认只运行精确 base 调用，不运行历史调试用的 `reduce_align_type` 参数分支。
即使小算子返回成功，也需要后续真实量化层与完整模型验证。
preflight 保存 automatic / explicit quantization 两条路径；不能只凭 checkpoint 的`quant_method=fp8` 就认定已选择正确的 FP4 专家加载类。

## 可选构造与 TP2 组件

只有一张空闲设备时，可以加构造检查：

```bash
afd_python=/usr/local/python3.11.10/bin/python
"$afd_python" -B openspec/changes/afd-v41-hccl-text-eager/evidence/portable_qualification.py \
  --python "$afd_python" \
  --checkpoint /path/to/DeepSeek-V4.1-Flash \
  --checks preflight,norm-native,mxfp,construction \
  --device 0 --output-dir /path/to/results/dsv41-target-002
```

构造检查沿用已记录的 reduced config，分别构造 A decoder 与 A wrapper，保留参数 manifest、配置差异、ModelState/raw-token 能力和显存数据。
它不加载 checkpoint 权重，不进行 forward；缩小 Engram 表不代表真实 Engram 已验收。

有四个空闲逻辑设备时，可加入均匀 TP2 配对组件。`--tp-devices` 顺序固定为 `A0,A1,F0,F1`，请按目标机器拓扑选择，优先使用不同 physical NPU：

```bash
afd_python=/usr/local/python3.11.10/bin/python
"$afd_python" -B openspec/changes/afd-v41-hccl-text-eager/evidence/portable_qualification.py \
  --python "$afd_python" \
  --checkpoint /path/to/DeepSeek-V4.1-Flash \
  --checks preflight,norm-native,mxfp,construction,tp \
  --device 0 --tp-devices 0,2,4,6 \
  --output-dir /path/to/results/dsv41-target-003
```

TP 检查创建 A/F 各自 native world/TP=2，运行真实 HCCL 三路输入及返回、受控计算后的 native TP allreduce、STOP/ACK，并在 pair 关闭后核对 native TP 组仍存活。
工具生成一个小 launcher，仅把用户提供的 checkpoint 路径传给现有探针及其子进程；原探针 SHA 一并保存，不改生产文件或模型 guard。
**这仍然是受控计算组件，未运行 native MoE，不证明 TP8 或完整模型可用。**

锁定 A5 profile 未声明 `RMS_NORM_CAST`，严格 norm 会按原始能力检查返回失败。
这不说明 A5 原生 decoder 的 packed-cache norm 路径不可用；严格 norm 和显式 fallback 都没有运行该路径。`norm-native` 不以该 capability 为前置要求，保留实际原生分支选择：
wheel 与 packed-cache 条件均为真时运行 `npu_rms_norm` 分支，否则由原生方法选择 DeviceOperator 或 RMSNorm fallback。报告中的 `native_norm_selection` 区分这些分支；wheel 缺失时的成功不代表 wheel 分支已验证。
原生 top-k 的算子注册可用性与执行成功分别保存在 `native_router_topk_available` 和`native_router_topk_success`；未注册时后者为 `null`，表示尚未验证 top-k。
已注册但执行失败会令检查失败并保留原始异常。norm、gate 均使用合成权重，`norm-native` 成功不代表完整路由或完整模型验收。

若目标缺少此能力或编译扩展，可另开新目录采集显式 fallback：

```bash
afd_python=/usr/local/python3.11.10/bin/python
"$afd_python" -B openspec/changes/afd-v41-hccl-text-eager/evidence/portable_qualification.py \
  --python "$afd_python" --checkpoint /path/to/DeepSeek-V4.1-Flash \
  --checks preflight,norm-fallback --device 0 \
  --output-dir /path/to/results/dsv41-target-fallback-001
```

fallback 允许进入显式关闭 wheel 路径的 decoder RMSNorm 分支，记录该隔离分支与`npu_rms_norm` 的比较；它不是 A5 packed-cache decoder 路径的验收。
gate 使用合成 BF16 权重及其 `.float()` 独立副本，两条受控 FP32 projection也使用同一 BF16 权重展开后的值。A5 原生 gate adaptor 的 `16cast32` 路径忽略prepared FP32 activation，报告会分别保留原生调用与受控 projection 的差异，不能把两次原生 gate 结果相同当成 FP32 传输可省略的证明。
编译扩展可用时仍尝试 native top-k；任何算子失败原样保留，不自动换算子或降级。
fallback 成功不能替代融合 norm、native top-k 或完整模型验收；保留此前严格检查的失败包。

## 返回文件与退出码

命令结束会打印输出目录和同名 `.tar.gz` 路径。请完整保留该压缩包；
其中包括 `summary.json`、`preflight.json`、逐检查日志和 JSON、代表张量、探针 SHA、选择的环境脚本 SHA，以及 NPU 前后状态。
不会打包 checkpoint 权重、完整环境变量或凭据。

退出码 0 仅表示本次选择的组件检查都成功；退出码 1 表示至少一项失败、超时或环境准备异常，压缩包仍生成；参数错误由 argparse 返回 2。
输出 `full_model_qualification=false` 始终保留，不能据退出码 0 发布完整模型支持。

完整原生 / AFD 对照仍需后续开发与验收：相同 checkpoint/tokenizer、文本 eager、V2、DSpark off、保留 Engram/shared-KV；AFD 使用 HCCL、FFN-side gate、U1，关闭 graph/SP/EP/DP/PP/PCP/DCP，按实际权重/HBM 预算选择相同 A/F native TP。
需要进一步确认正确量化类、原生 FP4/GEMM、自定义算子、冷加载与峰值显存，再验证 prefill/decode、连续/并发、请求结束与资源回收。
当前不提供通过绕开 guard 启动的完整服务 recipe。

## 950DT 8 卡的下一步

2026-11-09 用户确认 950DT、8 卡、每卡显示 98304 MB。
按 96 GiB/卡估算，先做 [原生 TP8 完整 checkpoint 控制组](A5_NATIVE_EAGER_ZH.md)，
AFD 候选为 A4/F4，两侧各 native TP4；具体容量和未知项见
[A5 预算](a5-topology-budget-20261109.json)。
原生命令继续使用 `/usr/local/python3.11.10/bin/python`，不要求 venv 或 ATB 脚本。
这是独立的后续完整模型验证，组件采集器仍不会自动启动它或解除 AFD guard。
