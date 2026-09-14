# DeepSeek-V4 AFD 第一期 A5 验证指导书

## 1. 适用范围

本文只用于单台 8 卡 Ascend 950DT A5。执行前提是原指导书第 2.2 节安装和第 3 节
H0 审计已经完成。不要重复安装 Python、CANN、vLLM 或 vLLM-Ascend，也不要在本机
执行双 A3、多节点、A8F8、A8F4 或 A4F8 章节。

第一期只做 MTP-off 功能门禁，不生成 golden，不进行逐 token 比对。MTP N1/N2/N3
不在本次单 A5 范围，后续叠加 dSpark 时另行制定组合验证矩阵；最终精度测试在全部
功能开发结束后另行执行。本次固定项如下：

| 项目 | 固定值 |
|---|---|
| vLLM | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `3da28f9414583d2d0b672a8f06d1fae142404bda` |
| afd-plugin | 新包 `manifest/versions.env` 中的 `AFD_TARGET_COMMIT`/`AFD_TARGET_TREE` |
| CANN | 只使用 `config.env` 指定的绝对路径；`EXPECTED_CANN_VERSION` 保持为空 |
| 模型 | `/home/models/DeepSeek-V4-Flash-MXFP8` 的官方原始 `DeepSeek-V4-Flash` 配置 |
| 硬件 | 单机 8 卡，device ordinal 为 0-7 |
| AFD | `P2pHcclAFDConnector`、TP1、`ENABLE_MTP=0` |

## 2. 本次要执行的项目

按下列顺序执行。前四项是功能门禁，最后一项是容量项：

| 顺序 | 项目 | NPU | 预期结论 |
|---|---|---|---|
| 1 | 官方 no-AFD、DP4、Graph、MTP-off | 0-3 | 模型加载、health、models 和请求成功 |
| 2 | A4F4 eager/U1/MTP-off | A: 0-3，F: 4-7 | AFD 基础门禁 |
| 3 | A4F4 Graph/U2/MTP-off | A: 0-3，F: 4-7 | Graph 和真实 U2 门禁 |
| 4 | A2F4 Graph/U2/MTP-off | A: 0-1，F: 2-5 | `F=kA` 门禁 |
| 5 | A4F2 Graph/U2/MTP-off | A: 0-3，F: 4-5 | `A=kF` 容量项 |

每个 AFD 点执行两次独立冷启动；每轮检查 ready、batch 1/8/32、取消请求后恢复、
启动与请求 fatal、U2 实际执行、停服和 NPU 清理。所有请求只检查 HTTP 和输出结构，
结果必须记录 `golden_checked=false`。

A4F2 若成功加载并通过全部功能门禁，记录为通过；若 FFN EP2 在模型加载阶段 OOM，
保留日志和 `npu-smi` 作为“容量阻塞”。不得降低模型、改权重或超卖来伪造通过。

## 3. 用新包升级验证脚本

将新 `slim-a5-reuse` 包复制到 A5。进入新包目录后，复用上一次已经验证过的
`config.env`。下面的 `OLD_BUNDLE_ROOT` 只需替换为上一次 A5 包的实际目录：

```bash
sha256sum -c dsv4-afd-hccl-manual-install-slim-a5-reuse-*.tar.gz.sha256
tar -xzf dsv4-afd-hccl-manual-install-slim-a5-reuse-*.tar.gz
cd dsv4-afd-hccl-manual-install-slim-a5-reuse-*
export BUNDLE_ROOT="$PWD"
export OLD_BUNDLE_ROOT="/替换为上一次A5包目录"
cp "$OLD_BUNDLE_ROOT/config.env" "$BUNDLE_ROOT/config.env"
sed -i \
  -e 's/^ENABLE_MTP=.*/ENABLE_MTP="0"/' \
  -e 's/^MTP_NUM_SPECULATIVE_TOKENS=.*/MTP_NUM_SPECULATIVE_TOKENS="1"/' \
  -e 's/^MTP_DRAFT_EXECUTION=.*/MTP_DRAFT_EXECUTION="eager"/' \
  "$BUNDLE_ROOT/config.env"
bash bin/00_print_config.sh
bash bin/install_all.sh
```

`a5-reuse` 不重装 env、不安装 Python 依赖、不重建两个上游仓库，只升级干净的独立
afd-plugin 目标目录。安装器只有在当前目标 tree 能匹配固定提交链时才会前进；有本地
改动或来源不明时会停止，不会 reset 或覆盖。

本次不需要再次执行 `install_a5_model_config.sh`。只有
`bin/00_print_config.sh` 或后续预检报告模型配置不符时，才停止并先核对上次安装产物，
不要直接覆盖模型配置。确认输出仍满足：

```text
CANN_ROOT=/usr/local/Ascend/cann-9.2.0       # 以现场实际路径为准
EXPECTED_CANN_VERSION=                       # 必须为空
MODEL_PATH=/home/models/DeepSeek-V4-Flash-MXFP8
SOC_VERSION=Ascend950DT_9582
ATTENTION_DEVICES=0,1,2,3
FFN_DEVICES=4,5,6,7
REUSE_VENV=1
INSTALL_PYTHON_DEPS=0
INSTALL_UPSTREAM_STACK=0
```

升级后重新加载运行环境。后续命令都在同一个 shell 中执行：

```bash
source "$BUNDLE_ROOT/bin/activate_runtime.sh"
cd "$AFD_PLUGIN_ROOT"
git status --short
bash tools/dsv4/run_phase1_a5_matrix.sh list-smoke
```

`git status --short` 必须为空；`list-smoke` 必须输出 4 个 MTP-off AFD case。

## 4. 运行前预检

停止其他 NPU 服务后执行：

```bash
source "$BUNDLE_ROOT/bin/activate_runtime.sh"
cd "$AFD_PLUGIN_ROOT"
bash tools/dsv4/run_phase1_a5_native_smoke.sh preflight
bash tools/dsv4/run_phase1_a5_matrix.sh preflight-smoke \
  a4f4_eager_u1_mtp_off \
  a4f4_graph_u2_mtp_off \
  a2f4_graph_u2_mtp_off \
  a4f2_graph_u2_mtp_off
```

预检会核对两个固定上游提交、三个源码导入根、干净工作树、模型路径、custom ops、唯一 CANN 路径和空闲 NPU。它不依赖 `ss`，也不校验 A5 的 CANN 版本字符串。

任一预检失败都不要继续启动模型。保存完整终端输出并回传。

## 5. 运行官方 no-AFD DP4 smoke

先建立本次统一输出根：

```bash
export A5_VALIDATION_ROOT="/data/validation/dsv4-phase1-a5-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$A5_VALIDATION_ROOT"
export PHASE1_NATIVE_OUTPUT_ROOT="$A5_VALIDATION_ROOT/native-dp4"
bash tools/dsv4/run_phase1_a5_native_smoke.sh run \
  2>&1 | tee "$A5_VALIDATION_ROOT/native-dp4.console.log"
```

脚本使用 NPU 0-3、DP4/TP1，按官方风格启动 no-AFD Graph/MTP-off 服务一次。它检查`/health`、`/v1/models` 和一个 completion，然后停服并检查 NPU 清理。此前 MTP N1 的成功结果可以保留为附加证据，但不能替代本次 MTP-off 基线。通过时应出现：

```text
[phase1-a5-native] completed: .../native-dp4
```

关键结果位于：

```text
native-dp4/runtime.env
native-dp4/models.json
native-dp4/functional_smoke.json
native-dp4/summary.env
native-dp4/server.log
native-dp4/npu-before.txt
native-dp4/npu-ready.txt
native-dp4/npu-after-stop.txt
```

## 6. 运行 3 个必须通过的 AFD 点

```bash
export PHASE1_OUTPUT_BASE="$A5_VALIDATION_ROOT/afd-required"
bash tools/dsv4/run_phase1_a5_matrix.sh smoke \
  a4f4_eager_u1_mtp_off \
  a4f4_graph_u2_mtp_off \
  a2f4_graph_u2_mtp_off \
  2>&1 | tee "$A5_VALIDATION_ROOT/afd-required.console.log"
```

矩阵脚本会强制清除外层 `config.env` 的 MTP 默认值，三个 case 均不传
`--enable-mtp`。这样旧配置中的 `ENABLE_MTP=1` 或 `MTP_DRAFT_EXECUTION=graph`
也不能把当前 A5 门禁改成 MTP 路径。

矩阵固定 `VLLM_SHUTDOWN_TIMEOUT_SECONDS=20`。`0` 在当前 vLLM 中表示立即 abort；
但 20 秒本身不能修复角色串行停机：若脚本等待 Attention 完全退出后才通知 FFN，
Attention 超时强杀 peer 时，仍在 `torch.npu.synchronize()` 的 FFN 会报 `507035`。
新脚本保持 Attention-first 协议顺序，连续向 Attention 和 FFN 的顶层进程发送优雅停机
请求，再分别等待；超时后才清理整个进程组。connector 释放同时改为幂等，首个真实
设备错误仍会上报，但不会在重复 close 时继续产生 `Invalid process group specified`。

取消请求改为流式请求。`curl=28` 后脚本立即从 `/metrics` 检查
`vllm:num_requests_running` 和 `vllm:num_requests_waiting`，两项连续两次为 0 才执行恢复
请求；恢复请求完成后再执行一次相同检查，然后才开始停服。这能区分“客户端已超时”
与“服务端请求确实已取消并归零”。`507035` 仍保留为 fatal，不做日志白名单。

三项必须全部返回 0。每个 case 目录必须包含 `cycle_1`、`cycle_2` 和
`validation_summary.json`；每轮必须包含：

```text
functional_smoke.json
cancellation.exitcode
cancellation_gate.json
cancellation_quiescence.metrics
cancellation_quiescence_gate.json
recovery.json
request_quiescence.metrics
request_quiescence_gate.json
cycle_summary.json
attention.log
ffn.log
npu_ready.txt
npu_after_cleanup.txt
```

`cancellation.exitcode` 的预期值为 28，两个 quiescence gate 必须分别为
`passed=true`、`running=0`、`waiting=0`、`stable_samples=2`，随后 `recovery.json`
必须通过。`cycle_summary.json` 中 `shutdown.coordinated` 必须为 `true`，`order` 必须为
`attention_request, ffn_request, attention_wait, ffn_wait`。Graph/U2 case 的
`ubatch_gate.observed_two_stages` 必须为 `true`。

失败证据目录不会被覆盖。修复后续跑第 6 节时应保留原来的
`A5_VALIDATION_ROOT`，只换一个新的输出目录，例如：

```bash
export PHASE1_OUTPUT_BASE="$A5_VALIDATION_ROOT/afd-required-r2"
bash tools/dsv4/run_phase1_a5_matrix.sh smoke \
  a4f4_eager_u1_mtp_off \
  a4f4_graph_u2_mtp_off \
  a2f4_graph_u2_mtp_off \
  2>&1 | tee "$A5_VALIDATION_ROOT/afd-required-r2.console.log"
```

第 5 节已经通过时不需要重跑。收集证据时将第 9 节的
`afd-required/smoke` 替换为实际成功目录，例如 `afd-required-r2/smoke`。

若旧包仅在 `a4f4_eager_u1_mtp_off` 的退出阶段出现下列组合：业务 smoke、取消恢复、
进程返回码和 NPU 清理均通过，但 Attention 日志在等待 20 秒后出现
`force killing remaining processes`，随后 FFN 报 `507035`，升级后先只重跑该点：

```bash
export PHASE1_OUTPUT_BASE="$A5_VALIDATION_ROOT/afd-eager-r2"
bash tools/dsv4/run_phase1_a5_matrix.sh smoke \
  a4f4_eager_u1_mtp_off \
  2>&1 | tee "$A5_VALIDATION_ROOT/afd-eager-r2.console.log"
```

两轮通过后，再用新的输出目录执行本节另外两个 Graph/U2 必过点。第 5 节和已通过的
安装/预检不需要重跑，旧失败目录不得删除。若新脚本在任一 quiescence gate 失败，先
回传两个 gate JSON、metrics 和 Attention 日志；这表示请求生命周期未归零。若两个
quiescence gate 均通过后仍出现 `507035`，回传 Attention/FFN 完整日志及 A5 设备侧
plog/slog，不能以第二轮偶然通过覆盖第一轮失败。

## 7. 单独运行 A4F2 容量项

```bash
export PHASE1_OUTPUT_BASE="$A5_VALIDATION_ROOT/afd-capacity"
set +e
bash tools/dsv4/run_phase1_a5_matrix.sh smoke a4f2_graph_u2_mtp_off \
  2>&1 | tee "$A5_VALIDATION_ROOT/afd-capacity.console.log"
export A4F2_EXITCODE=${PIPESTATUS[0]}
set -e
printf 'A4F2_EXITCODE=%s\n' "$A4F2_EXITCODE" \
  | tee "$A5_VALIDATION_ROOT/a4f2-result.env"
```

判定方法：

- `A4F2_EXITCODE=0` 且两轮 summary 均通过：容量项通过。
- 模型加载阶段出现明确 HBM/OOM，且日志和 NPU 快照完整：记录为容量阻塞。
- HCCL、Graph、请求、U2、取消恢复、fatal 或清理失败：这是功能失败，不能记为容量阻塞。

脚本即使失败也会保留已经创建的输出目录、case 退出码、日志和最后一次 `npu-smi`。

## 8. 快速检查结果

用已安装 venv 解析所有 summary，不要求系统安装 `jq`：

```bash
"$DSV4_RUNTIME_VENV/bin/python" - "$A5_VALIDATION_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in sorted(root.rglob("validation_summary.json")):
    data = json.loads(path.read_text())
    print(path)
    print("  passed=", data.get("passed"))
    print("  validation_mode=", data.get("validation_mode"))
    print("  golden_checked=", data.get("golden_checked"))
    for cycle in data.get("cycles", []):
        print(
            "  cycle=", cycle.get("cycle"),
            "passed=", cycle.get("passed"),
            "cancel=", cycle.get("cancellation_gate", {}).get("passed"),
            "cancel_idle=", cycle.get("cancellation_quiescence_gate", {}).get("passed"),
            "recovery_idle=", cycle.get("request_quiescence_gate", {}).get("passed"),
            "coordinated_shutdown=", cycle.get("shutdown", {}).get("coordinated"),
            "u2=", cycle.get("ubatch_gate", {}).get("observed_two_stages"),
            "cleanup=", cycle.get("npu_cleanup_gate", {}).get("passed"),
        )
PY
```

原生 `runtime.env` 必须为 `enable_mtp=0`，`summary.env` 必须为 `passed=1`、
`forced_stop=0`、`npu_cleanup_passed=1`。AFD summary 必须为
`validation_mode=functional_smoke`、`golden_checked=false`、`enable_mtp=false` 和
`mtp_draft_execution=null`。

## 9. 收集并回传证据

```bash
source "$BUNDLE_ROOT/bin/activate_runtime.sh"
cd "$AFD_PLUGIN_ROOT"
bash tools/dsv4/collect_phase1_validation.sh \
  "$A5_VALIDATION_ROOT/dsv4-phase1-a5-evidence.tar.gz" \
  "$A5_VALIDATION_ROOT/native-dp4" \
  "$A5_VALIDATION_ROOT/afd-required/smoke" \
  "$A5_VALIDATION_ROOT/afd-capacity/smoke"
sha256sum -c "$A5_VALIDATION_ROOT/dsv4-phase1-a5-evidence.tar.gz.sha256"
```

回传以下文件：

```text
dsv4-phase1-a5-evidence.tar.gz
dsv4-phase1-a5-evidence.tar.gz.sha256
native-dp4.console.log
afd-required.console.log
afd-capacity.console.log
a4f2-result.env
```

若某一步在创建预期目录前失败，先回传 console log，不要为了让收集器运行而创建伪造
summary。证据包会包含提交、CANN 路径、环境、Python 包、NPU 快照、结构化结果和截断
日志；不会打包 profiler raw。

## 10. 第一期完成条件

满足以下条件后，A5 第一期功能验证才可关闭：

1. 官方 no-AFD DP4 模型加载、health、models、请求和 NPU 清理通过。
2. 3 个必须 AFD 点两轮全部通过，batch 1/8/32、取消恢复和 fatal 门禁通过。
3. 2 个必过 Graph/U2 点都观测到真实 two-stage，而不是只配置了 `U_BATCHES=2`。
4. A4F2 得到“通过”或有完整 HBM 证据的“容量阻塞”结论。
5. 所有功能报告均明确 `golden_checked=false`，没有逐 token 精度声明。

最终精度、路径匹配 control、30/30 token exact、idle-resume、U3、正式性能和 12 卡
A8F4 均不在本次 A5 第一期执行范围。MTP 只在后续 dSpark 组合阶段重新纳入，不能用
本次 MTP-off 结果声明 dSpark + MTP 已通过。
