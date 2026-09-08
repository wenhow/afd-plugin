# DeepSeek-V4 AFD 第一期 A5 与双机验证指导书

## 1. 目标与固定口径

本文只关闭第一阶段功能门禁，不用于发布 U3 或性能结论。固定栈如下：

| 组件 | 固定值 |
|---|---|
| CANN | `9.0.0`，整个 shell 只能加载一个绝对 toolkit root |
| vLLM | `releases/v0.23.0`，`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend | `rfc/vllm_cann`，`3da28f9414583d2d0b672a8f06d1fae142404bda` |
| afd-plugin | 补丁包 `manifest/versions.env` 中的 `AFD_TARGET_COMMIT`/`AFD_TARGET_TREE` |
| 模型 | 同一份 DeepSeek-V4-Flash W8A8，`num_nextn_predict_layers=1` |
| 并行 | TP1；A/F 只验收 `A=kF`、`A=F`、`F=kA` 的整数比例 |
| MTP | `num_speculative_tokens=1/2/3`，最大值 3 |

第一阶段仍需完成的外部硬门禁：

1. A5 平台审计、独立安装和 5 份同栈、路径匹配 native control。
2. A5 standalone A8F8、A4F8、A8F4 的 eager/Graph、U1/U2、MTP N=1/2/3 代表矩阵。
3. A8F4 在高 HBM A5 上的真实模型加载和端到端请求。
4. 双机 PD Graph/U2 下的 A8F8 N2/N3、A4F8 N3、A8F4 N3。
5. 每个 PD 点的真实双 stage、FFN Graph 动态路由、取消后恢复、优雅退出、NPU 清理和第二次冷启动。
6. 路径匹配的 PD no-AFD control，以及每次冷启动 10 条 prompt x 3 轮的 30/30 token exact。

不属于第一阶段：U3、正式 P2 性能收益、A5 参数调优、非整数 A/F、TP/SP/CP/DCP/PP、TP3、非等量 TP2 和 TP2 full-draft Graph/U2/MTP 最大组合。

## 2. 补丁包安装

已经按 `/mnt/workspace/delivery/config.env.example` 安装过的两台 A3 使用文件名包含
`slim-dual-a3-reuse` 的专用包。它已固定现有安装目录、模型、CANN 9.0.0、NIC 和旧
AFD 种子提交；复用现有 venv 和两个上游 editable 安装，只把新版 afd-plugin 安装到
独立的 `/data/z00569729/code/afd-plugin-phase1-a5`。原
`/data/z00569729/code/afd-plugin` 不会被修改。

在两台 A3 使用同一个 tar 包和 SHA256 文件：

```bash
sha256sum -c dsv4-afd-hccl-manual-install-slim-dual-a3-reuse-*.tar.gz.sha256
tar -xzf dsv4-afd-hccl-manual-install-slim-dual-a3-reuse-*.tar.gz
cd dsv4-afd-hccl-manual-install-slim-dual-a3-reuse-*
sha256sum -c manifest/SHA256SUMS
bash bin/00_print_config.sh
bash bin/install_all.sh
```

上述已知双机没有必须手填的配置，也不需要重新安装 env。执行前仍需人工确认
`00_print_config.sh` 显示的目录、模型、`SOC_VERSION=ascend910_9362` 和
`NIC_NAME=enp23s0f3` 与本机一致；`HCCL_IF_IP` 留空时从 NIC 自动取本机 IPv4。只有
自动取址失败才在 `config.env` 填写本机 `HCCL_IF_IP`。若任一路径、SoC 或 NIC 已变化，
先修改对应项；不要修改固定 commit，也不要用旧 AFD 目录作为新目标目录。

A5 或其他新节点不使用该双机专用 profile，使用通用 `slim` 包，并填写
`CANN_ROOT`、`MODEL_PATH`、`PYTHON_BIN`、`SOC_VERSION`、`NIC_NAME`、必要时的
`HCCL_IF_IP`、安装/源码目录和实际可访问的 Git/pip 镜像。`CANN_ROOT` 必须指向
CANN 9.0.0 的唯一真实根目录；不得先 source 其他版本再继续。

安装完成后保存包目录和源码目录：

```bash
export BUNDLE_ROOT="$PWD"
source "$BUNDLE_ROOT/bin/activate_runtime.sh"
export AFD_PLUGIN_ROOT
```

轻量包会从分支基线提交应用二进制 patch，核对目标 tree，并生成内容确定的本地交付 commit。各节点的交付 commit 必须相同，且工作树必须干净。

## 3. H0 平台审计

每台 NPU 节点都执行，任何一项不一致就停止：

```bash
source "$BUNDLE_ROOT/bin/activate_runtime.sh"
bash "$BUNDLE_ROOT/bin/06_verify_install.sh"
git -C "$VLLM_ROOT" rev-parse HEAD
git -C "$VLLM_ASCEND_ROOT" rev-parse HEAD
git -C "$AFD_PLUGIN_ROOT" rev-parse HEAD
git -C "$VLLM_ROOT" status --short
git -C "$VLLM_ASCEND_ROOT" status --short
git -C "$AFD_PLUGIN_ROOT" status --short
readlink -f "$CANN_ROOT"
npu-smi info
```

通过条件：两个上游 commit 与第 1 节完全一致；三个 `status --short` 均为空；CANN 只有 9.0.0；NPU 健康；开始验证前没有其他 NPU 进程。把输出保存为文本，后续随证据包回传。

## 4. A5 同栈路径匹配 native control

不能把 MTP-off 的 token 文件跨路径用作 N2/N3 的 exact golden。speculative decoding 会改变 target 校验的执行 shape；即使 native 自身稳定，近似并列 logits 也可能在 MTP-off 和 N2 间选择不同 token。必须按 target/draft 执行模式与 MTP N 生成以下 5 份 no-AFD control：

| control key | target | draft | MTP |
|---|---|---|---|
| `eager_mtp_off` | eager | off | off |
| `eager_mtp_n1` | eager | eager | N1 |
| `eager_mtp_n2` | eager | eager | N2 |
| `graph_target_draft_eager_mtp_n2` | FULL_DECODE_ONLY | eager | N2 |
| `graph_target_draft_graph_mtp_n3` | FULL_DECODE_ONLY | Graph | N3 |

生成器逐点使用 8 卡冷启动、执行 10 条 prompt x 3 轮、正常停止并检查 NPU 清理。开始前本机不得存在任何其他 NPU 进程：

```bash
source "$BUNDLE_ROOT/bin/activate_runtime.sh"
cd "$AFD_PLUGIN_ROOT"
export MODEL_PATH HCCL_IF_IP GLOO_SOCKET_IFNAME HCCL_SOCKET_IFNAME
export PHASE1_GOLDEN_ROOT="/data/validation/dsv4-phase1-a5-native-controls"

bash tools/dsv4/run_phase1_native_controls.sh list
bash tools/dsv4/run_phase1_native_controls.sh preflight
bash tools/dsv4/run_phase1_native_controls.sh run
```

若中途失败，保留原目录；处理后可使用同一根目录只生成尚不存在的 control，例如：

```bash
bash tools/dsv4/run_phase1_native_controls.sh run eager_mtp_n2
```

每个 `golden_results.json` 必须是 `passed=true`、`rounds=3`、`prompt_count=10`、空 mismatch；metadata 中的 control key、target/draft、MTP N、CANN 和两个上游 commit 必须与表格及第 1 节一致。矩阵 preflight 会再次校验这些字段。PD 的 `NATIVE_GOLDEN_PATH` 可用 `$PHASE1_GOLDEN_ROOT/eager_mtp_off/golden_results.json` 提供固定 prompt 集；PD 输出判定仍必须使用第 7 节生成的路径匹配 control golden。

## 5. A5 standalone 门禁

矩阵脚本固定了 9 个代表点：

| 点 | 目的 |
|---|---|
| `a8f8_eager_u1_mtp_off` | A5 基础回归 |
| `a8f8_eager_u1_n1` | N1 兼容性 |
| `a8f8_eager_u2_n2` | eager U2、多 token |
| `a8f8_graph_u1_n2` | target Graph、draft eager |
| `a8f8_graph_u2_n3` | A8F8 最大一期组合、full draft Graph |
| `a4f8_eager_u1_n2` | `F=2A` fan-out 基础路径 |
| `a4f8_graph_u2_n3` | `F=2A` 最大一期组合 |
| `a8f4_eager_u1_n2` | `A=2F` 高 HBM 基础路径 |
| `a8f4_graph_u2_n3` | `A=2F` 最大一期组合 |

先执行 F0。F0 每点一次冷启动、1 轮、batch 1/8/32，不等待 30 分钟 idle：

```bash
source "$BUNDLE_ROOT/bin/activate_runtime.sh"
cd "$AFD_PLUGIN_ROOT"
export MODEL_PATH
export PHASE1_GOLDEN_ROOT
export PHASE1_OUTPUT_BASE="/data/validation/dsv4-phase1-a5-$(date +%Y%m%d_%H%M%S)"
bash tools/dsv4/run_phase1_a5_matrix.sh list
bash tools/dsv4/run_phase1_a5_matrix.sh preflight
bash tools/dsv4/run_phase1_a5_matrix.sh f0
```

F0 全部通过后执行 F1。F1 每点两次冷启动、10 条 prompt x 3 轮 serial exact、batch 1/8/32，并在第一轮加入 1800 秒 idle-resume：

```bash
bash tools/dsv4/run_phase1_a5_matrix.sh f1
```

定位失败时可只重跑一个点，但必须使用新的 `PHASE1_OUTPUT_BASE`：

```bash
export PHASE1_OUTPUT_BASE="/data/validation/dsv4-phase1-a5-retry-$(date +%Y%m%d_%H%M%S)"
bash tools/dsv4/run_phase1_a5_matrix.sh f0 a8f4_graph_u2_n3
```

每个 `validation_summary.json` 必须满足：`passed=true`；`golden` 指向该点对应的路径匹配 control；serial mismatch 为空；batch 1/8/32 的 `valid=true` 并保留各自 `token_exact_count`；topology/rank/capacity 与点名一致；U2 点观察到真实双 stage；两个 role return code 为 0；fatal marker 为空；每轮停止后 NPU cleanup 通过。batch exact 若仍复现 `UPSTREAM-DSV4-BI-001`，保留原始记录并单独分析；只有路径匹配 control 稳定而 AFD serial 新增分叉才阻塞插件门禁。不得使用 `ALLOW_NPU_PROCESSES` 绕过清理门禁。

## 6. 双机 PD 配置

以下将 Prefill/FFN 机记为 P/F，将 Attention 机记为 A；Proxy 可放在 P/F。两台机器安装同一个补丁包，模型、CANN、上游 commit 和 afd-plugin 交付 commit 必须一致。

双机专用包已附带按历史实跑配置生成的 `DUAL_A3_PD_COMMON.env.example`。在两台机器
各自执行一次 `init`，直接以该文件为模板：

```bash
export MATRIX="$AFD_PLUGIN_ROOT/tools/dsv4/mooncake_pd_manual/pd_graph_matrix.sh"
export CFG="/data/config/dsv4-phase1-pd"
export PD_GRAPH_MATRIX_COMMON_TEMPLATE="$BUNDLE_ROOT/DUAL_A3_PD_COMMON.env.example"
bash "$MATRIX" init "$CFG"
bash "$MATRIX" list "$CFG"
```

不要修改生成的 `*-role.env`。检查 `common.env` 中的 Prefill/Decode IP、NIC、模型、
Mooncake、CANN 和 `NATIVE_GOLDEN_PATH`；路径不变时无需手填。`init` 会把本机
afd-plugin HEAD 写入 `common.env`；两台机器的 `AFD_PD_COMMIT` 必须相同。每一轮在
两台机器设置同一个逻辑运行根：

```bash
export MATRIX_RUN_BASE="/data/run/dsv4-phase1-pd-r1"
```

一期 F1 使用 3 个路径匹配 control 和 4 个 AFD 点：

| control 点 | 对应 AFD 点 | 物理角色 |
|---|---|---|
| `control_graph_u2_mtp2_a8` | `afd_graph_u2_mtp2` | P8 + D8；P8 + A8F8 |
| `control_graph_u2_mtp3_a8` | `afd_graph_u2_mtp3`、`afd_graph_u2_split_a8f4_mtp3` | P8 + D8；P8 + A8F8；P8F4 + A8 |
| `control_graph_u2_mtp3_a4` | `afd_graph_u2_split_a4f8_mtp3` | P8 + D4；P8F8 + A4 |

把第 4 节的 `eager_mtp_off` control 放到 `common.env` 的 `NATIVE_GOLDEN_PATH`，仅用于提供同一 prompt 集。三个 PD control golden 分别存入 `${MATRIX_RUN_BASE}/f1-control/<路径键>/golden_results.json`；不要跨路径键复制，也不要用任一 native control 代替 PD control。

## 7. 生成路径匹配 control

以 `control_graph_u2_mtp2_a8` 为例，启动前先在对应节点执行 `check`：

```bash
# P/F 机
bash "$MATRIX" check "$CFG" control_graph_u2_mtp2_a8 prefill
# A 机
bash "$MATRIX" check "$CFG" control_graph_u2_mtp2_a8 decode
# Proxy 所在机
bash "$MATRIX" check "$CFG" control_graph_u2_mtp2_a8 proxy
```

按 Prefill、Decode、Proxy 顺序启动，在 Proxy 生成 control golden，再逆序停止和收集：

```bash
# P/F 机
bash "$MATRIX" start "$CFG" control_graph_u2_mtp2_a8 prefill
# A 机
bash "$MATRIX" start "$CFG" control_graph_u2_mtp2_a8 decode
# Proxy 所在机
bash "$MATRIX" start "$CFG" control_graph_u2_mtp2_a8 proxy
bash "$MATRIX" record-control "$CFG" control_graph_u2_mtp2_a8 proxy
bash "$MATRIX" stop "$CFG" control_graph_u2_mtp2_a8 proxy
# A 机
bash "$MATRIX" stop "$CFG" control_graph_u2_mtp2_a8 decode
bash "$MATRIX" collect "$CFG" control_graph_u2_mtp2_a8 decode
# P/F 机
bash "$MATRIX" stop "$CFG" control_graph_u2_mtp2_a8 prefill
bash "$MATRIX" collect "$CFG" control_graph_u2_mtp2_a8 prefill
```

将点名依次替换为 `control_graph_u2_mtp3_a8` 和 `control_graph_u2_mtp3_a4`，重复本节。每次 control 必须自身 30/30 稳定；若 native 与 control 不同，保留差异但不得用 native golden 直接替代 PD control。

## 8. 双机 AFD F0/F1

共置 A8F8 点 `afd_graph_u2_mtp2`、`afd_graph_u2_mtp3` 使用 P/F 机的 `prefill` 和 A 机的 `decode`。启动顺序与 control 相同。在 Proxy 先执行 `smoke`，再执行 `validate`：

```bash
# 两台 NPU 节点和 Proxy：启动前分别执行 check，然后按 prefill/decode/proxy 启动。
bash "$MATRIX" smoke "$CFG" afd_graph_u2_mtp2 proxy
bash "$MATRIX" validate "$CFG" afd_graph_u2_mtp2 proxy
bash "$MATRIX" evidence "$CFG" afd_graph_u2_mtp2 decode
```

split 点 `afd_graph_u2_split_a4f8_mtp3`、`afd_graph_u2_split_a8f4_mtp3` 使用 P/F 机的 `prefill_ffn` 和 A 机的 `attention`：

```bash
# P/F 机
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a4f8_mtp3 prefill_ffn
# A 机
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a4f8_mtp3 attention
# Proxy 所在机
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy

# 三个 check 都通过后再启动。
# P/F 机
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a4f8_mtp3 prefill_ffn
# A 机
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a4f8_mtp3 attention
# Proxy 所在机
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy
bash "$MATRIX" smoke "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy
bash "$MATRIX" validate "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy
# A 机
bash "$MATRIX" evidence "$CFG" afd_graph_u2_split_a4f8_mtp3 attention
```

四个 AFD 点都按逆序 `proxy -> decode/attention -> prefill/prefill_ffn` 停止。每个角色停止后执行对应的 `collect`。第一次全部通过后，把 `MATRIX_RUN_BASE` 改成 `...-r2`，重新生成 3 份 control golden，再完整执行 4 个 AFD 点，形成第二次独立冷启动证据。

每个 AFD 点必须同时满足：

1. Attention health 和全部 FFN connector loop ready。
2. `smoke`、取消请求后的 health recovery、batch 1/8/32 和 `validate` 通过。
3. 路径匹配 control 的 10 条 prompt x 3 轮达到 30/30 token exact。
4. `evidence` 覆盖全部 Attention rank，并观测到真实 U2 两 stage。
5. A4F8/A8F4 日志中的 rank、peer 和 per-peer token count 符合各自 fan-out/fan-in 方向。
6. 无 OOM、timeout、`Communication_Error`、`507015`、Python traceback 或 EngineCore fatal。
7. Attention 先退出、FFN 随后正常退出；停止后无残留端口和 NPU 进程。
8. 第二次冷启动结果与第一次一致。

## 9. 证据打包与回传

Standalone 验证使用统一收集器；它保留 JSON、环境、NPU 快照和截断日志，不包含 profiler 原始目录：

```bash
source "$BUNDLE_ROOT/bin/activate_runtime.sh"
cd "$AFD_PLUGIN_ROOT"
bash tools/dsv4/collect_phase1_validation.sh \
  /data/artifacts/dsv4-phase1-a5-evidence.tar.gz \
  "$PHASE1_GOLDEN_ROOT" \
  "$PHASE1_OUTPUT_BASE/f0" \
  "$PHASE1_OUTPUT_BASE/f1"
sha256sum -c /data/artifacts/dsv4-phase1-a5-evidence.tar.gz.sha256
```

PD 每个 `collect` 会打印一个小型归档及 `.sha256`。回传以下内容：

1. 包含 5 份 native control 和 9 点 standalone 结果的 A5 evidence tar 及 `.sha256`。
2. 两轮所有 control/AFD 角色的 `collect` 归档和 `.sha256`。
3. 两台机器 H0 审计文本。
4. 三份路径匹配 control golden。
5. 任何失败点的完整 `validation_summary.json`、role 日志尾部、首次 fatal 前后至少 200 行，以及当时的 `npu-smi info`。

不要只回传成功截图，也不要在失败后覆盖原 `RUN_ROOT`。收到上述材料后，可按 stack、启动、数据面、Graph 动态路由、token exact、生命周期和清理六类门禁逐项分析。
