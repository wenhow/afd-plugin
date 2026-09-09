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

### 1.1 先选择验证轨道

本文同时保留 A5 和双 A3 两条验证轨道，但同一组机器不需要顺序执行全部章节。
先按当前机器选择路径：

| 当前环境 | 必须执行 | 不在本机执行 | 能得到的结论 |
|---|---|---|---|
| A5 | 第 2、3、4、5、9 节的 A5 部分 | 第 6、7、8 节 | A5 native control 和 standalone AFD 门禁 |
| 双 A3，仅验证启动/F0 | 第 2、3、6、8.2、9 节的双 A3 部分 | 第 4、5、7、8.3 节 | 双机服务启动、health、smoke、取消后恢复；不含 token-exact |
| 双 A3，完整 F1 | 第 2、3、6、7、8、9 节的双 A3 部分 | 第 4、5 节 | 双机路径匹配 control、AFD 30/30 token-exact 和生命周期门禁 |

A5 和双 A3 没有验证产物依赖。两条轨道只共享仓库中版本固定的 10 条 prompt 清单：

```text
tools/dsv4/phase1_prompts.json
          ├── A5 第 4 节生成 5 份 native control -> 第 5 节 standalone AFD
          └── 双 A3 第 7 节生成 3 份 PD control -> 第 8.3 节双机 AFD
```

旧版文档让双 A3 的 `NATIVE_GOLDEN_PATH` 指向 A5 `eager_mtp_off`，实际只为读取其中
的 prompt，造成了不必要的交叉依赖。新版双 A3 配置直接指向上述 prompt-only JSON。
A5 native token 不会进入双 A3 流程；双 A3 的 exact golden 必须由第 7 节在相同 PD
拓扑和相同 Graph/MTP 组合上生成。

### 1.2 总体验收范围

第一阶段仍需完成的外部硬门禁：

1. A5 平台审计、独立安装和 5 份同栈、路径匹配 native control。
2. A5 standalone A8F8、A4F8、A8F4 的 eager/Graph、U1/U2、MTP N=1/2/3 代表矩阵。
3. A8F4 在高 HBM A5 上的真实模型加载和端到端请求。
4. 双机 PD Graph/U2 下的 A8F8 N2/N3、A4F8 N3、A8F4 N3。
5. 每个 PD 点的真实双 stage、FFN Graph 动态路由、取消后恢复、优雅退出、NPU 清理和第二次冷启动。
6. 路径匹配的 PD no-AFD control，以及每次冷启动 10 条 prompt x 3 轮的 30/30 token exact。

不属于第一阶段：U3、正式 P2 性能收益、A5 参数调优、非整数 A/F、TP/SP/CP/DCP/PP、TP3、非等量 TP2 和 TP2 full-draft Graph/U2/MTP 最大组合。

## 2. 补丁包安装（按平台选择）

### 2.1 双 A3 使用复用包

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

旧 `/data/z00569729/code/afd-plugin` 中已有的本地修改不需要删除、reset 或 stash。
新版安装器验证其 HEAD 仍为 `2164240...` 后，只读取已提交对象；本地修改继续留在旧
目录，并将 `status` 和 tracked diff 备份到
`/data/z00569729/run/dsv4-afd-phase1-install/state/afd-seed-*`。新目标目录保持独立且
必须干净。若 seed HEAD 不是固定提交，仍会停止，需要另行确认版本，不能强制切换。
双机已验证的 `dsv4-afd-p8-a16f8-dual-a3-graph-u2-cann900-20260904-r14.tar.gz`
本来就以 overlay 方式修改 `profiler.py`，因此该文件显示 `M` 是已知现场状态。新版
目标已包含 R14 profiler 能力及后续修复，不要把旧 overlay 再应用一次。

### 2.2 A5 或其他新节点使用通用包

A5 或其他新节点不使用该双机专用 profile，使用通用 `slim` 包，并填写
`CANN_ROOT`、`MODEL_PATH`、`PYTHON_BIN`、`SOC_VERSION`、`NIC_NAME`、必要时的
`HCCL_IF_IP`、安装/源码目录和实际可访问的 Git/pip 镜像。`CANN_ROOT` 必须指向
CANN 9.0.0 的唯一真实根目录；不得先 source 其他版本再继续。

### 2.3 两条轨道共同要求

精简容器没有 `ss` 不阻塞安装或验证，也不需要重装 env。端口门禁会按
`ss -> netstat -> /proc/net/tcp*` 自动回退；三种来源都不可用才视为环境缺失。

复用环境中的 vLLM-Ascend 版本前缀允许保留现场构建产生的
`0.19.1rc2.dev629`；审计固定其 `g3da28f941` 提交后缀，并在后续同时核验源码 HEAD
和实际 Python 导入根。因旧审计前缀不匹配而中断时不需要重装 env，换用新版包直接
重跑 `install_all.sh`；干净的既有一期目标目录会沿包内提交链升级。

供应商 CANN、NNAL/ATB 和 custom ops 的 `set_env` 在严格 Bash `set -u` 下可能读取
未定义变量。新版安装器会在 source 期间临时关闭 nounset 并立即恢复；不要为绕过
错误手工修改供应商脚本，也不需要在配置中伪造 `ASCEND_CUSTOM_OPP_PATH`。

安装完成后保存包目录和源码目录：

```bash
export BUNDLE_ROOT="$PWD"
source "$BUNDLE_ROOT/bin/activate_runtime.sh"
export AFD_PLUGIN_ROOT
```

轻量包会从分支基线提交应用二进制 patch，核对目标 tree，并生成内容确定的本地交付 commit。各节点的交付 commit 必须相同，且工作树必须干净。

## 3. H0 平台审计（A5 与双 A3 都执行）

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

## 4. A5 专属：同栈路径匹配 native control

**执行位置：A5。双 A3 操作者不要在 A3 上运行本节脚本。** 本节的 5 份 control
只供 A5 第 5 节使用，不向双 A3 交付任何文件。

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

每个 `golden_results.json` 必须是 `passed=true`、`rounds=3`、`prompt_count=10`、空 mismatch；metadata 中的 control key、target/draft、MTP N、CANN 和两个上游 commit 必须与表格及第 1 节一致。矩阵 preflight 会再次校验这些字段。

本节完成后，A5 留存全部 5 份 control，供第 5 节 standalone 验证使用。双 A3 不
复制这些文件；它直接使用仓库中的 `tools/dsv4/phase1_prompts.json`。

## 5. A5 专属：standalone AFD 门禁

**执行位置：A5。双 A3 完全跳过本节。** 本节读取第 4 节的 5 份 control，但不会
向第 6、7、8 节输出任何依赖文件。

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

## 6. 双 A3 专属：准备 PD 配置

**执行位置：两台 A3。A5 不执行本节。** 以下将 Prefill/FFN 机记为 P/F，将
Attention 机记为 A；Proxy 放在 P/F。两台机器必须安装同一个补丁包，并使用相同的
模型、CANN、两个上游 commit 和 afd-plugin 交付 commit。

### 6.1 先确认要做 F0 还是 F1

| 目标 | 仓库 prompt 清单 | 第 7 节 PD control | 第 8 节允许动作 |
|---|---|---|---|
| 启动定位/F0 | 包内已有，不读取 | 跳过 | `check`、`start`、`status`、`smoke`、`stop`、`collect` |
| 正式 F1 | 包内已有，直接读取 | 必须先生成 3 份 | F0 全部动作，加 `validate`、`evidence` |

正式 F1 的 prompt 来源随 afd-plugin 一起安装，默认路径为：

```text
${AFD_PLUGIN_ROOT}/tools/dsv4/phase1_prompts.json
```

两台机器确认文件存在；它的内容由同一 afd-plugin commit 保证一致：

```bash
test -s "$AFD_PLUGIN_ROOT/tools/dsv4/phase1_prompts.json"
sha256sum "$AFD_PLUGIN_ROOT/tools/dsv4/phase1_prompts.json"
```

双 A3 模板中的 `NATIVE_GOLDEN_PATH` 是为了兼容既有脚本保留的历史字段名；在新版
模板中它指向这个 prompt-only JSON，不代表依赖 A5 native golden。第 7 节会调用
双 A3 上的 no-AFD 服务生成 token。

### 6.2 初始化或升级配置

双机专用包包含 `DUAL_A3_PD_COMMON.env.example`。两台机器都先设置：

```bash
export MATRIX="$AFD_PLUGIN_ROOT/tools/dsv4/mooncake_pd_manual/pd_graph_matrix.sh"
export CFG="/data/config/dsv4-phase1-pd"
export PD_GRAPH_MATRIX_COMMON_TEMPLATE="$BUNDLE_ROOT/DUAL_A3_PD_COMMON.env.example"
export MATRIX_FAILURE_MODE=keep-shell
```

第一次创建配置时，两台机器分别执行：

```bash
bash "$MATRIX" init "$CFG"
bash "$MATRIX" list "$CFG"
```

如果 `$CFG` 已存在，不要重新 `init`；安装新版包后执行：

```bash
bash "$MATRIX" refresh-config "$CFG"
bash "$MATRIX" list "$CFG"
```

`refresh-config` 只允许沿交付提交链快进干净的 afd-plugin 目录；分叉提交或脏工作树会
报错并保留现场。`MATRIX_FAILURE_MODE=keep-shell` 会在动作失败后保留当前终端，因此
不能只看 shell 是否还在；看到任何 `ERROR` 或 `failed with status` 都必须停止当前点，
只有明确出现对应的 `preflight passed`、`ready` 或 `complete` 才能进入下一步。

不要修改生成的 `*-role.env`。只检查两台机器的 `common.env` 中以下字段：

1. Prefill/Decode IP 和 `NIC_NAME` 与机器一致。
2. `MODEL_PATH`、Mooncake、CANN 和 venv 指向同一套固定运行环境。
3. 两台机器的 `AFD_PD_COMMIT` 相同。
4. `NATIVE_GOLDEN_PATH` 指向
   `${AFD_PLUGIN_ROOT}/tools/dsv4/phase1_prompts.json`。

每轮在两台机器设置同一个逻辑运行根。第一轮使用：

```bash
export MATRIX_RUN_BASE="/data/run/dsv4-phase1-pd-r1"
```

### 6.3 control、AFD 点和物理角色映射

| PD control | AFD 点 | P/F 机角色 | A 机角色 |
|---|---|---|---|
| `control_graph_u2_mtp2_a8` | `afd_graph_u2_mtp2` | `prefill` | `decode`（A8F8 共置） |
| `control_graph_u2_mtp3_a8` | `afd_graph_u2_mtp3` | `prefill` | `decode`（A8F8 共置） |
| `control_graph_u2_mtp3_a8` | `afd_graph_u2_split_a8f4_mtp3` | `prefill_ffn`（P8F4） | `attention`（A8） |
| `control_graph_u2_mtp3_a4` | `afd_graph_u2_split_a4f8_mtp3` | `prefill_ffn`（P8F8） | `attention`（A4） |

三个 PD control 分别写到
`${MATRIX_RUN_BASE}/f1-control/<路径键>/golden_results.json`。不得跨路径键复制，也不得
用 A5 native golden 代替这些 PD control。

## 7. 双 A3 专属：生成路径匹配 PD control

**仅正式 F1 执行本节；只做启动/F0 时跳过。** 开始前必须满足：两台机器第 3、6 节
已通过；Proxy 所在机的 `NATIVE_GOLDEN_PATH` 指向仓库 prompt 清单；本轮使用新的
`MATRIX_RUN_BASE`；两台 NPU 机器没有上一点残留进程。

需要依次生成以下三份 control，不能少，也不能互相复用：

1. `control_graph_u2_mtp2_a8`
2. `control_graph_u2_mtp3_a8`
3. `control_graph_u2_mtp3_a4`

下面是第一个点的完整顺序。每条命令都在注释指定的机器执行：

```bash
# 1. 三个角色分别预检。
# P/F 机
bash "$MATRIX" check "$CFG" control_graph_u2_mtp2_a8 prefill
# A 机
bash "$MATRIX" check "$CFG" control_graph_u2_mtp2_a8 decode
# Proxy 所在 P/F 机
bash "$MATRIX" check "$CFG" control_graph_u2_mtp2_a8 proxy

# 2. 按 Prefill -> Decode -> Proxy 启动。
# P/F 机
bash "$MATRIX" start "$CFG" control_graph_u2_mtp2_a8 prefill
# A 机
bash "$MATRIX" start "$CFG" control_graph_u2_mtp2_a8 decode
# Proxy 所在 P/F 机
bash "$MATRIX" start "$CFG" control_graph_u2_mtp2_a8 proxy

# 3. Proxy 生成本路径的 no-AFD control。
bash "$MATRIX" record-control "$CFG" control_graph_u2_mtp2_a8 proxy

# 4. 按 Proxy -> Decode -> Prefill 逆序停止，每个角色随后 collect。
bash "$MATRIX" stop "$CFG" control_graph_u2_mtp2_a8 proxy
bash "$MATRIX" collect "$CFG" control_graph_u2_mtp2_a8 proxy
# A 机
bash "$MATRIX" stop "$CFG" control_graph_u2_mtp2_a8 decode
bash "$MATRIX" collect "$CFG" control_graph_u2_mtp2_a8 decode
# P/F 机
bash "$MATRIX" stop "$CFG" control_graph_u2_mtp2_a8 prefill
bash "$MATRIX" collect "$CFG" control_graph_u2_mtp2_a8 prefill
```

保持角色和顺序不变，把点名依次替换为 `control_graph_u2_mtp3_a8`、
`control_graph_u2_mtp3_a4`，各执行一遍完整流程。每份 control 必须自身 30/30 稳定。
后续 AFD 必须与对应的 PD control token-exact 一致；本流程不读取或比较 A5 token。

## 8. 双 A3 专属：AFD F0/F1

### 8.1 四个验证点

按第 6.3 节的表执行四个点。共置 A8F8 点由 `decode` 动作在 A 机内部连续拉起 FFN
和 Attention；split 点先由 P/F 机的 `prefill_ffn` 拉起 Prefill 和等待连接的 FFN，
再由 A 机的 `attention` 拉起 Attention。FFN 没有 HTTP 端口，必须以 `status` 输出的
connector loop 数量判断是否 ready。

### 8.2 F0：可跳过第 4、5、7 节的启动与请求验证

F0 不读取任何 golden，适合先定位安装、模型加载、Graph capture、HCCL、Mooncake 和
服务生命周期。不得在 F0 执行 `validate`，也不得把 F0 结果标记为 token-exact。

以下是当前 A4F8 点的完整 F0 顺序：

```bash
# 1. 三个角色分别预检。
# P/F 机
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a4f8_mtp3 prefill_ffn
# A 机
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a4f8_mtp3 attention
# Proxy 所在 P/F 机
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy

# 2. 三个 check 明确通过后，按 P/F -> A -> Proxy 启动。
# prefill_ffn 会让 FFN 在后台等待 Attention，因此该命令返回后立即启动 A 机。
# P/F 机
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a4f8_mtp3 prefill_ffn
# A 机
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a4f8_mtp3 attention
# Proxy 所在 P/F 机
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy

# 3. 分别确认 Prefill、全部 FFN loop、Attention health 和 Proxy health。
# P/F 机
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a4f8_mtp3 prefill_ffn
# A 机
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a4f8_mtp3 attention
# Proxy 所在 P/F 机
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy

# 4. Proxy 执行不依赖 golden 的 smoke 和取消后恢复。
bash "$MATRIX" smoke "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy

# 5. 按 Proxy -> Attention -> Prefill/FFN 逆序停止并收集。
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy
bash "$MATRIX" collect "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy
# A 机
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a4f8_mtp3 attention
bash "$MATRIX" collect "$CFG" afd_graph_u2_split_a4f8_mtp3 attention
# P/F 机
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a4f8_mtp3 prefill_ffn
bash "$MATRIX" collect "$CFG" afd_graph_u2_split_a4f8_mtp3 prefill_ffn
```

验证 A8F4 时把点名替换为 `afd_graph_u2_split_a8f4_mtp3`，角色不变。验证两个共置
A8F8 点时，P/F 机角色改为 `prefill`，A 机角色改为 `decode`；点名分别为
`afd_graph_u2_mtp2` 和 `afd_graph_u2_mtp3`。每个点必须重新执行 check、冷启动、
status、smoke、逆序停止和 collect，不能在运行中切换点名。

### 8.3 F1：依赖第 7 节的正式 token-exact 验收

先确认当前 `MATRIX_RUN_BASE` 下已经有第 7 节的三份 PD control。每个 AFD 点按 8.2
的角色映射重新冷启动；`smoke` 通过后，在停止服务之前执行该点对应的
`validate` 和 `evidence`：

```bash
# afd_graph_u2_mtp2 正在运行时，在 Proxy 和 A 机分别执行：
bash "$MATRIX" validate "$CFG" afd_graph_u2_mtp2 proxy
bash "$MATRIX" evidence "$CFG" afd_graph_u2_mtp2 decode

# afd_graph_u2_mtp3 正在运行时：
bash "$MATRIX" validate "$CFG" afd_graph_u2_mtp3 proxy
bash "$MATRIX" evidence "$CFG" afd_graph_u2_mtp3 decode

# afd_graph_u2_split_a4f8_mtp3 正在运行时：
bash "$MATRIX" validate "$CFG" afd_graph_u2_split_a4f8_mtp3 proxy
bash "$MATRIX" evidence "$CFG" afd_graph_u2_split_a4f8_mtp3 attention

# afd_graph_u2_split_a8f4_mtp3 正在运行时：
bash "$MATRIX" validate "$CFG" afd_graph_u2_split_a8f4_mtp3 proxy
bash "$MATRIX" evidence "$CFG" afd_graph_u2_split_a8f4_mtp3 attention
```

每一组命令只在对应点运行期间执行；不要同时启动四个点。完成 `validate/evidence` 后，
按 8.2 的逆序停止和 collect。第一轮全部通过后，两台机器切换到新的运行根：

```bash
export MATRIX_RUN_BASE="/data/run/dsv4-phase1-pd-r2"
```

随后重新执行第 7 节生成 3 份 control，再完整执行本节 4 个 AFD 点，形成第二次独立
冷启动证据。

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

### 9.1 A5 操作者

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

A5 只回传以下内容，不需要收集双 A3 的 PD 角色日志：

1. 包含 5 份 native control 和 9 点 standalone 结果的 A5 evidence tar 及 `.sha256`。
2. A5 的 H0 审计文本。

### 9.2 双 A3 操作者

PD 每个 `collect` 会打印一个小型归档及 `.sha256`。双 A3 不需要生成或回传 A5
standalone evidence；回传以下内容：

1. 两轮所有 control/AFD 角色的 `collect` 归档和 `.sha256`。
2. 两台机器 H0 审计文本。
3. 每轮 3 份路径匹配 PD control golden，共 6 份，路径中保留 `r1/r2` 标识。
4. 仓库 `tools/dsv4/phase1_prompts.json` 的 SHA256 和 afd-plugin commit。
5. 任何失败点的完整 `validation_summary.json`、role 日志尾部、首次 fatal 前后至少
   200 行，以及当时的 `npu-smi info`。

不要只回传成功截图，也不要在失败后覆盖原 `RUN_ROOT`。收到上述材料后，可按 stack、启动、数据面、Graph 动态路由、token exact、生命周期和清理六类门禁逐项分析。
