# DeepSeek-V4 M9 Mooncake PD 双机手工管理脚本

该目录把安装指南中的双机命令收敛为一个入口：

```bash
bash pd.sh <install|check|start|status|smoke|record-control|validate|profile-start|profile-check|profile-stop|stop|collect> config.env
```

update13 的双 A3 最短操作流程见
[`UPDATE13_RUNBOOK_ZH.md`](UPDATE13_RUNBOOK_ZH.md)。该手册当前作为全部本机功能
开发完成后的外部拓扑/F1 验收入口；本机开发不等待双 A3，也不继续使用 update10
的 native `GOLDEN_PATH` 作为 PD + AFD exact-token 门禁。

PD 分离 `P8+A16F8` 的 Graph+U2 比例验证使用
[`pd_graph_matrix.sh`](pd_graph_matrix.sh)，完整三组对照、双侧 Profile 和验收门禁见
[`DEEPSEEK_V4_AFD_P8_A16F8_DUAL_A3_GRAPH_U2_VALIDATION_GUIDE_ZH.md`](../../../docs/npu/DEEPSEEK_V4_AFD_P8_A16F8_DUAL_A3_GRAPH_U2_VALIDATION_GUIDE_ZH.md)。

## 1. 准备六个配置

control 和 AFD 各使用 Prefill、Decode、Proxy 三个配置。配置中的
`DEPLOYMENT_VARIANT` 分别设置为 `pd_control` 或 `pd_afd`，`NODE_ROLE` 分别为
`prefill`、`decode`、`proxy`。两种模式使用不同 `RUN_ROOT`，避免 PID 和日志覆盖。

在 A3-P 创建 Prefill 配置，例如：

```bash
bash pd.sh init /data/z00569729/config/pd-control-prefill.env
vi /data/z00569729/config/pd-control-prefill.env
```

control golden 只在 A3-P 生成并保存：

```text
/data/z00569729/validation/dsv4_m9_pd_control/golden_results.json
```

首次配置和当前 F0 阶段该文件不存在是正常的。全部计划功能开发完成、进入 F1 后，
才在 control 模式执行 `record-control`；随后 AFD 模式的 `validate` 自动读取并
检查该文件。

六个配置中的 `PREFILL_IP`、`DECODE_IP`、源码路径、Mooncake 安装模式和
`AFD_PD_COMMIT` 必须完全一致。矩阵 `init` 会把当前 afd-plugin checkout 的精确
40 位 HEAD 自动写入 `common.env`；直接复制 `config.env.example` 时必须手工替换
`CHANGE_ME`，不能填写分支名。镜像已经在目标 `VENV_ROOT` 内安装 Mooncake 0.3.9
时使用 `MOONCAKE_INSTALL_MODE=existing`，不需要传 wheel；否则使用 `wheel`。
安装包沿同一提交链升级后，矩阵 `check` 会自动刷新旧 `common.env` 中的
`AFD_PD_COMMIT` 并重新生成 role 配置。也可以先显式执行
`bash "$MATRIX" refresh-config "$CFG"`；分叉提交或脏工作树不会自动刷新。
所有 `check/start` 操作还要求 afd-plugin 工作树完全干净；验证内容必须先提交到该
40 位 HEAD，不能用未校验的 overlay 或本地 diff 改变实际运行代码。

Decode 拓扑支持两组固定值：首轮外部拓扑冒烟使用 `DECODE_DP_SIZE=8`、
`DECODE_TP_SIZE=1`；TP1 通过后使用 `DECODE_DP_SIZE=4`、
`DECODE_TP_SIZE=2` 验证 M8 冻结的 TP2 契约。Prefill 两种情况下都保持
DP2/TP4，Attention/FFN 都保持各 8 个物理 rank。

脚本还提供 `DECODE_EXECUTION_MODE`、`DECODE_U_BATCHES`、
`DECODE_ENABLE_MTP` 和 `DECODE_MTP_DRAFT_EXECUTION`。它们用于在 TP1/TP2 基线
后依次执行 eager/U2、Graph/U1、Graph/U2 和一 token MTP 的独立外部拓扑冒烟。
这套顺序不决定本机开发先后。F0 只验证功能和数据路径，暂不比较 golden；全部
计划功能完成后的 F1 才要求 control 与 AFD 配置逐项相同，并使用不同的
`RUN_ROOT` 和 `PD_CONTROL_GOLDEN_PATH`。配置入口存在不代表实模已通过；TP2
full-draft Graph U2 + MTP 仍显式拒绝，多 speculative token 仍不支持。

### 1.1 本机 F0-local：PD no-AFD control

单机 16 个逻辑 NPU 可以运行 Prefill 8 卡加 no-AFD Decode 8 卡，用于先完成 PD
control 的功能冒烟；完整 `Prefill + Decode Attention A8 + Decode FFN F8` 需要 24
个逻辑 NPU，不能在本机完成。

同机 control 的三个配置使用相同 `RUN_ROOT`、本机 IP 和 `pd_control`，并设置：

```bash
PREFILL_DEVICES="0,1,2,3,4,5,6,7"
ATTENTION_DEVICES="8,9,10,11,12,13,14,15"
ALLOW_COLOCATED_PD_CONTROL="1"
```

先在没有模型进程时分别完成本机 Mooncake round-trip。正式启动时依次启动 Prefill、
Decode、Proxy；Decode 的 `start` 只允许 PID 文件记录的 Prefill 子进程，并再次验证
两组设备严格不重叠。不得用 `ALLOW_NPU_PROCESSES=1` 代替该检查。服务就绪后在
Proxy 配置执行：

```bash
bash pd.sh smoke /path/to/pd-control-proxy.env
```

`smoke` 覆盖 batch 1/8/32、请求取消和取消后的成功恢复请求，产物明确记录
`golden_checked=0`。`record-control` 和 `validate` 留到后续 F1 正确性冻结。

## 2. 安装和预检

在 A3-P 和 A3-D 分别执行：

```bash
bash pd.sh install /data/z00569729/config/pd-control-prefill.env
bash pd.sh check /data/z00569729/config/pd-control-prefill.env
```

Decode 节点将配置文件换成 `pd-control-decode.env`。默认不修改系统包；系统依赖尚未
安装时，先手工安装，或者显式设置 `INSTALL_SYSTEM_PACKAGES=1`。脚本支持
`apt-get`、`dnf` 和 `yum`：Ubuntu 安装 `iproute2`，openEuler/RPM 系安装
`iproute`，二者都提供 `ip` 和 `ss`。root 容器直接安装，非 root 才调用
`sudo`。

离线镜像无法安装 `iproute` 时无需跳过门禁：新版脚本按 `ip -> ifconfig ->
netstat -ie -> Python 标准库` 检查网卡主 IPv4，按 `ss -> netstat ->
/proc/net/tcp*` 检查监听端口。配置中的 `NIC_NAME` 和角色 IP 仍必须准确；回退
模式不会放宽检查。

精简镜像没有 `hostname` 命令时，`collect` 会从
`/proc/sys/kernel/hostname`、Shell `HOSTNAME` 或 `uname -n` 获取主机名；无需为
收集验收包额外安装 `hostname` RPM。

`collect` 对 `netstat -lntp` 使用默认 10 秒超时；在大量 Mooncake/HCCL socket
下超时会自动改为保存 `/proc/net/tcp*`，不会阻塞证据包生成。F0 证据包同时包含
`batches.json` 和取消后的 `recovery.json`。

Mooncake 运行时必须预加载 `libjemalloc.so.2`。脚本会自动检查 Debian/Ubuntu 的
`/usr/lib/aarch64-linux-gnu`、openEuler/RPM 的 `/usr/lib64` 等常见位置，并回退
到 `ldconfig`；特殊镜像可在角色配置中设置绝对路径
`MOONCAKE_JEMALLOC=/custom/path/libjemalloc.so.2`。

若镜像使用 `/usr/local/Ascend/cann-9.0.0`，且 Mooncake 的两个配套库位于
`/usr/local/lib`，角色配置必须同时设置：

```bash
CANN_ROOT="/usr/local/Ascend/cann-9.0.0"
CANN_VERSION="9.0.0"
ATB_ROOT="/usr/local/Ascend/nnal/atb"
MOONCAKE_LIBRARY_DIR="/usr/local/lib"
```

运行检查会把目标 venv 的 `site-packages` 放到 CANN Python 路径之前，避免误用
CANN 目录中同名但不完整的 `mooncake` 包；随后验证两个配套库、`ldd`、Python
contract 和实际双 NPU round-trip。

`check` 还会对目标 venv 的 `torch_npu/lib/libop_plugin_atb.so` 执行 `ldd`，要求
`libatb.so` 和 Torch 动态库全部解析成功。外置 ATB 会在 venv 和 `torch/lib`
进入运行环境后加载，避免 `check` 通过而 worker 启动后才报 `libatb.so not found`。
加载 CANN/ATB 自带的 `set_env.sh` 时，脚本会临时关闭 Bash `nounset`，随后恢复
调用者原来的 `set -u` 状态；无需修改供应商脚本或手工设置 `ZSH_VERSION`。

角色启动会先激活通用运行时，再执行 Mooncake 门禁。交付版会保留调用者选择的
`DSV4_CANN_ROOT` 和 venv，确保第二次门禁仍使用配置中的 CANN，而不会回退到
`/mnt/workspace` 默认路径。

`existing` 只跳过 wheel 安装，不跳过验证。`check` 会核对版本、目标 venv、
动态库/CANN 路径，记录 `.so` 指纹，并执行本机两 NPU、2 MiB、两轮 Mooncake
round-trip。update12 使用当前节点配置的业务 IP 和 `NIC_NAME`，不再使用
loopback 模拟正式数据路径。A3-P 和 A3-D 生成的 `mooncake-libraries.sha256`
必须一致。只在
已经有独立证据且需要快速重跑时设置 `RUN_LOCAL_ROUNDTRIP=0`。

当前 F0-local 和功能开发阶段保持 `ENABLE_BATCH_INVARIANT=0`；batch-invariant
统一延后到全部计划功能开发完成后的 F1。届时六份配置切换到独立 BI venv，并设置
`VLLM_ASCEND_WORKTREE_MODE=batch_invariant_patch`、`ENABLE_BATCH_INVARIANT=1`
和 `BATCH_INVARIANT_OPP_ROOT`。脚本只接受交付补丁的两文件指纹，角色启动会自动
加载 OPP。完整步骤见
`docs/npu/DEEPSEEK_V4_BATCH_INVARIANT_DUAL_A3_VALIDATION_GUIDE_ZH.md`。

Proxy 配置只需执行 `check`。

## 3. 启动和验证

以 `pd_afd` 为例，按以下顺序在对应节点执行：

```bash
# A3-P
bash pd.sh start /data/z00569729/config/pd-afd-prefill.env

# A3-D；脚本会紧邻启动 FFN 和 Attention，并等待 8 个 FFN loop
bash pd.sh start /data/z00569729/config/pd-afd-decode.env

# Proxy 所在节点
bash pd.sh start /data/z00569729/config/pd-afd-proxy.env
bash pd.sh validate /data/z00569729/config/pd-afd-proxy.env
```

随时检查：

```bash
bash pd.sh status /data/z00569729/config/pd-afd-prefill.env
bash pd.sh status /data/z00569729/config/pd-afd-decode.env
bash pd.sh status /data/z00569729/config/pd-afd-proxy.env
```

`pd_control` 模式在 Proxy 上执行 `record-control`，只要求三轮内部稳定；
`pd_afd` 模式在 Proxy 上执行 `validate`，以 control golden 做 30/30 exact-token
比较，并覆盖 batch 1/8/32 和请求取消恢复。它不能读取远端 Decode 日志，因此
最终结论还要求 Decode 输出件中存在 KV transfer 和正确的模式 marker。

## 4. 停止顺序

```bash
bash pd.sh stop /data/z00569729/config/pd-afd-proxy.env
bash pd.sh stop /data/z00569729/config/pd-afd-decode.env
bash pd.sh stop /data/z00569729/config/pd-afd-prefill.env
```

脚本只停止自身 PID 文件记录、且命令行能够识别的 process group。默认不会
发送 KILL；TERM 超时时先检查进程，确认需要后再设置 `FORCE_KILL=1`。

## 5. 生成要回传的小输出件

每种模式的三个角色分别执行；下面以 `pd_afd` 为例：

```bash
bash pd.sh collect /data/z00569729/config/pd-afd-prefill.env
bash pd.sh collect /data/z00569729/config/pd-afd-decode.env
bash pd.sh collect /data/z00569729/config/pd-afd-proxy.env
```

收集动作不以日志内容作为成功条件。日志中即使存在 fatal marker，也会继续生成
归档；归档成功仅表示证据已保存，是否通过验收需要在收集后分析日志和结果文件。

每次输出两个文件：

```text
dsv4-m9-pd-control-<role>-<timestamp>.tar.gz
dsv4-m9-pd-afd-<role>-<timestamp>.tar.gz
dsv4-m9-pd-control-<role>-<timestamp>.tar.gz.sha256
dsv4-m9-pd-afd-<role>-<timestamp>.tar.gz.sha256
```

默认每个压缩包最大 2 MiB，三个角色合计不会超过约 6 MiB。内容只有：

- 三个源码 commit、关键 Python 包版本、Mooncake 安装模式和 `.so` 指纹；
- wheel 模式额外记录 Mooncake wheel SHA256；
- `status`、PID、监听端口和 `npu-smi info`；
- runtime check、本地 round-trip 结果；
- 每个角色日志最后 256 KiB、最近 50 条 KV transfer 证据和最近 200 条
  fatal marker；
- Proxy 的 smoke、golden、batch、取消恢复结果。

不会包含完整日志、模型、wheel、profiler、core dump、完整环境变量或 API key。
将三个 `.tar.gz` 和对应 `.sha256` 发回即可；通常实际总大小会明显低于上限。
