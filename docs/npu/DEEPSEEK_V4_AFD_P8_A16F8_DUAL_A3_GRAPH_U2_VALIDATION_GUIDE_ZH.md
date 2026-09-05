# DeepSeek-V4 P8+A16F8 双 A3 Graph+U2 逐步验证指导书

## 1. 目标、节点和操作规则

本指导书验证 PD 分离下的三种 Graph+U2 部署：

> **当前阻断（2026-09-05）**：2026-09-04 采集使用的 R14 overlay 会把 warmup 的
> AllToAllV split 固化到 Graph。split 实际随每层、每次请求的 expert routing 改变，live
> replay 不会重新执行 Python 预处理，因此该方案存在静默错算风险，已从提交中移除。
> 下文第 14 章数据仍可作为请求、U2 stage、流水与性能观测，但不是输出正确性证据。
> 新一轮 A16F8 Graph 验证必须先完成 MC2 capacity 或 AllToAllV graph-break 修复。

| 测试点 | 部署 | active NPU | 目的 |
| --- | --- | ---: | --- |
| `afd_graph_u2` | `P8 + [A8F8]` | 24 | 建立 Attention/FFN 同机时的当前基线 |
| `afd_graph_u2_split_a8f8` | `[P8F8] + A8` | 24 | 等量资源下测量 A/F 跨机拆分的代价或收益 |
| `afd_graph_u2_split_a16f8` | `[P8F8] + A16` | 32 | 测量 Attention 扩为两倍后的容量和资源效率 |

这三组组成一套控制变量实验，不只是分别验证服务能否启动：

1. `split A8F8` 对比 `colocated A8F8`：P/A/F 卡数和总 active NPU 都不变，   主要变量是 Attention 与 FFN 从同机通信变为跨机通信。该对比得到   `placement_penalty_split_a8f8_vs_colocated_a8f8`，用于量化跨机 HCCL、调度和部署位置变化带来的综合影响。
2. `split A16F8` 对比 `split A8F8`：两组都采用跨机 split，Prefill 和 FFN 数量不变，只把 Attention 从 8 增加到 16。每个 FFN rank 因此聚合两个 Attention  peer，即 `A:F=2:1` fan-in。该对比得到 `ratio_gain_a16f8_vs_split_a8f8`，用于判断增加 8 张 Attention NPU 后，系统容量是否扩展、FFN 是否成为瓶颈。
3. `split A16F8` 对比 `colocated A8F8`：得到 `end_to_end_a16f8_vs_colocated_a8f8`，回答目标拓扑相对当前部署的端到端变化。
   该对比同时改变了部署位置和资源数量，只能作为最终效果对照，不能单独用于归因。

最终可以得到三类结果：

- **运行和稳定性观测**：Graph+U2 是否完成 capture/replay，在线 `stage_count=2` 是否覆盖全部 Attention rank，三轮请求是否全部完成，以及是否存在 fatal、异常退出或 NPU 清理失败。只有再通过路径匹配 golden，才能升级为正确性结论。
- **性能和资源效率**：三轮 output throughput 均值与 CV、P50/P90/P99 TPOT，以及
  `output_tokens_per_second_per_npu`。这些是当前归档中已经具备并在第 14 章落表的指标。
- **流水重叠证据**：Profile 中 FFN Bubble/Free、长 receive 等待，以及 recv/compute、compute/send 的时间重叠变化，用来解释吞吐或时延变化来自哪里。

因此本轮只能测量跨机拆分和增加 Attention 资源后的候选变化，不能单凭当前样本确定
placement 代价或把变化归因于 A:F 比例。对比结果采用
`measurement_only_no_fixed_gain_threshold`，只给出测量值和变化比例；正式宣称“有收益”
还要求性能变化超过运行波动，并结合路径匹配 control 与 Profile 排除偶然波动、资源增加和
瓶颈转移。

两台机器固定为：

| 本文名称 | IP | 职责 |
| --- | --- | --- |
| A3-PF | `7.150.2.43` | Prefill、split FFN、Proxy、benchmark |
| A3-A | `7.150.7.206` | 共置 A8F8 Decode，或 split Attention |

执行规则：

1. 每个代码块上方都标明执行节点，只在该节点执行。
2. 不设置也不使用 `POINT` 变量，命令中始终写完整测试点名称。
3. 一个步骤返回非零时立即停止，不执行下一步。
4. `evidence` 只在 A3-A 执行：共置点用 `decode`，拆分点用 `attention`。
5. 性能和 Profile 是两轮独立实验，中间必须停止所有服务并冷启动。
6. 三个性能点全部完成后，才开启 Profile。
7. `collect-final` 发现日志 fatal 时只告警，仍会生成归档；归档成功仅表示证据已收集，不表示该轮验收通过，日志结论在归档后离线分析。

在执行性能流程前还有一个当前矩阵**没有自动化**的 P0 正确性前置门禁：

1. 在 clean commit 中完成并提交“所有支持 gear 均走 MC2”或“AllToAllV graph break”的
   安全 Graph 修复；禁止恢复静态 split cache，禁止用 force-load-balance 代替正确性修复。
2. 使用相同 vLLM、vLLM-Ascend、CANN、模型、PD 路径、拓扑、batch-invariant 配置和启动
   顺序，先证明路径匹配的 PD no-AFD control 跨冷启动稳定。
3. AFD eager 和 AFD Graph/U2 分别与该 control 完成 30/30 逐 token exact，并确认 eager 与
   Graph/U2 之间逐 token exact；同时覆盖两组不同 prompt、同 shape/不同 routing、多 layer、
   偏斜 routing、零接收 rank、最大 gear 和 batch 1/8/32。
4. 保存 control、eager、Graph/U2 的原始 token IDs、配置指纹、日志和 cleanup 证据。

`pd_graph_matrix.sh` 当前只自动化本指导书中的部署、性能、Profile 和证据收集，不会自动生成
或比较上述 golden。若没有单独完成并审阅这项前置门禁，只能查看历史结果或执行诊断，
**不得进入第 5 至第 11 章的性能/Profile 流程**。

满足前置门禁后的完整顺序只有下面这一条：

```text
准备环境
  -> 人工确认安全 Graph 修复与路径匹配 token-exact golden 前置门禁
  -> 性能1：共置 A8F8
  -> 性能2：split A8F8
  -> 性能3：split A16F8
  -> 生成性能对比
  -> Profile1：共置 A8F8
  -> Profile2：split A8F8
  -> Profile3：split A16F8
  -> 收集结果
```

## 2. 固定软件栈和交付包

本章 2.1 和 2.3 仅保留 2026-09-04 证据来源，**不得用于新的交付验证**。新一轮验证
直接从步骤 2.2 开始，在两台机器切到同一个 clean commit，不安装任何历史 overlay。`init` 会把当时
工作树的 40 位 HEAD 自动写入 `AFD_PD_COMMIT`，两台机器和生成配置三者必须完全一致。

两台机器必须使用相同的软件版本：

```text
CANN            9.0.0
vLLM            0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
vLLM-Ascend     3da28f9414583d2d0b672a8f06d1fae142404bda
afd-plugin      feat/dsv4-afd-graph-u2-multistream-all-on-v1
afd-plugin evidence base 2164240b31efc8605bf84cc45afc628996669554 + R14 overlay
Mooncake        0.3.9
model           /data/models/DeepSeek-V4-Flash-w8a8-mtp
```

以下文件只是 2026-09-04 R14 实验的历史证据包，不是新验证交付包，不得安装到新验证
工作树：

```text
/mnt/workspace/delivery/dsv4-afd-p8-a16f8-dual-a3-graph-u2-cann900-20260904-r14.tar.gz
/mnt/workspace/delivery/dsv4-afd-p8-a16f8-dual-a3-graph-u2-cann900-20260904-r14.tar.gz.sha256
```

新验证直接使用步骤 2.2 切换后的完整 clean 工作树，不能只复制 `pd_graph_matrix.sh`；它依赖
同一提交中的 `pd.sh`、角色启动脚本、性能工具、NPU 监控工具和 Profile 汇总工具。

### 步骤 2.1：历史 R14 归档信息（新验证不执行）

在当前验证机执行：

```bash
ssh z00569729@7.150.2.43 'mkdir -p /data/z00569729/packages'
ssh z00569729@7.150.7.206 'mkdir -p /data/z00569729/packages'
scp /mnt/workspace/delivery/dsv4-afd-p8-a16f8-dual-a3-graph-u2-cann900-20260904-r14.tar.gz* \
  z00569729@7.150.2.43:/data/z00569729/packages/
scp /mnt/workspace/delivery/dsv4-afd-p8-a16f8-dual-a3-graph-u2-cann900-20260904-r14.tar.gz* \
  z00569729@7.150.7.206:/data/z00569729/packages/
```

### 步骤 2.2：两台机器都切换 afd-plugin 分支

以下命令在 A3-PF 和 A3-A 分别执行一次：

```bash
cd /data/z00569729/code/afd-plugin
git status --short --branch
```

若输出包含本地修改，先保存：

```bash
git stash push -u -m "before-p8-a16f8-validation"
```

然后切换并更新分支：

```bash
git fetch origin feat/dsv4-afd-graph-u2-multistream-all-on-v1
git switch feat/dsv4-afd-graph-u2-multistream-all-on-v1 || \
  git switch --track -c feat/dsv4-afd-graph-u2-multistream-all-on-v1 \
  origin/feat/dsv4-afd-graph-u2-multistream-all-on-v1
git merge --ff-only origin/feat/dsv4-afd-graph-u2-multistream-all-on-v1
git rev-parse HEAD
```

最后一条在两台机器必须输出同一个 40 位 commit。历史 R14 证据使用：

```text
2164240b31efc8605bf84cc45afc628996669554
```

### 步骤 2.3：历史 R14 安装记录（新验证禁止执行）

以下命令在 A3-PF 和 A3-A 分别执行一次：

```bash
cd /data/z00569729/packages
sha256sum -c dsv4-afd-p8-a16f8-dual-a3-graph-u2-cann900-20260904-r14.tar.gz.sha256
tar -xzf dsv4-afd-p8-a16f8-dual-a3-graph-u2-cann900-20260904-r14.tar.gz \
  -C /data/z00569729/code/afd-plugin
test -f /data/z00569729/code/afd-plugin/afd_plugin/compat/patches/npu/moe_graph.py
grep -n '_afd_graph_splits_by_key' \
  /data/z00569729/code/afd-plugin/afd_plugin/compat/patches/npu/moe_graph.py
git -C /data/z00569729/code/afd-plugin status --short
```

以上命令只解释旧证据如何产生，不得在新验证中执行。新验证必须跳过步骤 2.1 和 2.3，
保持 afd-plugin 工作树干净；R14 中 Profile 生命周期工具的安全部分已由当前分支承接，
AllToAllV split-cache 则明确不承接。
R14 使用真正的无 step schedule 显式 Profile 窗口：服务启动和 ACL Graph capture阶段不创建 profiler；
`profile-start` 才通过 Attention 的本机 middleware 调用 engine client profile-start collective RPC，并以 AFD control payload 同步启动 FFN profiler。
调用 `start()` 时 profiler 立即从 `NONE` 进入 `RECORD`，不再等待 engine step 越过
`skip_first/wait/warmup`。历史 R14 绕过 torch_npu 生命周期 API 的异常吞掉包装；本地
Attention profiler 的底层 CANN 启动或停止失败会使 HTTP 请求和 shell 命令失败，远端 FFN
失败仍必须结合 FFN 日志与 raw-ready 门禁判断，不能把本地 HTTP 成功当成跨节点 ACK。
benchmark 结束后，`profile-stop` 先在每个启用 Profile 的 worker 上执行设备级同步，确保Graph U2 各 NPU stream 的已提交任务完成，再调用两端 worker 内的`profiler.stop()`。脚本等待非空 `device_*/data` 及 device、host 两类`end_info.done`。在线 worker 只写 raw CANN 数据，停服务后才用相同 CANN/venv
离线解析生成 `trace_view.json`。
历史 Profile 模式把 vLLM 内部 shutdown timeout 设为 240 秒，用于降低退出兜底路径强杀
worker 的概率；第 14 章证据仍出现强杀，因此该配置不是生命周期通过证明。
历史验证时，两台机器安装的是同一 R14 包。该包将采集参数对齐旧双机脚本使用的
vLLM-Ascend 原生配置：Level1、`data_simplification=True`、`record_shapes=False`，并拒绝与
`MSMONITOR_USE_DAEMON` 同时启用。Profile 生命周期日志提升为 WARNING，用于观察 API
collective、Attention 到 FFN 控制消息、worker PID/device、同步和 stop 耗时及 raw 文件现场。

R14 已在本机使用 vLLM 0.23/CANN 9.0.0 做最小真实 NPU 多流生命周期验证：4 个 stream 异步提交 96 次 NPU matmul，生成非空`device_0/data`、device/host 两类 `end_info.done` 和 `profiler_info.json`。离线`torch_npu.profiler.analyse()` 进一步生成了包含 MatMul kernel 的`trace_view.json`（462939 bytes）和 `kernel_details.csv`（30932 bytes）。该结果只证明 R14 配置和生命周期代码可运行，不能替代本指南的双机 AFD Profile 验收。

## 3. 生成并同步配置

新验证必须在 clean commit 上使用新的 `CFG` 目录执行 `init`，不要沿用任何历史 overlay
生成的角色 env。`init` 会把当前 40 位 HEAD 固定到配置中，并为本轮生成独立的运行与
Profile raw 路径。

### 步骤 3.1：在 A3-PF 生成配置

```bash
cd /data/z00569729/code/afd-plugin
MATRIX=tools/dsv4/mooncake_pd_manual/pd_graph_matrix.sh
CFG=/data/z00569729/config/pd-a16f8-graph-u2
bash "$MATRIX" init "$CFG"
bash "$MATRIX" list "$CFG"
```

如果配置目录已经由本轮验证生成，`init` 会拒绝覆盖。这种情况下不要删除目录，直接执行：

```bash
bash "$MATRIX" list "$CFG"
```

### 步骤 3.2：在 A3-PF 确认固定值

```bash
grep -E \
'^(MODEL_PATH|PREFILL_IP|DECODE_IP|NIC_NAME|CANN_ROOT|CANN_VERSION|AFD_PD_COMMIT|AFD_PROFILE_ENABLE|PERFORMANCE_CONCURRENCY|PERFORMANCE_NUM_PROMPTS)=' \
"$CFG/common.env"
```

必须确认：

```text
MODEL_PATH="/data/models/DeepSeek-V4-Flash-w8a8-mtp"
PREFILL_IP="7.150.2.43"
DECODE_IP="7.150.7.206"
NIC_NAME="enp23s0f3"
CANN_ROOT="/usr/local/Ascend/cann-9.0.0"
CANN_VERSION="9.0.0"
AFD_PD_COMMIT="<步骤 2.2 中 git rev-parse HEAD 的 40 位值>"
AFD_PROFILE_ENABLE="0"
```

2026-09-04 历史结果的复现主 workload 固定为：

```text
PERFORMANCE_CONCURRENCY="32"
PERFORMANCE_NUM_PROMPTS="128"
```

先用 C32/128 完成可比复现。C64/256 只能作为后续独立压力点，必须使用新的结果目录并明确
标记为不同 workload，不得与第 14 章 C32/128 数字直接计算收益。若需修改，只编辑
`common.env`，不要编辑生成的角色 env。

### 步骤 3.3：从 A3-PF 同步完整配置到 A3-A

在 A3-PF 执行：

```bash
ssh z00569729@7.150.7.206 \
  'mkdir -p /data/z00569729/config/pd-a16f8-graph-u2'
rsync -a "$CFG/" \
  z00569729@7.150.7.206:/data/z00569729/config/pd-a16f8-graph-u2/
```

### 步骤 3.4：两台机器分别准备当前 shell

在 A3-PF 和 A3-A 分别执行一次：

```bash
cd /data/z00569729/code/afd-plugin
MATRIX=tools/dsv4/mooncake_pd_manual/pd_graph_matrix.sh
CFG=/data/z00569729/config/pd-a16f8-graph-u2
```

关闭当前终端重新登录后，必须重新执行这三条。

## 4. 一次性环境和网络检查

### 步骤 4.1：A3-PF 检查本机地址

```bash
ip -o -4 addr show dev enp23s0f3
npu-smi info
ss -ltnp
```

输出中必须存在 `7.150.2.43`。

### 步骤 4.2：A3-A 检查本机地址

```bash
ip -o -4 addr show dev enp23s0f3
npu-smi info
ss -ltnp
```

输出中必须存在 `7.150.7.206`。

### 步骤 4.3：两台机器检查共享内存

```bash
df -h /dev/shm
find /dev/shm -maxdepth 1 -user "$USER" \
  \( -name 'psm_*' -o -name 'sem.mp-*' \) -printf '%f\n' | wc -l
```

`/dev/shm` 不能接近 100%。TBE 并行编译和 Python multiprocessing 都会使用该目录；
空间耗尽会表现为 Attention/FFN 长时间互等，随后报 `OSError: [Errno 28] No space left on device`。不要在服务运行时直接删除文件；先停止本用户的 AFD/vLLM 进程并确认没有打开句柄，再清理本用户遗留的 `psm_*`、`sem.mp-*` IPC 对象。

两机必须双向放通：API `8100/8910/9000`、AFD `29761`、Mooncake `30000-30007/30100-30115/15000-17000` 和 HCCL 基础端口 `50000/51000/52000`。
FFN 的 `8911` 不是 HTTP 健康接口，不要对它执行 HTTP 探测。

## 5. 性能轮次 1：共置 A8F8 基线

本节全部完成前不要进入第 6 节。配置必须保持 `AFD_PROFILE_ENABLE="0"`。

### 步骤 5.1：A3-PF 启动 Prefill

```bash
bash "$MATRIX" check "$CFG" afd_graph_u2 prefill
bash "$MATRIX" start "$CFG" afd_graph_u2 prefill
bash "$MATRIX" status "$CFG" afd_graph_u2 prefill
```

必须看到 `Prefill health: OK`。

### 步骤 5.2：A3-A 启动共置 Decode A8F8

```bash
bash "$MATRIX" check "$CFG" afd_graph_u2 decode
bash "$MATRIX" start "$CFG" afd_graph_u2 decode
bash "$MATRIX" status "$CFG" afd_graph_u2 decode
```

必须看到 Decode health 正常和 `FFN connector loops: 8/8`。

### 步骤 5.3：A3-PF 检查两端直连健康

```bash
curl --noproxy '*' -fsS -o /dev/null -w 'prefill HTTP %{http_code}\n' \
  http://7.150.2.43:8100/health
curl --noproxy '*' -fsS -o /dev/null -w 'decode HTTP %{http_code}\n' \
  http://7.150.7.206:8910/health
```

两条都必须输出 HTTP 200。

### 步骤 5.4：A3-PF 启动 Proxy

```bash
bash "$MATRIX" check "$CFG" afd_graph_u2 proxy
bash "$MATRIX" start "$CFG" afd_graph_u2 proxy
bash "$MATRIX" status "$CFG" afd_graph_u2 proxy
```

必须看到 `Proxy health: OK`。

### 步骤 5.5：A3-PF 执行功能 smoke

```bash
bash "$MATRIX" smoke "$CFG" afd_graph_u2 proxy
```

必须显示 F0 passed。若失败，停止本节，不启动 monitor。

### 步骤 5.6：两端复核服务健康

```bash
# A3-A
bash "$MATRIX" status "$CFG" afd_graph_u2 decode

# A3-PF
bash "$MATRIX" status "$CFG" afd_graph_u2 proxy
```

F0 smoke 只做 batch 1/8/32 功能检查，请求按批次串行提交，不能保证在线 U2 覆盖全部 Attention rank。此处不要执行 `evidence`；严格的 `8/8` U2 stage 证据在三轮 P2 并发负载完成后的步骤 5.9 收集。

### 步骤 5.7：两端分别启动 NPU monitor

A3-PF：

```bash
bash "$MATRIX" monitor-start "$CFG" afd_graph_u2 prefill
```

A3-A：

```bash
bash "$MATRIX" monitor-start "$CFG" afd_graph_u2 decode
```

### 步骤 5.8：A3-PF 执行三轮 P2

```bash
bash "$MATRIX" benchmark "$CFG" afd_graph_u2 p2
```

必须显示 `afd_graph_u2 p2 complete`。

### 步骤 5.9：A3-A 收集 stage 证据并停止 monitor

```bash
bash "$MATRIX" evidence "$CFG" afd_graph_u2 decode
bash "$MATRIX" monitor-stop "$CFG" afd_graph_u2 decode
```

### 步骤 5.10：A3-PF 停止 monitor

```bash
bash "$MATRIX" monitor-stop "$CFG" afd_graph_u2 prefill
```

### 步骤 5.11：按顺序停止本轮服务

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2 proxy
```

A3-A：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2 decode
```

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2 prefill
```

### 步骤 5.12：三端角色分别生成性能验收包

A3-PF：

```bash
bash "$MATRIX" collect-final "$CFG" afd_graph_u2 proxy
bash "$MATRIX" collect-final "$CFG" afd_graph_u2 prefill
```

A3-A：

```bash
bash "$MATRIX" collect-final "$CFG" afd_graph_u2 decode
```

三条都必须输出 `ARTIFACT=` 和 `SHA256_FILE=`。命令若同时打印 fatal warning，仍保留并回传归档，由离线分析判断该轮是否通过。完成后进入第 6 节。

## 6. 性能轮次 2：split A8F8

本节是新一轮冷启动，不复用第 5 节的进程。

### 步骤 6.1：A3-PF 启动 Prefill 和 FFN

```bash
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

此时 FFN 可以等待 Attention，暂时不要把 connector loop 未就绪当成失败。

### 步骤 6.2：A3-A 启动 Attention A8

```bash
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a8f8 attention
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a8f8 attention
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a8f8 attention
```

### 步骤 6.3：A3-PF 确认 FFN 连接完成

```bash
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

必须看到 Prefill health 正常和 `FFN connector loops: 8/8`。

### 步骤 6.4：A3-PF 启动 Proxy

```bash
curl --noproxy '*' -fsS http://7.150.2.43:8100/health >/dev/null
curl --noproxy '*' -fsS http://7.150.7.206:8910/health >/dev/null
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a8f8 proxy
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a8f8 proxy
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a8f8 proxy
```

### 步骤 6.5：A3-PF 执行 smoke

```bash
bash "$MATRIX" smoke "$CFG" afd_graph_u2_split_a8f8 proxy
```

### 步骤 6.6：两端复核服务健康

```bash
# A3-A
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a8f8 attention

# A3-PF
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

此处不要执行 `evidence`。F0 smoke 不保证触发并覆盖在线 U2；严格的 `8/8` stage 证据在三轮 P2 并发负载完成后的步骤 6.9 收集。

### 步骤 6.7：两端分别启动 monitor

A3-PF：

```bash
bash "$MATRIX" monitor-start "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

A3-A：

```bash
bash "$MATRIX" monitor-start "$CFG" afd_graph_u2_split_a8f8 attention
```

### 步骤 6.8：A3-PF 执行三轮 P2

```bash
bash "$MATRIX" benchmark "$CFG" afd_graph_u2_split_a8f8 p2
```

### 步骤 6.9：两端收尾 monitor

A3-A：

```bash
bash "$MATRIX" evidence "$CFG" afd_graph_u2_split_a8f8 attention
bash "$MATRIX" monitor-stop "$CFG" afd_graph_u2_split_a8f8 attention
```

A3-PF：

```bash
bash "$MATRIX" monitor-stop "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

### 步骤 6.10：按顺序停止服务

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a8f8 proxy
```

A3-A：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a8f8 attention
```

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

### 步骤 6.11：分别生成性能验收包

A3-PF：

```bash
bash "$MATRIX" collect-final "$CFG" afd_graph_u2_split_a8f8 proxy
bash "$MATRIX" collect-final "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

A3-A：

```bash
bash "$MATRIX" collect-final "$CFG" afd_graph_u2_split_a8f8 attention
```

完成后进入第 7 节。

## 7. 性能轮次 3：split A16F8

本节是目标拓扑，Attention 使用 A3-A 的 16 张 NPU。

### 步骤 7.1：A3-PF 启动 Prefill 和 FFN

```bash
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
```

### 步骤 7.2：A3-A 启动 Attention A16

当前 clean 分支在本步骤前是阻断状态：先完成并提交“所有支持 gear 走 MC2”或
“AllToAllV graph break”的安全修复，再执行以下命令。不得重新安装任何历史 overlay，也不得通过
force-load-balance 伪造生产正确性。

```bash
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a16f8 attention
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a16f8 attention
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a16f8 attention
```

Attention ready 后，在 A3-PF 执行以下检查：

```bash
FFN_LOG=/data/z00569729/run/dsv4-mooncake-pd-graph-matrix/afd_graph_u2_split_a16f8/logs/prefill_ffn/ffn.log
if grep -nE 'LocalScalarDenseNpu|Not allow to synchronize captured-stream|107027|Allocated size does not match required size' \
  "$FFN_LOG"; then
  echo 'ERROR: FFN Graph capture仍触发同步，停止本轮'
  false
else
  echo 'FFN capture synchronization marker absent; routing correctness not proven'
fi
```

该检查的非错误输出只表示本次日志没有命中已知的 capture 同步错误 marker，不证明 live
expert routing、AllToAllV split 或 Graph replay 数值正确。只有第 1 章的路径匹配 token-exact
前置门禁通过后，才允许把本步骤视为性能流程的一部分。

历史上，没有 overlay 时 A16F8 FFN 会在部分 Graph capture gear 的
`torch.repeat_interleave` 触发 `LocalScalarDenseNpu / 107027`。R2-R4 逐步加入
`output_size`、warmup split cache 和零输出静态 tensor，最终让 capture 与请求跑完。

复盘确认这不是安全修复：`input_splits/output_splits` 取决于动态 `topk_ids`，同一个
shape 在不同请求、不同 layer 可以路由到不同专家；FFN Graph replay 又不会重新进入
Python `_preprocess`。进程级共享 dispatcher 还会使同 shape 的后层 warmup 覆盖前层
split。给 key 增加 layer 只能修复覆盖，仍不能处理 live routing，所以 R14 已撤回。

历史 A8F8 `token_exact=1/1` 和 A16F8 128/128 只能证明样本可运行，不能覆盖同 shape/
不同 routing。新修复必须至少证明：FFN 最大聚合 gear 不落入动态 AllToAllV Graph；两组
不同 prompt 的 eager/Graph 输出一致；偏斜 routing、零接收 rank 和最大 gear 通过。完成前
日志检查即使输出 `OK`，也不能继续到性能验收。

### 步骤 7.3：A3-PF 确认 FFN 连接和启动 Proxy

```bash
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
curl --noproxy '*' -fsS http://7.150.2.43:8100/health >/dev/null
curl --noproxy '*' -fsS http://7.150.7.206:8910/health >/dev/null
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a16f8 proxy
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a16f8 proxy
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a16f8 proxy
```

FFN 必须为 `8/8`，Proxy health 必须为 OK。

### 步骤 7.4：A3-PF 执行 smoke

```bash
bash "$MATRIX" smoke "$CFG" afd_graph_u2_split_a16f8 proxy
```

### 步骤 7.5：两端复核服务健康

```bash
# A3-A
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a16f8 attention

# A3-PF
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
```

若在此处执行 `evidence` 得到 `0/16`，不代表 Graph/U2 失败：F0 smoke 只负责功能正确性，不保证让 16 个 Attention rank 都出现在线 `stage_count=2`。此处确认进程和 health 正常即可；严格的 `16/16` 证据在三轮 P2 并发负载完成后的步骤 7.8 收集，门禁仍保持全 rank，不做放宽。

### 步骤 7.6：两端分别启动 monitor

A3-PF：

```bash
bash "$MATRIX" monitor-start "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
```

A3-A：

```bash
bash "$MATRIX" monitor-start "$CFG" afd_graph_u2_split_a16f8 attention
```

### 步骤 7.7：A3-PF 执行三轮 P2

```bash
bash "$MATRIX" benchmark "$CFG" afd_graph_u2_split_a16f8 p2
```

### 步骤 7.8：两端收尾 monitor

A3-A：

```bash
bash "$MATRIX" evidence "$CFG" afd_graph_u2_split_a16f8 attention
bash "$MATRIX" monitor-stop "$CFG" afd_graph_u2_split_a16f8 attention
```

A3-PF：

```bash
bash "$MATRIX" monitor-stop "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
```

### 步骤 7.9：按顺序停止服务

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a16f8 proxy
```

A3-A：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a16f8 attention
```

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
```

### 步骤 7.10：分别生成性能验收包

A3-PF：

```bash
bash "$MATRIX" collect-final "$CFG" afd_graph_u2_split_a16f8 proxy
bash "$MATRIX" collect-final "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
```

A3-A：

```bash
bash "$MATRIX" collect-final "$CFG" afd_graph_u2_split_a16f8 attention
```

### 步骤 7.11：A3-PF 生成三组性能对比

```bash
bash "$MATRIX" compare-ratio "$CFG" p2
```

输出路径类似：

```text
/data/z00569729/run/dsv4-mooncake-pd-graph-matrix/comparison-af-ratio-p2-<timestamp>.json
```

确认对比文件生成后，才进入 Profile 阶段。

## 8. Profile 准备：两端打开 Profile

Profile 与前面三轮性能数据完全分开。不要在性能服务仍运行时修改配置。

### 步骤 8.1：确认所有服务和 monitor 已停止

A3-PF 和 A3-A 分别执行：

```bash
npu-smi info
ss -ltnp
```

确认没有本轮 Prefill、Decode、Attention、FFN、Proxy 和 monitor 进程。

### 步骤 8.2：A3-PF 打开 Profile

```bash
sed -i 's/^AFD_PROFILE_ENABLE=.*/AFD_PROFILE_ENABLE="1"/' "$CFG/common.env"
grep -q '^AFD_PROFILE_FINALIZE_TIMEOUT_SECONDS=' "$CFG/common.env" || \
  printf '\nAFD_PROFILE_FINALIZE_TIMEOUT_SECONDS="300"\n' >>"$CFG/common.env"
grep -q '^AFD_PROFILE_START_TIMEOUT_SECONDS=' "$CFG/common.env" || \
  printf '\nAFD_PROFILE_START_TIMEOUT_SECONDS="60"\n' >>"$CFG/common.env"
grep -q '^AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS=' "$CFG/common.env" || \
  printf '\nAFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS="240"\n' >>"$CFG/common.env"
grep -E '^AFD_PROFILE_(ENABLE|START_TIMEOUT_SECONDS|FINALIZE_TIMEOUT_SECONDS|VLLM_SHUTDOWN_TIMEOUT_SECONDS)=' "$CFG/common.env"
```

必须为：

```text
AFD_PROFILE_ENABLE="1"
AFD_PROFILE_START_TIMEOUT_SECONDS="60"
AFD_PROFILE_FINALIZE_TIMEOUT_SECONDS="300"
AFD_PROFILE_VLLM_SHUTDOWN_TIMEOUT_SECONDS="240"
```

当前 clean 分支使用无 step schedule 的手工 Profile 窗口，不读取
`AFD_PROFILE_SKIP_FIRST/WAIT/WARMUP/ACTIVE`。即使旧配置中仍残留这些变量也不会形成
schedule；建议删除，避免误以为采集窗口仍由 engine step 控制。

### 步骤 8.3：A3-PF 将新配置同步到 A3-A

```bash
rsync -a "$CFG/" \
  z00569729@7.150.7.206:/data/z00569729/config/pd-a16f8-graph-u2/
```

A3-A 执行确认：

```bash
cd /data/z00569729/code/afd-plugin
MATRIX=tools/dsv4/mooncake_pd_manual/pd_graph_matrix.sh
CFG=/data/z00569729/config/pd-a16f8-graph-u2
printf 'CFG=%s\n' "$CFG"
grep -E '^AFD_PROFILE_(ENABLE|START_TIMEOUT_SECONDS|FINALIZE_TIMEOUT_SECONDS|VLLM_SHUTDOWN_TIMEOUT_SECONDS)=' "$CFG/common.env"
bash "$MATRIX" print-config "$CFG" afd_graph_u2 decode | \
  grep -E '^(DEPLOYMENT_VARIANT|NODE_ROLE|AFD_PROFILE_ENABLE|AFD_PROFILE_ATTENTION_DIR|AFD_PROFILE_FFN_DIR)='
```

`common.env` 和 effective config 中的 `AFD_PROFILE_ENABLE` 都必须为 `1`；effective config
还必须显示 `DEPLOYMENT_VARIANT=pd_afd`、`NODE_ROLE=decode`。如果任一项不符，不要执行
`profile-start`。先确认当前 shell 的 `CFG` 是上述 clean 配置目录，再重新同步配置。

Profile 控制能力和 session 在服务启动阶段固定，真正的 Profile 对象只在
`profile-start` 时创建。若 Decode 曾以 `AFD_PROFILE_ENABLE=0` 启动，即使随后修改了
`common.env`，也必须先停止并重新启动 Decode；不能直接对旧进程执行 `profile-start`。

`attention`/`decode` 上的 `profile-start` 通过本机 HTTP API 发起请求；HTTP 非 200 能直接
证明本地 API 或 Attention collective 失败，但该返回值不是跨节点 FFN profiler 的完整 ACK。
split 部署必须先在 `prefill_ffn` 侧登记窗口，再由 Attention 发送控制消息，并在 benchmark
前回到 `prefill_ffn` 节点执行 `profile-check`，以非空且属于本 session 的 FFN raw 目录作为
远端已进入采集态的门禁。停止时同样先由 Attention 发出 stop，再由 `prefill_ffn` 的
`profile-stop` 等待 FFN device/host raw-ready 封口；不能只依据 Attention 本地 API 成功。

同一规则适用于 split 部署的 `attention` 和 `prefill_ffn`：两端都必须先同步
`AFD_PROFILE_ENABLE=1` 的配置，再分别冷启动。不要在服务运行后修改该开关。每次冷启动
会清理上一服务实例的 `profile-started.env`、`profile-finalized.env` 和
`profile-session.env`；`profile-start` 会用本实例新建的 session 文件确认启动时已打开
Profile，避免复用旧状态。

## 9. Profile 轮次 1：共置 A8F8

本轮不执行 smoke、不启动 NPU monitor、不执行 P2。

### 步骤 9.1：A3-PF 启动 Prefill

```bash
bash "$MATRIX" check "$CFG" afd_graph_u2 prefill
bash "$MATRIX" start "$CFG" afd_graph_u2 prefill
```

### 步骤 9.2：A3-A 启动共置 Decode

```bash
bash "$MATRIX" check "$CFG" afd_graph_u2 decode
bash "$MATRIX" start "$CFG" afd_graph_u2 decode
bash "$MATRIX" status "$CFG" afd_graph_u2 decode
```

### 步骤 9.3：A3-PF 启动 Proxy

```bash
bash "$MATRIX" check "$CFG" afd_graph_u2 proxy
bash "$MATRIX" start "$CFG" afd_graph_u2 proxy
bash "$MATRIX" status "$CFG" afd_graph_u2 proxy
```

### 步骤 9.4：A3-A 手工开启 Attention 和 FFN Profile

```bash
bash "$MATRIX" profile-start "$CFG" afd_graph_u2 decode
```

该命令只在 A3-A 执行。它通过 `127.0.0.1:8910/afd/profile/start` 调用 vLLM engine client 的
profile-start collective RPC；8 个 Attention worker 都参与 collective，只有配置的 Attention
DP0 创建 profiler。Attention 再通过 AFD control payload 通知 8 个 FFN worker，只有 FFN
DP0 创建 profiler。profiler 未配置 step schedule，`start()` 立即进入 `RECORD`。本地 API/
Attention 启动错误会使 HTTP 返回失败；对共置 FFN，命令还会轮询本机 FFN raw 根目录确认
其已启动，而不是把控制消息发送成功当作 FFN ACK。只有命令成功返回后才允许执行 9.5。

此时还没有跑 Profile 负载，目录只有少量初始化文件并不表示失败；9.5 运行时 raw
device 数据才会持续增长。矩阵默认将 CANN raw 写到节点本地
`/tmp/dsv4-pd-profile/<MATRIX_RUN_BASE basename>/<point>/{attention,ffn}`，日志、状态和
性能结果仍写入 `MATRIX_RUN_BASE`。这是因为 Python 可以在共享文件系统写入
`FRAMEWORK/torch.op_*`，并不代表 CANN 设备侧 writer 也能可靠写入同一文件系统。
可通过 `MATRIX_PROFILE_BASE` 显式选择其他容量充足的节点本地目录。

真正的完成门禁在 9.7：必须有非空 `device_*/data` 和两类 `end_info.done`，而不是
只看 `*_ascend_pt` 目录或 `FRAMEWORK` 的大小。

### 步骤 9.5：A3-PF 执行 Profile 专用负载

```bash
bash "$MATRIX" benchmark "$CFG" afd_graph_u2 profile
```

benchmark 在 A3-PF 执行，是因为请求入口 Proxy 部署在 A3-PF；实际 NPU Profile 始终由A3-A 上的 Attention/FFN worker 采集。两者不是同一个控制面。

### 步骤 9.6：A3-A 收集本轮 stage 证据

```bash
bash "$MATRIX" evidence "$CFG" afd_graph_u2 decode
```

### 步骤 9.7：A3-A 手工停止 Profile，保持服务运行

```bash
bash "$MATRIX" profile-stop "$CFG" afd_graph_u2 decode
```

该命令通过 Attention plugin 的 `/afd/profile/stop` 调用 engine client profile-stop collective RPC，并通过 AFD control payload 触发 FFN stop。Attention/FFN worker 均先执行设备级同步，再调用各自的 `profiler.stop()`。只有 Attention 和 FFN 都存在非空
`device_*/data`，并同时出现 device、host 两类 `end_info.done` 后命令才返回成功。
失败时不要执行服务 `stop`。

无 schedule 的 profiler 在手工 stop 时，torch_npu 2.10.0.post2 会输出通用提示
`Incorrect schedule: Stop profiler while current state is RECORD`。RECORD 和
RECORD_AND_SAVE 的 stop 都会执行 `stop_trace/finalize_trace/on_trace_ready`，因此仅改变
状态名称不能修复不完整采集；仍以上述 raw 门禁和 9.9 的离线产物门禁判断成败。
当前实现会在 Attention/FFN 日志中记录同步耗时、stop 耗时、实际 `prof_path`、raw 字节数
和封口标志数，shell 每 30 秒打印相同口径。

### 步骤 9.8：按顺序停止本轮服务

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2 proxy
```

A3-A：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2 decode
```

Profile 模式下该命令先检查本轮 `profile-finalized.env`；若 9.7 没有成功，会在发送TERM 前 fail-fast。门禁通过后，它只向 Attention/FFN 主管进程发送 TERM，由主管进程逐级关闭 worker；
退出后再次校验原始 Profile 存在非空 `device_*/data`，并同时存在
`device_*/end_info*.done` 和 `host/end_info.done`。
若 stop 报原始 Profile 不完整，不要继续 9.9，也不要设置`FORCE_KILL=1`，应先保留日志并排查退出链路。

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2 prefill
```

### 步骤 9.9：A3-A 离线生成并校验 trace_view.json

```bash
bash "$MATRIX" profile-analyse "$CFG" afd_graph_u2 decode
```

Attention/FFN worker 的 `tensorboard_trace_handler(..., analyse_flag=False)` 只完成原始
采集，避免 CANN 解析长时间阻塞推理 worker。服务停止并完成原始封口后，本步骤使用采集时相同的 CANN/torch_npu 运行时调用 `torch_npu.profiler.analyse()`，生成`ASCEND_PROFILER_OUTPUT/trace_view.json`，随后自动执行 `profile-summary`。
`msprof` 产物不作为该文件的替代品。

该入口会完成以下动作：

1. 只选择本轮 `profile-session.env` 之后生成的唯一 Attention/FFN Profile 根目录；
2. 先检查原始 `PROF_*` 下是否存在非空 `device_*/data`，并同时存在
   `device_*/end_info*.done` 和 `host/end_info.done`；
3. 从生成配置加载与采集一致的 CANN 9.0.0、ATB 和 venv，并核对   `profiler_info_0.json` 中的 `cann_version`；
4. 将已有的不完整 `ASCEND_PROFILER_OUTPUT` 和 parser `logs` 改名保留后再解析；
5. 同时要求 `kernel_details.csv`、`trace_view.json`、`communication.json` 非空，   并要求 `analyse.done` 存在，最后自动执行 `profile-summary`。

`analyse.done` 是“解析流程结束”标记，不是“所有 parser 成功”标记。只有`trace_view.json` 和 `analyse.done`、缺少 kernel/communication 的结果仍然失败。

`all_file.complete` 由离线 CANN export 阶段生成，不能作为执行 9.8 前的 raw 完整性门禁。若 `profile-analyse` 报 `Raw CANN capture is incomplete`，说明上述 device 或 host封口标志缺失；离线重跑 `analyse()` 无法补回未封口的数据。先保留该目录并检查：

```bash
source "$CFG/afd_graph_u2-decode.env"
find "$AFD_PROFILE_ATTENTION_DIR" -mindepth 1 -maxdepth 1 \
  -type d -name '*_ascend_pt' -print
PROFILE_ROOT=/replace/with/the/unique/current-session-ascend_pt
find "$PROFILE_ROOT" -type f -path '*/device_*/data/*' -size +0c -print | head
find "$PROFILE_ROOT" -type f \( -name 'end_info*.done' -o -name 'all_file.complete' \) -print
du -sh "$PROFILE_ROOT"
df -h "$PROFILE_ROOT"
grep -RniE 'ERROR|Failed|Traceback|ERR[0-9]+' "$PROFILE_ROOT/logs" | tail -n 100
```

当前 clean 实现若仍失败，在 A3-A 原样执行并回传以下输出；不需要先手工 `analyse()`：

```bash
source "$CFG/afd_graph_u2-decode.env"

grep -nE 'AFD NPU .*profiler|Incorrect schedule|Traceback|ERROR|ERR[0-9]+' \
  "$LOG_ROOT/attention.log" "$LOG_ROOT/ffn.log" | tail -n 500

find "$AFD_PROFILE_ATTENTION_DIR" "$AFD_PROFILE_FFN_DIR" -type f \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' | sort | tail -n 300
```

日志中的 `API stop request completed` 必须晚于 Attention local stop 和 engine collective；
FFN 必须出现 `received profiler stop payload` 及 `profiler stop payload completed`。每个启用
Profile 的 worker 还必须出现 stop synchronization completed、raw capture finalization
returned 以及 stop 后 raw 快照。若 API 已完成但某一侧没有对应 worker 日志，则是控制/RPC
链断开；若 worker stop 已返回但 marker 始终为 0，则转查相同 PID、相同时间窗口的 CANN
plog。

若没有非空 device 数据或任何 `end_info.done`，本轮 raw 不可恢复。保留失败目录，禁止安装
任何历史 overlay；让 A3-PF、A3-A 和执行 Proxy 的 shell 都设置一个新的相同运行根目录，在同一
clean commit 上从 9.1 重新开始：

```bash
export MATRIX_RUN_BASE=/data/z00569729/run/dsv4-mooncake-pd-graph-matrix-clean-retry1
```

### 步骤 9.10：A3-A 生成本轮 Profile 验收包

```bash
bash "$MATRIX" collect-final "$CFG" afd_graph_u2 decode
```

Profile 阶段不收 Proxy 或 Prefill 的 `collect-final`。

## 10. Profile 轮次 2：split A8F8

### 步骤 10.1：A3-PF 启动 Prefill 和 FFN

```bash
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

### 步骤 10.2：A3-A 启动 Attention

```bash
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a8f8 attention
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a8f8 attention
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a8f8 attention
```

### 步骤 10.3：A3-PF 检查 FFN 并启动 Proxy

```bash
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a8f8 proxy
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a8f8 proxy
```

### 步骤 10.4：两端手工开启 Profile

A3-PF 先登记 FFN Profile 窗口；该命令不访问 FFN HTTP，只校验 FFN 服务存活并写入本轮状态：

```bash
bash "$MATRIX" profile-start "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

A3-A 再启动 Attention profiler，并通过 AFD control payload 真正启动远端 FFN profiler：

```bash
bash "$MATRIX" profile-start "$CFG" afd_graph_u2_split_a8f8 attention
```

A3-PF 随后确认远端 FFN 已真正进入 CANN 采集态；该命令必须在 benchmark 前成功：

```bash
bash "$MATRIX" profile-check "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

### 步骤 10.5：A3-PF 执行 Profile 专用负载

```bash
bash "$MATRIX" benchmark "$CFG" afd_graph_u2_split_a8f8 profile
```

### 步骤 10.6：A3-A 收集 stage 证据

```bash
bash "$MATRIX" evidence "$CFG" afd_graph_u2_split_a8f8 attention
```

### 步骤 10.7：两端手工停止 Profile

A3-A 先触发 Attention，并通过控制面通知 FFN：

```bash
bash "$MATRIX" profile-stop "$CFG" afd_graph_u2_split_a8f8 attention
```

A3-PF 确认 FFN raw 已完成封口：

```bash
bash "$MATRIX" profile-stop "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

两条命令均成功后才继续。

### 步骤 10.8：按顺序停止本轮服务

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a8f8 proxy
```

A3-A：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a8f8 attention
```

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

### 步骤 10.9：两端分别生成并校验 trace_view.json

A3-PF 解析并校验 FFN：

```bash
bash "$MATRIX" profile-analyse "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

A3-A 解析并校验 Attention：

```bash
bash "$MATRIX" profile-analyse "$CFG" afd_graph_u2_split_a8f8 attention
```

若报原始 CANN capture 不完整，按 9.9 的诊断方式处理并重跑完整轮次 2。

### 步骤 10.10：两端分别生成 Profile 验收包

A3-PF：

```bash
bash "$MATRIX" collect-final "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

A3-A：

```bash
bash "$MATRIX" collect-final "$CFG" afd_graph_u2_split_a8f8 attention
```

## 11. Profile 轮次 3：split A16F8

### 步骤 11.1：A3-PF 启动 Prefill 和 FFN

```bash
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
```

### 步骤 11.2：A3-A 启动 Attention A16

```bash
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a16f8 attention
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a16f8 attention
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a16f8 attention
```

### 步骤 11.3：A3-PF 检查 FFN 并启动 Proxy

```bash
bash "$MATRIX" status "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
bash "$MATRIX" check "$CFG" afd_graph_u2_split_a16f8 proxy
bash "$MATRIX" start "$CFG" afd_graph_u2_split_a16f8 proxy
```

### 步骤 11.4：两端手工开启 Profile

A3-PF：

```bash
bash "$MATRIX" profile-start "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
```

A3-A：

```bash
bash "$MATRIX" profile-start "$CFG" afd_graph_u2_split_a16f8 attention
```

A3-PF 随后确认 FFN 已生成本轮 `*_ascend_pt/PROF_*` 原始根目录：

```bash
bash "$MATRIX" profile-check "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
```

### 步骤 11.5：A3-PF 执行 Profile 专用负载

```bash
bash "$MATRIX" benchmark "$CFG" afd_graph_u2_split_a16f8 profile
```

### 步骤 11.6：A3-A 收集 16-rank stage 证据

```bash
bash "$MATRIX" evidence "$CFG" afd_graph_u2_split_a16f8 attention
```

必须覆盖 `16/16` Attention rank。

### 步骤 11.7：两端手工停止 Profile

A3-A：

```bash
bash "$MATRIX" profile-stop "$CFG" afd_graph_u2_split_a16f8 attention
```

A3-PF：

```bash
bash "$MATRIX" profile-stop "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
```

两条命令均成功后才继续。

### 步骤 11.8：按顺序停止本轮服务

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a16f8 proxy
```

A3-A：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a16f8 attention
```

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
```

### 步骤 11.9：两端分别生成并校验 trace_view.json

A3-PF：

```bash
bash "$MATRIX" profile-analyse "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
```

A3-A：

```bash
bash "$MATRIX" profile-analyse "$CFG" afd_graph_u2_split_a16f8 attention
```

两条命令不能跨机器代执行。若报原始 CANN capture 不完整，按 9.9 诊断并重跑完整
轮次 3。

### 步骤 11.10：两端分别生成 Profile 验收包

A3-PF：

```bash
bash "$MATRIX" collect-final "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
```

A3-A：

```bash
bash "$MATRIX" collect-final "$CFG" afd_graph_u2_split_a16f8 attention
```

## 12. 失败时怎么做

`collect-final` 不会因为日志中存在 fatal marker 而拒绝打包。它会打印 warning，并把日志尾部和 `fatal-markers.txt` 一起放入归档。归档成功与业务验收通过是两件事。
下面的诊断 `collect` 用于其他命令失败或验收文件不完整的场景。

任何命令失败后：

1. 不执行下一条。
2. 不删除日志、Profile 或运行目录。
3. 先查看报错节点当前角色的 `status` 和当前日志软链接。
4. 将当前测试点按本节命令停止，再执行诊断 `collect`。

### 共置 A8F8 停止和诊断收集

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2 proxy || true
```

A3-A：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2 decode || true
```

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2 prefill || true
bash "$MATRIX" collect "$CFG" afd_graph_u2 proxy
bash "$MATRIX" collect "$CFG" afd_graph_u2 prefill
```

A3-A：

```bash
bash "$MATRIX" collect "$CFG" afd_graph_u2 decode
```

### split A8F8 停止和诊断收集

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a8f8 proxy || true
```

A3-A：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a8f8 attention || true
```

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a8f8 prefill_ffn || true
bash "$MATRIX" collect "$CFG" afd_graph_u2_split_a8f8 proxy
bash "$MATRIX" collect "$CFG" afd_graph_u2_split_a8f8 prefill_ffn
```

A3-A：

```bash
bash "$MATRIX" collect "$CFG" afd_graph_u2_split_a8f8 attention
```

### split A16F8 停止和诊断收集

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a16f8 proxy || true
```

A3-A：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a16f8 attention || true
```

A3-PF：

```bash
bash "$MATRIX" stop "$CFG" afd_graph_u2_split_a16f8 prefill_ffn || true
bash "$MATRIX" collect "$CFG" afd_graph_u2_split_a16f8 proxy
bash "$MATRIX" collect "$CFG" afd_graph_u2_split_a16f8 prefill_ffn
```

A3-A：

```bash
bash "$MATRIX" collect "$CFG" afd_graph_u2_split_a16f8 attention
```

注意：`afd_graph_u2-attention.env` 不存在是正常的。共置 `afd_graph_u2` 只有 `prefill/decode/proxy` 三个角色；只有两个 split 测试点具有 `attention` 角色。

## 13. 最终需要回传什么

回传以下信息：

1. 六个轮次每条 `collect-final` 输出的 `ARTIFACT` 和 `.sha256` 文件。
2. `comparison-af-ratio-p2-*.json`。
3. 三组 `performance_summary.json` 的绝对路径。
4. 共置点 Decode 的 `profile-summary.json`。
5. 两个 split 点各自 Attention 和 FFN 的 `profile-summary.json`。
6. 六个 Profile 原始根目录的绝对路径；完整 Profile 保留现场，不打进小包。

本指导书的**采集完整门禁**是三轮请求全部完成且没有 fatal；Profile 使用 CANN 9.0.0
解析、`with_stack=false`，并生成非空的 `kernel_details.csv`、`communication.json` 和
`trace_view.json`。通过这些条件只表示本轮性能/Profile 证据可供分析，不等于正式 P2 通过。

正式 P2 还必须同时满足：第 1 章的路径匹配 control/eager/Graph token-exact 正确性门禁；
正常停止、真实进程 rc、二次启动、取消恢复和 NPU cleanup 生命周期门禁；各比较点 CV
稳定性；预先固定的收益与尾延迟阈值；同总资源或明确的 active/reserved NPU 公平口径；以及
all-on/V1/off 的同代码消融。缺少任一项都不能创建性能 tag。

当前 `pd_graph_matrix.sh` 直接生成的是 all-on 配置，尚未自动生成 V1/off 的隔离配置与结果
目录。正式 P2 前必须先扩展该工具，或使用三个由 clean commit 生成、指纹明确且互不复用
状态目录的配置集；本指导书现有命令不能被解释为已经完成 all-on/V1/off 消融。

最终重点比较：

- split A16F8 相对 split A8F8 的绝对吞吐和 TPOT；
- FFN Bubble/Free 和长 receive 等待是否下降；
- recv/compute、compute/send 重叠是否增加；
- output tokens/s/NPU 是否改善；
- 性能增益是否超过两组运行的 CV 之和。

## 14. 2026-09-04 双 A3 实际验证结果

本轮实际使用 C32、128 prompts、输入 1024、输出 128、request rate infinite、seed 1024、
temperature 0、ignore EOS。固定栈为 CANN 9.0.0、vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、
vLLM-Ascend `3da28f9414583d2d0b672a8f06d1fae142404bda`、afd-plugin `2164240` 加
R14 overlay、Mooncake 0.3.9、torch 2.10 和 torch_npu 2.10.0.post2。五个 Graph/U2
开关均为 1，MTP 和 batch-invariant 均关闭。

### 14.1 功能与性能

三个点每轮均为 128/128 请求成功；Attention 在线 U2 stage 覆盖为 8/8、8/8、16/16。
双机均预留 32 张 NPU，active NPU 分别为 24、24、32。

这些是历史 R14 的运行观测，不是 Graph 输出正确性结果。本轮没有路径匹配 golden，且
R14 静态 AllToAllV split-cache 已因动态 expert routing 风险撤回；以下吞吐只能用于容量
方向和 Profile 机制分析，不能作为生产验收或性能 tag 依据。

| 点 | output token/s 原始三轮 | 均值 | CV | token/s/active NPU | TPOT P50/P90/P99 |
|---|---|---:|---:|---:|---:|
| 共置 A8F8 | 226.223/214.405/236.655 | 225.761 | 4.026% | 9.407 | 117.430/174.374/192.041 ms |
| split A8F8 | 176.784/231.413/249.349 | 219.182 | 14.080% | 9.133 | 131.892/148.103/156.636 ms |
| split A16F8 | 528.918/552.783/539.597 | 540.433 | 1.806% | 16.889 | 45.023/46.515/46.769 ms |

A16F8 的物理比例是 `A:F=2:1`，不是 4:1；U2 只使每个 FFN 收到
`2 Attention peers x 2 stages = 4` 个逻辑输入切片。A16F8 相对 split A8F8 的 output
throughput 为 `+146.568%`，token/s/active NPU 为 `+84.926%`，TPOT P50/P90/P99
分别下降 `65.864%/68.593%/70.141%`。这是强方向信号，但不是可发布的固定阈值收益：

- split A8F8 CV 为 14.080%，超过 10%；
- active NPU 同时从 24 增到 32；
- FFN `max_num_batched_tokens` 同时从 4096 增到 8192；
- 没有路径匹配的 no-AFD control，`acceptance_passed=null`。

因此不能把全部差值归因于 A:F=2:1、混合 DAG 或新增物理 stream。共置 A8F8 与 split
A8F8 的吞吐差 `-2.914%` 也小于波动口径，不能据此断言 placement penalty 为零。

### 14.2 双侧 Profile 与流水

六个 Profile 均由 CANN 9.0.0 采集和解析，`with_stack=false`，且生成非空的
`kernel_details.csv`、`communication.json` 和 `trace_view.json`。DP0 step-trace 为：

| 点/角色 | wall | Computing | Comm non-overlap | Comm overlap | Free | Bubble |
|---|---:|---:|---:|---:|---:|---:|
| 共置 A8 Attention | 109.544 s | 22.147 s | 67.747 s | 9.716 s | 19.650 s | 74.134 s |
| 共置 A8 FFN | 109.517 s | 39.650 s | 14.971 s | 8.053 s | 54.896 s | 22.155 s |
| split A8 Attention | 113.976 s | 21.506 s | 72.313 s | 8.145 s | 20.158 s | 74.588 s |
| split A8 FFN | 113.942 s | 14.772 s | 17.157 s | 5.033 s | 82.012 s | 20.696 s |
| split A16 Attention | 55.498 s | 18.756 s | 15.843 s | 3.585 s | 20.899 s | 18.752 s |
| split A16 FFN | 55.474 s | 16.124 s | 18.050 s | 3.520 s | 21.299 s | 20.500 s |

相对 split A8F8，A16F8 FFN Free 的绝对时间下降 `74.029%`，Bubble 绝对时间只下降
`0.944%`，non-overlap communication 绝对时间增加 `0.893 s`（`+5.205%`）。由于两组
wall 分别为 `113.942 s` 和 `55.474 s`，还必须同时看归一化占比：FFN Free/wall 从
`71.977%` 降到 `38.395%`（`-33.582 pp`），Bubble/wall 则从 `18.164%` 升到
`36.954%`（`+18.790 pp`，相对约 `+103.45%`）。因此主要改善是 FFN 更持续地获得输入、无任务空闲占比下降；
绝对 receive/sync Bubble 没有实质降低，其相对 wall 的负担反而上升，不能写成 Bubble 已消除。

最长 receive 的 197.892 ms 裁剪中，首个 FFN receive 等待 Attention producer 约
147.230 ms。数据到达后的 25.662 ms 活跃窗口内，FFN Computing 占 85.154%、Free
占 0.450%，约 60.756% 通信时间被计算覆盖；窗口中实际存在
`recv(S1) || compute(S0)`、`send(S0) || compute(S1)` 和跨 layer 的 send/recv/compute
重叠，说明混合 DAG 与多物理 stream 已生效。

A16F8 裁剪窗口主要出现 `MoeDistributeDispatchV2/CombineV2`（MC2），没有采样到
AllToAllV，因此该时间线仍能回答 MC2 窗口中的 stream overlap；它不能证明较大 gear
回退 AllToAllV 后的语义或性能正确。

该裁剪不能证明“单逻辑 recv stream 导致队头阻塞”：多个底层 `Notify_Wait` 已并行，且只
采到 Attention DP0 和 FFN DP0，没有第二个 Attention peer、多个 FFN rank、host enqueue
事件或可靠的跨机时钟同步。

### 14.3 生命周期状态与证据

请求、stage 和性能/Profile 观测有效，但 Graph 输出正确性与整套 F0-topology 均未通过。
14 个角色归档中
7 个有停机期 fatal marker，11 份日志尾部出现 `force killing remaining processes`；最终
`npu-smi` 没有残留进程。包内 `status.exitcode=1` 是服务停止后 health/status 的
NOT RUNNING，不是服务进程退出码；本轮没有独立记录真实进程 rc。

本机证据包：

```text
/mnt/workspace/log/201f96bcabb5446ea950ad241cd42202.zip
/mnt/workspace/log/91c03c33caa144539ef9438e7098e8be.zip
/mnt/workspace/log/29df8d29d6874b6899027810ac85072b.zip
```

六个完整 raw Profile 仍保留在两台采集机：

```text
/tmp/dsv4-pd-profile/dsv4-mooncake-pd-graph-matrix-r14-manual1/afd_graph_u2/attention/devserver-hps-e0117616-00030_484596_20260904094048901_ascend_pt
/tmp/dsv4-pd-profile/dsv4-mooncake-pd-graph-matrix-r14-manual1/afd_graph_u2/ffn/devserver-hps-e0117616-00030_483551_20260904094048903_ascend_pt
/tmp/dsv4-pd-profile/dsv4-mooncake-pd-graph-matrix-r14-manual1/afd_graph_u2_split_a8f8/attention/devserver-hps-e0117616-00030_510614_20260904102510658_ascend_pt
/tmp/dsv4-pd-profile/dsv4-mooncake-pd-graph-matrix-r14-manual1/afd_graph_u2_split_a8f8/ffn/devserver-hps-e0117616-00037_254102_20260904102510650_ascend_pt
/tmp/dsv4-pd-profile/dsv4-mooncake-pd-graph-matrix-r14-manual1/afd_graph_u2_split_a16f8/attention/devserver-hps-e0117616-00030_525438_20260904105542729_ascend_pt
/tmp/dsv4-pd-profile/dsv4-mooncake-pd-graph-matrix-r14-manual1/afd_graph_u2_split_a16f8/ffn/devserver-hps-e0117616-00037_275065_20260904105542730_ascend_pt
```

返回的 14 个角色 tar 未附各自外层 `.sha256`，且性能包不能独立证明 batch 1/8/32、取消/
恢复 smoke。最终交付前还要先修复 FFN Graph 动态路由，再保存这两类证据、修复优雅停止
并验证二次启动。

## 15. 本机方向性验证记录

在 all-on commit `2164240b31efc8605bf84cc45afc628996669554`、CANN 9.0.0 下，A1F1、A2F1、A4F1 均完成 Graph/U2 两 stage、两个动态 step、三物理流 capture 和 100 次 replay。

| 拓扑 | FFN compute-recv 重叠 replay | 重叠/compute | recv/replay | replay window |
| --- | ---: | ---: | ---: | ---: |
| A1F1 | 88/98 | 74.08% | 104.896 us | 10.923 ms |
| A2F1 | 98/98 | 99.56% | 159.777 us | 17.035 ms |
| A4F1 | 85/98 | 59.22% | 487.469 us | 20.472 ms |

证据目录：

```text
/mnt/workspace/validation/dsv4_afd_v023_cann900_graph_u2_multistream_ratio_a1f1_msprof_20260902
/mnt/workspace/validation/dsv4_afd_v023_cann900_graph_u2_multistream_ratio_a2f1_msprof_20260902
/mnt/workspace/validation/dsv4_afd_v023_cann900_graph_u2_multistream_a4f1_msprof_20260901_allon
```

该组件的 FFN compute 是 16-wide Add，不代表真实 MoE。最终结论必须来自本指导书的双 A3 split A8F8/A16F8 全模型对照，不能用本机组件 trace 外推。
