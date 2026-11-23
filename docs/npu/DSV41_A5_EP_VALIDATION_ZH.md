# DSV4.1 A5：TP1 / DP4 / EP4 验证

2026-11-20。参考锁定 DSV4 供体 `4ddd69a3`，当前完整权重资格使用以下拓扑。

| 部署 | Attention | FFN | 总卡数 |
| --- | --- | --- | --- |
| 原生对照 | 完整原生模型，TP1 / DP4 / EP4 | 无独立 F | 4 |
| AFD | A4，TP1 / DP4；保留 HC、Engram、KV/index | F4，TP1 / DP4 / EP4，原生完整 MoE | 8 |

AFD 两侧分别初始化各自的原生 DP/EP group；A/F v3 HCCL group 独立。
A 侧没有本地 routed experts，但保留相同的原生并行配置及 DP 调度。
原生 MoE 决定 F 的专家分片、gate FP32、shared experts 和 EP 通信；
AFD 只传 BF16 hidden、FP32 router input、原始 IDs，并回传完整 MoE 输出。
每个 F 从同一 parent plan 推导完整 DP token 向量。局部无有效 token 的 rank
仍计算传输 padding 并参与原生 EP；只有全局空 stage 可以整体跳过。
同一原生 DP wave 可以混合 target 和 idle dummy；全局 plan 选择 target，
各 A 保留自己的有效计数和 dummy 状态，A0 空闲也可协调；profile/warmup 仍严格一致。

EP 资格入口支持 eager/U1、Graph/U1、串行 eager/U2 和串行 Graph/U2。
Graph 使用 `FULL_DECODE_ONLY`。U2 使用原生 DBO，A/F 按 stage 串行推进，
对应 `launch-plan.json` 中 `stream_overlap=false`。各档可用 `N=0`（关闭 DSpark）、
`N=1` 或 `N=2`，草稿只在 A 侧以 eager 运行，关闭 adaptive verification。
PD 的 P 固定 eager/U1，D 使用上述四档。SP、EPLB 和 elastic EP 关闭。
A5 权重载入、真实 HCCL/EP、数值、连续请求和正常退出待本页验收。

## 准备和配置检查

使用含本次改动的插件 checkout，以及
[完整权重指导](DSV41_A5_FULL_CHECKPOINT_VALIDATION_ZH.md) 的既有 Python、vendor、
插件注册和源码预检步骤。源码锁仍为 vLLM `ced6857a` / Ascend `7e37e29a`。
本次不安装或切换上游依赖。使用原始 checkpoint，不裁剪专家或删除 Engram。

```bash
DSV41_PY=/usr/local/python3.11.10/bin/python
DSV41_MODEL=/home/models/DeepSeek-V4.1-Flash
DSV41_VENDOR=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer

"$DSV41_PY" -B tools/diagnostics/check_dsv41_full_checkpoint_config.py \
  --expert-parallel --model "$DSV41_MODEL" --output ep-config.json
```

配置检查实际构造原生和 A/F EngineArgs，禁止 NPU 初始化，不加载权重。
它验证 TP1、DP4、EP、原始量化和 CPU Engram 配置，不代表设备验收。
A 侧使用原生 `cpu_offload=true, dp_shared_memory=true, embedding_across_dp=false`；
F 在 native worker 构造前隔离 Engram。需要足够 host RAM 和原生 DP 共享内存容量；
专家分片约为四分之一的估算不替代完整 HBM 峰值实测。

## 原生 TP1 对照和 AFD 四档

按 `N=0`、`N=1`、`N=2` 依次执行。每个 N 在空闲节点生成一次原生
eager/U1 对照，供同 N 的 standalone AFD 和 PD D 复用。去掉 `--run` 可先审阅
`launch-plan.json`。原生使用普通 NPUWorker、TP1 的 64 query heads 和相同的
checkpoint、CPU Engram 与 `flashinfer_all2allv` 配置。

确认计划的 `request_suite=ep-rank-matrix-v2`。第一波 4 路并发向每个 DP rank
发一个请求，执行一次取消后，第二波 8 路并发向每 rank 发两个请求。最后单独向
DP1 发请求，覆盖 A0 空闲的路径。两轮各自使用相同 prompt，每个请求保留
`request-routing.json`。可另用 `--repeatability` 为 `N=0` 生成 eager/U1 或
Graph/U1 的串行重复性结果，原生和 AFD 均需带该参数。

```bash
N=0
"$DSV41_PY" -B tools/diagnostics/run_dsv41_full_checkpoint.py \
  --model "$DSV41_MODEL" --transformer-vendor "$DSV41_VENDOR" \
  --expert-parallel --native-control --spec-tokens "$N" \
  --attention-devices 0,1,2,3 --output-root "results/ep-native-n$N" --run
```

确认原生 `summary.json` 的 `request_passed` 为 true，填入
实际输出目录。`MODE` 依次取 `eager-u1`、`graph-u1`、`eager-u2`、`graph-u2`，
每次等上一个实例退出再启动。N 改变时使用对应的原生结果。

```bash
DSV41_EP_REFERENCE=/absolute/path/to/matching/native/result
MODE=eager-u1
"$DSV41_PY" -B tools/diagnostics/run_dsv41_full_checkpoint.py \
  --model "$DSV41_MODEL" --transformer-vendor "$DSV41_VENDOR" \
  --expert-parallel --mode "$MODE" --spec-tokens "$N" \
  --reference-run "$DSV41_EP_REFERENCE" \
  --attention-devices 0,1,2,3 --ffn-devices 4,5,6,7 \
  --output-root "results/ep-afd-$MODE-n$N" --run
```

如果原生矩阵已完成且进程退出超时，可直接复用已有归档做逐请求 token 诊断。
入口核对退出超时、清理后无残留进程、原生所用设备空闲、
checkpoint config/index 摘要以及 13 条原始响应。AFD 的 `summary.json`
通过 `native_reference.diagnostic_only=true` 记录该对照的范围，
并保留其 `clean_shutdown=false`。正式 EP 验收仍需正常退出的原生对照。

## 主机证据与 A5 判定

- 每档回传 `launch-plan.json`、预检、A/F 日志、`summary.json`、`request-matrix.json`
  和各请求的 `token-comparison.json`。确认 `negotiated.expert_parallel=true`、
  EP world=4、TP1、F 的专家分片、
  空闲 rank 参与、取消后请求成功、所有 A/F 进程正常退出以及八卡无残留进程。
- eager/U2 在 `summary.json` 中确认 `eager_u2_verification.passed=true`。核对
  请求矩阵期间同一事务的 A/F 各 rank 两阶段完成记录，以及 A 侧父输出合并。
- 定向矩阵的 Graph 档在 `summary.json` 中确认 `graph_verification.passed=true`，
  日志包含对应阶段数的抓图和全 rank 回放返回。结合定向请求、输出 token 和单路
  eager 事务确认回放及回退。仅有 eager 时，Graph 档仍缺回放证据。
- DSpark 的 target 校验宽度为 `N+1`，原生草稿复用同一批的 DP 调度结果。
  `request-matrix.json` 的 `acceptance_and_rejection_observed` 与
  `rejection_continuation_observed` 应均为 true，并核对首拒后继续解码的请求证据。

以上检查覆盖所列资格场景。完整模型精度和性能另行实测，官方 DP8/TP1/EP8
另有部署规模。

## EP + PD

P 和 D 各使用一个 A4/F4、TP1/DP4/EP4 实例。每个实例的 A/F 在同一台 A5，
两台机器上的设备号可以相同。P 固定 eager/U1，D 依次运行四档。每组 P、D 和
原生对照使用同一个 checkpoint 和相同的 `N`。从 `N=0` 开始，再分别验证 DSpark
`N=1` 和 `N=2`。复用前文同 N 的 `ep-rank-matrix-v2` 原生对照。

在 P 节点把 `DSV41_EP_P_IP` 设为 D 能访问的本机地址，再指定一个启动前
不存在的停止文件。P 示例使用默认 HTTP 18031、A/F 29600–29603 和 KV
握手 14579–14582。Attention 进程使用 `--kv-ip` 设置原生 Mooncake 的
`VLLM_HOST_IP`。Transfer Engine 实际 RPC 地址和端口随握手返回，也须能从
D 节点访问。

```bash
N=0
DSV41_EP_P_IP=192.0.2.10 # 换成 P 节点的实际地址
DSV41_EP_P_STOP="/absolute/path/to/ep-pd-n$N-finished.flag"
"$DSV41_PY" -B tools/diagnostics/run_dsv41_full_checkpoint.py \
  --model "$DSV41_MODEL" --transformer-vendor "$DSV41_VENDOR" \
  --expert-parallel --pd-role producer --mode eager-u1 --spec-tokens "$N" \
  --engine-id dsv41-ep-prefill --bind-host 0.0.0.0 --kv-ip "$DSV41_EP_P_IP" \
  --attention-devices 0,1,2,3 --ffn-devices 4,5,6,7 \
  --producer-stop-file "$DSV41_EP_P_STOP" --output-root "results/ep-pd-p-n$N" --run
```

P 打印 `P ready` 后，从 D 节点运行。示例给 D 偏移 1000，D 的 HTTP、
A/F 和 KV 预检端口分别为 19031、30600–30603 和 15579–15582。
如果 P/D 在同一台机器，使用互不重叠的两组八卡以及不冲突的端口偏移。
调整 P 偏移时，`--prefill-url` 中的端口随之改变。

```bash
N=0
DSV41_EP_P_IP=192.0.2.10 # 换成 P 节点的实际地址
DSV41_EP_D_IP=192.0.2.11 # 换成 D 节点的实际地址
DSV41_EP_REFERENCE=/absolute/path/to/matching/native/result
MODE=eager-u1
"$DSV41_PY" -B tools/diagnostics/run_dsv41_full_checkpoint.py \
  --model "$DSV41_MODEL" --transformer-vendor "$DSV41_VENDOR" \
  --expert-parallel --pd-role consumer --mode "$MODE" --spec-tokens "$N" \
  --engine-id dsv41-ep-decode --kv-ip "$DSV41_EP_D_IP" --port-offset 1000 \
  --attention-devices 0,1,2,3 --ffn-devices 4,5,6,7 \
  --prefill-url "http://$DSV41_EP_P_IP:18031" --reference-run "$DSV41_EP_REFERENCE" \
  --output-root "results/ep-pd-d-$MODE-n$N" --run
```

保留同一个 P，等待 D 当前档退出后，将 `MODE` 依次改为 `graph-u1`、`eager-u2`、
`graph-u2` 运行。D 四档都退出后，在 P 节点执行 `touch "$DSV41_EP_P_STOP"`，
让 P 正常清理。切换 N 时用对应的停止文件与原生对照重新启动 P、D。

对照每个 D 请求的 `request-routing.json`、`prefill-routing.json` 和
`prefill-response.json`。4+8 两波都将 D rank `r` 配到 P rank
`(r + 1) % 4`。最终单路请求从 P rank2 向 D rank1 拉取 KV。
检查 P 返回 `remote_port=14579+P rank`（使用偏移时加上 P 偏移）和完整
`afd_engram_history_v1`。回传 D 侧 `KV cache transfer for request` 日志、
`token-comparison.json` 和 P、D 两侧均正常退出的 `summary.json`。DSpark 档同时保留
P 响应的 `afd_dspark_source_v1`（含源请求和 generation）以及 D 的
`request-matrix.json`（含每步草稿提议、接受和拒绝后继续解码记录）。标准启动器
通过读取与正常排空检查完成请求。若另行运行记录 typed Ready、草稿分组读取和
DONE/ACK 的探针，同时回传其报告。D 的各档同时执行上一节对应的 U2、Graph
和 DSpark 证据检查。
