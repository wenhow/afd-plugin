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

## 2026-11-23 A5 eager/U1 结果与下一轮

真实权重、TP1/DP4/EP4、A4/F4、DSpark N=0。原生和 AFD 按 DP0–3
逐路预热，再串行执行首波。4 条预热和 13 条正式请求的输出 token、
logprobs 全部与原生一致，实际 DP rank 和逐步调度 token 数也一致。
此前分叉的 DP2 第 3 个 token 两侧均为 `3107`。流式取消后的 8 路并发和
最后的 DP1 单路通过。冷启动并发首波的数值一致性仍待确认。

原生和 AFD 的正常停机仍待验收。`SIGTERM` 后，服务主进程仍在等待子进程，
EngineCore 仍在等待请求。AFD 在强制清理阶段完成全 rank STOP/ACK，
清理后八卡空闲、无残留进程。EP6 继续保留待验状态。

本轮回传包括 AFD `e4c76132c6104a349c518fcf13f2020c.zip`、原生
`9bd053434a6e4bc6830cd254f22e30b0.gz` 和日志摘录
`e93a4ece8a0c4254a7d975624f139102.zip`。

### Graph/U1 N=0 功能预验证

确认上一实例已退出、0–7 卡空闲且启动端口已释放后，可以先采集
`graph-u1` 的设备证据。执行时需传入标准并发矩阵的原生 `N=0` 结果。
以下路径为此前使用过的对照目录示例，请替换为实际路径。
观察 `FULL_DECODE_ONLY` 捕获、A/F 全 rank 回放返回、eager 回退和取消后请求。
核对 `summary.json` 的 `graph_verification.passed`、`request_passed`
和 `clean_shutdown`，并保留 A/F 日志及进程清理结果。

```bash
DSV41_EP_REFERENCE=results/ep-native-n0/full-eager-u1-4v59wo3p
"$DSV41_PY" -B tools/diagnostics/run_dsv41_full_checkpoint.py \
  --model "$DSV41_MODEL" --transformer-vendor "$DSV41_VENDOR" \
  --expert-parallel --mode graph-u1 --spec-tokens 0 \
  --reference-run "$DSV41_EP_REFERENCE" \
  --attention-devices 0,1,2,3 --ffn-devices 4,5,6,7 \
  --output-root results/ep-afd-graph-u1-n0 --run
```

原生退出超时后的标准并发对照，可在验证完整请求矩阵后用于诊断。若其归档
仍标记进程残留或设备占用，独立 EP Graph/U1 入口会按 PID 和启动时刻
复查原生进程，并检查本轮 A/F 设备是否全部空闲。复查结果保存到
`native-reference-recheck.json`，原生 `clean_shutdown=false` 和退出码照原值记录。
请求完成后先核对 Graph 日志。停机前证据齐全时会写出 `graph-verification.json`，
若当时仍有事务日志在写入，则在归档阶段复核。功能以 `request_passed` 和
`graph_verification.passed` 判定。正常停机以 `clean_shutdown` 判定，
整体退出码仍会反映清理失败。
独立 EP Graph/U1 N=0 请求超时时，入口会先请求已注册的 Attention worker
输出 Python 栈，再发送退出信号。回传 `request-timeout-stacks/`、
`lifecycle.json` 和 A/F 日志。按 `transaction_id` 与 `peer_world_rank`
检查 `GRAPH_QUERY` 和 `GRAPH_REPORT` 的开始及返回记录，并核对 A 侧
`AFD HCCL graph execute` 的有效 token 数和 dummy 状态。
若首波有 DP rank 提前完成，零 token 的请求清理先由原生 runner 返回，
随后由 EngineCore 的 dummy step 参加同一轮 DP 协商。复测时核对该 rank
从 `valid_tokens=0 dummy_run=False` 进入 `mode=dummy`，以及其余请求继续返回。
独立 EP Graph/U1 N=0 会在有 token 差异时继续收集完整的 4+8+1 矩阵和
取消证据，保留 `request_passed=false`。正式数值对照需使用同 checkpoint、
同 N、同请求顺序与批形状的原生结果。
若 `SIGINT` 停机超时，可在上述命令加 `--attention-stop-signal SIGUSR2`，
由 Attention 根进程调用原生多 API server 退出回调。停机超时会在
`lifecycle.json` 记录 signal 状态，并把根进程和 worker 的栈追加到
`request-timeout-stacks/`。两个抓栈阶段分别记录在生命周期中。
EP7.2 在 EP6 收口后按本页验收项判定。

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

## eager/U1 N=0 定向诊断

首波请求出现 token 分叉或停机超时时，原生和 AFD 命令均加
`--ep-diagnostics`，使用独立输出目录并将 AFD 指向此次原生结果。
拓扑保持 TP1/DP4/EP4，继续运行原有 4+8+1 矩阵。纯 token 差异会记录在
`token-comparison.json`，其余请求继续执行，最终 `request_passed=false`。

若原生矩阵已完成，但归档在强制清理后仍记录进程或设备占用，使用同一
A5 容器执行 AFD 的 `--ep-diagnostics --reference-run`。入口会重新核对
归档中的 PID 与启动时刻，并检查本轮 A/F 全部设备。两项通过即可复用原生
响应。原始归档保留，`native-reference-recheck.json` 记录复核结果，
原生退出状态仍为 `clean_shutdown=false`。

- 在 `attention-server.log` 查找 `DSV41 EP request admitted`。用响应
  `id` 匹配 EngineCore 日志中带后缀的请求 ID，并核对记录的 `dp_rank`。
- 检查两个结果目录中响应的 `choices[0].logprobs.top_logprobs`。
  按 `first_output_mismatch.index` 取同一步的原生和 AFD top-5 候选，
  核对 top-1 与 top-2 的分数差。首个分歧之后的生成前缀已不同。
- `lifecycle.json` 的 `attention_signal_state` 保留 A 根进程的信号掩码和
  pending 状态。结合 A/F 日志中的 `DSV41 EP STOP` 记录定位进入停机、
  STOP 广播、各 peer 应答和最终确认。原生侧只检查信号和进程退出。
  A 根进程超时后，入口通过 pidfd 请求 A 根进程和已注册的 A EngineCore
  打印 Python 栈。回传 `shutdown-stacks/` 与 `lifecycle.json`。
  若内核不支持 pidfd 或进程已退出，生命周期记录对应原因。

若首波原生和 AFD 入队时机不同，在两边增加 `--ep-serial-first-wave`。
先用首波同一个 prompt 按 DP0 到 DP3 每次发一条预热，再按相同顺序
串行测四条请求。预热响应存于 `first-wave-warmup/`。对照使用本轮新原生
结果，并在 A 日志中查 `DSV41 EP scheduled tokens` 核对实测请求每步的
调度 token 数。第二波仍正常并发，可继续看取消后的请求与最终单路。
串行首波用于数值定位，正式并发矩阵需按原方案验证。

若 `SIGINT` 超时、而强制清理阶段已有全 rank STOP ACK，可复用已有的
原生诊断结果，在 AFD 命令增加 `--attention-stop-signal SIGTERM`。此选项
仅改变正常退出阶段发给 Attention 根进程的第一个信号。检查
`lifecycle.json` 是否在 `forced_cleanup` 之前出现 A 和 F 的退出记录，
以及 A/F 日志中是否有完整 STOP/ACK。即使请求矩阵有 token 分叉，也可
独立查看 `summary.json` 的 `clean_shutdown` 和 A/F 退出码。

若 `SIGINT` 和 `SIGTERM` 都未让根进程退出等待循环，使用
`--ep-diagnostics --ep-shutdown-smoke --attention-stop-signal SIGUSR2`。
这个模式只向 DP1 发一条请求，然后由 A 根进程的诊断处理器调用 vLLM
原有的多 API server 退出回调。该回调从正在运行的 `run_multi_api_server`
调用帧中取出。服务日志应出现
`DSV41 EP shutdown: invoking native vLLM signal handler from active serve frame`，再检查
STOP/ACK、F 自行退出和 A/F 退出码。此模式不传 `--reference-run`，
仅用于停机验证。

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
