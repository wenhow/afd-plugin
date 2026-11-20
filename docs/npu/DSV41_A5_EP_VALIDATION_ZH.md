# DSV4.1 A5：TP1 / DP4 / EP4 验证

2026-11-20：参考锁定 DSV4 供体 `4ddd69a3`，当前插件已实现独立的
TP1 / DP / EP eager/U1 路径。首个完整权重资格配置为：

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

当前是代码和 CPU 合约已实现的资格候选。A5 权重载入、真实 HCCL/EP、数值、
连续请求和正常退出仍待本页步骤验收。不能沿用 EP 关闭时的硬件证据。
EP 首版限制为对称 A/F、TP1、eager/U1、SP/EPLB/elastic EP 关闭；
EP 与 Graph、U2/DBO、DSpark、PD 组合在建组前明确拒绝，保留后续任务。

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

## 原生 TP1 对照和 AFD

在所选卡空闲时依次执行。先去掉 `--run` 可审阅 `launch-plan.json`。
两次都保留相同请求协议、原始权重和 Engram；原生对照采用普通原生 NPUWorker，
TP1 为原生 64 query heads，不使用 TP4 的 16→64 适配或其专项 probe。
`flashinfer_all2allv` 是两侧一致的配置值，实际 Ascend MoE 通信实现以运行记录为准。

```bash
"$DSV41_PY" -B tools/diagnostics/run_dsv41_full_checkpoint.py \
  --model "$DSV41_MODEL" --transformer-vendor "$DSV41_VENDOR" \
  --expert-parallel --native-control --repeatability \
  --attention-devices 0,1,2,3 --output-root results/ep-native --run
```

确认原生目录 `summary.json` 的 `request_passed`、`clean_shutdown` 和
`repeatability.json` 的 `completed`、`passed` 均为 true，再填入打印的实际目录：

```bash
DSV41_EP_NATIVE=/absolute/path/to/results/ep-native/full-eager-u1-XXXXXXXX

"$DSV41_PY" -B tools/diagnostics/run_dsv41_full_checkpoint.py \
  --model "$DSV41_MODEL" --transformer-vendor "$DSV41_VENDOR" \
  --expert-parallel --repeatability --reference-run "$DSV41_EP_NATIVE" \
  --attention-devices 0,1,2,3 --ffn-devices 4,5,6,7 \
  --output-root results/ep-afd --run
```

参考准入检查 TP、DP、EP 和 head adapter 声明，拒绝混用 TP4 或 EP-off 的结果。
保留现有预检、超时、进程监督及输出归档。F launcher 等待全部本地 EP engines
完成 STOP 清理；首个正常退出不会立即终止其余 engines，失败和清理超时仍退出非零。

## 回传与判定

回传两次打印目录的归档，包含 launch plan、preflight、各角色日志、summary、
repeatability 和 token comparison；保留错误 traceback 及正常退出记录。
AFD 启动记录应包含 `negotiated.expert_parallel=true`、TP1、A/F size=4、U1。
需确认 F 每 rank 的实际专家分片、各角色 EP world=4、无有效输入 rank 的持续参与、
重复/取消后请求、输入输出 token 对照以及退出后八卡无残留进程。

当前未宣称完整模型等价或性能通过；失败按此版本的源码和实际报告定位。
官方 DP8/TP1/EP8 另有部署规模；本页 DP4 是与 AFD F4 对齐的受控对照。
