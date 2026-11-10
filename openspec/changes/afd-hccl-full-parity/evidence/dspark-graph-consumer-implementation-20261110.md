# DSpark target Graph：q 传播与消费者候选

日期：2026-11-10。已完成软件候选冻结与实际依赖 CPU 回归：首轮九模块
551 项通过、4 项 fixture 失败，修复后完整 coordinator 模块 100 项通过。
两次是独立快照，不合并成一次全通过；本文不记录 NPU 或完整 DSpark 验收。
生产总开关保持关闭。

## 候选身份与边界

本批 8 个生产文件、13 个镜像/共享 fake 的原始冻结身份见
[21-file candidate](dspark-graph-consumer-candidate-20261110.json)，其 SHA256 为
`191a265f7a19e30425edcb848d522eb2ed8644e27e4cf36f2613226b1843ee06`。
其中 coordinator 镜像和共享 fake 随后交给独立审查 agent，仅追加原生外层
target/draft 捕获隔离测试；实际 CPU 快照使用最终文件哈希，保留此候选清单。

第一阶段已把原始、经过验证的 V4.1 DSpark `N+1` 固定为不可变
`DeepseekV41TargetShape`，在 F 清除 speculative/Engram 配置前取得，并显式传给
同步 FFN runner 和原生 MC2 容量计算。本批消费该形状；不恢复 F drafter，
不修改采样、接受/拒绝或请求/KV/Engram 所有权。

并行组件的所有权分别为：协议 agent 负责 `HcclPlanCapabilities`、HELLO 与
transport key 校验；root 负责固定 DSpark 准入和 UBatch runner 准入；本候选
负责 Graph key、原生切片投影、A 控制/协调器、F 服务和显式形状绑定。

## 目标宽度的传递

`target_query_len` 默认 1。协议的 q1 HELLO 省略该字段并保留旧编码；q>1
要求 Graph/counts v1、A/F 与 native TP 配对且为 TP1/2，在数据组建立前严格
比较所有 peer 的 q，不做降级。完整协议约束和独立证据见
[protocol contract](dspark-target-query-protocol-contract-20261110.md)。

| 消费者 | 输入和可观察约束 |
| --- | --- |
| `HcclGraphKey` | uniform q 等于 max query q，`T=R*q`，LoRA 仍为 0；现有 wire 字段布局不变 |
| `hccl_v41_graph` | capture 用 R 个 q 构建 CPU counts；实际 preview 使用原生 gather 的 ordered counts；固定均匀 q 才可选 FULL |
| `HcclV41GraphControl` | 从协商 caps 固定 q；native descriptor 的 maxq=None 规范化为 session q；预览、捕获和 key 必须一致 |
| A coordinator | 原始配置 helper、native `owner.decode_query_len` 与协商 q 相等；运行前再次核对原始形状未变化 |
| F graph service | 私有 F 配置仍无 spec/Engram；只从显式 `runner.target_shape` 恢复目标宽度，None 仅代表 q1 |
| stage bank | key 的 uniform/maxq 必须等于 caps q，防止直接绕开 coordinator 的公共 bank 分配入口 |
| F worker/runner bind | 三处现有 binding 显式传 `target_shape=self.target_shape`，共享 policy 不需要重新读取已清除的 spec |

Graph 准入使用 root 的 `validate_deepseek_v41_graph_speculation`：仅固定 DSpark、
draft eager、非 adaptive、无 PD。原生普通非 Graph 路径仍协商 q1，即使目标模型
保留了 q>1 形状。正常 connector factory 的默认调用也没有因本批而开放 q>1
生产路由。

## U2 不要求请求对齐

U2 沿原生 padded token midpoint 切分，允许同一 request 出现在两个 stage。
不能添加 `child_rows % q == 0` 或“每个 child 拥有完整 request”的限制。
锁定原生 `gpu/ubatch_utils.py::create_ubatch_slices` 还会把尾部 request start
夹到最后一个实际 request，因此实际短 batch 是否命中捕获 key，必须比较完整
request/token 几何，而不能仅比较真实 token 数。

以下 q3、捕获 R3/T9/U2 的结果由锁定源码推导；这两组 padded-short 输入尚未
作为独立实际依赖 CPU 小案例运行，不把表格作为运行证据：

| 实际输入，均 padded 到 R3/T9 | stage 0（request；token） | stage 1（request；token） | 与捕获几何 |
| --- | --- | --- | --- |
| R3/T9，counts=[3,3,3] | `[0,2); [0,4)` | `[1,3); [4,9)` | 相同 |
| R2/T6，counts=[3,3] | `[0,2); [0,4)` | `[1,3); [4,9)` | 相同，可合法 replay |
| R1/T3，counts=[3] | `[0,2); [0,4)` | `[0,3); [4,9)` | 不同，应重新选择 eager |

## 统一 CPU 回归

复用 root 的 [phase-two wrapper](run_a3_dspark_graph_cpu.py)，新 label
`geometry-001`，不创建另一套 identity/runner。快照锁住 21 个实际 installed
upstream source pin，在现有 A3 venv 运行以下九个完整镜像：

1. `tests/unit/distributed/test_hccl_graph.py`
2. `tests/unit/connectors/npu/test_hccl_graph_stage.py`
3. `tests/unit/v1/worker/npu/test_hccl_v41_graph.py`
4. `tests/unit/v1/worker/npu/test_hccl_v41_graph_control.py`
5. `tests/unit/v1/worker/npu/test_hccl_v41_graph_service.py`
6. `tests/unit/v1/worker/npu/test_hccl_v41_graph_coordinator.py`
7. `tests/unit/v1/worker/npu/test_hccl_v41_ffn_graph.py`
8. `tests/unit/v1/worker/npu/test_ffn_model_runner.py`
9. `tests/unit/v1/worker/npu/test_ffn_worker.py`

新 q2/q3 用例覆盖真实原生 CPU 切片、capture/dispatch 几何、可变长度 eager
选择、session/key/原始配置不一致拒绝、隔离 F 的 q 保留、warmup/capture/replay
与 STOP。共享 hccl fake 已改为传递真实 target shape，所以额外纳入 F runner、
worker 完整模块。coordinator 追加原生 outer capture 中 target/draft scope 隔离。
首次失败保留原始 XML、stdout、source manifest 与退出码，修复仅另开新 label。

本地已完成 21 文件 Ruff 和 Python 3.11 语法检查；纯 Graph key 镜像 65 项通过。
本地缺失运行依赖的 skip 不算 A3 实际依赖通过。实际 A3 两轮结果和全部 raw
哈希见 [geometry README](a3-dspark-graph-cpu-20261110/geometry-README.md) 及
[summary](a3-dspark-graph-cpu-20261110/geometry-validation-summary.json)。

`geometry-001` 的 4 项失败仅来自共享测试 helper：`dispatch_runtime` 将原生
第 5 个位置参数 `dp_size` 的 q1 常量误改成目标 q，导致 q2/q3 被真实 DP1
guard 正确拒绝。仅将该 fake 改为显式读取实际 parallel config 的 `dp_size`
和 `dp_rank`，没有改生产 guard。`geometry-002` 完整 coordinator 镜像 100 项
通过，包含 4 项 target/draft 隔离回归；其余 8 个完整镜像在首轮均已通过。
两轮 21 installed pins、完整 source bytes 均一致，8 个生产文件跨轮未修改。
并行 admission/protocol 回归仍保持独立结果和范围。

## 后续 q>1 组件 probe 的最小设计

此段仅供 root/架构 agent 审查；未修改既有 NPU probe，也未启动设备。

复用已经受监督的 native-manager/bank probe 和现有 venv，显式构造 q3 的
真实 base manager（`decode_query_len=3`）及同 q caps。先用 TP1 A/F 两设备，
沿原生 dummy `R=T/q` 和真实 `create_ubatch_slices` 建立 U1/U2 key。
warmup/capture valid=0，正式 replay 固定 key/地址、改变有效行，受控算术继续
逐层检查 BF16、独立 FP32、IDs、padding mask 和返回值；保留 native TP 前后
allreduce、STOP、空闲八卡前后样本、owned-PID 有界监督与首次失败。

Replay/miss 输入按上面的真实 request clamp 选择：R2/T6 对 R3/T9 可以命中；
R1/T3 的 request 几何变化才是明确的 shape-miss。其他 miss 仍使用实际 F cache
缺失或 session q 不匹配的协议负例，不能伪造“所有短输入都会 miss”。

该组件不需要 checkpoint、不构造 drafter、不调用真实 sampler；结果只能证明
q>1 的 Graph 几何/协商/受控 HCCL 与资源复用。完整 DSpark 的 proposed/accepted
tokens、拒绝回退、真实目标/草稿模型、HC/Engram/MoE、KV/history 和性能仍待
用户手工 A5 验证及后续完整模型入口。
