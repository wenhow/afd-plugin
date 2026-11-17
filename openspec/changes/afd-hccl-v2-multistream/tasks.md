# HCCL eager/Graph U2 多流与跨层流水：任务

全部为新增待完成项；只有实现、审查和验证证据齐全后勾选。任务状态只维护在本文件。

## 当前排期（2026-11-12）

遵循[总计划优先级](../../ROADMAP_ZH.md#engram-priority)：先完成三流、跨层流水、
buffer lifetime、异常 join、开关与独立组件 timeline；开发依赖 C4/C5 对应的非 Engram
组件。Engram producer/consumer 的流、event 和复用专项适配后置至其他 DSV4.1 功能
具备后的 [C7 2.3–2.4](../afd-hccl-full-parity/tasks.md#engram-medium)。完整模型重叠和
性能仍待该依赖及原有设备验收，不因延期提前放开生产开关；已有勾选及证据保留。

## 1. 实施与验收

- [x] 1.1 绘制 eager/Graph 的 A/F 数据和事件依赖图，映射 D01/D02/D04 与 C4 continuation（H05/H09/H10）。
- [x] 1.2 审查 lifetime、record_stream、event/缓冲复用、parent-stream 捕获和最终 join。
- [x] 1.3 实现 eager/U2 A 端收发流及延迟 continuation 等待，包含独立 FP32/IDs ready 依赖。
- [x] 1.4 实现 eager/U2 F 端 recv/compute/send 流，保持 stage/peer 消息顺序。
- [x] 1.5 实现 Graph compute fork/join、per-stage hybrid DAG，保持 warmup/capture parent-stream 约束。
- [x] 1.6 实现 Attention 三流计划与 F 接收流/跨层流水，图内事件及最终输出 join 完整。
- [x] 1.7 恢复六个 overlap 开关、0/1/default、模式隔离与 CROSS_LAYER→RECV_STREAM 依赖。
- [x] 1.8 恢复 stage diagnostics 与 FFN compute sync 定位能力，记录所有 rank 有效配置，诊断默认关闭。
- [x] 1.9 补充可观察依赖/输出、all-off、非法开关、event/事务复用和异常退出回归。
- [ ] 1.10 【高/待中】先对比独立 NPU 组件的 eager/U2 与 Graph/U2 开关组合；完整模型多轮结果及 Engram 流依赖待 C7 2.3–2.4。
- [ ] 1.11 【高/待中】先验证组件长运行、bucket/请求切换、allocator、空 stage 和退出；含 Engram buffer/event 的完整模型压力及生命周期验收后置。
- [ ] 1.12 【高/待中】先采集组件 A/F timeline 定位重叠与阻塞；完整模型吞吐/延迟/HBM/实际卡数及每卡效率在 Engram 和其他功能验收后收口，不用组件数字替代。

2026-11-10：1.2/1.5 的实现和审查见 `evidence/` 下 pipeline、consumer、
parent-stream 契约。实际依赖 CPU 两轮分别 130/240 项通过，父流 TLS 后续
定向 122 项通过；本地配置 seam 77 项通过。A3 的 hybrid 1/0 独立会话各通过
4 次 warmup、4 次 capture、12 次 replay 和 6 次 eager fallback，固定 key
覆盖 full/1/T-1 有效 token。它们验证受控算术组件及实际 stream/event/HCCL，
不等于真实 V4.1 Attention/Engram/MoE/full-checkpoint 或性能验收。
后续独立快照：五轴配置/消费者 460 项通过；eager A driver 38 项、
native session/runner 119 项、F eager 消费者 43 项、最终 owner 50 项均有
各自 CPU 证据。worker/factory 修复后的 014 为 114 项通过，身份边界与新增
签名的 015 为 134 项通过；历史失败/fixture 错误均保留，不拼接成一次总通过。
A3 五轴 11111/00000/10000/10010 四个独立 Graph 会话通过；实际 eager
ticket owner 的 TP1/TP2 各六个连续 parent 通过，包含空 stage、padding-only、
双层复用和正常 STOP。原生 A/F 消费者仍须完整 checkpoint 验收；这些组件
不证明 native HC/Engram/MoE、长运行 allocator 压力、实际重叠或性能。
以上为此前增量的边界；后续诊断和跨层调度补齐见下。Graph 物理流约束及
1.6 的最终实现审查见 2026-11-13 记录；不得把循环或开关称为已验证通信/计算重叠。

2026-11-10 后续：1.8 已补齐 eager serial/ticket、pre-KV、profile/dummy、
空 parent 和 Graph eager fallback 的 rank/transaction/actual stream 诊断，
保持默认关闭，FFN 显式同步限 live serial U1。018b/020 的独立 CPU 结果为
512/247；021 的 pre-KV49、coordinator81、UB74 全部通过，Execution 旧
profiler 夹具问题在 022 的完整36项关闭；022 的两个 A 镜像共享夹具问题
在 023 的完整23项关闭，原失败均保留。A3 ticket-owner TP1/TP2 每 rank
六个 parent 的真实诊断通过。这些是主机可观察协议和受控 NPU 组件证据，
不代表原生完整模型、Graph replay 内 Python 日志、物理流重叠或性能验收。
bulk_v1 当时已有独立 TP1 capture/replay 组件证据；当时 1.6/1.10–1.12 尚未收口。

2026-11-10 跨层增量：共享静态 cross_layer_v1 已供 A/F Graph 消费者使用，
双层/三层 CPU 的原始025/026分别保留失败与定向闭合；单次A3三层TP1组件
完成每rank 26事务（4 warmup、4 capture、12 replay、6 eager fallback），
31源码校验一致、8卡前后空闲、无超时/强杀。F 输出先复制到固定stage bank，
A 保持原生 continuation 后产生下一层输入。该组件的HCCL仍在capture parent，
静态动作表不是实测timeline；完整模型、物理通信侧流、TP2和性能尚待验收。
该增量当时未勾选1.6/1.10–1.12；后续软件工作转入C7的DSpark Graph组合。

2026-11-13 实现收口：1.6 的非 Engram 调度实现、独立审查及组件验证已具备，
本次勾选。供体 warmup/capture 的 send/recv 同样强制使用 NPUGraph parent，
replay 继承捕获图；compute 侧流、same-stage bank/event 依赖、cross_layer_v1
共享静态调度及最终 join 已实现。详见
[锁定供体与当前实现审查](../afd-hccl-full-parity/evidence/graph-parent-transport-parity-20261113.md)，
以及其中链接的三层 diagonal Graph、eager TP1/TP2 和正常 F service 独立证据。
不以开关名称额外要求 Graph 通信侧流，也不将该勾选计作物理重叠或完整模型验收。
1.10–1.12 继续待验：包括设备组合、长运行/allocator、A/F timeline、原生模型及性能；
Engram 相关新增适配仍按既定排期后置。

2026-11-13 timeline 组件增量：真实 F factory/daemon/Graph service × role-profiler
全关/全开各 23 parent，每个会话实际 60 MatMul（12 warmup、0 capture、36 replay、
12 eager）及 240 HCCL payload 任务，均按真实 host scope 与 trace/CSV 的
Model/Task/Stream/时间身份独立核对。首轮 CSV Step ID 的解析失败原样保留，
只离线重新解析原包，随后仅新跑全开；两轮 A/F 均退出 0、无强杀、八卡 idle。
见[审查与归档](../afd-hccl-full-parity/evidence/graph-ffn-profile-review-20261113.md)。
F 侧组件 timeline 不等于完整 A/F timeline，不证明物理流归属/重叠、长运行或性能；
1.10–1.12 仍保持 pending。
