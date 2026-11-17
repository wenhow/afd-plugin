# HCCL eager/Graph U2 多流与跨层流水：设计

状态：非 Engram 的 eager 收发流水、Graph compute fork/join、跨层调度和 A/F
执行体接入已实现并有独立组件证据；设备组合、长运行、timeline 及完整模型/性能
验收继续按 tasks.md 的 1.10–1.12 收口，Engram 新增适配按既定排期后置。
源码编号见 [来源索引](../../../.agents/dsv41/sources.lock.json)，能力编号见 [迁移矩阵](../../HCCL_CAPABILITY_MIGRATION_ZH.md)。

## 范围、源码和依赖

覆盖 H05/H09/H10/H14。eager 依赖 C4；Graph 依赖 C5，先同步正确再重叠。
对照 D01 的 PipelineEvents/GraphStreamPlan/pending 与 D04 的收算发/cross-layer，
模型执行以 C4 的目标 V4.1 骨架为准。

2026-11-12 排期：先使用 C4/C5 已具备的非 Engram 组件补齐流调度、lifetime、
join 和独立 timeline；Engram producer/consumer/event 的新增适配为中优先级，
其他 DSV4.1 功能具备后按 [C7 2.3–2.4](../afd-hccl-full-parity/tasks.md#engram-medium)实施。
最终模型与性能依赖完整 C4/C5 及 Engram 验收，不能用先行组件消除该前提，
详见[总计划](../../ROADMAP_ZH.md#engram-priority)。

## eager 数据依赖

A：compute(layer,stage) HC pre/norm → BF16/FP32/IDs ready → A2F send_done → F2A recv_done → 本 stage HC complete/pre_mix/下一层。
F：recv_done → native MoE compute_done → F2A send_done，另一 stage 可合法交错。
按 transaction/layer/stage/peer 管理 pending buffer/events，消费者真正需要结果时才等待。
正确 record_stream 并持有 BF16/FP32/IDs/接收/共享专家输出至最后消费者，避免 allocator 提前回收。
不能用每层 device synchronize 实现“安全重叠”；诊断同步只用于定位，性能报告说明其状态。

## Graph logical DAG 与 physical stream

保留 compute/send/recv 计划和 ready/compute_done/send_done/recv_done。
供体编译 lowering、warmup/capture 专门选择路径：HCCL 在 NPUGraph parent stream，模型 compute 可以 fork/join 到 side stream。
切换实际 NPU 流时同步 vLLM 已有的 current-stream TLS，并在退出/异常时恢复。
warmup/capture 各使用新的 stream/event owner，必须在进入实际图捕获前分配。
缓存以 frozen bundle 强持有原 pipeline/bank/native storage，replay 检查引用身份；
同一 key 不允许通过替换 pipeline 或递增 storage generation 热切调度。
策略变化须建立新的 session/cache/manager。
Attention-three-stream=1 不等于 capture 通信可任意置于侧流。
目标栈若允许更宽的流范围，先有捕获合法性与 timeline 证据；迁移目标是能力等价，不是机械复制 stream 身份。
hybrid DAG 使 stage 下一层仅等待本 stage 上层 receive-ready，无须等待另一 stage 全部工作。
F 接收流等待同 stage 前层 send，compute/send 分离；cross-layer 允许合法跨层交叠，
最后必须 join 所有 stage 末层 send 和 A 输出 compute，不留下未汇合子图/event。

## 开关、成本与回退

保留供体名字、显式 0/1、默认值和依赖，见能力矩阵。
五个 Graph 开关与 eager overlap 相互独立，Graph all-off 仍然是 Graph/U2。
FFN_CROSS_LAYER=1 依赖 FFN_RECV_STREAM=1；不满足时明确拒绝或有记录的一致禁用。
各 rank 记录最终值；验证全开、全关、逐项对照、非法值/依赖冲突。
保留 recipe full-decode-only:2 下 eager-overlap=0 或证明等价的模式隔离，防止 fallback 同时套两套调度。
回退到 C4/C5 同步路径仍保持协议/布局/数值；统计新 buffer、stream/event 数量和 HBM。

## 验收

CPU 验证可观察通信/依赖序列与输出；NPU 验证 tensor lifetime、多次 replay、请求/shape 切换、冷启动及退出。
同输入对比 eager/U2 overlap 开/关，Graph/U2 五项全开/全关。
A/F timeline 标出 compute、HCCL、wait 和 stage/layer；“有多条流”不能代替实际有效重叠证据。
报告总吞吐、TTFT/TPOT、实际卡数与每卡效率，无提升时定位串行点，不能因开关存在宣称收益。
