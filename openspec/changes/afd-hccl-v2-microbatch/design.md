# HCCL 多 stage 协议与 V4.1 U2/DBO：设计

状态：2026-11-09 已实现 rank/stage/peer 布局、独立 v3 parent codec 与全局 CPU 调度契约。
见 [布局记录](evidence/layout-contract-20261109.md)、
[协议记录](evidence/protocol-contract-20261109.md)及[独立审查](evidence/protocol-review-20261109.md)。
生产仍为 v2/U1；独立 v3 多 stage 传输已通过 A3 组件验证，模型状态及模型硬件验收未完成。
本轮实际建组/收发边界见 [同步组件记录](evidence/staged-transport-contract-20261109.md)。
源码编号见 [来源索引](../../../.agents/dsv41/sources.lock.json)，能力编号见 [迁移矩阵](../../HCCL_CAPABILITY_MIGRATION_ZH.md)。

## 能力、依赖与源码

覆盖 H01–H04/H11；依赖 C2 eager/U1。供体 D01–D07，目标 I01–I07/A10/A15/A16/U02/U03。
覆盖供体实际支持的 U1/U2；U 是 microbatch 数，不是 A/F rank 数。U>2 独立设计。

2026-11-12 排期：先补齐协议、HC/pre_mix、shared attention、归并和退出的非 Engram
部分；其他 DSV4.1 功能具备后，再按 [C7 Engram 专项](../afd-hccl-full-parity/tasks.md#engram-medium)
补齐配置准入及 producer/history/buffer/event。已有父 producer/views 保留。
开发组件不等待完整 C2/C4 Engram 验收，完整模型 U2 仍须满足全部原有条件；
详见[总计划](../../ROADMAP_ZH.md#engram-priority)。

## 多 stage 协议与拓扑

保留当前 HELLO→FORWARD→ACCEPT→多层 BF16/FP32/IDs/F2A→COMPLETE、idle STOP/ACK 语义。
先设计支持协商的协议版本/能力，不能仅解除 stage_idx=0 校验。
execution plan 描述 phase、stage/layer 顺序、每 stage/peer valid/padded/capacity、native TP 坐标、模式和 speculative 签名。
每 stage/peer 维护独立层游标与 pending，所有 stage 完成后才完成父 transaction；
按 stage 建独立通信域或证明等价隔离，所有 rank 建组顺序一致。
版本/shape/布局不兼容在 tensor 发送前失败。空 stage 是否发送 dummy 由双方计划共同确定。

恢复等配对、A=kF 聚合和 F=kA fanout（整数 k）的实际 token 投影、拼接、拆分和返回顺序。
处理 0/1/奇数 token 的 padding；物理卡数、DP token 布局、native TP coordinate 分开，不能重复归约。
最低对照为供体 TP1 整数比例、TP2 等 A/F；当前 V4.1 等 TP 配对继续回归，TP4 候选另列完整模型验收。
EP/SP 布局按原生 MoE 核实，不凭参数能解析就放开任意组合。

当前逐层 IDs/独立 FP32 可继续使用。迁入首层 IDs 缓存优化时，A/F 必须共同定义 transaction/stage/peer 生命周期和失效，不混用逐层发、首层收。

当前实现采用独立、显式选择的版本 3，不能向严格 v2 HELLO 增加字段后宣称透明协商。
v3 在相同模型/拓扑/TP 下协商较小容量与 stage 上限，固定逐层 BF16/独立 FP32/INT32 和
layer-major 顺序；只传 native counts，接收方复算 peer offsets。每个 rank 的 ACCEPT
绑定同一完整 parent SHA256；全体接受后才能进入 tensor 调度。输入按 layer/stage/peer 排序，
stage pending 独立；A 下一层等齐 fanout 返回，F 返回前收齐聚合输入，native TP 使用同样的就绪边界。
全空 stage 无 tensor，padding-only stage 保留工作；父 COMPLETE 等所有 stage/peer 完成。
STOP 仅 idle，abort 永久结束会话；重放保护到上限后要求重连。

`HcclPlanSession` 是全 parent 的 CPU 全局契约，不是任一 rank 的网络端点。
独立同步 adapter 已实现广播、全 rank ACK/COMPLETE 汇集、Gloo control 与每 stage 的
全 world HCCL 域：先 control/HELLO，后按 stage0/1 同序建组，逆序释放且不销毁 native TP。
COMPLETE 前全 rank 做设备同步；数据中途故障的取消与监督验收仍待补充。
该组件尚未接入 V2 模型 runner，不支持跨 stage pending/多流 overlap。
Graph phase/capture key 和 DSpark/speculative 签名保留在 C5/C7，不能直接使用当前 eager plan 表达。

## 模型执行与状态

用 V2 原生输入切分和输出 merge 为外壳，建立单线程 layer-major (layer,stage) 调度。
普通 U1 保持原生 forward；U2 在目标 V4.1 layer 骨架增加 prepare_remote_ffn/complete_remote_ffn 或等价 continuation，并标记 AFD 差异。
不能直接复制旧 V4 continuation tuple，也不能认为所有模式只换 MoE 类就够。

当前显式组件已实现 decoder prepare/complete、父 token shared views、父一次 Engram
producer 与 child lookup/mask、layer-major A/F executor 及 aux/MTP 归并，见
[模型组件记录](evidence/model-stage-contract-20261109.md)。A3 CPU seam 及真实张量 view
检查通过；native V2 run/factory/构造和完整模型仍未接入/验收，不能据此开放 U2。

每 stage 保存 hidden、residual/post/comb、ffn_pre/pre_mix、positions/IDs、aux、shared index/candidate state、Engram lookup/mask/events。
锁定原生 `shared_attention_state.reset()` 实际为 no-op；冲突来自 source/indexer 对同一
topk/candidate/lengths 存储前缀的写入。stage1 不能覆盖 stage0 的后续 consumer 仍需要的内容，
不能通过省略 reset 解决；必须建立独立的 stage 存储或证明读取与覆盖的事件顺序。
Engram 持久 buffer/producer 可先按父输入准备再切 view，但必须证明不会重写另一 stage 或重复提交历史生产者。
匹配 F2A 后才能 HC post、推进 pre_mix 与最终 collapse。输出/aux 按 V2 请求顺序合并并去 padding。
先同步 U2 证明语义，再由 C6 延迟等待并增加重叠，不以串行 U1 冒充 U2。

## 成本、回退与验收

按 stage/peer 预算组、buffer、Engram 状态、独立 FP32 的 HBM/带宽；只关闭 U2 即可回到 C2。
正常边界耗尽 pending，设备故障保留 terminal+进程监督，不销毁 native TP 的组。
CPU 证明消息序列/布局/归并；NPU 证明两 stage 多层模型、历史隔离、连续交易、empty/odd/fanout 和取消/退出。
