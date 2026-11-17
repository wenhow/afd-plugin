# HCCL V2 Graph 捕获、回放与 eager 回退：设计

状态：拟实施，契约审查与硬件证据待任务补齐。
源码编号见 [来源索引](../../../.agents/dsv41/sources.lock.json)，能力编号见 [迁移矩阵](../../HCCL_CAPABILITY_MIGRATION_ZH.md)。

## 能力、源码与依赖

覆盖 H06–H08/H11，依赖 C4 同步 U2。对照 D01 的 graph HCCL lowering、D03 capture/dispatch、D04 graph key/capture、D08 attention 参数更新；
目标 A10/A15/A16/U13/U14，以及 U02/U03 的 V4.1 producer。
主要验收 FULL_DECODE_ONLY，Graph/U1 为基线、Graph/U2 为必交付。
2026-11-12 排期：C4 同步 U2 的非 Engram 组件具备后，可先推进协调、图内通信、
buffer/key、attention 更新和回退；Engram producer/event/history 新增适配为中优先级，
待其他 DSV4.1 功能具备后按 [C7 2.4](../afd-hccl-full-parity/tasks.md#engram-medium)实施。
完整 Graph/U1/U2 模型验收继续等待该依赖，不以组件结果开放生产入口，
详见[总计划](../../ROADMAP_ZH.md#engram-priority)。
FULL/PARTIAL/MRV1 等供体未支持组合不自动扩展。prefill、低阈值或未捕获 decode shape 回到 eager U1/U2。

## 图内外与模式共识

图外：Gloo/JSON、HELLO/ACCEPT/COMPLETE/STOP、host shape/IDs 校验、native input/attention/Engram buffer 更新和图选择。
图内：实际 graph-visible HCCL tensor 操作、MoE、可捕获 attention/HC 和重放所需事件。
不得 capture 动态握手、tensor.item()、新建组或数据相关 host 分支。
BF16、独立 FP32、IDs、F2A 全部纳入稳定图输入/输出，不能只捕获 BF16。

核对实际 torch_npu/CANN _send/_recv 注册、签名和编译 lowering；
供体避开 dynamic-shape guard 的 workaround 基于旧版，先确认目标是否仍需，必要时在插件内最小修补。
图外 execution plan 明确 profile/warmup/capture/replay/eager、U、stage/peer shape、层和 speculative/协议签名。
每个 native descriptor 的每次 warmup/capture 都有匹配 F 工作；replay 不重复捕获阶段主机副作用。
graph key 区分 U、stage/peer shape/布局、TP/SP、层集、speculative 和 payload dtype/版本；
同总 token 不同 peer 分布不能误命中。valid 值是否入 key 以图依赖为准，避免无谓重捕获。
A/F 共同选择命中或 fallback；任一缺图则协商相同 eager 路径，禁止模式分裂。
buffer 地址稳定、图外原位填充 valid counts/IDs/mask/history，不按请求重新分配。

## V4.1 状态与捕获约束

EngramModelState 的 captured producer、lookback/query_start_loc/valid_token_count、
ExternalEvent 按真实 V2 生命周期 prime/refill/retire。不能在 producer 前错误 record，不能被另一 stage 提前 reset。
shared-KV/index/candidate/压缩态按 layer owner 更新，现有 MLA Graph 参数更新不等于 V4.1 更新正确。
dummy/capture 输入仍须有合法 token/hash/FP32；请求槽复用、prefix cache 不污染 replay。
先实现同步 Graph/U2，C6 再增加重叠。此时已要遵守 warmup/capture 的 parent-stream HCCL 规则和 compute fork/join。

## 成本、回退与验收

逐 shape 冷捕获后多次真 replay，验证 hit、同 bucket 不同 valid、同 total 不同 peer、odd/empty、
未捕获 shape、prefill→decode→fallback→replay。
记录 A/F mode/key、真实图内 HCCL、输出及状态；启动成功或图对象存在不够。
capture sizes 覆盖 target speculative query 和 F 聚合容量；统计冷捕获时间、pool/buffer HBM。
故障/取消不能复用未排空事件与 buffer；关闭 Graph 回到 C4。
