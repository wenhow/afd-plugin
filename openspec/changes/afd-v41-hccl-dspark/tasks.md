# 在 HCCL 文本路径上接入 V4.1 DSpark：任务

只有完成对应验证并记录证据后才勾选。2026-11-09 用户要求等待 A5 验证期间并行开发，
本轮推进源码契约和 CPU 适配；C2 完整模型结果仍是 C3 硬件验收的前提，
模型总 guard 保持关闭。范围见 [并行开发记录](evidence/parallel-development-20261109.md)。

## 当前排期（2026-11-12）

遵循[总计划优先级](../../ROADMAP_ZH.md#engram-priority)，DSpark 的 proposal、采样、
接受/拒绝、target-only、aux、KV 推进和异常生命周期继续高优先级。新增 Engram
lookback/history 修复与专项验证后置至其他 DSV4.1 功能具备后的
[C7 2.5](../afd-hccl-full-parity/tasks.md#engram-medium)；已有 native history/lookback
行为和证据保留。U2/Graph 其他组件可先接入，完整模型组合和性能仍待 Engram 依赖闭合。
普通旧 V4 MTP 后续开发继续暂停，不作为 Engram 专项开始条件。

## 1. 实施与验证

- [x] 1.1 复核原生/RFC draft 类选择与 checkpoint 的 DSpark 元数据，排除普通 MTP 配置混用。（2026-11-09 锁定源码与已回传 config 复核，见并行开发记录。）
- [x] 1.2 追踪 target/draft context hook，写出 target-only 远程消息序列，并确认是否存在多余宣告风险。（源码 scope 已确认；2026-11-09 A3 实际依赖下 target/draft context 回归通过，证据见恢复验证，不作为 DSpark 推理证明。）
- [x] 1.3 逐项核对 aux target layer IDs、捕获位置、mean/concat、SP gather、block size 和 draft stage 数。（保留原生实现；SP 仅核对源码，本轮不开放或验收 SP。）
- [ ] 1.4 评估 A 端完整 draft 的权重/workspace/HBM，固定验收拓扑。（已记录 7.388 GiB 编码权重及理想分片；RoPE 存储共享、峰值和拓扑待实测。）
- [ ] 1.5 完成 C3 架构和性能成本审查，保留关闭 DSpark 的独立回退。（配置/loader 补丁静态审查通过；完整资源与运行时审查未完成。）
- [ ] 1.6 实现插件侧必要能力和 metadata 路由，draft 不发布远程 F 工作，不修改上游 sampler。（最小实现及实际依赖下 loader/context 回归已通过；完整模型/资源放行待 A5，总 guard 关闭。）
- [x] 1.7 增加 target-only 消息、真实配置选择和空步/退出的 CPU 边界回归。（2026-11-09 A3 原生 API 六文件 456 passed、FFN worker/runner 生命周期 14 passed，均无 skip；见恢复验证。仅为 CPU/API 边界，不代表 DSpark NPU 推理通过。）
- [ ] 1.8 【高/待中】先完成 DSpark 输出、真实接受/拒绝、fallback 与 KV/采样历史的可独立验证部分；Engram lookback/history 的新增适配及联合 NPU 对照待 C7 2.5，整项不提前勾选。
- [ ] 1.9 验证不同请求长度、连续请求和退出，无额外 F recv；关闭 DSpark 后 C2 回归通过。
- [ ] 1.10 记录真实 acceptance 与卡数归一化性能，完成 eager/U1 基线说明；U2/Graph/PD 的完整交付在本项目 C4–C7 验收。

## 2. 完整 HCCL 迁移的增量范围（2026-11-09）

以下为本次新增任务，既有勾选与证据不变。详细依赖见 [能力矩阵](../../HCCL_CAPABILITY_MIGRATION_ZH.md)。

- [ ] 2.1 【高/待中】先接入 U2/Graph/多流的 target-only、aux merge、proposal 和接受/拒绝独立组件；Engram 与这些组合的整体验收待 C7 2.3–2.5，保留原始任务范围。
- [ ] 2.2 先与 C7 核对 DSV4.1 的五类 PD/DSpark 组合，Engram handoff 待 C7 2.6；普通旧 MTP 分支后续开发继续暂停，保留适用能力和既有改动，不阻塞 DSV4.1 排期。

2026-11-11 按用户要求优先DSpark，普通MTP后续开发暂停并保留本地改动。
本轮以364a19f archive加七个DSpark测试文件在A3验证原生proposal/Markov
host主干、target-only上下文，以及全接受/首拒/中拒后的native采样控制、history、
computed frontier、下一轮Engram lookback和KV收尾。003完整A runner33通过，
004完整coordinator117通过；两轮均57项安装源码一致、无device probe。
002的131通过/19夹具失败和003的148通过/2断言夹具失败原记录保留，不合并
为一次150项全绿。边界与raw见C7 evidence/a3-dspark-native-cpu-20261111。
设备rejection/history/gather、实际draft backbone/KV预计算和target捕获不在
本轮CPU证据范围；1.6/1.8/2.1保持待完成，A5完整模型仍后续集中验收。

同日继续U2组合CPU增量：1aac3e4 archive仅覆盖三个测试文件，实际native
U2prepare→父级RejectionSampler/history→A-local DSpark→KVpost，以及下一轮
Engram lookback均保留。dspark-u2-001完整两模块152通过、64项源码一致、
无skip/device，生产源码未改。真实输入/metadata/graph/draft KV/device kernel
为明确CPU边界；仅单轮target→next proposal，不认定连续NPU解码或2.1验收。

连续周期软件增量7c09186已交付：dspark-continuity-001同一snapshot完整两模块
158通过（A39/coordinator119）、64项安装源码一致、0fail/error/skip/device。
U1/U2真实native采样host链连续两轮，首轮状态经actual cached-request update
派生第二轮，独立logits改变接受/拒绝结果；另组U1/U2组件覆盖hit→miss/eager
完成→hit且三次draft均A-local。该三步未连接真实execute/sample/history完整链，
不把两组独立case合称该组合；原device/model边界不变，1.6/1.8/2.1仍待完成。

接续组合增量dspark-combined-001：同一真实nativeA owner完成FULL→NONE/eager
完成→FULL三个execute/sample/history/proposal/KV周期，后两轮输入派生自实际
采样状态；中间实际hidden/aux对独立CPU算术oracle，三轮最终history/lookback/
bonus边界与target-only草稿context和协议均检查。单snapshot完整A41/coordinator119
共160通过，64源码pin/137生产基底7c09186一致，0fail/error/skip/device；完整
设备Graph重算、真实Engram producer/SFAmetadata/KV/NPU接受率仍未验收，
CPU叶子边界与C7 tasks记录一致，1.6/1.8/2.1继续pending。
