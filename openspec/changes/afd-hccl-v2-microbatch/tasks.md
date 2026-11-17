# HCCL 多 stage 协议与 V4.1 U2/DBO：任务

全部为新增待完成项；只有实现、审查和验证证据齐全后勾选。任务状态只维护在本文件。
2026-11-09 按用户要求并行开发并恢复 A3 验证；本轮边界见
[布局契约](evidence/layout-contract-20261109.md)与
[多 stage 协议记录](evidence/protocol-contract-20261109.md)，硬件验收不提前勾选。

## 当前排期（2026-11-12）

遵循[总计划优先级](../../ROADMAP_ZH.md#engram-priority)：HC continuation、pre_mix、
shared attention、协议/拓扑、归并与取消组件先做。新增 Engram 准入、父 producer、
lookup/history/buffer/event 适配及专项设备实验为中优先级，待其他 DSV4.1 功能具备后
由 [C7 2.1–2.3](../afd-hccl-full-parity/tasks.md#engram-medium)推进。已有父级一次生产和
child views 保留；组件可先验证，完整 U2 模型仍保持 pending，不以现有 1.6 勾选证明
正常 V4.1 Engram/U2 配置入口已通过。

## 1. 实施与验收

- [x] 1.1 复核 V2 prepare/slice/merge 与供体序列，冻结 U1/U2、三类 peer 拓扑的行为矩阵（H02–H04）。（2026-11-09 锁定源码检查完成，实际两 stage 执行仍待后续任务。）
- [x] 1.2 设计可协商的多 stage plan、stage/peer 游标、empty/dummy、COMPLETE/STOP 与精度/IDs 契约（H01/H03）。（2026-11-09 eager parent 计划和全局 CPU 状态契约完成；Graph phase 与 speculative 扩展仍按 C5/C7 实施，不表示运行时已接入。）
- [x] 1.3 完成协议/状态架构审查，明确签名、stage group 顺序、最小补丁、内存成本及移除条件。（2026-11-09 counts/服务/bootstrap 独立审查及 native parent 契约复核完成；设备/模型绑定、唯一 transport 所有权和消费后复制失败终止问题已修复。见 [独立审查](evidence/staged-bootstrap-independent-review-20261109.md)、[父执行契约](evidence/native-parent-execution-contract-20261109.md)。此项是架构审查，不代表 A5 模型/offset 算子验收。）
- [x] 1.4 实现多 stage 协议/能力协商，保留 U1 与不兼容版本明确拒绝。（2026-11-09 独立 v3 HELLO、全 rank ACK/COMPLETE、stage 域及 U1/U2 真实组件通过；HELLO 不匹配拒绝通过，旧 v2/U1 保留。生产 factory 与 V2 模型接入仍在后续 1.6。）
- [x] 1.5 实现等配对、A=kF、F=kA 聚合/拆分/去 padding，正确映射 native TP 坐标与 token 数（H02）。（2026-11-09 组件实际收发接入布局；A3 的 A1/F1、A2/F1、A1/F2、A2/F2 native TP1/TP2 通过，含多 pair 非均匀/空/odd/padding-only 和 caller 归约。TP>1 不等规模仍拒绝，不表示完整模型验收。）
- [x] 1.6 实现 V2 U2 输入切分/child context 和单线程 layer-major，F 服务按相同序列执行（H03/H04）。（2026-11-09 native U1/profile/低阈值 fallback、真实各 A 计数提交、F v3 服务、正常 U2 配置的 backend bootstrap 和 STOP 生命周期已接线。最终集成 CPU 2242 passed；1A1F、2A2F/TP2 同 session U1/U2/empty/STOP 真实通信组件通过，见 [证据](evidence/a3-forward-counts-20261109/README.md)。当前限 DP1/PP1/CP1、A=F=native TP，模型 guard 保留；完整 checkpoint 验收仍由 1.7–1.12 跟踪。）
- [ ] 1.7 【高/待中】先补齐 V4.1 HC continuation、pre_mix、shared state 的 stage 所有权（H04/H11）；Engram buffer/event 专项待 C7 2.3，整项不提前完成。（2026-11-09 HC prepare/complete、四 shared buffer 无重叠 views、父 Engram 一次生产及 child lookup/mask views 已实现；CPU 与 A3 真实张量 view 检查通过。原生 HC/indexer/Engram producer/历史和双 consumer event 仍待目标算子/模型验收。）
- [ ] 1.8 实现 output/aux 归并，保持 raw IDs、独立 FP32 和原生归约；核对请求坐标与 padding。（2026-11-09 executor 已按父 token 坐标归并 hidden/aux/MTP，并接入 native U1/U2 入口；CPU 使用锁定原生 forward 顺序对照通过，A3 受控服务验证独立 FP32/IDs。真实模型归约与 aux 仍待 A5 对照。）
- [x] 1.9 增加两端协议、错误 stage/peer、奇数/空 stage、重复事务和取消/STOP 的 CPU 回归。（2026-11-13 六完整模块单轮 688P、0F/E/S，验收条件映射与独立审查通过，见 [CPU 收口审查](../afd-hccl-full-parity/evidence/c4-cpu-closure-review-20261113.md)。CPU abort/STOP 不代表能中止已阻塞的 HCCL collective，活动 device/请求取消仍待 C7；已有 [A3 故障组件](evidence/a3-failure-20261109/README.md)保留，完整模型/拓扑与联合验收继续由 1.10–1.12 跟踪。）
- [ ] 1.10 【高/待中】先验证 HC/shared-KV/index 与 U1/U2 输出的独立 NPU 组件；Engram 历史/事件及原 checkpoint 多层模型联合对照待 C7 2.3，受控组件不代替整项验收。
- [ ] 1.11 在 NPU 验证 TP1 聚合/fanout、TP2 等配对；当前 TP4 候选单列模型结果，未测拓扑 pending。（A3 已通过 TP1 聚合/fanout 与 TP2 等配对的受控算术组件；原生 MoE/TP4 完整 checkpoint 仍待 A5，不能勾选整项。）
- [ ] 1.12 【高/待中】先运行已有 connector/非 AFD 回归并记录资源/真实 stage 组件证据，供 C5/C6 非 Engram 开发使用；C2/C4 完整模型回归及生产 Graph/overlap 放行仍待 Engram 依赖与原有验收。（2026-11-09 最终 CPU 集成 2242 passed、2 skipped、4 deselected；C4 真实组件证据已留存，完整 C2/C4 模型验收待 A5。C5 独立通信图组件可先开发和验证，不据此开放生产 Graph/overlap。）
