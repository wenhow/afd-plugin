# HCCL 全能力组合、PD 与交付验收

## Why

单项 Graph/U2/多流成功不能证明 DSpark/PD/拓扑组合正确。供体还有五类 PD 场景、完整普通 MTP、capture-size 配置、诊断与 profiler，须在新框架上逐项有落点。

## What Changes

HCCL 全能力组合、PD 与交付验收，属于完整 HCCL 迁移的必交付阶段，详见 [迁移矩阵](../../HCCL_CAPABILITY_MIGRATION_ZH.md)。

2026-11-19 目标澄清：原始 checkpoint 保留 Engram，近期先闭合复用原生能力的
真实权重 eager/U1 正确性，再按依赖复核及验证 Graph/U1、DSpark/U1 和 PD/U1。
锁定 vLLM 拒绝 Engram+DBO/microbatch；U2 属于新增适配，仍保留在最终交付目标，
按中优先级审查实施。不将该限制扩大为所有 Engram 场景都不支持，也不把未被禁止
等同于 Ascend/AFD 已验收。[tasks 第 2 节](tasks.md#engram-medium)集中记录原生 U1
复用验证与新增缺口；既有非 Engram 组件证据和任务编号保留，最终范围不缩减。
具体阶段与开始条件以[总计划](../../ROADMAP_ZH.md#engram-priority)为准。

## Capabilities

### New Capabilities

- `afd-hccl-full-parity`: HCCL 全能力组合、PD 与交付验收。

### Modified Capabilities

无已发布主规格改名；保留当前 eager/U1 和既有 connector 行为。

## Impact

speculative/PD 插件兼容层、runner 必要修复、recipe/npu/P2pHcclAFDConnector、profiler/管理入口、测试与支持矩阵。

## Status

已有实现及 CPU/受控 NPU 组件证据，完整模型和组合验收尚未闭合。真实权重
eager/U1 指导已交付，历史 AFD 与原生首 token 差异仍待收口，不能仅凭请求成功
判为模型等价。A5 由用户手工验证，A3 metadata 旧失败保留但不继续追加设备定位。
本次仅修订目标、排期和工作流，不改运行时、不启动设备，也不增加任务勾选。
见 [设计](design.md)、[任务](tasks.md)、[总计划](../../ROADMAP_ZH.md)。
