# HCCL 全能力组合、PD 与交付验收

## Why

单项 Graph/U2/多流成功不能证明 DSpark/PD/拓扑组合正确。供体还有五类 PD 场景、完整普通 MTP、capture-size 配置、诊断与 profiler，须在新框架上逐项有落点。

## What Changes

HCCL 全能力组合、PD 与交付验收，属于完整 HCCL 迁移的必交付阶段，详见 [迁移矩阵](../../HCCL_CAPABILITY_MIGRATION_ZH.md)。

2026-11-12 排期调整：其他 DSV4.1 功能先具备，Engram 新增适配为中优先级后置，
由 [tasks 第 2 节](tasks.md#engram-medium)集中执行。先开发并验证非 Engram 组件，
完整模型及依赖 Engram 的 U2/Graph/DSpark/PD 验收在专项闭合后收口；最终范围不缩减。
具体先后与开始条件以[总计划](../../ROADMAP_ZH.md#engram-priority)为准。

## Capabilities

### New Capabilities

- `afd-hccl-full-parity`: HCCL 全能力组合、PD 与交付验收。

### Modified Capabilities

无已发布主规格改名；保留当前 eager/U1 和既有 connector 行为。

## Impact

speculative/PD 插件兼容层、runner 必要修复、recipe/npu/P2pHcclAFDConnector、profiler/管理入口、测试与支持矩阵。

## Status

已有实现及 CPU/受控 NPU 组件证据，完整模型和组合验收尚未闭合。当前持续推进非
Engram 功能接入：A3 正常 Worker、加载、首 C2 KV及AFD首prefill/sample已完成；
独立原生两轮及退役通过，AFD第二轮metadata仍失败，继续定位实际运行上下文；
DSpark 设备验证移至 A5，最小验证入口已交付。见 [设计](design.md)、[任务](tasks.md)、
[总计划](../../ROADMAP_ZH.md)。
