# HCCL eager/Graph U2 多流与跨层流水

## Why

供体通过 A compute/send/recv、F recv/compute/send、hybrid DAG 和跨层事件减少等待。迁移 stream 名称或开关不等于有效重叠，须将 V4.1 continuation、独立 FP32 和捕获约束接入同一依赖图。

## What Changes

HCCL eager/Graph U2 多流与跨层流水，属于完整 HCCL 迁移的必交付阶段，详见 [迁移矩阵](../../HCCL_CAPABILITY_MIGRATION_ZH.md)。

## Capabilities

### New Capabilities

- `afd-hccl-multistream`: HCCL eager/Graph U2 多流与跨层流水。

### Modified Capabilities

无已发布主规格改名；保留当前 eager/U1 和既有 connector 行为。

## Impact

HCCL connector/transport、V4.1 layer-major executor、NPU FFN runner、Graph/ubatch 适配、开关和 profiler 证据。

## Status

已进入实现，当前推进固定策略的 Graph compute fork/join 与真实 A/F 执行体接入。
eager 多流、完整五开关组合和性能验收仍按任务逐项完成。
见 [设计](design.md)、[任务](tasks.md)、[总计划](../../ROADMAP_ZH.md)。
