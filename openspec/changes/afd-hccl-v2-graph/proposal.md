# HCCL V2 Graph 捕获、回放与 eager 回退

## Why

eager IO 的主机握手和 tensor 校验不能直接捕获。v0.30 已有其他 connector 的 Graph 设施，但 HCCL、独立 FP32 和 V4.1 Engram 仍需专门接入。

## What Changes

HCCL V2 Graph 捕获、回放与 eager 回退，属于完整 HCCL 迁移的必交付阶段，详见 [迁移矩阵](../../HCCL_CAPABILITY_MIGRATION_ZH.md)。

## Capabilities

### New Capabilities

- `afd-hccl-graph`: HCCL V2 Graph 捕获、回放与 eager 回退。

### Modified Capabilities

无已发布主规格改名；保留当前 eager/U1 和既有 connector 行为。

## Impact

NPU A/F runner、HCCL protocol/transport、ubatch_runner、必要 NPU patch 与 V4.1 Engram/attention 适配。

## Status

计划待实施，本次不修改运行时。见 [设计](design.md)、[任务](tasks.md)、[总计划](../../ROADMAP_ZH.md)。
