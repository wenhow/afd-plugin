# HCCL 多 stage 协议与 V4.1 U2/DBO

## Why

当前 HcclForwardPlan 将 stage_idx 限制为 0，transport 只维护一个活动层/事务，F compute 固定 num_stages=1。已有 V2 UBatchRunner 不等于 HCCL 已支持 U2，必须迁移供体 stage 隔离、layer-major 和聚合/fanout。

## What Changes

HCCL 多 stage 协议与 V4.1 U2/DBO，属于完整 HCCL 迁移的必交付阶段，详见 [迁移矩阵](../../HCCL_CAPABILITY_MIGRATION_ZH.md)。

## Capabilities

### New Capabilities

- `afd-hccl-microbatch`: HCCL 多 stage 协议与 V4.1 U2/DBO。

### Modified Capabilities

无已发布主规格改名；保留当前 eager/U1 和既有 connector 行为。

## Impact

HCCL protocol/transport/config/connector，distributed、NPU A/F runner、ubatch_runner、V4.1 model 及既有模块测试。

## Status

2026-11-09 已开始布局层实现及 CPU 验证，现有 U1 使用共用的 HCCL rank 计算；
多 stage/多 peer wire 和模型执行入口尚未开放。见 [设计](design.md)、[任务](tasks.md)、[总计划](../../ROADMAP_ZH.md)。
