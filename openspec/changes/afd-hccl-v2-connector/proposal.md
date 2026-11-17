# 标准 HCCL P2P 接入 v0.30 V2

## Why

v0.30 没有 P2pHcclAFDConnector。旧分支有标准 HCCL 供体，但控制面、元数据和 F runner 契约已变化，整包复制会引入旧执行假设。

## What Changes

新增标准 P2P connector，接入工厂/配置/校验/通信组/V2 控制面；首版只支持同步 eager/U1/F gate，明确 ID 与精度协议。

## Capabilities

### New Capabilities

- `afd-hccl-transport`: 标准 HCCL P2P 接入 v0.30 V2。

### Modified Capabilities

无既有 OpenSpec 主规格修改；实现须保留现有支持矩阵的回归。

## Impact

connectors/npu/p2p_hccl.py（拟新增）、connectors/{base,factory,metadata}.py、distributed、NPU FFN runner、attention_metadata、config/validation 与对应测试。

## Status

C1 组件实现与 A3 验收完成，包括真实 core/client/manager 生命周期；不代表完整 V4.1 模型可运行。入口：[总计划](../../ROADMAP_ZH.md)、[设计](design.md)、[任务](tasks.md)、[证据](evidence/engine-validation.md)。

## 2026-11-09 完整迁移范围修订

本 change 保留原有实现/验收范围和进度。最终目标是 [HCCL 全能力迁移](../../HCCL_CAPABILITY_MIGRATION_ZH.md)，
新增 C4 microbatch/拓扑、C5 Graph、C6 多流、C7 DSpark/PD 组合均为本项目必交付。
本文件对 Graph/DBO/PD 的拒绝及 1A1F 限制描述的是历史组件范围；后续须按新计划有条件解除。C2 已扩展等 TP 配对/schema v2，不能将旧 schema v1 描述当作当前全部实现。
新增任务见 [tasks.md](tasks.md) 第 2 节；本次不把既有通过扩大为完整迁移通过。
