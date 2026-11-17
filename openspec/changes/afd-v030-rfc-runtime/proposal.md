# v0.30 与 RFC 运行时契约对齐

## Why

v0.30 AFD 已采用 V2，但 NPU patch 主要依据 8d4409d6；此次目标是 7e37e29a。模型身份、worker 初始化、loader 和 Engram 组的差异需要先明确，不能只放宽版本检查。

## What Changes

整理四版本台账；重建受影响的目标 patch；设计 V4.1 模型注册和角色加载；形成 FastAPI 阻塞处理记录。

## Capabilities

### New Capabilities

- `afd-rfc-runtime`: v0.30 与 RFC 运行时契约对齐。

### Modified Capabilities

无既有 OpenSpec 主规格修改；实现须保留现有支持矩阵的回归。

## Impact

afd_plugin/__init__.py、compat/npu、compat/patches/npu、NPU workers、config/validation；本 change 不修改上游。

## Status

C0 运行时契约与原生最小初始化阶段完成，进入 C1 HCCL 接入。392 项相关 CPU API 回归通过，单 rank 设备组及容器原生路由探针通过；完整 AFD 模型和端到端验收属于后续阶段。入口：[总计划](../../ROADMAP_ZH.md)、[设计](design.md)、[任务](tasks.md)、[最终记录](evidence/lifecycle-review.md)。

## 2026-11-09 完整迁移范围修订

本 change 保留原有实现/验收范围和进度。最终目标是 [HCCL 全能力迁移](../../HCCL_CAPABILITY_MIGRATION_ZH.md)，
新增 C4 microbatch/拓扑、C5 Graph、C6 多流、C7 DSpark/PD 组合均为本项目必交付。
新增任务见 [tasks.md](tasks.md) 第 2 节；本次不把既有通过扩大为完整迁移通过。
