# 在 HCCL 文本路径上接入 V4.1 DSpark

## Why

RFC 把 V4.1 draft 归一化为 DeepseekV41DSparkModel；DSpark 的层特征与 draft stage 不等于旧普通 MTP 的远程 FFN 次数。通用上下文 hook 可能产生多余控制消息，需通过协议测试核实。

## What Changes

target FFN 继续远程；完整 DSpark draft 在 A 端；对齐 aux features、状态推进及真实 acceptance，禁止 draft 触发 F 服务循环。

## Capabilities

### New Capabilities

- `afd-v41-dspark`: 在 HCCL 文本路径上接入 V4.1 DSpark。

### Modified Capabilities

无既有 OpenSpec 主规格修改；实现须保留现有支持矩阵的回归。

## Impact

V4.1 模型能力、attention_metadata hook、speculative 配置/runner 必要插件补丁和对应 DSpark NPU 用例；无上游源码修改。

## Status

2026-11-09 用户要求等待 A5 验证期间并行开发，当前准备源码适配及 CPU 边界。
C2 完整模型尚未验收，V4.1 总 guard 保持关闭，C3 硬件验收未执行。
入口：[总计划](../../ROADMAP_ZH.md)、[设计](design.md)、[任务](tasks.md)。

## 2026-11-09 完整迁移范围修订

本 change 保留原有实现/验收范围和进度。最终目标是 [HCCL 全能力迁移](../../HCCL_CAPABILITY_MIGRATION_ZH.md)，
新增 C4 microbatch/拓扑、C5 Graph、C6 多流、C7 DSpark/PD 组合均为本项目必交付。
新增任务见 [tasks.md](tasks.md) 第 2 节；本次不把既有通过扩大为完整迁移通过。
