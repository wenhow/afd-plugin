# V4.1 文本 eager 的角色模型适配

## Why

V4.1 的延迟 HC、Engram 与 shared attention state 改变了模型语义。只注册类名或替换 V4 decoder 会丢失状态、路由精度和历史 token。

## What Changes

复用 RFC 原生 wrapper/model/layer forward，在构造时划分 A/F 所有权，只以远程 proxy 替代 MoE；补齐权重与 V2 ModelState。

## Capabilities

### New Capabilities

- `afd-v41-text-eager`: V4.1 文本 eager 的角色模型适配。

### Modified Capabilities

无既有 OpenSpec 主规格修改；实现须保留现有支持矩阵的回归。

## Impact

拟新增 model_executor/models/npu/deepseek_v41.py，以及插件注册、角色构造、FFN runner 和必要 metadata；精度协议可能扩展 C1 payload。

## Status

进入实现阶段；锁定原生契约已核验，模型功能与数值/硬件验收仍待完成。入口：[总计划](../../ROADMAP_ZH.md)、[设计](design.md)、[任务](tasks.md)。

## 2026-11-09 完整迁移范围修订

本 change 保留原有实现/验收范围和进度。最终目标是 [HCCL 全能力迁移](../../HCCL_CAPABILITY_MIGRATION_ZH.md)，
新增 C4 microbatch/拓扑、C5 Graph、C6 多流、C7 DSpark/PD 组合均为本项目必交付。
“保留原生 forward、替换 MoE”是本 change 同步 U1 的边界；C4/C6 为异步执行增加目标 V4.1 continuation，并单独审查，不将旧 V4 tuple/aux 时机照搬。
新增任务见 [tasks.md](tasks.md) 第 2 节；本次不把既有通过扩大为完整迁移通过。
