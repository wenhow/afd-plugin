# v0.30 与 RFC 运行时契约对齐：任务

只有完成对应验证并记录证据后才勾选。已进入开发；记录见 [运行时契约审查](evidence/runtime-contract-review.md)、[CPU 验证](evidence/cpu-validation.md)、[loader 决策](evidence/loader-identity-review.md)、[A3 探针](evidence/a3-validation.md)、[F 配置隔离](evidence/ffn-role-review.md) 和 [F-only 构造](evidence/ffn-construction-review.md)。

## 当前排期（2026-11-12）

遵循[总计划优先级](../../ROADMAP_ZH.md#engram-priority)：已有勾选与证据保留，
非 Engram 增量先做；Engram 新增契约/准入适配为中优先级，其他 DSV4.1 功能具备后
由 [C7 2.1](../afd-hccl-full-parity/tasks.md#engram-medium)集中处理。混合项先做非 Engram
部分，整项仍待其 Engram 依赖闭合；不因排序放宽配置或模型 guard。

## 1. 实施与验证

- [x] 1.1 冻结四版本源码台账与 source index；逐条验证对象可读取，记录本地 HEAD 与只读状态。
- [x] 1.2 对 compat/patches/npu、compat/npu 和 NPU worker 建立符号/签名/行为差异表，分类机械、行为或架构变化。
- [x] 1.3 核实原生 loader 的 model_class 接口以及 model_arch_config/_model_info/_architecture 缓存契约，确定注册方案。
- [x] 1.4 列出 A/F 模块归属和初始化通信组；确认 F 不会参与 A 的 Engram 初始化或分配完整模型。配置隔离、归属/组表、F-only 实际构造、生产 loader 接线与单 rank 原生初始化已验证；全尺寸量化权重和完整模型初始化留后续 C2。
- [x] 1.5 复核 B1 FastAPI 求解冲突；记录可复现环境处理方案。2026-11-06 用户明确采用容器已部署版本，实测 FastAPI 0.123.10；按该环境继续开发和运行验收，声明冲突不再阻塞，见 [环境记录](evidence/container-environment-policy.json)。
- [x] 1.6 完成架构审查：factory、F-only 容器、loader、worker 构造、其余旧 patch 与 V2 生命周期的目标边界、成本和移除条件均已复核；本批独立复核无阻断发现，见 [生命周期审查](evidence/lifecycle-review.md)。
- [x] 1.7 完成本 change 的 runtime hook、F 配置隔离与 loader 注入、目标 patch/worker/V2 契约对齐；保留原生模型身份、Engram/ModelState/shared KV 所有权。V4.1 feature guard 保留；A 角色模型属于 C2，HCCL connector 属于 C1。
- [x] 1.8 在既有 validation_paths 内补充 A/F 构造、配置、loader 与当前配置作用域等边界测试；没有新增设备模拟作为验收。
- [x] 1.9 相关 A3 CPU API 测试 392 passed / 1 skipped；本机 CPU-only 可收集，Ruff/format/精确签名检查完成。中间探针与旧 fixture 失败及修复见生命周期记录，缺源码跳过单列。
- [x] 1.10 完成 C0 原生与 A/F 最小构造、单 rank 设备/HCCL/world/TP/PP/DP/MC2 初始化及清理，验证 F dummy loader；36 个关键源文件匹配锁定对象。完整引擎、非空 Engram group、真实权重与 forward 验收留 C1/C2，不在本项宣称通过。
- [x] 1.11 确认实现只修改当前 afd-plugin checkout，上游与 donor 保持原状；保存本批审查、快照和日志，允许进入 C1。

## 2. 完整 HCCL 迁移的增量范围（2026-11-09）

以下为本次新增任务，既有勾选与证据不变。详细依赖见 [能力矩阵](../../HCCL_CAPABILITY_MIGRATION_ZH.md)。

- [ ] 2.1 【高/待中】先复核 V2 slice/merge、Graph descriptor、torch_npu lowering 和 PD 接口，补齐 C4–C7 非 Engram 契约；Engram producer/event 与配置准入复核后置至 C7 2.1–2.4，整项待依赖闭合，原 C0 通过不覆盖新增范围。
- [ ] 2.2 对 C4–C7 的目标组件、最小补丁、性能/内存和移除条件完成增量架构审查；仅修改 afd-plugin。
