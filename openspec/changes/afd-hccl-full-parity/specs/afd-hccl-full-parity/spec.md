## ADDED Requirements

### Requirement: Engram adaptation scheduling

后续开发 MUST 先补齐其他 DSV4.1 功能的非 Engram 实现、审查和可独立验证部分，
之后才开始中优先级 Engram 新增适配；已有行为、证据和最终组合验收范围 SHALL 保留。

#### Scenario: Other functions are still being prepared

- **WHEN** 高优先级 DSV4.1 非 Engram 功能尚未具备
- **THEN** 继续这些功能，不新增 Engram 专项实现或专项设备实验，保留原生 Engram 和 A/F 所有权。

#### Scenario: Other functions are ready for Engram integration

- **WHEN** 非 Engram 高优先级实现、审查和独立验证具备，余项明确依赖 Engram 或最终整体验收
- **THEN** 开始 tasks 第 2 节中优先级工作，不要求先完成依赖 Engram 的整体验收，不把延期项删除或提前标为通过。

### Requirement: Speculation mapping

迁移 SHALL 保留适用 MTP 能力，并以原生 DSpark 语义接入 V4.1。

#### Scenario: Speculation mapping behavior

- **WHEN** V4.1 DSpark+Graph/U2 或旧模型 N 步 MTP
- **THEN** V4.1 draft 不发布 F 工作，旧模型执行完整 N 次协议，二者不混淆。

### Requirement: PD combination parity

本项目 MUST 覆盖供体五类 HCCL PD/DSpark/U2/Graph 组合。

#### Scenario: PD combination parity behavior

- **WHEN** Prefill 后 Decode 消费 cache 与历史
- **THEN** 按逻辑组件/请求坐标恢复 KV/index/Engram，真实请求与原生基线等价。

### Requirement: Operational parity

交付 SHALL 包含配置、诊断和 A/F profiler 控制。

#### Scenario: Operational parity behavior

- **WHEN** 显式关闭 overlap、启停 profiler、取消后继续服务
- **THEN** 有效值可见，trace 有效，业务与管理不被 F recv 阻塞。

### Requirement: Full migration completion

完整交付 MUST 逐行满足能力矩阵的实现或语义映射及所需验收。

#### Scenario: Full migration completion behavior

- **WHEN** 只通过 eager/U1 组件或计划检查
- **THEN** 整体未完成，不以降低矩阵或延期替代验收。
