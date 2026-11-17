## ADDED Requirements

### Requirement: Target-only remote FFN

DSpark 模式 MUST 仅将 target MoE 发布给远程 F，完整 draft 留 A。

#### Scenario: Target-only remote FFN behavior

- **WHEN** 一次 target verification 后执行 draft
- **THEN** F 只消费 target 的预期层调用，不等待 draft FFN 或 MTP phase-ready。

### Requirement: Native auxiliary features

aux features SHALL 使用原生目标层编号、捕获位置和聚合顺序。

#### Scenario: Native auxiliary features behavior

- **WHEN** 配置多个 DSpark target layers
- **THEN** 按 one-based 层序，在 Engram 注入前提取 HC mean 并按原生方式聚合。

### Requirement: Real speculative lifecycle

DSpark MUST 使用真实接受/拒绝和正确 KV/Engram history 推进。

#### Scenario: Real speculative lifecycle behavior

- **WHEN** 出现拒绝或 fallback
- **THEN** 最终生成和状态推进与原生同条件语义一致，无强制接受补丁。

### Requirement: Independent fallback

关闭 DSpark SHALL 恢复 C2 的文本 eager HCCL 行为。

#### Scenario: Independent fallback behavior

- **WHEN** speculative_config 关闭
- **THEN** 不加载 draft，不发送 draft 控制消息，C2 用例继续通过。
