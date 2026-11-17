## ADDED Requirements

### Requirement: Capture-safe transport

Graph MUST 捕获实际 tensor 交换，动态控制与 host 校验留在图外。

#### Scenario: Capture-safe transport behavior

- **WHEN** 多次 replay 更新后的输入
- **THEN** BF16/FP32/IDs/F2A 都正确交换，无图内 JSON/ACK/item 同步。

### Requirement: Coordinated graph identity

A/F SHALL 使用一致 stage/peer/speculative/布局图身份与模式。

#### Scenario: Coordinated graph identity behavior

- **WHEN** 总 token 相同但 peer 分布不同，或一端无图
- **THEN** 不误 replay，两端共同回退匹配的 eager。

### Requirement: Graph state updates

V4.1 Graph MUST 正确刷新 Engram history/event 与 shared-KV/index。

#### Scenario: Graph state updates behavior

- **WHEN** 同 bucket 的有效数或请求槽改变
- **THEN** 读取当前原位输入和历史，不覆盖另一 stage。

### Requirement: Decode fallback parity

FULL_DECODE_ONLY SHALL 保留 eager prefill、miss、阈值回退和 U1/U2。

#### Scenario: Decode fallback parity behavior

- **WHEN** prefill→hit→miss→再次 hit
- **THEN** 两端模式一致，无遗留接收，结果等价。
