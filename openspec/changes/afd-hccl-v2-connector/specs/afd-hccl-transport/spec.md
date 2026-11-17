## ADDED Requirements

### Requirement: Explicit HCCL selection

选择 P2pHcclAFDConnector SHALL 使用标准 HCCL P2P 路径。

#### Scenario: Explicit HCCL selection behavior

- **WHEN** 配置该 connector
- **THEN** 工厂返回该实现，未支持组合给出配置错误，不替换为 CAMP2P。

### Requirement: Matched transaction protocol

A 和 F MUST 对 target transaction、层序、token 数、dtype 和 IDs 策略达成一致。

#### Scenario: Matched transaction protocol behavior

- **WHEN** 连续执行两个不同 token 数的多层请求
- **THEN** 每层恰好消费匹配的数据，第二个请求不复用上个请求 IDs。

### Requirement: Capacity and padding

接收方 SHALL 按 rank 布局和 valid/padded 范围接收并验证容量。

#### Scenario: Capacity and padding behavior

- **WHEN** 合法 padding 或超容量输入出现
- **THEN** 合法数据还原有效区间，非法容量在发布不匹配传输前明确失败。

### Requirement: Lifecycle completion

控制面 MUST 定义空步、dummy 和 shutdown 的双端行为。

#### Scenario: Lifecycle completion behavior

- **WHEN** 空步后收到 shutdown
- **THEN** F 不进入不存在的 layer recv，服务可结束并释放其持有资源。

#### Scenario: Peer STOP reaches EngineCore

- **WHEN** F 接收并应答 peer STOP
- **THEN** 状态 RPC 报告正常完成，EngineCore 退出服务循环；清理失败仍上报。

#### Scenario: Failed service remains terminal

- **WHEN** F 计算或传输失败
- **THEN** 重复状态查询保留原始失败，后续清理不覆盖主异常，不在该会话重启；
  首版拒绝原生 fault tolerance 自动重试配置。

### Requirement: Preserved precision

传输 SHALL 保留原生 router 所需精度或使用经数值验证等价的重建。

#### Scenario: Preserved precision behavior

- **WHEN** FP32 tensor 含 BF16 无法表示的信息
- **THEN** 不能只发送 BF16 后通过 float() 声称等价。
