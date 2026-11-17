## ADDED Requirements

### Requirement: Eager pipeline dependencies

eager/U2 SHALL 用 event 连接收算发及 continuation。

#### Scenario: Eager pipeline dependencies behavior

- **WHEN** 另一 stage 在本 stage 等待 F2A 时推进
- **THEN** 结果只在正确 recv_done 后消费，FP32/IDs/lifetime 有效。

### Requirement: Capture stream legality

Graph 多流 MUST 满足 parent-stream 捕获和 fork/join。

#### Scenario: Capture stream legality behavior

- **WHEN** 开启 compute overlap 和三流计划
- **THEN** 无未汇合 side-stream HCCL，replay 事件依赖正确。

### Requirement: Switch parity

开关 SHALL 保留显式覆盖、严格值与跨层依赖。

#### Scenario: Switch parity behavior

- **WHEN** 全关或 CROSS_LAYER=1 且 RECV_STREAM=0
- **THEN** 全关仍为同步 Graph/U2，非法依赖在通信前处理。

### Requirement: Measured overlap

重叠/性能声明 MUST 有正确性和真实 timeline 证据。

#### Scenario: Measured overlap behavior

- **WHEN** 仅有多 stream 日志或 CPU 检查通过
- **THEN** 不声明有效重叠或收益已验收。
