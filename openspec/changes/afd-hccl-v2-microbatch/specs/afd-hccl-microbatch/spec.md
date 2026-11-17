## ADDED Requirements

### Requirement: Stage and peer isolation

计划 MUST 描述并隔离 stage/peer 布局、组、游标、IDs 与 pending 输出。

#### Scenario: Stage and peer isolation behavior

- **WHEN** 两个 stage token 数不同且连续执行两次事务
- **THEN** 两端按相同 transaction/layer/stage/peer 序列消费，无交叉或缓存污染。

### Requirement: Topology parity

HCCL SHALL 覆盖供体适用的等配对、A=kF 和 F=kA。

#### Scenario: Topology parity behavior

- **WHEN** peer 存在 0/1/奇数 token
- **THEN** 返回原 token 顺序，dummy 被裁剪，TP 坐标和有效行一致。

### Requirement: V41 stage ownership

U2 MUST 独立保存 V4.1 HC/pre_mix、Engram 和 shared state。

#### Scenario: V41 stage ownership behavior

- **WHEN** stage0 等待 F2A，stage1 推进
- **THEN** 不覆盖 stage0 状态，恢复后的结果等价。

### Requirement: Bounded microbatch scope

迁移 SHALL 真正执行 U1/U2 并正确归并输出。

#### Scenario: Bounded microbatch scope behavior

- **WHEN** 配置 U2 或未支持的 U>2
- **THEN** U2 两 stage 真实执行，U>2 在通信前明确拒绝。
