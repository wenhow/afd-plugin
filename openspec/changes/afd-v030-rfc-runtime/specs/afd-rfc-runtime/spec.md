## ADDED Requirements

### Requirement: Immutable upstream sources

实现 SHALL 使用锁定源码契约，并保持 vLLM 和 vLLM-Ascend 源码工作区只读。

#### Scenario: Immutable upstream sources behavior

- **WHEN** 本地上游 HEAD 与锁不同
- **THEN** 使用 git show 读取锁定快照，且前后 HEAD、status 与 tracked diff 不变。

### Requirement: Role-safe construction

模型初始化 MUST 在构造前确定角色和通信组，F 端只创建 FFN 所需模块。

#### Scenario: Role-safe construction behavior

- **WHEN** F worker 初始化 V4.1
- **THEN** 不等待 A 端 Engram group，不先分配完整 Attention/Engram 模型。

### Requirement: Consistent model identity

注册和 loader SHALL 保留 DeepseekV41ForCausalLM 的一致架构元数据与原生加载后处理。

#### Scenario: Consistent model identity behavior

- **WHEN** 选择 AFD V4.1 类
- **THEN** 原始模型能力仍可供 V2 与原生 loader 读取，非 AFD 模型选择不变。

### Requirement: Dependency qualification

环境验收 MUST 明确报告锁定依赖冲突及实际解决状态。

#### Scenario: Dependency qualification behavior

- **WHEN** FastAPI 要求不存在交集
- **THEN** 不能把安装失败或 --no-deps 安装报告为兼容性通过。

#### Scenario: Accepted container environment

- **WHEN** 用户明确选择容器内已部署的依赖版本
- **THEN** SHALL 记录实际版本并继续该环境的开发和运行验证，依赖声明冲突保留为记录，不阻止按实际行为验收对应路径。
