## ADDED Requirements

### Requirement: Native HC ownership

A 端 MUST 保留原生 HC/归一化、ffn_pre→pre_mix 与最终 collapse 语义。

#### Scenario: Native HC ownership behavior

- **WHEN** 远程 FFN 输出返回
- **THEN** 使用本层 residual/post/comb 恢复原生后处理，下一层收到对应 ffn_pre。

### Requirement: Native token state

文本服务 SHALL 保留 raw input tokens、Engram V2 ModelState 和 shared attention state。

#### Scenario: Native token state behavior

- **WHEN** 同一请求继续 decode 或请求槽复用
- **THEN** Engram 读取正确历史，新请求不继承旧请求状态。

### Requirement: Construction isolation

角色模型 MUST 在构造前裁剪模块归属，并按完整权重命名空间加载。

#### Scenario: Construction isolation behavior

- **WHEN** F 冷启动加载 V4.1 checkpoint
- **THEN** 只加载所需 router/experts，不分配全模型后再删除，也不误匹配 compressor 的 gate。

### Requirement: MoE equivalence

F 输出 SHALL 保留原生精度、IDs、shared experts、缩放和归约语义。

#### Scenario: MoE equivalence behavior

- **WHEN** 原生同输入 MoE 与 AFD MoE 比较
- **THEN** 在约定误差内输出一致，不重复 all-reduce 或丢失 FP32 router 输入。

### Requirement: Hardware-qualified text path

支持声明 MUST 基于目标 NPU 原生控制组和 AFD 同条件文本验收。

#### Scenario: Hardware-qualified text path behavior

- **WHEN** 仅有 CPU 或静态检查通过
- **THEN** 状态仍标记为待 NPU 验收，不宣称 dsv41-flash 已支持。

### Requirement: Uniform TP pairs

C2 SHALL 在 C1 的 TP1 配对基础上增加等规模 A/F native TP 配对。
每个 pair MUST 核对 TP size/rank，使用独立端口和 connector 组，保留 native TP 组及归约所有权。
首路径 MUST 拒绝 EP/SP、非 1 的 PP/DP/PCP/DCP、Graph/DBO 和推测解码。

#### Scenario: Independent pair lifetime

- **WHEN** 一个 TP pair 关闭且其他 pair 仍在工作
- **THEN** 其他 pair 与各角色 native TP 组仍有效，F EngineCore 等待全部 rank 正常 STOP 后才退出。

#### Scenario: Mismatched peer geometry

- **WHEN** 对端 TP size/rank 不匹配，或完整端口区间越界
- **THEN** 拒绝会话或配置，不进入模型数据传输。

### Requirement: Checkpoint quantization identity

V4.1 量化角色模型 MUST 使用表达 FP4 experts 的 deepseek_v4_fp8 配置，不能静默采用 generic FP8。

#### Scenario: Generic FP8 auto-selection

- **WHEN** 锁定版本自动选择 generic FP8 作为量化角色配置
- **THEN** 在模型构造/权重加载前拒绝，并提示显式指定 deepseek_v4_fp8。
