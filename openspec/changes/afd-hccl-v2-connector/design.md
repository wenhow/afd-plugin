# 标准 HCCL P2P 接入 v0.30 V2：技术设计

状态：首版 connector 注册、配置准入及 V2 F 服务循环已实现并通过 A3 组件验证。
原生 A provider/MoE 边界留待 C2，完整模型入口保持保护；故障进程退出验收继续执行。
源码标号见 [锁定阅读索引](../../../.agents/dsv41/sources.lock.json)。
## 源码事实与阅读入口

阅读 A02/A03/A04/A05/A06、D01，及 docs/design/module/connector_contracts.md。
D01 的 HCCL 供体包含 Graph/U2、旧 MTP header/phase-ready 和跨层状态；本 change 历史实现先提取 eager target 的收发与生命周期；完整 Graph/U2/多流语义现由 C4–C7 必须补齐。
旧首层发送/缓存 IDs 的行为必须与新 FFN 逐层调用同步决定。

## 协议和所有权

1. 初始化：遵循 v0.30 的 role_rank、global/local rank 分离，明确 A/F group 成员、后端与建组顺序。
2. 控制面：一次 target forward 携带 transaction、层范围、每 rank token 布局、valid/padded 数、容量、模式；F 据此执行匹配的 MoE 层序。
3. A2F：归一化 hidden states 与 token-aligned IDs，精度方案由 C2 的 DeviceOperator 对照决定；不能静默丢弃 FP32。
4. 首版逐层发送 IDs；如果采用一次发送缓存，则两端必须共同实现 transaction 级缓存、失效和 dummy/空步规则。
5. F2A：原生 MoE 完整输出，携带/恢复既有 transfer context；A 在同层 continuation 恢复后处理。
6. shutdown/error：控制面能唤醒 F 等待循环，所有通信组与缓冲区按所有者释放；不承诺底层失联时无限等待能够被 Python 安全打断。
   实际 HCCL 超时和进程退出语义需要 NPU 验证。

独立 HCCL schema v1 承载 execution plan，保留既有 AFDControlPayload wire。
组件限定 1A1F，F world rank=0、A=1，local NPU 单独指定；不规定完整 checkpoint 部署卡数。
初始化固定 A HELLO→F HELLO，FORWARD 后 F 先 ACCEPT，再传数据，末层后 F COMPLETE；
空步直接 COMPLETE，正常退出仅在 idle 边界 STOP/ACK。
具体约束、成本和已审查接口缺口见 [协议与架构审查](evidence/protocol-inventory.md)。
不依赖 CAMP2P 自定义算子，也不把所有 NPU connector 统一改为 HCCL。

## 成本、范围和回退

逐层 IDs 增加小量带宽，独立 FP32 tensor 则可能显著增加 A2F 流量，应测量。
首版显式拒绝此新路径未支持的 Graph/DBO/PD 组合；已有 connector 的范围保持不变。
回退为移除新 connector 注册和其独立路径。公共 payload 扩展保留默认兼容性。

## 首版 adapter 与服务循环

`P2pHcclAFDConnector` 使用独立 `begin_forward_plan/receive_forward_plan` 控制接口，
不伪装成旧 DP metadata control plane；旧 codec 保持原样。
Attention 的 `send_attn_output/recv_ffn_output` 对应真正分阶段 send/recv，
同一 context 绑定 transaction/layer/stage/padded，前一输出未消费不能进入下一层或 STOP。
F payload 新增默认 None 的 `hidden_states_fp32`，既有 connector 不受影响。

F runner 的 `execute_hccl_step` 直接按已接收的 typed plan 执行，绕开 CAMP2P token helpers；
显式传 IDs/FP32，保留 native Ascend forward context 和 profile 配置作用域。
F worker 收到 STOP 后自然退出线程，join 后再 close；接收或设备错误保持为终止状态，
不能被 shutdown event 吞掉或清空后再次启动。A 正常 shutdown 先请求 STOP/ACK。

EngineCore 通过专用状态 RPC 区分运行、peer STOP 和 terminal failure；
只有收到 STOP 才报告正常完成并进入 `SHUTTING_DOWN`。
正常完成后的清理错误须上报，已有主失败不被后续 stop 错误覆盖。
原生 fault tolerance 的重试与 terminal 会话不兼容，首版明确拒绝该配置。
group timeout 不保证失败设备上的 close 有界；失败会话交给 worker/EngineCore 进程监督，
不吞掉已有 507001 清理错误来制造正常退出。

配置限定 1A1F、V2、eager、BF16、TP/PP/DP/PCP/DCP=1、逻辑 U1、F gate，
拒绝 EP/EPLB/elastic/SP、编译/Graph、speculation、PD、LoRA。
原生 V4.1 注册的是 VL wrapper，文本路径使用原生 `language_model_only=True` 禁止多模态输入，
保留原始模型身份；默认启用多模态的配置拒绝进入组件。
首版 HBM 成本仍为逐层分配、IDs host 同步和传输额外 FP32，未做性能验收。

当前 [接线验证记录](evidence/service-validation.md) 明确区分工厂/F 服务组件与完整模型推理。
服务状态与进程退出的范围见 [生命周期记录](evidence/lifecycle-validation.md)。

## 验证

CPU 验证编码/解码、可观察消息序列、padding/rank 布局、容量、退出及非法配置。
NPU 使用受控 tensor 验证 BF16/FP32/IDs 和多层交换，再接模型。
空输入、warmup/dummy、两次连续 transaction、非法容量、shutdown 必须有明确结果。

## 2026-11-09 完整迁移范围修订

本 change 保留原有实现/验收范围和进度。最终目标是 [HCCL 全能力迁移](../../HCCL_CAPABILITY_MIGRATION_ZH.md)，
新增 C4 microbatch/拓扑、C5 Graph、C6 多流、C7 DSpark/PD 组合均为本项目必交付。
本文件对 Graph/DBO/PD 的拒绝及 1A1F 限制描述的是历史组件范围；后续须按新计划有条件解除。C2 已扩展等 TP 配对/schema v2，不能将旧 schema v1 描述当作当前全部实现。
新增任务见 [tasks.md](tasks.md) 第 2 节；本次不把既有通过扩大为完整迁移通过。
