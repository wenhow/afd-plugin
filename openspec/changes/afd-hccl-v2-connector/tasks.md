# 标准 HCCL P2P 接入 v0.30 V2：任务

只有完成对应验证并记录证据后才勾选。原 C0/C1 首版组件范围已完成；C1 历史证据仅指协议、connector、F 服务及真实 core/client/manager 组件，不代表完整模型或完整 HCCL 能力。第 2 节是本次新增待完成范围。

## 当前排期（2026-11-12）

遵循[总计划优先级](../../ROADMAP_ZH.md#engram-priority)：2.1/2.2 的协议、精度、
退出和独立组件复核为高优先级。Engram 专项后置至其他 DSV4.1 功能具备之后，
不阻塞独立 HCCL 组件工作，也不以组件通过解锁完整模型；原有勾选与证据保留。

## 1. 实施与验证

- [x] 1.1 从 D01 建立最小供体清单，排除 V1 runner、Graph/U2、普通 MTP 服务循环和无关环境变量，见 [清单与序列](evidence/protocol-inventory.md)。只提取 eager HCCL/Gloo 原语和资源所有权。
- [x] 1.2 固定 V2 A/F 初始化、单次 target、多层 A2F/F2A、空输入和 STOP/ACK 退出序列；逐层发送 BF16/独立 FP32/IDs，不复制首层 IDs 缓存；FORWARD/ACCEPT 后发数据，末层 COMPLETE。
- [x] 1.3 确定 A/F global/local/role rank 与通信组映射，明确第一版实际支持的拓扑和资源需求。首版组件 1A1F，A/F world rank=1/0、local NPU 独立；不规定完整模型卡数。见 [协议与审查](evidence/protocol-inventory.md)。
- [x] 1.4 定义 payload 的 valid/padded shape、容量、IDs dtype/范围和每层传输策略；wire 固定 BF16/独立 FP32/INT32/BF16、容量取 min、padding IDs=0。C2 仍须证明原生 FP32 来源和模型数值等价。
- [x] 1.5 完成首版组件与控制面架构审查，记录公共协议兼容性、带宽/缓冲开销和回退路径。独立 schema 不改旧 wire、不注册未接线组件；已审查并修复 terminal IO 状态与 INT32 上界比较。
- [x] 1.6 实现 connector 注册、typed extra config 与精确能力校验，不回退到 CAMP2P。真实 factory 构造通过；原生 VL wrapper 只在 language_model_only 文本模式准入，完整模型 guard 保留。见 [接线验证](evidence/service-validation.md)。
- [x] 1.7 实现 eager/U1 数据面、控制面和 V2 FFN 循环接入，保留非 HCCL 路径。真实 factory + 生产 F step/worker 完成 6 次交易、10 层计算；A 显式 typed plan 接口可用，原生 A provider/MoE proxy 在 C2 接入，当前不宣称完整模型运行。
- [x] 1.8 实现重复请求、空步、dummy/warmup、异常和 shutdown 的匹配协议及幂等释放。真实重复/empty/dummy/warmup/profile/STOP 与幂等 close 通过；专用状态 RPC、原生非阻塞管理队列与退出处理修复 core 空转/管理阻塞，正常清理错误上报、原始失败保留。真实 SyncMPClient、完整 EngineCoreProc 构造、原生 Executor 构造与 manager 的 normal/error/restart 验收通过；前端实际接收死亡帧，进程全部回收。worker/runner 装配仍受控，未验收模型；见 [完整 core 组件证据](evidence/engine-validation.md)。
- [x] 1.9 增加协议序列、IDs 对齐、容量和退出的 CPU 行为测试，遵循既有测试模块归属。纯 CPU 协议 104 通过，A3 执行新增配置/connector/runner/worker seam 用例；镜像测试由 connector/FFN runtime validation_paths 覆盖。
- [x] 1.10 运行受影响单元测试并对照基线确认已有 connector 行为没有回归。本批 connector/distributed 627 通过，NPU runner/config/相关 compat（含 EngineCore）472 通过；本地 engine/protocol 127 通过；17 个 GPU 用例的失败身份/消息与历史哈希快照完全相同，4 个旧自定义算子失败仍保留，见 [对照](evidence/engine-validation.md)。
- [x] 1.11 在 NPU 实际完成多层 tensor/IDs/精度收发、连续 transaction、正常退出和故障超时验证。多层 BF16/独立 FP32/IDs、连续交易与服务通过；本批 normal/error/restart 三轮通过，F exit 1 保留计算 cause，A 实际 HCCL watchdog SIGABRT，随后全新进程完成正常交易，无监督器超时/kill。仅组件验收，非失败组优雅清理或同进程恢复；完整 core 启动/监督集成在 1.8 验收，历史 507001 与各轮探针失败均保留。
- [x] 1.12 保存真实拓扑、软件版本、命令和完整日志；[组件接线](evidence/service-validation.md)、[生命周期](evidence/lifecycle-validation.md) 和 [完整 core 组件证据](evidence/engine-validation.md) 已保存；219 文件快照、36 个既有上游文件和 6 个 core/client/manager 关键文件已核验。正常退出码 0 与 manager 失败名的上游诊断不一致、两轮探针失败均保留。C1 组件验收完成，完整模型准入仍属于 C2。

## 2. 完整 HCCL 迁移的增量范围（2026-11-09）

以下为本次新增任务，既有勾选与证据不变。详细依赖见 [能力矩阵](../../HCCL_CAPABILITY_MIGRATION_ZH.md)。

- [ ] 2.1 对照当前 stage_idx=0/schema v2 与供体完整能力，冻结多 stage/peer/Graph 协议演进及兼容策略；由 C4/C5 实现。
- [ ] 2.2 复核既有 STOP/ACK、terminal 监督和独立 FP32 语义在 U2/Graph/多流下的延续；由 C4–C7 增量验收，不回填旧组件证据。
