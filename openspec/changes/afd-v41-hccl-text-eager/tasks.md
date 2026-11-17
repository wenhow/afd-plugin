# V4.1 文本 eager 的角色模型适配：任务

只有完成对应验证并记录证据后才勾选。C1 组件验收完成；C2 核心实现及组件验证已落地。A3 量化算子限制仍存在；用户回传的 [A5 组件证据](evidence/a5-20261107-manual-001/README.md)已确认 MXFP base、原生 wheel norm、缩小构造和 TP2 通过；最新 [原生 EP8 回传](evidence/a5-20261110-native-ep8-success-001/README.md)进一步确认部署环境完整 checkpoint 冷加载与首次文本请求通过。锁定源码对控制组、退出回收、完整数值及 AFD 端到端验收仍未完成。静态/CPU/组件不代表完整数值或端到端验收，见 [开发记录](evidence/development-status.md)。

## 当前排期（2026-11-12）

遵循[总计划优先级](../../ROADMAP_ZH.md#engram-priority)：量化/路由、A/F 冷加载、
HC/shared-KV 和生命周期的非 Engram 工作先做；1.8 Engram 专项为中优先级后置，
集中依赖 [C7 2.1–2.2](../afd-hccl-full-parity/tasks.md#engram-medium)。已有原生 Engram
可用于既有资格候选；若需新增 Engram 修复或专项实验，登记后等待中优先级，不通过
删除层或绕过拒绝继续。1.10–1.12 的整体验收仍保留真实前提及原始证据。

## 1. 实施与验证

- [x] 1.1 逐层追踪 U01/U02/U03 的 wrapper→model→decoder→MoE 调用，记录注入点和完整签名。复用 C0 契约，92 个方法的签名/行号/AST hash 与锁定 Git 对象一致；记录 class hooks、decoder 无 mlp hook 的构造限制、A 本地专家元数据及 Engram/shared KV 所有权，见 [原生契约](evidence/native-contract.md)。仅静态证据。
- [ ] 1.2 建立 A/F 模块、参数命名空间、通信组和 HBM 峰值预算表，确定可运行的最小拓扑。[48 shard 预算](evidence/resource-budget.md)已补 Engram 重编码和 native RoPE；用户确认 A5 8×96 GiB 后，补 [原生 TP8 与 A4/F4 TP4 的容量候选](evidence/a5-topology-budget-20261109.json)。TP2 组组件通过，完整峰值/最小拓扑仍未验收。
- [ ] 1.3 实测目标 DeviceOperator 的归一化双返回，决定独立 FP32 传输或等价重建，并记录 router/数值证据。[A3 fallback](evidence/precision-validation.md)及 [A5 原生 wheel norm 18 组](evidence/a5-20261107-manual-001/README.md)同值，仍独立传输 FP32；A5 可选 custom top-k 未注册/未执行，完整路由数值未验收。
- [x] 1.4 完成模型切分审查：保留原生 forward、延迟 pre_mix、Engram 与 shared KV；明确补丁删除条件。[独立静态审查](evidence/model-split-review.md)和缩小原生构造通过；不代表完整数值验收。
- [x] 1.5 实现 V4.1 专用角色构造/注册，保持模型身份、raw token 和 get_model_state_cls 能力。[A3 缩小 wrapper/loader/EngramModelState](evidence/a3-20261107-attention-construction-run4/results/wrapper/result.json)实际通过，模型总 guard 保持关闭。
- [x] 1.6 实现完整签名的远程 MoE proxy 和 F compute hook，保留 shared/routed 返回、缩放及 SP/TP 归约。首路径 non-SP，F 调用完整 native MoE；CPU seam 及 [TP2 传输/原生组组件](evidence/tp-validation.md)通过。未声称原生 MoE 量化数值、SP 或 TP8 验收。
- [x] 1.7 实现完整命名空间权重过滤及映射，保留原生量化加载后处理；核对 Engram/draft/HC 不误入 F。native loader seam 测试及 [角色审查](evidence/model-split-review.md)通过，补显式 DS FP4 量化选择保护；完整 checkpoint 冷加载仍在 1.11。
- [ ] 1.8 【中：后置】其他 DSV4.1 功能具备后，对齐 V2 Engram history、请求结束/复用、空步及 dummy 生命周期，验证其与共享 attention state 的请求隔离；专项实施见 C7 2.2，保留已有代码及证据。
- [x] 1.9 增加边界行为测试与已有 V4/V2、非 AFD 路径回归；不能用生产源码子串断言替代行为。[回归记录](evidence/development-status.md)：本地 808 passed/285 skipped；A3 模型/compat/V2 424 passed/1 skipped，connectors/distributed 652 passed/4 deselected。未代替设备/E2E 验收。
- [ ] 1.10 运行目标原生 V4.1 text eager，保存环境、checkpoint/config、拓扑、输出和内存控制组。A3 的 9.0.1/9.2.0 均拒绝 DynamicMxQuant；A5 MXFP base 已通过，完整控制组尚未通过。2026-11-09 第二次回传的 DS MXFP8 Linear odd-pair 问题已修正；第三次回传确认 Linear 探针 6 项通过、48/48 分片已读取，但 routed MoE 的 W2 scale `(384,5120,9)` 仍被按 8 列 reshape 而失败，见 [第三次诊断](evidence/a5-20261109-native-startup-003/diagnosis.json)。已在插件内扩展显式 odd-pair 修正至 MXFP4 MoE，并补 [单卡 MoE 探针与完整 TP8 操作说明](evidence/A5_NATIVE_EAGER_ZH.md)；后续 A5 MoE 探针前 6 项通过，W2 K=288 遇到 WeightNz 的 K=64×scale_pairs 校验，已补物理 W2/激活 K 对齐及生产 GMM2 hook，见 [后续诊断](evidence/a5-20261109-mxfp4-probe-001/diagnosis.json)，新 12 项仍待 A5 重跑。不修改上游源码。Ascend 实际 2db8549c 与锁定 7e37e29a 的相关源文件完全相同，源版本差异单独保留。完整路由、FP4 GEMM、HBM、首请求和 AFD 对照仍待验收。
  2026-11-10 更新：上述新 MoE 探针 12 项已回传通过，最大绝对误差均为 0，W2 K=288→320 及补丁 hash 已核对；第四次完整回传确认八个 rank 完成加载，随后在 profile 的融合 SwiGLU 拒绝 576-wide 输入，见 [本次记录](evidence/a5-20261110-native-startup-004/diagnosis.json)。已补各 gate/up 半轴 288→320 和新 12 项激活链路探针，待 A5 执行；启动器先探针、通过后才加载模型，日志全重定向。完整控制组 1.10 保持未完成，原先探针仅覆盖 2 个 synthetic experts 的 GMM 组件。
  2026-11-10 原生 EP8 更新：DP8/TP1/EP8 + CPU Engram shared memory 在 `/dev/shm=512G` 下完成八 rank 加载，health=200，首请求因 CompressorV2 动态符号解析失败返回 500。后续只读 [opapi 探测](evidence/a5-20261110-opapi-001/README.md)确认 custom_transformer 库可加载且两个符号均存在，但环境未导出 vendor 和库路径；启动器已补进程导入前的路径设置与加载前检查，首请求重跑仍待回传。符号查询不等于 kernel/NPU 验收，1.10 保持未完成。
  2026-11-10 原生首次请求通过：用户回传 [完整冷加载与首次文本请求](evidence/a5-20261110-native-ep8-success-001/README.md)，部署环境 afd-plugin=e7cfe618、Ascend=4ec6dc77，DP8/TP1/EP8 + shared CPU Engram、eager/DSpark off。8 rank 加载成功，HTTP 200，响应 ` Paris.`、token IDs `[11111,16,1]`，请求退出码 0，CompressorV2 符号异常未再现。此项实际环境的首次原生 smoke 已通过；锁定 Ascend 7e37e29a 的完整控制组、更多输入/数值和退出回收仍未验收，1.10 保持未完成。SIGINT 后出现退出阶段错误及 shared_memory 清理警告，退出设备快照仍有 worker，不能宣称资源已完全回收。AFD 对照仍待 1.11。
- [ ] 1.11 运行 HCCL AFD 同条件文本精度、冷加载、prefill/decode、连续/并发及请求结束用例。用户 A5 手工组件检查已通过，尚无完整 checkpoint 对照；[验证包说明](evidence/MANUAL_QUALIFICATION_ZH.md)继续保留组件范围，不绕过总 guard。
- [ ] 1.12 【高/待中】先完成非 Engram 的 F recv、异常退出与资源回收；Engram retire/group 清理及整体验收待 C7 2.2。C3 等其他功能的独立开发可先推进，硬件验收仍保留 C2 前置条件；性能在完整功能验收后测量。

## 2. 完整 HCCL 迁移的增量范围（2026-11-09）

以下为本次新增任务，既有勾选与证据不变。详细依赖见 [能力矩阵](../../HCCL_CAPABILITY_MIGRATION_ZH.md)。

- [ ] 2.1 【高/待中】先冻结供 C4 使用的 HC continuation、ffn_pre/pre_mix、shared state 和 prepare/complete 边界；Engram buffer/event 的新增适配后置至 C7 2.3，整项待该依赖闭合。
- [ ] 2.2 【高/待中】先核对精度、raw IDs、TP 配对及非 Engram ModelState 的多 stage/Graph 契约；Engram 对应部分待 C7 2.3–2.4，C2 完整通过仍仅代表 eager/U1。
