# HCCL V2 Graph 捕获、回放与 eager 回退：任务

全部为新增待完成项；只有实现、审查和验证证据齐全后勾选。任务状态只维护在本文件。

## 当前排期（2026-11-12）

遵循[总计划优先级](../../ROADMAP_ZH.md#engram-priority)：descriptor、A/F 模式共识、
buffer/key、HCCL 图、attention/shared-KV/index 更新和回退组件先做；依赖 C4 对应
非 Engram 组件，不等待 C4 的 Engram 整体验收才开发。Engram producer/event/history
新增适配与专项实验为中优先级，后置到 [C7 2.4](../afd-hccl-full-parity/tasks.md#engram-medium)。
1.8/1.11–1.13 的完整模型结论仍待该依赖，不扩大已有通信图/CPU 证据范围。

## 1. 实施与验收

- [x] 1.1 审查 V2 manager/descriptor、torch_npu lowering、Engram producer 和 V4.1 attention 更新契约（H06/H11）。（2026-11-09 锁定源码与实装 wheel 静态审查完成，见 [源码契约](evidence/source-contract-review-20261109.md)；Engram FULL 准备前必须先决定 fallback，独立通信图不代表模型状态验收。）
- [x] 1.2 设计图外控制、图内 tensor/事件、A/F warmup/capture/replay 序列，不直接 capture eager IO。（2026-11-10 图外 phase/全 rank ACK、实际计数与图内静态 tensor/事件边界已审查；基础 native manager 已验证配对顺序。见 [实现契约](evidence/graph-development-contract-20261110.md)，完整模型入口继续由后续任务验收。）
- [x] 1.3 审查 graph key、稳定 BF16/FP32/IDs/F2A buffer、stage/peer 布局与预算（H07）。（2026-11-10 key 包含原生 request/token 切片；独立审查覆盖 session/group 身份、缓存 generation、每 stage 跨层复用及借用 output。bank 每 rank 固定开销为 `(8H+9)T+4U` 字节，不乘层数；原生模型及 graph pool 另计，见 [实现契约](evidence/graph-development-contract-20261110.md)。）
- [x] 1.4 实现 HCCL graph send/recv 注册和最小兼容补丁，核对旧 shape workaround 是否仍需。（独立组件保留实际 torch-npu op 的静态 shape，修复独立 AFD rank 域的自身坐标判断；[基础 manager 006](evidence/a3-native-manager-20261109/README.md) 已完成 U1/U2 捕获、12 次 replay、6 次 eager fallback 和 STOP 后原生 TP 检查。未据此宣称 GE/Dynamo 或 ModelAclGraphManager/完整模型验收。）
- [ ] 1.5 实现 Graph/U1/U2 descriptor 的 profile/warmup/capture 配对，重复预热不遗留 F recv。（2026-11-10 显式 native coordinator、完整捕获序列共识、pre-KV eager 桥接与 A/F runner 生命周期已实现并独立审查；各自 CPU 快照已完成定向回归。普通工厂选择、完整模型和 NPU 生命周期仍待验收。锁定 Ascend 正常启动走 pre-KV `profile_run`，没有调用 GPU `profile_cudagraph_memory`；可选临时 KV 图内存 profiling 的双端生命周期另行设计，不能把 `profile_only` 透传算作验收。）
- [ ] 1.6 实现双端 key/mode 共识；miss、prefill、低阈值回退到匹配 eager U1/U2（H08）。
- [ ] 1.7 实现稳定 buffer 原位更新、有效 token/IDs/FP32 刷新与 capture 失败清理。
- [ ] 1.8 【高/待中】先补齐 shared-KV/index 与非 Engram metadata 的图更新和 patch 审查；Engram producer/event/history 新增适配后置至 C7 2.4，整项保持未完成。（2026-11-10 受控同步 callback、父级 metadata build、单次 captured Engram producer 与固定 stage views 已实现并完成源码/CPU seam 审查；完整模型入口组装、A5 packed-cache/offset 算子及历史/event 对照仍待验收，不能勾选整项。）
- [ ] 1.9 迁移 capture-size 自动推导/覆盖检查，区分 target query、U2 拆分和 F 聚合容量。（2026-11-10 显式覆盖检查直接消费 native resolved descriptors/U2 twins，验证 query-one、顺序和协商容量；paired TP 按每 rank 预算，未在 F 重新对总 token 做 rounding。已通过 CPU bank 实际字节核对，完整入口消费继续回归。）
- [ ] 1.10 补充模式/计划、同 total 不同 peer、fallback、stage 参数更新的 CPU 边界回归。（2026-11-10 稳定组件 integration-012 为 2609 passed、2 CUDA skips、4 device deselected，外层协调器明确排除；后续协调器 015 为 54 passed。016 未受影响的六模块 421 项通过，017 主入口 11 项和既有 execution 22 项通过，018 桥接其余 21 项及 019 最后六参数通过；失败首轮均保留，不能合并成一次全绿，见 [CPU 记录](evidence/a3-control-cpu-20261109/README.md)。普通工厂/完整拓扑入口接入后继续补对应边界。）
- [ ] 1.11 【高/待中】先完成非 Engram NPU 组件的冷捕获/replay 与 eager 对照；完整 Graph/U1/U2 模型输出及状态待 C7 2.4，组件结果不勾选整项。
- [ ] 1.12 【高/待中】先验证 odd/empty、bucket、miss、prefill/decode 与请求切换的独立组件；Engram 状态参与的完整模型组合待 C7 2.4。
- [ ] 1.13 【高/待中】先记录真实图内通信、mode/key、组件图内存/启动成本和退出，保持已有 connector/非 AFD 回归；完整模型资源与联合回归待 Engram 依赖闭合。

2026-11-13：正常 F factory/worker daemon/runner/Graph service 的 A3 设备链已补齐，
五轴全关/全开各23parent，包含U1/U2、4capture、12replay、3布局回退和正常STOP。
真实NPUGraph/default pool、nativecontext、bank复用及cache先于groups释放已核验，
见[服务审查](../afd-hccl-full-parity/evidence/graph-ffn-service-review-20261113.md)。
重型worker/runner构造、A完整入口、模型/KV/Engram/DSpark和性能不在该算术组件内，
1.5–1.13按各自完整条件继续pending；不能将正常F组件等同双端完整模型接入。
