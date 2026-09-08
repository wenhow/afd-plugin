# DeepSeek-V4 AFD A3 性能、非等量拓扑与 A5 适配路线

## 1. 文档定位

本文用于固化 DeepSeek-V4 AFD 在完成 CAMP2P eager/U1、Graph/U1，以及标准 HCCL P2P eager/U1、U2、Graph/U1 和 Graph/U2 正确性基线后的目标、开发顺序和验收门禁，供后续开发、验证、性能分析和 A5 迁移时直接使用。

文档状态：`2026-09-08`。CAMP2P eager/U2 已冻结为 `dsv4-afd-a3-eager-u2-v1`；标准 HCCL send/recv connector 已在提交 `9578dd2cb70f9f8db54673a70e8f45fde6479245` 完成 A3 A8F8 eager/U1、U2 正确性闭环。A3-P4 的 A8F8 未调优性能参照与 U1/U2 双侧 profile 已完成：三轮重复性通过，但 U2 在 C32 比 U1 回退 37.570%，因此当前只冻结参照协议，不冻结 U2 性能基线。

A3-P5 的 `A = k x F` 非等量协议和 A2F1/A4F2 NPU 组件验证已经完成。A3-P6 的 A8F4 实模加载在 64 GiB A3 上因 FFN EP4 专家权重峰值 HBM 不足而停止；A10F5 容量代理又被固定 vLLM-Ascend 的 256 experts/EP5 非均匀放置检查拒绝。该结论是当前硬件与固定栈组合的 E2E 门禁，不否定 connector 的非等量语义。A8F4 E2E 移到高 HBM 的 A5 实机验证；A3 保留现有 A8F8 同步 HCCL 性能参照，完成 MTP 功能门禁后再恢复新的调优和公平对照。

当前性能阶段明确不引入异步 HCCL：不使用 `isend/irecv`、后台通信线程或自定义异步传输 op，保留标准阻塞式 `torch.distributed.send/recv` 和现有 NPU 同步边界。第一轮同步优化已经减少重复 host 解析、device-to-host 标量读取和每层 forward-context 构造；A8F8/U1/C32 正式三轮均值达到 57.724 output token/s，相对 P4 均值提升 17.521%，CV 为 0.689%。该结果是 C32 候选收益，不代表 C1/C8、非 AFD 公平对照或整个 P7 已冻结。

旧栈的非等量 HCCL 与同步优化已经在提交 `0d2d52ae4a0e927c23db6762b0016555fcfd1baa`、tag `dsv4-afd-a3-sync-hccl-pre-v023-v1` 固化。后续开发已切换到 vLLM `releases/v0.23.0` 与 vLLM-Ascend `rfc/vllm_cann`；旧栈的 +17.521% 结论继续作为该优化在原固定栈上的有效证据，但不能直接当作目标栈性能数字。

目标栈功能迁移已经完成：同栈原生模型的 10 条 prompt 连续 3 轮稳定，AFD eager/U2 对同栈 golden 达到 30/30 逐 token 一致，并通过 batch 1/8/32 结构、真实双 stage、Attention 先停、FFN 后退、fatal 日志和 NPU 清理门禁。目标栈同参数 C32 性能复测也已完成：U1 三轮均值为 17.082 output token/s，U2 为 12.582 output token/s，U2 回退 26.342%。因此本阶段只冻结功能兼容性，不创建目标栈性能 tag。

标准 HCCL P2P Graph/U1 和 Graph/U2 功能适配均已在目标栈完成。A8F8 实模 F0 已完成；`A = k x F` 非等量 Graph 的 graph key、multi-peer capture/replay 和 Graph 外 IDs 也已完成 CPU/Mock 及 A2F1/A4F2 真实 NPU 组件闭环。U2 固定为两个 microbatch，Graph 模式仍只允许 `FULL_DECODE_ONLY`。A8F4 实模 F0 因 A3 EP4 HBM 不足留到 A5，因此非等量 Graph 当前冻结为 component functional snapshot，不宣称产品级 A8F4 或性能收益。Graph/U3 继续 fail-fast。完整报告见 `DEEPSEEK_V4_AFD_HCCL_P2P_GRAPH_U1_VALIDATION_REPORT_ZH.md`、`DEEPSEEK_V4_AFD_HCCL_P2P_GRAPH_U2_VALIDATION_REPORT_ZH.md` 和 `DEEPSEEK_V4_AFD_HCCL_P2P_GRAPH_UNEQUAL_COMPONENT_REPORT_ZH.md`。

MTP/speculative decoding 已纳入必交付范围。A3-P7M0 原生 MTP 基线和角色/权重契约、A3-P7M1 HCCL P2P eager/U1 + MTP、A3-P7M2 target Graph/U1 + draft eager MTP、A3-P7M3 eager/U2 + MTP，以及 A3-P7M4 target Graph/U2 + draft eager MTP 功能均已完成。M4 的 F0 达到 30/30 golden、batch 1/8/32、真实双 stage、capture/replay、正常停止和清理门禁；P1 128/128 成功，单轮 31.473 token/s、MTP acceptance rate 84.51%。该数字只作功能 guard，不创建性能 tag。M3 的 42.583% 回退仍登记在 `P8D-PERF-001`，但按“先补齐功能、后统一优化”的决策不再阻塞后续功能阶段。

A3-P7M5 已完成 eager `A = k x F` + MTP 的 connector 协议、CPU/Mock 和真实 NPU 组件闭环。随后 A3-P7M6 已完成非等量 target Graph/U1/U2 + eager draft MTP 的组件闭环：A2F1、A4F2 Graph capture/replay 通过，A4F2 Graph + MTP 组合也通过。A3 因 EP4 HBM 不足不能完成 A8F4 实模 E2E，因此两阶段都只冻结 component functional snapshot，不宣称 A8F4 产品级支持、不执行 P1。完整报告见 `DEEPSEEK_V4_AFD_HCCL_P2P_MTP_UNEQUAL_COMPONENT_REPORT_ZH.md` 和 `DEEPSEEK_V4_AFD_HCCL_P2P_GRAPH_UNEQUAL_COMPONENT_REPORT_ZH.md`。A3-P7M7 随后完成 full draft Graph U1/U2；该历史里程碑固定为 1 个 MTP layer 和 `num_speculative_tokens=1`。

2026-09-08 已在提交 `71168312eab4754b9c8d0ab87021be7b8701b448` 完成第一阶段新增的
M10/M11 本机开发门禁。M10 保持单 MTP layer，将
`num_speculative_tokens` 对外上限冻结为 3，并支持 `N=1/2/3` 的 eager/Graph、U1/U2
执行；M11 将 `P2pHcclAFDConnector` 扩展为 `A=kF`、`A=F`、`F=kA` 双向整数比例，
`F=kA` 采用 Attention token 连续均衡 scatter、FFN 计算、原序 gather，token 少于 fanout
时补零占位并在返回前丢弃。CANN 9.0.0 下 A1F2/A2F4 的 `N=2/3 x eager/Graph x
U1/U2` 共 16 个 NPU 组件项全部通过；A8F8 eager/U1/N2 与 A4F8 eager/U1/N2 实模
各完成 1 条 prompt、16 个输出 token 的同栈 golden exact、正常退出和 NPU 清理。随后
10 prompt 诊断确认 MTP-off 与 native N2 在近边界 token 上稳定不同，AFD N2 的 U1/U2
输出均与 native N2 达到 10/10 exact；提交 `4f052b92407eef50d5aa363282d6a961821a6b25`
已将 A5 standalone 改为 5 类路径匹配 native control，禁止跨执行路径复用金标。无并发
正式复跑中，native N2 达到 30/30 稳定，A8F8 eager/U2/N2 F0 的 serial 10/10、真实双
stage、双 role rc、fatal 和 NPU cleanup 通过；batch 32 exact 仍为已登记的上游问题。该结果
关闭两个新增门禁的本机范围，不替代 A8F4 高 HBM/A5、双机 PD 组合、完整 F1 或性能验收。

M9 在 2026-09-04 完成双 A3 的 TP1、MTP off、Graph/U2 数据面与性能/Profile 测量：
共置 A8F8、split A8F8、split A16F8 三点均完成三轮 C32 请求，所有 Attention rank
观测到真实 U2。A16F8 的物理 A:F 是 `2:1`；U2 使每个 FFN 对应四个 peer-stage 逻辑
输入切片，不能称为物理 4:1。A16F8 对 split A8F8 出现 `+146.568%` output throughput 和
`+84.926%` token/s/NPU 的强方向信号，但 split A8F8 CV 为 `14.080%`，比较还同时改变
active NPU 与 FFN token capacity；当前只记录 measurement，不创建性能 tag。停止期 fatal
和强杀门禁不干净，也意味着 Graph/U2 F0-topology 生命周期尚未冻结。采集时 R14 overlay
还使用了现已撤回的 AllToAllV warmup split cache；动态 expert routing 无法由 shape key
固定，因此本轮不能作为 Graph 输出正确性证明，只保留请求、流水和性能观测。按 FFN
Profile wall 归一化后，Free 从 `71.977%` 降至 `38.395%`，但 Bubble 从 `18.164%` 升至
`36.954%`，相对 bubble burden 约增加 `103.45%`；更多 Attention 填充了空闲，但没有
消除剩余通信等待。

本文不替代以下文档：

- `DEEPSEEK_V4_AFD_ADAPTATION_GUIDE_ZH.md`：完整适配背景和早期里程碑；
- `DEEPSEEK_V4_AFD_BASELINE_TAGS_SUMMARY_ZH.md`：已冻结 tag 的关键改动、原因和意义；
- `DEEPSEEK_V4_AFD_HCCL_P2P_VALIDATION_REPORT_ZH.md`：新 connector 的实现边界和 A8F8 正确性证据。

本文回答后续最关键的五个问题：

1. 当前只有 A3 环境时，哪些工作可以继续完成；
2. 如何证明开启 AFD 后有真实性能收益；
3. 为最终支持 A5，现在的代码需要保持哪些可迁移边界；
4. 标准 HCCL connector 如何支持 Attention/FFN 数量不相等；
5. A5 到位后，还必须完成哪些硬件相关适配和重新验收。

## 2. 最终目标和阶段性结论

### 2.1 最终目标

在昇腾服务器上为 DeepSeek-V4 建立可部署、可复现、可回退的 Attention/FFN 分离能力，并在相同模型、相同请求和可解释的资源口径下证明 AFD 的性能收益；最终在目标 A5 服务器上完成独立的正确性、稳定性和性能验收。

“能够运行”不是最终完成条件。正式验收必须同时满足：

- 正确性：输出 token IDs 与同平台非 AFD golden 一致；
- 生命周期：冷启动、二次启动、空闲恢复、异常退出和正常停止无残留；
- 性能：吞吐收益超过运行波动，尾延迟不出现不可接受的回退；
- 资源效率：同时报告总吞吐和 `tokens/s/NPU`，不能只用更多 NPU 与单实例比较；
- 可复现性：固定源码、运行栈、参数、拓扑、数据集和 profiling 解析版本。

### 2.2 当前阶段结论

当前可以且应该继续在 A3 上开发，先完成 AFD 的通用语义和 A3 性能闭环，再到 A5 上开发硬件差异部分。

```text
A3 当前阶段
  CAMP2P U1/U2 correctness 已完成并冻结
  -> 标准 HCCL P2P connector U1/U2 correctness 已完成
  -> 目标栈标准 HCCL P2P Graph/U1 correctness 已完成
  -> 目标栈标准 HCCL P2P Graph/U2 correctness 已完成
  -> 目标栈原生 MTP 基线和协议冻结（已完成）
  -> 目标栈 HCCL P2P eager/U1 + MTP correctness（M1 已完成）
  -> 目标栈 HCCL P2P target Graph/U1 + draft eager MTP correctness（M2 已完成）
  -> 目标栈 HCCL P2P eager/U2 + MTP correctness（M3 已完成，性能缺口保留）
  -> 目标栈 HCCL P2P target Graph/U2 + draft eager MTP correctness（M4 已完成）
  -> MTP eager 非等量协议/组件闭环（M5 已完成，A8F4 实模转 A5）
  -> Graph 非等量组件闭环（M6 已完成，A8F4 实模转 A5）
  -> full draft ACL Graph U1/U2（M7 已完成，30/30 golden）
  -> TP2（M8 已完成，证据保留但不作为第一阶段门禁）
  -> Mooncake PD（M9，第一阶段进行中）
  -> 多 speculative token（M10 本机门禁已完成，单 MTP layer，最大 N=3）
  -> HCCL P2P F=kA 反向非等量拓扑（M11 本机门禁已完成）
  -> 锁定 A8F8 U1/U2 性能参照和请求矩阵
  -> HCCL P2P 双向整数比例 fan-in/fan-out 组件闭环（已完成）
  -> A8F4 实模容量预检（A3 EP4 HBM 不足，转 A5）
  -> A8F8 阻塞式 HCCL profiling、同步调度优化和公平性能验收
  -> 冻结 A3 性能基线

A5 硬件到位后
  平台审计和独立运行栈
  -> 标准 HCCL send/recv 组件验证
  -> A=F、A=kF 与 F=kA 的 U1/U2 eager 回归
  -> 等量 A/F 的 eager U1/U2 与 target Graph/U1/U2 + MTP 回归
  -> 重新选择 A/F 比例并完成独立性能验收
```

A3 验收通过只说明实现语义和 A3 性能成立，不等于 A5 已支持，也不能将 A3 性能数字直接外推到 A5。

### 2.3 两阶段交付范围与新增功能里程碑

第一阶段交付 DeepSeek-V4 AFD 功能，不包含 U3 和正式性能收益。执行口径固定 TP1；TP/SP/CP/DCP/PP、TP3、非等量 TP2 和 TP2 最大 Graph+MTP 组合不作为第一阶段目标。
等量 DP4/TP2 eager/U1 的历史功能基线继续保留，但不阻塞第一阶段冻结。

第一阶段新增两个硬门禁，2026-09-08 本机开发状态如下：

1. **多 speculative token**：模型仍使用一个 MTP layer，`num_speculative_tokens` 对外最大值已冻结为 3；`N=1/2/3` 的执行、协议、Graph key/cache、组件回归已完成，A8F8 N2 的 U1/U2 输出与路径匹配 native N2 达到 10/10 exact；A5 无并发完整 F0/F1 待执行。
2. **取消 `A >= F` 方向限制**：首版扩展到双向整数比例，即 `A=kF`、`A=F`、   `F=kA`。A8F4、A8F8、A4F8 是三个代表点；非整数比例不在本次范围。双向 rank   mapping、数据/控制面、Graph/MTP 路径与 A1F2/A2F4 组件矩阵已完成，A4F8   eager/U1/N2 实模 smoke 已通过；A8F4 仍因 A3 HBM 限制留到高 HBM/A5。

两个新增门禁的本机范围已经完成，但第一阶段总体仍未冻结。后续顺序为：

```text
本机 CPU/Mock、NPU 组件和实模 smoke（M10/M11 已完成）
  -> 高 HBM A5：A8F4 实模
  -> 双机：PD + Graph/U2 + MTP 多 token、双向拓扑代表点和路径匹配 F1
  -> 第一阶段功能 tag

第二阶段
  U3
  -> 正式吞吐、延迟、tokens/s/NPU、稳定性和 Profile 验收
  -> 性能 tag
```

`F=kA` 已冻结为 scatter/gather 语义：一个 Attention rank 将连续 token 均衡切给连续的`k` 个 FFN rank，各 FFN 均参与计算，Attention 按相同 slice 顺序 gather output。若本地token 数小于 `k`，传输层补零使每个 FFN 至少收到一个 token，gather 后仅保留真实 token。
控制面从该 Attention source 向全部 FFN peer 发送同一 stage metadata；MTP header、FFN count 投影与 Graph key/cache 使用同一 per-peer layout。非整数比例继续 fail-fast。

2026-09-08 已补齐外部验证交付物，但尚未把“脚本可执行”升级为“硬件门禁通过”：

- A5 先生成 eager MTP-off/N1/N2、target Graph + draft eager N2、target/draft Graph N3 共 5 个路径匹配 native control；standalone 固定 9 个代表点，覆盖 A8F8 基础/N1、eager U2 N2、Graph U1 N2、Graph U2 N3，以及 A4F8/A8F4 的 eager U1 N2 和 Graph U2 N3；F1 固定两次冷启动、batch 1/8/32、serial 30/30 token exact、1800 秒 idle-resume、shutdown/fatal/NPU cleanup。
- 双机 PD 固定 3 个路径匹配 no-AFD control 和 4 个 AFD 点：A8F8 N2/N3、A4F8 N3、A8F4 N3；control golden 按 Attention DP、target/draft execution、U 数和 MTP N 隔离，不能跨路径复用。
- `pd.sh` 的部署约束已同步为双向整数 A/F 和 N1-N3，矩阵按拓扑动态生成 device list 与 FFN capacity；外部执行和证据回传步骤见 `DEEPSEEK_V4_AFD_PHASE1_A5_MULTI_NODE_VALIDATION_GUIDE_ZH.md`。
- 只有上述外部原始证据通过分析后，才能关闭 A5、PD Graph 动态路由、生命周期和 F1；当前不创建第一阶段功能 tag。

## 3. 已冻结基线

### 3.1 Tag 和定位

| Tag | Commit | 定位 |
|---|---|---|
| `dsv4-afd-eager-u1-v1` | `40981475a9270c9b79ebf5cfe46d375472ee0a06` | A8F8、eager、U1 正确性基线 |
| `dsv4-afd-graph-u1-v1` | `2ed98442351d4be96edbb315a6b6c8d00805bbc4` | A8F8、`FULL_DECODE_ONLY`、U1 Graph 与生命周期基线 |
| `dsv4-afd-a3-eager-u2-v1` | `1b5d011c830d66a2516ed647064fa571667761a3` | A8F8、eager、U2 正确性、生命周期与提交态 smoke 基线 |
| `dsv4-afd-a3-sync-hccl-pre-v023-v1` | `0d2d52ae4a0e927c23db6762b0016555fcfd1baa` | 旧固定栈的非等量 HCCL、同步热路径优化与迁移前 checkpoint |

以上 tag 均不改写。HCCL 后续阶段以提交 `9578dd2` 为开发起点，每个阶段独立提交并通过全部门禁后再打新 tag，失败阶段不打 tag。

标准 HCCL P2P connector 当前位于分支 `feat/dsv4-afd-hccl-p2p`、提交 `9578dd2cb70f9f8db54673a70e8f45fde6479245`。它是后续开发起点，但在性能门禁通过前不创建性能 tag。

### 3.2 历史 A3 固定运行栈（v0.26）

| 项目 | 固定值 |
|---|---|
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.1/cann-9.0.1`（历史基线） |
| Python venv | `/mnt/workspace/code/.venvs/afd-v026` |
| vLLM | `568afb3a13806beb53bb2e6bd518269357b237c0` |
| vLLM-Ascend | `80d8c194f7584b17fe08065ea99a130916f6b0e7` |
| 模型 | `/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp` |
| 插件 | `ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd` |
| A8F8 参照拓扑 | Attention NPU 0-7，FFN NPU 8-15 |
| A8F4 候选拓扑 | Attention NPU 0-7，FFN NPU 8-11；NPU 12-15 不计入服务资源 |
| 并行 | A8F8 为 Attention DP8、FFN DP8/EP8；A8F4 为 Attention DP8、FFN DP4/EP4；TP1、PP1、CP1、DCP1 |
| 确定性 | seed 1024、temperature 0 |

该表仅用于解释 v0.26 历史结果。当前开发遵守：

- 只修改 `afd-plugin`；
- 不修改固定 vLLM 和 vLLM-Ascend 源码；
- 不混入 CANN 9.0.1/9.1.0 或 vLLM 0.22.1 工作树；
- 每次验证前运行 `tools/dsv4/check_v023_vllm_cann_runtime.sh`；
- 每次验证后保存清理完成后的 `npu-smi info`。

### 3.3 目标开发运行栈

从 tag `dsv4-afd-a3-sync-hccl-pre-v023-v1` 之后，功能和性能验证使用以下目标栈。旧栈不删除，只用于回归和解释历史性能数据；两个栈的绝对性能不可混合计算收益。

| 项目 | 目标值 |
|---|---|
| CANN | `/mnt/workspace/code/.ascend/cann-9.0.0/cann-9.0.0` |
| Python venv | `/mnt/workspace/code/.venvs/afd-v023-vllm-cann` |
| vLLM 源码 | `/mnt/workspace/code/vllm-release-v0.23.0` |
| vLLM branch/commit | `releases/v0.23.0` / `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| vLLM-Ascend 源码 | `/mnt/workspace/code/vllm-ascend-rfc-vllm-cann` |
| vLLM-Ascend branch/commit | `rfc/vllm_cann` / `3da28f9414583d2d0b672a8f06d1fae142404bda` |
| 激活脚本 | `source tools/dsv4/activate_v023_vllm_cann_runtime.sh` |
| 环境门禁 | `tools/dsv4/check_v023_vllm_cann_runtime.sh` |

迁移只修改 `afd-plugin`，目标 vLLM 和 vLLM-Ascend 工作树保持干净。兼容层同时保留 0.23.0 和冻结的 0.26.0 路径，涉及 EngineCore、MoE loader、DSV4 构造、Ascend attention metadata 和 U2 ubatch metadata 的版本差异必须由测试覆盖，不能在上游源码中打临时 patch。

主要功能证据：

```text
/mnt/workspace/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json
/mnt/workspace/validation/dsv4_afd_v023_vllm_cann_e2e_u1_smoke_backend_fix/validation_summary.json
/mnt/workspace/validation/dsv4_afd_v023_vllm_cann_e2e_u2_full_final/validation_summary.json
```

#### 3.3.1 目标栈功能与性能结论

目标栈原生模型先生成同栈 golden，再由 AFD 做严格对照，避免把上游版本造成的 token 差异误判为 AFD 错误。原生 10 条 prompt 连续 3 轮稳定；AFD eager/U2 的串行请求 30/30 token IDs 完全一致，真实双 stage、batch 1/8/32 请求结构、启动、退出和清理均通过。

性能复测固定 A8F8、C32、输入 1024 token、输出 128 token、每轮 128 请求、3 轮、temperature 0、seed 1024，U1/U2 只改变 ubatch 数：

| 指标 | 目标栈 U1 | 目标栈 U2 | U2 相对 U1 |
|---|---:|---:|---:|
| output throughput | 17.082 token/s | 12.582 token/s | -26.342% |
| output throughput CV | 4.273% | 9.640% | 均通过 10% 重复性门禁 |
| output token/s/NPU | 1.068 | 0.786 | -26.342% |
| p50 TTFT | 12159.339 ms | 14883.743 ms | +22.406% |
| p50 TPOT | 1778.871 ms | 2558.760 ms | +43.842% |
| p99 TPOT | 2001.907 ms | 2854.412 ms | +42.585% |

U1 三轮原始吞吐为 17.004、16.229、18.012 token/s；U2 为 10.892、13.682、13.171 token/s。两组均为每轮 128/128 成功，U2 日志确认 `stage_count=2`，shutdown、fatal 和 NPU cleanup 门禁通过。

```text
/mnt/workspace/validation/dsv4_afd_v023_vllm_cann_perf_u1_c32_1k128_r3/performance_summary.json
/mnt/workspace/validation/dsv4_afd_v023_vllm_cann_perf_u2_c32_1k128_r3/performance_summary.json
```

目标栈 U1 比旧栈同参数、同同步优化的 57.724 token/s 低 70.408%。这说明切换上游栈后必须重新建立绝对性能基线，不能继承旧数字；它不推翻同步优化在旧栈 P4/P7 A/B 中已证明的 +17.521% 收益。若要量化该优化在目标栈上的独立贡献，仍需在目标栈做一次开启/关闭优化的同提交 A/B。

当前性能结论仍是“eager/U2 收益失败，Graph/U2 出现强候选信号但尚未正式验收”。HCCL P2P Graph/U1、Graph/U2、eager/U1 + MTP、target Graph/U1 + draft eager MTP、eager/U2 + MTP 和 target Graph/U2 + draft eager MTP 已作为独立功能里程碑完成。M1、M2、M3、M4 的 P1 分别为 28.280、22.835、16.238 和 31.473 output token/s，都只是轻量 guard；M4 的单轮改善不能覆盖 M3/P8D 已登记的等待缺口。P8C/P8D 在保持同步 `send/recv` 的前提下完成 comm stream 和单线程 `layer -> stage`，但 eager/U2 P1 仍比 U1 回退 46.197%。Graph/U2 MTP-off P1 达到 107.189 token/s，但只有一轮且执行模式改变，不能宣称性能收益。当前按功能优先进入 M9 Mooncake PD，功能组合闭环后再以 Graph/U1、Graph/U2、MTP on/off 和同预算 native Graph 三轮 P2 建立正式结论。M9 功能开发不引入异步 HCCL，也不做 A5 性能外推。

### 3.4 当前 profiling 观察基线

现有 Graph/U1 profile：

`/mnt/workspace/validation/dsv4_afd_graph_u1_dp0_profile_ceaf4f1_20260811_205830/profile_validation.json`

| Role | 平均计算 | 平均未重叠通信 | 平均 free | 平均 stage |
|---|---:|---:|---:|---:|
| Attention DP0 | 233.670 ms | 0.226 ms | 2.852 ms | 236.749 ms |
| FFN DP0 | 80.871 ms | 2.120 ms | 5.051 ms | 85.922 ms |

FFN 的 `free` 最大值为 50.261 ms，属于需要在后续稳定采集中确认的离群点。

这些数据只用于定位优化方向和与后续 trace 对比，不构成“AFD 已有性能收益”的证据。现有采集没有完成非 AFD 同口径对照，也没有完成 U2 重叠收益验证。

## 4. 当前架构边界

### 4.1 Connector 决策

DeepSeek-V4 现在有两条明确分开的 NPU 数据通路：

- `CAMP2pAFDConnector`：已有 A3 基线，hidden 路径使用 afd-plugin 自定义 A2E/E2A 算子；
- `P2pHcclAFDConnector`：当前性能主线，hidden、FFN output 和 DSV4 input IDs 全部使用 `torch.distributed.send/recv` 的 HCCL process group。

新增 connector 的原因不是模型语义不同，而是通信实现和调度约束不同。标准 HCCL 路径不加载、不调用 `torch.ops.vllm.afd_camp2p_send_attn_output()` 或 afd-plugin A2E/E2A 自定义算子，因而更贴近 A5 目标接口，也可以独立衡量 HCCL send/recv 下的 AFD 收益。

`P2pHcclAFDConnector` 的关键边界为：

- `AFDA2FTransferPayload.input_ids`；
- 每个 stage 独立的 `afd_ids` HCCL group；
- 每个 stage 独立的 hidden/output HCCL group；
- 预分配的 NPU `int32` IDs buffer；
- Attention 到 FFN 的一次性 IDs side channel；
- FFN hash layer 的 stage 级 IDs cache 生命周期；
- Gloo 控制面先传 DP token count，FFN 再按精确 shape 投递 HCCL receive；
- eager/U2 继续调用阻塞式 `send/recv` API，但由 Attention send/receive stream、FFN receive/compute/send stream 和逐 layer/stage event 建立依赖；U1、Graph U1 和 MTP 不进入该路径；
- DSV4 HCCL eager/U2 已由插件内单 host thread 按 `layer -> stage` 推进；F2A receive event 在同一 stage 进入下一层前建立 compute 依赖，不再依赖两个 Python 线程或 `dbo_yield` 交接；
- Graph/U1 编译期间将 hidden/output send/recv 降低为 torch-npu 注册的 HCCL `_send/_recv` op；eager 和 graph 外路径继续调用标准 `torch.distributed.send/recv`；
- Graph/U1 的 input IDs side channel 保持在 capture/replay 外，避免把动态长度的控制消息固化进图。

后续性能开发以 `P2pHcclAFDConnector` 为主线。CAMP2P 保留为已冻结回归基线，不把两条数据面实现在同一个 connector 内用条件分支混合。

### 4.2 A/F 非等量支持范围

当前标准 HCCL P2P DSV4 适配已经支持以下拓扑契约；等量门禁只继续保留在 CAMP2P 路径：

当前 `P2pHcclAFDConnector` 支持范围明确限定为双向整数比例：

```text
A >= F: A % F == 0, ratio = A / F
F > A:  F % A == 0, ratio = F / A
```

`A>=F` 时，一个 FFN rank 对应连续的 `ratio` 个 Attention rank，并按 role rank 顺序聚合
和切分；`F>A` 时，一个 Attention rank 对应连续的 `ratio` 个 FFN rank，并按连续均衡 token
slice scatter/gather。A=F 继续作为 ratio=1 的兼容路径。A2F1/A4F2 与 A1F2/A2F4 已完成
真实 NPU 组件闭环；A4F8 eager/U1/N2 实模 smoke 已通过，A8F4 仍是高 HBM A5 的首个
fan-in 非等量 E2E 目标。

Mooncake PD 的 split A16F8 也是物理 `A:F=2:1`，但它采用 P8F8 + A16、TP1，不能替代
standalone A8F4/EP4 的 A5 E2E 门禁。该双机点还含 R14 历史 overlay，只能作为运行和
Profile 观测，不能反向把非等量 FFN Graph 动态路由标记为已支持。

确定性映射为：

```text
A>=F: Attention rank a -> FFN rank floor(a / ratio)
      FFN rank f       -> Attention ranks [f * ratio, (f + 1) * ratio)
F>A:  FFN rank f       -> Attention rank floor(f / ratio)
      Attention rank a -> FFN ranks [a * ratio, (a + 1) * ratio)
AFD world        -> [F0 ... F(F-1), A0 ... A(A-1)]
```

`A>=F` 时，每个 FFN rank 必须按 Attention role rank 升序接收每个 peer 的 IDs 和 hidden，
按同一顺序拼接，计算后按原始 `seq_lens` 切分返回。`F>A` 时，Attention 以确定性连续均衡
slice 向全部 FFN peer 发送 IDs/hidden；token 少于 fanout 时补零占位，FFN output 按原 slice
返回后裁掉 dummy tail。两种方向的 IDs、hidden、output、MTP header 和 Graph key 必须共享
同一 peer layout。

这个范围是 DSV4 HCCL P2P 实现的阶段性产品契约，不是 HCCL 或 A3/A5 的底层限制。
CAMP2P 等其他 connector 继续保留 `A>=F` 约束。当前显式拒绝：

- 非整数 A/F 比例：需要非均匀 peer group、负载分配和更复杂的退出协议；
- Graph/U3：现有 target Graph 只验证 U1/U2，尚未定义第三个 stage 的 capture、key 和通信顺序。

因此文档中的“支持 A/F 非等量”均特指 `P2pHcclAFDConnector`、TP1、eager 或
`FULL_DECODE_ONLY` Graph、U1/U2 下的双向整数比例，不得扩展解读为任意比例或其他
connector。

### 4.3 当前未验证能力

标准 HCCL P2P eager/U1、U2、等量 A/F 的 Graph/U1、Graph/U2、等量 A8F8 eager/U1/U2 + MTP、target/draft full Graph U1/U2，以及双向整数比例的 eager/Graph/MTP 组件协议已完成当前门禁，但还没有形成性能 tag。以下能力仍不属于 HCCL 主线基线：

- Graph/U3；
- Attention 侧 gate；
- 非等量拓扑 + full draft Graph 实模 E2E；多 speculative token 和 `F=kA` 的本机门禁
  已完成，但双机 PD 组合与完整 F1 尚未冻结；
- Mooncake PD 已有 TP1、MTP off、Graph/U2 双机实模数据面证据，但优雅退出、F1、
  eager/U1、Graph/U1、TP2 和 MTP 组合尚未冻结；R14 AllToAllV 静态 split-cache 已撤回，
  FFN Graph 动态路由正确性仍是 P0；
- sequence parallel；
- A/F 非等量完整实模 E2E（A4F8 单请求 smoke 已通过；A8F4 仍受 A3 HBM 阻塞）；
- 超出 M8 已冻结 DP4/TP2 边界的 TP，以及 PP、SP、CP 或 DCP 大于 1；
- A5 实机 HCCL P2P 验证与调优。

## 5. A3 后续开发阶段

每阶段独立提交、独立验证。后续统一采用分级验收，不在每个功能阶段重复完整性能矩阵。

### 5.0 分级验收策略

| 级别 | 使用阶段 | 必须完成 | 不在本级完成 |
|---|---|---|---|
| F0-local 本机功能门禁 | M9 及后续功能开发 | CPU/Mock、配置矩阵、单机 NPU 组件、本机可执行的实模冒烟、batch、生命周期、fatal 日志和 NPU 清理 | golden、token exact、batch-invariant、跨机拓扑和性能结论 |
| F0-topology 外部拓扑门禁 | 依赖双机或 A5 的功能 | 目标拓扑启动、请求成功、真实数据路径、batch、取消、异常、shutdown、二次启动和资源清理 | golden、token exact、正式性能结论 |
| F1 正确性冻结门禁 | 全部计划功能开发完成后 | batch-invariant 专项、路径匹配 control/AFD、跨冷启动稳定性和 30/30 token exact | 正式性能结论 |
| P1 轻量性能 guard | 已能稳定 E2E 的中间阶段 | 单一固定负载的一次候选运行，检查成功率、OOM/timeout、HBM 和数量级回退 | 三轮统计、调参、正式收益结论和常规 profile |
| P2 正式性能验收 | 功能组合闭环后的 A3-P8 | 完整公平对照、至少三轮、波动门禁、双侧 profile 和收益归因 | 不再引入新功能或同时改变多个变量 |

从 M9 开始，F0 只回答“功能路径能否工作”，不包含 golden。F0-local 是进入下一实现阶段的硬门禁；依赖双机或 A5 的 F0-topology 不阻塞相互独立的本机功能开发，但在通过前不得声明对应拓扑已交付。F1 在全部计划功能开发完成后统一执行，未通过前不得创建正确性功能 tag 或进入正式 P2。M9 以前已经冻结的历史 F0 仍按各自报告中的原门禁解释，不追溯修改。

P1 只负责尽早发现灾难性回退，不用于证明性能收益；建议固定 A8F8、C32、输入 1024 token、精确输出 128 token、128 请求，完成预热后只测 1 轮，并复用最近的同模式基线。P1 必须满足请求 100% 成功、无 OOM/timeout；若配置 U2，还必须实际观测到双 stage。若 output throughput 相对最近可比基线回退超过 20%，或 HBM/等待出现异常，则暂停扩大功能范围并先定位。单轮 P1 数据不得用于调整正式收益阈值，也不得写成“AFD 已有性能收益”。

中间阶段不固定采集 profiler。只有 P1 出现超过 20% 的回退、异常 HCCL 等待、host 发射停顿或不明 HBM 增长时，才采集 Attention DP0 与 FFN DP0 的定向 profile；保持 `TORCH_PROFILER_WITH_STACK=0`，并使用与采集记录一致的 CANN 版本解析。功能阶段修复后只重跑 F0 和 P1，不补做完整 P2。

P2 才回答最终问题“开启 AFD 和 microbatch 后是否有性能收益”。至少同时完成：

- `HCCL P2P AFD U2` 对 `HCCL P2P AFD U1`：隔离 microbatch 的增量收益，并证明 U2 实际执行双 stage；
- `HCCL P2P AFD U2` 对同总 NPU 预算的非 AFD：证明 AFD + microbatch 组合的整体收益；
- eager 与 Graph 分开归因；Graph/U2 已完成功能门禁，但正式结论仍要求 Graph/U1、Graph/U2 和 native Graph 分别完成 P2；
- MTP off 先完成主结论，MTP on/off 作为独立维度报告 acceptance rate，不能把 speculative decoding 收益归因于 microbatch。

Mooncake PD Graph/U2 的正式 P2 主矩阵固定为以下资源点。`reserved NPU` 是两台 16 卡
A3 为整轮实验保留的总卡数，`active NPU` 是该测试点实际运行模型的卡数；两种口径都要
报告 token/s/NPU，不能用 16、24、32 张 active NPU 的原始吞吐直接互相证明收益：

| ID | 数据路径与物理拓扑 | placement | active/reserved NPU | 调度容量 | 对照目的 |
|---|---|---|---:|---|---|
| C0 | PD no-AFD，`P8+D8` | split PD | 16/32 | Decode `max_num_batched_tokens=4096`，`max_num_seqs=16` | 路径匹配 control；固定相同 Mooncake、Graph、MTP off、请求与启动顺序 |
| T1 | PD + AFD，`P8+[A8F8]` | A/F 共置 | 24/32 | Attention/FFN 均为 4096，`max_num_seqs=16` | 共置 A8F8 基线 |
| T2 | PD + AFD，`[P8F8]+A8` | A/F split | 24/32 | Attention/FFN 均为 4096，`max_num_seqs=16` | 与 T1 隔离 placement |
| T3 | PD + AFD，`[P8F8]+A16` | A/F split | 32/32 | Attention 4096、FFN 8192，`max_num_seqs=32` | 物理 A:F=2:1 扩展点 |

T1、T2、T3 各自在同一源码 commit 上执行下列三种显式开关预设，共 9 个 AFD 单元；C0
不使用 AFD 流水开关，只执行一次，因此主矩阵共 10 个单元。不得用切换不同历史分支代替
显式开关，也不得只保留结果最好的预设：

| 预设 | compute overlap | hybrid DAG | Attention 三流 | FFN recv stream | FFN cross-layer | 含义 |
|---|---:|---:|---:|---:|---:|---|
| all-on | 1 | 1 | 1 | 1 | 1 | 当前五项全开实验配置 |
| V1 | 1 | 1 | 0 | 0 | 0 | 混合 DAG、side compute；通信保持 V1 parent 映射和逐层 join |
| off | 0 | 0 | 0 | 0 | 0 | 同为 Graph/U2，但关闭本轮全部流水优化 |

T3 不只是增加 Attention rank：它还把每个 FFN rank 的接收容量从 4096 提到 8192，并把
`max_num_seqs` 从 16 提到 32。T2/T3 的比例比较必须记录实际 batch tokens、Graph bucket、
HBM 和 FFN capacity headroom；若要归因容量参数本身，需在 A8F8 上补相同 8192/32 的容量
敏感性对照。否则 T3 只能回答“资源和容量共同扩展后的系统效果”，不能回答纯 A:F 或纯
多流收益。

P2 使用第 6 章的 concurrency、长度、三轮波动、延迟、HBM 和 `tokens/s/NPU` 门禁。128K 继续作为容量、TTFT 和 HBM 专项，不混入短输入 decode/microbatch 收益结论。

| 阶段 | 主要交付 | 进入下一阶段的门禁 |
|---|---|---|
| A3-P0 | 固定非 AFD、AFD eager/U1、AFD Graph/U1 的性能实验协议 | 三种部署使用同一模型、请求和统计口径；结果可重复 |
| A3-P1 | DSV4 eager/U2 的 stage IDs 与执行语义 | 已通过：相关回归 116 项，真实双 stage 执行通过 |
| A3-P2 | eager/U2 A8F8 E2E | 已通过：golden、batch、双冷启动、30 分钟空闲恢复、profile 和清理通过 |
| A3-P3 | 新增标准 HCCL P2P connector | 已通过：A1F1/U2 组件 round-trip，A8F8 U1/U2 各 30/30 golden，batch 1/8/32、退出和清理通过 |
| A3-P4 | 锁定 A8F8 性能协议和未调优参照 | 已完成：U1/U2 三轮 CV 均低于 10%，双侧 profile 已由 CANN 9.0.1 解析；U2 C32 回退 37.570%，保留为调优对象 |
| A3-P5 | HCCL P2P `A = k x F` 非等量实现 | 已通过：A2F1/A4F2 真实 NPU 组件、两 stage/两 step、不同 peer token count、聚合/切分和 close 均通过 |
| A3-P6 | A8F4 eager/U1、U2 E2E 正确性 | A3 停止：EP4 模型构造 HBM 不足；A10F5 被固定栈 EP5 专家放置拒绝；A8F4 E2E 转 A5 |
| A3-P7 | A8F8 同步 HCCL profiling、调优和公平性能验收 | 已取得 C32 +17.521% 旧栈候选收益；后续调优等待 MTP-M1/M2，之后补 C1/C8、冷服务重复与非 AFD 公平对照 |
| A3-P7T | 迁移 vLLM 0.23 + `rfc/vllm_cann` | 功能已通过；U1/U2 三轮稳定，但 U2 回退 26.342%，只冻结功能兼容性 |
| A3-P7G | 目标栈标准 HCCL P2P Graph/U1 | 已通过：A8F8、等量 A/F、`FULL_DECODE_ONLY`、30/30 golden、batch 1/8/32、两次冷启动、capture/replay、退出和清理通过 |
| A3-P7G2 | 目标栈标准 HCCL P2P Graph/U2 | 已通过 F0 + P1：两次冷启动各 30/30 golden、batch 1/8/32、双 stage、capture/replay、退出和清理通过；P1 107.189 token/s 是单轮候选信号，不创建性能 tag |
| A3-P7M0 | 目标栈原生 MTP 基线与 AFD 协议设计 | 已通过：原生 MTP 启动，30/30 token IDs 与 MTP-off 一致，acceptance 198/264，真实 key/HBM/target hidden 和 AFD phase/message 契约已冻结；未解除 AFD 门禁 |
| A3-P7M1 | HCCL P2P eager/U1 + MTP | 已通过：A8F8 等量、`num_speculative_tokens=1`、30/30 golden、proposal/accept、batch 1/8/32、五次冷启动、30 分钟空闲恢复、退出/清理和 P1 单点 guard |
| A3-P7M2 | HCCL P2P target Graph/U1 + draft eager MTP | 已通过：30/30 golden、batch 1/8/32、两轮生命周期和 P1 guard；full draft Graph 因仅 6/30 被 fail-fast 禁用 |
| A3-P8 | 正式性能验收并冻结目标栈 A3 HCCL 基线 | 第一轮未通过：async scheduling off 修复 U1 host 调度退化，但同步 U2 相对 U1 回退 45.676%；停止扩大三轮和 MTP-on 矩阵，不创建性能 tag |
| A3-P8C/P8D | comm stream 与单线程 layer-major U2 | eager 功能门禁通过，P8D 相对 P8C 提升 10.099%，但相对 U1 仍回退 46.197%；问题保持 Open，Graph/U2 候选另按 P7G2 验收 |
| A3-P7M3 | HCCL P2P eager/U2 + MTP | F0 已通过；P1 为 16.238 token/s，相对 MTP/U1 回退 42.583%；只冻结功能，性能问题保留，后续扩展按功能优先决策独立验收 |
| A3-P7M4 | HCCL P2P target Graph/U2 + draft eager MTP | 已通过 F0 + P1：30/30 golden、batch 1/8/32、双 stage、capture/replay、128/128 P1、shutdown/fatal/cleanup；31.473 token/s 仅作单轮 guard |
| A3-P7M5 | eager 非等量拓扑 + MTP | 已通过 A1F1/A2F1/A4F2 真实 NPU 组件；A8F4 实模因 A3 HBM 留到 A5 |
| A3-P7M6 | Graph 非等量拓扑 | 已通过 graph key 隔离、A2F1/A4F2 两 stage capture/replay 和 A4F2 target Graph + eager MTP 组合组件；A8F4 实模 F0 留到 A5 |
| A3-P7M7 | full draft ACL Graph | 已通过：A8F8 U1/U2 各 30/30 golden、batch 1/8/32、A4F2 full-draft Graph 组件、动态 batch、128/128 P1、shutdown/fatal/cleanup；27.510 token/s 仅作单轮 guard |
| A3-P7M8 | HCCL P2P TP2 功能基线 | 已冻结等量 A8F8、DP4/TP2、eager/U1；TP2 full-draft Graph U2 最大组合保持 fail-fast |
| A3-P7M9 | Mooncake PD + AFD | TP1/MTP off/Graph U2 双 A3 数据面、三拓扑三轮测量及双侧 Profile 已完成；一期 3 control + 4 AFD 外部矩阵和证据收集脚本已就绪，动态路由、生命周期、F1 的硬件证据待回传；10 单元公平 P2 和物理 A:F 扫描属于后续性能阶段 |
| A3-P7M10 | 多 speculative token | 本机开发门禁已通过：保持单 MTP layer，对外最大 `N=3`；`N=1/2/3` 代码回归和 A1F2/A2F4 的 N2/N3 eager/Graph U1/U2 共 16 项 NPU 组件通过；A8F8 N2 的 U1/U2 输出均与 native N2 达到 10/10 exact。A5 需无并发生成 5 个路径 control 并完成 9 点 F0/F1，双机 N2/N3 硬件结果待回传 |
| A3-P7M11 | `F=kA` 双向非等量拓扑 | 本机开发门禁已通过：完成双向整数 rank mapping、scatter/gather、dummy padding、控制面、MTP/Graph；A1F2/A2F4 组件矩阵和 A4F8 eager/U1/N2 实模 smoke 通过；A4F8/A8F4 双机矩阵与 A5 指导书已就绪，A8F4 高 HBM 和完整 F1 证据仍保留 |

### 5.1 A3-P0：固定性能实验协议

先锁定实验协议，再开始调优或改变拓扑。以下“三轮”等正式统计要求只用于 P2；P1 使用 5.0 节的单点单轮 guard。至少固定：

- prompt 集合和输入/输出长度；短请求、常规长上下文与 128K 能力点分开统计；
- batch/concurrency 阶梯，至少保留 batch 1/8/32；
- 请求到达方式和预热请求数；
- seed、temperature 和最大输出 token 数；
- HBM 利用率、最大序列数和最大 batched tokens；
- U2 threshold、HCCL buffer 和调度参数；
- 每个点至少 3 轮稳定运行；
- 冷启动数据与稳态数据分开报告；
- 服务端吞吐、客户端延迟和 NPU 资源指标使用相同测量窗口。

必须保留以下三类初始对照：

1. 非 AFD eager/Graph 基线；
2. AFD eager/U1；
3. AFD Graph/U1。

后续只逐项加入 HCCL P2P eager/U1 和 U2，避免一次改变多个变量。

建议至少冻结以下长度类型，具体 token 数写入实验 manifest，不在跑数后调整：

| 类型 | 建议用途 | 约束 |
|---|---|---|
| 短输入 + 128/512 输出 token | decode 吞吐、TPOT 和 U2 稳态收益 | 输出必须足够长，避免只测启动和 TTFT |
| 8K/32K 输入 + 128 输出 token | 常规长上下文 TTFT、HBM 与吞吐 | 各 concurrency 独立记录 |
| 128K 输入 + 32/128 输出 token | 最大上下文能力、稳定性和 HBM 边界 | 先做 batch 1；不能作为唯一性能代表点 |

128K 只有在模型 `max_model_len`、KV cache 容量和固定运行栈共同允许时才进入正式矩阵；若因容量无法运行，应记录最大可稳定长度和失败原因，不能缩短输入后仍标记为 128K。

### 5.2 A3-P1：实现 eager/U2

建议分支：

```text
feat/dsv4-afd-eager-u2
```

本阶段已经完成。A3-P1 当时只在 eager 下解除 U2 门禁，Graph/U2 保持拒绝；该门禁后来由 A3-P7G2 独立解除。Attention 按 stage 的 token slice 发送 IDs，FFN 在单主线程中按 stage 预接收并限定 cache 生命周期。

本阶段需要完成：

关键交付包括：

1. 仅对已验证配置解除 DSV4 U2 门禁，继续拒绝 ubatch 数不等于 2；
2. 使用 `ubatch_slices[*].token_slice` 对 `input_ids` 做与 hidden states 完全相同的切分；
3. stage 0 和 stage 1 分别向自己的 `afd_ids` group 发送一次 IDs；
4. 保证每个 step/stage 的 IDs 消息数和 hidden 消息数严格对应；
5. FFN layer 0 分 stage 接收，layer 1/2 引用对应 cache，layer 3 起传 `None`；
6. step 完成、异常、取消和 shutdown 都在 `finally` 中清空 cache；
7. 请求不足以触发 U2 时保留 U1 fallback；
8. U1 原有路径和两个已冻结 tag 的行为不得回退。

实现时不硬编码 NPU 0-15 或 A8F8。角色数、rank、stage 数和 token slice 继续来自配置或运行上下文，为 A5 不同卡数保留空间。

### 5.3 A3-P1 测试门禁

CPU/Mock 测试至少覆盖：

- stage 0/1 使用刻意不同的 IDs 和 token 数；
- 每 step/stage 只发送一次 IDs；
- layer 0 接收，layer 1/2 复用，layer 3 后不可见；
- 连续两个 step 使用不同 IDs，无旧缓存污染；
- `-1` padding、词表上下界、空 tensor 和超 buffer token 数；
- send/recv 消息顺序和数量不匹配时显式失败；
- 异常、取消、connector close 后 cache/buffer 状态可重新使用；
- 未触发 U2 时 U1 fallback 行为不变；
- DSV4 以外模型和 connector 的配置验证不回退。

connector 组件测试至少覆盖：

- A1F1 的 stage 0/1 `int32` IDs round-trip；
- 两个 stage 使用不同 token count；
- 连续两个 step；
- hidden 和 IDs 的顺序一致；
- 异常取消与 connector close；
- 测试完成后 HCCL group 和进程均无残留。

### 5.4 A3-P2：eager/U2 E2E

部署继续使用 A8F8、DP8/TP1/EP8 和固定插件列表，关闭 MTP、Graph 和 PD。

硬门禁：

- 复用 Milestone 0 的 10 条 golden prompt；
- 连续 3 轮，共 30/30 请求逐 token 一致；
- batch 1/8/32 请求结构和 token 结果正确；
- 两轮冷启动均通过；
- 空闲 30 分钟后恢复；
- U1 fallback 和实际 U2 都被请求覆盖；
- Attention 先停，FFN 后退出；
- 两侧进程返回码为 0；
- fatal marker 为空；
- 端口、共享内存、HCCL 进程和 NPU 占用清理完成。

通过后建议冻结：

```text
dsv4-afd-a3-eager-u2-v1
```

### 5.5 A3-P3：标准 HCCL P2P 等量基线

本阶段已经在提交 `9578dd2cb70f9f8db54673a70e8f45fde6479245` 完成。验证产物为：

```text
/mnt/workspace/validation/dsv4_afd_hccl_p2p_component_fix_20260813
/mnt/workspace/validation/dsv4_afd_hccl_p2p_u1_correctness_20260813
/mnt/workspace/validation/dsv4_afd_hccl_p2p_u2_correctness_20260813
```

A8F8 eager/U1、U2 均达到 30/30 golden，batch 1/8/32、真实双 stage、退出和 NPU 清理通过。该提交是非等量开发的回归基准，不是性能收益结论。

### 5.6 A3-P4：锁定 A8F8 性能参照

本阶段已经完成。完整报告见
`DEEPSEEK_V4_AFD_HCCL_P2P_A3_P4_PERFORMANCE_REPORT_ZH.md`。

未调优参照的主要结论为：U1/U2 在 concurrency 1/8/32 的三轮 output
throughput CV 均低于 10%；U2 在 C1/C8 的差值没有超过波动，在 C32 从 U1
的 49.118 output tokens/s 降到 30.664 output tokens/s，回退 37.570%。双侧
20-step profile 表明 FFN U2 computing 没有退化，但 free 从 20.864 ms 墑到
304.625 ms，当前阻塞式 HCCL send/recv 与共享 compute stream 没有形成有效
stage 重叠。

这一阶段不改变 connector 拓扑语义，使用提交 `9578dd2` 固定性能实验 manifest，并取得未针对结果调参的 A8F8 U1/U2 参照。目的有两个：确认 benchmark 波动范围；为后续判断 A8F4 的收益、回退和新增通信开销提供同 connector 对照。

需要完成：

1. 以同一请求矩阵分别运行 A8F8 eager/U1、U2，每点至少 3 轮；
2. benchmark 稳态窗口与 profiler 窗口分开，不能把 profiler 开销计入正式吞吐；
3. 保存 output tokens/s、TTFT、TPOT、HBM、利用率和启动参数；
4. 采集 Attention role-local DP0 与 FFN role-local DP0，确认 trace 可由 CANN 9.0.1 解析；
5. 记录 3 轮波动区间；若波动超过预先写入 manifest 的阈值，先处理负载、日志、温度或 host 抖动，不进入拓扑对比；
6. 该阶段只产生“参照报告”，不因 U2 暂未获益而阻塞 A8F4 实现。

A3-P4 的参照重复性门禁固定为每个 concurrency 的 output throughput
CV 不超过 10%。这个阈值只判断三轮数据是否可作为未调优参照；后续声称
U2、A8F4 或其他优化有收益时，收益仍必须大于对应三轮的实际波动区间，
不能只因为通过 10% CV 门禁就判定性能提升。

### 5.7 A3-P5：实现 `A = k x F` 非等量 HCCL P2P

实现从 A2F1、A4F2 组件拓扑开始，再进入 A8F4。不能只删除等量门禁，必须作为一个完整的 rank、消息和 buffer 协议修改。

#### 5.7.1 Rank 与控制面

- 将公共 P2P topology 校验语义明确为 `A >= F` 且 `A % F == 0`，错误信息不再绑定 NCCL connector 名称；
- HCCL connector 复用 `AFDRankMapping.subgroup_index`、`subgroup_ranks` 和 `ratio`，但保持数据面只使用标准 `torch.distributed.send/recv`；
- Attention `a` 的数据目标为 `floor(a / ratio)`，不能继续使用 `role_rank` 直接作为 FFN rank；
- 每个 FFN 只接收一个完整 DP metadata payload；由确定的 Attention metadata sender 发送，其他 Attention rank 不发送控制消息；
- 每个 subgroup 的第一个 Attention peer（role rank `f * ratio`）作为 FFN rank `f` 的 metadata sender，FFN 只从这个固定 source 接收；
- group 创建顺序在所有 rank 上完全一致；U1/U2 仍各 stage 独立 data group 和 IDs group；
- 非法比例、rank 越界、组初始化失败和 connector 重建必须 fail-fast 并清理已创建 group。

#### 5.7.2 数据面与 IDs side channel

每个 FFN rank 对其 Attention peers 按 role rank 升序执行固定协议：

```text
layer 0, each peer:
  recv input IDs
  recv hidden

layer 1+, each peer:
  recv hidden

after FFN compute, each peer:
  send matching output slice
```

- 控制面的 per-Attention token count 生成 `seq_lens`，同时决定每个 peer 的 receive slice；
- FFN 预分配每 stage 的聚合 IDs/hidden buffer，直接向不重叠 slice 接收，避免每层创建 peer tensor 后再 `cat`；
- FFN 的聚合 buffer 容量按本 rank 所有 peer 的 token 上限校验；A8F4 recipe 必须相应设置 FFN `max_num_batched_tokens`，容量不足时在接收前失败，不能截断或临时超配；
- 聚合 IDs 与聚合 hidden 使用完全相同的 peer 顺序和 `seq_lens`；layer 0-2 的 FFN IDs cache 保存聚合视图，layer 3 起为 `None`；
- FFN output 按 `seq_lens` 切分并发回原 Attention peer，Attention 只从其映射 FFN 接收；
- `HCCLP2PTransferState` 保存 stage、peer ranks 和 `seq_lens`，send 端不重新猜测切分；
- U2 在同一 stage 进入下一层前等待匹配 F2A output event，并增加多 Attention peer 发送时序偏斜测试，证明不会因阻塞 send/recv 交叉等待；
- step 成功、异常、取消和 close 都清空 stage cache；peer 丢失必须在配置的 HCCL 超时内失败，不能无限等待。

#### 5.7.3 实现测试门禁

CPU/Mock 至少覆盖；以下项目已经通过：

- A1F1 兼容不回退，A2F1、A4F2、A1F2、A2F4 映射正确；
- 双向非整数比例、非 HCCL connector 的 `A<F`、零/负 role 数与越界 rank 明确拒绝；
- peer token count 刻意不同，IDs/hidden 聚合顺序和 output split 精确一致；
- layer 0 IDs 每 peer 每 stage 仅一条消息，layer 1/2 复用，layer 3 后为空；
- U1、U2、连续 step、不同 token count、`-1` padding、词表边界和 buffer 上限；
- 任一 peer 发送、接收或 FFN compute 异常后无旧 cache，close 可重入；
- HCCL connector 源码仍不引用 CAMP2P A2E/E2A 自定义 op。

NPU 组件覆盖 A2F1/A4F2 历史 fan-in，以及 A1F2/A2F4 fan-out：

- BF16 hidden/output 与 int32 IDs round-trip；
- U1 和两个 stage，连续两个不同 step；
- 每个 Attention peer 使用不同 token count 和可识别 tensor 内容；
- 人为延迟一个 Attention peer，验证固定接收顺序不互锁；
- 正常 close、异常取消、二次创建和 `npu-smi` 清理。

以上门禁已通过，`P2pHcclAFDConnector` 的 `A<F` fail-fast 已解除；非整数比例以及其他
connector 的 `A<F` fail-fast 继续保留。

#### 5.7.4 配置与部署脚本

- HCCL recipe 增加独立的 Attention/FFN role 数和 device list，启动前验证数量、重复 device 和越界 device；
- 两侧 `additional_config.afd` 必须写入相同的 `num_attention_ranks`、`num_ffn_ranks` 和 connector，只有 `role` 不同；
- Attention 服务的 DP size 等于 A，FFN 服务的 DP/EP size 等于 F；
- FFN `max_num_batched_tokens` 和 receive buffer 上限必须覆盖一个 subgroup 的聚合 token 数；
- readiness、PID、日志和清理循环分别按 A/F 数量生成，不再假定两侧都是 8 rank；
- manifest 保存逻辑 rank 到物理 NPU 的完整映射，并明确记录未参与服务的 NPU；
- 默认 A8F8 recipe 行为保持不变，A8F4 使用单独配置或显式参数，不能静默改变已有基线。

#### 5.7.5 完成状态与证据

P5 的历史 fan-in 实现和组件门禁已完成，M11 在其上完成双向扩展：

- `P2pHcclAFDConnector` 接受 `A>=F` 或 `F>A` 的双向整数比例；其他 connector 保持原约束；
- FFN 按 subgroup 聚合 IDs/hidden、保留同一 `peer_slices`，并按原 peer 顺序切分 output；
- fan-out 下 Attention 均衡 scatter/gather 并处理 dummy tail；控制 metadata 发给全部 FFN peers；
- role device list、DP/EP 和 FFN 聚合容量已参数化，默认 A8F8 行为不变；
- CPU/Mock 回归及 A2F1、A4F2、A1F2、A2F4 真实 HCCL round-trip 通过。

主要 NPU 证据：

```text
/mnt/workspace/validation/dsv4_afd_a3_p5_hccl_a2f1_20260817_1055/summary.json
/mnt/workspace/validation/dsv4_afd_a3_p5_hccl_a4f2_20260817_1105/summary.json
```

### 5.8 A3-P6：A8F4 eager/U1、U2 E2E

A8F4 的计划拓扑为 Attention DP8、FFN DP4/EP4、TP1、PP1、CP1、DCP1。进入 E2E 前先完成两侧 model load 和峰值 HBM 预检：FFN rank 数减少后，每 rank 的专家权重、量化 scale/offset 和运行 workspace 会增加；若模型加载或安全 HBM 余量不通过，不得靠提高超卖比例强行进入请求测试。

正确性和生命周期门禁与 A8F8 相同，并增加：

- 参数所有权、每 FFN rank 专家分片和 peak HBM 报告；
- 10 条 golden prompt x 3 轮，30/30 逐 token 与同平台非 AFD/M0 golden 一致；
- batch 1/8/32 覆盖不同 Attention peer token count；
- U1 fallback 和真实 U2 均有 runtime evidence；
- 至少两轮冷启动、30 分钟空闲恢复和二次启动；
- Attention 全部先停、FFN 后退出；单 peer 异常时在超时内退出且无 HCCL/NPU 残留；
- A8F8 回归用同一提交重跑，确认 ratio=1 路径未回退。

A8F2 不是本阶段硬门禁。只有 A8F4 正确性通过且 FFN HBM/profile 表明仍有余量时，才以独立实验评估 A8F2。

#### 5.8.1 A3 实机预检结论

2026-08-17 的 A3 预检没有进入 golden 请求阶段：

- A8F4 eager/U1 在 FFN 模型构造时 OOM；EP4 下每个 FFN rank 持有 64 个专家，约 60.62 GiB 已激活且不足以再分配 514 MiB。降低运行时 token buffer 或 `gpu_memory_utilization` 不能解决权重构造容量；
- A10F5 用作 2:1 且更低单 rank 专家数的容量代理时，固定 vLLM-Ascend 因 256 experts 无法均匀分配到 EP5 而 fail-fast：`allocated=52, placement=51`；
- 不使用冗余专家/EPLB 绕过，因为这会改变专家放置、内存和性能语义，超出当前 topology-only 验证范围。

证据目录：

```text
/mnt/workspace/validation/dsv4_afd_a3_p6_hccl_a8f4_u1_smoke_20260817_1120
/mnt/workspace/validation/dsv4_afd_a3_p6_hccl_a10f5_u1_smoke_20260817_1125
```

因此 A3-P6 的结论是“受硬件/固定栈容量阻塞”，不是 connector 正确性失败。A3 不再尝试用超卖或改变专家放置强行完成 A8F4；该 E2E 门禁保留给高 HBM A5。

### 5.9 A3-P7：等量/非等量联合 profiling 和调优

当前 A3 实际执行范围收敛为 A8F8；A8F4 联合 profile 等 A5 完成实模加载后再恢复。第一阶段仅做同步 HCCL 优化：

- 保留阻塞式 `torch.distributed.send/recv`；
- 保留 connector 当前的 `torch.npu.synchronize()` 完成语义；
- 不使用 `isend/irecv`、额外通信 stream、后台通信线程或自定义异步 op；
- 每个 step/stage 只解析一次 token counts 和 peer slices，43 层复用同一 stage runtime；
- NPU input IDs 热路径不再通过 `min().item()`/`max().item()`做两次 device-to-host 标量读取；CPU/Mock 边界仍保留值域检查；
- 当前 step 的控制 metadata 必须先更新，再接收 IDs；close 和每次 metadata 更新都清空旧 stage layout。

第一项同步优化只缓存 stage layout/token metadata，并移除 NPU input IDs 的 `min().item()`/`max().item()`。它已通过单测和 A4F2 NPU round-trip，但两个非 profiler C32 护栏分别为 45.656 和 46.800 token/s，低于 P4 均值，不能单独形成收益结论。对应 profile 确认 Attention 侧 `aten::item/_local_scalar_dense` 总耗时从约 1211.681 ms 降到 5.498 ms，说明昂贵回读确实被消除；同时 profile 暴露 FFN 每个 layer 仍重复构造相同 stage forward context，并重复读取相同 token-count 最大值。

第二项同步优化将 FFN Ascend forward context 改为每 step/stage 构造一次、43 层复用；每层只更新 input IDs、AFD metadata 和 MoE layer index。该实现不改变任何 HCCL 消息、顺序、stream 或同步边界。10/10 golden 逐 token 通过后，最终 A8F8 eager/U1、C32、1024 输入/128 精确输出、每轮 128 请求的正式三轮结果为：

| 指标 | P4 U1 C32 | P7 同步优化 | 变化 |
|---|---:|---:|---:|
| output throughput | 49.118 token/s | 57.724 token/s | +17.521% |
| output throughput CV | 0.260% | 0.689% | 均通过 10% 门禁 |
| output token/s/NPU | 3.070 | 3.608 | +17.521% |
| p50 TTFT | 3751.984 ms | 3736.812 ms | -0.404% |
| p50 TPOT | 622.147 ms | 530.345 ms | -14.756% |
| p90 TPOT | 667.083 ms | 574.315 ms | -13.907% |
| p99 TPOT | 670.795 ms | 579.259 ms | -13.646% |

三轮原始 throughput 为 57.277、57.653、58.243 token/s；每轮均为 128/128、无请求失败，fatal、shutdown 和 NPU cleanup 门禁全部通过。该结论只覆盖高负载 C32；P7 仍需补 C1/C8、冷服务重复和同资源非 AFD 对照，未创建性能 tag。

```text
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_a4f2_20260817_1135/summary.json
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_u1_guard_c32_retry_20260817_1150/performance_summary.json
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_u1_profile_20260817_120056/performance_summary.json
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_context_cache_golden_20260817_1315/validation_summary.json
/mnt/workspace/validation/dsv4_afd_a3_p7_sync_hccl_context_cache_c32_formal3_20260817_1345/performance_summary.json
```

部署、采集、解析必须分开执行：

1. 先确认目标服务已使用预期 commit 和参数启动；
2. 再启用 profiler 采集 Attention role-local DP0 和 FFN role-local DP0；
3. 最后使用与采集匹配的 CANN 9.0.1 工具解析原始目录；
4. 解析异常时保留原始产物，不覆盖或删除失败现场。

固定：

```bash
export TORCH_PROFILER_WITH_STACK=0
```

建议补充稳定的 `record_function` 区段：

- Attention compute；
- A2F send/receive；
- FFN compute；
- F2A send/receive；
- ubatch split、wait 和 merge；
- per-peer receive、aggregate view、output split 和 per-peer send。

需要统计：

- Attention/FFN 计算时间；
- A2F、F2A 延迟及未重叠部分；
- Attention/FFN overlap；
- FFN wait、free/bubble；
- stage 0/1 不均衡程度；
- 同一 FFN subgroup 内各 Attention peer 的到达偏斜、聚合等待和 FFN batch 放大收益；
- host `.item()`、同步 memcpy 和控制面开销；
- 每侧峰值 HBM、NPU 利用率和启动时间。

第一轮调优矩阵可以从以下值开始，但它们不是产品固定值：

```text
Topology: A8F8（A3）；A8F4（高 HBM A5）
Mode: U1, U2
U2 threshold: 16, 32, 48, 64, 96, 128
HCCL_BUFFSIZE: 在固定请求矩阵下做小范围扫描
```

异步 HCCL 不在当前矩阵中。若未来启用，必须另立里程碑并重新验证消息生命周期、异常取消、stream 同步、buffer 复用和 shutdown，不能把同步路径的正确性结论直接外推到异步实现。

调参顺序固定为 topology -> U1/U2 -> threshold -> HCCL buffer。每轮只改变一个维度；A8F8 和 A8F4 可以有不同最优 threshold，但必须共享同一请求、预热、测量窗口和正确性门禁。

性能热路径中的高频 warning、tensor `.tolist()` 和仅用于调试的 key 构造应降到 debug 或显式开关下。任何日志调整都必须保留错误和生命周期门禁所需信息。

### 5.10 A3-P7G：标准 HCCL P2P Graph/U1

本阶段已经完成，且没有沿用 CAMP2P Graph 的实现结论。P7G 冻结时的支持边界为 A8F8 等量拓扑、`FULL_DECODE_ONLY` 和 U1；Graph/U2 后来由 P7G2 独立解除，Graph 非等量随后由 M6 解除组件级门禁，Graph/U3 继续显式拒绝。

Graph 路径只在 `torch.compiler.is_compiling()` 时调用 torch-npu 注册的 HCCL `_send/_recv` op。shape 参数传 `None`，由输入/输出 tensor 决定 shape，避免符号 token 维度在编译时被专门化为首次请求长度。非编译路径仍是标准阻塞式 `torch.distributed.send/recv`。DSV4 input IDs 在 graph 外通过一次性 HCCL side channel 预传，FFN 的 layer 0/1/2 复用和 layer 3 后清理语义不变。

当前 A8F8 功能门禁结果：

- 10 条 prompt 连续 3 轮，共 30/30 token IDs 与目标栈原生 golden 一致；
- batch 1/8/32 请求结构和生成有效性通过；
- capture size 1/2/4/8 完成，Attention 8 个 rank 均有 `Replaying aclgraph` 证据；
- 独立 smoke 与完整门禁构成两次成功冷启动；
- Attention 先停、FFN 后停，fatal 日志和 NPU process table 清理门禁通过。

验证产物：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u1_smoke_20260818_112726
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u1_full_20260818
```

Graph/U1 功能基线已经由 tag `dsv4-afd-v023-hccl-graph-u1-v1` 冻结，A3-P7M1/M2 也已分别完成 eager 和 target Graph 下的 MTP 功能门禁。Graph/U2 已由下一小节独立解除门禁。当时不得把 U1/U2 的通过外推为 U3、full draft ACL Graph 或非等量 Graph 已支持；后续 M6 已解除非等量组件门禁，M7 已解除 full draft Graph U1/U2 门禁，U3 仍拒绝。

#### A3-P7G2：标准 HCCL P2P Graph/U2

本阶段已经完成，范围严格限定为 `P2pHcclAFDConnector`、A8F8 等量拓扑、
`FULL_DECODE_ONLY` 和两个 microbatch。Graph warmup 和真实 capture 都改为单线程
`layer -> stage 0 -> stage 1`，与 FFN connector loop 保持完全相同的 HCCL op 顺序；两个 stage
合并到同一个 NPUGraph。编译和真实 capture 使用 graph-visible HCCL `_send/_recv`，graph 外
仍是标准同步 `torch.distributed.send/recv`，未引入异步 HCCL。

DSV4 DSA compressed attention 显式跳过不适用的普通 MLA FIA workspace 预分配，但正常 ACL
Graph capture 仍启用。子 forward context 保存父 FULL Graph 状态，connector 的 stream/event
依赖在 capture 主 stream 上闭合，避免 stage-major 两线程造成 communicator 顺序不一致和
capture stream 未加入。

F0 连续两次冷启动均为 30/30 golden token exact，batch 1/8/32 有效，双 stage、capture
size 1/2/4/8、fatal、两侧 rc=0 和 NPU cleanup 全部通过。startup 分别为 414.229s 和
400.233s。验证产物：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_f0_20260820
```

P1 固定 C32、输入 1024、精确输出 128、128 请求，单轮 128/128 成功，output throughput
107.189 token/s、p50 TPOT 217.268 ms；Attention/FFN 最大 HBM 分别为 61,819/44,391 MiB。
相对 eager/U1 单点 30.615 token/s 高 250.116%，但执行模式不同且只有一轮，所以只记录为强
候选信号，不创建性能 tag、不宣称 AFD + microbatch 已有正式收益，也不固定采集 profiler。

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_u2_p1_20260820
```

完整实现原因、失败路径和证据见
`DEEPSEEK_V4_AFD_HCCL_P2P_GRAPH_U2_VALIDATION_REPORT_ZH.md`。eager/U2 + MTP 已在 M3
完成，target Graph/U2 + draft eager MTP 也已由 M4 独立解除门禁。在该阶段 Graph/U3、Graph
非等量和 full draft Graph 仍 fail-fast；后续非等量组件和 full draft Graph 已分别由 M6、M7
解除对应门禁，Graph/U3 仍拒绝。M3/P8D 的同步等待缺口继续登记，待功能组合闭环后统一执行
Graph/U1、Graph/U2、MTP on/off 和同预算 native Graph 的三轮 P2。

### 5.11 A3-P7M：HCCL P2P MTP 功能路线

MTP 是必交付能力，但必须作为独立功能路线实现，不能只删除 `speculative_config` 门禁。M1 已在 AFD DSV4 wrapper 中补齐角色化 MTP loader、target hidden-state buffer、proposer/verify 调用和跨 AF 的 MTP phase；目标 vLLM 0.23 与 vLLM-Ascend 的 MTP proposer、verify/rejection sampler 和 DeepSeek MTP loader 继续作为上游语义来源，afd-plugin 只实现角色所有权和 HCCL 数据面。

#### A3-P7M0：原生基线和协议冻结

本阶段已经完成。完整报告见
`DEEPSEEK_V4_AFD_MTP_M0_NATIVE_BASELINE_REPORT_ZH.md`，验证产物为：

```text
/mnt/workspace/validation/dsv4_v023_vllm_cann_native_mtp_m0_20260818
```

同一 vLLM 0.23 + `rfc/vllm_cann` 目标栈的非 AFD MTP 以 `num_nextn_predict_layers=1`、`num_speculative_tokens=1` 运行成功。10 条 prompt 连续 3 轮内部稳定，并与同栈 MTP-off 基线达到 30/30 最终 token IDs 一致；服务端统计累计 drafted 264、accepted 198，acceptance rate 为 75.0%。每 rank 权重由 MTP-off 的 44.4493 GiB 增至 47.4348 GiB，可用 KV cache 由约 7.76-7.77 GiB 降至约 4.74 GiB。

本阶段必须冻结以下契约：

- `mtp.*` checkpoint key 的 Attention/FFN 所有权，包括 weight/scale/offset 同角色约束；
- target hidden states 的生产者、消费者、shape、dtype、有效期和清理点；
- draft、target verify、rejection sampling 的调用顺序，以及哪一侧拥有 embedding、LM head 和 sampler；
- 若 MTP block 内的 MoE 继续保持严格 AF 分离，则为 MTP virtual layer 定义独立 layer/phase 标识、IDs、hidden 和 output 消息；不得把 MTP MoE 权重悄悄复制到 Attention role 来绕过协议；
- prefill、普通 decode、draft decode、verify 和 bonus token 的 token count/position 映射；
- 失败、请求取消和 shutdown 时 draft state、IDs cache、hidden buffer 与 HCCL group 的清理语义。

真实 checkpoint index 的 2,347 个 MTP key 已分类为 Attention 32、FFN 2,315。FFN 规则固定为 `mtp.<layer>.ffn.*`，其余 MTP key 归 Attention；weight/scale/offset 同角色。target hidden buffer 固定为 Attention 生产和消费的当前 step 有效前缀，当前 capacity 为 `[1024, 16384]` BF16。

MTP 使用独立 `phase=mtp`，不伪装成普通 decoder layer。M0 根据 target hidden 的 HC residual shape 暂定了 `[T,4,4096]` 传输；M1 真实执行证明该假设的边界位置不对：MTP Attention role 在远端 MoE 前已经执行 HC collapse，原生 MoE 只接受二维输入，所以线上 HCCL 边界修正为发送 post-HC `[T,4096]` hidden、接收同 shape output。connector 对 pre-HC 三维 tensor 显式拒绝，避免协议再次漂移。

当前原生 draft MoE 以 `is_draft_layer=True` 使用学习式 gate，不向 FFN 发送 IDs。MTP header 通过已有 IDs HCCL group 先发送 magic、speculative step、DP size 和每个 DP 的 token count，随后发送 hidden；返回方向只发送 output。若未来上游改为 hash router，必须新增 IDs side channel 并重新验收。该修正没有修改上游 vLLM/vLLM-Ascend。

#### A3-P7M1：eager/U1 + MTP

本阶段已经完成，范围严格限定为 `P2pHcclAFDConnector`、A8F8 等量拓扑、eager、U1、MTP method 和 `num_speculative_tokens=1`。实现包括：

- 按冻结的 key 分类加载 MTP 权重，保持 checkpoint iterator one-shot；
- 恢复并验证 target hidden-state buffer，不跨 step/request 复用旧内容；
- 扩展 HCCL payload/metadata，显式携带 MTP phase、speculative step 和 token count；MTP 学习式 gate 不消费 input IDs；
- 为 MTP virtual layer 建立确定的 header/hidden/output 消息顺序和独立二维预分配 buffer；
- proposal 数、accepted 数、bonus token 与最终输出 token IDs 可审计；
- 非 MTP、现有 eager/U1/U2 和 Graph/U1 回归不退化。

CPU/Mock 已覆盖 key 分类、权重归属、one-shot iterator、payload、cache、消息计数、二维 shape guard 和 fail-fast。A8F8 E2E 使用同栈 MTP-off golden，10 条 prompt 连续 3 轮共 30/30 最终 token IDs 一致；batch 8/32 分别 8/8、32/32 token exact。golden 运行累计 drafted 264、accepted 198，acceptance rate 75.0%。smoke、golden、batch 和 P1 构成四次成功冷启动，Attention 先停、FFN 后停、fatal 日志和 NPU 清理均通过。

P1 固定 C32、输入 1024、精确输出 128、128 请求，单轮 128/128 成功，output throughput 为 28.280 token/s，acceptance rate 为 85.70%，Attention/FFN 最大观测 HBM 分别为 59,650/44,253 MiB。相对最近同模式 MTP-off 三轮均值 17.082 token/s 没有灾难性回退；由于只有一轮且变量包含 MTP，本结果不能作为正式收益结论。验证产物和完整变更原因见 `DEEPSEEK_V4_AFD_HCCL_P2P_MTP_M1_VALIDATION_REPORT_ZH.md`。

#### A3-P7M2：target Graph/U1 + draft eager MTP

本阶段已经完成，完整报告见 `DEEPSEEK_V4_AFD_HCCL_P2P_MTP_M2_VALIDATION_REPORT_ZH.md`。支持边界为等量 A8F8、target `FULL_DECODE_ONLY`、draft `enforce_eager=true`、U1、1 个 MTP layer 和 `num_speculative_tokens=1`。普通 decoder graph key 区分 MTP on/off、speculative token 数、draft execution 和 DP token shape；普通 decoder 的 hash-router IDs 继续在 graph 外预传，MTP 学习式 gate 不消费 IDs。

Attention 和 FFN 在 target capture 时都只执行 target。原因是上游同一次 dummy call 会在 target 后继续执行 drafter，而标准阻塞式 HCCL 的 eager draft 无法跨越两侧 target graph context 的同步边界。capture 期间临时省略 eager drafter 不会漏 capture，因为 draft 明确不使用 ACL Graph；warmup 和在线请求仍完整执行 target、proposal、draft、verify。在线路径固定为 target graph replay 后执行当前 step 的 eager MTP，header 和 hidden 每 step 重新接收，不缓存旧 draft state。

完整 draft ACL Graph 曾作为探索路径实测：graph capture 和请求均可完成，但 30 条 golden 仅 6 条匹配，batch 1/8/32 的 token exact 分别为 1/1、2/8、13/32。该结果不能归因于 HCCL 死锁，也不能作为支持能力；当前 feature validation 显式要求 `draft enforce_eager=true`，拒绝 full draft Graph。

当前门禁结果：

- 目标栈 golden 10 条 prompt 连续 3 轮，30/30 最终 token IDs 一致；
- batch 1/8/32 的请求结构、输出长度和错误门禁通过；
- 两次连续冷启动均逐 token 匹配，startup 为 392.219s 和 424.199s；
- 两轮 shutdown、fatal 日志和 NPU cleanup 均通过；
- P1 C32、1024/128、128 请求为 128/128 成功，22.835 token/s；相对 M1 eager P1 的 28.280 token/s 回退 19.253%，未越过 20% 暂停阈值，但已接近边界；
- P1 最大观测 HBM 为 Attention 60,030 MiB、FFN 44,527 MiB，无 OOM/timeout。

验证产物：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m2_correctness_20260818_2310
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m2_lifecycle_20260818_2316
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m2_p1_20260818_2328
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m2_final_smoke_retry2_20260819
```

M2 只冻结功能基线，不创建性能 tag。随后 A3-P8/P8C/P8D 已完成 eager MTP-off 的第一轮性能定位，Graph/U2 也已完成独立 F0 + P1。target Graph + eager draft 的 19.253% 回退仍是后续 P2 的重点定位项；在三轮和公平对照完成前不得宣称 Graph 或 MTP 带来收益。

#### A3-P7M3：eager/U2 + MTP

本阶段功能实现和 F0 已完成。范围严格限定为 `P2pHcclAFDConnector`、A8F8 等量拓扑、target/draft eager、target decoder U2、一个合并后的 MTP U1 phase、1 个 MTP layer 和 `num_speculative_tokens=1`。

target U2 不能按 token 中点机械切分。MTP verify 的同一请求包含相关的 target/draft token；若二者落入不同 stage，稀疏/量化 Attention 会使用不同 kernel shape，真实 smoke 曾在第 12 个输出 token 开始偏离 golden。当前实现按请求边界切分 target decoder，保留同一请求的 token 对；8 个 DP rank 通过一次五行 all-reduce 同步总 token、padding、graph mode、decode 类型和 stage 0 token count，并向 FFN 发送两个 stage 的精确 per-rank token 向量。

若任一 DP rank 不能形成两个非空请求边界 stage，所有 rank 全局回退 U1，避免不同 rank 使用不同 stage 数。DP8 下启用真实 U2 至少需要 16 个并发请求；batch 1/8 和串行 golden 使用 U1 fallback，batch 32 与 P1 实际执行双 stage。target decoder 完成两个 stage 后，Attention 按 stage 顺序把 pre-HC residual 合并到当前 step 的 target hidden buffer；上游 proposer 随后只执行一次合并后的 MTP phase，MTP phase 不再拆 U2。

F0 结果为 30/30 串行 token exact，batch 1/8/32 均有效；batch 32 的 8 个 Attention rank 都记录两个 `(4,4,4,4,4,4,4,4)` stage。独立冷启动 404.230s，两侧返回码 0，fatal 日志和首次 NPU cleanup 通过。P1 固定 C32、输入 1024、精确输出 128、128 请求，128/128 成功，output throughput 16.238 token/s、p50 TTFT 7724.851ms、p50 TPOT 1746.781ms；Attention/FFN 峰值 HBM 为 60,539/44,819 MiB。

P1 相对 M1 eager/U1 + MTP 的 28.280 token/s 回退 42.583%，越过 20% 暂停线；相对最近 MTP-off layer-major U2 的 16.472 token/s 仅回退 1.423%，说明主要缺口仍是 target U2 的通信、stage 调度和等待成本，不能归因于新增 MTP phase。M3 只创建功能 tag，不创建性能 tag。该性能问题继续通过 `P8D-PERF-001` 跟踪；后续按功能优先决策继续 M4，但没有把暂停线取消或把单轮数据升级为正式性能结论。完整证据见 `DEEPSEEK_V4_AFD_HCCL_P2P_MTP_M3_VALIDATION_REPORT_ZH.md`。

MTP 扩展按“target Graph/U2 + draft eager -> eager 非等量拓扑 -> 非等量 target Graph + draft eager”的顺序分别立项；M4、M5、M6 已依次完成，full draft Graph 也已由 M7 独立完成。后续功能里程碑允许在保留 P1 风险记录的前提下继续，但 P2 和性能 tag 仍要求可比配置回到门禁内。M10 已完成多 speculative token 的本机门禁，双机 PD/F1 组合继续后移。

#### A3-P7M4：target Graph/U2 + draft eager MTP

本阶段已经完成，范围严格限定为 `P2pHcclAFDConnector`、A8F8 等量拓扑、target `FULL_DECODE_ONLY`、两个 target microbatch、draft `enforce_eager=true`、1 个 MTP layer 和 `num_speculative_tokens=1`。MTP phase 继续为合并后的单次 eager phase，不进入 U2，也不抓 draft ACL Graph。

实现解决了两个只在 Graph/U2 + MTP 组合中出现的问题：

1. 启动抓图时，最小 U2 capture shape 可能回落到已存在的 U1 graph key。Attention 会 replay 旧图；FFN 必须同步 replay 对应 graph，而不能因 key 重复直接跳过，否则 Attention 的 HCCL send 没有匹配 receive；
2. 在线新 stage shape 只能由 Attention wrapper 在模型调用内发现，此时 FFN 已按 step metadata 选择 eager。若 Attention 单边动态抓图，会造成 A/F 模式不一致。当前只允许同步 startup dummy run 创建 U2 graph key；在线 key miss 将 target 整步降为 eager/U2，已捕获 key 继续 Graph/U2。

F0 达到 30/30 串行 golden token exact，batch 1/8/32 有效，batch 32 观测真实双 stage；8 个 Attention rank 完成 capture/replay，shutdown、fatal log 和 NPU cleanup 通过。验证产物：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m4_f0_20260820_182146
```

P1 固定 C32、输入 1024、精确输出 128、128 请求，结果为 128/128 成功、0 failed、31.473 output token/s、p50 TTFT 5939.420 ms、p50 TPOT 1144.432 ms，MTP acceptance rate 84.51%；Attention/FFN 峰值 HBM 为 61,437/45,119 MiB。日志、双 stage 和 cleanup gate 全部通过：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_m4_p1_final_20260820_192421
```

这是单轮 P1 功能 guard，不是性能基线。不得把它与 M3 或 MTP-off Graph/U2 的单点直接解释为 Graph、MTP 或 microbatch 收益；正式结论仍由三轮 P2、MTP on/off、Graph/U1/U2 和同预算 native 对照给出。完整证据见 `DEEPSEEK_V4_AFD_HCCL_P2P_MTP_M4_VALIDATION_REPORT_ZH.md`。

#### A3-P7M5：eager 非等量拓扑 + MTP 组件闭环

本阶段的 A3 component functional snapshot 已完成。范围限定为
`P2pHcclAFDConnector`、eager、`A >= F`、`A % F == 0`、target decoder U1/U2、
一个合并的 MTP stage 0、1 个 MTP layer 和 `num_speculative_tokens=1`。该阶段冻结时
Graph 非等量仍 fail-fast；该限制随后由 M6 独立解除。

MTP header 的 token count 输入仍是 Attention world 的 A 长度向量。每个 Attention rank
在发送前按连续 subgroup 投影为 F 长度汇总向量，同时保留 header 中的本地 token count；
映射到同一 FFN rank 的所有 Attention peer 都发送相同的 F 长度汇总向量。FFN 按 role rank
升序接收 `ratio` 个 header，要求 magic、speculative step 和汇总向量完全一致，并检查 peer
本地 token 总和等于汇总向量中当前 FFN rank 的值。随后 FFN 使用相同 peer 顺序接收 hidden、
执行一次合并 MoE，再复用 receive state 按原始 `seq_lens` 拆分 output。

header 产生的 MTP layout 只能消费一次：重复 header、没有 header 的 hidden receive、跨 step
残留、peer header 不一致和 subgroup 总数不一致都会失败；正常 hidden receive 在进入通信前
取走 layout，connector close 会清空未消费状态。协议仍只使用标准同步
`torch.distributed.send/recv`，没有引入 CAMP2P 自定义 op、`isend/irecv` 或后台通信线程。

验证结果：

- CPU/Mock：connector 59/59、recipe 58/58、feature validation 44/44；
- A1F1 等量回归：两个 stage、两个 step、MTP header/hidden/output、close 全部通过；
- A2F1：FFN 每 step 聚合两个 Attention peer，两个不同 MTP token 分布均通过；
- A4F2：两个独立 FFN subgroup 同时完成 fan-in/fan-out，六个进程返回码均为 0；
- 首次 A2F1 的测试识别值 `1001` 在 BF16 中舍入为 `1000`，导致测试断言失败；协议收发已完成。
  修正为 BF16 可精确表示的值后通过，同时验证工具增加任一 worker 失败即取消其余 worker，
  避免错误路径等待完整 timeout。

验证产物：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_equal_a1f1_20260820_203703/summary.json
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_unequal_a2f1_20260820_203437/summary.json
/mnt/workspace/validation/dsv4_afd_v023_hccl_mtp_unequal_a4f2_20260820_203549/summary.json
```

本阶段没有在 A3 运行 A8F4 实模 golden、batch、生命周期或 P1。原因仍是 A3-P6 已确认的
FFN EP4 专家权重峰值 HBM 不足，而不是 connector 协议未通过。A5 到位后必须先跑 A8F4
实模 F0，生成 A5 同平台原生 MTP golden，再决定是否进入 A8F4 P1/P2。因此 M5 只允许创建
带 `component` 的功能 tag，不能创建 `perf` tag，也不能宣称非等量 MTP 已有性能收益。

#### A3-P7M6：Graph 非等量拓扑组件闭环

本阶段已完成 `P2pHcclAFDConnector` 的 `A = k x F`、`FULL_DECODE_ONLY`、U1/U2
Graph 功能实现。已有 multi-peer HCCL 收发循环保持不变，核心修复是 FFN Graph key 从
F 长度聚合 token 数改为 A 长度的精确 peer token layout。原因是 `[2,3,4,5]` 与
`[1,4,3,6]` 的 FFN 聚合都为 `[5,9]`，但每个 `_recv/_send` 的 slice shape 不同；若只按
聚合值复用 Graph，会用错误的 peer shape replay。

Graph connector 初始化现在显式加载 torch-npu 2.10.0.post2 自带的
`npu_define::_send/_recv` 注册模块，不再依赖编译器先加载模块的偶然顺序。input IDs 仍在
Graph 外一次性传输；hidden/output 在一个 NPUGraph 内按稳定 peer rank 和 slice 顺序捕获。
eager、CAMP2P、MTP header 协议和同步 HCCL API 均未改变。

验证结果：

- CPU/Mock 覆盖同聚合不同 peer layout 的 key 隔离、A2F1 Graph feature/recipe 接受、
  multi-peer Graph op 顺序和 Attention-side gate 继续拒绝；
- A2F1：两 stage、两 eager step、Graph capture、更新静态输入后的 replay、Graph 外 IDs、
  双 Attention peer fan-in/fan-out 和全部进程 close 通过；
- A4F2：两个 FFN subgroup 同时完成不同 peer shape 的 capture/replay，六个进程返回码为 0；
- A4F2 + MTP：target Graph capture/replay 与每 step 的 eager MTP phase 组合通过。

验证产物：

```text
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_unequal_a2f1_20260821_m6_retry3/summary.json
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_unequal_a4f2_20260821_m6/summary.json
/mnt/workspace/validation/dsv4_afd_v023_hccl_graph_mtp_unequal_a4f2_20260821_m6/summary.json
```

本阶段仍是 component functional snapshot。A3 无法加载 A8F4 FFN EP4 实模，因此没有
30/30 golden、batch 1/8/32、生命周期或 P1/P2 结论；这些门禁保留到高 HBM A5。
下一功能阶段曾规划为 full draft ACL Graph；该阶段已经由 M7 完成，已知 6/30 问题提升为
U1/U2 各 30/30。M8 TP 也已冻结；当前直接进入 M9 Mooncake PD。
SP/CP/DCP 和 PP 统一后移到 M9 之后，不作为 PD 功能开发的前置条件。

#### A3-P7M7：full draft ACL Graph

本阶段已经完成 A8F8 target/draft `FULL_DECODE_ONLY` Graph U1/U2 实模闭环，并完成 A4F2
非等量 full-draft Graph 组件验证。target 和 MTP virtual layer 使用独立 FFN Graph cache；
MTP header 在 Graph 外由 control-plane metadata 写入稳定 NPU buffer，Graph 内只记录标准
HCCL header/hidden/output 收发。

live target 完成全部请求时可能不执行 proposer，因此 Attention 只在 draft runnable 真正调用时
发送 phase marker。marker 同时携带本轮实际 Graph replay/eager 模式；未命中已捕获 draft
descriptor 时两侧共同 eager，禁止 Attention 单边动态 capture。FFN MTP Graph key 将每个
Attention peer 的 live count 向上归一到实际 capture bucket，例如 `6 -> 8`，非等量拓扑仍
逐 peer 保留 HCCL slice shape。

F0 的 U1/U2 均达到 30/30 串行 golden，batch 1/8/32 有效，U2 观测两个 target stage；
A4F2 的 6 个 worker 完成 target Graph 和 MTP Graph capture/replay。正式 P1 为 C32、
1024/128、128 请求，128/128 成功，27.510 output token/s，MTP acceptance rate 86.75%，
双 stage、fatal、shutdown 和 cleanup 全部通过。相关回归 316/316 通过。

完整报告见 `DEEPSEEK_V4_AFD_HCCL_P2P_MTP_FULL_DRAFT_GRAPH_VALIDATION_REPORT_ZH.md`。
M7 只创建功能 tag，不创建性能 tag；A8F4 实模 full-draft Graph 继续留到 A5。

#### A3-P7M8：HCCL P2P TP2 功能基线

M8 已完成并达到进入 M9 的既定功能门禁。标准 HCCL P2P connector 现在接受 TP1/TP2；
TP2 首版限定等量物理 A/F、两侧相同 TP2 和 `physical ranks = DP x TP`。A8F8 部署由
原来的两侧 DP8/TP1 变为两侧 DP4/TP2，Attention rank `i` 仍只与 FFN rank `i`
交换 IDs、hidden 和 output。CAMP2P TP2、非等量 TP2 和 TP3 继续 fail-fast。

control payload 显式携带 TP，FFN 从完整 payload 更新 Graph/warmup/MTP phase，MTP 的
逻辑 DP token count 按 TP rank 复制到物理 peer。部署入口显式选择
`flashinfer_all2allv`，以遵守固定 vLLM-Ascend 栈在 `enable_sp=False` 时关闭隐式
sequence-parallel MoE 的约定；本阶段没有启用 SP。

验证结果：

- A2F2、DP1/TP2 的 eager、eager + MTP、full Graph/MTP 组件 capture/replay 全部通过；
- 原生 NPU0-7、DP4/TP2 的 10 条 prompt x 3 轮结果稳定，形成 TP2 golden；
- AFD A8F8、DP4/TP2、eager/U1 实模 F0 达到 30/30 token IDs 精确一致；
- batch 1/8/32、fatal log、Attention 先停、FFN 后停和 NPU 清理门禁通过。

补充的 `TP2 + FULL_DECODE_ONLY + U2 + full-draft MTP` 最大组合没有通过：Attention
完成 Graph capture/replay 且观测到两个 stage，但 FFN Graph warmup 触发 AICore
507015/非法内存访问。该精确组合已 fail-fast，不纳入 M8 发布边界，也不得写成已经支持。
M8 冻结的是 TP2 eager/U1 实模基线及 TP2 connector/control 契约，不是全部功能的 TP2
笛卡尔积。

完整报告见 `DEEPSEEK_V4_AFD_HCCL_P2P_TP2_VALIDATION_REPORT_ZH.md`。M8 只创建功能
tag，不创建性能 tag。下一功能阶段直接进入 M9 Mooncake PD；SP/CP/DCP 和 PP 后移。

### 5.12 A3-P8：同步 HCCL 第一轮性能门禁

第一轮已完成同预算 native 双实例三轮基线、AFD U1/U2 调度 A/B 和双侧 profile。native DP8 x 2 的 C32 三轮均值为 116.952 token/s，CV 4.058%。关闭目标 vLLM async scheduling 后，AFD U1 C32 单轮为 30.615 token/s，相对旧的 auto/on 三轮均值 17.082 提升 79.228%；Attention `Preparing` 从约 1066.969 ms 降到 1.411 ms，证明 host/DP 调度退化已被定位。

相同 async scheduling off 配置下，U2 C32 单轮只有 16.631 token/s，比 U1 回退 45.676%，虽然双 stage 和清理门禁均通过。profile 显示 U2 的 Attention communication 增加 121.5%、bubble 增加 110.0%，FFN stage 增加 115.1%；dispatch、combine 和 grouped matmul 调用数均翻倍。当前两个 microbatch 使用同一 compute stream，阻塞式 HCCL send/recv 没有与另一 microbatch 的计算重叠，所以不能用继续调 threshold 或重复三轮弥补结构性串行。

因此 P8 状态为“第一轮未通过”：不创建性能 tag，不继续 MTP-on P2，也不把功能通过写成性能收益。完整数据、归因和证据目录见 `DEEPSEEK_V4_AFD_A3_P8_SYNC_HCCL_PERFORMANCE_REPORT_ZH.md`。

后续有两个互斥选择：

1. 若继续保持完全阻塞式 HCCL，则冻结 async scheduling off 的同步 U1 作为功能/调试基线，停止 U2 性能路线；
2. 若目标仍是 AFD + microbatch 性能收益，则新立“标准 HCCL P2P NPU 通信流重叠”里程碑，保持 HCCL API 不变，引入 comm stream、event 依赖和双 microbatch 状态机；先以 trace 证明真实 overlap，再恢复 P2、Graph 和 MTP 性能矩阵。

### 5.13 A3-P8C：同步 API 的 NPU comm stream

P8C 已完成第一版实现，通信接口没有改成异步：hidden/output 仍由 `torch.distributed.send/recv` 传输，不使用 `isend/irecv`、后台通信线程或自定义异步 HCCL op。新增范围严格限定为 eager/U2 decoder：

- Attention 建立 A2F send stream、F2A receive stream，以及逐 layer/stage 的 compute/send/receive event；
- FFN 建立 receive、compute、send stream，下一层同 stage 的 receive 等待上一层 send event；
- DSV4 input IDs 在两个 ubatch 线程启动前按 stage 一次性预传，避免两个线程争抢同一 IDs side channel；
- receive 使用独立 output tensor，避免 send 尚未完成时原地覆盖 Attention 输入；
- buffer 通过 `record_stream` 绑定生命周期，close 清空 stream、event 和 pending transfer；
- U1、Graph U1 和 MTP 保持原路径，不扩大功能边界。

功能门禁通过：41 个 connector 单测、目标 Ascend runtime 回归、A2F1 两阶段/两 step 组件测试，以及 A8F8 eager/U2 batch32。batch32 下 8 个 Attention rank 均记录 `stage_count=2`，golden 结果一致，fatal、shutdown 和 NPU cleanup 通过。

P1 固定 C32、输入 1024、精确输出 128、128 请求，结果为 14.961 output token/s、p50 TPOT 2128.415 ms。相对旧同步 U2 的 16.631 token/s 回退 10.043%，相对 U1 的 30.615 token/s 回退 51.133%，因此不进入三轮 P2，也不创建性能 tag。

双侧 CANN 9.0.1 profile 证明实现不是“空 stream”：Attention 每 step 平均 overlap 从旧 U2 的 0 增至 35.551 ms，未重叠 communication 从 1925.313 降至 1669.388 ms，bubble 从 1814.378 降至 1187.816 ms；FFN overlap 从 0 增至 13.557 ms。但 FFN free 从 592.080 增至 949.095 ms，新增 overlap 没有转化为端到端吞吐。FFN 计算 kernel 已从旧默认 stream 46 移至专用 stream 43；HCCL task 仍由 process-group 内部 stream 执行，符合保持标准同步 API 的约束。

P8C 提出的两个 DBO Python 线程/`dbo_yield` 替换建议已经由 P8D 实施，结果见下一节。

证据目录：

```text
/mnt/workspace/validation/dsv4_afd_a3_comm_stream_component_a2f1_20260819_170945
/mnt/workspace/validation/dsv4_afd_a3_comm_stream_u2_batch32_20260819_173904
/mnt/workspace/validation/dsv4_afd_a3_comm_stream_u2_p1_c32_1k128_20260819_174750
/mnt/workspace/validation/dsv4_afd_a3_comm_stream_u2_profile_20260819_181639
```

### 5.14 A3-P8D：单线程 layer-major U2

P8D 将 DeepSeek-V4 HCCL eager/U2 的两个 Attention ubatch 线程替换为一个插件内 host loop，按 `layer -> stage` 连续推进。connector 延迟消费 F2A receive event，直到同一 stage 即将进入下一层；decoder layer 拆分 remote MoE 与 HC post，保证任何 output 消费都发生在 event wait 之后。U1、Graph、MTP、其他 connector 和 wire protocol 不变，底层仍是同步 `torch.distributed.send/recv`。

P8D 定向功能门禁已通过目标 CPU/Mock 回归、A2F1 两阶段/两 step 组件测试和 A8F8 batch32；8 个 Attention rank 均实际执行两个 stage，单条 golden、fatal、shutdown 和 NPU cleanup 通过。该候选没有重跑完整 30/30 golden 与 batch 1/8，因为随后 P1 已经否决性能候选。

同口径 P1 为 16.472 output token/s、p50 TPOT 1960.470 ms，相对 P8C 的 14.961 提升 10.099%，但相对旧同步 U2 低 0.958%，相对 U1 的 30.615 低 46.197%。按 P1 门禁停止，不扩大为三轮。

20-step 双侧 profile 显示 Attention 未重叠通信从 P8C 的 1669.388 降至 1398.570 ms，但 FFN free 从 949.095 增至 1273.094 ms，Attention bubble 也升至 1430.170 ms。kernel 数没有减少；Attention send wait 明显下降，但 F2A receive wait 上升，证明线程/GIL 交接不是剩余主瓶颈，等待被移动而未被消除。

eager/U2 的后续性能优化继续保持同步 API，并保留两项定向待办：

1. 对 DP0-7 同时采集/对齐 layer-stage 到达时间，定位每轮最慢 Attention/FFN rank 和等待传播链；
2. 设计更粗粒度的同步状态机，减少逐 layer/stage 控制消息、host `send/recv` 调用和 event wait 次数，但不改变 IDs/hidden/output 顺序、精确 shape、cache 生命周期和异常清理。

在功能扩展期间不扫描 eager threshold、不单独启动 eager P2 三轮。Graph/U2、MTP eager/U2 和 target Graph/U2 + eager draft MTP 均已完成独立 F0 + P1；其中 MTP eager/U2 的 P1 仍触发 20% 暂停线。当前按功能优先继续独立能力边界，全部功能组合闭环后再做定向 profile 和结构性修复，并统一恢复 Graph/U1、Graph/U2、native Graph 和 MTP on/off 的 P2。

性能缺口登记为 `P8D-PERF-001`，并由 M3 P1 再次确认。允许冻结 Graph/U2 和 MTP/U2 functional snapshot 供定位复用，但这些 tag 都不是性能基线；关闭问题仍以同口径 U1、三轮 P2 和同预算 native 门禁为准。

```text
/mnt/workspace/validation/dsv4_afd_a3_layer_major_component_a2f1_20260819_193218
/mnt/workspace/validation/dsv4_afd_a3_layer_major_u2_batch32_20260819_194535
/mnt/workspace/validation/dsv4_afd_a3_layer_major_u2_p1_c32_1k128_20260819_195445
/mnt/workspace/validation/dsv4_afd_a3_layer_major_u2_profile_20260819_202039
```

## 6. A3 性能验收方法

本章定义 P2 正式性能验收，只在目标功能组合完成后执行。中间功能阶段只运行 5.0 节定义的 F0 与 P1，不重复本章完整矩阵。

### 6.1 比较矩阵

最终至少比较：

| 方案 | 作用 |
|---|---|
| 非 AFD eager | 基础执行对照 |
| 非 AFD Graph | 非 AFD 最佳稳态对照 |
| AFD eager/U1 | 分离本身的成本和收益 |
| CAMP2P AFD eager/U1、U2 | 已冻结功能对照，不代表 HCCL 主线性能 |
| HCCL P2P AFD eager/U1 | 分离通信成本与 U2 对照 |
| HCCL P2P AFD eager/U2 | 当前候选交付形态，验证双阶段重叠收益 |
| HCCL P2P AFD Graph/U1 | 已通过功能门禁；用于衡量 Graph 对 host 发射与稳态执行的影响 |
| HCCL P2P eager/Graph U1 + MTP | MTP 功能通过后加入；与同模式 MTP-off 及 native MTP 对照，报告 acceptance rate |
| HCCL P2P A8F4 eager/U1 | 非等量聚合/切分成本与 U2 对照 |
| HCCL P2P A8F4 eager/U2 | 候选节省 FFN 资源形态，验证吞吐保持和资源效率 |

### 6.2 公平资源口径

A8F8 使用 16 个 NPU，A8F4 使用 12 个 NPU。只把 A8F8 与一个 8-NPU 非 AFD 实例比较，会把资源增加带来的收益误认为 AFD 架构收益；把 A8F4 直接称为“同资源非 AFD 对照”同样不成立，因为固定 DSV4 非 AFD 基线没有 12-NPU 等价布局。

因此必须同时报告以下口径：

| 对比 | 说明 |
|---|---|
| AFD A8F8 vs 非 AFD 单个 8-NPU 实例 | 反映单服务扩展能力和延迟变化 |
| AFD A8F8 vs 两个非 AFD 8-NPU 实例的总和 | 反映相同 16-NPU 总预算下的资源效率 |
| HCCL A8F4 vs HCCL A8F8 | 反映减少 4 个 FFN rank 后的吞吐保持、延迟和 HBM 变化 |
| HCCL A8F4 的 tokens/s/NPU vs A8F8/非 AFD | 反映归一化资源效率；必须同时给出绝对吞吐 |

如果后续有至少 24 张同型 NPU，可增加两个 A8F4 实例与三个非 AFD 8-NPU 实例的精确 24-NPU 总预算对照。当前 16-NPU A3 环境不能伪造该对照，也不能只按比例外推总吞吐。

所有结果必须同时给出：

- request/s；
- output tokens/s；
- output tokens/s/NPU；
- TTFT；
- TPOT p50/p90/p99；
- 峰值 HBM；
- NPU 利用率；
- A2F/F2A 和 overlap/bubble 指标。

### 6.3 建议门禁

以下是开始正式实验前应确认的建议门禁，不是适用于所有产品场景的永久阈值：

1. 正确性、生命周期和清理门禁必须 100% 通过；
2. HCCL P2P eager/U2 相对同参数 HCCL P2P eager/U1 的 output tokens/s 提升不低于 10%；
3. p99 TPOT 相对选定基线的回退不超过 5%；
4. AFD 相对同总 NPU 预算非 AFD 的收益必须大于 3 轮稳定运行的波动区间；
5. 不允许通过减少输出 token、改变 batch、降低 golden 覆盖或放宽错误检查获得收益；
6. 若吞吐增加但 `tokens/s/NPU` 明显下降，必须明确记录为扩容收益，不能表述为资源效率收益；
7. A8F4 eager/U2 相对 A8F8 eager/U2 的绝对 output tokens/s 建议保持不低于 95%，同时 `tokens/s/NPU` 建议提升不低于 20%；
8. A8F4 的 FFN 峰值 HBM 必须保留预先定义的安全余量，不能以临界 OOM 状态通过吞吐门禁。

第 7 项是首轮建议值，不是已证实结论。最终阈值应在 A3-P4 的 A8F8 参照完成后、A8F4 正式性能跑数前写入验证脚本或实验 manifest，跑完后不根据结果反向调整门禁。

### 6.4 A3 性能完成定义

A3 性能阶段只有在以下材料齐全时才完成：

- 全部对照方案的启动参数；
- 每个点 3 轮原始客户端结果；
- 服务端日志和返回码；
- 环境与精确 git commit；
- Attention/FFN profile 原始产物和解析结果；
- HBM、利用率和清理后的 `npu-smi info`；
- 一份结论明确区分“吞吐扩展”“资源效率”和“延迟”的报告。

通过后建议冻结：

```text
dsv4-afd-a3-hccl-p2p-perf-v1
```

tag message 和报告必须写明最终选择的 A/F 比例。若 A8F4 正确性通过但性能门禁未通过，可以冻结仅用于回归的 `dsv4-afd-a3-hccl-p2p-unequal-v1`，但不得使用 `perf` 命名。

## 7. 面向 A5：现在就要保持的设计边界

本文暂按“A5 指 Atlas A5 / Ascend 950 系列”规划。实际服务器 SKU、SoC 字符串和单机 NPU 数以目标机器审计结果为准。

### 7.1 已知的软件基础和当前缺口

固定 vLLM-Ascend commit 已包含：

- `A5DeviceAdaptor`；
- `SOC_VERSION` 以 `ascend950` 开头时的构建识别；
- 多个 Attention/MoE 算子的 `ascend950` 注册。

但这只说明固定上游存在 A5 基础，不说明 afd-plugin 已支持 A5。当前主线选择标准 HCCL send/recv，A5 不再以前置移植 afd-plugin 自定义 A2E/E2A kernel 为目标；这减少了以下 A3 专用实现对 A5 的阻塞：

- `csrc/npu/build_aclnn.sh` 只接受 `910c`/`ascend910_93*`；
- `a2e_def.cpp` 和 `e2a_def.cpp` 只注册 `ascend910_93`；
- `tools/dsv4/install_plugin.sh` 默认 `SOC_VERSION=ascend910_9362`；
- kernel 中存在 192 KB UB、48 core、window offset 等 A3 假设；
- ACLNN host 侧 HCCL server type 对 910B 和其他 SoC 走不同分支，A5 行为尚未实测。

这些限制仍适用于 CAMP2P 备选路径，但不进入 HCCL P2P 主线。另一方面，也不能因为代码只调用公共 `torch.distributed.send/recv` 就宣称 A5 已支持；A5 的驱动、固件、CANN、torch-npu、HCCL P2P 能力和目标拓扑仍必须实机验证。

### 7.2 A3 开发期间必须做到

- U2 stage、rank 和 token slice 逻辑保持硬件无关；
- 不在模型/connector 核心路径硬编码 A8F8、NPU 0-15 或单机卡数；
- role 数、buffer token 上限、U2 threshold 和 HCCL 配置保持可配置；
- 非等量映射只依赖逻辑 role rank，不依赖物理 device ordinal；`ratio`、peer ranks 和 `seq_lens` 由单一 topology 对象派生；
- HCCL backend、P2P send/recv 或目标拓扑不支持时显式 fail-fast；
- A3/A5 使用独立 venv、CANN 根目录、构建输出、启动 recipe 和验证目录；
- CPU/Mock 测试不依赖 A3 核数或物理 device ordinal；
- performance manifest 记录 SoC、驱动、固件、CANN、torch-npu、拓扑和 NUMA；
- connector 核心只依赖 PyTorch distributed HCCL 公共接口，不引用 CAMP2P 自定义 op；
- 上游 patch 继续标注固定 commit 和 AFD patch marker，便于 A5 栈变化时重放差异。

### 7.3 当前不要提前做的 A5 修改

没有 A5 硬件和匹配工具链时，不应凭 A3 结果猜测并提交以下变更：

- CAMP2P A2E/E2A kernel 的 UB 分配、核数和通信 window；
- A5 特定私有 IPC/SDMA 路径；
- U2 threshold、HCCL buffer 和 Graph capture size；
- CPU/NUMA 绑核和物理 rank 映射。

这些修改需要在 A5 上通过最小组件测试和 profile 驱动。

## 8. A5 到位后的独立适配阶段

### 8.1 A5-H0：硬件与运行栈审计

先记录实际机器，不沿用“A5 应该是什么”的假设。至少保存：

```bash
npu-smi info
npu-smi info -t board -i 0
npu-smi info -t topo
uname -m
```

若目标驱动不支持某个 `npu-smi` 子命令，保存等价的板卡和拓扑查询结果。

同时记录：

- 服务器完整 SKU；
- 实际 SoC 名称和 `SOC_VERSION`；
- 单机 NPU 数、每卡 HBM 和健康状态；
- 驱动、固件、CANN 和 torch-npu 版本；
- A5 对应 ops 包；
- CPU 架构、socket、NUMA 与 NIC/NPU 亲和关系；
- 单机还是跨机 A/F 拓扑。

A5 使用独立运行栈，并按目标产品支持矩阵固定版本。不要直接把 A3 的 CANN 根目录、venv 或已编译 `ascend910_93` 产物复制过去。

### 8.2 A5-H1：标准 HCCL P2P 运行栈门禁

需要完成：

1. 按 A5 产品支持矩阵建立独立 CANN、torch-npu、vLLM 和 vLLM-Ascend 环境；
2. 验证 `torch.distributed` HCCL process group 可创建、销毁和二次创建；
3. 验证 `send/recv` 支持 BF16 hidden/output 和 int32 input IDs；
4. 验证一个 FFN 与多个 Attention peer 的 communicator 建立、固定消息顺序和超时恢复；
5. 确认单机或跨机目标拓扑的 rank、NIC、NUMA 与链路能力；
6. unsupported backend、dtype、拓扑或栈版本明确失败。

只有产品决定重新启用 CAMP2P 备选路径时，才单独建立 A5 自定义算子移植里程碑；它不阻塞 HCCL P2P 主线。

### 8.3 A5-H2：kernel 和通信最小验证

按由小到大的顺序验证：

1. A1F1 IDs `int32` round-trip；
2. A1F1 hidden HCCL send/recv round-trip；
3. A2F1/A4F2 多 peer IDs、hidden 聚合和 output split round-trip；
4. 不同 peer token count、`-1` padding 和 buffer 边界；
5. 连续两个 step 和两个 stage；
6. 多 rank 单机；
7. 若产品拓扑跨机，再增加跨节点 HCCL 测试；
8. 异常取消、connector close 和重新启动。

本阶段重点实测：

- BF16/int32 send/recv 正确性和消息顺序；
- HCCL group 创建、销毁和错误恢复；
- 单链路带宽、时延与多 stage 并发行为；
- HCCL buffer、超时和网络接口选择；
- rank 到物理 NPU/NIC 的映射。

### 8.4 A5-H3：模型正确性回归

A5 必须先生成同平台非 AFD golden，不能只拿 A3 token 文件代替 A5 基线。随后依次验证：

1. Attention/FFN 角色构造和权重所有权；
2. layer 0、2、3、42 的单层/loopback 等价；
3. eager/U1；
4. eager/U2；
5. `A=kF` 与 `F=kA` 双向整数比例的 eager/U1、U2；
6. 冷启动、二次启动、batch、空闲恢复和严格关闭；
7. 等量和双向整数非等量 A/F 的 HCCL P2P Graph/U1、Graph/U2 回归；Graph/U3 另立里程碑。
8. 先生成 A5 原生 MTP golden，再回归 HCCL P2P 等量 eager/U1/U2 + MTP 和 Graph/U1/U2 + MTP；随后以 A8F4、A4F8 或实际选定的双向整数拓扑完成非等量 MTP 的 golden、batch、生命周期和 P1；不得直接复用 A3 MTP token 文件，也不得用 A3 组件结果替代 A5 实模 F0。

若 A5 单机有 16 个 NPU，先验证 A8F8，再在 HBM 允许时验证 A8F4，并回归 A4F8；若只有 8 个 NPU，先验证 A4F4，再评估 A4F2/A2F4。非等量候选必须满足较大侧是较小侧的整数倍。实际角色映射必须根据 `npu-smi` 拓扑和 NUMA/NIC 关系决定，不能只按 device ordinal 对半切分，也不能在未测 HBM 前假定更少 FFN rank 一定可行。

### 8.5 A5-H4：重新调优和性能验收

A5 需要重新扫描：

- U2 threshold；
- HCCL buffer；
- Attention/FFN 角色数与 `A/F ratio`；
- MTP speculative token 数、acceptance rate 和 proposer/verify 开销；
- CPU/NUMA 绑核；
- 单机/跨机 rank 布局。

profiling 仍遵守：部署、采集、解析分离；`TORCH_PROFILER_WITH_STACK=0`；Attention/FFN role-local DP0；使用与 A5 采集环境匹配的 CANN parser。

A5 使用与 A3 相同的公平资源口径，但重新生成全部数字和门禁结论。通过后再创建带 A5 标识的独立 tag，不能复用 A3 性能 tag 代表 A5。

## 9. M9 Mooncake PD 集成顺序

M8 TP 功能基线冻结后，下一功能阶段直接进入 M9 Mooncake PD。SP、CP、DCP
和 PP 全部后移，不作为 M9 的前置能力。M9 继续只修改 `afd-plugin`；若目标
Mooncake 接口与固定 vLLM/vLLM-Ascend 栈不兼容，先在插件兼容层适配并记录
上游差异，不直接修改两个固定上游工作树。

M9 分为功能门禁和性能门禁，二者不得混为一个准入条件。

进入 M9 功能开发的门禁：

- M8 的 HCCL P2P TP2 组件、同栈原生 golden 和至少 eager/U1 实模 F0 通过；
- TP1 行为无回退，TP2 rank/peer/control payload 契约已冻结；
- Attention、FFN 生命周期和自动清理稳定；
- 已明确 Mooncake PD 的 prefill/decode 角色、KV transfer、AF 子拓扑和启动顺序。

M9 按“本机先完成能完成的开发和验证，外部拓扑与确定性后置”的顺序推进：

1. 审计固定目标栈中的 Mooncake connector/API 和已有部署脚本，冻结 P/D/AF
   进程、rank、端口、NIC 与 KV ownership；
2. 建立不加载实模的最小 PD + AF connector 生命周期与 metadata 组件测试；
3. 在本机依次完成 TP1/TP2、eager/U1/U2、Graph/U1/U2 和一 token MTP 的
   配置、CPU/Mock、真实 NPU 组件及可执行实模 F0-local；
4. 本机 F0-local 只检查启动、请求成功、真实 stage/消息路径、batch、取消、异常、
   shutdown、二次启动和资源清理，暂不生成或比较 golden；
5. 全部计划功能代码和 F0-local 完成后，再在双 A3 上完成 TP1、TP2 及各组合的
   F0-topology；每次只增加一个变量并保存独立产物；
6. 最后统一启用 batch-invariant，生成路径匹配的 PD control，执行跨冷启动稳定性
   和 control/AFD 30/30 token exact 的 F1 正确性冻结门禁；
7. F1 通过后才创建 M9 正确性功能 tag，并进入正式 P2 性能验收。

截至 `2026-08-28`，M9 第 1 步审计和首个代码门禁已经落地：固定目标栈注册的
KV connector 为 `MooncakeHybridConnector`；PD 只连接 Prefill 与 Decode Attention，
Decode FFN 不配置 KV connector。插件当前开放 `P2pHcclAFDConnector + TP1/TP2 +
eager/U1 + MTP off + kv_consumer`，并校验 `engine_id`、Mooncake 端口以及
Prefill/Decode DP/TP 元数据。TP1 使用 Decode DP8/TP1；TP2 使用 M8 已冻结的等量
A8F8、Decode DP4/TP2 rank/peer/control payload 契约。结构化配置生成器、
Prefill/Attention/Proxy recipe 和运行库预检已经加入。手工管理脚本和 no-AFD
control 现已具备 eager/U2、Graph/U1、Graph/U2 和一 token MTP 的逐项配置入口，
并把模式写入 control golden metadata，防止不同路径误比较；TP2 full-draft Graph
U2 + MTP 继续 fail-fast。上述新增项都只是代码/配置门禁已开放，下一步先完成
本机 F0-local 和剩余功能开发；双 A3 F0-topology 已进入后续外部验收队列，但不
阻塞彼此独立的本机功能开发。外部拓扑未通过前不得写成双机实模交付完成。

当前 A3 环境门禁已经补齐：`libgoogle-glog0v6t64`、`libjsoncpp25`、`libjemalloc2`
和本地 Ascend Mooncake 0.3.9 wheel 已安装，扩展由目标 `afd-v023-vllm-cann`
venv 自身提供；`TransferEngine` import、`MooncakeHybridConnector` 请求 metadata、
Prefill/Decode DP/TP 解析及 CANN 9.1.0 泄漏检查均通过。最小 import 矩阵还确认
`torch_npu + Mooncake` 必须预加载 `libjemalloc.so.2`，否则进程退出阶段会触发堆损坏；
PD launchers 通过运行门禁继承该 preload，standalone AF 和 FFN 不受影响。安装 wheel
使用现有本地构建产物，未修改 Mooncake、vLLM 或 vLLM-Ascend 工作树。两进程真实
NPU 组件也已通过：NPU0/NPU1 使用 `P2PHANDSHAKE + ascend` 注册 2 MiB buffer，
连续两次同步传输返回 0、逐字节一致，双方均正常析构退出。下一步在本机完成全部
可执行的组合 F0-local；完整 `Prefill DP2/TP4 + Decode A8F8` 需要 24 个逻辑 NPU，
本机 16 个逻辑 NPU 无法同时容纳，因此只把该完整路径保留到双 A3 F0-topology。

当前 contract 证据包括：插件 Mooncake feature gate 13/13、recipe/config/EngineCore
71/71、固定 vLLM-Ascend `MooncakeHybridConnector` 5/5 和通用 Mooncake connector
92/92、真实两进程 NPU round-trip 2/2，完整 Ascend runner 回归也通过。阶段汇总保存在：

```text
/mnt/workspace/validation/dsv4_afd_v023_mooncake_pd_m9_contract_20260821_181148/summary.json
```

该历史汇总状态是 `real_transfer_component_passed_f0_pending`。它证明当时的组件门禁，
不证明当前 M9 F0-local、双机 F0-topology 或 F1 已完成，也不能据此创建 M9 正确性
功能 tag。

截至 `2026-08-28`，本机 M9 F0-local 基础闭环已完成，且明确不执行 golden：

- CPU/Mock、诊断和手工工具回归 `36/36`，Mooncake PD 功能矩阵 `46/46`；
- Mooncake NPU0 -> NPU1 连续两次 2 MiB 同步传输返回 0 且逐字节一致；
- HCCL 组件覆盖 TP1/TP2、非等量 AF、eager/Graph、U2 和一 token MTP，四组
  组合全部通过；
- 同机 PD no-AFD control 使用 Prefill NPU0-7 DP2/TP4 和 Decode NPU8-15
  DP8/TP1，完成两次完整冷启动；每次 batch 1/8/32、请求取消和取消后恢复均通过，
  Decode 每轮记录 43 条成功 Mooncake KV transfer；
- 两轮均正常按 Proxy、Decode、Prefill 顺序停止，停止后 8100/8910/9000 均关闭，
  `npu-smi info` 无 NPU 进程；全过程 `ENABLE_BATCH_INVARIANT=0`、
  `golden_checked=false`。

本轮还补齐了同机 control 的严格设备不重叠/进程归属门禁、无 golden 的 `smoke`
入口，以及有超时和 `/proc/net/tcp*` 回退的小型证据收集。结果汇总位于：

```text
/mnt/workspace/validation/dsv4_afd_m9_local_f0_20260828_112300/summary.json
```

上述结果只冻结“本机可验证的 M9 基础设施和组件闭环”。本机没有同时运行
Prefill 8 + Attention 8 + FFN 8，因此不能据此声明完整 PD + AFD F0-topology
通过，也不创建 M9 功能 tag。完整三段路径和各组合仍留到双 A3；一 token MTP
之后的多 speculative token 已由 M10 关闭本机门禁，PD 组合仍需单独验收。

### 9.1 2026-09-04 双 A3 Graph/U2 验证结果

本轮使用 CANN 9.0.0、vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、
vLLM-Ascend `3da28f9414583d2d0b672a8f06d1fae142404bda`、afd-plugin
`2164240b31efc8605bf84cc45afc628996669554 + R14 overlay` 和 Mooncake 0.3.9。
三点均为 TP1、MTP off、`FULL_DECODE_ONLY`、Graph/U2；双机物理资源均预留 32 张
NPU，但 active NPU 分别是 24、24 和 32：

| 测试点 | 拓扑 | output token/s 三轮均值 | CV | token/s/active NPU | TPOT P50/P90/P99 |
|---|---|---:|---:|---:|---:|
| 共置 A8F8 | `P8 + [A8F8]` | 225.761 | 4.026% | 9.407 | 117.430/174.374/192.041 ms |
| split A8F8 | `[P8F8] + A8` | 219.182 | 14.080% | 9.133 | 131.892/148.103/156.636 ms |
| split A16F8 | `[P8F8] + A16` | 540.433 | 1.806% | 16.889 | 45.023/46.515/46.769 ms |

每轮 128/128 请求完成，三个测试点的 U2 stage 证据分别覆盖 8/8、8/8、16/16
Attention rank。A16F8 的物理比例是 `A:F=2:1`；每个 FFN 在 U2 下接收两个
Attention peer 乘两个 stage，共四个逻辑输入切片，不能把逻辑切片数写成物理 4:1。

这里的“完成”只指请求、stage 和采集流程完成，不是输出正确性验收。R14 为绕过部分
FFN Graph capture 的 `LocalScalarDenseNpu` 错误，曾把 warmup 的 AllToAllV split 按 shape
缓存；审计确认 split 实际依赖每个 layer/请求的动态 `topk_ids`，而 replay 不会再次执行
Python 预处理。该 overlay 已撤回且不进入提交。A16F8 Profile 窗口主要观察到 MC2
`MoeDistributeDispatchV2/CombineV2`，仍可用于解释流水，但可达的大 gear AllToAllV 路径
没有 golden 覆盖，不能据此冻结 FFN Graph 正确性。

A16F8 相对 split A8F8 的 output throughput 为 `+146.568%`，token/s/active NPU 为
`+84.926%`，TPOT P50/P90/P99 分别下降 `65.864%/68.593%/70.141%`。变化远大于
两组 CV 之和，是强方向信号；但该对比同时增加 8 张 active Attention NPU，并把 FFN
`max_num_batched_tokens` 从 4096 调到 8192，且 split A8F8 的 CV 超过 10%。因此该轮只记
`measurement_only_no_fixed_gain_threshold`，不能把全部改善归因于 2:1、混合 DAG 或
新增物理 stream。共置与 split A8F8 的均值差 `-2.914%` 小于波动口径，也不能据此断言
跨机 placement 没有代价。

六个 CANN 9.0.0 Profile root 都通过产物与 parser 交叉检查，`with_stack=false`。DP0
step-trace 的核心分解如下；`wall = Computing + Communication(Not Overlapped) + Free`：

| 测试点/角色 | wall | Computing | Comm non-overlap | Comm overlap | Free | Bubble |
|---|---:|---:|---:|---:|---:|---:|
| 共置 A8 Attention | 109.544 s | 22.147 s | 67.747 s | 9.716 s | 19.650 s | 74.134 s |
| 共置 A8 FFN | 109.517 s | 39.650 s | 14.971 s | 8.053 s | 54.896 s | 22.155 s |
| split A8 Attention | 113.976 s | 21.506 s | 72.313 s | 8.145 s | 20.158 s | 74.588 s |
| split A8 FFN | 113.942 s | 14.772 s | 17.157 s | 5.033 s | 82.012 s | 20.696 s |
| split A16 Attention | 55.498 s | 18.756 s | 15.843 s | 3.585 s | 20.899 s | 18.752 s |
| split A16 FFN | 55.474 s | 16.124 s | 18.050 s | 3.520 s | 21.299 s | 20.500 s |

绝对秒数必须同时按各自 Profile wall 归一化。split A8F8 到 split A16F8 的 FFN wall
从 113.942 s 缩短到 55.474 s，因此 Bubble 秒数近似不变并不表示其相对负担不变：

| FFN 指标 | split A8F8 | split A16F8 | 归一化变化 |
|---|---:|---:|---:|
| `Free / wall` | 71.977% | 38.395% | -33.582 个百分点 |
| `Bubble / wall` | 18.164% | 36.954% | +18.790 个百分点；相对 bubble burden 约 +103.45% |

相对 split A8F8，A16F8 的 FFN Free 下降 `74.029%`，而 Bubble 绝对值只下降
`0.944%`，non-overlap communication 反而增加 `0.893 s`（`+5.205%`）。因此吞吐改善的主要解释是
FFN 更持续地得到输入；但在缩短后的 wall 中，剩余 Bubble 占比反而约翻倍，不能表述成
“receive/sync bubble 已被消除”，也不能仅以绝对 Bubble 秒数判断比例扩展已经解决等待。
最长 receive 裁剪窗口
进一步确认同一窗口存在 recv/compute、compute/send 和前一 send/下一 recv 重叠；从第一个
Attention send 开始的活跃窗口内 FFN Computing 占 `85.154%`，约 `60.756%` 通信时间被
计算覆盖。四个高层输入接收也与 `2 peers x 2 stages` 一致。

当前证据不支持“单逻辑 recv stream 导致队头阻塞”的确定结论：裁剪中多个底层
`Notify_Wait` 已并行，高层 receive 完成虽呈串行，但只采集 Attention DP0 和 FFN DP0，
缺少第二个 Attention peer、多个 FFN rank、host enqueue/event 和跨机时钟同步。下一轮应
直接补这些证据，而不是仅依据一条最长 receive 外推全局。

本轮请求、流水和测量观测有效，但 Graph 正确性与 F0-topology 生命周期都没有冻结。
14 个角色归档中 7 个包含停机期
fatal marker，11 份日志尾部出现强杀；最终 `npu-smi` 已无残留进程。归档中的
`status.exitcode=1` 只是停止后 health/status 的 NOT RUNNING，不是服务进程退出码，现有产物
也没有独立记录真实进程 rc。发布前必须修复优雅停止并完成二次启动，同时补路径匹配的 PD
no-AFD control、30/30 token exact F1、稳定的 split A8F8，以及 all-on/V1/off 公平消融。
原始证据归档为：

```text
/mnt/workspace/log/201f96bcabb5446ea950ad241cd42202.zip
/mnt/workspace/log/91c03c33caa144539ef9438e7098e8be.zip
/mnt/workspace/log/29df8d29d6874b6899027810ac85072b.zip
```

### 9.2 上游确定性遗留与 AFD 验收边界

双 A3 验证中暴露的数值问题不归属 `afd-plugin`，登记为两个独立上游遗留：

| ID | 现象与归属 | 当前处理 |
|---|---|---|
| `UPSTREAM-DSV4-BI-001` | vLLM-Ascend batch-invariant ReduceSum 不支持 DeepSeek-V4 HC 的非末轴求和，并错误覆盖无 `dim` 的 `aten::sum`；归属 vLLM-Ascend 适配层及官方自定义 OPP 交付 | 状态 `deferred_after_functional_development`；保留两文件补丁、独立 venv、自定义 OPP 和验证包，全部计划功能开发完成后统一复验 |
| `UPSTREAM-DSV4-SHORT-EXTEND-001` | Mooncake KV-only 交接后 Decode 进行 N-1 short-extend 重算，和 one-shot Prefill 属于不同执行路径；旧 native golden 可能稳定地只匹配 21/30 | 保留为 vLLM-Ascend DSA/PD 路径等价性问题，不要求 AFD 插件把当前稳定输出改回旧路径 |

以下是全部功能开发完成后的 F1 正确性冻结门禁，不是当前 F0-local 的进入条件：

1. 同一路径先满足三轮和跨冷启动稳定；不稳定时不能进入 token exact 验收；
2. `PD no-AFD control` 与 `PD + AFD` 必须使用同一 Prefill/Decode 路径、拓扑、
   batch-invariant 配置和启动顺序，逐请求 token IDs 要求 30/30；
3. 当前 PD 路径与旧 direct native golden 的差异只记录，不作为 AFD 失败；
4. 只有 `PD no-AFD` 稳定而 `PD + AFD` 相对它发生新增分叉，才归属
   `afd-plugin` 并阻塞 M9；
5. 上述上游遗留不阻塞 M9 代码组合开发；F1 未通过前不创建 M9 正确性功能 tag，
   也不开始正式性能验收。

延后执行的双 A3 batch-invariant 补丁、安装、两次 10 x 3 和小包收集步骤见
`DEEPSEEK_V4_BATCH_INVARIANT_DUAL_A3_VALIDATION_GUIDE_ZH.md`。

standalone AF 的 `P8D-PERF-001` 在 M9 期间继续保持 Open，但不阻塞上述功能
开发。进入 M9 性能结论和性能 tag 前，仍必须补齐：AFD 相对非 AFD 的公平
资源对照、A2F/F2A/FFN wait/bubble profile、三轮稳定性，以及 PD control 对照。
PD 拓扑是新的独立变量，不能用 standalone AF 数字直接替代生产拓扑结论。
U3 仍不纳入当前路线。

### 9.3 F1 后的物理 A:F 受控扫描

物理比例扫描不得与当前 P0 修复并行得出性能结论。只有以下三个门禁全部通过后才开始：

1. FFN Graph 动态路由 P0 已关闭；同一 Graph key 下不同 `topk_ids`、zero-to-nonzero
   路由、所有可达 capture gear 均通过 eager/Graph 对照和 30/30 token exact；
2. 路径匹配的 PD no-AFD control 与 PD + AFD 已通过跨冷启动稳定性和 F1；
3. 正常停止、异常停止、二次启动和空闲恢复均无 fatal、强杀、残留进程、端口或 NPU
   占用，真实进程返回码已进入归档。

门禁通过后，以固定 `F=16` 执行物理 `A:F=1:1 -> 2:1 -> 4:1` 扫描：

| 物理比例 | 目标 AF 拓扑 | AF active NPU | 说明 |
|---|---:|---:|---|
| 1:1 | A16F16 | 32 | 固定 FFN EP16 的比例基线 |
| 2:1 | A32F16 | 48 | 每个 FFN 对接 2 个 Attention peer |
| 4:1 | A64F16 | 80 | 每个 FFN 对接 4 个 Attention peer；不是当前 A16F8 的外推结果 |

三点固定同一代码、模型、CANN/Mooncake、TP、Graph/U2、流水预设、请求长度和到达模型，
通过同一 concurrency sweep 找到各自稳定饱和区；每点至少三轮，并同时报告 active/reserved
NPU、吞吐、token/s/NPU、TPOT、CV、HBM、FFN `Free/wall`、`Bubble/wall`、receive wait 和
通信重叠。FFN 接收容量必须按 fan-in 的正确性上限配置并完整记录；容量随比例变化时，另在
较低比例补同容量敏感性对照，不能把 buffer/capture gear 变化归因于 A:F。

`A64F16` 的 AF 数据面本身需要 80 张 active NPU，尚未计入 Prefill，明显超出当前两台
16 卡 A3 的 32 卡总资源。当前双 A3 只实测到 A16F8（物理 2:1），既没有固定 F16，也没有
物理 4:1；因此不能从 A16F8 的吞吐、Free 或 Bubble 趋势推导 A64F16。4:1 必须在具备足够
节点、网络和 Prefill 资源的 scale-out 环境重新完成 F0-topology、F1、Profile 和 P2。

## 10. 分支、Tag 和产物规范

### 10.1 建议分支和 Tag

| 阶段 | 建议分支/Tag |
|---|---|
| eager/U2 开发 | `feat/dsv4-afd-eager-u2` |
| A3 eager/U2 基线 | `dsv4-afd-a3-eager-u2-v1` |
| HCCL P2P connector 开发 | `feat/dsv4-afd-hccl-p2p` |
| HCCL P2P 非等量开发 | `feat/dsv4-afd-hccl-p2p-unequal` |
| A3 HCCL P2P 非等量正确性基线 | `dsv4-afd-a3-hccl-p2p-unequal-v1` |
| v0.23 HCCL P2P eager 非等量 MTP 组件基线 | `dsv4-afd-v023-hccl-mtp-unequal-component-v1` |
| v0.23 HCCL P2P full draft Graph 功能基线 | `dsv4-afd-v023-hccl-mtp-full-draft-graph-v1` |
| v0.23 HCCL P2P TP2 功能基线 | `dsv4-afd-v023-hccl-tp2-v1` |
| v0.23 Mooncake PD + AFD 开发 | `feat/dsv4-afd-mooncake-pd` |
| v0.23 Mooncake PD Graph/U2 多流验证 | `feat/dsv4-afd-graph-u2-multistream-all-on-v1` |
| A3 HCCL P2P 性能验收 | `dsv4-afd-a3-hccl-p2p-perf-v1` |
| vLLM 0.23 + `rfc/vllm_cann` 功能兼容基线 | `dsv4-afd-v023-vllm-cann-eager-u2-functional-v1` |
| A5 基线 | 在实际硬件和版本确认后使用 `dsv4-afd-a5-*` 命名 |

每个 tag 应为 annotated tag，tag message 至少包含：

- 基线用途；
- 精确 commit；
- SoC/拓扑；
- CANN、vLLM、vLLM-Ascend；
- U1/U2、eager/Graph 和 connector；
- 主验证产物路径。

### 10.2 验证产物目录

统一使用：

```text
/mnt/workspace/validation/dsv4_afd_a3_<stage>_<commit>_<timestamp>
/mnt/workspace/validation/dsv4_afd_a5_<stage>_<commit>_<timestamp>
```

每个目录至少包含：

- `environment.txt`：环境变量、CANN 和 Python 包；
- `git.txt`：afd-plugin/vLLM/vLLM-Ascend commit 和 worktree 状态；
- `npu_before.txt`、`npu_after.txt`；
- `command.txt` 或结构化启动参数；
- Attention/FFN 日志和返回码；
- golden/batch/性能原始请求结果；
- `validation_summary.json`；
- profiler 原始目录、解析目录和 `profile_validation.json`；
- 清理检查结果。

## 11. 风险和停止条件

遇到以下情况时停止扩大测试规模，先修复当前阶段：

- token 与 golden 不一致；
- IDs/hidden 消息计数或 stage 对应关系不确定；
- 非等量拓扑下 peer 顺序、`seq_lens` 或 output 回传目标不唯一；
- 当前 eager 路径跨 step 使用旧 IDs；
- 出现 NaN/Inf、shape 或 dtype 不一致；
- Attention/FFN 任一侧非 0 退出；
- 冷启动后存在残留进程、端口或 NPU 占用；
- CANN 路径混入其他版本；
- profiler 采集栈和 parser 版本不匹配；
- 性能收益只在单轮出现，或小于稳定运行波动；
- A8F4 FFN rank 的模型加载、专家 workspace 或聚合通信 buffer 接近 OOM；
- A5 上只完成 HCCL 组件验证，没有完成模型和 E2E 验证。

## 12. 后续恢复工作时的最短检查清单

1. 确认当前分支、HEAD、已冻结 tag 和 worktree 状态；迁移前 checkpoint 为 `dsv4-afd-a3-sync-hccl-pre-v023-v1`；
2. 激活目标运行栈并运行 `tools/dsv4/check_v023_vllm_cann_runtime.sh`，同时确认两个目标上游工作树干净；
3. 确认同栈原生 golden 和最近一个 AFD `validation_summary.json`，不要再用旧栈 token IDs 判断目标栈正确性；
4. A3-P4 已完成；先阅读 `DEEPSEEK_V4_AFD_HCCL_P2P_A3_P4_PERFORMANCE_REPORT_ZH.md`，不要把当前 U2 当作性能基线；
5. A3-P5 已完成；恢复时先核对 A2F1/A4F2 `summary.json` 和相关回归，不重复改写 topology 协议；
6. A3-P6 的 A8F4 已确认受 EP4 HBM 阻塞，A10F5 受固定栈 EP5 放置阻塞；不要通过超卖或 EPLB 改变语义绕过；
7. 旧栈 C32 同步优化和 A3-P7T 目标栈 U1/U2 复测均已完成；目标栈 U2 回退 26.342%，明确禁止打性能 tag；
8. 目标栈 HCCL P2P Graph/U1 和 Graph/U2 功能已经通过；恢复时先核对两个专项报告、Graph/U2 F0/P1 汇总和 graph replay 日志，不重复修改已通过的 lowering 与 layer-major capture；
9. A3-P7M0 已完成；恢复时先核对 M0 报告、`mtp_weight_contract.json` 和 30/30 对照，不重复生成原生基线；
10. A3-P7M1、M2、M3 和 M4 功能已完成；恢复时核对四份报告、M4 的 30/30 golden、请求边界 U2、低并发 U1 fallback、startup graph key 和在线 miss 整步 eager fallback 证据；M3 P1 相对 MTP/U1 回退 42.583%，M4 的 31.473 token/s 单轮不能关闭该问题；
11. A3-P7M5/M6 的 eager/Graph 非等量组件闭环已完成；恢复时核对 M5 的 A1F1/A2F1/A4F2 与 M6 的 A2F1/A4F2/Graph+MTP `summary.json` 和两份专项报告，不把 component tag 当成 A8F4 E2E 或性能基线；
12. A3-P8/P8C/P8D 的 eager 性能缺口 `P8D-PERF-001` 仍为 Open；Graph/U2 P1 的 107.189 token/s 和 M4 P1 的 31.473 token/s 都不能直接关闭该问题。功能组合闭环后再做 Graph/U1、Graph/U2、MTP on/off 和同预算 native Graph 三轮 P2；
13. A3-P7M7 full draft ACL Graph 已完成；恢复时核对专项报告、U1/U2 各 30/30、A4F2 组件、capture bucket、128/128 P1 和 cleanup，不再沿用旧的 6/30 结论；
14. A3-P7M8 TP2 功能基线已完成；恢复时核对专项报告、三个 TP2 组件产物、原生 DP4/TP2 golden 和 A8F8 eager/U1 30/30 F0；不要把失败的 TP2 full-draft Graph U2 最大组合写成支持；
15. 当前功能里程碑为 M9 Mooncake PD；TP1/MTP off/Graph U2 已完成双 A3 三拓扑请求、三轮性能和双侧 DP0 Profile 观测。A16F8 是物理 2:1；按 wall 归一化后 FFN Free 从 71.977% 降至 38.395%，Bubble 却从 18.164% 升至 36.954%，不能宣称 bubble 已消除。R14 AllToAllV 静态 split-cache 已撤回，FFN Graph 动态路由是当前 P0。顺序固定为：先实现 graph-safe 动态路由并重跑 golden，再关闭优雅停止/二次启动门禁，再完成路径匹配 PD control 和 30/30 F1；随后执行 C0 加 T1/T2/T3 x all-on/V1/off 的 10 单元正式 P2，并补其他执行组合；上述门禁全部完成后才做固定 F16 的物理 1:1/2:1/4:1 扫描。A64F16 超出当前双 16 卡 A3，不能从 A16F8 外推；
16. `UPSTREAM-DSV4-BI-001` 和 `UPSTREAM-DSV4-SHORT-EXTEND-001` 归属 vLLM-Ascend/执行路径，不作为 afd-plugin 代码缺陷；batch-invariant 和 golden/token exact 统一后移到全部计划功能开发完成后的 F1，只有路径匹配 control 稳定而 AFD 相对它发生新增分叉才阻塞插件正确性冻结；
17. TP/SP/CP/DCP/PP、TP3、非等量 TP2 和 TP2 最大 Graph+MTP 组合不作为第一阶段门禁；
18. M10/M11 本机开发门禁已完成：最大 `N=3`，A1F2/A2F4 的 N2/N3 eager/Graph U1/U2
    共 16 项 NPU 组件在精确源码栈通过；A8F8 N2 的 U1/U2 输出均与 native N2 达到
    10/10 exact，A4F8 eager/U1/N2 实模 smoke 通过。组件证据位于
    `/mnt/workspace/validation/phase1_cann900_exact_3e88ad2`；恢复时不得退回单 token、
    `A>=F` 或跨 target/draft/MTP 路径复用 golden 的假设；
19. A5 到位后从硬件审计和独立工具链开始，不复用 A3 二进制；先无并发生成 5 份路径
    匹配 native control，再执行 9 点 standalone F0/F1；
20. 每次阶段完成都保存日志、原始数据、解析结果和清理证据。

## 13. 一句话路线

在 CANN 9.0.0、vLLM 0.23 + `rfc/vllm_cann` 目标栈已经完成 HCCL P2P eager U1/U2、
Graph/U1/U2、单 token MTP M0-M7、双向整数比例组件与 TP2/M8 历史基线。M10/M11
已关闭本机开发门禁：单 MTP layer 最大 `N=3`，A1F2/A2F4 16 项组件通过，A8F8 N2
的 U1/U2 输出均与路径匹配 native N2 达到 10/10 exact，A4F8 eager/U1/N2 实模 smoke
通过。M9 的
TP1/MTP off/Graph U2 已完成双 A3 三拓扑运行和 Profile 观测，但动态路由、优雅退出和
路径匹配 F1 未冻结。第一阶段下一步在 A5 生成 5 类 native control，完成 9 点 standalone、
高 HBM A8F4、双机 PD 组合和 F1；第一阶段
功能 tag 完成后，第二阶段再做 U3 和正式性能
收益；TP/SP/CP/DCP/PP、TP3、非等量 TP2 和 TP2 最大 Graph+MTP 不作为第一阶段门禁。
