# feat/dsv4-afd HCCL 完整能力迁移矩阵

修订日期：2026-11-19。供体锁定 feat/dsv4-afd@4ddd69a391dfb5b3e362c7601399e538a31de702。
目标实现以 v0.30 + RFC/V4.1 的契约重建，允许改变类/文件组织；不允许以 eager/U1 通过代替全能力交付。
本表是范围/覆盖表，任务进度仍只在各 change 的 tasks.md；表中“供体存在”不等于当前目标 NPU 已验收。
本次审视起点 HEAD 为 c1f9ccc7fd7e5ba09f243ce37824af90bb0030ea；I01–I07 固定到该提交作为可重现的缺口证据。
工作区存在同时推进的 C3 改动，不能假定当前文件全部等于该 HEAD。核查基于锁定 Git 对象和起点工作区，本次修订未编辑运行时，也未使用历史性能数据证明新能力。

## 当前排期：先验证原生 Engram/U1，新增 Engram 适配单列

按 [2026-11-19 目标澄清](ROADMAP_ZH.md#engram-priority)，近期先闭合原始量化 checkpoint
保留原生 Engram 的 eager/U1 正确性，再按依赖验证 Graph/U1、DSpark/U1 和 PD/U1。
新增 Engram 适配仍按中优先级审查实施，原有独立组件证据保留。

当前锁定 vLLM 明确拒绝 Engram+DBO/microbatch，Ascend 普通路径也归零相关参数；
因此 U2 真实模型是新增适配目标。该限制不直接禁止 Graph/U1、DSpark/U1 或 PD/U1，
但这些路径仍需各自的锁定接口、状态及 NPU 证据，不直接判为支持。
U1 不限定卡数，CPU offload 不关闭 Engram，dummy/BF16/无 Engram 组件不能代表完整权重。

| 能力范围 | 当前复用验证与独立组件 | 需要审查实施的新增 Engram 依赖 |
| --- | --- | --- |
| H01–H04：协议、拓扑、U2、continuation | 原生 Engram eager/U1 真实权重对照、A/F 权重与状态；原有非 Engram U2 组件 | Engram+U2 准入、父 producer、stage lookup/history/event，见 C7 2.1–2.3 |
| H05–H11：多流与 Graph | 复核及验证 Graph/U1 的原生 Engram 接口；保留非 Engram U2/图/流组件 | U1 复核发现的实际图缺口，以及 U2 图输入、producer/event、replay/fallback，见 C7 2.3–2.4 |
| H12–H13：DSpark/PD | 在各自前置满足后验证 DSpark/U1、PD/U1 的真实历史和请求链；保留组件结果 | 新增 history/handoff 缺口及 U2 组合，见 C7 2.5–2.6；U1 不等待 U2 先完成 |
| H14–H15：运维、完整验收 | 按精确 U1 场景记录正确性及必要清理；profiler/启停/性能专项后置 | U2 完整模型、最终组合与性能收口，见 C7 2.7 及原 1.8–1.12 |

上述 Engram 待办的唯一专项清单是 [C7 第 2 节](changes/afd-hccl-full-parity/tasks.md#engram-medium)，
复用验证与新增适配分别记录。阶段 U1 交付不等于完整 H01–H15 迁移；最终 U2、Graph、
DSpark/PD 范围保持，混合任务满足全部条件才勾选。不得用关闭 Engram/受控计算替代原
checkpoint 的数值、历史或性能验收。普通旧 V4 MTP 保持暂停，恢复时单列回归。

## 一、上一版遗漏的实质

- Graph/U2/DBO、PD 被写成延期项，导致“先做 U1”变成了“只交付 U1”。
- “保留原生 forward，只换 MoE”适用于同步边界；要保留旧 layer-major 多流能力，需要目标 V4.1 的 continuation/prepare/complete 执行接口。
- 当前 C1 typed plan 限定 stage_idx=0、F num_stages=1，不能直接复用已有 CAMP2P UBatchRunner 就宣称 U2 支持。
- 只迁移 Python send/recv 不能覆盖图内通信、捕获协议、稳定 buffer、按 peer 图缓存和 fallback。
- Engram producer/持久 buffer/ExternalEvent 与 shared attention state 是新增状态所有者；U2/Graph/多流都必须逐 stage 对齐。
- 原生 MTP 与 V4.1 DSpark 语义不同，要逐分支映射而非全盘删去或强行 alias。

## 二、能力→源码→实现任务→验收

任务进度以各 change 的 tasks.md 为准，以下仅定义最终覆盖关系。C0–C7 既有证据
按原范围保留；本次排期调整不扩大已通过范围，也不把 Engram 依赖从最终矩阵删除。

| 编号  | 必迁能力与供体依据                                                                                                 | 当前缺口 / 目标落点                                                                        | 任务入口及验收                                                                                                                                     |
| --- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| H01 | BF16/IDs/输出、控制/层序、组和退出生命周期；D01、D04、D09                                                                    | 已有 BF16+独立 FP32+IDs 的 U1 组件；必须覆盖新 stage/peer/图模式，保留协商/STOP/故障终止语义                  | [C4 1.2–1.4](changes/afd-hccl-v2-microbatch/tasks.md)，[C5 1.2–1.7](changes/afd-hccl-v2-graph/tasks.md)；协议与 NPU 数据逐项验证                       |
| H02 | A=kF 聚合、F=kA fanout、等配对、非均匀 peer token、padding/反向拆分；D01/D04/D09/D14                                       | 当前是等 TP 坐标的双端 pair，不具备完整多 peer 数据投影；扩展 topology/control/layout                     | [C4 1.1/1.5/1.11](changes/afd-hccl-v2-microbatch/tasks.md)；TP1 整数比例、TP2 等规模，0/1/奇数 token                                                    |
| H03 | 独立 stage HCCL 组、U1/U2、DBO 阈值/切分/child context、归并；D01/D03/D06/D07/D10                                      | 扩展 stage_idx=0 plan、单活动游标、F num_stages=1；接入 V2 slice/merge                         | [C4 1.2–1.6/1.8–1.10](changes/afd-hccl-v2-microbatch/tasks.md)；两 stage 真执行、empty/odd、请求坐标和输出一致                                              |
| H04 | 单线程 layer-major、延迟 FFN completion、最终 drain；D02 forward_ubatches_layer_major/_complete_pending_stage       | 目标增加 V4.1 continuation，保存 residual/post/comb 和 ffn_pre→pre_mix，不能套旧 V4 tuple       | [C4 1.6–1.10](changes/afd-hccl-v2-microbatch/tasks.md)；本 stage 接收后恢复、无状态串扰                                                                  |
| H05 | eager/U2 A compute/send/recv、F recv/compute/send 的显式 event；D01/D04                                        | 当前顺序 IO 无该流水；FP32/IDs 同时参与 ready/lifetime 管理                                       | [C6 1.1–1.4/1.9–1.12](changes/afd-hccl-v2-multistream/tasks.md)；开关开/关同结果，timeline 有真实重叠                                                     |
| H06 | graph-visible HCCL send/recv 注册、编译 lowering、旧动态 shape guard 处理；D01:78/99/121、D09                          | eager host 校验/握手不能捕获；检查目标 torch_npu/CANN ABI，按需最小 patch                            | [C5 1.1–1.4](changes/afd-hccl-v2-graph/tasks.md)；真图内 BF16/FP32/IDs/F2A 及重复 replay                                                           |
| H07 | warmup/capture/replay 协同、稳定 buffer、精确 peer graph key、speculative 签名；D03/D04/D10                           | A/F descriptor/phase/key 一致；每次 warmup 匹配 F，buffer 原位刷新                             | [C5 1.3/1.5–1.9](changes/afd-hccl-v2-graph/tasks.md)；同 total 不同 peer 不误图，shape/容量/内存验收                                                      |
| H08 | FULL_DECODE_ONLY、U1/U2、live miss→eager、prefill/阈值回退和 attention graph 更新；D03/D08/D10/D14                   | 使用 V2 Graph 接口重建；不能拿 MLA 或 CAMP2P 原有路径充当 V4.1 HCCL                                 | [C5 1.6/1.8–1.13](changes/afd-hccl-v2-graph/tasks.md)；hit/miss/prefill 连续切换，双端模式一致                                                          |
| H09 | Graph compute fork/join、per-stage receive-ready、hybrid DAG；D01/D02                                        | 新增 stage-local HC/Engram/共享态依赖；避免层级大同步                                             | [C6 1.1/1.2/1.5](changes/afd-hccl-v2-multistream/tasks.md)；捕获合法、依赖正确、重复回放                                                                   |
| H10 | Attention 三流计划、F recv/compute/send、接收流和跨层流水；D01/D04                                                       | parent-stream capture 特例与逻辑三流分离；F 最终 join 所有 stage send                            | [C6 1.2/1.5–1.11](changes/afd-hccl-v2-multistream/tasks.md)；开关依赖、event/lifetime、实际 overlap                                                  |
| H11 | stage-local attention metadata、HC/aux；D02/D07/D10；V4.1 还需 U02/U03/U12                                     | shared_attention_state.reset、Engram 持久 buffer/producer/events、shared-KV/index 均需隔离 | [C4 1.7/1.8/1.10](changes/afd-hccl-v2-microbatch/tasks.md)，[C5 1.8/1.11/1.12](changes/afd-hccl-v2-graph/tasks.md)；history、slot 复用和 graph 更新 |
| H12 | MTP merged stage0、N 步 header/phase-ready、target graph+eager/full draft；DSpark A-local；D01/D03/D04/D10/D11 | V4.1 按 RFC DSpark；既有普通 MTP 模型保留适用能力/回归，逐分支说明去向，不能错搬给 V4.1                          | [C3](changes/afd-v41-hccl-dspark/tasks.md)，[C7 1.1/1.3/1.4](changes/afd-hccl-full-parity/tasks.md)；真实接受/拒绝及完整调用序列                           |
| H13 | 五类 PD+AFD/DSpark/U2/Graph、layer-addressed KV transfer 与 async admission；D12–D15/D18                       | v0.23 patch 需审查目标等价接口；V4.1 shared KV/index 与 Engram handoff 必须闭合                   | [C7 1.2/1.5/1.9](changes/afd-hccl-full-parity/tasks.md)；真实 P→A→F 请求/精度，不用 healthcheck 替代                                                    |
| H14 | 六个 overlap、两个 diagnostics、A/F profiler 和阶段日志；D01/D04/D14–D17                                              | 显式 0/default/依赖、各 rank 有效值、profiler 管理不被 F recv 阻塞                                 | [C6 1.7/1.8](changes/afd-hccl-v2-multistream/tasks.md)，[C7 1.6/1.7](changes/afd-hccl-full-parity/tasks.md)；all-on/off、RPC、trace             |
| H15 | checkpoint/拓扑/capture-size preflight、冷启动/取消/退出/故障后重启；D09/D10/D14/D15                                      | 保留 C1 terminal 监督语义，扩展到所有代表组合；重新预算完整模型和图内存                                         | [C7 1.6/1.8–1.12](changes/afd-hccl-full-parity/tasks.md)；功能+数值+timeline+同卡数性能完整证据                                                           |

## 三、主要阅读代码

源码标号、精确提交及路径可由 [sources.lock.json](../.agents/dsv41/sources.lock.json) 查阅。以下链接全部固定到供体提交，源码存在性已本地核对。

| 阅读目的 | 供体代码/关键符号 | v0.30 目标接入点 |
| --- | --- | --- |
| 通信/事件/图 lowering | [D01 p2p_hccl.py](https://github.com/wenhow/afd-plugin/blob/4ddd69a391dfb5b3e362c7601399e538a31de702/afd_plugin/connectors/npu/p2p_hccl.py)：_graph_hccl_send/_recv、HCCLAttentionGraphStreamPlan、recv_attn_output_streamed | [当前 connector](../afd_plugin/connectors/npu/p2p_hccl.py)、[protocol](../afd_plugin/connectors/npu/hccl_protocol.py)、[transport](../afd_plugin/connectors/npu/hccl_transport.py) |
| layer-major 与 continuation | [D02 deepseek_v4.py](https://github.com/wenhow/afd-plugin/blob/4ddd69a391dfb5b3e362c7601399e538a31de702/afd_plugin/model_executor/models/deepseek_v4.py)：forward_ubatches_layer_major、_forward_ubatches_graph_compute_pipeline、_complete_pending_stage | [目标 V4.1 adapter](../afd_plugin/model_executor/models/npu/deepseek_v41.py)，按 RFC 原生 V4.1 骨架重建 |
| A 侧准备、capture/dispatch | [D03 attention_model_runner.py](https://github.com/wenhow/afd-plugin/blob/4ddd69a391dfb5b3e362c7601399e538a31de702/afd_plugin/v1/worker/npu/attention_model_runner.py)：_warmup_and_capture、dispatch_cudagraph、IDs pretransfer | [V2 A runner](../afd_plugin/v1/worker/npu/attention_model_runner_v2.py)、[UBatchRunner](../afd_plugin/v1/worker/npu/ubatch_runner.py)、[V2 patch](../afd_plugin/compat/patches/npu/mrv2_ubatching.py) |
| F 侧调度/多流/图 key | [D04 ffn_model_runner.py](https://github.com/wenhow/afd-plugin/blob/4ddd69a391dfb5b3e362c7601399e538a31de702/afd_plugin/v1/worker/npu/ffn_model_runner.py)：_initialize_ffn_stream_pipeline、_make_graph_key、_mtp_ffn_forward_sequence、capture_model | [当前 F runner](../afd_plugin/v1/worker/npu/ffn_model_runner.py)：execute_hccl_step 当前固定 U1 |
| stage-local metadata/归并 | D05–D08 的 ubatching、npu_ubatch_wrapper、ubatch_utils、mla_graph；D10 测试 | V2 slice/prepare、child context、merge；V4.1 graph 参数另需 U02/U03/U12 |
| PD/配置/观测 | D11 feature_validation、D12/D13 dspark_pd*、D14/D15 recipe、D16 profiler、D17 F worker | 当前 NPU capability guard、原生 KV connector、操作 recipe 与 profiler 管理 |

供体的 _complete_pending_stage 在 V4 结果后采集某些 aux；RFC V4.1 在目标层入口、Engram 前采集。迁移的是调度能力，不能照抄采集时机。

## 四、执行依赖示意

下面是逻辑依赖，实际物理流随 eager/Graph capture 路径选择；Graph parent-stream 例外仍须遵守。

```mermaid
flowchart LR
    A["A: HC pre / norm / Engram-ready"] --> S["A2F: BF16 + FP32 + IDs"]
    S --> R["F recv_done"] --> F["F native MoE"] --> O["F2A send_done"]
    O --> D["A recv_done"] --> C["本 stage HC complete / pre_mix"]
    C --> N["本 stage 下一层"]
```

U2 有两套独立 stage 状态；某 stage 的等待间隙可以推进另一 stage。逻辑阶段存在并不证明设备已经重叠，C6 仍需 timeline。

## 五、开关语义也属于迁移范围

| 名称 | 供体默认 | 必保留/核对的语义 |
| --- | --- | --- |
| AFD_HCCL_EAGER_U2_STREAM_OVERLAP | 1 | eager/U2 独立开关；显式 0 仍执行 U2 |
| AFD_HCCL_GRAPH_U2_COMPUTE_OVERLAP | 1 | Graph compute fork；关闭不代表关闭 Graph |
| AFD_HCCL_GRAPH_U2_HYBRID_DAG | 1 | per-stage receive-ready 依赖，在 compute pipeline 有效时生效 |
| AFD_HCCL_GRAPH_U2_ATTENTION_THREE_STREAM | 1 | 逻辑 compute/send/recv 映射，warmup/capture 保留 parent-stream 例外 |
| AFD_HCCL_GRAPH_U2_FFN_RECV_STREAM | 1 | F 独立接收流，不能错误复用另一 stage 的 ready |
| AFD_HCCL_GRAPH_U2_FFN_CROSS_LAYER | 1 | 依赖 FFN_RECV_STREAM=1，最终 join 末层 sends |
| AFD_HCCL_STAGE_DIAGNOSTICS | 0 | stage/layer/模式诊断，支持 U1 与 U2 |
| AFD_HCCL_FFN_COMPUTE_SYNC_DIAGNOSTICS | 0 | 定位同步，默认关闭，性能结果披露是否开启 |

仅接受 0/1。供体 recipe 的 full-decode-only:2 将 eager overlap 置 0，五项 Graph 开关独立；
显式全关后仍为 Graph/U2，不能以运行 eager 冒充全关对照。各 rank 最终生效值必须可核对。
保留模式隔离和依赖条件，允许有经审查证明的 V2 等价实现。

## 六、最终验收格子与完成定义

以下是最终目标矩阵，按“真实权重 eager/U1 → 各自依赖满足的 U1 组合 → 新增
Engram/U2 适配与联合验收”逐格推进，不表示当前已开放全部组合。原生不支持的
Engram+microbatch 是待解决的适配边界；未解决时保持 pending，不降低完整迁移定义。

| 组 | 必测模式 | 重点 |
| --- | --- | --- |
| 同步基础 | native、AFD eager/U1、AFD eager/U2（overlap=0） | 同 checkpoint 数值、raw IDs/FP32、HC/Engram/KV/history、真实 U2 |
| eager 多流 | eager/U2 overlap=1 与 0 | 延迟消费、lifetime、timeline、吞吐/TPOT/TTFT |
| Graph | FULL_DECODE_ONLY U1/U2，五开关全开/全关及逐项对照 | 冷捕获/真实回放、parent-stream、hit/miss、prefill/阈值回退、图内存 |
| DSpark | eager U1/U2、target Graph/U2 + eager draft | aux 位置/merge、真实接受/拒绝、KV/history、target-only F |
| PD 五组合 | eager/U1、DSpark/U1、Graph/U2、DSpark eager/U2、DSpark Graph/U2 | P/A/F 链、shared cache/index/Engram handoff、prefix/cancel/后续请求 |
| 拓扑 | 等配对、A=kF、F=kA；TP1/TP2 供体范围及当前 V4.1 等 TP 候选 | 非均匀 peer shape、odd/empty、容量、F2A 拆分、TP/SP 原生归约 |
| 普通 MTP 回归 | 适用旧模型 N=1/2/3、merged stage0、target/draft graph 分支 | 完整 N 次交换；V4.1 按原生 DSpark 映射，不强行开启 MTP |
| 生命周期/运维 | 代表格子的冷加载、并发/取消、STOP、故障新进程重启、profiler | 真实业务结果、资源回收、无额外 recv；不承诺错误 HCCL 组同进程自愈 |

每格记录实际硬件/卡数、软件锁、checkpoint/量化、tokenizer、请求/采样参数、命令、退出码、原始日志和结果。
性能比较固定工作量与部署规模，报告总吞吐、TTFT/TPOT、实际卡数/每卡效率、HBM、Graph 冷启动；DSpark 另报真实 acceptance。
矩阵中的组合是交付验证范围，不是“任意并行维度笛卡尔积全支持”的声明；只有已验证的精确组合进入支持表。
源码不支持项应有精确证据，待实现/待验收项保持 pending。不能把供体已有能力降成后续需求，不能把 CPU/mock/历史 V4 结果计为新 V4.1 NPU 通过。

## 七、范围边界

完整迁移限定本表的 HCCL 语义及其执行/PD/观测依赖。Window/CAM 是其他 connector；其独立特性不因同文件出现而自动迁入。
供体本身限制 U2=2 stages、主要图模式 FULL_DECODE_ONLY、TP/拓扑有条件；
不自动新增任意 U、任意 TP 比例、PP/CP、视觉或故障组自愈。
旧类名、线程组织和 v0.23 私有接口可以用 V2 等价实现替代，但必须保留能力、配置行为和验收映射。
