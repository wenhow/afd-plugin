# DSV4.1-Flash AFD 适配计划（v0.30 主线，完整迁移 HCCL 能力）

## 整体进度总览（2026-11-19）

**总目标仍是完成 DSV4.1-Flash 的 AFD 开发与验证，并迁移供体完整 HCCL 能力：
eager/U1/U2、DBO、Graph 捕获/回放/回退、多流/拓扑、DSpark、PD、诊断和性能验收。**
当前已有基础适配和多项 CPU/NPU 组件证据；**完整 AFD 模型输出等价尚未通过，项目未完成。**

2026-11-19 继续开发增量：已接入保留原始量化/CPU Engram 的 TP4 完整权重候选，
覆盖 DSpark/U1、U2、多流、Graph/U1/U2 和 PD；补齐父 Engram 图外 producer、
稳定输入及 P→D 原始历史声明。1,433 项 CPU 回归与 55 个真实首次 EngineArgs 配置通过，
NPU 均未初始化。源码/架构记录见 [C7 本轮证据](changes/afd-hccl-full-parity/evidence/full-checkpoint-integration-20261119/README.md)。
这些是默认关闭的候选入口；A5 首 token 差异和每个完整模型格子仍待用户验证。

**2026-11-19 目标澄清：近期先闭合保留原生 Engram 的真实权重 eager/U1 正确性，
再按依赖验证 U1 下的 Graph、DSpark 和 PD；Engram+U2/DBO 列为新增适配。**
此前“Engram 后置”仅约束新增适配，不表示所有含 Engram 的场景都不支持或都要等待。
无 Engram 的 dummy/BF16 组件证据继续保留，不能替代原始量化 checkpoint 的验收。

### 模型能力与当前交付目标

原始 checkpoint 配有 Engram，完整模型验收保留其层、权重和原生历史语义；CPU offload
只改变表的存储位置。A 拥有 Engram，F 构造前隔离。U1 指不拆 microbatch，不限定卡数。

| 真实权重场景（均保留 Engram） | 已知边界 | 计划目标 |
| --- | --- | --- |
| eager/U1 | 已有原生路径及 AFD 资格入口；历史 AFD 与原生从首 token 起不同 | 当前首要验收目标：固定 checkpoint、软件、输入/采样及可比拓扑，定位差异并验证连续请求 |
| Graph/U1 | DBO/microbatch 限制没有直接禁止它；不据此推定 Ascend/AFD 已支持 | U1 正确性基线后，复核锁定图接口并验证 capture/replay/fallback 与 Engram 生命周期 |
| DSpark/U1、PD/U1 | 该条限制没有直接禁止它们；仍需分别核对 proposal/接受拒绝、history 与 P→D 接入 | 按独立依赖逐项验证；发现新增 Engram 缺口后归入 C7 专项，不等待 U2 先完成 |
| eager/U2、DBO 及 Graph/U2、含 U2 的 DSpark/PD | 当前锁定 vLLM 明确拒绝 Engram+microbatch | 保留最终目标；作为插件新增适配，先评估契约、审查实现，再做真实模型联合验收 |

上游依据为 [Engram limitations](https://github.com/vllm-project/vllm/blob/main/docs/features/engram.md#limitations)
和锁定 vLLM `ced6857a` 的
[`_resolve_and_verify_engram_config`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/config/vllm.py#L1220)：
配置了 Engram 层且 `use_ubatching` 为真时直接拒绝。锁定 Ascend `7e37e29a` 的
[`platform.py`](https://github.com/vllm-project/vllm-ascend/blob/7e37e29a7803d6897a7113088e2b771f83205bc0/vllm_ascend/platform.py#L1014)
也会在普通路径归零 DBO/ubatch 参数。main 文档不是 NPU 支持声明，也不改变源码锁。

近期交付与完整迁移分别报告：U1 精确场景验收通过后可独立记录，U2 的未完成项仍保留，
不得将阶段交付称为 H01–H15 全部完成。当前完整模型输出等价、完整组合和性能仍 pending。
此前计划澄清已进入实施；本轮实现插件内状态接入并执行 CPU 检查，没有启动新的 A3/A5 推理实验。

2026-11-17 的非 Engram 组件收口安排和已交付入口保留。A3 metadata 原因未确认，
暂停追加 A3 设备定位；性能和服务启停专项继续后置，测试必要清理保留。

| 总览     | 当前进度                                                                                                                                                                                                                                                                                             |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 任务清单   | **109 项：已完成 54 项，未完成 55 项；勾选比例 49.5%**                                                                                                                                                                                                                                                           |
| 实现状态 | 角色模型、HCCL、U2、Graph、多流、DSpark/PD 已有实现和有限组件证据；逐项以 C7 原始记录为准，完整权重支持不由组件结果推定 |
| 完整模型验收 | A5 受控 AFD eager/U1 请求已运行成功，但与原生基线从首 token 起不同，**输出等价未通过**                                                                                                                                                                                                                                        |
| 性能验收   | 尚未完成；依赖功能、数值和状态正确性先通过                                                                                                                                                                                                                                                                            |
| 当前工作位置 | **C7：原生 Engram eager/U1 真实权重对照与首 token 差异收口；U1 组合按依赖推进，U2 新增适配单列。** 具体任务及历史证据见 [C7](changes/afd-hccl-full-parity/tasks.md)；A5 由用户手工验证，profiler、启停和性能后置 |

当前真实权重入口见 [A5 完整 checkpoint 验证](../docs/npu/DSV41_A5_FULL_CHECKPOINT_VALIDATION_ZH.md)，
新增十二种 standalone 与五类 PD 的可配置候选及同 TP4 原生对照；
先收口 eager/U1 首 token，再逐格验收。下述 11-17 组件交付记录保留，
其后实际 A5 报告及事后审计以 [C7 证据记录](changes/afd-hccl-full-parity/tasks.md)为准。

11-17 的组件统一入口见 [A5 非 Engram 最小功能验证](../docs/npu/DSV41_A5_FUNCTIONAL_VALIDATION_ZH.md)：
一条命令顺序运行7场景，正常原生对照 + AFD四模式（含多流）+ DSpark/PD两组合。
默认关闭的正常Worker组合资格入口、Graph/U2跨层PD及后台异常归属已完成本轮源码收口；
真实CPU检查342P/0F/E/S，9个首次EngineArgs配置（原生1、A4、F4）通过且NPU均未初始化。
A3第二轮metadata原因未确认，按用户要求暂缓追加设备定位。A5实际执行、完整检查点数值及
Engram相关验收仍pending，不提前增加任务勾选。

此前A5待验入口已推送到 `feat/dsv41-flash-hccl-v030@6b4d6c5`，远端 SHA 已核对：
[DSpark 两场景](../docs/npu/DSV41_A5_DSPARK_FUNCTIONAL_ZH.md)覆盖拒绝后续轮/Ready 与 Graph/U2取消/新请求，
[同 TP4 单层 MoE 对照](../docs/npu/DSV41_A5_AFD_MOE_PARITY_ZH.md)继续定位完整模型首 token 差异。
源码、CPU及干净提交树检查已通过；A5实际设备结果待回传。A3独立正常Worker候选已完成
真实CPU门禁87F→256P/0S、A/F首EngineArgs配置、正常WorkerWrapper/init_device/runner构造，
随后正常dummy模型加载和HELLO/STOP也已实测通过，两端退出0、八卡空闲。
A保留原生Attention/HC，F为原生完整MoE。两层纯SWA候选的KV规划失败已保留；
随后[三层 `[0,0,2]` 首C2候选](changes/afd-hccl-full-parity/evidence/a3-text-normal-worker-20261116/prepared-c2-001/README.md)
真实CPU门禁274P/0F/E/S、A侧原生KV规划/分配/绑定通过：64个物理block、五个逻辑组，
真实int8/FP16 index tuple及FP32 ring沿用原生共享布局；188.186s、两端退出0、八卡空闲。
上述KV轮没有Scheduler或forward/sample；随后[正常执行002](changes/afd-hccl-full-parity/evidence/a3-text-normal-worker-20261116/prepared-execute-002/README.md)
已完成三层prefill、三次远程MoE、首采样721和Scheduler回填，decode在原生
SparseFlashMlaMetadata AICPU失败，整轮未通过。[同步定位003](changes/afd-hccl-full-parity/evidence/a3-text-normal-worker-20261116/prepared-execute-003/README.md)
确认失败在decode C2 metadata内，尚未进入本轮模型forward。
[独立正常原生对照](changes/afd-hccl-full-parity/evidence/a3-text-normal-worker-20261116/prepared-native-control-001/README.md)
已完成721、14两轮及自然退役；79参数的实际设备SHA、首4096B hidden和前三次metadata显式输入与AFD一致。
[补齐query长度004](changes/afd-hccl-full-parity/evidence/a3-text-normal-worker-20261116/prepared-execute-004/README.md)
仍失败，因此未将补参作为生产修复。随后[原输入上下文005](changes/afd-hccl-full-parity/evidence/a3-text-normal-worker-20261116/prepared-execute-005/README.md)
复现失败：实际同线程/device/stream、推理模式开启，输入format2/连续/offset0且无观测错误；
尚未定位AICPU内部具体检查分支。原件及进程自有CANN日志保留，不据部分执行
开放完整模型U2/Graph/PD。见[入口结果](changes/afd-hccl-full-parity/evidence/a3-text-normal-worker-20261116/README.md)。

统计直接取下表链接的八份 `tasks.md`，以 `[x]` 为完成；**49.5% 是任务勾选比例，
不是代码完成率、工时完成率或 NPU 验收比例**。任务大小不同，且存在跨阶段联合验收，
不能用测试条数或已到 C7 推算整体百分比。未勾选也不都等于尚未开发。
各 change 的 `tasks.md` 仍是唯一任务进度表；本页是截至上述日期的派生概览，任务变化时同步刷新。

**最新软件增量（2026-11-15）：** [PD 配置](changes/afd-hccl-full-parity/evidence/dspark-pd-cross-layer-config-review-20261115/README.md)
完整 CPU 模块 234 项通过；[跨层候选](changes/afd-hccl-full-parity/evidence/a3-dspark-native-live-cross-layer-20261115/README.md)
225 项协议核对/64 项生产选择、[eager 多流候选](changes/afd-hccl-full-parity/evidence/a3-dspark-native-live-eager-streams-20261115/README.md)
87 项/32 项及独立归档审计通过；[混合 U2/U1 观测汇总](changes/afd-hccl-full-parity/evidence/eager-mixed-summary-20261115/README.md)
已修复，Windows/Linux 各 52 个 CLI 场景通过。这些不是本组合的 NPU 验收。

随后修复 [HTTP profiler 迟到回复污染](changes/afd-hccl-full-parity/evidence/h14-http-rpc-terminal-review-20261115/README.md)：
失败后禁止复用 HCCL 多进程 RPC 通道，入口限定到 HCCL，并要求显式插件清单保留 `afd`。
继续补齐 [profiler 中断与原异常保留](changes/afd-hccl-full-parity/evidence/profiler-baseexception-review-20261115/README.md)
及 [F STOP 后导出中断上报](changes/afd-hccl-full-parity/evidence/profiler-ffn-baseexception-review-20261115/README.md)。
该轮冻结源码的 Linux 全部 CPU 单测 **1987P/525S，0F/E**；缺失依赖的跳过项不计原生运行时通过。
原生 IPC、实际退出和设备验证仍 pending，已冻结包含这些修复的 A3 CPU 后继包004。
部署清单同步补齐[全部插件 Python 源身份](changes/afd-hccl-full-parity/evidence/recipe-complete-source-20261115/README.md)，
当前149个模块/152项记录，避免新增补丁漏记；目标安装与部署仍需独立核验。
另修复[两部署CLI输出覆盖源码/输入](changes/afd-hccl-full-parity/evidence/recipe-output-source-protection-20261115/README.md)：
同一回归的10个E2E方法通过，包含18个输出边界与原有52个日志汇总CLI场景。

**当前功能工作（2026-11-16）：** 已收集 live-004/005/006 的真实结果。
[控制帧修复](changes/afd-hccl-full-parity/evidence/a3-ready-control-device-20261116/README.md)
完成真实 Torch CPU 默认设备边界 RED/GREEN，完整连接器模块58项通过；标量证据写入也已修正。
[live-006](changes/afd-hccl-full-parity/evidence/a3-dspark-native-live-prefix-20261115/live-006/README.md)
到达原生目标元数据，发现受控 CSA2 配置不一致；已按锁定源码修正，并
[补齐原生元数据算子包](changes/afd-hccl-full-parity/evidence/a3-ready-qli-metadata-20261116/README.md)。
十算子包及原生 AICPU 输出已补齐；live-007 前进到 F 原生 context，暴露后台线程缺少
inference mode，[最小修复](changes/afd-hccl-full-parity/evidence/a3-ready-inference-mode-20261116/README.md)
完成两例 RED 和完整 worker 模块83项 CPU GREEN。
live-008 暴露原生 KV consumer 在纯文本声明下仍需要 tokenizer；验证配置已修正。
live-009 观察到完整 Ready 前沿、真实读取/DONE、整池核对通过；P 功能报告写完后原生退出异常，
整轮仍失败。decode-001 已进入 D 首轮采样/提案，旧随机小模型产生114而非验证包假定的7；
后继在执行前声明合成 Markov 参数，继续通过原生提案/拒绝采样验证，不替换输出。
decode-002 的真实接受路径两轮完成，decode-003 的拒绝后第三轮也完成；均核对实际
草稿、采样、计算位置、累计历史及缓存写入保护，自然达到声明的输出上限。
两轮的四角色功能报告均成功，但 P 报告后原生退出异常，整体仍失败；这些是合成权重
组件观察，不代表真实检查点精度或完整运行验收。
[eager-lifecycle-001](changes/afd-hccl-full-parity/evidence/a3-dspark-native-live-prefix-20261115/eager-lifecycle-001/README.md)
已实际启动取消/连续请求/KV槽复用与 eager U2多流组合；D/F2在原生 V2配置校验阶段因
DSpark及多模态DBO限制失败，尚未构造owner或执行目标生命周期，不能计为功能通过。
插件自有DSpark/U2最小配置接入已实现并经源码审查，本地相邻回归270P/3S；
新原生校验模块因缺依赖跳过，不计通过。A3已启动的旧校验CPU对照实际4F，
精确复现原生spec/MM拒绝；新校验的原生GREEN及设备功能验证转A5，不再追加A3 DSpark推理。
**按用户11-16最新安排，A3接下来推进非DSpark：普通文本U2的真实配置前门、
普通PD及U2/Graph/多流正常接入链。** DSpark既有A3原始证据保留，后续A5场景集中精简。
普通文本/noSpec U2 已完成[原生配置方法对照](changes/afd-hccl-full-parity/evidence/a3-text-u2-config-admission-20261116/README.md)：
相同最终补丁及测试在旧 runtime 下4F，安装后普通文本33项边界及1项签名检查共34P/0S；
只变更 runtime 安装，446源码及191安装身份前后不变。
随后真实 EngineArgs 首次声明U2的 A/F×eager/Graph 四项完成4拒绝→4通过，
保留MM身份/有效U2，NPU均未初始化；447源码/191安装身份不变。
继而首声明PD consumer的 A/F×eager/Graph 四项也通过，保留原tokenizer配置、
attested connector、equal TP及原生MM embeds策略，未初始化NPU；两类配置合计八个独立进程检查。
上述八项host检查不含训练参数加载。随后新增D/F两个受控H16首构造host进程通过，
并把同一工厂返回的配置接到[普通PD Graph设备链](changes/afd-hccl-full-parity/evidence/a3-native-pd-graph-engine-args-20261116/README.md)：
首eager/U2、U2/U1回放、取消/幸存请求及新C读取/请求槽复用实际通过；P/D/F均0，八卡idle。
五轮数值/累计整池字节和35个父事务/每端355canonical帧独立审计通过，14项语义篡改拒验。
受控owner仍未执行完整Worker构造或训练参数加载；该正常接缝继续开发，完整等价未通过。
只跑定位功能缺口所需的回归；profiler、性能及服务启停
专项后置，不作为功能推进的前置门槛。每轮测试必要的资源清理仍保留。任务总数仍为 **54/109**。

### 非 DSpark 功能当前状态（2026-11-16 核对）

| 功能 | 已实际完成的验证 | 仍需完成 |
| --- | --- | --- |
| HCCL 基础与角色切分 | 正常 factory/F 服务、多层 BF16/独立 FP32/IDs、连续事务、TP1/TP2 通信组件；A/F 构造及参数隔离已有证据 | 完整检查点的路由、归约和输出等价；A5 AFD 与原生首 token 差异仍未闭合 |
| 普通 PD | 真实 P→D KV/整池核对、Scheduler、D 执行采样、取消后新 ID 及块清零已有 A3 通过证据 | 正常完整 Worker/模型入口及完整矩阵；在途读取抢占不在既有结果范围 |
| U2 | TP1 聚合/fanout、TP2 等配对、同 session U1/U2/empty 组件已通过；普通 text/noSpec 配置方法4F→34P、原八项首构造host通过；新增D/F首构造已接到PD设备组件，首eager/U2及后续Graph回放通过 | 完整Worker/模型构造仍待；完整原生 HC/shared-KV/index 数值与归并未闭合 |
| Graph | 四个原生 U1/U2 图及动态输入/KV回放；普通 PD 的 U2 回放、取消后幸存请求切 U1、新 C 同 owner 读取并复用 B 请求槽后 eager→U1 replay 已通过 | 正常完整 Worker/模型入口、SparseFA 与状态联合对照；新 C 结果仍为受控算术组件 |
| 多流 | eager TP1/TP2 流和 ticket、Graph 开关组合及跨层组件已有 NPU 证据 | 完整模型多轮输出/状态、请求和 bucket 切换；timeline/性能专项后置 |

上述通过项均保留各自组件边界，不等于完整检查点验收。普通文本 U2 配置前门的host对照已完成；
[普通 PD Graph 新请求/请求槽复用](changes/afd-hccl-full-parity/evidence/a3-native-pd-graph-replace-20261116/README.md)
亦实际通过：C复用B槽6、真实第三份KV/DONE、首eager后U1回放，P/D/F退出0、8卡空闲。
普通U2首构造现已进入受控设备组件；A3继续[真实Worker构造的独立设计](changes/afd-hccl-full-parity/evidence/text-u2-normal-worker-design-20261116.md)，
保留默认完整模型门禁。DSpark设备功能统一在A5验证。

### 各阶段完成了多少、还差什么

| 阶段 / 任务入口 | 已完成 / 总数 | 未完成 | 已有工作 | 主要剩余 |
| --- | ---: | ---: | --- | --- |
| [C0：运行时与契约](changes/afd-v030-rfc-runtime/tasks.md) | 11 / 13 | 2 | 初始源码/接口/环境复核 | 新增 U2/Graph/PD/Engram 契约及整体增量架构审查收口 |
| [C1：HCCL connector](changes/afd-hccl-v2-connector/tasks.md) | 12 / 14 | 2 | eager/U1 协议、传输和生命周期组件 | U2/Graph/多流协议演进与联合验收 |
| [C2：V4.1 角色模型 / eager](changes/afd-v41-hccl-text-eager/tasks.md) | 6 / 14 | 8 | A/F 构造、MoE 边界、权重过滤、独立 FP32 传输；A5 资格请求已运行 | 输出差异定位、完整路由/容量/连续并发对照、Engram 生命周期与清理 |
| [C3：DSpark](changes/afd-v41-hccl-dspark/tasks.md) | 4 / 12 | 8 | target/draft 隔离、proposal/采样/连续轮次 CPU 组合；A5 原生 N=1 基线有接受率 | AFD DSpark 真模型、资源/接受拒绝/回退/历史与性能，及 U2/Graph/PD 组合 |
| [C4：U2 / microbatch](changes/afd-hccl-v2-microbatch/tasks.md) | 7 / 12 | 5 | stage/peer 协议、continuation、归并和 TP1/TP2 受控 NPU 组件 | 原生模型 HC/shared-KV/Engram 联合对照、活动取消、完整拓扑与回归 |
| [C5：Graph](changes/afd-hccl-v2-graph/tasks.md) | 4 / 13 | 9 | 图内通信组件、协调/key/稳定 buffer/回退实现及组件证据 | 正常完整入口、模型状态更新、真实模型 capture/replay/miss、资源与联合回归 |
| [C6：多流](changes/afd-hccl-v2-multistream/tasks.md) | 9 / 12 | 3 | eager 三流、Graph parent/fork/join、跨层接入、开关与诊断，独立组件证据 | 模型开关对照、长运行与 A/F timeline/性能 |
| [C7：组合与交付](changes/afd-hccl-full-parity/tasks.md) | 1 / 19 | 18 | DSpark/PD 非 Engram 接入、异常/重启、profiler、冻结清单和配置日志等增量 | 部署观测闭环、Engram 7 项、完整 standalone/PD 矩阵、性能与交付审计 |
| **合计** | **54 / 109** | **55** | **已有实现和组件证据继续保留** | **按实际完成条件收口，不提前勾选** |

C0/C1 的基础任务完成后追加了完整迁移任务，所以仍各余 2 项。
C7 的任务较大：即使某部分代码和组件已通过，只要整项包含的模型/状态/组合验收未闭合，仍不勾选。

### 剩余工作与执行顺序

剩余 **55 项 = C0–C6 的 37 项 + C7 的 18 项**。C7 的 18 项进一步分为：

| C7 剩余任务组 | 数量 | 任务编号 / 执行安排 |
| --- | ---: | --- |
| 高优先级功能实现与独立验证 | 4 | 1.3、1.5、1.6、1.10：DSpark/PD、功能配置、并发/取消/随后请求；1.10 的服务启停/重启部分后置 |
| 后置 profiler 专项 | 1 | 1.7：管理入口、采集启停与 trace；不阻塞功能推进 |
| Engram 专项 | 7 | 2.1–2.7：复用原生 U1 的验证纳入当前目标；新增准入、U2/Graph、多流及 DSpark/PD 缺口仍按中优先级审查实施 |
| 完整模型与最终交付验收 | 4 | 1.8、1.9、1.11、1.12：standalone、五种 PD 组合、性能、H01–H15 审计 |
| 分支映射及暂停的旧 V4 MTP | 2 | 1.1 是混合映射任务，1.4 为旧 MTP 后续开发；旧 MTP 保持暂停，不阻塞 Engram 开始 |

上表是任务归属统计；C0–C6 的未完成项与 C7 有依赖和联合验收，不能把 55 项理解为
55 个相互独立、尚未编码的功能。实际继续开发按以下顺序：

1. **先闭合真实权重 eager/U1 正确性。** 保留原生 Engram，以
   [完整权重指导](../docs/npu/DSV41_A5_FULL_CHECKPOINT_VALIDATION_ZH.md)收集新锁下的
   载权、输入/输出、连续请求和正常清理证据。历史首 token 差异仍是阻断项；同 TP4
   [MoE 对照](../docs/npu/DSV41_A5_AFD_MOE_PARITY_ZH.md)只用于定位，不代替整模型结果。
2. **按依赖扩展 U1 验证。** 逐项核对 Graph/U1、DSpark/U1、PD/U1 的锁定接口，
   在各自基线和准入满足后验证图执行、接受/拒绝、history/handoff 与随后请求。
   这些路径不被 Engram+microbatch 的拒绝条件一并排除，也不因未被禁止就算已支持。
3. **收口独立组件的实际缺口。** 保留非 Engram 的 U2/Graph/多流及 DSpark/PD 证据，
   修复必要的正常接入和错误传播，不反复重跑已有组件以代替完整模型验收。
4. **评估并实施新增 Engram 适配。** 高优先级可独立工作具备后，按 C7 2.1–2.6
   复核实际缺口：U2 准入、父 producer、stage 视图、buffer/event、Graph 与历史交接。
   新增补丁先架构审查，保留非 AFD 原生拒绝；未闭合时 U2 完整模型仍 pending。
   U1 数值基线是 U2 完整模型对照的前置，不要求完整 U2 矩阵先通过才允许开展设计。
5. **逐格验收及交付。** 已通过的 U1 精确场景可独立记录；原定 U2/Graph/DSpark/PD、
   拓扑及 H01–H15 最终范围保留，全部满足才报告完整迁移。性能和启停专项最后安排。
   需要用户手工 A5 验证时先交付最小指导，等待期间继续不依赖设备结果的审查和 CPU 工作。

最新补齐 **正常 F factory→worker daemon→runner→Graph service 的真实 A3 组件**：
五轴全关/全开各完成 23 个 parent（4 warmup、4 capture、12 replay、3 eager 回退），
真实 NPUGraph/default pool、独立 FP32/IDs、U2 bank 地址复用和 STOP/join 通过；
正常退出后八卡 idle。见[服务审查](changes/afd-hccl-full-parity/evidence/graph-ffn-service-review-20261113.md)。
该组件跳过重型 worker/runner 构造，MoE 为受控算术，不等于完整模型、物理重叠或性能验收。
同 TP4 的新 A5 对照已通过 A3 导入/注册/配置，数值计算仍待 A5。

后续[Graph 故障/重启](changes/afd-hccl-full-parity/evidence/graph-ffn-failure-review-20261113.md)
通过预期异常传播、缓存保留和全新进程恢复组件；故障轮实际由监督终止对端，不称正常清理。
另修复 F 在 STOP_ACK 后、profiler 尚未导出时过早返回结束的问题：确定性 RED 两例均复现，
GREEN 五完整 CPU 模块单轮 222 项通过；A3 实际 EngineCore/MP/HCCL/profiler 中
自然导出 7.03 秒后才报告结束，随后 stop RPC 0.26 秒，进程正常退出、八卡 idle。
见[STOP/export 审查](changes/afd-hccl-full-parity/evidence/profiler-stop-poll-review-20261113.md)。
这些仍是各自冻结的组件证据，不替代完整模型和最终组合验收。

新增[真实 F Graph × profiler 组件](changes/afd-hccl-full-parity/evidence/graph-ffn-profile-review-20261113.md)：
全关/全开各 23 个 parent，每会话 trace/CSV 核对 60 次 MatMul、240 个 HCCL
任务，其中 36 次 MatMul 来自真实 replay；STOP 后 owner 导出、退出及八卡 idle
通过。首轮 CSV Step ID 沿用 capture 导致的后处理失败已保留，修正归属后直接
离线重读原包；没有重复全关设备。A 侧 timeline、物理重叠与性能仍未证明。
DSpark/PD 已接入路径的[前次源码复查](changes/afd-hccl-full-parity/evidence/dspark-pd-lifecycle-source-review-20261113.md)
保留其检查范围；后续实际 native bind/deferred-load 检查已发现正常 P 的登记时序问题，
见下方新增证据。2026-11-15 的[后继源码复查与验证设计](changes/afd-hccl-full-parity/evidence/dspark-native-pd-graph-successor-design-20261115.md)
确认当前窄域 D-side Graph/U1/U2 配置、loaded consumer/native KV、aux 归并及采样发布
已有接线，旧记录不能沿用为“未接线”的当前结论。组合设备与完整模型验收仍待完成；
完整模型门禁继续关闭，不能把组件实现当作整体验收。

Graph 外层[父事务观测](changes/afd-hccl-full-parity/evidence/graph-host-observations-review-20261113.md)
已补齐实际 reservation/replay/finish/失败记录与严格逐 rank 汇总。最新六完整
CPU 模块单轮 380P、0F/E/S，双端串行 codec/control 的五个父事务摘要一致；
既有 52 与新增 34 个 CLI 场景分别通过，真实 CPU 记录经 codec 派生清单后也
通过公开汇总；startup 为构造前载体上的生产方法读取。独立真实 F Graph 组件
另有 23 parent/90 条 host 记录，A/F 正常退出、八卡 idle。原容量夹具失败和未知 host 错误
误报成功均保留并修复。实际部署、NPU Graph 与完整组合验收仍待完成，继续
推进合法 P-prefill/U2 产物到独立 PD consumer 的数据合同，保留正常组合门禁。

后续[U2 prefill 数据合同](changes/afd-hccl-full-parity/evidence/prefill-u2-pd-data-contract-review-20261113.md)
已通过两个定向用例及九完整 CPU 模块单轮 561P、0F/E/S：2/5/6 请求实际跨 6+7 stage，
原始 parent/三层 aux 经投影、typed read/DONE 到 D 实际 sample/proposal。受控 leaf 与
独立 no-PD/U2、U1/PD 配置的边界明确，尚不开放正常组合。完整回归另修正手工
staged 绑定被配置日志破坏的问题，原失败保留。
正常异步 P 的[登记修复](changes/afd-hccl-full-parity/evidence/async-p-admission-review-20261113.md)
已从 4F/1P 复现到相同五例全通过：在 bind 提前登记 source，保持原生延后 load，
同批晚到 post 不复活 DONE/取消，修改已绑定 metadata 会拒绝。完整模块/签名回归
571P/1F 后，仅修正缺字段夹具的整个 KV 模块 109P 通过，各批独立记录。
后续 [Ready 顺序修复](changes/afd-hccl-full-parity/evidence/p-ready-frontier-review-20261113.md)
已在两项真实 RED 后通过五项定向回归：最终投影/Ready 移到 native sample/proposal/post
之后，pending 重入/关闭与原始失败保留；同时修复单 token P 的输出上限及
[最终 SWA 页回收](changes/afd-hccl-full-parity/evidence/p-final-retirement-review-20261113.md)。
仅 P 侧 exact HCCL U2 配置条件、同 owner load/KV/原生采样/Ready 的
[CPU 接入](changes/afd-hccl-full-parity/evidence/p-producer-u2-review-20261113.md)已通过；
随后十完整 CPU 模块及三项签名单轮 **721P、0F/E/S**，
见[合并审计](changes/afd-hccl-full-parity/evidence/p-quiescence-combined-validation-20261113.json)。
随后[两真实进程的最终错误报告](changes/afd-hccl-full-parity/evidence/p-report-gloo-review-20261114.md)
在实际 CPU Gloo 上通过正常与单 rank proposal 失败场景，三个受影响完整 CPU 模块
另轮 276P、0F/E/S。后续[D/U2 接入](changes/afd-hccl-full-parity/evidence/pd-consumer-u2-design-20261114.md)
通过同 owner 两轮 native sample/proposal、首次 Q1/U1 fallback→cached Q3/U2 与 Q3
跨 child/接受拒绝检查；十完整 CPU 模块及三项签名单轮 736P、0F/E/S。
后续[失败 PD owner 的公开 shutdown 边界](changes/afd-hccl-full-parity/evidence/pd-failed-shutdown-design-20261114.md)
由 U1/U2 两项真实 RED 复现，最小两处生产检查后，十一完整 CPU 模块及四项签名
同轮 755P、0F/E/S。后续[Graph/PD 组合](changes/afd-hccl-full-parity/evidence/pd-consumer-graph-design-20261114.md)
已补齐 exact D target-only FULL_DECODE_ONLY 的配置门禁：同测试 RED 40P/29F →
GREEN 69P；原生 PD handoff→eager 首轮→cached FULL 的 U1/U2 及 U2 异常保留
三个 CPU 场景通过。13 完整模块及 4 签名契约同快照单轮 **1057P/0F/E/S**。
FULL 输出和 graph handle 是预录 CPU 输入，不据此宣称同一 PD owner 启动 capture
或真实图中 KV 读写。继续 U2 FULL 元数据和实际 A capture/replay；Engram、设备绑定和完整模型资格仍按
原条件控制。CPU 回归不能替代设备/模型任务，勾选数保持 54/109，不增加 A5 手工场景。

随后[U2 FULL 元数据差异](changes/afd-hccl-full-parity/evidence/u2-full-metadata-design-20261114.md)
已修复：真实压缩缓存 builder 的 capture/live FULL/eager 六场景 RED 4P/2F →
同测试 GREEN 6P，仅补一个原生模式参数。包含通用 NPU 路径的十五完整模块及四项
签名单轮 **1223P/0F/E/S**；仍是 CPU 元数据范围，继续实际 A capture/replay 设备链路。

同日[正常 F 服务的 query3 Graph 组件](changes/afd-hccl-full-parity/evidence/graph-ffn-q3-service-design-20261114.md)
完成 A3 实测：真实 factory/daemon/runner/default NPUGraph，query3 每 rank 24 个父事务、
query1 同快照回归 23 个，正常 STOP 后八卡空闲。query3 包含 U2 4/5 行 capture、
同 key 4/2 短 batch replay 和请求布局 miss→eager。155 项安装源码前后核对通过。
这是两层受控算术和 F 服务的设备组件，A 仍使用 base manager；不代表 PD/KV、
真实 DSpark proposal/接受率、完整模型、Engram 或性能验收，不改变任务勾选。

随后[原生 A Graph 组件](changes/afd-hccl-full-parity/evidence/native-attention-graph-design-20261114.md)
完成 A3 实测：真实 ModelState/input/metadata/Graph manager 与原生 capture factory，
四个 U1/U2 图、十二次新输入/位置及同地址 KV 变化 replay，各 rank 21 个 parent，
STOP 后八卡空闲。标准库独立重算输入/KV 坐标、逐步 BF16/FP32 输出和 KV 摘要通过；
156 项安装身份、143 份生产闭包及 CANN/Triton 原文件前后核对。前三轮初始化与
CANN 布局失败保留，第四轮仅在 scratch 镜像补编译缓存身份路径后通过。
Attention/HC 仍是受控算术、owner 为 constructor-free，尚未覆盖公开 execute/sample、
PD、真实 SparseFA 或模型等价。当前继续公开执行、取消/零 token、fallback/随后 replay；
整体仍为 54/109 项已勾选，不增加 A5 手工场景。

同一 A owner 的[公开执行/采样生命周期](changes/afd-hccl-full-parity/evidence/native-attention-lifecycle-design-20261114.md)
随后在 A3 首轮通过：两次空 parent 原生退休、两次新请求 prefill eager、两次 cached
FULL，并真实执行 native Sampler/AsyncOutput/history；A/F 共各 27 个 parent。
取消后新请求复用同一槽、改读自己的物理块；四份有效输出、历史及 KV 摘要经独立
标准库重算/核验，159 项安装身份通过，正常退出后八卡空闲。只验证受控
Attention/HC/LMHead，不据此闭合模型/PD/DSpark/Engram 组合；继续原始异常、STOP
前关闭拒绝和全新进程恢复场景。

此前 **C7 1.6 的逐 rank 汇总工具已完成首个 eager/U2 范围**，
52 个 CLI 场景及 A3 TP1/TP2 真实组件记录通过校验，每 rank 六个 parent 正常完成；
**C7 1.5/1.10 修复 PD 失败后继续执行排队请求的问题**，六个完整 CPU 模块单轮
333 项通过。详见[观测审查](changes/afd-hccl-full-parity/evidence/hccl-observations-review-20261113.md)
和[排队失败审查](changes/afd-hccl-full-parity/evidence/pd-retained-failure-review-20261113.md)。
这些是不同范围的独立结果，不相加；实际部署和完整组合验收仍待完成。
C6 1.6 经[供体流约束与实现审查](changes/afd-hccl-full-parity/evidence/graph-parent-transport-parity-20261113.md)
确认调度实现及组件证据已齐，本次勾选；C4 1.9 经[CPU 协议审查](changes/afd-hccl-full-parity/evidence/c4-cpu-closure-review-20261113.md)确认六完整模块 688P、0F/E/S 后关闭；**总数为 54 / 109，未完成 55 项**。
C6 1.10–1.12 的组合、长运行及 timeline/性能继续待验，不以增量测试数量提前勾选整项。

<details>
<summary>实施与验证历史详情（展开查看原始增量及证据）</summary>

更新：2026-11-14。当前开发指向 **C7 `afd-hccl-full-parity`**，优先 DSpark 等其他
DSV4.1 功能；Engram 新增适配调整为**中优先级，其他功能具备后开始**，见
[后续排期](#engram-priority)。
C2–C6 的完整模型和部分设备验收尚未闭合，进入 C7 不表示此前阶段已全部完成。
C4 的 U2 协议/runner、C5 的 Graph 协调与回退、C6 的多流基础均已有实现及 CPU/
受控 NPU 组件证据；完整模型输出、Engram/KV 状态、物理流重叠与性能仍待验收。
DSpark 已交付原生 proposal/采样历史、Graph 命中/回退及 PD 接收和请求接入组合；
首步 target/sample/proposal CPU 组合在 A3 修正测试夹具后，三个完整模块单轮
193 项通过、125 项安装源码一致；同一请求的第二轮调度组合八个完整 CPU 模块
单轮 561 项通过、127 项安装源码一致；后续完成请求结束/取消后的同 ID 本地新请求
及 target 入口异常 CPU 边界。现有十四个完整 CPU 模块单轮 1015 项通过、138 项
安装源码一致，包括独立 F 本地启动握手入口和退出诊断。完整模型默认及 PD
Graph/U2 门禁仍关闭；仅显式开放受控 eager/U1、DSpark off 的 A5 资格候选，
普通旧 V4 MTP 后续开发按用户优先级暂停。各 change 的 tasks.md 是唯一进度表，
不能以阶段指针或测试用例数代替验收。

本次继续开发补齐已注册 DSpark PD 的采样/proposal 失败边界：保留已更新历史和
KV、传播原异常并拒绝继续复用失败 owner。15 个完整 CPU 模块单轮 1042 项通过，
141 项安装源码与锁定 Git 一致；见 [审查和证据](changes/afd-hccl-full-parity/evidence/dspark-pd-sampling-failure-review-20261112.md)。
这是 C7 1.3/1.5/1.10 的非 Engram 增量，整项及完整设备验收仍 pending。

后续同轮已补齐原生异步输出失败监督与同 ID 第二次远端 PD 读取；A3 真正执行
HCCL COMPLETE 后的 idle STOP/ACK 和全新进程重启，两轮 A/F 均正常退出。
另修复 F 管理 RPC 失败后的迟到回复风险：通道失效后禁止后续 collective 清理，
由原生 executor 监督进程。最终 16 个 CPU 模块单轮 1107 项通过、142 份锁定源码一致；
独立 A3 Unix/UniProc/F daemon/HCCL/profiler 组合完成 start/stop，trace/CSV 对齐
8 个实际 MatMul kernel，正常退出后八卡 idle。见 [异步与 PD 证据](changes/afd-hccl-full-parity/evidence/dspark-output-failure-review-20261112.md)
及 [H14 审查](changes/afd-hccl-full-parity/evidence/h14-rpc-terminal-review-20261112.md)。
这些是 CPU/设备组件证据；C7 1.7、完整模型、Engram 及性能任务保持 pending。
随后 A3 同一 F owner 的 profiler 重启/active STOP 也通过：两个 prefix 各 8 个
实际 MatMul kernel，第二份由 STOP 后的 owner.finish 导出；导出 3.75 秒、stop RPC
4.10 秒，A/F exit0、无强制回收且八卡 idle。该次观测不保证其他规模的导出耗时，
随后真实 MP 共享内存 RPC 组件也完成两周期/最终 STOP：145 份安装源码一致、
190 份 trace 文件校验、两个 prefix 各 8 个 MatMul kernel；管理进程与设备 owner
分离，原生 executor 在显式 30 秒宽限内正常退出，八卡 idle。原生默认队列容量和
5 秒退出宽限的失败轮次保留，见 [MP 审查](changes/afd-hccl-full-parity/evidence/h14-multiproc-review-20261112.md)。
后续真实 EngineCore 控制循环/输入队列/清理与上述 MP 组件组合也通过：152项源码、
170份trace文件核验，正常 SystemExit、engine cleanup及全部子进程自然退出、八卡idle。
001 缺少原生 manager 平台初始化而在最后allocator清理失败，002仅补已有native hook
后通过，原失败保留；见[EngineCore审查](changes/afd-hccl-full-parity/evidence/h14-engine-core-review-20261112.md)。
原生构造/前端、失败注入、完整模型与性能仍待验收，C7 1.7 保持 pending。

部署预检的 CPU planner 已通过本地20边界及 A3 原生13组方法对照；后续 eager
family HELLO 的三个完整 CPU 模块484P，真实四 rank 的单 F 配置不一致在 HCCL
数据组/warmup 前全员拒绝，配置一致时双 stage warmup/串行算术/STOP通过。
原生TP组仍可用、全部正常退出且八卡idle，见[协商审查](changes/afd-hccl-full-parity/evidence/eager-family-review-20261113.md)。
两阶段装配已实现并完成[合同审查](changes/afd-hccl-full-parity/evidence/eager-bootstrap-review-20261113.md)，
完整模型准入不变，C7 1.6 保持 pending：
11完整CPU模块单轮728P、152源码一致，覆盖实际A load/KV和F工厂/线程/清理入口。
独立A3 connector/bootstrap组件在TP1/TP2均通过warmup、串行profile、受控KVcallback
后同session六U2 parent/tickets/diagnostics/STOP，每rank16个精确payload检查，全部正常
退出且八卡idle。不把CPU入口与设备callback拼成真实模型/nativeKV验收。
后续actual F runner/consumer在独立设备线程的TP1/TP2也通过：每F rank16次真实forward
context与native/TLS compute stream精确，六parent+STOP后七次恢复ambient，全部exit0、
nativeTP仍可用且八卡idle；[原包审计](changes/afd-hccl-full-parity/evidence/eager-ffn-validation-20261113.json)
明确不含runner/worker构造或真实daemon/nativeKV/MoE，不扩展为完整服务验收。

逐 rank 启动配置观测增量也已完成：正常 staged A/F 在 HELLO/prepare 成功后，
用原生 INFO 通道记录冻结的八项 configured 值和协商身份，两个诊断开关全关时仍可见。
A3 configuration-002 的同批11完整CPU模块763P、0F/E/S，154份锁定安装源码匹配，
见[原包审计](changes/afd-hccl-full-parity/evidence/eager-configuration-validation-20261113.json)。
这是 KV/线程启动之前的 session 配置记录，不是模型 ready 或 actual overlap；
旧 P2p U1 已在后续增量接入冻结快照与单次配置日志，详见下述最新审计；实际启动与
完整逐 rank 汇总仍待补齐，C7 1.6 保持 pending。

后续已新增[双角色冻结清单工具](../tools/diagnostics/prepare_dsv41_hccl_recipe.py)和
[recipe编制入口](../recipe/npu/P2pHcclAFDConnector/deepseek_v41/README.md)：复用预检结果
生成A/F实际argv、同一八项快照、端口和输入/源码身份，保留Graph/U2及显式0。
A3单轮37个stdlib CLI场景及12份清单/24次实际native A/F参数解析通过，158份安装源码
一致，前后八卡idle；未创建engine/model或执行NPU，见[最新审查](changes/afd-hccl-full-parity/evidence/legacy-configuration-review-20261113.md)。
该standalone投影明确拒绝DSpark/PD、跨host和多TP world，不绕过准入；实际启动、
完整rank/actual事件汇总仍缺，1.6保持pending。最新native-004已将legacy和A/F
日志入口纳入18份插件源身份，结果与旧native-003分开保存，不相加计数。

旧U1接入的最新A3 legacy-002回归为12完整CPU模块842P、0F/E/S，154份锁定
安装源码匹配；[原包审计](changes/afd-hccl-full-parity/evidence/legacy-configuration-validation-20261113.json)
核验了归档、测试/生产快照和3入口签名。正常A/F冻结一次八项配置，手工None未知，
registry替换保留；HELLO成功后经native INFO单次记录真实pair坐标，不改wire/streams。
首轮834P/8F仅修两个测试fake，生产字节一致；CPU不替代模型/NPU/完整矩阵验收。

A5 与开发并行，按[最小两场景说明](../docs/npu/DSV41_A5_PARALLEL_MINIMAL_ZH.md)
回传的原生 eager 基线一个 32-token 请求及原生 DSpark N=1 两个顺序 32-token 请求
已通过，三个输出逐 token 一致；DSpark 每请求真实 draft 19、接受 13，详见
[2026-11-12 结果](changes/afd-hccl-full-parity/evidence/a5-native-dspark-n1-20261112/README.md)。
退出清理未闭合；A5 实际 Ascend `4ec6dc7` 与开发锁 `7e37e29` 不同，结果限定于
实测环境，不扩展为 AFD、PD、Graph/U2、N=2/3 或性能验收。沿用 DP8/TP1/EP8 和
CPU Engram 布局，操作入口见[原生说明](changes/afd-v41-hccl-text-eager/evidence/A5_NATIVE_EAGER_ZH.md)。

补充包四份安装源码均已核验为实际 `4ec6dc7` 字节，新增依赖身份确认，开发锁保持。
[受控 AFD eager/U1 请求](../docs/npu/DSV41_AFD_EAGER_QUALIFICATION_ZH.md)也已回传：
packed Attention 四例通过、HTTP200/32-token 请求完成、进程均 exit0/八卡 idle；
与既有原生基线从首 token 起不同，输出等价未通过。四 rank 清理 INFO 标记缺失的
日志问题已修复，随下一项有区分度的验证复核；不为日志单独重复完整模型请求。
[真实单层 MoE 的 native TP1/TP4 对照](../docs/npu/DSV41_A5_MOE_PARITY_ZH.md)
已于2026-11-13完成，见[回传分析](changes/afd-hccl-full-parity/evidence/a5-moe-parity-20261113/README.md)：
五个rank exit0、前后八卡空闲，prefill/decode的gate logits逐项一致；MoE输出相对L2
误差分别0.436075%/0.449229%，最大绝对误差均0.25。未预设容忍度或验收等价。
精简报告没有独立top-k/shared/routed/归约中间结果，不能将完整AFD首token差异归因
到某一步；也未独立审计完整源码/权重身份。原始产物在A5保留，本轮无需重跑该场景。
后续若需验证AFD边界，应优先同TP布局的native/AFD对照；AFD DSpark N1继续等待
输出等价与单独准入。开发先推进C7 1.6冻结recipe清单和逐rank观测闭环。

### 2026-11-09 历史增量

2026-11-09 并行开发更新：C3 guarded 准备代码已交付至 `eb41916`；C4 布局和 parent
协议已交付至 `acf5398`，本轮继续 [实际传输组件](changes/afd-hccl-v2-microbatch/evidence/staged-transport-contract-20261109.md)。
当时 active change 为 C4，独立 v3 parent codec、CPU 事务状态机及同步多 stage/peer
真实传输组件已实现；A3 的 TP1 配对/聚合/fanout 和 TP2 等配对组件已通过。
V4.1 continuation、shared views、父 Engram 与显式 A/F stage runner 已实现并交付至
`efae6df`，见 [模型执行组件](changes/afd-hccl-v2-microbatch/evidence/model-stage-contract-20261109.md)。
新增 A/F CPU seam 分组 90/45 通过，真实 NPU tensor views 通过；native V2 run/factory、
真实 runner 构造及完整模型 U2 尚未接入/验收。用户已通知 A3 恢复：原生 API 456 passed，既有 U1/TP2
HCCL 组件通过，见 [恢复验证](changes/afd-v41-hccl-dspark/evidence/a3-restored-20261109/README.md)。
C2 完整模型及 C3 DSpark 硬件验收仍待 A5；C5–C7 的项目范围不变。

</details>

## 目标与边界

在 afd-plugin 的 release/v0.30.0rc2 基线上，为 DeepSeek-V4.1-Flash 增加 Attention/FFN 分离能力。
首个验证里程碑为 **ModelRunner V2 + 文本 + eager + 单 microbatch（U1）+ P2pHcclAFDConnector + FFN 侧 gate**，先关闭 DSpark。
**最终交付包含 `feat/dsv4-afd@4ddd69a3` 的 HCCL 能力迁移：U1/U2、DBO/microbatch、eager 多流、FULL_DECODE_ONLY Graph 捕获/回放/回退、Graph/U2 多流与跨层流水、多 peer 拓扑，以及 DSpark/PD 组合和观测诊断。eager/U1 通过不能作为整体交付完成。**
完整覆盖关系见 [HCCL 能力迁移矩阵](HCCL_CAPABILITY_MIGRATION_ZH.md)。这是对上一版把 Graph/U2/PD 延期的范围修正；新增能力均待实现和验收，不改变已完成任务的证据。
U1 表示一个 microbatch，不表示一张卡。部署卡数由 checkpoint、A/F 权重和 Engram/KV/HCCL 内存预算确定。

只修改当前 afd-plugin 仓库。vLLM、vLLM-Ascend 以及旧 AFD checkout 均作为只读参考。
初始规划轮只新增计划、项目技能和本地检查 harness。用户已要求开始实现并授权 A3 环境验证，当前沿用本计划推进；不重复设置额外的权限确认流程。
提交和推送以用户明确授权范围为准，代码与操作说明已交付至开发分支。2026-11-09 用户要求等待 A5 验证期间并行开发，当前推进 C4 实际传输与 V4.1 stage 状态接入；C3 原生 API 已在 A3 恢复后通过，完整模型与 DSpark 硬件验收仍按依赖执行，见 [C3 并行记录](changes/afd-v41-hccl-dspark/evidence/parallel-development-20261109.md)和 [C4 传输记录](changes/afd-hccl-v2-microbatch/evidence/staged-transport-contract-20261109.md)。

## 固定源码

| 组件 | 指定引用 | 本计划锁定提交 | 用途 |
| --- | --- | --- | --- |
| afd-plugin | wenhow/afd-plugin release/v0.30.0rc2 | 39017a137d7809bd0ab7cda4a86697e2a6b7a522 | 唯一实现基线 |
| vLLM | v0.30.0 | ced6857afa0ea7b2e3f0846a62e1394e90f15607 | 只读上游 |
| vLLM-Ascend | rfc/vllm_cann | 7e37e29a7803d6897a7113088e2b771f83205bc0 | 只读目标快照 |
| 原 v0.30 AFD 的 Ascend 参考 | 文档/patch 注释 | 8d4409d6256d8a6729140ddcc0d1889e3f96cdd6 | RFC 差异对照 |
| HCCL 供体 | feat/dsv4-afd | 4ddd69a391dfb5b3e362c7601399e538a31de702 | 完整能力的语义基准 |

完整来源与阅读索引见 [sources.lock.json](../.agents/dsv41/sources.lock.json)。
RFC 是分支名，本计划使用已核对的固定快照，不自动跟随分支最新提交；需要更新时先刷新锁和差异评估。
本地上游 HEAD 可以不同于目标快照，阅读时使用 git show <SHA>:<path>，不切换上游工作区。

## 路线选择与原理

选择 v0.30 主线：当前基线已有 V2 worker、独立 FFN runner、角色模型、控制面和模块级测试。
如果以 feat/dsv4-afd 为主线，需要同时承担旧 runner 到 V2 的迁移和 V4.1 模型适配。
保留 v0.30 V2 执行框架，将旧 HCCL 的执行与调度能力重建到 V2。迁移范围包括 connector、模型执行器、A/F runner、配置、启动脚本和诊断；仅复制 connector 文件无法完成迁移。
旧 V1 类、v0.23 私有 API 和函数组织方式无需照搬；Graph/U2、多流、普通 MTP 的适用行为必须有目标实现或明确的模型语义映射，不能因为不复制旧类而删去能力。

原始 `39017a1` 基线的 connector factory 注册 NCCL、DeepEP、CAMP2P 和 Async CAM，没有 P2pHcclAFDConnector；当前 C1/C2 已增加 eager/U1 的 typed-plan 路径，尚不具备供体的完整执行能力。
CAMP2P 虽然也使用 HCCL 通信域，但有不同的自定义算子、布局和拓扑约束，因此本计划明确新增标准 HCCL P2P 路径。
不要通过替换名称、放宽校验或默认回退到 CAMP2P 宣称完成 HCCL 支持。

V4.1 的 HC 是多路残差流。原生 decoder 在 FFN 前完成 HC 聚合和归一化，FFN 后将结果与本地 residual、post、comb 混合。
它还返回 ffn_pre，作为下一层 pre_mix，最终执行 HC collapse。AFD 的远程边界应落在 MoE 调用处：A 端保留这些局部状态和原生 layer/model forward，F 端只执行本层完整原生 MoE，再把结果送回。
这里的 continuation 指恢复本次 MoE 调用后的计算位置和状态，不等于 speculative decoding 的 draft。

Engram 基于原始 token 与历史状态增强特征；shared-KV/index 拓扑在多层间传递 attention 状态。
两者需要保留在 A 端原生 V2 生命周期中，不能把 V4.1 当作 V4 改类名，也不能丢弃原始 token IDs。

<a id="engram-priority"></a>

## 后续开发优先级（2026-11-19，区分原生 U1 与新增 Engram 适配）

本次澄清“Engram 后置”的对象是新增适配。已有原生 Engram/U1 的复用、真实权重
正确性和可独立开展的 U1 验证不等待 U2；11-12 的新增适配中优先级安排保留。
功能和正确性优先，profiler、性能和启停专项后置；完整 HCCL 最终范围保持。

| 优先级 | 后续工作 | 开始与完成边界 |
| --- | --- | --- |
| 高，当前 | 原始量化 checkpoint + 原生 Engram 的 eager/U1 输出差异、连续请求及清理；按依赖复核 Graph/U1、DSpark/U1、PD/U1；独立非 Engram 组件实际缺口 | 复用已有能力和资格入口，先固定对照条件；U1 不等待 Engram/U2 开发完成；未验收的组合不提前放行 |
| 中，新增适配 | Engram+U2/DBO 准入、父 producer 与 stage 视图、buffer/event；U1/Graph/DSpark/PD 复核发现的新增 Engram 缺口 | 按 [C7 2.1–2.7](changes/afd-hccl-full-parity/tasks.md#engram-medium)审查实施；U1 复用验证与新增代码分别记录，不能仅移除校验 |
| 低，功能后 | profiler 控制/trace、服务启动/停止/退出/重启专项及性能/timeline评测 | 已有代码和证据保留，不作为功能推进或 Engram 开始条件；功能验证中的必要清理仍执行 |
| 最终收口 | 原生与 AFD 完整模型、全部 U1/U2/Graph/DSpark/PD 组合及最终交付 | 等对应功能前置满足，逐格收口；性能和运维专项按降低后的优先级补齐 |

“其他功能具备”指上表高优先级工作已有可审阅实现、接口审查及可独立执行的验证，
剩余组合缺口明确归属于 Engram 或等待其接入的整体验收；不是要求先把依赖 Engram
的完整模型任务勾完，避免形成循环依赖。普通旧 V4 MTP 后续开发继续暂停，不作为
开始 DSV4.1 Engram 专项的前提。

已有原生 Engram、A 侧所有权、F 构造前隔离、权重及证据均保留。U1 的真实权重验证
是当前工作，设备验证由用户按最小指导执行；若需新增 Engram 修复或准入放宽，先登记
具体契约和依赖，按中优先级审查实施，不因此暂停其他可独立进行的 U1 验证。
不得通过删除 Engram 层、修改 checkpoint 或绕过配置拒绝宣称完整模型通过；
不把受控组件、CPU fake 或无 Engram 结果扩展为原 checkpoint 验收。

各 tasks.md 用“高/待中”标识混合任务：非 Engram 部分先做，整项依赖中优先级时仍不勾选。
原生 U1 复用验证与新增 Engram 适配在同一任务内分阶段记录；既有勾选和历史记录不改写。恢复开发先读本节和 active
change 的优先级说明，不机械地将首个未勾选项视为当前可执行任务。届时重新比较锁定源码
与上游能力，优先复用可用接口；不自动升级源码锁，也不把上游 GPU DBO 支持当成 NPU AFD 验收。

## 分阶段实施与准入条件

| 阶段 | OpenSpec change | 实施重点 | 完成条件 |
| --- | --- | --- | --- |
| C0 | [afd-v030-rfc-runtime](changes/afd-v030-rfc-runtime/proposal.md) | RFC 契约、模型身份、角色初始化；扩展 V2 调度/Graph 契约复核 | 旧任务保留；新增多 stage/Graph 契约审查完成 |
| C1 | [afd-hccl-v2-connector](changes/afd-hccl-v2-connector/proposal.md) | 已有 eager/U1 组件；冻结可扩展协议演进边界 | 已有组件证据保留；不能代表 HCCL 全能力 |
| C2 | [afd-v41-hccl-text-eager](changes/afd-v41-hccl-text-eager/proposal.md) | V4.1 角色模型、FP32、Engram、冷加载与文本精度 | 完整 NPU eager/U1 验收，不只构造/TP 组件 |
| C3 | [afd-v41-hccl-dspark](changes/afd-v41-hccl-dspark/proposal.md) | target 远程、draft 留 A；DSpark/U1 语义 | 真实接受/拒绝、aux 与 KV/history；组合验收在 C7 |
| C4 | [afd-hccl-v2-microbatch](changes/afd-hccl-v2-microbatch/proposal.md) | 多 stage 协议、U2/DBO、layer-major continuation、拓扑布局 | U1/U2、奇数/空 stage、多 peer、状态隔离和输出归并正确 |
| C5 | [afd-hccl-v2-graph](changes/afd-hccl-v2-graph/proposal.md) | FULL_DECODE_ONLY、U1/U2 warmup/capture/replay、miss→eager | A/F 图与控制计划匹配、稳定 buffer、Engram/KV 更新、真回放证据 |
| C6 | [afd-hccl-v2-multistream](changes/afd-hccl-v2-multistream/proposal.md) | eager 收算发、Graph compute/三流计划/hybrid DAG、F 跨层 | 开关全开/全关/依赖组合正确；timeline 证明实际重叠 |
| C7 | [afd-hccl-full-parity](changes/afd-hccl-full-parity/proposal.md) | DSpark×U2×Graph×PD、旧 MTP 回归、诊断与完整支持矩阵 | 迁移矩阵每行有可审阅实现/等价映射与对应验收，无无主延期项 |

C0/C1 的历史完成状态仅覆盖当时 eager/U1 组件范围，新增范围通过追加未完成任务追踪。
eager/U1 完整模型验收依赖 C0/C1 → C2；Graph/U1、DSpark/U1、PD/U1 分别核对其
C5、C3、C7 对应契约，不机械等待 C4/U2 整项闭合。最终包含 U2 的完整矩阵仍需
C4 → C5 → C6 → C7 的相应联合验收；C3 的硬件基线依赖 C2。
开发排期按上节执行：原生 Engram eager/U1 正确性先收口，U1 组合按各自依赖验证，
独立组件可并行推进，再补新增 Engram 适配和最终完整模型依赖链。C6 的 eager/Graph 组件分别依赖 C4/C5 对应组件，
不要求提前完成其 Engram 整体验收；未满足真实前置语义时仍不能打开生产优化开关。
每个 change 的 tasks.md 是唯一任务进度表；[session.json](../.agents/dsv41/session.json) 保留当前阶段指针，不重置已完成记录。

## 关键适配决定与待验证项

1. **注册和构造**：保留 DeepseekV41ForCausalLM 身份，以及一致的 model_arch_config、_model_info、_architecture。
   复用原生 VL wrapper 的 language_model_cls、LLM 的 model_cls 和 model 的 decoder_layer_cls 注入点。
   文本也要保留 requires_raw_input_tokens=True 和 get_model_state_cls 转发。
   F 端在构造前隔离 Engram 通信组和非 FFN 模块，不采用完整模型加载后删除模块的方式。

2. **角色初始化**：核对 worker 初始化、ensure_model_parallel_initialized 与 init_ascend_model_parallel 的实际调用。
   F 端在创建原生模型前形成局部角色视图，避免跨 A/F 创建不匹配的 Engram group；不原地污染共用配置。
   优先使用原生 loader 的显式 model_class 注入能力，保留权重格式和量化后处理；接口可用性由 C0 核实。

3. **控制面和 tensor**：一次 target forward 宣告层范围、transaction、valid/padded token 数、rank 布局及容量。
   两端必须采用相同的层序、IDs 策略和 dtype。旧 HCCL 的首层 IDs 缓存与 V2 的逐层接收必须统一。
   当前 C1/C2 每层发送 token-aligned IDs 和独立 FP32。完整迁移须形成按 transaction/stage/peer 定义的图安全策略；可保留逐层传输或经证明等价的首层缓存，但不能出现 A 每层发、F 仅首层收的混用。
   C4 将当前仅允许 stage_idx=0 的 plan、单活动层游标扩展为显式多 stage 描述；C5 将 JSON/ACK、host 校验和元数据更新置于图外。
   空输入、dummy/warmup、异常和 shutdown 都要有成对协议，不靠接收端猜测。

4. **FP32 路由**：原生 MoE 接口含 hidden_states_fp32 和 already_sequence_parallel。
   当前 AFD V4 proxy 只转发低精度 hidden_states；V4.1 必须证明 router 输入一致。
   当前实现保留独立 FP32 payload；不得在 U2/Graph/多流迁移中退化为 BF16.float()。
   若后续证明特定 DeviceOperator 可等价重建，须另有数值/路由证据；优化前仍按 BF16+FP32+IDs 的真实带宽、buffer 与流依赖预算。

5. **权重和返回值**：按完整命名空间区分 attn、ffn/mlp、Engram、HC、共享专家和 draft。
   保留原生 .ffn. 到 .mlp. 映射和量化处理；不能以 gate 子串过滤，以免误伤 indexer/compressor。
   F 端保留 routed/shared experts 的原生返回结构、缩放和归约，A 端不重复归约。

6. **DSpark**：使用 RFC 的 DeepseekV41DSparkModel；完整 draft 留在 A 端，只有 target MoE 使用 HCCL。
   核对通用 forward-context metadata hook 是否误向 F 宣告 draft，并用回归测试验证；这目前是待审查风险。
   保留原生进入目标层、Engram 注入之前的 aux capture、one-based layer IDs 和 HC mean/concat。
   dspark_block_size、draft stage 数与远程 FFN 调用次数不是同一个概念，不复用旧普通 MTP 固定循环。


7. **microbatch 模型执行**：先覆盖供体实际的 U1/U2（两个 stage），不把 U2 当作任意 U 的证明。
   采用 V2 原生输入切分/输出归并与单线程 layer-major 调度，保持每个 stage 的 residual/post/comb、ffn_pre/pre_mix、共享 attention state、Engram lookup/event 和 aux 独立。
   eager/U1 可继续使用原生 forward；异步切分处需以目标 V4.1 forward 为骨架增加 prepare/complete 或等价 continuation，不能继续声称“所有模式只换一个 MoE 类即可”。
   锁定原生 shared state 的 reset 是 no-op；真实冲突是 source/indexer 共用 tensor 存储前缀，
   Engram 也持有持久 buffer，不能简单交错调用两个完整 forward 或通过省略 reset 处理隔离。

8. **Graph 与流**：FULL_DECODE_ONLY 是供体主要图模式，包含 decode replay 和 prefill/未捕获 shape 的 eager U1/U2 回退。
   A/F 共同协商 warmup/capture/replay、stage/peer shape 和 speculative 签名；不让 F 在 A 回退时误 replay。
   控制面在图外，HCCL lowering、BF16/FP32/IDs/F2A tensor 和事件依赖在图内可重放。
   Graph warmup/capture 时要遵守通信 parent-stream 约束，保留模型 compute fork/join；“三流开关=1”不等于每个阶段通信都可随意移到侧流。
   详细流依赖、graph key 和各开关条件见迁移矩阵及 C5/C6 设计。

9. **完整拓扑和组合**：恢复供体的等规模配对、A=kF 聚合、F=kA fanout、非均匀 peer token 数、padding 和返回拆分；同 TP 坐标映射与原生 TP/EP 布局必须明确。
   供体 recipe 的 TP1 整数比例、TP2 等 A/F 是最低对照范围，当前 V4.1 TP4 容量候选另需模型验收；不据此承诺任意 TP 比例、PP/CP 或任意 U。
   PD+AFD、PD+DSpark、PD+Graph/U2、PD+DSpark/U2、PD+DSpark+Graph/U2 均列入 C7，不能再以“另立后续项目”结束本次迁移。

## 已发现阻塞与资源约束

B1：锁定 vLLM 要求 fastapi[standard]>=0.133.0,<0.137.0，RFC Ascend 要求 fastapi<0.124.0，声明范围没有交集。
2026-11-06 用户明确沿用容器内已部署版本；实际核实为 **FastAPI 0.123.10**，见 [环境记录](changes/afd-v030-rfc-runtime/evidence/container-environment-policy.json)。
本次采用该容器环境继续开发和运行验收，不再以声明范围冲突阻塞进度；不修改上游依赖元数据或源码锁定提交。
pip check 的声明冲突与实际运行结果分别记录；容器路径的可用性通过真实 API、组件及后续端到端验证判断。

先估算 A/F 权重、Engram 大表、KV、算子 workspace、HCCL buffer 和预填充峰值，再决定部署规模。
不预设完整模型一定能用 1A1F，也不将增加卡数带来的容量视为 AFD 性能收益。

## 验证与回归

| 层次 | 证明内容 | 本轮状态 |
| --- | --- | --- |
| 计划检查 | 文件、链接、source object、任务、技能格式、上游只读状态 | 用本地 harness 验证 |
| CPU/契约 | 消息序列、stage/peer 布局、回退与生命周期 | 保留 C0/C1/C2 原有证据，新增能力逐项补证 |
| NPU 组件 | HCCL、真实 stage/peer、Graph/多流依赖 | 已有组件不覆盖新增 Graph/U2/多流；新增项待验收 |
| NPU 端到端 | 原生与 AFD 同 checkpoint、同请求/解码参数的精度与稳定性 | 待硬件验证 |
| 性能 | 总吞吐、TTFT、TPOT、实际卡数、每卡效率和部署规模 | 功能合格后测量 |

现有测试遵循 AGENTS.md：断言可观察协议/输出，不断言内部 helper 次数，不用源代码子串证明功能。
测试加入既有模块的 validation_paths；硬件行为进入 NPU/E2E 用例，CPU fake 不能替代设备证明。
保留 v0.30 已有 V4/V2、CAMP2P、Async CAM、非 AFD 路径，首版新增支持范围不等于删除旧功能。

Graph/U2/DBO、多流与 HCCL 既有 PD 组合已纳入本次必交付范围。Window/CAM 等其他 connector 优化、视觉输入，以及供体本身未支持的任意 U/PP/CP 等扩展不自动并入。
供体源码有分支、开关或测试不代表在新依赖上已通过验收。完整迁移须逐条覆盖 [能力矩阵](HCCL_CAPABILITY_MIGRATION_ZH.md)，CPU 正确性、真实 NPU 功能和性能证据分开记录。
性能至少比较 native、AFD eager/U1、AFD eager/U2（overlap 开/关）、AFD Graph/U2（全开/全关），再逐开关归因；使用相同 checkpoint/拓扑/请求参数。

## 开发工作流入口

阅读 [工作流说明](../docs/npu/DSV41_V030_HCCL_WORKFLOW_ZH.md)，保留 session 当前阶段，
按[后续排期](#engram-priority)选择高优先级未完成任务，再推进中优先级 Engram 与最终组合验收。
采用项目技能约束决策、OpenSpec 描述行为和任务、harness 检查可恢复状态与上游只读边界。
这是本地轻量组合，不声称已安装完整 OpenSpec/Superpowers 插件或已通过官方 CLI 验证。

方法参考：[OpenSpec workflow](https://github.com/Fission-AI/OpenSpec/blob/main/docs/workflows.md)
和 [Superpowers writing-plans](https://github.com/obra/superpowers/blob/main/skills/writing-plans/SKILL.md)。
采用其“先形成可审阅设计、按可验证小任务推进”的思想；本项目实际范围以用户约束和此计划为准。
