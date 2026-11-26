# HCCL 全能力组合、PD 与交付验收：任务

只有实现、审查和验证证据齐全后勾选。任务状态只维护在本文件，既有勾选和历史证据保留。

## 当前排期（2026-11-19）

遵循[总计划优先级](../../ROADMAP_ZH.md#engram-priority)：当前先复用已有原生 Engram
闭合真实权重 eager/U1 输出正确性，Graph/U1、DSpark/U1 和 PD/U1 按各自依赖推进。
锁定 vLLM 明确拒绝 Engram+DBO/microbatch，U2 真实模型属于新增适配；不把该限制
扩大成所有 Engram 场景都不支持。新增 Engram 适配仍按中优先级审查实施。
profiler、性能及服务启停/重启专项后置；必要资源清理、取消/随后请求和错误传播保留。
已有实现、接口/CPU 和 NPU 组件证据按原范围保留；源码准入与设备验收分别记录。

2026-11-19 用户继续授权完整功能开发，A5 手工同步验证；PD 设备安排稍后确定，
先提供可配置脚本。本轮已补齐 TP4 组合入口、父 Engram 图外 producer/稳定输入、
FFN 启动前角色校验及 PD 原始历史声明，详见
[源码审查与验证](evidence/full-checkpoint-integration-20261119/README.md)。
独立非 Engram 组件已有可审阅实现和验证，剩余完整权重缺口明确归属本轮状态接入，
因此进入第 2 节新增适配；不等待依赖它的完整矩阵先验收。
1,433 项 CPU 回归和 55 个真实首次 EngineArgs 配置通过，NPU 均未初始化。
真实权重 eager/U1 首 token 差异仍待 A5 同 TP4 原生对照，其他组合也未获得完整模型验收。

2026-11-19 首批完整权重矩阵回传：原生 TP4 与 AFD A4/F4 都停在预检，
25 份锁源、checkpoint 配置/索引与结构检查通过；当前 Python 缺少
`vllm-afd-plugin` 安装元数据及 `afd` 入口。未载权或执行请求，不能闭合 2.2/1.8。
已补充同 Python、无依赖解析/下载、跳过算子编译的 editable 注册及只读核验步骤；
详见[归档审阅](evidence/a5-full-checkpoint-plugin-registration-20261119/README.md)。

随后原生 `full-eager-u1-1g5_98ms` 的插件/25 份源码/checkpoint 预检通过，但
设备资源检查发现卡 0–7 的现存 vLLM 进程，未启动 probe/服务或发送请求。
AFD 因该参考的 `request_passed=false`、`clean_shutdown=false` 被拒绝，未进入模型执行。
已增强参考失败报错并补充资源与参考成功条件，详见
[本轮审阅](evidence/a5-native-resource-gate-20261119/README.md)，2.2/1.8 保持 pending。

原生 `full-eager-u1-yj67qgqv` 随后通过预检和四项 A5 packed Attention probe，
TP4 真实量化权重及 CPU Engram 载入完成，17:08:02 HTTP 应用启动完成。
启动器未通过就绪检查或发送生成请求，17:20:13 用户手动中断；缺少 summary 和最终
资源快照，不能认定原生参考或清理通过。已补齐直连服务、health.json 及等待进度；
代理干扰在主机 HTTP 回归中可复现，但不是本包已证实根因。见
[本轮证据](evidence/a5-native-readiness-20261119/README.md)，2.2/1.8 仍 pending。

按用户要求，随后将就绪检查切换为直接调用 `curl --disable --noproxy '*'`，保留
health.json 和等待进度；HTTP 200 与 curl exit 0 同时满足才通过，生成/取消/PD 继续
使用无代理 opener。缺少 curl 会立即报错，连接/HTTP/超时错误保留在报告中。
见[主机验证](evidence/a5-curl-readiness-cpu-20261119/README.md)，不改变 A5 任务状态。

2026-11-20 原生 `full-eager-u1-yqxpk_w6` 的预检/packed probe/curl HTTP 200 均通过，
首波双请求进入模型后因 wheel 限制 `N1 must be 64 ... got 16` 返回 HTTP 500。
原生对照未接入 AFD 已有的 per-Attention head adapter；已补齐显式控制 worker，
仅在原生载权后替换本实例的 backend，原生模型/ModelRunner/MoE/Engram 保留。
计划标记相同 adapter，参考准入核对标记；新增父 worker 锁源检查。
19 项真实依赖 CPU 适配/选择契约通过，NPU 未初始化；完整请求和正常退出仍待 A5。
本包清理后八卡 idle、无残留，但触发强制清理，2.2/1.8 不勾选。
见[归档、架构审查与验证](evidence/a5-native-tp4-heads-20261120/README.md)。

2026-11-20 随后回传 `full-eager-u1-pqdgncjt`：AFD 两个 HTTP 200 均返回 32 token，
但分别从第 1、第 13 个 token 与原生分歧；A/F 正常退出、八卡回收，无强制清理/残留。
参考原生 `full-eager-u1-7z0p69ts` 自身重复相同请求也在这两个位置分歧，不能直接归因
AFD/Engram。本轮增加独立 `--repeatability`：单活动请求串行、预热、取消前/后各三轮，
严格聚合输入/输出差异，默认复用 token-only 采样路径；可选 top-5 分数单独建参考。
参考套件/参数/稳定性在 AFD 启动前核对，普通矩阵失败门保持。
主机协议/CLI 检查通过，设备复测待用户；2.2/1.8 及完整矩阵勾选不变。见
[本轮输入、工具与验证](evidence/a5-eager-u1-repeatability-20261120/README.md)。

2026-11-20 新回传三份组合及 `full-eager-u2-a3guux9z`：DSpark eager/U1 N=1 与
Graph/U2 N=2 在原生 draft 的 packed Attention 抛出 16-head 拒绝；已在 A 侧
载入后为实际 draft 实例选择现有 head adapter，保留原生 draft/权重/KV/采样。
新增锁源检查覆盖原生 `dspark.py`。CPU load seam 修复前 2F/4P，修复后四个完整
模块及签名表工作区回归 201P、交付范围冻结复核 190P，均 0F/E/S、NPU 未初始化；
主机诊断检查通过。
普通 eager/U2 与 Graph/U1 均生成两个 32-token 响应但输出对照失败，前者实际执行
32 个 U2 parent；与旧 AFD eager/U1 也分别从第 5/14 个 token 分歧。
原生自身重复性尚未闭合，未改动未定位的数值路径；补 Graph/U1 串行重复诊断，
沿用稳定原生参考继续定位。用户已授权本轮提交并推送；
1.3、1.8、2.2–2.5 仍 pending，未执行本轮 A5 复测。
见[修复、架构审查及四份归档分析](evidence/a5-dspark-heads-eager-u2-20261120/README.md)。

随后 A5 原生串行归档 `full-eager-u1-pz23il_b` 与 top-5 归档 `full-eager-u1-5zwijbpu`
均完成 14 个请求并正常回收，数列 prompt 在取消前已分歧。top-5 归档的首步候选分数
在生成历史为空时已经不同。本轮增加原生 TP4 可选首两步 trace，采集各 rank
的 logits processor 输入摘要和原始 logits，供确定下一处排查边界。
A5 `full-eager-u1-zywlcax3`（`beea901`）预检、算子探针及载权通过，首条预热请求
因 trace 在包装层访问 `logits_processor` 而返回 HTTP 500，未产生 trace。
按锁定 `vl_model.py` 的 `compute_logits` 委托，改用
`language_model.logits_processor`。锁定委托方法与 CPU 张量回归修复前复现 2 项
属性错误，修复后通过。实机 trace 待复测，2.2/1.8 和其余组合的验收状态保留。

| 执行顺序 | 对应原任务 | 阶段边界与验收前置 |
| --- | --- | --- |
| 先完成真实权重 eager/U1 对照 | 2.2、1.8，关联 C2 | 原始量化 checkpoint、A 侧原生 CPU Engram、F 隔离；历史首 token 差异未闭合，单次请求成功不等于等价 |
| 再按依赖验证 U1 组合 | 2.4–2.6 的 U1 部分、1.3、1.5、1.8–1.10 | 核对 Graph/U1、DSpark/U1、PD/U1 的原生接口及准入；分别验证 replay/fallback、接受拒绝与 history/handoff，不等待 U2 先完成 |
| 收口独立组件实际缺口 | 1.3、1.5、1.6、1.10 | 保留 dummy/BF16/无 Engram 组件结果，不替代真实量化模型；不重复已完成组件来累计模型验收 |
| 审查实施新增 Engram 能力 | 2.1–2.6 的新增部分 | 高优先级独立工作具备后处理 U2 准入、父 producer/stage 视图及图/历史缺口；原生拒绝之外的插件扩展需审查和实际对照 |
| 完整矩阵与交付 | 2.7、1.8–1.12 | 已通过 U1 精确场景可单列；U2 与 H01–H15 最终范围保留，未完成格子仍 pending，性能/启停专项最后安排 |

当前真实权重入口为 [完整 checkpoint 指导](../../../docs/npu/DSV41_A5_FULL_CHECKPOINT_VALIDATION_ZH.md)。
A5 设备由用户手工执行；等待报告时继续独立实现、审查和 CPU 工作。本轮已实施运行时代码。
以下 11-16/11-17 安排保留为背景，具体执行顺序以上表为准。

用户11-16随后调整验证分工：**DSpark相关设备功能验证改在A5，A3推进非DSpark功能**。
用户11-17进一步明确：**先完成暂不含Engram的组合功能开发，再集中到A5整体验证，场景尽量精简**。
A3 metadata失败保留为尚未解释的环境/调用差异，不再以该问题阻塞本轮功能收口或追加
A3设备定位。先核对实际正常入口和组合代码缺口，完成必要修改、审查和CPU回归，随后
交付可直接更新执行的A5统一入口/指导；A5结果未回传前不报告设备或完整检查点验收通过。
既有A3 DSpark证据保留，不追加DSpark推理场景；后续A5提案/接受拒绝/PD与U2/Graph组合集中整理最少场景。当前A3优先普通文本U2的真实配置入口、普通PD及U2/Graph/多流正常接入链。
正常Worker/加载/首C2 KV和AFD首prefill/sample已实测完成，首hidden与独立原生相同；
原生两轮及退役通过，AFD第二轮C2 metadata仍失败，补query长度004未解决；保留证据，
按最新要求暂缓追加A3设备定位。
A5同TP4单层MoE及DSpark两场景已提交推送至`feat/dsv41-flash-hccl-v030@6b4d6c5`，设备结果待用户回传；以上增量不提前勾选依赖完整组合验收的任务。

第 2 节区分**当前原生 U1 复用验证**与**中优先级新增适配**。前者不等待 Engram/U2，
后者在高优先级独立实现/审查/验证具备后开始；不要求先勾完依赖它的 1.8–1.12，
避免循环依赖。U1 数值基线是 U2 完整模型对照的前置，不阻止先做独立设计和 CPU 检查。
普通旧 V4 MTP 后续开发继续暂停。

“高/待中”任务先做非 Engram 部分，完整勾选仍等第 2 节对应依赖和原有验收；
原生 Engram 行为、A 所有权/F 隔离及已完成适配保留，不通过删层、改 checkpoint或绕过配置拒绝宣称完成。各 change 原有任务仍负责本阶段整体验收，本节只集中管理Engram 新增适配的执行进度，不在其他 tasks.md 重复登记同一专项的完成状态。

## 1. 高优先级实施与最终组合验收（原任务编号保留）

- [ ] 1.3 【源码已接入，联合验收待 A5】完成 DSpark×U2×Graph×多流的 target-only、aux merge、真实接受/拒绝、KV/采样历史和异常生命周期；本轮补齐完整权重组合准入与父状态接入，Engram 联合设备验收见 2.3–2.5，整项保持 pending。
  固定 N=5 已接入完整权重 A/F 启动与配置检查。EP Graph 用现有协议协商 Q=6，
  TP1/DP4/EP4 的原生 eager 与四档 AFD 真机结果待
  [A5 验证](../../../docs/npu/DSV41_A5_EP_VALIDATION_ZH.md#n5-原生对照与-afd-验证)。
  2026-11-26 原生对照在载权后整表比较 embedding 触发 NPU OOM，启动未就绪。
  原生 EP DSpark 对照使用分块 CPU 精确比较兼容，A5 复跑待验。
  后续归档在四卡通过载权后，原生对照的 KV 预算为 −1.73 至 −1.74 GiB
  （显存比例 0.90）。原生预算可单独配置，0.95 复跑待验。
- [ ] 1.5 【源码已接入，联合验收待 A5】完成目标 v0.30 PD 请求接入、KV/index、layer/component 和退出适配，不启用过时补丁或改上游；本轮补齐 Engram 原始 history 声明与校验，设备 handoff 验收见 2.6。
- [ ] 1.6 迁移 recipe/preflight/capture-size/拓扑校验和所有 overlap/诊断值记录，保留显式 0 覆盖。
- [ ] 1.7 【低：功能后】对齐 A/F profiler 控制与 F 管理 RPC，验证启停/step/trace 不阻塞业务（H14）；已有实现保留，不阻塞功能开发验证。
- [ ] 1.10 【高/待中；启停后置】优先并发/取消/随后请求及影响功能的错误传播；STOP专项/故障退出/新进程重启降低优先级，测试必要清理仍执行。Engram retire/history 联合验证待 2.2/2.5/2.6，旧 MTP 恢复后单列结果。
- [ ] 1.8 【当前 U1；完整矩阵待中】在 NPU 验证 standalone eager/U1/U2、Graph/U1/U2，含 DSpark 开/关、代表拓扑和长运行（H15）；先完成原生 Engram eager/U1 数值对照，按依赖验证 U1 组合并单列证据，完整 U2 矩阵待 2.1–2.5/2.7，整项不提前勾选。
- [ ] 1.9 【按依赖验证 U1；完整矩阵待中】在 NPU 验证五种 PD 组合，同 checkpoint/tokenizer，覆盖 prefix、KV/index 和 Engram history；先核对 PD/U1 的原生历史接入并在条件满足时验证，不等待 U2；新增 handoff 缺口及完整 U2 组合待 2.6–2.7。
- [ ] 1.11 【低：功能后；待中】按同条件完整矩阵采集输出/接受率/timeline/冷启动/HBM 与卡数归一化性能，等 Engram 及其他前置功能验收后执行，不用当前组件数字填充。
- [ ] 1.12 【收口：待中】审计 H01–H15 实现/等价映射/证据及运行说明；Engram 依赖与未验收格子保持 pending，满足完整条件才报告完成，旧 MTP 暂停状态继续单列。
- [x] 1.2 审查目标 PD 与 shared cache/index/Engram handoff 及旧 patch 差异，明确需补齐接口（H13）。

<a id="engram-medium"></a>

## 2. Engram 工作：原生 U1 复用验证与中优先级新增适配

2026-11-19 澄清原编号 2.1–2.7 的执行边界：已有原生 Engram/U1 验证是当前目标，
新增 Engram 代码和 U2 准入仍按中优先级审查实施。已有实现和证据保留，只补复核后的
实际缺口。main 文档的 microbatch 限制不代表所有 U1 组合都不支持；未被该条禁止也
不代表已在 NPU 验收。锁定版本变更另行评估，不自动跟随 main；补丁注明成本、作用域
与移除条件。各任务中的 U1 及 U2 证据分阶段记录，全部原条件满足才勾选整项。

- [x] 2.1 【新增 U2 准入：源码/架构/CPU 完成】高优先级独立工作具备后，复核锁定 vLLM/Ascend 对 Engram+DBO/microbatch 的明确拒绝、配置重建及 worker 校验链，设计插件内最小兼容接入并完成架构审查；保留非 AFD 拒绝语义与上游只读边界，不以单纯删除校验放行。见[本轮审查](evidence/full-checkpoint-integration-20261119/README.md)：1,433 CPU 通过，55 个真实首次 EngineArgs 及重复 native post-init 均通过、NPU 未初始化；其余完整模型验收仍 pending。
- [ ] 2.2 【当前：原生 U1 复用验证；新增修复按中优先级】先保留原生 Engram 完成真实权重 eager/U1 的可比基线、输出差异和连续请求验证；复核 A/F 初始化、组/权重归属、请求结束/复用、chunk/dummy/空步、异常 retire 与资源回收，新增缺口登记后按中优先级审查修复；以真实历史和设备对照收口 C2 1.8/1.12，保留 A 所有权及 F 构造前隔离。
- [ ] 2.3 【父准备/视图已复核，联合 NPU 待验】补齐 eager/U2 父级一次 producer、lookup/mask 子视图、lookback 请求坐标、buffer lifetime 和双 consumer event 契约；含切分跨请求/跨 chunk、odd/empty、多轮及与 HC/pre_mix/shared-KV 的联合 NPU 对照，闭合 C4 1.7/1.10 的 Engram 依赖。
- [ ] 2.4 【图外 TP4 producer/稳定输入已接入，NPU 待验】U1 基线后验证原生 Engram 图路径；已补实际 producer、稳定输入及共享事件消费边界，保留 native retire；仍需真实 capture/replay/miss→eager、多流、请求/bucket 切换对照，闭合 C5 1.8 与 C6 的 Engram 依赖。
- [ ] 2.5 【原生 history 已复用，NPU 待验】验证 U1 的接受/拒绝/bonus/回退后 Engram history 与下一轮 lookback，以及 U2/Graph 组合；覆盖同 worker 连续轮次、取消/结束后的槽复用，真实 target/proposal/采样和设备结果对应，闭合 C3 1.8/2.1 与 C7 1.3 的 Engram 依赖。
- [ ] 2.6 【原始历史声明/校验已接入，NPU 待验】验证 U1 Prefill→Decode 的原生 Engram 历史、请求/token 坐标和 prefix/缓存接入，以及 U2 等 PD 五组合，覆盖取消/随后请求；F 不拥有该历史，不以 KV 到达替代 Engram 正确性，闭合 C7 1.5/1.9 的 Engram 依赖。PD 设备待定，使用可配置启动脚本。
- [ ] 2.7 【中：后置】在前六项实际缺口闭合后，记录 Engram 对各模式的 NPU 对照/生命周期证据、资源成本与兼容补丁保留/移除结论，更新精确支持边界，作为原 1.8–1.12 完整矩阵验收的输入；不提前报告性能或全量交付通过。

## 3. 分支映射与普通旧 V4 MTP（保持原暂停策略）

- [ ] 1.1 先登记 DSV4.1 DSpark/PD 分支去向；旧 MTP 的既有映射与适用能力保留，新增旧 MTP 开发继续暂停，不作为第 2 节开始前提（H12）。
- [ ] 1.4 【暂停：旧 MTP】保留适用旧模型完整 MTP eager/target graph/full-draft graph 的已有改动，后续开发按原用户优先级暂停；V4.1 不强行开启 MTP，恢复后单列验收。

## 4. 历史实施与证据记录（保留原范围）

2026-11-16 按用户功能优先要求恢复真实 DSpark/PD 验证。live-004 保留旧 draft 分类失败；
[live-005 的控制帧与标量写入问题](evidence/a3-ready-control-device-20261116/README.md)已修复，完整连接器模块58项CPU通过，四种标量/空张量写入通过。
[live-006](evidence/a3-dspark-native-live-prefix-20261115/live-006/README.md)确认前述错误消失，进入原生目标元数据；受控 ratio4 与 V4.1 分类不一致，
后继按真实 C2 约束修正 ratio2/heads64/topk512/FP32 state。
[原生十算子包](evidence/a3-ready-qli-metadata-20261116/README.md)构建通过且26,600只读输入不变，其 AICPU 输出缺口已按原生 ENABLE_AICPU 选项补齐（build-004），27,666只读输入不变。
live-007 前进到 F native context 后发现线程 inference mode 缺口；
[最小修复](evidence/a3-ready-inference-mode-20261116/README.md)同测试两例RED、完整worker83项GREEN，Ready-007仅叠相同执行函数守卫。live-008 中 P 已完成真实提案和来源发布，D 读取前的原生 Scheduler 因处理器需要 tokenizer 而失败；原生 KV consumer 强制开启 embedding 输入，纯文本声明不免除该初始化。已修正验证配置，不改上游消费端策略。live-009 已观察到完整 Ready 前沿、D读取/DONE、整池核对通过，但 P 功能报告写完后发生原生退出异常，整轮仍失败。decode-001 已到 D 首轮采样与提案，实际草稿114与验证包假定的7不符；后继在执行前声明合成 Markov 参数并由原生 loader加载，保留提案/拒绝采样及完整核对，后继实际结果另记。
该后继实际运行 decode-002/003：接受路径两轮、拒绝后继续第三轮均完成，
真实草稿7、采样输出、计算位置5/7/9、累计历史与缓存写入保护符合预声明；
两个场景四角色功能报告均成功，输出上限自然结束。但 P 报告后仍发生原生退出异常，整体仍失败；仅记录受控合成权重组件的功能观察，不升为完整模型或整体运行通过。
[eager-lifecycle-001](evidence/a3-dspark-native-live-prefix-20261115/eager-lifecycle-001/README.md)已实际运行取消/连续请求/KV槽复用与 eager U2多流组合，源444/生产145/安装身份191及OPP前后不变、8卡前后空闲。D/F2在原生V2配置准入时被DSpark及多模态DBO限制拒绝，尚未构造owner、读取Ready或执行target；原始supervisor因F2退出1终止三个peer。
本轮不是历史P报告后abort，亦无生命周期功能通过。另发现外层引用仅非运行元数据才有的observer字段会误拒，已在后继驱动修正，原始包及结果保留。AFD插件自有DSpark/U2的最小配置校验接入已实现并经源码复制合同/独立审查；相邻本地回归270P/3S，新native模块因缺Torch跳过，不计通过。已启动的A3配置CPU旧代码对照red-002实际4F，精确复现A/F×eager/Graph四项原生spec/MM拒绝，446源码/191安装身份前后不变。
red-001仅上传超时，未启动pytest。按用户最新分工，native GREEN和DSpark设备功能转A5，不再启动A3 DSpark推理；当前转普通text/noSpec U2配置前门及非DSpark功能。
服务专项不作为功能推进前提。
随后完成[普通text/noSpec U2原生配置方法](evidence/a3-text-u2-config-admission-20261116/README.md)
最小扩展及源码审查：同最终补丁/测试/共享fake，旧runtime四项正向4F精确复现MM拒绝；
仅安装runtime后，五个普通text selector的33项边界加唯一签名检查共34P、0F/E/S。
两轮446源码和191安装身份前后不变，没有选择DSpark测试或运行设备推理。
本地相邻完整模块270P/3S。
随后真实EngineArgs首次声明U2的host检查，旧runtime A/F×eager/Graph四项MM拒绝，
只更换runtime后四项首构造通过，MM身份与U2声明保留、NPU前后未初始化；447源码/191安装身份不变。
首轮red-001两项Graph为MM拒绝、两项eager为验证脚本capture-size类型错误，原始混合失败保留；
修正后的red-002与green-001才是同最终脚本的完整对照，不将其计作pytest/JUnit。
无PD四项之后，首声明attested PD consumer配置的A/F×eager/Graph四项亦通过，
保留原tokenizer配置/skipFalse、equalTP1/DP1/PP1及原生MM embeds策略，NPU前后未初始化；
447源码与191安装身份不变，对比无PD GREEN仅验证脚本变化，所有生产字节相同。
受控小配置无训练参数加载，也未构造KV runtime/传输引擎；完整Worker/模型设备执行仍待另验，不据此勾选整项。

同日继续完成[普通PD Graph取消后新C及请求槽复用](evidence/a3-native-pd-graph-replace-20261116/README.md)：
replace-001沿用原会话，在A取消/B自然结束与真实空步退役后接入第三份真实KV；
C实际复用B的请求槽6，输入35/位置4 eager后，以输入7/位置5复用B-only原U1 key回放，
原生采样历史[7]→[7,7]后自然结束并空步退役。原owner/Scheduler/Graph/cache/bank/receiver/TE地址保持。
实际P第三DONE、五轮BF16/独立FP32/IDs/采样、两池各36864B累计oracle及36parent/每端365canonical帧复核通过。
P/D/F和外层均退出0，84.698s，无timeout/kill，八卡前后空闲；418源码/165安装身份及toolchain/Mooncake前后不变。
145生产字节继承cancel002，未叠当前其他增量；不是完整Worker/SparseFA/训练检查点验收，
也未把上述EngineArgs前门接到设备owner；物理块复用保证、同ID及transfer/target并发不在范围。
1.5/1.10及完整U2/Graph任务仍保留原始联合验收条件，不提前勾选。
构建、CPU、静态审查均不替代设备链路通过。profiler/性能/服务启停专项未启动，
未新增A5手工场景；任务勾选保持54/109。

随后继续完成[真实EngineArgs首声明进入普通PD Graph设备链](evidence/a3-native-pd-graph-engine-args-20261116/README.md)：
D/F同一工厂首声明H16/HC1/两层、U2和FULL_DECODE_ONLY，保留原生MM身份、tokenizer、
attested consumer及F正常隔离；原生尺寸缓存getter一致。两host进程通过且NPU未初始化，
继而同423源/146生产快照的entry-001在A3真实通过，77.7225s，P/D/F及外层0，
无timeout/kill，前后八卡idle，165安装身份及toolchain/Mooncake未变。
首声明prefill threshold2使首轮走eager/U2（两stage各一行），之后U2回放、取消A、
B继续至自然结束，再接新C真实KV并复用B槽6，eager后命中既有U1图回放/退役。
五轮BF16/独立FP32/IDs/history、两池各36864B累计全byte、35parent和每端355canonical帧
独立核验通过；14项真实语义篡改拒验，主流程新进程重跑审计报告逐字节一致。
生产只在145父模块上更新已审runtime并加ubatching补丁，其余144份字节相同；
chunked prefill从首构造合法声明True，不声称同旧后改配置完全相同。
受控owner仍未执行完整Worker构造/训练参数加载或SparseFA，正常入口下一步见
[独立设计](evidence/text-u2-normal-worker-design-20261116.md)；原联合任务不提前勾选，
项目仍54/109。DSpark设备仍留A5，profiler/性能/服务专项后置。

同日按用户要求直接交付 A5 验证入口：同 TP4 native/AFD F 单层 MoE 对照
`2e046e347de42fdcbf0c015f36e3f61b2091122f`、[DSpark 两场景](../../../docs/npu/DSV41_A5_DSPARK_FUNCTIONAL_ZH.md)
`6b4d6c58c534a788a9691385687c56b6e598329f` 已推送至 `feat/dsv41-flash-hccl-v030`，
独立 `ls-remote` 返回同一最终 SHA。MoE 从独立冻结包运行；DSpark 入口直接读取仓库自带
纯源码包，不依赖未提交生产代码或历史临时目录。后者 archive SHA
`311cafc72c81dd06f09a67fe903527db3820089e23a39192f971e022cc5f7c3b`，180 源、
146 生产模块逐字节保留父包、194 个锁定原生 Python 身份；A5 差异限定为四个验证
helper 后继和一个新 helper，包含首声明64头、原生False custom-op策略及544字节packed SWA。
[独立审计](evidence/a5-dspark-packed-codec-review-20261116/README.md)通过1319项标量检查、
5项实际源码变异拒验及归档/源码身份复核；Windows/Linux两 metadata 场景及精确提交树
`111cdba56afebcb7e1792b32022f6cbf4f76ec26` 的干净导出入口实际通过，未导入设备运行时。
这些是源码交付和CPU证据；同TP4 MoE数值、A5原生packed writer/P→D、64头及两个
DSpark场景的设备结果仍pending，不提前勾选任务。A3继续独立正常Worker候选的门禁与构造链，
不因A5等待停止非DSpark开发；旧资格范围不变，U2/Graph/PD不由新候选一次放开。

随后新增[正常Worker受控候选](evidence/text-normal-worker-source-review-20261116.md)：
独立 `AFD_DSV41_DUMMY_QUALIFICATION` 默认关闭、精确0/1、与旧A5资格互斥，
仅A1/F1、BF16 dummy、H512/HC4/64头、两层SWA `[0,0]`、eager/U1，显式禁止
U2/Graph/PD/speculation/Engram。旧A5资格及其他既有校验helper AST保持。
[A3真实CPU门禁](evidence/a3-text-normal-worker-20261116/README.md)同最终测试/fake/probe
对照：旧feature下256项中87F/169P，新feature下256P/0F/E/S；A/F两真实EngineArgs
首声明亦从旧guard RuntimeError变为成功，原生U1的num_ubatches为0，保留MM身份及
chunked prefill首声明True。两轮仅生产feature_validation不同，其他145生产字节同父，
449源码/191安装身份前后不变；三个子进程NPU均前后未初始化。
随后同一工厂真实进入 [WorkerWrapper→A/F Worker.init_device](evidence/a3-text-normal-worker-20261116/prepared-device-001/README.md)：
device001两角色分别在npu:0/npu:1构造正常A/F runner，338.448s，均退出0、
无timeout/kill/survivor，正常wrapper shutdown与native groups清理成功，八卡前后idle。
451源码/191安装身份/1068现成active package产物/1404 CANN及ATB身份前后不变；
实际映射既有CANN9.2扩展，无旧parent package或CANN9.0.1混入。
本轮不加载模型、不分配KV、不执行推理或A/F配对HCCL；独立结果审计另存同目录。
随后 [load-001](evidence/a3-text-normal-worker-20261116/prepared-load-001/README.md)
真实并行调用正常A/F Worker.load_model；F经原生空KV planner和wrapper初始化进入HELLO，
A正常shutdown发送STOP，F公开poll等到STOP及owner退出。整轮192.015613s，两端退出0，
无timeout/kill/survivor/cleanup failure，八卡前后idle。A正常Attention/HC模型35参数、
27271344B；F原生完整MoE模型14参数、28360768B，设备及有限性读回通过。
每端四条canonical控制消息完全对应，零FORWARD；453源码/146生产同父、191安装/
1068 active/1404 CANN及ATB身份前后不变。DummyModelLoader直接初始化，不读取检查点，
所记load_weights方法仅为定义来源；不声称跨角色权重等价。下一步接A正常KV和首轮执行，
本轮不分配A KV、forward/sample、Engram或完整模型精度，任务勾选仍54/109。

继而[两层纯SWA的KV-001](evidence/a3-text-normal-worker-20261116/prepared-kv-001/README.md)
实测失败：191.424522s，A退出1/F退出0、八卡空闲。原生V41共享池把单类型SWA规格当作
复合full-context规格，未进入A KV分配；原始失败及独立审计保留，不计功能通过。
必要性审查后，纯SWA专用分派/allocator候选归档并从活动路径撤回，未安装到生产。
下一步显式转入实际目标的三层 `[0,0,2]` 首C2功能：沿原生long-KV/index/ring/SWA
物理元组、zeroer元数据和正常Worker初始化，不修改原生planner/allocator/binder。
新增受控资格只允许精确三层组合、KV/index源 `[2]`、candidate源20，保持默认关闭和其余
eager/U1/BF16/dummy边界；旧两层load-only及旧A5资格保留。仅该资格helper改变，其他生产
函数AST同父；本机274项中50P/224S，缺依赖跳过不计原生通过。
随后[同源C2候选](evidence/a3-text-normal-worker-20261116/prepared-c2-001/README.md)
在A3真实CPU门禁274P/0F/E/S，两次真实EngineArgs首构造退出0、三个子进程NPU前后均False；
同archive的设备轮188.186065s，A/F均退出0、无timeout/kill/survivor/cleanup failure、八卡空闲。
正常Worker完成原生long-KV/index/ring/SWA规划、分配及绑定：8MiB单descriptor、64物理block、
五逻辑组、六carriers，实际int8 K/FP16 scale tuple和FP32 ring共享原生物理tile；
A58参数40,055,816B、F21参数42,541,152B，KV初始化前后参数逐项保持。
455源码/191安装/1068 active/1404 CANN及ATB身份前后不变，撤回的纯SWA补丁未安装。
独立审计重读原始host/device归档及完整8MiB字节，根代理新进程复算报告SHA同为
`ca5cda221d373f29608454f9494acbee4847f2749c943d4e30c7d7aca5858bf8`。
此轮只初始化zeroer元数据，未执行fresh-block清零、Scheduler、forward、sample或数值等价；
下一步接原生调度、prefill/decode及空步retire，不提前开放U2/Graph/PD或完整模型，任务仍54/109。
首轮执行候选已保留为仅静态prepared-execute-001，455父源及146生产全等，新增probe/driver
后的457源、444 Python编译及10项stdlib CLI检查通过，补齐原生MM/Worker的6个锁源pin。
[A3只读依赖检查](evidence/a3-normal-c2-qli-readiness-20261117/README.md)实际确认旧完整十算子
包缺少C2 forward所需两个精确 `aclnnVllmAscendQuantLightningIndexerV2` 动态导出；普通CANN
同类API不替代，旧包不启动本场景。后继在独立目录补建原生7e完整十一算子vendor，显式冻结
新环境描述后才运行真实Scheduler；不修改旧C2证据或切换算法，不计已执行forward。
随后原生十一算子构建/独立安装实际均退出0、无存活子进程，37540原输入不变；1081文件包
保留原20个API并新增精确两个QLI导出，实际kernel/JSON/OPP与libopapi四项原件另存并重算SHA。
显式新环境70417471冻结1082项package/record与1404项原CANN/ATB，791个旧non-vendor文件
和两AICPU产物保持；状态仅built-not-loaded。[执行002](evidence/a3-text-normal-worker-20261116/prepared-execute-002/README.md)
冻结458源/146生产全父同、444 Python/197安装源码pin。实际execute-001整轮失败：
224.032s，A退出1/F退出-9，管理端清理owned F PID898852；非timeout，最终无survivor、八卡空闲。
已完成原生三层prefill/三次远程MoE/首采样721及Scheduler回填，随后decode在
SparseFlashMlaMetadata自定义AICPU报param invalid，异步在后续HC/RmsNorm处抛出；
未完成第二采样、自然停止或请求退役。原生AICPU字节同旧包，不据host异步栈认定HC根因。
下一轮只加同步定位与metadata实际输入观测；保留失败原件，不计正常执行或数值验收通过。
随后[同步定位003](evidence/a3-text-normal-worker-20261116/prepared-execute-003/README.md)
实际221.910555s，A退出-6/F退出-9（监督清理PID904338）、非timeout、无survivor、八卡空闲。
prefill/sample721再次完成；第三次真实metadata调用精确为decode C2：cu_q=[0,1]、
原KV=[5]、压缩KV=[2]、余数=[1]、ratio2，原生AICPU内部同步报param invalid。
全部源/依赖身份保持；首轮两次metadata返回，未完成decode/第二采样/退役。
下一步独立首次EngineArgs无AFD的正常原生对照，保留实际79参数并与A58/F21逐项核对，
不先换算子、改输入或以异步host栈猜根因，不增加A5手工场景。
随后[独立原生对照001](evidence/a3-text-normal-worker-20261116/prepared-native-control-001/README.md)
实测通过：214.308542s，单原生进程退出0、无timeout/kill/survivor、八卡空闲；正常prefill/
decode输出721、14，自然长度停止并退役。首次EngineArgs无AFD、运行时无afd_plugin导入，
460源及3753只读身份不变。实际完整79参数与AFD003的A58/F21精确映射后SHA等全同，
首hidden/token721逐字节相同；前三次metadata实参也全同，但第三次原生返回、AFD失败。
下一轮隔离可选query-length None→empty边界，仅诊断补充由真实cu_q差分的等价长度，
尚不接受为生产修复或完整AFD数值通过，不改变上游、原生算子或其余实参。
该[query长度隔离004](evidence/a3-text-normal-worker-20261116/prepared-execute-004/README.md)
实际223.134711s，A退出1/F退出0、无timeout/kill/survivor、八卡空闲。原输入均未传
seqused_q，诊断实际转发int32/NPU的[4]、[4]、[1]；前两次返回，第三次decode C2仍由
SparseFlashMlaMetadata内部同步报param invalid。首prefill/sample721及4096B hidden与003
一致，未进入decode模型forward/第二采样/自然退役；补参未解决该失败，不进入生产。
458源和3753只读身份不变。下一步只读观测实际stream、线程、推理模式及小输入的
stride/storage offset/NPU格式，保留原实参；不增加A5手工场景或将部分结果提升为验收。
随后[原输入上下文005](evidence/a3-text-normal-worker-20261116/prepared-execute-005/README.md)
实测217.817924s，A退出-6/F退出-9（监督清理owned F PID919141），非timeout、无survivor、
八卡空闲。三次调用同MainThread/device0/stream、inference=True/grad=False；小输入均
format2/连续/offset0，观测errors为空，第三次decode C2仍失败。79参数设备SHA、首721和
4096B hidden与原生一致，原生前三次显式实参相同；source/3753只读身份保持。
独立原包审计与根代理复算同SHA`e97951ef58f024862a025fea484049040df9cb8d3f3574f4b850489f20910ead`。
结束后只读补采本轮进程自有CANN日志，核对stream47/task942及custom so装载成功；尚未
得到具体KERNEL_LOG_ERROR。根因未闭合，不改生产或将首轮成功计为正常完整执行通过。

2026-11-15 连续收口 [profiler 中断边界](evidence/profiler-baseexception-review-20261115/README.md)
及 [F daemon 错误发布](evidence/profiler-ffn-baseexception-review-20261115/README.md)：
Session 后台中断保持 failed，已有执行异常不被导出或日志失败替换；F 在 STOP 后导出中断
不再错误报告退出/清理成功。Session 同最终测试 RED18F/34P→GREEN52P，Linux52P；
F 完整模块正常导入、真实 Python 线程的6个补充 host 场景完成RED/GREEN，原生依赖测试待A3。
最新 Linux 全部 CPU 单测1987P/525S、0F/E，388源前后不变；首次收集器包名错误的原始
JUnit保留，修正后的完整记录为 linux-all-cpu-002，跳过项不计原生或NPU验收。
独立A3后继包004冻结429源/145生产/191身份，更新4个生产边界及5个完整测试模块，尚未发送执行。
旧设备包不变，1.7/1.10仍pending，任务勾选仍54/109。

同日 [两部署CLI输出保护](evidence/recipe-output-source-protection-20261115/README.md)
修复报告覆盖自身输入、插件源码或source lock后仍记录旧SHA的问题。相同测试/输入旧代码
11个碰撞场景失败，新代码完整10个E2E方法通过，含18个输出场景及原52个汇总CLI；
Linux3.12另对受影响的18个输出场景验证通过（2方法，未重跑52汇总场景）；
只修改planner/prepare及既有E2E，不启动设备，不提前闭合1.6部署验收。

2026-11-15 [同一eager会话的U1回退汇总](evidence/eager-mixed-summary-20261115/README.md)
修复离线观测入口将每个parent固定为U2的问题。保留U2 ticket链，新增实际U1串行
body、显式F同步、TP2本地空行continuation pair边界；没有生成parent完成或设备完成结论。
相同最终测试在旧工具上5个合法输入失败；Windows3.14与WSL3.12各52CLI符合预期，
独立Windows复跑同样通过。既有52个eager/配置及34个Graph场景另批通过。
本增量只改工具及E2E，生产/协议/准入未变；构造日志不计NPU或正常部署验收，1.6仍待闭合。

2026-11-15 [eager U2 多流后继](evidence/a3-dspark-native-live-eager-streams-20261115/README.md)
已冻结436源/145生产/191身份；D/F2真实eager bootstrap、同session的原生KV初始化、
staged执行及取消/存活/随后请求接线已经审查。另修复组件owner初始化前漏绑transport，
不把该源码检查算作协议核对器覆盖。87项正反例及32项实际生产selection通过；
归档内12源闭包复跑、190个锁定Git对象和13项篡改拒验通过。
生产与跨层包相同，完整38模块回归及组合NPU均尚未执行；host记录不证明物理重叠。
既有Graph47/76/139/225项核对器回归通过，冻结旧包不变。继续非Engram实际缺口，
A3恢复后先收集live-004，任务勾选不变，不新增A5手工场景。

2026-11-15 [Graph 跨层后继](evidence/a3-dspark-native-live-cross-layer-20261115/README.md)
已完成独立源码审阅、225 项实际 codec/诊断协议正反例和 64 项生产 selection 检查；
读取 A/F 各自实际 bank 顺序及 U2 warmup/capture 动作，要求 U1 串行、U2 跨层，
保留旧 bool 坐标误接受 RED。冻结 434 源/145 生产/191 安装身份，完整归档、14 源可执行
闭包重跑、190 个锁定 Git 对象与 11 项篡改拒验通过，独立审计 JSON 逐字节一致。
145 生产仍为 144 份旧 CPU GREEN 加单独配置 GREEN；
[38 完整模块加 35 签名的 CPU 回归包](evidence/a3-dspark-cross-layer-regression-cpu-20261115/README.md)
已准备、独立核验 427 源/419 Python 编译，但尚未执行 pytest。
没有本组合的 NPU、物理重叠或性能结果；A3 转发未恢复，先收集 live-004。
继续 eager U2 多流与同一 DSpark/PD owner 生命周期组合，任务勾选不变。

2026-11-15 [Graph 全轴 stage_serial 后继](evidence/dspark-native-pd-axes-successor-design-20261115.md)
已实现；真实 emitter 复现普通 body 缺 logical/effective 字段的四个合法输入失败，
修正为仅 configuration 检查这些字段后，[独立核对](evidence/dspark-native-pd-axes-review-20261115/README.md)
139项正反例及64项生产选择组合符合预期，原始失败和SHA核验的重构闭包保留。
[prepared-001](evidence/a3-dspark-native-live-axes-20261115/README.md)已冻结433源/145生产/191身份；
生产为144份旧CPU基线加1份独立234项配置GREEN，不称一次全145模块回归。
完整归档、190个Git对象、归档核对闭包及10项篡改拒验通过，独立审计JSON逐字节一致。
它改变hybrid/F等待与join，未选择diagonal跨层交换，也无设备、物理重叠或性能验收。
继续[跨层实际顺序入口](evidence/dspark-native-pd-cross-layer-successor-design-20261115.md)，勾选不变。

2026-11-15 接续[PD consumer Graph/U2 跨层配置接缝](evidence/dspark-pd-cross-layer-config-review-20261115/README.md)：
既有 A/F wavefront 消费者已实现，但 PD 校验只允许 stage_serial；现仅允许精确
consumer/native U2 的 cross_layer_v1，保留 U1/bulk 拒绝、角色 preflight 及其余
checkpoint/draft/布局约束，完整模型门禁不变。同测试真实 RED 为8F/14P，单一生产
条件修复后完整 CPU 配置模块234P、0F/E/S；root独立整模块及归档 RED/GREEN 重放通过。
该证据只证明 CPU 配置结果，不证明 paired wire、设备 wavefront 或完整模型。
继续全轴 stage_serial 组合及独立跨层顺序入口，未完成项保持 pending。

2026-11-15 继续[Graph/U2 compute 轴与真实 DSpark/PD 生命周期接入](evidence/dspark-native-pd-compute-successor-design-20261115.md)：
A/F2 在构造时固定同一 compute 开关，U2 使用原生产 pipeline，U1 和 eager 保持串行；
捕获、回放的 bank/stream/20 个 event owner 身份与实际 bank 独立交叉核对，
HCCL 仍沿用 Graph parent stream。新增[独立核对](evidence/dspark-native-pd-compute-review-20261115/README.md)
76 项正反例及 32 个生产 selection 组合均符合预期，root 新进程复跑结果逐字节相同。
pipeline/数值/请求观察作为合成核对输入，不是设备模拟验收；没有 NPU 执行、
物理重叠或性能结果。[prepared-001](evidence/a3-dspark-native-live-compute-20261115/README.md)
已冻结433源/145生产/191安装身份；完整归档审计重取190个Git对象，归档内核对及8项
篡改拒验通过，独立新进程复跑审计结果逐字节相同。继续其余多流组合，任务勾选不变。

2026-11-15 继续[同一原生 D owner 的取消/存活/后续请求](evidence/dspark-native-pd-lifecycle-successor-design-20261115.md)：
独立第三 P source、A/B 两轮 Q3 后真实取消 A、B 单请求继续并完成、空步退休后 C
重新 typed read/首次与缓存验证、真实 worker slot 复用的工具已实现；未规定物理块必然复用。
按锁定原生 FIA/Graph 合同区分单请求有效3行与捕获填充6行，保留完整父 hidden/aux
与地址，只将有效行纳入独立数值预期；每轮从真实 SchedulerOutput 取前置坐标，
public get_block_ids 观察当前 lookahead 分配。独立[监督器核对](evidence/dspark-native-pd-lifecycle-review-20261115/README.md)
复现3个坐标误接受后修复，47项离线正反例全部符合；root新进程复跑JSON逐字节一致。
这些是源码实现及CPU harness证据，未执行真实取消/采样/传输/Graph或NPU，不勾选任务。
该场景 prepared-001 已冻结432源/145生产/191安装身份，归档内47项核对及实际190个
Git对象重取通过；独立新进程复跑审计JSON逐字节一致。继续 Graph compute 轴独立组合，
完整模型/准入和Engram后置边界不变。

2026-11-15 后续[Graph/U1 与双请求 U2 核对及源包审查](evidence/dspark-native-pd-graph-u2-review-20261115/README.md)：
两条不同原始 prompt 的真实 P source、同一 D owner 原生 Scheduler/execute/sample/propose、
全父 aux 归并及双 stage F2 观察已实现；N2 draft 始终 eager，目标 Graph/U2 由首次配置声明。
原 Graph/U1 prepared-001 通过真实 emitter 复现 mixed schema 的 KeyError 与 eager scope
事件不匹配，仅保留为历史未执行候选。修正后的 U1 两政策 prepared-002 和 U2 constant7
prepared-001 均431源/145既有CPU GREEN生产/191安装身份；完整源包审计实际只读重取190个Git对象，
各包闭包分别通过21/27项 actual-codec/真实emitter离线正反例及logging核对，未执行设备。
新增严格事务顺序、双层及3/6行约束，拒绝重复空事务、未知host、伪完成边界和空层列表。
继续取消/存活请求/随后请求的同owner生命周期；A3恢复后先收集live-004，再顺序补设备。
这些工具仍采用受控target/FFN算术，不代表完整AFD构造、正常准入、Engram或性能通过，勾选不变。

2026-11-15 继续[DSpark/PD Graph/U1 后继入口](evidence/dspark-native-pd-graph-successor-design-20261115.md)：
D/F2 在首次 EngineArgs 构造时声明 FULL_DECODE_ONLY/N2，draft 及 P/F1 保持 eager；
真实 factory/HELLO、同一 bootstrap/native KV/ModelAclGraphManager/capture 接线已实现，
接受拒绝驱动继续复用原生 Scheduler、sample/propose 及整池保护。F2 用真实 worker daemon/
GraphService，逐层设备观察 copy_ 随图捕获、完成后读回，并明确计入额外开销。
v3 双向观测核对 actual GRAPH_QUERY/REPORT/DECISION/ACK、FORWARD/COMPLETE/STOP，
父事务数量从实际零步推导；离线 actual-codec 合成输入核对器通过27项，不能计作设备通过。
同轮复查修正“组合未接线”的旧判断；正常部署门禁和任务勾选不变，继续独立审查、源包及两请求U2。

2026-11-15 离线增量：Ready 三次真实 typed-query 的观测入口及独立 P/D/F1/F2
两轮原生 D 接受/拒绝工具已实现并完成源码/静态审查；正常协议的离线核对器
完成 3 个正例、17 个拒验反例。三个[下一轮准备包](evidence/dspark-live-offline-preparation-20261115/README.md)
均为428文件、145生产与已审49项CPU GREEN源包相同；只读重取190个锁定Git文件，
另保留1项既有profiler身份。A3转发断开，尚无上述场景的设备结果，live-004实际
结果与清理仍待收集；不据此新增CPU/NPU通过数或勾选1.3/1.5/1.10。

同日继续完成[原生proposal返回后host异常](evidence/a3-dspark-post-proposal-failure-20261115/prepared-001/README.md)
的独立P/F工具与冻结包：真实proposal原body、原sentinel/traceback、五次typed
not-ready、原pending及model/KV资源保留和双PID受控SIGKILL合同已实现并审查。
独立离线11个codec及13个outer gate反例/正例通过，root复跑JSON字节一致；
429文件中145生产与既有CPU GREEN相同。无NPU执行，不称kernel故障、自然清理
或quiescence。另更正collector真实outer字典/inner字符串tick归一，实际Linux CPU
RED/GREEN与独立复核已保留；正常下一轮包改为prepared-002，旧001原件不覆盖。
继续核对Graph/U2组合非Engram实际接入缺口，整体验收与任务勾选不变。

2026-11-13 Graph 父事务观测增量：A/F 原 owner 现在记录实际 reservation、
replay 调用/返回、finish 调用/返回、eager scope 返回及失败，复用冻结开关，
默认关闭，不改 wire/准入/body/流/fence。旧生产三个外层场景 RED 3F；新生产
六完整 CPU 模块单轮 376P，补普通 sink 失败及共享容量 fixture 后另轮六完整
模块 380P，均 0E/S。collect-002 用 A 实际发送消息驱动 F，五个父事务的
身份/key/decision 摘要一致；为串行 CPU 协议证据，无正常启动日志或 NPU。
汇总工具新增严格 host 清单，最终既有 52 与新 34 个 CLI 场景分别通过；修复
未知 host 错误被静默忽略的问题，原误报成功及首轮容量夹具失败均保留。见
[审查](evidence/graph-host-observations-review-20261113.md)和
[原包核对](evidence/graph-host-observations-validation-20261113.json)。
这些仍不证明实际部署、设备完成、完整组合或性能，1.6 保持 pending。
随后[单次真实 F 设备组件](evidence/graph-host-device-review-20261113.md)完成
23 parent、90 条 host/235 条 body 记录，12 次 replay 无 Python body，A/F 0/0、
无 timeout/kill、八卡 idle。142 份插件 Python 与上述 380P 快照一致；A 仍为
受控 base manager，未执行 A coordinator，不能据此声称双端正常部署通过。

2026-11-13 F Graph × profiler 新增独立设备组件：可选 probe 模式复用真实
factory/daemon/runner/default NPUGraph/pool 与 deferred F role-profiler。
全关/全开各 23 parent、A/F 均退出 0，无 timeout/kill，前后八卡 idle；每会话
实际 60 次 AI_CORE MatMul、240 个 AI_CPU HCCL payload，按 parent host scope、
真实 ProfilerStep 与完整 Model/Task/Stream/时间身份核对 CSV。36 次 replay
MatMul 不经过 Python 模型/aten::mm；STOP 时尚未导出，owner finish 后 closed
才管理 stop/cache/group close。首轮实际设备成功但 CSV Step ID 沿用 capture
使后处理失败；原记录保留，只离线重读原 tar，随后仅新跑全开。见
[审查](evidence/graph-ffn-profile-review-20261113.md)和
[独立核对](evidence/graph-ffn-profile-validation-20261113.json)。
142 份插件 Python 与既有 STOP-poll GREEN 源快照相同，154 安装源码核对一致；
没有新增 CPU 通过条数或 A5 手工场景。物理 parent/side 映射、有效重叠、A 侧
timeline、完整模型与性能继续待验，1.7/1.8/1.10–1.12 不因组件结果提前勾选。

同日 [DSpark/PD 生命周期源码复查](evidence/dspark-pd-lifecycle-source-review-20261113.md)
确认已接入的 Ready 前设备同步、取消/同 ID 复用及 queued failure 边界，无新增
可证实的叶子接口漏洞。Graph+PD、DSpark+PD+U2 仍受非 Engram 专项谓词限制，
属于未闭合组合准入/集成范围，不改写成只差硬件或 Engram。继续补组合接口检查
及 Graph 外层父事务观测；保留生产门禁，1.3/1.5/1.6/1.10 继续 pending。

2026-11-10 增量：H14 已实现延迟 profiler owner、固定状态/控制 RPC、显式HTTP endpoint plugin 和 headless F 本地 Unix socket 入口。A3 CPU 的 API025 为62项通过，headless027为91项通过，源校验分别15/17项一致；真实 Unix socket、主循环清理与执行 owner 的 pending/applied 分开记录。独立 NPU role-owner002 的四轮采集均有真实 AI_CORE MatMul/CSV 对应证据，但仍是受控会话组件，不是完整 A/F 服务或管理入口到 NPU 的链路。完整模型业务不阻塞、真实服务启停/trace、压力和性能尚未验收，1.7/1.8/1.10/1.11 保持待完成。

2026-11-10 DSpark Graph 增量：F 已在隔离 spec/Engram 前保留不可变 target
N+1 并用于原生 MC2 容量（001四核心模块218通过，额外Async旧夹具失败在独立
003完整55项闭合，原始001/002均保留）。严格 target Graph+eager draft 准入与
A/F shape binding 的admission002为323项通过；HELLO查询宽度协议的protocol001
为321项通过，均21个真实上游源码校验一致、无device probe。首次admission001
是供体源码误分类导致的预检拒绝、零测试执行，不能算通过。几何/缓存/消费者、
target/draft capture隔离、q>1受控NPU组件及真实proposal/接受率/KV/history仍需
各自证据；正常工厂Graph选择与生产完整模型门禁未开放，1.3继续待完成。

2026-11-10 消费者增量：固定 q 已贯通 Graph key、原生 U2 几何、A control/
coordinator、F service、stage bank 与角色 binding。geometry001 为九模块
551通过/4夹具失败；修复共享夹具的 native DP-size 位置参数后，geometry002
完整 coordinator 100通过，8个运行文件未变化，均21项真实上游源码一致、无
device probe。四个真实 native outer target→draft capture 隔离用例通过；
不将两次记录合成为一次全绿回归。q3受控NPU组件另行记录；正常启动自动装配、
真实proposal/接受拒绝/KV/history、A5和组合性能仍待完成，1.3不勾选。

2026-11-10 q3 NPU组件增量：三层 cross_layer_v1、五Graph轴均1、TP1 A/F、
设备0/2，一次受监督实测通过；每rank27事务（4warmup/4capture/13replay/
6eager），含R2/T6同key回放及R1/T3布局回退。31项实际源码一致、八卡前后
空闲、无超时/强杀、native TP前后[1.0]。仅受控算术/Graph/HCCL组件，
不等于真实DSpark接受率、HC/Engram/MoE、KV/history、A5或性能验收。

2026-11-10 H13源码审查完成：31个锁定Git文件的SHA重新核对；shared physical
pages/index/alias、native Engram历史、async lookahead和hybrid失败路径均已映射。
不迁移原生已有等价的旧v0.23补丁。缺口是实际materialized布局证明及pre-transfer
consumer、DSpark prompt-draft上下文协议和Graph/PD native生命周期组合；接口、
所有权和成本见evidence/hccl-combination-source-gap-20261110.md及JSON。
1.2仅代表审查完成，1.5/1.9和PD设备验收继续pending。

2026-11-10 正常 Graph 装配增量：A load/profile/原生 KV 初始化和 headless F
start RPC 已接入同一 cache/session 的正式 owner；q1/q3×U1/U2、单次加载、
KV 复用拒绝、STOP 后禁止重启等 CPU seam 已闭合。bootstrap002 为884通过/
6夹具失败；仅修正3个测试文件后，bootstrap003四个完整模块192通过，其余10个
模块在002已有698通过。两轮生产字节相同、21源码项一致、无device probe，
不合成为一次890全绿。提交1a79b8f，完整模型门禁仍关闭，1.3保持待完成。

2026-11-10 H13布局增量：typed materialized KV/page/index/alias只读采集、严格
JSON比较及完整请求block映射检查已实现；原生真实CPU views的layout001两完整
模块59通过（43 core/CLI +16 native），27锁定源码一致，无device probe。
提交f83dcd8。当前工具未注册buffer或传输KV；实际connector pre-transfer
consumer、P/D数据与history/Engram验证仍未完成，1.5/1.9继续pending。

2026-11-10 交付范围按用户最新要求仅代码和操作说明。前述完整本地证据提交
f83dcd8/1a79b8f已用backup/dsv41-local-evidence-20261110-1a79b8f保留，源码等值
交付为bd57c33/e0b1315；新增raw/XML/environment/result/identity仍只留本地。
Graph顺序配置f005c37在A3十完整模块864通过、21源码项一致；默认mapping/HELLO
保持原样，非默认只Graph，准确绑定到所有peer和本地owner，原NPUprobe未改。
真实AFD→Ascend→GPU→ActiveKVConnector零计算路径测试08355ba的完整A模块
27通过、29源码项一致；首轮19通过/8CPU平台夹具失败保留。该结果不包含正向
Graph模型计算或采样，不能将target COMPLETE解释成native KV/采样全部完成。

2026-11-10 H13 serving consumer软件增量610be1b已实现外部Mooncake连接器，
原生内存注册/地址发布/唯一batched-read前分别校验typed proof、immutable
page/stride/group和完整window-tail block映射，保留原生DONE/失败收尾。
adapter001为145通过/31共享HF夹具setup错误；仅修该fake后adapter002完整
connector模块43通过。两轮33源码项一致、生产代码相同、无device probe；不合成
一次176全绿。当前仅eager/no-spec组件、完整模型guard仍关闭；真实Mooncake
设备注册/传输、Graph/DSpark PD和请求历史仍待验收，1.5/1.9继续pending。

2026-11-10 H12前置缺口：旧V4 gate-on-FFN的A/F两端已保留原生
独立FP32路由输入，缺省None仍沿原生分支。old-v4-fp32-001三个完整
CPU模块40通过，含新增四个协议往返用例，34项锁定源码与安装字节一致；
无device probe。该增量不表示旧V4 HCCL/MTP已准入或完成真实量化MoE
验收；N步proposal、F consumer、END/STOP和原生采样仍需后续增量，1.4保持待完成。

2026-11-10 native FULL/KV 增量cff7f50：真实AFD→Ascend→GPU execute和
原生FULL manager、ActiveKV/Engram prepare/finish的CPU生命周期用例已补齐。
native-kv-full-replay-001为27通过/2个CPU backend选择夹具失败；仅修fake后
002完整模块29通过，37项锁定源码与安装字节一致，生产代码未变、无device
probe。成功execute允许返回None，失败不伪造COMPLETE/KV完成；该结果不含
真实设备Graph、sample_tokens或请求历史验收，1.3/1.8继续待完成。

2026-11-10 PD真实设备首轮transfer-001：固定源码/二进制/model config后，
P/D TP1、设备0/2的空请求、非连续映射、window-tail映射及alias冲突拒绝四项
数据/DONE检查均通过；但D进程在解释器退出时SIGABRT（exit -6），P退出0，
故整轮FAILED。无监督超时/强杀，八卡前后空闲。原始结果不可改写，布局不匹配
第二轮未启动。正在处理插件receiver的合作退出边界；尚无C++栈，不能将原因
归结为某个engine析构。底层TransferEngine有unregister_memory API，但原生
Worker/GlobalTE wrapper无停止/关闭接口。1.5/1.9/1.10保持待完成。

2026-11-10 native FULL采样增量d6f7f29：同一owner的execute(None)后继续真实
native sample_tokens/sample/postprocess/ActiveKV post与继承AsyncOutput.get_output，
成功返回token与KV结果并清理绑定；采样异常保留未post的KV和历史，未追加HCCL
事务。native-kv-full-sample-001完整模块31通过，42项锁定源码一致；仅两个测试
文件变更，无生产/guard/device改动。head、q1无拒绝历史算子和copy构造是明确CPU
边界，不构成NPU Graph、DSpark拒绝/接受或真实Engram历史验收，1.3/1.8继续待完成。

2026-11-11 用户明确优先DSpark：暂停普通旧V4 MTP后续wire/normalcaller开发，
保留其已有本地改动和独立证据。优先推进V4.1原生DSpark proposal、真实接受/拒绝、
KV/Engram history及U2/Graph组合；不把普通MTP的N步/F侧draft协议套用于DSpark。
已启动的PD组件退出验证继续收尾；A5完整模型和组合矩阵留后续集中手工验收。

2026-11-11 PD组件退出增量364a19f：D接收线程按整native batch关闭准入，
以实际DONE/ACK为已接纳任务完成边界，私有sentinel退出并join；失败保留owner。
retirement-003完整连接器CPU57通过/33源码一致；先前001为95通过/1夹具失败
（其中39签名通过），002为旧生产字节57通过，三轮独立保留不拼总数。
新retirement-transfer-001与retirement-mismatch-001两个独立A3 NPU组件会话均
通过：四种数据/冲突case、布局负例地址发布前拒绝、整slab/DONE、接收线程退出、
P/D退出码0、无超时/强杀、八卡前后空闲，33源码/2二进制/模型配置身份一致。
原transfer-001 SIGABRT仍为独立失败；新路径不注销内存/不关闭engine、不验证
原生模型allocator、Engram历史或完整PD服务。1.5/1.9/1.10保留待完成。

2026-11-11 DSpark原生CPU增量：以已交付364a19f的Git archive仅覆盖本轮
七个测试/共享fake文件，排除暂停的普通MTP工作区改动。dspark-cycle-001仅本地
prepare未执行；首个实际A3CPU轮次002为131通过/19夹具失败/0error/0skip，
38项安装源码一致，无device probe。失败分别是draft NONE-manager CPU边界
缺字段，以及target模型CPU边界缺原生getter；先保留原日志，再修夹具重跑。
回归仍不证明真实draft backbone、KV写入、NPU接受率或整模型运行。

随后003完整两模块148通过/2断言夹具失败，57项源码一致；其中A runner完整
33项通过，包含q3三请求的全接受/首拒/中拒、native proposal、history与下一轮
Engram lookback/KV收尾，以及proposal异常保留状态。仅修coordinator测试将
profile/dummy合法None元数据与target共享None哨兵误判为泄漏的问题后，004
完整coordinator117项通过、57项源码一致。两轮独立保留，不合称单轮150全绿。
最终七个测试文件在004上传清单与本地逐hash一致，生产使用364a19f archive、
暂停MTP未混入、无device probe。native proposal/Markov host主干保持原生，
rejection/count/history/lookback是明确greedy CPU设备叶子替代；target捕获输出
为预供夹具，draft模型/KV预计算为CPU边界。1.3/1.8仍待真实NPU组合验收。

软件交付：DSpark CPU回归与独立说明ff7fd08、原生A5显式--dspark入口及说明
1aac3e4已推送到feat/dsv41-flash-hccl-v030，独立ls-remote核对完整SHA
1aac3e436ae60719e2be466582a0a003cfffa2d2。默认DSpark-off基线保留；新入口仅
手工指定N1/2/3时追加原生配置/详细指标，先做N1，未运行A5或解除AFDguard。
新增raw/identity/环境证据均留本地，旧操作说明只补链接。继续推进native U2
prepare与父级采样/历史/KV组合；不以本次推送视为1.3/1.8完成。

2026-11-11 DSpark U2软件增量：仅三个既有测试文件从已交付1aac3e4 archive
覆盖，保留native Hccl/NPU/GPU UBatchRunner.prepare、actual request slicing/
FULL子context、同一Engram ModelState和TargetDeviceMetadata.build状态机。
q3/R3按4/5两段执行准备后，原生父RejectionSampler/history/proposal/KV序列
保持一次；公开FORWARD布局、全接收/首拒/中拒token、历史、next lookback、
metadata/lookup lease退休、异常保留KV和无draft F事务已覆盖。
dspark-u2-001首轮完整A35+coordinator117共152通过/0fail/error/skip，64项
安装源码一致、无device probe；生产137个Python文件仍逐字节等于1aac3e4。
输入materialization/metadata operator/executor、inert graph replay、draft模型/
KV预计算和greedy采样kernel仍明确CPU边界。该结果不代表连续设备解码、完整
模型、真实Graph/多流、Engram gather kernel或性能验收，1.3/1.8继续pending。

U2三测试文件与说明已提交dffeb4e并推送，独立ls-remote确认
dffeb4ebf7e8cbb8d2271379c500c688e6e6e63a；新增验证raw/环境/identity未交付。
当前开发优先DSpark，后续软件增量为连续target/proposal周期及Graph miss/fallback
与采样/历史组合；真实backbone与完整模型依赖A5集中验收。

2026-11-11 DSpark连续周期增量：以dffeb4e Git archive仅覆盖四个既有测试/
共享fake文件，普通MTP改动排除。U1/U2沿用同一请求、ModelState和KV owner，
第一轮真实draft/bonus/computed派生第二轮freshbatch，actualCachedRequestData
经过原生update_requests更新CPU镜像，第二套独立targetlogits改变接受/拒绝结果。
断言两轮token、完整history/computed/total、下一轮lookback与KV收尾；第二轮
proposal失败保留已更新history、前一draft与未postKV，不伪造回滚。

独立U1/U2组件用例覆盖hit/miss-eager-completes/hit，middle使用实际原生prepare/
AFD模型wrapper/continuation及独立unsplit算术输出参考；FP32/rawIDs协议可见，
每父COMPLETE后原生DSparkproposal无新增F事务，retainedGraph entry/storage/value
保持。ready/missing/ready是脚本化F可用性，不证明F实际recapture或终止故障恢复；
三步组件未连接完整nativeexecute/sample/history链，不能与两轮用例合称该组合。

dspark-continuity-001首轮完整A39/coordinator119共158通过/0fail/error/skip，
64项安装源码一致，无device probe；137个生产Python字节仍等于dffeb4e。
四文件双方独立源码复核及root快审无阻断，Ruff/format/3.11AST与本地规划harness
通过，upstreams保持未改。CPU输入/metadata/devicekernel、inertFULLreplay、draft
模型/KV预计算及greedy拒绝算子边界沿用；不代表NPU连续解码、实际Graph/KV/
Engramkernel、完整模型或性能验收，1.3/1.8继续pending。新增raw/identity不交付。

连续周期四测试和说明已提交并推送7c091860c7cd1b779ed50da03d9926d90aeabea6，
独立ls-remote核对一致；本次恰为五路径，无新增raw或环境文件。接续开发同一
真实A execute/sample/history owner与Graph miss/eager完成之间的组合衔接；
已交付的连续两轮与三步组件回归不作为该完整组合已通过的证据。

2026-11-11 DSpark三轮组合软件增量：同一真实A runner/model/EngramModelState/
requeststate/manager/cache，nativeexecute dispatch由F脚本ready/missing/ready
选择FULL/NONE/FULL，每轮实际execute(None)→原生sample/rejection/history→
A-localproposal→KVpost；后两轮输入经真实CachedRequestData/update_requests，
从上一轮真实last/draft/computed派生。中间ExecuteModelState公开hidden/aux与
独立unsplitCPU算术参考逐值比较后，原生sampler消费其原值；3套独立logits使
三请求接受/拒绝轮换，最终完整history、computed/total与nextlookback精确检查。
每execute返回先检查Engram lease退休/history未变/KV仍bound，sample后KV清，
draft公开context无target sidecar/AFDmetadata，无新增F事务，retainedGraph
storage/value和key不变。

仅扩四个既有测试文件：FULL/NONE input/metadata/KVcontext叶子、正确替换旧
CPUstate的公开hookhandle、bundle前固定vocab128CPU权重和只读unsplit参考。
U2 NONE首child snapshot打开CPUlease，未运行实际SFA metadata executor；无
Engram原生空producer分支、FULL惯性预供输出和原devicekernel替代均明确。
dspark-combined-001首轮完整A41/coordinator119共160通过/0fail/error/skip，
64项安装源码一致、137生产Python仍逐字节等于7c09186，无device/service/install。
FastAPI保持0.123.10；不证明NPU Graph新输入重算、真实Engram/KV或模型接受率。
1.3/1.8仍pending，新增raw/identity只留本地，后续A5集中验收。

三轮组合四测试和统一说明已提交并推送234863677fc6fcdd29cdd0c8b3d13f2e80b742d7，
独立ls-remote完整SHA一致、HEAD与origin ahead/behind为0/0；新增raw/环境/identity
未交付。Root独立解析单轮XML160、逐项核64锁定Git源码pin、137生产基底字节及
四staged/uploaded源hash，3.11AST/Ruff/format、说明链接/Bash语法与规划harness
通过，upstreams保持不变。保持DSpark优先，普通MTP仍暂停；设备Graph、真实
Engram/KV和完整模型结果继续待A5集中验收，不以本次软件交付勾选1.3/1.8。

2026-11-11 按用户重申继续自主开发，优先DSpark、A5集中手工验收。A3只读
preflight-001 SSH可达，Python3.12.9/torch2.10.0/torch-npu2.10.0.post4/
vLLM0.30.0+empty/Ascend7e37e29a7/FastAPI0.123.10保持，64锁定源码一致，未
运行device/service/install/pytest；干净基底137生产Python保持未改。

DSpark统一HTTP请求采集工具已实现：连接既有native服务、先health、顺序单序列
固定seed请求、逐轮保存token IDs/finish reason/原生详细metrics和前后Prometheus
全文。验证accepted/drafted/histogram/聚合与native比率，允许accepted=0；缺指标
不补0，HTTP错误/截断保留body，首失败停后续completion并保存exit1。新建独立
目录，已有结果保留；不启动/停止服务，不安装依赖。启动身份采集增加实际
stats.py/Completion协议源码hash，操作说明新增短命令与重点回传文件。
本地受控HTTP真实CLI13场景及URL参数5场景通过，响应fixture由ced6857a原生
RequestSpecDecodeMetrics精确AST生成；3.11AST/Ruff/format、Bash语法/CLI9负例、
两guide链接/Bash块检查通过。独立审查无阻断，6d4578a5f1f7f84149644c8d5455f118784080ee
四路径工具/说明提交已push，独立ls-remote完整SHA一致，新增raw/环境/identity
未交付。本地HTTP证据不代表A5、AFD、NPU或完整模型通过。

并行PD源码核查确认V41 draft是真SWA而SFA上层仅MLA；未来Mooncake consumer
必须按P实际computed prefix（含压缩N−1语义）在execute返回前同步投影draftSWA，
保原group/block identity，并在D绑定源generation与后分配的本地generation；
target+draft整页/全TP就绪后才可运行。完整垂直设计和精确源位置保留在
evidence/dspark-mooncake-prefix-consumer-state-review-20261111.md；未生成无人
消费codec、未解除guard，也未为当前允许路径不可达的SFA取消进度打补丁。
本轮核心继续真实native capture factory/sideDAG→同owner DSpark采样与KV组合。

2026-11-11 native factory组合已闭合：保留GPU.capture_model→ACL.capture→ModelCuda.capture/create_forward_fn、真实Ascend dummy/InputBuffers、U2 prepare/begin_capturable_run、ModelWithContext和store_capture_output；仅base.capture、device任务/stream/event/算子/远端F为明确CPU叶子，replay句柄惰性不重跑Python。
q3 U2按4/5，stage_serial/compute=1/hybrid=0，capture全padding F返回0，hidden/aux与独立未分段参考逐值一致，再由同owner execute(None)→native采样/拒绝/history→本地DSpark→KV post消费；成功及draft异常均检查存储保留/隔离。
Factory metadata只在真实prepare打开，native begin/finish及manager finally幂等退休；运行期lookup/replay仍由原execute收尾。无生产runtime改动。

A3 dspark-native-factory-001首次单轮162通过/0失败/0错误/0跳过（A43+coor119），66项安装源码一致，137生产字节等于6d4578a5，四overlay冻结=current=upload。
FastAPI保持0.123.10，无device/service/install。Root独立核XML、66锁定Git pins、137基底与四index/upload字节；独立状态审查无阻断，3.11AST/Ruff/format、说明链接/Bash块和harness/上游保护均通过。五路径测试/说明提交并推送ee0fdfe0467f8d036590a3aad08c2478f9b146cb，独立ls-remote完整SHA一致。新增raw/环境/identity仍只留本地，不以CPU结果替代NPU Graph/真实模型接受率。
1.3/1.8仍pending，下一生产增量继续审查DSpark×Mooncake请求级prefix ready消费门。

2026-11-11 DSpark × Mooncake PD 软件切片：真实加载后绑定 target/draft，原生 KV 初始化前保留 draft 组，注册后锁定实际缓存视图。P 每个分块从 native KVCacheBlocks 活列表冻结 allocation（含 SWA null/回收复用），execute 收尾后调用原生 lower context writer 投影 BF16 auxiliary 并同步；TP 全员报告成功后发布 Ready。D 按原始 prompt 摘要、source/destination generation、all-group blocks/schema/TP ordinal 获取读取租约，实际同步读成功及 typed DONE/ACK 后才本地完成；不明 DMA drain 保留双方资源。请求 ID 复用、取消、bounded tombstone、legacy DONE 拒绝及 native Worker 释放前的 connector shutdown 已补齐。仅 eager/U1、固定验证、CP/PP/DP1、同 P/D TP；P 中途抢占明确 terminal，不实现恢复。

request-ready-001=428P15F/84pins，002=449P2F/85pins，003=449P2F/85pins 均保留原始失败。最终004新冻结单轮六完整模块451P/0F-E-S（prefix39/connector89/config117/signatures38/A49/coordinator119），85项实际安装源码一致；root 独立 XML+锁定Git+138运行文件(base137+4生产delta)+13上传与index字节逐项复核通过。native projection/组发现/KV allocator/请求协议保持原生，RoPE/scatter/stream sync、target compute、socket/TE 为明确CPU边界；无device probe/包安装/服务启动。源码和15项代码/测试/说明单独交付；raw/identity/环境记录不推送，普通MTP工作区13项原改动已校验保留。完整模型guard继续关闭，1.3/1.5/1.8/1.9/1.10不因CPU结果勾选；真实NPU/完整模型/性能及PD×U2/Graph需后续独立验收。

本切片15个文件已提交并推送7600c093f08c25858303948f90b0a0c0301d35ee，独立ls-remote返回同完整SHA；提交后的138个runtime及13overlay也与004实际快照逐字节一致。root复核记录见evidence/dspark-pd-request-ready-root-verification-20261111.md，新增raw/evidence仍只留本地。后续优先补DSpark PD与既有U2/Graph生命周期组合；A5实际模型/读取/接受率/性能继续集中手工验收。

2026-11-11 独立 P/D CPU 缓存增量：两端使用独立存储，真实接收器生成地址批次，CPU byte-copy 边界搬运 target/draft 整页；独立 oracle 检查全部 D 页、未写区域、P 页不变及 typed DONE/ACK。完整 connector91+A49 单轮140通过、0失败/错误/跳过，85项安装源码与锁定Git一致，138个生产Python文件保持7600c093。root独立核对XML/source/runtime/index和冻结/current/upload字节；3.11AST/Ruff、说明链接、Bash语法及harness通过。提交0ab7eb2e24e879e5385cc7050294a3deb75ffde5并独立
ls-remote确认，13项暂停改动保持不变，raw/环境/identity未推送。此轮不构造D模型，不证明真实DMA或attention；完整模型guard及1.3/1.5/1.8/1.9/1.10仍pending。

2026-11-11 原生D调度增量：保留实际Scheduler构造、connector factory、Request、
KVManager、schedule/update_from_output；Ascend配置初始化/getter为CPU边界。
检查单token prompt回退computed=0、普通prompt的computed=Nprompt−1与DSpark
物理占位行；取消后的KV保持到接收完成。worker完成通知是明确的外部输入，本轮
未串联实际typed receiver/ActiveKV/D模型计算。native-scheduler-001完整
connector94+A49单轮143通过、0失败/错误/跳过，103项安装源码与锁定Git一致，
138个生产文件保持0ab7eb2e。root核XML/source/runtime/index，3.11AST/Ruff/format、
链接/Bash/harness通过，13项暂停改动保持。提交并推送
c4197e68d4e05d04569b3a4002bff70b3cb26bb0，独立ls-remote确认，新增raw/identity
仍只留本地。完整模型guard与1.3/1.5/1.8/1.9/1.10仍pending；下一步串联零token
execute、ActiveKV typed read自然完成通知与同一Scheduler恢复。

2026-11-11 自然PD完成组合继续：独立D存储/model identity/consumer在注册前固定，
同一ActiveKV的真实零token execute接纳读取，真实队列handler执行GET_META、
GET_READY、CPU整页copy及typed DONE/ACK，再由下一零token输出自然通知Scheduler。
首轮001为182P2F（空KV输出None与源export夹具），002为182P2F（旧chunk声明
与合法P/D SWA数量差），003为183P1F（真正读取前的Query逐组等长约束）。原始
失败和独立只读诊断均保留，不合并结果。003实际C5/N5最后draft组源allocation
(1,2,3)、P发布(2,3)、D只分配(8)，独立窗口坐标要求P3→D8；全部copy未开始，
Query构造抛错，engine.reads为空。不能用缩短Ready或放宽整页oracle掩盖。

生产最小修正保持完整Ready/lease身份，Query要求组数一致且各D组长度不超过P
发布组；receiver现有exact effective-tail/layout/alias/draft非空校验均保留。
新增tail/非draft空组wire往返与missing/extra/overrun拒绝公共边界，第四轮004
完整prefix44+connector94+A51单轮189项通过、0失败/错误/跳过。root独立解析XML，
逐项核103安装源码/锁定Git、138运行文件base+exact1生产delta及六冻结/current/
upload字节；无device/service/install。global full-model及PD Graph/U2 guard不变，
1.3/1.5/1.8/1.9/1.10仍pending。

上述自然完成/滑窗Query修复已以七路径提交并推送
780f9263d65fe8ab17edc0b53897b42178a754ae，独立ls-remote完整SHA一致，ahead/behind
为0/0。六overlay提交字节等于004实际快照，13项暂停路径仍保持；新增raw/identity/
环境/失败诊断只留本地。后续软件切片把同一Scheduler的真实NewRequestData交给同一
D owner的原生RequestState/BlockTables/add_requests/update/gather，不执行target。

原生worker接入002单轮prefix44+connector94+A51共189通过、0失败/错误/跳过。
真实NewRequestData在同一Downer进入native add/update/gather，public prompt/history/
computed/draft0、全部5组含SWA null的绝对块表、prefill分类与完整D页字节均检查。
native KVBlockZeroer真实构造/跨度与去重保留，仅复制和清零kernel为CPU叶子；新块ID
不含null0且与已传输前缀不相交，按AttentionSpec tensor精确跨度清零，tuple index/
scale与padding保留。首次001187P2F是__new__ fixture漏掉正常zeroer初始化，独立
诊断壳自身TypeError如实单列；锁定Ascend Worker1215-1221已正常接入此初始化，
没有据此修改上游或认定生产缺陷。116项安装源码/root锁定Git一致、138生产文件
全等780f9263基底、两个冻结/current/upload源一致；无device/service/install。
CPU memory/staging/zeroing叶子及实际Backend logical/kernel block size选择明确，
不执行input preparation/target/sample/proposal/F。完整模型/PD Graph-U2 guard
保持，1.3/1.5/1.8/1.9/1.10继续pending；后续软件组合需bind前一致target/draft构造。

该原生接入增量三路径已提交并推送58035a1e15f81cab6209d3c362013e6cc9819510，
root独立ls-remote完整SHA、ahead/behind0/0，以及两个提交overlay等于002实际快照
已核；13项暂停路径哈希不变。3.11AST/Ruff/格式、说明链接/Bash语法与staged
范围检查通过；新增raw/identity/环境/失败诊断未推送。继续开发bind前一致微型
target/draft+真实D缓存读取+native first-input/sample/proposal组合，不能称仅剩A5。

2026-11-11 首步组合软件增量：在schema/consumer注册前构造一致的三层target、
三aux、两draft层、N5微型CPU模型；各target层独立SWA并保留原生共享KV/index规则。
新增实际prepare_inputs/原生绝对块表gather和slot wrappers、真实EngramModelState
构造/prepare_attn/TargetDeviceMetadata.run_build/finish、真实NONE manager eager
dispatch，以及同一owner的execute(None)→原生Sampler/RejectionSampler→history→
DSpark context projection/forward/Markov的组合用例。模型/metadata/kernel/输出复制
为明确CPU叶子，无checkpoint/设备/远端F；独立BF16/FP32逐步参考检查hidden、3aux、
pre-head residual和完整D页。fresh block zeroer跨度oracle共用，实际Sampler输出
直方图按原生history kernel更新，首批draft为真实worker初始0值，不注入接受长度。

该新增组合尚未取得实际依赖回归结果：native-input001在SSH tempfile前返回255，
第二次只读SSH实际stderr为Connection closed by UNKNOWN port 65535；未上传、
未执行pytest、无XML，远端安装源码检查0项。冻结001快照及原始失败只留本地；
后续首步代码继续完成静态审查，不能把此前接收/接入两轮189P计作本轮通过。
本地6模块collect为0items/6skips（缺torch/numpy，工具返回exit1），仅证明依赖门可收集；
3.11AST/Ruff/格式/说明Bash语法、规划harness434files/8changes/102tasks/56objects/
upstreams unchanged分别记录。新增125项锁版Git identity是待实测门禁，未标安装匹配。
说明扩为8完整模块，环境恢复后补跑后再继续多轮调度/错误/设备组合。
完整模型和PD Graph/U2准入仍关闭，1.3/1.5/1.8/1.9/1.10/1.12保持pending；不称
开发完成或只剩A5。按用户持续策略，仅代码/测试/操作说明交付，新增证据不推送。

2026-11-12 A3恢复连接后继续DSpark优先。首步组合001八模块553P/6setupError；
002/003三模块分别187P/6setupError（逐层spec、padded页尺寸及旧fixture覆盖三层compress_ratios）；004为189P/4setupError（NONE manager必读capture-size）；005
为189P/4F（__new__ owner漏正常metadata计数）；006为191P/2F（draft eager装饰器初始化字段）。均保留原始快照和日志，不合并计数，也未据此修改生产或上游。
按锁定源码补齐测试夹具后，007三完整模块单轮193P/0F/E/S、125安装源码一致；
138生产Python仍全等b157684，实际target hidden/3aux/prehead和D整页参考通过。

随后同一PD请求第二轮组合001八完整模块单轮561P/0F/E/S、127安装源码一致。
实际首轮sample→Scheduler.update_from_output→take_draft_token_ids→update_draft→cached schedule→原生prepare/execute/sample/proposal，未复用手工SchedulerOutput。
无grammar的-1占位只给Scheduler定长度，真实输入从持久RequestState读取7..12。
独立oracle从第二轮前D页、当轮zero列表及完整绝对块表推导hidden/aux/prehead/整页；
实际首拒后history、computed/total、计数、position=N的新draftcontext及P源页不变均检查。
微型模型关闭Engram、CPU Attention/HC/MoE/greedy等叶子、无F/设备/服务/安装；
第二轮只覆盖首拒，Markov序列不作为draft Attention/HC数值验收。1.3/1.5/1.8/1.9/1.10/1.12仍pending，完整模型及PD Graph/U2门禁不变，后续仍需请求终结/取消/故障。

root独立解析561 XML、逐项核127安装/锁定Git源码、138生产base字节及四冻结/上传/current测试源；harness的head_commit取自外层仓库当时c169edf，实际archive是b157684，
二者只差两份A5操作说明，未据其head字段代替逐文件校验。13项暂停普通MTP路径哈希保持，Ruff/format/3.11AST、文档链接/Bash语法及规划harness通过，上游只读。

用户要求A5同步且精简：最小两场景说明已以c169edfcce3f43b38fc645d4539768ec0be6e7df推送并独立ls-remote确认。原生eager基线一个32token请求，再全新DSpark N1服务两个
相同32token请求，检查正常退出/资源回收；不扩N2/3、AFD/PD/Graph-U2。用户可并行手工执行，开发继续。新增raw/环境/identity不推送，普通MTP后续仍暂停。

本轮四测试/共享fake及验证说明五路径已提交并推送e65a0cc9f17151a99fd0e724adccfec4b5dcd738，独立ls-remote一致、ahead/behind为0/0。
提交四测试源与561P实际冻结/上传字节一致，13项暂停改动哈希不变，index为空。
新增raw/identity及本地路线图/tasks未推送；A5最小说明已在同分支，可同步更新验证。

2026-11-12 用户回传 A5 最小两场景结果：原生 eager 基线一个32-token请求、原生DSpark N=1两个顺序32-token请求均HTTP200/请求exit0，三个输出token IDs完全一致。两个DSpark请求各19步/draft19/accepted13，直方图[6,13]，接受率68.4211%、mean length1.68421；逐步数组和原始Prometheus增量独立复核一致。
两请求分别由DP engine7、3执行，不作为同一worker跨请求状态验证。

结果限定于实际A5 DP8/TP1/EP8、CPU共享Engram环境；Ascend实际HEAD沿用4ec6dc7722ec4f7f8005a86570238ac4377c5d43，与开发锁7e37e29不同。
DSpark采集15源码hash中13匹配，量化w4a8_mxfp4与DSpark speculator两个不同；
未改源码锁或上游。基线SIGINT后launcher130，有退出错误/残留worker快照，随后DSpark启动前八卡无NPU进程，但中间回收过程无记录；DSpark退出码/退出后快照未回传。原包、hash、离线核验及边界见[本地A5结果](evidence/a5-native-dspark-n1-20261112/README.md)。
仅记录原生短请求与真实proposal/接受拒绝通过，不勾选AFD/PD/Graph/U2、内部KV/Engram、退出、完整精度或性能任务，1.3/1.8/1.9/1.10/1.11继续pending。
本轮未运行设备/服务/安装，原包及新增证据仅本地留存，未提交或推送。

2026-11-12 继续开发：在既有真实 Scheduler→native target/sample/proposal 两步组合上新增完成/取消后的同 ID、同 slot、同 owner 本地新 prefill，以及 target 入口Python 异常后的 terminal 拒重试。新请求不做第二次远端 lease，Engram 关闭；不称在途 DMA 取消、mid-target 部分写入回滚或新进程恢复。001=561P4F，修正 penalties关闭时原生不重建未使用 histogram 的错误断言；002=890P2F，补齐故障用例共享的sampling CPU 初始化。003 十个完整模块单轮892P/0F-E-S、133项实际安装源码一致；
两轮原始失败保留，不拼接通过数。138个生产Python文件中仅feature_validation.py为显式资格准入增量，其余与交付e65a0cc archive逐字节一致。13项暂停普通MTP改动保持不变；本地CPU-only收集43P118skip仅证明无依赖可收集，不能替代A3结果。

完整V4.1默认仍拒绝；仅AFD_DSV41_EAGER_QUALIFICATION=1检查受控eager/U1候选：
V2/text/BF16、HCCL、F gate、A4/F4 TP4、DP/PP/CP1、EP/SP/Graph/DSpark/PD关闭，A侧CPU Engram且nonshared，F由get_npu_ffn_config提前隔离。复用实际rank的原有
HCCL校验，不修改loader/执行/通信协议。当前为资格配置代码，不提供可执行完整服务recipe或正式支持声明。候选预算和后续首个off请求→N1请求说明已准备。

已只读取得实际Ascend4ec6的量化/speculator源码，下载字节hash与用户上传的installed记录一致。前者含shared_weights与MegaMoE布局/恢复差异，后者含draft MoE私有配置/current-config差异；新增依赖ascend_forward_context.py和ops/fused_moe/dataclass/fused_experts.py需installed字节确认。新增stdlib四文件采集工具与最小说明，不启动服务/设备、不安装或改上游；既有三个原生请求不重跑。
退出收尾仍待用户补齐。1.3/1.5/1.8/1.9/1.10/1.11/1.12保持pending。

上述8项代码/测试/操作说明已提交并推送
3b277e20d20d173540ac0e9c3b32b0d8bb12f55f，独立ls-remote返回同完整SHA，ahead/behind为0/0。新增raw/identity、用户回传包及本地tasks未推送；暂停普通MTP 13路径不变。892P实际快照四个overlay与提交字节一致，默认完整模型拒绝不变。

2026-11-12 用户补充源码包已核验：量化、DSpark speculator、forward_context和fused_experts四份安装源码都符合已审查的实际Ascend4ec6版本，新增依赖身份闭合。
无新增退出/设备记录，不据此勾选NPU或完整模型任务；原始包及身份仅本地留存。

新增受控A5 AFD eager/U1启动器：默认stdlib源码/checkpoint预检，固定16份关键上游源码及config/index身份、48shard头结构；不读取tensor payload、不证明权重值。
F独立入口继承native SyncMPClient完成本地frontend握手，复用native进程管理；
避免独立--headless等待外部frontend。只覆盖A4/F4 TP4、DSpark/PD/Graph/U2 off。
先并发启动F/A，F接收循环+Ahealth后只发一个32token请求，再由A主进程SIGINT触发STOP，等待F自然退出。新增peerSTOP、engine及四rank cleanup日志协议；
late-loaded FFN准备路径同样初始化实例STOP状态，防止非HCCL路径AttributeError。
原生worker/engine teardown保持原owner，超时只回收自身PID/start-time身份进程，forced cleanup不记正常退出。操作说明已收敛为一个新增A5场景。

A3实际依赖CPU：001为SSH沙箱传输失败，未执行pytest；002完整14模块单轮1015P。
签名表新增monitor hook的一行lint注释修正后，003最终冻结单轮仍1015P、0F/E/S，138项安装源码逐项匹配锁定Git；140个生产文件仅4个声明的入口/诊断增量，其余均与3b277e2 archive一致。新F入口18case、签名39case，A/F worker 12/54case。
另13项预检文件fixture与8项Linux真实Popen/HTTP/SIGINT/owned cleanup行为通过；
后者用fixture子进程，不是模型/HCCL设备模拟验收。A3真实A/F完整argv parser通过，没有create_engine_config或启动服务/设备。13项暂停普通MTP路径哈希不变，上游只读。
1.3/1.5/1.8/1.9/1.10/1.11/1.12仍pending，等待对应实际完整模型与硬件证据。

本轮15项代码/测试/操作说明已提交并推送
700c0485b063cf13bdec93954c3222aa4ff844f4，独立ls-remote完全一致、ahead/behind0/0。
提交15个blob均与限定暂存内容一致，003实际CPU冻结的9项overlay与index一致；
13项暂停工作路径逐字节未改。新增采集原包、环境/identity、CPU日志与本地规划
记录均未推送。完整A5 AFD及其DSpark资格仍待手工实际验收。

2026-11-12 A5 首个完整AFD候选回传，archive SHA256 287e75b728d857f87c1c5e88ecc2102ebe09fdc41523cbc7e0f6fb6ac923b7ef，实际HEAD700c048且source干净、preflight通过。A/F同一根因：资格校验冗余拒绝is_multimodal_model，忽略native language-model-only仍保留MultiModalConfig。
删除该重复条件，复用已有HCCL text/encoder-decoder校验；新增A/F×语言通过、视觉拒绝、encoder-decoder拒绝6例，使用native ModelConfig属性与真实MultiModalConfig，仅CPU配置投影，不称完整checkpoint配置构造。默认关闭及其余候选限制保持。

A3最终冻结v41-text-guard-001完整14模块单轮1021P、0F/E/S，138项安装源码与锁定Git一致。140生产文件中仅feature_validation显式增量，其余均与700c048 archive一致。13项暂停普通MTP路径保持，上游只读。该A5失败没有请求、无peerSTOP验收；两角色exit1、forced_cleanup=false、无存活进程、后快照八卡idle。无需调整CANN/FastAPI/模型config或重跑原生三请求，修复交付后重试同一AFD off短请求。
原始包、raw/identity及CPU证据仅本地留存。完整模型与硬件相关task仍pending。

上述两项代码/测试与两份操作说明已提交并推送faba3a848b8eee05cbc3566791d94482e6f8d023，独立ls-remote一致、ahead/behind0/0。
4个提交blob与暂存冻结一致，实际1021P的两overlay与index一致，index为空；
13个暂停路径不变，新增raw/identity及本地tasks未推送。A5下一次仍执行同一受控AFD eager/U1 off单请求，不据CPU配置修复标记完整模型验收。

2026-11-12 后续 A5 包：faba3a8 / preflight通过 / 源码干净；A/F加载完成、F loop及Ahealth就绪，首请求HTTP500。实际ds41 mixed-quant Sparse MLA只接受64个Q头，
TP4本地16头被拒绝。新增实例级packed backend子类，16个Q/sink头补至64，native计算后裁回连续16头，原生KV/indices/metadata等待/sinkFP32转换及共享状态保留。
构造和qsmla签名与锁定源一致；独立架构审查通过，声明4倍Attention头计算及临时HBM成本。
五份新增依赖4ec6/7e37字节一致，实际安装只读预检扩至21份；没有更改开发锁。

A3冻结packed-attention-002，15模块单轮1035P、0F/E/S；140项上游源码匹配，141个插件Python文件为faba3a8+2项声明生产增量。另预检fixture13项、Linux真实监督/HTTP/信号fixture10项通过（probe先成功退出才启动F/A；失败/超时不启动服务）。
同一A5命令新增单卡小张量真实packed算子四例，通过并idle后才执行原AFD单请求；
本地/A3没有执行A5算子，不据CPU合同勾选任何设备/完整模型/正常退出任务。

本包失败后FORWARD已公告，F可阻塞recv，STOP idle-boundary保护不应放宽。
保留forced_cleanup=true、clean_shutdown=false、F=-9、A=0、无survivor、八卡idle的失败分类；优雅中途取消需独立双方协议。原始包和身份/CPU证据仅本地留存。

本轮10份代码/测试/设计与操作说明已提交推送401290370fefbc0581848d1b740fff77b8fc4b11；独立ls-remote同SHA、ahead/behind0/0。
10个提交blob逐一匹配限定暂存冻结，1035P的4项overlay与index一致，index为空。
原包、CPU/环境identity、tasks及暂停MTP路径均不在提交中。旧暂停baseline中11项仍匹配，AGENTS及workflow已有的priority文档差异保持；交付前13项快照提交后未改。
下一步用户在A5更新该分支，仍运行同一--run命令，回传统一tar.gz；真实算子、完整AFD短请求及正常退出待审阅，DSparkN1暂不据此开放。

2026-11-12 按刷新后的排期继续 C7 1.3/1.5/1.10 非 Engram 生命周期：补齐已注册DSpark PD owner 在原生 sample/proposal 失败后的 terminal 边界。target head 失败保留未推进历史；draft backbone 失败保留原生已推进历史/已写 context KV，原异常对象继续传播；下一次 execute 或已有待采样状态不得复用该 owner，无待采样状态仍返回 native None。无新增 HCCL/PD 消息、同步、回滚或重建，准入不变。

red-001 是沙箱 SSH 失败、未执行测试；red-002 为49P/20setupError，补齐共享scheduler fake 的真实 Ascend 配置导入绑定；red-003为63P/6F，其中2例draft外层hook未被原生compile入口调用，改在真实embedding边界注入。最终相同六例在未修复的red-004上63P/6F、无error/skip；修复后 sampling-003 的15完整CPU模块单轮1042P、0F/E/S，141安装上游源码与锁定Git一致，141生产Python中仅A runner一份变化。
原始失败均保留，不拼通过数。暂停普通MTP内容保留，签名表仅叠加本轮一行契约。
本机无依赖收集117P/51skip、Ruff/format及本地审计通过；无安装、设备或服务。

详见 [采样异常审查](evidence/dspark-pd-sampling-failure-review-20261112.md)和[逐项验证](evidence/dspark-pd-sampling-failure-validation-20261112.json)。当前只证明
CPU微型模型的同步异常边界；AsyncOutput返回后的错误、多rank监督、设备退出/重启、Engram history及完整矩阵继续待验。Engram专项仍后置，1.3/1.5/1.8–1.12保持pending。
本轮仅本地开发和隔离A3 CPU验证，未提交或推送，A5原待验范围不扩展。

2026-11-12 用户后续A5包22e0cb6...，archive SHA256ce12ce263dbf26673b836a76dd069d26bb31b55089406514be60bb000c4268bc。
实际HEAD4012903、21源预检通过且source干净。真实packed Attention四例通过，16头适配与native64保留输出最大差0；AFD首个32-token请求HTTP200、finish=length。
一次完整短请求可执行，不等于完整精度/DSpark/性能验收。A/F及probe均exit0、无forced cleanup/无survivors、八卡idle；peerSTOP、engine和launcher标记存在。
原summary缺四rank worker清理INFO，clean_shutdown=false，保持原始判定。

输入10个token与既有DP8/TP1/EP8原生基线一致，但输出自index0不同。AFD为TP4、
DP1/EPoff、CPU Engram非共享；原因未定，不归因为正常误差或某一模块，不勾选
输出等价/完整矩阵。下一步优先隔离差异，再开放AFD DSparkN1；现有包不用仅为
补日志重复跑同一场景。API EngineDead发生于请求成功及engine teardown后，
保留为退出链异常证据，不误记本次请求失败或无异常退出。

日志缺口修复仅把F worker logger接入vLLM已配置的INFO handler，原root WARNING
会丢INFO；不修改STOP/join/native shutdown或成功条件，不放宽launcher四rank检查。
测试取消被测logger强制INFO，在rootWARNING/nativeINFO下捕获已发日志，保留
STOP缺失/native失败禁止成功标记。最终同一fixture red-004为52P/2F，修复004为
54P、0F/E/S，141项安装源码匹配锁定Git。002/003为测试捕获层级/stream绑定问题，
实际修复stdout已出现标记，原失败记录保留，不与最终结果拼计。无设备/安装/服务。
仅准备交付本轮2项代码测试+2项操作说明；已有26项dirty开发保持，不混交普通MTP
或其他待交付C7/诊断改动。原包及全部CPU/source identity仅本地保存。

本轮日志代码/测试和两份操作说明共4路径已提交推送
f26d112654d696f07dc4dd1c7a13c6586efa0fcc，独立ls-remote一致、ahead/behind0/0。
4个commit blob匹配冻结，54P的2overlay与index一致，index为空；raw/identity/task
不在提交。初次push仅LFS可选lock verify网络失败，以单命令临时关闭验证重试成功，
没有改仓库配置或强推。26项既有dirty中25项哈希保持；另一个已有a3-failure采集
脚本在本轮期间另有工作区变更，保留其当前内容且不纳入交付。现有输出差异、
四rank实际清理标记及DSpark完整模型资格继续pending，本包不要求立即重跑。

2026-11-12 持续开发增量：原生 AsyncOutput 完成错误经 UniProc future/EngineCore
阻止 scheduler 成功输出，补齐 12 项成功/失败对照，保留已推进 history/draft/KV 与
完成 PD 消息；WorkerProc 仅验证实际 SUCCESS/FAILURE 响应组装，不称 MP 监督验收。
另补两项 finish/cancel 后同 ID 的第二次远端 PD：同注册 owner、新 source/destination
generation/lease、旧 GETREADY/DONE 隔离、第二次实际读字节与 D 恢复。001/002 CPU
各 1054P/2setupError，修正共享 CPU platform/PCP/TP 初始化后窄范围两项通过；原始失败保留。
详见 [异步/复用审查](evidence/dspark-output-failure-review-20261112.md) 与
[验证记录](evidence/dspark-output-failure-validation-20261112.json)。

A3 output-failure-003 在 F0/A2、各 TP1、两层/U2 真 HCCL parent 后注入 host 输出错误，
真实 A prepare_shutdown 与 F STOP/ACK 完成；全新 PID/通信组第二轮 parent/STOP 通过。
两轮 A/F exit0，无 forced cleanup，native TP 前后[1.0]、最终八卡 idle。
001 沙箱 SSH 未运行设备；002 测试 F runner 缺现有可选诊断字段、parent 前失败，A -9
受监督回收后八卡 idle；003 修正探针初始化，不是放宽生产 STOP 门禁。该证据不执行
AsyncOutput/模型，不能与 CPU 拼成完整模型恢复，也不证明 blocked recv/device fault 自愈。

H14 源码审查发现 native MP RPC 超时/部分失败后的迟到回复无 request ID；真实原生
executor/future/WorkerProc 确定性 CPU 复现 profile 超时后 status 错收[None]。
修复 F profiler/start/poll/STOP 的异常及中断后通道失效，禁止后续 collective，保留原异常，
交原生 executor 清理进程；两个 F 初始化入口均设置标志，失败不发布 clean marker。
最终 h14-terminal-002 的 16 完整 CPU 模块单轮1107P、0F/E/S，142安装源码逐项一致；
001为1106P/1F，仅旧测试仍期望失败后STOP，修正协议期望后重跑全量，生产字节不变。

A3 profiler-control-001 的 Unix socket→native UniProc→实际 F daemon→HCCL/profiler
通过；四个两层/U2 parent 和16次精确 MatMul oracle，两个 active parent trace/CSV
对齐8个实际AI_CORE MatMul kernel；owner稳定且与管理线程区分，start/stop pending
在后续parent边界应用。A/F exit0、无forced cleanup、nativeTP前后正常、八卡idle。
143安装源码（含已核验torch-npu profiler）通过，95trace文件哈希一致；初次Windows
长路径解包失败，完整116文件归档已取回，无重复设备运行。
见 [H14审查](evidence/h14-rpc-terminal-review-20261112.md) 和
[逐项验证](evidence/h14-rpc-terminal-validation-20261112.json)。
H14 EngineCore/MP IPC设备组合、第二次profiler restart及active最终STOP同步导出与
五秒join仍待验；1.7及1.3/1.5/1.8–1.12继续pending。Engram仍后置，旧MTP保留暂停，
A5仍沿用一个packed算子+AFD eager/U1 off请求/STOP单命令；无提交或推送。

后续 A3 profiler-restart-001 在同一 F owner 完成两周期与 active 最终 STOP：六个
两层/U2 parent、24 次精确 MatMul 检查、累计 steps5；两个 prefix 各8个实际
AI_CORE MatMul 事件且CSV匹配。STOP前第二prefix无trace_view、仍requested/applied，
STOP后真实owner.finish导出；190trace文件哈希、143安装源码通过。A/F exit0、无kill、
nativeTP前后[1.0]、八卡idle。owner.finish=3.7497s，stop RPC=4.1041s，A ACK=0.00154s，
本次未超现有5sjoin；不由单次组件结果保证其他负载耗时，不新增运行时等待策略。
详见 [重启与STOP证据](evidence/h14-profiler-restart-validation-20261112.json)。
上述此前待验的重启/activeSTOP仅在该UniProc设备组件范围闭合；H14 EngineCore/MP IPC、
完整模型及性能仍待验，1.7及其他完整任务保持pending。最终CPU仍为同一1107P单轮，
探针扩展未改变三份已验证生产overlay。最多两个子agent执行审查/探针，未提交或推送。

2026-11-12 后续 H14 真实 MP 组件 `profiler-mp-restart-003` 通过：Unix 控制进入
native MessageQueue/MultiprocExecutor/WorkerProc，独立 F device PID 内的 RPC 线程
与 profiler owner 线程区分；237 次分发、两端完整 shm handles 一致。六个两层/U2
parent、24 次精确 MatMul、两周期累计 steps5；两个 prefix 各8个真实 AI_CORE kernel
并与CSV一致，第二trace在最终STOP后由owner.finish导出。145份安装源码、190份trace
哈希及原始归档核验；A/F和独立device子进程exit0，无TERM/KILL、无forced cleanup、
nativeTP前后[1.0]且八卡idle。owner.finish=3.7005s、stopRPC=4.0637s，均为本次host观测。
原生 executor 显式30s进程退出宽限下自然退出；001原生响应环240MiB超/dev/shm的52MiB
空闲，002默认5s宽限触发子进程SIGTERM，均保留失败，不追认通过。003显式两端真实
MQ为64KiB×10；不覆盖原生默认容量/队列initializer、native构造/worker.shutdown、
EngineCore或完整模型/Engram/Graph/性能。共享探针仅新增可选入口与PID合同；三份
已验证生产overlay字节不变。详见[MP审查](evidence/h14-multiproc-review-20261112.md)
和[原包逐项审计](evidence/h14-multiproc-validation-20261112.json)。1.7继续pending；
A5不增加profiler场景，优先单层真实MoE nativeTP1/TP4隔离首token输出差异。

2026-11-12 单层 MoE 定位包已准备：保留原生40层配置，只构造/加载第0层实际
MoE 的2312项checkpoint tensor（7,258,804,992字节），顺序 native TP1/TP4；
相同确定性输入、gate/output完整张量、各rank和25项安装源码原文均归档。
未改Engram、未加载AFD插件，不预设数值阈值，也不把单层结果当作模型等价。
A3 setup-003 实际包导入及两种TP配置均通过，未set_device、建通信组或读权重；
先前setup-002原生导入循环失败保留。最终wrapper-004仅重复无设备的元数据/打包
检查并通过；两轮前后八卡idle。详见[准备审查](evidence/moe-parity-preparation-review-20261112.md)
及[setup原包审计](evidence/moe-parity-setup-validation-20261112.json)。
独立150文件包基于f26d112、仅显式加入诊断依赖，不混入暂停MTP或待验H14 overlay；
包SHA256为5a24753b232b663e3a0d5b8c58233feec0480e02be5fcbc71cf27d4355ab334d。
已告知用户按[一次A5操作](../../../docs/npu/DSV41_A5_MOE_PARITY_ZH.md)运行并回传原包，
不重跑此前原生/完整AFD请求。A5结果及完整输出等价继续pending，AFD DSpark N1不开放；
独立A3 EngineCore控制循环验证继续推进，不因手工A5等待而停止开发。

后续 `profiler-engine-restart-002` 的真实 EngineCore busy-loop/input handler/
finally shutdown 与 MP IPC/独立F设备进程组合通过；WAKEUP被消费、peerSTOP切换
SHUTTING_DOWN、socket移除、两次幂等STOP、engine cleanup和native child自然退出。
六个两层/U2 parent、24次精确MatMul、双prefix各8kernel/CSV，152安装源码和170trace
哈希核验，A/F/device均0、无强制回收且八卡idle。生产三份overlay仍匹配1107P。
001在native executor/device退出后因manager fixture漏原生平台hook，真实cleanup触发
NPU allocator断言，保留失败；002只补EngineArgs原有pre_register_and_update，使用
已锁定Ascend allocator适配，不改生产、不吞清理错误。见[审查](evidence/h14-engine-core-review-20261112.md)
和[审计](evidence/h14-engine-core-validation-20261112.json)。显式30s进程宽限和64KiB×10
MQ仍为组件条件；构造/前端/信号、REQUESTED、UTILITY失败及完整模型/性能未验收，
1.7不勾选。A5仍只等待已经准备的单层MoE对照，不增加H14手工场景；继续1.6部署预检。

部署前[recipe planner](../../../tools/diagnostics/plan_dsv41_hccl_recipe.py)已实现：三层
capture预测与whole-parent覆盖、真实CPU拓扑mapping、八个开关raw/configured/有前提
phase投影，显式0保留；不启动、不修改准入。禁site20个CLI边界单轮通过并保留102项
原始产物哈希；A3 native-002的13组实际原生CPU配置/候选/切分方法对照也通过，152源码
匹配、11方法文件逐项比锁定Git、八卡前后idle、无设备执行。native窄候选与recipe完整
覆盖策略明确分开，见[审查](evidence/recipe-plan-review-20261112.md)、[本地](evidence/recipe-plan-local-validation-20261112.json)
和[原生对照](evidence/recipe-plan-native-validation-20261112.json)。真实启动recipe/eager自动
owner装配及逐rank生效值尚缺，1.6仍pending。下一步先补两侧eager family HELLO一致性，
避免A warmup/F serial receive错位；不提前开启完整模型/U2 guard，不进入新增Engram适配。

2026-11-13 eager HELLO 一致性增量：新增可省略的 eager_pipeline_version，0保留既有
v3字节、1要求pairedTP1/2+U2；peer必须精确匹配，不min降级。A3 hello-002的三个
完整CPU模块单轮484P、0F/E/S，152安装源码匹配。真实四rank/A-F各TP2的单F0不一致
全员在Gloo HELLO拒绝，无数据HCCL组/warmup/parent；一致组双stage真实warmup、
串行受控U1/U2算术与STOP成功。全部exit0、无kill、关闭后nativeTP[3.0]、两轮前后
八卡idle；001沙箱SSH未执行远端工作。见[审查](evidence/eager-family-review-20261113.md)
及[原包审计](evidence/eager-family-validation-20261113.json)。自动bootstrap、真实ticket/
模型/Engram/性能未由此验收；继续两阶段装配，1.6及完整矩阵不勾选。

用户授权的A5单层MoE验证包和操作说明共两路径已提交推送
501f9d3c4389d797109db65c619a15c1c8b95717，独立ls-remote核对一致、ahead/behind0/0。
150文件/447350字节包及逐文件manifest核验，包SHA仍为5a24753b...334d；说明新增在
仓库根git pull后直接解包运行。原始A3证据和其他dirty开发未混入该交付；A5结果仍待回传。

2026-11-13 A5 单层 MoE 回传在源码采集时 exit1：旧定位逻辑未覆盖 editable
Ascend 的 import finder，报缺少 `vllm_ascend/models/deepseek_v41/model.py`。
尚未启动TP阶段或载权/计算；无强制回收、无owned残留，未产生数值结果。
按用户要求改为仓库直接维护并运行探针/入口，顶层 import spec 定位实际源码，
不执行包初始化；补记三份探针依赖SHA。CPU七项源码/manifest边界、两入口help、
Ruff和本地harness通过，上游只读状态保持。见[失败与修复记录](evidence/a5-moe-parity-source-failure-20261113.json)。
本次修复不表示A5载权、FP4算子、TP collective或模型输出等价通过；单层数值与
完整模型仍pending，AFD DSpark N1门禁不变。仅诊断源码与操作说明纳入本次交付。
上述三路径已提交推送至 `feat/dsv41-flash-hccl-v030@ed19f38ef964f08b4ee090a1c4d3bf5e330c1b3d`，
独立 `git ls-remote` SHA一致、ahead/behind 0/0；原有其他dirty工作与本地证据未混入。

2026-11-13 后续A5 TP1已通过before检查并完成原生单rank组初始化，但在MoE构造时
通用 `RoutedExperts.__init__` 拒绝 `n_shared_experts`，尚未载权。根因为探针未执行
原生worker的 `register_ascend_customop(config)`，只导入ops不能完成OOT类注册。
探针现补齐原生dummy/customop注册，配置阶段记录并检查实际分发类；源码归档增加
custom_op/worker，构造后记录真实runner/experts类。A3现有安装包通过真实注册/配置
边界验证：首次注册前参数拒绝、注册后AscendRoutedExperts接受，AscendMoERunner
选择正确，TP4配置幂等注册通过；三份实际类源码SHA与锁定Git一致。无临时源码树、
无组/载权/核执行；原始[日志与审计](evidence/moe-registry-validation-20261113.json)保留。
Ruff/CLI help/本地harness通过，上游不变。此前一次guard因只读git status超时失败，
后续完整check成功，未更改上游。A5完整MoE与数值仍pending；只交付探针和说明两路径。
两路径已推送 `6f54c1d21d67cffe3a70019882227f36d59d0749`，独立ls-remote核对一致、
ahead/behind 0/0，其他dirty工作及本地证据保留。

2026-11-13 A5后续已返回load_layer并进入MoE forward，普通sqrtsoftplus原生路由
报 `_C_ascend.moe_gating_top_k_hash` 未注册。此前OOT类注册不包含编译扩展加载。
探针现在在set_device后、通信组/载权前调用原生kernel路径bootstrap并导入
`vllm_ascend_C`，记录二进制路径/大小/SHA和schema，检查16个实际调用参数及
PrivateUse1实现；保留原异常，不改路由、权重、format政策或Engram。
锁定router调用与C++绑定参数静态对齐；A3现有环境find_spec返回扩展缺失，未绑定
设备、导入扩展或执行算子，不能验证正向加载。见[源码与扩展检查](evidence/moe-extension-review-20261113.json)。
Ruff、两个CLI help、metadata无运行时导入及本地harness通过，上游不变；A5扩展
是否存在、设备执行与TP1/TP4数值仍pending。本次只交付探针与说明两路径。
已推送 `e2f0471b5bbdf7a29844118ead4354d1501ae27a`，独立ls-remote核对SHA一致、
ahead/behind 0/0，两路径工作区干净，其他本地工作保留。

2026-11-13 eager两阶段装配已实现并审查：正常A/F创建connector前一次settings快照，
HELLO family后prepare；A preprofile串行、真实KV返回后同session绑定；F daemon前
prepare/bind，actualexecute覆盖receive/target/profile/STOP的native与TLS父流并恢复。
失败terminal保留资源，不发STOP/销毁PG；成功nativecleanup后释放引用。旧manual
family0及Graph独立owner保留，无Engram/完整模型guard改动，签名AST六项对齐。
见[装配审查](evidence/eager-bootstrap-review-20261113.md)。

A3 integration-004的11完整CPU模块单轮728P、0F/E/S，152源码一致；002/003分别
719P9F、727P1F，只修新夹具初始化时机、STOP_ACK和rank，production三份字节未变。
[CPU原包审计](evidence/eager-integration-validation-20261113.json)含冻结源码/原始失败。
独立bootstrap-006的真实TP1/TP2 connector→HELLO1→warmup→串行profile→受控KV
callback→同session六U2 parent/ticket/diagnostics→STOP成功；每rank16个精确payload
检查，2/4rank全部exit0/no kill/nativeTP仍可用，两组前后8卡idle。四模块202P、152
源码和[设备原包](evidence/eager-bootstrap-validation-20261113.json)独立记录。004/005
配置探针错误在建组前失败并自然退出，后只改探针；失败记录保留。不把受控KVcallback
和CPU自动入口拼成真实模型/原生KV验收，1.6及完整矩阵仍pending。继续真实F设备
consumer/daemon集成，A5仍只等待单层MoE结果，无新增手工场景；本轮runtime未提交。

后续 A3 ffn-002 的 actual F runner/consumer 独立设备线程也通过TP1/TP2：管理线程准备session，服务线程刻意使用不同ambient，真实execute/Ascend forwardcontext消费六parent和第七次STOP；每F rank16次完整BF16/原FP32/rawID/output/context/native与TLScompute stream精确、compute milestones各16，七次ambient恢复、管理stream保持。
全部2/4进程exit0、无kill，nativeTP前后健康，两组前后八卡idle。152安装源码和source/result归档核验，production与728P完全匹配，不重复CPU计数。001为沙箱SSH无远端执行。见[原包审计](evidence/eager-ffn-validation-20261113.json)和装配审查补充。
受控thread不等于真实workerdaemon/startup/shutdown；无model/MoE/nativeKV/Engram/Graph/性能验收。1.6及完整任务继续pending，下一步补齐recipe和逐rank配置观测。

2026-11-13 启动逐rank配置记录已实现并审查：正常staged A/F在HELLO和prepare/bind成功后经既有native INFO记录同一冻结快照的八项configured和协商rank/device/geometry，
两个诊断开关关闭时仍输出；显式0保留、manual None未知、环境后改不重新读取。
记录在A native KV/F Thread.start之前，不作model-ready或actual overlap证明。
独立configuration-002的11完整CPU模块单轮763P、0F/E/S，154安装源码匹配；新增24导出/11日志边界由同批验证，source/result归档逐项核验。见[审查补充](evidence/eager-bootstrap-review-20261113.md)
和[原包审计](evidence/eager-configuration-validation-20261113.json)。001只有本地source归档，无远端身份/结果，不计入；日志增量未重复NPU流水线，旧设备结果保留各自源码身份。
旧P2p U1 launcher、实际staged recipe和逐rank配置/实际执行汇总仍缺，1.6不勾选。
本轮runtime未提交，A5仍只需已有单层MoE对照，不追加手工场景。

2026-11-13 A5 设备绑定报 userDevId=2 越界、有效范围 [0,1)；用户确认 shell 为`ASCEND_RT_VISIBLE_DEVICES=0`，继承单卡限制不能完成 TP4。
说明现显式为原命令设置四卡可见；--devices 明确为进程内可见编号，探针在 TP1
绑定/建组/载权前检查全部四个编号与实际 device_count 并记录选择，入口记录继承值。
setup-only 保持不查询设备数量。Ruff/format、两个 CLI 无运行时导入、diff check 和
本地 harness 通过，上游不变。见[设备可见性记录](evidence/moe-device-visibility-review-20261113.json)。
当时修正环境后的设备绑定、编译扩展和 MoE 数值仍待原场景回传。
三个诊断路径已提交推送 `bca7ba9f4157c16b5de9ca41d5420f81d71f0761`，其他 dirty 开发与本地证据未混入。

2026-11-13 A5 单层原生 MoE TP1/TP4 已回传 summary 与精简 result，实际执行不是 setup-only，五个 rank 全部 exit0、107.51 秒、前后八卡 idle，无强制清理或残留进程。探针/入口 SHA 与实际记录提交 e2f0471 一致；八项 gate logits 比较完全相等，MoE 输出 prefill/decode 最大绝对误差均 0.25，相对 L2 误差分别0.436075% / 0.449229%。TP4 rank 输出一致等约束由成功探针路径检查；没有独立读取完整 rank 报告或原始二进制重算。见[原始 JSON 与审计](evidence/a5-moe-parity-20261113/README.md)。
本次闭合该组件的执行回传等待，未预设容忍度，不等于数值等价验收；实际 top-k、shared/routed expert 与归约误差来源、此前完整模型首 token 差异仍未定位。
完整 AFD/Engram/性能与整项验收继续 pending；未新增 A5 场景或修改运行时代码。

2026-11-13 用户回传后继续1.6：新增planner→A/F冻结启动清单工具及recipe编制示例，
两角色argv/cwd/设备拓扑、八项规范0/1、输入/源码SHA和完整AFD/native/API端口区间
均明确记录。全0保留FULL_DECODE_ONLY/U2，传raw buckets/native max，不误传descriptor；
qualification=0不绕过guard，FlashComm1环境显式0防止native OR覆盖SP off意图。
按真实standalone F合同拒绝DSpark/PD、跨host和多TP world；其余native defaults未
复刻旧A5资格配方。独立审查无阻断，工具最终SHA8d0c5a2b...df13。
A3 native-003单轮37个-S CLI场景、12份成功清单/24次实际native A/F解析全部通过，
158安装源码与锁定对象一致、八卡前后idle；source/result归档和13源身份逐项核验。
本地CLI004的37场景单列，不累加；001本地身份准备失败、002沙箱SSH未远端执行，
旧本地001的raw/normalized断言修正只改harness，失败保留。见[审查](evidence/recipe-launch-review-20261113.md)
与[原包审计](evidence/recipe-launch-validation-20261113.json)。未创建engine/model/组或
执行NPU算子，编制成功不代表运行准入；实际启动、完整staged rank配置/actual事件
汇总和旧U1观测仍缺，1.6不勾选。本轮无新增A5手工场景，后续归因优先固定同TP4
的native MoE与实际AFD F构造/载权/context对照，再决定是否加入传输。

2026-11-13 继续1.6旧U1观测：正常DSV41 HCCL A/F一次冻结八项配置，U1 eager按
既有解析保持false；factory继续实际registry，仅实际P2p实例初始化前绑定非空快照。
手工None未知，其他model/connector及显式F注入不解析环境。A/F在成功HELLO及
prepare/bind后经native INFO单次记录，包括已初始化F；legacy使用真实双peer
world_size2/F0-A1与单列TP pair_index，不改wire/STOP/streams/组所有权或门禁。
独立审查及3个锁定Ascend入口签名审计无阻断；新增41 connector/factory及17 A/F
展开case。A3 legacy-002同批12完整CPU模块842P、0F/E/S，154安装源码匹配；
source/result原包、所有overlay与当前字节逐项核验，见[审查](evidence/legacy-configuration-review-20261113.md)
与[原包审计](evidence/legacy-configuration-validation-20261113.json)。首轮legacy-001
834P/8F是两个新fixture缺registration身份及6线程误入set_device(cpu)，只修两fake，
9份生产overlay逐字节一致，失败原包保留。不把合成selector配置或CPU thread/STOP
当真实模型/设备验收；日志早于KV/Thread.start，不是model-ready/actual overlap。
清单工具后续SHA d3c4fbd1...babb补齐18份插件源身份，A3 native-004重新37个CLI场景
及12清单/24次native A/F参数解析通过、158安装源码匹配、八卡前后idle；与历史
native-003分开归档，见[最新清单审计](evidence/recipe-launch-legacy-validation-20261113.json)。
本地CLI005的37场景不与远端相加。C7 1.6仍不勾选：继续完整逐rank配置/actual
parent事件汇总及实际启动，完整模型/Engram/Graph-U2/DSpark-PD组合边界不变。
本轮无新增A5手工场景，普通旧V4 MTP后续继续暂停。

2026-11-13 持续收口部署观测和 PD 异常队列：新增标准库逐 rank 汇总 CLI，
读取冻结 recipe 和原始 SHA 绑定的独立 rank 文件，校验八项配置、协商坐标与
显式 visible namespace；旧 U1 pair、Graph 配置与 eager/U2 actual 范围分开。
只有显式 expected parent 库存才验证 live eager/U2 paired TP1/TP2 的首末层
事件、空 stage、post/loan/compute/reuse/finish 因果关系以及跨坐标的 accepted
posting/bank reuse 顺序。最终 CLI005 单轮 52 场景通过；早期40/44/50场景和
harness中断产物保留，不累加。真实 A3 observe-001 的 TP1/TP2 原始 startup/
stage 导出和最终 v4 离线汇总均通过，每 rank 六 parent、16次精确 payload检查，
A146/F178条 stage records，全部2/4进程exit0、正常STOP、无kill、nativeTP
关闭前后健康、两组前后八卡idle，154安装源码一致。九份生产 overlay 与此前842P
字节匹配，不重复该CPU批次；本轮设备组件无model/MoE/nativeKV构造，prepared
argv未启动。见[观测审查](evidence/hccl-observations-review-20261113.md)和
[最终原包审计](evidence/hccl-observations-005-validation-20261113.json)。
记录没有session/PID/source绑定，caller归属和SHA不能识别同内容跨run；host
事件不证明Graph replay、设备完成、物理重叠或完整模型。1.6及完整矩阵仍pending。

同轮修复 native PD peer dispatch 捕获异常后继续执行已排队请求的问题：在既有
receiver task-entry锁内检查sticky failure，早于DSpark GETREADY/lease/read/
finished发布；正常关闭admission仍drain已接受任务，不新增abort/rollback或清理。
RED实际native peer queue三场景1P2F复现第二目标错误完成；GREEN六完整CPU模块
单轮333P、0F/E/S，154安装源码一致，锁定原生_handle_request签名精确匹配，
source/result归档与当前三文件逐字节核验、独立审查无阻断。见
[排队失败审查](evidence/pd-retained-failure-review-20261113.md)和
[RED/GREEN原包审计](evidence/pd-retained-failure-validation-20261113.json)。
失败页/lease保留，其他peer已穿过入口的read仍需进程监督；无真实DMA/NPU故障
恢复或完整DSpark/PD/Engram验收，1.5/1.10不勾选。两批不同范围结果不相加；
109任务仍52已完成/57待完成。该批次未新增A5场景，普通旧V4 MTP后续继续暂停。

2026-11-13 后续补齐正常 F Graph 服务的真实设备链：
`create_npu_runtime_connector` → worker transport binding/HELLO → 实际 service
factory/bind → daemon → runner → Graph service。A3 service-002 五轴00000/11111
各完成每rank23个parent（4warmup/4capture/12replay/3布局eager回退），F使用默认
NPUGraph/global pool、真实native forward context/MC2 mask、稳定bank与实际U2
compute pipeline；A使用原生基础manager和独立算术oracle。STOP后daemon join、
cache早于groups释放、nativeTP仍健康，两组全部exit0、无kill且前后八卡idle。
154安装源码和374份source manifest核验通过；九份生产overlay与此前842P一致，
该CPU批次未重跑。001仅SSH255、未启动设备。见
[服务审查](evidence/graph-ffn-service-review-20261113.md)和
[原包审计](evidence/graph-ffn-service-validation-20261113.json)。
该组件跳过worker/runner重型构造，MoE为受控算术，A端不是正常完整模型入口；
不含真实模型、nativeKV/Engram/DSpark或性能，1.3/1.6/1.8/1.10仍pending。

数值定位另新增[同TP4 native/AFD F单层场景](../../../docs/npu/DSV41_A5_AFD_MOE_PARITY_ZH.md)：
固定TP4，使用真实F层构造/载权过滤入口/原生FP4后处理/计算边界，包含同权重控制和
独立FP32路由输入。A3已通过两阶段导入、注册与配置，实际数值仍需A5；不重复此前
原生TP1/TP4，也不把constructor-free单层组件称为完整F构造/runner/HCCL验收。

该诊断的[setup审查](evidence/moe-afd-review-20261113.md)已独立核对四套各38项
实际归档源码（27原生Git对象、11插件/诊断源），另有154项远端安装源码检查；
两阶段各仅一个rank0配置进程，不是TP4collective或FP4验收。独立包151文件中
149份Python与setup冻结字节一致，见[包审计](evidence/moe-afd-delivery-validation-20261113.json)。

2026-11-13 Graph异常生命周期组件也已闭合该独立范围：failure-001先通过20个
parent（4warmup/4capture/12replay），首个真实布局miss的eager算术回调抛错后，
两次poll及public stop持续传播同一原cause；cache/transport terminal且不关闭，
bank保留，无ACK/正常group清理。F实际exit1，对端A由inner监督SIGTERM后
SIGKILL（exit-9），不记为正常清理或设备自愈；A失败JSON缺失如实保留。
随后全新进程23parent正常完成并退出，两组前后八卡idle。源码/结果包、154安装
检查及新旧PID身份已核验，见[故障审查](evidence/graph-ffn-failure-review-20261113.md)。
该轮仍使用poll修复前的独立冻结runtime，不反推后续源改动已通过同一设备运行。

随后修复实际F poll提前宣称STOP结束的问题：STOP_ACK后owner仍同步finish/export时
返回True，只有owner退出且再次核对retained error后才返回False；原cause与直接stop
5秒保护保留。确定性CPU RED为2F，GREEN五完整模块单轮222P/0F/E/S，154安装
源码检查通过。真实A3 EngineCore→nativeMP RPC→F daemon→HCCL/profiler两周期
组件也通过，最终owner导出7.03秒、随后stop RPC0.26秒，pollFalse时closed=true且
thread已退出；全部进程正常退出、无kill且八卡idle。见
[STOP/export审查](evidence/profiler-stop-poll-review-20261113.md)。原生MP进程退出宽限
显式30秒与worker5秒join不同，不把历史MP002的退出宽限失败当作本轮RED；CPU、
设备组件和历史结果不相加，1.7/1.10完整模型与联合任务继续pending。

2026-11-13 继续 DSpark/PD 组合接口：合法 no-PD/U2 P 产物到独立合法 U1/PD
consumer 的数据合同，两定向用例及九完整 CPU 模块单轮 561P、0F/E/S；真实请求
2/5/6 跨 6+7 stage，原始 parent/三层 aux、每请求投影地址、typed read/DONE 及
D 实际 sample/proposal 已核验。前期夹具/预期失败均保留。完整回归另发现并修复
手工 staged transport 绑定被配置日志破坏的问题，原 negotiation 测试未改变。
见[数据合同审查](evidence/prefill-u2-pd-data-contract-review-20261113.md)。
该 CPU Attention leaf、独立配置组件及预先登记不代表正常 U2+PD 准入。

正常异步 P 后续另发现 source 仅在 deferred start_load 登记，首次完整 prefix
却已在 execute 返回前发布 Ready。真实 ActiveKVConnector RED 五例4F/1P，
最小 bind 提前登记修复后相同五例5P：晚到 post 不复活 DONE/取消，同对象 source/cancel
修改拒绝，native 延迟 load 不变。见[异步 P 审查](evidence/async-p-admission-review-20261113.md)。
完整模块/bind/shutdown/签名验证为 whole-001 571P/1F 后，修正缺字段夹具的
整个 KV 模块 kv-001 109P 通过；全部生产和其他夹具一致，批次不拼接为单轮全绿。
后续发现 P sample/proposal 在当前 Ready 后仍会写注册 draft KV，随后完成
[最终 projection/Ready 后移](evidence/p-ready-frontier-review-20261113.md)：
两真实 RED、五项定向 GREEN；pending 重入/关闭和原始 proposal 失败/retry 拒绝一并通过。
另修复[单 token 输出上限](evidence/p-single-token-review-20261113.md)与
[最终 SWA 回收](evidence/p-final-retirement-review-20261113.md)，保留 source/export
严格比较及现有 lease/DONE/cancel 释放；不是把已释放页重新登记为安全。
同 owner P/U1/U2 的[配置与原生执行 CPU 组件](evidence/p-producer-u2-review-20261113.md)
已通过定向；十完整 CPU 模块及三项签名合并单轮 **721P、0F/E/S**，
见[合并审计](evidence/p-quiescence-combined-validation-20261113.json)。首轮 719P/2F
仅两项新增负例的原生调度/整页边界预期错误，修正后全部生产及 fixture 字节不变。
2026-11-14 后续[两真实进程最终报告验证](evidence/p-report-gloo-review-20261114.md)
通过实际 CPU Gloo 正常和单 rank draft 失败两场景：原始 metadata 绑定前广播，
实际报告向量交换，错误两侧 terminal/无 Ready/拒绝重试；Gloo 正常销毁不代表
失败 owner 恢复或模型/KV 清理。三个受共享 fixture 改动影响的完整 CPU 模块
另轮 276P、0F/E/S。源包与 721P 的 143 份生产字节一致，四处历史本地差异保留
并明确排除于此次冻结执行。当前继续 D 侧 DSpark/PD eager/U2 接入；1.3/1.5/1.10
的设备/完整模型组合仍未完成，Graph/Engram 或模型资格不因此开放，不增加 A5 手工场景。

同日[D/U2 接入](evidence/pd-consumer-u2-design-20261114.md)完成配置/FFN 隔离的
六新 D 正例 RED→GREEN 143P，实际同 owner handoff→target/sample/propose 两轮
及 draft 失败定向 3P；十完整 CPU 模块及三签名同轮 736P、0F/E/S。短 prompt 首轮
Q1/U1 fallback 后下一 cached 轮进入 Q3/U2，非短 prompt 两轮均真实 split1+2；
第二轮实际 RejectionSampler 在声明的 CPU logits 下接受一 draft 后拒绝下一，
history、aux、KV 及原 cause 保留均核对。前两轮错误测试预期保留，生产未为此修改。
CPU Attention 是有限 prepared-prefix 算术，不宣称 U1/U2 causal DSA 或 checkpoint
数值等价。继续[失败 PD owner 的 worker 释放前检查](evidence/pd-failed-shutdown-design-20261114.md)，
1.3/1.5/1.10 完整设备/Engram/模型验收继续 pending。

2026-11-14 [失败 PD owner 释放前检查](evidence/pd-failed-shutdown-design-20261114.md)
已完成 U1/U2 两项真实 RED→GREEN，公开 worker 在 native KV retirement 和 STOP
之前拒绝 terminal consumer；P pending 错误优先级保持。正常 U1/U2 和部分构造
对照均通过，十一完整 CPU 模块及四项签名同轮 **755P、0F/E/S**。RED001 两项 U1
setup 错误保留；RED002 与 GREEN 的 tests/fakes 全字节相同，仅 runner/worker
两处生产变化。独立原包审计和签名/生命周期审查通过，不宣称物理页释放、NPU
故障恢复或完整模型验收。继续 Graph/PD 组合；1.3/1.5/1.10 保持 pending。

2026-11-14 [F 服务 query3 Graph 设备组件](evidence/graph-ffn-q3-service-design-20261114.md)
完成：正常 F factory→daemon→runner→default NPUGraph，query3 每 rank 24 个父事务，
同快照 query1 回归 23 个。分别 4 warmup / 4 capture / 13 或 12 replay / 3 eager；
query3 包含 U2 4/5 行 capture、同 key valid 4/2 短 batch 和请求布局 miss 回退。
原 checkpoint 的真实 native DSpark 元数据先独立验证，另声明两层受控组件元数据，
相同 shape3 在隔离前保留到实际 F HELLO/key/bank/context。155 项安装源码和
143 production 文件闭包核对通过；两次正常 STOP、无内外强制清理、八卡前后空闲。
首轮 q3-001 的新增 eager observer 字段错误及内部监督清理保留，一行探针修复后通过，
无 production 改动。A 是 base manager/受控算术，不代表 native A/PD/KV、实际
DSpark proposal/接受率、Engram、完整模型或性能验收；1.3/1.6/1.8 保持 pending。

2026-11-14 [D Graph/PD 组合接入](evidence/pd-consumer-graph-design-20261114.md)
完成本轮独立 CPU 范围：完整 PD/Graph 三入口一致收窄为 exact plugin DSpark D、
FULL_DECODE_ONLY、eager fixed draft、stage_serial；P 保持 eager，原拓扑/缓存/休眠/
权重/offload 限制保留。配置 RED 40P/29F → 同测试 GREEN 69P；最终执行组件
cpu-012 3P（U1/U2 handoff→首轮 eager→cached FULL/native sampler/proposal，U2
FULL 异常原 cause 与 owner 保留）。同快照 13 完整模块及 4 签名单轮 1057P/0F/E/S。
预录 Graph bundle/输出及带 NPU 标签的 CPU allocator 是声明边界，不证明 NPU
准入、同一 PD owner 启动 capture、设备图 KV 数值或完整模型。失败不承诺 KV 回滚，
只保留失败时原生准备后的数据并拒绝重试/shutdown。完整1.3/1.5/1.10继续pending；
当前继续 U2 FULL 元数据和真实 A capture/replay，不新增 A5 手工场景。

2026-11-14 [U2 FULL 压缩缓存元数据修复](evidence/u2-full-metadata-design-20261114.md)：
插件 NPU UBatchRunner 的原生 build_attn_metadata 调用补回
`full_graph_mode=cudagraph_mode == CUDAGraphMode.FULL`，与锁定原生 ModelState
合同对齐；不把 for_capture 当作 live FULL 标记。真实 native V4.1 builder 经公开
U2 prepare，index_k/long_kv × capture/live FULL/eager 六场景 RED 4P/2F →
同测试 GREEN 6P。结合最终 Graph/PD 双向协议断言及两个通用 NPU 模块，十五完整
模块和四项签名单轮 **1223P/0F/E/S**，154 项安装源码与 143 production 闭包核对。
仅一行 production 变化，无新增 wire/流/同步/集合通信；这是 CPU 元数据回归修复，
不声称设备图算术故障已复现或完整模型等价。C5/C7 完整设备/组合验收保持 pending。

2026-11-14 [原生 A Graph 动态输入/KV 设备组件](evidence/native-attention-graph-design-20261114.md)
已通过：真实 ModelState、RequestState/输入/BlockTables、原生 V4.1 metadata builder、
ModelAclGraphManager 与 coordinator→native capture factory→NPUGraph，四个 U1/U2 图
各三次 same-key replay。新 IDs/positions、随后仅同地址 KV 内容变化均产生独立
标准库 BF16/FP32 oracle 预期的新结果，原生 query/seq/slot 与完整 KV 摘要同时核对。
A/F 各 21 个 parent，STOP/ACK/cache/group/daemon 正常退出、八卡前后空闲。
156 项安装身份、143 份生产闭包与 CANN/Triton 原文件前后 SHA 通过。前三轮原始
初始化/布局失败保留，最终仅 scratch CANN 镜像兼容编译 cache 身份路径，无安装或
生产修改。constructor-free owner/受控 Attention/HC、关闭 query tiling 的公开 builder
选项是明确边界，不代表 native SparseFA、公开 execute/sample、PD/DSpark/Engram、
完整模型或性能验收。1.3/1.5/1.6/1.8/1.10 保持 pending；继续同一 A owner 的公开
执行/原生采样、取消/空 parent、eager fallback 和随后 replay，不增加 A5 场景。

随后[同 owner 公开执行/采样生命周期](evidence/native-attention-lifecycle-design-20261114.md)
在 A3 首轮通过：6 个新增 parent，含 2 次原生 finished/取消空轮、2 次 prefill
eager 与 2 次 cached FULL；真实 LMHead/LogitsProcessor、Sampler、AsyncOutput 和
history/postprocess，无人工清 ExecuteModelState。四次独立 BF16/FP32 输出、token7
采样/历史、slot0 复用及 block1→2 的 KV 所有权核对；GPU computed4/5/2/3 与原生
CPU 准备快照0/4/0/2分别记录，不误称二者同步。A/F 各 27 parent、正常 STOP/关闭、
八卡前后空闲，159 安装身份、143 production、72/79 CANN 工具链核对及独立审计通过。
保持 constructor-free/受控算术与只读 KV 边界，完整模型、原生 SparseFA 写入、
PD/DSpark/Engram、性能均未验收。继续公开执行原始异常与 STOP 前 shutdown 拒绝、
旧 owner 保留和全新进程恢复，1.3/1.5/1.8/1.10 继续 pending。


2026-11-14 [原生 A 异常与全新进程恢复](evidence/native-attention-failure-design-20261114.md)
完成 A3 组件验证：failure-001 的 A 已完成 25 parent，第 26 个 eager 注入原始
RuntimeError；再次 execute 及 prepare_shutdown/shutdown 均拒绝，不提前释放 KV。
258 条实际 codec/plan digest 记录证实前 25 次有 COMPLETE，失败事务没有 COMPLETE
或 STOP。A/F exit 1/-15，F 由监督器 SIGTERM；缺失的 F rank JSON 不作为成功证据。
相同 395 份源内容的 recovery-001 在全新进程完成 27 parent、4 图、12 replay 和
4 次输出，并正常 STOP/daemon/group close。159 安装身份、143 生产闭包及 72/79
CANN 工具链核对通过；正常恢复轮无 timeout/kill，前后八卡空闲。
这是受控 Attention 异常和全新进程恢复，没有设备故障、回滚或同 owner 恢复结论。
Worker.shutdown、PD/DSpark/Engram 和完整模型保持待验；1.3/1.5/1.7/1.10 不提前勾选。


2026-11-14 [原生 A profiler 组件](evidence/native-attention-profile-design-20261114.md)
在 A3 完成 profile-001：27 parent、4 图、12 次 replay、6 次公开 execute 和
4 次真实 sample/head，正常 STOP/daemon/group close，A/F 均退出 0，前后八卡空闲。
真实 deferred A session 从 step0 到6，active=7，STOP 后由 owner finish/export。
独立 trace/CSV 核对 4 MatMul、24 send+8 receive 的 Model/Task/Stream/时间身份；
4 个非空公开执行 scope 各包含一个 head 和八个 HCCL。实际 7 个 ProfilerStep，
scope 可跨相邻 step；不把 CSV Step Id 当 parent。398 份源、143 生产、159 安装
身份及 72/79 工具链核对通过，BF16/FP32/KV/历史 oracle 沿用独立核验。
Windows 长路径收集失败和 parser 初始误判 aten::mm 的记录保留，均仅对同一原包
离线修复；实际 CPU head 为 aten::matmul，没有重跑 NPU 来掩盖后处理失败。
F profiler 在本轮关闭，不宣称 A/F 联合 timeline、有效重叠、完整模型或性能。
1.7/1.8/1.10–1.12 仍 pending，继续 PD 接入。

2026-11-14 [原生合法 PD 传输与关闭](evidence/native-pd-transfer-design-20261114.md)
先完成 native planner/allocator/reshape 的 storage-001。transfer-003 虽已传输并
核对全部物理 bytes，但 D 退出 SIGABRT、P 被监督器终止，整轮仍失败。发现 target-only
D 跳过现有 receiver retirement 后，插件仅扩展消费者 shutdown 入口：注册前保持
原生行为；注册后等待真正 DONE/ACK、join receiver/executor；失败保留资源。
CPU RED 为 2P/3F，GREEN 为5P，相关完整模块为814P、0F/E/S。随后 transfer-004
在 A3 完成真实跨进程页传输、全部字节核验和 D receiver 退休，P/D 自然退出均为0，
无 timeout/kill，前后八卡空闲。143 份生产仅上述 shutdown 文件变化，159 安装身份、
72/79 工具链和9份共享库身份核对通过。此处仍无模型 execute/sample、Scheduler
分配/完成/恢复、DSpark、Engram 或 Graph+PD 结论；继续补 Scheduler 接入，
1.5/1.9/1.10 不提前勾选，也不新增 A5 手工场景。

随后[原生 Scheduler 接入](evidence/native-pd-scheduler-design-20261114.md)在 A3
完成 scheduler-002：BalanceScheduler 实际分配三个组的目标块1/2/3，真实
ActiveKV 异步输出回传 update_from_output 后恢复首个 D target；prompt5的
external frontier 为4，NewRequestData computed4、尾 token 调度1，Request
乐观推进5分别记录，不称模型已经执行。两个物理页池全部 bytes、未选页和
来源不变核验通过；163安装身份、401源清单及143份未变生产闭包一致。
两端自然退出0，无 timeout/kill，八卡前后空闲；D receiver/executor 与 Scheduler
关闭后才 caller 通知 P。首轮容器比较错误及 D异常/P监督退出原包完整保留，
不追认为通过。继续 D 模型 execute/sample 和 A/F profiler 联合观测；
1.5/1.7/1.9/1.10 的整体条件未齐，仍不勾选。

Scheduler 原包已通过[独立审计](evidence/native-pd-scheduler-review-20261114.md)，
可复跑脚本和五个拒验反例一并冻结。随后进入
[真实 D 首 target 执行/采样接入](evidence/native-pd-attention-owner-design-20261114.md)：
保持真实三组 SchedulerOutput/页传输，新增独立 F daemon；明确使用公开 staged
组件绑定而非宣称 eager/U1 生产 factory 的正常选择，失败运行单独归档，验收待独立核对。

同日 [A/F 双端 profiler 同会话组件](evidence/native-attention-joint-profile-design-20261114.md)
设备两端27 parent、正常退出0/0和八卡 idle。原 supervisor 按host scope归属设备任务
而失败的记录保留；仅对同原包离线重读，用实际 launch/connection/model/device完成事件
核对，F56 MatMul+224HCCL与 A4head+32HCCL的 trace/CSV身份一致，全F任务唯一归属。
独立审计及14个拒验反例通过，没有NPU重跑；6个F MatMul跨host scope事实保留。
不据两端不同窗口声明物理重叠、性能或完整模型；1.7及联合任务仍不提前勾选。

后续 D 首 target 模型组件`model-008`已通过A3及
[独立审计](evidence/native-pd-model-validation-20261114.json)：真实三组分配/传输后，
D token15/position4读取收到的SWA prefix row3，经真实F daemon两层计算，独立
BF16/FP32 oracle、采样token7/history及同Scheduler完成/空轮退休核对；执行后整池不变。
P/D/F均自然退出0，八卡idle；403源/143未变production/163安装身份和13个拒验反例通过。
此前七轮失败保留。仍为显式staged U1和受控模型，下一步继续正常factory legacyU1选择链，
不据此勾选完整PD/DSpark/Graph/模型验收。

随后正常 eager/U1 factory 组件`legacy-002`在A3通过
[独立核验](evidence/native-pd-legacy-model-independent-validation-20261114.json)、
[审查](evidence/native-pd-legacy-model-independent-review-20261114.md)与
[可复跑审计](evidence/native-pd-legacy-model-independent-audit-20261114/README.md)：
实际registry选择P2p，原生forward经真实RemoteMoE和legacy F daemon完成唯一target1事务，
两端各七条canonical消息对应、真实TE/全页/独立BF16-FP32/采样history和退休通过。
P/D/F自然退出0，无timeout/内外kill，八卡前后空闲；143生产与model008相同，163安装身份、
CANN72/79、Mooncake9份二进制身份及十一项拒验反例核对。legacy001原数值失败/异常退出
保留；仅受控norm输出clone修正了组件alias假设，没有生产变化。
codec记录是成功收发后消息的canonical重编码而非raw socket抓包；五个Graph settings
默认True但eager inactive，driver axes00000不代表真实配置或重叠。仍是constructor-free/
受控算术组件，不关闭完整PD/DSpark/Graph/模型任务，统计保持54/109。
下一项为真实取消A后，同owner/Scheduler/receiver/TE以新ID B继续handoff及模型执行，
在上述 legacy-002 记录时仍在开发；后续实测结果独立记录如下。

2026-11-14 同 owner 请求取消/替换组件 `reuse-001` 已通过 A3 及
[独立审计](evidence/native-pd-request-reuse-validation-20261114.json)，
[设计和范围](evidence/native-pd-request-reuse-design-20261114.md)与
[可复跑审计包](evidence/native-pd-request-reuse-audit-20261114/README.md)已归档。
A 真实采样后仍为 RUNNING，再经公开取消、empty retirement；B 使用新 ID、不同 prompt/source
block，在同 Scheduler/模型/RequestState/receiver/TE 上继续真实 handoff、execute/sample 和
LENGTH_CAPPED 退休。实际三组页为 A `[1,2,3]`、B `[3,2,1]`，request slot 均为7；
两阶段全池字节、独立 BF16/FP32、有效 history、四个 F payload、两次 FORWARD 和最终 STOP
核对通过。P/D/F 自然退出0，无 timeout/内外 kill，前后八卡空闲；406 源清单、143 未变生产、
163 安装身份、CANN72/79、Mooncake9 及11项拒验反例通过。它只验证不同 ID 在 read/DONE/sample
之后取消再替换，不等于在途 read 抢占、同 ID 防重放或完整 PD/DSpark/Graph/模型验收；
下一项为 WAITING 分配后、首次 ActiveKV 提交前取消的独立边界，任务勾选仍保持不变。

同日 WAITING 取消边界 `pending-001` 已通过 A3
[独立 validation](evidence/native-pd-pending-cancel-validation-20261114.json)、
[review](evidence/native-pd-pending-cancel-review-20261114.md) 与
[可复跑审计](evidence/native-pd-pending-cancel-audit-20261114/README.md)。
A 原始零 token schedule 在 public abort 后仍由真实 owner 执行；20份空轮中第2份执行
退休通知，第20份真实接收完成才释放原分配/删除请求，没有 target/sample/F parent。
B 在同 Scheduler/模型/receiver/TE 继续独立新 prefix，三组真实 block 复用、整池累积
字节、BF16/FP32/有效history、唯一 B FORWARD 与两次 native DONE 均核对；P/D/F
自然0、无强杀、八卡 idle，407清单/143未变生产/163安装身份/CANN72-79/Mooncake9
及12项拒验反例通过。A 未入模型请求槽，不将 B 的槽7称为 A 槽复用；此项仍不是在途
read 抢占、同ID安全、Graph/DSpark/Engram 或完整模型验收，不改变任务勾选。

同日关闭 DSpark 的 PD + Q1 Graph/U2 组件 `graph-004` 通过 A3 及
[独立审核](evidence/native-pd-target-graph-independent-review-20261114.md)、
[原包审计](evidence/native-pd-target-graph-validation-20261114.json)和
[标准库复跑](evidence/native-pd-target-graph-independent-audit-20261114/README.md)。
仅一个配置门禁文件相对既有生产基线变化；此前四完整 CPU 模块 515P。
同已注册 owner 在请求前真实 capture 四份 bucket2/4×U1/U2 图，两条原生 PD 请求
首轮 eager、后两轮 FULL/U2 replay；96个BF16隐藏值、FP32路由、12次受控层KV写入、
两整池累计字节、真实sample/history和退休均通过，两次replay无PythonAttention新增。
P/D/F自然0、77.6478s、前后八卡idle；143生产与CPU GREEN逐字节相同，163安装源码、
CANN72/79、Mooncake9前后一致；11项证据篡改实际拒验。
001的ID类型断言失败、002/003的新fixture闭包绕过既有stage binding而数值失败均保留；
004改为原生的self.shared_state读写，不追认生产错误或旧noPD证据失效。
它不验收checkpoint/SparseFA/Engram/DSpark、完整serving入口、物理重叠或性能。

随后 [P 原生调度与执行产出链](evidence/native-pd-producer-design-20261114.md)
`producer-005` 通过 A3 及[独立审核](evidence/native-pd-producer-review-20261114.md)、
[原包审计](evidence/native-pd-producer-validation-20261114.json)。P 使用真实 native
Scheduler/NPUModelRunner/decoder/sample，prompt5原生截4、实际生成SWA KV；从真实
EngineCoreOutput取envelope，下一实际zero execute登记sender；D保持原prompt5、external4，
读取P生成的有效row3后交真实F。逐层BF16/FP32、P/D全池、D/F协议、history均核对；
真实DONE前持块、之后请求删除及KV usage由0→0.375→0，P/D/F自然0、77.9306s、八卡idle。
143生产与graph004相同，163安装源码/CANN72-79/Mooncake9一致，8项篡改实际拒验。
首四轮fixture/校验失败保留。原生V4.1 list carrier与通用zeroer存在边界，本组件实际
没有eligible zeroing segments，fresh-block zeroing明确未验收，继续单列适配与验证；
P sender线程join/TE显式close、checkpoint/SparseFA及完整模型任务仍未据此验收。
任务勾选仍为原状态。

随后 fresh-block 清零适配通过[独立审核](evidence/native-pd-zeroing-review-20261114.md)、
[原包审计](evidence/native-pd-zeroing-validation-20261114.json)和
[补丁架构审核](evidence/native-kv-zeroing-design-review-20261114.md)：原生V4.1的
`kv_cache=[Tensor/tuple]`被通用zeroer跳过，A3同一用例在旧143份生产源码上出现
64/64 hidden差异、最大12；仅增加两模块并接入runtime/PD worker的145份生产源码后，
相同原生Scheduler fresh IDs完整清零，P→D→F数值、整池字节、DONE持块/释放均通过。
P/D/F自然0、78.1615s、八卡idle；165安装源码前后一致，8项实际篡改拒验。
CPU `cpu-002` 四完整模块及唯一新增签名项371P、0F/E/S，145生产逐字节与设备GREEN相同。
CPU首轮344P/2F来自设备基线保留的旧Graph准入/receiver退休测试，恢复既已验收的CPU
测试闭包后通过；不将两轮拼接，原失败保留。该主设备场景压缩比2，不代替其他布局。

独立 [ratio4/128 tile 设备组件](evidence/native-kv-zeroing-tiles-review-20261114.md)
及[原包审计](evidence/native-kv-zeroing-tiles-validation-20261114.json)进一步验证：
相同145生产、真实原生planner/分配/绑定，原zeroer保持list载体skip，适配后仍调用
原生Triton清零API；4个底层storage在before/native/adapted三个阶段共25,608,192字节
逐字节核对，选中1/3整块为0，其他块及PD对齐前后padding均不变，ratio128的2字节
FP16索引scale真实覆盖。原carrier/group/view/alias不变，165源码/CANN72-79前后一致，
31.2712s自然退出、八卡idle，9项篡改拒验。它不证明完整Worker入口、A5 packed ABI、
checkpoint/SparseFA/Engram、DSpark proposal或性能；继续PD Graph取消/随后请求及
DSpark真实设备投影缺口，原有任务勾选保持不变。

同日 PD Graph 取消后的幸存请求组件 `cancel-002` 通过
[独立审核](evidence/native-pd-graph-cancel-review-20261114.md)、
[原包审计](evidence/native-pd-graph-cancel-validation-20261114.json)及
[可复跑审计包](evidence/native-pd-graph-cancel-audit-20261114/manifest.json)。
两条原生 PD 请求先 eager、再 FULL/U2 replay；公开取消 g0 后，同一 owner 的 g1
切到已有 FULL/U1 图继续真实采样，最终 empty retirement。三轮 valid/padded 行数为
2/2、2/2、1/2；最后一轮原始 padding 保留，仅有效首行参与数值和 history 核验。
两整池字节、取消页保护、独立 BF16/FP32、实际完成通知及两端315帧/31父事务一致；
P/D/F 自然0，78.9250s，无强杀，八卡 idle，145生产及165安装源身份不变，13项实际篡改拒验。
首轮 observer 比较 padded 全量与有效行导致的失败/异常退出保留；冻结摘要中的
`graph_mode=NONE` 和通用 cancellation=false 为继承字段，不用来判断实际执行。
实际 FULL 配置、图控制、公开取消与历史/字节证据单列核验。未据此验收在途 read 抢占、
DSpark/Engram、完整 Worker/模型、物理重叠或性能，原任务勾选不变。

继续 [P auxiliary 原生产出](evidence/native-pd-producer-aux-design-20261114.md)和
[DSpark 原生投影算子](evidence/dspark-native-projection-ops-design-20261114.md)：
`aux-001` A3 自然0、80.3400s，166安装源身份前后相同；
`ops-001` 实际缺少 `_C_ascend.npu_scatter_nd_update_sk` 注册而失败，未执行 RoPE，
其[失败原样记录](evidence/dspark-native-projection-ops-failure-20261114.json)。
正在自有 scratch 内构建锁定 Ascend 原始算子及扩展，既有 venv/CANN/OPP 保持只读；
构建成功也须另行执行设备数值验证，再接真实 loaded projection、Ready 与 P/D read。

`aux-001` 随后通过[独立审核](evidence/native-pd-producer-aux-review-20261114.md)、
[原包审计](evidence/native-pd-producer-aux-validation-20261114.json)与13项实际篡改拒验，
主流程另行运行冻结审计包得到逐字节相同 validation。两份 `[4,16]` BF16 aux 各128字节，
分别独立重建为进入 target0/1 前的 residual HC 均值；真实 native sample 清除 owner pending
state 后，保留的原 ExecuteModelState、InputBatch 坐标、aux tensor 地址和字节均未改变。
417源/122结果成员、145原生产、166安装身份、CANN72/79、Mooncake9与原生P→D/F/DONE均核验。
HC通道使用相同受控值，未证明非平凡HC均值或完整checkpoint；无draft/projection/接受率验收。
继续[原生 lower writer 与 draft cache 接入](evidence/dspark-native-loaded-projection-design-20261114.md)，
当前新增入口通过标准库导入、CLI负例、AST与Ruff，设备待隔离原生算子通过后执行；任务勾选不变。

随后[锁定原生扩展/OPP隔离构建](evidence/dspark-native-projection-ops-build-20261114/README.md)
完成，4476原件、共享输入保持只读；实际 `ops-004` 与 `model-002` 通过
[独立审查](evidence/dspark-native-loaded-projection-review-20261114.md)、
[原包审计](evidence/dspark-native-loaded-projection-validation-20261114.json)与6项实际篡改拒验。
前者896字节scatter和96字节RoPE逐字节符合独立参考；后者两底层各2,134,016字节完整保护
未选中区间，8行投影实际0 BF16 ULP（预设界限2未改），真实HCCL组销毁且两轮均自然0/八卡idle。
169安装源、145冻结生产、867隔离包、CANN72/79、Mooncake9前后相同；主流程复跑审计结果相同。
归档输入重放仍不等于live P原state，也未验收Speculator/proposal/Ready/read/完整模型。
head16仅用于这次投影；下一真实draft attention须遵守原生head512及HC4 ABI，继续新隔离算子
依赖构建和真实Speculator加载/绑定，不替换其Attention/HC/MoE叶子。旧失败全部保留，勾选不变。

并行修复[局部DSpark异常cause丢失](evidence/dspark-prefix-cause-review-20261114.md)：
真实CPU RED只因公开异常链为None失败；同测试/fixture/driver、生产只加原异常cause后，
完整consumer模块及native sample失败接缝45项通过。独立审计及主流程复跑相同；未更改wire、
分配、同步或owner生命周期，也未混入上面的冻结设备生产快照。

继续完成[完整 CANN/ATB 9.2 的锁定原生构建](evidence/dspark-native-draft-ops-build-20261114/README.md)：
4476原Git对象、1065产物及25,532只读输入逐项核验，主流程复跑构建审计得到同一结果。
原9.0.1缺宏、9.2旧头优先、新API与重入direct-kernel对象失败均保留，不改原生源或共享环境。
随后真实 `speculator-004` A3 自然0、84.0280s，原生31参数/16,686,524字节、9cache层、
实际LBNHC协商和draft group3绑定通过，[独立原包审计](evidence/dspark-native-speculator-audit-20261114/README.md)
及7反例通过；主流程复跑 validation 相同。前三个验证程序前置/断言失败不改写。

[原生 proposal-001](evidence/dspark-native-proposal-design-20261114.md) 首轮A3自然0、92.5491s，
真实HC/Attention/MoE/Markov未替换，Q2 ids[7,0]/pos[4,5]/seq6，返回[[7,7]]；
4096字节[4,512] BF16前缀对独立参考最大0ULP（预设2不变），两底层各1,179,648字节，
仅draft的6行6144字节允许写，其他物理字节、aux与绑定地址保护通过。
145生产匹配cause-GREEN后继，180安装身份/1065包/9.2工具链/只读自有JIT seed不变，
真实HCCL组销毁、前后八卡idle，无timeout/kill。独立审计已重算全部前缀、两完整池并拒绝10个语义反例。
该轮aux及InputBatch仍为组件输入，各group共用block1；不推断live target KV、完整草稿数值、
接受拒绝loop、正常AFD Worker或Ready/read。继续[实时P接入](evidence/dspark-native-live-prefix-design-20261115.md)，
保留正常生产sample→propose→consumer→publish和真实分组分配；中优先级Engram与旧MTP策略不变，任务不提前勾选。

实时接入审查发现[原生注意力发现与KV层集合不等](evidence/dspark-native-draft-layer-names-design-20261115.md)：
原生draft同时注册无KV spec的Attention helper与持有SWA的cache carrier，旧consumer误判为不匹配。
真实CPU RED为1通过/1失败；同测试、fixture与driver，单一consumer构造校验修复后49项全通过，
155安装身份和两快照不变。修复按原生公开get_kv_cache_spec分类，仍拒绝漏报KV和未注册层；
不改原生draft集合、wire、分配、同步或发布时序。继续A3实时P采样提案/Ready/read联调，任务不提前勾选。
[独立原包审计](evidence/dspark-draft-layers-audit-20261115/README.md)及主流程复跑结果相同，
10个静态分类输入和8项证据篡改另列，不计入49项CPU；前两轮live配置前置失败仍未到consumer设备接缝。

第三轮补齐既有AFD配置namespace初始化后，验证程序仍误判原生engine-ID自动UUID后缀；
该前置断言已修，原始失败保留。[live-004](evidence/a3-dspark-native-live-prefix-20261115/live-004/README.md)
冻结启动后SSH中断，结果/清理/设备状态尚未核实；恢复连接须先收集同一原包再决定重跑。
继续本地实现[真实Ready边界观测](evidence/dspark-live-ready-frontier-design-20261115.md)和
[四角色D执行/接受拒绝](evidence/dspark-live-decode-design-20261115.md)，
保留原生draft、sampler、缓存分配与独立D–F2连接；未执行部分不记设备通过。

2026-11-15 连续开发补齐 [H14 HTTP 失败通道](evidence/h14-http-rpc-terminal-review-20261115/README.md)：
锁定原生 worker 回复不带 RPC ID，真实 ASGI/原生方法 CPU 队列边界复现超时及部分 rank
失败后 GET 误收 profile 的 None。EngineCore 现对 NPU HCCL Multiproc 的固定 profiler
失败先置原生 executor failed，再通知 native fatal callback；后续管理和 execute 拒绝，
F 清理不再发 STOP，原异常保持。HTTP 初始化同时拒绝非 HCCL A/F 配置，范围与保护一致。
核心同 tests/fake RED 17F/55P/6S → GREEN72P/6S；准入同源 RED4F/26P/5S → GREEN30P/5S。
Linux 四完整模块172P/7S；全 CPU 单测1960P/525S，0F/E；跳过项不算原生运行时验收。
两组实际ASGI对照与独立源码审查通过，不含native MP IPC、真实引擎退出或NPU。
429源/145生产/191安装身份的独立A3 CPU包已准备，尚未发送/执行，旧设备包未刷新。
C7 1.7/1.10及完整模型任务继续pending；项目仍54/109，无新增A5手工场景。

C7 1.6 的[部署源码清单](evidence/recipe-complete-source-20261115/README.md)不再手选18项，
改为全部149个插件Python文件加2CLI/source-lock共152项。公开CLI52场景仍5合法通过、
47错误拒绝；独立四次真实prepare确认新源自动纳入、缺锁及不可读源明确失败，且不导入
设备运行时。仅checkout源码身份，不核验目标安装/二进制/权重，旧冻结包不改写，任务不勾选。

H14 后继复查补齐 explicit VLLM_PLUGINS 必须保留 afd general plugin 的初始化检查，
避免只加载 endpoint 而假定另一 EngineCore 进程已有保护；未设置清单保留原生默认。
同tests/fake RED3F/33P/5S → GREEN36P/5S，独立复跑一致；最终 Linux 四完整模块
178P/7S，真实 ASGI 两组对照重跑通过。前述1960P全CPU结果仍属于末次门禁前快照。
A3 CPU包002仅比001改API/test/driver三项，另外426项相同，尚未发送/执行。
格式收尾后的包003仅比002改测试排版及driver对应pin，测试AST等价、145生产字节不变；
独立读取429源、421Python编译和191计划身份核验通过，旧回归结果与归档保留。


### 2026-11-17：非 Engram 组合接入收口与 A5 最小集中验证

按用户要求先完成非Engram组合实现/审查及CPU检查。补齐默认关闭的
`AFD_DSV41_FUNCTIONAL_QUALIFICATION=1`正常Worker资格入口：原生U1/U2、
FULL_DECODE_ONLY Graph、eager draft DSpark、attested PD；保留BF16/dummy三层C2、
TP1、最多两请求及draft/cache/Engram边界，旧资格入口与默认拒绝不变。
纳入已有Graph/U2 `cross_layer_v1` PD准入及F daemon BaseException公开归属修复。
P中途抢占按既有当前切片terminal处理，不新增恢复协议。暂停的普通旧V4 MTP未纳入交付源码。

真实隔离CPU门禁最终342P/0F/E/S；原生1、A4、F4共9个首次EngineArgs声明/配置检查
全部返回0，NPU初始化前后均False。源码审查无阻断。本地完整unit、最终封包/CLI与规划
检查见本轮证据记录；这些结果不作设备验收证据。

新增正常原生/A/F Worker三层C2 probe，经过实际Worker factory、加载、原生KV规划、
公开compile_or_warm_up、双请求两轮、取消A/B继续、正常退役和追加C；不写死正常模型token。
Graph/U2/多流成功要求实际阶段记录，不凭配置开关判通过。统一入口额外比较原生79参数与
A58+F21实际字节及采样序列。A5默认7场景：原生eager/U1、AFD四模式（U2带多流）、
DSpark/PD拒绝后继续/Ready，以及Graph/U2多流跨层接受/取消/新请求。

执行指导：[DSV41_A5_FUNCTIONAL_VALIDATION_ZH.md](../../../docs/npu/DSV41_A5_FUNCTIONAL_VALIDATION_ZH.md)。
所有场景共享同一146生产模块冻结源码，保留原生A5库与Python来源检查，不包含A3二进制。
A5设备结果待用户回传；完整模型、Engram、全五PD矩阵、拓扑/长运行和性能/启停验收保持原状态。
本轮不提前勾选1.3/1.5/1.8/1.9/1.10/1.12；总任务计数仍54/109，不能当作代码完成百分比。

最终交付[证据](evidence/non-engram-functional-20261117/README.md)：完整本地unit
1805P/698S/0F/E；真实依赖CPU追加873P/0S/0F/E，NPU初始化前后False。
历史快照的未交付ordinary-MTP签名项已从交付表剔除，工作区暂停改动保留；
已有profiler/管理RPC异常传播修复同步保留，以保持生产、测试及A5包一致，不新增设备专项。
373生产/单测路径与真实CPU源码一致，146生产模块与A5包逐字节一致。

### 2026-11-17：A5集中验证回传，原生源码预检阻塞

用户回传ZIP已[归档分析](evidence/a5-native-source-preflight-20261117/README.md)。
AFD功能包SHA正确；194份原生Python源码中189匹配，差异仅为Ascend的五份源码，
vLLM 126/126匹配。失败发生于原生运行库导入前，7个功能场景均未开始，不能计为功能
通过或设备推理回归。软件包版本后缀不是当前editable源码Git身份。

其中forward context/DSpark speculator有既有A5精确源码，确认包含MegaMoE/PD策略、
draft配置及关联导入变化；ascend_config/platform/utils仍需实际字节。
新增stdlib只读采集器，定向归档差异源码及实际HEAD/status/diff，保留严格源码门禁，
不替换共享环境、修改上游或扩大运行时准入。用户可复用已有失败JSON，无需重跑7场景。
实际源码收齐后的兼容审查与A5功能验证保持pending；本轮不变更任务勾选。

### 2026-11-17：A5原生源码已匹配，预检JIT阶段超时

用户回退后[本次回传](evidence/a5-native-runtime-timeout-20261117/README.md)确认
194份原生源码全部匹配；进程`timed_out=true`、退出`-9`，日志仍在原生导入/JIT编译。
原120秒预检被外层终止，七场景尚未开始。清理采样保留一个存活PID，不能将旧轮算通过。
验证入口新增独立`--preflight-timeout`（默认600秒）及明确超时错误，同步更新冻结包；
188源码仅入口改变，146生产模块和194份原生身份锁不变，原包与失败原件归档保留。
默认600/显式900两次七场景metadata通过且未导入runtime，三个非法参数拒绝；
CPU错误报告接缝重放确认超时/非超时均失败且零场景，不计设备证据。
实际预检完成、清理与A5功能重跑仍待用户验证，任务勾选保持原状。

### 2026-11-17：A5预检通过，修复正常场景的配置初始化断言

[回传结果与修复](evidence/a5-config-device-init-20261117/README.md)：原生预检194源匹配，
40.43秒退出0且无残留；原生首场景配置已完成，但NPU初始化后被验证probe的CPU断言阻断，
模型尚未加载。修复仅在真实`--run`配置前绑定指定卡、记录并核对配置前后设备，
保留独立`--run-config`的CPU未初始化检查和原生Worker流程。
冻结包仅probe一项改变，146生产及原生194身份锁/launcher不变；原件与父包保留。
六项合成CPU报告接缝、七场景metadata、185Python编译及Ruff通过；
这些不计设备证据。A5后续Worker/加载/推理和七场景仍待重跑，原任务勾选不变。

### 2026-11-17：A5模型已加载，修复参数观测的内部格式读取

[回传日志及修复](evidence/a5-parameter-readback-20261117/README.md)：原生Worker与加载已完成，
参数SHA采集的CPU复制拒绝内部格式；KV与推理未开始。本次角色通信组已销毁且无清理异常。
共享parameter_record只将detached观测张量非原地转ND后读取，原参数与SHA/finite对照保留；
准备器同步刷新helper，冻结包仅该源码成员改变，146生产及194原生身份锁/入口不变。
六项真实CPU参数哈希与原件保持检查、七场景metadata、185Python编译、194Git对象核对、
Ruff及本地规划/保护检查通过。原日志和父包保留；A5格式转换、KV、推理及全部设备场景待重跑，
不提前变更任务勾选。

### 2026-11-17：A5首轮QLI拒绝64头，修正受控Indexer声明

[原始日志与本轮检查](evidence/a5-qli-geometry-20261117/README.md)确认原生模型、
参数读取、KV与公开warmup完成，首prefill在QLI metadata失败，未进入forward/采样。
真实DSL错误要求N1=32/N2=1/D=128；实际indexer权重证明受控声明为64头。
正常原生/A/F首次配置统一indexer32/128，普通Attention仍64头、维度512；
同批DSpark声明同步。functional资格入口增加该精确候选，旧dummy及默认限制保留。
冻结包同步，旧日志、模型配置和父包保留，上游只读工作区不变。

真实依赖CPU同测试RED36F/350P到GREEN386P/0F/E/S；原生1、A4、F4共9个
首次配置全部通过，实际indexer32/128且NPU初始化前后均False。
本地CPU56P/330S，跳过不计通过；七场景metadata、185Python编译、194Git身份、
Ruff及规划/保护检查通过。A5新几何与后续设备场景仍待重跑，不变更任务勾选。

### 2026-11-17：集中入口补齐Transformer vendor路径

用户回传CompressorV2库或符号未找到；此前原生入口已有vendor环境处理，集中入口
只继承环境，遗漏显式参数。新增可选`--transformer-vendor`，检查现有libcust_opapi.so
路径，在子进程启动前前置ASCEND_CUSTOM_OPP_PATH和LD_LIBRARY_PATH；
预检、原生/A/F及DSpark继承同一环境，父环境及共享安装不改写。
首次声明和实际两组路径记录在summary，指导默认命令同步该参数及新入口SHA。

[本轮CPU及源码记录](evidence/a5-transformer-vendor-20261117/README.md)：四项stdlib
CLI/真实子进程环境检查通过，含七场景metadata、无参数继承和错误路径拒绝。
新冻结包只变更launcher，146生产与194原生身份锁不变；185Python编译、Ruff及
规划/保护检查通过。实际A5库符号、CompressorV2与七场景仍待重跑，不变更任务勾选。

### 2026-11-17：CompressorV2最小输入宽度

[本轮失败与CPU检查](evidence/a5-compressor-width-20261117/README.md)记录新上传日志：
原生加载/KV/warmup通过，首prefill进入CompressorV2后报507035/向量341 UB越界，
请求无已完成步骤且清理非全通过。日志模型H=512，低于官方算子公开H>=1024约束。
普通Worker原生及A/F在首次配置统一H=1024，D512和Index32/D128保留；权重观察按
实际配置核对embedding/lm_head/mHC。functional入口仅增加固定1024候选，旧dummy
限制保留；独立受控DSpark组件不改写，不推断其执行原生CompressorV2。

同一真实依赖CPU测试由旧校验RED36F/392P变为GREEN428P/0F/E/S；9组EngineArgs
均通过且实际H1024，NPU初始化前后均False。七场景metadata/4项CLI、185Python编译、
194原生Git身份、Ruff与规划/保护检查通过。源码、默认包及锁同步，保留父包和失败证据。
A5越界是否消除、采样及七场景组合仍待更新重跑，不变更设备验收任务勾选。

### 2026-11-17：H1024原生异步错误与同步诊断

[新日志及检查](evidence/a5-native-sync-debug-20261117/README.md)确认H1024已生效，
首轮仍失败；Bisheng报507035，外层dsa_v41_forward/event栈提示异步定位不准确。
没有已完成请求或采样，worker清理也非全通过；不能确认具体失败算子，不能宣称H调整
已解决设备异常。新增`--diagnose-native`，仅原生eager/U1子进程设置同步执行，
保留预检/清理与失败退出；跳过A/F、Graph/U2及DSpark，并明确诊断不代表七场景通过。

本地与A3隔离目录六项stdlib CLI检查均通过，覆盖单原生子进程实际同步变量、
默认七场景metadata与父环境保持；A3测试进程未导入任何NPU运行时。
185Python编译、194原生Git身份、146生产字节不变、Ruff与规划/保护检查通过。
源码和默认冻结包同步，下一步由A5同步异常栈定位具体算子；设备验收勾选不变。

### 2026-11-17：A5 MoE 路由小专家形状

[同步日志和补充summary](evidence/a5-routing-width-20261117/README.md)确认
194份原生Python源码匹配、同步开关生效；首轮实际失败于
`_C_ascend.moe_gating_top_k_hash`，gate为[8,1024]，尚无采样且Worker shutdown失败，
最终八卡空闲。锁定A5内核的32元素padding缓冲与完整64元素向量读取存在小形状风险；
未证明A5扩展的C++构建来源，根因及修复效果仍需设备验证。
普通Worker原生/A/F首次声明统一64专家、top-k2，保留H1024与Index32/D128并记录实际值；
functional准入仅增加64专家候选，历史dummy和独立DSpark组件继续保持原8专家边界。
新CPU测试对父版本36F/436P，修复后472P/0F/E/S；九组真实EngineArgs配置通过且实际
64专家，NPU初始化前后False。本地/A3各六项CLI、185Python编译、194原生Git身份及
Ruff/规划/上游保护通过。默认冻结包同步，145份其他生产字节和原生身份不变。
下一步由A5先重跑同步原生，再执行七场景；不变更任何设备验收勾选。

### 2026-11-18：64专家失败与路由单算子隔离

[新日志及本轮检查](evidence/a5-router-isolation-20261118/README.md)确认64专家、
H1024和同步开关生效，仍在原生hash router报507035/341，尚无推理完成且shutdown失败。
上一轮调整未解决设备故障；重新核对编译入口及同目录regbase实现，保留已安装内核来源未知。
新增`--diagnose-router`，六个独立进程比较64/384专家、top-k2/6、FP32/BF16及一个vision-bias
对照，保留原生bootstrap/调用和CPU数值参考；不加载Worker/模型，不建立通信组。
失败后先检查所属进程回收和八卡空闲再继续，汇总失败保持非零退出；缺失实际算子结果也失败。
保存每项输入、schema、结果/同步栈及实际加载库SHA。本地与A3各10项CPU检查通过，
包括六项metadata、诊断模式互斥、缺失结果拒绝与失败后独立项保留；未导入NPU运行时。
186Python编译、194原生Git身份、146生产字节保持及Ruff/规划/上游保护通过。
默认冻结包和入口SHA同步，A5需按新指引回传单算子结果；启动及七场景验收仍待，勾选不变。

### 2026-11-18：六项路由对照通过与原生真实输入诊断

[本轮上传和CPU检查](evidence/a5-router-real-inputs-20261118/README.md)记录六项独立A5算子
全部通过，ID精确一致，FP32最大绝对误差1.1920928955078125e-7、BF16为0；
六项实际加载库身份一致，194原生Python源码匹配。按用户要求排除NPU占用提示，
保留原始summary失败和占用观察；不据此宣称此前原生507035/341已解决。
普通随机64专家输入可运行，保持现有模型几何；原生诊断增加真实路由输入、布局、有限值统计
和加载库身份保存，失败且所属进程回收后用新进程重放有限值的完整失败调用。
重放成功不覆盖原生失败；记录不完整/非有限时明确跳过，不改变原生数学或模型参数。
显式占用忽略参数仅限诊断，默认七场景限制保持。本地与A3各15项入口检查、
A3真实CPU PyTorch六项捕获/错误/有限值/布局/嵌套调用/观察器恢复检查通过，NPU未初始化。
186Python编译、194Git身份、146生产字节保持及Ruff/规划/上游保护通过，默认冻结包同步。
真实A5输入捕获/重放、原生首轮及七场景仍待用户环境运行，设备验收任务勾选不变。

### 2026-11-18：真实路由输入重放通过与导入前vendor准备

[两份新上传及本轮记录](evidence/a5-router-import-order-20261118/README.md)确认真实FP32
logits `[8,64]`与bias `[64]`均连续、偏移0且全部有限；原生仍报507035/341且shutdown失败，
同一输入独立A5重放成功，CPU参考ID精确一致、权重最大绝对误差5.960464477539063e-8。
所属进程已回收；按用户要求排除占用观察，不改写原始失败。
原生额外加载customize vendor库，但其是否提供失败符号尚未证实，未确定设备根因。
锁定C++扩展在dlopen时固化vendor路径；将已安装vLLM-Ascend自带vendor的优先级准备
提前到两个A5探针的实际运行时导入前，保留外部Transformer/其他目录和原生数学。
增加首次调用前库、路由符号导出及推理上下文记录，符号探针只读已加载handle，不代表实际缓存绑定。
本地/A3各17项入口检查、A3七项真实CPU捕获/符号检查及真实输入参考通过，NPU未初始化。
186Python编译、194Git身份、146生产字节保持与Ruff/规划/上游保护通过，默认冻结包同步。
导入时序调整能否修复A5首轮仍待重跑，七场景及全部设备验收任务勾选不变。

### 2026-11-18：A5 原生 vendor 前置诊断与请求生命周期通过

[两份成功上传与独立核对](evidence/a5-native-vendor-first-pass-20261118/README.md)
确认默认冻结包d0a32a80、194份原生Python源码匹配；原生NPUWorker/V2、BF16 dummy
三层C2、H1024/E64/top-k2、Index32-D128在npu:0完成六轮执行/采样。
十八次路由输入完整有限且调用均返回；首次PT与前次失败输入SHA完全一致，
已安装vendor在扩展导入前前置，原外部Transformer/FLA目录保留。
两轮后取消A、B继续结束、追加C并结束，Scheduler/Worker请求清空且复用退役请求槽位。
六份hidden及十八份路由输入SHA、加载/进度/最终报告一致；Worker和通信组正常清理，
进程退出0，无超时、强杀、所属残留或清理异常。占用观察按用户要求排除并原样保留。
成功时仍加载customize库，未把单个导出来源当作C++缓存绑定的直接证明。
本轮确认vendor前置后的原生诊断路径通过；summary仍为diagnostic_only且仅一个场景。
下一步完整七场景的原生/AFD参数及采样对照、U2/Graph/多流、DSpark/PD仍待A5执行；
完整训练检查点、Engram与性能未验收，整项任务勾选保持不变。

### 2026-11-18：AFD U1 通过与 U2 child 源 RoPE 初始化修复

[三份上传、独立对照及CPU检查](evidence/a5-u1-pass-u2-source-rope-20261118/README.md)
确认U1 A/F均退出0，六轮执行/采样、取消及随后请求和角色清理通过；
对此前保存的原生诊断基线，A58+F21共79份canonical参数身份及A/B/C输出完全一致。
本轮局部上传未含完整source/summary和U1外层监督报告，独立对照不改写原始报告标志。
U2握手后、首轮计算前在C2 metadata缺完整源RoPE表；F侧peer closed/loop失败为连带错误。
外层退出1，无超时、强杀或所属残留，失败与清理错误均保留。
锁定上游只准备原生attn_groups，插件创建的独立child builder漏掉同一初始化。
在子builder创建后调用原生prepare_source_rope，保留同步eager、FULL任务owner及stage状态隔离。
新增三项CPU契约修复前均失败，修复后相关290项全部通过，NPU未初始化；
本地17项入口检查、186Python编译、194Git身份、Ruff和规划/上游保全通过。
冻结包仅更新该生产文件（另145份生产字节保持）及provenance/manifest；
A5 U2修复效果、后续Graph/DSpark/PD及完整七场景仍待重跑，整项任务勾选保持不变。


### 2026-11-18：A5 DSpark P sender 退役

[原始失败、实现审查和 CPU 验证](evidence/a5-dspark-sender-retirement-20261118/README.md)
记录 c1107179 批次的普通 AFD 五场景组件结果和 DSpark 退出失败。P 进程
`corrupted double-linked list` / -6 保持失败；正常场景仍是受控 dummy 参数。
生产修改要求 registry/drain 后停止并 join sender 的原生 socket owner；不改变
TE/注册/报文所有权。213 项 CPU KV/DSpark/签名检查通过且 NPU 初始化前后均 false；
19 项标准库入口检查通过。此前 333 项广回归触发 NPU 初始化，不计作 CPU 证明。
仅刷新一个生产文件，其余 145 份保持；新增 `--dspark-only`，A5 退出修复仍待确认。

### 2026-11-18：DSpark Graph/U2 第一个 eager 批次的观察器误判

[ZIP 分析和冻结后继](evidence/a5-dspark-graph-observer-20261118/README.md)：
用户反馈 reject 场景成功，本 ZIP 不含其原始报告；不能独立复核该成功或完整七场景。
Graph/U2 run `2961f4d3e38240e9ad96aeb2ae8e705f` 的 capture、KV 字节/物理页
检查返回成功；首次异常是 eager 返回后顶层 forward 计数预期 2、实际 0。
HCCL U2 使用直接 stage 路径，观察器改为 core embedding 入口，不改生产执行。
固定父 helper 哈希后生成最小后继，保留 eager 2/1 与 replay 0 的严格计数和
原有数值/存储/报文检查；146 份生产源码保持字节一致。新增标准库 host 观察器
回归先复现旧断言，再与入口检查合计 21 项通过；186 Python 编译、194 原生源码
pin 核对通过。本地 host fixture 不证明 NPU stage、replay 或多流正确性。
原始后续 heap corruption、D=-6 和三个 forced kill 保留为未解决设备失败。
Graph/U2 replay/取消/替换及正常退出仍待 A5；1.3/1.8 等整项状态保持不变。

### 2026-11-19：reject 成功确认与 Graph/U2 stage 观察器修正

[当前两 ZIP、顶层 summary 与修正验证](evidence/a5-dspark-stage-observer-20261119/README.md)
确认 source SHA 匹配 04da709：reject process=0、functional accepted=true、
四角色 exit=0，无强制终止/存活子进程。P sender socket owner join 已返回。
详细 P/D 最终报告未随 reject ZIP 提供，具体提案/接受拒绝计数不从本包独立复算。
整批失败来自 Graph/U2 run `24d698d379344c7a8ef541279205964b`：capture/KV
保持已返回，第一轮 eager 后 embedding 计数 1、token_rows=6，而两层 stage0/1
的 host compute_returned 均存在。上轮观察器遗漏 native MRV2 父批次 prepare
embeds 分支；此入口亦可在 replay 前运行，不能作为 stage 执行计数。
仅修改冻结 helper 观察点为受控首层 Attention forward，保留 eager U2=2/U1=1、
replay=0 和原检查，新增逐 stage 3 行校验及断言前最后批次记录。146 份生产及
native identity 字节保持，194 pins 核对、186 Python 编译、确定性重建和 21 项
标准库 host/入口检查通过。原始 heap corruption、D=-6、内部三个 forced kill
保持失败；Graph replay/取消替换/正常退出待 A5，1.3/1.8 等整项不提前勾选。

### 2026-11-19：Graph/U2 已越过旧观察器断言，等待内层最终失败报告

[当前部分证据和顶层失败摘要改进](evidence/a5-dspark-graph-final-audit-20261119/README.md)
记录用户 9d66ef97 ZIP：source SHA 匹配 51a0954，reject accepted=true、四角色
exit=0；Graph run `9ead09c9335949019bf137a913ff0774` 出现第三次 KV 传输和
consumer-verified 标记，提供日志未见旧断言/heap corruption，但顶层 Graph
process=1。该标记早于 observer 及 supervisor 的最终验证；不能判整体通过。
缺少场景内 summary 和 P/D/F-consumer 最终 JSON，已请求既有报告，无需重跑；
未知失败不猜测为 runtime 或审计错误。入口仅补 failed scene failure_summary，
传递内层 traceback/退出码/角色错误并保持原失败。23 项标准库检查、冻结编译/
pins/确定性及本地 harness 通过，146 份生产与 Graph helper 字节保持。
当前 Graph/U2 验收和根因仍待缺失报告，1.3/1.8 等整项状态不变。

### 2026-11-19：补齐 Graph/U2 内层报告，修正设备映射审计并离线复核

[原始报告、完整复核与修复](evidence/a5-dspark-device-audit-20261119/README.md)
确认 run `9ead09c9335949019bf137a913ff0774` 四角色 success=true、exit=0，
无超时/强制终止，Graph/U2 eager/replay、取消替换及正常退出的驱动检查均已返回。
最终失败为离线校验写死 A3 Attention/FFN 卡号 2/6，本次启动参数实际为 1/3；
pipeline 审计亦有同类常量。六份冻结 helper 改由 supervisor 启动参数按角色传递
必填期望卡号，保留原协议/数值/KV/host/pipeline 全部校验，146 份生产及 native
identity 字节保持。旧实际执行快照复现同一失败；修正后真实既有报告完整事后复核
通过，错误期望卡号仍拒绝。25 项本地标准库检查、186 Python 编译、194 pins、
确定性重建与 Ruff 通过；本地规划/只读保护结果随证据保存。
原始失败 summary 不改写，无新 NPU 执行；本项定位与既有报告复核无需重跑。
受控非 Engram Graph/U2/PD/DSpark 组件组合已有真实设备记录及独立事后审计，
新版本入口生成成功顶层 summary 尚待后续自然运行，完整权重/Engram/物理重叠/
性能及 1.3/1.8 等更大组合任务不提前勾选。

### 2026-11-19：完整权重验证指导与当前源码预检对齐

[完整权重指导](../../../docs/npu/DSV41_A5_FULL_CHECKPOINT_VALIDATION_ZH.md)
复用既有 eager/U1 A4/F4 TP4、原始 safetensors 和 A 侧原生 CPU Engram 资格入口。
预检原四份旧 4ec6dc77 哈希与当前锁不符，按只读 Git 对象对齐 7e37e29a，21 份原生
身份全部核对；保留 checkpoint/config/index 与默认拒绝，不加兼容哈希或改上游。
诊断入口增加输入 token 核对、首个输出分歧和无参考/无效参考状态，失败后仍保留
独立 token-comparison.json 并汇入 summary；单短请求不自动判模型等价。
具体源码/标准库回归及交付边界见
[本轮证据](evidence/a5-full-checkpoint-guide-20261119/README.md)。
生产代码、七场景冻结包及 Engram/U2/Graph 准入保持；没有新 NPU 执行。
当前待用户按新指导回传实际载权/短请求/正常退出和原生参考；历史首 token 差异
及完整数值/连续请求仍未闭合，1.3/1.8 等整项勾选不变。

### 2026-11-19：澄清 Engram 原生 U1 与新增 U2 目标，修订计划

用户此前将 Engram 理解为所有场景均不支持；本轮依据官方
[limitations](https://github.com/vllm-project/vllm/blob/main/docs/features/engram.md#limitations)
及锁定 vLLM `ced6857a` 的
[配置拒绝](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/config/vllm.py#L1220)
明确：原始模型保留 Engram，原生明确限制的是 DBO/microbatch。当前先复用已有路径
验证真实权重 eager/U1，历史首 token 差异仍待解决；Graph/U1、DSpark/U1、PD/U1
按各自前置复核验证，未被该条禁止不等于已支持。U2/DBO 保留最终目标，新增准入、
producer/stage、图及 history 缺口按中优先级审查实施。阶段 U1 可单列，不替代 H01–H15。

本次同步总计划、能力矩阵、C7 proposal/design/tasks、工作流/证据技能及会话范围指针。
原任务编号、完成勾选、源码锁、原始证据和未提交实现保留；本轮只有计划/工作流改动，
未运行模型或设备实验。校验仅证明规划一致及上游只读保护，不计模型或 NPU 验收。

### 2026-11-20：TP1 / DP4 / EP4 HCCL 支持

- [x] EP1：核对供体与锁定原生 MoE / DP 调用边界，记录架构审查。
- [x] EP2：实现 EP 协商、TP1/DP 角色编号与 eager/U1 准入；保持旧 TP4 路径。
- [x] EP3：F 从全局 plan 推导 DP token 向量；idle/dummy rank 参与原生 MoE 和 EP。
- [x] EP4：提供原生 TP1/DP4/EP4 对照与 A4/F4 验证入口、配置检查及指导。
- [x] EP5：通过 CPU 协议、角色/配置/执行边界回归、语法及上游只读保护并归档证据。
- [x] EP6：A5 原始权重 EP 载权、连续请求、空闲 DP 与正常退出的功能验收。2026-11-24 确认 13 条有效响应、取消后续跑、四个 F rank 的 STOP/ACK 和八卡释放。
- [ ] EP7：EP 与 Graph/U2/DSpark/PD 的组合实现及设备验收。
  - [x] EP7.1：接通 EP + Graph/U1 的全局桶、异长 eager 回退和原生 A/F DP context，完成配置及 CPU 回归。
  - [x] EP7.2：A5 EP + Graph/U1、DSpark N=0 功能验收。2026-11-24 核对双端捕获、全 rank 回放、空闲 rank 回退、取消后请求与正常退出。
  - [ ] EP7.3：A5 验证 EP + DSpark N=1/2 的 target/draft DP 同步、实际采样及提交 history。
  - [ ] EP7.4：A5 验证 EP + PD 的跨 DP rank KV/index/history 交接，覆盖 DSpark N=0/1/2。
  - [ ] EP7.5：A5 验证串行 eager/Graph U2 的双 stage 与 Engram producer，覆盖独立 AFD 和 PD 的 D。
  - [x] EP7.6：接通串行 eager/Graph U2、DSpark N=1/2 和单节点 PD 的代码及验收入口。
  - [ ] EP7.7：在锁定原生依赖下复跑最终源码的 CPU 合约与完整 EngineArgs 矩阵。

架构与冻结证据见 [EP CPU 审查](evidence/ep-cpu-20261120/README.md)：1859 项 CPU
合约、79 项本地诊断入口及原生/A/F 三组真实 EP EngineArgs 通过，无 NPU 初始化。
EP1–EP5 只表示代码/CPU/指导完成；EP6/EP7 和原有完整迁移验收状态保持待验证。

随后按用户“提交并推送”授权准备独立交付树；从 `f29b74d9` 只应用本轮 EP 增量，
排除其他未提交实现/测试后，1761 项 CPU、79 项本地诊断入口及三组真实 EngineArgs
仍通过，见同目录 `delivery-validation.json`。原始工作区及 1859 项开发结果保留；
此复核不新增 NPU 验收或改变 EP6/EP7 状态。

用户要求按上述顺序继续。组合源码开发和 CPU 回归与 EP6 的 A5 基线验证并行，
设备验收从 EP6 开始，随后按组合分别执行。

EP7.1 的 [冻结证据](evidence/ep-graph-u1-20261120/README.md) 包含 1977 项 CPU 回归、
38 项本地诊断入口测试、43 项子测试和五组真实 EngineArgs。A5 Graph/U1 入口按 DP rank
定向发送请求，设备验证仍由 EP7.2 记录。

2026-11-24 已按功能范围核对 EP6 和 EP7.2 的 A5 回传。见
[EP 指南的功能验收](../../../docs/npu/DSV41_A5_EP_VALIDATION_ZH.md#ep-功能验收2026-11-24)
所列归档。原始归档保留各自的 token 比较、请求判定和退出码。
后续用 `--ep-functional-only` 验证 `N=0` 的串行 eager/U2，再验证 Graph/U2。

EP7.6 的 DSpark 接入检查了锁定 Ascend `7e37e29a` 的
`worker/v2/spec_decode/dspark/speculator.py::propose` 在普通分片草稿路径复用 target 的
`dp_sync`，profiling dummy 则清空同步状态。A 保留原生 proposal 和 history 退休，F
保留 target 验证与 draft 隔离。P 使用 eager/U1，D 可运行 eager/Graph U1/U2。
详细的角色与性能审查见 [设计](design.md#ep--串行-u2dspark-和-pd)，
A5 命令与验收条件见 [TP1/DP4/EP4 指南](../../../docs/npu/DSV41_A5_EP_VALIDATION_ZH.md)。

2026-11-23 的 [A5 原生 N=0 记录](evidence/a5-native-ep-n0-20261123.md)
确认 TP1/DP4/EP4 完整权重推理与 13 条定向矩阵响应。原生四 API 进程组退出
超时并触发强制清理，八卡清理后空闲。EP6 保持待验。资格脚本允许清理完成后
复用已有原生响应继续 AFD token 诊断。
