# HCCL 全能力组合、PD 与交付验收：设计

状态：已有实现与有限组件证据，剩余契约、完整模型及组合验收按任务补齐。
源码编号见 [来源索引](../../../.agents/dsv41/sources.lock.json)，能力编号见 [迁移矩阵](../../HCCL_CAPABILITY_MIGRATION_ZH.md)。

## 范围、依赖与源码

依赖 C2/C3/C4/C5/C6，闭合 H12–H15 并审计 H01–H11。
2026-11-19 开发顺序按[总计划优先级](../../ROADMAP_ZH.md#engram-priority)：先复用原生
Engram 完成真实权重 eager/U1 对照；Graph/U1、DSpark/U1、PD/U1 按各自依赖复核验证，
不等待 U2 先完成。当前 vLLM 的拒绝条件是 Engram 层存在且 `use_ubatching` 为真，
没有直接禁止上述 U1 组合，但也不证明它们在 Ascend/AFD 已支持。
新增 Engram 准入、stage/图/历史缺口仍是中优先级，先完成契约与架构审查再实施；
Engram+U2/DBO 保留为最终适配目标，不能仅删校验或调整启动参数视作实现。
上述依赖是逐场景模型验收前置，不阻止独立组件和 CPU 工作。Engram 工作归属
[tasks 2.1–2.7](tasks.md#engram-medium)，原生 U1 复用验证与新增适配分阶段记录；
混合任务仍满足全部条件才勾选。旧 MTP 后续开发继续暂停。
D03/D04 的普通 MTP、D12/D13 的 PD patch、D14/D15 的 recipe 是语义参考，不能直接复制旧 v0.23 API。
本阶段是必交付，不把 PD、Graph+DSpark 再列为无主延期项。

## DSpark 和普通 MTP 映射

V4.1 使用 RFC DeepseekV41DSparkModel，完整 draft 留 A，target Graph/U2 + eager draft 是最终必验目标；
近期先核对和验证 DSpark/U1 的真实 proposal/接受拒绝及 Engram history，Graph/U2 组合待新增适配闭合。
若目标原生提供可捕获 draft，再按该接口验证，不伪造旧远程 MTP。
每 stage 在正确入口采集 target aux 并归并，draft 不发布 F 工作、不覆盖 target Graph 状态。
microbatch、block size、proposal、draft stage 数和远程 MoE 次数分开。

已注册 DSpark PD owner 的异常边界覆盖 target execute 与后续原生 sample/proposal。
采样/proposal 抛错时保留已有 history/KV 并将 consumer 置为 terminal，禁止下一次
execute 或消费已有待采样状态；没有待采样状态时沿用原生 None。原异常继续传播，
不新增回滚、重建、collective 或资源释放。AsyncOutput 返回后的错误仍由原生
engine/worker 监督，不能据本边界宣称多 rank 故障恢复或设备验收。实现与 CPU
范围见 [采样失败审查](evidence/dspark-pd-sampling-failure-review-20261112.md)。
原生 AsyncOutput/UniProc/EngineCore 已有输出完成错误传播，新增 CPU 契约验证
失败批次不提交成功输出并保留已推进 history/KV，不新增重复包装。同 ID 第二次
远端 PD 读取验证新 generation/lease 与旧消息隔离。真实 HCCL 的 idle STOP/ACK
及全新进程重启另由 A3 组件验证，不能拼接为完整模型故障恢复验收，见
[异步输出与请求复用](evidence/dspark-output-failure-review-20261112.md)。

PD receiver 的失败状态还必须约束已经进入 native peer 队列的后续任务。原生 peer
循环捕获异常后继续队列，因此仅关闭新批次 admission 不足以形成 terminal 边界。
插件在每个任务进入既有 owner 锁后检查 sticky failure，早于 DSpark GETREADY、
lease、read 与 finished 发布；普通 admission 关闭仍允许已接受任务正常 drain。
其他 peer 已经开始的 read 不由此取消，失败分配和 lease 保留给既有进程监督，
不新增 abort/rollback/cleanup。见[排队失败审查](evidence/pd-retained-failure-review-20261113.md)。

逐分支登记供体普通 MTP N 次（有效范围 1–3）、merged stage0、phase-ready/header、
eager/target graph/full-draft graph 的去向。
对原生支持普通 MTP 的既有模型保留适用能力和回归，迁移完整次数、形状和生命周期。
V4.1 原生只接受 DSpark，普通 MTP 只能标为有源码证据的模型语义映射，不强行新增 V4.1 MTP；
不能一句“不需要旧 MTP”就删除公共 connector 能力。
图 key 含 speculative 签名；N 写进 metadata 不会自动让一次捕获交换变成 N 次。

## PD 五组合

本次验收包含：
1. PD+AFD eager/U1。
2. PD+AFD+DSpark eager/U1。
3. PD+AFD Graph/U2。
4. PD+AFD+DSpark eager/U2。
5. PD+AFD+DSpark Graph/U2。

同时保留 standalone eager/U1/U2、Graph/U1/U2。
PD 的 KV connector 与 A/F activation HCCL 分离，F 不因 PD 拥有请求 KV/Engram 历史。
核对 Prefill→Decode shared-KV/index、compressor/SWA、Engram history、prefix 和 target/draft cache owner。
按逻辑 layer/component、dtype、stride/block geometry 对齐，不能 zip 异构 group/注册数组。
旧 dspark_pd.py/dspark_pd_layout.py 有 v0.23 特定逻辑/版本门禁，先检查 v0.30 等价行为；
已被上游覆盖的补丁不重复，缺失行为只在 afd-plugin 最小重建，不修改上游或单放开版本判断。
若原生 PD metadata 无法传递所需 Engram history，保留为最终验收阻断，按中优先级
tasks 2.6 后置解决；继续其他 PD/KV/index 组件工作，不假装该依赖不适用或已完成。
记录 HIXL/本地资源路径、proxy/listener/端口/路由与 P/A/F 锁定环境；healthcheck 不能替代请求成功。

## 操作、观测和完成条件

迁移 recipe 模式、U2 阈值、rank/device/TP 映射、checkpoint 族检查、capture-size 推导/覆盖。
恢复所有 rank overlap/diagnostic 生效值记录、A/F profiler 启停/step/按 rank trace；
管理 RPC 不因 F 常驻 recv 阻塞，按 v0.30 实际接口适配。
部署观测由独立标准库 CLI 读取冻结 recipe 与逐 rank 原始记录，校验八项配置、
协商坐标，并可按显式 expected transaction 清单检查 eager/U2 首末层的 host
事件因果链、空 stage 和 parent 完成。Graph/serial 只作配置汇总，不能用 Python
capture 日志证明 replay。记录没有 session/PID/source 绑定，run 归属和设备可见
映射由调用方保证；payload、真实设备完成及物理重叠各自需要独立证据。见
[观测审查](evidence/hccl-observations-review-20261113.md)。实际 prepared argv 的
完整模型启动与完整逐模式观测尚未闭合，1.6 不因该组件汇总而勾选。
锁定 native executor 顺序消费未带请求 ID 的回复；profiler/start/poll/STOP RPC
失败或中断后将 F 管理通道置为失效，禁止后续 collective 清理并由原生 executor
监督进程。正常已经 ACK 的 peer STOP 流程保留；错误通道不能发出清理成功日志。
Unix/UniProc/F daemon/HCCL/profiler 已有 A3 启停、同 owner 重启和 active STOP
导出的有限组件证据；五秒 join 仅有该规模实测，没有普遍耗时保证。后续实际
EngineCore busy-loop/shutdown 与 MP IPC 成功链也已完成 A3 组件验证，见
[EngineCore 审查](evidence/h14-engine-core-review-20261112.md)；原生构造/前端、
失败注入与完整模型仍待验收。
冷加载、prefill/decode、并发/取消/随后请求、STOP/错误后全新进程重启覆盖代表 eager 与 Graph/U2 格子。
设备失败保留 terminal+进程监督，不宣称错误 HCCL 组同进程自愈。

逐行填写能力矩阵的实现/等价映射、测试/日志和支持状态；源码有分支/历史 A5 只证明供体存在该行为。
同硬件同请求比较 native、eager/U1、eager/U2 开/关、Graph/U2 全开/全关，再测 DSpark/PD。
报告 TTFT/TPOT、总吞吐、真实 acceptance、实际卡数/每卡效率、HBM/冷捕获成本。
未验收格子保持 pending，不通过缩小矩阵或 C1 完成来宣告整体完成。

## 2026-11-20：TP1 / DP / EP 架构增量

用户授权参考锁定供体 `4ddd69a3` 开发 EP。复用现有 v3 staged HCCL 传输和
原生 DeepseekV41MoE，不复制旧 V1 runner，不修改 vLLM / Ascend。
首版为对称 A/F、TP1、DP=角色 rank 数、eager/U1；真实权重资格入口固定 A4/F4。

- HELLO 增加默认关闭的 `expert_parallel`；所有 peers 必须一致，EP-off 字节保持。
- EP 要求 local counts v1、max_stages=1、Graph/pipeline=0；不支持的组合建组前拒绝。
- AFD role_rank 和原生 WORLD rank 都包含 DP；engine-local `parallel.rank` 保持 0。
- A0 收集每个 A 的真实计数，所有 F 从同一 parent 推导完整角色 DP token 向量。
- 同一原生 DP wave 可以混合 target 和 idle dummy，包括 A0 空闲的情况。全局 plan
  有 target 才选 target，否则为 dummy；idle dummy 必须 valid=0。profile/warmup 严格一致。
- 仅全局空 stage 可以跳过；局部无有效 token 的 F 仍调用原生 MoE，参与 EP 通信。
- 原生 MoE 管理专家分片、shared experts、FP32 gate 和 EP 通信；AFD 不重复 reduce。
- A 保留 HC/pre_mix、Engram ModelState、shared KV/index 和 U1 DPMetadata；CPU Engram
  使用原生 DP 共享。F 的 DSpark/PD 原始声明在隔离前拒绝，不能剥离后误准入。
- 原生 TP1/DP4/EP4 对照与 F4 对齐；TP1 不使用 TP4 的 16→64 适配或其专项 probe。

原生 manager.monitor_engine_liveness 在第一个 engine 退出后终止其余 engines。
EP launcher 改为等待所有本地 F 正常退出，首个成功后用 shutdown timeout 限制剩余清理；
失败和超时仍交由原生 client 清理。真实 EP 集合通信退出证据保持待 A5 验收。

代价为每个 parent 的现有 v3 counts/control barrier 和完整 hidden/FP32/ID 传输；
F 的 DP tensor 每 stage 构造一次。EP all-to-all 由原生选择；HBM 峰值需要实测。
本增量修改 AFD 自有边界，无新增上游函数复制或猴补丁。若原生 runner 提供可直接
消费的 A/F MoE 服务边界，再迁移并删除自有同步 glue。CPU 合约通过不代表设备验收。

## EP + Graph/U1

单节点 A4/F4、TP1/DP4/EP4 的完整权重资格入口加入 FULL_DECODE_ONLY。只使用
固定的 U1 target decode bucket。捕获和回放要求所有 A/F rank 接受相同的 graph key。
F 的 bank 绑定相同编号的 A peer。每个 F 在捕获前用同一 parent plan 构造完整 DP
token 向量，原生 MoE 使用该向量执行专家通信。A 的 U1 child 和 parent 保留固定
bucket 的原生 DPMetadata，Engram 输入准备和退休由 A 的 ModelState 持有。

运行时先由原生 DP dispatcher 同步模式与 bucket，A/F 再协商缓存可用性。
`GRAPH_QUERY.eager_padded_tokens=null` 表示本次 EP runtime 回退使用各 rank 实际行数，
沿用随后已有的 counts 往返。捕获阶段、捕获列表预检和 EP-off 的行数仍为精确数组。
有 idle dummy 时走 eager，空闲的 A0 仍能协调。若原生 FULL 被缓存协商降级，
所有 A 在输入准备前用同一个原生 dispatcher 再进行一次 eager DP 同步，代价为该次
回退增加一轮原生 DP collective。原生 eager 已同步时直接保留其 metadata。

补丁复核对应 vLLM `ced6857a` 的 `vllm/v1/worker/gpu/dp_utils.py`
`dispatch_cg_and_sync_dp`，保持原 13 参数与返回约定。U1 准备沿用 Ascend
`7e37e29a` 的 `EngramModelState.prepare_attn` 签名与现有委托，固定向量与
`worker/v2/aclgraph_utils.py` 中的原生捕获形状对齐。上游提供可保留最终 DP
metadata 的预准备回调后，可移除相应作用域。设备集体通信回放与时序由 A5 验证。

## EP + 串行 U2、DSpark 和 PD

EP 支持 eager 与 FULL_DECODE_ONLY 的 U1/U2。U2 的 HCCL 包序为逐层、逐 stage，
同一 stage 上各 F 使用同一份 DP padded-token 向量执行原生 MoE。单个 DP rank
零有效行时仍参与。Graph 在捕获和回放前协商所有 A/F rank 的 stage 数和静态桶，
缓存不一致或有 idle dummy 时统一走 eager。U2 的回退在输入准备前通过第二次
原生 DP 同步确定 stage 数，随后从各 A 的真实行数构造 HCCL plan。额外的原生
同步只出现在此回退分支。A 保留 parent Engram producer，由现有 child 切片使用。

固定 DSpark N=1/2 复用锁定 Ascend 的 eager proposer、完整 native InputBatch 和
parent DP 同步结果。A 保存 target aux 并完成 proposal、采样和 history 退休。
Graph 的 target 桶按 1+N 行对齐，draft 仍为 eager。F 保留 target 宽度用于
捕获准入，并在构造前隔离 draft 与 Engram。

单节点 PD 以 P eager/U1 导出 KV、index 和已提交的 Engram history。P/D 的 TP、
DP 与 PP 在准入时检查，DSpark draft schema 记录 EP 的 DP 宽度。D 的 MRV2
作用域保留原生跨 DP 同步，避免 Mooncake 默认优化在 EP 所需的 target/draft
同步前跳过集合操作。F 仅执行 target MoE。验收用不同的 P/D DP rank 验证
实际传输，D 可选 eager/Graph U1/U2。

补丁审查对照 vLLM `ced6857a` 的 `vllm/v1/worker/gpu/dp_utils.py` 和
MRV2 对 `should_skip_allreduce_across_dp_group` 的导入别名。仅对通过准入的
EP Mooncake D 强制保留原生同步，其余配置调用锁定实现。后续若原生 PD
可按 EP 消费者保留 target/draft 同步，移除该作用域。正常 decode 的串行
stage 无额外 stream/event；stage padding、第二次同步及 PD 数据传输的
设备耗时由 A5 测量。
