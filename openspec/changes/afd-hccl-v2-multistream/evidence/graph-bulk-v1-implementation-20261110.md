# Graph bulk_v1 同层输入前沿：实现契约

日期：2026-11-10。基线 HEAD：`c279c4f6f95445b9e764d2487829af39b3a757a3`。
这是实现者的源码自审，独立审查和实际依赖 CPU 结果另行附录。
完整 20 个源码/镜像/共享 fake 的 SHA256 见
`graph-bulk-v1-local-validation-20261110.json`。本轮不改 upstream、生产模型 guard、
默认连接器选择、环境变量读取，也不提交或推送。

## 行为和调用者

现有 `HcclGraphEndpoint` 已提供图内 send/receive 算子；本轮消费这些端点，
没有新增底层 batch API。HCCL 算子全部保留在 capture parent stream。
新 DAG 仅在显式协商 `graph_exchange_order="bulk_v1"`、Graph U2 且存在实际
compute pipeline 时成立。它为同层 F 接收下一 stage 输入时排入上一 stage
计算提供依赖结构，不等于物理 send/recv 侧流或完整跨层流水实现。

以两个 positive stage 为例，单层 parent 端点顺序是：

| 角色 | Parent HCCL 顺序 | Compute stream |
| --- | --- | --- |
| A | send BF16/FP32/IDs S0 → send BF16/FP32/IDs S1 → recv output S0 → recv output S1 | 先构造本层所有 stage；下一层逐 stage 等自己的上一层 receive，或沿 hybrid=false 的整层 parent fence |
| F | recv BF16/FP32/IDs S0 → recv BF16/FP32/IDs S1 → send output S0 → send output S1 | recv S0 后排入 compute S0；随后排入 recv S1/compute S1，最后 parent 才等 compute S0 发结果 |

A 的下一层仍等上一层所有 wire 已在 host 上排入，F 的下一层仍在本层全部
send 排入后开始。`ffn_recv_stream`/`ffn_cross_layer` 原有 event/reuse/join
策略继续有效，但当前 bulk 顺序不据此升级为跨层实际通信重叠。

实际调用链：

* A：coordinator `_before_factory` → 协商的 reservation → bank/pipeline →
  `HcclV41GraphExecution` → 单次 native `ModelWithContext`/GraphParent →
  `DeepseekV41GraphStages._run_pipelined` → 全 `attention_send` → 全
  `attention_receive` → continuation/tail join → 原生父级 HC/MTP/aux merge。
* F：GraphService `_resolve`/`_prepare_resources` → bank/pipeline →
  `HcclV41FFNGraph.run` → 全 receive+native `compute_ffn_output` 排入 → 全
  `_send_output` → 原有 layer/final join → owner 在 body 外完成事务。

## Wire 和 session 固定身份

`HcclPlanCapabilities.graph_exchange_order` 默认 `stage_serial`。默认值不编码
到 HELLO，因此既有 serial HELLO 的字节和字段集合保持不变。显式 bulk 需要
Graph/counts v1、两个 stage 的容量、paired native TP1 或 TP2；能力的原有
counts 约束同时要求 A=F=nativeTP。双方必须完全相等，不取最低共同值。

原有 `HcclStagedTransport._hello` 直接消费新的 `negotiate`，不需要修改其
状态机：coordinator 收齐各 rank 的完整 HELLO 帧后统一拒绝错误，HELLO
成功才创建 stage HCCL groups。旧 strict reader 对新可选字段拒绝，不会把
bulk 当 serial 接受。原有无效 frame 长度/后端断连的监督边界保持不变。

bank 的 `exchange_order` 是由完整 negotiated capabilities 和 key 推导的只读
属性。U1 key 总是 `stage_serial`，eager fallback 也继续走现有 eager executor。
pipeline 在构造时冻结完整 capabilities，coordinator/service 同样保存 session
能力值，后续改 wire 顺序会在新 QUERY/prepare 前拒绝；缓存资源继续要求 exact
bank/body/pipeline/session/group/key/storage 归属。改变顺序需要新的 transport、
cache 和 native manager，generation miss 不授权同 key 重捕获。

启动 preflight 只将完整有序 capture list 的摘要和首 key 发给 F。因此 bulk
session 的所有 A/F 必须预先具备 U2 compute 能力（显式 policy，或 settings
compute=true），即使第一张图是 U1。缺能力时诚实报告 unsupported，所有
peer 统一执行并退休零行 EAGER parent，跳过捕获；不让首个 U1 图掩盖后续
U2 不支持。实际 U1 graph body 依旧串行，可无 side pipeline。U1-only 且不
需要 bulk 的调用者应保持默认 `stage_serial`。直接缺 pipeline 的 U2 body
也会在 producer/wire 前拒绝，factory 边界有同样的防御。

## 事件、payload 和异常

bank 仍每 stage 持有 BF16、原生 FP32、int32 IDs、output 四个 disjoint 固定
payload；FP32 不从 BF16 重建。`attention_send` 沿原路径检查 shape/dtype/
device/contiguous，copy 原输入后仅将 padding IDs 置零。F 原生 MoE 结果继续
copy 到 output bank 并 mask padding，A `attention_receive` 返回同一 borrowed
output tensor。positive padding-only stage 仍全部执行；静态零行 stage 不排
wire、compute、record/wait event。

新增 pipeline split scope 严格约束：A 全 stage compute 后才按序 send，全
stage send 后才按序 receive；F receive 和 send 分别按 layer/stage 次序，
本层全部 receive/compute 排入后才允许 send。原 serial combined scope 在
bulk U2 下拒绝。事件全部在图外预分配，每个 event 每次 host 录制只 record
一次，保留原有最终 join；调度器在 run 内不发送 control wire，不分配
stream/event/通信 buffer，也不做 query、item 或 device synchronize。

bulk A 将新产出的 x/原生 FP32、positions、IDs、Engram lookup/mask、hidden、
pre_mix、borrowed output 和 continuation 四个 tensor 交给 pipeline 的强引用/
record_stream 前沿。F 的每 stage 原生 output 在其 deferred send 之前由
`retain_compute_outputs` 保持，send 后沿 parent consumer 保持。前一个 stage
output 不会因第二个 stage compute 失败而被释放。失败 run 为 sticky terminal，
保留 stream/event/bank/当前临时 tensor，外层 cache/session failure 同样保留
资源供监督；不宣称取消 HCCL，也不在 tensor 部分失败后改为 eager 重试。

与 serial 相比，F 本层最多同时保留 U 个原生 MoE 输出，而非收到一个即发送
一个；现有四 payload bank 不扩容。新增 event scope 复用原已分配的 per-step
send/receive events，没有额外通信 stream。实际 HBM、allocator 压力和吞吐
成本尚需设备/模型测量。默认 serial 不增加这些延迟输出的 retention 调用。

## 验证边界

本地已运行 20 文件 Ruff、Python 3.11 AST，以及七个镜像的 CPU-only collect：
147 个纯 codec 测试通过，六个 runtime 镜像因缺 torch 被 importorskip。
该结果不验证 torch/NPU 事件、Graph 算子或 HCCL。

新增有意义的 seam 覆盖包括 HELLO 默认字段省略/严格不降级、非法 order/
unsupported topology、A/F 成对 packet 次序、每层最后一次 receive 早于第一次
compute→parent send wait、U1 仍 serial、空 stage、原始 FP32/IDs/padding、
native continuation/aux/MTP CPU 输出、缺 pipeline 的 preflight、session 内改
order 和失败时 deferred output 强引用。CPU host recorder 只记录调用和实际
CPU tensor 运算，不模拟 device 调度。

实际依赖 CPU 已保存 `a3-graph-bulk-20261110/graph-bulk-019a`：552 通过、
一个旧 service 夹具失败。改为真实 oversized-peer key 后，独立 019b 的
service 64 项全通过；生产源码未变。分模块/原失败/最终定向结果见该目录
README 与 validation-summary，不拼接成一次总通过。独立 source review
另行记录。随后 root 独立审查并授权的单个 `bulk-tp1-001` NPU 会话首次通过：
devices0,2、axes11111、31 pins match、8 卡前后 idle、无 timeout/强杀，双方
各 4 warmup/4 capture/12 replay/6 eager fallback，native TP 前后 `[1.0]`。
实际跨 stage group 的 bulk parent 顺序和图捕获/回放仅在该受控算术组件范围
得到证据；numeric parent handle、timeline、TP2 bulk 和静态零行 stage 的
device 用例均不在本次报告内。详见 `a3-graph-bulk-20261110/README.md`。
设备门禁需先复核 A3 idle/source pins/受监督窗口，明确全角色配对，然后单独
验证 bulk warmup/capture/replay、同 stage 双层复用、padding/empty/桶切换和
正常 STOP。现有 serial 或 eager isend 证据不能替代 bulk 的跨 stage HCCL
enqueue/Graph 合法性。timeline、native V4.1 HC/Engram/MoE 和完整 checkpoint
均独立验收，当前不将 C6 1.6/1.10/1.11/1.12 标为完成。
