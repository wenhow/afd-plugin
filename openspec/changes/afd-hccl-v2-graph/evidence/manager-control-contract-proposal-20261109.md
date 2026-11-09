# C5 native manager 接入：控制契约提案（2026-11-09）

状态：只读设计提案，尚未冻结 wire/schema，也没有修改 C4 counts cap1 或运行时代码。
已选定保守两轮时序（pre-prepare mode/key 决策，再 post-prepare actual counts 核对），
先保证正确性，额外 host 控制轮次的成本留待后续实测优化。
前置通信组件的最终验收见 [native-002](a3-static-graph-20261109/native-002/README.md)。
本提案把下一步实际消费者定为 native manager / F graph cache，不先增加闲置 codec。
early owner、FIA padding、request layout 与 slot storage 的精确源码依据见
[pre-prepare 独立审查](pre-prepare-dispatch-contract-review-20261109.md)。

## 1. 必须早于 Engram 准备决定 fallback

锁定 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` 的
`vllm/v1/worker/gpu/model_runner.py:1685-1700` 先 dispatch，1710 起准备 InputBatch，
1790 调用 U2 prepare，1862 调用 ModelState.prepare_inputs，1901-1909 FULL 分支直接
run_fullgraph；eager 的 set_forward_context 在其后另一个分支。FULL replay 不会再调用
Python 模型/现有 C4 parent-provider，也不能靠现有 provider 自动发布 parent。

Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0` 的 EngramModelState
`engram/model_state.py:130-141` 在支持 captured producer 的 FULL route 只写固定 query/
valid coordinates，不创建 eager `_engram_inputs`。因此在 C4 post-prepare 的 submit
时才发现某个 F 缺图，并把执行模式临时改 NONE，会缺 producer；不能作为 fallback 实现。

建议分成两个图外阶段：**dispatch 后、ModelState/attention preparation 前**先对
candidate phase/key/availability 达成共识；然后仍由 native 准备真正输入/metadata，
沿用真实 local counts round 校验实际 stage layout。图 miss 的统一 eager 决策必须先于
第一阶段结束。无法在早期形成可信 key 的请求直接选择 eager，不能先准备 FULL 再猜。

U2 额外约束：`gpu/ubatch_utils.py:49-75` 的 create_ubatch_slices 使用真实
num_scheduled_tokens / padded tokens / padded requests；common
`vllm/v1/worker/ubatch_utils.py:166-217` 的 token split 是 padded/U midpoint，request
slice 仍依赖真实 query 分布。不能只复制 A0 counts，也不能把 scheduler dict 顺序假定为
native batch 顺序。

独立追查找到更小的实际来源：GPU `model_runner.py:1217-1267` 的
`gather_batch_req_state(self, scheduler_output: SchedulerOutput, dummy_run: bool)
-> tuple["BatchReqState | None", int | None]` 在 dispatch 之前返回 batch-order
`num_scheduled_tokens, num_tokens, req_ids, idx_mapping_np`。Ascend override
`worker/v2/model_runner.py:615-636` 仅额外处理本范围已排除的 PD 情况，签名无返回注解。
可在 execute scope 记录该**原实例真实返回**，随后在 imported dispatch seam 用 native
pure slicing helper 预览；无需重新排序或虚构 all-ones batch。dummy 返回 None，必须按
原生 dummy/capture 规则另行处理。预览与真正 native prepare 的相等验证仍是必要边界。

初期 spec off / CP1 / PD off / FULL uniform q=1 下，真实 `BatchReqState` 与 descriptor
可在 metadata 副作用之前计算：`T=max(state.num_tokens,desc.num_tokens)`，
`R=max(len(state.req_ids),desc.num_reqs or 0)`。调用相同的
`maybe_create_ubatch_slices(True,state.num_scheduled_tokens,T,R,U)`，再保留原生
create_ubatch_slices 的 request-start clamp 到 realR-1、request-stop clamp 到 paddedR。
Ascend FIA pad 与 slot 扩展需要同时遵守：非 uniform 不直接套用此 R 公式，提前 eager。
U1 用 parent 单 slice。真正调用的 owner 仍是 Ascend 原实例方法，没有复制 GPU gather
或多执行一次 gather；返回 tuple 的原生字段保留给后续 execute 使用。

具体反例已由独立源码审查确认：realR=4、paddedT/R=8、U2 时，trailing stage 的
request slice 为 `3:8`（5行）；capture dummy8requests 的同一阶段为 `4:8`（4行）。
即使总 T=8、U=2、各 stage token rows 一样，也必须 key miss/提前 eager。仅检查
`2*min(raw_tokens) >= bucket` 或 native descriptor token/U 不能覆盖这个 layout 差异。
graph key 应保留实际 stage request_start/stop 与 token_start/stop；slot/block 的值由
native 在 persistent storage 原位更新，key 另绑定 storage generation，禁止跨重分配。

descriptor 是 frozen dataclass，最终模式需由 dispatch 返回正确新对象，不能在 prepare
wrapper 事后改名。现 C4 `use_async_cam_dbo_dispatch` 的 DP1 路径在 U1 结果上仅替换
`num_ubatches=2`，因为只支持 eager；C5 必须先决定 U2 eligibility，再真正调用
`manager.dispatch(..., num_ubatches=2)` 选择已捕获 U2 entry（参照锁定 native
`gpu/dp_utils.py:123-129`）。不能把 U1 cache hit 改名为 U2。need_eager/profile 仍强制
NONE。global miss 时按原生 forced-eager 规则返回 descriptor，保留真实 U1/U2 threshold
结果，不能用一次 `cg_mode=NONE` 替换掩盖 request/query/padding 改变。

## 2. 拟冻结的数据与能力边界

不把 `phase` 混入现有 parent.mode：mode 继续表达 target/dummy/profile 等执行用途；
phase 单独表达 eager/warmup/capture/replay。每次 native warmup/capture 都是新 parent
transaction，即使 descriptor/key 相同也不能跳过对应 F 工作。

建议新增严格协商 `graph_execution_version=0/1`：0 不编码额外字段、保留 C4 bytes；
1 必须所有 peer 相同，不能 min 降级。初期继续 DP1、PP1、CP1、paired native TP、
text/BF16、spec off、FULL_DECODE_ONLY，不扩 production guard。

候选对象（仅设计名，不是已存在 Python API）：

- `GraphCandidate`：requested phase、完整 native descriptor 投影、canonical graph key、
  本 rank cache availability（ready/missing/unsupported）。key 为 None 时只能 eager。
- `GraphDecision`：reservation/transaction ID、统一 phase、选定 key（eager 时可无）、
  descriptor/布局摘要、fallback reason；不能仅广播一枚 bool。
- `GraphExecution`：将最终 decision 绑定到实际 HcclParentPlan、包含 phase/key 的总 digest。
  native actual counts 必须匹配 reserved layout 才可 ACCEPT。

key 至少区分 descriptor 的 U/token/request/query/LoRA 字段、native TP、target layer
集合、payload版本/dtypes、每 stage peer/rank/offset/padded rows，以及影响图输入形状的
stage request/query 布局。valid/history/IDs 若纯 tensor data 不入 key，但图内消费地址和
shape 必须固定。全空 stage skip pattern 入 key；valid0/padded>0 仍有通信。

cache entry 自身保有 graph + exchanges/HC continuation + Engram/metadata views/events
的完整 lifetime bundle。availability=ready 只表示**完全捕获、同 generation、未失败且
仍持有全部 buffers**的 key；pending capture 或图对象存在均不够。

## 3. solicit-first 顺序与 STOP/错误

需扩展现有 idle service 的可识别控制状态。拟新增一个 QUERY / REPORT 回合，之后仍用
带 digest 的 FORWARD / all-peer ACCEPT / global ACCEPT。所有新字段/kind 只在
graph_execution_version=1 使用，旧 C4 service 不接收半个 Graph round。

1. 非 A0 的 A 与 F 都先等 QUERY 或 STOP，不主动发送 availability。
2. A0 广播有界 QUERY（reservation ID、candidate phase/key/descriptor）。各 A 对照自己
   native candidate，F runner 通过自身 graph cache 回答；transport 不拥有模型/cache。
3. A0 drain 每个完整 REPORT 后决定模式。replay 要求所有 rank 同 key ready；任一 miss/
   unsupported 均统一 eager。A candidate descriptor 不一致属于配置/状态错误，不能当
   普通 cache miss 隐藏。
4. 统一 DECISION 必须 ACK/全局提交后才允许 native 按选定模式准备。随后真实 counts
   round 形成最终 parent plan；每个 A 再核自己的 column/实际 layout。final digest
   覆盖 reservation、phase、key 和完整 actual plan，不能只用原 counts digest。
5. warmup/capture 必须两端都执行相应 body。capture 中出现“某端已有 key”时，首轮
   选择提前拒绝或显式统一 recapture generation；不能局部 return 或混合 capture/replay。
6. 只有 idle 边界允许 STOP。STOP-before-QUERY 复用现有所有 A/F 两阶段 STOP/ACK；
   已提交 reservation 后 STOP 属于误序，禁止有的 rank 先退而有的继续准备/capture。

完整错误帧按已有 drain→global ERROR 策略；坏 frame length/断链或 capture/设备异常
置 terminal 并交 watchdog/进程监督。图内没有 JSON/ACK/sync。不可在一端 capture
失败后假称通过不对称 eager fallback 恢复同一 session。

reserve 与最终 counts round 之间若 native 准备失败，peer 可能正等第二回合；首轮按
terminal supervision 收尾，不引入未经证明的网络 abort。若要支持可恢复 reservation
cancel，需另定完整握手和负例，不顺带加一条单向 CANCEL。

## 4. capture/replay 对接方案与 host cursor

native CudaGraphManager.capture（`gpu/cudagraph_utils.py:398-488`）先调用
create_forward_fn(desc,warmup=True)，执行 warmup，再用 fresh factory 准备 capture。
不能在返回的 forward_fn 内执行 parent COMPLETE，因为实际 capture 时它运行在图内。

一个可审查的小 seam：在 capture 的 factory 边界完成**前一个** phase（若有），再
委托 native fresh preparation，为当前 phase reserve/submit，返回仅含 tensor body 的
callback。下一次 factory 进入时，或整个 native capture 成功返回后，才在图外同步并
提交前一个 phase 的 COMPLETE。必须在 fresh preparation 写 buffer 前完成前一 phase，
不能到新的输入准备后才等旧图。异常不发送正常 COMPLETE。上游 wrapper 的精确签名、
临时绑定恢复和 delegation 例外需要另做 patch review；本轮不实现全局 monkeypatch。

U2 原生 begin_capturable_run 启动 full-model handoff threads，不能直接套在现有同步
V4.1 stage body 上。新同步可捕获 callback 应保留 native per-stage prepare 和 capture
stream，但不启动重复 full-model threads，也不调用 C4 host-cursor transport methods。
这是模型/UBatch owner 的增量，不属于 tensor exchange 的自动能力。

FULL replay 的图外 hook需先取得已接受的 GraphExecution，原位更新固定输入/metadata，
委托 native manager 的 producer/replay/attention更新/retirement，再将本地 captured
schedule 标为完成，最后 parent COMPLETE。host cursor 只能由 validated graph schedule
**整体**提交一次；禁止伪造逐层 send/receive，也不能允许 eager cursor 部分前进后再 replay。
device completion 仍必须在 COMPLETE 之前证明。图内不推进 Python iterator。

## 5. Engram 与下一轮可检查行为

保留 native EngramModelState history/coordinates 与 Ascend replay owner。支持的
captured producer 应每个父图一次，再对 stage 切 lookup/mask；U2 不重复消费同一个
ExternalEvent wait/reset pair。graph-external producer route 必须证明唯一消费点和错误
retirement。不能仅捕获独立 MoE 后宣称 V4.1 graph。

建议下一轮先做有实际 manager/cache consumer 的 phase agreement 软件 seam，覆盖：
某 F 缺图→所有 A 在 FULL准备前 eager；A1 descriptor 不同→全体拒绝；重复 capture key；
warmup/capture 各一次配对；同 total 不同 stage/peer/request布局 miss；STOP-before-query；
完整坏 REPORT drain；reservation 后 native prepare 异常保留资源；replay 不调用模型
Python仍完成 host cursor/COMPLETE。上述行为测试不得断言内部 helper call counts。

只有 protocol/API 经独立审查冻结、early mode owner 被精确定位后，才改 C4 稳定代码。
随后分开验 native manager、Dynamo/GE、真实 V4.1 producer/attention；A5 手工结果回来后
统一做完整 weights/model 验收。当前通信组件的 native NPUGraph 通过不勾这些任务。
