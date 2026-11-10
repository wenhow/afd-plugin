# H12/H13：普通 MTP 与 PD 组合的锁版源码缺口审计

本轮仅只读审查锁定 Git 对象，新增本文及配套 JSON；没有修改运行代码、上游、资格门禁或任务状态，没有导入 native 模块、启动设备或执行测试。结论是实现接口与所有权审计，不是 MTP、PD、Graph PD 或全模型验收。

源编号、完整文件 SHA256、AST 规范化签名和带原行号摘录见 [配套 source JSON](hccl-combination-source-gap-20261110.json)。V=`ced6857afa0ea7b2e3f0846a62e1394e90f15607`；A=`7e37e29a7803d6897a7113088e2b771f83205bc0`；D=`4ddd69a391dfb5b3e362c7601399e538a31de702`。当前插件比较基线 P 单独固定为已发布 `c153a4f372aad238f1849082408b5295996572a2`，不把并行开发的 DSpark Graph phase1/phase2 工作树混入这个快照；phase1 的独立实现和 CPU 历史见 [单独审查](dspark-target-shape-independent-review-20261110.md)。

## 可保留的原生职责和仍需实现的接口

| 项目 | 锁版已有职责 | 当前真实差异 / 最小下一项 |
| --- | --- | --- |
| async PD speculative lookahead | V01 `scheduler.py:1131–1137` 在异步 KV load 时将 effective lookahead 置零 | donor D05 的 v0.23 allocation patch 已有原生等价，不迁移 |
| hybrid KV load failure | V01 `:3087–3109` 遍历所有 KV groups、排除 null block、整请求回退；`:3166–3179` 还有 request-ID failed-recving 路径；`:3181–3250` 保留 fail/recompute 策略 | 不复活 donor group0-only 补丁；以后应以真实 connector 返回的 failure 类型验证原生路径 |
| V4.1 shared KV/index/alias 注册 | A06 `cache_config.py:147–178,248–274` 定义 physical page；A09 `_reconstruct_shared_pages:114–175` 结合 descriptor 和实际 tensor views，`:1931–1980` 注册整页 | 已能复用本地注册，不应把 index/scale 的 inner view 当独立整页；仍缺远端具名语义布局一致性核对 |
| Engram 请求历史 | V01 `:1324–1334` 对新/恢复请求传完整 `_all_token_ids`；V02 `:1120–1158` 建立 request state；A08 `prepare_engram_inputs:99–154` 从历史与 computed-token 数生成 lookback | 不新增 F-side history/Engram，也没有证据要求把 Engram lookup 临时 tensor 作为 PD KV 发送；需实际验证入口完整历史、重试/恢复和 native 一次退休 |
| target/draft sampling 状态 | V02 `sample_tokens:1997–2181` 持有 execute state、accepted/rejected、request postprocess、proposal 和最后 KV post-forward | 保持 A native owner；AFD target transaction 完成不等于 sampling / proposal 已完成 |
| V4.1 DSpark prompt draft-KV | A10/A04 有具名 draft groups、prompt chunks、generation、target+draft ready 和 TP join 前置条件 | 该通路由 `SfaRemoteD2HConnector` 选择；V4.1 当前压缩模型与其 D-side sparse-offload 资格冲突，不能直接放开 DSpark PD |
| 普通 MTP 完整 N 步 | V07/A03 已提供 V2 proposal、prefill/decode 两个 manager 和 Ascend merged FULL decode | 当前 P 没有 donor role-aware MTP model/远端 draft phase consumer；目标 DSpark 软件增量不构成 H12 普通 MTP 迁移 |

## H13：共享页已经原生处理，远端布局还没有证明

A06 `get_layer_tuples:147–178` 将 `long_kv_cache`、index cache、可选 folded index 放入有序页，同时包含 compressor state、target SWA 和可选 `.mtp.` draft SWA aliases；capacity 是各视图所需页空间的最大值。`get_deepseek_v41_kv_cache_config:248–274` 使用 `offset=0, layer_stride=0, block_stride=page_size`，同页 alias 与全局 block-ID pool 都属于 native allocator。A09 `register_kv_caches:1931–1980` 根据真实 runtime view 验证物理页基址，传输注册的是完整 padded page。插件不需要复制这些缓存，也不能按 attention 的“每层”数量另造独立地址数组。

但 A09 `MooncakeAgentMetadata:96–105` 只有 engine/endpoint、地址列表、block size/count、block lengths、SSM sizes 和 runtime ID；没有 component 名、dtype/stride、共享关系或逻辑 group 映射。`_validate_remote_runtime:1146–1155` 仅比较 Mooncake wheel/native binary runtime ID；`_get_remote_metadata:1157–1188` 不校验 P/D component layout。`_transfer_kv_cache_all_groups:813–884` 按本地 group、stride/length 和本地/远端地址数组位置构造读取。由此只能确认本地页组织正确，不能推出任意 P/D 配置的远端页面逐组件一致。

特别是 P 不加载 draft、D 加载 draft 时，A06 的 draft SWA aliases 和页 capacity 可能不同；不能仅以相同 checkpoint、TP 或压缩参数判定相同物理布局。当前 P01 `validate_deepseek_v41_pd:90–151` 的精确 MooncakeHybrid eager/no-spec/same-layout 配置资格，以及 `get_npu_ffn_config:155–192` 的 F private KV/spec/Engram 隔离，是受控配置边界；并非运行时远端布局验收。原全局模型门禁仍是另一层资格。

建议下一独立软件增量是**由实际 KV 配置和 materialized views 生成布局证明，并由真实 pre-transfer consumer 使用**，先保持 eager/no-spec 资格。可实施接口草案：

```python
collect_pd_layout(
    kv_cache_config: KVCacheConfig,
    kv_caches: Mapping[str, KVCacheValue],
) -> PdLayoutSnapshot

validate_pd_transfer(
    local: PdLayoutSnapshot,
    remote: PdLayoutSnapshot,
    blocks: PdTransferBlocks,
) -> PdTransferPlan
```

这两个名称和类型仅为建议，尚未新增运行代码。`KVCacheValue` 应精确表示 native tensor / tensor tuple；不可用任意 dict 或 checkpoint 配置假冒实际 tensor 布局。snapshot 分开保存可比较的语义与本地 opaque 地址：component 全名、逻辑 group、spec 类型、block-size/compression/window、dtype、每 block 形状与字节 stride、inner-page offset、共享页/alias 关系、实际可用 block 范围。跨进程地址值不要求相等，num_blocks 也不必相等；逐请求 src/dst block IDs 必须各自合法且完整覆盖。任何缺 component、次序/offset/stride/alias 不一致或越界都应在**首个数据读取之前**整体拒绝，不能边读边发现。

首个 observable consumer 可是独立 P/D 布局采集与比较工具，加上使用实际 typed `KVCacheConfig`/runtime CPU views 的 seam tests；它只证明布局判定行为。若作为 serving 保护，还必须接到实际 connector handshake/transfer 前沿，避免做未消费 codec。V08 `KVConnectorFactory.get_connector_class:95–128` 提供 `kv_connector_module_path` 外部 connector 扩展，第三参必须为真实 `kv_cache_config`；这比恢复 v0.23 全局 monkeypatch 更合适。A09 outer constructor `:1303–1318` 硬编码创建 inner worker，而 `register_kv_caches` 内部 `:1981–2037` 已启动 sender/receiver threads：只在 outer registration 返回后附加一个检查，无法自行证明 metadata 发布与传输线程的先后。若要实现 adapter，应先冻结 inner metadata schema/注册前 hook 的窄 patch 或显式 worker composition，标记复制差异并做独立架构审查；本轮没有证明一个零 patch 的 serving hook 已存在。

这项成本是每次 KV generation/registration 的 O(component 数) 布局收集、一次协商和每请求 O(block 数) 边界验证，不应进入每 token 算子体。未来上游若提供具名布局握手和 validation hook，应移除 plugin adapter，保留 native allocator/transfer ownership。

## H13：Graph / DSpark PD 不能由上述资格顺带开启

V03 `ActiveKVConnector` 的真实链路是 `pre_forward` 绑定 metadata、必要同步加载，`finish_forward`，再在 native sample 的尾部 `post_forward` 启动 pending async load、等待保存、返回 transfer/invalid-block/failed-request 状态并清 metadata；`no_forward:125–133` 专门处理只维护 KV 的调用。A13 `execute_model:503–532` 先处理 preemption，成功/失败均在 `finally` 调 ModelState `finish_execution`，成功 live 调用后再进入 `_maybe_send_draft_kv:555–581`。这些操作都不能放进 capture tensor body，也不能因为 Python model 在 replay 被跳过而省掉。

因此 Graph PD 的最小后续 seam 应验证真实 native execute/no-forward/sample 三个边界仍各自运行、metadata/slot/KV storage generation 被 capture key/owner 约束、加载完成前不开始模型、transfer failure/cancel 保留 native request retirement。先有这个 owner seam 和布局契约，才考虑精确组合资格。当前 P02 `hccl_config.py:269–274` 明确拒 Graph KV transfer，不能把 DSpark target query-length phase2 当作 PD 放行依据。

DSpark PD 还有独立的 draft 上下文协议。A10 `uses_sfa_dspark_kv_transfer:25–47` 仅选择 SFA connector（含 MultiConnector 子项）；`get_pd_dspark_aux_layer_ids:50–75` 要求 P eager、CP1、关闭本地 prefix caching。`send_dspark_prefill_kv:222–304` 只消费真实 prompt 行，检查有序 aux schema、chunk offset/generation、draft group block 覆盖，调用 A04 `initialize_local_context:169–213` 在 P 本地写 draft KV，最后才发 draft pages。`DSparkContextReceiver:363–451` 分别记录 target ready 与完整 draft prompt ready，处理 stale/duplicate/backpressure/failure，再由 connector 汇合 D TP ranks；不能用 target KV copy 完成或一次 `aux_hidden_states` 张量传输替代。

A11 `SfaRemoteD2HConnector:75–90` 要求 D sparse offload enabled；A12 `SparseKVOffloadConfig._validate_preconditions:1761–1762` 又拒绝带 `compress_ratios` 的模型，而 A07 V4.1 `build_layer_plan:522–573` 必需该字段。因此当前锁版原生 SFA DSpark-PD 接口不能直接用于 V4.1。后续要么由上游提供 V4.1 支持，要么为 plain MooncakeHybrid 做完整、具名且 generation-safe 的 draft-context adapter；两者都超出删除一个插件 guard 的范围。

Engram 本身继续保持 native 请求历史 owner：A08 从 `req_states.all_token_ids.gpu` 和 `num_computed_tokens.gpu` 取 lookback，FULL 模式只刷新 persistent producer inputs，eager 留待 forward-context 内执行，`finish_execution:65–72` 负责退休 lookup/events。P/D proxy 是否完整传递 prompt/已生成历史、resume/retry 是否保持一致必须由实际请求链验证；本次源码不证明某个外部 proxy 已做到。

## H12：MRV1 的单 runnable 不能代替 V2 的完整 proposal

V4.1 不存在 classic MTP checkpoint 结构：V05 `speculative.py:1344–1355` 明确拒绝并要求 DSpark。A01 `init_speculator:40–45` 先选择 `AscendDSparkSpeculator`，普通 `method='mtp'` 才选择 `AscendMTPSpeculator:58–67`。H12 的普通 MTP 迁移应限定 native 支持的旧模型，不把 V4.1 强行映射成 MTP，也不把 A-local DSpark proposal 记为 F remote-MTP 完成。

普通 MTP 的 native owner 链是 V02 `execute_model_state` → `sample_tokens` → native sampler 返回 `num_sampled/num_rejected` → `postprocess_sampled` → `speculator.propose` → `req_states.draft_tokens` → native KV `post_forward`。`sample_tokens` 合法返回 `None` 的情形和 `execute_model` 留待 sample 的情形均应保留。插件不能自己重算接受/拒绝数、以 target transaction COMPLETE 代替 proposal 完成，或把草稿输出提前写入 request history。

V07 `AutoRegressiveSpeculator.propose:204–392` 有两段不同几何：

1. **prefill** 接收真实 target InputBatch、attn metadata、slot mappings、hidden/aux、accepted/rejected 等，沿用 target 查询布局并生成第一枚 draft；其 FULL graph 归 `prefill_cudagraph_manager`。
2. 若 N>1，先 `prepare_decode_inputs`，decode 每请求 q=1，再走 `_multi_step_decode` 生成剩余 N-1 枚。A03 `capture:396–444` 将这 N-1 步整体 capture；`_multi_step_decode:492–512` 在 FULL 时只 replay 一次，NONE/PIECEWISE 才委托原生逐步循环。

V06 `MTPSpeculator:38–59` 还保留可选 index top-k compact/skip hooks；它们是否启用取决于 native model/config，不能对所有 MTP 模型假定启用。A05 `deepseek_v4/mtp.py:124–178` 的 V2 模型输出会 flatten HC residual，`:216–249,284–301` 将 forward feedback 与 compute-logits/head 处理分开。普通 MTP 角色 wrapper 必须按该新结构保留 HC/attention/index/head 归属和返回值，不能复制 donor 的旧输出折叠时机。

donor D01 `attention_model_runner.py:1998–2042` 使用 MRV1 `.drafter._runnable` 的单 `ACLGraphWrapper`；D03 的 role-aware MTP forward 用隐式 modulo-N 调用计数和 static step-zero Graph wire。D02 `ffn_model_runner.py:1221–1306` 在 target 后服务完整 N 次 merged stage0 草稿 MoE。当前 P 没有这些 role-aware MTP model/`attach_afd_connector`/remote draft phase consumer，且当前 V2 没有 donor 的那个单 runnable 生命周期。直接搬旧 phase-ready wrapper 会漏 prefill/decode 的不同 native manager、FULL 整段 N-1 replay，以及 model feedback/head 契约。

建议 H12 分两次受控代码增量，而不是扩大 V4.1 guard：

| 次序 | 需要真正消费的接口 | 必须可观察的行为 |
| --- | --- | --- |
| 先旧模型 eager/U1 | pinned native MTP model 的 role-specific wrapper；A 绑定实际 `speculator.model`，F 私有 draft-MoE loader；proposal owner 在真实 native prefill/每个后续 decode 调用前协商包含 proposal identity、segment、logical step、draft-layer、真实 rows 的事务 | 一个真实 native N-step proposal 对应完整 F 服务；accepted/rejected/feedback/token-state 仍由 native A 更新；N=1/多步、profile/dummy/取消/异常和下一 target 不串阶段 |
| 再 target-U2 / draft Graph | target 两段输出先 native merge；draft prefill 使用真实 target 布局，decode 使用 q=1；独立 cache keys 表示 prefill 和合并 N-1 FULL decode，并在首个 data op 前共同决定 replay/eager | target U2 不把普通 draft 凭空拆成同样 U2；一端 cache miss 两端前置 eager，不能执行半段后 fallback；capture 不向 F 发 live request，STOP/失败保留 owner/Work/buffer |

预期修改范围是 plugin-owned MTP model wrapper、NPU attention/speculator binding、F runner/loader、typed phase-aware control 与镜像 seam tests；既有 `hccl_plan` target-only 协议不能以一个未消费的 `phase` 字段冒充实现。第一项也需要源签名/构造/权重命名空间审查及明确 capability 协商。N 次 MoE 的通信成本与模型本身相同数量级，额外控制协商应在 proposal segment/parent 外，不能放进逐 token 算子；单独图池/缓冲成本必须按两 native managers 实测。

## 本轮结论和未关闭项

可立即推进的软件是实际布局证明及其 consumer，或普通旧模型 eager/U1 MTP 的完整 native proposal/service owner；二者都不依赖 A5 才能写 CPU/interface 行为。但 V4.1 native MoE/DSpark、真实 P/D 传输与请求历史、Graph PD、DSpark PD、旧模型完整 N-step 运行和性能仍需要各自设备证据。本文不扩大任何资格，不将任务勾选，也没有把旧 donor patch 存在当作 v0.30 已支持。
