# C5 Graph 源码契约审查与首个实现边界（2026-11-09）

本记录覆盖任务 1.1–1.3 的静态设计输入；不勾选任务，不打开 Graph/model guard。
没有执行 Graph capture/replay、GE 编译或 V4.1 模型推理。A3 的 eager C4 HCCL
组件结果不能替代本记录所列 Graph 验收。所有上游通过锁定 Git object 只读查看。

## 1. 精确来源与签名

| 来源 | 精确 revision | 路径 / 符号 / 行号 |
|---|---|---|
| vLLM | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | `vllm/v1/worker/gpu/cudagraph_utils.py:63-76`，不可变 `BatchExecutionDescriptor` |
| vLLM | 同上 | 同文件 `CudaGraphManager.capture:398-488`、`dispatch:496-527`、`run_fullgraph:529-542` |
| vLLM | 同上 | 同文件 `ModelCudaGraphManager.capture:585-747`、`run_fullgraph:749-762` |
| vLLM | 同上 | `vllm/v1/worker/gpu/ubatch_utils.py:329-387`，`UBatchRunner.prepare(self, input_batch: InputBatch, block_tables: tuple[torch.Tensor, ...], slot_mappings: torch.Tensor, cg_mode: CUDAGraphMode = CUDAGraphMode.NONE, for_capture: bool = False) -> UBatchState` |
| vLLM | 同上 | 同文件 `begin_capturable_run(self, model: Any, model_inputs: dict[str, Any], ubatch_state: UBatchState, for_capture: bool = False) -> Callable[[], Any]`，432-524；`abort_pending_run:526-533` |
| Ascend | `7e37e29a7803d6897a7113088e2b771f83205bc0` | `vllm_ascend/worker/v2/aclgraph_utils.py:138-300`，`ModelAclGraphManager`；`ModelWithContext.forward:315-337` |
| Ascend | 同上 | `vllm_ascend/models/deepseek_v41/engram/model_state.py:48-183`，`EngramModelState` |
| Ascend | 同上 | `vllm_ascend/models/deepseek_v41/model.py:1297-1360`，`can_capture_engram_producer` / `captured_engram_inputs`；1362-1546，retire/buffers/events；`AscendDeepseekV41LLMForCausalLM.forward:1739-1777` |
| Ascend | 同上 | `vllm_ascend/models/deepseek_v41/engram/embedding.py:358-365`，`wait_lookup(done, external: bool = False) -> None` |
| Ascend | 同上 | `vllm_ascend/attention/sfa_v1.py:184-203,288-327,792-806,961-986,1159-1168`，persistent Sparse MLA metadata / support / graph builders / no-op `update_graph_params` |
| donor | `4ddd69a391dfb5b3e362c7601399e538a31de702` | `afd_plugin/connectors/npu/p2p_hccl.py:73-141,1727-1788`，graph ops / registration / parent stream |
| donor | 同上 | `afd_plugin/v1/worker/cuda_graph.py:104-132`，`make_ffn_graph_key`；`afd_plugin/v1/worker/npu/ffn_model_runner.py:729-742,1385-1402,1480-1516`，key / duplicate capture / F graph cache |

主要 blob 的 SHA256（对 `git show` 原始 bytes）：

| 路径 | SHA256 |
|---|---|
| vLLM `gpu/cudagraph_utils.py` | `6e9c042890603535e300a40df8ee159dbed1058a64a83ae50ff0329e332e05ff` |
| vLLM `gpu/ubatch_utils.py` | `5092e2e58af0561c8dc12e62847cfc6dda9ca20dfbd33e7f0cd77a83fe87f7c5` |
| Ascend `worker/v2/aclgraph_utils.py` | `b2271eb4aab5876e85370897236a99ece7e9a9541b05f2fc0dd923b32bb8193d` |
| Ascend `engram/model_state.py` | `677cb13bb86ad80a154e0ff9a43620aa3344e7e93170a05e27aba0cc1f10aa8b` |
| Ascend V4.1 `model.py` | `989aec6db1e344dca76159dcbef0d5876874facedaa36b47e993146d5b8a2b9f` |
| donor `connectors/npu/p2p_hccl.py` | `594ac53c8c8008ba05da4d12db1c38511e94fdcb8e13340322a0587e940df0ef` |

## 2. Native manager 是执行协议，不能只复制 graph key

`BatchExecutionDescriptor` 的字段为 `cg_mode, num_tokens, num_reqs,
uniform_token_count, max_query_len, num_active_loras, num_ubatches`。
manager 根据这些字段选择 FULL/NONE；未捕获 shape 返回 eager descriptor。
仅用总 token 数作 AFD key 会丢失 U、request/query 约束及 peer 布局。

每个 FULL descriptor 实际执行：`create_forward_fn(desc,warmup=True)` → 一次
`forward_fn(NONE)` → **重新** `create_forward_fn(desc,warmup=False)` → 图内一次
`forward_fn(NONE)`。图内传入 NONE 不表示本次不是 capture；Ascend
`ModelWithContext` 从 stream capture 状态设置 capturing。F 必须逐次配对，不可用
descriptor 去重跳过第二个通信 body。重复 key 如选择 replay，双端仍必须各 launch 一次；
donor 1385-1402 行已有处理，直接移植“existing key 就 return”会让 A 的 HCCL 等不到 F。

U2 `create_forward_fn` 先调用 `prepare(..., cg_mode=FULL, for_capture=True)`，再
`begin_capturable_run(..., for_capture=True)`；threads 在图外到达 barrier，回调在
`capture_stream` 内释放并 join。失败调用 `abort_pending_run`。原生 handoff 中某个
thread 死亡仍可能令 sibling 停驻（原生注释 477-483），不能把 abort 当任意故障的
可靠网络取消。C4 同步 stage engine 没有使用该图协议，当前 prepare 也明确拒绝 FULL。
C5 需明确新的同步可捕获 tensor body；不能删除 C4 guard 后继承 native full-model threads。

## 3. Torch NPU：两个后端的 shape 语义不可混淆

由 `c4_staged_probe` 在现有 A3 环境只读采集；没有导入 torch、启动设备任务或修改包。
环境为 `torch 2.10.0` / `torch_npu 2.10.0.post4`，site 前缀：
`/mnt/workspace/code/.venvs/afd-v030-dsv41-c0/lib/python3.12/site-packages`。
这是实装包源码证据，不能宣称锁定仓库中已固定该 wheel，更不能代表 A5 的 wheel。
采集元数据与摘录见 [只读来源记录](torch-npu-lowering-source-20261109.json)；
记录保留完整文件 hash，但未本地归档源码全文，部分 converter 签名已标明归一化。

| 实装文件 | SHA256 / 关键行 |
|---|---|
| `torch_npu/dynamo/npugraph_ex/ops/_hcom_send_recv.py` | `95a7b9816b91f1fb5d43fe482366351592506985d1440055b6a21ab7b903c8ce`；14-15/70-71 schema，47-64/104-122 trace wrappers |
| `torch_npu/dynamo/torchair/_ge_concrete_graph/ge_converter/experimental/hcom_send_recv.py` | `f34c259696f9b152f8de40235647554b766f7e33296ae44d0dd3f17e697e082e`；51-65 send converter，125-141 recv converter |
| `torch_npu/dynamo/npugraph_ex/_utils/adjust_traceable_collective_remaps.py` | `d1121d36ce556eb70dd4b347d9610891c2987c4b1ffa0e94dc3f4b3e6179af3b`；44-72 trace remaps |
| `torch_npu/dynamo/npugraph_ex/npu_fx_compiler.py` | `926818c66e7fae1cc5191ff994935b746c46fcd9e6fba23e1938a0f84fe1382c`；777-795 send/recv pass，CANN >=8.6.0 检查（旧版仅 logger.error） |
| `torch/distributed/distributed_c10d.py` | `31155344233ae77360a630bbeacf1b43cfa18f1923cf2103cd5b3142db39ec5c`；2540/2572 send/recv 签名 |

实装 trace wrappers 接受 `(tensor, dst/src=None, group=None, tag=0,
group_dst/src=None)`；非 tracing 调用 c10d；tracing 向 `npu_define._send/_recv`
传入 rank list、group tag、tag、group coordinate，**并传入 `shape=tensor.shape`**。
recv 返回 tensor 后 `copy_` 到目标。ops 有 Meta/PrivateUse1 实现及 side-effect 标记。
PrivateUse1 body 的确不使用 `shape`，但 **GE converter 的 HcomReceive 使用
`shape=shape, dtype=tensor.dtype`**，同时保留不可剪枝 side effect。

donor 针对 2.10.0.post2 + 旧 vLLM 编译器将可选 shape 设 None；源码注释的
“unused”不能推广到两个编译后端。当前最小实现先用静态 contiguous buffers 与原生
trace wrappers，保留 shape；仅在复现 dynamic-shape specialization guard 后考虑
局部 workaround，并分别验 npugraph_ex 和 GE。不能凭 schema 可选就认定两者兼容。
本轮没有查看 CANN 二进制实现；需在目标 CANN/wheel 上验证真实 lowering 与执行。

## 4. V4.1 producer、事件与 attention 的图内外分界

`EngramModelState` 保有固定地址 lookback window、独立 `max_reqs+1` query buffer、
单元素 valid-count tensor。FULL 且支持 captured producer 时，prepare 只原位填
coordinates，不得先在 Python 提交 lookup 或 record ExternalEvent。native outer
model forward 在 `captured_engram_inputs` 内捕获 hash/lookup、consumer waits 及
auxiliary-stream join（1739-1777 / 1314-1360）。支持条件包括 slotless、非 SP、PP1，
每个 Engram table dp_size/tp_size 均为1，不能从 native model TP 数推断 table TP 数。

不支持 captured producer 的 route 使用 capacity lookup/mask buffers 与按 padded-token
bucket 建立的 ExternalEvent；producer 必须写完才 record，consumer wait 后 reset。
仅捕获 BF16 FFN hidden states、忽略 lookback/valid-count/FP32/IDs 或事件，不构成模型图。
`ModelAclGraphManager.run_fullgraph` 保留原生 prepare/replay/metadata retirement；
`EngramModelState.finish_execution` 仍是父执行的 retirement owner。

U2 不能把同一 native ExternalEvent wait/reset pair 每个 stage 消费一次。同步 Graph/U2
最小设计优先捕获一个父 producer，再把 lookup/mask 切到每个 stage；首个消费前完成必要
wait，父 compute stream 必须在结束前 join。若走 graph-external producer route，需明确
父事件唯一消费点或为 stage 分配独立事件，不能复制 eager `engram_prepared=True` 语义。

四个 shared attention tensors 的 reset 为 no-op（native model.py:502-515），固定
storage 是图地址的一部分。C4 token-slice views 可成为每个 Graph key 的固定 stage view；
capture/replay 的 stage offsets、source/consumer 跨层共享关系不得改变。SFA 的
`update_graph_params` 是 no-op，真实更新由 persistent metadata 的原位 prepare/图内
kernel 实现；不能迁移旧 MLA update 调用后宣称 V4.1 已更新。尤其 Sparse MLA
valid_mask/topk_lengths、sinks、block table 及 A5 packed indexer 长度必须单独观察。

## 5. 最小可实施组件与第一个可观察 seam

建议先实现 backend-local **tensor-only HCCL exchange body + 固定 buffers**，直接供
一项真实 NPU 双进程 Graph component probe 使用；暂不接 model factory/guard，也不先加
无人调用的 Graph codec。其 schedule 从已验证 parent layout 生成，显式保持 layer-major
stage 顺序、每 stage data group、peer rank/offset。初版仅 DP1 / A-size=F-size=native-TP，
同步单 parent stream，U1 后 U2；后续 C6 再做 overlap。

图外负责：HELLO/counts/父 plan 及模式共识、输入/IDs 合法性校验、buffers 分配、图选择、
原位刷新、COMPLETE/STOP。图内只包含 BF16/独立 FP32/int32 IDs 的 HCCL tensor ops、
native compute 及相应 events/copies。不得调用当前 `receive_layer`（每次 alloc/IDs item
检查）、`_advance`（Python iterator）或 `finish_forward`（device synchronize + Gloo）。
必须把“固定设备 schedule 已执行”提交到 host cursor 的行为放在图执行完成之后，并防止
同一个 accepted parent 既走 eager cursor 又走 captured schedule。

首个 seam 的可接受结果：两端同时 warmup/capture，各自缓存一张图；重复 replay 至少
三次，每次原位改 BF16、非 BF16 可重建的 FP32 值、IDs 与有效行；输出随新数据变化，
padding 不泄漏；打印实际 HCCL op/capture/replay 记录及 buffers 地址、key、stage顺序。
捕获失败不继续复用该 session；安全 quiescence 未证实时交进程监督，不假装可广播 abort。
该 probe 验证通信图组件；下一步再接 native manager warmup/capture callbacks 和真实 MoE，
然后才接 V4.1 captured producer/attention 状态，不能把算术 probe 称 checkpoint 验收。

Graph key 最小包含完整 native descriptor（或显式等价投影）、U、payload版本/dtypes、
native TP、layer集合、每 stage 的 `(A-rank,F-rank,offset,padded_rows)`。全空 stage 的
skip pattern 必须入 key；valid0/padded>0 仍有 wire dummy，不能与无通信混用。valid值若
仅作为固定 tensor 数据影响 mask，不必入 key；任何 Python valid 分支影响 schedule 时
必须重构或入 key。不得把 transaction ID/请求ID放 key。双端任一 miss 均选择相同 eager
fallback；这个共识应扩展现有 solicited counts round，不采用可能与 STOP 对发的主动报告。

按固定 stage buffers 复用、不复制每层激活估算：每 rank 每 stage，A输入 BF16+FP32+
int32IDs 与 BF16 F2A 合计 `rows * (8*hidden_size + 4)` bytes；F按其 aggregate rows
同式估算。另计 native graph pool、HC continuation、Engram、attention metadata和events。
若保留多 key 多 buffer，预算按所有常驻 key 相加；不能用单 token bucket 数替代 peer/key数。

## 6. 后续需要验证，当前未证明

- A5 实装 wheel、CANN 9.2.0 与 A3 wheel/lowering 是否相同；实际 send/recv graph lowering。
- npugraph_ex 与 GE 分别 capture/replay 静态 shape，是否复现 donor 动态 shape guard。
- U2 非零 offset views 的 A5 packed indexer/attention 图 ABI；图内 source/consumer更新。
- 相同 bucket 不同 valid/history/request slot 的 Engram 结果、event唯一消费及失败 retirement。
- native descriptor warmup/capture、重复 key、任一端 miss、prefill fallback 的双端配对。
- 全模型 weights/MoE、cold capture HBM/时间、连续 replay 及 eager 数值对照。

本次仅新增本审查证据，未修改运行时代码、任务状态或上游。建议组件移除点：目标
torch_npu 公开稳定 graph collective API、上游 V2 暴露 capture/replay callbacks 后删除对应
私有 seam；保留 AFD wire/layout 和 role ownership 的独立边界。
