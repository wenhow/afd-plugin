# V4.1 AFD eager/U1：受控资格配置

本页提供首次完整 A/F 硬件资格验证的受控启动器。**A/F 已完成加载并健康就绪，首个请求仍失败；
尚未通过完整文本推理与正常退出验收。** 默认只做源码与 checkpoint 预检；只有显式 `--run` 才启动服务。
本批仍为 DSpark 关闭的第一道门，完整模型默认拒绝策略保留。

## 显式候选范围

仅在 worker 环境中显式设置 `AFD_DSV41_EAGER_QUALIFICATION=1` 时，现有
`fail_if_unsupported_npu_afd_features` 才检查下列候选。变量未设置或为 `0`
仍走原来的拒绝路径；其他值报错。调用方继续执行原有 worker/runtime 校验。

| 项目 | 第一批限定 |
| --- | --- |
| 模型执行 | V2、纯文本、BF16、eager、逻辑 U1 |
| A/F 通信 | `P2pHcclAFDConnector`、F 侧完整 gate、每个 native rank 与对端配对 |
| 设备候选 | A4/F4，两侧 native TP4；A、F 使用独立四卡及原生 TP world |
| 其他并行 | DP/PP/PCP/DCP=1；EP/SP/EPLB/DBO/Graph 关闭 |
| 载入 | 原始完整 checkpoint，显式 `deepseek_v4_fp8`，native auto/safetensors loader |
| A 侧 Engram | `cpu_offload=true`、`dp_shared_memory=false`、`embedding_across_dp=false` |
| F 侧状态 | 原生 worker 构造前隔离 Engram；speculation、PD 不进入 F |
| 请求 | max_num_seqs=1，同步调度；model/token 上限均不超过 1024 |
| 本批关闭 | DSpark、PD、prefix cache、稀疏 KV offload、LoRA、sleep、权重更新、细粒度 TP、shared-expert DP、Async CAM layered GMM |

候选直接复用现有 HCCL 校验，包括实际 `parallel_config.rank`，不以固定 rank=0
替代真实执行 rank。A 的原生 HC/pre_mix、Engram ModelState、shared cache 和采样
继续由原生 runner 持有。F 的私有配置经 `get_npu_ffn_config` 生成；不能直接修改
原始目标配置来删除 A 的 Engram。门禁不改变 loader、模型执行或通信协议。

DP1/PCP1 不允许沿用已成功原生 DP8/EP8 的 `dp_shared_memory=true`：锁定的
Ascend 配置与 embedding 构造都要求共享组大于一个 rank。此候选中四个 A rank
分别持有其 native TP head shard，F 不持有 Engram。

## 容量与对照边界

依据既有 [checkpoint 预算](../../openspec/changes/afd-v41-hccl-text-eager/evidence/a5-topology-budget-20261109.json)
重算 CPU offload 候选，数值为容量估计，未测加载峰值：

| 项目 | GiB |
| --- | ---: |
| Engram host 表总量 | 188.833133 |
| 每 A rank 的 host 表均值 | 47.208283 |
| 每 A rank HBM 参考（含旧 20 GiB RoPE 假设） | 21.926063 |
| 每 F rank HBM 参考 | 68.005807 |
| 同 TP4 原生完整模型 CPU offload HBM 参考 | 89.931871 |

以上未计实际复制参数、NZ/layout、加载临时分配、KV/index/history、量化与 MoE
workspace、HCCL 和进程开销；host 还需额外加载内存。A 暂定卡 0–3、F 卡 4–7，
最终映射须核查机器拓扑和逻辑/物理卡号，不在门禁中硬编码物理卡号。

同 TP4 原生参考距离 96 GiB 仅余约 6.068 GiB，且超过 `0.90` 对应的 86.4 GiB
预算，因此不能宣称该控制组可运行。已通过的原生 DP8/TP1/EP8 是功能参考；其 TP、
EP 和 Engram 共享方式与本候选不同，不能用于固定并行条件的性能归因。

## 已核验的源码与本批边界

开发锁仍是 vLLM `ced6857a` / Ascend `7e37e29a`。用户成功原生环境的 Ascend 为
`4ec6dc7722ec4f7f8005a86570238ac4377c5d43`；2026-11-12 补充包中的量化、DSpark
speculator、`ascend_forward_context.py` 和 `fused_experts.py` 四份实际安装源码均与该
版本一致。新增依赖身份已确认，不需要再次采集。未更新开发锁，也未拼接上游安装。

首批 EP 关闭，MegaMoE 不参与该 AFD 候选。量化新增的 `shared_weights` 入参和对应
dataclass 已一并审查。DSpark 的 draft MoE 私有配置/current-config 差异保留给后续
N=1 准入，本次不执行 DSpark。预检固定核对 vLLM 八份与 Ascend 十三份关键源码，
并非整仓等价证明；版本号不能代替源码字节。

## 最小 A5 操作：只新增一个 AFD 请求

使用现有 CANN 环境，不需要 venv、`atb-env` 或安装依赖；Python 为
`/usr/local/python3.11.10/bin/python`，FastAPI 沿用已部署版本。先按
[分支更新说明](DSV41_A5_PARALLEL_MINIMAL_ZH.md#1-更新分支)更新到
`feat/dsv41-flash-hccl-v030`。八张卡必须空闲，A 固定可见卡 0–3、F 固定 4–7。
两侧各自 TP4；此映射须与本机实际卡号及拓扑一致。

2026-11-12 的 `700c048` 首次启动在 A/F 资格检查阶段误报
`V4.1 eager qualification requires a text-only model`，未发出请求。原因是原生
`--language-model-only` 保留 `MultiModalConfig`，`is_multimodal_model` 仍为 true。
修复后由现有 HCCL 校验检查 `language_model_only`，继续拒绝视觉与 encoder-decoder；
无需修改模型 config、移除多模态字段或调整 CANN/FastAPI。遇到该报错时先按上述
说明更新分支，再原样执行下面命令。旧结果目录保留，新一轮自动创建不同目录。

后续运行已通过 A/F 加载、F 接收循环和 A `/health`，但首个请求返回 HTTP 500：
`N1 must be 64 (the only supported query-head count), got 16`。根因是 native A5
mixed-quant Sparse MLA 轮子仅接受 64 个 Q 头，而 TP4 的每个 A rank 只有 16 个。
这是 Attention 计算约束；加载和健康就绪不能证明 A/F 推理或正常退出通过。

本次插件兼容仅在 A Attention 实例选中原生 `MixedQuantPackedCacheOps` 时启用：
将 16 个 Q/sink 头补齐为 64，调用原生 `qsmla`，再截取原 16 头的连续输出。
KV、索引、metadata 等待和 sink FP32 转换沿用原生；不修改上游或轮子。
五份新增原生依赖在部署版本 `4ec6dc77` 与开发锁 `7e37e29a` 中字节相同，
已加入只读预检；仍须核对环境实际安装文件。CPU 1035 项合同回归通过（含新增 14 项），
**A5 packed 算子数值、完整请求及正常退出尚待本轮验证**。

此兼容会令 TP4 Attention 头计算量扩大到原来的 4 倍，并增加临时 Q/output/crop。
它是功能资格候选；不得据此宣称性能合格。详见
[适配设计](../design/module/dsv41_packed_cache.md)。请求中途失败时，已发布的 FORWARD
事务可能仍令 F 阻塞接收；保留 STOP 的 idle-boundary 保护和强制回收标记，不将其当作正常退出。

第一步仅预检，不启动服务、不加载算子库、不访问 NPU：

```bash
/usr/local/python3.11.10/bin/python -B tools/diagnostics/run_dsv41_a5_afd.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --output-root /mnt/workspace/tmp
```

模型或 vendor 路径不同时填实际路径。工具打印独立结果目录；要求退出码 0、
`preflight.json` 的 `passed=true`。检查固定 config/index 身份、48 个 shard 头及索引
范围、Python、插件入口、源码和 checkout 状态。不读取 tensor payload，不证明权重
内容或完整模型正确性。不能以替换文件、放宽哈希或修改门禁绕过预检失败。

第二步仍只执行下面一个命令。启动器重新预检后，先在独立进程用单卡小张量检查真实 packed 算子：3 token/1 token × 原始 KV/压缩 KV，比较原生 64 头与适配后 16 头输出；不加载模型、不建立通信组。通过并确认设备空闲后，再自动完成 A/F 启动、一个请求、退出和归档。算子检查失败会保留 JSON/日志并停止，不加载大模型：

```bash
/usr/local/python3.11.10/bin/python -B tools/diagnostics/run_dsv41_a5_afd.py \
  --model /home/models/DeepSeek-V4.1-Flash \
  --transformer-vendor /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer \
  --output-root /mnt/workspace/tmp \
  --run
```

如既有原生基线 `response.json` 在本机，可在上述命令追加
`--reference-response /absolute/path/to/native/response.json`，比较 32 个输出 token IDs；
不提供时 `reference_token_ids_match=null`，不能记为数值等价通过。
请求固定沿用原生测试的 prompt、temperature=0、seed=0、max_tokens=32、ignore_eos=true，
无需再开终端手工发送。正常流程自动退出，不需要 Ctrl+C。

## 启动和退出判定

F 使用插件独立入口 `afd_plugin.entrypoints.npu_ffn`，继承原生 `SyncMPClient` 的
本地 ZMQ 握手与资源生命周期，不建立 F HTTP 服务。不能直接改为独立原生
`serve --headless`：该路径会等待外部前端握手。F 启动后立即启动 A，不能等待 F
就绪后才启动 A，否则可能卡住 HCCL 配对。A 健康检查和 F 接收循环都就绪后才请求。

退出时只向 A 主进程发送 SIGINT，由 A 发 STOP，等待 F 自然结束。原生 shutdown
等待时间和 worker 等待均显式设为 180 秒；外层分别给 A/F 最多 240 秒。
正常退出要求：

- A、F 退出码均为 0；F 记录 peer STOP、engine cleanup 和四个 rank cleanup 标记。
- F 启动器记录全部 engine 的零退出码及 native client 清理完成。
- 没有强杀日志、没有本次存活进程，退出后八卡均无 NPU 进程。

超时或请求失败仍先尝试 A 驱动的退出；必要时只回收本次创建并核实 PID/start-time
身份的进程。任何强制回收都会记录 `forced_cleanup=true`，`clean_shutdown=false`，
不会报告正常退出。启动前发现其他 NPU 进程则直接停止，不清理其他服务。
设备 idle 输出格式无法识别也停止，由回传日志定位。

## 最少回传文件及下一步

执行结束打印 `diagnostics: ...tar.gz`。优先回传这个单一归档，包含：

- `summary.json`、`preflight.json`、`launch-plan.json` 和 `lifecycle.json`。
- `request.json`、`response.json`，以及使用参考时的 `reference-response.json`。
- `attention-server.log`、`ffn-server.log`、算子探针、资源与前后 NPU 快照。
- `packed-attention-probe.json`、`packed-attention-probe-server.log`：轻量算子的四例结果、比较误差、版本和源码哈希；失败时含具体阶段和异常栈。

预检失败时先回传 `preflight.json`；不用启动服务收集后续文件。归档不含模型权重。
所有原始结果保留在环境，代码分支只交付代码、测试和操作说明。

本门通过后才单独开放 AFD DSpark N=1，再以同一 worker 的两个连续短请求检查真实
proposal/acceptance/rejection 和状态。Graph/U2、PD、并发、精度和性能分别验收。
此前原生三条短请求无需重跑；其退出记录仍按
[原生补充说明](DSV41_A5_PARALLEL_MINIMAL_ZH.md)收尾。
