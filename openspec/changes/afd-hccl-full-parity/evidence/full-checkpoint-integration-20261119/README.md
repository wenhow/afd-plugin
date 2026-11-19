# 完整 checkpoint 组合接入与契约审查（2026-11-19）

本轮按用户“继续功能开发、A5 同步手工验证”的要求实施。原始量化权重、
CPU offload Engram、A 所有权/F 构造前隔离保留；只修改 afd-plugin。
非 Engram 的四模式、DSpark/PD 协议已有独立实现与组件证据；余下完整模型
缺口已定位到 TP4 准入、Engram 父 producer/图事件和原始历史交接，可以开始
对应新增适配。不以等待这些适配的完整矩阵作为其开始条件。

当前结果是受控完整权重候选入口及状态接入，**不是 A5 完整模型通过**。
历史 eager/U1 首 token 差异未关闭；新增同 TP4 原生对照用于去除旧 DP/EP
部署条件差异，不能据此认定已找到根因。原生层、checkpoint 和量化参数未删改。
普通旧 V4 MTP 开发仍暂停，profiler、性能和服务启停专项仍后置。

## 补丁与状态归属审查

锁：vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、Ascend
`7e37e29a7803d6897a7113088e2b771f83205bc0`。使用只读 Git 对象核对，
没有切换或修改上游 checkout。签名与 native resolver AST 核对见
[patch-contracts.json](patch-contracts.json)。所有新差异都有原因、PATCH 标记、
成本及移除条件；没有新的可变全局状态。

| 接入点 | 锁定原生契约和实现 | 运行成本与移除条件 |
| --- | --- | --- |
| U2 配置准入 | `vllm/config/vllm.py:VllmConfig._resolve_and_verify_engram_config(self)->None`。复制原函数，仅在 Engram+U2 拒绝条件中排除经过范围检查的显式 AFD full candidate；native model/parallel/load 校验全部保留，非 AFD/未开启标志仍拒绝 | 仅 host 配置判断；原生支持同等父准备契约后移除 |
| A 模型状态选择 | `vl_model.py:AscendDeepseekV41ForCausalLM.get_model_state_cls(self)`，返回继承 native EngramModelState 的插件状态；不复制请求/token/KV 状态 | 只增加 producer 路由布尔值；原生支持 staged TP Engram 后移除 |
| TP4 图 producer | `engram/model_state.py:EngramModelState.__init__(self,vllm_config,model,encoder_cache,device)`、`prepare_engram_dummy_inputs(self,num_reqs:int,num_tokens:int)->dict[str,Any]`、`prepare_engram(self)->dict[str,Any]`，签名保持。TP>1 时 native `can_capture_engram_producer` 不允许图内 collective，改在父 replay 前调用一次 native producer，使用 NONE/plain events，父流等待后只把稳定 lookup/mask 地址交给图 | 每 replay 一次 native producer 与相应流等待；不承诺该路径性能。native 支持 TP collective 图 producer 后移除 |
| native eager 与 history | 保留 `prepare_engram_inputs` 从 `req_states.all_token_ids.gpu` 和已提交 `num_computed_tokens.gpu` 收集 lookback；保留 native forward hook 与 `finish_execution(failed=...)` 退役。未复制 speculative accept/reject 的历史修正逻辑 | 无额外整段 history 副本；等待 A5 原生 proposal/拒绝后下一轮与取消复用验证 |
| U2 图/多流消费者 | 已有父一次准备、按 token slice 的 lookup/mask 视图继续复用。图外生产已在父流 join，不把 one-shot ExternalEvent 交给两个 child wait/reset。保留 native HC/pre_mix、aux、KV/index 所有权 | 复用持久 native buffer；失败仍由 native execute finally 退役。原生 stage/event hook 可表达该边界后替换 |
| TP4 传输 | stage layout 对每个 `r` 推导 `A[4+r]↔F[r]`；包仍为 BF16 hidden、原始 FP32 router 输入、int32 raw IDs。扩展 eager pipeline、Graph DSpark query width 和 cross-layer 的成对 TP4 候选；不新增 TP 归约 | 相同的每坐标点对点包；native MoE 负责 TP reduction。TP1/2/4 白名单保留，TP3 负例与 peer/count 不一致继续拒绝 |
| FFN 启动 | `validate_ffn_configuration` 先验证原始 DSpark/PD 声明，再用 `get_npu_ffn_config` 的私有角色视图进行 V2/runtime/feature 检查。原始配置交给 worker；worker 在隔离前保留 target shape | 无 draft/Engram/KV consumer 在 F 构造。默认旧入口继续要求 DSpark/PD 关 |
| PD 原始历史 | `kv_layout_connector.py` 对 native `on_new_request(self,request:Request)->None` 及 `request_finished_all_groups(self,request:Request,block_ids:BlockIds)->tuple[bool,dict[str,Any]\|None]` 的既有 adapter 增量。P 在 native truncation 前保存原 prompt 的长度/摘要/source 身份，返回 `afd_engram_history_v1`；D 在 native pull 前比较完整原始 IDs，再由本地 native slotless history 重建 | 每请求 O(prompt tokens) host hash、恒定大小声明；不传 lookup 表、不向 F 传 Engram 历史。不替代 Ready/读取/DONE，也不提供 peer 认证。原生保留并验证等价声明后移除 |

架构边界审查未发现需要改动上游或新建 history 所有者的阻断项。源码准入和
CPU 契约可接入设备验证；图捕获合法性、CPU table 的真实 TP lookup、量化输出、
接受/拒绝后的历史以及 PD 回收仍须以下 A5 指导验收。

## 验证记录

全部快照在独立 scratch 内运行，不安装包、不加载模型或运行 DSpark 推理。
CPU 驱动显式阻断 `torch.npu._lazy_init`。环境为 torch 2.10.0、torch-npu
2.10.0.post4、vLLM 0.30.0+empty、Ascend 0.1.dev1+g7e37e29a7；A3 CPU
结果不是 A5 原生扩展、载权或通信通过。

| 记录 | 结果与边界 |
| --- | --- |
| `cpu-001` | 首轮夹具 N=5 与签名导入失败保留；既有图执行夹具误触 native synchronize，after initialized=true，因此不记 CPU-only 通过 |
| `cpu-002` | 阻断初始化后保留 14 个失败：4 个实际 TP4 准入缺口、1 个 P 夹具块容量不足、9 个既有 CPU 图夹具同步边界缺口；NPU before/after=false |
| `cpu-003` | 1,403 项、0F/E/S，NPU before/after=false。含成对 TP4 的 wire roundtrip、四坐标 FFN 消费/Graph endpoint，及 Engram/PD/config 全模块 |
| `cpu-004` | 新 FFN 测试夹具缺 device_config 导致 12F；真实 EngineArgs 找出 23 个 F 原始配置误拒绝。失败原件保留，未转为通过 |
| `cpu-005` | **1,433 项、0F/E/S**，NPU before/after=false；真实 54 个 A/F EngineArgs 场景通过。额外 native 对照误调用 AFD 专用 checker 的单项工具失败保留 |
| `config-006` | 修正 native checker 调用范围后，**55/55 个首次 EngineArgs 构造通过**，四模式/DSpark N=0/1/2、PD producer/consumer、原生 TP4；声明的 ubatching/Graph/Engram/量化/PD 实际字段均核对，原 checkpoint config 字节不变，NPU before/after=false |
| `config-007` | 最终配置检查另覆盖 EngineCore 握手的重复 native `__post_init__()`，**55/55 首次构造及重验证均通过**，实际模式保持、原 config 不变、NPU before/after=false。与 cpu-005 的生产 Python 字节一致；只更新了诊断工具 |
| `host-008` | Engram 边界测试规范整理后 **5/5 通过**，NPU before/after=false；只断言公开输出及流同步协议 |
| `host-009` | Engram 与 Graph 两模块最终 **73/73 通过**，NPU before/after=false；将禁止的 producer capture 作为失败边界，不断言内部 helper 调用次数 |
| `delivery` | 用户授权推送后的独立暂存快照：**1,433/1,433 CPU、312/312 主机测试、55/55 真实配置及重复原生校验通过**；NPU before/after=false。148 份生产 Python，不含暂停的旧 V4 MTP；A5 待验 |

最终 host CLI/HTTP 流程检查包含完整矩阵计划、可配置 PD/卡号/地址/端口、关闭多流、
原生 token 比较，以及“拒绝出现在最后一轮时不得通过”的反例。原 launcher 的既有
流程一并回归；详见 [host-tests.xml](host-tests.xml)。本地 ruff、diff whitespace
和规划 harness 另行通过，后者不是官方 OpenSpec 或 NPU 验收。
开发工作区的 152 份生产 Python 与 cpu-005 通过快照逐字节一致；最终测试文件以
host-009 manifest 为准。配置工具与 config-007 一致。该阶段源码身份、检查命令和
结果见 [final-validation.json](final-validation.json)，其中未提交状态为当时记录。

用户随后授权提交、推送。交付暂存区排除暂停的旧 V4 MTP 代码及共享文档/签名表中的
对应增量；按暂存区独立导出快照复验，结果及精确源码 manifest 见
[delivery/validation.json](delivery/validation.json)。历史失败与成功记录均保留；
本提交只收入日志、摘要、JUnit 和 manifest，原始源码压缩包与测试源码副本保留本地。

CPU tests 在模块边界断言协议、稳定输出地址/值、native 配置结果和错误，不把
CPU fake 当作哈希算子、device event、HCCL 或实际数值正确性。新测试分别由
model_integration、compatibility_and_patches、ffn_entrypoint 和既有 connector/distributed
模块的 validation_paths 管理。继承签名另做锁源审查，pytest 仍只有一个补丁签名表。

## 可执行交付与待验

入口：[run_dsv41_full_checkpoint.py](../../../../../tools/diagnostics/run_dsv41_full_checkpoint.py)。
指导：[A5 完整权重矩阵](../../../../../docs/npu/DSV41_A5_FULL_CHECKPOINT_VALIDATION_ZH.md)。
CLI 支持四模式、DSpark N=0/1/2、多流 0/1，及 P/D 的卡号、HTTP/KV 地址、
端口偏移和 engine ID。每格使用同 TP4 原生对照，双并发两波、取消及随后请求、
长 prompt/chunked prefill；DSpark 要求接受、拒绝和拒绝后继续轮次同时出现。
P/D 放置尚未指定，不假定设备已落实。

完整权重预检在原 U1 的 21 份 native 源码之外增加 Engram config、EngineArgs、
ACL graph manager 和 Mooncake connector 四份锁源；不修改旧证据的期望哈希。
默认只写启动计划，`--run` 才执行预检和设备流程。

所有 A5 格子目前待回传：eager/U1 首 token/连续输出；eager/U2 多流；Graph/U1/U2
真实 capture/replay/fallback；DSpark 接受/拒绝/下一轮 history；五种 PD 组合的
原始 prompt/handoff、Ready/DONE、取消/复用。保持请求检查与完整验收标志分离，
没有据 CPU 或一次 HTTP 成功开放默认完整模型支持。C7 2.2–2.7 及 1.8/1.9 等不提前勾选。
