# C4 显式 native entry 独立审查（2026-11-09）

结论：本轮增量可作为显式、同步、DP1/PP1/CP1、eager U2 的工厂及原生 `run` 入口保留。审查发现的 native TP/role 绑定缺口已在本轮修正；复核修正后没有发现此受控范围内仍未解决的静态阻塞。该结论不开放生产 V4.1/U2 guard，也不构成真实模型、Engram 算子、HCCL 模型执行或 overlap 验收。

## 快照与方法

只读审查三项工作区 diff，读取依赖及锁定 Git 对象；审查者仅新增本文，不修改生产代码、测试、上游或 Git。观察到的工作区 HEAD 为 `9b8a41e6169a3e29e3203190be69cdaa10b41077`，审查对象是下列未提交文件内容，不能只以 HEAD 复现。

| 文件 | SHA256（修正后） |
| --- | --- |
| `afd_plugin/compat/patches/npu/mrv2_ubatching.py` | `1e53b24475946f0cc83d6a05b5a83af5d4746287bd9ba04f357021df27905ecb` |
| `afd_plugin/v1/worker/npu/hccl_v41_ubatch_runner.py` | `7d9ec420eb7d15192819e47587e37116e846dde7967998ea615c284b95ea3435` |
| `tests/unit/v1/worker/npu/test_hccl_v41_ubatch_runner.py` | `ba7ac5e1af8e325a7d404f8a1ebae635e31a27f479d7ac1d5f3edc075e414d9d` |

来源以 `.agents/dsv41/sources.lock.json` 为准：

- vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。
- vLLM-Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`。
- donor `4ddd69a391dfb5b3e362c7601399e538a31de702` 本轮未移植新函数；不能用 donor V4 tuple 替代 V4.1 状态契约。

使用 `git diff -- <三项文件>`、`git -C <上游> show <锁定SHA>:<path>`、`git grep` 检查实际来源。另以 Python AST 比较参数/返回注解并检查 Python 3.11 语法，命令退出 0。这是审查脚本，不是新增的源码文本单元测试；本审查没有运行 NPU 或模型。

## 接口和最小 seam

锁定 vLLM `vllm/v1/worker/gpu/ubatch_utils.py`：

```python
# :538
def maybe_build_ubatch_runner(
    vllm_config: VllmConfig, device: torch.device, model_state: ModelState,
    attn_groups: list[list[AttentionGroup]], kv_cache_config: KVCacheConfig,
    max_num_reqs: int,
) -> UBatchRunner | None: ...

# UBatchRunner.prepare :329
def prepare(
    self, input_batch: InputBatch, block_tables: tuple[torch.Tensor, ...],
    slot_mappings: torch.Tensor, cg_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    for_capture: bool = False,
) -> UBatchState: ...

# UBatchRunner._make_forward_contexts :389
def _make_forward_contexts(
    self, ubatch_slices: UBatchSlices, attn_metadata: list[dict[str, Any]],
    slot_mappings_by_layer: list[dict[str, torch.Tensor]],
    is_padding: list[torch.Tensor | None],
) -> list[ForwardContext]: ...

# UBatchRunner.run :423
def run(
    self, model: Any, model_inputs: dict[str, Any], ubatch_state: UBatchState,
) -> Any: ...
```

子类三个 override 的参数、默认值与返回注解与锁定源一致。局部工厂参数一致；其返回拼写 `ubatch_utils.UBatchRunner | None` 指向同一原生类型。宽 `Any` 保留在原生接口边界，实际 stage driver 使用已有的具体输入类型。没有新建第二个 upstream signature 测试模块。

`mrv2_ubatching.py:110` 为已有 plugin scope 增加 keyword-only `hccl_transport=None`；未改变被替换的 native factory 的六参数调用。`model_runner.py:682` 实际调用的是本轮替换的 alias。新增分支在 native 工厂 admission 前选择子类，子类再明确验证所有新增受控条件；省略 transport 时继续执行旧 admission。原生工厂在 DP1 返回 `None` 的行为仍由无 transport 回归保护。

补丁原因、功能、原生签名、新增 keyword、移除条件均在函数邻近注释中说明；新增行为有 PATCH START/END 标记。没有 `_original_*` 转发作为新执行路径，也没有复制 native `execute_model`、请求调度或 ModelState 生命周期。

## 已修正的真实发现：native TP 与 transport 坐标

最初子类构造只核对 Attention role，没有把 transport 声明的 TP/role topology 与当前 worker 的 native 权重坐标绑定。相同 hidden shape 不能发现 TP size 或 role rank 错配；错误坐标可以把 activation 送给其他 native 权重分片，或使 native collective 成员和消息成员不同。这在开放该显式入口前是实质性正确性缺口。

修正后 `hccl_v41_ubatch_runner.py:88` 起在 native 分配前拒绝：

- DP size 不为 1，或 DP rank 不为 0。
- `transport.negotiated_capabilities.native_tp_size != parallel.tensor_parallel_size`。
- `capabilities.attention_size != parallel.tensor_parallel_size`。
- `parallel.rank != transport.role_rank`。

锁定 vLLM `vllm/v1/worker/worker_base.py:86` 由实际 worker 构造参数设置 `parallel_config.rank`；本轮已要求 DP1/PP1/PCP1/DCP1，因此该 rank 对应 native TP 坐标。没有对未定义的 DP 偏移作推断。F 端已独立要求 DP1、FFN role size/native TP/rank 对齐，本轮 A 端限制与其兼容。

镜像测试 `test_explicit_factory_rejects_unproven_native_group_mappings` 覆盖上述五种拒绝边界，并断言没有消息或完成回执。最终 fixture 的 TP-size 用例比较合法 transport TP1 与 config TP2；rank 用例比较合法 TP2 内的 native rank1 与 role rank0，能单独命中坐标错配。已有 DP1/DP3 `prepare` 子视图用例通过绕过 constructor 建立 CPU fixture，仅证明切分/child context 的可复用性，不能解释为 DP3 native entry 已支持。

## guard、传输与状态归属

构造要求 P2pHccl、Attention、自身及 transport role 一致、`use_ubatching`、U2、`enforce_eager`、配置 Graph NONE，以及 PP1/PCP1/DCP1/non-SP、原生 `EngramModelState`。`prepare` 再拒绝 Graph/capture 及未实现的并行布局；`run` 再拒绝未初始化、已关闭或 terminal 的 transport。未接受 parent plan 的运行在 payload 前失败。

锁定原生 `model_runner.py:1790` 先 `ubatch_runner.prepare`，`:1862` 后执行父 `model_state.prepare_inputs`。Ascend `engram/model_state.py:74` 的普通 `prepare_attn` 本来会记录 `_cg_mode`，本地 child metadata seam 暂时绕过它，因此本地 `prepare` 把 eager NONE 保持到成功后的父 `prepare_inputs` 是必要行为；不能在 `finally` 无条件恢复旧 Graph 值。失败时恢复此前 mode，与现有生产者状态保持一致。

原生 `model_runner.py:1935` 在父 forward context 中调用 `.run`。本地 `.run` 进入同一个 `run_staged`，`:258` 只消费一次 native `prepare_engram()`；调用显式 stage driver 避免重复触发整个模型 `nn.Module.__call__` pre-hook。lookup、mask、pending/event 按父 token slice 投影到孩子；没有新增孩子历史提交或重建父持久 buffer。

锁定 Ascend `vllm_ascend/worker/v2/model_runner.py:486` 的 native `execute_model` 在 `:530` 的 `finally` 调用唯一 `finish_execution(failed=...)`；`engram/model_state.py:65` 才退役 lookup/events 并调用父状态结束。本轮子类没有再退役。直接调用显式 `run_staged` 的测试只验证其组件责任，不能代替该 native finally 的实机覆盖。

transport 的创建、initialize、accept plan、idle STOP/close 仍由 caller 管理。子类只保留实例引用；不建新 native TP group，不销毁 caller group。成功在全部 stage 输出归并后调用一次父 `finish_forward`；异常设 terminal 并保留异常交给 backend timeout/外部监督，不在 payload 已半途发送时伪造网络 abort/COMPLETE。

工厂 scope 在 `finally` 同时恢复 `model_runner.maybe_build_ubatch_runner` 和 `ubatch_utils.create_sm_control_context`，失败构造也不泄漏 alias。它沿用已有初始化期的进程全局 scope，前提仍是 worker 初始化串行；本轮没有把该 scope 用于并发请求执行。

## 测试 seam 与成本

镜像测试在运行时导入前 `importorskip` numpy/torch/vLLM/torch_npu/Ascend；CPU-only 缺依赖时可收集。setup 复用 `tests/unit/fakes`。切分测试保留锁定原生 slicing/context，只替代设备 RoPE 和平台 kwargs；断言子输入、KV/slot/padding、序列重定位以及父 mode 观察值。模型 seam 断言返回及 aux、layer-major 消息、父 context/state 恢复、terminal/COMPLETE 边界。没有源码 substring 断言、内部 helper 调用次数或伪装 NPU 数值证明。

工厂 fixture 明确替换 `NPUUBatchRunner.__init__`，因此其证据限于选择、native `.run` 入口、拒绝和 alias 恢复，不能称为真实 constructor/builder/设备分配成功。实际 native constructor A3 probe 与本报告分开。测试路径由 `docs/design/module/attention_runtime.md` 的 `validation_paths` 明确覆盖。

工厂注入与持有 transport 引用没有新增 tensor copy/通信；调用现有同步 layer-major driver。继承 native constructor 仍预分配 U2 query-start/seq-lens、stage metadata builders，以及原生 comm/capture 两条 stream（同步 driver 不使用它们做 overlap）。父 Engram producer/lookup 不按 U2 重复准备；四个共享 attention tensor 继续采用已经独立审查的父 slice views。已有 v3 每 stage 通信域及 BF16/FP32/IDs/F2A 成本仍存在，没有在本轮被减少或证明能重叠。

移除条件：上游提供 NPU runner/factory 注入及同步 stage execution callback、父 ModelState mode preparation hook 后，替换相应局部 scope/override。当前继承保留原生输入与 builder 升级接口，避免复制完整 runner。

## 增量边界与剩余工作

生产 `attention_model_runner_v2.py:370` 仍以 `use_npu_ubatch_factory(self)` 调用，没有注入 transport。锁定 `vllm/v1/worker/gpu/dp_utils.py:214` 的 dispatch 在 DP1 直接返回 U1 descriptor；工厂能显式构造 U2 不会自动改变该 dispatch。该补丁没有收集真实父计划、启用 U2 选择/fallback 或开放现有 model guard。

后续仍需 native actual-constructor/metadata builder 证据、完整 `execute_model` 输入与 parent plan 集成、真实 V4.1 双 stage 多层算子及 Engram 跨交易/empty/odd 测试、A/F TP 对齐的 HCCL 模型执行、异常监督和连续交易验收。A3 张量 view/传输组件结果不替代 A5 checkpoint 完整模型；A5 量化启动修正也是独立验证项。Graph、线程/stream overlap、DP>1 native mapping、DSpark/PD 不在本次静态通过范围。
