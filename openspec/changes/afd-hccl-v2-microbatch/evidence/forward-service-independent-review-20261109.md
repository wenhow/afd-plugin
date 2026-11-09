# F service 与 shared policy 独立审查（2026-11-09）

结论：下列修复后的显式 FFN v3 service/binding 增量没有发现尚未解决的静态阻塞。最初两个真实绑定缺口（设备、模型能力）已修复并复核。该判断仅覆盖显式组件及现有 native ownership，不代表 production factory/bootstrap 已接通，也不是完整 checkpoint、native MoE、Graph 或端到端推理验收。

## 对象与快照

独立审查者只读这些文件及锁定上游，未编辑 root 所拥有的生产/测试文件；本文是独立新增证据。修复后、automatic bootstrap 前的 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `afd_plugin/connectors/npu/hccl_config.py` | `8666b451c94abe6ca6da7f7179ee37c216c1d28f6b7043a87d802471031192fa` |
| `afd_plugin/v1/worker/npu/ffn_worker.py` | `5d2c3f10de0bf6b420fccf9bd9a56e089af545e27cc853c8885e3440bb3f6124` |
| `afd_plugin/v1/worker/npu/ffn_model_runner.py` | `e1603fd984109293e73afb3af0ffd5ab062f1b86ad5d33c44b76a1aa1973e4f2` |
| `tests/unit/v1/worker/npu/test_ffn_worker.py` | `933e841a622473555bd91113fae7f3f6296c15abf792dac8a2d63f548b02d417` |
| `tests/unit/connectors/npu/test_hccl_config.py` | `19e379a112fb93d56a3c3fe552d2ebde0b3bccb8948681a074fd24cd0b255dd6` |

来源保持锁定：vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`、Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`，通过 `git show` 只读。重要边界是 Ascend `vllm_ascend/worker/worker.py:365` 的 `shutdown(self) -> None` 仅在 worker 安全结束后调用 runner.shutdown；`:384` `_init_device(self)` 将 logical rank 映射成 visible device 后返回 device，不能用未映射的 local_rank 代替实际 device 身份。

## 修复前发现与修复核对

1. **必须绑定实际 NPU 设备。** 初稿只验证 role/TP/rank，未比较 `transport.local_device` 与 worker/runner 的实际 device。`HcclStagedTransport.receive_layer` 按 transport 设备分配输入，而 native 权重位于 worker 设备，错误注入会在全体 ACCEPT 后才出现跨设备算子错误。修正后的 shared binding 要求 `device.type == "npu"` 且 `device.index == transport.local_device`。该值按真实 device 核对，不假设 logical rank 与 visible device 恒等。
2. **必须绑定当前模型及容量。** 初稿未把注入的 hidden/vocab/layers 与 native 模型配置绑定，合法但属于另一模型的能力可以 HELLO/ACCEPT 成功，随后才形状或 layer index 失败。修正后从 `model_config.hf_text_config.hidden_size/vocab_size/num_hidden_layers` 生成当前模型的预期能力，要求完整 `tuple(range(num_hidden_layers))` 一致，且实际 capability 的 max_tokens 不超过 `min(connector_extra.max_tokens or scheduler.max_num_batched_tokens, scheduler.max_num_batched_tokens)`。较小容量允许协商，不扩大原生预算。

`validate_staged_hccl_binding` 先调用 shared runtime policy 与 `caps.validate()`，再核对设备、native rank、未关闭/未终结/无已接受 plan、模型/拓扑/计数协议版本和容量。F binder 还要求同步 `AFDNPUFFNModelRunner`、FFN transport、未启动 thread、未初始化原 connector、没有已有绑定；错误配置在建组或计算前可拒绝。

## 共享策略与签名

旧公开 `validate_hccl_runtime(vllm_config: VllmConfig, afd_config: AFDConfig, role_rank: int) -> HcclCapabilities` 保持参数和旧 v2 返回类型，委托 plugin 自有 private `_validate_hccl_runtime(..., staged=False)`。新增 staged runtime policy 返回 `HcclPlanCapabilities`，计数版本固定 1，角色规模来自 AFD config，保留 native TP 坐标。

private staged 分支只允许 logical U1/native U2，并独立拒绝 speculation。公共路径仍拒绝 DP/PP/PCP/DCP 非 1、EP/EPLB/elastic/fault tolerance、SP、非 BF16、非 text decoder、compile/Graph、LoRA 与 PD；没有因 v3 service 分支绕开这些条件。旧 v2/U1 仍拒绝 DBO/ubatching，旧 Attention speculation 路径保留自己的既有验证。实现没有复制或篡改 U1 配置来使 U2 穿过旧 connector。

本次没有新增上游函数 patch 或修改 native worker/runner 签名；新增绑定和 step 属于 plugin 自有 API，保持直接字段访问和 concrete transport/runner 类型。

## 控制与资源归属

- `initialize_from_config`/`start_ffn_server_loop` 绑定后初始化同一个 staged transport，跳过 legacy wire 初始化；未绑定时继续旧逻辑。
- loop 的一次 `execute_hccl_v41_step(transport) -> bool` 是 receive parent → 既有完整 F executor → 全部输出及父 COMPLETE。False 仅来自已确认的 STOP，不能把异常或未加载模型当作成功/STOP。非空 parent 未加载模型保留 RuntimeError；全空 parent 可只完成控制握手。
- `execute_hccl_v41_parent` 继续拥有完整 native MoE 调用及原生 TP reduction；worker 不新增 all_reduce。parent COMPLETE 中的设备 synchronize 已发生，不重复伪造异步完成。
- 正常停机由 A 发 STOP；F ACK/global ACK 后 thread 返回，worker join 完成才 close。RPC poll 仅在 peer STOP 后返回 False。初始化了 transport 但未收到 STOP 时，shutdown 拒绝并保留资源。
- 计算或通信失败保留 `_ffn_loop_error`，poll/start/stop 均重抛其原因；不会清掉错误后重启，也不会在仍有 blocking receive/device work 时销毁 group 或释放模型。backend 超时和进程监督负责中途故障，未实现可恢复的网络 abort。
- `AFDNPUFFNWorker.shutdown` 先完成上述安全停止，再调用 native worker.shutdown；native runner.shutdown 释放模型/图资源的顺序没有复制进 service 线程。未初始化的 legacy connector 不参与 v3 close。

## 测试与实机证据的范围

镜像 worker 测试在 CPU seam 中覆盖空父 U1/U2 连续事务、COMPLETE/STOP、失败跨 poll/stop 持续暴露、未 STOP 保留资源、role/TP/DP/eager 及新增设备/hidden/vocab/layers/capacity 拒绝。共享 policy 测试覆盖 legacy U1 不放宽及 staged 的 EP/SP/EPLB/Graph partition/LoRA/PD/speculation 负例。断言协议消息、错误原因及资源最终状态，不把 helper 调用次数作为行为。

A3 验证 agent 已报告 `a3-forward-counts-20261109/counts-cpu-002/results/cpu-forward-counts.log/xml` 的 324 项实际依赖 CPU 通过；其中包含此前 F step/service 和 counts 冻结版本。该批次先于本记录最后 shared-policy/model-device 修正，不能替代修正后的最终回归。

同 agent 已报告真实 HCCL `service-1a1f-002`、`service-2a2f-tp2-001`：同 session 五个 U1/U2/空父、所有 A submit/stop、真实 runner step 方法、BF16/FP32/IDs 和 caller native TP 组件通过。这些是受控算术与通信组件验证，未运行完整 V4.1 native MoE/checkpoint。记录原始结果的任务与本静态审查分开维护。

## 成本、剩余软件与移除条件

新 binder/shared policy 只在构造/初始化边界读取配置，不复制权重、改变 token 精度或创建额外 native TP 域。F loop 的每父新增分发不复制 scheduler/Engram；实际通信及 COMPLETE synchronize 成本仍由现有 v3 transport 承担。终结失败保留资源是故障监督边界，不能当正常关闭的 HBM 回收证据。

当前仍是显式 instance 注入。正常 U2 startup 在旧 connector 构造阶段会被 U1 validation 提前拒绝，post-construction binder 不能解决它。下一项可实现软件是 backend-local 的真实 staged connector/transport composition，在旧 Factory 构造前按 exact V4.1/HCCL/spec-off 选择，保留旧 v2 与 closed modelguard；不能继承伪装 v2，不能复制成 U1 config 欺骗 validator。

自动选择必须复用同一 transport，A 的 control_plane=None admission 显式允许新 connector，F 在 EngineCore start service 时初始化，以配对 A load_model 结束后的 rendezvous。profile-before-KV 保留原生 Engram hook 的一次消费；能力协商可能把 max_stages 下调到 1，A 初始化后必须再次核验并在新父选择前处理，不能盲目按配置 2 切分。

上游提供独立 FFN service/runner 注入及 native staged HCCL 生命周期后，可移除局部 factory/binder/controller。Graph、overlap、DP>1/不等角色 native MoE、DSpark/PD 需要后续单独契约和验收。
