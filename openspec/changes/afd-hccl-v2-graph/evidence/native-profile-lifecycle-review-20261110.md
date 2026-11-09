# 锁定 NPU 启动生命周期与 C5 cache 所有权审查

**当前正常 NPU 启动不需要新增 profile cache epoch 协议。** Ascend 的内存 profiling 是 eager `profile_run()`；正式 KV 分配后才执行一次 graph capture。GPU 的“临时 KV → profiling capture → teardown → 正式 KV → capture”是另一条 helper 路径，锁定 Ascend worker 没有调用它。下一步应接好正常启动的 eager/graph 双端配对，保持当前同 key 禁止重复捕获的边界。

本报告只读 `git show` 锁定对象，不运行模型、CPU 测试或硬件。精确 SHA256、行号窗口和当时的工作区源码哈希见 [source audit](native-profile-source-audit-20261110.json)。vLLM 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，Ascend 为 `7e37e29a7803d6897a7113088e2b771f83205bc0`。工作区 C5 runtime 仍在开发，source audit 不代表它已通过执行验证；此前 CPU 013/014 的结果另见 [证据总览](a3-control-cpu-20261109/README.md)。

| 正常启动阶段 | 锁定调用点与行为 |
|---|---|
| 计算 KV 预算 | vLLM `engine/core.py:308` 调 executor 的 `determine_available_memory`。Ascend `worker.py:557-627` 调 `model_runner.profile_run()`，记录 activation/non-torch/weights；没有调用 GPU graph-memory helper。显式 KV 内存配置的 fast path 在 572-586 也只是 `profile_run()`。 |
| eager profile | Ascend `worker/v2/model_runner.py:584-613` 负责 MC2 profiling 前置和作用域，然后调用基类。vLLM `gpu/model_runner.py:955-970` 是 `_dummy_run(max_num_tokens, skip_attn=True, is_profile=True)`，随后 sampler、同步、释放临时输出和 GC；不是 FULL graph capture。 |
| 分配正式 KV | vLLM `engine/core.py:319-354` 计算 KV 配置并调 `initialize_from_config`。Ascend `worker.py:1194-1201` 调 `initialize_kv_cache`；Ascend V2 `model_runner.py:352-422` 保留原生 shared-cache/KV 规划、manager wrapper 与 device-metadata 重新绑定。 |
| warmup 与正式 capture | vLLM `engine/core.py:356` 调 `compile_or_warm_up_model`。Ascend `worker.py:882-892` 先按 compile sizes 做 dummy run、kernel warmup，然后在非 eager 配置调用一次 `capture_model()`。892 的普通调用没有 `profile_only=True`。 |
| graph 内存报告 | Ascend `worker.py:894-936` 使用正式 capture 返回的 `npugraph_memory_bytes` 给后续运行的 KV 内存配置建议；不能替换成 GPU helper 已执行或 A/F 共享一份内存预算的说法。 |

固定源码入口：[Ascend worker](https://github.com/vllm-project/vllm-ascend/blob/7e37e29a7803d6897a7113088e2b771f83205bc0/vllm_ascend/worker/worker.py#L557)、[Ascend V2 runner](https://github.com/vllm-project/vllm-ascend/blob/7e37e29a7803d6897a7113088e2b771f83205bc0/vllm_ascend/worker/v2/model_runner.py#L352)、[vLLM engine](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/engine/core.py#L305)、[vLLM profile run](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu/model_runner.py#L938)。

当前 C5 资源 owner 的约束是合理的，不能用清空字典绕过：

- 原生 A runner/manager 拥有正式 KV、attention groups、block tables、ModelState、captured output 和 native graph。coordinator 冻结这些对象身份；A cache 的资源 bundle 还保留 KV、manager、execution/bank 的引用。应在正式 KV 完成后绑定 coordinator，不能把前一轮绑定改指向新 KV。
- F service 拥有其独立的 graph、stage bank、contexts 和 graph pool，没有 A 的临时/正式 KV。F cache 的 fresh 检查和同 key 拒绝，保证旧 storage 不会被当成新捕获。A 的 KV 预算不能直接扣除 F 设备上的 bank bytes；逻辑 tensor bytes 也不是 allocator/graph-pool 实测峰值。
- `HcclGraphCache.close()` 先同步、reset graph，再释放 pending/ready resources；失败保留资源。最终 F service 只有在 acknowledged idle STOP 后才能关闭 cache，然后由 worker 关闭插件 process groups。它是最终服务退出 API，不是可复用的 capture-round reset。

适合 C5 task 1.5 的最小后续软件范围：

1. **先覆盖 pre-KV profile 与 compile dummy。** Graph 协议会让 F service 首先接收 graph reservation；因此 A 的 profile/dummy 也要先发布共同的 eager reservation，再走既有 eager tensor driver，不能在 F 等待 GRAPH_QUERY 时直接发送旧 FORWARD。当前 `HcclV41GraphControl` 已允许 `manager=None`，也有 `reserve_eager`；可用一个明确的启动阶段 owner，避免要求尚不存在的 ModelAclGraphManager。此处是接线建议，并非已有启动链验收。
2. **正式 KV 后才绑定 native coordinator。** 在正常 `capture_model` 入口选用正式 runner 的 manager、EngramModelState、KV 和 ubatch runner，做一次 fresh-cache/ordered-sequence preflight，再进入原生 factory 的 warmup/capture 配对。原生 ModelWithContext 继续负责 device metadata 的 begin/finish，插件不复制其生命周期。
3. **只保留一次正式 capture。** profile/dummy eager 不发布 graph cache entry，所以第一次正式 capture 仍有 fresh A/F cache。当前无需清理 profile graph、重建 HCCL groups 或添加 epoch wire 字段。重复捕获继续明确不支持，不应通过删除 pending/ready 或伪造 generation 来放行。
4. **先做启动顺序 CPU seam，再做对应设备增量。** 观察真实入口的 `profile_run → initialize_from_config → compile dummy → capture warmup/capture → target replay/fallback → STOP`；检查各阶段实际发送的协议、完整 COMPLETE、正式对象身份和错误时资源保留。CPU seam 不证明 Engram、HC/indexer、MoE 或设备 graph 行为。

若未来明确启用 GPU graph-memory helper，才需要另立双端 epoch 设计。它目前不是正常 NPU 启动的完成条件，且存在以下独立兼容问题：

- vLLM `cudagraph_utils.py:986` 调 `runner.initialize_kv_cache(..., is_profiling=True)`；锁定 Ascend 和当前 AFD 的 `initialize_kv_cache` 都没有该参数。直接继承调用会在该 keyword 处失败。也不能仅删除 keyword，因为原生基类用它选择 profiling 配置与 NO_OP KV connector（`gpu/model_runner.py:705-761`）。
- helper 在 878-949 切换全局 throwaway graph pool，采样 FULL descriptor prefix，并在 finally 清理 wrapper/manager；990-1026 解除 layer KV、runner KV/attention/manager、GC 和 empty-cache。`capture_model(profile_only=True)` 本身主要跳过 workspace lock，**不包含外层完整 teardown**。
- 只传 `profile_only` 不能清除插件 A cache 对临时 KV/manager 的额外引用，也不会把 F runner 在构造时绑定的 `graph_pool` 切成 throwaway pool。旧引用残留会改变释放时点和内存估计。
- 真正需要此功能时，应给 idle A/F 共同确认的 capture purpose/epoch、当轮 sequence digest、局部 storage generation、fresh resource bundle；在所有 parent COMPLETE 且双端同步成功后，结束 profiling epoch、释放插件 keepalive，再允许原生 teardown 和正式 KV 构造。不能用最终 STOP 代替轮次边界，也不能把本地 generation 当成跨 rank 指针标识。任何清理失败继续终止 session 并保留依赖；这要求审查完整 helper 的 finally 顺序，不能只补一个 `profile_only` 透传。

GPU helper 依据：[profile/teardown](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu/cudagraph_utils.py#L847)、[capture 与 workspace lock](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/worker/gpu/model_runner.py#L998)。这组未来要求保持为条件性设计，不建议现在为未启用路径实现新控制协议。
