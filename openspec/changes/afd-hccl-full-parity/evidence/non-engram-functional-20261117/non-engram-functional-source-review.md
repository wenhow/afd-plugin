# 非 Engram 功能资格与 P 抢占边界：独立源码审查

结论：当前候选未见阻断性源码问题。此结论覆盖启动资格与所有权接线，不代表新的 A5 设备组合已通过。

## P source 抢占恢复的范围

当前 `openspec/changes/afd-hccl-full-parity/tasks.md:606` 明确写明“P 中途抢占明确 terminal，不实现恢复”。能力矩阵 H13 和当前 1.10 要求 prefix、取消、随后请求及功能错误传播，没有要求内存压力抢占后的同请求恢复。因此本轮保留该明确限制，不把恢复协议作为已缺失的必需实现；取消并真实完成旧 lease 后的新 generation 接入是另一条已实现路径。

锁定 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` 的 Scheduler：

- `scheduler.py:733–808` 分配失败后挑选 RUNNING victim，先检查本地 in-flight fence，再调用 `_preempt_request`。
- `:1477–1522` 的 `_preempt_request` 先 `_free_request_blocks`，随后清 computed/spec 状态、放回 waiting 并登记 preempted ID。
- `:2603–2619` 的 free fence 只使用 native scheduled/processed sequence；它不认识插件的远端读取 lease。
- `:1416/:1436` 到生成 SchedulerOutput/connector metadata 时，旧页已经释放，且本次 schedule 可能已经重新分配这些页。

当前 `kv_layout_connector.py:665–678` 在 metadata seam 对 source preemption 进入 terminal，阻止发布旧 allocation 和后续调度继续执行；它不是抢占前的租约冻结钩子。不能只删这个拒绝来声称恢复，也不能在此处重新标 Ready。P 正常 source 被限制为一次输出，完整投影/采样后转交 delayed-free handoff；已经交付给接收端的页仍由真实 DONE/ACK 退休。该事实不推广为任意在途读抢占安全。

只读 donor `4ddd69a391dfb5b3e362c7601399e538a31de702` 的 `compat/patches/npu/dspark_pd.py` 只有异步远端分配时 lookahead 处理和 hybrid KV 加载失败的整请求 fail；没有可直接移植的 generation/lease 抢占恢复实现。

将来若明确扩范围，最小正确接缝必须在 native `_preempt_request` 释放前取得 source 所有权：未 Ready/无 lease 才能废弃旧 generation，从位置 0 完整重新投影；任何在途 lease 都必须保留原物理页到真实 DONE。当前 post-allocation metadata seam 太晚。需要真实 native Scheduler 内存压力 RED：旧 allocation 被释放并复用后，旧 query 仍不可获取、新 generation 完整投影前不可 Ready，真实 DONE 前原 source 页不得被新执行覆盖。当前已有 `test_preempted_source_cannot_publish_metadata_or_resume_old_allocation` 只验证 terminal 拒绝，不能充当恢复测试。

## 新默认关闭的功能资格

审查基准是 `.tmp/non-engram-functional-20261117-parent` 三份原始备份；旧 A5 eager 资格函数及其它旧 V4 函数 AST 均保持不变。

新 `AFD_DSV41_FUNCTIONAL_QUALIFICATION` 仅允许精确 `0/1`，默认仍拒绝 V4.1 正常执行；与旧 eager/dummy 资格互斥。候选继续固定 BF16 dummy、H512/HC4/64 Q heads/head512、三层 `[0,0,2]`、source `[2]`、candidate source20、词表4096、8专家/top2、1或2请求、batch16/maxlen128、chunked prefill、TP/DP/PP/CP及角色约束。旧两层 load-only 候选没有借此获得新模式。

选择 legacy/staged 的逻辑复用现有真实 HCCL validators：eager U1、eager U2、FULL_DECODE_ONLY U1/U2 各走其原合同，不以资格函数改写运行模式。PD 限 exact attestation module，Graph P 仍拒绝。DSpark 限 same-checkpoint native normalizer、固定验证、eager draft、N≤2、MTP1、aux IDs `[1,2,3]` 及有界专家/投影几何；Engram 仍禁用。F 在保留 target query shape 后验证原始 PD view，再删除 Attention 所有的 draft/PD/Engram；资格函数不会让 F 接管 drafter。

审查发现并由 root 补齐的边界：`SpeculativeConfig.kv_cache_dtype` 会由 locked `dspark/utils.py` 复制进 draft cache，必须独立于 target cache 检查。候选现只接受 `None/auto/bfloat16`，且声明 `spec.quantization=None`。量化项是候选范围限制：Ascend 的 same-checkpoint loader wrapper 实际复用 target quant config，不能将这一条称为已证实的独立量化执行漏洞。

## 最终封包的另两份生产差分

相对 normal execute005：

- `runtime_config.py` 仅允许 exact PD consumer Graph/U2 使用 `cross_layer_v1`；U1 仍仅 serial，编译 NONE、FULL_DECODE_ONLY、equal TP、插件 attestation 与所有其它原条件保留。实际 pipeline policy 仍由 capture preflight 核验。
- `ffn_worker.py` 只将 daemon 的原始 `BaseException` 保存给公开 poll/shutdown；普通 CAM event shutdown 抑制仍要求 `isinstance(exc, Exception)`。既有 daemon `torch.inference_mode()` 保持，新增只是说明 TLS 边界的注释。

没有吸入普通 V4 MTP 后继改动。前者不分配缓存、设备或 RPC；后者不增加同步/设备工作。异常保存不等于取消已在途 device work，也不承诺同进程恢复。

## 已读取的真实 CPU 原件

独立实际读取 `.tmp/functional-gate-cpu-20261117-green003/results.tar.gz`：完整 feature 模块 **342 passed、0 failed/error/skipped**；native eager-U1、A/F 各四模式的9次首次 EngineArgs 构造均 rc0，所有 NPU initialized 标志为 false，source unchanged 为 true。本审查没有再次执行远端任务。

当前 feature 源与该 CPU 包仅 Ruff 换行不同，完整 AST 相等；当前 test/fake 与该包逐字节相等。旧首轮 335P/5F 原件保留：两个 Graph P 在 F 隔离前被原 validator 拒绝、三个错误 regex；不把修正测试前的该轮称为 GREEN。

未运行 SSH、NPU、A5 或新的推理。本结论不替代 normal DSpark/PD 的真实原生 KV/Attention/HC/MoE、accept/reject、取消/槽复用及跨角色参数/输出核验。
