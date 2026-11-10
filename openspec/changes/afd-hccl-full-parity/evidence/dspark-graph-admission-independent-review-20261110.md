# DSpark target Graph phase2：窄准入独立审查

结论：本次受控配置/构造增量未发现阻断项。审查者没有编辑被审运行代码或测试，没有启动设备；已直接读取原始 CPU XML、stdout、identity、source-check，并核对当前字节与 immutable snapshot。这里接受的是 explicit target Graph 组件的配置与构造契约，不是普通生产 model guard 开放，也不是完整 DSpark Graph/采样、NPU 或 PD 验收。

## 冻结范围

| 文件 | SHA256 |
| --- | --- |
| `afd_plugin/compat/npu/runtime_config.py` | `74c206ae788d30722678c882e1d1d1401498f674c09351ce0e1f2c9c4ed62bf0` |
| `afd_plugin/connectors/npu/hccl_config.py` | `7981e31baa9e3004d07b94cd1cd42c87a251dac522b0fb2cbe1a6e4f24272660` |
| `afd_plugin/v1/worker/npu/hccl_v41_ubatch_runner.py` | `5eb91545be95489a7357de70ae6620cc1cbbdc22fd35c8e6971b2301654af6d7` |
| `tests/unit/compat/npu/test_runtime_config.py` | `a071c994815497e597f1cb49c29a0751c1aad11d6e16c2242b20599f7772e091` |
| `tests/unit/connectors/npu/test_hccl_config.py` | `2341c7eadaaef9c2a3bb458b47038be7fae6e7e8bed8768e5a458ce293b7d4b3` |
| `tests/unit/v1/worker/npu/test_hccl_v41_ubatch_runner.py` | `dcc80aa354ffc1d58a97a592bfec88bbec6c6cfba6117ff726ef15649ee090f4` |
| `tests/unit/fakes/vllm_config.py` | `19cae0453579c57d7734d06e57e01683ad5af9dc522f9b97d5316725022054b0` |

`feature_validation.py` 保持 `f36266695f24c04f977fb8df5bccc4f3c9b4525e0ff5be311aa4e305bb96e4ff`，无本次 diff。锁版 native 仍是 vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607` / Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0`。phase1 retained target shape / native MC2 的独立边界见 [前一份审查](dspark-target-shape-independent-review-20261110.md)，本次不以配置测试替代它或扩大其硬件结论。

## 源码审查

`runtime_config.validate_deepseek_v41_graph_speculation:67–86` 先调用原 resolver，保留 exact V4.1、same-checkpoint DSpark、normalized draft architecture、相同 TP、真实 rejection、ordered aux IDs 和 draft metadata 验证，再要求 `speculative.enforce_eager is True`、adaptive verification off、无 KV-transfer/PD。原生 draft/proposal 留在 A。非 V4.1 返回 `None`，不是给其它模型添加新的宽泛资格。

`hccl_config.validate_staged_hccl_runtime:77–120` 只有 `graph=True` 才派生/恢复 target q；A 的显式 `target_shape` 必须与其原始 config resolver 结果相同，F 的私有 spec-off config 可使用构造时保留的 immutable shape，且仍校验 token capacity。`graph=False` 的 capability q 始终为 1，传入 target shape 不改变 eager wire 契约。Graph 恢复 F target q 不会把 drafter/spec/Engram 重新放回 F config。

`validate_staged_hccl_binding:140–185` 将 shape 传给预期 capability 计算，再比较 q、原 model dimensions/layers、native TP、role-size、counts/Graph versions、rank/device、idle/terminal/closed 和 capacity。F 的原始 spec 仍必须在 role isolation 时移除；`_validate_hccl_runtime:287–299` 不接受 F speculative config，Graph PD 也仍拒绝。原 DP1/PP1/CP1、非 EP/SP/LoRA、BF16、compilation NONE/FULL_DECODE_ONLY 等限制均保留。

`HcclV41UBatchRunner.__init__:124–170` 用同一 Graph speculation validator，要求真实 A q 与 transport negotiated q 相同，已有 explicit `graph_owner.transport is transport`、counts/Graph v1、paired native TP 和 native rank 检查不变。native constructor 参数/返回声明未改变；新增校验位于现有 AFD patch markers 内，不复制 native execute 或 sample。新 config helper 与已有 validation helper 的 plugin-only keyword `target_shape` 有明确参数归属说明；不是上游签名变更。

这些校验只做 CPU 配置读取、shape/capability 比较，位于构造/绑定前沿，不在 tensor capture body 或每层 HCCL 内执行。它们没有新 registry、mutable global、drafter 构造或额外归约。若未来 native 为分离 F runner 提供正式 target-shape/config owner，应替换该 plugin-only 注入，保留严格端点协商；不需要恢复已废弃的 MRV1 draft wrapper。

镜像 tests 对外断言 config 保持、显式返回 q、A/F capability 相等、错误类别以及构造后首个 prepared-FULL 边界；没有以内部 helper call counts 作为证明。共享 fake 的 eager-draft 字段显式为 True，用于已限定组件资格，不意味着 native draft 的所有默认配置均满足此资格。

## 实际 CPU 证据

直接核对 [admission-002 identity](a3-dspark-graph-cpu-20261110/admission-002/identity.json)、[XML](a3-dspark-graph-cpu-20261110/admission-002/results/cpu.xml)、[stdout](a3-dspark-graph-cpu-20261110/admission-002/results/cpu.stdout.log)、[summary](a3-dspark-graph-cpu-20261110/admission-002/results/summary.json) 和 [source-check](a3-dspark-graph-cpu-20261110/admission-002/results/source-check.json)：

- snapshot HEAD `06b9d620379944eb30ec41e99746f8e6f5ef6637`，working-tree manifest `b9acdb04d9542107713a36b9d0c98d3719b2c928a2cf572103c307acabbbffba`；identity SHA256 `443c224f3053b5902a45f26f40efc0ccd58937d713b9fe39e1574dcad3811f1a`。
- 三个完整 modules：runtime config **106**、HCCL config **139**、UBatch runner **78**，合计 **323 passed / 0 failed / 0 error / 0 skipped**；XML 用时 1.718s，exit 0。当前上表七文件及原 feature guard 均与该 snapshot manifest 一致。
- 21 个 installed vLLM/Ascend 源文件校验全部 true；`source_files_match=true`，`device_probe_started=false`。这些是实际依赖下 config/constructor/native-CPU seam 行为，没有 native model/采样或设备执行。
- 前次 [admission-001 source-check](a3-dspark-graph-cpu-20261110/admission-001/results/source-check.json) 保留：21 native pins true，但 harness 把 donor AFD `p2p_hccl.py` 错当 installed upstream pin而失败，**没有执行 tests**。002 修正源身份分类后独立运行；未将 001 改写为测试通过或运行时回归。

本次没有覆盖新增 q>1 的真实多端 HELLO、early geometry/key/controller、capture/replay tensor bank、完整 native sampling、native DSpark proposal 或 A/F 服务入口贯通；这些由其独立 phase2 组件与实际设备/模型证据关闭。Graph PD、draft Graph、adaptive verification、未接受拓扑以及总模型门禁都不因这 323 个 CPU cases 获得资格。
