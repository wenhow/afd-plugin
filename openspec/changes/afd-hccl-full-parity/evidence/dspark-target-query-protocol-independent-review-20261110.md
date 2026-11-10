# Target query capability：独立协议审查

对实现者冻结的五文件做只读审查，未发现阻止本轮 capability/CPU 生命周期组件继续验证的源码问题。没有编辑运行代码、重跑测试或启动设备。本记录与 [实现者契约](dspark-target-query-protocol-contract-20261110.md) 分开，不把独立 source review 当 NPU 或完整 DSpark 验收。

| 路径 | SHA256 |
| --- | --- |
| `afd_plugin/distributed/hccl_plan.py` | `4ae86a77ccd36a2dff905526c66309cd3e1947c7dc1801d7b8177acc9f063c47` |
| `afd_plugin/connectors/npu/hccl_staged_transport.py` | `43e7c1ae034108f44a878d070d934d4d7658cd8fc6ac47cb9e3e22ef6d4b54e4` |
| `tests/unit/distributed/test_hccl_plan.py` | `e1a96f4254cab40af96a359a812d3d8a088a933ceaa81c8c1aedfbf8ceb31d62` |
| `tests/unit/connectors/npu/test_hccl_staged_transport.py` | `80fcb4df31cb59ec288c9bce644303649ce2b7c1f7f1e3fbf18528cb22f2a652` |
| `tests/unit/fakes/hccl.py` | `8e17f0cda846d70fc55cad743febe212a371353b60b3fb35a17974c72eb42237` |

`HcclPlanCapabilities.target_query_len` 追加在 dataclass 尾部，默认 1，不改变旧 positional 字段含义。`validate:108–163` 用现有 strict-integer helper 拒绝 bool、非整数、非正数和超 `max_tokens`；q>1 必须 counts v1/Graph v1、paired native TP1/TP2。`negotiate:165–208` 将 q 作为严格相等项，只有原有 token/stage capacity 可以取 min；双方 q 都已不超过自身 capacity，取 min 后也不会产生小于 q 的新 ceiling。

`encode_hccl_plan_message:513–522` 在 q=1 时不编码新字段，`_decode_capabilities:574–609` 缺省为 1；原 q1 golden HELLO 字节测试保留，q>1 是显式可拒绝扩展，不静默降级。旧 strict reader 遇未知字段会在 HELLO 前沿失败。此次没有更改原 JSON/frame schedule、target payload 的 BF16/独立 FP32/raw IDs 或 stage 计数规则，也没有要求每个 child token slice 整除 q。

`HcclStagedTransport.initialize:257–276` 仍先 control group→完整 HELLO→stage data groups。coordinator `_hello:288–315` 对完整 semantic 错误收完所有 HELLO 后广播 ERROR；query-width mismatch 沿已有 terminal/close 路径收尾，不分配数据 PG。断链/I/O 错误仍是有界 transport 故障，不承诺网络 abort 或任意 peer 继续执行。

`_validate_graph_candidate:395–411` 在 coordinator 调 resolver 前、非 coordinator A/F 收 query 后调用，要求 keyed warmup/capture/replay 的 uniform count 和 max-query 均等于 negotiated q。错误走已有前置 ERROR / report-rejection 路径；尚未创建本轮 native inputs/producer。无 key 的 honest eager/empty preflight 保持原协议。STOP 的 idle/reservation/actual-count 边界没有本次更改。

新 tests 使用真实公开 `initialize()`、真实 codec、HELLO negotiation 和状态转换，仅替换 PG factory、destroy 和 set-device 三个环境边界；覆盖 A0/A1/F0/F1 的 mismatch 与匹配配置、warmup/capture/replay 错 key 在 native resolver 前拒绝、q1 wire bytes、q 范围/拓扑/version 与对称协商。transcript 顺序和 PG 创建/销毁属于该公开生命周期契约；这些 CPU records 不是实际 HCCL process groups。

已直接读取 [本地记录](dspark-target-query-protocol-local-20261110.json) 和其 [原始 XML](dspark-target-query-protocol-local-20261110.xml)：pure codec **208 passed**；因本机缺 torch，transport mirror 整模块 **1 skip**。该 skip 不算实际行为通过。此刻实际依赖 transport/HELLO CPU 回归仍待 parent 的 immutable snapshot；后续结果应另行 append 并匹配同一五文件字节。

运行成本只是在 HELLO 多一个可选整数、Graph query 多两次标量比较，无逐层 payload/归约/设备内存成本。target shape/key/geometry、F isolated config 恢复、真实 A/F worker consumer 和 native sampling 仍由其各自增量负责；生产 guard、PD、draft Graph 和未接受拓扑没有因此开放。未来 native A/F capability 若统一提供 target q，应替换该 plugin-owned 字段，同时维持 explicit、exact agreement 和旧 q1 wire 契约。

## Actual-dependency CPU follow-up: protocol-001

Independently read `a3-dspark-graph-cpu-20261110/protocol-001/identity.json`,
`results/source-check.json`, `results/summary.json`, and `results/cpu.xml`.
The immutable snapshot is based on HEAD
`06b9d620379944eb30ec41e99746f8e6f5ef6637`; all five reviewed file hashes above
match its manifest and the current files. All 21 installed-source checks are
true. XML records **321 passed** (208 codec and 113 transport), zero failures,
errors, or skips, in 0.708 seconds. The command returned zero and explicitly
reports `device_probe_started=false` and `source_files_match=true`.

This closes the earlier pending actual-dependency CPU transport/HELLO check for
these exact bytes. It does not exercise NPU HCCL, native model inference, or
DSpark proposal/verification. The local missing-torch skip above remains part of
the historical record rather than being relabeled as a pass.
