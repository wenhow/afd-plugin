# A5 U1 通过与 U2 子批次源 RoPE 初始化修复（2026-11-18）

三个原始 ZIP 按文件名原样保存，SHA 和核对结果见 [assessment.json](assessment.json)。
用户两处标题均写了 FFN；实际 `8c282423171142818bdacce74b45ed6a.zip` 是 U1 Attention，
`6c4d6f688c8f4106a85897c3ff665742.zip` 是 U1 FFN 加 pair 汇总，
`89ceeac6f8f74cfcabe3aeeb86948d81.zip` 是 U2 多流失败的 A/F、pair 和监督进程产物。
按用户要求排除占用提示，原始错误及清理失败保持。

## U1 实际通过范围

U1 pair `success=true`，Attention/FFN 均退出 0，实际设备分别为 npu:0/npu:1。
模型为 BF16 dummy、三层 C2 `[0,0,2]`、H1024/E64/top-k2、Index32-D128，Engram 关闭。
正常 A/F Worker、加载、KV、warmup、真实远程 MoE 及六轮请求执行均完成。
A 两轮后取消，B 继续结束，C 随后接入并结束；A/F 清理报告均正常。

独立对照前次保存的原生诊断报告 `a5-native-vendor-first-pass-20261118`：
A58 + F21 共 79 份 canonical 参数的 shape/dtype/numel/bytes/SHA/finite 全部一致，
六份 Attention hidden 的文件长度和 SHA 与当前报告匹配，三请求输出完全相同：

| 请求 | 原生和本次 AFD U1 的输出 IDs |
| --- | --- |
| functional-A | 239, 14 |
| functional-B | 1140, 382, 3962, 2736 |
| functional-C | 1605, 3825 |

这支持该 BF16 dummy、拓扑与请求场景下的 U1 功能和有限数值对照。
原始角色报告的 equivalence 字段仍原样保留，独立对照不回写它们。
本轮未提供整套 summary/runtime-source 预检及 U1 外层监督报告；
不能据局部包确认本轮完整冻结身份或七场景通过，也不扩大为训练检查点精度验收。

## U2 首个失败及原因

U2 A/F 已完成 HELLO，尚无首轮计算控制消息。Attention 在第一次 prepare 时失败：

```text
NPUUBatchRunner._prepare_attn -> build_attn_metadata
AscendDSAV41MetadataBuilder.build
AssertionError: Enable device metadata before building compressor metadata
```

实际是新 builder 的完整源 RoPE 表未准备；异常文字不意味着应在 eager 路径开启异步任务。
锁定 Ascend `7e37e29a7803d6897a7113088e2b771f83205bc0` 的
`worker/v2/model_runner.py:415` 在正常 KV 初始化后调用 `prepare_v41_source_rope(self)`；
`worker/v2/utils.py:85-94` 只遍历原生 `runner.attn_groups`。
AFD 在 `NPUUBatchRunner.__init__` 为各 child 创建了独立 `stage_attn_groups` 和 builder，
但没有对这些新实例执行同一准备。eager metadata build 因而在 `dsa_v41.py:1302` 拒绝。
FFN 随后的 Gloo peer closed、loop failed 和 shutdown 失败是 A 失败后的连带错误，
保留原样；U2 外层退出 1，无超时/强杀/所属残留进程。

## 修改与架构审查

改动位于插件 `afd_plugin/v1/worker/npu/ubatch_runner.py` 的既有初始化方法。
每个 child 的 builder 创建后，对 `AscendDSAV41MetadataBuilder` 调用其公共
`prepare_source_rope()`；其他 builder 不受该分支影响。
遵循原生 helper 的延迟导入，避开 model/cache 注册循环。
无上游方法复制或新 monkey patch；已有构造参数和签名不变，增加原因/移除说明及差异标记。

- 原生 `dsa_v41.py:850-859` 的准备只获取按 Attention layer 注册的完整源表。
  `rope_dsv4.py:216` 返回原有表；这里不复制表或分配新 KV/模型参数。
- 工作发生在构造时，每个 child/builder 一次；不增加逐请求循环、同步或通信。
- 每个 builder 自己的可变 ring/metadata buffer 继续独立，源 RoPE 是原生共享输入。
  不修改原生 HC/pre_mix、stage continuation、F gate、FP32/IDs 或请求/Engram ownership。
- 不调用 `enable_device_metadata()`，eager 同步构建和 FULL 原有 task owner 边界保持。
- 原生缺表错误继续传播，不吞异常或用零值回退。
- 后续当上游支持 Ascend 子 builder 初始化时，删除本处准备，复用上游构造契约。

冻结工具新增该文件的显式 overlay，保证环境获得修复。
与父包 d0a32a80 相比，生产代码仅此一份变化，其他 145 份生产字节保持；
另更新 provenance 和 manifest。新包 SHA：
`c11071793ca85dd7082556552a23cf1ab21e37d6ef612ddc1a13c079547b0fb6`。
入口 SHA 保持；186 Python 编译、194 原生 Git 身份核对通过。

## CPU 验证与待验范围

`cpu-red/`：新增三项构造/源表消费/缺表传播检查在修复前全部失败，失败符合预期。
`cpu-green-final/`：含上述三项的 U2 prepare、Graph coordinator、execution、
唯一签名表合计 290 项全部通过，NPU 初始化前后均为 false，传输源码 SHA 保持。
CPU builder fake 只保留原生源表获取方法，不分配或模拟设备 buffers/算子。
本地另外通过 17 项冻结 CLI/metadata 检查、Ruff、规划及上游保全。

`cpu-green/` 的首次广检查为 287 pass/3 fail：CPU 控制器覆盖 PYTHONPATH 丢掉 CANN 的
`acl` 模块目录，导致三个 Worker 签名导入失败。修正控制器保留原环境后全 290 项通过；
失败原件保留，没有安装软件包或修改共享环境。

CPU 验证不执行 C2/NPU 内核；修复后的 A5 U2/Graph/多流设备效果仍待重跑。
完整七场景、训练检查点、Engram、物理重叠与性能仍待验收，任务整项勾选不变。
更新分支后按[集中指引](../../../../../docs/npu/DSV41_A5_FUNCTIONAL_VALIDATION_ZH.md)
运行完整七场景到全新目录，回传整个目录以保留同轮 source/summary 和监督结果。

复核与复跑代码均保留在本目录：`assess_archives.py` 核对原始上传；
`run_cpu_validation.py` + `cpu_native_runner.py` 使用既有 A3 依赖执行隔离 CPU 检查。
