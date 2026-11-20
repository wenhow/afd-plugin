# DSpark draft 头数修复与完整 eager/U2 归档分析

2026-11-20 用户要求修复三份组合归档的问题，并分析新增
`C:/Users/zq/Downloads/full-eager-u2-a3guux9z.tar.gz`。
本轮修改限于 afd-plugin 工作区；未安装依赖、未启动 A5。
实现与 CPU 检查完成后，用户明确要求将本轮修复提交并推送。
已有脏文件及其他未请求改动保留，上游和 donor 保持只读。

## 已修复：原生 DSpark draft 漏接 TP4 head adapter

`full-eager-u1-em3yen9n` 实际为 DSpark N=1，`full-graph-u2-io6ueywd` 为
DSpark N=2。两者请求均在原生 draft proposal 的 Attention 抛出：

```text
ValueError: N1 must be 64 (the only supported query-head count), got 16
```

target 已有 per-Attention 适配，但 native draft 由独立的
`DeepseekV41DSparkAttention`（继承 SWA Attention）构造，没有经过 target 构造器。
错误栈直接进入 native `qsmla`，已有四项 packed probe 不覆盖真实 draft 选择。
此前离线展开及调用链见本地
`results/analysis-full-checkpoint-20261120/README.md`。

在 `AFDNPUAttentionModelRunnerV2.load_model` 的 native load 返回后、worker profiling、
KV 和 capture 前，遍历实际 `self.speculator.model`，只为 V4.1+DSpark 且精确使用
native packed backend 的 draft Attention 实例安装既有
`AFDMixedQuantPackedCacheOps`。日志记录实际适配层数。
没有替换 draft 模型或完整本地 MoE，也没有增加 F 事务或改动 proposal/sampling。
A3 的 None、其他 backend、其他模型和其他 speculative method 保留原行为。

该修复属于所有调用该加载入口的 DSpark eager/Graph、U1/U2 组合的共同前置；
CPU 检查通过不证明这些组合的设备请求或接受拒绝已通过。
代价是一次 draft 模块遍历及既有 adapter 的 4 倍 query-head 算量，KV/HCCL 不扩展。
待 wheel 原生支持 16 heads 时删除该选择及 adapter。
精确上游签名、所有权和移除审查见 [review.json](review.json)。
完整权重 preflight 新增 `dspark.py` 的锁定 Git 哈希，见
[source-pins.json](source-pins.json)，不修改受检环境或降低源文件检查。

## 新归档：eager/U2 执行完成，严格输出对照失败

归档 SHA256：`824f9cdd9817d01c71c45e8cabd9c1f71a4ed995b63c3fd0439126ef0af32837`。
实际 HEAD 为 `d3a6f27baae2d7651806f4c0d9887328360e66d9`，
mode=eager-u2、DSpark 关闭、A4/F4 TP4、原始量化 safetensors、CPU offload Engram，
stream overlap 开启。26 份受检源文件全部匹配，packed probe 与健康检查通过。

两次请求均 HTTP 200，各返回 32 token、`finish_reason=length`，没有 Python 异常。
输入 token 与原生参考一致，输出首次分歧（从 1 开始）为：

| 请求 | 首次分歧 | 原生 token | eager/U2 token |
| --- | --- | --- | --- |
| request-00，工作记录 | 第 1 个 | 33213 | 1999 |
| request-01，数列 | 第 13 个 | 14 | 16 |

A/F 各有 128 条 U2 `parent_finish_returned`：四个 role rank 各 32 条，
对应同一批 32 个 parent transaction，`num_stages=2`、overlap 有效值为 true。
因此确实进入并完成 U2，不能因最终失败将其描述为配置拒绝或未执行双 stage；
这些 host 边界日志不单独证明物理重叠或性能收益。

启动器因首波输出对照失败返回 1，没有执行取消、第二波或长输入。
`request_passed=false`、`hardware_acceptance=false`；正常清理通过，A/F/probe exit 0、
F rank 0–3 标记齐全、无强制清理/残留，最终八卡 idle。
原始文件哈希、逐项对照和事件计数见 [received-results.json](received-results.json)。

另重新读取旧 AFD eager/U1 `full-eager-u1-pqdgncjt` 原始归档：
U2 与它的输入相同，两条输出分别从第 5/14 个 token 分歧，28/17 个输出位置不同，
并非全序列一致。本轮过程中曾将相似前缀误述为全序列相同，已按原始 token 纠正。

## 数值差异仍未闭合，新增 Graph/U1 串行诊断

`full-graph-u1-3hewjwkz` 已完成实际 capture/replay 和两个 HTTP 200、各 32 token，
但分别从第 1/2 个 token 与原生参考分歧。它与新 eager/U2 属于输出一致性失败，
不属于上述 draft shape 异常。
原生参考 `full-eager-u1-7z0p69ts` 本身在重复相同输入时也分别从第 1/13 个 token
分歧；中间存在流式取消，现有资料不足以唯一归因取消、Engram、Graph、HCCL 或量化。
本轮没有对未定位的数值路径作推测性修改，也没有将差异视为可忽略误差。

现有 `--repeatability` 扩展为 standalone eager/U1 或 Graph/U1、DSpark 关闭。
Graph/U1 仍使用 FULL_DECODE_ONLY capture/replay，原生控制继续使用 eager/U1；
两者使用同一串行请求协议及稳定原生参考。客户端和 max-num-seqs 都为 1，
预热、取消前后多轮响应完整保留；纯 token 差异继续采集，最终仍严格失败。
普通双请求矩阵不变。U2 仍拒绝该串行协议，避免将 U1 fallback 当作 U2 验收。

设备后续顺序沿用[完整权重指导](../../../../../docs/npu/DSV41_A5_FULL_CHECKPOINT_VALIDATION_ZH.md)：
先原生 eager/U1 串行稳定，再 AFD eager/U1 对照，随后 Graph/U1 串行对照。
DSpark 修复的真实 draft/kernel 与接受拒绝结果需在后续 A5 组合中复核。
1.3、1.8、2.2–2.5 和完整矩阵任务仍 pending，不能宣称所有问题已经解决。

## 验证记录

- 修复前：新增 public load seam 测试在真实依赖 CPU 环境中为 2F/4P，失败均为
  16-head 拒绝，见 `cpu-red-001/results/junit.xml`。
- 工作区回归：A runner、packed adapter、native control worker 和唯一上游签名表四个完整
  模块共 201P，0F/E/S，见 `cpu-green-002/results/`。该快照包含其他本地改动，
  因此另冻结 HEAD `825fb438` 加本轮 11 个交付文件，排除其他未交付改动后复核。
- 交付范围复核：相同四个模块共 190P，0F/E/S，见 `cpu-delivery-001/results/`。
  相差的 11 项来自未交付的签名表改动。两次检查的 NPU 前后均未初始化，
  受检 native pins 一致、快照源码未变。交付 manifest 的 `overlays` 明确列出范围。
- CPU 驱动在授权的 A3 临时工作区运行，不安装依赖；拦截 `_lazy_init` 并将设备
  synchronize 替换为 host no-op。CPU headwise 算术和 host 生命周期不模拟 packed
  device kernel，也不构成设备同步、capture/replay 或 HCCL 正确性的验收。
- 中间扩大检查的 `cpu-green-001` 有 27F/5E：31 项触及未隔离的设备 synchronize，
  另 1 项 native load fake 没有发布实际生命周期所需的 speculator。
  已修正 CPU 驱动和共享 fake 后重跑，未通过删除测试或放开 NPU 初始化绕过。
- 主机诊断两模块共 31 个测试方法（JUnit tests 含 subtest 为 66），0F/E/S，
  覆盖源文件准入、Graph/U1 串行计划、原生参考协议、HTTP、取消前后对照和严格失败。
  见 [host-tests.xml](host-tests.xml)。
- Ruff、格式、diff 检查和本地 harness 规划/上游保持检查另记录在 validation.json。
  harness 不等于官方 OpenSpec 验证、运行时包验收或完整模型验收。

CPU 驱动保留在 [cpu_runner.py](cpu_runner.py)。仓库保留 manifest、汇总、修复前与最终
JUnit；中间 `cpu-green-001` 的大型失败日志和全部 `source.tar.gz` 仅保留在本地与授权
A3 临时目录，避免将其他未交付源码随测试快照推送。`remote.json` 记录临时目录和快照哈希。
当前状态为实现与 CPU 通过；A5 数值、DSpark 修复及组合验收待实际结果。
